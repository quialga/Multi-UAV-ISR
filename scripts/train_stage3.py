"""
scripts/train_stage3.py — Train the Stage 3 CTDE recurrent GNN policy.

Partial-observability + GRU actor + centralised (full-state) GNN critic.
Design: docs/stage3_design.md.

Same PPO outer loop as Stage 1/2 with three additions:

1. Rollouts come from ``RecurrentVectorPursuitEnv`` — obs dict now
   carries two extra keys (``bb_edge_visible``, ``rb_edge_visible``).
2. Hidden state is trainer-owned: initialised to zeros, advanced by
   the actor's GRU each step, reset to zeros on env-done boundaries.
3. Optional warm-start of the CTDE critic path from a Stage 2 GNN
   checkpoint via ``policy.load_stage2_critic(...)``.

Run:
    python scripts/train_stage3.py
    python scripts/train_stage3.py --n-rollouts 5 --no-eval   # smoke

Outputs go under ``runs/stage3/{timestamp}/``:
    - tb/                 (TensorBoard event files)
    - checkpoint_*.pt     (periodic snapshots)
    - final.pt            (last rollout)
    - train_log.txt       (mirror of stdout)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.tensorboard import SummaryWriter

from isr.configs.stage3_default import STAGE3_DEFAULTS
from isr.env.pursuit_env    import PursuitEnv
from isr.agents.heuristics  import (
    GreedyPursuer, RandomAgent,
    run_from_nearest_uav, stationary_red,
)
from isr.agents.gnn_ctde_policy import GNNCTDEPolicy
from isr.train.graph_buffer     import RecurrentGraphRolloutBuffer
from isr.train.ppo              import ppo_update_ctde
from isr.train.vec_env          import RecurrentVectorPursuitEnv


BASE_OBS_KEYS = (
    "blue_features", "red_features",
    "bb_edge_features", "rb_edge_features",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    d = STAGE3_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Env
    p.add_argument("--n-blue",         type=int,   default=d["n_blue"])
    p.add_argument("--n-red",          type=int,   default=d["n_red"])
    p.add_argument("--arena-size",     type=float, default=d["arena_size"])
    p.add_argument("--max-steps",      type=int,   default=d["max_steps"])
    p.add_argument("--capture-radius", type=float, default=d["capture_radius"])
    p.add_argument("--sensor-radius",  type=float, default=d["sensor_radius"],
                   help="Actor's sensor range for partial observability.")
    # PPO
    p.add_argument("--n-envs",         type=int,   default=d["n_envs"])
    p.add_argument("--rollout-steps",  type=int,   default=d["rollout_steps"])
    p.add_argument("--n-rollouts",     type=int,   default=d["n_rollouts"])
    p.add_argument("--n-epochs",       type=int,   default=d["n_epochs"])
    p.add_argument("--mb-size",        type=int,   default=d["mb_size"])
    p.add_argument("--lr",             type=float, default=d["lr"])
    p.add_argument("--clip-eps",       type=float, default=d["clip_eps"])
    p.add_argument("--ent-coef",       type=float, default=d["ent_coef"])
    p.add_argument("--vf-coef",        type=float, default=d["vf_coef"])
    p.add_argument("--max-grad-norm",  type=float, default=d["max_grad_norm"])
    p.add_argument("--gamma",          type=float, default=d["gamma"])
    p.add_argument("--gae-lambda",     type=float, default=d["gae_lambda"])
    p.add_argument("--target-kl",      type=float, default=d["target_kl"],
                   help="Per-epoch mean-KL early stop.  Pass 0 to disable.")
    p.add_argument("--aux-hidden-coef", type=float,
                   default=d["aux_hidden_coef"],
                   help="Coef for belief-state distillation MSE loss "
                        "(0 disables — baseline Stage 3).  See "
                        "docs/stage3_design.md follow-up.")
    p.add_argument("--freeze-critic", action="store_true",
                   help="Freeze the entire CTDE critic (encoder + trunk + "
                        "head) at the warm-started weights.  The aux target "
                        "then becomes a stable Stage 2 oracle instead of a "
                        "drifting one.  Requires --warm-start-critic to be "
                        "set to a valid path.  value_loss is still computed "
                        "as a diagnostic but no longer updates the critic.")
    # Logging / eval
    p.add_argument("--log-interval",   type=int,   default=d["log_interval"])
    p.add_argument("--save-interval",  type=int,   default=d["save_interval"])
    p.add_argument("--no-eval",        action="store_true",
                   help="skip heuristic-baseline calibration printout at start")
    # Red-policy mixing
    p.add_argument("--red-policy-mix", default="stationary:1,random:1,run:1",
                   help="Comma-separated 'name:weight' pairs specifying the "
                        "per-episode red-policy distribution.")
    # GNN
    p.add_argument("--d-hidden",     type=int, default=d["d_hidden"])
    p.add_argument("--n-msg-rounds", type=int, default=d["n_msg_rounds"])
    # Warm-start
    p.add_argument("--warm-start-critic",
                   default=d["warm_start_critic"],
                   help="Path to a Stage 2 checkpoint to seed the CTDE critic. "
                        "Pass '' or 'none' to skip warm-start.")
    # Run management
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--device",         default="cpu",
                   help="cpu / cuda / cuda:0 — Stage 3 strongly prefers GPU.")
    p.add_argument("--run-name",       default=None,
                   help="override the timestamp directory under runs/stage3/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Baseline helper (same as Stage 1)
# ---------------------------------------------------------------------------

def evaluate_heuristic_baseline(
    agent_factory,
    env_kwargs:  Dict,
    red_policy,
    n_episodes:  int,
    seed_base:   int = 20_000,
) -> Dict[str, float]:
    returns, caught, steps = [], [], []
    for ep in range(n_episodes):
        env = PursuitEnv(**env_kwargs, red_policy=red_policy,
                         seed=seed_base + ep)
        obs_d, _ = env.reset(seed=seed_base + ep)
        blue = agent_factory()
        total = 0.0
        while env.agents:
            actions = {a: blue.act(obs_d[a], env, a) for a in env.agents}
            obs_d, rew_d, term_d, trunc_d, info_d = env.step(actions)
            total += rew_d[env.possible_agents[0]]
        snap = env.state_snapshot()
        returns.append(total)
        caught.append(int((~snap["red_active"]).sum()))
        steps.append(int(snap["t"]))
    return {
        "mean_return": float(np.mean(returns)),
        "mean_caught": float(np.mean(caught)),
        "mean_steps":  float(np.mean(steps)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_device(obs_dict, device):
    return {k: torch.from_numpy(v).float().to(device) for k, v in obs_dict.items()}


def _split_views(obs_t):
    """Split a partial-obs dict into (partial_obs, full_state)."""
    partial_obs = obs_t
    full_state  = {k: obs_t[k] for k in BASE_OBS_KEYS}
    return partial_obs, full_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir  = Path("runs") / "stage3" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    log_file = open(run_dir / "train_log.txt", "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Run dir: {run_dir}")
    log(f"Args: {vars(args)}")

    # ---- Env kwargs ---------------------------------------------------
    env_kwargs = dict(
        n_blue         = args.n_blue,
        n_red          = args.n_red,
        arena_size     = args.arena_size,
        max_steps      = args.max_steps,
        capture_radius = args.capture_radius,
        sensor_radius  = args.sensor_radius,
    )

    # ---- Red-policy mix parse ----------------------------------------
    red_policy_mix = []
    for chunk in args.red_policy_mix.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, w = chunk.split(":")
        red_policy_mix.append((name.strip(), float(w)))
    log(f"red_policy_mix: {red_policy_mix}")

    # ---- Heuristic baselines (one-off) --------------------------------
    if not args.no_eval:
        log("Computing heuristic baselines for reference (fully-observable env)...")
        # Baselines run the heuristics against the FULL env (they use
        # the flat per-agent obs, not the graph obs).  Drop sensor_radius
        # so those runs still see all reds.
        env_kwargs_full = {k: v for k, v in env_kwargs.items()
                           if k != "sensor_radius"}
        for name, factory, red_name, red in [
            ("Greedy vs Run",        GreedyPursuer, "run",        run_from_nearest_uav),
            ("Greedy vs Stationary", GreedyPursuer, "stationary", stationary_red),
            ("Random vs Run",        lambda: RandomAgent(seed=args.seed),
                                     "run",        run_from_nearest_uav),
        ]:
            r = evaluate_heuristic_baseline(factory, env_kwargs_full, red, n_episodes=20)
            log(f"  {name:<22}: mean_return={r['mean_return']:+7.2f}  "
                f"caught={r['mean_caught']:.2f}/{args.n_red}  "
                f"steps={r['mean_steps']:.1f}")
            writer.add_scalar(f"baselines/{name.replace(' ', '_')}",
                              r["mean_return"], 0)

    # ---- Vec env ------------------------------------------------------
    vec_env = RecurrentVectorPursuitEnv(
        n_envs              = args.n_envs,
        env_kwargs          = env_kwargs,
        base_seed           = args.seed,
        episode_buffer_size = 256,
        red_policy_mix      = red_policy_mix,
    )
    n_agents   = vec_env.n_agents
    action_dim = vec_env.action_dim

    # ---- Policy + optimiser ------------------------------------------
    policy = GNNCTDEPolicy(
        n_blue        = vec_env.n_blue,
        n_red         = vec_env.n_red,
        blue_feat_dim = vec_env.blue_feat_dim,
        red_feat_dim  = vec_env.red_feat_dim,
        edge_feat_dim = vec_env.edge_feat_dim,
        action_dim    = action_dim,
        d_hidden      = args.d_hidden,
        n_msg_rounds  = args.n_msg_rounds,
        init_log_std  = STAGE3_DEFAULTS.get("init_log_std", 0.0),
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    log(f"Policy: GNNCTDEPolicy d_hidden={args.d_hidden} "
        f"rounds={args.n_msg_rounds} params={n_params}")

    # ---- Warm-start critic from a Stage 2 checkpoint -----------------
    warm_path = args.warm_start_critic
    warm_ok = False
    if warm_path and warm_path.lower() not in ("", "none"):
        if not Path(warm_path).exists():
            log(f"WARNING: warm_start_critic path {warm_path} does not exist. "
                f"Continuing with random critic init.")
        else:
            n_copied = policy.load_stage2_critic(warm_path)
            log(f"Warm-started CTDE critic from {warm_path} — copied {n_copied} tensors.")
            warm_ok = True
    else:
        log("No critic warm-start (random init for the critic path).")

    # ---- Optional: freeze the critic ---------------------------------
    # When on, the critic is a fixed Stage 2 oracle throughout Stage 3
    # training.  All aux belief-state distillation runs against this
    # stable target (option A in docs/stage3_design.md follow-up).
    if args.freeze_critic:
        if not warm_ok:
            raise ValueError(
                "--freeze-critic requires --warm-start-critic to be a valid "
                "path (otherwise you would freeze a randomly initialised "
                "critic, which would produce meaningless value estimates "
                "and a meaningless aux target)."
            )
        for p_ in policy.critic_encoder.parameters():
            p_.requires_grad = False
        for p_ in policy.critic_trunk.parameters():
            p_.requires_grad = False
        for p_ in policy.critic_head.parameters():
            p_.requires_grad = False
        n_frozen    = sum(p_.numel() for p_ in policy.parameters()
                          if not p_.requires_grad)
        n_trainable = sum(p_.numel() for p_ in policy.parameters()
                          if p_.requires_grad)
        log(f"Critic frozen: {n_frozen} params frozen, {n_trainable} trainable.")

    args_dict_saved = {**vars(args), "policy_type": "gnn_ctde"}

    # ---- Rollout buffer ----------------------------------------------
    buffer = RecurrentGraphRolloutBuffer(
        rollout_steps = args.rollout_steps,
        n_envs        = args.n_envs,
        n_agents      = n_agents,
        action_dim    = action_dim,
        obs_feature_dims={
            "blue_features":    vec_env.blue_feat_dim,
            "red_features":     vec_env.red_feat_dim,
            "bb_edge_features": vec_env.edge_feat_dim,
            "rb_edge_features": vec_env.edge_feat_dim,
        },
        obs_token_counts={
            "blue_features":    vec_env.n_blue,
            "red_features":     vec_env.n_red,
            "bb_edge_features": vec_env.n_bb_edges,
            "rb_edge_features": vec_env.n_rb_edges,
        },
        n_bb_edges = vec_env.n_bb_edges,
        n_rb_edges = vec_env.n_rb_edges,
        d_hidden   = args.d_hidden,
        device     = device,
    )

    log(f"\nStarting Stage 3 training: {args.n_rollouts} rollouts x "
        f"{args.rollout_steps} steps x {args.n_envs} envs x {n_agents} agents")

    # target_kl: 0 means disabled.
    target_kl_arg = args.target_kl if args.target_kl and args.target_kl > 0 else None

    obs_np = vec_env.reset(seed=args.seed)
    hidden = policy.initial_hidden(n_envs=args.n_envs, device=device)
    t_start = time.time()
    global_step = 0

    for rollout in range(args.n_rollouts):
        buffer.reset()

        for step in range(args.rollout_steps):
            obs_t = _dict_to_device(obs_np, device)
            partial_obs, full_state = _split_views(obs_t)
            with torch.no_grad():
                action_t, log_p_t, _, value_t, new_hidden = \
                    policy.get_action_and_value(partial_obs, full_state, hidden)
            action_np = action_t.cpu().numpy().astype(np.float32)
            next_obs_np, reward_np, done_np, _ = vec_env.step(action_np)

            buffer.add(
                obs        = {k: obs_t[k] for k in BASE_OBS_KEYS},
                actions    = action_t,
                log_probs  = log_p_t,
                values     = value_t,
                rewards    = torch.from_numpy(reward_np).to(device),
                dones      = torch.from_numpy(done_np).to(device),
                bb_visible = obs_t["bb_edge_visible"],
                rb_visible = obs_t["rb_edge_visible"],
                hidden     = hidden,
            )

            # Advance hidden for next step; reset rows where the env
            # just terminated (auto-reset returned the fresh obs).
            done_t = torch.from_numpy(done_np).to(device).view(-1, 1, 1)
            hidden = new_hidden * (1.0 - done_t)

            obs_np = next_obs_np
            global_step += args.n_envs

        # Bootstrap V from the state after the last step in the rollout.
        with torch.no_grad():
            last_obs_t = _dict_to_device(obs_np, device)
            _, last_full_state = _split_views(last_obs_t)
            last_value = policy.critic_forward(last_full_state)
        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        update_metrics = ppo_update_ctde(
            policy          = policy,
            optimizer       = optimizer,
            buffer          = buffer,
            clip_eps        = args.clip_eps,
            ent_coef        = args.ent_coef,
            vf_coef         = args.vf_coef,
            max_grad_norm   = args.max_grad_norm,
            n_epochs        = args.n_epochs,
            mb_size         = args.mb_size,
            value_clip      = STAGE3_DEFAULTS["value_clip"],
            normalize_adv   = STAGE3_DEFAULTS["normalize_adv"],
            target_kl       = target_kl_arg,
            aux_hidden_coef = args.aux_hidden_coef,
        )

        ep_stats = vec_env.recent_episode_stats()
        if (rollout + 1) % args.log_interval == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            aux_str = (f"  aux={update_metrics['aux_hidden_loss']:.4f}"
                       if args.aux_hidden_coef > 0 else "")
            log(
                f"[rollout {rollout+1:>4d}/{args.n_rollouts}]  "
                f"steps={global_step:>9d}  sps={sps:>6.0f}  "
                f"epR={ep_stats['mean_return']:+7.2f}+-{ep_stats['std_return']:.2f}  "
                f"caught={ep_stats['mean_caught']:.2f}/{args.n_red}  "
                f"pol={update_metrics['policy_loss']:+.4f}  "
                f"val={update_metrics['value_loss']:.4f}  "
                f"ent={update_metrics['entropy']:.3f}  "
                f"kl={update_metrics['approx_kl']:.4f}  "
                f"clip={update_metrics['clip_frac']:.3f}  "
                f"eps={update_metrics['n_epochs_run']}"
                f"{aux_str}"
            )
        writer.add_scalar("rollout/mean_return", ep_stats["mean_return"], global_step)
        writer.add_scalar("rollout/mean_caught", ep_stats["mean_caught"], global_step)
        writer.add_scalar("rollout/mean_length", ep_stats["mean_length"], global_step)
        writer.add_scalar("ppo/policy_loss",     update_metrics["policy_loss"],     global_step)
        writer.add_scalar("ppo/value_loss",      update_metrics["value_loss"],      global_step)
        writer.add_scalar("ppo/entropy",         update_metrics["entropy"],         global_step)
        writer.add_scalar("ppo/approx_kl",       update_metrics["approx_kl"],       global_step)
        writer.add_scalar("ppo/clip_frac",       update_metrics["clip_frac"],       global_step)
        writer.add_scalar("ppo/n_epochs_run",    update_metrics["n_epochs_run"],    global_step)
        writer.add_scalar("ppo/aux_hidden_loss", update_metrics["aux_hidden_loss"], global_step)

        if (rollout + 1) % args.save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_{rollout+1:05d}.pt"
            torch.save({
                "policy_state": policy.state_dict(),
                "rollout":      rollout + 1,
                "global_step":  global_step,
                "args":         args_dict_saved,
            }, ckpt_path)

    torch.save({
        "policy_state": policy.state_dict(),
        "rollout":      args.n_rollouts,
        "global_step":  global_step,
        "args":         args_dict_saved,
    }, run_dir / "final.pt")

    elapsed = time.time() - t_start
    log(f"\nStage 3 training done.  Elapsed: {elapsed/60:.1f} min")
    log(f"Final checkpoints under {run_dir}")
    log_file.close()
    writer.close()


if __name__ == "__main__":
    main()
