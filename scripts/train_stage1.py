"""
scripts/train_stage1.py — Train the blue GNN + CTDE-critic policy.

Standard recurrent PPO loop (recurrence lands in Stage 3):

    for rollout in range(n_rollouts):
        1. Collect rollout_steps transitions across n_envs envs.
        2. Compute GAE advantages + returns.
        3. Run n_epochs of minibatch SGD over the buffer.
        4. Log to stdout + TensorBoard.
        5. Save checkpoint on save_interval.

The MLP + flat-obs training path was removed in the July-2026 cleanup
after the Stage 2 scaling experiment (see docs/stage2_results.md).
The filename ``train_stage1.py`` is kept for continuity of the
docs/README references; going forward this trains the GNN backbone
used by every subsequent stage.

Run:
    python scripts/train_stage1.py
    python scripts/train_stage1.py --n-rollouts 200 --no-eval     # quick smoke

Outputs go under ``runs/stage1/{timestamp}/``:
    - tb/                 (TensorBoard event files)
    - checkpoint_*.pt     (periodic model snapshots)
    - final.pt            (last rollout)
    - train_log.txt       (mirror of stdout)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

# Make `python scripts/train_stage1.py` work from the repo root without
# ``pip install -e .`` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.tensorboard import SummaryWriter

from isr.configs.stage1_default import STAGE1_DEFAULTS
from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import (
    GreedyPursuer, RandomAgent,
    run_from_nearest_uav, stationary_red,
)
from isr.agents.gnn_policy import GNNActorCritic
from isr.train.graph_buffer import GraphRolloutBuffer
from isr.train.ppo    import ppo_update
from isr.train.vec_env import VectorPursuitEnv


# ---------------------------------------------------------------------------
# CLI parsing — most flags pre-loaded from STAGE1_DEFAULTS
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    d = STAGE1_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Env
    p.add_argument("--n-blue",         type=int,   default=d["n_blue"])
    p.add_argument("--n-red",          type=int,   default=d["n_red"])
    p.add_argument("--arena-size",     type=float, default=d["arena_size"])
    p.add_argument("--max-steps",      type=int,   default=d["max_steps"])
    p.add_argument("--capture-radius", type=float, default=d["capture_radius"])
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
    # Logging / eval
    p.add_argument("--log-interval",   type=int,   default=d["log_interval"])
    p.add_argument("--save-interval",  type=int,   default=d["save_interval"])
    p.add_argument("--no-eval",        action="store_true",
                   help="skip the one-off heuristic-baseline printout at start "
                        "(saves ~30s on smoke runs)")
    # v2: red policy mixing during training
    p.add_argument("--red-policy-mix", default="stationary:1,random:1,run:1",
                   help="Comma-separated 'name:weight' pairs specifying the "
                        "per-episode red-policy distribution.  Names: "
                        "stationary / random / run.")
    # GNN hyperparameters
    p.add_argument("--d-hidden", type=int, default=64,
                   help="Node embedding dimension for the GNN policy.")
    p.add_argument("--n-msg-rounds", type=int, default=2,
                   help="Rounds of message passing in the GNN policy.")
    # Run management
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--device",         default="cpu",
                   help="cpu / cuda / cuda:0 — GNN training strongly prefers GPU.")
    p.add_argument("--run-name",       default=None,
                   help="override the timestamp directory under runs/stage1/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Heuristic-baseline reference (printed once at start for calibration)
# ---------------------------------------------------------------------------

def evaluate_heuristic_baseline(
    agent_factory,
    env_kwargs:  Dict,
    red_policy,
    n_episodes:  int,
    seed_base:   int = 20_000,
) -> Dict[str, float]:
    """Run a heuristic blue baseline (RandomAgent / GreedyPursuer) for comparison."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Run dir
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir  = Path("runs") / "stage1" / run_name
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

    # ---- Heuristic baselines (one-off calibration reference) ----------
    if not args.no_eval:
        log("Computing heuristic baselines for reference...")
        for name, factory, red_name, red in [
            ("Greedy vs Run",        GreedyPursuer, "run",        run_from_nearest_uav),
            ("Greedy vs Stationary", GreedyPursuer, "stationary", stationary_red),
            ("Random vs Run",        lambda: RandomAgent(seed=args.seed),
                                     "run",        run_from_nearest_uav),
        ]:
            r = evaluate_heuristic_baseline(factory, env_kwargs, red, n_episodes=20)
            log(f"  {name:<22}: mean_return={r['mean_return']:+7.2f}  "
                f"caught={r['mean_caught']:.2f}/{args.n_red}  "
                f"steps={r['mean_steps']:.1f}")
            writer.add_scalar(f"baselines/{name.replace(' ', '_')}",
                              r["mean_return"], 0)

    # ---- Vec env ------------------------------------------------------
    vec_env = VectorPursuitEnv(
        n_envs              = args.n_envs,
        env_kwargs          = env_kwargs,
        base_seed           = args.seed,
        episode_buffer_size = 256,
        red_policy_mix      = red_policy_mix,
    )
    n_agents   = vec_env.n_agents
    action_dim = vec_env.action_dim

    # ---- Policy + optimiser ------------------------------------------
    policy = GNNActorCritic(
        n_blue        = vec_env.n_blue,
        n_red         = vec_env.n_red,
        blue_feat_dim = vec_env.blue_feat_dim,
        red_feat_dim  = vec_env.red_feat_dim,
        edge_feat_dim = vec_env.edge_feat_dim,
        action_dim    = action_dim,
        d_hidden      = args.d_hidden,
        n_msg_rounds  = args.n_msg_rounds,
        init_log_std  = STAGE1_DEFAULTS["init_log_std"],
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    log(f"Policy: GNNActorCritic d_hidden={args.d_hidden} "
        f"rounds={args.n_msg_rounds} params={n_params}")

    # Save policy_type + hyperparams into the checkpoint args so the
    # loader can reconstruct the network without duplicating dims.
    args_dict_saved = {
        **vars(args),
        "policy_type": "gnn",
    }

    # ---- Rollout buffer ----------------------------------------------
    buffer = GraphRolloutBuffer(
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
        device        = device,
    )

    log(f"\nStarting GNN training: {args.n_rollouts} rollouts x "
        f"{args.rollout_steps} steps x {args.n_envs} envs x {n_agents} agents")

    obs_np = vec_env.reset(seed=args.seed)
    t_start = time.time()
    global_step = 0

    for rollout in range(args.n_rollouts):
        buffer.reset()

        for step in range(args.rollout_steps):
            obs_t = _dict_to_device(obs_np, device)
            with torch.no_grad():
                action_t, log_p_t, _, value_t = policy.get_action_and_value(obs_t)
            action_np = action_t.cpu().numpy().astype(np.float32)
            next_obs_np, reward_np, done_np, _ = vec_env.step(action_np)
            # Vec env returns per-agent rewards (n_envs, n_agents).  Agents
            # are NOT identical any more — control effort (and, if enabled,
            # crash / clearance) is individual — so column 0 is one agent's
            # reward, not the team's.  Stage 1 is a shared-reward learner and
            # wants a per-env scalar: take the MEAN over agents, which is
            # also how vec_env accumulates episode returns.
            reward_np = reward_np.mean(axis=1)
            buffer.add(
                obs       = obs_t,
                actions   = action_t,
                log_probs = log_p_t,
                values    = value_t,
                rewards   = torch.from_numpy(reward_np).to(device),
                dones     = torch.from_numpy(done_np).to(device),
            )
            obs_np = next_obs_np
            global_step += args.n_envs

        with torch.no_grad():
            last_obs_t = _dict_to_device(obs_np, device)
            _, _, _, last_value = policy.get_action_and_value(last_obs_t)
        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        update_metrics = ppo_update(
            policy        = policy,
            optimizer     = optimizer,
            buffer        = buffer,
            clip_eps      = args.clip_eps,
            ent_coef      = args.ent_coef,
            vf_coef       = args.vf_coef,
            max_grad_norm = args.max_grad_norm,
            n_epochs      = args.n_epochs,
            mb_size       = args.mb_size,
            value_clip    = STAGE1_DEFAULTS["value_clip"],
            normalize_adv = STAGE1_DEFAULTS["normalize_adv"],
        )

        ep_stats = vec_env.recent_episode_stats()
        if (rollout + 1) % args.log_interval == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            log(
                f"[rollout {rollout+1:>4d}/{args.n_rollouts}]  "
                f"steps={global_step:>9d}  sps={sps:>6.0f}  "
                f"epR={ep_stats['mean_return']:+7.2f}+-{ep_stats['std_return']:.2f}  "
                f"caught={ep_stats['mean_caught']:.2f}/{args.n_red}  "
                f"pol={update_metrics['policy_loss']:+.4f}  "
                f"val={update_metrics['value_loss']:.4f}  "
                f"ent={update_metrics['entropy']:.3f}  "
                f"kl={update_metrics['approx_kl']:.4f}  "
                f"clip={update_metrics['clip_frac']:.3f}"
            )
        writer.add_scalar("rollout/mean_return", ep_stats["mean_return"], global_step)
        writer.add_scalar("rollout/mean_caught", ep_stats["mean_caught"], global_step)
        writer.add_scalar("rollout/mean_length", ep_stats["mean_length"], global_step)
        writer.add_scalar("ppo/policy_loss", update_metrics["policy_loss"], global_step)
        writer.add_scalar("ppo/value_loss",  update_metrics["value_loss"],  global_step)
        writer.add_scalar("ppo/entropy",     update_metrics["entropy"],     global_step)
        writer.add_scalar("ppo/approx_kl",   update_metrics["approx_kl"],   global_step)
        writer.add_scalar("ppo/clip_frac",   update_metrics["clip_frac"],   global_step)

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
    log(f"\nGNN training done.  Elapsed: {elapsed/60:.1f} min")
    log(f"Final checkpoints under {run_dir}")
    log_file.close()
    writer.close()


if __name__ == "__main__":
    main()
