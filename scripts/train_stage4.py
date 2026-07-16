"""
scripts/train_stage4.py — Train the Stage 4 CTDE + belief-map policy.

Extends the Stage 3 recurrent CTDE trainer with:

1. Rollouts come from ``Stage4VectorPursuitEnv`` — obs dict carries
   ``blue_features``, ``bb_edge_features``, ``belief_maps``,
   ``obstacle_positions``, ``true_occupancy``.  No red-side or
   visibility-mask keys.
2. Policy is ``GNNStage4Policy`` — CNN belief encoders (actor +
   critic) + blue-only GNN + GRU.  Cold-started critic.
3. PPO update is ``ppo_update_stage4`` — same clipped objective as
   Stage 3, plus a diagnostic BCE metric on the fused belief.

Run:
    python scripts/train_stage4.py
    python scripts/train_stage4.py --n-rollouts 5 --no-eval   # smoke

Outputs go under ``runs/stage4/{timestamp}/``:
    - tb/                 (TensorBoard event files)
    - checkpoint_*.pt     (periodic snapshots)
    - final.pt            (last rollout)
    - best.pt             (peak checkpoint by --best-ckpt-metric)
    - train_log.txt       (mirror of stdout)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.tensorboard import SummaryWriter

from isr.configs.stage4_default import STAGE4_DEFAULTS
from isr.env.pursuit_env    import PursuitEnv
from isr.agents.heuristics  import (
    GreedyPursuer, RandomAgent,
    run_from_nearest_uav, stationary_red,
)
from isr.agents.gnn_stage4_policy import GNNStage4Policy
from isr.train.graph_buffer       import Stage4RolloutBuffer
from isr.train.ppo                import ppo_update_stage4
from isr.train.vec_env            import Stage4VectorPursuitEnv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    d = STAGE4_DEFAULTS
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Env
    p.add_argument("--n-blue",         type=int,   default=d["n_blue"])
    p.add_argument("--n-red",          type=int,   default=d["n_red"])
    p.add_argument("--arena-size",     type=float, default=d["arena_size"])
    p.add_argument("--max-steps",      type=int,   default=d["max_steps"])
    p.add_argument("--capture-radius", type=float, default=d["capture_radius"])
    p.add_argument("--sensor-radius",  type=float, default=d["sensor_radius"])
    # Stage 4 obstacle + belief knobs
    p.add_argument("--n-obstacles",           type=int,   default=d["n_obstacles"])
    p.add_argument("--obstacle-radius-min",   type=float, default=d["obstacle_radius_min"])
    p.add_argument("--obstacle-radius-max",   type=float, default=d["obstacle_radius_max"])
    p.add_argument("--obstacle-spawn-clearance",
                   type=float, default=d["obstacle_spawn_clearance"])
    p.add_argument("--belief-grid-size", type=int,   default=d["belief_grid_size"])
    p.add_argument("--belief-channels",  type=int,   default=d["belief_channels"])
    p.add_argument("--belief-clip",      type=float, default=d["belief_clip"])
    p.add_argument("--p-tp",             type=float, default=d["p_TP"])
    p.add_argument("--p-fp",             type=float, default=d["p_FP"])
    p.add_argument("--ray-step-size",    type=float, default=d["ray_step_size"])
    p.add_argument("--belief-encoder-out-dim",
                   type=int, default=d["belief_encoder_out_dim"])
    p.add_argument("--belief-window-size", type=int,
                   default=d.get("belief_window_size", 0),
                   help="Side K of the ego-centric window fed to the "
                        "actor's belief CNN.  0 -> disabled (fall back "
                        "to global belief_maps).")
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
    p.add_argument("--target-kl",      type=float, default=d["target_kl"])
    # LR schedule
    p.add_argument("--lr-schedule",    default=d.get("lr_schedule", "linear"),
                   choices=("constant", "linear"))
    p.add_argument("--lr-min-frac",    type=float, default=d.get("lr_min_frac", 0.1))
    # Best-ckpt tracking
    p.add_argument("--best-ckpt-metric",       default=d.get("best_ckpt_metric",
                                                              "mean_return"),
                   choices=("mean_return", "mean_caught"))
    p.add_argument("--best-ckpt-min-delta",    type=float,
                   default=d.get("best_ckpt_min_delta", 0.05))
    p.add_argument("--best-ckpt-min-episodes", type=int,
                   default=d.get("best_ckpt_min_episodes", 32))
    # Logging / eval
    p.add_argument("--log-interval",   type=int,   default=d["log_interval"])
    p.add_argument("--save-interval",  type=int,   default=d["save_interval"])
    p.add_argument("--no-eval",        action="store_true")
    p.add_argument("--red-policy-mix", default="stationary:1,random:1,run:1")
    # GNN
    p.add_argument("--d-hidden",       type=int, default=d["d_hidden"])
    p.add_argument("--n-msg-rounds",   type=int, default=d["n_msg_rounds"])
    p.add_argument("--share-hidden-via-gnn", action="store_true",
                   default=d.get("use_hidden_in_gnn", True))
    # Run management
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--run-name",       default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_heuristic_baseline(
    agent_factory, env_kwargs: Dict, red_policy,
    n_episodes: int, seed_base: int = 20_000,
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


def _to_device(np_arr, device):
    return torch.from_numpy(np_arr).float().to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir  = Path("runs") / "stage4" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    log_file = open(run_dir / "train_log.txt", "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Run dir: {run_dir}")
    log(f"Args: {vars(args)}")

    env_kwargs = dict(
        n_blue                  = args.n_blue,
        n_red                   = args.n_red,
        arena_size              = args.arena_size,
        max_steps               = args.max_steps,
        capture_radius          = args.capture_radius,
        sensor_radius           = args.sensor_radius,
        n_obstacles             = args.n_obstacles,
        obstacle_radius_min     = args.obstacle_radius_min,
        obstacle_radius_max     = args.obstacle_radius_max,
        obstacle_spawn_clearance= args.obstacle_spawn_clearance,
        use_belief_maps         = True,
        belief_grid_size        = args.belief_grid_size,
        belief_channels         = args.belief_channels,
        belief_clip             = args.belief_clip,
        p_TP                    = args.p_tp,
        p_FP                    = args.p_fp,
        ray_step_size           = args.ray_step_size,
        belief_window_size      = args.belief_window_size,
    )

    # Red policy mix parsing.
    red_policy_mix = []
    for chunk in args.red_policy_mix.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, w = chunk.split(":")
        red_policy_mix.append((name.strip(), float(w)))
    log(f"red_policy_mix: {red_policy_mix}")

    # Heuristic baselines (full-obs — Greedy still cheats past belief
    # noise; sensor-aware Greedy already handles sensor_radius but
    # doesn't yet know about obstacles; treated as a rough reference).
    if not args.no_eval:
        env_kwargs_baseline = {k: v for k, v in env_kwargs.items()
                                if k not in ("sensor_radius", "use_belief_maps",
                                             "belief_grid_size", "belief_channels",
                                             "belief_clip", "p_TP", "p_FP",
                                             "ray_step_size",
                                             "obstacle_spawn_clearance")}
        log("Computing heuristic baselines for reference "
            "(NOTE: baselines use sensor_radius=None; belief-map "
            "training runs in partial-obs env)...")
        for name, factory, red in [
            ("Greedy vs Run",        GreedyPursuer,        run_from_nearest_uav),
            ("Greedy vs Stationary", GreedyPursuer,        stationary_red),
        ]:
            r = evaluate_heuristic_baseline(
                factory, env_kwargs_baseline, red, n_episodes=10,
            )
            log(f"  {name:<22}: mean_return={r['mean_return']:+7.2f}  "
                f"caught={r['mean_caught']:.2f}/{args.n_red}  "
                f"steps={r['mean_steps']:.1f}")
            writer.add_scalar(f"baselines/{name.replace(' ', '_')}",
                              r["mean_return"], 0)

    # Vec env.
    vec_env = Stage4VectorPursuitEnv(
        n_envs             = args.n_envs,
        env_kwargs         = env_kwargs,
        base_seed          = args.seed,
        episode_buffer_size = 256,
        red_policy_mix     = red_policy_mix,
    )
    n_agents   = vec_env.n_agents
    action_dim = vec_env.action_dim

    # Policy.
    policy = GNNStage4Policy(
        n_blue                = vec_env.n_blue,
        blue_feat_dim         = vec_env.blue_feat_dim,
        edge_feat_dim         = vec_env.edge_feat_dim,
        action_dim            = action_dim,
        d_hidden              = args.d_hidden,
        n_msg_rounds          = args.n_msg_rounds,
        init_log_std          = STAGE4_DEFAULTS.get("init_log_std", 0.0),
        belief_channels       = args.belief_channels,
        belief_grid_size      = args.belief_grid_size,
        belief_encoder_out_dim= args.belief_encoder_out_dim,
        belief_window_size    = args.belief_window_size,
        use_hidden_in_gnn     = args.share_hidden_via_gnn,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    log(f"Policy: GNNStage4Policy d_hidden={args.d_hidden} "
        f"rounds={args.n_msg_rounds} params={n_params}")
    log("Critic COLD-STARTED (Stage 3 checkpoints not compatible with "
        "Stage 4 obs dict / critic input shape).")

    args_dict_saved = {**vars(args), "policy_type": "gnn_stage4"}

    # Buffer.
    buffer = Stage4RolloutBuffer(
        rollout_steps  = args.rollout_steps,
        n_envs         = args.n_envs,
        n_agents       = n_agents,
        action_dim     = action_dim,
        blue_feat_dim  = vec_env.blue_feat_dim,
        edge_feat_dim  = vec_env.edge_feat_dim,
        n_bb_edges     = vec_env.n_bb_edges,
        belief_channels= args.belief_channels,
        belief_grid    = args.belief_grid_size,
        belief_window  = args.belief_window_size,
        d_hidden       = args.d_hidden,
        device         = device,
    )

    log(f"\nStarting Stage 4 training: {args.n_rollouts} rollouts x "
        f"{args.rollout_steps} steps x {args.n_envs} envs x {n_agents} agents")

    target_kl_arg = args.target_kl if args.target_kl and args.target_kl > 0 else None

    obs_np = vec_env.reset(seed=args.seed)
    hidden = policy.initial_hidden(n_envs=args.n_envs, device=device)
    t_start = time.time()
    global_step = 0

    best_metric_val: float = float("-inf")
    best_ckpt_path = run_dir / "best.pt"

    def _current_lr_frac(r: int) -> float:
        if args.lr_schedule == "constant" or args.n_rollouts <= 1:
            return 1.0
        progress = r / max(args.n_rollouts - 1, 1)
        return 1.0 + (args.lr_min_frac - 1.0) * progress

    for rollout in range(args.n_rollouts):
        lr_frac = _current_lr_frac(rollout)
        cur_lr  = args.lr * lr_frac
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        buffer.reset()

        for step in range(args.rollout_steps):
            # Tensorise the current obs.
            partial_obs = {
                "blue_features":    _to_device(obs_np["blue_features"], device),
                "bb_edge_features": _to_device(obs_np["bb_edge_features"], device),
                "belief_maps":      _to_device(obs_np["belief_maps"], device),
            }
            if "belief_windows" in obs_np:
                partial_obs["belief_windows"] = _to_device(
                    obs_np["belief_windows"], device,
                )
            full_state = {
                "blue_features":    partial_obs["blue_features"],
                "bb_edge_features": partial_obs["bb_edge_features"],
                "true_occupancy":   _to_device(obs_np["true_occupancy"], device),
            }

            with torch.no_grad():
                action_t, log_p_t, _, value_t, new_hidden = \
                    policy.get_action_and_value(partial_obs, full_state, hidden)
            action_np = action_t.cpu().numpy().astype(np.float32)

            next_obs_np, reward_np, done_np, _ = vec_env.step(action_np)

            buffer.add(
                blue_features    = partial_obs["blue_features"],
                bb_edge_features = partial_obs["bb_edge_features"],
                belief_maps      = partial_obs["belief_maps"],
                true_occupancy   = full_state["true_occupancy"],
                actions          = action_t,
                log_probs        = log_p_t,
                values           = value_t,
                rewards          = torch.from_numpy(reward_np).to(device),
                dones            = torch.from_numpy(done_np).to(device),
                hidden           = hidden,
                belief_windows   = partial_obs.get("belief_windows", None),
            )

            done_t = torch.from_numpy(done_np).to(device).view(-1, 1, 1)
            hidden = new_hidden * (1.0 - done_t)

            obs_np = next_obs_np
            global_step += args.n_envs

        # Bootstrap V from post-rollout state.
        with torch.no_grad():
            last_full_state = {
                "blue_features":    _to_device(obs_np["blue_features"], device),
                "bb_edge_features": _to_device(obs_np["bb_edge_features"], device),
                "true_occupancy":   _to_device(obs_np["true_occupancy"], device),
            }
            last_value = policy.critic_forward(last_full_state)
        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        update_metrics = ppo_update_stage4(
            policy         = policy,
            optimizer      = optimizer,
            buffer         = buffer,
            clip_eps       = args.clip_eps,
            ent_coef       = args.ent_coef,
            vf_coef        = args.vf_coef,
            max_grad_norm  = args.max_grad_norm,
            n_epochs       = args.n_epochs,
            mb_size        = args.mb_size,
            value_clip     = STAGE4_DEFAULTS["value_clip"],
            normalize_adv  = STAGE4_DEFAULTS["normalize_adv"],
            target_kl      = target_kl_arg,
        )

        ep_stats = vec_env.recent_episode_stats()
        if (rollout + 1) % args.log_interval == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            lr_str = (f"  lr={cur_lr:.2e}"
                      if args.lr_schedule != "constant" else "")
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
                f"  bce_uav={update_metrics['per_uav_bce']:.3f}"
                f"  bce_fused={update_metrics['fused_bce']:.3f}"
                f"{lr_str}"
            )

        writer.add_scalar("rollout/mean_return", ep_stats["mean_return"], global_step)
        writer.add_scalar("rollout/mean_caught", ep_stats["mean_caught"], global_step)
        writer.add_scalar("rollout/mean_length", ep_stats["mean_length"], global_step)
        writer.add_scalar("ppo/policy_loss",  update_metrics["policy_loss"],  global_step)
        writer.add_scalar("ppo/value_loss",   update_metrics["value_loss"],   global_step)
        writer.add_scalar("ppo/entropy",      update_metrics["entropy"],      global_step)
        writer.add_scalar("ppo/approx_kl",    update_metrics["approx_kl"],    global_step)
        writer.add_scalar("ppo/clip_frac",    update_metrics["clip_frac"],    global_step)
        writer.add_scalar("ppo/n_epochs_run", update_metrics["n_epochs_run"], global_step)
        writer.add_scalar("ppo/per_uav_bce",  update_metrics["per_uav_bce"],  global_step)
        writer.add_scalar("ppo/fused_bce",    update_metrics["fused_bce"],    global_step)
        writer.add_scalar("ppo/lr",           cur_lr,                         global_step)

        # Best-ckpt tracking.
        n_completed = int(ep_stats.get("n_completed", 0))
        if n_completed >= args.best_ckpt_min_episodes:
            metric_val = float(ep_stats.get(args.best_ckpt_metric, 0.0))
            if metric_val >= best_metric_val + args.best_ckpt_min_delta:
                best_metric_val = metric_val
                torch.save({
                    "policy_state": policy.state_dict(),
                    "rollout":      rollout + 1,
                    "global_step":  global_step,
                    "args":         args_dict_saved,
                    "best_metric":  {
                        "name":  args.best_ckpt_metric,
                        "value": metric_val,
                        "n_completed": n_completed,
                    },
                }, best_ckpt_path)
                writer.add_scalar("rollout/best_metric", metric_val, global_step)

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
    log(f"\nStage 4 training done.  Elapsed: {elapsed/60:.1f} min")
    log(f"Final checkpoints under {run_dir}")
    log_file.close()
    writer.close()


if __name__ == "__main__":
    main()
