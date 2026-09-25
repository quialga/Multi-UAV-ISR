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
    run_from_nearest_uav, stationary_red, random_red,
)
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
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
    p.add_argument("--red-v-max", type=float, default=None,
                   help="Red top speed; None keeps RED_TARGET.v_max (1.0) "
                        "against the blue's 1.5, the setting every result "
                        "in docs/stage4_results.md was measured under.  At "
                        "1.0 a stern chase closes 0.5/step and covers 100 "
                        "units of a 130 m arena in an episode, so pure "
                        "pursuit wins without coordination -- measured, not "
                        "argued: the uncoordinated ObsGreedy scores 2.90/3. "
                        "Raise it to make cutting a target off the only way "
                        "to catch it.  CHANGES THE TASK: results are not "
                        "comparable across values.")
    # Stage 4 obstacle + belief knobs
    p.add_argument("--n-obstacles",           type=int,   default=d["n_obstacles"])
    p.add_argument("--n-red-min", type=int, default=d.get("n_red_min", None),
                   help="Lower bound for per-episode active red count "
                        "(capacity = --n-red). Unset => fixed at capacity. "
                        "Enables variable-count training + generalisation.")
    p.add_argument("--n-obstacles-min", type=int,
                   default=d.get("n_obstacles_min", None),
                   help="Lower bound for per-episode obstacle count "
                        "(capacity = --n-obstacles). Unset => fixed.")
    p.add_argument("--moving-obstacle-fraction", type=float,
                   default=d.get("moving_obstacle_fraction", 0.0),
                   help="Fraction (0..1) of obstacles that patrol back-"
                        "and-forth along one axis, bouncing off walls. "
                        "0 = all static.")
    p.add_argument("--obstacle-speed", type=float,
                   default=d.get("obstacle_speed", 0.0),
                   help="Patrol speed (m/step) for moving obstacles.")
    p.add_argument("--obstacle-belief-decay", type=float,
                   default=d.get("obstacle_belief_decay", 1.0),
                   help="Forgetting on the obstacle belief channel "
                        "(<1 fades a moving obstacle's stale trail; "
                        "1.0 = off, static behaviour).")
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
    # Phase A prediction step (enemy channel).
    p.add_argument("--enemy-belief-decay", type=float,
                   default=d.get("enemy_belief_decay", 1.0),
                   help="Forgetting factor gamma (1.0 = off).")
    p.add_argument("--enemy-belief-diffusion", type=float,
                   default=d.get("enemy_belief_diffusion", 0.0),
                   help="Motion-model spread p_move (0.0 = off).")
    p.add_argument("--sensor-pos-noise-std", type=float,
                   default=d.get("sensor_pos_noise_std", 1.0),
                   help="Live-sensor position accuracy (m) for visible "
                        "targets; continuous measurement replaces the "
                        "cell-centre peak on that blue's rb edge.")
    p.add_argument("--no-track-occlusion", dest="track_occlusion",
                   action="store_false",
                   default=d.get("track_occlusion", True),
                   help="Let live tracks see THROUGH obstacles (pre-fix "
                        "behaviour). By default a live track needs clear "
                        "line of sight, like the belief map.")
    p.add_argument("--no-track-detection", dest="track_detection",
                   action="store_false",
                   default=d.get("track_detection", True),
                   help="Make live-track detection perfect (pre-fix). By "
                        "default a live track requires the same p_TP draw "
                        "the belief map uses; on a miss the track coasts on "
                        "the memory path.")
    p.add_argument("--sensor-vel-noise-std", type=float,
                   default=d.get("sensor_vel_noise_std", 0.1),
                   help="Doppler measurement noise. Keep SMALL vs "
                        "--sensor-pos-noise-std (radar measures velocity "
                        "from phase, not by differencing positions).")
    p.add_argument("--track-conf-min", type=float,
                   default=d.get("track_conf_min", 0.5),
                   help="Live-track confidence at max sensor range (SNR "
                        "proxy; range-only so it never leaks true-vs-false). "
                        "1.0 = flat conf 1.0 everywhere (pre-fix).")
    p.add_argument("--sensor-noise-range-growth", type=float,
                   default=d.get("sensor_noise_range_growth", 1.0),
                   help="Range growth of measurement noise: sigma(r) = "
                        "sigma_base * (1 + g (r/R)^2). The accuracy half of "
                        "the same SNR falloff as --track-conf-min. "
                        "0 = range-independent noise (pre-fix).")
    # Tracker path (docs/tracker_observation.md)
    p.add_argument("--clutter-rate", type=float,
                   default=d["clutter_rate"],
                   help="False plots per blue per scan in raw_detections() "
                        "(tracker path only; the belief path models false "
                        "alarms per cell with --p-fp).")
    p.add_argument("--obstacle-radius-noise-std", type=float,
                   default=d["obstacle_radius_noise_std"],
                   help="Obstacle radius measurement noise (m, range-scaled) "
                        "in raw_obstacle_detections() (tracker path only).")
    p.add_argument("--use-staleness", action="store_true",
                   help="maintain the staleness field and give the ACTOR "
                        "R*R region nodes [staleness, searchable] -- the "
                        "where-have-we-not-looked signal NEITHER observation "
                        "carries today (docs/stage4_results.md 9.3: the "
                        "belief map's own fallback peak lands on cells being "
                        "swept right now, so it is clutter, not search).")
    p.add_argument("--staleness-regions", type=int, default=5,
                   help="R: the staleness field is aggregated to R x R "
                        "region nodes.  At sensor_radius 40 in a 130 m "
                        "arena, R=5 gives 26 m regions a blue sweeps from "
                        "their centre.")
    p.add_argument("--actor-obs", choices=("belief", "tracker"),
                   default=d["actor_obs"],
                   help="Source of the ACTOR's red and obstacle nodes: the "
                        "belief map (control) or the red/obstacle trackers. "
                        "The critic's ground-truth graph is the same in both.")
    p.add_argument("--red-motion-ckpt", type=str,
                   default=d["red_motion_ckpt"],
                   help="Learned red motion model for the red tracker "
                        "(--actor-obs tracker); omit for constant velocity.")
    p.add_argument("--tracker-red-slots", type=int,
                   default=d["tracker_red_slots"])
    p.add_argument("--tracker-obstacle-slots", type=int,
                   default=d["tracker_obstacle_slots"])
    p.add_argument("--tracker-sigma-cutoff", type=float,
                   default=d["tracker_sigma_cutoff"],
                   help="Red tracks with position sd above this (m) are "
                        "not shown to the actor.")
    p.add_argument("--eval-interval", type=int, default=25,
                   help="Every N rollouts, DETERMINISTICALLY evaluate the "
                        "learned policy (the metric comparable to Stage 3; "
                        "the training caught stat is stochastic and lower). "
                        "0 disables.")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per red policy in the deterministic eval.")
    p.add_argument("--warm-start-critic",
                   default=d.get("warm_start_critic", None),
                   help="Path to a Stage 1/2 GNN checkpoint to warm-start "
                        "the critic (Stage 3's stabiliser).  'none' to "
                        "cold-start.")
    p.add_argument("--reset-log-std", type=float, default=None,
                   help="After warm-starting, overwrite the actor's "
                        "log_std with this value (sigma = exp(x)).  Applies "
                        "AFTER the copy, which is the only place it works: "
                        "load_full_stage4 copies every shape-matching "
                        "tensor, actor_log_std included, so a checkpoint's "
                        "own sigma otherwise wins over anything the "
                        "constructor set.  Needed because PPO SAMPLES while "
                        "eval takes the mean: a behaviour-cloned actor "
                        "arrives with log_std still at its init 0.0 "
                        "(sigma 1.0, near-uniform on a [-1,1]^2 box), so its "
                        "rollouts perform far below the policy itself -- "
                        "measured 1.70/3 sampled against 2.52/3 "
                        "deterministic -- and PPO would compute advantages "
                        "on wrecked trajectories.  docs/stage4_results.md "
                        "6.5 records entropy settling around sigma 0.17-0.2 "
                        "on this task, so -1.2 (sigma 0.30) is a sane "
                        "fine-tuning width.")
    p.add_argument("--warm-start-full",
                   default=None,
                   help="Path to a converged Stage 4 checkpoint (e.g. "
                        "runs/stage4/obstacles_v1/best.pt) to warm-start "
                        "BOTH actor and critic.  The crash-avoidance run "
                        "starts here so the policy already flies + catches; "
                        "training only has to LEARN to avoid crashes.  "
                        "Overrides --warm-start-critic when set.")
    # ----- Crash avoidance ------------------------------------------------
    p.add_argument("--crash-obstacle-penalty", type=float,
                   default=d["crash_obstacle_penalty"],
                   help="Per-agent per-step penalty for crashing into an "
                        "obstacle (soft-stop, no termination). 0 = off.")
    p.add_argument("--crash-blue-penalty", type=float,
                   default=d["crash_blue_penalty"],
                   help="Per-agent per-step penalty for a blue-blue "
                        "collision. 0 = off.")
    p.add_argument("--blue-collision-radius", type=float,
                   default=d["blue_collision_radius"],
                   help="Distance (m) below which two blues count as "
                        "collided.")
    p.add_argument("--clearance-weight", type=float,
                   default=d.get("clearance_weight", 0.0),
                   help="Dense per-step clearance/barrier shaping magnitude "
                        "(backlog §16): penalises being within "
                        "--clearance-margin of an obstacle surface, growing "
                        "inward so its gradient points toward open space. "
                        "0 = off.")
    p.add_argument("--clearance-margin", type=float,
                   default=d.get("clearance_margin", 4.0),
                   help="Metres beyond an obstacle surface where clearance "
                        "shaping acts.")
    p.add_argument("--clearance-ally-weight", type=float,
                   default=d.get("clearance_ally_weight", 0.0),
                   help="Blue<->blue barrier magnitude (same falloff, "
                        "'surface' = blue_collision_radius). Needed with "
                        "--clearance-weight or crashes just migrate "
                        "obstacle->ally. 0 = off.")
    p.add_argument("--clearance-ally-margin", type=float,
                   default=d.get("clearance_ally_margin", 3.0),
                   help="Metres beyond the collision radius where the ally "
                        "barrier acts (tighter than obstacles so it doesn't "
                        "fight converge-on-a-red coordination).")
    # Parallelism
    p.add_argument("--n-workers", type=int, default=0,
                   help="Env-stepping worker processes (0 = in-process, "
                        "the previous behaviour).  Shards --n-envs across "
                        "subprocesses; combine with a larger --n-envs to "
                        "use all CPU cores.")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads for the trainer process "
                        "(0 = torch default).  With --n-workers > 0, "
                        "capping this (e.g. cores - n_workers) avoids "
                        "thread oversubscription during rollouts.")
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
    p.add_argument("--aux-hidden-coef", type=float,
                   default=d.get("aux_hidden_coef", 0.0),
                   help="MSE(actor_h_blue, critic_h_blue.detach()) "
                        "coefficient.  0 disables.  Stage 3 opt-C used 0.1.")
    # LR schedule
    p.add_argument("--lr-schedule",    default=d.get("lr_schedule", "linear"),
                   choices=("constant", "linear"))
    p.add_argument("--lr-min-frac",    type=float, default=d.get("lr_min_frac", 0.1))
    # Best-ckpt tracking
    p.add_argument("--best-ckpt-metric",       default=d.get("best_ckpt_metric",
                                                              "mean_return"),
                   choices=("mean_return", "mean_caught",
                            "det_caught", "det_composite"),
                   help="Which metric selects best.pt.  'mean_return' / "
                        "'mean_caught' are the STOCHASTIC training stats "
                        "(mean_return mis-selects on crash-penalty runs — "
                        "caution lowers return, so it peaks at ~rollout 1).  "
                        "'det_caught' / 'det_composite' use the DETERMINISTIC "
                        "eval (robust; require --eval-interval > 0).  "
                        "det_composite = det_caught − λ·(obstacle+ally crash "
                        "events) — the crash-aware selector for penalty runs.")
    p.add_argument("--best-ckpt-crash-lambda", type=float,
                   default=d.get("best_ckpt_crash_lambda", 0.5),
                   help="λ for the 'det_composite' metric: how many caught "
                        "reds one crash EVENT is worth trading.  Only used "
                        "when --best-ckpt-metric det_composite.")
    p.add_argument("--best-ckpt-min-delta",    type=float,
                   default=d.get("best_ckpt_min_delta", 0.05))
    p.add_argument("--best-ckpt-min-episodes", type=int,
                   default=d.get("best_ckpt_min_episodes", 32))
    # Logging / eval
    p.add_argument("--log-interval",   type=int,   default=d["log_interval"])
    p.add_argument("--save-interval",  type=int,   default=d["save_interval"])
    p.add_argument("--no-eval",        action="store_true")
    p.add_argument("--red-policy-mix", default="stationary:1,random:1,run:1",
                   help="weighted mix the red policy is sampled from PER "
                        "EPISODE.  Training against a single red collapses "
                        "out of distribution -- see docs/red_policy_mixing.md "
                        "for the measurement that motivated this.")
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


def saved_args(args) -> dict:
    """The ``args`` dict a checkpoint carries.

    Reward shape is config-only (not argparse) but must be recorded, so a
    checkpoint pins the reward it was trained under.  Sister scripts
    (``train_bc.py``, ``compare_observation_quality.py``) build env configs
    through this same function: building them from the parser alone lets
    ``env_kwargs_from_checkpoint`` fall through to its PRE-feature defaults
    — ``step_cost`` 0.05 instead of the 0.033 this config actually uses —
    and silently evaluate in a different reward regime from training.
    """
    return {
        **vars(args), "policy_type": "gnn_stage4_v6",
        "catch_reward":     STAGE4_DEFAULTS["catch_reward"],
        "step_cost":        STAGE4_DEFAULTS["step_cost"],
        "uncaught_penalty": STAGE4_DEFAULTS["uncaught_penalty"],
        "action_cost_coef": STAGE4_DEFAULTS["action_cost_coef"],
        "p_TP_obstacle":    STAGE4_DEFAULTS["p_TP_obstacle"],
        "p_FP_obstacle":    STAGE4_DEFAULTS["p_FP_obstacle"],
    }


def saved_args_from_flags(flags: str) -> dict:
    """``saved_args`` for a command line given as a string, so another
    script describes an env the way a training run is launched."""
    import shlex
    argv_backup = sys.argv
    try:
        sys.argv = ["train_stage4.py"] + shlex.split(flags)
        return saved_args(_parse_args())
    finally:
        sys.argv = argv_backup


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
        caught.append(int(snap.get("n_red_start", len(snap["red_active"]))
                          - int(snap["red_active"].sum())))
        steps.append(int(snap["t"]))
    return {
        "mean_return": float(np.mean(returns)),
        "mean_caught": float(np.mean(caught)),
        "mean_steps":  float(np.mean(steps)),
    }


def _to_device(np_arr, device):
    return torch.from_numpy(np_arr).float().to(device)


@torch.no_grad()
def evaluate_policy_deterministic(
    policy, env_kwargs: Dict, red_policy, n_episodes: int, device,
    seed_base: int = 30_000,
) -> Dict[str, float]:
    """
    Roll the LEARNED policy DETERMINISTICALLY (action = distribution
    mean, no exploration noise) — the metric comparable to Stage 3's
    eval.  The training-loop ``caught`` stat is stochastic and always
    lower; this is the number to judge convergence by.
    """
    returns, caught, steps = [], [], []
    obs_crashes, ally_crashes = [], []
    for ep in range(n_episodes):
        env = PursuitEnv(**env_kwargs, red_policy=red_policy,
                         seed=seed_base + ep)
        env.reset(seed=seed_base + ep)
        hidden = policy.initial_hidden(1, device)
        total = 0.0
        # DISTINCT crash EVENTS (rising edges), not crash-steps: a blue
        # that sits inside an obstacle for many steps counts ONCE, until
        # it leaves and re-enters.  Counted per blue via mask diffs.
        ep_obs_crash = 0
        ep_ally_crash = 0
        prev_obs_mask = np.zeros(len(env.possible_agents), dtype=bool)
        prev_ally_mask = np.zeros(len(env.possible_agents), dtype=bool)
        agents = env.possible_agents
        while env.agents:
            obs = env.structured_belief_observation()
            obs_t = {k: _to_device(v, device).unsqueeze(0)
                     for k, v in obs.items()}
            partial_obs, _ = split_stage4_obs(obs_t)
            mean, hidden = policy.act_deterministic(partial_obs, hidden)
            a_np = mean.squeeze(0).cpu().numpy().astype(np.float32)  # (n_blue, 2)
            actions = {agents[i]: a_np[i] for i in range(len(agents))}
            _, rew_d, _, _, _ = env.step(actions)
            total += rew_d[agents[0]]
            o_mask = env._last_obstacle_crash_mask
            a_mask = env._last_blue_crash_mask
            ep_obs_crash += int(np.count_nonzero(o_mask & ~prev_obs_mask))
            ep_ally_crash += int(np.count_nonzero(a_mask & ~prev_ally_mask))
            prev_obs_mask, prev_ally_mask = o_mask, a_mask
        snap = env.state_snapshot()
        returns.append(total)
        caught.append(int(snap.get("n_red_start", len(snap["red_active"]))
                          - int(snap["red_active"].sum())))
        steps.append(int(snap["t"]))
        obs_crashes.append(ep_obs_crash)
        ally_crashes.append(ep_ally_crash)
    return {
        "mean_return":           float(np.mean(returns)),
        "mean_caught":           float(np.mean(caught)),
        "mean_steps":            float(np.mean(steps)),
        "mean_obstacle_crashes": float(np.mean(obs_crashes)),
        "mean_blue_crashes":     float(np.mean(ally_crashes)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
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
        n_red_min               = args.n_red_min,
        n_obstacles_min         = args.n_obstacles_min,
        moving_obstacle_fraction= args.moving_obstacle_fraction,
        obstacle_speed          = args.obstacle_speed,
        obstacle_belief_decay   = args.obstacle_belief_decay,
        obstacle_radius_min     = args.obstacle_radius_min,
        obstacle_radius_max     = args.obstacle_radius_max,
        obstacle_spawn_clearance= args.obstacle_spawn_clearance,
        # The belief map is only computed when it is the actor's source.
        use_belief_maps         = args.actor_obs == "belief",
        belief_grid_size        = args.belief_grid_size,
        belief_channels         = args.belief_channels,
        belief_clip             = args.belief_clip,
        p_TP                    = args.p_tp,
        p_FP                    = args.p_fp,
        ray_step_size           = args.ray_step_size,
        enemy_belief_decay      = args.enemy_belief_decay,
        enemy_belief_diffusion  = args.enemy_belief_diffusion,
        sensor_pos_noise_std    = args.sensor_pos_noise_std,
        # Per-channel sensor quality — config-only, like the reward shape
        # (it defines the task).  Enemy inherits p_TP / p_FP above.
        p_TP_obstacle           = STAGE4_DEFAULTS["p_TP_obstacle"],
        p_FP_obstacle           = STAGE4_DEFAULTS["p_FP_obstacle"],
        track_occlusion         = args.track_occlusion,
        track_detection         = args.track_detection,
        sensor_vel_noise_std    = args.sensor_vel_noise_std,
        track_conf_min          = args.track_conf_min,
        sensor_noise_range_growth = args.sensor_noise_range_growth,
        clutter_rate            = args.clutter_rate,
        obstacle_radius_noise_std = args.obstacle_radius_noise_std,
        actor_obs               = args.actor_obs,
        red_motion_ckpt         = args.red_motion_ckpt,
        tracker_red_slots       = args.tracker_red_slots,
        tracker_obstacle_slots  = args.tracker_obstacle_slots,
        tracker_sigma_cutoff    = args.tracker_sigma_cutoff,
        # Reward shape — config-only (no CLI flags: these define the task,
        # they are not per-run knobs).  Recorded in the saved args below so
        # a checkpoint still pins the reward it was trained under.
        catch_reward            = STAGE4_DEFAULTS["catch_reward"],
        step_cost               = STAGE4_DEFAULTS["step_cost"],
        uncaught_penalty        = STAGE4_DEFAULTS["uncaught_penalty"],
        action_cost_coef        = STAGE4_DEFAULTS["action_cost_coef"],
        crash_obstacle_penalty  = args.crash_obstacle_penalty,
        crash_blue_penalty      = args.crash_blue_penalty,
        blue_collision_radius   = args.blue_collision_radius,
        clearance_weight        = args.clearance_weight,
        clearance_margin        = args.clearance_margin,
        clearance_ally_weight   = args.clearance_ally_weight,
        clearance_ally_margin   = args.clearance_ally_margin,
        use_staleness           = args.use_staleness,
        staleness_regions       = args.staleness_regions,
        red_v_max               = args.red_v_max,
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
                                             "obstacle_spawn_clearance",
                                             "actor_obs", "red_motion_ckpt",
                                             "tracker_red_slots",
                                             "tracker_obstacle_slots",
                                             "tracker_sigma_cutoff")}
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

    # Vec env — in-process by default; --n-workers > 0 shards the env
    # pool across subprocesses (same semantics, parallel stepping).
    if args.n_workers > 0:
        from isr.train.subproc_vec_env import SubprocStage4VecEnv
        vec_env = SubprocStage4VecEnv(
            n_envs              = args.n_envs,
            env_kwargs          = env_kwargs,
            base_seed           = args.seed,
            episode_buffer_size = 256,
            red_policy_mix      = red_policy_mix,
            n_workers           = args.n_workers,
        )
        log(f"Vec env: {args.n_envs} envs across {vec_env.n_workers} "
            f"worker processes.")
    else:
        vec_env = Stage4VectorPursuitEnv(
            n_envs             = args.n_envs,
            env_kwargs         = env_kwargs,
            base_seed          = args.seed,
            episode_buffer_size = 256,
            red_policy_mix     = red_policy_mix,
        )
    n_agents   = vec_env.n_agents
    action_dim = vec_env.action_dim

    # Policy (v6: typed GNN + obstacle nodes, no CNN).
    policy = GNNStage4Policy(
        n_blue            = vec_env.n_blue,
        n_red             = vec_env.n_red,
        n_obs             = vec_env.n_obstacles,
        blue_feat_dim     = vec_env.blue_feat_dim,
        red_feat_dim      = 4,   # [conf, Sxx, Syy, Sxy]  (velocity cov)
        obs_feat_dim      = 5,   # [placed/conf, radius/L, Sxx, Syy, Sxy]
        # The actor's graph sizing comes from the env: identical to the
        # critic's in belief mode, fixed slots and wider uncertainty
        # features in tracker mode.
        actor_n_red        = vec_env.actor_n_red,
        actor_n_obs        = vec_env.actor_n_obstacles,
        actor_red_feat_dim = vec_env.actor_red_feat_dim,
        actor_obs_feat_dim = vec_env.actor_obs_feat_dim,
        edge_feat_dim     = vec_env.edge_feat_dim,
        action_dim        = action_dim,
        d_hidden          = args.d_hidden,
        n_msg_rounds      = args.n_msg_rounds,
        init_log_std      = STAGE4_DEFAULTS.get("init_log_std", 0.0),
        use_hidden_in_gnn = args.share_hidden_via_gnn,
        # Coverage path, actor only; 0 leaves the policy as it was.
        n_region          = (args.staleness_regions ** 2
                             if args.use_staleness else 0),
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    log(f"Policy: GNNStage4Policy (v6) d_hidden={args.d_hidden} "
        f"rounds={args.n_msg_rounds} n_obs={vec_env.n_obstacles} "
        f"params={n_params}")

    # Warm-start.  --warm-start-full (a converged Stage 4 ckpt) copies
    # BOTH actor and critic and takes precedence; otherwise fall back to
    # the critic-only Stage 1/2 warm-start (Stage 3's stabiliser).
    wf = args.warm_start_full
    ws = args.warm_start_critic
    if wf and str(wf).lower() != "none" and Path(wf).exists():
        n_copied = policy.load_full_stage4(wf, log=log)
        log(f"Actor+Critic WARM-STARTED (full) from {wf} "
            f"({n_copied} tensors copied).")
    elif wf and str(wf).lower() != "none":
        log(f"WARN: --warm-start-full ckpt not found ({wf}); "
            f"falling back to critic-only warm-start.")
        wf = None
    if not (wf and str(wf).lower() != "none"):
        if ws and str(ws).lower() != "none" and Path(ws).exists():
            n_copied = policy.load_pretrained_critic(ws)
            log(f"Critic WARM-STARTED from {ws} ({n_copied} tensors copied).")
        else:
            if ws and str(ws).lower() != "none":
                log(f"WARN: warm-start ckpt not found ({ws}); cold-starting.")
            log("Critic COLD-STARTED (typed GNN, no CNN).")

    # Exploration width, AFTER the warm-start copy (which would otherwise
    # overwrite it with the checkpoint's own).  See --reset-log-std.
    if args.reset_log_std is not None:
        with torch.no_grad():
            was = policy.actor_log_std.detach().clone()
            policy.actor_log_std.fill_(float(args.reset_log_std))
        log(f"actor log_std reset {was.tolist()} -> "
            f"{policy.actor_log_std.detach().tolist()}  "
            f"(sigma {float(np.exp(args.reset_log_std)):.3f})")

    # Build the optimizer AFTER warm-start so Adam's moment buffers are
    # fresh for the loaded weights.
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    args_dict_saved = saved_args(args)

    # Buffer (generic dict-of-tensors).
    buffer = Stage4RolloutBuffer(
        rollout_steps = args.rollout_steps,
        n_envs        = args.n_envs,
        n_agents      = n_agents,
        action_dim    = action_dim,
        d_hidden      = args.d_hidden,
        device        = device,
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
    # Det-based selectors need the deterministic eval to actually run.
    det_based_metric = args.best_ckpt_metric in ("det_caught", "det_composite")
    if det_based_metric and args.eval_interval <= 0:
        raise SystemExit(
            f"--best-ckpt-metric {args.best_ckpt_metric} needs "
            f"--eval-interval > 0 (got {args.eval_interval}); best.pt would "
            f"never be written.  Use a stochastic metric or enable eval.")

    def _maybe_save_best(metric_val: float, rollout: int, extra: dict) -> None:
        """Save best.pt iff metric_val beats the running best by >=
        min_delta.  One tracker / one file, whatever the metric — a run
        uses a single metric, so there is no cross-metric contamination."""
        nonlocal best_metric_val
        if metric_val >= best_metric_val + args.best_ckpt_min_delta:
            best_metric_val = metric_val
            torch.save({
                "policy_state": policy.state_dict(),
                "rollout":      rollout + 1,
                "global_step":  global_step,
                "args":         args_dict_saved,
                "best_metric":  {"name": args.best_ckpt_metric,
                                 "value": metric_val, **extra},
            }, best_ckpt_path)
            writer.add_scalar("rollout/best_metric", metric_val, global_step)

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
            # Tensorise the whole v6 graph obs, then split into
            # actor / critic dicts.
            obs_t = {k: _to_device(v, device) for k, v in obs_np.items()}
            partial_obs, full_state = split_stage4_obs(obs_t)

            with torch.no_grad():
                (action_t, log_p_t, _, value_t, new_hidden,
                 _actor_hb, _critic_hb) = \
                    policy.get_action_and_value(partial_obs, full_state, hidden)
            action_np = action_t.cpu().numpy().astype(np.float32)

            next_obs_np, reward_np, done_np, _ = vec_env.step(action_np)

            buffer.add(
                obs       = obs_t,
                actions   = action_t,
                log_probs = log_p_t,
                values    = value_t,
                rewards   = torch.from_numpy(reward_np).to(device),
                dones     = torch.from_numpy(done_np).to(device),
                hidden    = hidden,
            )

            done_t = torch.from_numpy(done_np).to(device).view(-1, 1, 1)
            hidden = new_hidden * (1.0 - done_t)

            obs_np = next_obs_np
            global_step += args.n_envs

        # Bootstrap V from post-rollout state.
        with torch.no_grad():
            last_obs_t = {k: _to_device(v, device) for k, v in obs_np.items()}
            _, last_full_state = split_stage4_obs(last_obs_t)
            last_value, _ = policy.critic_forward(last_full_state)
        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        update_metrics = ppo_update_stage4(
            policy          = policy,
            optimizer       = optimizer,
            buffer          = buffer,
            clip_eps        = args.clip_eps,
            ent_coef        = args.ent_coef,
            vf_coef         = args.vf_coef,
            max_grad_norm   = args.max_grad_norm,
            n_epochs        = args.n_epochs,
            mb_size         = args.mb_size,
            value_clip      = STAGE4_DEFAULTS["value_clip"],
            normalize_adv   = STAGE4_DEFAULTS["normalize_adv"],
            target_kl       = target_kl_arg,
            aux_hidden_coef = args.aux_hidden_coef,
        )

        ep_stats = vec_env.recent_episode_stats()

        # Tracking diagnostic, from whichever source the actor reads.
        #   belief mode : mean belief-peak-to-true-red distance (m);
        #   tracker mode: the same distance for the tracks the actor is
        #     shown, plus NEES (honest uncertainty; 4.0 = calibrated for a
        #     4-D state) and NIS over the rollout.
        def _mean_of(vals):
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        nees = nis = float("nan")
        if args.actor_obs == "tracker":
            diags = [d for d in vec_env.tracker_diagnostics() if d]
            track_err = _mean_of([d["track_error_m"] for d in diags])
            nees = _mean_of([d["nees"] for d in diags])
            nis = _mean_of([d["nis"] for d in diags])
        else:
            track_err = _mean_of(vec_env.belief_track_errors())

        if (rollout + 1) % args.log_interval == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            lr_str = (f"  lr={cur_lr:.2e}"
                      if args.lr_schedule != "constant" else "")
            aux_str = (f"  aux={update_metrics['aux_hidden_loss']:.3f}"
                       if args.aux_hidden_coef > 0 else "")
            crash_on = (args.crash_obstacle_penalty > 0
                        or args.crash_blue_penalty > 0)
            crash_str = (
                f"  crash(o={ep_stats['mean_obstacle_crashes']:.2f}"
                f"/a={ep_stats['mean_blue_crashes']:.2f})"
                if crash_on else "")
            cons_str = (f"  nees={nees:4.1f}  nis={nis:4.1f}"
                        if args.actor_obs == "tracker" else "")
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
                f"eps={update_metrics['n_epochs_run']}  "
                f"trk={track_err:5.1f}m{cons_str}"
                f"{crash_str}{aux_str}{lr_str}"
            )

        writer.add_scalar("rollout/mean_return", ep_stats["mean_return"], global_step)
        writer.add_scalar("rollout/mean_caught", ep_stats["mean_caught"], global_step)
        writer.add_scalar("rollout/mean_length", ep_stats["mean_length"], global_step)
        writer.add_scalar("crash/obstacle_rate",
                          ep_stats["mean_obstacle_crashes"], global_step)
        writer.add_scalar("crash/ally_rate",
                          ep_stats["mean_blue_crashes"], global_step)
        writer.add_scalar("ppo/policy_loss",  update_metrics["policy_loss"],  global_step)
        writer.add_scalar("ppo/value_loss",   update_metrics["value_loss"],   global_step)
        writer.add_scalar("ppo/entropy",      update_metrics["entropy"],      global_step)
        writer.add_scalar("ppo/approx_kl",    update_metrics["approx_kl"],    global_step)
        writer.add_scalar("ppo/clip_frac",    update_metrics["clip_frac"],    global_step)
        writer.add_scalar("ppo/n_epochs_run", update_metrics["n_epochs_run"], global_step)
        writer.add_scalar("ppo/lr",           cur_lr,                         global_step)
        if args.aux_hidden_coef > 0:
            writer.add_scalar("ppo/aux_hidden_loss",
                              update_metrics["aux_hidden_loss"], global_step)
        if not np.isnan(track_err):
            tag = "tracker" if args.actor_obs == "tracker" else "belief"
            writer.add_scalar(f"{tag}/track_error_m", track_err, global_step)
        if not np.isnan(nees):
            writer.add_scalar("tracker/nees", nees, global_step)
        if not np.isnan(nis):
            writer.add_scalar("tracker/nis", nis, global_step)

        # Deterministic evaluation of the LEARNED policy — the metric
        # comparable to Stage 3 (the training `caught` above is
        # stochastic and always lower).
        if args.eval_interval > 0 and (rollout + 1) % args.eval_interval == 0:
            det = {}
            det_ocrash, det_acrash = [], []
            for name, red in (("stationary", stationary_red),
                              ("random", random_red(seed=12345)),
                              ("run", run_from_nearest_uav)):
                r = evaluate_policy_deterministic(
                    policy, env_kwargs, red, args.eval_episodes, device,
                )
                det[name] = r["mean_caught"]
                det_ocrash.append(r["mean_obstacle_crashes"])
                det_acrash.append(r["mean_blue_crashes"])
                writer.add_scalar(f"eval_det/caught_{name}",
                                  r["mean_caught"], global_step)
            det_mean = float(np.mean(list(det.values())))
            writer.add_scalar("eval_det/caught_mean", det_mean, global_step)
            # Distinct crash EVENTS (deterministic, per episode, averaged
            # over the three red policies) — a blue camping in an obstacle
            # counts ONCE, unlike the per-step-summed training-loop stat.
            # The clean crash metric the stochastic training stat can't
            # give.  Only surfaced when crash penalties are on.
            crash_on = (args.crash_obstacle_penalty > 0
                        or args.crash_blue_penalty > 0)
            det_ocrash_m = float(np.mean(det_ocrash))
            det_acrash_m = float(np.mean(det_acrash))
            writer.add_scalar("eval_det/obstacle_crashes", det_ocrash_m,
                              global_step)
            writer.add_scalar("eval_det/ally_crashes", det_acrash_m,
                              global_step)
            crash_str = (f"  crash(o={det_ocrash_m:.2f}/a={det_acrash_m:.2f})"
                         if crash_on else "")
            log(
                f"    [det eval @ {rollout+1}]  "
                f"caught  stat={det['stationary']:.2f}  "
                f"rand={det['random']:.2f}  run={det['run']:.2f}  "
                f"mean={det_mean:.2f}/{args.n_red}{crash_str}"
            )

            # Best-ckpt tracking on DETERMINISTIC metrics — only updatable
            # here, where the det eval exists.  det_composite trades caught
            # against crash EVENTS so caution can't tank the selector the
            # way mean_return does.
            if det_based_metric:
                if args.best_ckpt_metric == "det_caught":
                    sel = det_mean
                else:  # det_composite
                    sel = det_mean - args.best_ckpt_crash_lambda * (
                        det_ocrash_m + det_acrash_m)
                _maybe_save_best(sel, rollout, {
                    "det_caught":           det_mean,
                    "det_obstacle_crashes": det_ocrash_m,
                    "det_ally_crashes":     det_acrash_m,
                })

        # Best-ckpt tracking on STOCHASTIC metrics (det metrics are handled
        # at eval time above).
        if not det_based_metric:
            n_completed = int(ep_stats.get("n_completed", 0))
            if n_completed >= args.best_ckpt_min_episodes:
                metric_val = float(ep_stats.get(args.best_ckpt_metric, 0.0))
                _maybe_save_best(metric_val, rollout,
                                 {"n_completed": n_completed})

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
    vec_env.close()
    log_file.close()
    writer.close()


if __name__ == "__main__":
    main()
