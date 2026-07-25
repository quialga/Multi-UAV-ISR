"""
scripts/eval_stage4_counts.py — entity-count generalisation eval.

Measures how a trained Stage 4 policy performs at red / obstacle counts
that may differ from what it trained on — the "train on 2-4, evaluate
zero-shot on 6 (or 10)" test that the variable-entities work unlocks.

Why a dedicated script (not just a flag on training): the number of
entities the model can process is a CONSTRUCTION-time capacity (it sizes
the graph's node/edge buffers), while the ACTIVE count is observed, not
told.  To evaluate at a count outside the trained capacity we rebuild
the policy at the eval capacity and reload the SAME weights — which
works because every learned tensor is count-agnostic (per-node/edge
MLPs + masked-mean pool; edge-index buffers are non-persistent and
rebuilt).  Deterministic eval only runs the ACTOR, so even a pre-pool
checkpoint (whose critic trunk wouldn't transfer) evaluates fine.

Run:
    # sweep red counts 2,4,6,8 on a policy trained at some capacity
    python scripts/eval_stage4_counts.py \
        --checkpoint runs/stage4/<run>/best.pt --n-red 2 4 6 8

    # also vary obstacle count, more episodes
    python scripts/eval_stage4_counts.py --checkpoint <ckpt> \
        --n-red 3 6 --n-obstacles 4 8 --n-episodes 40

    # team-size (n_blue) generalisation — count-agnostic too
    python scripts/eval_stage4_counts.py --checkpoint <ckpt> \
        --n-blue 3 5 8 --n-red 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> isr
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts   -> train_stage4

from isr.agents.gnn_stage4_policy import GNNStage4Policy
from isr.agents.heuristics import (
    run_from_nearest_uav, stationary_red, random_red,
)
from isr.configs.stage4_default import STAGE4_DEFAULTS
from train_stage4 import evaluate_policy_deterministic


def _env_kwargs_from_train_args(
    ta: Dict, n_blue: int, n_red: int, n_obstacles: int,
) -> Dict:
    """Rebuild the Stage 4 env config from a checkpoint's saved args,
    with the entity COUNTS (blues / reds / obstacles) overridden.
    Perception knobs (sensor, belief, crash penalties, moving obstacles)
    are carried over so the eval matches the training regime; `.get`
    defaults keep older checkpoints (missing newer knobs) working.
    Counts are FIXED at the override value (no *_min) so every episode
    has exactly that many."""
    g = ta.get
    return dict(
        n_blue                  = n_blue,
        n_red                   = n_red,
        arena_size              = ta["arena_size"],
        max_steps               = ta["max_steps"],
        capture_radius          = ta["capture_radius"],
        sensor_radius           = g("sensor_radius", 40.0),
        n_obstacles             = n_obstacles,
        obstacle_radius_min     = g("obstacle_radius_min", 5.0),
        obstacle_radius_max     = g("obstacle_radius_max", 15.0),
        obstacle_spawn_clearance= g("obstacle_spawn_clearance", 10.0),
        moving_obstacle_fraction= g("moving_obstacle_fraction", 0.0),
        obstacle_speed          = g("obstacle_speed", 0.0),
        obstacle_belief_decay   = g("obstacle_belief_decay", 1.0),
        use_belief_maps         = True,
        belief_grid_size        = g("belief_grid_size", 26),
        belief_channels         = g("belief_channels", 2),
        belief_clip             = g("belief_clip", 10.0),
        p_TP                    = g("p_tp", 0.85),
        p_FP                    = g("p_fp", 0.15),
        ray_step_size           = g("ray_step_size", 2.5),
        enemy_belief_decay      = g("enemy_belief_decay", 0.99),
        enemy_belief_diffusion  = g("enemy_belief_diffusion", 0.2),
        sensor_pos_noise_std    = g("sensor_pos_noise_std", 1.0),
        crash_obstacle_penalty  = g("crash_obstacle_penalty", 0.0),
        crash_blue_penalty      = g("crash_blue_penalty", 0.0),
        blue_collision_radius   = g("blue_collision_radius", 2.0),
    )


def _build_policy_at(
    ta: Dict, n_blue: int, n_red: int, n_obs: int, device,
) -> GNNStage4Policy:
    """Instantiate the policy at the EVAL capacity (blues / reds /
    obstacles) and soft-load the checkpoint weights (all count-agnostic
    -> the actor transfers in full; any non-matching critic tensor is
    named and left at init but is unused by the deterministic actor
    eval)."""
    pol = GNNStage4Policy(
        n_blue            = n_blue,
        n_red             = n_red,
        n_obs             = n_obs,
        d_hidden          = ta.get("d_hidden", 64),
        n_msg_rounds      = ta.get("n_msg_rounds", 2),
        init_log_std      = STAGE4_DEFAULTS.get("init_log_std", 0.0),
        use_hidden_in_gnn = ta.get("share_hidden_via_gnn", True),
    ).to(device)
    return pol


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Stage 4 checkpoint .pt (best.pt / final.pt)")
    p.add_argument("--n-blue", type=int, nargs="+", default=None,
                   help="blue team sizes to sweep (default: the trained "
                        "n_blue). The team size is count-agnostic too, so "
                        "the same weights evaluate on more/fewer UAVs.")
    p.add_argument("--n-red", type=int, nargs="+", default=None,
                   help="red counts to sweep (default: the trained n_red)")
    p.add_argument("--n-obstacles", type=int, nargs="+", default=None,
                   help="obstacle counts to sweep (default: the trained "
                        "n_obstacles)")
    p.add_argument("--n-episodes", type=int, default=20,
                   help="episodes per (count, red-policy) cell")
    p.add_argument("--seed-base", type=int, default=30_000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = args.checkpoint.resolve()
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ckpt["args"]

    blue_counts = args.n_blue      or [ta["n_blue"]]
    red_counts  = args.n_red       or [ta["n_red"]]
    obs_counts  = args.n_obstacles or [ta.get("n_obstacles", 0)]
    red_policies = [
        ("stationary", stationary_red),
        ("random",     random_red(seed=12_345)),
        ("run",        run_from_nearest_uav),
    ]

    print(f"Checkpoint: {ckpt_path}")
    print(f"  trained at n_blue={ta['n_blue']} n_red={ta['n_red']} "
          f"n_obstacles={ta.get('n_obstacles', 0)}  "
          f"(n_red_min={ta.get('n_red_min')}, "
          f"n_obstacles_min={ta.get('n_obstacles_min')})")
    print(f"  sweeping n_blue={blue_counts}  n_red={red_counts}  "
          f"n_obstacles={obs_counts}  "
          f"({args.n_episodes} episodes/cell, deterministic)\n")

    # Header: one column per red policy + a mean, reporting caught/N.
    col = "{:>22}"
    hdr = f"{'n_blue':>6} {'n_red':>6} {'n_obs':>6} | " + " ".join(
        col.format(f"caught/N vs {name}") for name, _ in red_policies
    ) + col.format("mean frac")
    # Pre-flight: report the weight transfer ONCE (it's identical for
    # every count -- the actor + pooled critic trunk are count-invariant),
    # then silence it in the sweep.  Deterministic eval runs only the
    # actor, so any skipped critic tensor is harmless.
    _notes: list = []
    _build_policy_at(ta, blue_counts[0], red_counts[0], obs_counts[0],
                     device).load_full_stage4(str(ckpt_path), log=_notes.append)
    if _notes:
        print("Weight transfer (deterministic eval uses the ACTOR only; "
              "critic tensors are unused):")
        for ln in _notes:
            print(" ", ln)
        print()
    _silent = lambda *_a: None

    print(hdr)
    print("-" * len(hdr))

    for n_blue in blue_counts:
        for n_obs in obs_counts:
            for n_red in red_counts:
                pol = _build_policy_at(ta, n_blue, n_red, n_obs, device)
                pol.load_full_stage4(str(ckpt_path), log=_silent)
                pol.eval()
                ek = _env_kwargs_from_train_args(ta, n_blue, n_red, n_obs)

                fracs = []
                cells = []
                for name, rp in red_policies:
                    m = evaluate_policy_deterministic(
                        pol, ek, rp, args.n_episodes, device,
                        seed_base=args.seed_base,
                    )
                    frac = m["mean_caught"] / max(n_red, 1)
                    fracs.append(frac)
                    cells.append(col.format(
                        f"{m['mean_caught']:.2f}/{n_red} ({frac:.2f})"))
                mean_frac = float(np.mean(fracs))
                print(f"{n_blue:>6} {n_red:>6} {n_obs:>6} | "
                      + " ".join(cells) + col.format(f"{mean_frac:.2f}"))

    print("\nfrac = mean reds caught / n_red (comparable across counts).")


if __name__ == "__main__":
    main()
