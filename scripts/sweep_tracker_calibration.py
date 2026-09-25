"""
scripts/sweep_tracker_calibration.py — can the red tracker be calibrated
at a faster red?  Measured answer: no, and that is the useful part.

**What NEES is for.**  NEES weights the estimation error by the
covariance the filter claims: for a 4-D state it averages **4.0** when the
filter is honest about its own uncertainty.  Above 4 it is
over-confident — the real error is larger than the covariance admits —
and the actor is handed a confidence nobody earned.

**Why this exists.**  `docs/stage4_results.md §13.2` blamed `nees` 8-9 at
red 1.4 on `vel_prior_std` not having been raised with `--red-v-max`, and
`stage4_backlog.md §21.3` had asked for exactly that.  Both were wrong,
and this script is what showed it.

**Two knobs, measured, neither one fixes it:**

* ``vel_prior_std`` is the BIRTH prior — how fast a newly detected track
  might be moving.  It washes out within a few updates, once the filter
  reaches the steady state its process noise dictates.  Moving it
  1.0 -> 2.8 changed NEES by ~1 out of ~24 of error.
* ``a_max`` sets the process noise (``sigma_a = a_max * sqrt(2)``, and it
  is used for nothing else — `tracker.py`), so it IS the steady-state
  lever.  Swept 1.0 -> 6.0, i.e. sigma_a 1.41 -> 8.49: NEES bottoms out
  at **11.6** and rises in both directions, while track error goes
  3.8 -> 15.0 m and captures fall 2.92 -> 2.56.

**Why inflating Q cannot work here.**  ``sigma_a`` admits more *white*
acceleration.  The error against ``run_from_nearest_uav`` is *systematic*:
the evader turns away from the nearest blue, consistently, and a
constant-velocity model predicts "straight on" every time.  Isotropic
noise does not correct a directional bias — it only blurs the track,
which is what the rising track error shows.

**So the fix is a better motion model, not a better-tuned one** —
`stage4_backlog.md §15`, the trained red-motion adapter that is built and
has never been switched on.  The practical conclusion for now is to change
nothing: the shipped ``a_max=1.0`` is already the best operating point for
captures.

Run:
    python scripts/sweep_tracker_calibration.py         --checkpoint runs/stage4/ppo_red14_v1/best.pt
    python scripts/sweep_tracker_calibration.py ... --param vel_prior_std
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from isr.agents.heuristics import run_from_nearest_uav           # noqa: E402
from isr.agents.policy_loader import (                           # noqa: E402
    build_trained_agent, env_kwargs_from_checkpoint, load_policy,
)
from isr.env.pursuit_env import PursuitEnv                       # noqa: E402

NEES_TARGET = 4.0     # 4-D state [x, y, vx, vy]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--red-v-max", type=float, default=1.4)
    p.add_argument("--param", choices=("a_max", "vel_prior_std"),
                   default="a_max",
                   help="which RED_TRACKER_CONFIG scale to sweep.  Default "
                        "a_max because it sets the process noise "
                        "(sigma_a = a_max*sqrt(2)) and therefore the "
                        "STEADY-STATE covariance; vel_prior_std is only the "
                        "BIRTH prior and washes out after a few updates -- "
                        "measured, moving it 1.0 -> 2.8 changed NEES by 1.2 "
                        "out of 24 of error.")
    p.add_argument("--priors", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 3.0, 4.5, 6.0])
    p.add_argument("--n-episodes", type=int, default=25)
    p.add_argument("--seed-base", type=int, default=50_000)
    p.add_argument("--out-json", type=Path,
                   default=Path("runs/vel_prior_sweep.json"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_kwargs = env_kwargs_from_checkpoint(ckpt["args"])
    policy = load_policy(args.checkpoint, torch.device("cpu"))
    print(f"checkpoint {args.checkpoint} (rollout {ckpt.get('rollout')}), "
          f"red v_max {args.red_v_max}, {args.n_episodes} matched-seed "
          f"episodes per row\n")

    header = (f"{args.param:>9}  {'nees':>7}  {'|nees-4|':>9}  {'nis':>6}  "
              f"{'trk (m)':>8}  {'caught':>7}  {'steps':>7}")
    print(header)
    print("-" * len(header))

    out: Dict[str, Dict[str, float]] = {}
    for prior in args.priors:
        kw = dict(base_kwargs, red_v_max=args.red_v_max,
                  **{f"tracker_{args.param}": prior})
        nees: List[float] = []
        nis: List[float] = []
        trk: List[float] = []
        caught: List[int] = []
        steps: List[int] = []
        for ep in range(args.n_episodes):
            seed = args.seed_base + ep
            env = PursuitEnv(**kw, red_policy=run_from_nearest_uav, seed=seed)
            env.reset(seed=seed)
            blue = build_trained_agent(policy, torch.device("cpu"),
                                       deterministic=True)
            while env.agents:
                env.step({a: blue.act(None, env, a) for a in env.agents})
            d = env.tracker_diagnostics()
            for key, acc in (("nees", nees), ("nis", nis),
                             ("track_error_m", trk)):
                v = d.get(key, float("nan"))
                if not np.isnan(v):
                    acc.append(v)
            snap = env.state_snapshot()
            caught.append(int((~snap["red_active"]).sum()))
            steps.append(int(snap["t"]))

        row = {"nees": float(np.mean(nees)) if nees else float("nan"),
               "nis": float(np.mean(nis)) if nis else float("nan"),
               "track_error_m": float(np.mean(trk)) if trk else float("nan"),
               "mean_caught": float(np.mean(caught)),
               "se_caught": float(np.std(caught) / np.sqrt(len(caught))),
               "mean_steps": float(np.mean(steps))}
        out[f"{prior:.2f}"] = row
        print(f"{prior:9.2f}  {row['nees']:7.2f}  "
              f"{abs(row['nees'] - NEES_TARGET):9.2f}  {row['nis']:6.2f}  "
              f"{row['track_error_m']:8.2f}  "
              f"{row['mean_caught']:5.2f}+-{row['se_caught']:.2f}  "
              f"{row['mean_steps']:7.1f}")

    best = min(out, key=lambda k: abs(out[k]["nees"] - NEES_TARGET))
    top = max(out, key=lambda k: out[k]["mean_caught"])
    print(f"\n  nees closest to {NEES_TARGET}: {args.param} = {best} "
          f"(nees {out[best]['nees']:.2f}, caught {out[best]['mean_caught']:.2f})")
    print(f"  most captures:              {args.param} = {top} "
          f"(nees {out[top]['nees']:.2f}, caught {out[top]['mean_caught']:.2f})")
    if best != top:
        print("  -> they DISAGREE; calibration and performance are not the "
              "same objective here, and that is the finding.")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "red_v_max": args.red_v_max,
         "n_episodes": args.n_episodes, "nees_target": NEES_TARGET,
         "param": args.param, "by_value": out}, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
