"""
scripts/sweep_red_speed.py — does this task reward coordination at all?

`docs/design.md §3.6` set blue 1.5 against red 1.0 so that "pure pursuit
is winnable in principle but not trivial — coordination among blue agents
has to provide the extra edge".  The measurements in
`docs/stage4_results.md` say the second half of that did not come true:
`ObsGreedy` is pure pursuit with **no coordination whatsoever** — five
blues all steer at their own nearest track, which is frequently the same
one — and it scores **2.90/3** (§10), 97% of the maximum.

The arithmetic agrees.  A blue behind a red closes at 1.5 − 1.0 = 0.5 per
step; over a 200-step episode that is 100 units of closing in a 130 m
arena, so a stern chase essentially always succeeds and nobody ever has
to cut anybody off.

This sweeps red's top speed and measures where that stops being true.
The quantity of interest is not any single score but the **gap** between
what a coordination-free rule achieves and what the task allows:

* while `ObsGreedy` ≈ `Greedy` ≈ ceiling, coordination buys nothing and
  no amount of RL will find any, because there is none to find;
* where `ObsGreedy` falls away, cutting a target off starts to be the
  only way to catch it, and that gap is the headroom a learned policy
  could actually win.

`Greedy` (true positions, range-gated) is carried as the information
control: it degrades for the same kinematic reason, so the two falling
together is the task getting harder, not the observation getting worse.

**Changes the task.**  Nothing measured here is comparable with §6–§11,
which are all at red 1.0.  This is a diagnostic sweep, not a new baseline.

Run:
    python scripts/sweep_red_speed.py --n-episodes 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                               # noqa: E402

import scripts.train_stage4 as train_stage4                      # noqa: E402
from isr.agents.heuristics import (                              # noqa: E402
    GreedyPursuer, ObservationGreedyPursuer, RandomAgent,
    run_from_nearest_uav,
)
from isr.agents.policy_loader import env_kwargs_from_checkpoint  # noqa: E402
from scripts.evaluate_trained import run_episode                 # noqa: E402

# §10/§11 geometry: tracker observation with the coverage path, which is
# the arm the project is building on.
BASE_FLAGS = (
    "--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 "
    "--belief-grid-size 26 --max-steps 200 --actor-obs tracker "
    "--use-staleness --staleness-regions 5"
)
BLUE_V_MAX = 1.5      # isr/env/entities.py


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speeds", type=float, nargs="+",
                   default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                   help="red top speeds to sweep (blue is 1.5)")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=50_000)
    p.add_argument("--out-json", type=Path,
                   default=Path("runs/red_speed_sweep.json"))
    args = p.parse_args()

    # Only the EVADER is informative here: a stationary or random red does
    # not use the speed it is given, so a sweep over it measures nothing.
    blue_factories = {
        "Random":    lambda: RandomAgent(seed=0),
        "Greedy":    lambda: GreedyPursuer(),
        "ObsGreedy": lambda: ObservationGreedyPursuer(),
    }

    print(f"blue v_max {BLUE_V_MAX}, red = run_from_nearest_uav, "
          f"{args.n_episodes} matched-seed episodes per cell\n")
    header = (f"{'red v_max':>9}  {'closing':>8}  "
              + "  ".join(f"{b:>18}" for b in blue_factories))
    print(header)
    print("-" * len(header))

    out: Dict[str, Dict] = {}
    for v in args.speeds:
        env_kwargs = env_kwargs_from_checkpoint(
            train_stage4.saved_args_from_flags(
                f"{BASE_FLAGS} --red-v-max {v}"))
        row: Dict[str, Dict[str, float]] = {}
        for name, factory in blue_factories.items():
            caught: List[int] = []
            steps: List[int] = []
            for ep in range(args.n_episodes):
                seed = args.seed_base + ep
                _, c, n, _ = run_episode(factory(), run_from_nearest_uav,
                                         env_kwargs, seed)
                caught.append(c)
                steps.append(n)
            row[name] = {"mean_caught": float(np.mean(caught)),
                         "se_caught": float(np.std(caught)
                                            / np.sqrt(len(caught))),
                         "mean_steps": float(np.mean(steps))}
        cells = "  ".join(
            f"{row[b]['mean_caught']:8.2f}+-{row[b]['se_caught']:.2f} "
            f"({row[b]['mean_steps']:5.1f})" for b in blue_factories)
        print(f"{v:9.2f}  {BLUE_V_MAX - v:8.2f}  {cells}")
        out[f"{v:.2f}"] = row

    # The headline: how much a coordination-free rule leaves on the table.
    print(f"\n{'red v_max':>9}  {'ObsGreedy':>10}  {'vs Greedy':>10}  "
          f"{'vs 3.00':>9}")
    for v in args.speeds:
        r = out[f"{v:.2f}"]
        og = r["ObsGreedy"]["mean_caught"]
        print(f"{v:9.2f}  {og:10.2f}  "
              f"{og - r['Greedy']['mean_caught']:+10.2f}  {og - 3.0:+9.2f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"blue_v_max": BLUE_V_MAX, "red": "run_from_nearest_uav",
         "n_episodes": args.n_episodes, "seed_base": args.seed_base,
         "base_flags": BASE_FLAGS, "by_red_v_max": out}, indent=2),
        encoding="utf-8")
    print(f"\nJSON: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
