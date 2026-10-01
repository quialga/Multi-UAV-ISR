"""
scripts/eval_red_count_sweep.py — the degradation CURVE, not a point.

The hypothesis attention is built for is not "scores higher" but "degrades
more slowly as the number of live targets varies".  A single operating point
cannot test that; two curves can.  So this sweeps the active red count and
reports the slope of captured FRACTION against count, which is the quantity
the comparison turns on.

Why a fraction and not raw captures: raw captures rise with the count simply
because there is more to catch, so a flat raw line already means degradation.
The fraction normalises that away.

Why this works at all: in tracker mode the actor's red nodes are the fixed
tracker SLOTS, not the active reds (``tracker_red_slots``, 8), so the actor
graph is identical at every count and ONE policy evaluates across the sweep
without rebuilding.  ``tests/test_variable_counts.py`` pins that.

Both arms share a time budget (``max_steps`` from the checkpoint), and that
budget is deliberately NOT raised for larger counts: measured on
``AssignGreedy`` at 5v5, the value of coordination SHRINKS as steps grow
(+0.43 at 200 steps, +0.40 at 300, +0.33 at 400), because given enough time
uncoordinated pursuit catches everything anyway.  Time pressure is what makes
allocation matter, so it stays fixed.

Run:
    python scripts/eval_red_count_sweep.py --ckpt runs/stage4/X/final.pt
    python scripts/eval_red_count_sweep.py --ckpt ... --counts 1,2,3,4,5 --episodes 100
    python scripts/eval_red_count_sweep.py --heuristic assign --ckpt ...  # baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.agents.heuristics import (AssignmentGreedyPursuer,
                                   ObservationGreedyPursuer,
                                   run_from_nearest_uav)
from isr.agents.policy_loader import (build_trained_agent,
                                      env_kwargs_from_checkpoint, load_policy)
from isr.env.pursuit_env import PursuitEnv


def run_count(make_blue, kw, n_red: int, n_ep: int, seed_base: int):
    # n_red_min cleared: the sweep wants EXACTLY n_red active, not a sample.
    # Leaving it set would make every column measure the same mixture.
    kw = dict(kw, n_red=n_red, n_red_min=None)
    caught, steps = [], []
    for ep in range(n_ep):
        seed = seed_base + ep
        env = PursuitEnv(**kw, red_policy=run_from_nearest_uav, seed=seed)
        env.reset(seed=seed)
        blue = make_blue()
        while env.agents:
            env.step({a: blue.act(None, env, a) for a in env.agents})
        s = env.state_snapshot()
        caught.append(int((~s["red_active"]).sum()))
        steps.append(int(s["t"]))
    c = np.array(caught, dtype=float)
    return dict(n_red=n_red, caught=float(c.mean()),
                se=float(c.std(ddof=1) / np.sqrt(len(c))),
                frac=float(c.mean() / n_red),
                frac_se=float(c.std(ddof=1) / np.sqrt(len(c)) / n_red),
                steps=float(np.mean(steps)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="the policy, or (with --heuristic) just the env config")
    p.add_argument("--heuristic", choices=("obs", "assign"), default=None,
                   help="evaluate a heuristic in the checkpoint's env instead")
    p.add_argument("--counts", default="1,2,3,4,5")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--red-v-max", type=float, default=None)
    p.add_argument("--seed-base", type=int, default=91_000,
                   help="keep fixed across checkpoints or the curves are unpaired")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    kw = env_kwargs_from_checkpoint(ck["args"])
    if a.red_v_max is not None:
        kw["red_v_max"] = float(a.red_v_max)

    if a.heuristic:
        cls = {"obs": ObservationGreedyPursuer,
               "assign": AssignmentGreedyPursuer}[a.heuristic]
        make_blue = lambda: cls()
        label = f"{a.heuristic}Greedy (env from {a.ckpt})"
    else:
        dev = torch.device("cpu")
        policy = load_policy(a.ckpt, dev)
        make_blue = lambda: build_trained_agent(policy, dev, deterministic=True)
        label = a.ckpt

    counts = [int(x) for x in a.counts.split(",")]
    print(f"{label}")
    print(f"red v_max {kw.get('red_v_max')}, max_steps {kw['max_steps']}, "
          f"{a.episodes} episodes, seed_base {a.seed_base}\n")
    print(f"{'n_red':>6}  {'caught':>14}  {'fraction':>14}  {'steps':>7}")
    print("-" * 50)
    rows = []
    for n in counts:
        r = run_count(make_blue, kw, n, a.episodes, a.seed_base)
        rows.append(r)
        print(f"{n:6d}  {r['caught']:6.2f} +- {r['se']:4.2f}  "
              f"{r['frac']:6.3f} +- {r['frac_se']:5.3f}  {r['steps']:7.1f}",
              flush=True)

    # Least-squares slope of fraction against count: the degradation rate the
    # two arms are compared on.  Reported with its standard error so a
    # difference between arms can be judged rather than eyeballed.
    x = np.array([r["n_red"] for r in rows], dtype=float)
    y = np.array([r["frac"] for r in rows], dtype=float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se_slope = (np.sqrt((resid ** 2).sum() / (n - 2)
                        / ((x - x.mean()) ** 2).sum()) if n > 2 else float("nan"))
    print(f"\nfraction slope: {slope:+.4f} per extra red  (se {se_slope:.4f})")
    print("  less negative = degrades more slowly, which is the hypothesis")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(ckpt=a.ckpt, heuristic=a.heuristic,
                           episodes=a.episodes, seed_base=a.seed_base,
                           red_v_max=kw.get("red_v_max"),
                           rows=rows, slope=float(slope),
                           slope_se=float(se_slope)), f, indent=2)


if __name__ == "__main__":
    main()
