"""
scripts/eval_checkpoint.py — one checkpoint, one number, with an error bar.

Exists so an unattended batch can report its own results instead of leaving
a training log to be read by hand.  The training log's per-eval numbers are
25 episodes with no error bar, and docs/stage4_results.md §15.4 is the record
of that being misleading: two runs whose last-eight log means differed by
0.16 turned out identical at 2.72 ± 0.08 when measured head to head.

The env is rebuilt from the checkpoint's own ``args`` so a policy is always
evaluated in the regime it trained in; ``--red-v-max`` overrides only that
one knob, for cross-speed checks.

Run:
    python scripts/eval_checkpoint.py --ckpt runs/stage4/X/final.pt
    python scripts/eval_checkpoint.py --ckpt ... --red-v-max 1.4 --episodes 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Same as train_stage4.py / train_bc.py: run as a script from anywhere without
# needing the repo on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.agents.heuristics import (run_from_nearest_uav, stationary_red,
                                   random_red)
from isr.agents.policy_loader import (build_trained_agent,
                                      env_kwargs_from_checkpoint, load_policy)
from isr.env.pursuit_env import PursuitEnv

RED_KINDS = ("run", "stationary", "random")


def _red(kind: str, seed: int):
    if kind == "run":
        return run_from_nearest_uav
    if kind == "stationary":
        return stationary_red
    return random_red(seed=seed)


def evaluate(ckpt: str, red_kind: str, n_ep: int, red_v_max, seed_base: int,
             n_obstacles=None):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = env_kwargs_from_checkpoint(ck["args"])
    if red_v_max is not None:
        kw["red_v_max"] = float(red_v_max)
    if n_obstacles is not None:
        # Removing obstacles from a policy TRAINED with them is the forgetting
        # check, and it works without a shape error: the env then emits no
        # obstacle keys, so the encoder's has_obs is False and the ob channel
        # contributes nothing.  The question it answers is "with no obstacles
        # present, does it still pursue as well as before it learned to dodge".
        kw["n_obstacles"] = int(n_obstacles)
    dev = torch.device("cpu")
    policy = load_policy(ckpt, dev)
    caught, steps, rets = [], [], []
    for ep in range(n_ep):
        seed = seed_base + ep
        env = PursuitEnv(**kw, red_policy=_red(red_kind, seed), seed=seed)
        env.reset(seed=seed)
        blue = build_trained_agent(policy, dev, deterministic=True)
        tot = 0.0
        while env.agents:
            _o, rew, _t, _tr, _i = env.step(
                {a: blue.act(None, env, a) for a in env.agents})
            tot += float(sum(rew.values())) / max(len(rew), 1)
        s = env.state_snapshot()
        caught.append(int((~s["red_active"]).sum()))
        steps.append(int(s["t"]))
        rets.append(tot)
    c = np.array(caught, dtype=float)
    return dict(
        caught=float(c.mean()),
        # ddof=1: this is a sample SE, and with n=25 the difference from the
        # population form is not negligible.
        se=float(c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else 0.0,
        steps=float(np.mean(steps)),
        ret=float(np.mean(rets)),
        n=len(c),
        red_v_max=float(kw["red_v_max"]) if kw.get("red_v_max") else None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--red-v-max", type=float, default=None,
                   help="override the checkpoint's red speed (cross-speed check)")
    p.add_argument("--n-obstacles", type=int, default=None,
                   help="override the obstacle count; 0 on an obstacle-trained "
                        "policy is the forgetting check")
    p.add_argument("--reds", default="run",
                   help=f"comma-separated subset of {RED_KINDS}, or 'all'")
    p.add_argument("--seed-base", type=int, default=70_000,
                   help="matched seeds across checkpoints -- keep it fixed "
                        "when comparing runs, or the comparison is unpaired")
    p.add_argument("--json", default=None, help="also write results here")
    a = p.parse_args()

    kinds = RED_KINDS if a.reds == "all" else tuple(
        k.strip() for k in a.reds.split(","))
    out = {}
    print(f"{a.ckpt}   {a.episodes} episodes, seed_base {a.seed_base}")
    print(f"{'red':>12}  {'caught':>14}  {'steps':>7}  {'return':>8}")
    print("-" * 48)
    for k in kinds:
        r = evaluate(a.ckpt, k, a.episodes, a.red_v_max, a.seed_base,
                     a.n_obstacles)
        out[k] = r
        print(f"{k:>12}  {r['caught']:6.2f} +- {r['se']:4.2f}  "
              f"{r['steps']:7.1f}  {r['ret']:8.2f}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"ckpt": a.ckpt, "episodes": a.episodes,
                       "red_v_max": a.red_v_max, "results": out}, f, indent=2)


if __name__ == "__main__":
    main()
