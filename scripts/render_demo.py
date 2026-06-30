"""
scripts/render_demo.py — Render one pursuit episode.

Run a single episode with chosen blue/red heuristic policies and either
display the animation interactively or save it as a GIF.

Examples
--------
Interactive (opens a matplotlib window):
    python scripts/render_demo.py --blue greedy --red run

Save to GIF for the README / portfolio:
    python scripts/render_demo.py --blue greedy --red run --save out/greedy_vs_run.gif

Random blue against stationary red — sanity-check the renderer wiring:
    python scripts/render_demo.py --blue random --red stationary --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

# Allow running as `python scripts/render_demo.py` from repo root without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import (
    RandomAgent, GreedyPursuer,
    stationary_red, random_red, run_from_nearest_uav,
)
from isr.utils.render import animate_episode


BLUE_POLICIES = {
    "random": lambda seed: RandomAgent(seed=seed),
    "greedy": lambda seed: GreedyPursuer(),
}
RED_POLICIES = {
    "stationary": lambda seed: stationary_red,
    "random":     lambda seed: random_red(seed=seed),
    "run":        lambda seed: run_from_nearest_uav,
}


def run_episode(
    blue_factory: Callable,
    red_factory:  Callable,
    seed:         int,
    n_blue:       int,
    n_red:        int,
    max_steps:    int,
) -> Tuple[PursuitEnv, list, float]:
    """Run one episode; return (env, snapshots, total_reward)."""
    env = PursuitEnv(
        n_blue=n_blue, n_red=n_red, max_steps=max_steps,
        red_policy=red_factory(seed), seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    blue = blue_factory(seed)

    snaps = [env.state_snapshot()]
    total = 0.0
    while env.agents:
        actions = {a: blue.act(obs[a], env, a) for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
        total += rew["blue_0"]                   # shared team reward
        snaps.append(env.state_snapshot())
    return env, snaps, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blue", choices=BLUE_POLICIES, default="greedy",
                        help="blue team policy")
    parser.add_argument("--red",  choices=RED_POLICIES,  default="run",
                        help="red team policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-blue", type=int, default=3)
    parser.add_argument("--n-red",  type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--trail-len", type=int, default=5)
    parser.add_argument("--save", default=None,
                        help="output .gif path; omit to show interactively")
    args = parser.parse_args()

    env, snaps, total = run_episode(
        BLUE_POLICIES[args.blue], RED_POLICIES[args.red],
        seed=args.seed, n_blue=args.n_blue, n_red=args.n_red,
        max_steps=args.max_steps,
    )
    n_caught = int((~snaps[-1]["red_active"]).sum())

    print(f"Episode: blue={args.blue}  red={args.red}  seed={args.seed}")
    print(f"  steps     : {snaps[-1]['t']}")
    print(f"  caught    : {n_caught}/{args.n_red}")
    print(f"  return    : {total:+.2f}")
    print(f"  frames    : {len(snaps)}")

    if args.save:
        save_dir = Path(args.save).parent
        if save_dir and not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        print(f"  saving GIF -> {args.save}  (fps={args.fps})")

    animate_episode(
        snaps, env.arena_size,
        save_path=args.save, fps=args.fps, trail_len=args.trail_len,
    )

    if args.save:
        print("  done.")


if __name__ == "__main__":
    main()
