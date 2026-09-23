"""
scripts/compare_observation_quality.py — belief map vs tracker, with no
policy in the way.

``docs/stage4_results.md §7`` compares the two actor observations through
two TRAINED policies, and §8 then shows both policies sit at ~25% of what
their own observation supports.  A comparison of representations read off
two policies that far below their input's ceiling measures the policies,
not the representations.

``ObservationGreedyPursuer`` consumes exactly the actor's enemy graph
(``rb_edge_features`` + ``rb_edge_visible`` from
``structured_belief_observation()``) and nothing else, so running it under
``--actor-obs belief`` and ``--actor-obs tracker`` asks the question
directly: **how much pursuit performance is extractable from each
observation, with no learning involved?**

``Random`` and ``Greedy`` are the controls.  Neither reads the actor
observation — ``Greedy`` reads ``state_snapshot()``, ``Random`` reads
nothing — so their rows must come out the same in both modes up to
episode noise.  If they do, the ObsGreedy delta is the observation; if
they do not, the two modes are not the same env and the delta is
confounded (belief maps carry their own RNG draws, and the tracker path
adds clutter plots).

That control is what caught the shared red-RNG bug now fixed in
``evaluate_trained.eval_matrix``: it made the `Random` column depend on
the episode lengths of the rows above it, so the control failed on that
one column while passing to the digit on the other two.

Defaults reproduce the §7 head-to-head geometry: arena 130, 5 blue /
3 red, no obstacles, ``max_steps 200``, §4b sensor model on.  Everything
else comes from the TRAINER's own argument parser, so a config here is
the same config a training run would get — no second copy of the
defaults to drift.

Run:
    python scripts/compare_observation_quality.py --n-episodes 50
    python scripts/compare_observation_quality.py --train-args "--arena-size 200 --n-obstacles 4"

No checkpoint is needed: none of the three baselines learns.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train_stage4 as train_stage4                      # noqa: E402
from isr.agents.heuristics import (                              # noqa: E402
    GreedyPursuer, ObservationGreedyPursuer, RandomAgent,
    run_from_nearest_uav, stationary_red, random_red,
)
from isr.agents.policy_loader import env_kwargs_from_checkpoint  # noqa: E402
from scripts.evaluate_trained import eval_matrix, _fmt_cell      # noqa: E402

# The §7 / §8 geometry.  Kept as one string so it reads as the command a
# training run was launched with, which is how docs/stage4_results.md
# records configurations.
DEFAULT_TRAIN_ARGS = (
    "--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 "
    "--belief-grid-size 26 --max-steps 200"
)


def env_kwargs_for_mode(train_args: str, mode: str) -> Dict:
    """Build the env config a TRAINING RUN with these flags would get.

    Via ``train_stage4.saved_args_from_flags``, not the parser alone: the
    reward shape lives in the config rather than in argparse, so a dict
    built from the parser makes ``env_kwargs_from_checkpoint`` fall back to
    its pre-feature defaults (``step_cost`` 0.05 against the 0.033 in use)
    and compare under a reward no run has trained with.  It changes no
    `caught` number here — none of these baselines reads the reward — but
    it would make every `mean_return` incomparable with §7/§8.
    """
    return env_kwargs_from_checkpoint(
        train_stage4.saved_args_from_flags(
            f"{train_args} --actor-obs {mode}"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-episodes", type=int, default=50,
                   help="episodes per (blue, red) cell, per mode")
    p.add_argument("--seed-base", type=int, default=50_000,
                   help="shared across blue policies AND across the two "
                        "modes, so the two tables are matched-seed")
    p.add_argument("--train-args", default=DEFAULT_TRAIN_ARGS,
                   help="trainer flags describing the env to compare under "
                        "(--actor-obs is supplied by this script)")
    p.add_argument("--out-json", type=Path,
                   default=Path("runs/observation_quality.json"))
    args = p.parse_args()

    # seed -> policy, so the red noise depends only on the episode seed.
    red_factories = {
        "Stationary":     lambda seed: stationary_red,
        "Random":         lambda seed: random_red(seed=seed),
        "RunFromNearest": lambda seed: run_from_nearest_uav,
    }
    blue_factories = {
        "Random":    lambda: RandomAgent(seed=0),
        "Greedy":    lambda: GreedyPursuer(),
        "ObsGreedy": lambda: ObservationGreedyPursuer(),
    }

    all_results: Dict[str, Dict] = {}
    for mode in ("belief", "tracker"):
        env_kwargs = env_kwargs_for_mode(args.train_args, mode)
        print(f"\n=== actor_obs = {mode} ===")
        print(f"env_kwargs={env_kwargs}")
        t0 = time.time()
        results = eval_matrix(blue_factories, red_factories, env_kwargs,
                              n_episodes=args.n_episodes,
                              seed_base=args.seed_base)
        print(f"  {args.n_episodes} episodes/cell in {time.time() - t0:.1f}s")

        red_names = list(red_factories)
        header = f"{'Blue':>10}  " + "  ".join(f"{r:>25}" for r in red_names)
        print(header)
        print("-" * len(header))
        for b in blue_factories:
            cells = [_fmt_cell(results[b][r]) for r in red_names]
            print(f"{b:>10}  " + "  ".join(f"{c:>25}" for c in cells))
        all_results[mode] = {"env_kwargs": {k: str(v) for k, v in
                                            env_kwargs.items()},
                             "results": results}

    # ---- The comparison: mean caught over the three red policies --------
    print("\n=== mean caught / n_red, averaged over the three reds ===")
    print(f"{'Blue':>10}  {'belief':>8}  {'tracker':>8}  {'delta':>8}")
    for b in blue_factories:
        means = {}
        for mode in ("belief", "tracker"):
            cells = all_results[mode]["results"][b]
            means[mode] = sum(c["mean_caught"] for c in cells.values()) / len(cells)
        print(f"{b:>10}  {means['belief']:8.3f}  {means['tracker']:8.3f}  "
              f"{means['tracker'] - means['belief']:+8.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"n_episodes": args.n_episodes, "seed_base": args.seed_base,
         "train_args": args.train_args, "modes": all_results},
        indent=2), encoding="utf-8")
    print(f"\nJSON: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
