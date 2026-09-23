"""
tests/test_eval_matrix_seeding.py — the evaluation matrix must be
matched-seed, including its RANDOM red.

``eval_matrix`` compares blue policies cell by cell on shared episode
seeds.  That guarantee used to hold only for the env: ``red_factories``
took red POLICIES, and ``random_red(seed=0)`` is a closure over one
``default_rng``, so a single stream was shared across every episode AND
every blue row.  Its draws therefore depended on how many steps the
preceding rows had consumed — and episode length is the very thing the
matrix measures, so a faster blue policy silently changed the reds the
next row faced.

The failure was quiet: every number looked plausible, the `Random` column
was simply not comparable across rows.  It surfaced only as a control in
``compare_observation_quality.py``, where `Stationary` and
`RunFromNearest` reproduced digit-for-digit across two sweeps and
`Random` did not.

The property below is the cheap statement of the fix: the matrix must not
depend on the ORDER of its blue rows.  It fails on the old code and
passes on the new.

Run:
    pytest tests/test_eval_matrix_seeding.py -v
"""
from __future__ import annotations

from isr.agents.heuristics import (
    GreedyPursuer, RandomAgent, random_red, run_from_nearest_uav,
    stationary_red,
)
from scripts.evaluate_trained import eval_matrix
from scripts.train_stage4 import saved_args_from_flags
from isr.agents.policy_loader import env_kwargs_from_checkpoint

FLAGS = ("--arena-size 130 --n-blue 3 --n-red 2 --n-obstacles 0 "
         "--belief-grid-size 26 --max-steps 40 --actor-obs belief")

RED_FACTORIES = {
    "Stationary":     lambda seed: stationary_red,
    "Random":         lambda seed: random_red(seed=seed),
    "RunFromNearest": lambda seed: run_from_nearest_uav,
}


def _env_kwargs():
    return env_kwargs_from_checkpoint(saved_args_from_flags(FLAGS))


def _run(blue_factories):
    return eval_matrix(blue_factories, RED_FACTORIES, _env_kwargs(),
                       n_episodes=4, seed_base=50_000)


def test_row_order_does_not_change_the_numbers():
    """Two blue policies with very different episode lengths — Greedy
    finishes early, Random times out — evaluated in both orders."""
    fast_first = _run({"Greedy": lambda: GreedyPursuer(),
                       "Random": lambda: RandomAgent(seed=0)})
    slow_first = _run({"Random": lambda: RandomAgent(seed=0),
                       "Greedy": lambda: GreedyPursuer()})

    for blue in ("Greedy", "Random"):
        for red in RED_FACTORIES:
            a = fast_first[blue][red]
            b = slow_first[blue][red]
            assert a == b, (blue, red, a, b)


def test_repeating_the_same_matrix_is_deterministic():
    """No hidden stream carries across calls either."""
    factories = {"Greedy": lambda: GreedyPursuer(),
                 "Random": lambda: RandomAgent(seed=0)}
    assert _run(factories) == _run(factories)


def test_the_random_red_actually_varies_with_the_seed():
    """Guard against the trivial way to pass the two tests above: a red
    factory that ignores its seed would be perfectly reproducible and
    perfectly useless.  Different seed bases must give different
    episodes."""
    factories = {"Greedy": lambda: GreedyPursuer()}
    a = eval_matrix(factories, RED_FACTORIES, _env_kwargs(),
                    n_episodes=4, seed_base=50_000)["Greedy"]["Random"]
    b = eval_matrix(factories, RED_FACTORIES, _env_kwargs(),
                    n_episodes=4, seed_base=77_000)["Greedy"]["Random"]
    assert a != b, (a, b)
