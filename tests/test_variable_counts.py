"""
tests/test_variable_counts.py — per-episode active-count sampling must survive
the checkpoint round trip.

``train_stage4.py`` has always passed ``n_red_min`` / ``n_obstacles_min`` to
the env, but ``env_kwargs_from_checkpoint`` did not read them.  So every
consumer that rebuilds an env from a checkpoint -- ``train_bc.py``, every
evaluation script, the renderer -- silently ran at FIXED capacity.  That is a
regime change, not a detail: a policy trained on 1-5 reds would only ever be
measured at 5, and the generalisation result the variable-count feature exists
to produce could not be measured at all.

Run:
    pytest tests/test_variable_counts.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.policy_loader import env_kwargs_from_checkpoint
from isr.env.pursuit_env import PursuitEnv
from scripts.train_stage4 import saved_args_from_flags

FLAGS = ("--n-blue 5 --n-red 5 --n-red-min 1 --n-obstacles 0 "
         "--sensor-radius 40 --actor-obs tracker --use-staleness "
         "--max-steps 200 --arena-size 130")


def test_n_red_min_round_trips():
    saved = saved_args_from_flags(FLAGS)
    assert saved["n_red_min"] == 1
    kw = env_kwargs_from_checkpoint(saved)
    assert kw["n_red_min"] == 1, "dropped on the read path"
    assert kw["n_red"] == 5


def test_the_env_actually_samples_the_active_count():
    """The round trip is pointless if the env then ignores it."""
    kw = env_kwargs_from_checkpoint(saved_args_from_flags(FLAGS))
    counts = set()
    for ep in range(40):
        env = PursuitEnv(**kw, seed=1000 + ep)
        env.reset(seed=1000 + ep)
        counts.add(int(env._red_active.sum()))
    assert len(counts) > 1, f"active count never varied: {counts}"
    assert max(counts) <= kw["n_red"]
    assert min(counts) >= 1


def test_a_pre_flag_checkpoint_stays_fixed_at_capacity():
    """Checkpoints written before the keys existed must not start sampling."""
    saved = saved_args_from_flags(FLAGS)
    del saved["n_red_min"]
    kw = env_kwargs_from_checkpoint(saved)
    assert kw["n_red_min"] is None
    env = PursuitEnv(**kw, seed=0)
    env.reset(seed=0)
    assert int(env._red_active.sum()) == kw["n_red"]


def test_n_obstacles_min_round_trips_too():
    saved = saved_args_from_flags(
        "--n-blue 3 --n-red 2 --n-obstacles 4 --n-obstacles-min 1 "
        "--sensor-radius 40 --max-steps 100 --arena-size 130")
    assert saved["n_obstacles_min"] == 1
    assert env_kwargs_from_checkpoint(saved)["n_obstacles_min"] == 1


def test_the_actor_graph_width_does_not_depend_on_the_red_count():
    """What makes the eval sweep possible at all: in tracker mode the actor's
    red nodes are the fixed tracker SLOTS, not the active reds, so one policy
    evaluates at any count without rebuilding."""
    widths = set()
    for n_red in (1, 3, 5):
        kw = env_kwargs_from_checkpoint(saved_args_from_flags(
            FLAGS.replace("--n-red 5", f"--n-red {n_red}")
                 .replace("--n-red-min 1 ", "")))
        env = PursuitEnv(**kw, seed=0)
        env.reset(seed=0)
        widths.add((env.actor_n_red, env.structured_belief_observation()
                    ["rb_edge_visible"].shape[0]))
    assert len(widths) == 1, f"actor graph changed with the red count: {widths}"
