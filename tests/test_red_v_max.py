"""
tests/test_red_v_max.py — red's top speed as an env parameter.

It used to be ``RED_TARGET.v_max``, a module constant applied directly in
``step()``, so the speed ratio that decides whether this task rewards
coordination at all was not configurable and not recorded in any
checkpoint.

What must hold:

* the default is unchanged, or every number in docs/stage4_results.md
  silently moves;
* the parameter actually binds red's motion, and binds ONLY red's;
* it survives the round trip through a checkpoint, since a sweep over it
  produces results that are meaningless without knowing which value
  produced them.

Run:
    pytest tests/test_red_v_max.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.policy_loader import env_kwargs_from_checkpoint
from isr.env.entities import BLUE_UAV, RED_TARGET
from isr.env.pursuit_env import PursuitEnv
from scripts.train_stage4 import saved_args_from_flags

BASE = dict(n_blue=2, n_red=2, n_obstacles=0, arena_size=130.0,
            max_steps=60, capture_radius=3.0, sensor_radius=40.0)


def _flat_out_red(v_max, n_steps=40, seed=0):
    """Drive every red at full throttle in +x and read its top speed."""
    def always_east(blue_pos, red_pos, red_active, *a, **kw):
        out = np.zeros_like(red_pos, dtype=np.float32)
        out[:, 0] = 1.0
        out[~red_active] = 0.0
        return out

    env = PursuitEnv(**BASE, red_policy=always_east, red_v_max=v_max,
                     seed=seed)
    env.reset(seed=seed)
    # Park the blues far away so nothing is captured mid-run.
    env._blue_pos = np.full((2, 2), 5.0, dtype=np.float32)
    env._red_pos = np.array([[10.0, 60.0], [10.0, 100.0]], dtype=np.float32)
    top = 0.0
    for _ in range(n_steps):
        env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
        env._blue_pos = np.full((2, 2), 5.0, dtype=np.float32)
        top = max(top, float(np.abs(env._red_vel).max()))
    return top


def test_default_is_unchanged():
    """The setting every published result was measured under."""
    env = PursuitEnv(**BASE, seed=0)
    assert env.red_v_max == RED_TARGET.v_max == 1.0
    assert BLUE_UAV.v_max == 1.5, "the 1.5 the closing-rate argument uses"


def test_the_parameter_binds_reds_speed():
    for v in (0.5, 1.0, 1.4):
        top = _flat_out_red(v)
        assert abs(top - v) < 1e-3, (v, top)


def test_it_does_not_touch_blue():
    """Blue must still cap at BLUE_UAV.v_max whatever red is given."""
    env = PursuitEnv(**BASE, red_v_max=1.4, seed=1)
    env.reset(seed=1)
    for _ in range(40):
        env.step({a: np.array([1.0, 0.0], dtype=np.float32)
                  for a in env.agents})
    assert float(np.abs(env._blue_vel).max()) <= BLUE_UAV.v_max + 1e-4


def test_it_round_trips_through_a_checkpoints_args():
    """A sweep is uninterpretable if the checkpoint does not say which
    speed produced it."""
    kw = env_kwargs_from_checkpoint(saved_args_from_flags(
        "--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 "
        "--max-steps 200 --actor-obs tracker --red-v-max 1.3"))
    assert kw["red_v_max"] == 1.3
    env = PursuitEnv(**kw, seed=0)
    assert env.red_v_max == 1.3


def test_pre_flag_checkpoints_keep_the_old_speed():
    """An args dict written before the flag existed must rebuild at 1.0."""
    kw = env_kwargs_from_checkpoint(saved_args_from_flags(
        "--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 "
        "--max-steps 200 --actor-obs tracker"))
    assert kw["red_v_max"] is None
    assert PursuitEnv(**kw, seed=0).red_v_max == RED_TARGET.v_max


def test_a_faster_evader_is_measurably_harder_to_catch():
    """The premise of the whole sweep: pure pursuit degrades as the
    closing rate shrinks.  If this does not hold, the speed knob is not
    doing what the closing-rate argument says."""
    from isr.agents.heuristics import GreedyPursuer, run_from_nearest_uav

    def caught_at(v, n_ep=12):
        total = 0
        for ep in range(n_ep):
            env = PursuitEnv(**dict(BASE, max_steps=200),
                             red_policy=run_from_nearest_uav,
                             red_v_max=v, seed=4000 + ep)
            env.reset(seed=4000 + ep)
            blue = GreedyPursuer()
            while env.agents:
                env.step({a: blue.act(None, env, a) for a in env.agents})
            total += int((~env.state_snapshot()["red_active"]).sum())
        return total / n_ep

    slow, fast = caught_at(1.0), caught_at(1.45)
    assert slow > fast, (slow, fast)
