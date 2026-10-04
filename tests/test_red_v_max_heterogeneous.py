"""
tests/test_red_v_max_heterogeneous.py — a PER-RED top speed.

``red_v_max`` was one number for every evader, so a blue never had to work
out which targets it could still run down.  ``red_v_max_min`` turns it into
a range sampled once per episode and held for it, mirroring ``n_red_min``.

Deliberately NOT exposed as a policy feature: a target's top speed is not
something a sensor reports, so declaring it would be a sixth privileged leak
(docs/tracker_observation.md).  The policy must infer it from the velocity
its own tracker estimates.

What must hold:

* the default is unchanged, or every result in docs/stage4_results.md moves;
* the cap binds PER RED, not just per episode -- a vector that exists but is
  applied as a scalar would pass a shape check and change nothing;
* speeds are constant WITHIN an episode (different evaders, not accelerating
  ones) and reproducible from the seed;
* it survives the checkpoint round trip, since a policy trained against
  mixed speeds evaluated against uniform ones is a silent regime change;
* blue speed is untouched.

Run:
    pytest tests/test_red_v_max_heterogeneous.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.agents.policy_loader import env_kwargs_from_checkpoint
from isr.env.entities import BLUE_UAV, RED_TARGET
from isr.env.pursuit_env import PursuitEnv
from scripts.train_stage4 import saved_args_from_flags

BASE = dict(n_blue=3, n_red=4, n_obstacles=0, arena_size=130.0,
            max_steps=60, capture_radius=3.0, sensor_radius=40.0)


def _flat_out(env, steps=40):
    """Drive every red at full throttle on +x and return peak |vx| per red."""
    env.reset(seed=0)
    peak = np.zeros(env.n_red)
    for _ in range(steps):
        if not env.agents:
            break
        env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
        peak = np.maximum(peak, np.abs(env._red_vel[:, 0]))
    return peak


def _throttle_red(_bp, rp, _ra, _op, _orad, _L):
    return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(rp), 1))


def test_the_default_is_unchanged():
    env = PursuitEnv(**BASE, seed=0)
    env.reset(seed=0)
    assert env.red_v_max_min is None
    assert np.allclose(env._red_v_max_vec, RED_TARGET.v_max)


def test_speeds_differ_between_reds():
    env = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=1.0, seed=0)
    env.reset(seed=0)
    v = env._red_v_max_vec
    assert len(np.unique(v)) == env.n_red, f"speeds not distinct: {v}"
    assert v.min() >= 1.0 and v.max() <= 1.5


def test_the_cap_BINDS_per_red():
    """The test that matters: a per-red vector that is applied as a scalar
    would satisfy every shape check and change nothing physical.  Drive all
    reds flat out and check each one's peak speed tracks ITS OWN cap."""
    env = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=0.6,
                     red_policy=_throttle_red, seed=0)
    env.reset(seed=0)
    caps = env._red_v_max_vec.copy()
    assert len(np.unique(caps)) == env.n_red
    peak = _flat_out(env)
    # Each red reaches its own cap and never exceeds it.
    assert np.all(peak <= caps + 1e-5), f"exceeded cap: {peak} vs {caps}"
    assert np.all(peak >= caps - 1e-4), f"did not reach cap: {peak} vs {caps}"
    # And the ORDERING is preserved, which a scalar cap could not produce.
    assert np.argsort(peak).tolist() == np.argsort(caps).tolist()


def test_speeds_are_constant_within_an_episode():
    env = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=1.0,
                     red_policy=_throttle_red, seed=0)
    env.reset(seed=0)
    first = env._red_v_max_vec.copy()
    for _ in range(20):
        if not env.agents:
            break
        env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
    np.testing.assert_array_equal(first, env._red_v_max_vec)


def test_speeds_are_reproducible_from_the_seed():
    a = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=1.0, seed=7)
    b = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=1.0, seed=7)
    a.reset(seed=7)
    b.reset(seed=7)
    np.testing.assert_array_equal(a._red_v_max_vec, b._red_v_max_vec)


def test_speeds_are_resampled_between_episodes():
    env = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=1.0, seed=0)
    env.reset(seed=0)
    first = env._red_v_max_vec.copy()
    env.reset(seed=1)
    assert not np.array_equal(first, env._red_v_max_vec)


def test_blue_speed_is_untouched():
    env = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=0.6, seed=0)
    env.reset(seed=0)
    for _ in range(40):
        if not env.agents:
            break
        env.step({a: np.array([1.0, 0.0], dtype=np.float32)
                  for a in env.agents})
    assert np.abs(env._blue_pos).max() < 1e9     # sanity: ran at all
    assert np.abs(env._blue_vel[:, 0]).max() <= BLUE_UAV.v_max + 1e-5


def test_a_min_above_the_max_is_refused():
    with pytest.raises(AssertionError, match="red_v_max_min"):
        PursuitEnv(**BASE, red_v_max=1.0, red_v_max_min=1.4, seed=0)


def test_it_round_trips_through_a_checkpoint():
    saved = saved_args_from_flags(
        "--n-blue 3 --n-red 4 --n-obstacles 0 --sensor-radius 40 "
        "--red-v-max 1.5 --red-v-max-min 1.0 --max-steps 60 --arena-size 130")
    assert saved["red_v_max_min"] == 1.0
    kw = env_kwargs_from_checkpoint(saved)
    assert kw["red_v_max_min"] == 1.0
    env = PursuitEnv(**kw, seed=0)
    env.reset(seed=0)
    assert len(np.unique(env._red_v_max_vec)) == env.n_red


def test_a_pre_flag_checkpoint_keeps_uniform_speeds():
    saved = saved_args_from_flags(
        "--n-blue 3 --n-red 4 --n-obstacles 0 --sensor-radius 40 "
        "--red-v-max 1.4 --max-steps 60 --arena-size 130")
    del saved["red_v_max_min"]
    kw = env_kwargs_from_checkpoint(saved)
    env = PursuitEnv(**kw, seed=0)
    env.reset(seed=0)
    assert np.allclose(env._red_v_max_vec, 1.4)


def test_the_speed_is_not_handed_to_the_policy():
    """It would be a sixth privileged leak.  Two envs whose reds differ ONLY
    in top speed must produce identical observations at reset, because at
    reset nothing has moved yet and so nothing has revealed it."""
    slow = PursuitEnv(**BASE, red_v_max=1.5, red_v_max_min=1.5,
                      actor_obs="tracker", use_belief_maps=False, seed=3)
    fast = PursuitEnv(**BASE, red_v_max=3.0, red_v_max_min=3.0,
                      actor_obs="tracker", use_belief_maps=False, seed=3)
    o_slow, _ = slow.reset(seed=3)
    o_fast, _ = fast.reset(seed=3)
    a = slow.structured_belief_observation()
    b = fast.structured_belief_observation()
    for k in a:
        if isinstance(a[k], np.ndarray) and k in b:
            np.testing.assert_allclose(
                a[k], b[k], atol=1e-6,
                err_msg=f"{k} reveals the red's top speed before it moves")
