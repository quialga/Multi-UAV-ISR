"""
tests/test_gazebo_kinematics.py — the bridge's command integration
must be BIT-IDENTICAL to the env's.

gazebo/kinematics.integrate_cmd re-implements the velocity half of
``PursuitEnv._integrate`` (per-axis v_max clip + wall-stop) for the
Gazebo bridge nodes.  Any drift between the two is a silent
sim-to-sim parity leak, so we fuzz them against each other directly.
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from gazebo.kinematics import integrate_cmd


def test_integrate_cmd_matches_env_integrate():
    env = PursuitEnv(n_blue=3, n_red=2, arena_size=130.0, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(7)
    for _ in range(200):
        # Positions biased toward walls so the wall-stop branch fires.
        pos = rng.uniform(-0.5, 130.5, size=(4, 2)).astype(np.float32)
        pos = np.clip(pos, 0.0, 130.0)
        vel = rng.uniform(-1.5, 1.5, size=(4, 2)).astype(np.float32)
        acc = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
        v_max = float(rng.choice([1.0, 1.5]))

        _, env_vel = env._integrate(pos.copy(), vel.copy(), acc, v_max)
        bridge_vel = integrate_cmd(pos.copy(), vel.copy(), acc,
                                   v_max, env.dt, env.arena_size)
        assert np.array_equal(env_vel, bridge_vel)


def test_wall_stop_zeroes_only_offending_axis():
    """An agent pressed against x=0 keeps its y motion (slide) and can
    accelerate away in +x next tick — the env's 'no bounce, back off'
    semantics."""
    pos = np.array([[0.0, 50.0]], dtype=np.float32)
    vel = np.array([[-1.5, 1.0]], dtype=np.float32)
    acc = np.array([[-1.0, 0.0]], dtype=np.float32)
    out = integrate_cmd(pos, vel, acc, 1.5, 1.0, 130.0)
    assert out[0, 0] == 0.0        # into-wall component zeroed
    assert out[0, 1] == 1.0        # tangential slide survives
    # Next tick, accelerating away works immediately.
    out2 = integrate_cmd(pos, out, np.array([[1.0, 0.0]], np.float32),
                         1.5, 1.0, 130.0)
    assert out2[0, 0] == 1.0
