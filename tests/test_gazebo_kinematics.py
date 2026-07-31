"""
tests/test_gazebo_kinematics.py — the bridge's motion contract must be
TRAJECTORY-IDENTICAL to the env's.

``gazebo/kinematics.integrate_cmd`` re-implements the env's per-step
motion rule (per-axis v_max clip + wall arrival + obstacle rollback)
for the Gazebo bridge.  Any drift is a silent sim-to-sim parity leak
the policy would experience as a world subtly different from the one
it trained in, so we march multi-step trajectories through BOTH
implementations and require identical positions and velocities at
every step:

    env    : pos, vel = _integrate(...); then
             pos, vel, _ = _clip_positions_from_obstacles(...)
    bridge : exec, state, hit = integrate_cmd(...);
             pos += exec * dt; vel = state

(The bridge splits each step into exec_vel — what Gazebo executes to
land on the env's next position — and state_vel — the env's post-step
velocity.  Equality of the resulting trajectories is exactly the
faithfulness claim.)
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.env.pursuit_env import PursuitEnv
from gazebo.kinematics import integrate_cmd


@pytest.fixture(scope="module")
def env():
    e = PursuitEnv(n_blue=3, n_red=2, arena_size=130.0, n_obstacles=4,
                   seed=0)
    e.reset(seed=0)
    return e


def _env_step(env, pos, vel, acc, v_max):
    new_pos, new_vel = env._integrate(pos.copy(), vel.copy(), acc, v_max)
    new_pos, new_vel, hit = env._clip_positions_from_obstacles(
        new_pos, pos.copy(), new_vel)
    return new_pos, new_vel, hit


def _bridge_step(env, pos, vel, acc, v_max):
    exec_, state, hit = integrate_cmd(
        pos.copy(), vel.copy(), acc, v_max, env.dt, env.arena_size,
        env._obstacle_pos, env._obstacle_r)
    return (pos + exec_ * env.dt).astype(np.float32), state, hit


def test_trajectories_identical_under_fuzz(env):
    """60 random 40-step trajectories, biased to hit walls and pillars."""
    rng = np.random.default_rng(11)
    for traj in range(60):
        K = 4
        # Start some trajectories right next to obstacles/walls.
        pos = rng.uniform(0.0, 130.0, size=(K, 2)).astype(np.float32)
        if traj % 3 == 1:
            k = traj % env.n_obstacles
            pos[0] = env._obstacle_pos[k] + env._obstacle_r[k] + 1.0
        if traj % 3 == 2:
            pos[0] = (1.0, 65.0)
        vel = np.zeros((K, 2), dtype=np.float32)
        p_e, v_e = pos.copy(), vel.copy()
        p_b, v_b = pos.copy(), vel.copy()
        v_max = float(rng.choice([1.0, 1.5]))
        for _ in range(40):
            acc = rng.uniform(-1.0, 1.0, size=(K, 2)).astype(np.float32)
            p_e, v_e, hit_e = _env_step(env, p_e, v_e, acc, v_max)
            p_b, v_b, hit_b = _bridge_step(env, p_b, v_b, acc, v_max)
            np.testing.assert_allclose(p_b, p_e, atol=1e-4)
            np.testing.assert_allclose(v_b, v_e, atol=1e-4)
            assert np.array_equal(hit_b, hit_e)


def test_wall_arrival_lands_exactly_on_boundary(env):
    """Env semantics: an agent 1 m from the wall at full speed lands ON
    the wall (position clipped to 0) with that velocity component
    zeroed.  The bridge must reproduce both, via a partial-speed
    exec_vel and a zeroed state_vel."""
    pos = np.array([[1.0, 50.0]], dtype=np.float32)
    vel = np.array([[-1.5, 1.0]], dtype=np.float32)
    acc = np.array([[-1.0, 0.0]], dtype=np.float32)
    exec_, state, _ = integrate_cmd(pos, vel, acc, 1.5, 1.0, 130.0)
    assert exec_[0, 0] == -1.0          # partial approach: lands at x=0
    assert state[0, 0] == 0.0           # env's stored velocity
    assert state[0, 1] == 1.0           # tangential slide survives
    # Next tick, accelerating away works immediately (no bounce, no
    # stickiness) — the env's "agent learns to back off" contract.
    pos2 = pos + exec_ * 1.0
    assert pos2[0, 0] == 0.0
    exec2, state2, _ = integrate_cmd(
        pos2, state, np.array([[1.0, 0.0]], np.float32), 1.5, 1.0, 130.0)
    assert exec2[0, 0] == 1.0


def test_obstacle_entry_cancels_whole_move(env):
    """Env semantics: a move landing inside a pillar rolls back
    entirely and zeroes the WHOLE velocity (soft-crash)."""
    c = env._obstacle_pos[0]
    r = float(env._obstacle_r[0])
    pos = (c + np.array([r + 0.5, 0.0])).reshape(1, 2).astype(np.float32)
    vel = np.array([[-1.5, 0.3]], dtype=np.float32)
    acc = np.array([[-1.0, 0.0]], dtype=np.float32)
    exec_, state, hit = integrate_cmd(
        pos, vel, acc, 1.5, 1.0, 130.0, env._obstacle_pos, env._obstacle_r)
    assert bool(hit[0])
    assert np.all(exec_[0] == 0.0)      # no movement at all
    assert np.all(state[0] == 0.0)      # both components zeroed
