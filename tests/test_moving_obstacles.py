"""
tests/test_moving_obstacles.py — patrolling (reciprocating) obstacles.

A fraction of obstacles move back-and-forth along one axis, bouncing
off the arena walls (deterministic, no response to blues).  They reuse
the existing per-agent obstacle crash penalty; blues are never
destroyed.  Everything defaults OFF (static), byte-preserving prior
behaviour.

Run:
    pytest tests/test_moving_obstacles.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red


def _env(**kw):
    base = dict(
        n_blue=3, n_red=2, n_obstacles=4, arena_size=100.0, max_steps=200,
        sensor_radius=40.0, use_belief_maps=True, red_policy=stationary_red,
    )
    base.update(kw)
    return PursuitEnv(**base)


def _acts(env):
    return {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}


# ---------------------------------------------------------------------------
# Defaults / setup
# ---------------------------------------------------------------------------

def test_default_obstacles_are_static():
    env = _env()   # no motion knobs
    env.reset(seed=0)
    assert np.all(env._obstacle_vel == 0.0)
    p0 = env._obstacle_pos.copy()
    for _ in range(10):
        env.step(_acts(env))
    assert np.allclose(env._obstacle_pos, p0)


def test_fraction_controls_number_moving():
    env = _env(moving_obstacle_fraction=0.5, obstacle_speed=3.0)
    env.reset(seed=1)
    n_move = int(np.count_nonzero(np.any(env._obstacle_vel != 0.0, axis=1)))
    assert n_move == round(0.5 * len(env._obstacle_pos))
    # Movers travel along exactly one axis.
    for v in env._obstacle_vel[np.any(env._obstacle_vel != 0, axis=1)]:
        assert np.count_nonzero(v) == 1
        assert abs(abs(v[np.nonzero(v)[0][0]]) - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Kinematics: bounce + in-bounds
# ---------------------------------------------------------------------------

def test_obstacles_stay_in_bounds_and_bounce():
    env = _env(moving_obstacle_fraction=1.0, obstacle_speed=5.0)
    env.reset(seed=2)
    L = env.arena_size
    v_start = env._obstacle_vel.copy()
    flipped = np.zeros(len(env._obstacle_pos), dtype=bool)
    for _ in range(200):
        env.step(_acts(env))
        lo = env._obstacle_r[:, None]
        hi = L - env._obstacle_r[:, None]
        assert np.all(env._obstacle_pos >= lo - 1e-3)
        assert np.all(env._obstacle_pos <= hi + 1e-3)
        flipped |= np.any(np.sign(env._obstacle_vel) != np.sign(v_start), axis=1)
    # Over 200 steps at speed 5 every mover must have bounced at least once.
    assert flipped.all()


def test_moving_obstacle_grid_tracks_motion():
    """The belief-truth obstacle grid must be recomputed as obstacles
    move (it was cached-once for static obstacles)."""
    env = _env(moving_obstacle_fraction=1.0, obstacle_speed=5.0)
    env.reset(seed=3)
    g0 = env._obstacle_grid.copy()
    for _ in range(10):
        env.step(_acts(env))
    assert not np.array_equal(g0, env._obstacle_grid)


# ---------------------------------------------------------------------------
# Features: obstacle velocity reaches the graph
# ---------------------------------------------------------------------------

def test_seen_moving_obstacle_track_carries_velocity():
    env = _env(moving_obstacle_fraction=1.0, obstacle_speed=4.0,
               sensor_pos_noise_std=0.0)
    env.reset(seed=4)
    mov = 0
    # Park a blue on top of the moving obstacle so it's a live track.
    env._blue_pos[0] = env._obstacle_pos[mov].copy()
    pos, vel, conf, _r, _i = env._build_obstacle_tracks()
    i = int(np.linalg.norm(pos - env._obstacle_pos[mov], axis=1).argmin())
    assert conf[i] == 1.0
    assert np.allclose(vel[i], env._obstacle_vel[mov])   # own-radar Doppler


def test_unseen_obstacle_track_has_zero_velocity():
    env = _env(moving_obstacle_fraction=1.0, obstacle_speed=4.0)
    env.reset(seed=4)
    env._blue_pos[:] = np.array([1.0, 1.0])   # far from every obstacle
    # (obstacles are spawned with wall clearance, so none sit at the corner)
    _pos, vel, _conf, _r, _i = env._build_obstacle_tracks()
    assert np.allclose(vel, 0.0)              # memory tracks carry no Doppler


# ---------------------------------------------------------------------------
# Crash penalty + belief decay
# ---------------------------------------------------------------------------

def test_moving_obstacle_sweeps_into_blue_triggers_crash():
    env = _env(moving_obstacle_fraction=1.0, obstacle_speed=3.0,
               crash_obstacle_penalty=2.0)
    env.reset(seed=1)
    mov = 0
    d = env._obstacle_vel[mov] / np.linalg.norm(env._obstacle_vel[mov])
    # Park blue 0 just ahead of the obstacle's path; others clear.
    env._blue_pos[0] = env._obstacle_pos[mov] + d * (env._obstacle_r[mov] + 6.0)
    env._blue_pos[1] = np.array([2.0, 2.0])
    env._blue_pos[2] = np.array([2.0, 98.0])
    hits = 0
    for _ in range(6):
        _, _, _, _, info = env.step(_acts(env))
        hits += info["blue_0"]["obstacle_crashes"]
    assert hits >= 1


def test_obstacle_belief_decay_fades_channel():
    env = _env(moving_obstacle_fraction=1.0, obstacle_speed=4.0,
               obstacle_belief_decay=0.5)
    env.reset(seed=5)
    # Seed the obstacle channel with a few detection steps.
    for _ in range(5):
        env.step(_acts(env))
    before = float(np.abs(env._belief_maps[1]).sum())
    # Freeze sensing far away so only the predict-step decay acts.
    env._blue_pos[:] = np.array([1.0, 1.0])
    env.sensor_radius = 0.1
    env._predict_enemy_belief()
    after = float(np.abs(env._belief_maps[1]).sum())
    assert after < before
