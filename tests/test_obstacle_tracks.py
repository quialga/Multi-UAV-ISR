"""
tests/test_obstacle_tracks.py — verifies the obstacle live-sensor
refinement (crash-avoidance follow-up).

Before: the actor's obstacle node position was always the belief-map
PEAK — grid-quantised to ~half a cell.  Now, when a blue currently
senses an obstacle, the actor gets the precise OWN-RADAR position
(true centre + sensor noise); only when NO blue senses it does it fall
back to the coarse peak.  This lets the policy hug obstacle boundaries
tightly instead of keeping a quantisation-sized safety margin.

Run:
    pytest tests/test_obstacle_tracks.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red


def _seeded_env(noise=0.0, seed=2):
    env = PursuitEnv(
        n_blue=3, n_red=2, n_obstacles=4, arena_size=130.0, max_steps=40,
        sensor_radius=40.0, use_belief_maps=True,
        sensor_pos_noise_std=noise, red_policy=stationary_red,
        # These tests measure POSITION precision of a seen track; pin the
        # p_TP draw and the range-based confidence so a random radar miss
        # (or an SNR-scaled conf) cannot flake them.
        track_detection=False, track_conf_min=1.0,
        sensor_noise_range_growth=0.0,
    )
    env.reset(seed=seed)
    # A few steps so the obstacle belief channel is populated (the
    # unseen-fallback peaks need a seeded map).
    acts = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    for _ in range(5):
        env.step(acts)
    return env


def _nearest_track(tracks, target):
    d = np.linalg.norm(tracks - target, axis=1)
    i = int(d.argmin())
    return i, float(d[i])


def test_seen_obstacle_uses_precise_position_not_peak():
    """A blue parked next to obstacle 0 -> the obstacle track equals its
    TRUE centre to sub-cell precision (noise off), with confidence 1.0,
    and is strictly better than the grid-quantised cell centre the old
    peak path would have produced."""
    env = _seeded_env(noise=0.0)
    o0 = env._obstacle_pos[0].copy()
    cs = env.belief_cell_size

    # Blue 0 within sensor range of obstacle 0; the others far away.
    env._blue_pos[0] = o0 + np.array([env._obstacle_r[0] + 3.0, 0.0])
    env._blue_pos[1] = np.array([120.0, 120.0])
    env._blue_pos[2] = np.array([120.0, 5.0])

    pos, _vel, conf, _r, _i = env._build_obstacle_tracks()
    i, err = _nearest_track(pos, o0)

    assert err < 1e-4, f"seen obstacle not refined to true centre (err={err})"
    assert conf[i] == 1.0

    # Refinement genuinely beats grid quantisation: the true centre does
    # not sit on a cell centre for this seed, so the precise track differs
    # from the quantised peak the old path would have emitted.
    quantised = (np.floor(o0 / cs) + 0.5) * cs
    assert np.linalg.norm(pos[i] - quantised) > 1e-3


def test_unseen_obstacle_falls_back_to_coarse_peak():
    """With every blue far from obstacle 0, its track comes from the
    belief peak and is only grid-accurate (materially worse than the
    precise position a live sensor would give)."""
    env = _seeded_env(noise=0.0)
    o0 = env._obstacle_pos[0].copy()
    cs = env.belief_cell_size

    env._blue_pos[:] = np.array([120.0, 120.0])   # all blues far from o0
    pos, _vel, conf, _r, _i = env._build_obstacle_tracks()
    _, err = _nearest_track(pos, o0)

    # A grid peak cannot be sub-cell accurate; the precise path (other
    # test) hits 0.  Half a cell diagonal is the tightest a peak can be.
    assert err > 0.5 * cs


def test_seen_beats_unseen_precision():
    """Same obstacle, same map: precise (seen) position is strictly
    closer to truth than the peak (unseen) fallback."""
    env = _seeded_env(noise=0.0)
    o0 = env._obstacle_pos[0].copy()

    env._blue_pos[0] = o0 + np.array([env._obstacle_r[0] + 3.0, 0.0])
    env._blue_pos[1] = np.array([120.0, 120.0])
    env._blue_pos[2] = np.array([120.0, 5.0])
    seen_pos, _, _, _, _ = env._build_obstacle_tracks()
    _, seen_err = _nearest_track(seen_pos, o0)

    env._blue_pos[:] = np.array([120.0, 120.0])
    unseen_pos, _, _, _, _ = env._build_obstacle_tracks()
    _, unseen_err = _nearest_track(unseen_pos, o0)

    assert seen_err < unseen_err


def test_sensor_noise_bounds_seen_position():
    """With noise on, the seen obstacle track is a noisy measurement of
    the true centre — tight (within a few sigma), far better than the
    ~half-cell peak quantisation."""
    sigma = 1.0
    env = _seeded_env(noise=sigma)
    o0 = env._obstacle_pos[0].copy()

    env._blue_pos[0] = o0 + np.array([env._obstacle_r[0] + 3.0, 0.0])
    env._blue_pos[1] = np.array([120.0, 120.0])
    env._blue_pos[2] = np.array([120.0, 5.0])

    errs = []
    for _ in range(50):
        pos, _vel, conf, _r, _i = env._build_obstacle_tracks()
        i, err = _nearest_track(pos, o0)
        assert conf[i] == 1.0
        errs.append(err)
    # Mean 2-D Gaussian offset magnitude is sigma*sqrt(pi/2) ~ 1.25;
    # allow generous slack but stay well under a coarse-peak error.
    assert np.mean(errs) < 3.0 * sigma


def test_no_obstacles_returns_empty_tracks():
    env = PursuitEnv(n_blue=3, n_red=2, n_obstacles=0, arena_size=130.0,
                     sensor_radius=40.0, use_belief_maps=True,
                     red_policy=stationary_red)
    env.reset(seed=0)
    pos, _vel, conf, _r, _i = env._build_obstacle_tracks()
    assert pos.shape == (0, 2)
    assert conf.shape == (0,)


# ---- Obstacle RADIUS feature (Q1: actor must see the surface location) ----

def test_seen_obstacle_track_reports_true_radius():
    """A blue parked next to obstacle 0 -> its live track carries the TRUE
    measured radius, not the command prior."""
    env = _seeded_env(noise=0.0)
    o0 = env._obstacle_pos[0].copy()
    env._blue_pos[0] = o0 + np.array([env._obstacle_r[0] + 3.0, 0.0])
    env._blue_pos[1] = np.array([120.0, 120.0])
    env._blue_pos[2] = np.array([120.0, 5.0])

    pos, _vel, conf, r, _i = env._build_obstacle_tracks()
    i, _ = _nearest_track(pos, o0)
    assert conf[i] == 1.0
    assert np.isclose(r[i], env._obstacle_r[0], atol=1e-4)


def test_unseen_obstacle_track_uses_command_prior_radius():
    """With every blue far from obstacle 0, its memory track reports the
    surveyed field's mean radius (command prior), not a true measurement."""
    env = _seeded_env(noise=0.0)
    o0 = env._obstacle_pos[0].copy()
    env._blue_pos[:] = np.array([120.0, 120.0])          # all blues far from o0
    prior = 0.5 * (env.obstacle_radius_min + env.obstacle_radius_max)

    pos, _vel, conf, r, _i = env._build_obstacle_tracks()
    i, _ = _nearest_track(pos, o0)
    assert conf[i] < 1.0                                  # memory, not live
    assert np.isclose(r[i], prior, atol=1e-4)


def test_obstacle_node_feature_is_2d_with_normalised_radius():
    """Both actor and critic obstacle node features are [placed/conf,
    radius/arena_size]; the critic's radius channel is exact per slot."""
    env = _seeded_env(noise=0.0)
    obs = env.structured_belief_observation()
    ob  = obs["obstacle_features"]
    tob = obs["true_obstacle_features"]
    assert ob.shape[-1] == 2 and tob.shape[-1] == 2
    # Critic true graph is in obstacle order: slot o == obstacle o.
    for o in range(len(env._obstacle_pos)):
        assert tob[o, 0] == 1.0                            # placed
        assert np.isclose(tob[o, 1], env._obstacle_r[o] / env.arena_size,
                          atol=1e-6)
    # Radius channel is normalised into a sane range.
    placed = tob[:, 0] > 0.5
    assert np.all(tob[placed, 1] > 0.0) and np.all(tob[placed, 1] < 1.0)
