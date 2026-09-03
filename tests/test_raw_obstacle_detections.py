"""
tests/test_raw_obstacle_detections.py — identity-free obstacle returns.

``_build_obstacle_tracks`` loops over TRUE obstacle indices and reports the
exact RADIUS when seen — no measurement noise at all — on top of the same
perfect association ``raw_detections`` already fixed for reds.
``raw_obstacle_detections()`` emits what a real sensor would: position,
radius, Doppler, line of sight and per-return sigmas, with NO obstacle
label.

Run:
    pytest tests/test_raw_obstacle_detections.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red


def _env(**kw):
    base = dict(
        n_blue=3, n_red=1, n_obstacles=2, arena_size=130.0, max_steps=100,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
        sensor_noise_range_growth=1.0, red_policy=stationary_red, seed=0,
    )
    base.update(kw)
    env = PursuitEnv(**base)
    env.reset(seed=base["seed"])
    return env


def _clear(env, dist=15.0):
    """One obstacle in the open, `dist` from blue 0; the other obstacle and
    every other blue parked far away."""
    env._obstacle_pos = np.array([[65.0, 65.0], [10.0, 120.0]], dtype=np.float32)
    env._obstacle_r = np.array([8.0, 4.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = env._obstacle_pos[0] + np.array([dist, 0.0], dtype=np.float32)
    return env


def test_schema_and_no_identity_leak():
    env = _clear(_env(track_detection=False))
    dets = env.raw_obstacle_detections()
    assert len(dets) >= 1
    d = dets[0]
    assert set(d) == {"blue", "z_pos", "z_range", "z_radius", "z_radial",
                      "los", "sigma_pos", "sigma_radius", "sigma_radial",
                      "truth_id"}
    assert d["blue"] == 0
    assert np.isclose(np.linalg.norm(d["los"]), 1.0, atol=1e-5)


def test_radius_is_exact_by_default():
    """obstacle_radius_noise_std = 0 (default): radius is exact, matching
    _build_obstacle_tracks's pre-existing behaviour, and sigma_radius is
    reported as 0 so a consumer can tell."""
    env = _clear(_env(track_detection=False))
    d = env.raw_obstacle_detections()[0]
    assert np.isclose(d["z_radius"], env._obstacle_r[0], atol=1e-6)
    assert d["sigma_radius"] == 0.0


def test_radius_noise_when_enabled():
    env = _clear(_env(track_detection=False, obstacle_radius_noise_std=1.0))
    seen = [env.raw_obstacle_detections()[0]["z_radius"] for _ in range(40)]
    assert not np.allclose(seen[0], seen[1]), "radius is not being noised"
    assert np.isclose(np.mean(seen), env._obstacle_r[0], atol=0.6)   # unbiased
    d = env.raw_obstacle_detections()[0]
    assert d["sigma_radius"] > 0.0


def test_radius_noise_grows_with_range():
    # dist is from the CENTRE (_clear's convention); obstacle 0's radius is
    # 8 m, so both values must clear that to keep the blue outside the disk.
    near = _clear(_env(track_detection=False, obstacle_radius_noise_std=1.0),
                 dist=13.0).raw_obstacle_detections()[0]
    far = _clear(_env(track_detection=False, obstacle_radius_noise_std=1.0),
                dist=38.0).raw_obstacle_detections()[0]
    assert far["sigma_radius"] > near["sigma_radius"]
    assert far["sigma_pos"] > near["sigma_pos"]


def test_doppler_near_zero_for_a_static_obstacle():
    env = _clear(_env(track_detection=False, sensor_vel_noise_std=0.0))
    d = env.raw_obstacle_detections()[0]
    assert np.isclose(d["z_radial"], 0.0, atol=1e-5)


def test_doppler_reflects_a_moving_obstacle():
    env = _env(n_obstacles=1, track_detection=False, sensor_vel_noise_std=0.0)
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r = np.array([8.0], dtype=np.float32)
    env._obstacle_vel = np.array([[0.6, 0.0]], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = np.array([50.0, 65.0], dtype=np.float32)   # due west
    d = env.raw_obstacle_detections()[0]
    # LOS points from blue toward the obstacle (east, +x): radial speed
    # should read the full +0.6, since velocity is along the LOS.
    assert np.isclose(d["z_radial"], 0.6, atol=1e-4)


def test_same_detection_chain_as_the_track_builder():
    """Range gate, near-surface occlusion and the p_TP_obstacle draw must
    match _build_obstacle_tracks, or the tracker is scored in a different
    sensor regime."""
    env = _clear(_env(track_detection=False), dist=60.0)
    assert env.raw_obstacle_detections() == []

    # Two obstacles from the start, so _obstacle_vel stays correctly sized:
    # a second, LARGER obstacle sits directly between blue 0 and obstacle 0.
    env = _env(n_obstacles=2, track_detection=False)
    env._obstacle_pos = np.array([[65.0, 65.0], [42.0, 65.0]], dtype=np.float32)
    env._obstacle_r = np.array([10.0, 15.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = np.array([20.0, 65.0], dtype=np.float32)
    dets = env.raw_obstacle_detections()
    assert all(d["truth_id"] != 0 for d in dets if d["blue"] == 0), (
        "the far obstacle should be occluded by the nearer, larger one")

    env = _clear(_env(p_TP=0.5, track_detection=True))
    hits = sum(int(len(env.raw_obstacle_detections()) > 0) for _ in range(2000))
    assert abs(hits / 2000 - 0.5) < 0.05


def test_one_return_per_detecting_blue():
    env = _env(n_obstacles=1, track_detection=False)
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r = np.array([8.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = np.array([50.0, 65.0], dtype=np.float32)
    env._blue_pos[1] = np.array([65.0, 50.0], dtype=np.float32)
    dets = env.raw_obstacle_detections()
    assert sorted(d["blue"] for d in dets) == [0, 1]


def test_no_obstacles_returns_empty():
    env = _env(n_obstacles=0, track_detection=False)
    assert env.raw_obstacle_detections() == []


def test_runs_over_a_full_episode():
    env = _env(n_blue=5, n_red=2, n_obstacles=4, track_detection=True)
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(60):
        if not env.agents:
            break
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                  for a in env.agents})
        for d in env.raw_obstacle_detections():
            total += 1
            assert np.all(np.isfinite(d["z_pos"]))
            assert 0 <= d["truth_id"] < env.n_obstacles
    assert total > 0, "no obstacle detections at all over an episode"
