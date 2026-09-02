"""
tests/test_raw_detections.py — identity-free sensor returns.

``_build_enemy_tracks`` loops over TRUE red indices, measures
``self._red_pos[r]``, and groups returns for fusion by ``detect[:, r]``.
That hands the policy a PERFECT data association — which return belongs to
which target, plus a slot->target mapping stable across steps.  Real radar
gives none of that; establishing identity IS the tracking problem.

``raw_detections()`` emits what a real sensor would, so an actual tracker
can be built and scored against ground truth.  These tests pin the two
properties that matter: it runs the SAME detection chain as the track
builder, and it leaks no identity beyond the clearly-marked evaluation key.

Run:
    pytest tests/test_raw_detections.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.agents.heuristics import stationary_red


def _env(**kw):
    base = dict(
        n_blue=3, n_red=2, n_obstacles=1, arena_size=130.0, max_steps=100,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
        sensor_noise_range_growth=1.0, red_policy=stationary_red, seed=0,
    )
    base.update(kw)
    env = PursuitEnv(**base)
    env.reset(seed=base["seed"])
    return env


def _clear(env, dist=15.0):
    """One red in the open, `dist` from blue 0; obstacle and other blues far."""
    env._obstacle_pos = np.array([[10.0, 120.0]], dtype=np.float32)
    env._obstacle_r = np.array([4.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._red_active[:] = False
    env._red_active[0] = True
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._red_pos[0] = np.array([50.0 + dist, 50.0], dtype=np.float32)
    env._red_vel[0] = np.array([0.6, -0.3], dtype=np.float32)
    return env


def test_schema_and_no_identity_leak():
    """A detection carries only what a sensor knows.  The single identity
    field is namespaced as evaluation-only."""
    env = _clear(_env(track_detection=False))
    dets = env.raw_detections()
    assert len(dets) >= 1
    d = dets[0]
    assert set(d) == {"blue", "z_pos", "z_range", "z_radial", "los",
                      "sigma_pos", "sigma_radial", "truth_id"}
    # Everything except truth_id must be sensor-derivable.
    assert d["blue"] == 0                       # our own UAV: known
    assert d["z_pos"].shape == (2,)
    assert d["los"].shape == (2,)
    assert np.isclose(np.linalg.norm(d["los"]), 1.0, atol=1e-5)


def test_detections_are_noisy_measurements_not_truth():
    env = _clear(_env(track_detection=False))
    seen = [env.raw_detections()[0]["z_pos"].copy() for _ in range(30)]
    seen = np.stack(seen)
    assert not np.allclose(seen[0], seen[1]), "position is not being noised"
    # Unbiased around the true position.
    assert np.allclose(seen.mean(axis=0), env._red_pos[0], atol=0.8)


def test_radial_velocity_is_the_los_projection():
    """A sensor measures Doppler along its own line of sight — not the full
    2-D velocity."""
    env = _clear(_env(track_detection=False, sensor_vel_noise_std=0.0,
                      sensor_noise_range_growth=0.0))
    d = env.raw_detections()[0]
    delta = env._red_pos[0] - env._blue_pos[0]
    u = delta / np.linalg.norm(delta)
    assert np.isclose(d["z_radial"], float(env._red_vel[0] @ u), atol=1e-4)


def test_same_detection_chain_as_the_track_builder():
    """Range gate, occlusion and the p_TP draw must match the track path,
    or the tracker would be scored in a different sensor regime."""
    # Out of range -> nothing.
    env = _clear(_env(track_detection=False), dist=60.0)
    assert env.raw_detections() == []

    # Occluded -> nothing, even well inside range.
    env = _env(track_detection=False)
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r = np.array([10.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._red_active[:] = False
    env._red_active[0] = True
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = np.array([45.0, 65.0], dtype=np.float32)
    env._red_pos[0] = np.array([85.0, 65.0], dtype=np.float32)
    assert env.raw_detections() == []

    # p_TP governs the rate.
    env = _clear(_env(p_TP=0.5, track_detection=True))
    hits = sum(int(len(env.raw_detections()) > 0) for _ in range(2000))
    assert abs(hits / 2000 - 0.5) < 0.05


def test_noise_sigmas_are_reported_and_grow_with_range():
    """A tracker needs the per-return covariance, and it must reflect the
    same SNR falloff the rest of the sensor model uses."""
    near = _clear(_env(track_detection=False), dist=5.0).raw_detections()[0]
    far = _clear(_env(track_detection=False), dist=38.0).raw_detections()[0]
    assert far["sigma_pos"] > near["sigma_pos"]
    assert far["sigma_radial"] > near["sigma_radial"]
    assert near["sigma_pos"] > 0.0


def test_one_return_per_detecting_blue():
    """Two blues that both see the target produce TWO returns — the raw
    material the velocity fusion needs, and the association problem the
    tracker has to solve."""
    env = _env(track_detection=False)
    env._obstacle_pos = np.array([[10.0, 120.0]], dtype=np.float32)
    env._obstacle_r = np.array([4.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._red_active[:] = False
    env._red_active[0] = True
    tgt = np.array([65.0, 65.0], dtype=np.float32)
    env._red_pos[0] = tgt
    env._blue_pos[:] = np.array([120.0, 10.0], dtype=np.float32)
    env._blue_pos[0] = tgt + np.array([15.0, 0.0], dtype=np.float32)
    env._blue_pos[1] = tgt + np.array([0.0, 15.0], dtype=np.float32)
    dets = env.raw_detections()
    assert sorted(d["blue"] for d in dets) == [0, 1]
    # Non-collinear lines of sight -> the pair determines the full velocity.
    los = np.stack([d["los"] for d in dets])
    assert abs(float(los[0] @ los[1])) < 0.9


def test_caught_and_inactive_reds_produce_nothing():
    env = _clear(_env(track_detection=False))
    env._red_active[:] = False
    assert env.raw_detections() == []


def test_runs_over_a_full_episode():
    env = _env(n_blue=5, n_red=3, n_obstacles=4,
               red_policy=run_from_nearest_uav, track_detection=True)
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(60):
        if not env.agents:
            break
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                  for a in env.agents})
        for d in env.raw_detections():
            total += 1
            assert np.all(np.isfinite(d["z_pos"]))
            assert env._red_active[d["truth_id"]], "return from a dead red"
    assert total > 0, "no detections at all over an episode"


# ---- Clutter (false returns) ---------------------------------------------

def test_no_clutter_by_default():
    """clutter_rate = 0 keeps raw_detections byte-identical to before."""
    env = _clear(_env(track_detection=False))
    for _ in range(50):
        assert all(d["truth_id"] >= 0 for d in env.raw_detections())


def test_clutter_rate_is_the_poisson_mean_per_blue():
    """Each blue emits Poisson(clutter_rate) false plots per scan."""
    rate = 2.0
    env = _env(n_blue=3, n_obstacles=0, track_detection=False,
               clutter_rate=rate)
    env._red_active[:] = False                 # isolate: no real returns
    env._blue_pos[:] = np.array([65.0, 65.0], dtype=np.float32)
    counts = []
    for _ in range(400):
        dets = env.raw_detections()
        assert all(d["truth_id"] == -1 for d in dets)
        counts.append(len(dets))
    # 3 blues x rate each; sampling is in-disk and arena-clipped, so the
    # observed mean is a slight under-count.
    assert 0.8 * 3 * rate < np.mean(counts) <= 3 * rate * 1.05


def test_clutter_is_a_well_formed_return():
    """A false plot must be indistinguishable from a real one to a tracker:
    same fields, unit LOS, positive sigmas, position on the arena."""
    env = _env(n_blue=1, n_obstacles=0, track_detection=False,
               clutter_rate=8.0)
    env._red_active[:] = False
    env._blue_pos[0] = np.array([65.0, 65.0], dtype=np.float32)
    dets = env.raw_detections()
    assert dets, "expected clutter"
    for d in dets:
        assert set(d) == {"blue", "z_pos", "z_range", "z_radial", "los",
                          "sigma_pos", "sigma_radial", "truth_id"}
        assert np.isclose(np.linalg.norm(d["los"]), 1.0, atol=1e-5)
        assert d["sigma_pos"] > 0.0
        assert 0.0 <= d["z_pos"][0] <= env.arena_size
        assert 0.0 <= d["z_pos"][1] <= env.arena_size
        assert d["z_range"] <= env.sensor_radius * 1.2


def test_clutter_lies_inside_the_sensor_disk():
    env = _env(n_blue=1, n_obstacles=0, track_detection=False,
               clutter_rate=6.0)
    env._red_active[:] = False
    bp = np.array([65.0, 65.0], dtype=np.float32)
    env._blue_pos[0] = bp
    for _ in range(40):
        for d in env.raw_detections():
            assert np.linalg.norm(d["z_pos"] - bp) <= env.sensor_radius + 1e-4


def test_clutter_respects_occlusion():
    """A false alarm still needs line of sight — it is a threshold crossing
    in a real beam, not a ghost that sees through walls."""
    env = _env(n_blue=1, n_obstacles=1, track_detection=False,
               clutter_rate=40.0, sensor_radius=40.0)
    env._red_active[:] = False
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r = np.array([14.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    bp = np.array([40.0, 65.0], dtype=np.float32)
    env._blue_pos[0] = bp
    n = 0
    for _ in range(40):
        for d in env.raw_detections():
            n += 1
            assert not bool(env._rays_occluded_by_obstacles(
                bp, d["z_pos"][None, :])[0]), "clutter behind an obstacle"
    assert n > 0


def test_clutter_doppler_is_meaningless_but_present():
    """A false alarm carries a Doppler (it IS a range-Doppler cell crossing),
    it just means nothing — so it must not be a constant zero."""
    env = _env(n_blue=1, n_obstacles=0, track_detection=False,
               clutter_rate=10.0)
    env._red_active[:] = False
    env._blue_pos[0] = np.array([65.0, 65.0], dtype=np.float32)
    vals = [d["z_radial"] for _ in range(30) for d in env.raw_detections()]
    assert len(vals) > 20
    assert np.std(vals) > 0.1, "clutter Doppler is degenerate"
