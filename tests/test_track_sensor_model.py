"""
tests/test_track_sensor_model.py — live tracks obey the SAME sensor model
as the belief map.

The live-track gate used to be range-only, so a target inside
``sensor_radius`` produced a ``conf = 1.0`` measurement EVERY step and even
THROUGH obstacles — while the belief map, fed by the *same* radar, applied
``p_TP``/``p_FP`` and an exact line-of-sight test.  A real radar is one
chain: detect (P_d) -> measure (position + Doppler, both noisy).  You
cannot miss the detection and still report an accurate position.

Covered here:
  * occlusion gate      — no live track through a wall
  * detection draw      — a red in the clear is detected ~p_TP of scans,
                          and a miss falls back to the memory path
  * Doppler noise       — velocity is measured, not exact
  * range-based conf    — SNR proxy, depends only on range (never on
                          true-vs-false, which would leak ground truth)
  * range-based noise    — the accuracy half of the same SNR falloff
  * pre-fix escape hatch — the old behaviour is still reproducible

Run:
    pytest tests/test_track_sensor_model.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.agents.heuristics import stationary_red


def _env(**kw):
    base = dict(
        n_blue=1, n_red=1, n_obstacles=1, arena_size=130.0, max_steps=100,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        sensor_pos_noise_std=1.0, red_policy=stationary_red, seed=0,
    )
    base.update(kw)
    env = PursuitEnv(**base)
    env.reset(seed=base["seed"])
    return env


def _hidden_geometry(env):
    """Blue at (45,65), obstacle at (65,65) r=10, red at (85,65): the red is
    40 m away (inside sensor range) but fully behind the obstacle."""
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r = np.array([10.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[0] = np.array([45.0, 65.0], dtype=np.float32)
    env._red_pos[0] = np.array([85.0, 65.0], dtype=np.float32)
    return env


def _clear_geometry(env, dist=20.0):
    """Red in the clear, `dist` m from the blue, obstacle parked far away."""
    env._obstacle_pos = np.array([[10.0, 120.0]], dtype=np.float32)
    env._obstacle_r = np.array([5.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[0] = np.array([40.0, 40.0], dtype=np.float32)
    env._red_pos[0] = np.array([40.0 + dist, 40.0], dtype=np.float32)
    return env


# --------------------------------------------------------------------- #
#  Occlusion gate
# --------------------------------------------------------------------- #

def test_no_live_track_through_an_obstacle():
    """A red hidden behind a wall must NOT produce a live track, even though
    it is well inside sensor range."""
    env = _hidden_geometry(_env(track_detection=False))   # isolate occlusion
    assert bool(env._rays_occluded_by_obstacles(
        env._blue_pos[0], env._red_pos[0][None, :])[0])
    _pos, conf, red, _tv, _tc = env._build_enemy_tracks()
    assert red[0] == -1, "occluded red still produced a LIVE track"
    assert conf[0] < 1.0, "occluded red reported full confidence"


def test_pre_fix_sees_through_walls():
    """Escape hatch: the old range-only behaviour is reproducible."""
    env = _hidden_geometry(_env(track_occlusion=False, track_detection=False))
    _pos, conf, red, _tv, _tc = env._build_enemy_tracks()
    assert red[0] == 0 and conf[0] == 1.0


def test_clear_line_of_sight_still_tracks():
    """The occlusion gate must not suppress a red in the open."""
    env = _clear_geometry(_env(track_detection=False))
    _pos, conf, red, _tv, _tc = env._build_enemy_tracks()
    assert red[0] == 0 and conf[0] > 0.0


# --------------------------------------------------------------------- #
#  Detection draw (p_TP)
# --------------------------------------------------------------------- #

def test_detection_rate_matches_p_TP():
    """A red in the clear is reported on ~p_TP of scans — the same statistic
    the belief map uses — not 100% of them."""
    env = _clear_geometry(_env(p_TP=0.85, track_conf_min=1.0))
    hits = sum(int(env._build_enemy_tracks()[2][0] == 0) for _ in range(4000))
    rate = hits / 4000
    assert abs(rate - 0.85) < 0.03, f"detection rate {rate:.3f} != p_TP 0.85"


def test_detection_miss_falls_back_to_memory_not_truth():
    """On a miss there is NO measurement: the slot must not carry a
    live track backed by the red (it coasts on the belief path)."""
    env = _clear_geometry(_env(p_TP=0.5))
    saw_miss = False
    for _ in range(200):
        _pos, _conf, red, _tv, _tc = env._build_enemy_tracks()
        if red[0] == -1:
            saw_miss = True
            break
    assert saw_miss, "never missed a detection at p_TP=0.5"


def test_pre_fix_detection_is_perfect():
    """Escape hatch: track_detection=False restores 100% detection."""
    env = _clear_geometry(_env(p_TP=0.5, track_detection=False))
    for _ in range(50):
        assert env._build_enemy_tracks()[2][0] == 0


# --------------------------------------------------------------------- #
#  Doppler noise
# --------------------------------------------------------------------- #

def test_doppler_velocity_is_noisy_but_unbiased():
    v_true = np.array([0.7, -0.4], dtype=np.float32)
    env = _env(sensor_vel_noise_std=0.1, sensor_noise_range_growth=0.0)
    meas = np.array([env._measured_vel(v_true) for _ in range(4000)])
    assert not np.allclose(meas[0], meas[1]), "velocity is not being noised"
    assert np.allclose(meas.mean(axis=0), v_true, atol=0.02)   # unbiased
    assert abs(meas.std(axis=0).mean() - 0.1) < 0.02           # right sigma


def test_zero_vel_noise_returns_exact_velocity():
    v = np.array([0.3, 0.2], dtype=np.float32)
    env = _env(sensor_vel_noise_std=0.0)
    assert np.allclose(env._measured_vel(v), v)


# --------------------------------------------------------------------- #
#  Range-based confidence (SNR proxy)
# --------------------------------------------------------------------- #

def test_confidence_falls_with_range():
    """Closer target -> stronger return -> higher confidence."""
    near = _clear_geometry(_env(track_detection=False, track_conf_min=0.5),
                           dist=5.0)._build_enemy_tracks()[1][0]
    far = _clear_geometry(_env(track_detection=False, track_conf_min=0.5),
                          dist=38.0)._build_enemy_tracks()[1][0]
    assert near > far > 0.0
    assert near <= 1.0 + 1e-6


def test_confidence_stays_strictly_positive():
    """conf == 0 is reserved for dead/padding slots, so a live track at max
    range must never reach it."""
    env = _clear_geometry(_env(track_detection=False, track_conf_min=0.5),
                          dist=39.9)
    conf = env._build_enemy_tracks()[1][0]
    assert conf >= 0.5 - 1e-6 and conf > 0.0


def test_conf_min_one_is_flat_confidence():
    """Escape hatch: track_conf_min=1.0 reproduces flat conf 1.0."""
    for dist in (5.0, 38.0):
        env = _clear_geometry(_env(track_detection=False, track_conf_min=1.0),
                              dist=dist)
        assert env._build_enemy_tracks()[1][0] == 1.0


def test_confidence_does_not_leak_true_vs_false():
    """Confidence must be a pure function of RANGE.  Two reds at the same
    range get the same confidence regardless of anything else about them —
    the property that stops conf from revealing ground truth."""
    env = _env(n_red=2, track_detection=False, track_conf_min=0.4)
    env._obstacle_pos = np.array([[10.0, 120.0]], dtype=np.float32)
    env._obstacle_r = np.array([5.0], dtype=np.float32)
    env._recompute_obstacle_grid()
    env._blue_pos[0] = np.array([65.0, 65.0], dtype=np.float32)
    env._red_pos[0] = np.array([65.0 + 20.0, 65.0], dtype=np.float32)
    env._red_pos[1] = np.array([65.0, 65.0 + 20.0], dtype=np.float32)
    conf = env._build_enemy_tracks()[1]
    assert np.isclose(conf[0], conf[1], atol=1e-6)


# --------------------------------------------------------------------- #
#  Range-based measurement noise (accuracy half of the SNR story)
# --------------------------------------------------------------------- #

def test_measurement_noise_grows_with_range():
    """Confidence and accuracy are two faces of the same SNR falloff: if a
    distant track is less trusted it must also be less accurate."""
    env = _env(sensor_pos_noise_std=1.0, sensor_noise_range_growth=1.0)
    R = env.sensor_radius
    assert np.isclose(env._noise_scale(0.0), 1.0)
    assert np.isclose(env._noise_scale(R), 2.0)        # doubles at max range
    assert env._noise_scale(R / 2) < env._noise_scale(R)

    p = np.zeros(2, dtype=np.float32)
    near = np.array([env._measured_pos(p, 0.0) for _ in range(3000)]).std()
    far = np.array([env._measured_pos(p, R) for _ in range(3000)]).std()
    assert far > near * 1.5, f"noise did not grow with range ({near}->{far})"


def test_noise_growth_off_is_range_independent():
    """Escape hatch: growth 0 => constant sigma at every range."""
    env = _env(sensor_pos_noise_std=1.0, sensor_noise_range_growth=0.0)
    assert np.isclose(env._noise_scale(0.0), 1.0)
    assert np.isclose(env._noise_scale(env.sensor_radius), 1.0)


# --------------------------------------------------------------------- #
#  Velocity is only reported by a blue that actually detected the target
# --------------------------------------------------------------------- #

def test_velocity_only_from_detecting_blue():
    """A blue that did not detect the red must report zero Doppler on its
    rb edge — it cannot measure what it never saw."""
    env = _hidden_geometry(_env(n_blue=1, track_detection=False))
    env._red_vel[0] = np.array([0.5, 0.5], dtype=np.float32)
    obs = env.structured_belief_observation()
    # Occluded -> no detection -> the rb edge carries no velocity signal.
    assert env._last_red_detect is not None
    assert not bool(env._last_red_detect[0, 0])
    assert np.allclose(obs["rb_edge_features"][0][2:4], 0.0, atol=1e-6)


def test_detection_mask_shape_and_reset():
    env = _clear_geometry(_env(track_detection=False))
    env._build_enemy_tracks()
    assert env._last_red_detect.shape == (env.n_blue, env.n_red)
    env._build_obstacle_tracks()
    assert env._last_obs_detect.shape == (env.n_blue, env.n_obstacles)


def test_full_step_runs_with_realistic_sensor():
    """End-to-end smoke: the realistic chain steps cleanly in a normal env."""
    env = PursuitEnv(
        n_blue=5, n_red=3, n_obstacles=4, arena_size=130.0, max_steps=60,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
        track_conf_min=0.5, sensor_noise_range_growth=1.0,
        red_policy=run_from_nearest_uav, seed=3,
    )
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(40):
        if not env.agents:
            break
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                  for a in env.agents})
    obs = env.structured_belief_observation()
    assert np.all(np.isfinite(obs["rb_edge_features"]))
    assert np.all(np.isfinite(obs["red_features"]))


def test_velocity_is_command_fused_and_shared():
    """Velocity is now a COMMAND-LAYER product, shared with every blue
    exactly like position — because it is genuinely derivable: each radar
    measures only the radial component, but two non-collinear radials
    determine the full 2-D vector.  What differs per blue is nothing; what
    varies is how much is KNOWN, which travels in the covariance feature.
    """
    env = PursuitEnv(
        n_blue=2, n_red=1, n_obstacles=1, arena_size=130.0, max_steps=60,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        sensor_pos_noise_std=0.0, sensor_vel_noise_std=0.0,
        moving_obstacle_fraction=1.0, obstacle_speed=2.0,
        track_detection=False, track_conf_min=1.0,
        sensor_noise_range_growth=0.0,
        red_policy=stationary_red, seed=5,
    )
    env.reset(seed=5)
    o = env._obstacle_pos[0].copy()
    r0 = float(env._obstacle_r[0])
    env._red_pos[0] = np.array([5.0, 5.0], dtype=np.float32)
    env._blue_pos[0] = o + np.array([r0 + 8.0, 0.0], dtype=np.float32)  # detects
    env._blue_pos[1] = np.array([125.0, 125.0], dtype=np.float32)       # far

    obs = env.structured_belief_observation()
    assert env._last_obs_detect is not None
    assert bool(env._last_obs_detect[0, 0]), "near blue should detect it"
    assert not bool(env._last_obs_detect[1, 0]), "far blue should not"

    # ob edges are (s outer, b inner) -> slot 0 gives [blue0, blue1].
    ob = obs["ob_edge_features"]
    # Both blues carry the SAME fused object velocity (rel_vel differs only
    # by each blue's own motion, and both are stationary here).
    assert np.allclose(ob[0][2:4], ob[1][2:4], atol=1e-6), (
        "fused velocity is a shared command product")


def test_velocity_covariance_reports_what_is_known():
    """The covariance is the informative half: with ONE detector it says the
    tangential direction is unknown (~prior); with two non-collinear ones it
    collapses.  A memory track sits at the prior in both directions."""
    def cov(n_det):
        env = PursuitEnv(
            n_blue=3, n_red=1, n_obstacles=0, arena_size=130.0, max_steps=60,
            capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
            sensor_pos_noise_std=0.0, sensor_vel_noise_std=0.05,
            sensor_noise_range_growth=0.0, vel_prior_std=1.0,
            track_detection=False, red_policy=stationary_red, seed=7)
        env.reset(seed=7)
        tgt = np.array([65.0, 65.0], dtype=np.float32)
        env._red_pos[0] = tgt
        env._red_vel[0] = np.array([0.7, -0.4], dtype=np.float32)
        env._blue_pos[:] = np.array([5.0, 5.0], dtype=np.float32)   # far
        if n_det >= 1:
            env._blue_pos[0] = tgt + np.array([18.0, 0.0], dtype=np.float32)
        if n_det >= 2:
            env._blue_pos[1] = tgt + np.array([0.0, 18.0], dtype=np.float32)
        _p, _c, _r, _v, vcov = env._build_enemy_tracks()
        return vcov[0]

    c0, c1, c2 = cov(0), cov(1), cov(2)
    # 0 detectors -> the prior in both directions (normalised to 1).
    assert np.allclose(c0[:2], 1.0, atol=1e-3)
    # 1 detector -> one direction measured, the other still at the prior.
    assert min(c1[0], c1[1]) < 0.05 and max(c1[0], c1[1]) > 0.9
    # 2 non-collinear -> both directions collapse.
    assert c2[0] < 0.05 and c2[1] < 0.05
