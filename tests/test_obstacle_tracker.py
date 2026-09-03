"""
tests/test_obstacle_tracker.py — Kalman tracking for obstacles.

Hand-built, deterministic scenarios (see tests/test_tracker.py's rationale:
aggregate metrics tell you THAT something broke, these tell you WHAT).

Run:
    pytest tests/test_obstacle_tracker.py -v
"""
from __future__ import annotations

import numpy as np

from isr.tracking.kalman import joseph_update
from isr.tracking.obstacle_tracker import ObstacleTrack, ObstacleTracker


def det(blue_pos, true_pos, true_vel, true_r, blue=0,
        sigma_pos=1.0, sigma_radius=0.0, sigma_radial=0.02, rng=None,
        truth_id=0):
    blue_pos = np.asarray(blue_pos, dtype=np.float32)
    true_pos = np.asarray(true_pos, dtype=np.float32)
    true_vel = np.asarray(true_vel, dtype=np.float32)
    d = true_pos - blue_pos
    n = float(np.linalg.norm(d))
    los = (d / n) if n > 1e-6 else np.zeros(2, dtype=np.float32)
    z_pos = true_pos.copy()
    z_r = float(true_r)
    zr = float(true_vel @ los)
    if rng is not None:
        z_pos = z_pos + rng.normal(0.0, sigma_pos, 2).astype(np.float32)
        if sigma_radius > 0.0:
            z_r += float(rng.normal(0.0, sigma_radius))
        zr += float(rng.normal(0.0, sigma_radial))
    return {"blue": blue, "z_pos": z_pos.astype(np.float32),
            "z_range": float(np.linalg.norm(z_pos - blue_pos)),
            "z_radius": z_r, "z_radial": zr, "los": los.astype(np.float32),
            "sigma_pos": float(sigma_pos), "sigma_radius": float(sigma_radius),
            "sigma_radial": float(sigma_radial), "truth_id": truth_id}


def _tracker(**kw):
    base = dict(dt=1.0, sigma_a=0.1, vel_prior_std=1.0, confirm_hits=2,
                max_misses=5)
    base.update(kw)
    return ObstacleTracker(**base)


# --------------------------------------------------------------------- #
#  Shared Kalman primitive, exercised at a non-4-D size
# --------------------------------------------------------------------- #

def test_joseph_update_is_dimension_agnostic():
    """The obstacle tracker's whole justification for reusing
    isr.tracking.kalman.joseph_update (rather than re-deriving the Joseph
    form) is that it works for ANY state size — proven here at 5-D, not
    just the red tracker's 4-D."""
    x = np.array([10.0, 5.0, 0.3, -0.1, 8.0])
    P = np.eye(5) * 2.0
    H = np.zeros((2, 5)); H[0, 0] = H[1, 1] = 1.0
    R = np.eye(2) * 1.0
    z = np.array([10.5, 5.2])
    x_new, P_new, nis, loglik = joseph_update(x, P, H, R, z)
    assert x_new.shape == (5,)
    assert P_new.shape == (5, 5)
    assert np.allclose(P_new, P_new.T)
    assert np.all(np.linalg.eigvalsh(P_new) > 0)
    assert nis >= 0.0
    assert np.isfinite(loglik)


# --------------------------------------------------------------------- #
#  Scenario 1 — a single static obstacle: position and radius converge
# --------------------------------------------------------------------- #

def test_static_obstacle_position_and_radius_converge():
    """Two non-collinear blues: a single observer's LOS constrains only ONE
    velocity component via Doppler, leaving the perpendicular axis to drift
    on process noise alone over many steps (see
    test_single_blue_static_obstacle_can_drift_off_gate) -- not the
    convergence behaviour this test means to exercise."""
    trk = _tracker()
    rng = np.random.default_rng(0)
    p = np.array([65.0, 65.0]); v = np.array([0.0, 0.0]); r = 9.0
    b0, b1 = np.array([65.0, 40.0]), np.array([40.0, 65.0])
    errs_pos, errs_r = [], []
    for _ in range(30):
        dets = [det(b0, p, v, r, blue=0, sigma_pos=1.0, sigma_radius=1.0, rng=rng),
                det(b1, p, v, r, blue=1, sigma_pos=1.0, sigma_radius=1.0, rng=rng)]
        trk.step(dets)
        assert len(trk.tracks) == 1, "a single obstacle must not spawn extras"
        errs_pos.append(np.linalg.norm(trk.tracks[0].pos - p))
        errs_r.append(abs(trk.tracks[0].radius - r))
    assert trk.tracks[0].confirmed
    assert np.mean(errs_pos[-10:]) < 1.0    # beats the raw 1 m noise
    assert np.mean(errs_r[-10:]) < 0.6      # beats the raw 1 m radius noise
    assert np.linalg.norm(trk.tracks[0].vel) < 0.3, "static obstacle: v ~ 0"


def test_exact_radius_is_taken_immediately():
    """sigma_radius = 0 (the default sensor setting): the first return
    already IS the truth, so the track's radius must match from step 1,
    not converge gradually."""
    trk = _tracker()
    p = np.array([65.0, 65.0]); r = 11.3
    d = det(np.array([65.0, 45.0]), p, np.zeros(2), r, sigma_radius=0.0)
    trk.step([d])
    assert np.isclose(trk.tracks[0].radius, r, atol=1e-6)


# --------------------------------------------------------------------- #
#  Scenario 2 — velocity from ONE blue circling over TIME, not a single
#  instant: a real advantage of a persistent filter over one-shot fusion.
# --------------------------------------------------------------------- #

def test_single_blue_recovers_velocity_by_circling_over_time():
    """A lone observer nearly perpendicular to the motion cannot see it
    INSTANTANEOUSLY (Doppler ~ 0 on that one LOS) -- but if that same blue
    sweeps to a DIFFERENT angle a few steps later, the persistent filter
    accumulates both radial equations across TIME and recovers the full
    velocity, without ever needing two blues at once."""
    trk = _tracker()
    rng = np.random.default_rng(1)
    p = np.array([65.0, 65.0]); v = np.array([0.5, 0.0]); r = 8.0

    # Step 1: observer due SOUTH -> LOS ~ (0, 1), nearly perpendicular to v.
    trk.step([det(np.array([65.0, 40.0]), p, v, r, sigma_pos=0.3,
                  sigma_radial=0.02, rng=rng)])
    v_after_one_angle = trk.tracks[0].vel.copy()

    # Steps 2+: the SAME blue has moved to observe from the EAST (LOS ~
    # (-1, 0), now well-aligned with v) -- a realistic "it flew past".
    for _ in range(10):
        trk.step([det(np.array([95.0, 65.0]), p, v, r, sigma_pos=0.3,
                      sigma_radial=0.02, rng=rng)])
    assert np.linalg.norm(trk.tracks[0].vel - v) < 0.15, (
        f"velocity should converge from accumulated single-blue angles: "
        f"{trk.tracks[0].vel} vs {v} (one-angle estimate was "
        f"{v_after_one_angle})")


# --------------------------------------------------------------------- #
#  Scenario 3 — occlusion: coast and reacquire, identity preserved
# --------------------------------------------------------------------- #

def test_track_coasts_through_occlusion_and_reacquires():
    trk = _tracker(max_misses=5)
    rng = np.random.default_rng(2)
    p = np.array([65.0, 65.0]); v = np.zeros(2); r = 9.0
    b0, b1 = np.array([65.0, 45.0]), np.array([45.0, 65.0])
    for _ in range(6):
        trk.step([det(b0, p, v, r, blue=0, rng=rng),
                  det(b1, p, v, r, blue=1, rng=rng)])
    tid = trk.tracks[0].id
    for _ in range(4):
        trk.step([])
        assert len(trk.tracks) == 1, "track died during a short gap"
    assert trk.tracks[0].id == tid
    for _ in range(5):
        trk.step([det(b0, p, v, r, blue=0, rng=rng),
                  det(b1, p, v, r, blue=1, rng=rng)])
    assert len(trk.tracks) == 1
    assert trk.tracks[0].id == tid, "reacquisition created a NEW identity"


def test_track_dies_after_max_misses():
    trk = _tracker(max_misses=3)
    rng = np.random.default_rng(3)
    p = np.array([65.0, 65.0]); v = np.zeros(2); r = 9.0
    b0, b1 = np.array([65.0, 45.0]), np.array([45.0, 65.0])
    for _ in range(5):
        trk.step([det(b0, p, v, r, blue=0, rng=rng),
                  det(b1, p, v, r, blue=1, rng=rng)])
    assert len(trk.tracks) == 1
    for _ in range(4):
        trk.step([])
    assert len(trk.tracks) == 0


def test_single_blue_static_obstacle_can_drift_off_gate():
    """Documented limitation, not a bug (same class as the red tracker's
    test_single_blue_crossing_geometry_is_unobservable): a LONE observer's
    LOS constrains only the RADIAL velocity component via Doppler, leaving
    the perpendicular axis free to random-walk on process noise alone.
    Over enough steps this can walk the predicted position outside the
    gate for an otherwise-static obstacle, fragmenting the track.  Fixed
    the same way -- a second, non-collinear observer."""
    def run(two_blues, seed=3, steps=5):
        trk = _tracker(max_misses=3)
        rng = np.random.default_rng(seed)
        p = np.array([65.0, 65.0]); v = np.zeros(2); r = 9.0
        b0, b1 = np.array([65.0, 45.0]), np.array([45.0, 65.0])
        for _ in range(steps):
            dets = [det(b0, p, v, r, blue=0, rng=rng)]
            if two_blues:
                dets.append(det(b1, p, v, r, blue=1, rng=rng))
            trk.step(dets)
        return len(trk.tracks)

    assert run(two_blues=False) > 1, "expected the single-LOS drift to fragment"
    assert run(two_blues=True) == 1, "a second LOS should prevent the drift"


# --------------------------------------------------------------------- #
#  Scenario 4 — the duplicate-birth trap (same as the red tracker)
# --------------------------------------------------------------------- #

def test_two_blues_seeing_one_new_obstacle_create_ONE_track():
    trk = _tracker()
    p = np.array([65.0, 65.0]); v = np.zeros(2); r = 9.0
    trk.step([det(np.array([65.0, 45.0]), p, v, r, blue=0),
              det(np.array([90.0, 65.0]), p, v, r, blue=1)])
    assert len(trk.tracks) == 1, (
        f"one new obstacle produced {len(trk.tracks)} tracks (duplicate birth)")


def test_two_separated_new_obstacles_create_two_tracks():
    trk = _tracker()
    trk.step([det(np.array([10.0, 10.0]), np.array([50.0, 50.0]),
                  np.zeros(2), 8.0, blue=0, truth_id=0),
              det(np.array([10.0, 10.0]), np.array([95.0, 50.0]),
                  np.zeros(2), 6.0, blue=0, truth_id=1)])
    assert len(trk.tracks) == 2


# --------------------------------------------------------------------- #
#  A moving (patrolling) obstacle: position tracks a real trajectory
# --------------------------------------------------------------------- #

def test_moving_obstacle_position_tracks_constant_velocity():
    trk = _tracker(sigma_a=0.1)
    rng = np.random.default_rng(4)
    p = np.array([30.0, 65.0]); v = np.array([1.0, 0.0]); r = 6.0
    blue = np.array([30.0, 40.0])
    errs = []
    for _ in range(25):
        p = p + v
        d = det(blue, p, v, r, sigma_pos=0.5, sigma_radial=0.02, rng=rng)
        trk.step([d])
        errs.append(np.linalg.norm(trk.tracks[0].pos - p))
    assert np.mean(errs[-8:]) < 0.6
    assert np.linalg.norm(trk.tracks[0].vel - v) < 0.3


# --------------------------------------------------------------------- #
#  Numerics
# --------------------------------------------------------------------- #

def test_covariance_stays_symmetric_positive_definite():
    trk = _tracker()
    rng = np.random.default_rng(5)
    p = np.array([65.0, 65.0]); v = np.array([0.2, 0.1]); r = 8.0
    blues = [np.array([40.0, 65.0]), np.array([90.0, 65.0]),
             np.array([65.0, 95.0])]
    for _ in range(60):
        p = p + v
        trk.step([det(b, p, v, r, blue=i, sigma_radius=0.5, rng=rng)
                  for i, b in enumerate(blues)])
        P = trk.tracks[0].P
        assert np.allclose(P, P.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(P) > 0)


def test_radius_process_noise_is_exactly_zero():
    """The cleanest part of this design: a rigid obstacle's TRUE radius
    does not change, so Q_r must be EXACTLY 0, not merely small."""
    trk = _tracker()
    assert trk.Q[4, 4] == 0.0
    assert np.all(trk.Q[4, :] == 0.0) and np.all(trk.Q[:, 4] == 0.0)


def test_no_detections_ever_is_a_no_op():
    trk = _tracker()
    for _ in range(10):
        trk.step([])
    assert trk.tracks == []
