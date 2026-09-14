"""
tests/test_obstacle_merge.py — duplicate merge and the static-obstacle
configuration of the obstacle tracker.

Two failures seen at the training configuration (docs/tracking_diagnostics.md
§9.6): tracks of static obstacles DRIFTING on a velocity estimated from
noise, and DUPLICATE confirmed tracks beside a good one.  The static
configuration (sigma_a = 0, velocity pinned at 0) removes the drift; the
merge removes duplicates that converge onto the same obstacle.

Run:
    pytest tests/test_obstacle_merge.py -v
"""
from __future__ import annotations

import numpy as np

from isr.tracking.obstacle_tracker import CHI2_99, ObstacleTrack, ObstacleTracker

NEVER = 10 ** 9


def _track(pos, r, t=0, sd=0.3, sd_r=0.3, confirmed=True):
    x = np.array([pos[0], pos[1], 0.0, 0.0, r], dtype=np.float64)
    P = np.diag([sd ** 2, sd ** 2, 1e-6, 1e-6, sd_r ** 2])
    tr = ObstacleTrack(x, P, t)
    tr.confirmed = confirmed
    tr.history = [True]
    return tr


def _static_tracker(**kw):
    base = dict(sigma_a=0.0, vel_prior_std=0.0, confirm_hits=3,
                confirm_window=4, confirm_deadline=True, max_misses=NEVER)
    base.update(kw)
    return ObstacleTracker(**base)


def _obstacle_det(blue_pos, centre, r, blue=0, sigma_pos=1.0, sigma_radius=2.0):
    blue_pos, centre = np.asarray(blue_pos, float), np.asarray(centre, float)
    d = centre - blue_pos
    los = d / np.linalg.norm(d)
    return {"blue": blue, "z_pos": centre.astype(np.float32),
            "z_range": float(np.linalg.norm(d)), "z_radius": float(r),
            "z_radial": 0.0, "los": los.astype(np.float32),
            "sigma_pos": sigma_pos, "sigma_radius": sigma_radius,
            "sigma_radial": 0.1, "truth_id": 0}


# --------------------------------------------------------------------- #
#  Merge
# --------------------------------------------------------------------- #

def test_merge_is_off_by_default():
    trk = _static_tracker()
    trk.tracks = [_track((50, 50), 10, t=0), _track((50.1, 50), 10, t=5)]
    trk.step([])
    assert len(trk.tracks) == 2


def test_duplicates_of_one_obstacle_merge_into_the_older_track():
    trk = _static_tracker(merge_chi2=CHI2_99[3])
    old = _track((50.0, 50.0), 10.0, t=0, sd=0.5)
    new = _track((50.2, 49.9), 10.1, t=7, sd=0.2)
    trk.tracks = [old, new]
    trk.step([])
    assert trk.tracks == [old], "the older confirmed track survives"
    assert np.allclose(old.pos, (50.2, 49.9)), "and keeps the more certain estimate"


def test_distinct_obstacles_are_never_merged():
    """The closest two real obstacles can be: radii 5 + 5 and a 1 m gap."""
    trk = _static_tracker(merge_chi2=CHI2_99[3])
    a = _track((50.0, 50.0), 5.0, t=0, sd=1.5, sd_r=2.0)
    b = _track((61.0, 50.0), 5.0, t=3, sd=1.5, sd_r=2.0)
    trk.tracks = [a, b]
    trk.step([])
    assert len(trk.tracks) == 2


def test_confirmed_track_is_preferred_over_an_older_tentative():
    trk = _static_tracker(merge_chi2=CHI2_99[3])
    tent = _track((50.0, 50.0), 10.0, t=0, confirmed=False)
    tent.born_at = trk.t + 1                # keep it within its deadline
    conf = _track((50.1, 50.0), 10.0, t=0)
    conf.born_at = -5
    trk.tracks = [tent, conf]
    trk.step([])
    assert trk.tracks == [conf]


# --------------------------------------------------------------------- #
#  Static configuration
# --------------------------------------------------------------------- #

def test_static_config_does_not_drift_while_unobserved():
    trk = _static_tracker()
    centre, r = (80.0, 60.0), 10.0
    for k in range(4):                      # two observers, 4 scans
        trk.step([_obstacle_det((40.0, 60.0), centre, r, blue=0),
                  _obstacle_det((80.0, 20.0), centre, r, blue=1)])
    (tr,) = trk.confirmed_tracks()
    pos, P = tr.pos.copy(), tr.P.copy()
    for _ in range(60):
        trk.step([])
    assert np.allclose(tr.vel, 0.0, atol=1e-6)
    assert np.allclose(tr.pos, pos)
    assert np.allclose(tr.P, P), "no process noise: covariance must not grow"


def test_default_config_does_drift_on_a_noisy_velocity():
    """The failure the static configuration removes: an estimated velocity
    from noise carries the track away while nobody observes it."""
    rng = np.random.default_rng(0)
    trk = ObstacleTracker(confirm_hits=3, confirm_window=4,
                          confirm_deadline=True, max_misses=NEVER)
    centre, r = np.array([80.0, 60.0]), 10.0
    for _ in range(4):
        dets = []
        for b, bp in enumerate(((40.0, 60.0), (80.0, 20.0))):
            d = _obstacle_det(bp, centre + rng.normal(0, 1.0, 2), r, blue=b)
            d["z_radial"] = float(rng.normal(0, 0.1))
            dets.append(d)
        trk.step(dets)
    (tr,) = trk.confirmed_tracks()
    pos = tr.pos.copy()
    for _ in range(60):
        trk.step([])
    assert np.linalg.norm(tr.pos - pos) > 1.0
