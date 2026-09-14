"""
tests/test_coverage_misses.py — counting a confirmed track's misses only
where it should have been seen.

Two layers:
* ``sensor_coverage``: the geometry — range, the k-sigma margin, occlusion.
* ``MultiTargetTracker.step(..., coverage=...)``: the accounting — a phantom
  sitting in view dies fast, a target out of view keeps coasting, and
  ``max_coast_steps`` still bounds the latter.

With no coverage passed, behaviour is unchanged; the existing suite
(including the bit-exact Gaussian-Sum non-regression) covers that.

Run:
    pytest tests/test_coverage_misses.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.tracking import MultiTargetTracker
from isr.tracking.coverage import sensor_coverage
from tests.test_tracker import det

R = 40.0
TIGHT = np.eye(2) * 1.0          # sd 1 m
LOOSE = np.eye(2) * 100.0        # sd 10 m


# --------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------- #

def test_inside_range_is_covered_outside_is_not():
    cov = sensor_coverage([[0.0, 0.0]], R)
    assert cov(np.array([20.0, 0.0]), TIGHT)
    assert not cov(np.array([60.0, 0.0]), TIGHT)


def test_margin_scales_with_the_tracks_own_uncertainty():
    """dist + k*sd must fit inside the disk.  At 30 m, sd 1 m fits
    (30 + 2 = 32 <= 40); sd 10 m does not (30 + 20 = 50)."""
    cov = sensor_coverage([[0.0, 0.0]], R, k_sigma=2.0)
    assert cov(np.array([30.0, 0.0]), TIGHT)
    assert not cov(np.array([30.0, 0.0]), LOOSE)


def test_margin_uses_the_largest_axis_of_the_covariance():
    """An elongated ellipse is only certainly inside if its LONG axis is."""
    cov = sensor_coverage([[0.0, 0.0]], R, k_sigma=2.0)
    P = np.diag([1.0, 100.0])                  # sd 1 m in x, 10 m in y
    assert not cov(np.array([30.0, 0.0]), P)


def test_any_blue_is_enough():
    cov = sensor_coverage([[0.0, 0.0], [100.0, 0.0]], R)
    assert cov(np.array([90.0, 0.0]), TIGHT)


def test_occlusion_blocks_coverage_unless_another_blue_has_line_of_sight():
    def wall_at_x10(frm, pts):
        # Anything seen across x = 10 from the left is blocked.
        return (frm[0] < 10.0) & (pts[:, 0] > 10.0)

    blocked = sensor_coverage([[0.0, 0.0]], R, occluded=wall_at_x10)
    assert not blocked(np.array([20.0, 0.0]), TIGHT)

    other_side = sensor_coverage([[0.0, 0.0], [30.0, 0.0]], R,
                                 occluded=wall_at_x10)
    assert other_side(np.array([20.0, 0.0]), TIGHT)


# --------------------------------------------------------------------- #
#  Tracker accounting
# --------------------------------------------------------------------- #

BLUE_A = np.array([20.0, 60.0])
BLUE_B = np.array([60.0, 20.0])
V = np.array([0.5, 0.0])


def _tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=2,
                confirm_window=3, max_misses=3, max_misses_tentative=2)
    base.update(kw)
    return MultiTargetTracker(**base)


def _confirmed_at(trk, p):
    for _ in range(4):
        p = p + V
        trk.step([det(BLUE_A, p, V, blue=0), det(BLUE_B, p, V, blue=1)])
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed
    return p


def test_without_coverage_every_miss_counts():
    trk = _tracker()
    _confirmed_at(trk, np.array([40.0, 40.0]))
    for _ in range(4):
        trk.step([])
    assert len(trk.tracks) == 0


def test_confirmed_track_in_view_dies_on_the_short_budget():
    """The phantom case: sensors keep looking where the track should be."""
    always_seen = lambda pos, P: True
    trk = _tracker()
    _confirmed_at(trk, np.array([40.0, 40.0]))
    for _ in range(4):
        trk.step([], coverage=always_seen)
    assert len(trk.tracks) == 0, "a track nobody sees while in view survived"


def test_confirmed_track_out_of_view_keeps_coasting():
    """The fled-target case: its misses are uninformative and do not count."""
    never_seen = lambda pos, P: False
    trk = _tracker()
    _confirmed_at(trk, np.array([40.0, 40.0]))
    for _ in range(40):
        trk.step([], coverage=never_seen)
    assert len(trk.tracks) == 1, "a track out of every sensor's view was killed"
    assert trk.tracks[0].misses == 0
    assert trk.tracks[0].steps_since_hit == 40


def test_max_coast_steps_still_bounds_an_out_of_view_track():
    never_seen = lambda pos, P: False
    trk = _tracker(max_coast_steps=10)
    _confirmed_at(trk, np.array([40.0, 40.0]))
    for _ in range(10):
        trk.step([], coverage=never_seen)
    assert len(trk.tracks) == 1
    trk.step([], coverage=never_seen)
    assert len(trk.tracks) == 0, "coast cap was not enforced"


def test_tentative_tracks_count_every_miss_regardless_of_coverage():
    """A tentative that left view right after birth must not freeze."""
    never_seen = lambda pos, P: False
    trk = _tracker()
    trk.step([det(BLUE_A, [40.0, 40.0], V)])
    assert not trk.tracks[0].confirmed
    for _ in range(3):
        trk.step([], coverage=never_seen)
    assert len(trk.tracks) == 0


def test_a_hit_resets_both_counters():
    in_view = lambda pos, P: True
    trk = _tracker(max_misses=5)
    p = _confirmed_at(trk, np.array([40.0, 40.0]))
    for _ in range(2):
        p = p + V
        trk.step([], coverage=in_view)
    tr = trk.tracks[0]
    assert tr.misses == 2 and tr.steps_since_hit == 2
    p = p + V
    trk.step([det(BLUE_A, p, V, blue=0), det(BLUE_B, p, V, blue=1)],
             coverage=in_view)
    assert tr.misses == 0 and tr.steps_since_hit == 0


def test_env_coverage_mirrors_the_detection_range_and_occlusion():
    e = PursuitEnv(n_blue=1, n_red=1, n_obstacles=1, arena_size=130.0,
                   max_steps=50, sensor_radius=40.0,
                   red_policy=run_from_nearest_uav, seed=0)
    e.reset(seed=0)
    e._blue_pos[:] = np.array([[20.0, 65.0]], dtype=np.float32)
    e._obstacle_pos = np.array([[45.0, 65.0]], dtype=np.float32)
    e._obstacle_r = np.array([8.0], dtype=np.float32)
    cov = e.track_coverage(k_sigma=2.0)
    assert cov(np.array([20.0, 80.0]), TIGHT), "clear LOS, in range"
    assert not cov(np.array([57.0, 65.0]), TIGHT), "behind the obstacle"
    assert not cov(np.array([100.0, 65.0]), TIGHT), "out of range"
