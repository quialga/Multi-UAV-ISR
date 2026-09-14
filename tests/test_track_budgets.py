"""
tests/test_track_budgets.py — separate miss budgets for TENTATIVE and
CONFIRMED tracks, in both the red and the obstacle tracker.

Why the split exists: a confirmed track should coast long (the learned
motion model holds a coasting red ~5 m from truth after 51-80 misses), a
tentative one — most often born from a single clutter plot — should die
fast.  A single shared budget cannot do both.

The scenario that matters most is the last red-tracker test: under a long
shared budget, a stale tentative track sits in the arena with an
ever-growing gate and ABSORBS the first real detection that wanders into
it, so a real target is tracked by a phantom's identity.

Run:
    pytest tests/test_track_budgets.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.tracking import MultiTargetTracker
from isr.tracking.obstacle_tracker import ObstacleTracker
from tests.test_obstacle_tracker import det as obs_det
from tests.test_tracker import det as red_det

BLUE_A = np.array([20.0, 60.0])
BLUE_B = np.array([60.0, 20.0])


def _red(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=2,
                confirm_window=3, max_misses=80)
    base.update(kw)
    return MultiTargetTracker(**base)


def _obs(**kw):
    base = dict(dt=1.0, sigma_a=0.1, vel_prior_std=1.0, confirm_hits=2,
                confirm_window=3, max_misses=80)
    base.update(kw)
    return ObstacleTracker(**base)


# --------------------------------------------------------------------- #
#  Defaults preserve existing behaviour
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("make", [_red, _obs])
def test_unset_tentative_budget_falls_back_to_the_shared_one(make):
    trk = make(max_misses=7)
    assert trk.max_misses_tentative == 7


@pytest.mark.parametrize("make", [_red, _obs])
def test_negative_tentative_budget_is_rejected(make):
    with pytest.raises(AssertionError):
        make(max_misses_tentative=-1)


# --------------------------------------------------------------------- #
#  Red tracker
# --------------------------------------------------------------------- #

def test_red_tentative_track_dies_on_its_own_short_budget():
    """One spurious return, then silence.  With a long shared budget the
    phantom lingers; with a short tentative budget it is gone after
    max_misses_tentative + 1 empty steps."""
    spurious = red_det(BLUE_A, [40.0, 40.0], [0.0, 0.0], truth_id=-1)

    shared = _red()
    split = _red(max_misses_tentative=2)
    for trk in (shared, split):
        trk.step([spurious])
        assert len(trk.tracks) == 1 and not trk.tracks[0].confirmed

    for _ in range(3):
        shared.step([])
        split.step([])
    assert len(shared.tracks) == 1, "shared budget should keep the phantom"
    assert len(split.tracks) == 0, "tentative outlived its short budget"


def test_red_confirmed_track_keeps_the_long_budget():
    trk = _red(max_misses_tentative=2)
    p, v = np.array([50.0, 50.0]), np.array([0.5, 0.0])
    for _ in range(4):                                   # establish + confirm
        p = p + v
        trk.step([red_det(BLUE_A, p, v, blue=0), red_det(BLUE_B, p, v, blue=1)])
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed
    for _ in range(30):                                  # well past 2
        trk.step([])
    assert len(trk.tracks) == 1, "confirmed track died on the tentative budget"


def test_red_confirmed_track_still_dies_past_the_long_budget():
    trk = _red(max_misses=5, max_misses_tentative=2)
    p, v = np.array([50.0, 50.0]), np.array([0.5, 0.0])
    for _ in range(4):
        p = p + v
        trk.step([red_det(BLUE_A, p, v, blue=0), red_det(BLUE_B, p, v, blue=1)])
    assert trk.tracks[0].confirmed
    for _ in range(6):
        trk.step([])
    assert len(trk.tracks) == 0, "confirmed track outlived max_misses"


def test_stale_phantom_absorbs_a_real_target_under_a_shared_budget():
    """The motivating failure.  A single clutter plot at X births a
    tentative track.  Ten steps later a real target arrives at X.  Under a
    long shared budget the phantom is still alive, its gate has grown with
    every predict, and it takes the real detection — the real target ends
    up tracked under the phantom's identity.  With a short tentative budget
    the phantom is long gone and the real target gets its own track."""
    x = np.array([40.0, 40.0])
    t_arrival = 10

    def run(tracker):
        tracker.step([red_det(BLUE_A, x, [0.0, 0.0], truth_id=-1)])   # t=0
        for _ in range(t_arrival - 1):
            tracker.step([])
        v = np.array([0.3, 0.0])
        p = x.copy()
        for _ in range(4):
            tracker.step([red_det(BLUE_A, p, v, blue=0, truth_id=0),
                          red_det(BLUE_B, p, v, blue=1, truth_id=0)])
            p = p + v
        near = [t for t in tracker.tracks if np.linalg.norm(t.pos - p) < 5.0]
        assert len(near) == 1, f"expected one track on the target: {tracker.tracks}"
        return near[0]

    phantom_owner = run(_red())
    assert phantom_owner.born_at == 1, (
        "under a shared long budget the stale phantom should have taken "
        "the real target (born at the clutter step)")

    own = run(_red(max_misses_tentative=2))
    assert own.born_at > t_arrival - 1, (
        "with a short tentative budget the real target must get a track "
        "born when it actually arrived, not inherit a phantom")


# --------------------------------------------------------------------- #
#  Obstacle tracker
# --------------------------------------------------------------------- #

def test_obstacle_tentative_track_dies_on_its_own_short_budget():
    spurious = obs_det(BLUE_A, [40.0, 40.0], [0.0, 0.0], 5.0, truth_id=-1)
    shared = _obs()
    split = _obs(max_misses_tentative=2)
    for trk in (shared, split):
        trk.step([spurious])
    for _ in range(3):
        shared.step([])
        split.step([])
    assert len(shared.tracks) == 1
    assert len(split.tracks) == 0


def test_seen_obstacle_is_not_forgotten_during_a_long_absence():
    """A static obstacle does not go anywhere, so a confirmed one must
    survive a long stretch unobserved even with a short tentative budget —
    the policy-facing config for obstacles is 'remember what you saw'."""
    trk = _obs(max_misses=1000, max_misses_tentative=2)
    c = np.array([70.0, 70.0])
    for _ in range(3):
        trk.step([obs_det(BLUE_A, c, [0.0, 0.0], 8.0, blue=0),
                  obs_det(BLUE_B, c, [0.0, 0.0], 8.0, blue=1)])
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed
    for _ in range(300):
        trk.step([])
    assert len(trk.tracks) == 1, "confirmed obstacle was forgotten"
    assert np.linalg.norm(trk.tracks[0].x[:2] - c) < 5.0, (
        "a static obstacle's estimate should not wander while unobserved")
