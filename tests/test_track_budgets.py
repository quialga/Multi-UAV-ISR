"""
tests/test_track_budgets.py — how tentative and confirmed tracks are deleted,
in both the red and the obstacle tracker.

Two rules, deliberately separate:

* CONFIRMED tracks die on ``max_misses`` (with a coverage model, counting
  only misses where they should have been seen), bounded by
  ``max_coast_steps``.  They should coast long.
* TENTATIVE tracks, with ``confirm_deadline``, get exactly one confirmation
  window and are deleted if they have not confirmed by its end.  They
  should die fast: most are born from a single clutter plot.

The deadline replaced a separate consecutive-miss budget for tentatives,
which overlapped with M-of-N (its sensible value is fixed by the window, and
set independently it could turn 3-of-4 into 3-of-3) and left a gap: a
tentative ALTERNATING hit and miss never confirmed and never died.  Both
are pinned below, as is the failure that motivated treating tentatives
differently at all: a stale phantom absorbing a real target.

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


def _drive(trk, pattern, v=(0.5, 0.0)):
    """Step a single target through a hit/miss PATTERN ('T' = both blues
    detect it, 'F' = no detections).  Returns the tracker."""
    p, v = np.array([40.0, 40.0]), np.array(v)
    for ch in pattern:
        p = p + v
        dets = ([red_det(BLUE_A, p, v, blue=0), red_det(BLUE_B, p, v, blue=1)]
                if ch == "T" else [])
        trk.step(dets)
    return trk


# --------------------------------------------------------------------- #
#  Default: original behaviour, one shared budget
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("make", [_red, _obs])
def test_deadline_is_off_by_default(make):
    assert make().confirm_deadline is False


def test_without_the_deadline_a_tentative_lives_as_long_as_max_misses():
    trk = _red(max_misses=5)
    trk.step([red_det(BLUE_A, [40.0, 40.0], [0.0, 0.0], truth_id=-1)])
    for _ in range(5):
        trk.step([])
    assert len(trk.tracks) == 1
    trk.step([])
    assert len(trk.tracks) == 0


# --------------------------------------------------------------------- #
#  The motivating failure
# --------------------------------------------------------------------- #

def test_stale_phantom_absorbs_a_real_target_without_the_deadline():
    """A single clutter plot at X births a tentative track.  Ten steps later
    a real target arrives at X.  With a long shared budget the phantom is
    still alive, its gate grown by every predict, and it takes the real
    detection — the real target ends up tracked under the phantom's
    identity.  With the deadline the phantom is long gone and the real
    target gets a track of its own."""
    x = np.array([40.0, 40.0])
    t_arrival = 10

    def run(tracker):
        tracker.step([red_det(BLUE_A, x, [0.0, 0.0], truth_id=-1)])   # t=1
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
        "without the deadline the stale phantom should have taken the real "
        "target (born at the clutter step)")

    own = run(_red(confirm_deadline=True))
    assert own.born_at > t_arrival - 1, (
        "with the deadline the real target must get a track born when it "
        "actually arrived, not inherit a phantom")


# --------------------------------------------------------------------- #
#  The deadline
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("pattern", ["TTT", "TTFT", "TFTT"])
def test_deadline_lets_every_3_of_4_pattern_confirm(pattern):
    """The deadline falls on the step that completes the first window, and
    promotion is checked first — so every pattern that satisfies 3-of-4
    within its first window still confirms."""
    trk = _drive(_red(confirm_hits=3, confirm_window=4, confirm_deadline=True),
                 pattern)
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed


def test_deadline_deletes_a_tentative_that_fails_its_first_window():
    """TFFT has only 2 hits in the first window of 4.  The last hit is
    absorbed by the tentative, which is then deleted at age 3 — and a target
    still there is simply re-born from its next detection."""
    trk = _drive(_red(confirm_hits=3, confirm_window=4, confirm_deadline=True),
                 "TFFT")
    assert len(trk.tracks) == 0
    trk.step([red_det(BLUE_A, [45.0, 40.0], [0.5, 0.0], blue=0),
              red_det(BLUE_B, [45.0, 40.0], [0.5, 0.0], blue=1)])
    assert len(trk.tracks) == 1 and not trk.tracks[0].confirmed, (
        "the next detection should start a fresh tentative")


def test_deadline_on_2_of_3_falls_at_age_2():
    assert len(_drive(_red(confirm_hits=2, confirm_window=3,
                           confirm_deadline=True), "TFT").tracks) == 1
    assert len(_drive(_red(confirm_hits=2, confirm_window=3,
                           confirm_deadline=True), "TFF").tracks) == 0


def test_deadline_ignores_max_misses_for_tentatives():
    """The clash the deadline exists to remove.  With max_misses = 0 applied
    to tentatives, any miss kills one, so 3-of-4 silently becomes 3-of-3 and
    TFTT can never confirm.  With the deadline, tentatives answer to the
    window alone and TFTT confirms."""
    legacy = _drive(_red(confirm_hits=3, confirm_window=4, max_misses=0), "TFTT")
    assert not any(t.confirmed for t in legacy.tracks), (
        "without the deadline max_misses=0 should have killed the tentative")

    trk = _drive(_red(confirm_hits=3, confirm_window=4, max_misses=0,
                      confirm_deadline=True), "TFTT")
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed


def test_alternating_tentative_lingers_without_the_deadline_but_not_with_it():
    """The gap a miss budget cannot close.  Alternating hit and miss never
    gathers 3 hits in 4 (no confirmation) and never chains more than one
    miss (no death), so without the deadline it lives indefinitely.  With
    it, no tentative ever outlives its first window."""
    window = 4
    pattern = "TF" * 12

    legacy = _drive(_red(confirm_hits=3, confirm_window=window, max_misses=3),
                    pattern)
    assert any(not t.confirmed and legacy.t - t.born_at > window - 1
               for t in legacy.tracks), "expected a lingering tentative"

    trk = _red(confirm_hits=3, confirm_window=window, confirm_deadline=True)
    p, v = np.array([40.0, 40.0]), np.array([0.5, 0.0])
    for ch in pattern:
        p = p + v
        trk.step([red_det(BLUE_A, p, v, blue=0), red_det(BLUE_B, p, v, blue=1)]
                 if ch == "T" else [])
        for t in trk.tracks:
            assert t.confirmed or trk.t - t.born_at <= window - 1, (
                f"tentative aged {trk.t - t.born_at} outlived its window")


def test_deadline_does_not_touch_confirmed_tracks():
    trk = _drive(_red(confirm_hits=3, confirm_window=4, confirm_deadline=True),
                 "TTT" + "F" * 30)
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed


def test_confirmed_track_still_dies_past_max_misses():
    trk = _drive(_red(confirm_hits=3, confirm_window=4, confirm_deadline=True,
                      max_misses=5), "TTT" + "F" * 6)
    assert len(trk.tracks) == 0, "confirmed track outlived max_misses"


# --------------------------------------------------------------------- #
#  Obstacle tracker
# --------------------------------------------------------------------- #

def test_obstacle_deadline_deletes_a_tentative_that_fails_its_first_window():
    trk = _obs(confirm_hits=3, confirm_window=4, confirm_deadline=True)
    c = np.array([70.0, 70.0])
    for ch in "TFFT":
        trk.step([obs_det(BLUE_A, c, [0.0, 0.0], 8.0, blue=0),
                  obs_det(BLUE_B, c, [0.0, 0.0], 8.0, blue=1)]
                 if ch == "T" else [])
    assert len(trk.tracks) == 0


def test_obstacle_spurious_tentative_dies_by_the_deadline_despite_huge_max_misses():
    trk = _obs(max_misses=1000, confirm_deadline=True)
    trk.step([obs_det(BLUE_A, [40.0, 40.0], [0.0, 0.0], 5.0, truth_id=-1)])
    for _ in range(3):
        trk.step([])
    assert len(trk.tracks) == 0


def test_seen_obstacle_is_not_forgotten_during_a_long_absence():
    """A static obstacle does not go anywhere, so a confirmed one must
    survive a long stretch unobserved — the policy-facing config for
    obstacles is 'remember what you saw', while the deadline still clears
    spurious tentatives quickly."""
    trk = _obs(max_misses=1000, confirm_deadline=True)
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
