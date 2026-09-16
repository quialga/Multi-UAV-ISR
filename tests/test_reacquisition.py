"""
tests/test_reacquisition.py — re-acquisition with confirmation
(MultiTargetTracker reacquire_after).

After a few coasted scans a single return on a confirmed track is more
likely clutter than the target, and accepting it resets the track onto the
false plot.  With ``reacquire_after`` a track that long without a hit is
LOST for association: returns near it start a tentative, which must pass
M-of-N and, when it confirms, takes over the lost track's id.

Run:
    pytest tests/test_reacquisition.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.tracking import MultiTargetTracker
from tests.test_tracker import det

BLUE = np.array([0.0, 0.0])
STILL = (0.0, 0.0)


def _tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=3,
                confirm_window=4, confirm_deadline=True, max_misses=30)
    base.update(kw)
    return MultiTargetTracker(**base)


def _confirm(trk, pos=(20.0, 0.0), scans=5):
    for _ in range(scans):
        trk.step([det(BLUE, pos, STILL)])
    (tr,) = trk.confirmed_tracks()
    return tr


def _coast(trk, scans):
    for _ in range(scans):
        trk.step([])


def _clutter(pos):
    return det(BLUE, pos, STILL, truth_id=-1)


# --------------------------------------------------------------------- #

def test_invalid_value_rejected():
    with pytest.raises(ValueError):
        _tracker(reacquire_after=0)


def test_off_by_default_a_single_plot_reattaches_after_a_long_coast():
    trk = _tracker()
    tr = _confirm(trk)
    _coast(trk, 5)
    trk.step([_clutter((26.0, 0.0))])
    assert tr.steps_since_hit == 0, "original behaviour: the plot is a hit"


def test_a_lost_track_does_not_take_a_single_plot():
    trk = _tracker(reacquire_after=4)
    tr = _confirm(trk)
    _coast(trk, 5)
    trk.step([_clutter((26.0, 0.0))])
    assert tr.steps_since_hit == 6, "a lost track must not be reset by one plot"
    assert np.allclose(tr.pos, (20.0, 0.0), atol=0.5)
    assert len(trk.tracks) == 2 and not trk.tracks[1].confirmed


def test_a_short_gap_still_associates_directly():
    """reacquire_after=4: after 2 coasted scans the next return is the 3rd
    scan since the hit, below the threshold, so it is a normal hit."""
    trk = _tracker(reacquire_after=4)
    tr = _confirm(trk)
    _coast(trk, 2)
    trk.step([det(BLUE, (20.0, 0.0), STILL)])
    assert trk.tracks == [tr] and tr.steps_since_hit == 0


def test_the_threshold_is_scans_since_the_hit():
    trk = _tracker(reacquire_after=4)
    tr = _confirm(trk)
    _coast(trk, 3)                       # the next return is scan 4: lost
    trk.step([det(BLUE, (20.0, 0.0), STILL)])
    assert tr.steps_since_hit == 4
    assert len(trk.tracks) == 2


def test_real_reacquisition_confirms_and_keeps_the_identity():
    trk = _tracker(reacquire_after=4)
    tr = _confirm(trk)
    old_id = tr.id
    _coast(trk, 6)
    for k in range(3):                   # 3-of-4 needs three hits
        trk.step([det(BLUE, (20.0, 0.0), STILL)])
        if k < 2:
            assert len(trk.confirmed_tracks()) == 1, "not confirmed yet"
    (conf,) = trk.confirmed_tracks()
    assert conf.id == old_id, "the re-acquired track takes over the old id"
    assert conf is not tr and tr not in trk.tracks
    assert len(trk.tracks) == 1
    assert conf.steps_since_hit == 0


def test_clutter_that_never_confirms_leaves_the_lost_track_alone():
    trk = _tracker(reacquire_after=4)
    tr = _confirm(trk)
    _coast(trk, 5)
    trk.step([_clutter((26.0, 0.0))])
    trk.step([_clutter((12.0, 5.0))])
    _coast(trk, 3)
    assert trk.tracks == [tr], "clutter tentatives die at their deadline"
    assert np.allclose(tr.pos, (20.0, 0.0), atol=0.5)


def test_absorbs_only_a_consistent_lost_track_and_the_closest_one():
    trk = _tracker(reacquire_after=4)
    for _ in range(5):
        trk.step([det(BLUE, (20.0, 0.0), STILL, blue=0),
                  det(BLUE, (30.0, 0.0), STILL, blue=1),
                  det(BLUE, (150.0, 0.0), STILL, blue=2)])
    near, closer, far = sorted(trk.confirmed_tracks(), key=lambda t: t.pos[0])
    _coast(trk, 6)
    for _ in range(3):
        trk.step([det(BLUE, (28.0, 0.0), STILL)])
    ids = {t.id for t in trk.confirmed_tracks()}
    assert closer.id in ids and closer not in trk.tracks, "closest absorbed"
    assert near in trk.tracks, "only one lost track is absorbed"
    assert far in trk.tracks, "an inconsistent lost track is untouched"

