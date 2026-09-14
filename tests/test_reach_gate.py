"""
tests/test_reach_gate.py — the kinematic reach gate (max_target_speed).

A coasting track's chi^2 gate grows with its predicted covariance, far
faster than a target can move, so it catches clutter that no real target
could have produced; the hit then resets the track and it lives on clutter.
The reach gate bounds association by physics instead: a return joins a
track only if it lies within max_target_speed * scans_since_hit of where
the track stood at its last hit, plus measurement margin.

Run:
    pytest tests/test_reach_gate.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.tracking import MultiTargetTracker
from tests.test_tracker import det

SPEED = float(np.sqrt(2.0))          # red v_max 1 per axis -> sqrt(2) speed
BLUE = np.array([0.0, 0.0])


def _tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=2,
                max_misses=5)
    base.update(kw)
    return MultiTargetTracker(**base)


def _confirmed_stationary_track(trk, pos=(20.0, 0.0), scans=5):
    for _ in range(scans):
        trk.step([det(BLUE, pos, (0.0, 0.0))])
    (tr,) = trk.tracks
    assert tr.confirmed
    return tr


def _coast(trk, scans):
    for _ in range(scans):
        trk.step([])


def _clutter(pos):
    return det(BLUE, pos, (0.0, 0.0), truth_id=-1)


# --------------------------------------------------------------------- #

def test_off_by_default():
    trk = _tracker()
    assert trk.max_target_speed is None
    tr = _confirmed_stationary_track(trk)
    assert trk._within_reach(tr, _clutter((180.0, 0.0)))


def test_non_positive_speed_rejected():
    with pytest.raises(ValueError):
        _tracker(max_target_speed=0.0)


def test_coasting_track_without_the_gate_swallows_clutter():
    """The failure being fixed: after 3 coasted scans the CV gate is wide
    enough to take a plot 12 m away, which no stationary-or-slow target
    could have produced, and the hit resets the track."""
    trk = _tracker()
    tr = _confirmed_stationary_track(trk)
    _coast(trk, 3)
    trk.step([_clutter((32.0, 0.0))])
    assert len(trk.tracks) == 1 and trk.tracks[0] is tr
    assert tr.steps_since_hit == 0, "clutter was accepted as a hit"


def test_the_gate_rejects_clutter_out_of_reach():
    trk = _tracker(max_target_speed=SPEED)
    tr = _confirmed_stationary_track(trk)
    _coast(trk, 3)
    trk.step([_clutter((32.0, 0.0))])
    assert tr in trk.tracks
    assert tr.steps_since_hit == 4, "clutter must not count as a hit"
    assert len(trk.tracks) == 2, "the rejected plot starts its own tentative"
    assert np.allclose(tr.last_hit_pos, (20.0, 0.0), atol=1.0)


def test_a_target_at_full_diagonal_speed_is_still_reacquired():
    """The gate must never cut a real target: moving at the fastest speed
    the env allows, re-detected after a long gap, it joins its old track."""
    trk = _tracker(max_target_speed=SPEED, max_misses=10)
    vel = np.array([1.0, 1.0])
    p = np.array([20.0, 20.0])
    for _ in range(5):
        p = p + vel
        trk.step([det(BLUE, p, vel)])
    (tr,) = trk.tracks
    for _ in range(6):
        p = p + vel
        trk.step([])
    p = p + vel
    trk.step([det(BLUE, p, vel)])
    assert trk.tracks == [tr]
    assert tr.steps_since_hit == 0


def test_reach_grows_with_scans_since_the_last_hit():
    trk = _tracker(max_target_speed=SPEED, reach_k_sigma=3.0)
    tr = _confirmed_stationary_track(trk, pos=(0.0, 0.0))
    tr.last_hit_pos = np.zeros(2)
    tr.last_hit_sd = 0.0
    margin = 3.0 * 1.0                        # det sigma_pos = 1, sd_hit = 0
    for ssh in (0, 4):
        tr.steps_since_hit = ssh
        reach = SPEED * (ssh + 1) + margin
        assert trk._within_reach(tr, _clutter((reach - 0.05, 0.0)))
        assert not trk._within_reach(tr, _clutter((reach + 0.05, 0.0)))


def test_margin_uses_both_the_return_and_the_track_uncertainty():
    trk = _tracker(max_target_speed=SPEED, reach_k_sigma=2.0)
    tr = _confirmed_stationary_track(trk, pos=(0.0, 0.0))
    tr.last_hit_pos, tr.last_hit_sd, tr.steps_since_hit = np.zeros(2), 4.0, 0
    far = det(BLUE, (SPEED + 2.0 * 5.0 - 0.05, 0.0), (0.0, 0.0),
              sigma_pos=3.0, truth_id=-1)          # hypot(3, 4) = 5
    assert trk._within_reach(tr, far)


def test_anchor_moves_on_hits_only():
    trk = _tracker(max_target_speed=SPEED)
    tr = _confirmed_stationary_track(trk, pos=(20.0, 0.0))
    anchor = tr.last_hit_pos.copy()
    _coast(trk, 2)
    assert np.array_equal(tr.last_hit_pos, anchor), "a miss moved the anchor"
    trk.step([det(BLUE, (21.0, 0.0), (0.0, 0.0))])
    assert tr.steps_since_hit == 0
    assert tr.last_hit_pos[0] > anchor[0]
