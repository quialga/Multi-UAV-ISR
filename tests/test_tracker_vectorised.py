"""
tests/test_tracker_vectorised.py — the batched gate and merge search must
compute exactly what the per-pair versions did.

Both loops scaled with the number of Gaussian-Sum components, which only
started to matter once a motion model branched for real (docs
§11.7): the gate scored every (track, return) pair against every component
in Python, and the reduce inverted a 4x4 for every pair of components,
repeatedly.  The batched versions are equivalence-checked here against the
straightforward implementations they replace.

Run:
    pytest tests/test_tracker_vectorised.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.tracking.tracker import (
    MultiTargetTracker, Track, _Component, _mahalanobis2_between,
)
from tests.test_tracker import det

BLUE = np.array([0.0, 0.0])


def _tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=2,
                max_misses=5)
    base.update(kw)
    return MultiTargetTracker(**base)


def _random_component(rng, spread=30.0):
    x = np.concatenate([rng.uniform(0, 200, 2), rng.uniform(-1.5, 1.5, 2)])
    A = rng.normal(size=(4, 4))
    P = A @ A.T * rng.uniform(0.05, spread) + np.eye(4) * 1e-3
    return _Component(float(rng.uniform(0.05, 1.0)), x, P)


def _random_track(rng, n_comp, t=0):
    comps = [_random_component(rng) for _ in range(n_comp)]
    tr = Track(comps[0].x, comps[0].P, t=t)
    tr.components = comps
    tr.confirmed = True
    return tr


def _random_dets(rng, m, blue=0):
    return [det(BLUE, rng.uniform(0, 200, 2), rng.uniform(-1, 1, 2),
                blue=blue, sigma_pos=float(rng.uniform(0.5, 3.0)),
                sigma_radial=float(rng.uniform(0.05, 0.3)))
            for _ in range(m)]


def _reference_gate(trk, idxs, dets):
    """What the per-pair loop computed, verbatim."""
    n, m = len(trk.tracks), len(idxs)
    cost = np.zeros((n, m))
    gate = np.zeros((n, m), dtype=bool)
    thresh = [trk._gate_threshold(dets[k]) for k in idxs]
    for i, tr in enumerate(trk.tracks):
        if trk._is_lost(tr):
            continue
        for j, k in enumerate(idxs):
            d2, c = trk._gate_pos(tr, dets[k])
            cost[i, j] = c
            gate[i, j] = d2 <= thresh[j]
    return cost, gate


def _reference_pair(trk, comps):
    """What the double loop in _reduce picked, verbatim."""
    best = None
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            d2 = _mahalanobis2_between(comps[i], comps[j])
            if d2 <= trk.merge_gate and (best is None or d2 < best[0]):
                best = (d2, i, j)
    return None if best is None else (best[1], best[2])


# --------------------------------------------------------------------- #
#  Gate
# --------------------------------------------------------------------- #

def test_gate_matrix_matches_the_per_pair_loop():
    rng = np.random.default_rng(0)
    for _ in range(40):
        trk = _tracker()
        trk.tracks = [_random_track(rng, int(rng.integers(1, 7)))
                      for _ in range(int(rng.integers(1, 6)))]
        dets = _random_dets(rng, int(rng.integers(1, 6)))
        idxs = list(range(len(dets)))
        cost, gate = trk._gate_matrix(idxs, dets)
        ref_cost, ref_gate = _reference_gate(trk, idxs, dets)
        np.testing.assert_array_equal(gate, ref_gate)
        np.testing.assert_allclose(cost, ref_cost, rtol=1e-9, atol=1e-9)


def test_gate_matrix_keeps_lost_tracks_out():
    rng = np.random.default_rng(1)
    trk = _tracker(reacquire_after=4)
    trk.tracks = [_random_track(rng, 3) for _ in range(3)]
    trk.tracks[1].steps_since_hit = 10           # lost
    dets = _random_dets(rng, 4)
    _cost, gate = trk._gate_matrix(list(range(len(dets))), dets)
    assert not gate[1].any()
    ref_cost, ref_gate = _reference_gate(trk, list(range(len(dets))), dets)
    np.testing.assert_array_equal(gate, ref_gate)


def test_gate_matrix_falls_back_for_doppler_gating():
    rng = np.random.default_rng(2)
    trk = _tracker(doppler_gating=True)
    trk.tracks = [_random_track(rng, 4) for _ in range(3)]
    dets = _random_dets(rng, 3)
    idxs = list(range(len(dets)))
    cost, gate = trk._gate_matrix(idxs, dets)
    ref_cost, ref_gate = _reference_gate(trk, idxs, dets)
    np.testing.assert_array_equal(gate, ref_gate)
    np.testing.assert_allclose(cost, ref_cost, rtol=1e-12, atol=1e-12)


def test_gate_matrix_handles_no_tracks_and_no_returns():
    trk = _tracker()
    cost, gate = trk._gate_matrix([], [])
    assert cost.shape == (0, 0) and gate.shape == (0, 0)


# --------------------------------------------------------------------- #
#  Merge search
# --------------------------------------------------------------------- #

def test_closest_pair_matches_the_double_loop():
    rng = np.random.default_rng(3)
    for gate in (0.25, 2.0, 4.0, 1e9):
        trk = _tracker(merge_gate=gate)
        for _ in range(30):
            comps = [_random_component(rng, spread=8.0)
                     for _ in range(int(rng.integers(2, 9)))]
            assert trk._closest_pair(comps) == _reference_pair(trk, comps)


def test_closest_pair_is_none_when_nothing_is_close_enough():
    rng = np.random.default_rng(4)
    trk = _tracker(merge_gate=1e-9)
    comps = [_random_component(rng) for _ in range(4)]
    assert trk._closest_pair(comps) is None


def test_closest_pair_tie_break_is_the_first_in_scan_order():
    """Identical distances: the (i < j) scan order decides, as before."""
    trk = _tracker(merge_gate=100.0)
    P = np.eye(4)
    a = _Component(0.4, np.zeros(4), P)
    b = _Component(0.3, np.array([1.0, 0, 0, 0]), P)
    c = _Component(0.3, np.array([2.0, 0, 0, 0]), P)   # same gap b->c
    comps = [a, b, c]
    assert trk._closest_pair(comps) == (0, 1) == _reference_pair(trk, comps)


def test_reduce_still_merges_and_caps():
    trk = _tracker(merge_gate=4.0, max_components=3)
    P = np.eye(4) * 0.5
    twins = [_Component(0.2, np.zeros(4), P),
             _Component(0.2, np.array([0.05, 0, 0, 0]), P)]
    far = [_Component(0.2, np.array([50.0 * k, 0, 0, 0]), P) for k in (1, 2, 3)]
    out = trk._reduce(twins + far)
    assert len(out) == 3
    assert sum(c.w for c in out) == pytest.approx(1.0)
