"""
tests/test_actor_graph.py — trackers -> actor node slots
(isr.tracking.actor_graph).

Pins the agreed observation rules: confirmed tracks only, a position-sd
cut-off, overflow keeps the smallest sd, readout from the dominant
component with the whole mixture's moment-matched covariance, binary
presence, bounded uncertainty features, and the estimated (not true)
obstacle field.

Run:
    pytest tests/test_actor_graph.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.tracking import actor_graph as ag
from isr.tracking.coverage import disk_occluder, segment_disk_occluded
from isr.tracking.obstacle_tracker import ObstacleTrack
from isr.tracking.tracker import Track, _Component

CUT, K = 40.0, 3
SCALE_P, SCALE_V, COAST = 40.0, 1.0, 80


def _red(pos=(50.0, 50.0), vel=(0.5, 0.0), sd=2.0, confirmed=True, ssh=0):
    P = np.diag([sd ** 2, sd ** 2, 0.25, 0.25])
    tr = Track(np.array([*pos, *vel], dtype=float), P, t=0)
    tr.confirmed = confirmed
    tr.steps_since_hit = ssh
    return tr


def _obstacle(pos=(80.0, 60.0), r=10.0, sd=0.5, sd_r=0.4, confirmed=True):
    P = np.diag([sd ** 2, sd ** 2, 1e-12, 1e-12, sd_r ** 2])
    tr = ObstacleTrack(np.array([*pos, 0.0, 0.0, r]), P, t=0)
    tr.confirmed = confirmed
    return tr


def _slots(tracks, k=K):
    return ag.red_slots(tracks, k, CUT, SCALE_P, SCALE_V, COAST)


# --------------------------------------------------------------------- #
#  Reds
# --------------------------------------------------------------------- #

def test_tentative_tracks_are_not_nodes():
    s = _slots([_red(confirmed=False), _red(pos=(10, 10))])
    assert s["present"].tolist() == [1.0, 0.0, 0.0]
    assert np.allclose(s["pos"][0], (10, 10))


def test_tracks_above_the_sigma_cutoff_are_dropped():
    s = _slots([_red(sd=CUT + 0.1), _red(pos=(20, 20), sd=CUT)])
    assert s["present"].sum() == 1.0
    assert np.allclose(s["pos"][0], (20, 20))


def test_cutoff_uses_the_largest_axis():
    P = np.diag([1.0, (CUT + 1.0) ** 2, 0.25, 0.25])
    tr = Track(np.array([5.0, 5.0, 0, 0]), P, t=0)
    tr.confirmed = True
    assert _slots([tr])["present"].sum() == 0.0


def test_overflow_keeps_the_smallest_sigma():
    tracks = [_red(pos=(i, i), sd=sd) for i, sd in enumerate((9.0, 1.0, 5.0, 3.0, 7.0))]
    s = _slots(tracks)
    assert s["present"].tolist() == [1.0, 1.0, 1.0]
    assert [tuple(p) for p in s["pos"]] == [(1, 1), (3, 3), (2, 2)]


def test_padding_is_all_zero():
    s = _slots([_red()])
    for key in ("pos", "vel", "feats", "present"):
        assert not np.any(s[key][1:])


def test_readout_is_the_dominant_component_and_cov_the_whole_mixture():
    tr = _red()
    a = _Component(0.7, np.array([50.0, 50.0, 0.5, 0.0]), np.diag([1.0, 1.0, 0.1, 0.1]))
    b = _Component(0.3, np.array([56.0, 50.0, 0.0, 0.5]), np.diag([1.0, 1.0, 0.1, 0.1]))
    tr.components = [a, b]
    s = _slots([tr])
    assert np.allclose(s["pos"][0], (50.0, 50.0))
    assert np.allclose(s["vel"][0], (0.5, 0.0))
    P = ag.mixture_covariance(tr)
    # Between-component spread along x: 0.7*0.3*6^2 = 7.56, plus 1.
    assert P[0, 0] == pytest.approx(1.0 + 7.56)
    assert s["feats"][0, 1] == pytest.approx(np.tanh(np.sqrt(8.56) / SCALE_P), rel=1e-5)
    assert s["feats"][0, 2] == pytest.approx(np.tanh(1.0 / SCALE_P), rel=1e-5)


def test_feature_layout_and_bounds():
    tr = _red(sd=4.0, ssh=20)
    s = _slots([tr])
    f = s["feats"][0]
    assert f.shape == (ag.RED_FEAT_DIM,)
    assert f[0] == 1.0
    assert f[1] == pytest.approx(np.tanh(4.0 / SCALE_P), rel=1e-5)
    assert f[3] == pytest.approx(0.0)
    assert f[4] == pytest.approx(np.tanh(0.5 / SCALE_V), rel=1e-5)
    assert f[7] == pytest.approx(20 / COAST)
    assert _slots([_red(ssh=500)])["feats"][0, 7] == 1.0, "staleness is capped"
    assert np.all(np.abs(s["feats"]) <= 1.0)


def test_covariance_features_correlation():
    P = np.array([[4.0, 3.0], [3.0, 9.0]])
    sx, sy, rho = ag.cov2_features(P, scale=10.0)
    assert sx == pytest.approx(np.tanh(0.2))
    assert sy == pytest.approx(np.tanh(0.3))
    assert rho == pytest.approx(0.5)
    assert ag.cov2_features(np.zeros((2, 2)), 1.0) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------- #
#  Obstacles
# --------------------------------------------------------------------- #

def test_obstacle_slots_use_the_estimated_radius_and_confirmed_tracks():
    tracks = [_obstacle(r=12.3), _obstacle(pos=(10, 10), confirmed=False)]
    s = ag.obstacle_slots(tracks, 4, arena_size=200.0, pos_scale=SCALE_P,
                          vel_scale=SCALE_V)
    assert s["present"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert s["radius"][0] == pytest.approx(12.3)
    f = s["feats"][0]
    assert f.shape == (ag.OBS_FEAT_DIM,)
    assert f[1] == pytest.approx(12.3 / 200.0)
    assert f[2] == pytest.approx(np.tanh(0.4 / ag.RADIUS_SD_SCALE), rel=1e-5)


def test_obstacle_overflow_keeps_the_smallest_sigma():
    tracks = [_obstacle(pos=(i * 30.0, 0.0), sd=sd) for i, sd in enumerate((3.0, 0.2, 1.0))]
    s = ag.obstacle_slots(tracks, 2, 200.0, SCALE_P, SCALE_V)
    assert [p[0] for p in s["pos"]] == [30.0, 60.0]


def test_confirmed_obstacle_geometry():
    tracks = [_obstacle(pos=(1, 2), r=5.0), _obstacle(confirmed=False)]
    c, v, r = ag.confirmed_obstacle_geometry(tracks)
    assert c.tolist() == [[1.0, 2.0]] and r.tolist() == [5.0] and v.shape == (1, 2)
    c, v, r = ag.confirmed_obstacle_geometry([])
    assert c.shape == (0, 2) and r.shape == (0,)


# --------------------------------------------------------------------- #
#  Occlusion on estimated disks
# --------------------------------------------------------------------- #

def test_disk_occluder_matches_the_exact_test():
    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 200, (6, 2))
    radii = rng.uniform(5, 15, 6)
    occ = disk_occluder(centres, radii, margin=2.5)
    for _ in range(20):
        o = rng.uniform(0, 200, 2)
        pts = rng.uniform(0, 200, (30, 2))
        assert np.array_equal(occ(o, pts),
                              segment_disk_occluded(o, pts, centres, radii, 2.5))


def test_disk_occluder_blocks_behind_a_disk_only():
    occ = disk_occluder(np.array([[50.0, 0.0]]), np.array([10.0]), margin=2.5)
    got = occ(np.array([0.0, 0.0]), np.array([[100.0, 0.0], [20.0, 0.0], [100.0, 40.0]]))
    assert got.tolist() == [True, False, False]
