"""
tests/test_motion_batching.py — batching the learned motion model's calls.

Batching exists only for speed, so the one thing that matters is that it
changes NOTHING else: the batched categoricals, the branches built from
them, and the tracks the tracker ends up with must match the one-call-per-
component path.  Float32 batched matrix kernels can differ from batch-of-
one kernels in the last bits, hence tolerances rather than exact equality.

Run:
    pytest tests/test_motion_batching.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from isr.agents.learned_red_motion import LearnedRedMotion
from isr.agents.red_motion_gnn import RedMotionGNN
from isr.tracking import MultiTargetTracker
from tests.test_tracker import det

L = 130.0
BLUE_CAP, OBS_CAP = 6, 5


def _adapter(**kw) -> LearnedRedMotion:
    torch.manual_seed(0)
    model = RedMotionGNN(n_blue=BLUE_CAP, n_red=1, n_obs=OBS_CAP)
    ad = LearnedRedMotion(model, BLUE_CAP, OBS_CAP, arena_size=L, **kw)
    ad.set_context(
        blue_pos=np.array([[30.0, 40.0], [90.0, 70.0], [60.0, 110.0]]),
        blue_vel=np.array([[0.5, 0.0], [0.0, -0.4], [-0.3, 0.2]]),
        obs_pos=np.array([[65.0, 65.0]]), obs_vel=np.zeros((1, 2)),
        obs_r=np.array([10.0]))
    return ad


class PerCallOnly:
    """Exposes ONLY __call__, so the tracker cannot find predict_batch and
    takes the one-call-per-component path."""

    def __init__(self, ad):
        self.ad = ad

    def __call__(self, x, P):
        return self.ad(x, P)


def _states(k, seed=0):
    rng = np.random.default_rng(seed)
    xs = [np.concatenate([rng.uniform(10, 120, 2), rng.uniform(-1, 1, 2)])
          for _ in range(k)]
    Ps = [np.diag(rng.uniform(0.5, 4.0, 4)) for _ in range(k)]
    return xs, Ps


# --------------------------------------------------------------------- #
#  Adapter
# --------------------------------------------------------------------- #

def test_batched_probabilities_match_one_at_a_time():
    ad = _adapter()
    xs, _ = _states(7)
    pos = np.stack([x[:2] for x in xs])
    vel = np.stack([x[2:] for x in xs])
    batched = ad.predict_probs_batch(pos, vel)
    single = np.stack([ad.predict_probs(p, v) for p, v in zip(pos, vel)])
    assert batched.shape == single.shape
    assert np.allclose(batched, single, atol=1e-6)


def test_batched_probabilities_match_when_blues_are_truncated_per_sample():
    """More blues than the trained capacity: each sample keeps its OWN
    nearest blues, which differ between samples — the batch must not share
    one truncation across the rows."""
    ad = _adapter()
    rng = np.random.default_rng(1)
    ad.set_context(blue_pos=rng.uniform(0, L, (BLUE_CAP + 3, 2)),
                   blue_vel=rng.uniform(-1, 1, (BLUE_CAP + 3, 2)))
    xs, _ = _states(5, seed=2)
    pos = np.stack([x[:2] for x in xs])
    vel = np.stack([x[2:] for x in xs])
    single = np.stack([ad.predict_probs(p, v) for p, v in zip(pos, vel)])
    assert np.allclose(ad.predict_probs_batch(pos, vel), single, atol=1e-6)


def test_batched_branches_match_one_at_a_time():
    ad = _adapter()
    xs, Ps = _states(6, seed=3)
    batched = ad.predict_batch(xs, Ps)
    assert len(batched) == len(xs)
    for (x, P), got in zip(zip(xs, Ps), batched):
        want = ad(x, P)
        assert len(got) == len(want)
        for (wg, xg, Pg), (ww, xw, Pw) in zip(got, want):
            assert wg == pytest.approx(ww, abs=1e-6)
            assert np.allclose(xg, xw, atol=1e-6)
            assert np.allclose(Pg, Pw, atol=1e-6)


def test_empty_batch_is_empty():
    ad = _adapter()
    assert ad.predict_batch([], []) == []
    assert ad.predict_probs_batch(np.zeros((0, 2)), np.zeros((0, 2))).shape[0] == 0


def test_batch_requires_a_context():
    torch.manual_seed(0)
    ad = LearnedRedMotion(RedMotionGNN(n_blue=BLUE_CAP, n_red=1, n_obs=OBS_CAP),
                          BLUE_CAP, OBS_CAP, arena_size=L)
    xs, Ps = _states(2)
    with pytest.raises(RuntimeError, match="set_context"):
        ad.predict_batch(xs, Ps)


# --------------------------------------------------------------------- #
#  Tracker
# --------------------------------------------------------------------- #

class CountingBatch:
    """Batch-capable wrapper that counts how often each path is used."""

    def __init__(self, ad):
        self.ad, self.batch_calls, self.single_calls = ad, 0, 0

    def __call__(self, x, P):
        self.single_calls += 1
        return self.ad(x, P)

    def predict_batch(self, xs, Ps):
        self.batch_calls += 1
        return self.ad.predict_batch(xs, Ps)


def _scenario(motion, steps=25):
    """Three targets seen by two blues, with a gap so tracks coast — every
    PREDICT sees several tracks and, while coasting, grown covariances."""
    trk = MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0,
                             confirm_hits=2, max_misses=10,
                             motion_model=motion, max_components=8)
    b0, b1 = np.array([20.0, 60.0]), np.array([60.0, 20.0])
    ps = [np.array([40.0, 40.0]), np.array([80.0, 50.0]), np.array([50.0, 90.0])]
    vs = [np.array([0.5, 0.2]), np.array([-0.4, 0.3]), np.array([0.2, -0.5])]
    for t in range(steps):
        ps = [p + v for p, v in zip(ps, vs)]
        dets = [] if 10 <= t < 14 else [
            det(b, p, v, blue=i, truth_id=k)
            for k, (p, v) in enumerate(zip(ps, vs))
            for i, b in enumerate((b0, b1))]
        trk.step(dets)
    return trk


def test_tracker_with_batching_matches_the_per_call_path():
    per_call = _scenario(PerCallOnly(_adapter()))
    batched = _scenario(_adapter())
    assert len(per_call.tracks) == len(batched.tracks) > 0
    for a, b in zip(per_call.tracks, batched.tracks):
        assert a.confirmed == b.confirmed and a.misses == b.misses
        assert len(a.components) == len(b.components)
        assert np.allclose(a.x, b.x, atol=1e-5)
        assert np.allclose(a.P, b.P, atol=1e-5)


def test_tracker_makes_one_batched_call_per_step_with_tracks():
    counter = CountingBatch(_adapter())
    trk = _scenario(counter, steps=12)
    assert counter.single_calls == 0, "batch path fell back to per-call"
    # The first step has no tracks yet, so nothing to predict.
    assert counter.batch_calls == 11


def test_models_without_predict_batch_still_work():
    """Lambdas and plain callables keep the original path."""
    ad = _adapter()
    trk = _scenario(lambda x, P: ad(x, P), steps=8)
    assert trk.tracks
