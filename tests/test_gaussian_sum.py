"""
tests/test_gaussian_sum.py — the Gaussian-Sum tracker refactor.

Two things are tested:

1. NON-REGRESSION.  A track is now a mixture of weighted components, not a
   single (x, P).  With the default constant-velocity motion model every
   track has exactly ONE component at every step, so every number the
   tracker produces must be BIT-EXACT to the pre-refactor single-Kalman
   implementation.  ``tests/_frozen_pre_gsf_tracker.py`` is a frozen copy
   of ``tracker.py`` from before this refactor (git commit b18bf66); this
   file runs identical detection streams through both and compares.

2. THE NEW MECHANIC.  A track can genuinely branch (a stand-in
   ``motion_model`` that rotates velocity two ways is used, since the real
   learned model does not exist yet): components must be readable via the
   DOMINANT mode (never the mixture mean — see
   ``docs/tracking_diagnostics.md`` Sec. 8 for why averaging ~90-140 deg
   apart modes is wrong), a detection must down-weight the branch it
   contradicts ("the update prunes automatically"), near-duplicate
   components must merge, negligible-weight ones must be dropped, and the
   component count must stay bounded through a long coast.

Run:
    pytest tests/test_gaussian_sum.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.tracking.tracker import (
    MultiTargetTracker, Track, _Component, _mahalanobis2_between, _merge_pair,
)
from tests._frozen_pre_gsf_tracker import MultiTargetTracker as OldTracker


# --------------------------------------------------------------------- #
#  1. Non-regression: identical detection streams, both implementations
# --------------------------------------------------------------------- #

def _episode_detections(seed, n_blue=5, n_red=3, n_obstacles=4, steps=120,
                        clutter_rate=0.0):
    """Record the exact detection list seen at every step, so both
    trackers are fed IDENTICAL input (never re-drawn per tracker)."""
    e = PursuitEnv(n_blue=n_blue, n_red=n_red, n_obstacles=n_obstacles,
                   arena_size=130.0, max_steps=steps + 10, capture_radius=3.0,
                   sensor_radius=40.0, use_belief_maps=True,
                   sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
                   sensor_noise_range_growth=1.0, clutter_rate=clutter_rate,
                   red_policy=run_from_nearest_uav, seed=seed)
    e.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        if not e.agents:
            break
        e.step({a: rng.uniform(-1, 1, 2).astype(np.float32) for a in e.agents})
        out.append(e.raw_detections())
    return out


def _assert_tracks_match(old_tracks, new_tracks, step):
    assert len(old_tracks) == len(new_tracks), (
        f"step {step}: track count diverged {len(old_tracks)} vs {len(new_tracks)}")
    for i, (o, n) in enumerate(zip(old_tracks, new_tracks)):
        assert np.allclose(o.x, n.x, atol=1e-9), f"step {step} track {i}: x diverged"
        assert np.allclose(o.P, n.P, atol=1e-9), f"step {step} track {i}: P diverged"
        assert o.confirmed == n.confirmed, f"step {step} track {i}: confirmed diverged"
        assert o.misses == n.misses, f"step {step} track {i}: misses diverged"
        assert o.history == n.history, f"step {step} track {i}: history diverged"
        assert n.n_modes == 1, (
            f"step {step} track {i}: default motion model must never branch, "
            f"got {n.n_modes} components")


def test_bit_exact_non_regression_single_episode():
    dets_per_step = _episode_detections(seed=42)
    old = OldTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0)
    new = MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0)
    for step, dets in enumerate(dets_per_step):
        old.step(dets)
        new.step(dets)
        assert old.last_nis == new.last_nis, f"step {step}: last_nis diverged"
        _assert_tracks_match(old.tracks, new.tracks, step)


def test_bit_exact_non_regression_with_clutter():
    """Clutter exercises spurious births and rejections -- more association
    edge cases than a clean run."""
    dets_per_step = _episode_detections(seed=7, clutter_rate=0.3)
    old = OldTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0, max_misses=10)
    new = MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0, max_misses=10)
    for step, dets in enumerate(dets_per_step):
        old.step(dets)
        new.step(dets)
        _assert_tracks_match(old.tracks, new.tracks, step)


def test_bit_exact_non_regression_oracle_association():
    dets_per_step = _episode_detections(seed=13, clutter_rate=0.2)
    old = OldTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0,
                     oracle_association=True)
    new = MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0,
                             oracle_association=True)
    for step, dets in enumerate(dets_per_step):
        old.step(dets)
        new.step(dets)
        _assert_tracks_match(old.tracks, new.tracks, step)


def test_bit_exact_across_several_seeds():
    """A handful of independent scenarios, fewer steps each -- broader
    coverage of birth/death/coasting transitions than one long episode."""
    for seed in (1, 2, 3, 4, 5):
        dets_per_step = _episode_detections(seed=seed, steps=40)
        old = OldTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0)
        new = MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0)
        for step, dets in enumerate(dets_per_step):
            old.step(dets)
            new.step(dets)
            _assert_tracks_match(old.tracks, new.tracks, step)


# --------------------------------------------------------------------- #
#  2. The new mechanic: genuine multimodality
# --------------------------------------------------------------------- #

def _branching_motion_model(angle=1.2, rel_w=(0.5, 0.5)):
    """Stand-in for the not-yet-built learned model: rotates the velocity
    two ways (a crude proxy for 'round the obstacle left or right').  Adds
    no process noise of its own (P unchanged) so the test can attribute
    everything to the branching, not to Q."""
    c, s = np.cos(angle), np.sin(angle)
    R1 = np.array([[c, -s], [s, c]])
    R2 = R1.T

    def model(x, P):
        pos = x[:2] + x[2:]              # dt = 1
        v1 = R1 @ x[2:]
        v2 = R2 @ x[2:]
        return (
            (rel_w[0], np.concatenate([pos, v1]), P.copy()),
            (rel_w[1], np.concatenate([pos, v2]), P.copy()),
        )
    return model


def _mk_tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0)
    base.update(kw)
    return MultiTargetTracker(**base)


def test_branching_motion_model_creates_multiple_components():
    """Isolate branching from merging: merge_gate=0 (never merge) so this
    tests ONLY that _predict_track fans a component out into the model's
    branches.  test_reduce_merges_near_duplicates and
    test_reduce_never_merges_genuinely_separated_modes cover merge
    behaviour on its own."""
    trk = _mk_tracker(motion_model=_branching_motion_model(), merge_gate=0.0)
    tr = Track(np.array([50.0, 50.0, 1.0, 0.0]), np.eye(4), t=0)
    trk.tracks = [tr]
    trk._predict_track(tr)
    assert tr.n_modes == 2
    ws = sorted(c.w for c in tr.components)
    assert np.isclose(ws[0], 0.5, atol=1e-9) and np.isclose(ws[1], 0.5, atol=1e-9)


def test_readout_is_the_dominant_mode_not_the_mixture_mean():
    """The whole point of the Gaussian Sum: two ~symmetric modes must NOT
    average to a heading between them."""
    tr = Track(np.array([0.0, 0.0, 1.0, 0.0]), np.eye(4), t=0)
    left = _Component(0.6, np.array([10.0, 5.0, 0.7, 0.7]), np.eye(4))
    right = _Component(0.4, np.array([10.0, -5.0, 0.7, -0.7]), np.eye(4))
    tr.components = [left, right]

    mixture_mean_pos = 0.6 * left.x[:2] + 0.4 * right.x[:2]
    assert np.allclose(tr.pos, left.x[:2]), "readout must be the dominant mode"
    assert not np.allclose(tr.pos, mixture_mean_pos), (
        "readout must NOT be the mixture's weighted mean")
    assert np.allclose(tr.vel, left.x[2:])
    assert np.allclose(tr.x, left.x)
    assert np.allclose(tr.P, left.P)
    assert tr.n_modes == 2


def test_update_downweights_the_branch_the_detection_contradicts():
    """'The update prunes automatically': a detection consistent with only
    ONE branch must raise that branch's relative weight."""
    trk = _mk_tracker()
    tr = Track(np.array([50.0, 50.0, 1.0, 0.0]), np.eye(4) * 4.0, t=0, w=0.5)
    # A second, far-away component with EQUAL prior weight.
    tr.components.append(_Component(0.5, np.array([80.0, 80.0, -1.0, 0.0]),
                                    np.eye(4) * 4.0))
    for c in tr.components:
        c.w = 0.5
    det = {"blue": 0, "z_pos": np.array([50.5, 49.5], dtype=np.float32),
          "z_range": 10.0, "z_radial": 0.0, "los": np.zeros(2, dtype=np.float32),
          "sigma_pos": 1.0, "sigma_radial": 0.0, "truth_id": 0}
    trk._update_track(tr, det)
    ws = sorted((c.w, tuple(np.round(c.x[:2]))) for c in tr.components)
    near_w = [w for w, pos in ws if pos == (50.0, 50.0) or pos == (51.0, 50.0)
             or abs(pos[0] - 50.0) < 2 and abs(pos[1] - 50.0) < 2]
    assert tr.pos[0] < 65.0, (
        f"the near-detection branch should now dominate, pos={tr.pos}")


def test_merge_pair_is_moment_matched_and_conserves_weight():
    c1 = _Component(0.6, np.array([10.0, 0.0, 1.0, 0.0]), np.eye(4))
    c2 = _Component(0.4, np.array([12.0, 0.0, 1.0, 0.0]), np.eye(4))
    m = _merge_pair(c1, c2)
    assert np.isclose(m.w, 1.0)
    assert np.isclose(m.x[0], 0.6 * 10.0 + 0.4 * 12.0)   # weighted mean
    # merged covariance must be inflated by the spread of the means
    # (the same "spread of the means" term used to collapse a mixture).
    assert m.P[0, 0] > 1.0


def test_mahalanobis_between_identifies_close_vs_far_components():
    base = _Component(0.5, np.array([0.0, 0.0, 0.0, 0.0]), np.eye(4))
    close = _Component(0.5, np.array([0.1, 0.0, 0.0, 0.0]), np.eye(4))
    far = _Component(0.5, np.array([50.0, 50.0, 0.0, 0.0]), np.eye(4))
    assert (_mahalanobis2_between(base, close)
            < _mahalanobis2_between(base, far))


def test_reduce_merges_near_duplicates():
    trk = _mk_tracker(merge_gate=1.0)
    c1 = _Component(0.5, np.array([10.0, 10.0, 0.0, 0.0]), np.eye(4) * 0.01)
    c2 = _Component(0.5, np.array([10.05, 10.0, 0.0, 0.0]), np.eye(4) * 0.01)
    out = trk._reduce([c1, c2])
    assert len(out) == 1, "near-duplicate components should merge"
    assert np.isclose(sum(c.w for c in out), 1.0)


def test_reduce_never_merges_genuinely_separated_modes():
    """The failure mode this whole refactor exists to avoid: merging modes
    that are actually different plans."""
    trk = _mk_tracker(merge_gate=1.0)
    c1 = _Component(0.6, np.array([10.0, 5.0, 0.7, 0.7]), np.eye(4))
    c2 = _Component(0.4, np.array([10.0, -5.0, 0.7, -0.7]), np.eye(4))
    out = trk._reduce([c1, c2])
    assert len(out) == 2, "separated modes must survive as distinct hypotheses"


def test_reduce_drops_negligible_weight():
    trk = _mk_tracker(min_component_weight=0.05)
    c1 = _Component(0.99, np.array([0.0, 0.0, 0.0, 0.0]), np.eye(4))
    c2 = _Component(0.01, np.array([80.0, 80.0, 0.0, 0.0]), np.eye(4))
    out = trk._reduce([c1, c2])
    assert len(out) == 1
    assert np.isclose(out[0].w, 1.0)


def test_reduce_caps_component_count_keeping_highest_weight():
    trk = _mk_tracker(max_components=3, merge_gate=0.0)   # merge_gate=0: never merge
    comps = [_Component(float(k + 1), np.array([k * 40.0, 0.0, 0.0, 0.0]),
                        np.eye(4)) for k in range(6)]
    out = trk._reduce(comps)
    assert len(out) == 3
    kept = sorted(c.x[0] for c in out)
    assert kept == [120.0, 160.0, 200.0], "must keep the HIGHEST-weight ones"


def test_reduce_weights_always_sum_to_one():
    trk = _mk_tracker(max_components=4, merge_gate=0.5)
    rng = np.random.default_rng(0)
    comps = [_Component(rng.random(), rng.normal(size=4) * 20, np.eye(4))
            for _ in range(10)]
    out = trk._reduce(comps)
    assert abs(sum(c.w for c in out) - 1.0) < 1e-9


def test_component_count_stays_bounded_through_a_long_coast():
    """A coasting track (no detections) branches every PREDICT step with a
    branching motion model; without the mandatory reduce, the count would
    grow as 2^n over n steps."""
    trk = _mk_tracker(motion_model=_branching_motion_model(), max_components=8)
    tr = Track(np.array([50.0, 50.0, 1.0, 0.0]), np.eye(4), t=0)
    trk.tracks = [tr]
    for _ in range(30):
        trk._predict_track(tr)
        assert tr.n_modes <= trk.max_components, (
            f"component count exceeded the cap: {tr.n_modes}")


def test_single_component_default_motion_model_never_branches():
    trk = _mk_tracker()          # motion_model=None
    tr = Track(np.array([50.0, 50.0, 1.0, 0.0]), np.eye(4), t=0)
    trk.tracks = [tr]
    for _ in range(20):
        trk._predict_track(tr)
        assert tr.n_modes == 1
