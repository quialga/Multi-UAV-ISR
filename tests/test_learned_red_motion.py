"""
tests/test_learned_red_motion.py — the trained red-motion model wired into
the tracker's ``motion_model`` plug point.

The load-bearing test here is the TRAIN/SERVE one: the adapter must build
byte-identical network inputs to the training path for the same physical
situation.  Everything else in this file is contract checking.

Run:
    pytest tests/test_learned_red_motion.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_red_motion_dataset import collect_episode          # noqa: E402
from isr.agents.learned_red_motion import (                     # noqa: E402
    LearnedRedMotion, _INPUT_KEYS,
)
from isr.agents.red_motion_features import (                    # noqa: E402
    N_BINS, N_MAGNITUDE_BINS, V_NORM, ZERO_CLASS, featurize_shard,
)
from isr.agents.red_motion_gnn import RedMotionGNN              # noqa: E402
from isr.tracking.tracker import MultiTargetTracker             # noqa: E402

L = 130.0
BLUE_CAP, OBS_CAP = 6, 5


def _adapter(**kw) -> LearnedRedMotion:
    """An UNTRAINED network is fine for every contract here — none of these
    tests assert anything about prediction quality, only about shapes,
    conventions and the mixture algebra."""
    model = RedMotionGNN(n_blue=BLUE_CAP, n_red=1, n_obs=OBS_CAP)
    return LearnedRedMotion(model, BLUE_CAP, OBS_CAP, arena_size=L, **kw)


def _context_from_sample(s: dict) -> dict:
    """Invert the collector's ego-centric normalisation back to absolute
    world coordinates — what a live caller would pass to set_context."""
    rp = s["red_pos"].astype(np.float64)
    nb, no = int(s["n_blue"]), int(s["n_obs_placed"])
    return dict(
        blue_pos=rp + s["blue_rel_pos"][:nb].astype(np.float64) * L,
        blue_vel=s["blue_rel_vel"][:nb].astype(np.float64) * V_NORM,
        obs_pos=rp + s["obs_rel_pos"][:no].astype(np.float64) * L,
        obs_vel=s["obs_rel_vel"][:no].astype(np.float64) * V_NORM,
        obs_r=s["obs_radius"][:no].astype(np.float64) * L,
    )


# --------------------------------------------------------------------- #
#  The one that matters
# --------------------------------------------------------------------- #

def test_adapter_features_match_the_training_path_exactly():
    """Same physical situation -> byte-identical network inputs.

    Train/serve skew is the defining failure mode for this adapter, and it
    would be silent: the model would simply predict worse at inference
    than in validation, with nothing pointing at the cause.  Here a real
    collector sample is inverted back to absolute world coordinates, fed
    through set_context, and the resulting tensors compared against
    featurize_shard applied to the ORIGINAL sample.
    """
    rng = np.random.default_rng(0)
    samples = collect_episode(
        rng, ep_id=0, steps=40, n_blue_range=(2, BLUE_CAP),
        n_red_range=(1, 3), n_obs_range=(0, OBS_CAP),
        blue_cap=BLUE_CAP, obs_cap=OBS_CAP, arena_size=L, p_deterministic=0.3)
    assert samples

    ad = _adapter()
    for s in samples[::7]:
        want = featurize_shard(
            {k: np.asarray(v)[None, ...] for k, v in s.items()}, arena_size=L)
        ad.set_context(**_context_from_sample(s))
        got = featurize_shard(
            ad._pack(s["red_pos"].astype(np.float64),
                    s["red_vel"].astype(np.float64)), arena_size=L)
        for k in _INPUT_KEYS:
            assert np.allclose(got[k], want[k], atol=1e-6), (
                f"{k} differs between the training and inference paths")


# --------------------------------------------------------------------- #
#  Mixture contract
# --------------------------------------------------------------------- #

def test_branches_form_a_valid_weighted_mixture():
    ad = _adapter(max_branches=4)
    ad.set_context(blue_pos=np.array([[40.0, 60.0], [90.0, 70.0]]),
                  blue_vel=np.array([[1.0, 0.0], [0.0, -1.0]]))
    x = np.array([65.0, 65.0, 0.5, -0.3])
    P = np.diag([4.0, 4.0, 1.0, 1.0])
    branches = ad(x, P)
    assert 1 <= len(branches) <= 5          # <= max_branches modes + ZERO
    for w, xb, Pb in branches:
        assert w > 0.0
        assert xb.shape == (4,) and Pb.shape == (4, 4)
        assert np.all(np.isfinite(xb)) and np.all(np.isfinite(Pb))
        assert np.allclose(Pb, Pb.T), "covariance not symmetric"
        assert np.min(np.linalg.eigvalsh(Pb)) > 0, "covariance not PD"


def test_branch_weights_sum_to_one_so_no_weight_flows_between_parents():
    """The tracker sets w_child = w_parent * rel_w and then normalises
    GLOBALLY across every child of every parent.

    That is only safe if EVERY call returns the same total mass. If one
    parent's branches summed to 1.0 and another's to 0.5, the global step
    would quietly move weight from the second parent to the first --
    with no evidence favouring it, purely because its branch set captured
    less of its own distribution. A top-k selector keyed on "the cells
    holding >=90% of the mass" breaks this, since the count needed varies
    with the state.
    """
    ad = _adapter(max_branches=4)
    rng = np.random.default_rng(0)
    for _ in range(25):
        n_blue = int(rng.integers(1, BLUE_CAP + 1))
        ad.set_context(
            blue_pos=rng.uniform(5, 125, (n_blue, 2)),
            blue_vel=rng.uniform(-1, 1, (n_blue, 2)),
            obs_pos=rng.uniform(20, 110, (2, 2)),
            obs_vel=np.zeros((2, 2)),
            obs_r=rng.uniform(4, 12, 2))
        x = np.concatenate([rng.uniform(5, 125, 2), rng.uniform(-1, 1, 2)])
        total = sum(w for w, _, _ in ad(x, np.diag([4.0, 4.0, 1.0, 1.0])))
        assert abs(total - 1.0) < 1e-5, (
            f"branch weights sum to {total}, so this call would lose weight "
            f"to parents whose branches summed higher")


def test_branches_keep_all_of_the_probability_mass():
    """The defining property of basin splitting over top-k: NOTHING is
    discarded.  Truncating to the k most probable cells threw away ~80% of
    the distribution and made the branch covariance badly overconfident
    (measured NEES 6.6 against a target of 4.0)."""
    ad = _adapter(max_branches=3)
    ad.set_context(blue_pos=np.array([[45.0, 45.0]]), blue_vel=np.zeros((1, 2)))
    probs = ad.predict_probs(np.array([65.0, 65.0]), np.array([0.3, 0.2]))
    total = sum(w for w, _, _ in ad._basins(probs))
    assert np.isclose(total, 1.0, atol=1e-6), (
        f"basins hold {total:.4f} of the mass, not all of it")


def test_basin_mean_and_spread_match_the_distribution_they_summarise():
    """Each branch must be the true first and second moment of its own
    basin -- otherwise the mixture is not a faithful reduction."""
    ad = _adapter(max_branches=1, sigma_a_model=0.0)
    ad.set_context(blue_pos=np.array([[45.0, 45.0]]), blue_vel=np.zeros((1, 2)))
    probs = ad.predict_probs(np.array([65.0, 65.0]), np.array([0.3, 0.2]))
    basins = ad._basins(probs)
    grid_basins = [b for b in basins if np.linalg.norm(b[1]) > 1e-12]
    assert len(grid_basins) == 1, "max_branches=1 should give one grid basin"
    w, mu, cov = grid_basins[0]

    # With a single basin every grid cell belongs to it, so the moments are
    # just those of the whole grid part of the categorical.
    p = probs[:ZERO_CLASS] / probs[:ZERO_CLASS].sum()
    a = ad._cell_accel
    assert np.allclose(mu, p @ a, atol=1e-9)
    dev = a - mu
    want = (dev * p[:, None]).T @ dev + np.einsum("n,nij->ij", p, ad._cell_quant)
    assert np.allclose(cov, want, atol=1e-9)


def test_more_branches_resolve_finer_modes_without_losing_mass():
    """Raising the cap splits the SAME distribution more finely; it never
    changes how much of it is represented."""
    ctx = dict(blue_pos=np.array([[30.0, 30.0]]), blue_vel=np.array([[1.0, 1.0]]))
    one, four = _adapter(max_branches=1), _adapter(max_branches=4)
    one.set_context(**ctx)
    four.set_context(**ctx)
    probs = one.predict_probs(np.array([65.0, 65.0]), np.array([0.0, 0.0]))
    b1, b4 = one._basins(probs), four._basins(probs)
    assert len(b4) >= len(b1)
    assert np.isclose(sum(w for w, _, _ in b1), 1.0, atol=1e-6)
    assert np.isclose(sum(w for w, _, _ in b4), 1.0, atol=1e-6)


def test_a_bimodal_categorical_yields_two_separated_branches():
    """The whole reason the tracker is a Gaussian SUM: two opposed modes
    must survive as two branches whose means point in opposite directions,
    instead of being averaged onto a heading the evader will never fly.

    Driven by a synthetic categorical rather than the network, so the test
    asserts the SPLITTING logic and does not depend on what an untrained
    model happens to predict.
    """
    ad = _adapter(max_branches=4, sigma_a_model=0.0)
    probs = np.zeros(N_BINS)
    top = N_MAGNITUDE_BINS - 1                       # |a| ~ a_max row
    for h, mass in ((0, 0.5), (18, 0.5)):            # 180 degrees apart
        probs[h * N_MAGNITUDE_BINS + top] = mass
    basins = [b for b in ad._basins(probs) if b[0] > 1e-9]
    assert len(basins) == 2, f"expected 2 modes, got {len(basins)}"
    (w1, m1, _), (w2, m2, _) = basins
    assert np.isclose(w1, 0.5) and np.isclose(w2, 0.5)
    cos = (m1 @ m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
    assert cos < -0.9, f"modes collapsed toward each other (cos {cos:.2f})"


def test_prediction_matches_the_env_integration_exactly():
    """A branch must advance its state the way PursuitEnv ACTUALLY does.

    The env uses ``v' = clip(v + a*dt, -v_max, v_max)`` then
    ``p' = p + v'*dt`` -- so the position gain is dt^2, not the textbook
    dt^2/2, and the velocity is capped AXIS-WISE.  Getting either wrong is
    invisible in a single step and ruinous over a coast: the textbook gain
    predicts half a metre short per step, and an uncapped velocity reaches
    80 m/s over an 80-step coast, throwing the estimate a thousand metres
    outside a 130 m arena.  Both were measured before this test existed.
    """
    from isr.env.entities import RED_TARGET

    ad = _adapter(max_branches=8, dt=1.0)
    ad.set_context(blue_pos=np.array([[20.0, 20.0]]), blue_vel=np.zeros((1, 2)))
    for x in (np.array([65.0, 65.0, 1.0, 0.0]),      # already at v_max
              np.array([65.0, 65.0, -0.4, 0.9]),
              np.array([65.0, 65.0, 0.0, 0.0])):
        for _, xb, _ in ad(x, np.eye(4) * 1e-9):
            # Recover the acceleration this branch applied, then re-derive
            # the state through the env's own formula.
            a = (xb[2:] - x[2:]) / ad.dt
            v = np.clip(x[2:] + a * ad.dt, -RED_TARGET.v_max, RED_TARGET.v_max)
            assert np.allclose(xb[2:], v, atol=1e-9)
            assert np.allclose(xb[:2], x[:2] + v * ad.dt, atol=1e-9)


def test_velocity_never_exceeds_the_red_speed_cap():
    """The cap is what stops a coasting track accelerating without bound."""
    from isr.env.entities import RED_TARGET

    ad = _adapter(max_branches=4)
    ad.set_context(blue_pos=np.array([[10.0, 10.0]]), blue_vel=np.zeros((1, 2)))
    x = np.array([65.0, 65.0, 0.0, 0.0])
    for _ in range(200):                       # a long coast, no detections
        branches = ad(x, np.eye(4))
        x = max(branches, key=lambda b: b[0])[1]
    assert np.all(np.abs(x[2:]) <= RED_TARGET.v_max + 1e-9), (
        f"velocity ran away to {x[2:]} over a long coast")


def test_learned_branch_is_tighter_than_the_constant_velocity_default():
    """The reason to plug this in at all: the default Q assumes the
    acceleration is entirely unknown (sigma_a = a_max*sqrt(2)), whereas a
    branch has already committed to a cell, so its residual spread must be
    materially smaller."""
    ad = _adapter()
    ad.set_context(blue_pos=np.array([[30.0, 90.0]]), blue_vel=np.zeros((1, 2)))
    x, P = np.array([65.0, 65.0, 0.4, 0.4]), np.zeros((4, 4))
    default = MultiTargetTracker(dt=1.0, a_max=1.0)
    q_pos = default.Q[0, 0]
    for _, _, Pb in ad(x, P):
        assert Pb[0, 0] < q_pos, "branch no tighter than assuming nothing"


def test_branch_covariance_is_anisotropic_along_the_acceleration():
    """Within a cell the spread is NOT isotropic, and which axis is coarser
    is a fact about THIS grid, not a guess.

    With 36 heading bins and 5 magnitude bins: the magnitude bin is 0.2
    wide, while the heading arc at |a| = a_max is 2*pi/36 = 0.175.  So the
    RADIAL (magnitude) spread is the larger one at full acceleration --
    and since the tangential term scales with |a| while the radial does
    not, radial dominates everywhere on this grid.
    """
    ad = _adapter(sigma_a_model=0.0)          # isolate the quantisation term
    C = ad._branch_cov(np.array([1.0, 0.0]))
    along, across = C[0, 0], C[1, 1]
    assert not np.isclose(along, across), "cell spread came out isotropic"
    assert along > across, "magnitude bin is the coarser axis on a 36x5 grid"


def test_tangential_spread_scales_with_the_acceleration_magnitude():
    """A fixed heading ARC is a wider absolute spread the faster you push,
    so the across-track term must grow with |a| while the radial term --
    set by the magnitude bin width -- stays put."""
    ad = _adapter(sigma_a_model=0.0)
    small = ad._branch_cov(np.array([0.1, 0.0]))
    large = ad._branch_cov(np.array([0.9, 0.0]))
    assert large[1, 1] > small[1, 1] * 50, "tangential term did not scale"
    assert np.isclose(large[0, 0], small[0, 0]), "radial term should not scale"


def test_zero_class_is_its_own_branch_with_no_acceleration():
    """"Do not accelerate" is not a direction, so it cannot belong to any
    heading basin -- it has to come through as a separate branch."""
    ad = _adapter(max_branches=2, sigma_a_model=0.2)
    probs = np.zeros(N_BINS)
    probs[0 * N_MAGNITUDE_BINS + 4] = 0.7
    probs[ZERO_CLASS] = 0.3
    basins = ad._basins(probs)
    zero = [b for b in basins if np.linalg.norm(b[1]) < 1e-12]
    assert len(zero) == 1, "ZERO class did not survive as its own branch"
    w, mu, cov = zero[0]
    assert np.isclose(w, 0.3)
    assert np.allclose(mu, 0.0)
    assert np.allclose(cov, ad.sigma_a_model ** 2 * np.eye(2)), (
        "the ZERO class has no cell geometry, so only model error applies")


# --------------------------------------------------------------------- #
#  Context handling
# --------------------------------------------------------------------- #

def test_refuses_to_predict_without_a_context():
    """Falling back to constant velocity would make a WIRING bug look like
    a model-quality result -- the hardest kind to diagnose."""
    ad = _adapter()
    with pytest.raises(RuntimeError, match="set_context"):
        ad(np.array([65.0, 65.0, 0.0, 0.0]), np.eye(4))


def test_truncates_to_the_nearest_blues_not_arbitrary_ones():
    """The red policy reads only its NEAREST blue, so when a scenario has
    more blues than the trained capacity the nearest ones are exactly the
    informative subset."""
    ad = _adapter()
    far = np.array([[5.0, 5.0]] * BLUE_CAP)
    near = np.array([[64.0, 64.0]])
    ad.set_context(blue_pos=np.vstack([far, near]),
                  blue_vel=np.zeros((BLUE_CAP + 1, 2)))
    red = np.array([65.0, 65.0])
    packed = ad._pack(red, np.zeros(2))
    kept = packed["blue_rel_pos"][0, :int(packed["n_blue"][0])] * L + red
    assert any(np.allclose(k, near[0]) for k in kept), (
        "dropped the nearest blue, which is the one the policy reads")
    assert int(packed["n_blue"][0]) == BLUE_CAP


def test_empty_blue_context_is_rejected():
    ad = _adapter()
    with pytest.raises(ValueError, match="nothing to flee"):
        ad.set_context(blue_pos=np.zeros((0, 2)), blue_vel=np.zeros((0, 2)))


def test_context_without_obstacles_is_valid():
    ad = _adapter()
    ad.set_context(blue_pos=np.array([[50.0, 50.0]]), blue_vel=np.zeros((1, 2)))
    branches = ad(np.array([65.0, 65.0, 0.0, 0.0]), np.eye(4))
    assert branches


# --------------------------------------------------------------------- #
#  End to end
# --------------------------------------------------------------------- #

def test_tracker_runs_with_the_learned_motion_model():
    """Smoke: the tracker must accept these branches and keep a track alive
    across steps with a genuinely branching model (max_components > 1)."""
    ad = _adapter(max_branches=3)
    tr = MultiTargetTracker(dt=1.0, motion_model=ad, max_components=8)
    rng = np.random.default_rng(1)
    truth = np.array([60.0, 60.0])
    for _ in range(12):
        ad.set_context(blue_pos=np.array([[30.0, 30.0], [90.0, 40.0]]),
                      blue_vel=np.zeros((2, 2)))
        truth = truth + np.array([0.6, 0.4])
        # Position-only return: a zero LOS with sigma_radial 0 is the
        # tracker's own "no Doppler on this plot" convention.
        dets = [dict(z_pos=truth + rng.normal(0, 0.5, 2), sigma_pos=1.0,
                    los=np.zeros(2), sigma_radial=0.0, z_radial=0.0,
                    blue=0, truth_id=0)]
        tr.step(dets)
    assert tr.tracks, "no track survived a branching motion model"
    est = tr.tracks[0].x[:2]
    assert np.linalg.norm(est - truth) < 12.0, (
        f"track drifted to {est} against truth {truth}")
