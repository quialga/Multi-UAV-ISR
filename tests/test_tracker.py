"""
tests/test_tracker.py — multi-target tracker: hand-built scenarios.

These are deterministic, millisecond-scale, and the correct answer is known
BY CONSTRUCTION.  Aggregate metrics (MOTA/IDF1) tell you *that* something
broke; these tell you *what*.  Each scenario targets one failure mode of the
association loop.

Run:
    pytest tests/test_tracker.py -v
"""
from __future__ import annotations

import itertools

import numpy as np

from isr.tracking import MultiTargetTracker, linear_sum_assignment, solve_gated


# --------------------------------------------------------------------- #
#  Detection factory — mirrors PursuitEnv.raw_detections()
# --------------------------------------------------------------------- #

def det(blue_pos, target_pos, target_vel, blue=0,
        sigma_pos=1.0, sigma_radial=0.05, rng=None, truth_id=0):
    blue_pos = np.asarray(blue_pos, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    target_vel = np.asarray(target_vel, dtype=np.float32)
    d = target_pos - blue_pos
    n = float(np.linalg.norm(d))
    los = (d / n) if n > 1e-6 else np.zeros(2, dtype=np.float32)
    z = target_pos.copy()
    zr = float(target_vel @ los)
    if rng is not None:
        z = z + rng.normal(0.0, sigma_pos, 2).astype(np.float32)
        zr += float(rng.normal(0.0, sigma_radial))
    return {"blue": blue, "z_pos": z.astype(np.float32),
            "z_range": float(np.linalg.norm(z - blue_pos)),
            "z_radial": zr, "los": los.astype(np.float32),
            "sigma_pos": float(sigma_pos), "sigma_radial": float(sigma_radial),
            "truth_id": truth_id}


def _tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=2,
                max_misses=5)
    base.update(kw)
    return MultiTargetTracker(**base)


# --------------------------------------------------------------------- #
#  Assignment primitive
# --------------------------------------------------------------------- #

def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(120):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        C = rng.random((n, m)) * 10.0
        r, c = linear_sum_assignment(C)
        mine = C[r, c].sum()
        k = min(n, m)
        best = min(
            sum(C[i, j] for i, j in zip(rows, cols))
            for rows in itertools.permutations(range(n), k)
            for cols in itertools.permutations(range(m), k)
        )
        assert abs(mine - best) < 1e-9


def test_gated_solver_drops_forbidden_pairs():
    cost = np.array([[1.0, 50.0], [50.0, 1.0]])
    gate = np.array([[True, False], [False, False]])
    pairs = solve_gated(cost, gate)
    assert pairs == [(0, 0)], "only the gated-in pair may survive"


# --------------------------------------------------------------------- #
#  Scenario 1 — a single target moving in a straight line
# --------------------------------------------------------------------- #

def test_single_target_is_tracked_and_velocity_converges():
    trk = _tracker()
    rng = np.random.default_rng(0)
    p = np.array([50.0, 50.0]); v = np.array([0.8, -0.4])
    blue = np.array([20.0, 60.0])
    errs = []
    for t in range(25):
        p = p + v
        trk.step([det(blue, p, v, rng=rng)])
        assert len(trk.tracks) == 1, "a single target must not spawn extras"
        errs.append(np.linalg.norm(trk.tracks[0].pos - p))
    assert trk.tracks[0].confirmed
    # Late error should beat the raw 1 m measurement noise (filtering works).
    assert np.mean(errs[-10:]) < 1.0
    assert np.linalg.norm(trk.tracks[0].vel - v) < 0.25


# --------------------------------------------------------------------- #
#  Scenario 2 — two targets crossing: the classic ID-switch trap
# --------------------------------------------------------------------- #

def test_two_crossing_targets_keep_their_identities():
    trk = _tracker()
    rng = np.random.default_rng(1)
    # TWO blues at ~90 deg: with a single observer nearly perpendicular to
    # the motion the velocity is unobservable (Doppler sees ~0 of it), which
    # would test physics rather than association.  See
    # test_single_blue_crossing_geometry_is_unobservable.
    b0 = np.array([60.0, 5.0])
    b1 = np.array([5.0, 60.0])
    pA = np.array([40.0, 60.0]); vA = np.array([1.0, 0.0])
    pB = np.array([80.0, 60.0]); vB = np.array([-1.0, 0.0])
    id_a = id_b = None
    for t in range(40):
        pA = pA + vA
        pB = pB + vB
        trk.step([det(b0, pA, vA, blue=0, rng=rng, truth_id=0),
                  det(b0, pB, vB, blue=0, rng=rng, truth_id=1),
                  det(b1, pA, vA, blue=1, rng=rng, truth_id=0),
                  det(b1, pB, vB, blue=1, rng=rng, truth_id=1)])
        if len(trk.tracks) == 2 and id_a is None and t > 3:
            # label by which track is nearer each truth, once settled
            d0 = [np.linalg.norm(tr.pos - pA) for tr in trk.tracks]
            id_a = trk.tracks[int(np.argmin(d0))].id
            id_b = [tr.id for tr in trk.tracks if tr.id != id_a][0]
    ids = {tr.id for tr in trk.tracks}
    assert id_a is not None
    assert len(trk.tracks) == 2, f"expected 2 tracks, got {trk.tracks}"
    assert ids == {id_a, id_b}, (
        f"identities were not preserved through the crossing: {ids} "
        f"vs {{{id_a}, {id_b}}}")


# --------------------------------------------------------------------- #
#  Scenario 3 — occlusion: the track must COAST and survive
# --------------------------------------------------------------------- #

def test_track_coasts_through_a_detection_gap_and_reacquires():
    trk = _tracker(max_misses=5)
    rng = np.random.default_rng(2)
    p = np.array([50.0, 50.0]); v = np.array([1.0, 0.0])
    blue = np.array([50.0, 10.0])
    for _ in range(6):                      # establish
        p = p + v
        trk.step([det(blue, p, v, rng=rng)])
    tid = trk.tracks[0].id
    P_before = np.trace(trk.tracks[0].P[:2, :2])

    for _ in range(4):                      # occluded: NO detections
        p = p + v
        trk.step([])
        assert len(trk.tracks) == 1, "track died during a short gap"
    assert trk.tracks[0].id == tid, "identity lost while coasting"
    # Coasting must INFLATE the uncertainty, not pretend to still know.
    assert np.trace(trk.tracks[0].P[:2, :2]) > P_before
    # And it should still be roughly right, because it kept the velocity.
    assert np.linalg.norm(trk.tracks[0].pos - p) < 8.0

    for _ in range(5):                      # reacquire
        p = p + v
        trk.step([det(blue, p, v, rng=rng)])
    assert len(trk.tracks) == 1
    assert trk.tracks[0].id == tid, "reacquisition created a NEW identity"


def test_track_dies_after_max_misses():
    trk = _tracker(max_misses=3)
    rng = np.random.default_rng(3)
    p = np.array([50.0, 50.0]); v = np.array([1.0, 0.0])
    b0 = np.array([50.0, 10.0]); b1 = np.array([10.0, 50.0])
    for _ in range(5):
        p = p + v
        trk.step([det(b0, p, v, blue=0, rng=rng),
                  det(b1, p, v, blue=1, rng=rng)])
    assert len(trk.tracks) == 1
    for _ in range(4):
        trk.step([])
    assert len(trk.tracks) == 0, "track outlived max_misses"


# --------------------------------------------------------------------- #
#  Scenario 4 — the duplicate-birth trap
# --------------------------------------------------------------------- #

def test_two_blues_seeing_one_new_target_create_ONE_track():
    """If births were processed inside the per-blue loop, each observing
    blue would spawn its own track for the same physical target."""
    trk = _tracker()
    p = np.array([65.0, 65.0]); v = np.array([0.5, 0.5])
    trk.step([det(np.array([45.0, 65.0]), p, v, blue=0),
              det(np.array([65.0, 45.0]), p, v, blue=1)])
    assert len(trk.tracks) == 1, (
        f"one new target produced {len(trk.tracks)} tracks (duplicate birth)")


def test_birth_from_two_non_collinear_returns_recovers_velocity():
    """A cluster with two non-collinear lines of sight is born already
    knowing its velocity; one return falls back to the prior."""
    p = np.array([65.0, 65.0]); v = np.array([0.7, -0.4])
    two = _tracker()
    two.step([det(np.array([45.0, 65.0]), p, v, blue=0, sigma_radial=0.02),
              det(np.array([65.0, 45.0]), p, v, blue=1, sigma_radial=0.02)])
    assert np.linalg.norm(two.tracks[0].vel - v) < 0.15

    one = _tracker()
    one.step([det(np.array([45.0, 65.0]), p, v, blue=0, sigma_radial=0.02)])
    # Only the radial component is observable from a single line of sight.
    u = (p - np.array([45.0, 65.0])); u = u / np.linalg.norm(u)
    assert np.linalg.norm(one.tracks[0].vel - float(v @ u) * u) < 0.15


def test_two_separated_new_targets_create_two_tracks():
    trk = _tracker()
    trk.step([det(np.array([10.0, 10.0]), np.array([50.0, 50.0]),
                  np.array([0.0, 0.0]), blue=0, truth_id=0),
              det(np.array([10.0, 10.0]), np.array([95.0, 50.0]),
                  np.array([0.0, 0.0]), blue=0, truth_id=1)])
    assert len(trk.tracks) == 2


# --------------------------------------------------------------------- #
#  Lifecycle + numerics
# --------------------------------------------------------------------- #

def test_m_of_n_confirmation():
    trk = _tracker(confirm_hits=3, confirm_window=4)
    rng = np.random.default_rng(4)
    p = np.array([50.0, 50.0]); v = np.array([0.5, 0.0])
    blue = np.array([50.0, 20.0])
    p = p + v
    trk.step([det(blue, p, v, rng=rng)])
    assert not trk.tracks[0].confirmed, "confirmed on a single hit"
    for _ in range(2):
        p = p + v
        trk.step([det(blue, p, v, rng=rng)])
    assert trk.tracks[0].confirmed
    assert trk.confirmed_tracks() == trk.tracks


def test_covariance_stays_symmetric_positive_definite():
    """Joseph form: many sequential updates must not destroy P."""
    trk = _tracker()
    rng = np.random.default_rng(5)
    p = np.array([50.0, 50.0]); v = np.array([0.6, 0.3])
    blues = [np.array([10.0, 10.0]), np.array([90.0, 10.0]),
             np.array([50.0, 95.0])]
    for _ in range(80):
        p = p + v
        trk.step([det(b, p, v, blue=i, rng=rng) for i, b in enumerate(blues)])
        P = trk.tracks[0].P
        assert np.allclose(P, P.T, atol=1e-9), "P lost symmetry"
        assert np.all(np.linalg.eigvalsh(P) > 0), "P lost positive-definiteness"


def test_process_noise_has_independent_axes():
    """Q must treat the two axes as INDEPENDENT.

    Each per-axis [p, v] block is rank 1 by construction — one scalar
    acceleration drives both, so position and velocity noise are perfectly
    correlated WITHIN an axis.  That is correct DWNA.  What must NOT happen
    is coupling ACROSS axes: writing Q as G G^T with a single scalar
    G = [dt^2/2, dt^2/2, dt, dt] gives a rank-1 4x4, asserting a_x and a_y
    are the same random variable.
    """
    trk = _tracker()
    assert np.linalg.matrix_rank(trk.Q) == 2      # one per axis, not 1, not 4
    for i, j in ((0, 1), (0, 3), (1, 2), (2, 3)):  # cross-axis terms
        assert np.isclose(trk.Q[i, j], 0.0), f"Q couples axes at {(i, j)}"
    # Within an axis the correlation is exactly 1 (single driving noise).
    assert np.isclose(trk.Q[0, 2] ** 2, trk.Q[0, 0] * trk.Q[2, 2])


def test_sigma_a_defaults_to_amax_over_sqrt2():
    """Constant-magnitude, varying-direction acceleration => Var per axis is
    a_max^2/2 (measured 0.704 vs 0.707), not a_max^2/3."""
    trk = _tracker(a_max=1.0)
    assert abs(trk.sigma_a - 1.0 / np.sqrt(2.0)) < 1e-9


def test_no_detections_ever_is_a_no_op():
    trk = _tracker()
    for _ in range(10):
        trk.step([])
    assert trk.tracks == []


# --------------------------------------------------------------------- #
#  Documented limitation — not a bug, physics
# --------------------------------------------------------------------- #

def test_single_blue_crossing_geometry_is_unobservable():
    """A lone observer nearly perpendicular to the motion cannot estimate
    velocity, and the track fragments.  This is EXPECTED, and it is why the
    scenarios above use two blues.

    Doppler measures only the LOS component: with the blue at (50,10) and a
    target near y=50 moving in +x, the line of sight is ~(0.025, 1) so the
    true velocity (1,0) projects to 0.025 — essentially nothing.  The only
    other velocity cue is differencing noisy positions, whose error is
    sqrt(2)*sigma_pos/dt = 1.41 m/step, LARGER than the target's own speed.
    So the estimate can take the wrong sign, the prediction walks away, the
    detection falls outside the gate, and a fresh track is born.

    The fix is geometric, not algorithmic: a second, non-collinear observer.
    That is the same result as the velocity-fusion work — and it is exactly
    the emergent incentive for the team to spread out.
    """
    trk = _tracker()
    rng = np.random.default_rng(3)
    p = np.array([50.0, 50.0]); v = np.array([1.0, 0.0])
    blue = np.array([50.0, 10.0])          # LOS almost perpendicular to v
    for _ in range(6):
        p = p + v
        trk.step([det(blue, p, v, blue=0, rng=rng)])
    # Velocity along the unobservable (tangential) axis is not recovered.
    assert abs(trk.tracks[0].vel[0] - v[0]) > 0.5

    # Adding a second, non-collinear observer fixes it.
    trk2 = _tracker()
    rng2 = np.random.default_rng(3)
    p2 = np.array([50.0, 50.0])
    b1 = np.array([10.0, 50.0])
    for _ in range(6):
        p2 = p2 + v
        trk2.step([det(blue, p2, v, blue=0, rng=rng2),
                   det(b1, p2, v, blue=1, rng=rng2)])
    assert abs(trk2.tracks[0].vel[0] - v[0]) < 0.35, (
        "two non-collinear observers should recover the tangential velocity")


def test_Q_matches_the_closed_form_DWNA():
    """Pin Q against the literature form, so it cannot drift:

        Q = sigma_a^2 [[dt^4/4,  0,       dt^3/2,  0     ],
                       [0,       dt^4/4,  0,       dt^3/2],
                       [dt^3/2,  0,       dt^2,    0     ],
                       [0,       dt^3/2,  0,       dt^2  ]]

    Two independent scalar accelerations, one per axis — NOT one shared.
    """
    from isr.tracking.tracker import _dwna_Q
    for dt in (0.5, 1.0, 2.0):
        for sa in (0.3, 1.0 / np.sqrt(2.0), 1.0):
            expected = sa ** 2 * np.array([
                [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                [dt ** 3 / 2, 0, dt ** 2, 0],
                [0, dt ** 3 / 2, 0, dt ** 2],
            ])
            assert np.allclose(_dwna_Q(dt, sa), expected), (dt, sa)
