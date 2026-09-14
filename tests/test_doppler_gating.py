"""
tests/test_doppler_gating.py — gating on position + Doppler jointly.

What Doppler gating is for: a return at the right PLACE with the wrong
RADIAL SPEED is not the track's return — in particular, on a scan where the
real return was MISSED, a clutter plot inside the position gate should not
hijack the track.

What it CANNOT do here, found by these tests: with the constant-velocity
model's calibrated process noise the predicted velocity is so uncertain
(sd ~1.41 m/s) that the Doppler term can never reject a clutter plot on its
own.  It helps only when the motion model predicts velocity confidently.
Both facts are pinned below.

Everything is off by default; the existing suite (including the bit-exact
Gaussian-Sum non-regression) covers that path.

Run:
    pytest tests/test_doppler_gating.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.tracking import MultiTargetTracker
from isr.tracking.tracker import CHI2_99
from tests.test_tracker import det

# LOS from BLUE_A to the target is along +x, so the target's radial speed
# along it is large (0.8) and a sign-flipped Doppler is a big innovation.
BLUE_A = np.array([10.0, 50.0])
BLUE_B = np.array([50.0, 10.0])
V = np.array([0.8, 0.0])


def _tracker(**kw):
    base = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0, confirm_hits=2,
                max_misses=5)
    base.update(kw)
    return MultiTargetTracker(**base)


def _establish(trk, steps=10):
    """A single well-observed target: position AND velocity settled."""
    p = np.array([40.0, 50.0])
    for _ in range(steps):
        p = p + V
        trk.step([det(BLUE_A, p, V, blue=0, sigma_radial=0.05, truth_id=0),
                  det(BLUE_B, p, V, blue=1, sigma_radial=0.05, truth_id=0)])
    assert len(trk.tracks) == 1 and trk.tracks[0].confirmed
    return p


# --------------------------------------------------------------------- #
#  Measurement construction
# --------------------------------------------------------------------- #

def test_joint_measurement_stacks_the_radial_row():
    trk = _tracker(doppler_gating=True)
    d = det(BLUE_A, [40.0, 50.0], V, sigma_pos=1.5, sigma_radial=0.1)
    H, R, z = trk._gate_measurement(d)
    assert H.shape == (3, 4)
    assert np.allclose(H[:2], [[1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.allclose(H[2], [0, 0, *d["los"]])
    assert np.allclose(np.diag(R), [1.5 ** 2, 1.5 ** 2, 0.1 ** 2])
    assert np.allclose(z, [*d["z_pos"], d["z_radial"]])
    assert trk._gate_threshold(d) == pytest.approx(CHI2_99[3])


def test_off_by_default_uses_the_position_gate():
    trk = _tracker()
    d = det(BLUE_A, [40.0, 50.0], V)
    H, _, _ = trk._gate_measurement(d)
    assert H.shape == (2, 4)
    assert trk._gate_threshold(d) == pytest.approx(trk.gate_chi2)


def test_return_without_doppler_falls_back_to_the_position_gate():
    """sigma_radial 0 means no usable Doppler — the same condition the
    update applies — so even with gating on, the 2-D gate is used."""
    trk = _tracker(doppler_gating=True)
    d = det(BLUE_A, [40.0, 50.0], V, sigma_radial=0.0)
    H, _, _ = trk._gate_measurement(d)
    assert H.shape == (2, 4)
    assert trk._gate_threshold(d) == pytest.approx(trk.gate_chi2)


# --------------------------------------------------------------------- #
#  Behaviour
# --------------------------------------------------------------------- #

def test_consistent_return_is_still_accepted():
    """Doppler gating must not start rejecting the target's own returns."""
    trk = _tracker(doppler_gating=True)
    p = _establish(trk)
    tid = trk.tracks[0].id
    for _ in range(5):
        p = p + V
        trk.step([det(BLUE_A, p, V, blue=0, sigma_radial=0.05, truth_id=0),
                  det(BLUE_B, p, V, blue=1, sigma_radial=0.05, truth_id=0)])
        assert len(trk.tracks) == 1, "a consistent return spawned a new track"
        assert trk.tracks[0].misses == 0
    assert trk.tracks[0].id == tid


def _tight_motion_model(sigma_a=0.05):
    """Stand-in for a motion model that predicts velocity WELL: constant
    velocity with a small process noise.  Doppler gating can only
    discriminate when the predicted velocity is this confident."""
    from isr.tracking.tracker import _dwna_Q
    F = np.eye(4)
    F[0, 2] = F[1, 3] = 1.0
    Q = _dwna_Q(1.0, sigma_a)
    return lambda x, P: [(1.0, F @ x, F @ P @ F.T + Q)]


def _hijack_attempt(trk):
    """Establish a track, then a scan where the real return is MISSED and a
    clutter plot sits exactly at the predicted position with a sign-flipped
    Doppler (radial -0.8 vs +0.8).  Returns the original track after it."""
    p = _establish(trk) + V
    tid = trk.tracks[0].id
    trk.step([det(BLUE_A, p, -V, blue=0, sigma_radial=0.05, truth_id=-1)])
    return next(t for t in trk.tracks if t.id == tid)


def test_constant_velocity_process_noise_leaves_doppler_nothing_to_reject():
    """A LIMITATION, pinned so nobody expects otherwise.  The calibrated CV
    process noise is sigma_a = a_max * sqrt(2), so after PREDICT the
    velocity variance is Q_vv = 2.0 per axis (sd ~1.41 m/s).  Clutter
    Doppler lies in [-1, 1] and a red's predicted radial speed within
    ~1.41, so the Doppler term adds at most ~2.9 to d^2 — never enough to
    cross the 3-dof threshold (11.3).  This is physics, not tuning: the red
    accelerates up to 1 m/s^2 every step, so its radial speed genuinely can
    change that much between scans.

    So with the constant-velocity model, Doppler gating cannot stop a
    clutter plot from hijacking a track."""
    original = _hijack_attempt(_tracker(doppler_gating=True))
    assert original.misses == 0, (
        "CV process noise was expected to make the Doppler term too weak "
        "to reject the plot -- if this now fails, re-derive the argument")


def test_doppler_gating_prevents_a_hijack_when_velocity_is_well_predicted():
    """The mechanism does work once the prerequisite holds.  With a motion
    model that predicts velocity confidently, the same sign-flipped plot is
    rejected: the track records a miss and the plot births its own
    tentative track.  Without gating, the same tight model is hijacked."""
    gated = _hijack_attempt(_tracker(doppler_gating=True,
                                     motion_model=_tight_motion_model()))
    assert gated.misses == 1, "well-predicted track was hijacked despite gating"

    ungated = _hijack_attempt(_tracker(doppler_gating=False,
                                       motion_model=_tight_motion_model()))
    assert ungated.misses == 0, (
        "without gating the plot is expected to be assigned -- otherwise "
        "this test no longer demonstrates what gating adds")


def test_gate_is_scored_against_the_predicted_covariance():
    """Gating happens AFTER predict inside step().  Scoring a return against
    the post-UPDATE covariance instead (which is much tighter) would make
    Doppler gating look far stronger than it is — exactly the mistake an
    earlier version of this test made."""
    trk = _tracker(doppler_gating=True)
    p = _establish(trk) + V
    wrong = det(BLUE_A, p, -V, blue=0, sigma_radial=0.05, truth_id=-1)
    tr = trk.tracks[0]
    d2_updated, _ = trk._gate_pos(tr, wrong)       # NOT what step() does
    trk._predict_track(tr)
    d2_predicted, _ = trk._gate_pos(tr, wrong)     # what step() does
    assert d2_updated > trk._gate_threshold(wrong)
    assert d2_predicted <= trk._gate_threshold(wrong)
