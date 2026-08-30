"""
tests/test_mot_metrics.py — the MOT harness must be right before it is used
to judge anything.  Small hand-built sequences with known answers.
"""
from __future__ import annotations

import numpy as np

from isr.tracking.metrics import ConsistencyAccumulator, MOTAccumulator


def test_perfect_tracking_scores_perfectly():
    acc = MOTAccumulator(match_dist=5.0)
    for t in range(10):
        p = np.array([[float(t), 0.0], [0.0, float(t)]])
        acc.update([0, 1], p, ["a", "b"], p)
    s = acc.summary()
    assert s["MOTA"] == 1.0 and s["MOTP"] == 0.0
    assert s["IDSW"] == 0.0 and s["IDF1"] == 1.0
    assert s["recall"] == 1.0 and s["MT"] == 1.0


def test_missed_target_counts_as_false_negative():
    acc = MOTAccumulator(match_dist=5.0)
    for _ in range(10):
        acc.update([0], np.array([[0.0, 0.0]]), [], np.zeros((0, 2)))
    s = acc.summary()
    assert s["FN"] == 10 and s["FP"] == 0 and s["recall"] == 0.0
    assert s["MOTA"] == 0.0            # 1 - 10/10
    assert s["ML"] == 1.0


def test_spurious_hypothesis_counts_as_false_positive():
    acc = MOTAccumulator(match_dist=5.0)
    for _ in range(10):
        acc.update([], np.zeros((0, 2)), ["ghost"], np.array([[9.0, 9.0]]))
    assert acc.summary()["FP"] == 10


def test_identity_switch_is_detected():
    """Same GT object, hypothesis id changes halfway -> exactly one IDSW."""
    acc = MOTAccumulator(match_dist=5.0)
    for t in range(5):
        acc.update([0], np.array([[0.0, 0.0]]), ["a"], np.array([[0.0, 0.0]]))
    for t in range(5):
        acc.update([0], np.array([[0.0, 0.0]]), ["b"], np.array([[0.0, 0.0]]))
    s = acc.summary()
    assert s["IDSW"] == 1.0
    assert s["IDF1"] < 1.0, "IDF1 must penalise the identity change"


def test_beyond_the_gate_is_not_a_match():
    acc = MOTAccumulator(match_dist=5.0)
    acc.update([0], np.array([[0.0, 0.0]]), ["a"], np.array([[50.0, 0.0]]))
    s = acc.summary()
    assert s["FN"] == 1 and s["FP"] == 1, "a far hypothesis is FP, not a match"


def test_fragmentation_counts_reacquisitions():
    acc = MOTAccumulator(match_dist=5.0)
    z = np.array([[0.0, 0.0]])
    for _ in range(3):
        acc.update([0], z, ["a"], z)
    for _ in range(3):
        acc.update([0], z, [], np.zeros((0, 2)))     # lost
    for _ in range(3):
        acc.update([0], z, ["a"], z)                 # reacquired
    assert acc.summary()["Frag"] == 1.0


def test_nees_recovers_the_state_dimension():
    """With a correct covariance, NEES should average the state dim (4)."""
    rng = np.random.default_rng(0)
    P = np.diag([2.0, 3.0, 0.5, 0.7])
    L = np.linalg.cholesky(P)
    cons = ConsistencyAccumulator()
    x_hat = np.zeros(4)
    for _ in range(6000):
        cons.add_nees(x_hat, P, L @ rng.normal(size=4))
    s = cons.summary(state_dim=4)
    assert abs(s["NEES"] - 4.0) < 0.25
