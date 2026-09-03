"""
isr/tracking/kalman.py — shared Kalman-filter primitive.

Extracted from ``isr.tracking.tracker.MultiTargetTracker._update`` so the
obstacle tracker (``isr.tracking.obstacle_tracker``) can reuse the exact
same, already-verified math instead of a second hand-copied version.  The
Joseph-form update is dimension-agnostic — it only ever needed ``np.eye(4)``
because the red tracker's state happens to be 4-D; written as
``np.eye(len(x))`` it is bit-identical for that case (verified by
``tests/test_gaussian_sum.py``'s non-regression suite) and correct for any
other state size, e.g. the obstacle tracker's 5-D ``[px, py, vx, vy, r]``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def joseph_update(
    x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray, z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """One Kalman update, Joseph form, for a state of any dimension.

    P = (I-KH) P (I-KH)^T + K R K^T rather than (I-KH)P: the short form is
    only valid for the optimal gain in exact arithmetic, and loses symmetry
    / positive-definiteness under many sequential updates (both trackers
    apply several per step, for hundreds of steps).  Joseph costs one
    extra product and is stable for any K.

    Also returns the innovation's NIS (Mahalanobis d^2) and log-likelihood
    — the latter is what a Gaussian-Sum track (see ``tracker.py::Track``)
    uses to Bayes-reweight its components; a single-hypothesis consumer
    (the obstacle tracker) can ignore it.
    """
    n = x.shape[0]
    y = np.atleast_1d(z) - H @ x
    S = H @ P @ H.T + R
    Si = np.linalg.inv(S)
    nis = float(y @ Si @ y)
    sign, logdet = np.linalg.slogdet(S)
    k = y.shape[0]
    loglik = -0.5 * (nis + (logdet if sign > 0 else 0.0)
                     + k * np.log(2.0 * np.pi))
    K = P @ H.T @ Si
    x_new = x + K @ y
    A = np.eye(n) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)          # kill drift in symmetry
    return x_new, P_new, nis, loglik
