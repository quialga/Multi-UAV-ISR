"""
isr/tracking/coverage.py — "would this target have been seen?", for the
tracker's miss accounting.

A miss is only evidence against a track's existence if the target WOULD
have been detected had it been there.  Where no sensor could see, a miss
says nothing.  This is the idea behind track-existence filters with a
detection probability (the IPDA family), in its simplest form: instead of
a continuous existence probability updated with P_D, a confirmed track's
miss counter only advances where P_D is high.

The motivating case is clutter.  A clutter plot that confirms as a phantom
sits where the plot appeared — inside some blue's sensor disk — so the
sensors keep looking at it and seeing nothing, and its misses count.  A
real target that has fled out of range is outside every disk, so its
misses do not, and it keeps coasting.

Blue positions and the sensor footprint are the platform's OWN state, not
privileged information.  Occlusion needs obstacle geometry: in evaluation
the env passes its exact test; in deployment it would come from the
obstacle tracker's estimates.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

Occluded = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sensor_coverage(
    blue_pos:      np.ndarray,
    sensor_radius: float,
    occluded:      Optional[Occluded] = None,
    k_sigma:       float = 2.0,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Build ``covered(pos, P_pos) -> bool`` for one scan.

    True when some blue would CERTAINLY have had the target in view: the
    target's uncertainty region, taken as ``k_sigma`` times the largest
    position standard deviation around the predicted position, lies wholly
    inside that blue's sensor disk, and the line of sight from that blue to
    the predicted position is clear.

    The margin is deliberately conservative.  A track so uncertain that it
    might be outside every disk does not have its miss counted — which also
    means a phantom that drifts toward a disk edge while its covariance
    grows can escape the rule.  That trade is made on purpose: counting a
    miss for a target that was never visible kills real tracks.

    Occlusion is tested to the predicted position only, not across the
    whole uncertainty region — a simplification.

    Parameters
    ----------
    blue_pos      (N, 2) positions of the observing blues this scan
    sensor_radius detection range, metres
    occluded      ``(from (2,), points (K, 2)) -> (K,) bool``, True where
                  the ray is blocked; None means no occlusion
    k_sigma       how many position standard deviations must fit inside
    """
    blues = np.asarray(blue_pos, dtype=np.float64).reshape(-1, 2)
    R = float(sensor_radius)

    def covered(pos: np.ndarray, P_pos: np.ndarray) -> bool:
        p = np.asarray(pos, dtype=np.float64).reshape(2)
        sd = float(np.sqrt(max(np.linalg.eigvalsh(
            np.asarray(P_pos, dtype=np.float64)).max(), 0.0)))
        dist = np.linalg.norm(blues - p, axis=1)
        inside = np.nonzero(dist + k_sigma * sd <= R)[0]
        if inside.size == 0:
            return False
        if occluded is None:
            return True
        for b in inside:
            if not bool(occluded(blues[b], p[None, :])[0]):
                return True
        return False

    return covered
