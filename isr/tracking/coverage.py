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
privileged information.  Occlusion needs obstacle geometry: the env's
evaluation helper passes its exact test on the TRUE obstacles; the actor
path (``PursuitEnv`` with ``actor_obs="tracker"``) builds the same test on
the obstacle tracker's ESTIMATES with ``disk_occluder``.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

Occluded = Callable[[np.ndarray, np.ndarray], np.ndarray]


def segment_disk_occluded(
    origin:  np.ndarray,   # (2,)
    points:  np.ndarray,   # (K, 2)
    centres: np.ndarray,   # (n, 2)
    radii:   np.ndarray,   # (n,)
    margin:  float,
) -> np.ndarray:
    """Exact analytic segment-disk occlusion: True for each ray
    origin -> point that enters any disk more than ``margin`` metres before
    reaching the point.  The geometry is documented on
    ``PursuitEnv._rays_occluded_by_obstacles``, which calls this.  Inputs
    are used in their own dtype, so the env's float32 result is unchanged
    by the extraction."""
    K = points.shape[0]
    if centres is None or len(centres) == 0:
        return np.zeros(K, dtype=bool)

    d = points - origin[None, :]                        # (K, 2)
    seg_len = np.linalg.norm(d, axis=-1)                # (K,)
    seg_len = np.maximum(seg_len, 1e-6)

    oc = centres - origin[None, :]                      # (n, 2)
    oc_len2 = np.sum(oc * oc, axis=-1)                  # (n,)

    dot = d @ oc.T                                      # (K, n)
    t_hat = dot / (seg_len ** 2)[:, None]
    proj_len2 = (t_hat * seg_len[:, None]) ** 2
    perp2 = oc_len2[None, :] - proj_len2

    disc = radii[None, :] ** 2 - perp2
    intersects = disc > 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    half_chord_t = sqrt_disc / seg_len[:, None]
    t1 = t_hat - half_chord_t
    t2 = t_hat + half_chord_t

    t_cut = 1.0 - (margin / seg_len)
    blocked = intersects & (t2 > 0.0) & (t1 < t_cut[:, None])
    return np.any(blocked, axis=1)


def disk_occluder(centres: np.ndarray, radii: np.ndarray,
                  margin: float) -> Occluded:
    """An ``occluded(from, points)`` test against a given set of disks —
    e.g. the obstacle tracker's estimated obstacles."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(radii, dtype=np.float64).reshape(-1)

    def occluded(origin: np.ndarray, points: np.ndarray) -> np.ndarray:
        return segment_disk_occluded(
            np.asarray(origin, dtype=np.float64).reshape(2),
            np.asarray(points, dtype=np.float64).reshape(-1, 2),
            c, r, float(margin))

    return occluded


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
