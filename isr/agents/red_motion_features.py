"""
isr/agents/red_motion_features.py — featuriser and action discretiser for
the learned red-motion model (docs Sec. 8).

Deliberately ONE module used by BOTH paths — training (from collector
shards) and inference (from tracker hypotheses).  Duplicating this logic
is the classic train/serve skew bug, and it would be especially easy to
hit here because the collector stores some fields normalised and some raw
(``red_pos``/``red_vel`` are raw by design, everything else is already
divided by ``arena_size`` or ``V_NORM``).

Edge features follow the env's own 7-D convention exactly
(``_build_obs``/``GNNEncoder``):

    rel_pos(2) + rel_vel(2) + range(1) + bearing_cos_sin(2)

with ``rel_pos`` pointing from the RECEIVER to the sender, ``rel_vel`` the
sender's velocity relative to the receiver's, and the bearing measured
between the receiver's OWN velocity and the direction to the sender (so it
answers "is this thing ahead of me or behind me").

Action discretisation (docs Sec. 8.4): a joint categorical over
(heading, magnitude), PLUS one dedicated ZERO class.  The zero class is
not decoration: 1.78% of collected samples have ``|a|`` exactly 0, where
the heading is genuinely undefined (``arctan2(0, 0)`` is meaningless), and
"do not accelerate" is a real discrete outcome the tracker should be able
to receive as a branch.  Measured bin occupancy on the collected data
justifies uniform bins on both axes: headings land at 2.63-2.92% per bin
against 2.78% uniform, and magnitudes spread 8/17/23/22/15% across five
equal bins.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from isr.env.entities import BLUE_UAV

V_NORM = BLUE_UAV.v_max      # shared velocity normaliser, as the collector used

N_HEADING_BINS = 36
N_MAGNITUDE_BINS = 5
N_BINS = N_HEADING_BINS * N_MAGNITUDE_BINS + 1     # + the ZERO class
ZERO_CLASS = N_BINS - 1
ZERO_EPS = 1e-6              # |a| below this is the ZERO class


# --------------------------------------------------------------------- #
#  Edge features — the one primitive both paths share
# --------------------------------------------------------------------- #

def edge_features(rel_pos: np.ndarray, rel_vel: np.ndarray,
                 receiver_vel: np.ndarray) -> np.ndarray:
    """The env's 7-D edge convention, vectorised over leading axes.

    ``rel_pos``  (..., 2) receiver -> sender, already arena-normalised
    ``rel_vel``  (..., 2) sender velocity minus receiver velocity, V_NORM-normalised
    ``receiver_vel`` (..., 2) the receiver's own velocity, V_NORM-normalised
    """
    rng = np.linalg.norm(rel_pos, axis=-1, keepdims=True)
    # Bearing between the receiver's heading and the direction to the
    # sender.  Zeros when either vector is degenerate -- a well-defined
    # "bearing is meaningless here" signal, matching _bearing_features.
    u_rel = rel_pos / np.maximum(rng, 1e-9)
    v_norm = np.linalg.norm(receiver_vel, axis=-1, keepdims=True)
    u_vel = receiver_vel / np.maximum(v_norm, 1e-9)
    cos = np.sum(u_rel * u_vel, axis=-1, keepdims=True)
    # sin from the 2-D cross product, so the bearing is SIGNED (left/right
    # of my heading), not just an unsigned angle.
    sin = (u_vel[..., 0:1] * u_rel[..., 1:2]
          - u_vel[..., 1:2] * u_rel[..., 0:1])
    degenerate = (rng < 1e-9) | (v_norm < 1e-9)
    cos = np.where(degenerate, 0.0, cos)
    sin = np.where(degenerate, 0.0, sin)
    return np.concatenate([rel_pos, rel_vel, rng, cos, sin], axis=-1)


# --------------------------------------------------------------------- #
#  Action discretisation
# --------------------------------------------------------------------- #

def accel_to_bin(accel: np.ndarray) -> np.ndarray:
    """Map raw accelerations (..., 2) to a flat class index (...,).

    Grid index = heading_bin * N_MAGNITUDE_BINS + magnitude_bin, with the
    ZERO class reserved for |a| ~ 0 (heading undefined).
    """
    accel = np.asarray(accel, dtype=np.float64)
    mag = np.linalg.norm(accel, axis=-1)
    ang = np.arctan2(accel[..., 1], accel[..., 0])          # [-pi, pi)
    h = np.floor((ang + np.pi) / (2 * np.pi) * N_HEADING_BINS).astype(np.int64)
    h = np.clip(h, 0, N_HEADING_BINS - 1)
    m = np.floor(np.clip(mag, 0.0, 1.0) * N_MAGNITUDE_BINS).astype(np.int64)
    m = np.clip(m, 0, N_MAGNITUDE_BINS - 1)                 # |a| == 1 -> top bin
    out = h * N_MAGNITUDE_BINS + m
    return np.where(mag < ZERO_EPS, ZERO_CLASS, out)


def bin_to_accel(idx: np.ndarray) -> np.ndarray:
    """Inverse: the representative acceleration at a class's CENTRE.

    Used when turning a predicted class into a Gaussian-Sum branch
    (docs Sec. 8.5).  The ZERO class maps to exactly [0, 0].
    """
    idx = np.asarray(idx, dtype=np.int64)
    h = idx // N_MAGNITUDE_BINS
    m = idx % N_MAGNITUDE_BINS
    ang = (h + 0.5) / N_HEADING_BINS * 2 * np.pi - np.pi
    mag = (m + 0.5) / N_MAGNITUDE_BINS
    out = np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=-1)
    return np.where((idx == ZERO_CLASS)[..., None], 0.0, out)


def soft_targets(idx: np.ndarray, heading_eps: float = 0.10,
                magnitude_eps: float = 0.05) -> np.ndarray:
    """Soft label distribution (..., N_BINS) with neighbour smoothing.

    The heading axis is CIRCULAR: bin 35 and bin 0 are 10 degrees apart,
    not unrelated classes, so a heading landing near a boundary should not
    be an all-or-nothing assignment.  Magnitude is merely ORDINAL (no
    wraparound), so it gets lighter, non-circular smoothing.  The ZERO
    class is discrete and unambiguous -- no smoothing at all.
    """
    idx = np.asarray(idx, dtype=np.int64)
    flat = idx.reshape(-1)
    out = np.zeros((flat.size, N_BINS), dtype=np.float32)

    is_zero = flat == ZERO_CLASS
    out[is_zero, ZERO_CLASS] = 1.0

    grid = ~is_zero
    if np.any(grid):
        gi = flat[grid]
        h = gi // N_MAGNITUDE_BINS
        m = gi % N_MAGNITUDE_BINS
        rows = np.nonzero(grid)[0]
        core = 1.0 - 2.0 * heading_eps - 2.0 * magnitude_eps
        out[rows, h * N_MAGNITUDE_BINS + m] = core
        for dh in (-1, 1):                       # circular in heading
            hn = (h + dh) % N_HEADING_BINS
            out[rows, hn * N_MAGNITUDE_BINS + m] += heading_eps
        for dm in (-1, 1):                       # clipped in magnitude
            mn = np.clip(m + dm, 0, N_MAGNITUDE_BINS - 1)
            out[rows, h * N_MAGNITUDE_BINS + mn] += magnitude_eps
        # Clipping at the magnitude edges folds mass back onto the core;
        # renormalise so every row is a proper distribution.
        out[rows] /= out[rows].sum(axis=1, keepdims=True)

    return out.reshape(*idx.shape, N_BINS)


# --------------------------------------------------------------------- #
#  Training path: collector shards -> network inputs
# --------------------------------------------------------------------- #

def featurize_shard(d: Dict[str, np.ndarray], arena_size: float = 130.0
                    ) -> Dict[str, np.ndarray]:
    """Turn one loaded ``.npz`` shard into network inputs + targets.

    Each stored sample is ONE (step, red) pair, so every graph here has
    exactly one red node.  That is exact, not an approximation: with no
    red->red edges (see ``RedMotionGNN``'s docstring for why the current
    adversary justifies none), the computation factorises per red, so a
    batch of single-red graphs is identical to processing them jointly.
    """
    n = d["accel"].shape[0]
    blue_cap = d["blue_rel_pos"].shape[1]
    obs_cap = d["obs_rel_pos"].shape[1]

    # red_vel is stored RAW (the collector documents it as absolute, for
    # reference); everything else already arrives normalised.
    red_vel = d["red_vel"].astype(np.float64) / V_NORM
    speed = np.linalg.norm(red_vel, axis=-1, keepdims=True)
    red_feats = np.concatenate(
        [red_vel, speed, d["wall_dist"].astype(np.float64)], axis=-1
    )[:, None, :]                                        # (n, 1, 7)

    blue_active = (np.arange(blue_cap)[None, :] < d["n_blue"][:, None])
    blue_feats = blue_active.astype(np.float64)[..., None]           # (n, B, 1)

    obs_mask = d["obs_mask"]
    obs_feats = np.stack(
        [obs_mask.astype(np.float64), d["obs_radius"].astype(np.float64)],
        axis=-1)                                                     # (n, O, 2)

    rv = red_vel[:, None, :]                             # broadcast receiver vel
    b2r = edge_features(
        d["blue_rel_pos"].astype(np.float64),
        d["blue_rel_vel"].astype(np.float64) - rv,       # sender minus receiver
        np.broadcast_to(rv, (n, blue_cap, 2)))           # (n, B, 7)
    o2r = edge_features(
        d["obs_rel_pos"].astype(np.float64),
        d["obs_rel_vel"].astype(np.float64) - rv,
        np.broadcast_to(rv, (n, obs_cap, 2)))            # (n, O, 7)

    return dict(
        red_feats=red_feats.astype(np.float32),
        blue_feats=blue_feats.astype(np.float32),
        obs_feats=obs_feats.astype(np.float32),
        b2r_edge_feats=b2r.astype(np.float32),
        o2r_edge_feats=o2r.astype(np.float32),
        b2r_active=blue_active.astype(np.float32),
        o2r_active=obs_mask.astype(np.float32),
        target=accel_to_bin(d["accel"]).astype(np.int64),
        accel=d["accel"].astype(np.float32),
        episode_id=d["episode_id"],
    )
