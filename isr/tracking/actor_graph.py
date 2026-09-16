"""
isr/tracking/actor_graph.py — the trackers as the actor's observation.

Turns the red and obstacle trackers' state into the fixed-size node slots
the actor's GNN consumes, replacing the belief-map path
(``PursuitEnv._build_enemy_tracks`` / ``_build_obstacle_tracks``), which
leaks ground truth: one slot per TRUE red, the true index behind a live
slot, the true radius of a seen obstacle, and memory slot counts equal to
the number of unseen entities.  Here the number of present nodes is the
number of tracks the trackers hold — nothing the sensors did not report.

Design (docs/tracker_observation.md):

* Only CONFIRMED tracks become nodes; tentative tracks never do.
* A red track is dropped once its position sd (largest eigen-sd of the
  mixture's moment-matched position covariance) exceeds ``sigma_cutoff``:
  beyond the sensor radius a blue flying to the readout no longer finds the
  target reliably (docs/tracking_diagnostics.md §6.4).
* More tracks than slots: keep the smallest sd.
* Readout: position and velocity of the DOMINANT component (the tracker's
  own readout), uncertainty of the WHOLE mixture (moment-matched).
* Presence is binary — node feature 0 and the edge visibility mask — and
  uncertainty travels as separate features, so no single number has to be
  both a mask and a measure.

The tracker configurations the observation was measured with live here as
well, so the env and the evaluation scripts build the same trackers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from isr.tracking.obstacle_tracker import CHI2_99 as OBS_CHI2_99
from isr.tracking.obstacle_tracker import ObstacleTracker
from isr.tracking.tracker import MultiTargetTracker

# Red node:      [present, sd_px, sd_py, rho_p, sd_vx, sd_vy, rho_v, staleness]
RED_FEAT_DIM = 8
# Obstacle node: [present, r/L, sd_r, sd_px, sd_py, rho_p, sd_vx, sd_vy, rho_v]
OBS_FEAT_DIM = 9

# Red tracker: the chosen configuration of docs/tracking_diagnostics.md §6.2
# plus re-acquisition with confirmation (§6.4).  In-view miss budget 3
# (coverage-aware), coast cap 80.
RED_TRACKER_CONFIG = dict(
    dt=1.0, a_max=1.0, vel_prior_std=1.0,
    confirm_hits=3, confirm_window=4, confirm_deadline=True,
    max_misses=3, max_coast_steps=80, reacquire_after=4,
)
# Coverage margin (k * sd) for the red tracker's miss accounting.
COVERAGE_K_SIGMA = 2.0
# Learned red motion model adapter.  sigma_a_model CALIBRATED against NEES
# at this configuration and FROZEN (docs/tracking_diagnostics.md §11.6):
# with oracle association the mean NEES is 4.08 against a target of 4.0 for
# a 4-D state (median 3.06 against 3.36).  The previous placeholder 0.35
# was over-cautious (NEES 2.97 / 2.24).  Re-calibrate if the geometry, the
# red policy mix or the model itself changes.
LEARNED_MOTION_CONFIG = dict(max_branches=4, sigma_a_model=0.20)

# Obstacle tracker (§9.6): 3-of-4 with deadline, never forgets a confirmed
# obstacle, merges duplicates.  STATIC model when obstacles are known not to
# move; with patrolling obstacles the default sigma_a (0.1) is required.
_OBSTACLE_BASE = dict(confirm_hits=3, confirm_window=4, confirm_deadline=True,
                      max_misses=10 ** 9, merge_chi2=OBS_CHI2_99[3])
OBSTACLE_TRACKER_STATIC = dict(_OBSTACLE_BASE, sigma_a=0.0, vel_prior_std=0.0)
OBSTACLE_TRACKER_MOVING = dict(_OBSTACLE_BASE)

# Scale of the radius sd feature (m): the base radius noise agreed for the
# training configuration.
RADIUS_SD_SCALE = 2.0


def make_red_tracker(motion_model=None) -> MultiTargetTracker:
    kw = dict(RED_TRACKER_CONFIG)
    if motion_model is not None:
        kw.update(motion_model=motion_model, max_components=8)
    return MultiTargetTracker(**kw)


def make_obstacle_tracker(static: bool) -> ObstacleTracker:
    return ObstacleTracker(**(OBSTACLE_TRACKER_STATIC if static
                              else OBSTACLE_TRACKER_MOVING))


# --------------------------------------------------------------------- #
#  Moments and covariance features
# --------------------------------------------------------------------- #

def mixture_covariance(track) -> np.ndarray:
    """Moment-matched covariance of a red track's whole Gaussian mixture
    (4x4).  For a single component it is that component's covariance."""
    comps = track.components
    if len(comps) == 1:
        return np.asarray(comps[0].P, dtype=np.float64)
    w = np.array([c.w for c in comps], dtype=np.float64)
    w = w / w.sum()
    xs = np.stack([np.asarray(c.x, dtype=np.float64) for c in comps])
    mean = w @ xs
    P = np.zeros((xs.shape[1], xs.shape[1]))
    for wi, c, xi in zip(w, comps, xs):
        d = xi - mean
        P += wi * (np.asarray(c.P, dtype=np.float64) + np.outer(d, d))
    return P


def max_sd(P2: np.ndarray) -> float:
    return float(np.sqrt(max(np.linalg.eigvalsh(P2).max(), 0.0)))


def cov2_features(P2: np.ndarray, scale: float) -> Tuple[float, float, float]:
    """(sd_x, sd_y, rho) of a 2x2 covariance, sds squashed as
    tanh(sd / scale) into [0, 1) and the correlation in [-1, 1]."""
    sxx, syy, sxy = float(P2[0, 0]), float(P2[1, 1]), float(P2[0, 1])
    sx, sy = np.sqrt(max(sxx, 0.0)), np.sqrt(max(syy, 0.0))
    rho = sxy / (sx * sy) if sx * sy > 1e-12 else 0.0
    return (float(np.tanh(sx / scale)), float(np.tanh(sy / scale)),
            float(np.clip(rho, -1.0, 1.0)))


# --------------------------------------------------------------------- #
#  Slots
# --------------------------------------------------------------------- #

def select_red_tracks(tracks: Sequence, k: int, sigma_cutoff: float
                      ) -> List[Tuple[object, np.ndarray, float]]:
    """Confirmed tracks with position sd <= sigma_cutoff, smallest sd
    first, at most ``k``.  Returns ``(track, mixture_cov, sd)`` triples."""
    chosen = []
    for tr in tracks:
        if not tr.confirmed:
            continue
        P = mixture_covariance(tr)
        sd = max_sd(P[:2, :2])
        if sd <= sigma_cutoff:
            chosen.append((tr, P, sd))
    chosen.sort(key=lambda c: (c[2], c[0].id))
    return chosen[:k]


def red_slots(tracks: Sequence, k: int, sigma_cutoff: float,
              pos_scale: float, vel_scale: float,
              max_coast_steps: int) -> Dict[str, np.ndarray]:
    """Fixed-size red slots.  Keys: ``pos`` (k,2), ``vel`` (k,2),
    ``feats`` (k, RED_FEAT_DIM), ``present`` (k,) in {0, 1}.
    Padding slots are all zero."""
    pos = np.zeros((k, 2), dtype=np.float32)
    vel = np.zeros((k, 2), dtype=np.float32)
    feats = np.zeros((k, RED_FEAT_DIM), dtype=np.float32)
    present = np.zeros((k,), dtype=np.float32)
    for s, (tr, P, _sd) in enumerate(select_red_tracks(tracks, k, sigma_cutoff)):
        pos[s] = tr.pos
        vel[s] = tr.vel
        feats[s] = (1.0,
                    *cov2_features(P[:2, :2], pos_scale),
                    *cov2_features(P[2:4, 2:4], vel_scale),
                    min(tr.steps_since_hit / float(max_coast_steps), 1.0))
        present[s] = 1.0
    return dict(pos=pos, vel=vel, feats=feats, present=present)


def obstacle_slots(tracks: Sequence, k: int, arena_size: float,
                   pos_scale: float, vel_scale: float,
                   radius_scale: float = RADIUS_SD_SCALE
                   ) -> Dict[str, np.ndarray]:
    """Fixed-size obstacle slots from CONFIRMED obstacle tracks, smallest
    position sd first.  Keys: ``pos``, ``vel`` (k,2), ``radius`` (k,),
    ``feats`` (k, OBS_FEAT_DIM), ``present`` (k,)."""
    conf = [(tr, max_sd(tr.P[:2, :2])) for tr in tracks if tr.confirmed]
    conf.sort(key=lambda c: (c[1], c[0].id))
    pos = np.zeros((k, 2), dtype=np.float32)
    vel = np.zeros((k, 2), dtype=np.float32)
    radius = np.zeros((k,), dtype=np.float32)
    feats = np.zeros((k, OBS_FEAT_DIM), dtype=np.float32)
    present = np.zeros((k,), dtype=np.float32)
    for s, (tr, _sd) in enumerate(conf[:k]):
        pos[s] = tr.pos
        vel[s] = tr.vel
        radius[s] = tr.radius
        sd_r = float(np.sqrt(max(tr.P[4, 4], 0.0)))
        feats[s] = (1.0, tr.radius / arena_size,
                    float(np.tanh(sd_r / radius_scale)),
                    *cov2_features(tr.P[:2, :2], pos_scale),
                    *cov2_features(tr.P[2:4, 2:4], vel_scale))
        present[s] = 1.0
    return dict(pos=pos, vel=vel, radius=radius, feats=feats, present=present)


def confirmed_obstacle_geometry(tracks: Sequence
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centres (n,2), velocities (n,2), radii (n,)) of the confirmed
    obstacle tracks — the estimated obstacle field the actor path uses
    wherever it needs obstacles (motion-model context, occlusion)."""
    conf = [tr for tr in tracks if tr.confirmed]
    if not conf:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    return (np.array([tr.pos for tr in conf], dtype=np.float64),
            np.array([tr.vel for tr in conf], dtype=np.float64),
            np.array([max(tr.radius, 0.0) for tr in conf], dtype=np.float64))
