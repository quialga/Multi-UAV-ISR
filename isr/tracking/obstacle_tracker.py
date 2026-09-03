"""
isr/tracking/obstacle_tracker.py — Kalman tracking for obstacles.

Same motivation as ``tracker.py``'s red tracker: ``_build_obstacle_tracks``
loops over TRUE obstacle indices and reports the exact radius when seen —
a perfect data association PLUS a noiseless size measurement handed to the
policy. This module consumes ``PursuitEnv.raw_obstacle_detections()``
instead — position, radius, Doppler, line of sight and per-return sigmas,
with NO obstacle label — and maintains tracks with persistent ids, exactly
the association problem the red tracker solves.

Why a SEPARATE tracker rather than reusing ``MultiTargetTracker``:

* Obstacles carry a RADIUS the state must hold; reds do not. State is
  ``[px, py, vx, vy, r]`` (5-D), not the red tracker's ``[px, py, vx, vy]``.
* Obstacles do not evade — there is no "flee the nearest blue"
  discontinuity, so there is currently no MEASURED reason to think an
  obstacle's motion is multimodal (see ``docs/tracking_diagnostics.md``
  Sec. 9). Building in the red tracker's Gaussian-Sum machinery here would
  be speculative generality with no evidence behind it — same discipline
  as everywhere else in this module: measure before building. A single
  Gaussian per track is the right amount of machinery today.

What IS shared: the Joseph-form Kalman update
(``isr.tracking.kalman.joseph_update``, dimension-agnostic) and the
Hungarian solver (``isr.tracking.assignment.solve_gated``) — the same
primitives, not a second hand-copied implementation.

The per-step loop mirrors the red tracker's, minus the Gaussian-Sum steps:

    1. PREDICT  every track: x- = Fx, P- = F P F^T + Q  (radius: F_r = 1,
                Q_r = 0 exactly — see the RADIUS note below)
    2. GATE     Mahalanobis d^2 of every (track, detection) pair, against
                the prediction
    3. ASSOCIATE  Hungarian PER OBSERVING BLUE (same reasoning as the red
                tracker: one return per obstacle per blue per scan, but a
                track can take several returns, one per observer)
    4. UPDATE   sequential Kalman per assigned pair: position, radius,
                then Doppler
    5. BIRTH    cluster unassigned returns ACROSS blues; one cluster = one
                tentative track, radius seeded from the group's own
                measurements
    6. COAST / DIE / PROMOTE

RADIUS is the cleanest part of this design: a rigid obstacle's TRUE radius
does not change over an episode, so `Q_r = 0` is not an approximation the
way DWNA's white-noise acceleration is for the red — it is exactly correct,
and the filter's radius variance shrinks monotonically with every sighting,
never diverging.

VELOCITY: `sigma_a` for obstacle acceleration must be measured, not
inherited from the red's value — a patrolling obstacle's motion is
constant-velocity between wall bounces (near-zero acceleration) with an
instantaneous velocity-sign FLIP exactly AT a bounce, a fundamentally
different statistic from the red's continuously-manoeuvring pursuit.  See
``docs/tracking_diagnostics.md`` Sec. 9 for the measurement and for the
known, un-modelled limitation this leaves: a bounce is a discontinuity a
constant-velocity KF cannot predict through, and it is corrected only
AFTER the fact by the next few updates disagreeing with the prediction —
exactly the same class of gap the ``motion_model`` plug point exists to
close for the red tracker, and left for the same reason (measure the real
cost before building for it — the moving-obstacle fraction defaults to 0
in every stage4 config to date, so this has not yet cost anything).

The ``motion_model`` plug point is carried here too, for interface
symmetry with the red tracker and because a future FAST, SMALL interceptor
obstacle (proportional navigation, "for later" per docs Sec. 9) may need
one — not because anything needs it today.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from isr.tracking.assignment import solve_gated
from isr.tracking.kalman import joseph_update

CHI2_99 = {1: 6.635, 2: 9.210, 4: 13.277, 5: 15.086}


def _dwna_Q5(dt: float, sigma_a: float) -> np.ndarray:
    """Process noise for the 5-D obstacle state ``[px, py, vx, vy, r]``:
    the SAME discrete white-noise-acceleration block as the red tracker's
    ``_dwna_Q`` embedded in the position/velocity rows, with radius left
    EXACTLY untouched (Q_r = 0 — see the module docstring's RADIUS note).
    """
    q_pp = dt ** 4 / 4.0
    q_pv = dt ** 3 / 2.0
    q_vv = dt ** 2
    Q = np.zeros((5, 5), dtype=np.float64)
    for i, j in ((0, 2), (1, 3)):          # (px,vx) and (py,vy) blocks
        Q[i, i] = q_pp
        Q[i, j] = Q[j, i] = q_pv
        Q[j, j] = q_vv
    return Q * (sigma_a ** 2)


class ObstacleTrack:
    """One hypothesis about one physical obstacle.  Single Gaussian — see
    the module docstring for why this does not need a Gaussian Sum today.
    State: x = [px, py, vx, vy, r]."""

    _next_id = 0

    def __init__(self, x: np.ndarray, P: np.ndarray, t: int) -> None:
        ObstacleTrack._next_id += 1
        self.id = ObstacleTrack._next_id
        self.x = np.asarray(x, dtype=np.float64).reshape(5)
        self.P = np.asarray(P, dtype=np.float64).reshape(5, 5)
        self.born_at = t
        self.history: List[bool] = []
        self.misses = 0
        self.confirmed = False

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def radius(self) -> float:
        return float(self.x[4])

    @property
    def hits(self) -> int:
        return int(sum(self.history))

    def __repr__(self) -> str:
        tag = "conf" if self.confirmed else "tent"
        return (f"ObstacleTrack(id={self.id} {tag} pos={np.round(self.pos, 1)} "
                f"r={self.radius:.1f} vel={np.round(self.vel, 2)} "
                f"misses={self.misses})")


class ObstacleTracker:
    """Kalman + gating + per-blue Hungarian + M-of-N lifecycle, for
    obstacles. See the module docstring for the full design rationale."""

    def __init__(
        self,
        dt: float = 1.0,
        sigma_a: float = 0.1,
        vel_prior_std: float = 1.0,
        gate_chi2: float = CHI2_99[2],
        confirm_hits: int = 2,
        confirm_window: int = 3,
        max_misses: int = 5,
        birth_cluster_dist: float = 6.0,
        oracle_association: bool = False,
        motion_model=None,
    ) -> None:
        # Plug point for a learned/explicit motion model (e.g. one that
        # predicts a wall bounce) — see the module docstring.  A callable
        # (x, P) -> (x_pred, P_pred); None uses constant-velocity + DWNA.
        self.motion_model = motion_model
        # EVALUATION ONLY — see MultiTargetTracker's identical parameter.
        self.oracle_association = bool(oracle_association)
        self.dt = float(dt)
        # sigma_a measured from the EXISTING patrol/bounce physics
        # (_move_obstacles), not inherited from the red's value: between
        # bounces acceleration is exactly 0 (99.1% of steps measured);
        # AT a bounce it is an instantaneous |a| = 2*obstacle_speed
        # velocity-sign flip (0.9% of steps) — a discrete regime change a
        # continuous-noise KF cannot represent, not a small perturbation.
        # A NEES sweep (docs/tracking_diagnostics.md Sec. 9) found static
        # obstacles are INSENSITIVE to sigma_a (NEES ~2.9 flat from 0.005
        # to 0.3), while moving (patrolling) ones are not: sigma_a=0.005
        # gives NEES in the MILLIONS at a bounce (catastrophically
        # overconfident), 0.05 still 2x too confident (NEES 11.4), 0.1
        # calibrates well (NEES 4.96 vs a target of 5.0) at negligible cost
        # to the static case (2.91 vs 2.87).  A discrete bounce is still
        # NOT predicted through — it is corrected only by the next few
        # updates disagreeing with the (temporarily wrong) prediction; see
        # the module docstring's VELOCITY note.
        self.sigma_a = float(sigma_a)
        # Gaussian speed prior for the velocity ridge at birth — same role
        # as the red tracker's vel_prior_std (the characteristic scale of
        # "how fast could this possibly be moving", NOT necessarily 0 even
        # for an obstacle: a patrolling one has real speed).  Makes a
        # single non-collinear-poor return well-posed instead of leaving
        # the tangential direction unconstrained.  RADIUS needs no such
        # prior — it is a scalar measured directly, never rank-deficient
        # the way a single line-of-sight velocity measurement is.
        self.vel_prior_std = float(vel_prior_std)
        self.gate_chi2 = float(gate_chi2)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window = int(confirm_window)
        self.max_misses = int(max_misses)
        self.birth_cluster_dist = float(birth_cluster_dist)

        self.F = np.eye(5)
        self.F[0, 2] = self.F[1, 3] = self.dt
        # F[4, 4] (radius) is already 1 from np.eye — no dynamics.
        self.Q = _dwna_Q5(self.dt, self.sigma_a)
        self.H_pos = np.zeros((2, 5)); self.H_pos[0, 0] = self.H_pos[1, 1] = 1.0
        self.H_r = np.zeros((1, 5)); self.H_r[0, 4] = 1.0

        self.tracks: List[ObstacleTrack] = []
        self.t = 0
        self.last_nis: List[float] = []
        self._oracle_map: Dict[int, int] = {}

    # ---------------- gating ---------------- #

    def _gate_pos(self, tr: ObstacleTrack, det: Dict):
        R = np.eye(2) * det["sigma_pos"] ** 2
        y = det["z_pos"].astype(np.float64) - self.H_pos @ tr.x
        S = self.H_pos @ tr.P @ self.H_pos.T + R
        Si = np.linalg.inv(S)
        d2 = float(y @ Si @ y)
        sign, logdet = np.linalg.slogdet(S)
        return d2, d2 + (logdet if sign > 0 else 0.0)

    # ---------------- birth ---------------- #

    def _fuse_birth(self, group: Sequence[Dict]):
        """Initialise a track from one cluster of simultaneous returns.

        Position: inverse-variance weighted mean, same as the red tracker.
        Radius: inverse-variance weighted mean of the group's OWN radius
        measurements; if any return in the group is EXACT
        (``sigma_radius = 0``, the default when ``obstacle_radius_noise_std``
        is off), that value is taken as the truth outright rather than
        blended with noisier ones.
        Velocity: WLS on the RADIAL components with a Gaussian speed prior
        (``vel_prior_std``) as the ridge term — identical construction to
        the red tracker's ``_fuse_radial_velocity``, making a single
        non-collinear-poor return well-posed instead of leaving the
        tangential direction unconstrained.
        """
        w = np.array([1.0 / max(d["sigma_pos"] ** 2, 1e-9) for d in group])
        pos = np.sum([d["z_pos"] * wi for d, wi in zip(group, w)], axis=0) / w.sum()
        P_pos = np.eye(2) / w.sum()

        sigmas_r = [float(d["sigma_radius"]) for d in group]
        if any(s <= 0.0 for s in sigmas_r):
            # obstacle_radius_noise_std = 0 (the default): the FIRST exact
            # return is the truth, not something to blend with a prior.
            exact = next(d["z_radius"] for d, s in zip(group, sigmas_r)
                        if s <= 0.0)
            r, P_r = float(exact), 1e-9
        else:
            rw = np.array([1.0 / s ** 2 for s in sigmas_r])
            r = float(np.sum([d["z_radius"] * wi for d, wi in zip(group, rw)])
                     / rw.sum())
            P_r = float(1.0 / rw.sum())

        s_prior2 = max(self.vel_prior_std, 1e-6) ** 2
        U = np.stack([np.asarray(d["los"], dtype=np.float64) for d in group])
        sig = np.array([max(d["sigma_radial"], 1e-6) for d in group])
        rmeas = np.array([d["z_radial"] for d in group], dtype=np.float64)
        W = np.diag(1.0 / sig ** 2)
        P_vel = np.linalg.inv(U.T @ W @ U + np.eye(2) / s_prior2)
        vel = P_vel @ U.T @ W @ rmeas

        x = np.concatenate([pos, vel, [r]])
        P = np.zeros((5, 5))
        P[:2, :2] = P_pos
        P[2:4, 2:4] = P_vel
        P[4, 4] = P_r
        return x, P

    @staticmethod
    def _cluster(dets: List[Dict], radius: float) -> List[List[Dict]]:
        groups: List[List[Dict]] = []
        for d in dets:
            for g in groups:
                if any(np.linalg.norm(d["z_pos"] - o["z_pos"]) <= radius
                       for o in g):
                    g.append(d)
                    break
            else:
                groups.append([d])
        return groups

    # ---------------- the per-step loop ---------------- #

    def step(self, detections: Sequence[Dict]) -> None:
        self.t += 1
        self.last_nis = []

        # 1. PREDICT ------------------------------------------------------
        for tr in self.tracks:
            if self.motion_model is None:
                tr.x = self.F @ tr.x
                tr.P = self.F @ tr.P @ self.F.T + self.Q
            else:
                tr.x, tr.P = self.motion_model(tr.x, tr.P)

        dets = list(detections)
        assigned_det = set()
        hit_tracks = set()

        by_blue: Dict[int, List[int]] = {}
        for k, d in enumerate(dets):
            by_blue.setdefault(int(d["blue"]), []).append(k)

        assignments: List[tuple] = []
        if self.oracle_association:
            by_id = {tr.id: i for i, tr in enumerate(self.tracks)}
            for k, d in enumerate(dets):
                if int(d["truth_id"]) < 0:
                    continue
                tid = self._oracle_map.get(int(d["truth_id"]))
                if tid is not None and tid in by_id:
                    assignments.append((by_id[tid], k))
        else:
            for _blue, idxs in sorted(by_blue.items()):
                if not self.tracks:
                    break
                n, m = len(self.tracks), len(idxs)
                cost = np.zeros((n, m))
                gate = np.zeros((n, m), dtype=bool)
                for i, tr in enumerate(self.tracks):
                    for j, k in enumerate(idxs):
                        d2, c = self._gate_pos(tr, dets[k])
                        cost[i, j] = c
                        gate[i, j] = d2 <= self.gate_chi2
                for i, j in solve_gated(cost, gate):
                    assignments.append((i, idxs[j]))

        # 4. UPDATE — sequential: position, radius, then Doppler.
        for i, k in assignments:
            tr = self.tracks[i]
            d = dets[k]
            R_p = np.eye(2) * d["sigma_pos"] ** 2
            tr.x, tr.P, nis, _ll = joseph_update(
                tr.x, tr.P, self.H_pos, R_p, d["z_pos"].astype(np.float64))
            self.last_nis.append(nis)

            if d["sigma_radius"] > 0.0:
                R_r = np.array([[d["sigma_radius"] ** 2]])
                tr.x, tr.P, nis, _ll = joseph_update(
                    tr.x, tr.P, self.H_r, R_r, np.array([d["z_radius"]]))
                self.last_nis.append(nis)

            u = np.asarray(d["los"], dtype=np.float64)
            if np.linalg.norm(u) > 1e-6 and d["sigma_radial"] > 0.0:
                H_d = np.array([[0.0, 0.0, u[0], u[1], 0.0]])
                R_d = np.array([[d["sigma_radial"] ** 2]])
                tr.x, tr.P, nis, _ll = joseph_update(
                    tr.x, tr.P, H_d, R_d, np.array([d["z_radial"]]))
                self.last_nis.append(nis)

            assigned_det.add(k)
            hit_tracks.add(i)

        # 5. BIRTH ----------------------------------------------------
        leftovers = [dets[k] for k in range(len(dets)) if k not in assigned_det]
        if self.oracle_association:
            leftovers = [d for d in leftovers if int(d["truth_id"]) >= 0]
        for group in self._cluster(leftovers, self.birth_cluster_dist):
            x, P = self._fuse_birth(group)
            tr = ObstacleTrack(x, P, self.t)
            self.tracks.append(tr)
            if self.oracle_association:
                self._oracle_map[int(group[0]["truth_id"])] = tr.id

        # 6. COAST / DIE / PROMOTE.
        survivors: List[ObstacleTrack] = []
        for i, tr in enumerate(self.tracks):
            if tr.born_at == self.t:
                tr.history.append(True)
                tr.misses = 0
            elif i in hit_tracks:
                tr.history.append(True)
                tr.misses = 0
            else:
                tr.history.append(False)
                tr.misses += 1
            tr.history = tr.history[-self.confirm_window:]
            if not tr.confirmed and tr.hits >= self.confirm_hits:
                tr.confirmed = True
            if tr.misses <= self.max_misses:
                survivors.append(tr)
        self.tracks = survivors

    # ---------------- readout ---------------- #

    def confirmed_tracks(self) -> List[ObstacleTrack]:
        return [t for t in self.tracks if t.confirmed]

    def reset(self) -> None:
        self.tracks = []
        self.t = 0
