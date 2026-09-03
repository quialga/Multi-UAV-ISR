# NON-REGRESSION FIXTURE — frozen copy of tracker.py from before the
# Gaussian-Sum refactor (git commit b18bf66).  Used ONLY by
# tests/test_gaussian_sum.py to prove the refactor is bit-exact for the
# default (single-Gaussian) case.  Do not import this anywhere else, and
# do not "fix" it when tracker.py changes again -- it is meant to stay
# frozen at that commit.
"""
isr/tracking/tracker.py — multi-target tracker over identity-free returns.

Why this exists: ``PursuitEnv._build_enemy_tracks`` loops over TRUE red
indices, measures ``_red_pos[r]``, and groups returns for fusion by
``detect[:, r]``.  That is a PERFECT data association handed over by the
simulator, plus a slot->target mapping stable across steps.  The
``track_red = -1`` on memory tracks is the other face of it: identity
vanishes the moment the oracle cannot help.  There is no association layer,
and its absence is hidden by the oracle.

This module is that layer.  It consumes ``PursuitEnv.raw_detections()`` —
position, Doppler, line of sight and per-return sigmas, with NO target
label — and maintains tracks with persistent ids.

Per step (the ordering matters; see the two notes below):

    1. PREDICT  every track:  x- = Fx,  P- = F P F^T + Q
    2. GATE     Mahalanobis d^2 of every (track, detection) pair, ALL
                against the prediction
    3. ASSOCIATE  Hungarian PER OBSERVING BLUE
    4. UPDATE   sequential Kalman per assigned pair (position, then
                Doppler), Joseph form
    5. BIRTH    cluster the unassigned returns ACROSS blues; one cluster =
                one tentative track
    6. COAST / DIE / PROMOTE

Note 1 — gate against the prediction, never mid-update.  Gating detection 2
against a state already corrected by detection 1 makes the association
order-dependent: the first update moves the state and can push a perfectly
valid second return outside the gate.  Associate once, from the prior.

Note 2 — Hungarian PER BLUE, not global.  A radar yields at most one return
per target per scan, so within one blue's returns the matching is one-to-one
(exactly what Hungarian enforces).  ACROSS blues a track must be able to
take several returns, one per observer — that is the non-collinear geometry
that determines velocity.  One global Hungarian would force one-to-one over
everything and throw all but one return per track away.

Note 3 — births happen after ALL blues are associated.  Spawning inside the
per-blue loop would create one track per observing blue for the same new
target.  Unassigned returns are clustered first, and a cluster with >= 2
non-collinear lines of sight is born with its velocity already fused.

State per track: x = [px, py, vx, vy].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from isr.tracking.assignment import solve_gated

# Chi-square 99% quantiles.
CHI2_99 = {1: 6.635, 2: 9.210, 4: 13.277}


def _dwna_Q(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise acceleration process noise, INDEPENDENT axes.

    Per axis the [p, v] block is sigma_a^2 * [[dt^4/4, dt^3/2],
                                              [dt^3/2, dt^2   ]].
    Writing it as G G^T with a single scalar G = [dt^2/2, dt^2/2, dt, dt]
    would be rank-1 — it would assert that a_x and a_y are the SAME random
    variable, perfectly correlated.  They are not.
    """
    q_pp = dt ** 4 / 4.0
    q_pv = dt ** 3 / 2.0
    q_vv = dt ** 2
    Q = np.zeros((4, 4), dtype=np.float64)
    for i, j in ((0, 2), (1, 3)):          # (px,vx) and (py,vy) blocks
        Q[i, i] = q_pp
        Q[i, j] = Q[j, i] = q_pv
        Q[j, j] = q_vv
    return Q * (sigma_a ** 2)


class Track:
    """One hypothesis about one physical target."""

    _next_id = 0

    def __init__(self, x: np.ndarray, P: np.ndarray, t: int) -> None:
        Track._next_id += 1
        self.id = Track._next_id
        self.x = np.asarray(x, dtype=np.float64).reshape(4)
        self.P = np.asarray(P, dtype=np.float64).reshape(4, 4)
        self.born_at = t
        self.history: List[bool] = []      # hit / miss, most recent last
        self.misses = 0                    # CONSECUTIVE misses
        self.confirmed = False

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def hits(self) -> int:
        return int(sum(self.history))

    def __repr__(self) -> str:
        tag = "conf" if self.confirmed else "tent"
        return (f"Track(id={self.id} {tag} pos={np.round(self.pos, 1)} "
                f"vel={np.round(self.vel, 2)} misses={self.misses})")


class MultiTargetTracker:
    """Kalman + gating + per-blue Hungarian + M-of-N lifecycle."""

    def __init__(
        self,
        dt: float = 1.0,
        a_max: float = 1.0,
        vel_prior_std: float = 1.0,
        sigma_a: Optional[float] = None,
        gate_chi2: float = CHI2_99[2],
        confirm_hits: int = 2,
        confirm_window: int = 3,
        max_misses: int = 5,
        birth_cluster_dist: float = 6.0,
        oracle_association: bool = False,
        motion_model=None,
    ) -> None:
        # Plug point for a LEARNED transition model.  A callable
        # (x, P) -> (x_pred, P_pred); None uses the constant-velocity F/Q.
        # Swapping it is the whole migration path: gating, association,
        # lifecycle and metrics stay untouched.
        self.motion_model = motion_model
        # EVALUATION ONLY.  Reads det["truth_id"] to associate perfectly,
        # which isolates FILTER quality from ASSOCIATION quality: the gap
        # between oracle and real association IS the cost of associating.
        # Never enable it for anything a policy consumes.
        self.oracle_association = bool(oracle_association)
        self.dt = float(dt)
        # sigma_a is a STANDARD DEVIATION, not a bound.
        #
        # Step 1 — the white-noise value.  This red normalises its
        # acceleration to unit magnitude, so only the DIRECTION varies:
        # a_i = a_max*cos(theta), hence Var[a_i] = a_max^2/2 and
        # sigma_white = a_max/sqrt(2) = 0.707.  Measured 0.704.  (The
        # a_max/sqrt(3) of a UNIFORM magnitude does not apply here.)
        #
        # Step 2 — the correlation correction.  DWNA assumes WHITE noise,
        # and this acceleration is not: measured lag-1 rho = 0.546 (per
        # track; a naive estimate that concatenates tracks is meaningless).
        # For an AR(1)-like acceleration the variance of the sum of N
        # increments is
        #     Var = N s^2 (1 + 2 sum_{k=1..N-1} (1 - k/N) rho^k)
        # against N s^2 for white noise, i.e. an inflation of 2.4-3.1 over
        # 5-20 step horizons, so sigma_a should be scaled by 1.55-1.77.
        # An independent NEES sweep put the optimum at ~2x the white-noise
        # value.  Theory and measurement agree within ~15%, so:
        #
        #     sigma_a = a_max * sqrt(2)   (= 2x the white-noise value)
        #
        # NEES 5.29 -> 4.91 against a target of 4.0.  Note this does NOT
        # change recall (a Q sweep over 16x moved it 0.30-0.31): Q scales
        # the covariance, not the predicted MEAN.  It buys calibration and
        # fewer false positives, nothing more.
        self.sigma_a = float(a_max * np.sqrt(2.0)) if sigma_a is None \
            else float(sigma_a)
        self.vel_prior_std = float(vel_prior_std)
        self.gate_chi2 = float(gate_chi2)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window = int(confirm_window)
        self.max_misses = int(max_misses)
        self.birth_cluster_dist = float(birth_cluster_dist)

        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = self.dt
        self.Q = _dwna_Q(self.dt, self.sigma_a)
        self.H_pos = np.zeros((2, 4)); self.H_pos[0, 0] = self.H_pos[1, 1] = 1.0

        self.tracks: List[Track] = []
        self.t = 0
        self.last_nis: List[float] = []      # NIS of this step's updates
        self._oracle_map: Dict[int, int] = {}   # truth_id -> track id

    # ---------------- Kalman primitives ---------------- #

    @staticmethod
    def _update(x, P, H, R, z):
        """Joseph-form Kalman update.

        P = (I-KH) P (I-KH)^T + K R K^T rather than (I-KH)P: the short form
        is only valid for the optimal gain in exact arithmetic, and loses
        symmetry / positive-definiteness under the many sequential updates
        this tracker performs (up to n_blue x 2 per step, for hundreds of
        steps).  Joseph costs one extra product and is stable for any K.
        """
        y = np.atleast_1d(z) - H @ x
        S = H @ P @ H.T + R
        Si = np.linalg.inv(S)
        nis = float(y @ Si @ y)
        K = P @ H.T @ Si
        x_new = x + K @ y
        A = np.eye(4) - K @ H
        P_new = A @ P @ A.T + K @ R @ K.T
        P_new = 0.5 * (P_new + P_new.T)          # kill drift in symmetry
        return x_new, P_new, nis

    def _gate_pos(self, tr: Track, det: Dict):
        """Mahalanobis d^2 and the NLL cost of pairing a track with a
        POSITION return, evaluated at the prediction."""
        R = np.eye(2) * det["sigma_pos"] ** 2
        y = det["z_pos"].astype(np.float64) - self.H_pos @ tr.x
        S = self.H_pos @ tr.P @ self.H_pos.T + R
        Si = np.linalg.inv(S)
        d2 = float(y @ Si @ y)
        # cost = d^2 + ln|S|: the log-determinant stops very uncertain
        # tracks from hoovering up every detection just because their gate
        # is wide.
        sign, logdet = np.linalg.slogdet(S)
        return d2, d2 + (logdet if sign > 0 else 0.0)

    # ---------------- birth ---------------- #

    def _fuse_birth(self, group: Sequence[Dict]):
        """Initialise a track from one cluster of simultaneous returns.

        Position: inverse-variance weighted mean.  Velocity: weighted least
        squares on the RADIAL components with a Gaussian speed prior (ridge)
        — identical to the command-layer fusion, so a cluster with >= 2
        non-collinear lines of sight is born already knowing its velocity,
        and a single return falls back to the prior in the unobserved
        direction instead of blowing up.
        """
        w = np.array([1.0 / max(d["sigma_pos"] ** 2, 1e-9) for d in group])
        pos = np.sum([d["z_pos"] * wi for d, wi in zip(group, w)], axis=0) / w.sum()
        P_pos = np.eye(2) / w.sum()

        s_prior2 = max(self.vel_prior_std, 1e-6) ** 2
        U = np.stack([np.asarray(d["los"], dtype=np.float64) for d in group])
        sig = np.array([max(d["sigma_radial"], 1e-6) for d in group])
        r = np.array([d["z_radial"] for d in group], dtype=np.float64)
        W = np.diag(1.0 / sig ** 2)
        P_vel = np.linalg.inv(U.T @ W @ U + np.eye(2) / s_prior2)
        vel = P_vel @ U.T @ W @ r

        x = np.concatenate([pos, vel])
        P = np.zeros((4, 4))
        P[:2, :2] = P_pos
        P[2:, 2:] = P_vel
        return x, P

    @staticmethod
    def _cluster(dets: List[Dict], radius: float) -> List[List[Dict]]:
        """Greedy single-link clustering of unassigned returns by position."""
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

        # 2-3. GATE + ASSOCIATE, per observing blue.  Everything is scored
        # against the PREDICTION, before any update is applied.
        by_blue: Dict[int, List[int]] = {}
        for k, d in enumerate(dets):
            by_blue.setdefault(int(d["blue"]), []).append(k)

        assignments: List[tuple] = []      # (track_idx, det_idx)
        if self.oracle_association:
            # EVALUATION ONLY — associate straight from the labels, so the
            # gap against real association measures exactly what associating
            # costs.  Perfect association also means perfect CLUTTER
            # REJECTION (truth_id < 0): telling a false alarm from a target
            # is part of the association problem, so the oracle gets it
            # right by construction and the real tracker has to earn it.
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

        # 4. UPDATE — sequential, position then Doppler, Joseph form.
        for i, k in assignments:
            tr = self.tracks[i]
            d = dets[k]
            R_p = np.eye(2) * d["sigma_pos"] ** 2
            tr.x, tr.P, nis = self._update(tr.x, tr.P, self.H_pos, R_p,
                                           d["z_pos"].astype(np.float64))
            self.last_nis.append(nis)
            # Doppler is LINEAR in the state once the line of sight is
            # known: h(x) = [0, 0, ux, uy] x.  No EKF needed.
            u = np.asarray(d["los"], dtype=np.float64)
            if np.linalg.norm(u) > 1e-6 and d["sigma_radial"] > 0.0:
                H_d = np.array([[0.0, 0.0, u[0], u[1]]])
                R_d = np.array([[d["sigma_radial"] ** 2]])
                tr.x, tr.P, nis = self._update(tr.x, tr.P, H_d, R_d,
                                               np.array([d["z_radial"]]))
                self.last_nis.append(nis)
            assigned_det.add(k)
            hit_tracks.add(i)

        # 5. BIRTH — after ALL blues, so one new target does not spawn one
        # track per observing blue.
        leftovers = [dets[k] for k in range(len(dets)) if k not in assigned_det]
        if self.oracle_association:
            # Perfect association never births a track on clutter.
            leftovers = [d for d in leftovers if int(d["truth_id"]) >= 0]
        for group in self._cluster(leftovers, self.birth_cluster_dist):
            x, P = self._fuse_birth(group)
            tr = Track(x, P, self.t)
            self.tracks.append(tr)
            if self.oracle_association:
                self._oracle_map[int(group[0]["truth_id"])] = tr.id

        # 6. COAST / DIE / PROMOTE.
        survivors: List[Track] = []
        for i, tr in enumerate(self.tracks):
            if tr.born_at == self.t:          # just born this step
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

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]

    def reset(self) -> None:
        self.tracks = []
        self.t = 0
