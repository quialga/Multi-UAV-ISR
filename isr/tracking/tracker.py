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

Per step (the ordering matters; see the notes below):

    1. PREDICT  every track's every COMPONENT through the motion model,
                then prune/merge (see Note 4)
    2. GATE     Mahalanobis d^2 of every (track, detection) pair, ALL
                against the prediction, using the BEST-matching component
    3. ASSOCIATE  Hungarian PER OBSERVING BLUE
    4. UPDATE   sequential Kalman per assigned pair, EVERY component,
                then Bayes-reweight the components by measurement
                likelihood (see Note 5), then prune/merge again
    5. BIRTH    cluster the unassigned returns ACROSS blues; one cluster =
                one tentative track (always unimodal at birth)
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

Note 4 — a track is a GAUSSIAN SUM, not a single Gaussian, and the reduce
step (prune negligible weights, merge near-duplicates, cap the count) is
MANDATORY after every predict, not just after an update.  A coasting track
(no detection to Bayes-reweight it) branches every PREDICT step; with
nothing to prune the hypothesis count is unbounded after a long gap.  See
``docs/tracking_diagnostics.md`` Sec. 8 for why a single Gaussian is wrong
here (with modes ~115 deg apart the mean lands on a heading the evader will
never fly) and for the measured branching factor that sizes the cap.

Note 5 — the update step Bayes-reweights EVERY component of a track by that
component's measurement likelihood, in log-space (numerically necessary:
diverged components can differ in likelihood by many orders of magnitude).
This is what makes "the update prunes automatically": a branch the
detection contradicts is down-weighted, not merely left alone, and the
explicit reduce step only has to clean up what a genuinely ambiguous
observation could not resolve on its own.

Non-regression guarantee: with the default constant-velocity motion model
(``motion_model=None``), PREDICT always emits exactly ONE branch per
existing component, so no track ever exceeds 1 component, the reduce step
is a no-op every time, and the log-space reweight of a single component is
`exp(0)/exp(0) = 1` — inert.  Every number this tracker produces is then
bit-for-bit identical to the single-Kalman-filter implementation it
replaces.  See ``tests/test_gaussian_sum.py``.

State per component: x = [px, py, vx, vy].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from isr.tracking.assignment import solve_gated

# Chi-square 99% quantiles.
CHI2_99 = {1: 6.635, 2: 9.210, 4: 13.277}

# A motion model may return several weighted branches per component:
# (relative_weight, x_pred, P_pred).  Relative weights need not sum to 1 —
# they are renormalised against the parent component's own weight.
MotionModel = "Callable[[np.ndarray, np.ndarray], Sequence[Tuple[float, np.ndarray, np.ndarray]]]"


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


# --------------------------------------------------------------------- #
#  Gaussian Sum primitives
# --------------------------------------------------------------------- #

class _Component:
    """One weighted Gaussian hypothesis inside a Track's mixture."""

    __slots__ = ("w", "x", "P")

    def __init__(self, w: float, x: np.ndarray, P: np.ndarray) -> None:
        self.w = float(w)
        self.x = np.asarray(x, dtype=np.float64).reshape(4)
        self.P = np.asarray(P, dtype=np.float64).reshape(4, 4)

    def copy(self) -> "_Component":
        return _Component(self.w, self.x.copy(), self.P.copy())


def _mahalanobis2_between(c1: _Component, c2: _Component) -> float:
    """Squared Mahalanobis distance between two components' means, using
    their AVERAGED covariance — a cheap, symmetric proxy for "are these
    the same mode, estimated slightly differently" used only to decide
    whether to MERGE.  It is not used for anything that must be exact."""
    d = c1.x - c2.x
    P_avg = 0.5 * (c1.P + c2.P)
    try:
        Pi = np.linalg.inv(P_avg)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(d @ Pi @ d)


def _merge_pair(c1: _Component, c2: _Component) -> _Component:
    """Moment-matched merge of two components judged to be the SAME mode.

    This is the same "spread of the means" formula that would collapse an
    entire mixture to one Gaussian (see docs/tracking_diagnostics.md Sec 8
    for why doing that across GENUINELY separated modes is wrong — the mean
    lands on a heading the evader will never fly).  Applied only to
    near-duplicates gated by ``_mahalanobis2_between``, it is the opposite
    case: two estimates of the one hypothesis, safe to fold together.
    """
    w = c1.w + c2.w
    if w <= 0.0:
        return c1.copy()
    mu = (c1.w * c1.x + c2.w * c2.x) / w
    d1 = c1.x - mu
    d2 = c2.x - mu
    P = (c1.w * (c1.P + np.outer(d1, d1))
       + c2.w * (c2.P + np.outer(d2, d2))) / w
    return _Component(w, mu, P)


class Track:
    """One hypothesis about one physical target — a GAUSSIAN SUM (a mixture
    of weighted Gaussian components), so genuinely multimodal beliefs are
    representable rather than averaged away.

    Two independent sources of multimodality motivate this (see
    ``docs/tracking_diagnostics.md`` Sec. 8 for the measurements): our own
    belief over the target's state, pushed through the evader policy's
    "flee the NEAREST blue" discontinuity (epistemic — present even for a
    deterministic evader, measured 45-78% bimodal depending on how lost the
    track is); and the evader's own committed choices, e.g. rounding an
    obstacle left or right (aleatoric — measured 20% of instants, modes
    ~80 deg apart).  Averaging two ~90-140 deg-separated modes produces a
    heading between them that the evader will never fly.

    Readout is always the DOMINANT (highest-weight) component's mean/cov —
    NEVER the mixture's weighted mean — for exactly that reason.

    Non-regression: with the default constant-velocity motion model there
    is always exactly ONE component, and every property below reduces
    EXACTLY to the pre-Gaussian-Sum single-Kalman-filter Track this
    replaces.
    """

    _next_id = 0

    def __init__(self, x: np.ndarray, P: np.ndarray, t: int,
                w: float = 1.0) -> None:
        Track._next_id += 1
        self.id = Track._next_id
        self.components: List[_Component] = [_Component(w, x, P)]
        self.born_at = t
        self.history: List[bool] = []      # hit / miss, most recent last
        self.misses = 0                    # CONSECUTIVE misses
        self.confirmed = False

    @property
    def _dominant(self) -> _Component:
        return max(self.components, key=lambda c: c.w)

    @property
    def x(self) -> np.ndarray:
        """Dominant component's state.  Read-only: PREDICT/UPDATE mutate
        ``components`` directly, never this property."""
        return self._dominant.x

    @property
    def P(self) -> np.ndarray:
        return self._dominant.P

    @property
    def pos(self) -> np.ndarray:
        return self._dominant.x[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        return self._dominant.x[2:].copy()

    @property
    def n_modes(self) -> int:
        return len(self.components)

    @property
    def hits(self) -> int:
        return int(sum(self.history))

    def __repr__(self) -> str:
        d = self._dominant
        tag = "conf" if self.confirmed else "tent"
        extra = f" +{len(self.components) - 1} modes" if len(self.components) > 1 else ""
        return (f"Track(id={self.id} {tag} pos={np.round(d.x[:2], 1)} "
                f"vel={np.round(d.x[2:], 2)} misses={self.misses}{extra})")


class MultiTargetTracker:
    """Kalman + gating + per-blue Hungarian + M-of-N lifecycle, over a
    Gaussian-Sum state per track."""

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
        max_components: int = 8,
        min_component_weight: float = 1e-3,
        merge_gate: float = 4.0,
    ) -> None:
        # Plug point for a LEARNED transition model.  A callable
        # (x, P) -> [(rel_weight, x_pred, P_pred), ...] — one branch per
        # mode the model predicts (e.g. per discretised heading bin holding
        # >=90% of the mass; see docs/tracking_diagnostics.md Sec. 8).
        # Relative weights need not sum to 1 (renormalised against the
        # parent's own weight).  None -> a single constant-velocity branch,
        # which is what makes this tracker reduce exactly to a plain KF.
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
        # fewer false positives, nothing more.  (Nothing here changes with
        # the Gaussian-Sum refactor: this Q is still the per-component
        # process noise used by the DEFAULT single-branch motion model.)
        self.sigma_a = float(a_max * np.sqrt(2.0)) if sigma_a is None \
            else float(sigma_a)
        self.vel_prior_std = float(vel_prior_std)
        self.gate_chi2 = float(gate_chi2)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window = int(confirm_window)
        self.max_misses = int(max_misses)
        self.birth_cluster_dist = float(birth_cluster_dist)
        # Gaussian-Sum bookkeeping.  Inert under the default motion model
        # (which never branches, so no track ever holds >1 component) —
        # these only start doing anything once a branching motion_model is
        # plugged in.  max_components=8 gives generous headroom over the
        # measured branching factor (mean ~2, p90 3-4; see
        # docs/tracking_diagnostics.md Sec. 8).  merge_gate is a squared
        # Mahalanobis distance between two components' means: a
        # placeholder default, not yet tuned against a real branching
        # model.
        self.max_components = int(max_components)
        self.min_component_weight = float(min_component_weight)
        self.merge_gate = float(merge_gate)

        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = self.dt
        self.Q = _dwna_Q(self.dt, self.sigma_a)
        self.H_pos = np.zeros((2, 4)); self.H_pos[0, 0] = self.H_pos[1, 1] = 1.0

        self.tracks: List[Track] = []
        self.t = 0
        self.last_nis: List[float] = []      # NIS of this step's updates
        self._oracle_map: Dict[int, int] = {}   # truth_id -> track id

    # ---------------- Kalman primitive (per component) ---------------- #

    @staticmethod
    def _update(x, P, H, R, z):
        """Joseph-form Kalman update for ONE component.

        P = (I-KH) P (I-KH)^T + K R K^T rather than (I-KH)P: the short form
        is only valid for the optimal gain in exact arithmetic, and loses
        symmetry / positive-definiteness under the many sequential updates
        this tracker performs (up to n_blue x 2 per step, for hundreds of
        steps).  Joseph costs one extra product and is stable for any K.

        Also returns the innovation's log-likelihood, needed to
        Bayes-reweight a track's components (Note 5 above).  For a
        single-component track this is computed but never changes x/P —
        the softmax reweight of one term is `exp(0)/exp(0) = 1` — so it is
        numerically inert for the default (non-branching) motion model.
        """
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
        A = np.eye(4) - K @ H
        P_new = A @ P @ A.T + K @ R @ K.T
        P_new = 0.5 * (P_new + P_new.T)          # kill drift in symmetry
        return x_new, P_new, nis, loglik

    # ---------------- Gaussian-Sum reduce (prune + merge + cap) -------- #

    def _reduce(self, components: List[_Component]) -> List[_Component]:
        """Renormalise, drop negligible weights, merge near-duplicates,
        cap the count.  See Note 4 on why this must run after every
        PREDICT, not only after an update.

        No-op for a length-1 list (every arithmetic step below is either
        skipped by a length guard or divides/multiplies by the same total,
        i.e. by 1.0 after the first renormalisation) — the non-regression
        path for the default motion model.
        """
        comps = list(components)
        tot = sum(c.w for c in comps)
        if tot <= 0.0 or not comps:
            return comps
        for c in comps:
            c.w /= tot

        if len(comps) > 1:
            survivors = [c for c in comps if c.w >= self.min_component_weight]
            comps = survivors or comps        # never prune down to nothing

        while len(comps) > 1:
            best = None
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    d2 = _mahalanobis2_between(comps[i], comps[j])
                    if d2 <= self.merge_gate and (best is None or d2 < best[0]):
                        best = (d2, i, j)
            if best is None:
                break
            _, i, j = best
            merged = _merge_pair(comps[i], comps[j])
            comps = [c for k, c in enumerate(comps) if k not in (i, j)] + [merged]

        if len(comps) > self.max_components:
            comps.sort(key=lambda c: -c.w)
            comps = comps[:self.max_components]

        tot = sum(c.w for c in comps)
        if tot > 0.0:
            for c in comps:
                c.w /= tot
        return comps

    def _predict_track(self, tr: Track) -> None:
        new_components: List[_Component] = []
        for c in tr.components:
            if self.motion_model is None:
                branches: Sequence[Tuple[float, np.ndarray, np.ndarray]] = (
                    (1.0, self.F @ c.x, self.F @ c.P @ self.F.T + self.Q),
                )
            else:
                branches = self.motion_model(c.x, c.P)
            for rel_w, xb, Pb in branches:
                new_components.append(_Component(c.w * rel_w, xb, Pb))
        tr.components = self._reduce(new_components)

    def _gate_pos(self, tr: Track, det: Dict):
        """Mahalanobis d^2 and the NLL cost of pairing a track with a
        POSITION return, evaluated at the prediction — the BEST (minimum
        cost) match over the track's components, so a multimodal track is
        gated in if ANY of its hypotheses is consistent with the return.
        For a single-component track this is the only candidate, so the
        result is identical to the pre-Gaussian-Sum implementation.
        """
        R = np.eye(2) * det["sigma_pos"] ** 2
        z = det["z_pos"].astype(np.float64)
        best_d2, best_cost = None, None
        for c in tr.components:
            y = z - self.H_pos @ c.x
            S = self.H_pos @ c.P @ self.H_pos.T + R
            Si = np.linalg.inv(S)
            d2 = float(y @ Si @ y)
            # cost = d^2 + ln|S|: the log-determinant stops very uncertain
            # tracks from hoovering up every detection just because their
            # gate is wide.
            sign, logdet = np.linalg.slogdet(S)
            cost = d2 + (logdet if sign > 0 else 0.0)
            if best_cost is None or cost < best_cost:
                best_d2, best_cost = d2, cost
        return best_d2, best_cost

    def _update_track(self, tr: Track, det: Dict) -> None:
        """Update EVERY component with this detection, then Bayes-reweight
        the components by their measurement likelihood (Note 5).  Appends
        the DOMINANT (post-reweight highest-weight) component's NIS to
        ``self.last_nis`` — for a single-component track that IS the only
        component, so this is the same statistic the pre-Gaussian-Sum
        tracker reported.
        """
        R_p = np.eye(2) * det["sigma_pos"] ** 2
        z_pos = det["z_pos"].astype(np.float64)
        u = np.asarray(det["los"], dtype=np.float64)
        has_doppler = np.linalg.norm(u) > 1e-6 and det["sigma_radial"] > 0.0
        if has_doppler:
            H_d = np.array([[0.0, 0.0, u[0], u[1]]])
            R_d = np.array([[det["sigma_radial"] ** 2]])
            z_rad = np.array([det["z_radial"]])

        log_w = np.empty(len(tr.components))
        best_i, best_logw, best_nis = 0, -np.inf, []
        for i, c in enumerate(tr.components):
            x1, P1, nis1, ll1 = self._update(c.x, c.P, self.H_pos, R_p, z_pos)
            total_ll, nis_list = ll1, [nis1]
            if has_doppler:
                x1, P1, nis2, ll2 = self._update(x1, P1, H_d, R_d, z_rad)
                total_ll += ll2
                nis_list.append(nis2)
            c.x, c.P = x1, P1
            log_w[i] = np.log(max(c.w, 1e-300)) + total_ll
            if log_w[i] > best_logw:
                best_i, best_logw, best_nis = i, log_w[i], nis_list

        # Log-space softmax: numerically necessary once components have
        # diverged enough for raw likelihood ratios to under/overflow.
        # Order-preserving, so best_i/best_nis picked above (by log_w)
        # already identify the POST-reweight dominant component.
        m = float(np.max(log_w))
        w = np.exp(log_w - m)
        tot = float(w.sum())
        if tot > 0.0:
            for c, wi in zip(tr.components, w):
                c.w = float(wi / tot)

        self.last_nis.extend(best_nis)
        tr.components = self._reduce(tr.components)

    # ---------------- birth ---------------- #

    def _fuse_birth(self, group: Sequence[Dict]):
        """Initialise a track from one cluster of simultaneous returns.

        Position: inverse-variance weighted mean.  Velocity: weighted least
        squares on the RADIAL components with a Gaussian speed prior (ridge)
        — identical to the command-layer fusion, so a cluster with >= 2
        non-collinear lines of sight is born already knowing its velocity,
        and a single return falls back to the prior in the unobserved
        direction instead of blowing up.  Always produces ONE component —
        a new track starts unimodal; multimodality emerges only from
        PREDICT branching over subsequent steps.
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

        # 1. PREDICT (per component, then reduce) -------------------------
        for tr in self.tracks:
            self._predict_track(tr)

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

        # 4. UPDATE — sequential per assigned pair; every component of the
        # track is updated and Bayes-reweighted (Note 5).
        for i, k in assignments:
            tr = self.tracks[i]
            self._update_track(tr, dets[k])
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
