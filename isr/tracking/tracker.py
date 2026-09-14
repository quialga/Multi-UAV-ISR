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
from isr.tracking.kalman import joseph_update

# Chi-square 99% quantiles.
CHI2_99 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277}

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


def _max_sd(P_pos: np.ndarray) -> float:
    """Standard deviation along the most uncertain axis of a 2x2 covariance."""
    return float(np.sqrt(max(np.linalg.eigvalsh(P_pos).max(), 0.0)))


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
        # CONSECUTIVE misses that COUNT toward death.  Without a coverage
        # model every miss counts; with one, a confirmed track's misses
        # count only when it should have been seen (see step()).
        self.misses = 0
        # Steps since the last hit, counting EVERY miss — the track's plain
        # staleness, whatever the coverage model decided about each miss.
        self.steps_since_hit = 0
        self.confirmed = False
        # Where the track stood at its last hit, and how sure it was then —
        # the anchor of the kinematic reach gate (MultiTargetTracker
        # max_target_speed).  Refreshed on every hit, never on a miss.
        self.last_hit_pos = np.asarray(x, dtype=np.float64)[:2].copy()
        self.last_hit_sd = _max_sd(np.asarray(P, dtype=np.float64)[:2, :2])

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
        doppler_gating: bool = False,
        gate_chi2_doppler: float = CHI2_99[3],
        max_coast_steps: Optional[int] = None,
        confirm_deadline: bool = False,
        max_target_speed: Optional[float] = None,
        reach_k_sigma: float = 3.0,
    ) -> None:
        # Plug point for a LEARNED transition model.  A callable
        # (x, P) -> [(rel_weight, x_pred, P_pred), ...] — one branch per
        # mode the model predicts (see docs/tracking_diagnostics.md Sec. 8).
        # None -> a single constant-velocity branch, which is what makes
        # this tracker reduce exactly to a plain KF.
        #
        # WEIGHTS MUST SUM TO THE SAME TOTAL FOR EVERY CALL — normally 1,
        # i.e. the branches account for ALL of the model's predicted mass.
        # An earlier note here said they "need not sum to 1", which is
        # true only in the harmless case: `_predict_track` sets
        # w_child = w_parent * rel_w and `_reduce` then normalises
        # GLOBALLY across every child of every parent.  If one parent's
        # branches summed to 1.0 and another's to 0.5, that global step
        # would silently move weight from the second parent to the first —
        # not because any evidence favoured it, but because its branch set
        # captured less of its own distribution.  A top-k branch selector
        # that keeps "the cells holding >=90% of the mass" does exactly
        # this, since the number of cells needed varies with the state.
        #
        # Verified for the basin-splitting model in
        # isr/agents/learned_red_motion.py, whose branches keep all the
        # mass by construction: max |1 - sum(rel_w)| = 1.8e-7 over 528
        # calls, so the renormalisation is inert.
        self.motion_model = motion_model
        # EVALUATION ONLY.  Reads det["truth_id"] to associate perfectly,
        # which isolates FILTER quality from ASSOCIATION quality: the gap
        # between oracle and real association IS the cost of associating.
        # Never enable it for anything a policy consumes.
        self.oracle_association = bool(oracle_association)
        self.dt = float(dt)
        # sigma_a is a STANDARD DEVIATION, not a bound.
        #
        # SCOPE: this is the process noise of the DEFAULT constant-velocity
        # motion model ONLY.  self.Q is built from it and is referenced in
        # exactly one place -- the `motion_model is None` branch of
        # _predict_track -- so when a learned motion model is plugged in,
        # Q is never used and this value is irrelevant.  Do not confuse it
        # with LearnedRedMotion.sigma_a_model (0.35), which is that model's
        # own per-branch error term.  They answer different questions:
        #   sigma_a       = 1.414  "I know nothing about the acceleration"
        #   sigma_a_model = 0.35   "the model predicted it; how wrong is it"
        # The ~4x ratio between them IS the learned model's value,
        # quantified -- and the mechanical reason a coasted track holds 5 m
        # of error over 80 steps instead of 53 m (docs Sec. 11.5).
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
        # NEES 5.29 -> 4.91 against a target of 4.0 (deterministic red,
        # 5 m match gate, at the DEFAULT coast budget).
        #
        # Step 3 — the scope of "Q does not change recall".  That was
        # measured at max_misses=5 and holds there: a 16x sweep moves
        # recall 0.320-0.321, with FP minimised exactly at the adopted
        # value.  It does NOT generalise to a long coast budget.  Re-run at
        # max_misses=80 (stochastic red, 20 m gate) the same sweep moves
        # recall 0.387-0.454, about 17% relative:
        #
        #   x_white   0.5    1.0    2.0    4.0    8.0
        #   recall  0.454  0.442  0.428  0.411  0.387
        #   FP       1253    896    777    712    688
        #   MOTA     0.09   0.18   0.20   0.20   0.18
        #
        # The stated mechanism is still right — Q scales the covariance,
        # not the predicted MEAN — but the conclusion does not follow once
        # tracks coast: the covariance sets the GATE, the gate decides
        # whether a re-detection joins the old track or starts a new one,
        # and that decides recall.  A smaller Q buys recall by keeping
        # narrow-gated tracks alive, at a catastrophic FP cost.
        #
        # a_max*sqrt(2) remains the right value at both budgets (best MOTA
        # at 2-4x white, and NEES 3.91 at max_misses=80).  See
        # docs/tracking_diagnostics.md Sec. 11.5 for the coast-budget
        # analysis.
        #
        # (Nothing here changes with the Gaussian-Sum refactor: this Q is
        # still the per-component process noise used by the DEFAULT
        # single-branch motion model.)
        self.sigma_a = float(a_max * np.sqrt(2.0)) if sigma_a is None \
            else float(sigma_a)
        self.vel_prior_std = float(vel_prior_std)
        self.gate_chi2 = float(gate_chi2)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window = int(confirm_window)
        self.max_misses = int(max_misses)
        # Gate (and cost) on POSITION + DOPPLER jointly, instead of position
        # alone.  The radial velocity is already fused in the UPDATE; this
        # also uses it to decide WHETHER a return belongs to a track.
        # Clutter carries a meaningless Doppler, so a false plot must now
        # match the track's predicted radial speed as well as its position
        # — which both protects real tracks from having a clutter plot
        # assigned to them on a scan where the real return was missed, and
        # makes a clutter-born tentative need one more coincidence to
        # confirm.
        #
        # The discrimination is only as good as the PREDICTED velocity, and
        # with the constant-velocity model it is none at all: the calibrated
        # process noise gives Q_vv = sigma_a^2 = 2.0 per axis after every
        # predict (sd ~1.41 m/s), while clutter Doppler lies in [-1, 1], so
        # the Doppler term can add at most ~2.9 to d^2 against a threshold
        # of 11.3.  That is physics rather than tuning — the red accelerates
        # up to 1 m/s^2 each step, so its radial speed really can change
        # that much between scans.  It can only help with a motion model
        # whose velocity prediction is confident, and then only as safely
        # as that confidence is calibrated (an overconfident model would
        # reject a real manoeuvre).  See tests/test_doppler_gating.py.
        #
        # Joint 3-D innovation [px, py, radial] with the full cross
        # covariance, gated at chi^2 with 3 dof; returns without usable
        # Doppler fall back to the 2-D position gate.  Off by default, so
        # existing behaviour and the bit-exact non-regression tests hold.
        self.doppler_gating = bool(doppler_gating)
        self.gate_chi2_doppler = float(gate_chi2_doppler)
        # Absolute cap on steps since the last hit, whatever the coverage
        # model decides.  Needed once a coverage model is in use: a
        # confirmed track coasting OUT of coverage then accrues no counted
        # misses at all and would otherwise live forever.  None = no cap,
        # which without a coverage model is redundant anyway (misses and
        # steps_since_hit are then the same counter).
        self.max_coast_steps = (None if max_coast_steps is None
                                else int(max_coast_steps))
        # CONFIRMATION DEADLINE: a tentative track gets exactly one full
        # confirmation window.  If it has not confirmed by the step its
        # history first fills the window (age confirm_window - 1), it is
        # deleted; a target still there is simply re-born from its next
        # detection.  That is "evaluate M-of-N once, on the first window".
        #
        # When on, the deadline is the ONLY rule that deletes a tentative —
        # max_misses then governs confirmed tracks alone.  Letting max_misses
        # also cut tentatives would reopen the clash this replaces: a small
        # max_misses (e.g. the coverage-aware in-view budget of 3) would
        # delete tentatives before their window ends, silently turning
        # 3-of-4 into something stricter.
        #
        # Why it replaced a separate consecutive-miss budget for tentatives
        # (docs/tracking_diagnostics.md §6.1):
        # * no free parameter — the deadline IS the window, so it cannot
        #   disagree with confirm_hits/confirm_window (a budget of 0 turned
        #   3-of-4 into 3-of-3; a large one kept hopeless tentatives alive);
        # * it closes a gap a miss budget cannot: a tentative that
        #   ALTERNATES hit and miss never gathers enough hits to confirm
        #   and never chains enough misses to die, so under any budget >= 1
        #   it lived indefinitely — measured 0.23 such tentatives per step
        #   at clutter 0.2, 1.2 at 0.5; zero with the deadline.
        # Off by default: tentatives then fall under max_misses like every
        # other track, the original behaviour.
        self.confirm_deadline = bool(confirm_deadline)
        # KINEMATIC REACH GATE: a return can join a track only if the target
        # could physically have got there since the track's last hit —
        #     |z - last_hit_pos| <= max_target_speed * k * dt
        #                           + reach_k_sigma * sqrt(sigma_z^2 + sd_hit^2)
        # with k the scans since that hit, sigma_z the return's position sd
        # and sd_hit the track's position sd right after the hit.  Applied
        # ON TOP of the chi^2 gate, never instead of it.
        #
        # Why: the chi^2 gate is only as tight as the predicted covariance,
        # and that grows much faster than any target can move — the CV model
        # has sd 7 m after 3 coasted scans (gate ~22 m) for a red that can
        # have moved at most ~4 m.  At clutter 0.2 a coasting track caught a
        # false plot inside that gate, the hit reset its misses and shrank
        # its covariance, and it lived on clutter alone: 16% of confirmed CV
        # tracks with sd <= 10 m were more than 40 m from their target
        # (learned model 5%), and none at clutter 0.  Such a track looks
        # CONFIDENT, so no sd threshold can filter it
        # (docs/tracking_diagnostics.md §6.3).
        #
        # max_target_speed is a SPEED, not a per-axis bound: PursuitEnv caps
        # velocity axis-wise at v_max, so a red reaches sqrt(2) * v_max on a
        # diagonal.  None disables the gate (the original behaviour).
        if max_target_speed is not None and max_target_speed <= 0.0:
            raise ValueError("max_target_speed must be positive or None")
        self.max_target_speed = (None if max_target_speed is None
                                 else float(max_target_speed))
        self.reach_k_sigma = float(reach_k_sigma)
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
        # Used ONLY when motion_model is None -- see the SCOPE note on
        # sigma_a above.
        self.Q = _dwna_Q(self.dt, self.sigma_a)
        self.H_pos = np.zeros((2, 4)); self.H_pos[0, 0] = self.H_pos[1, 1] = 1.0

        self.tracks: List[Track] = []
        self.t = 0
        self.last_nis: List[float] = []      # NIS of this step's updates
        self._oracle_map: Dict[int, int] = {}   # truth_id -> track id

    # ---------------- Kalman primitive (per component) ---------------- #

    @staticmethod
    def _update(x, P, H, R, z):
        """Joseph-form Kalman update for ONE component — see
        ``isr.tracking.kalman.joseph_update`` (shared with the obstacle
        tracker; this is a thin, byte-identical wrapper kept so every
        internal call site here stays unchanged).

        Also returns the innovation's log-likelihood, needed to
        Bayes-reweight a track's components (Note 5 above).  For a
        single-component track this is computed but never changes x/P —
        the softmax reweight of one term is `exp(0)/exp(0) = 1` — so it is
        numerically inert for the default (non-branching) motion model.
        """
        return joseph_update(x, P, H, R, z)

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

    def _predict_all(self) -> None:
        """PREDICT every track.

        A motion model that also exposes ``predict_batch(xs, Ps)`` — one
        list of branches per ``(x, P)``, in order — is asked about every
        component of every track in a SINGLE call.  The branches, their
        order and the per-track reduce are exactly those of the one-call-
        per-component path below; only the number of calls changes.  That
        matters for the learned model, where each individual call costs
        far more in per-sample overhead than in arithmetic.
        """
        batch = getattr(self.motion_model, "predict_batch", None)
        if batch is None or not self.tracks:
            for tr in self.tracks:
                self._predict_track(tr)
            return
        owner, xs, Ps = [], [], []
        for ti, tr in enumerate(self.tracks):
            for c in tr.components:
                owner.append((ti, c.w))
                xs.append(c.x)
                Ps.append(c.P)
        branch_lists = batch(xs, Ps)
        if len(branch_lists) != len(xs):
            raise RuntimeError(
                f"predict_batch returned {len(branch_lists)} branch lists "
                f"for {len(xs)} components")
        children: List[List[_Component]] = [[] for _ in self.tracks]
        for (ti, w_parent), branches in zip(owner, branch_lists):
            for rel_w, xb, Pb in branches:
                children[ti].append(_Component(w_parent * rel_w, xb, Pb))
        for tr, comps in zip(self.tracks, children):
            tr.components = self._reduce(comps)

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
        return, evaluated at the prediction — on POSITION, or on position +
        DOPPLER when ``doppler_gating`` is on (see ``_gate_measurement``).
        Takes the BEST (minimum cost) match over the track's components, so
        a multimodal track is gated in if ANY of its hypotheses is
        consistent with the return.  With gating off and a single component
        the result is identical to the pre-Gaussian-Sum implementation.

        Costs of different dimension (2-D vs 3-D) are not comparable, but
        within one blue's Hungarian problem every return has the same
        Doppler availability in practice (a line of sight, and one shared
        ``sensor_vel_noise_std``), so the columns stay homogeneous.
        """
        H, R, z = self._gate_measurement(det)
        best_d2, best_cost = None, None
        for c in tr.components:
            y = z - H @ c.x
            S = H @ c.P @ H.T + R
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

    def _gate_measurement(self, det: Dict):
        """(H, R, z) for gating: position only, or position + Doppler.

        The joint form stacks the position rows with the radial row
        ``[0, 0, u_x, u_y]``, so ``H P H^T`` carries the position-velocity
        cross covariance — a track whose position and velocity errors are
        correlated is scored consistently rather than as two independent
        tests.  Doppler is used only when the return has a line of sight
        and a positive radial sigma, the same condition the update uses.
        """
        sp2 = float(det["sigma_pos"]) ** 2
        z_pos = det["z_pos"].astype(np.float64)
        u = np.asarray(det["los"], dtype=np.float64)
        if (self.doppler_gating and np.linalg.norm(u) > 1e-6
                and det["sigma_radial"] > 0.0):
            H = np.zeros((3, 4))
            H[0, 0] = H[1, 1] = 1.0
            H[2, 2:] = u
            R = np.diag([sp2, sp2, float(det["sigma_radial"]) ** 2])
            z = np.array([z_pos[0], z_pos[1], float(det["z_radial"])])
            return H, R, z
        return self.H_pos, np.eye(2) * sp2, z_pos

    def _within_reach(self, tr: Track, det: Dict) -> bool:
        """Kinematic reach gate — see ``max_target_speed`` in __init__.
        Called during association, BEFORE this scan's hits are booked, so
        ``steps_since_hit + 1`` is the number of scans since the last hit."""
        if self.max_target_speed is None:
            return True
        k = tr.steps_since_hit + 1
        margin = self.reach_k_sigma * float(np.hypot(det["sigma_pos"],
                                                     tr.last_hit_sd))
        reach = self.max_target_speed * k * self.dt + margin
        dist = float(np.linalg.norm(
            det["z_pos"].astype(np.float64) - tr.last_hit_pos))
        return dist <= reach

    def _gate_threshold(self, det: Dict) -> float:
        """chi^2 threshold matching the dimension ``_gate_measurement``
        chose for this return."""
        H, _, _ = self._gate_measurement(det)
        return self.gate_chi2_doppler if H.shape[0] == 3 else self.gate_chi2

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

    def step(self, detections: Sequence[Dict], coverage=None) -> None:
        """One scan.

        ``coverage`` — optional callable ``(pos (2,), P_pos (2,2)) -> bool``
        answering "would a target here certainly have been in some sensor's
        view this scan?" (see ``isr.tracking.coverage``).  When given, a
        CONFIRMED track's miss counts toward death only if the answer is
        yes; a miss where nobody could have seen it says nothing about
        whether the track exists.  None keeps the original behaviour, in
        which every miss counts.
        """
        self.t += 1
        self.last_nis = []

        # 1. PREDICT (per component, then reduce) -------------------------
        self._predict_all()

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
                thresh = [self._gate_threshold(dets[k]) for k in idxs]
                for i, tr in enumerate(self.tracks):
                    for j, k in enumerate(idxs):
                        d2, c = self._gate_pos(tr, dets[k])
                        cost[i, j] = c
                        gate[i, j] = (d2 <= thresh[j]
                                      and self._within_reach(tr, dets[k]))
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
            if tr.born_at == self.t or i in hit_tracks:   # born or hit
                tr.history.append(True)
                tr.misses = 0
                tr.steps_since_hit = 0
                tr.last_hit_pos = tr.pos
                tr.last_hit_sd = _max_sd(tr.P[:2, :2])
            else:
                tr.history.append(False)
                tr.steps_since_hit += 1
                # A confirmed track's miss counts only where it should have
                # been seen.  Tentative tracks always count, so that without
                # the deadline one that left coverage right after birth
                # still dies under max_misses instead of freezing.
                if (coverage is None or not tr.confirmed
                        or coverage(tr.pos, tr.P[:2, :2])):
                    tr.misses += 1
            tr.history = tr.history[-self.confirm_window:]
            if not tr.confirmed and tr.hits >= self.confirm_hits:
                tr.confirmed = True
            # Promotion is checked first, so the step that completes the
            # first window is still a tentative's last chance to confirm.
            if tr.confirmed:
                alive = tr.misses <= self.max_misses
            elif self.confirm_deadline:
                # The deadline alone — see __init__ for why max_misses must
                # not also cut tentatives.
                alive = self.t - tr.born_at < self.confirm_window - 1
            else:
                alive = tr.misses <= self.max_misses
            if (self.max_coast_steps is not None
                    and tr.steps_since_hit > self.max_coast_steps):
                alive = False
            if alive:
                survivors.append(tr)
        self.tracks = survivors

    # ---------------- readout ---------------- #

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]

    def reset(self) -> None:
        self.tracks = []
        self.t = 0
