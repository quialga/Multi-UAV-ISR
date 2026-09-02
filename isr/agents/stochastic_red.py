"""
isr/agents/stochastic_red.py — a scripted evader with REALISTIC randomness.

``run_from_nearest_uav`` is deterministic: given the state, its acceleration
is a function.  That is convenient but not what a real adversary does, and
it distorts everything downstream — a predictor trained against it has no
aleatoric uncertainty to represent, so a mixture model has nothing to earn.

Two design constraints, both measured rather than assumed:

1. **The noise must be TEMPORALLY CORRELATED, not iid.**  One step of red
   motion is 1 m = 0.20 belief cells, and the acceleration-induced spread is
   0.5 m = 0.10 cells.  Per-step iid jitter is therefore SUB-CELL: the
   belief map cannot represent it and any predictor averages it straight
   out.  It would add variance to the training target while changing no
   behaviour worth predicting.  What is visible, and what a real vehicle
   actually does, is drift that PERSISTS over several steps.
2. **Some of it must be COMMITTED and DISCRETE.**  Rounding an obstacle to
   the left or to the right are two distinct plans, not two samples from a
   blob around "straight at it".  Averaging them gives the one heading the
   evader would never fly.  A committed side choice is what produces
   genuine bimodality — and bimodality is the whole reason a mixture (or a
   particle) representation beats a single Gaussian.

The deterministic policy already has lag-1 acceleration autocorrelation
rho = 0.546 (it tracks smoothly-moving blues), so the added noise is shaped
to preserve that structure rather than whiten it.

Three independent, separately-tunable sources:

* ``heading_noise_std`` / ``heading_rho`` — an AR(1) (Ornstein-Uhlenbeck)
  perturbation of the chosen heading.  Continuous "imprecision" with a
  tunable correlation time.  rho = 0 recovers iid (deliberately available,
  to demonstrate it is invisible); rho -> 1 is a near-constant per-episode
  bias.
* ``commit_prob`` / ``commit_steps`` — when inside an obstacle's influence
  band, with probability ``commit_prob`` the evader picks a SIDE and holds
  it for ``commit_steps``, deflecting its heading tangentially instead of
  radially.  This is the bimodal source.
* ``speed_jitter`` — multiplicative noise on the acceleration magnitude,
  breaking the "always exactly max effort" artefact.

All default to 0 / off, so with no arguments this is byte-identical to
``run_from_nearest_uav``.

Usage::

    red = StochasticRed(heading_noise_std=0.35, commit_prob=0.5)
    env = PursuitEnv(..., red_policy=red)

The env calls ``reset(n_red)`` at episode start when the policy exposes it
(duck-typed), so the per-red state does not leak across episodes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from isr.env.pursuit_env import run_from_nearest_uav

INFLUENCE = 12.0        # must match run_from_nearest_uav's influence band


def _rotate(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]],
                    dtype=np.float32)


class StochasticRed:
    """``run_from_nearest_uav`` plus correlated, partly committed randomness."""

    def __init__(
        self,
        heading_noise_std: float = 0.0,
        heading_rho:       float = 0.85,
        commit_prob:       float = 0.0,
        commit_steps:      int   = 8,
        commit_angle:      float = 1.1,      # radians of tangential deflection
        speed_jitter:      float = 0.0,
        seed:              Optional[int] = None,
    ) -> None:
        self.heading_noise_std = float(heading_noise_std)
        self.heading_rho       = float(heading_rho)
        self.commit_prob       = float(commit_prob)
        self.commit_steps      = int(commit_steps)
        self.commit_angle      = float(commit_angle)
        self.speed_jitter      = float(speed_jitter)
        self._rng = np.random.default_rng(seed)
        self._phase: Optional[np.ndarray] = None    # AR(1) heading offset
        self._side:  Optional[np.ndarray] = None    # -1 / +1 / 0 (uncommitted)
        self._left:  Optional[np.ndarray] = None    # steps left on the commit

    # ------------------------------------------------------------------ #

    def reset(self, n_red: int, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._phase = np.zeros(n_red, dtype=np.float64)
        self._side = np.zeros(n_red, dtype=np.int8)
        self._left = np.zeros(n_red, dtype=np.int32)

    def _ensure(self, n_red: int) -> None:
        if self._phase is None or len(self._phase) != n_red:
            self.reset(n_red)

    # ------------------------------------------------------------------ #

    def __call__(self, blue_pos, red_pos, red_active,
                 obstacle_pos=None, obstacle_r=None, arena_size=None):
        a = run_from_nearest_uav(blue_pos, red_pos, red_active,
                                 obstacle_pos, obstacle_r, arena_size)
        n_red = a.shape[0]
        self._ensure(n_red)

        has_obs = (obstacle_pos is not None and len(obstacle_pos) > 0
                   and obstacle_r is not None)

        for i in range(n_red):
            if not red_active[i]:
                self._side[i] = 0
                self._left[i] = 0
                continue
            v = a[i]
            if float(np.linalg.norm(v)) < 1e-8:
                continue

            # --- committed side choice (the BIMODAL source) --------------
            if self.commit_prob > 0.0 and has_obs:
                if self._left[i] > 0:
                    self._left[i] -= 1
                else:
                    surf = (np.linalg.norm(obstacle_pos - red_pos[i], axis=1)
                            - obstacle_r)
                    if np.any(surf < INFLUENCE):
                        # Entering an influence band: commit to a side, or
                        # explicitly decline to (so it is not deterministic
                        # that a band always triggers a deflection).
                        if self._rng.random() < self.commit_prob:
                            self._side[i] = 1 if self._rng.random() < 0.5 else -1
                            self._left[i] = self.commit_steps
                        else:
                            self._side[i] = 0
                    else:
                        self._side[i] = 0
                if self._left[i] > 0 and self._side[i] != 0:
                    v = _rotate(v, self.commit_angle * float(self._side[i]))

            # --- AR(1) heading drift (correlated "imprecision") ----------
            if self.heading_noise_std > 0.0:
                rho = np.clip(self.heading_rho, 0.0, 0.999)
                # Stationary AR(1): the innovation is scaled by sqrt(1-rho^2)
                # so the marginal std stays heading_noise_std whatever rho is.
                # Without that factor, changing the correlation would also
                # change the amount of noise, confounding the two knobs.
                self._phase[i] = (rho * self._phase[i]
                                  + np.sqrt(1.0 - rho ** 2)
                                  * self._rng.normal(0.0, self.heading_noise_std))
                v = _rotate(v, float(self._phase[i]))

            # --- magnitude jitter ---------------------------------------
            if self.speed_jitter > 0.0:
                v = v * float(np.clip(
                    1.0 + self._rng.normal(0.0, self.speed_jitter), 0.0, 1.0))

            a[i] = v
        return np.clip(a, -1.0, 1.0).astype(np.float32)
