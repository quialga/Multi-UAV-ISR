"""
isr/agents/learned_red_motion.py — the trained red-motion model wired into
the Gaussian-Sum tracker's ``motion_model`` plug point (docs Sec. 8.5).

The tracker calls ``motion_model(x, P)`` once per COMPONENT and expects
``[(rel_weight, x_pred, P_pred), ...]``.  This adapter turns the network's
joint categorical over (heading, magnitude) into exactly that: the cells
holding most of the mass each become one branch, carrying the acceleration
that cell represents.

Why this can work at all — the asymmetry that makes it legitimate:
``run_from_nearest_uav`` reads BLUE positions and OBSTACLE geometry.  Blue
positions are our OWN drones, known exactly at inference; obstacles come
from the separate obstacle tracker.  So the adversary's inputs are
observable to us even though its action is not.  Only the red's own state
is uncertain, and that uncertainty is already carried by the mixture the
tracker maintains.

NO INPUT AVERAGING, deliberately.  An earlier plan sampled M hypotheses
from ``N(x, P)`` and marginalised the predictions.  That double-counts:
each tracker COMPONENT already is one hypothesis about where the red is,
and the mixture over components is precisely ``P(s_t | O_t)``.  The right
question per component is the pointwise one the network was trained on —
"if the red were exactly HERE, what would it do?" — so the featuriser runs
at the component's own mean and the tracker's own machinery does the
marginalising.

TRAIN/SERVE SKEW is the obvious failure mode for a module like this, so it
is designed out rather than tested for: this file builds the collector's
field dict and calls the SAME ``featurize_shard`` the training path uses.
There is no second copy of the normalisation conventions to drift.
``tests/test_learned_red_motion.py`` pins the equivalence anyway.

The residual per-branch covariance is NOT yet tuned.  ``sigma_a_model``
below is a placeholder standing in for the network's own error; the honest
value comes from a NEES sweep against real rollouts, exactly as
``sigma_a`` was tuned for the obstacle tracker.  Until then, treat any
downstream number from this adapter as provisional.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from isr.agents.red_motion_features import (
    N_HEADING_BINS, N_MAGNITUDE_BINS, V_NORM, ZERO_CLASS,
    bin_to_accel, featurize_shard,
)
from isr.agents.red_motion_gnn import RedMotionGNN
from isr.env.entities import RED_TARGET

_INPUT_KEYS = ("red_feats", "blue_feats", "b2r_edge_feats",
              "obs_feats", "o2r_edge_feats", "b2r_active", "o2r_active")


def load_red_motion_model(path: str, device: str = "cpu"
                          ) -> Tuple[RedMotionGNN, int, int]:
    """Rebuild a ``RedMotionGNN`` from a training checkpoint.

    Returns the model alongside the padding capacities it was trained
    with — the caller needs them, since a scenario with more blues than
    the trained capacity has to be truncated (see ``set_context``).
    """
    ck = torch.load(path, map_location=device, weights_only=True)
    model = RedMotionGNN(n_blue=ck["n_blue"], n_red=1, n_obs=ck["n_obs"],
                         d_hidden=ck["d_hidden"], n_msg_rounds=ck["msg_rounds"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, int(ck["n_blue"]), int(ck["n_obs"])


class LearnedRedMotion:
    """``motion_model`` adapter: network categorical -> Gaussian-Sum branches.

    Usage, once per tracker step::

        motion.set_context(blue_pos, blue_vel, obs_pos, obs_vel, obs_r)
        tracker.step(detections, t)

    ``set_context`` must precede every step: the prediction is conditional
    on where the blues are RIGHT NOW, and silently reusing a stale context
    would degrade the model in a way that looks like the model being bad.
    Calling the adapter without a context raises rather than falling back
    to constant velocity, for the same reason.
    """

    def __init__(
        self,
        model:            RedMotionGNN,
        blue_cap:         int,
        obs_cap:          int,
        dt:               float = 1.0,
        a_max:            float = 1.0,
        v_max:            float = RED_TARGET.v_max,
        arena_size:       float = 130.0,
        max_branches:     int = 4,
        sigma_a_model:    float = 0.35,
        device:           str = "cpu",
    ) -> None:
        self.model = model
        self.blue_cap = int(blue_cap)
        self.obs_cap = int(obs_cap)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.L = float(arena_size)
        # A cap on MODES, not on cells: every heading bin is assigned to a
        # kept peak, so raising or lowering this never discards mass, it
        # only changes how finely the modes are resolved.
        self.max_branches = int(max_branches)
        # Stands in for the NETWORK's own error, on top of the within-bin
        # quantisation spread computed per branch below.  Placeholder --
        # see the module docstring.
        self.sigma_a_model = float(sigma_a_model)
        self.device = device

        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = self.dt
        # Control/noise gain for state [px, py, vx, vy].
        #
        # dt^2, NOT the textbook dt^2/2.  PursuitEnv._integrate advances
        # the position with the NEW velocity:
        #     v' = clip(v + a*dt),   p' = p + v'*dt
        # so p' = p + v*dt + a*dt^2.  Using dt^2/2 here predicts half a
        # metre short per step at full acceleration, which compounds over
        # a coast -- the model would be blamed for an integration mismatch.
        self.G = np.array([[self.dt ** 2, 0.0],
                          [0.0, self.dt ** 2],
                          [self.dt, 0.0],
                          [0.0, self.dt]])

        # Within-bin spread of a uniform variable: width / sqrt(12).  The
        # RADIAL term is the magnitude bin's width; the TANGENTIAL term is
        # the heading bin's arc, which scales with |a| (a 10 degree bin is
        # a wider absolute spread for a large acceleration than a small
        # one), so it is applied per branch.
        self._sd_radial = (self.a_max / N_MAGNITUDE_BINS) / np.sqrt(12.0)
        self._sd_tangential_per_a = (2 * np.pi / N_HEADING_BINS) / np.sqrt(12.0)

        # Per-cell acceleration and quantisation covariance are FIXED
        # properties of the grid, so they are built once rather than per
        # component per step.
        cells = np.arange(ZERO_CLASS)
        self._cell_accel = bin_to_accel(cells) * self.a_max        # (180, 2)
        self._cell_quant = np.stack(
            [self._quantisation_cov(a) for a in self._cell_accel])  # (180,2,2)

        self._ctx: Optional[dict] = None

    # ----------------------------------------------------------------- #

    def set_context(
        self,
        blue_pos: np.ndarray,                       # (n_blue, 2) metres
        blue_vel: np.ndarray,                       # (n_blue, 2) m/s
        obs_pos:  Optional[np.ndarray] = None,      # (n_obs, 2)
        obs_vel:  Optional[np.ndarray] = None,      # (n_obs, 2)
        obs_r:    Optional[np.ndarray] = None,      # (n_obs,)
    ) -> None:
        """Record the situation every prediction this step is conditioned on.

        Blues beyond the trained capacity are truncated to the NEAREST
        ones once a query position is known (see ``_pack``), not dropped
        arbitrarily: the red policy reads only its nearest blue, so the
        nearest-k are exactly the ones that carry signal.
        """
        blue_pos = np.asarray(blue_pos, dtype=np.float64).reshape(-1, 2)
        blue_vel = np.asarray(blue_vel, dtype=np.float64).reshape(-1, 2)
        if len(blue_pos) != len(blue_vel):
            raise ValueError("blue_pos and blue_vel disagree on count")
        if len(blue_pos) == 0:
            raise ValueError("no blues in context: the red policy is "
                            "undefined with nothing to flee from")
        if obs_pos is None or len(obs_pos) == 0:
            obs_pos = np.zeros((0, 2))
            obs_vel = np.zeros((0, 2))
            obs_r = np.zeros((0,))
        else:
            obs_pos = np.asarray(obs_pos, dtype=np.float64).reshape(-1, 2)
            obs_r = np.asarray(obs_r, dtype=np.float64).reshape(-1)
            obs_vel = (np.zeros_like(obs_pos) if obs_vel is None
                      else np.asarray(obs_vel, dtype=np.float64).reshape(-1, 2))
        self._ctx = dict(blue_pos=blue_pos, blue_vel=blue_vel,
                        obs_pos=obs_pos, obs_vel=obs_vel, obs_r=obs_r)

    # ----------------------------------------------------------------- #

    def _pack(self, red_pos: np.ndarray, red_vel: np.ndarray) -> dict:
        """Build ONE collector-shaped sample, ready for ``featurize_shard``.

        Field names, normalisers and padding conventions are the
        collector's; the featuriser then applies exactly the transform it
        applies during training.
        """
        c = self._ctx
        L, cap_b, cap_o = self.L, self.blue_cap, self.obs_cap

        bp, bv = c["blue_pos"], c["blue_vel"]
        if len(bp) > cap_b:                       # keep the NEAREST cap_b
            keep = np.argsort(np.linalg.norm(bp - red_pos, axis=1))[:cap_b]
            bp, bv = bp[keep], bv[keep]
        n_blue = len(bp)

        op, ov, orad = c["obs_pos"], c["obs_vel"], c["obs_r"]
        if len(op) > cap_o:                       # nearest by SURFACE distance
            keep = np.argsort(
                np.linalg.norm(op - red_pos, axis=1) - orad)[:cap_o]
            op, ov, orad = op[keep], ov[keep], orad[keep]
        n_obs = len(op)

        blue_rel_pos = np.zeros((1, cap_b, 2), dtype=np.float32)
        blue_rel_vel = np.zeros((1, cap_b, 2), dtype=np.float32)
        blue_rel_pos[0, :n_blue] = (bp - red_pos) / L
        blue_rel_vel[0, :n_blue] = bv / V_NORM

        obs_rel_pos = np.zeros((1, cap_o, 2), dtype=np.float32)
        obs_rel_vel = np.zeros((1, cap_o, 2), dtype=np.float32)
        obs_radius = np.zeros((1, cap_o), dtype=np.float32)
        obs_mask = np.zeros((1, cap_o), dtype=bool)
        if n_obs:
            obs_rel_pos[0, :n_obs] = (op - red_pos) / L
            obs_rel_vel[0, :n_obs] = ov / V_NORM
            obs_radius[0, :n_obs] = orad / L
            obs_mask[0, :n_obs] = True

        return dict(
            accel=np.zeros((1, 2), dtype=np.float32),     # unused at inference
            red_pos=red_pos.astype(np.float32)[None, :],
            red_vel=red_vel.astype(np.float32)[None, :],
            blue_rel_pos=blue_rel_pos, blue_rel_vel=blue_rel_vel,
            obs_rel_pos=obs_rel_pos, obs_rel_vel=obs_rel_vel,
            obs_radius=obs_radius, obs_mask=obs_mask,
            wall_dist=np.array([[red_pos[0], L - red_pos[0],
                                red_pos[1], L - red_pos[1]]],
                              dtype=np.float32) / L,
            n_blue=np.array([n_blue]), n_obs_placed=np.array([n_obs]),
            episode_id=np.array([0]),
        )

    def predict_probs(self, red_pos: np.ndarray, red_vel: np.ndarray
                      ) -> np.ndarray:
        """The network's full categorical at one exact state, shape (N_BINS,)."""
        if self._ctx is None:
            raise RuntimeError(
                "set_context() must be called before predicting: the red's "
                "action depends on where the blues are this step")
        f = featurize_shard(self._pack(np.asarray(red_pos, dtype=np.float64),
                                      np.asarray(red_vel, dtype=np.float64)),
                           arena_size=self.L)
        with torch.no_grad():
            logits = self.model(*[torch.from_numpy(f[k]) for k in _INPUT_KEYS])
            return torch.softmax(logits[0, 0], dim=-1).numpy().astype(np.float64)

    # ----------------------------------------------------------------- #

    def _quantisation_cov(self, accel: np.ndarray) -> np.ndarray:
        """Spread of the true acceleration WITHIN one cell, in world axes.

        A cell names a region, not a point.  Anisotropic and aligned with
        the acceleration: radial spread set by the magnitude bin's width,
        tangential by the heading arc.  Which axis is coarser is a fact
        about the grid, not a guess — on 36x5 the magnitude bin (0.2) beats
        the arc at |a| = a_max (2*pi/36 = 0.175), and the tangential term
        shrinks with |a| while the radial does not, so RADIAL dominates
        throughout.
        """
        mag = float(np.linalg.norm(accel))
        if mag < 1e-9:                       # ZERO class: no cell geometry
            return np.zeros((2, 2))
        u = accel / mag                                  # radial unit
        t = np.array([-u[1], u[0]])                      # tangential unit
        return (self._sd_radial ** 2 * np.outer(u, u)
               + (mag * self._sd_tangential_per_a) ** 2 * np.outer(t, t))

    def _branch_cov(self, accel: np.ndarray) -> np.ndarray:
        """Single-cell residual covariance: quantisation + model error."""
        return (self._quantisation_cov(accel)
               + self.sigma_a_model ** 2 * np.eye(2))

    # ----------------------------------------------------------------- #

    def _basins(self, probs: np.ndarray) -> List[Tuple[float, np.ndarray,
                                                      np.ndarray]]:
        """Split the categorical into MODES, keeping ALL of the mass.

        Two things make the obvious "take the top-k cells" wrong, and both
        were measured rather than reasoned about:

        * Training uses SOFT LABELS that deliberately smear mass onto
          neighbouring bins, so the k most probable cells are almost always
          ONE mode sampled k times, not k modes.  Fed to the tracker they
          are near-duplicates, the merge gate correctly folds them into a
          single component, and the mixture machinery does nothing at all
          (measured: mean 1.01 components per track).
        * With 181 classes and a peak around 7%, the top 4 cells hold only
          ~20% of the mass.  Truncating there discards the tails, so the
          branch covariance reflects a far narrower distribution than the
          model actually predicted — measured NEES 6.6 against a target of
          4.0, i.e. badly overconfident, which is the direction that kills
          tracks.

        So: find local maxima of the HEADING marginal, keep the strongest
        ``max_branches`` of them, and assign EVERY heading bin to its
        nearest kept peak.  No mass is discarded; each branch is a genuine
        mode carrying its own basin's weight, mean and spread.

        Heading, not the joint grid, defines the modes: the multimodality
        this adversary actually produces is a left/right commitment, which
        is a split in DIRECTION.  Magnitude is ordinal and unimodal within
        a direction, so splitting on it would manufacture branches that
        differ only in effort — exactly the near-duplicates to avoid.
        """
        grid = probs[:ZERO_CLASS].reshape(N_HEADING_BINS, N_MAGNITUDE_BINS)
        ph = grid.sum(axis=1)                       # heading marginal

        prev, nxt = np.roll(ph, 1), np.roll(ph, -1)     # circular neighbours
        peaks = np.nonzero((ph >= prev) & (ph >= nxt))[0]
        if len(peaks) == 0:                             # perfectly flat
            peaks = np.array([int(ph.argmax())])
        peaks = peaks[np.argsort(-ph[peaks])][:self.max_branches]

        # Circular distance from every heading bin to every kept peak.
        d = np.abs(np.arange(N_HEADING_BINS)[:, None] - peaks[None, :])
        d = np.minimum(d, N_HEADING_BINS - d)
        owner = d.argmin(axis=1)                        # (36,) kept-peak index

        out = []
        for k in range(len(peaks)):
            rows = np.nonzero(owner == k)[0]
            w_cells = grid[rows].reshape(-1)            # (len(rows)*5,)
            w = float(w_cells.sum())
            if w <= 0.0:
                continue
            idx = (rows[:, None] * N_MAGNITUDE_BINS
                  + np.arange(N_MAGNITUDE_BINS)[None, :]).reshape(-1)
            a = self._cell_accel[idx]                   # (n, 2)
            p = w_cells / w
            mu = p @ a
            dev = a - mu
            # Mixture covariance over the basin: the spread of the cell
            # means PLUS each cell's own within-cell spread, then the
            # model's isotropic error.  Dropping the first term is exactly
            # the overconfidence the top-k version suffered.
            cov = (dev * p[:, None]).T @ dev
            cov = cov + np.einsum("n,nij->ij", p, self._cell_quant[idx])
            out.append((w, mu, cov + self.sigma_a_model ** 2 * np.eye(2)))

        p_zero = float(probs[ZERO_CLASS])
        if p_zero > 0.0:
            # Genuinely discrete: "do not accelerate" is not a direction,
            # so it cannot belong to any heading basin.
            out.append((p_zero, np.zeros(2),
                       self.sigma_a_model ** 2 * np.eye(2)))
        return out

    def _advance(self, x: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """Advance a state exactly as ``PursuitEnv._integrate`` would.

        The velocity cap is AXIS-WISE (matching the env, which clips the
        components, not the magnitude) and is what stops a coasting track
        accelerating without bound: applied every step, an uncapped unit
        acceleration reaches 80 m/s over an 80-step coast and throws the
        estimate a thousand metres outside a 130 m arena.  Measured
        exactly that before the cap was added.

        Arena clipping is deliberately NOT applied.  It is a hard
        non-linearity that would bias the mixture at the boundary, and a
        track whose mean has left the arena is already lost by any useful
        tolerance.
        """
        v = np.clip(x[2:] + accel * self.dt, -self.v_max, self.v_max)
        return np.concatenate([x[:2] + v * self.dt, v])

    def __call__(self, x: np.ndarray, P: np.ndarray
                 ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """One Gaussian-Sum branch per MODE of the predicted categorical."""
        probs = self.predict_probs(x[:2], x[2:])
        FPFt = self.F @ P @ self.F.T
        out: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for w, accel, cov_a in self._basins(probs):
            # Mean through the env's exact (clipped) kinematics; covariance
            # through the linear gain.  Where the cap binds, the true
            # spread is SMALLER than the linear one, so this errs
            # conservative -- the safe direction for a gate.
            x_pred = self._advance(x, accel)
            P_pred = FPFt + self.G @ cov_a @ self.G.T
            out.append((float(w), x_pred, P_pred))
        return out
