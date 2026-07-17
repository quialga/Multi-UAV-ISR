"""
isr/env/pursuit_env.py — Stage 1 pursuit-evasion environment.

PettingZoo ParallelEnv.  ``N_blue`` UAVs (the agents) chase ``N_red``
targets (env-internal, scripted policy in Stage 1) inside a 2D
continuous arena.

This is the regression bedrock for the curriculum — the simplest
possible MARL pursuit task with continuous actions.  Every later
stage swaps one thing at a time while keeping the rest constant.

See ``docs/design.md §3`` for the full spec (arena, kinematics,
observation, action, reward, hyperparameters, acceptance criterion).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from isr.env.entities import BLUE_UAV, RED_TARGET


# ---------------------------------------------------------------------------
# Default Stage 1 red policy: every red flees its nearest blue UAV.
# ---------------------------------------------------------------------------

def run_from_nearest_uav(
    blue_pos:   np.ndarray,   # (N_blue, 2)
    red_pos:    np.ndarray,   # (N_red,  2)
    red_active: np.ndarray,   # (N_red,) bool
) -> np.ndarray:
    """
    Scripted red policy used by ``PursuitEnv`` in Stage 1.

    Each *active* red target computes the unit vector pointing **away**
    from its nearest blue UAV and uses that as its acceleration command
    (magnitude exactly 1.0 — the policy is "always run at max
    effort").  Caught reds (``active=False``) return zero acceleration.

    The function is pure (no hidden state) so it's safe to swap out for
    a learned policy in Stage 4 by passing a different callable to the
    env's ``red_policy`` constructor argument.

    Returns
    -------
    (N_red, 2) float32 — per-red acceleration commands.
    """
    n_red = red_pos.shape[0]
    out = np.zeros((n_red, 2), dtype=np.float32)
    for i in range(n_red):
        if not red_active[i]:
            continue
        diffs = red_pos[i] - blue_pos              # vectors blue -> this red
        dists = np.linalg.norm(diffs, axis=1)
        nearest = int(np.argmin(dists))
        d_vec = diffs[nearest]
        norm = float(np.linalg.norm(d_vec))
        if norm > 1e-8:
            out[i] = d_vec / norm                  # unit vector AWAY from nearest blue
    return out


# ---------------------------------------------------------------------------
# Stage 1 environment
# ---------------------------------------------------------------------------

class PursuitEnv(ParallelEnv):
    """
    Stage 1 pursuit-evasion environment (full observability, scripted red).

    Parameters
    ----------
    n_blue          Number of blue UAVs (PettingZoo agents).
    n_red           Number of red targets (env-internal in Stage 1).
    arena_size      Side length of the square arena (units).
    max_steps       Episode horizon — truncate after this many steps.
    capture_radius  A red is caught when within this Euclidean distance
                    of any blue UAV.
    dt              Simulation step size (units of time).
    red_policy      Callable ``(blue_pos, red_pos, red_active) -> (N_red, 2)``
                    returning per-red acceleration commands.  Defaults to
                    ``run_from_nearest_uav``.  Stage 4 will pass a learned
                    policy here without further env changes.
    seed            Initial RNG seed.

    Observation (per blue UAV, v2)
    ------------------------------
    Ego-centric world-frame representation motivated by TERL (see
    ``docs/stage1_analysis.md``).  All positional features are
    relative to this UAV's own position; all velocities are relative
    to this UAV's own velocity; each entity carries both Cartesian
    (rel_pos, rel_vel) and polar (range, bearing) auxiliary features.
    Every blue's obs is structurally identical from its own
    perspective — no ``self_idx_onehot`` needed.  See
    ``_build_obs`` docstring for the exact layout.  Total dim for
    N_blue=3, N_red=2: 38.

    Action (per blue UAV)
    ---------------------
    Continuous 2D acceleration command in [-1, 1]^2.

    Reward
    ------
    Shared team reward, computed once per step and given to every blue
    UAV.  See ``docs/design.md §3.7``.
    """

    metadata = {"render_modes": ["human"], "name": "pursuit_v0"}

    def __init__(
        self,
        n_blue:                   int                = 3,
        n_red:                    int                = 2,
        arena_size:               float              = 100.0,
        max_steps:                int                = 200,
        capture_radius:           float              = 3.0,
        dt:                       float              = 1.0,
        red_policy:               Optional[Callable] = None,
        seed:                     Optional[int]      = None,
        sensor_radius:            Optional[float]    = None,
        # ---- Stage 4: obstacles + belief map (all optional) --------------
        n_obstacles:              int   = 0,
        obstacle_radius_min:      float = 5.0,
        obstacle_radius_max:      float = 15.0,
        obstacle_spawn_clearance: float = 10.0,
        use_belief_maps:          bool  = False,
        belief_grid_size:         int   = 26,
        belief_channels:          int   = 2,
        belief_clip:              float = 10.0,
        p_TP:                     float = 0.85,
        p_FP:                     float = 0.15,
        ray_step_size:            float = 2.5,
        belief_window_size:       int   = 0,   # 0 = ego-centric windows disabled
    ):
        """
        sensor_radius: Stage 3 partial-observability knob.  When None
            (default), the env is fully observable (Stage 1/2 behaviour).
            When set to a positive value, each blue UAV only "sees"
            entities within this Euclidean distance.  Visibility is
            per-receiver (each blue has its own view), exposed via
            ``structured_partial_observation()`` as per-edge masks so
            the graph tensor shape stays constant across timesteps.
            The full-state ``structured_observation()`` remains
            unaffected — used by the CTDE critic.

        Stage 4 params
        --------------
        n_obstacles                Number of static circular obstacles
                                   placed at reset.  Default 0 = no
                                   obstacles (Stage 1-3 behaviour byte-
                                   preserved).
        obstacle_radius_min / _max Uniform sampling range for obstacle
                                   radii, in metres.
        obstacle_spawn_clearance   Minimum distance blues and reds must
                                   spawn from any obstacle boundary.
        use_belief_maps            When True, allocate and update per-UAV
                                   log-odds belief maps every step.
                                   Costs some compute; default False.
        belief_grid_size           H = W for the belief tensor.  With
                                   arena_size = 130 and default 26,
                                   each cell is 5 m across.
        belief_channels            Number of channels in the belief
                                   tensor.  Convention: ch 0 = enemy,
                                   ch 1 = obstacle.
        belief_clip                Absolute cap on stored log-odds.
                                   ±10 -> P in ~[5e-5, 1 - 5e-5].
        p_TP, p_FP                 Sensor detection reliability.
                                   Same values for both channels in v1.
        ray_step_size              Ray-cast step for occlusion, in metres.
        """
        super().__init__()
        self.n_blue         = int(n_blue)
        self.n_red          = int(n_red)
        self.arena_size     = float(arena_size)
        self.max_steps      = int(max_steps)
        self.capture_radius = float(capture_radius)
        self.dt             = float(dt)
        self.red_policy     = red_policy or run_from_nearest_uav
        self.sensor_radius: Optional[float] = (
            float(sensor_radius) if sensor_radius is not None else None
        )

        # ---- Stage 4 knobs ------------------------------------------------
        self.n_obstacles              = int(n_obstacles)
        self.obstacle_radius_min      = float(obstacle_radius_min)
        self.obstacle_radius_max      = float(obstacle_radius_max)
        self.obstacle_spawn_clearance = float(obstacle_spawn_clearance)
        self.use_belief_maps          = bool(use_belief_maps)
        self.belief_grid_size         = int(belief_grid_size)
        self.belief_channels          = int(belief_channels)
        self.belief_clip              = float(belief_clip)
        self.p_TP                     = float(p_TP)
        self.p_FP                     = float(p_FP)
        self.ray_step_size            = float(ray_step_size)
        self.belief_window_size       = int(belief_window_size)
        # Derived quantities.
        self.belief_cell_size = self.arena_size / max(1, self.belief_grid_size)
        # Log-odds evidence constants — same for both channels in v1.
        # Clamp to (eps, 1-eps) so a degenerate p_TP or p_FP passed by
        # tests / users doesn't blow up log arithmetic.  The tiny
        # clamp does not change behaviour in normal operating ranges.
        _eps = 1e-6
        p_tp_c = min(max(self.p_TP, _eps), 1.0 - _eps)
        p_fp_c = min(max(self.p_FP, _eps), 1.0 - _eps)
        self._L_detect    = float(np.log(p_tp_c / p_fp_c))
        self._L_no_detect = float(np.log((1.0 - p_tp_c) / (1.0 - p_fp_c)))

        self._rng = np.random.default_rng(seed)

        # PettingZoo agent set: blue UAVs only.  Red is scripted inside
        # the env in Stage 1; Stage 4 will refactor or override
        # ``red_policy`` without touching the agent set.
        self.possible_agents = [f"blue_{i}" for i in range(self.n_blue)]
        self.agents = list(self.possible_agents)

        # Observation dim — see the ``_build_obs`` docstring for the
        # v2 layout (ego-centric world-frame relative + polar auxiliary
        # features per entity, plus wall distances and speed).
        # v2 total = 8 * n_red + 7 * (n_blue - 1) + 8
        # (per-red: rel_pos 2 + rel_vel 2 + range 1 + bearing_cs 2 + active 1)
        # (per-teammate: rel_pos 2 + rel_vel 2 + range 1 + bearing_cs 2)
        # (self+global: vel 2 + speed 1 + walls 4 + time 1)
        self._obs_dim = (
            8 * self.n_red
            + 7 * (self.n_blue - 1)
            + 2 + 1 + 4 + 1
        )

        self._action_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,),
                          dtype=np.float32)
            for a in self.possible_agents
        }

        # ---- Stage 2 graph dims + fixed edge index tables ------------
        # The structured obs used by the GNN policy is not per-agent —
        # the graph is the same at each timestep; only the "acting
        # agent" interpretation differs.  We expose the graph as a
        # dict of arrays via ``structured_observation()``.  See
        # ``_build_structured_obs`` docstring for the layout.
        self.blue_feat_dim: int = 8
        self.red_feat_dim:  int = 1
        self.edge_feat_dim: int = 7
        # Fixed edge indices (blue-blue bidirectional, red-blue directed).
        self.bb_edge_src, self.bb_edge_dst = self._build_bb_edge_indices()
        self.rb_edge_src, self.rb_edge_dst = self._build_rb_edge_indices()
        self.n_bb_edges = len(self.bb_edge_src)   # N_blue * (N_blue - 1)
        self.n_rb_edges = len(self.rb_edge_src)   # N_red * N_blue

        # Mutable state — initialised in reset().
        self._blue_pos:   Optional[np.ndarray] = None
        self._blue_vel:   Optional[np.ndarray] = None
        self._red_pos:    Optional[np.ndarray] = None
        self._red_vel:    Optional[np.ndarray] = None
        self._red_active: Optional[np.ndarray] = None
        self._t:          int                  = 0
        # Diagnostic captured from the last step (rendering / logging).
        self._last_n_caught: int = 0

        # ---- Stage 4 mutable state (obstacles + belief) ------------------
        # ``_obstacle_pos``  (n_obs, 2)   float32 -- centres
        # ``_obstacle_r``    (n_obs,)     float32 -- radii
        # ``_belief_maps``   (n_blue, C, H, W) float32 -- per-UAV log-odds
        # All None when Stage 4 features are disabled.
        self._obstacle_pos: Optional[np.ndarray] = None
        self._obstacle_r:   Optional[np.ndarray] = None
        self._belief_maps:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  PettingZoo API                                                     #
    # ------------------------------------------------------------------ #

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        L = self.arena_size

        # Stage 4: place obstacles BEFORE sampling entity positions so
        # spawn clearance can reject bad draws.
        self._place_obstacles()

        # Uniform random initial positions with obstacle clearance.
        # No minimum-separation constraint between blue and red in
        # Stage 1 — caught-at-spawn is rare given typical L/r ratios
        # and the policy will learn to handle it.  We can add a
        # min-separation constraint later if it matters empirically.
        self._blue_pos = self._sample_free_positions(self.n_blue)
        self._blue_vel = np.zeros((self.n_blue, 2), dtype=np.float32)
        self._red_pos  = self._sample_free_positions(self.n_red)
        self._red_vel  = np.zeros((self.n_red, 2), dtype=np.float32)
        self._red_active = np.ones(self.n_red, dtype=bool)
        self._t = 0
        self._last_n_caught = 0
        self.agents = list(self.possible_agents)

        # Stage 4: zero belief maps.
        if self.use_belief_maps:
            self._belief_maps = np.zeros(
                (self.n_blue, self.belief_channels,
                 self.belief_grid_size, self.belief_grid_size),
                dtype=np.float32,
            )
            # First observation update happens before returning obs so
            # the initial belief map reflects step-0 observations.
            self._update_belief_maps()

        obs = {a: self._build_obs(i) for i, a in enumerate(self.possible_agents)}
        info = {a: {} for a in self.possible_agents}
        return obs, info

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],   # obs
        Dict[str, float],        # rewards
        Dict[str, bool],         # terminations
        Dict[str, bool],         # truncations
        Dict[str, Dict],         # infos
    ]:
        # 1. Pack blue actions into a (N_blue, 2) array, clipping to the
        #    action box.  Missing entries default to zeros (defensive —
        #    PettingZoo expects every alive agent to act each step).
        blue_a = np.zeros((self.n_blue, 2), dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                blue_a[i] = np.clip(
                    np.asarray(actions[agent], dtype=np.float32),
                    -1.0, 1.0,
                )

        # 2. Red scripted action (only active reds will produce non-zero).
        red_a = self.red_policy(self._blue_pos, self._red_pos, self._red_active)
        red_a = np.clip(red_a.astype(np.float32), -1.0, 1.0)
        red_a[~self._red_active] = 0.0  # defensive

        # 3. Integrate blue kinematics (double-integrator with axis-wise
        #    velocity cap and arena wall stop).
        prev_blue_pos = self._blue_pos.copy()
        self._blue_pos, self._blue_vel = self._integrate(
            self._blue_pos, self._blue_vel, blue_a, BLUE_UAV.v_max,
        )
        # Stage 4: reject moves that land inside any obstacle (soft
        # crash — position rolled back, velocity zeroed; explicit crash
        # penalty is a backlog item, see docs/stage4_backlog.md §1).
        if self.n_obstacles > 0:
            self._blue_pos, self._blue_vel = self._clip_positions_from_obstacles(
                self._blue_pos, prev_blue_pos, self._blue_vel,
            )

        # 4. Integrate red kinematics.
        prev_red_pos = self._red_pos.copy()
        self._red_pos, self._red_vel = self._integrate(
            self._red_pos, self._red_vel, red_a, RED_TARGET.v_max,
        )
        if self.n_obstacles > 0:
            self._red_pos, self._red_vel = self._clip_positions_from_obstacles(
                self._red_pos, prev_red_pos, self._red_vel,
            )
        # Caught reds stay frozen at their last position.
        self._red_vel[~self._red_active] = 0.0

        # 5. Capture check — vectorised pairwise distance.
        n_caught_this_step = 0
        if self._red_active.any():
            # (N_red, N_blue) pairwise distances
            dists = np.linalg.norm(
                self._red_pos[:, None, :] - self._blue_pos[None, :, :],
                axis=-1,
            )
            min_dists = dists.min(axis=1)                       # (N_red,)
            newly_caught = self._red_active & (min_dists <= self.capture_radius)
            n_caught_this_step = int(newly_caught.sum())
            self._red_active &= ~newly_caught
            # Zero velocities of newly caught reds (cosmetic, but
            # keeps the obs clean).
            self._red_vel[newly_caught] = 0.0
        self._last_n_caught = n_caught_this_step

        # 6. Reward — see docs/design.md §3.7.
        catch_bonus = 10.0 * n_caught_this_step
        action_cost = 0.01 * float(np.sum(blue_a ** 2))   # sum over UAVs and axes
        step_cost   = 0.05
        r = catch_bonus - action_cost - step_cost

        # 7. Termination check.
        self._t += 1
        all_caught = bool(not self._red_active.any())
        time_up    = self._t >= self.max_steps
        terminated = all_caught
        truncated  = time_up and not all_caught
        if terminated or truncated:
            n_uncaught = int(self._red_active.sum())
            r += -5.0 * n_uncaught                          # terminal penalty
            self.agents = []                                # PettingZoo convention

        # Stage 4: update belief maps AFTER movement + capture so
        # observations reflect the new state.  A caught red will no
        # longer contribute to the enemy channel's ground truth this
        # step, so its cells' P(enemy) will decay from now on.
        if self.use_belief_maps:
            self._update_belief_maps()

        # 8. Pack the per-agent dicts.  Shared team reward — every
        #    blue agent gets the same r.
        rewards     = {a: float(r) for a in self.possible_agents}
        terminateds = {a: terminated for a in self.possible_agents}
        truncateds  = {a: truncated  for a in self.possible_agents}
        infos = {
            a: {
                "n_caught_this_step": n_caught_this_step,
                "n_red_remaining":    int(self._red_active.sum()),
                "t":                  self._t,
            }
            for a in self.possible_agents
        }
        obs = {a: self._build_obs(i) for i, a in enumerate(self.possible_agents)}
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        """No built-in render — see ``isr.utils.render`` for matplotlib."""
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    #  Inspector / helpers                                                #
    # ------------------------------------------------------------------ #

    def state_snapshot(self) -> Dict[str, Any]:
        """
        Read-only snapshot of full env state.  Used by the renderer and
        by tests that need to assert on positions / velocities directly.
        """
        snap = {
            "t":           self._t,
            "blue_pos":    self._blue_pos.copy(),
            "blue_vel":    self._blue_vel.copy(),
            "red_pos":     self._red_pos.copy(),
            "red_vel":     self._red_vel.copy(),
            "red_active":  self._red_active.copy(),
            "n_caught_last_step": self._last_n_caught,
        }
        # Stage 4: obstacles + belief maps (only when populated).
        if self._obstacle_pos is not None:
            snap["obstacle_pos"] = self._obstacle_pos.copy()
            snap["obstacle_r"]   = self._obstacle_r.copy()
        if self._belief_maps is not None:
            snap["belief_maps"] = self._belief_maps.copy()
        return snap

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _integrate(
        self,
        pos:    np.ndarray,
        vel:    np.ndarray,
        accel:  np.ndarray,
        v_max:  float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Double-integrator step with axis-wise velocity cap and arena
        wall stop.  Returns ``(new_pos, new_vel)``.

        Wall handling: when a position update would leave the arena,
        we clip the position to the wall and zero the velocity
        component on that axis.  No bounce — keeps reasoning simple
        and the agent learns to back off.
        """
        new_vel = np.clip(vel + accel * self.dt, -v_max, v_max).astype(np.float32)
        new_pos = pos + new_vel * self.dt
        for axis in (0, 1):
            below = new_pos[:, axis] < 0.0
            above = new_pos[:, axis] > self.arena_size
            new_pos[below, axis] = 0.0
            new_pos[above, axis] = self.arena_size
            new_vel[below | above, axis] = 0.0
        return new_pos.astype(np.float32), new_vel

    # ------------------------------------------------------------------ #
    #  Observation (v2: ego-centric world-frame relative + polar)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bearing_features(
        self_vel:    np.ndarray,     # (2,)
        rel_pos:     np.ndarray,     # (K, 2)
        ranges:      np.ndarray,     # (K,)
        eps:         float = 1e-6,
    ) -> np.ndarray:
        """
        Compute (cos θ, sin θ) per entity, where θ is the angle between
        the acting UAV's velocity vector and the world-frame relative
        position vector to that entity.

        Convention: θ = 0 iff the entity lies directly ahead of the
        UAV's current motion; cos θ ~ +1 means "in front", cos θ ~ -1
        means "behind", sin θ > 0 means "to the left of forward" (2D
        cross-product z-component).

        When the UAV is essentially still (|v| < eps), θ is undefined
        — we return zeros as a well-defined signal that "bearing is
        not meaningful right now" (paired with the ``speed`` scalar in
        the obs so the network can learn to discount it).
        """
        out = np.zeros((rel_pos.shape[0], 2), dtype=np.float32)
        speed = float(np.linalg.norm(self_vel))
        if speed < eps:
            return out
        vel_dir = self_vel / speed                              # (2,)
        safe_ranges = np.where(ranges > eps, ranges, 1.0)
        rel_dir = rel_pos / safe_ranges[:, None]                # (K, 2)
        cos_b = rel_dir @ vel_dir                               # (K,) dot
        sin_b = (vel_dir[0] * rel_dir[:, 1]
                 - vel_dir[1] * rel_dir[:, 0])                  # (K,) 2D cross-z
        mask = ranges > eps
        out[mask, 0] = cos_b[mask]
        out[mask, 1] = sin_b[mask]
        return out

    def _build_obs(self, blue_idx: int) -> np.ndarray:
        """
        Build the observation vector for blue UAV ``blue_idx``.

        Ego-centric world-frame representation (v2).  All positional
        features are relative to this UAV's own position; all
        velocities are relative to this UAV's own velocity.  A polar
        (range + bearing) pair complements the Cartesian relatives per
        entity — following the TERL feature vocabulary (see
        ``docs/stage1_analysis.md`` for rationale).

        Layout (concatenated float32):

            For each red target j (fixed order, N_red entries):
                rel_pos_xy      (2)  = (red_pos_j - self_pos) / arena_size
                rel_vel_xy      (2)  = (red_vel_j - self_vel) / v_max_blue
                range           (1)  = |rel_pos_xy| * arena_size / arena_size
                bearing_cos_sin (2)  = (cos θ, sin θ) between self_vel and rel_pos
                active_flag     (1)  = 1 if red still active else 0
            → 8 D per red

            For each teammate t (excluding self, fixed order in
            teammate_idx = [i for i != blue_idx], N_blue-1 entries):
                rel_pos_xy      (2)
                rel_vel_xy      (2)
                range           (1)
                bearing_cos_sin (2)
            → 7 D per teammate

            Self features:
                self_vel_xy_normalised (2)
                self_speed_normalised   (1)
                wall_distances          (4) = (x, L-x, y, L-y) / L

            Global:
                time_remaining          (1) = (max_steps - t) / max_steps

        No ``self_idx_onehot`` — every blue's obs is structurally
        identical from its own perspective, so parameter-shared PPO
        needs no symmetry-breaking feature.

        Total dim for N_blue=3, N_red=2 = 16 + 14 + 8 = 38.
        """
        L = self.arena_size
        v_max_self = BLUE_UAV.v_max

        self_pos = self._blue_pos[blue_idx]                     # (2,)
        self_vel = self._blue_vel[blue_idx]                     # (2,)

        # ---- Per-red block --------------------------------------------
        rel_red_pos = self._red_pos - self_pos                  # (N_red, 2)
        rel_red_vel = self._red_vel - self_vel                  # (N_red, 2)
        red_ranges  = np.linalg.norm(rel_red_pos, axis=1)       # (N_red,)
        red_bearings = self._bearing_features(self_vel,
                                              rel_red_pos, red_ranges)

        # Normalise: positions by L, velocities by v_max_self.
        rel_red_pos_n = (rel_red_pos / L).astype(np.float32)
        rel_red_vel_n = (rel_red_vel / v_max_self).astype(np.float32)
        red_ranges_n  = (red_ranges  / L).astype(np.float32)

        # Zero out caught reds (defensive — active_flag also carries
        # the signal, but keeps the obs clean of stale data).
        inactive_mask = ~self._red_active
        rel_red_pos_n[inactive_mask] = 0.0
        rel_red_vel_n[inactive_mask] = 0.0
        red_ranges_n[inactive_mask]  = 0.0
        red_bearings[inactive_mask]  = 0.0

        red_active_flag = self._red_active.astype(np.float32)

        red_block = np.concatenate([
            rel_red_pos_n.flatten(),         # 2 * N_red
            rel_red_vel_n.flatten(),         # 2 * N_red
            red_ranges_n,                    #     N_red
            red_bearings.flatten(),          # 2 * N_red
            red_active_flag,                 #     N_red
        ]).astype(np.float32)

        # ---- Per-teammate block ---------------------------------------
        teammate_idx = [i for i in range(self.n_blue) if i != blue_idx]
        tm_pos = self._blue_pos[teammate_idx]                   # (N_blue-1, 2)
        tm_vel = self._blue_vel[teammate_idx]
        rel_tm_pos = tm_pos - self_pos
        rel_tm_vel = tm_vel - self_vel
        tm_ranges  = np.linalg.norm(rel_tm_pos, axis=1)
        tm_bearings = self._bearing_features(self_vel,
                                             rel_tm_pos, tm_ranges)

        rel_tm_pos_n = (rel_tm_pos / L).astype(np.float32)
        rel_tm_vel_n = (rel_tm_vel / v_max_self).astype(np.float32)
        tm_ranges_n  = (tm_ranges  / L).astype(np.float32)

        tm_block = np.concatenate([
            rel_tm_pos_n.flatten(),          # 2 * (N_blue - 1)
            rel_tm_vel_n.flatten(),          # 2 * (N_blue - 1)
            tm_ranges_n,                     #     (N_blue - 1)
            tm_bearings.flatten(),           # 2 * (N_blue - 1)
        ]).astype(np.float32)

        # ---- Self + global block --------------------------------------
        self_vel_n = (self_vel / v_max_self).astype(np.float32)
        self_speed_n = np.array(
            [float(np.linalg.norm(self_vel)) / v_max_self],
            dtype=np.float32,
        )
        walls = np.array([
            self_pos[0] / L,                 # → left  wall (x=0)
            (L - self_pos[0]) / L,           # → right wall
            self_pos[1] / L,                 # → bottom wall (y=0)
            (L - self_pos[1]) / L,           # → top   wall
        ], dtype=np.float32)
        time_remaining = np.array(
            [(self.max_steps - self._t) / self.max_steps],
            dtype=np.float32,
        )

        return np.concatenate([
            red_block,        # 8 * N_red
            tm_block,         # 7 * (N_blue - 1)
            self_vel_n,       # 2
            self_speed_n,     # 1
            walls,            # 4
            time_remaining,   # 1
        ]).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Structured observation (v3 / Stage 2 — for the GNN policy)         #
    # ------------------------------------------------------------------ #

    def _build_bb_edge_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fixed directed edges between every pair of distinct blue nodes.
        Order: for src in 0..N-1, for dst in 0..N-1 if dst != src.
        For N_blue = 3 -> src = [0,0,1,1,2,2], dst = [1,2,0,2,0,1].
        """
        src, dst = [], []
        for s in range(self.n_blue):
            for d in range(self.n_blue):
                if s != d:
                    src.append(s); dst.append(d)
        return (np.asarray(src, dtype=np.int64),
                np.asarray(dst, dtype=np.int64))

    def _build_rb_edge_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fixed directed edges from every red to every blue.
        Order: for src in 0..N_red-1, for dst in 0..N_blue-1.
        For N_red=2, N_blue=3 -> src=[0,0,0,1,1,1], dst=[0,1,2,0,1,2].
        """
        src, dst = [], []
        for r in range(self.n_red):
            for b in range(self.n_blue):
                src.append(r); dst.append(b)
        return (np.asarray(src, dtype=np.int64),
                np.asarray(dst, dtype=np.int64))

    def _edge_features_for(
        self,
        src_pos:  np.ndarray,   # (E, 2) sender positions
        src_vel:  np.ndarray,   # (E, 2) sender velocities
        dst_pos:  np.ndarray,   # (E, 2) receiver positions
        dst_vel:  np.ndarray,   # (E, 2) receiver velocities
    ) -> np.ndarray:
        """
        Compute the shared 7-D edge feature layout
        ``[rel_pos_xy (2), rel_vel_xy (2), range (1), bearing_cs (2)]``
        for an arbitrary set of directed edges.

        The bearing is defined in the **sender's** frame (angle between
        sender_vel and rel_pos = dst - src), following the design in
        docs/stage2_gnn_design.md §2.2.  All positional / velocity
        features are normalised the same way as in the flat obs
        (positions by ``arena_size``, velocities by ``BLUE_UAV.v_max``).
        """
        L = self.arena_size
        v_max = BLUE_UAV.v_max
        rel_pos = dst_pos - src_pos                           # (E, 2)
        rel_vel = dst_vel - src_vel                           # (E, 2)
        ranges  = np.linalg.norm(rel_pos, axis=1)             # (E,)
        # Bearing in sender frame — reuse the same helper.
        bearing = np.zeros((rel_pos.shape[0], 2), dtype=np.float32)
        for i in range(rel_pos.shape[0]):
            bearing[i] = self._bearing_features(
                src_vel[i], rel_pos[i:i + 1], ranges[i:i + 1],
            )[0]
        return np.concatenate([
            (rel_pos / L).astype(np.float32),
            (rel_vel / v_max).astype(np.float32),
            (ranges / L).astype(np.float32).reshape(-1, 1),
            bearing,
        ], axis=1).astype(np.float32)

    def _velocity_edge_features_for(
        self,
        src_vel: np.ndarray,   # (E, 2)  sender velocities  (e.g. reds)
        dst_vel: np.ndarray,   # (E, 2)  receiver velocities (e.g. blues)
    ) -> np.ndarray:
        """
        Velocity-only edge features for the Stage 4 v5.1 rb_edges.

        Produces 4-D features: ``[rel_vel (2), src_vel (2)]`` normalised
        by ``BLUE_UAV.v_max``.  NO position information -- that lives
        exclusively in the noisy Bayesian belief map, per the realistic
        sensor split (Doppler radar precisely measures velocity;
        position is uncertain).

        - ``rel_vel = src_vel - dst_vel``: closing/opening motion, useful
          for pursuit geometry once the policy knows WHERE the enemy is.
        - ``src_vel`` (enemy absolute velocity): tells the policy where
          the enemy is heading in world frame, for lead-pursuit.
        """
        v_max = BLUE_UAV.v_max
        rel_vel = src_vel - dst_vel                          # (E, 2)
        return np.concatenate([
            (rel_vel / v_max).astype(np.float32),
            (src_vel / v_max).astype(np.float32),
        ], axis=1).astype(np.float32)

    def _build_structured_obs(self) -> Dict[str, np.ndarray]:
        """
        Build the graph-structured observation for the GNN policy.

        Layout:
        - ``blue_features``    : (N_blue, 8)  intrinsic blue node feats
          per row = [vel_xy_norm (2), speed_norm (1), wall_dists (4),
                     time_remaining (1)]
        - ``red_features``     : (N_red,  1)  intrinsic red node feats
          per row = [active_flag (1)]
        - ``bb_edge_features`` : (n_bb, 7)    blue-blue edge feats
          ordered per self.bb_edge_src / self.bb_edge_dst
        - ``rb_edge_features`` : (n_rb, 7)    red-blue edge feats
          ordered per self.rb_edge_src / self.rb_edge_dst

        The graph *structure* (edge index tables) is fixed at env
        construction time and exposed as ``self.{bb,rb}_edge_{src,dst}``
        — no need to include it in every obs.
        """
        L = self.arena_size
        v_max = BLUE_UAV.v_max
        t = self._t

        # ---- Blue node features (N_blue, 8) --------------------------
        blue_vel_n = (self._blue_vel / v_max).astype(np.float32)       # (N_blue, 2)
        blue_speed = np.linalg.norm(self._blue_vel, axis=1) / v_max    # (N_blue,)
        wall_dists = np.stack([
            self._blue_pos[:, 0] / L,
            (L - self._blue_pos[:, 0]) / L,
            self._blue_pos[:, 1] / L,
            (L - self._blue_pos[:, 1]) / L,
        ], axis=1).astype(np.float32)                                  # (N_blue, 4)
        time_col = np.full(
            (self.n_blue, 1),
            (self.max_steps - t) / self.max_steps,
            dtype=np.float32,
        )                                                              # (N_blue, 1)
        blue_features = np.concatenate([
            blue_vel_n,                                                # 2
            blue_speed.reshape(-1, 1).astype(np.float32),              # 1
            wall_dists,                                                # 4
            time_col,                                                  # 1
        ], axis=1)                                                     # (N_blue, 8)

        # ---- Red node features (N_red, 1) ----------------------------
        red_features = self._red_active.astype(np.float32).reshape(-1, 1)

        # ---- Blue-Blue edge features (n_bb, 7) -----------------------
        bb_src_pos = self._blue_pos[self.bb_edge_src]
        bb_src_vel = self._blue_vel[self.bb_edge_src]
        bb_dst_pos = self._blue_pos[self.bb_edge_dst]
        bb_dst_vel = self._blue_vel[self.bb_edge_dst]
        bb_edge_features = self._edge_features_for(
            bb_src_pos, bb_src_vel, bb_dst_pos, bb_dst_vel,
        )

        # ---- Red-Blue edge features (n_rb, 7) ------------------------
        rb_src_pos = self._red_pos[self.rb_edge_src]
        rb_src_vel = self._red_vel[self.rb_edge_src]
        rb_dst_pos = self._blue_pos[self.rb_edge_dst]
        rb_dst_vel = self._blue_vel[self.rb_edge_dst]
        rb_edge_features = self._edge_features_for(
            rb_src_pos, rb_src_vel, rb_dst_pos, rb_dst_vel,
        )
        # Zero edge features for edges to/from caught reds — the network
        # can still see active_flag on the red node.
        for i, r in enumerate(self.rb_edge_src):
            if not self._red_active[r]:
                rb_edge_features[i] = 0.0

        return {
            "blue_features":     blue_features,
            "red_features":      red_features,
            "bb_edge_features":  bb_edge_features,
            "rb_edge_features":  rb_edge_features,
        }

    def structured_observation(self) -> Dict[str, np.ndarray]:
        """
        Public accessor for the Stage 2 graph-structured obs.

        Unlike the flat per-agent ``observation_space`` used by the
        PettingZoo Gym API, this returns one dict describing the whole
        graph at the current timestep.  The GNN policy's forward pass
        consumes this once and produces action distributions for every
        blue node simultaneously.
        """
        return self._build_structured_obs()

    # ------------------------------------------------------------------ #
    #  Stage 3 partial observability (sensor_radius)                       #
    # ------------------------------------------------------------------ #

    def _compute_edge_visibility(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-edge visibility masks under ``sensor_radius``.

        Convention (see docs/stage3_design.md §2.2):
        - A bb edge ``(i -> j)`` is visible iff
          ``distance(blue_j, blue_i) <= sensor_radius`` — the RECEIVER
          is what matters.  Because Euclidean distance is symmetric,
          bb visibility for a pair (i, j) is symmetric across the two
          directed edges (i->j and j->i are both visible together).
        - A rb edge ``(r -> b)`` is visible iff
          ``distance(blue_b, red_r) <= sensor_radius``.

        When ``sensor_radius`` is None (fully observable), both masks
        are all ones (Stage 2 behaviour, preserved).

        Returns
        -------
        bb_visible : (n_bb_edges,) float32
        rb_visible : (n_rb_edges,) float32
        """
        n_bb = self.n_bb_edges
        n_rb = self.n_rb_edges
        if self.sensor_radius is None:
            return (np.ones(n_bb, dtype=np.float32),
                    np.ones(n_rb, dtype=np.float32))

        R = self.sensor_radius
        # bb: receiver = bb_edge_dst.  Distance receiver-to-sender.
        bb_recv = self._blue_pos[self.bb_edge_dst]
        bb_send = self._blue_pos[self.bb_edge_src]
        bb_dists = np.linalg.norm(bb_recv - bb_send, axis=1)
        bb_visible = (bb_dists <= R).astype(np.float32)

        # rb: receiver blue = rb_edge_dst, sender red = rb_edge_src.
        rb_recv = self._blue_pos[self.rb_edge_dst]
        rb_send = self._red_pos[self.rb_edge_src]
        rb_dists = np.linalg.norm(rb_recv - rb_send, axis=1)
        rb_visible = (rb_dists <= R).astype(np.float32)
        # Caught reds should also not appear in the actor's view even
        # if physically near — active_flag on the red node still
        # exists but their outgoing edges are hidden.
        rb_visible = rb_visible * self._red_active[self.rb_edge_src].astype(np.float32)

        return bb_visible, rb_visible

    # ------------------------------------------------------------------ #
    #  Stage 4: obstacles + belief maps                                    #
    # ------------------------------------------------------------------ #

    def _place_obstacles(self) -> None:
        """
        Rejection-sample ``n_obstacles`` non-overlapping circular
        obstacles inside the arena.  Called from ``reset()``.

        Placement rules:
        - Radius drawn uniformly from
          ``[obstacle_radius_min, obstacle_radius_max]``.
        - Position sampled uniformly with a wall clearance of
          ``obstacle_radius_max`` metres.
        - New obstacle rejected if it overlaps an existing one (a 1 m
          minimum gap between boundaries).
        - Up to 200 attempts per obstacle; if placement fails, fewer
          obstacles land -- exposed via ``len(self._obstacle_pos)``.
        """
        if self.n_obstacles > 0:
            L = self.arena_size
            max_r = self.obstacle_radius_max
            placed_pos: list = []
            placed_r:   list = []

            max_attempts = 200
            for _ in range(self.n_obstacles):
                for _attempt in range(max_attempts):
                    r = float(self._rng.uniform(
                        self.obstacle_radius_min, self.obstacle_radius_max,
                    ))
                    lo = max_r
                    hi = L - max_r
                    if hi <= lo:
                        break
                    pos = self._rng.uniform(lo, hi, size=2).astype(np.float32)
                    ok = True
                    for pp, pr in zip(placed_pos, placed_r):
                        if float(np.linalg.norm(pos - pp)) < r + pr + 1.0:
                            ok = False
                            break
                    if ok:
                        placed_pos.append(pos)
                        placed_r.append(r)
                        break

            if placed_pos:
                self._obstacle_pos = np.stack(placed_pos, axis=0).astype(np.float32)
                self._obstacle_r   = np.asarray(placed_r, dtype=np.float32)
            else:
                self._obstacle_pos = np.zeros((0, 2), dtype=np.float32)
                self._obstacle_r   = np.zeros((0,),   dtype=np.float32)
        else:
            self._obstacle_pos = np.zeros((0, 2), dtype=np.float32)
            self._obstacle_r   = np.zeros((0,),   dtype=np.float32)

        # ---- Precompute static caches for the belief update ---------------
        # Cell centres (W, H, 2), used every step for sensor-disk masks
        # and ray endpoints.  Computed once at reset since the grid is
        # fixed for the episode.
        H = W = self.belief_grid_size
        cs = self.belief_cell_size
        xs = (np.arange(W) + 0.5) * cs
        ys = (np.arange(H) + 0.5) * cs
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._cell_centres = np.stack(
            [gx, gy], axis=-1,
        ).astype(np.float32)                                        # (W, H, 2)
        self._cell_centres_flat = self._cell_centres.reshape(-1, 2)  # (W*H, 2)

        # Obstacle-cell mask (W, H) bool, cached because obstacles are
        # static.  Used by _true_occupancy every step.
        if len(self._obstacle_pos) > 0:
            diffs = self._cell_centres_flat[:, None, :] - self._obstacle_pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)                  # (W*H, n_obs)
            inside = np.any(dists <= self._obstacle_r[None, :], axis=1)
            self._obstacle_grid = inside.reshape(W, H).astype(np.float32)
        else:
            self._obstacle_grid = np.zeros((W, H), dtype=np.float32)

    def _positions_in_any_obstacle(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorised: return a boolean mask (K,) that's True iff the
        corresponding position lies inside any obstacle disk.  Empty
        obstacle list returns all-False.
        """
        if self._obstacle_pos is None or len(self._obstacle_pos) == 0:
            return np.zeros(positions.shape[0], dtype=bool)
        # (K, n_obs) distances
        diffs = positions[:, None, :] - self._obstacle_pos[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        # Inside iff dist <= r for any obstacle.
        return np.any(dists <= self._obstacle_r[None, :], axis=1)

    def _sample_free_positions(self, k: int) -> np.ndarray:
        """
        Uniformly sample ``k`` positions in the arena that are outside
        every obstacle by at least ``obstacle_spawn_clearance`` metres.
        Falls back to unclearance-checked uniform sampling after
        ``max_attempts`` if the obstacle geometry is pathological.
        """
        L = self.arena_size
        if self._obstacle_pos is None or len(self._obstacle_pos) == 0:
            return self._rng.uniform(0.0, L, size=(k, 2)).astype(np.float32)

        out = np.zeros((k, 2), dtype=np.float32)
        max_attempts = 100
        for i in range(k):
            for _ in range(max_attempts):
                p = self._rng.uniform(0.0, L, size=2).astype(np.float32)
                # Clearance check against all obstacles.
                diffs = p[None, :] - self._obstacle_pos
                dists = np.linalg.norm(diffs, axis=1)
                if bool(np.all(dists >= self._obstacle_r + self.obstacle_spawn_clearance)):
                    out[i] = p
                    break
            else:
                # Fallback: accept any position not literally inside an obstacle.
                out[i] = self._rng.uniform(0.0, L, size=2).astype(np.float32)
        return out

    def _clip_positions_from_obstacles(
        self,
        new_pos:  np.ndarray,   # (K, 2) post-integration
        prev_pos: np.ndarray,   # (K, 2) pre-integration
        vel:      np.ndarray,   # (K, 2) post-integration velocity
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each entity whose new position lies inside any obstacle,
        roll the position back to its previous position and zero the
        velocity.  Simple "hit an obstacle, stop" model.  Crash penalty
        is a backlog item.
        """
        collided = self._positions_in_any_obstacle(new_pos)
        if bool(np.any(collided)):
            new_pos = new_pos.copy()
            vel     = vel.copy()
            new_pos[collided] = prev_pos[collided]
            vel[collided]     = 0.0
        return new_pos, vel

    def _true_occupancy(self) -> np.ndarray:
        """
        Ground-truth occupancy grid of shape ``(2, H, W)`` -- always
        exactly 2 channels: enemy + obstacle.  Used by the critic and
        by the diagnostic BCE.  It does NOT track the deterministic
        ally / self channels that live in the actor's belief tensor
        (channels 2-3), because those are already fully known and
        the critic reads ally positions via ``blue_features`` anyway.
        """
        H = W = self.belief_grid_size
        cs = self.belief_cell_size
        grid = np.zeros((2, H, W), dtype=np.float32)

        # Channel 0: enemy (active reds).
        for r in range(self.n_red):
            if not self._red_active[r]:
                continue
            cx = int(self._red_pos[r, 0] / cs)
            cy = int(self._red_pos[r, 1] / cs)
            if 0 <= cx < W and 0 <= cy < H:
                grid[0, cx, cy] = 1.0

        # Channel 1: obstacle -- cached once at reset (obstacles static).
        grid[1] = self._obstacle_grid
        return grid

    def _cell_occluded_by_obstacle(
        self,
        uav_pos:     np.ndarray,   # (2,)
        cell_centre: np.ndarray,   # (2,)
    ) -> bool:
        """
        Return True iff the straight line from ``uav_pos`` to
        ``cell_centre`` passes through any obstacle (strictly between
        the endpoints).  The cell centre itself is allowed to be an
        obstacle -- the sensor CAN observe obstacle cells to add
        evidence to the obstacle channel; only cells BEHIND another
        obstacle along the ray are hidden.
        """
        if self._obstacle_pos is None or len(self._obstacle_pos) == 0:
            return False
        diff = cell_centre - uav_pos
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return False
        n_steps = max(2, int(np.ceil(dist / self.ray_step_size)) + 1)
        # Sample points strictly between the endpoints.  We EXCLUDE k=n_steps-1
        # (the endpoint) so the sensor can observe the first obstacle cell it
        # sees; we EXCLUDE k=0 (the UAV) trivially.
        ts = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]   # (n_steps - 1,)
        pts = uav_pos[None, :] + ts[:, None] * diff[None, :]   # (n_steps-1, 2)
        # For each interior point, distance to all obstacles.
        # (n_pts, n_obs)
        dists = np.linalg.norm(
            pts[:, None, :] - self._obstacle_pos[None, :, :], axis=-1,
        )
        # Occluded if ANY interior point is inside ANY obstacle.
        return bool(np.any(dists < self._obstacle_r[None, :]))

    def _update_belief_maps(self) -> None:
        """
        Vectorised Bayesian log-odds update on ``self._belief_maps``.
        See ``docs/stage4_design.md §3.4`` for the algorithm.

        Channel layout (v2, 4 channels by default):
        - 0: P(enemy)    -- Bayesian log-odds from noisy sensor
        - 1: P(obstacle) -- Bayesian log-odds from noisy sensor
        - 2: ally_positions -- DETERMINISTIC overlay (perfect GPS via TDL)
        - 3: self_position  -- DETERMINISTIC overlay (own GPS)

        Only channels 0-1 are updated by Bayesian log-odds and clipped
        to ``[-belief_clip, +belief_clip]``.  Channels 2-3 are reset
        each step and written directly from ground-truth positions --
        they represent perfect self and ally position knowledge, which
        is what modern ISR drones have from GPS + Tactical Data Link.
        """
        if self._belief_maps is None or self.sensor_radius is None:
            return

        R  = self.sensor_radius
        centres = self._cell_centres                # (W, H, 2)  cached at reset
        C_total    = self.belief_channels
        C_bayesian = min(2, C_total)                # channels 0..1 if present
        truth = self._true_occupancy()              # (2, W, H) -- always 2

        n_obs = 0 if self._obstacle_pos is None else int(self._obstacle_pos.shape[0])

        for i in range(self.n_blue):
            uav_pos = self._blue_pos[i]

            # 1. Cells in sensor disk: (W, H) bool.
            dist_to_cells = np.linalg.norm(centres - uav_pos, axis=-1)
            in_disk = dist_to_cells <= R
            if not bool(np.any(in_disk)):
                continue

            cx_idx, cy_idx = np.where(in_disk)      # (K,) each
            K = int(cx_idx.shape[0])
            candidate_centres = centres[cx_idx, cy_idx]     # (K, 2)

            # 2. Batched ray-cast occlusion check across ALL candidates.
            if n_obs > 0:
                diffs = candidate_centres - uav_pos          # (K, 2)
                dists = np.linalg.norm(diffs, axis=-1)       # (K,)
                # Enough interior samples to resolve any obstacle along
                # the LONGEST ray at ``ray_step_size`` granularity.
                max_dist = float(dists.max())
                n_samples = max(1, int(np.ceil(max_dist / self.ray_step_size)))
                # Interior sample fractions in (0, 1); endpoints excluded so
                # the sensor CAN observe an obstacle boundary cell directly.
                if n_samples <= 1:
                    ts = np.array([0.5], dtype=np.float32)
                else:
                    ts = np.linspace(0.0, 1.0, n_samples + 1,
                                     dtype=np.float32)[1:-1]
                # (K, S, 2) sample points along each ray.
                sample_pts = (
                    uav_pos[None, None, :]
                    + ts[None, :, None] * diffs[:, None, :]
                )                                             # (K, S, 2)
                # Distance from each sample to each obstacle: (K, S, n_obs).
                obs_diffs = (
                    sample_pts[:, :, None, :]
                    - self._obstacle_pos[None, None, :, :]
                )
                obs_dists = np.linalg.norm(obs_diffs, axis=-1)
                # A candidate is occluded if any of its samples lies
                # strictly inside any obstacle.
                occluded = np.any(
                    obs_dists < self._obstacle_r[None, None, :],
                    axis=(1, 2),
                )                                             # (K,)
                visible = ~occluded
                cx_idx = cx_idx[visible]
                cy_idx = cy_idx[visible]
                K = int(cx_idx.shape[0])
                if K == 0:
                    continue

            # 3. Vectorised Bernoulli sensor sampling across BAYESIAN channels only.
            if C_bayesian > 0:
                truth_vis = truth[:C_bayesian, cx_idx, cy_idx]     # (C_b, K)
                uniforms  = self._rng.random((C_bayesian, K)).astype(np.float32)
                p_detect  = np.where(
                    truth_vis > 0.5,
                    np.float32(self.p_TP),
                    np.float32(self.p_FP),
                )                                                   # (C_b, K)
                detected = uniforms < p_detect                      # (C_b, K) bool
                evidence = np.where(
                    detected,
                    np.float32(self._L_detect),
                    np.float32(self._L_no_detect),
                )                                                   # (C_b, K)

                # 4. Apply evidence to this UAV's belief map (Bayesian channels).
                for c in range(C_bayesian):
                    self._belief_maps[i, c, cx_idx, cy_idx] += evidence[c]

        # Clip log-odds -- ONLY on Bayesian channels; deterministic
        # channels 2-3 stay in {0, 1} untouched.
        if C_bayesian > 0:
            np.clip(
                self._belief_maps[:, :C_bayesian],
                -self.belief_clip, self.belief_clip,
                out=self._belief_maps[:, :C_bayesian],
            )

        # 5. Deterministic overlays for channels 2 (allies) and 3 (self).
        #    Each is reset to zero every step, then set to 1.0 at the
        #    ground-truth cell(s) -- perfect GPS + TDL model.
        if C_total >= 3:
            self._belief_maps[:, 2] = 0.0
            cs = self.belief_cell_size
            H = W = self.belief_grid_size
            # Ally channel: same for every UAV -- all allies visible on TDL.
            for j in range(self.n_blue):
                cx = int(self._blue_pos[j, 0] / cs)
                cy = int(self._blue_pos[j, 1] / cs)
                if 0 <= cx < W and 0 <= cy < H:
                    self._belief_maps[:, 2, cx, cy] = 1.0

        if C_total >= 4:
            self._belief_maps[:, 3] = 0.0
            cs = self.belief_cell_size
            H = W = self.belief_grid_size
            # Self channel: DIFFERENT per UAV -- each sees only itself.
            for i in range(self.n_blue):
                cx = int(self._blue_pos[i, 0] / cs)
                cy = int(self._blue_pos[i, 1] / cs)
                if 0 <= cx < W and 0 <= cy < H:
                    self._belief_maps[i, 3, cx, cy] = 1.0

    def _extract_belief_peaks(self, K: int, channel_idx: int = 0) -> np.ndarray:
        """
        Vectorised top-K peak extraction from a specified channel of
        each UAV's belief map.  Returns per-detection (rel_dx, rel_dy,
        P_conf) triples in normalised units.

        Rationale: v5.1 showed the CNN can't extract usable position
        info from the belief-map tensor.  Explicit peak detection is
        what real ISR trackers do -- output a list of tracks with
        (position, confidence).  The peaks are still noisy (they
        inherit the belief map's noise), so the sensor-physics story
        holds; we just give the policy a more digestible form.

        Parameters
        ----------
        K : int
          Number of peaks to extract per UAV.
        channel_idx : int
          Which belief map channel to extract from.  0 = P(enemy),
          1 = P(obstacle) in the v5.2+ layout.

        Returns
        -------
        peaks : (N_blue, K, 3) float32
          Channels: (dx / arena_size, dy / arena_size, sigmoid(log_odds))
          dx, dy are relative from the UAV to the peak cell centre.
        """
        assert self._belief_maps is not None
        assert 0 <= channel_idx < self._belief_maps.shape[1]
        L = self.arena_size
        cs = self.belief_cell_size
        H = W = self.belief_grid_size

        # Extract log-odds for the requested channel and sigmoid to prob.
        chan_lo = self._belief_maps[:, channel_idx, :, :]
        chan_p  = 1.0 / (1.0 + np.exp(-chan_lo))
        flat = chan_p.reshape(self.n_blue, -1)              # (N, H*W)

        # Top-K per UAV (vectorised across UAVs).
        if flat.shape[1] <= K:
            topk_idx = np.tile(np.arange(flat.shape[1]),
                                (self.n_blue, 1))[:, :K]
        else:
            topk_idx = np.argpartition(flat, -K, axis=1)[:, -K:]

        topk_vals = np.take_along_axis(flat, topk_idx, axis=1)  # (N, K)
        # Sort each row in descending value order.
        order = np.argsort(-topk_vals, axis=1)
        topk_idx  = np.take_along_axis(topk_idx,  order, axis=1)
        topk_vals = np.take_along_axis(topk_vals, order, axis=1)

        # Flat index -> (cx, cy).
        cx = topk_idx // W                                    # (N, K)
        cy = topk_idx %  W
        # Cell centres in world coords.
        px = (cx.astype(np.float32) + 0.5) * cs
        py = (cy.astype(np.float32) + 0.5) * cs
        # Relative to each UAV.
        dx = (px - self._blue_pos[:, 0:1]) / L                # (N, K)
        dy = (py - self._blue_pos[:, 1:2]) / L

        out = np.stack([dx, dy, topk_vals.astype(np.float32)], axis=-1)
        return out.astype(np.float32)

    def _extract_belief_windows(self, K: int) -> np.ndarray:
        """
        Ego-centric KxK crop of each UAV's belief map centred on that
        UAV's own cell.  Cells outside the arena are zero-padded (so a
        UAV near a wall gets zeros filling the out-of-bounds portion of
        its window).

        Returns
        -------
        windows : (N_blue, C, K, K) float32
        """
        assert self._belief_maps is not None
        assert K >= 1
        r = K // 2
        C = self.belief_channels
        H = W = self.belief_grid_size
        cs = self.belief_cell_size

        # Zero-pad the belief tensor spatially by r on each side.
        padded = np.pad(
            self._belief_maps,
            ((0, 0), (0, 0), (r, r), (r, r)),
            mode="constant", constant_values=0.0,
        )                                     # (N_blue, C, H+2r, W+2r)

        out = np.zeros((self.n_blue, C, K, K), dtype=np.float32)
        for i in range(self.n_blue):
            cx = int(np.clip(self._blue_pos[i, 0] / cs, 0, W - 1))
            cy = int(np.clip(self._blue_pos[i, 1] / cs, 0, H - 1))
            # In padded coords the UAV cell is at (cx + r, cy + r);
            # window starts r cells before that -> at (cx, cy) in
            # padded coords -- so the window slice is [cx:cx+K, cy:cy+K].
            out[i] = padded[i, :, cx:cx + K, cy:cy + K]
        return out

    def structured_belief_observation(self) -> Dict[str, np.ndarray]:
        """
        Stage 4 observation dict.  Extends the Stage 2/3 fully-observable
        structured obs with per-UAV belief maps, obstacle positions, and
        ground-truth occupancy (training-only key).

        Notable removals versus Stage 3:
        - ``red_features`` and ``rb_edge_features`` -- reds live only in
          the belief map's enemy channel.
        - ``bb_edge_visible`` and ``rb_edge_visible`` -- no per-edge
          gating in Stage 4 (allies communicate via GPS-style
          continuous comms; enemies enter via the belief map).

        Kept from Stage 3:
        - ``blue_features`` and ``bb_edge_features`` -- unchanged
          semantics.

        New keys:
        - ``belief_maps``       : (N_blue, C, H, W) float32 log-odds
        - ``obstacle_positions``: (N_obs, 3) float32 = (x, y, radius)
        - ``true_occupancy``    : (C, H, W) float32 binary,
          training-only.  The actor must NOT read this key.
        """
        base = self._build_structured_obs()
        # v5.1: velocity-only rb_edges.
        # Position of enemies lives ONLY in the noisy Bayesian belief
        # map -- feeding precise position through rb_edges would let
        # the policy bypass the belief map entirely (radar/EO position
        # estimates are noisy in reality; only the fused belief
        # tracker output is available to the operator).  Velocity is
        # measured directly by Doppler radar and comes through
        # precisely when the target is visible.
        _bb_vis, rb_vis = self._compute_edge_visibility()
        rb_edge_vel = self._velocity_edge_features_for(
            src_vel=self._red_vel[self.rb_edge_src],
            dst_vel=self._blue_vel[self.rb_edge_dst],
        )
        out: Dict[str, np.ndarray] = {
            "blue_features":     base["blue_features"],
            "bb_edge_features":  base["bb_edge_features"],
            "red_features":      base["red_features"],
            "rb_edge_features":  rb_edge_vel,
            "rb_edge_visible":   rb_vis,
        }

        # Belief maps.  If disabled, still return a zero-filled tensor
        # so downstream callers can rely on the schema.
        if self._belief_maps is not None:
            out["belief_maps"] = self._belief_maps.copy()
        else:
            out["belief_maps"] = np.zeros(
                (self.n_blue, self.belief_channels,
                 self.belief_grid_size, self.belief_grid_size),
                dtype=np.float32,
            )

        # Obstacle positions in (x, y, r) form.
        if self._obstacle_pos is not None and len(self._obstacle_pos) > 0:
            out["obstacle_positions"] = np.concatenate([
                self._obstacle_pos,
                self._obstacle_r.reshape(-1, 1),
            ], axis=1).astype(np.float32)
        else:
            out["obstacle_positions"] = np.zeros((0, 3), dtype=np.float32)

        # Ground-truth occupancy for the CTDE critic + diagnostic BCE.
        out["true_occupancy"] = self._true_occupancy()

        # Ego-centric belief windows (Phase 4 v3).  Only included when
        # belief_window_size > 0.  Downstream: actor consumes
        # ``belief_windows`` in place of ``belief_maps``; belief_maps
        # is still returned for the diagnostic BCE + the CTDE critic
        # can still see it if needed.
        if self.belief_window_size > 0 and self._belief_maps is not None:
            out["belief_windows"] = self._extract_belief_windows(
                self.belief_window_size,
            )

        # Belief-map peak detections (Phase 4 v5.3).  Top-K cells with
        # highest P per UAV, as (dx, dy, conf) triples.  Two channels:
        #   - belief_peaks_enemy    : from log-odds channel 0
        #     (K = n_red -- one detection slot per real target)
        #   - belief_peaks_obstacle : from log-odds channel 1
        #     (K = n_obstacles -- one slot per real obstacle)
        # Both come from the noisy Bayesian belief map -- consistent
        # sensor-physics story.  The former ``obstacle_positions``
        # ground-truth field is deliberately NOT consumed by the
        # actor, only kept in the obs for logging / diagnostics.
        if self._belief_maps is not None:
            out["belief_peaks_enemy"] = self._extract_belief_peaks(
                self.n_red, channel_idx=0,
            )
            # Obstacles: ensure we always emit at least 1 slot even
            # in the n_obstacles=0 case so downstream tensor shapes
            # stay consistent within a training run.  The confidence
            # channel signals "nothing here" when there are no real
            # obstacles (P → 0 as sensor keeps observing "no obstacle").
            n_obs_peaks = max(1, self.n_obstacles)
            if self._belief_maps.shape[1] >= 2:
                out["belief_peaks_obstacle"] = self._extract_belief_peaks(
                    n_obs_peaks, channel_idx=1,
                )

        return out

    def structured_partial_observation(self) -> Dict[str, np.ndarray]:
        """
        Partial-observability variant of ``structured_observation()``
        for the Stage 3 actor.

        Returns the Stage 2 dict extended with two extra keys:
        - ``bb_edge_visible`` : (n_bb, ) float32, 1.0 if the bb edge is
          within sensor range of its receiver, 0.0 otherwise.
        - ``rb_edge_visible`` : (n_rb, ) float32, same convention.

        Consumers (the Stage 3 actor's GNN) multiply message tensors by
        the visibility masks before aggregation so hidden edges
        contribute zero — same tensor shape as the fully-observable
        graph, just with masked contributions.

        The Stage 2 fields (node features, edge features) are
        unchanged; the mask is a *separate* signal.  If a downstream
        consumer wants the raw obs it can still call
        ``structured_observation()`` to get the full-state graph
        (used by the CTDE critic).
        """
        obs = self._build_structured_obs()
        bb_visible, rb_visible = self._compute_edge_visibility()
        obs["bb_edge_visible"] = bb_visible
        obs["rb_edge_visible"] = rb_visible
        return obs
