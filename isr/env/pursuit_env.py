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
        n_blue:         int                = 3,
        n_red:          int                = 2,
        arena_size:     float              = 100.0,
        max_steps:      int                = 200,
        capture_radius: float              = 3.0,
        dt:             float              = 1.0,
        red_policy:     Optional[Callable] = None,
        seed:           Optional[int]      = None,
        sensor_radius:  Optional[float]    = None,
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
        # Uniform random initial positions, zero initial velocities.
        # No minimum-separation constraint between blue and red in
        # Stage 1 — caught-at-spawn is rare given typical L/r ratios
        # and the policy will learn to handle it.  We can add a
        # min-separation constraint later if it matters empirically.
        self._blue_pos = self._rng.uniform(0.0, L,
                                           size=(self.n_blue, 2)).astype(np.float32)
        self._blue_vel = np.zeros((self.n_blue, 2), dtype=np.float32)
        self._red_pos  = self._rng.uniform(0.0, L,
                                           size=(self.n_red, 2)).astype(np.float32)
        self._red_vel  = np.zeros((self.n_red, 2), dtype=np.float32)
        self._red_active = np.ones(self.n_red, dtype=bool)
        self._t = 0
        self._last_n_caught = 0
        self.agents = list(self.possible_agents)

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
        self._blue_pos, self._blue_vel = self._integrate(
            self._blue_pos, self._blue_vel, blue_a, BLUE_UAV.v_max,
        )

        # 4. Integrate red kinematics.
        self._red_pos, self._red_vel = self._integrate(
            self._red_pos, self._red_vel, red_a, RED_TARGET.v_max,
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
        return {
            "t":           self._t,
            "blue_pos":    self._blue_pos.copy(),
            "blue_vel":    self._blue_vel.copy(),
            "red_pos":     self._red_pos.copy(),
            "red_vel":     self._red_vel.copy(),
            "red_active":  self._red_active.copy(),
            "n_caught_last_step": self._last_n_caught,
        }

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
