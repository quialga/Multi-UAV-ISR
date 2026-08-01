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
    blue_pos:     np.ndarray,           # (N_blue, 2)
    red_pos:      np.ndarray,           # (N_red,  2)
    red_active:   np.ndarray,           # (N_red,) bool
    obstacle_pos: Optional[np.ndarray] = None,   # (N_obs, 2)
    obstacle_r:   Optional[np.ndarray] = None,   # (N_obs,)
    arena_size:   Optional[float] = None,        # square side; enables walls
) -> np.ndarray:
    """
    Scripted red policy used by ``PursuitEnv``.

    Each *active* red target flees along the unit vector pointing
    **away** from its nearest blue UAV (max-effort evasion).  Caught
    reds (``active=False``) return zero acceleration.

    Collision-avoidance (Stage 4 obstacles): when obstacle geometry is
    supplied, a short-range REPULSION term from nearby obstacle
    surfaces is blended into the flee vector, so a fleeing red steers
    *around* obstacles instead of pinning itself against them (which
    would make it trivially cornerable — a strawman adversary).  The
    repulsion is limited to obstacles within ``influence`` metres of
    the boundary and falls off with distance; the result is renormalised
    to unit magnitude (still max-effort).  With no obstacles supplied
    the behaviour is byte-identical to the original.

    Wall repulsion (``arena_size`` given): the SAME treatment for the
    arena boundary.  A red fleeing straight away from a pursuer otherwise
    runs into the wall, has its perpendicular velocity clipped, and slides
    along the boundary — pinning itself exactly like the obstacle case and
    letting blue learn a degenerate wall-trapping counter.  An inward push
    from each of the four walls within ``influence`` metres is blended into
    the same repulsion term, so a cornered red curves back toward open
    space.  With ``arena_size=None`` (the default) wall repulsion is off
    and the behaviour is byte-identical to callers that don't pass it.

    The function is pure (no hidden state).

    Returns
    -------
    (N_red, 2) float32 — per-red acceleration commands.
    """
    n_red = red_pos.shape[0]
    out = np.zeros((n_red, 2), dtype=np.float32)

    has_obs = (obstacle_pos is not None and len(obstacle_pos) > 0
               and obstacle_r is not None)
    influence = 12.0        # metres beyond the surface where repulsion acts
    w_repulse = 1.5         # weight of repulsion vs the flee direction

    for i in range(n_red):
        if not red_active[i]:
            continue
        # 1. Flee direction: unit vector away from the nearest blue.
        diffs = red_pos[i] - blue_pos              # vectors blue -> this red
        dists = np.linalg.norm(diffs, axis=1)
        nearest = int(np.argmin(dists))
        d_vec = diffs[nearest]
        norm = float(np.linalg.norm(d_vec))
        flee = d_vec / norm if norm > 1e-8 else np.zeros(2, dtype=np.float32)

        # 2. Obstacle repulsion: sum of outward pushes from nearby disks,
        #    each scaled by how deep inside the influence band the red is.
        repulse = np.zeros(2, dtype=np.float32)
        if has_obs:
            oc = red_pos[i] - obstacle_pos          # (N_obs, 2) centre -> red
            od = np.linalg.norm(oc, axis=1)         # (N_obs,)  centre distance
            surf = od - obstacle_r                   # distance to the surface
            for o in range(obstacle_pos.shape[0]):
                if surf[o] < influence and od[o] > 1e-6:
                    strength = (influence - surf[o]) / influence   # in (0, 1]
                    strength = float(np.clip(strength, 0.0, 1.0))
                    repulse += strength * (oc[o] / od[o])          # outward unit

        # 2b. Wall repulsion: inward push from each of the four arena walls
        #     within the influence band — same falloff as obstacles, added
        #     to the same term.  Corners get pushes from two walls at once
        #     (a diagonal inward nudge), which is exactly what un-pins a
        #     cornered red.
        if arena_size is not None:
            rx, ry = float(red_pos[i, 0]), float(red_pos[i, 1])
            walls = (
                (rx,              np.array([ 1.0,  0.0], dtype=np.float32)),  # left
                (arena_size - rx, np.array([-1.0,  0.0], dtype=np.float32)),  # right
                (ry,              np.array([ 0.0,  1.0], dtype=np.float32)),  # bottom
                (arena_size - ry, np.array([ 0.0, -1.0], dtype=np.float32)),  # top
            )
            for dist, inward in walls:
                if dist < influence:
                    strength = float(np.clip(
                        (influence - dist) / influence, 0.0, 1.0))
                    repulse += strength * inward

        vec = flee + w_repulse * repulse
        v_norm = float(np.linalg.norm(vec))
        if v_norm > 1e-8:
            out[i] = vec / v_norm                   # renormalise: max effort
        else:
            out[i] = flee                           # degenerate: keep fleeing
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
        # ----- Phase A: Bayesian prediction step on the enemy channel -----
        # Both default OFF (1.0 / 0.0) so pre-Phase-A behaviour is
        # byte-preserved unless explicitly enabled (stage4 config
        # enables them).
        enemy_belief_decay:       float = 1.0,   # gamma; < 1 = forgetting
        enemy_belief_diffusion:   float = 0.0,   # p_move; > 0 = motion spread
        # Live-sensor position accuracy (m) for VISIBLE targets: the rb
        # edge of a blue that can see the associated red uses the true
        # position + N(0, sigma^2) instead of the cell-centre grid peak.
        sensor_pos_noise_std:     float = 1.0,
        # ----- Crash penalties (per-agent shaped reward) ------------------
        # Both default 0.0 (off) so the shared-team-reward behaviour is
        # byte-preserved unless enabled.  When > 0, a crashing blue takes
        # an INDIVIDUAL reward hit r_crash_i (r_i = r_team + r_crash_i);
        # the episode is NOT terminated (soft stop, backlog §1 v1).
        crash_obstacle_penalty:   float = 0.0,   # magnitude; applied as -x
        crash_blue_penalty:       float = 0.0,   # magnitude; applied as -x
        blue_collision_radius:    float = 2.0,   # m; < capture_radius (3)
        # ----- Clearance / barrier shaping (backlog §16 #1) ---------------
        # Dense per-agent shaping that penalises being within
        # ``clearance_margin`` metres of an obstacle SURFACE, growing as the
        # blue approaches and continuing to grow INSIDE the disk — so its
        # gradient points continuously toward open space (a proactive "keep
        # distance" AND a "get out, this way" signal the flat crash penalty
        # can't give).  Both default OFF (weight 0.0) so behaviour is
        # byte-preserved.  See docs/stage4_backlog.md §16.
        clearance_weight:         float = 0.0,   # per-step magnitude; 0 = off
        clearance_margin:         float = 8.0,   # m band beyond the surface
        # ----- Variable entity counts (generalisation) --------------------
        # ``n_red`` / ``n_obstacles`` are the PADDED CAPACITY (fixed tensor
        # shapes).  When a *_min is set, each reset samples the actual
        # active count uniformly in [min, capacity]; the unused slots are
        # padded inactive (reds: _red_active=False; obstacles: unplaced).
        # None (default) => always at capacity, byte-preserving old runs.
        n_red_min:                Optional[int] = None,
        n_obstacles_min:          Optional[int] = None,
        # ----- Moving obstacles (backlog §4, simplest kinematics) ---------
        # A fraction of obstacles patrol back-and-forth along one axis,
        # bouncing off the arena walls (deterministic, no response to
        # blues).  Both default 0 => all obstacles static, byte-preserving
        # the pre-motion behaviour.  Crashing into a moving obstacle uses
        # the SAME per-agent crash penalty; blues are never destroyed.
        moving_obstacle_fraction: float = 0.0,   # 0..1 of placed obstacles
        obstacle_speed:           float = 0.0,   # m/step along the axis
        # Optional forgetting on the OBSTACLE belief channel so a moving
        # obstacle's stale "comet trail" fades (1.0 = off; enable when
        # obstacles move).  Mirrors enemy_belief_decay on channel 0.
        obstacle_belief_decay:    float = 1.0,
    ):
        """
        sensor_radius: Stage 3 partial-observability knob.  When None
            (default), the env is fully observable (Stage 1/2 behaviour).
            When set to a positive value, each blue UAV only "sees"
            entities within this Euclidean distance.  Visibility is
            per-receiver (each blue has its own view), computed by
            ``_compute_edge_visibility()`` and consumed as per-edge
            masks (e.g. ``bb_edge_visible`` in the Stage 4 obs) so the
            graph tensor shape stays constant across timesteps.
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
        use_belief_maps            When True, allocate and update the
                                   command-layer fused log-odds belief
                                   map every step (env-level latent;
                                   see doctrine note on
                                   ``_update_belief_maps``).  Costs
                                   some compute; default False.
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
        ray_step_size              Occlusion margin, in metres: obstacle
                                   boundary cells within this depth of
                                   the ray's disk entry point remain
                                   observable (the analytic segment-disk
                                   test cuts the ray this far before the
                                   cell centre).
        enemy_belief_decay         Phase A forgetting factor gamma for the
                                   enemy channel: L <- gamma * L each step
                                   before the sensor update.  1.0 = off.
                                   Pulls stale evidence (both signs)
                                   toward log-odds 0 = "unknown".
        enemy_belief_diffusion     Phase A motion-model spread p_move:
                                   the fraction of each cell's probability
                                   mass that moves to its 8 neighbours per
                                   step (isotropic random-walk prediction,
                                   applied in PROBABILITY space via a 3x3
                                   convolution).  0.0 = off.  Calibrate to
                                   target kinematics: p_move ~
                                   v_red_max * dt / cell_size.
        sensor_pos_noise_std       Live-sensor position accuracy (m).
                                   When a red is inside a blue's sensor
                                   disk, that blue's rb edge uses the
                                   TRUE position + N(0, sigma^2) instead
                                   of the cell-centre grid peak — real
                                   radar reports continuous measurements;
                                   the grid is only the fusion/memory
                                   layer.  Without this the endgame is
                                   cell-quantised (up to 3.5 m error on a
                                   5 m grid vs 3 m capture radius).
        """
        super().__init__()
        self.n_blue         = int(n_blue)
        self.n_red          = int(n_red)
        # Variable-count lower bounds (None => fixed at capacity).
        self.n_red_min = None if n_red_min is None else int(n_red_min)
        if self.n_red_min is not None:
            assert 0 <= self.n_red_min <= self.n_red, (
                f"n_red_min ({self.n_red_min}) must be in [0, n_red={self.n_red}]"
            )
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
        self.n_obstacles_min = (
            None if n_obstacles_min is None else int(n_obstacles_min)
        )
        if self.n_obstacles_min is not None:
            assert 0 <= self.n_obstacles_min <= self.n_obstacles, (
                f"n_obstacles_min ({self.n_obstacles_min}) must be in "
                f"[0, n_obstacles={self.n_obstacles}]"
            )
        self.moving_obstacle_fraction = float(moving_obstacle_fraction)
        assert 0.0 <= self.moving_obstacle_fraction <= 1.0
        self.obstacle_speed           = float(obstacle_speed)
        assert self.obstacle_speed >= 0.0
        self.obstacle_belief_decay    = float(obstacle_belief_decay)
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
        # Phase A prediction-step knobs (enemy channel only).
        self.enemy_belief_decay       = float(enemy_belief_decay)
        self.enemy_belief_diffusion   = float(enemy_belief_diffusion)
        assert 0.0 < self.enemy_belief_decay <= 1.0
        assert 0.0 <= self.enemy_belief_diffusion < 1.0
        self.sensor_pos_noise_std     = float(sensor_pos_noise_std)
        assert self.sensor_pos_noise_std >= 0.0
        # Crash-penalty knobs (per-agent shaped reward).
        self.crash_obstacle_penalty   = float(crash_obstacle_penalty)
        self.crash_blue_penalty       = float(crash_blue_penalty)
        self.blue_collision_radius    = float(blue_collision_radius)
        self.clearance_weight         = float(clearance_weight)
        self.clearance_margin         = float(clearance_margin)
        assert self.crash_obstacle_penalty >= 0.0
        assert self.crash_blue_penalty >= 0.0
        assert self.blue_collision_radius >= 0.0
        assert self.clearance_weight >= 0.0
        assert self.clearance_margin > 0.0
        # Diagnostic: crash counts for the LAST step (rendering/logging).
        self._last_obstacle_crashes: int = 0
        self._last_blue_crashes:     int = 0
        # Per-blue crash occupancy masks for the LAST step (for rising-edge
        # event counting; see _step).  All-False before the first step.
        self._last_obstacle_crash_mask = np.zeros(n_blue, dtype=bool)
        self._last_blue_crash_mask      = np.zeros(n_blue, dtype=bool)
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
        #
        # NOTE (Stage 4): obstacles are deliberately NOT part of this
        # flat per-agent obs.  It is the legacy Stage 1-3 MLP / Gym-API
        # path (also used by heuristic baselines, which are
        # obstacle-unaware).  Stage 4 policies consume the structured
        # graph obs (``structured_belief_observation``), where obstacles
        # appear as typed nodes with ob_edges.
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
        self._n_red_start: int                 = self.n_red
        self._t:          int                  = 0
        # Diagnostic captured from the last step (rendering / logging).
        self._last_n_caught: int = 0

        # ---- Stage 4 mutable state (obstacles + belief) ------------------
        # ``_obstacle_pos``  (n_obs, 2)   float32 -- centres
        # ``_obstacle_r``    (n_obs,)     float32 -- radii
        # ``_belief_maps``   (C, H, W)    float32 -- command-layer fused
        #   log-odds track picture (env-level latent; see doctrine note
        #   on ``_update_belief_maps`` and docs/stage4_backlog.md §13)
        # All None when Stage 4 features are disabled.
        self._obstacle_pos: Optional[np.ndarray] = None
        self._obstacle_r:   Optional[np.ndarray] = None
        self._obstacle_vel: Optional[np.ndarray] = None   # (n_obs, 2) patrol vel
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

        # Variable-count: sample how many obstacles / reds are ACTIVE this
        # episode (capacity stays fixed for tensor shapes; the rest are
        # padded inactive).  With *_min = None the count is fixed = cap.
        if self.n_obstacles_min is None:
            n_obs_target = self.n_obstacles
        else:
            n_obs_target = int(
                self._rng.integers(self.n_obstacles_min, self.n_obstacles + 1)
            )
        if self.n_red_min is None:
            n_red_active = self.n_red
        else:
            n_red_active = int(
                self._rng.integers(self.n_red_min, self.n_red + 1)
            )
        self._n_red_start = n_red_active   # for the caught metric under padding

        # Stage 4: place obstacles BEFORE sampling entity positions so
        # spawn clearance can reject bad draws.
        self._place_obstacles(n_obs_target)

        # Uniform random initial positions with obstacle clearance.
        # No minimum-separation constraint between blue and red in
        # Stage 1 — caught-at-spawn is rare given typical L/r ratios
        # and the policy will learn to handle it.  We can add a
        # min-separation constraint later if it matters empirically.
        self._blue_pos = self._sample_free_positions(self.n_blue)
        self._blue_vel = np.zeros((self.n_blue, 2), dtype=np.float32)
        self._red_pos  = self._sample_free_positions(self.n_red)
        self._red_vel  = np.zeros((self.n_red, 2), dtype=np.float32)
        # First n_red_active reds are live; the rest are padding (inactive
        # from step 0, treated exactly like caught reds everywhere).
        self._red_active = np.zeros(self.n_red, dtype=bool)
        self._red_active[:n_red_active] = True
        self._t = 0
        self._last_n_caught = 0
        self.agents = list(self.possible_agents)

        # Stage 4: zero the command-layer fused belief map (C, H, W).
        if self.use_belief_maps:
            self._belief_maps = np.zeros(
                (self.belief_channels,
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
        # Pass obstacle geometry so obstacle-aware red policies (e.g.
        # run_from_nearest_uav's collision-avoidance) can steer around
        # them.  Policies that don't use it accept and ignore the args.
        red_a = self.red_policy(
            self._blue_pos, self._red_pos, self._red_active,
            self._obstacle_pos, self._obstacle_r, self.arena_size,
        )
        red_a = np.clip(red_a.astype(np.float32), -1.0, 1.0)
        red_a[~self._red_active] = 0.0  # defensive

        # 2.5. Advance patrolling obstacles first (they respond to no one),
        #    so blue/red collision checks below run against the updated
        #    obstacle positions.  No-op unless moving obstacles are enabled.
        if self.n_obstacles > 0:
            self._move_obstacles()

        # 3. Integrate blue kinematics (double-integrator with axis-wise
        #    velocity cap and arena wall stop).
        prev_blue_pos = self._blue_pos.copy()
        self._blue_pos, self._blue_vel = self._integrate(
            self._blue_pos, self._blue_vel, blue_a, BLUE_UAV.v_max,
        )
        # Stage 4: reject moves that land inside any obstacle (soft
        # crash — position rolled back, velocity zeroed).  The collision
        # mask feeds the per-agent crash penalty below (episode is NOT
        # terminated, backlog §1 v1).
        blue_obstacle_crash = np.zeros(self.n_blue, dtype=bool)
        if self.n_obstacles > 0:
            (self._blue_pos, self._blue_vel,
             blue_obstacle_crash) = self._clip_positions_from_obstacles(
                self._blue_pos, prev_blue_pos, self._blue_vel,
            )

        # 4. Integrate red kinematics.
        prev_red_pos = self._red_pos.copy()
        self._red_pos, self._red_vel = self._integrate(
            self._red_pos, self._red_vel, red_a, RED_TARGET.v_max,
        )
        if self.n_obstacles > 0:
            self._red_pos, self._red_vel, _ = self._clip_positions_from_obstacles(
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
            # A confirmed capture is a perfect observation: wipe the
            # dead target's belief blob so the peak extractor cannot
            # re-latch onto its stale evidence.
            if (self.use_belief_maps and self._belief_maps is not None
                    and n_caught_this_step > 0):
                self._clear_enemy_belief_at(self._red_pos[newly_caught])
        self._last_n_caught = n_caught_this_step

        # 6. Reward.
        # 6a. TEAM (shared) component — see docs/design.md §3.7.
        catch_bonus = 10.0 * n_caught_this_step
        action_cost = 0.01 * float(np.sum(blue_a ** 2))   # sum over UAVs and axes
        step_cost   = 0.05
        r_team = catch_bonus - action_cost - step_cost

        # 6b. Blue-blue collisions (per-agent): any two blues within
        #     blue_collision_radius each take the ally-crash penalty.
        blue_ally_crash = np.zeros(self.n_blue, dtype=bool)
        if self.crash_blue_penalty > 0.0 and self.n_blue > 1:
            bb = np.linalg.norm(
                self._blue_pos[:, None, :] - self._blue_pos[None, :, :], axis=-1,
            )
            np.fill_diagonal(bb, np.inf)
            blue_ally_crash = np.any(bb <= self.blue_collision_radius, axis=1)

        self._last_obstacle_crashes = int(blue_obstacle_crash.sum())
        self._last_blue_crashes     = int(blue_ally_crash.sum())
        # Per-blue boolean masks for this step (occupancy, not events).
        # A rising edge across steps = one DISTINCT crash event; callers
        # that want an event count (vs the per-step-summed counts above)
        # diff these against the previous step's masks.
        self._last_obstacle_crash_mask = blue_obstacle_crash
        self._last_blue_crash_mask      = blue_ally_crash

        # 6c. Per-agent crash shaping: r_crash_i (applied as a negative).
        r_crash = np.zeros(self.n_blue, dtype=np.float32)
        if self.crash_obstacle_penalty > 0.0:
            r_crash -= self.crash_obstacle_penalty * blue_obstacle_crash
        if self.crash_blue_penalty > 0.0:
            r_crash -= self.crash_blue_penalty * blue_ally_crash

        # 6d. Per-agent CLEARANCE / barrier shaping: r_clear_i (backlog §16).
        #     A dense, smooth penalty for being within clearance_margin of an
        #     obstacle SURFACE.  "Depth into the band" t is 0 at the margin
        #     edge, 1 at the surface, and > 1 INSIDE the disk, so the penalty
        #     keeps growing inward and its position-gradient points OUTWARD
        #     everywhere in range — the proactive "keep clear" and the
        #     "get out, this way" signal.  Summed over obstacles (a blue in a
        #     tight gap feels both).  Off (byte-identical) when weight = 0.
        r_clear = np.zeros(self.n_blue, dtype=np.float32)
        if (self.clearance_weight > 0.0 and self.n_obstacles > 0
                and self._obstacle_pos is not None
                and len(self._obstacle_pos) > 0):
            # (N_blue, N_obs) signed distance to each obstacle surface.
            cd = np.linalg.norm(
                self._blue_pos[:, None, :] - self._obstacle_pos[None, :, :],
                axis=-1,
            ) - self._obstacle_r[None, :]
            t = (self.clearance_margin - cd) / self.clearance_margin  # depth
            t = np.clip(t, 0.0, None)                                 # 0 outside band
            r_clear = -self.clearance_weight * t.sum(axis=1).astype(np.float32)

        # 7. Termination check.
        self._t += 1
        all_caught = bool(not self._red_active.any())
        time_up    = self._t >= self.max_steps
        terminated = all_caught
        truncated  = time_up and not all_caught
        if terminated or truncated:
            n_uncaught = int(self._red_active.sum())
            r_team += -5.0 * n_uncaught                     # terminal penalty (team)
            self.agents = []                                # PettingZoo convention

        # Stage 4: update belief maps AFTER movement + capture so
        # observations reflect the new state.  A caught red will no
        # longer contribute to the enemy channel's ground truth this
        # step, so its cells' P(enemy) will decay from now on.
        if self.use_belief_maps:
            self._update_belief_maps()

        # 8. Pack the per-agent dicts.  r_i = r_team (shared) + r_crash_i +
        #    r_clear_i (individual).  With crash penalties and clearance
        #    weight off, both are 0 and every agent gets the identical shared
        #    reward as before.
        rewards     = {
            a: float(r_team + r_crash[i] + r_clear[i])
            for i, a in enumerate(self.possible_agents)
        }
        terminateds = {a: terminated for a in self.possible_agents}
        truncateds  = {a: truncated  for a in self.possible_agents}
        infos = {
            a: {
                "n_caught_this_step": n_caught_this_step,
                "n_red_remaining":    int(self._red_active.sum()),
                "t":                  self._t,
                "obstacle_crashes":   self._last_obstacle_crashes,
                "blue_crashes":       self._last_blue_crashes,
                "clearance_penalty":  float(-r_clear.sum()),  # >=0 magnitude
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
            # Active reds at episode START (== capacity unless n_red_min set).
            # caught this episode = n_red_start - red_active.sum(); using
            # (~red_active).sum() would miscount padded reds as caught.
            "n_red_start": int(getattr(self, "_n_red_start", self.n_red)),
        }
        # Stage 4: obstacles + belief maps (only when populated).
        if self._obstacle_pos is not None:
            snap["obstacle_pos"] = self._obstacle_pos.copy()
            snap["obstacle_r"]   = self._obstacle_r.copy()
            if self._obstacle_vel is not None:
                snap["obstacle_vel"] = self._obstacle_vel.copy()
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

    def _place_obstacles(self, n_target: Optional[int] = None) -> None:
        """
        Rejection-sample ``n_target`` non-overlapping circular
        obstacles inside the arena (``n_target`` defaults to the full
        ``n_obstacles`` capacity).  Called from ``reset()``.

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
        if n_target is None:
            n_target = self.n_obstacles
        if self.n_obstacles > 0:
            L = self.arena_size
            max_r = self.obstacle_radius_max
            placed_pos: list = []
            placed_r:   list = []

            max_attempts = 200
            for _ in range(n_target):
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

        # ---- Patrol velocities for moving obstacles -----------------------
        # A random subset patrols back-and-forth along ONE axis (x or y),
        # bouncing off the arena walls.  The rest stay static (vel 0).
        n_placed = len(self._obstacle_pos)
        self._obstacle_vel = np.zeros((n_placed, 2), dtype=np.float32)
        if (n_placed > 0 and self.obstacle_speed > 0.0
                and self.moving_obstacle_fraction > 0.0):
            n_move = int(round(self.moving_obstacle_fraction * n_placed))
            if n_move > 0:
                movers = self._rng.choice(n_placed, size=n_move, replace=False)
                for o in movers:
                    axis = int(self._rng.integers(0, 2))          # 0 = x, 1 = y
                    sign = float(self._rng.choice([-1.0, 1.0]))
                    self._obstacle_vel[o, axis] = sign * self.obstacle_speed

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

        # Obstacle-cell mask (W, H) — cached at reset; recomputed each step
        # only when obstacles move (see _recompute_obstacle_grid).
        self._recompute_obstacle_grid()

    def _recompute_obstacle_grid(self) -> None:
        """(Re)build the (W, H) obstacle occupancy mask from the current
        obstacle positions.  Called once at reset (static case) and once
        per step when obstacles move so the belief-truth channel + the
        occlusion truth track the moving disks."""
        H = W = self.belief_grid_size
        if self._obstacle_pos is not None and len(self._obstacle_pos) > 0:
            diffs = self._cell_centres_flat[:, None, :] - self._obstacle_pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)                  # (W*H, n_obs)
            inside = np.any(dists <= self._obstacle_r[None, :], axis=1)
            self._obstacle_grid = inside.reshape(W, H).astype(np.float32)
        else:
            self._obstacle_grid = np.zeros((W, H), dtype=np.float32)

    def _move_obstacles(self) -> None:
        """Advance patrolling obstacles one step: reciprocating motion
        along their axis, bouncing when the disk edge reaches an arena
        wall.  No-op when nothing moves.  Refreshes the obstacle grid."""
        if (self._obstacle_vel is None
                or not np.any(self._obstacle_vel)):
            return
        L = self.arena_size
        self._obstacle_pos = self._obstacle_pos + self._obstacle_vel * self.dt
        # Bounce: keep each disk fully inside [r, L - r]; flip the velocity
        # component that hit a wall.
        lo = self._obstacle_r[:, None]                 # (n_obs, 1)
        hi = L - self._obstacle_r[:, None]
        below = self._obstacle_pos < lo
        above = self._obstacle_pos > hi
        self._obstacle_pos = np.clip(self._obstacle_pos, lo, hi)
        self._obstacle_vel[below | above] *= -1.0
        self._recompute_obstacle_grid()

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each entity whose new position lies inside any obstacle,
        roll the position back to its previous position and zero the
        velocity ("hit an obstacle, stop" soft-crash model).

        Returns (new_pos, vel, collided) -- the boolean (K,) collision
        mask lets the caller apply a per-agent crash penalty (the
        episode is NOT terminated, backlog §1 v1).
        """
        collided = self._positions_in_any_obstacle(new_pos)
        if bool(np.any(collided)):
            new_pos = new_pos.copy()
            vel     = vel.copy()
            new_pos[collided] = prev_pos[collided]
            vel[collided]     = 0.0
        return new_pos, vel, collided

    def _true_occupancy(self) -> np.ndarray:
        """
        Ground-truth occupancy grid of shape ``(2, H, W)`` -- always
        exactly 2 channels: enemy + obstacle.  Used by the critic and
        by the diagnostic BCE.  It does NOT track the deterministic
        overlay channels that may live in the actor's belief tensor
        (ally at channel 2, and the legacy self channel at channel 3
        when belief_channels=4), because those positions are already
        fully known -- the critic reads ally positions via
        ``blue_features`` anyway.
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

    def _rays_occluded_by_obstacles(
        self,
        uav_pos:      np.ndarray,   # (2,)
        cell_centres: np.ndarray,   # (K, 2)
    ) -> np.ndarray:
        """
        EXACT analytic segment-disk occlusion test (no sampling).

        For each ray uav_pos -> cell_centre, returns True iff the ray
        enters any obstacle disk strictly BEFORE reaching the cell —
        specifically, more than ``ray_step_size`` metres before the
        cell centre.  That margin preserves the original semantics:
        the sensor CAN observe the first obstacle boundary cell it
        sees (to accumulate evidence on the obstacle channel); only
        cells buried BEHIND an obstacle surface are hidden.

        Replaces the earlier sampled ray-march, which could miss a
        grazing chord shorter than the sample spacing.

        Geometry per (ray k, obstacle o):
          d      = cell - uav                       (ray vector)
          t_hat  = (c_o - uav)·d / |d|²             (closest approach param)
          perp²  = |c_o - uav|² - (t_hat |d|)²      (line-to-centre dist²)
          the ray intersects the disk iff perp² < r²; the entry point
          parameter is t1 = t_hat - sqrt(r² - perp²)/|d|.
          Occluded iff the intersection interval [t1, t2] overlaps
          (0, t_cut) with t_cut = 1 - margin/|d|.

        Returns
        -------
        occluded : (K,) bool
        """
        K = cell_centres.shape[0]
        if self._obstacle_pos is None or len(self._obstacle_pos) == 0:
            return np.zeros(K, dtype=bool)

        d = cell_centres - uav_pos[None, :]                # (K, 2)
        seg_len = np.linalg.norm(d, axis=-1)               # (K,)
        seg_len = np.maximum(seg_len, 1e-6)

        oc = self._obstacle_pos - uav_pos[None, :]         # (n_obs, 2)
        oc_len2 = np.sum(oc * oc, axis=-1)                 # (n_obs,)
        r = self._obstacle_r                               # (n_obs,)

        # Closest-approach parameter of each obstacle centre along each
        # ray: t_hat (K, n_obs) = (d · oc) / |d|².
        dot = d @ oc.T                                     # (K, n_obs)
        t_hat = dot / (seg_len ** 2)[:, None]

        # Perpendicular (line-to-centre) distance squared.
        proj_len2 = (t_hat * seg_len[:, None]) ** 2        # (K, n_obs)
        perp2 = oc_len2[None, :] - proj_len2               # (K, n_obs)

        disc = r[None, :] ** 2 - perp2                     # (K, n_obs)
        intersects = disc > 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        half_chord_t = sqrt_disc / seg_len[:, None]        # in ray params
        t1 = t_hat - half_chord_t                          # entry
        t2 = t_hat + half_chord_t                          # exit

        # Cut the ray ``ray_step_size`` metres before the cell centre so
        # the first obstacle boundary cell stays observable.
        t_cut = 1.0 - (self.ray_step_size / seg_len)       # (K,)

        blocked = intersects & (t2 > 0.0) & (t1 < t_cut[:, None])
        return np.any(blocked, axis=1)                     # (K,)

    def _predict_enemy_belief(self) -> None:
        """
        Phase A Bayesian PREDICTION step on the enemy channel (channel
        0) of the global belief map.  Runs before the sensor update so
        each step is a proper predict -> update cycle.

        Two enemy-channel components below; the obstacle channel (1) is
        touched only when ``obstacle_belief_decay < 1`` (moving obstacles,
        handled first, decay-only) — for STATIC obstacles its prediction
        is identity and channel 1 is left alone:

        1. DECAY (temporal forgetting):  L <- gamma * L.
           Pulls all log-odds toward 0 (= P 0.5 = "unknown").  Stale
           positive peaks fade (ghost tracks die), and saturated
           negatives relax (previously-"cleared" regions become
           re-acquirable — the enemy could have moved in since).

        2. DIFFUSION (isotropic random-walk motion model).  The
           prediction equation b'(i) = sum_j P(i|j) b(j) is LINEAR IN
           PROBABILITY, not in log-odds, so we convert:
               P = sigmoid(L)  ->  P' = K * P  ->  L = logit(P')
           with a 3x3 kernel derived from p_move
           (= enemy_belief_diffusion):
               centre       1 - p_move
               4-orthogonal p_move * 0.20 each
               4-diagonal   p_move * 0.05 each
           Border handling: edge-replicate padding (the target cannot
           leave the arena; reflecting mass at the walls).

        Effect over unobserved time: a peak fades in confidence while
        spreading into a widening blob around the last sighting —
        the same qualitative behaviour as a full Bayes filter's prior
        under an unknown-heading motion model.
        """
        if self._belief_maps is None:
            return
        gamma  = self.enemy_belief_decay
        p_move = self.enemy_belief_diffusion

        # Obstacle-channel forgetting: when obstacles MOVE, their stale
        # detections leave a "comet trail" of high-confidence cells behind
        # the disk.  A geometric decay toward 0 fades the trail so the peak
        # extractor (memory-track fallback) doesn't lock onto ghosts.  No
        # diffusion (reciprocating motion isn't a random walk); the live-
        # sensor refinement handles near-field accuracy.  Off by default
        # (obstacle_belief_decay = 1.0 -> obstacles treated as static).
        if self.obstacle_belief_decay < 1.0 and self.belief_channels >= 2:
            Lo = self._belief_maps[1]
            Lo *= self.obstacle_belief_decay
            np.clip(Lo, -self.belief_clip, self.belief_clip, out=Lo)

        if gamma >= 1.0 and p_move <= 0.0:
            return

        L = self._belief_maps[0]

        if gamma < 1.0:
            L *= gamma

        if p_move > 0.0:
            # Log-odds -> probability (clamped for logit stability).
            P = 1.0 / (1.0 + np.exp(-L))
            # 3x3 convolution via padded shifts (no scipy dependency).
            Pp = np.pad(P, 1, mode="edge")
            centre = 1.0 - p_move
            w_orth = p_move * 0.20
            w_diag = p_move * 0.05
            P_new = (
                centre * Pp[1:-1, 1:-1]
                + w_orth * (Pp[:-2, 1:-1] + Pp[2:, 1:-1]
                            + Pp[1:-1, :-2] + Pp[1:-1, 2:])
                + w_diag * (Pp[:-2, :-2] + Pp[:-2, 2:]
                            + Pp[2:, :-2] + Pp[2:, 2:])
            )
            eps = 1e-6
            P_new = np.clip(P_new, eps, 1.0 - eps)
            L[:] = np.log(P_new / (1.0 - P_new))

        np.clip(L, -self.belief_clip, self.belief_clip, out=L)

    def _clear_enemy_belief_at(self, positions: np.ndarray) -> None:
        """
        Reset the enemy-channel belief to log-odds 0 ("unknown") in a
        disk around each given position.

        Called on capture: a confirmed kill is a perfect observation —
        the dead target's stale blob must not survive to re-capture a
        track slot in the peak extractor.  We reset to UNKNOWN rather
        than negative because the capture says nothing about OTHER
        reds moving through the same area later.

        Clear radius = capture_radius + 2 cells (the blob may have
        diffused wider than the capture point).
        """
        if self._belief_maps is None or positions.shape[0] == 0:
            return
        clear_r = self.capture_radius + 2.0 * self.belief_cell_size
        d = np.linalg.norm(
            self._cell_centres[:, :, None, :]
            - positions[None, None, :, :], axis=-1,
        )                                   # (W, H, n_pos)
        mask = np.any(d <= clear_r, axis=-1)
        self._belief_maps[0][mask] = 0.0

    def belief_track_error(self) -> float:
        """
        Diagnostic: mean distance (metres) from each extracted enemy
        peak to the nearest ACTIVE red.  Measures how well the belief
        map is tracking vs lagging the true targets.  Returns NaN when
        no reds are active or the belief map is disabled.
        """
        if self._belief_maps is None:
            return float("nan")
        active = np.where(self._red_active)[0]
        if len(active) == 0:
            return float("nan")
        n_act = int(len(active))
        peak_pos, conf = self._extract_belief_peaks(
            n_act, channel_idx=0, k_extract=n_act, nms_radius_cells=2,
        )
        real = conf > 0.0
        if not np.any(real):
            return float("nan")
        d = np.linalg.norm(
            peak_pos[real][:, None, :] - self._red_pos[active][None, :, :],
            axis=-1,
        )                                   # (n_real, n_active)
        return float(d.min(axis=1).mean())

    def _update_belief_maps(self) -> None:
        """
        Vectorised Bayesian log-odds update on the belief map
        ``self._belief_maps`` of shape (C, H, W).

        Doctrine (see docs/stage4_backlog.md §13).  The belief map is
        NOT a per-UAV object; it is the shared operational picture
        maintained by the MISSION COMMAND layer -- an environment-
        level latent state used for target tracking and evaluation,
        not the instantaneous knowledge available to any single UAV.
        Every UAV's sensor evidence is added into the same log-odds
        tensor because Bayesian fusion of independent sensors IS
        log-odds addition, and command has (in the current model)
        access to every UAV's raw returns for fusion regardless of
        whether pairs of UAVs can currently talk to each other.

        This puts the belief map in the same category as
        ``true_occupancy`` -- both are environment-level latents.
        The difference is that ``true_occupancy`` is the ground-truth
        state the CTDE critic sees, whereas the belief map is the
        NOISY track picture command has fused from sensor returns
        (what an operator staring at a track display would see).

        Consequence: the per-step GNN ally-comms mask
        ``bb_edge_visible`` (sensor-radius-gated) is a separate
        thing entirely -- it models per-step UAV-to-UAV messaging,
        not the command-layer fusion.  A UAV out of comms with
        peers can still contribute evidence to the map because
        command receives its raw sensor returns over the C2
        downlink.  Backlog §13b (command-link gating on peaks)
        and §7 (decoupled ``comms_radius``) together open the
        door to studying degraded-network scenarios.

        Channel layout (2 channels):
        - 0: P(enemy)    -- Bayesian log-odds from noisy sensors
        - 1: P(obstacle) -- Bayesian log-odds from noisy sensors
        Ally/self positions are NOT belief channels — they are precise
        and flow through the graph (blue_features / bb_edges).

        Occlusion is the exact analytic segment-disk test
        (``_rays_occluded_by_obstacles``), not a sampled ray-march.
        """
        if self._belief_maps is None or self.sensor_radius is None:
            return

        # Phase A: predict -> update.  Prediction (decay + diffusion on
        # the enemy channel) runs BEFORE the sensor evidence so each
        # step is a proper Bayes-filter cycle.  No-op when both knobs
        # are at their off defaults.
        self._predict_enemy_belief()

        R  = self.sensor_radius
        centres = self._cell_centres                # (W, H, 2)  cached at reset
        C = min(2, self.belief_channels)            # Bayesian channels
        truth = self._true_occupancy()              # (2, W, H) -- always 2

        for i in range(self.n_blue):
            uav_pos = self._blue_pos[i]

            # 1. Cells in sensor disk: (W, H) bool.
            dist_to_cells = np.linalg.norm(centres - uav_pos, axis=-1)
            in_disk = dist_to_cells <= R
            if not bool(np.any(in_disk)):
                continue

            cx_idx, cy_idx = np.where(in_disk)      # (K,) each
            candidate_centres = centres[cx_idx, cy_idx]     # (K, 2)

            # 2. Exact analytic occlusion test across all candidates.
            occluded = self._rays_occluded_by_obstacles(
                uav_pos, candidate_centres,
            )
            visible = ~occluded
            cx_idx = cx_idx[visible]
            cy_idx = cy_idx[visible]
            K = int(cx_idx.shape[0])
            if K == 0:
                continue

            # 3. Vectorised Bernoulli sensor sampling per channel.
            truth_vis = truth[:C, cx_idx, cy_idx]              # (C, K)
            uniforms  = self._rng.random((C, K)).astype(np.float32)
            p_detect  = np.where(
                truth_vis > 0.5,
                np.float32(self.p_TP),
                np.float32(self.p_FP),
            )                                                  # (C, K)
            detected = uniforms < p_detect                     # (C, K) bool
            evidence = np.where(
                detected,
                np.float32(self._L_detect),
                np.float32(self._L_no_detect),
            )                                                  # (C, K)

            # 4. Fuse this UAV's evidence into the SHARED map.
            for c in range(C):
                self._belief_maps[c, cx_idx, cy_idx] += evidence[c]

        # Clip log-odds.
        np.clip(
            self._belief_maps[:C],
            -self.belief_clip, self.belief_clip,
            out=self._belief_maps[:C],
        )

    def _extract_belief_peaks(
        self,
        K: int,
        channel_idx: int = 0,
        k_extract: Optional[int] = None,
        nms_radius_cells: int = 2,
        exclude_cells: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Peak extraction with greedy NON-MAXIMUM SUPPRESSION from one
        channel of the GLOBAL fused belief map.

        Two fixes over the naive top-K-cells version:

        1. NMS: after picking the highest cell, all cells within
           ``nms_radius_cells`` (Chebyshev) of it are suppressed before
           picking the next.  Without this, the 2nd/3rd highest cells
           are almost always NEIGHBOURS of the same strong blob
           (especially with Phase A diffusion widening blobs), so one
           well-tracked target could eat every slot and mask the
           others.  One blob -> one track.

        2. Known entity count: only ``k_extract`` real peaks are
           extracted; the remaining ``K - k_extract`` slots are PADDED
           with confidence 0 (downstream, zero-confidence slots get
           zeroed edges and vis 0 — the Stage 3 caught-red
           convention).  The team legitimately knows how many enemies
           remain: it performed the captures itself.

        Parameters
        ----------
        K : int
          Total slot count (fixed tensor shape: n_red / n_obstacles).
        channel_idx : int
          0 = P(enemy), 1 = P(obstacle).
        k_extract : int, optional
          Number of REAL peaks to extract (e.g. number of still-active
          reds).  Defaults to K.
        nms_radius_cells : int
          Chebyshev suppression radius, in cells.
        exclude_cells : list of (cx, cy), optional
          Cells to pre-seed the suppression set with — peaks within
          ``nms_radius_cells`` of any excluded cell are skipped.  Used
          so belief (memory) peaks do not duplicate live detection
          tracks whose reds are already occupying a slot.

        Returns
        -------
        peak_pos : (K, 2) float32 — peak cell centres in WORLD coords
          (padded slots hold (0, 0), but their conf is 0 so consumers
          gate them out).
        conf     : (K,)   float32 — sigmoid(log_odds) per slot, real
          peaks sorted descending, padded slots 0.
        """
        assert self._belief_maps is not None
        assert 0 <= channel_idx < self._belief_maps.shape[0]
        if k_extract is None:
            k_extract = K
        k_extract = max(0, min(int(k_extract), K))
        cs = self.belief_cell_size
        H = W = self.belief_grid_size

        chan_lo = self._belief_maps[channel_idx]            # (H, W)
        chan_p  = 1.0 / (1.0 + np.exp(-chan_lo))
        flat = chan_p.reshape(-1)                           # (H*W,)

        peak_pos = np.zeros((K, 2), dtype=np.float32)
        conf     = np.zeros((K,),   dtype=np.float32)

        if k_extract > 0:
            order = np.argsort(-flat)                       # desc
            # Seed the suppression set with the excluded cells so their
            # neighbourhoods are skipped (but they do NOT consume slots).
            suppress = list(exclude_cells) if exclude_cells else []
            picked_cells: list = []
            for idx in order:
                cx = int(idx) // W
                cy = int(idx) %  W
                if any(max(abs(cx - px), abs(cy - py)) <= nms_radius_cells
                       for (px, py) in suppress):
                    continue
                s = len(picked_cells)
                peak_pos[s, 0] = (cx + 0.5) * cs
                peak_pos[s, 1] = (cy + 0.5) * cs
                conf[s] = flat[idx]
                picked_cells.append((cx, cy))
                suppress.append((cx, cy))
                if len(picked_cells) == k_extract:
                    break

        return peak_pos, conf

    def _build_enemy_tracks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Populate the K = n_red enemy track slots DETECTION-FIRST.

        Doctrine: a directly-observed target is the strongest, cleanest
        signal — it must never wait on, or be filtered by, the memory
        (belief-map) layer.  So:

        1. LIVE tracks — one slot per active red that ANY blue currently
           sees, seeded directly from that red (fused measured position,
           conf 1, backing red index known).  No peak->red association:
           the slot IS the red.  This removes the many-to-one collapse,
           the belief-map lag, and the NMS-collapse of visible targets.
        2. MEMORY tracks — belief-map peaks for the reds NOBODY sees,
           with the live-track cells excluded so a visible red's
           residual blob cannot also spawn a phantom memory track.
        3. Remaining slots stay conf 0 (dead / padding).

        Returns
        -------
        track_pos : (K, 2) float32 — world position (measured for live,
          cell centre for memory, (0,0) for padding).
        track_conf: (K,)   float32 — 1.0 live, peak conf memory, 0 pad.
        track_red : (K,)   int32   — backing red index for LIVE tracks,
          -1 for memory / padding (never measured).
        """
        K = self.n_red
        track_pos  = np.zeros((K, 2), dtype=np.float32)
        track_conf = np.zeros((K,),   dtype=np.float32)
        track_red  = np.full((K,), -1, dtype=np.int32)

        active = np.where(self._red_active)[0]
        if len(active) == 0 or self._belief_maps is None:
            return track_pos, track_conf, track_red

        # Which active reds does ANY blue currently see?
        if self.sensor_radius is None:
            vis_mask = np.ones(len(active), dtype=bool)
        else:
            d = np.linalg.norm(
                self._red_pos[active][:, None, :]
                - self._blue_pos[None, :, :], axis=-1,
            )                                   # (n_active, n_blue)
            vis_mask = d.min(axis=1) <= self.sensor_radius
        vis_reds    = active[vis_mask]
        unseen_reds = active[~vis_mask]

        cs = self.belief_cell_size
        slot = 0

        # 1. Live tracks: fused measured position (true + noise).
        for r in vis_reds:
            if slot >= K:
                break
            pos = self._red_pos[r].astype(np.float32)
            if self.sensor_pos_noise_std > 0.0:
                pos = pos + self._rng.normal(
                    0.0, self.sensor_pos_noise_std, size=2,
                ).astype(np.float32)
            track_pos[slot]  = pos
            track_conf[slot] = 1.0
            track_red[slot]  = int(r)
            slot += 1

        # 2. Memory tracks: belief peaks for the unseen reds, excluding
        #    cells already claimed by live tracks.
        n_mem = min(len(unseen_reds), K - slot)
        if n_mem > 0:
            exclude = [
                (int(track_pos[s, 0] / cs), int(track_pos[s, 1] / cs))
                for s in range(slot)
            ]
            peaks_pos, peaks_conf = self._extract_belief_peaks(
                n_mem, channel_idx=0, k_extract=n_mem,
                nms_radius_cells=2, exclude_cells=exclude,
            )
            for i in range(n_mem):
                if peaks_conf[i] <= 0.0:
                    continue
                track_pos[slot]  = peaks_pos[i]
                track_conf[slot] = peaks_conf[i]
                track_red[slot]  = -1        # memory: never measured
                slot += 1

        return track_pos, track_conf, track_red

    def _enemy_graph_from_tracks(
        self,
        track_pos:  np.ndarray,   # (K, 2)
        track_conf: np.ndarray,   # (K,)
        track_red:  np.ndarray,   # (K,) int; -1 = memory / pad
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the actor-side enemy node + edge features from the K
        detection-seeded tracks.  Same 7-D edge layout / ordering as
        ``_obstacle_graph_from_tracks`` (for s in K, for b in N).

        Per (track s, blue b): POSITION is always ``track_pos[s]`` -- the
        single shared/fused track position (measured+noise for a live
        track, belief cell centre for a memory track).  VELOCITY is the
        only per-blue difference:
        - live track (track_red[s] >= 0) whose red is within b's sensor
          range -> own-sensor Doppler velocity.
        - otherwise (out of range, or memory / padding) -> zero velocity.

        Doctrine — where these positions and velocities come from
        (see docs/stage4_backlog.md §13).  The track positions are
        drawn from the belief map, which lives at the MISSION COMMAND
        layer -- not per-UAV.  What each UAV receives is command's
        fused track output plus, when in sensor range, its OWN radar
        Doppler measurement of the same target:

        * POSITION for every blue = the shared command-layer track
          (measured+noise for a live track, belief-map cell centre for
          a memory track).  We model the C2 downlink of track updates
          as always-on for the actor; backlog §13b would gate this
          per-UAV via ``command_link_visible`` for a degraded-C2
          study.
        * VELOCITY = your OWN radar's Doppler if you can see the
          target now, else 0.  Doppler is a first-person measurement
          that command cannot deliver to a UAV that lacks its own
          sensor lock.

        So the realistic split is: "command shares POSITION with
        everyone; VELOCITY is what your own radar tells you right now".

        Terminology aside -- TDL = Tactical Data Link (Link 16, Link
        22, SADL, ...), the standardised military radio networks that
        carry the common tactical picture.  Earlier docstrings said
        "position broadcast over TDL"; strictly speaking, tracks
        travel over the C2 downlink from command to each UAV; the
        UAV-to-UAV TDL is a separate channel modelled here by
        ``bb_edge_visible`` (per-step GNN messaging, sensor-radius-
        gated).  Backlog §7 tracks decoupling the messaging range
        into its own ``comms_radius`` knob.
        """
        N = self.n_blue
        K = track_pos.shape[0]
        n_edges = K * N

        src_pos = np.zeros((n_edges, 2), dtype=np.float32)
        src_vel = np.zeros((n_edges, 2), dtype=np.float32)
        dst_pos = np.zeros((n_edges, 2), dtype=np.float32)
        dst_vel = np.zeros((n_edges, 2), dtype=np.float32)
        edge_vis = np.zeros((n_edges,), dtype=np.float32)

        # POSITION is always the ONE shared/fused track position (the
        # value seeded in _build_enemy_tracks: measured+noise for a live
        # track, belief cell centre for a memory track).  The ONLY
        # per-blue difference is VELOCITY: a blue that can see the
        # backing red gets its own-sensor Doppler; everyone else gets 0.
        # This keeps the TDL doctrine exact -- position is the shared
        # network track, velocity is own-sensor -- with no second,
        # inconsistent position draw.
        for s in range(K):
            r = int(track_red[s])
            for b in range(N):
                e = s * N + b
                src_pos[e] = track_pos[s]
                dst_pos[e] = self._blue_pos[b]
                dst_vel[e] = self._blue_vel[b]
                edge_vis[e] = track_conf[s]
                if r >= 0 and (
                    self.sensor_radius is None
                    or np.linalg.norm(self._red_pos[r] - self._blue_pos[b])
                    <= self.sensor_radius
                ):
                    # This blue sees the backing red -> own-sensor Doppler.
                    src_vel[e] = self._red_vel[r]
                # else: memory / out-of-range -> velocity stays 0.

        edge_feat = self._edge_features_for(src_pos, src_vel, dst_pos, dst_vel)
        edge_feat[edge_vis <= 0.0] = 0.0     # zero padded/dead slots
        node_feat = track_conf.reshape(K, 1).astype(np.float32)
        return node_feat, edge_feat, edge_vis

    def _build_obstacle_tracks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Actor-side obstacle "tracks", mirroring ``_build_enemy_tracks``:
        for every obstacle a blue currently senses, use the precise
        OWN-RADAR position (true centre + ``sensor_pos_noise_std``
        noise); for the rest, fall back to the coarse belief-map peak
        (command's memory of the surveyed obstacle field).

        Why refine at all when obstacles are static?  Under the crash
        penalty, a blue that wants to graze an obstacle to cut a corner
        needs the boundary to sub-cell accuracy; the grid-quantised peak
        (~half a belief cell of error) forces a conservative safety
        margin it cannot otherwise resolve.  A live radar return removes
        that quantisation exactly when the blue is close enough for it to
        matter.  Velocity stays 0 (static), so only POSITION improves.

        Doctrine (same as the enemy tracks, backlog §13): the refined
        position is a shared command-layer track — if ANY blue senses
        the obstacle it becomes a "live track" downlinked to every blue,
        else it is command memory.  Range-gated only (no occlusion
        test), matching ``_build_enemy_tracks``.

        Returns ``(obs_pos (n_obstacles, 2), obs_vel (n_obstacles, 2),
        obs_conf (n_obstacles,))`` in the same slot convention
        ``_extract_belief_peaks`` uses: real tracks first, unplaced slots
        padded with conf 0.  ``obs_vel`` is the obstacle's own-radar
        velocity when it is a live (seen) track, else 0 (a memory track
        carries no Doppler) — static obstacles are simply always 0.
        """
        n_obs  = self.n_obstacles
        obs_pos  = np.zeros((n_obs, 2), dtype=np.float32)
        obs_vel  = np.zeros((n_obs, 2), dtype=np.float32)
        obs_conf = np.zeros((n_obs,),   dtype=np.float32)
        placed = 0 if self._obstacle_pos is None else len(self._obstacle_pos)
        if placed == 0:
            return obs_pos, obs_vel, obs_conf

        # Which placed obstacles does ANY blue currently see?  Distance-
        # only gate, exactly as the enemy track builder does.
        idx = np.arange(placed)
        if self.sensor_radius is None:
            seen_mask = np.ones(placed, dtype=bool)
        else:
            d = np.linalg.norm(
                self._obstacle_pos[:placed][:, None, :]
                - self._blue_pos[None, :, :], axis=-1,
            )                                    # (placed, n_blue)
            seen_mask = d.min(axis=1) <= self.sensor_radius
        seen   = idx[seen_mask]
        unseen = idx[~seen_mask]

        cs = self.belief_cell_size
        slot = 0

        # 1. Live tracks: precise own-sensor position (true + noise).
        for o in seen:
            if slot >= n_obs:
                break
            pos = self._obstacle_pos[o].astype(np.float32)
            if self.sensor_pos_noise_std > 0.0:
                pos = pos + self._rng.normal(
                    0.0, self.sensor_pos_noise_std, size=2,
                ).astype(np.float32)
            obs_pos[slot]  = pos
            if self._obstacle_vel is not None:
                obs_vel[slot] = self._obstacle_vel[o]   # own-radar Doppler
            obs_conf[slot] = 1.0
            slot += 1

        # 2. Memory tracks: belief peaks for the unseen obstacles,
        #    excluding cells already claimed by a live track.
        n_mem = min(len(unseen), n_obs - slot)
        if (n_mem > 0 and self._belief_maps is not None
                and self.belief_channels >= 2):
            exclude = [
                (int(obs_pos[s, 0] / cs), int(obs_pos[s, 1] / cs))
                for s in range(slot)
            ]
            peaks_pos, peaks_conf = self._extract_belief_peaks(
                n_mem, channel_idx=1, k_extract=n_mem,
                nms_radius_cells=3, exclude_cells=exclude,
            )
            for i in range(n_mem):
                if peaks_conf[i] <= 0.0:
                    continue
                obs_pos[slot]  = peaks_pos[i]
                obs_conf[slot] = peaks_conf[i]     # memory track: vel stays 0
                slot += 1

        return obs_pos, obs_vel, obs_conf

    def _obstacle_graph_from_tracks(
        self,
        track_pos: np.ndarray,   # (K, 2) world coords (shared track picture)
        track_vel: np.ndarray,   # (K, 2) obstacle velocity (0 unless moving+seen)
        conf:      np.ndarray,   # (K,)   track confidences
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Actor-side OBSTACLE node + edge features from the obstacle tracks
        (``_build_obstacle_tracks``): a live own-sensor position when a
        blue senses the obstacle, else the belief-map peak.  For a MOVING
        obstacle the live track also carries its own-radar velocity so the
        ``rel_vel`` edge feature lets blues anticipate the sweep; static
        (or memory-track) obstacles keep velocity 0.  Same 7-D layout /
        (s outer, b inner) ordering as the enemy graph.
        """
        N = self.n_blue
        K = track_pos.shape[0]
        n_edges = K * N

        src_pos = np.zeros((n_edges, 2), dtype=np.float32)
        src_vel = np.zeros((n_edges, 2), dtype=np.float32)
        dst_pos = np.zeros((n_edges, 2), dtype=np.float32)
        dst_vel = np.zeros((n_edges, 2), dtype=np.float32)
        edge_vis = np.zeros((n_edges,), dtype=np.float32)

        for s in range(K):
            for b in range(N):
                e = s * N + b
                src_pos[e] = track_pos[s]
                src_vel[e] = track_vel[s]
                dst_pos[e] = self._blue_pos[b]
                dst_vel[e] = self._blue_vel[b]
                edge_vis[e] = conf[s]

        edge_feat = self._edge_features_for(src_pos, src_vel, dst_pos, dst_vel)
        edge_feat[edge_vis <= 0.0] = 0.0
        node_feat = conf.reshape(K, 1).astype(np.float32)
        return node_feat, edge_feat, edge_vis

    def _true_obstacle_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ground-truth obstacle node + edge features for the CTDE critic.

        Node count is fixed at ``self.n_obstacles`` (the configured
        value) so tensor shapes are stable across a run even when
        rejection sampling places fewer.  Missing slots are inactive
        (node feature 0, zero edges).  Obstacles are static -> object
        velocity 0.
        """
        n_obs = self.n_obstacles
        N = self.n_blue
        placed = 0 if self._obstacle_pos is None else len(self._obstacle_pos)

        node_feat = np.zeros((n_obs, 1), dtype=np.float32)
        n_edges = n_obs * N
        src_pos = np.zeros((n_edges, 2), dtype=np.float32)
        src_vel = np.zeros((n_edges, 2), dtype=np.float32)   # true obstacle vel
        dst_pos = np.zeros((n_edges, 2), dtype=np.float32)
        dst_vel = np.zeros((n_edges, 2), dtype=np.float32)

        for o in range(n_obs):
            has = o < placed
            if has:
                node_feat[o, 0] = 1.0
            for b in range(N):
                e = o * N + b
                dst_pos[e] = self._blue_pos[b]
                dst_vel[e] = self._blue_vel[b]
                # src_pos = obstacle centre if placed; else = blue pos
                # (rel_pos 0) so the padded edge is neutral before zeroing.
                src_pos[e] = self._obstacle_pos[o] if has else self._blue_pos[b]
                # True obstacle velocity (0 for static; the critic gets it
                # exactly, no Doppler gating).
                if has and self._obstacle_vel is not None:
                    src_vel[e] = self._obstacle_vel[o]

        edge_feat = self._edge_features_for(src_pos, src_vel, dst_pos, dst_vel)
        # Zero the edges of padded (unplaced) obstacle slots.
        for o in range(n_obs):
            if o >= placed:
                edge_feat[o * N:(o + 1) * N] = 0.0
        return node_feat, edge_feat

    def structured_belief_observation(self) -> Dict[str, np.ndarray]:
        """
        Stage 4 (v6) observation dict — the proven Stage 3 typed-GNN
        graph, extended with obstacle nodes.  No CNN, no raw belief
        tensor fed to the policy.

        The ACTOR consumes a belief-derived graph: enemy/obstacle
        positions come from top-K PEAK detections on the shared
        command-layer belief map (noisy, see doctrine note on
        ``_update_belief_maps`` and backlog §13); enemy edge velocity
        from radar (precise when the associated red is in that blue's
        sensor range, else 0); obstacle velocity is the own-radar Doppler
        when a moving obstacle is seen, else 0 (static / memory track).
        Per-edge visibility = track confidence.

        The CRITIC (CTDE) consumes the ground-truth graph: true red and
        obstacle node/edge features, no masks.

        Keys
        ----
        Shared precise:
        - ``blue_features``     (N_blue, 8)
        - ``bb_edge_features``  (n_bb, 7),  ``bb_edge_visible`` (n_bb,)

        Actor (belief-derived):
        - ``red_features``      (N_red, 1),  ``rb_edge_features`` (n_rb, 7),
          ``rb_edge_visible`` (n_rb,)
        - ``obstacle_features`` (N_obs, 1),  ``ob_edge_features`` (n_ob, 7),
          ``ob_edge_visible`` (n_ob,)         [only if n_obstacles > 0]

        Critic (ground truth):
        - ``true_red_features`` (N_red, 1),  ``true_rb_edge_features`` (n_rb, 7)
        - ``true_obstacle_features`` (N_obs, 1),
          ``true_ob_edge_features`` (n_ob, 7)  [only if n_obstacles > 0]

        Diagnostics (not consumed by the policy):
        - ``belief_maps`` (C, H, W) global fused, ``true_occupancy`` (2, H, W)
        """
        base = self._build_structured_obs()   # true blue/red/bb/rb + masks
        bb_vis, rb_vis_true = self._compute_edge_visibility()

        out: Dict[str, np.ndarray] = {
            # Shared precise.
            "blue_features":    base["blue_features"],
            "bb_edge_features": base["bb_edge_features"],
            "bb_edge_visible":  bb_vis,
            # Critic ground truth (reds).
            "true_red_features":     base["red_features"],
            "true_rb_edge_features": base["rb_edge_features"],
        }

        # ---- Actor enemy graph: DETECTION-SEEDED tracks ------------------
        # Live tracks (visible reds) seeded directly from the sensor;
        # remaining slots filled from belief-map memory peaks.  See
        # _build_enemy_tracks.  Dead slots are conf-0 padded.
        track_pos, track_conf, track_red = self._build_enemy_tracks()
        red_node, rb_edge, rb_vis = self._enemy_graph_from_tracks(
            track_pos, track_conf, track_red,
        )
        out["red_features"]    = red_node
        out["rb_edge_features"] = rb_edge
        out["rb_edge_visible"]  = rb_vis

        # ---- Obstacle graph (only when obstacles are configured) ---------
        if self.n_obstacles > 0:
            # Live own-sensor position when a blue senses the obstacle,
            # else the belief-map peak (command memory).  See
            # _build_obstacle_tracks for the doctrine.
            obs_pos, obs_vel, obs_conf = self._build_obstacle_tracks()
            ob_node, ob_edge, ob_vis = self._obstacle_graph_from_tracks(
                obs_pos, obs_vel, obs_conf,
            )
            out["obstacle_features"] = ob_node
            out["ob_edge_features"]  = ob_edge
            out["ob_edge_visible"]   = ob_vis
            # Critic ground truth (obstacles).
            true_ob_node, true_ob_edge = self._true_obstacle_graph()
            out["true_obstacle_features"] = true_ob_node
            out["true_ob_edge_features"]  = true_ob_edge

        # ---- Diagnostics (telemetry; NOT fed to the policy) --------------
        if self._belief_maps is not None:
            out["belief_maps"] = self._belief_maps.copy()
        else:
            out["belief_maps"] = np.zeros(
                (self.belief_channels,
                 self.belief_grid_size, self.belief_grid_size),
                dtype=np.float32,
            )
        out["true_occupancy"] = self._true_occupancy()
        return out

