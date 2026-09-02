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

from typing import Any, Callable, Dict, List, Optional, Tuple

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
        # ----- Reward shape (was hard-coded in _step) ---------------------
        #   r_i = r_team + r_crash_i + r_clear_i - action_cost_i
        #   r_team = catch_reward * n_caught - step_cost
        #            [- uncaught_penalty * n_uncaught at episode end]
        # SHARED terms are mission-level (a catch is a team success, time
        # pressure applies to everyone); control effort is INDIVIDUAL.
        # Defaults are the historical hard-coded values.
        catch_reward:             float = 10.0,
        step_cost:                float = 0.05,
        uncaught_penalty:         float = 5.0,
        action_cost_coef:         float = 0.01,
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
        # ----- Per-channel sensor quality (backlog §8) --------------------
        # A concrete 15 m obstacle and a small, actively-evading drone are
        # NOT equally detectable, yet both channels shared one (p_TP, p_FP).
        # This became load-bearing once live tracks started obeying p_TP:
        # obstacles were dropping out of live tracking ~15% of scans, which
        # flickers their perceived SURFACE (what clearance shaping keys on).
        # Each defaults to None -> fall back to the shared p_TP / p_FP, so
        # callers that pass only the base pair are byte-preserved.
        # Channel 0 = enemy, channel 1 = obstacle.
        p_TP_enemy:               Optional[float] = None,
        p_FP_enemy:               Optional[float] = None,
        p_TP_obstacle:            Optional[float] = None,
        p_FP_obstacle:            Optional[float] = None,
        ray_step_size:            float = 2.5,
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
        # ----- Live-track sensor realism ----------------------------------
        # A live track comes from the SAME radar that feeds the belief map,
        # so it must obey the same detection chain: detect (p_TP) -> measure.
        # Previously the track gate was range-only, so a target inside
        # sensor_radius produced a conf-1.0 measurement EVERY step and even
        # THROUGH obstacles, while the belief map applied p_TP/p_FP and an
        # exact occlusion test off the same sensor.
        #   track_occlusion  — require clear line of sight for a live track.
        #   track_detection  — require the p_TP draw to fire; on a miss there
        #                      is NO measurement and the track coasts on the
        #                      belief/memory path (as an unseen target does).
        # Both default True (physically coherent).  Set False to reproduce
        # pre-fix runs.
        track_occlusion:          bool  = True,
        track_detection:          bool  = True,
        # Doppler (radial-velocity) measurement noise.  Real radar measures
        # velocity from phase, far more precisely than differencing noisy
        # positions would allow, so keep this SMALL (~0.1) relative to
        # sensor_pos_noise_std.  0.0 = perfect velocity (pre-fix).
        sensor_vel_noise_std:     float = 0.0,
        # Live-track confidence floor at max sensor range (SNR proxy: return
        # power falls ~R^-4, so a distant track is a weaker, less trustworthy
        # return).  conf(r) = c_min + (1-c_min)(1-(r/R)^2), using the range to
        # the NEAREST detecting blue (best SNR wins).  Depends ONLY on range —
        # never on whether the target is real, which would leak ground truth.
        # 1.0 (default) = flat conf 1.0 everywhere (pre-fix).
        track_conf_min:           float = 1.0,
        # Range growth of MEASUREMENT NOISE — the other half of the same SNR
        # story as track_conf_min: if a distant return is less trustworthy,
        # it must also be less ACCURATE, or the model is incoherent.  Real
        # cross-range error = R * sigma_angle with sigma_angle ~ 1/sqrt(SNR),
        # so error grows steeply with range; this exposes it as a tunable
        # multiplier rather than the raw exponent:
        #     sigma(r) = sigma_base * (1 + growth * (r/R)^2)
        # 0.0 (default) = range-independent noise (pre-fix).  1.0 = noise
        # doubles at max sensor range.  Applies to BOTH position and Doppler.
        sensor_noise_range_growth: float = 0.0,
        # Gaussian speed prior used as the ridge term in the command-layer
        # velocity fusion (see _fuse_radial_velocity).  Physically "a target
        # cannot outrun this speed", so it should be ~the fastest entity's
        # v_max.  It makes the fusion solvable for ANY detector geometry and
        # supplies the honest fallback in an unobserved direction.
        vel_prior_std:            float = 1.0,
        # False returns (clutter) in raw_detections(), for the TRACK path
        # only.  Modelled at the PLOT level, not the cell level: the belief
        # map's per-cell p_FP would inject ~one false return per resolution
        # cell (~200 in a 40 m disk), but a real radar's plot extractor
        # (CFAR + clustering) already suppresses most cell-level false
        # alarms, so plot clutter is far sparser.  Poisson mean per blue per
        # scan; each false plot is uniform in that blue's sensor disk (and
        # occlusion-gated, like a real return) with a meaningless Doppler.
        # 0.0 (default) = no clutter, so raw_detections stays byte-identical.
        clutter_rate:             float = 0.0,
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
        clearance_margin:         float = 4.0,   # m band beyond the surface
        # Blue<->blue barrier: same falloff, but the "surface" is the
        # blue_collision_radius.  Needed alongside the obstacle term — with
        # obstacle clearance alone, blues avoiding obstacles crowd into the
        # same clear space and just MIGRATE crashes obstacle->ally (observed
        # on clearance_fixed_v1).  A tighter margin than obstacles so it does
        # not fight normal converge-on-a-red coordination.  0 = off.
        clearance_ally_weight:    float = 0.0,   # per-step magnitude; 0 = off
        clearance_ally_margin:    float = 3.0,   # m band beyond the collision radius
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
        # Per-channel resolution (None -> the shared base pair).
        self.p_TP_enemy    = float(p_TP if p_TP_enemy    is None else p_TP_enemy)
        self.p_FP_enemy    = float(p_FP if p_FP_enemy    is None else p_FP_enemy)
        self.p_TP_obstacle = float(p_TP if p_TP_obstacle is None else p_TP_obstacle)
        self.p_FP_obstacle = float(p_FP if p_FP_obstacle is None else p_FP_obstacle)
        self.catch_reward             = float(catch_reward)
        self.step_cost                = float(step_cost)
        self.uncaught_penalty         = float(uncaught_penalty)
        self.action_cost_coef         = float(action_cost_coef)
        assert self.action_cost_coef >= 0.0
        self.ray_step_size            = float(ray_step_size)
        self.track_occlusion          = bool(track_occlusion)
        self.track_detection          = bool(track_detection)
        self.sensor_vel_noise_std     = float(sensor_vel_noise_std)
        self.track_conf_min           = float(track_conf_min)
        self.sensor_noise_range_growth = float(sensor_noise_range_growth)
        self.vel_prior_std             = float(vel_prior_std)
        self.clutter_rate              = float(clutter_rate)
        assert self.vel_prior_std > 0.0
        assert self.clutter_rate >= 0.0
        assert self.sensor_vel_noise_std >= 0.0
        assert self.sensor_noise_range_growth >= 0.0
        assert 0.0 < self.track_conf_min <= 1.0, (
            "track_conf_min must be in (0, 1]; 0 is reserved for dead/padded "
            "track slots")
        # Per-(blue, target) detection masks for THIS step, set by the track
        # builders and read by _fuse_radial_velocity to pick which platforms
        # contribute a radial measurement.  Also the diagnostic the sensor
        # tests assert on to check the occlusion / p_TP gating.
        self._last_red_detect: Optional[np.ndarray] = None   # (n_blue, n_red)
        self._last_obs_detect: Optional[np.ndarray] = None   # (n_blue, n_obs)
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
        self.clearance_ally_weight    = float(clearance_ally_weight)
        self.clearance_ally_margin    = float(clearance_ally_margin)
        assert self.crash_obstacle_penalty >= 0.0
        assert self.crash_blue_penalty >= 0.0
        assert self.blue_collision_radius >= 0.0
        assert self.clearance_weight >= 0.0
        assert self.clearance_margin > 0.0
        assert self.clearance_ally_weight >= 0.0
        assert self.clearance_ally_margin > 0.0
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
        # Per-channel versions (index 0 = enemy, 1 = obstacle) used by the
        # belief update; the scalars above stay for the shared base pair.
        tp_ch = np.array([self.p_TP_enemy, self.p_TP_obstacle], dtype=np.float64)
        fp_ch = np.array([self.p_FP_enemy, self.p_FP_obstacle], dtype=np.float64)
        tp_ch = np.clip(tp_ch, _eps, 1.0 - _eps)
        fp_ch = np.clip(fp_ch, _eps, 1.0 - _eps)
        self._p_TP_ch = tp_ch.astype(np.float32)
        self._p_FP_ch = fp_ch.astype(np.float32)
        self._L_detect_ch    = np.log(tp_ch / fp_ch).astype(np.float32)
        self._L_no_detect_ch = np.log(
            (1.0 - tp_ch) / (1.0 - fp_ch)).astype(np.float32)

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

        # A STATEFUL red policy (e.g. StochasticRed, whose noise is
        # temporally correlated and whose side-commitments persist across
        # steps) must not carry that state into the next episode.  Duck-typed
        # so plain function policies are untouched.
        if hasattr(self.red_policy, "reset"):
            self.red_policy.reset(self.n_red)

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
        catch_bonus = self.catch_reward * n_caught_this_step
        step_cost   = self.step_cost
        r_team = catch_bonus - step_cost

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

        # 6b'. Per-agent CONTROL EFFORT: r_action_i.  Control effort is a
        #      first-person cost (your own actuators), so each blue pays for
        #      its OWN action only — it used to be summed over the whole team
        #      and charged to everyone via r_team, so ~81% of what an agent
        #      "paid" was its teammates' manoeuvring.  The expected policy
        #      gradient w.r.t. a_i is UNCHANGED (d/da_i of the teammates'
        #      terms is 0), so this is pure variance reduction, not a change
        #      of objective.  It also removes an n_blue dependence: the old
        #      form scaled the penalty each agent felt with team size, which
        #      fought the count-agnostic design.
        action_cost = (self.action_cost_coef
                       * np.sum(blue_a ** 2, axis=1)).astype(np.float32)

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
            r_clear -= self.clearance_weight * t.sum(axis=1).astype(np.float32)

        # 6e. Blue<->blue CLEARANCE: the same barrier, "surface" = the
        #     collision radius.  Without it, obstacle clearance alone just
        #     bunches blues together and migrates crashes obstacle->ally.
        #     Symmetric (both blues in a close pair feel it); self-pairs
        #     excluded.  Off (byte-identical) when weight = 0.
        if self.clearance_ally_weight > 0.0 and self.n_blue > 1:
            bb = np.linalg.norm(
                self._blue_pos[:, None, :] - self._blue_pos[None, :, :], axis=-1,
            ) - self.blue_collision_radius            # signed dist to collision
            ta = (self.clearance_ally_margin - bb) / self.clearance_ally_margin
            ta = np.clip(ta, 0.0, None)
            np.fill_diagonal(ta, 0.0)                 # exclude self-pair
            r_clear -= self.clearance_ally_weight * ta.sum(axis=1).astype(np.float32)

        # 7. Termination check.
        self._t += 1
        all_caught = bool(not self._red_active.any())
        time_up    = self._t >= self.max_steps
        terminated = all_caught
        truncated  = time_up and not all_caught
        if terminated or truncated:
            n_uncaught = int(self._red_active.sum())
            r_team += -self.uncaught_penalty * n_uncaught   # terminal penalty (team)
            self.agents = []                                # PettingZoo convention

        # Stage 4: update belief maps AFTER movement + capture so
        # observations reflect the new state.  A caught red will no
        # longer contribute to the enemy channel's ground truth this
        # step, so its cells' P(enemy) will decay from now on.
        if self.use_belief_maps:
            self._update_belief_maps()

        # 8. Pack the per-agent dicts:
        #        r_i = r_team + r_crash_i + r_clear_i - action_cost_i
        #    SHARED (r_team): catch bonus + step cost — the mission-level
        #    terms that genuinely apply to the whole team.
        #    INDIVIDUAL: crash, clearance, and control effort — all
        #    first-person quantities.  Agents therefore differ whenever any
        #    of them acts (control effort is never identically zero in
        #    training), so consumers must NOT assume a shared scalar; take
        #    the mean over agents for a team-level figure.
        rewards     = {
            a: float(r_team + r_crash[i] + r_clear[i] - action_cost[i])
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
    #  Raw sensor returns (identity-free) — input to a real tracker       #
    # ------------------------------------------------------------------ #

    def raw_detections(self) -> List[Dict[str, Any]]:
        """
        This step's sensor returns, WITHOUT target identity.

        Why this exists: ``_build_enemy_tracks`` currently loops over TRUE red
        indices, measures ``self._red_pos[r]``, and groups returns for fusion
        by ``detect[:, r]``.  That means the simulator hands the policy a
        perfect data association — which detection belongs to which target,
        and a slot->target mapping that is stable across steps.  Real radar
        gives none of that: a return is a position and a Doppler, with no
        label.  Establishing identity IS the tracking problem.

        This method emits exactly what a real sensor would, so an actual
        tracker (association + filtering) can be built and measured against
        ground truth.  It runs the SAME detection chain as the track builder
        (range gate -> line of sight -> p_TP draw) and applies the same
        range-scaled measurement noise, so the two are directly comparable.

        Each detection is a dict:
            blue        int   — the observing UAV (known: it is ours)
            z_pos       (2,)  — measured position (true + range-scaled noise)
            z_range     float — measured range to the observer
            z_radial    float — radial (Doppler) speed along the LOS, noisy
            los         (2,)  — unit line of sight, observer -> detection
            sigma_pos   float — 1-sigma position noise for THIS return
            sigma_radial float — 1-sigma Doppler noise for THIS return
            truth_id    int   — TRUE target index for a real return; -1 for a
                                CLUTTER (false) return.  For EVALUATION ONLY
                                (MOT metrics / association scoring).  A
                                tracker must never read this.

        Clutter: when ``clutter_rate > 0``, each blue also emits Poisson-many
        FALSE returns per scan, uniform in its sensor disk, occlusion-gated
        like a real return, carrying a meaningless (random) Doppler.  These
        make association HARD — without them every return belongs to some
        real target and the matcher barely errs, so the gate / M-of-N /
        birth logic looks better than it is.  Clutter is at the PLOT level,
        not the belief map's per-cell level (see ``clutter_rate``).
        """
        out: List[Dict[str, Any]] = []
        if self._red_pos is None:
            return out

        active = np.where(self._red_active)[0]
        red_act = self._red_pos[active] if len(active) else np.zeros((0, 2))

        for b in range(self.n_blue):
            bp = self._blue_pos[b]

            # --- real returns -------------------------------------------
            if len(active):
                d = np.linalg.norm(red_act - bp[None, :], axis=1)
                ok = (np.ones(len(active), dtype=bool)
                      if self.sensor_radius is None else d <= self.sensor_radius)
                if self.track_occlusion and ok.any():
                    ok &= ~self._rays_occluded_by_obstacles(bp, red_act)
                if self.track_detection and ok.any():
                    ok &= self._rng.random(len(active)) < self.p_TP_enemy
                for j in np.where(ok)[0]:
                    r = int(active[j])
                    rng_m = float(d[j])
                    z_pos = self._measured_pos(self._red_pos[r], rng_m)
                    delta = self._red_pos[r] - bp
                    nrm = float(np.linalg.norm(delta))
                    los = (delta / nrm) if nrm > 1e-6 else np.zeros(2, np.float32)
                    z_rad = float(self._red_vel[r] @ los)
                    s_rad = self.sensor_vel_noise_std * self._noise_scale(rng_m)
                    if s_rad > 0.0:
                        z_rad += float(self._rng.normal(0.0, s_rad))
                    out.append(self._make_return(
                        b, bp, z_pos, los, z_rad,
                        self.sensor_pos_noise_std * self._noise_scale(rng_m),
                        s_rad, truth_id=r))

            # --- clutter (false returns) --------------------------------
            if self.clutter_rate > 0.0 and self.sensor_radius is not None:
                n_c = int(self._rng.poisson(self.clutter_rate))
                for _ in range(n_c):
                    # Uniform in the disk: r ~ R*sqrt(u), theta ~ U[0, 2pi).
                    rad = self.sensor_radius * np.sqrt(self._rng.random())
                    th = self._rng.random() * 2.0 * np.pi
                    pt = bp + rad * np.array([np.cos(th), np.sin(th)],
                                             dtype=np.float32)
                    if np.any(pt < 0.0) or np.any(pt > self.arena_size):
                        continue                     # off the arena
                    if self.track_occlusion and bool(
                            self._rays_occluded_by_obstacles(bp, pt[None, :])[0]):
                        continue                     # a false alarm needs LOS too
                    delta = pt - bp
                    nrm = float(np.linalg.norm(delta))
                    los = (delta / nrm) if nrm > 1e-6 else np.zeros(2, np.float32)
                    scale = self._noise_scale(float(nrm))
                    # A false alarm is a threshold crossing in a range-Doppler
                    # cell, so it carries a MEANINGLESS Doppler, not none.
                    z_rad = float(self._rng.uniform(-self.vel_prior_std,
                                                    self.vel_prior_std))
                    out.append(self._make_return(
                        b, bp, pt, los, z_rad,
                        self.sensor_pos_noise_std * scale,
                        self.sensor_vel_noise_std * scale, truth_id=-1))
        return out

    def _make_return(self, b, bp, z_pos, los, z_rad, s_pos, s_rad, truth_id):
        return {
            "blue": int(b),
            "z_pos": np.asarray(z_pos, dtype=np.float32),
            "z_range": float(np.linalg.norm(np.asarray(z_pos) - bp)),
            "z_radial": float(z_rad),
            "los": np.asarray(los, dtype=np.float32),
            "sigma_pos": float(s_pos),
            "sigma_radial": float(s_rad),
            "truth_id": int(truth_id),          # EVALUATION ONLY; -1 = clutter
        }

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
            centre = 1.0 - p_move
            w_orth = p_move * 0.20
            w_diag = p_move * 0.05

            def _conv3(A: np.ndarray) -> np.ndarray:
                """3x3 motion kernel via padded shifts (no scipy dep).
                Edge-replicate padding = the target cannot leave the arena."""
                Ap = np.pad(A, 1, mode="edge")
                return (
                    centre * Ap[1:-1, 1:-1]
                    + w_orth * (Ap[:-2, 1:-1] + Ap[2:, 1:-1]
                                + Ap[1:-1, :-2] + Ap[1:-1, 2:])
                    + w_diag * (Ap[:-2, :-2] + Ap[:-2, 2:]
                                + Ap[2:, :-2] + Ap[2:, 2:])
                )

            # Log-odds -> probability (clamped for logit stability).
            P = 1.0 / (1.0 + np.exp(-L))

            # Obstacle-aware diffusion: an enemy cannot move THROUGH or INTO
            # an obstacle, so probability mass must not leak into masked
            # cells.  Rather than dropping that mass (which would silently
            # destroy probability), renormalise each source cell's outgoing
            # kernel weight over its VALID neighbours only — a reflecting
            # boundary at the obstacle wall.  With no obstacles, valid == 1
            # everywhere, f == 1, and this reduces exactly to the plain
            # convolution (byte-identical).
            valid = None
            og = getattr(self, "_obstacle_grid", None)
            if og is not None:
                valid = (og <= 0.5).astype(np.float32)
                if valid.all():
                    valid = None            # no masked cells -> fast path
            if valid is not None:
                f = _conv3(valid)           # fraction of weight landing valid
                P_src = np.where(valid > 0.0, P, 0.0) / np.maximum(f, 1e-6)
                P_new = _conv3(P_src) * valid
            else:
                P_new = _conv3(P)

            eps = 1e-6
            P_new = np.clip(P_new, eps, 1.0 - eps)
            L[:] = np.log(P_new / (1.0 - P_new))

        np.clip(L, -self.belief_clip, self.belief_clip, out=L)
        self._pin_enemy_belief_in_obstacles()

    def _pin_enemy_belief_in_obstacles(self) -> None:
        """
        Enemy-channel cells inside an obstacle are KNOWN-EMPTY, not
        "unknown".  Reds are kinematically clipped out of obstacle disks
        (``_clip_positions_from_obstacles``), so P(enemy | inside obstacle)
        is 0 — yet those cells are permanently occluded, so the sensor
        update never reaches them and they sit near log-odds 0 (= P 0.5,
        "unknown").  That makes them ~5.8 log-odds HIGHER than properly
        cleared open space, turning obstacle interiors into probability
        sinks that attract phantom peaks (measured: ~6.6% of extracted
        peaks landed inside an obstacle, tens of metres from any real red).

        Pinning them to -belief_clip encodes what we actually know and
        removes the bias.  Channel 1 (obstacle) is untouched — being
        inside an obstacle is exactly where that channel SHOULD be high.
        """
        og = getattr(self, "_obstacle_grid", None)
        if self._belief_maps is None or og is None:
            return
        mask = og > 0.5
        if mask.any():
            self._belief_maps[0][mask] = -self.belief_clip

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
                self._p_TP_ch[:C, None],
                self._p_FP_ch[:C, None],
            )                                                  # (C, K)
            detected = uniforms < p_detect                     # (C, K) bool
            evidence = np.where(
                detected,
                self._L_detect_ch[:C, None],
                self._L_no_detect_ch[:C, None],
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
        # Obstacle interiors are known-empty on the ENEMY channel, not
        # "unknown".  Re-pin after fusion so a moving obstacle's current
        # footprint is always enforced.  See _pin_enemy_belief_in_obstacles.
        self._pin_enemy_belief_in_obstacles()

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

    def _noise_scale(self, rng_m: float) -> float:
        """Measurement-noise multiplier at measurement range ``rng_m``.

        The same SNR falloff that lowers a distant track's CONFIDENCE also
        degrades its ACCURACY, so both must move together or the sensor
        model contradicts itself:  sigma(r) = sigma_base * (1 + g (r/R)^2).
        Returns 1.0 when the growth knob is off or range is unbounded.
        """
        g = self.sensor_noise_range_growth
        if g <= 0.0 or self.sensor_radius is None or not np.isfinite(rng_m):
            return 1.0
        x = float(np.clip(rng_m / self.sensor_radius, 0.0, 1.0))
        return 1.0 + g * x * x

    def _fuse_radial_velocity(
        self,
        target_pos: np.ndarray,          # (2,)
        true_vel:   np.ndarray,          # (2,)
        blue_idx:   np.ndarray,          # (M,) indices of DETECTING blues
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Command-layer velocity fusion from per-sensor RADIAL measurements.

        Physics: a Doppler radar measures only the component of a target's
        velocity ALONG its line of sight.  The tangential component is not
        observable from one platform.  But command receives radials from
        several platforms, and two NON-COLLINEAR lines of sight determine
        the full 2-D vector:

            [u_1^T; u_2^T] v = [r_1; r_2]        (netted / multistatic radar)

        so a fused velocity IS derivable — which is why it can be shared
        exactly like the fused position (see backlog §13c).

        Weighted least squares with a Gaussian speed prior (ridge):

            Sigma = (U^T W U + I / vel_prior_std^2)^-1     W = diag(1/sigma_b^2)
            v_hat = Sigma U^T W r

        * ``W`` weights each detector by its own measurement noise, which
          already grows with range (``_noise_scale``) — closer platforms
          count for more, automatically.
        * The ridge term is the prior "a target cannot outrun
          ``vel_prior_std``".  It makes the system solvable for ANY
          geometry (one detector, or collinear ones) with no special case,
          and it acts only where the measurements are uninformative: with
          two good baselines it contributes a fraction of a percent.
        * GDOP is emergent — near-collinear baselines make ``U^T W U``
          near-singular and the tangential variance grows on its own.

        Returns ``(v_hat (2,), Sigma (2, 2))``.  With no detectors the
        answer is the prior: zero mean, ``vel_prior_std^2`` isotropic.
        """
        s_prior2 = max(self.vel_prior_std, 1e-6) ** 2
        prior_prec = np.eye(2, dtype=np.float64) / s_prior2
        if len(blue_idx) == 0:
            return (np.zeros(2, dtype=np.float32),
                    (np.linalg.inv(prior_prec)).astype(np.float32))

        d = target_pos[None, :] - self._blue_pos[blue_idx]      # (M, 2)
        rng_m = np.linalg.norm(d, axis=1)
        u = d / np.maximum(rng_m, 1e-6)[:, None]                # unit LOS
        # Per-detector radial measurement + its own noise (range-scaled).
        sig = np.array(
            [max(self.sensor_vel_noise_std * self._noise_scale(float(rm)), 1e-6)
             for rm in rng_m], dtype=np.float64)
        r_true = u @ np.asarray(true_vel, dtype=np.float64)
        r_meas = r_true + self._rng.normal(0.0, sig)

        W = np.diag(1.0 / sig ** 2)
        Sigma = np.linalg.inv(u.T @ W @ u + prior_prec)
        v_hat = Sigma @ u.T @ W @ r_meas
        return v_hat.astype(np.float32), Sigma.astype(np.float32)

    @staticmethod
    def _cov_features(Sigma: np.ndarray, s_prior2: float) -> np.ndarray:
        """Pack a 2x2 velocity covariance into the 3 unique entries the
        policy sees, normalised by the prior variance so they land in
        ~[-1, 1] (1 = "no better than the prior" in that direction)."""
        return np.array(
            [Sigma[0, 0] / s_prior2, Sigma[1, 1] / s_prior2,
             Sigma[0, 1] / s_prior2], dtype=np.float32)

    def _measured_vel(self, true_vel: np.ndarray,
                      rng_m: float = float("inf")) -> np.ndarray:
        """Own-sensor Doppler measurement = true velocity + Gaussian noise,
        the noise growing with measurement range (see ``_noise_scale``).

        Real radar derives velocity from phase over the coherent processing
        interval, which is far more precise than differencing noisy
        positions, so ``sensor_vel_noise_std`` should stay small relative to
        ``sensor_pos_noise_std``.  NOTE: this models the measurement as
        isotropic in 2-D; a real radar measures only the RADIAL component
        well (see backlog "radial/tangential Doppler").
        """
        v = np.asarray(true_vel, dtype=np.float32)
        if self.sensor_vel_noise_std <= 0.0:
            return v
        sigma = self.sensor_vel_noise_std * self._noise_scale(rng_m)
        return (v + self._rng.normal(0.0, sigma, size=2).astype(np.float32)
                ).astype(np.float32)

    def _measured_pos(self, true_pos: np.ndarray,
                      rng_m: float = float("inf")) -> np.ndarray:
        """Fused measured position = true position + Gaussian noise, the
        noise growing with measurement range (see ``_noise_scale``)."""
        p = np.asarray(true_pos, dtype=np.float32)
        if self.sensor_pos_noise_std <= 0.0:
            return p
        sigma = self.sensor_pos_noise_std * self._noise_scale(rng_m)
        return (p + self._rng.normal(0.0, sigma, size=2).astype(np.float32)
                ).astype(np.float32)

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
        s_prior2   = max(self.vel_prior_std, 1e-6) ** 2
        track_pos  = np.zeros((K, 2), dtype=np.float32)
        track_conf = np.zeros((K,),   dtype=np.float32)
        track_red  = np.full((K,), -1, dtype=np.int32)
        track_vel  = np.zeros((K, 2), dtype=np.float32)
        # Velocity covariance (3 unique entries, normalised by the prior
        # variance).  A memory/padding slot carries the PRIOR: nothing is
        # known beyond the speed bound -> [1, 1, 0].
        track_vcov = np.tile(np.array([1.0, 1.0, 0.0], dtype=np.float32),
                             (K, 1))

        active = np.where(self._red_active)[0]
        if len(active) == 0 or self._belief_maps is None:
            return track_pos, track_conf, track_red, track_vel, track_vcov

        # Which active reds does ANY blue currently DETECT?  One detection
        # chain, same statistics as the belief map: in range -> clear line of
        # sight -> the p_TP draw fires.  A miss yields NO measurement at all
        # (the radar simply did not report), so the track coasts on the
        # memory path below, exactly like an out-of-range target.
        n_act    = len(active)
        red_act  = self._red_pos[active]                      # (n_act, 2)
        vis_mask = np.zeros(n_act, dtype=bool)
        best_rng = np.full(n_act, np.inf, dtype=np.float32)
        detect   = np.zeros((self.n_blue, self.n_red), dtype=bool)
        for b in range(self.n_blue):
            d = np.linalg.norm(red_act - self._blue_pos[b][None, :], axis=1)
            ok = (np.ones(n_act, dtype=bool) if self.sensor_radius is None
                  else d <= self.sensor_radius)
            if self.track_occlusion and ok.any():
                ok &= ~self._rays_occluded_by_obstacles(
                    self._blue_pos[b], red_act)
            if self.track_detection and ok.any():
                ok &= self._rng.random(n_act) < self.p_TP_enemy
            vis_mask |= ok
            best_rng = np.where(ok, np.minimum(best_rng, d), best_rng)
            detect[b, active] = ok
        self._last_red_detect = detect

        # Live-track confidence: SNR proxy from the range to the NEAREST
        # detecting blue (strongest return wins the fused track).
        if self.track_conf_min >= 1.0 or self.sensor_radius is None:
            live_conf = np.ones(n_act, dtype=np.float32)
        else:
            x = np.clip(best_rng / self.sensor_radius, 0.0, 1.0)
            live_conf = (self.track_conf_min
                         + (1.0 - self.track_conf_min) * (1.0 - x * x)
                         ).astype(np.float32)

        vis_reds    = active[vis_mask]
        live_confs  = live_conf[vis_mask]
        unseen_reds = active[~vis_mask]

        cs = self.belief_cell_size
        slot = 0

        # 1. Live tracks: fused measured position (true + range-scaled noise
        #    from the best-SNR detector).
        live_rngs = best_rng[vis_mask]
        for r, c, rm in zip(vis_reds, live_confs, live_rngs):
            if slot >= K:
                break
            track_pos[slot]  = self._measured_pos(self._red_pos[r], float(rm))
            track_conf[slot] = float(c)
            track_red[slot]  = int(r)
            # Command-layer velocity fusion from the RADIAL measurements of
            # every blue that detected this red (see _fuse_radial_velocity).
            v_hat, Sig = self._fuse_radial_velocity(
                self._red_pos[r], self._red_vel[r], np.where(detect[:, r])[0])
            track_vel[slot]  = v_hat
            track_vcov[slot] = self._cov_features(Sig, s_prior2)
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
                # Memory confidence is scaled into [0, track_conf_min) so a
                # LIVE track (which floors at track_conf_min) always
                # outranks a remembered one.  Both numbers land in the same
                # feature slot AND gate the GNN messages via edge_vis, but
                # they answer different questions: live conf is an SNR /
                # accuracy proxy, peak conf is P(target in this cell).
                # Measured, the gap is real — live tracks sit at ~1.8 m
                # position error, memory peaks at ~19-40 m even when their
                # raw conf reads >0.9 — so without this an unrescaled peak
                # would out-shout a live contact.  Within-path ordering is
                # preserved (peak conf does track its own error), and at
                # track_conf_min = 1.0 this reduces to the old behaviour
                # exactly (live 1.0, memory = raw peak conf).
                track_conf[slot] = peaks_conf[i] * self.track_conf_min
                track_red[slot]  = -1        # memory: never measured
                slot += 1                    # vel/vcov stay at the prior

        return track_pos, track_conf, track_red, track_vel, track_vcov

    def _enemy_graph_from_tracks(
        self,
        track_pos:  np.ndarray,   # (K, 2)
        track_conf: np.ndarray,   # (K,)
        track_red:  np.ndarray,   # (K,) int; -1 = memory / pad
        track_vel:  np.ndarray,   # (K, 2) command-FUSED velocity
        track_vcov: np.ndarray,   # (K, 3) its covariance (prior-normalised)
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
            for b in range(N):
                e = s * N + b
                src_pos[e] = track_pos[s]
                # VELOCITY is now the command-FUSED estimate, shared with
                # every blue exactly like POSITION.  A single radar measures
                # only the radial component, but two non-collinear radials
                # determine the full 2-D vector, so a fused velocity IS
                # derivable at command (backlog §13c).  How much is actually
                # known travels with it in the covariance node feature.
                src_vel[e] = track_vel[s]
                dst_pos[e] = self._blue_pos[b]
                dst_vel[e] = self._blue_vel[b]
                edge_vis[e] = track_conf[s]

        edge_feat = self._edge_features_for(src_pos, src_vel, dst_pos, dst_vel)
        edge_feat[edge_vis <= 0.0] = 0.0     # zero padded/dead slots
        # Node feature: [conf, Sxx, Syy, Sxy] — the velocity covariance tells
        # the policy WHICH DIRECTION it is ignorant about (1 = no better than
        # the speed prior), which a scalar confidence would discard.
        node_feat = np.concatenate(
            [track_conf.reshape(K, 1), track_vcov], axis=1).astype(np.float32)
        return node_feat, edge_feat, edge_vis

    def _build_obstacle_tracks(self):
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

        Returns ``(obs_pos, obs_vel, obs_conf, obs_r, obs_idx)`` in the same
        slot convention ``_extract_belief_peaks`` uses: real tracks first,
        unplaced slots padded with conf 0.  ``obs_vel`` is the obstacle's
        own-radar velocity when it is a live (seen) track, else 0 (a memory
        track carries no Doppler) — static obstacles are simply always 0.
        ``obs_r`` is the obstacle RADIUS: the true measured value for a seen
        track, the surveyed field's mean (command prior) for a memory track.
        ``obs_idx`` maps slot -> backing obstacle index (-1 for memory/pad),
        mirroring ``track_red``; the graph builder uses it to apply PER-BLUE
        Doppler.  NOTE ``obs_vel`` here is the best-detector measurement kept
        for diagnostics/tests — the GRAPH does not consume it, because
        velocity is a first-person measurement (see
        ``_obstacle_graph_from_tracks``).
        The actor needs it because both the crash penalty and the clearance
        barrier are defined on the SURFACE (centre_dist − radius), which a
        centre-only track cannot locate when radii vary.
        """
        n_obs  = self.n_obstacles
        obs_pos  = np.zeros((n_obs, 2), dtype=np.float32)
        obs_vel  = np.zeros((n_obs, 2), dtype=np.float32)
        obs_conf = np.zeros((n_obs,),   dtype=np.float32)
        obs_r    = np.zeros((n_obs,),   dtype=np.float32)   # obstacle RADIUS
        obs_idx  = np.full((n_obs,), -1, dtype=np.int32)    # slot -> obstacle
        s_prior2 = max(self.vel_prior_std, 1e-6) ** 2
        obs_vcov = np.tile(np.array([1.0, 1.0, 0.0], dtype=np.float32),
                           (n_obs, 1))                      # prior by default
        # Command's prior on obstacle size for memory tracks (unseen): the
        # belief peak has lost which physical obstacle it is, so use the
        # surveyed field's mean radius.  A live (seen) track overrides this
        # with the true measured radius below.
        radius_prior = 0.5 * (self.obstacle_radius_min + self.obstacle_radius_max)
        placed = 0 if self._obstacle_pos is None else len(self._obstacle_pos)
        if placed == 0:
            return obs_pos, obs_vel, obs_conf, obs_r, obs_idx, obs_vcov

        # Which placed obstacles does ANY blue currently DETECT?  Same
        # detection chain as the enemy tracks / belief map: in range ->
        # clear line of sight -> the p_TP draw fires.
        idx  = np.arange(placed)
        opos = self._obstacle_pos[:placed]
        orad = self._obstacle_r[:placed]
        seen_mask = np.zeros(placed, dtype=bool)
        best_rng  = np.full(placed, np.inf, dtype=np.float32)
        odet = np.zeros((self.n_blue, n_obs), dtype=bool)
        for b in range(self.n_blue):
            bp = self._blue_pos[b]
            d  = np.linalg.norm(opos - bp[None, :], axis=1)
            ok = (np.ones(placed, dtype=bool) if self.sensor_radius is None
                  else d <= self.sensor_radius)
            if self.track_occlusion and ok.any():
                # Cast to the NEAR-SURFACE point, not the centre: a disk
                # would otherwise always occlude itself.  The occlusion
                # margin lets the first boundary surface be observed.
                u    = (opos - bp[None, :]) / np.maximum(d, 1e-6)[:, None]
                near = (opos - orad[:, None] * u).astype(np.float32)
                ok &= ~self._rays_occluded_by_obstacles(bp, near)
            if self.track_detection and ok.any():
                ok &= self._rng.random(placed) < self.p_TP_obstacle
            seen_mask |= ok
            best_rng = np.where(ok, np.minimum(best_rng, d), best_rng)
            odet[b, :placed] = ok
        self._last_obs_detect = odet

        # Live-track confidence: SNR proxy from range to nearest detector.
        if self.track_conf_min >= 1.0 or self.sensor_radius is None:
            o_conf = np.ones(placed, dtype=np.float32)
        else:
            x = np.clip(best_rng / self.sensor_radius, 0.0, 1.0)
            o_conf = (self.track_conf_min
                      + (1.0 - self.track_conf_min) * (1.0 - x * x)
                      ).astype(np.float32)

        seen   = idx[seen_mask]
        unseen = idx[~seen_mask]

        cs = self.belief_cell_size
        slot = 0

        # 1. Live tracks: precise own-sensor position (true + noise).
        for o in seen:
            if slot >= n_obs:
                break
            rm = float(best_rng[o])
            obs_pos[slot] = self._measured_pos(self._obstacle_pos[o], rm)
            # Command-layer velocity fusion, same chain as the enemy tracks.
            v_true = (self._obstacle_vel[o] if self._obstacle_vel is not None
                      else np.zeros(2, dtype=np.float32))
            v_hat, Sig = self._fuse_radial_velocity(
                self._obstacle_pos[o], v_true, np.where(odet[:, o])[0])
            obs_vel[slot]  = v_hat
            obs_vcov[slot] = self._cov_features(Sig, s_prior2)
            obs_r[slot]    = self._obstacle_r[o]        # true measured radius
            obs_conf[slot] = float(o_conf[o])
            obs_idx[slot]  = int(o)                     # backing obstacle
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
                # Scaled below the live floor — see _build_enemy_tracks.
                obs_conf[slot] = peaks_conf[i] * self.track_conf_min
                obs_r[slot]    = radius_prior      # command prior (unknown exact)
                slot += 1              # obs_idx -1, vel/vcov stay at prior

        return obs_pos, obs_vel, obs_conf, obs_r, obs_idx, obs_vcov

    def _obstacle_graph_from_tracks(
        self,
        track_pos: np.ndarray,   # (K, 2) world coords (shared track picture)
        track_vel: np.ndarray,   # (K, 2) command-FUSED velocity
        conf:      np.ndarray,   # (K,)   track confidences
        track_r:   np.ndarray,   # (K,)   obstacle radius (true if seen, prior else)
        track_vcov:np.ndarray,   # (K, 3) velocity covariance (prior-normalised)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Actor-side OBSTACLE node + edge features from the obstacle tracks
        (``_build_obstacle_tracks``): a live own-sensor position when a
        blue senses the obstacle, else the belief-map peak.  Same 7-D layout
        / (s outer, b inner) ordering as the enemy graph — and the same
        POSITION-vs-VELOCITY split:

        * POSITION is the ONE shared command-layer track (same for every
          blue), so all blues share a single correlated measurement error
          rather than being able to average independent draws.
        * VELOCITY is PER-BLUE own-sensor Doppler, reported only by a blue
          that actually DETECTED this obstacle this scan, with noise scaled
          by ITS range.  Velocity is a first-person measurement; command
          cannot deliver Doppler to a UAV without its own sensor lock.
          Undetected / memory / static -> zero.
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
                src_vel[e] = track_vel[s]      # command-fused, shared
                dst_pos[e] = self._blue_pos[b]
                dst_vel[e] = self._blue_vel[b]
                edge_vis[e] = conf[s]

        edge_feat = self._edge_features_for(src_pos, src_vel, dst_pos, dst_vel)
        edge_feat[edge_vis <= 0.0] = 0.0
        # Node feature: [conf, radius / arena_size].  conf stays at index 0
        # (used as the placed/active mask downstream).  Radius normalised by
        # the arena side to match the position scaling in the edges.
        # [conf, radius/L, Sxx, Syy, Sxy]
        node_feat = np.concatenate(
            [np.stack([conf, track_r / self.arena_size], axis=-1), track_vcov],
            axis=1).astype(np.float32)
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

        # [placed, radius/L, Sxx, Syy, Sxy] — covariance slots stay 0: the
        # critic sees ground truth, so its velocity is certain.
        node_feat = np.zeros((n_obs, 5), dtype=np.float32)
        n_edges = n_obs * N
        src_pos = np.zeros((n_edges, 2), dtype=np.float32)
        src_vel = np.zeros((n_edges, 2), dtype=np.float32)   # true obstacle vel
        dst_pos = np.zeros((n_edges, 2), dtype=np.float32)
        dst_vel = np.zeros((n_edges, 2), dtype=np.float32)

        for o in range(n_obs):
            has = o < placed
            if has:
                node_feat[o, 0] = 1.0
                node_feat[o, 1] = self._obstacle_r[o] / self.arena_size
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
            # Critic ground truth (reds).  The actor's red node feature is
            # [conf, Sxx, Syy, Sxy]; the critic sees TRUTH, so its covariance
            # slots are ZERO — no uncertainty — keeping both encoders at the
            # same red_feat_dim.
            "true_red_features": np.concatenate(
                [base["red_features"],
                 np.zeros((base["red_features"].shape[0], 3), dtype=np.float32)],
                axis=1),
            "true_rb_edge_features": base["rb_edge_features"],
        }

        # ---- Actor enemy graph: DETECTION-SEEDED tracks ------------------
        # Live tracks (visible reds) seeded directly from the sensor;
        # remaining slots filled from belief-map memory peaks.  See
        # _build_enemy_tracks.  Dead slots are conf-0 padded.
        (track_pos, track_conf, track_red,
         track_vel, track_vcov) = self._build_enemy_tracks()
        red_node, rb_edge, rb_vis = self._enemy_graph_from_tracks(
            track_pos, track_conf, track_red, track_vel, track_vcov,
        )
        out["red_features"]    = red_node
        out["rb_edge_features"] = rb_edge
        out["rb_edge_visible"]  = rb_vis

        # ---- Obstacle graph (only when obstacles are configured) ---------
        if self.n_obstacles > 0:
            # Live own-sensor position when a blue senses the obstacle,
            # else the belief-map peak (command memory).  See
            # _build_obstacle_tracks for the doctrine.
            (obs_pos, obs_vel, obs_conf, obs_r,
             _obs_idx, obs_vcov) = self._build_obstacle_tracks()
            ob_node, ob_edge, ob_vis = self._obstacle_graph_from_tracks(
                obs_pos, obs_vel, obs_conf, obs_r, obs_vcov,
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

