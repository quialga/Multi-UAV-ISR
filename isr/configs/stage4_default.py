"""
isr/configs/stage4_default.py — Stage 4 hyperparameters.

Mirrors docs/stage4_design.md §8.  Extends STAGE3_DEFAULTS with the
belief-map / obstacle / sensor-noise knobs.  Override via CLI flags
in scripts/train_stage4.py rather than editing this file.
"""
from __future__ import annotations

from isr.configs.stage3_default import STAGE3_DEFAULTS


STAGE4_DEFAULTS = {
    **STAGE3_DEFAULTS,

    # ----- Belief map (mission-command Bayesian tracker) -----------------
    # ONE belief map maintained at the MISSION COMMAND layer -- an
    # environment-level latent used for target tracking and evaluation,
    # NOT the instantaneous per-UAV view (see doctrine note on
    # PursuitEnv._update_belief_maps and docs/stage4_backlog.md §13).
    # Bayesian fusion of independent sensor returns is log-odds
    # addition, so command's picture accumulates all UAVs' observations.
    # The POLICY never sees the raw grid: the env extracts top-K peaks
    # per channel and reconstructs the typed GNN graph (rb_edges from
    # channel 0 = P(enemy); ob_edges from channel 1 = P(obstacle)).
    # No CNN.  Grid at 26x26 (5 m cells): the earlier 2.5 m refinement
    # was chasing a convergence problem that turned out to be
    # architectural, not resolution.
    "use_belief_maps":     True,
    "belief_grid_size":    26,     # H = W (arena_size / cell_size = 130 / 5)
    # 2 channels: {P(enemy) Bayesian, P(obstacle) Bayesian}.  Ally/self
    # overlays are gone — those positions are precise and already flow
    # through the graph (blue_features / bb_edges).
    "belief_channels":     2,
    "belief_clip":         10.0,   # log-odds clip

    # ----- Phase A: Bayesian prediction step (enemy channel) -------------
    # Each env step runs predict -> update on the enemy channel:
    #   decay:     L <- gamma * L   (stale evidence -> "unknown";
    #              ghost tracks fade, cleared regions re-acquirable)
    #   diffusion: p_move of each cell's probability mass spreads to its
    #              8 neighbours (isotropic random-walk motion model,
    #              applied in probability space).
    # Obstacle channel untouched (static -> prediction = identity).
    # Set 1.0 / 0.0 to recover pre-Phase-A behaviour exactly.
    # CALIBRATED to red kinematics (v_max 1 m/s, dt 1 s, cell 5 m):
    # a red crosses at most ~0.2-0.28 cells/step, so p_move 0.2; the
    # original 0.4 implied ~2 m/s targets and made unobserved tracks
    # go mushy within 3-4 steps.  decay 0.99 (confidence half-life ~69
    # steps) suits 350-step episodes; 0.97 (~23 steps) was too
    # forgetful.
    "enemy_belief_decay":     0.99,
    "enemy_belief_diffusion": 0.2,

    # Live-sensor position accuracy (m) for VISIBLE targets: a blue
    # that can see the associated red gets the true position +
    # N(0, sigma^2) on its rb edge instead of the cell-centre grid
    # peak.  Fixes the endgame: cell-quantised tracks are up to 3.5 m
    # off on a 5 m grid, vs a 3 m capture radius -- flying exactly to
    # the peak could still miss the capture.  Radar range accuracy of
    # ~1 m at close range is realistic.
    "sensor_pos_noise_std":   1.0,
    # Live-track sensor realism: a live track comes from the SAME radar that
    # feeds the belief map, so it obeys the same detect(p_TP) -> measure
    # chain and the same line-of-sight test.  Set the booleans False to
    # reproduce pre-fix runs (range-only gate, conf 1.0, sees through walls).
    "track_occlusion":        True,
    "track_detection":        True,
    # Doppler noise — keep SMALL vs sensor_pos_noise_std (radar measures
    # velocity from phase, not by differencing noisy positions).
    "sensor_vel_noise_std":   0.1,
    # Live-track confidence floor at max range (SNR proxy).  Range-only, so
    # it never leaks true-vs-false.  1.0 = flat conf (pre-fix).
    "track_conf_min":         0.5,
    # The other half of the same SNR story: a distant return is not just
    # less trustworthy, it is less ACCURATE.  sigma(r) = sigma_base *
    # (1 + g (r/R)^2); 1.0 = measurement noise doubles at max range.
    "sensor_noise_range_growth": 1.0,

    # ----- Sensor model --------------------------------------------------
    "p_TP":                0.85,
    "p_FP":                0.15,
    # Per-channel override (backlog §8).  A concrete 15 m obstacle is far
    # easier to detect than a small, actively-evading drone, yet both
    # channels shared one (p_TP, p_FP).  This became load-bearing once live
    # tracks started obeying p_TP: obstacles dropped out of live tracking
    # ~15% of scans, flickering their perceived SURFACE — which is exactly
    # what clearance shaping keys on.  Enemy deliberately INHERITS the pair
    # above (0.85/0.15) so this stays a physics correction, not a difficulty
    # change.  Also gets most of §18's benefit without tracker state.
    "p_TP_obstacle":       0.95,
    "p_FP_obstacle":       0.05,
    # Occlusion margin (m): the analytic segment-disk test cuts each
    # ray this far before the cell centre, so obstacle boundary cells
    # within this depth stay observable.  (Name kept from the old
    # sampled ray-march for CLI compatibility.)
    "ray_step_size":       2.5,

    # ----- Obstacles -----------------------------------------------------
    "n_obstacles":            4,
    "obstacle_radius_min":    5.0,
    "obstacle_radius_max":    15.0,
    "obstacle_spawn_clearance": 10.0,

    # ----- Variable entity counts (generalisation) ----------------------
    # n_red / n_obstacles above are the PADDED CAPACITY.  Setting a *_min
    # makes each episode sample the active count in [min, capacity]; the
    # masked-mean critic pool makes the value head count-agnostic so a
    # single policy generalises across (and beyond) the trained counts.
    # None = fixed at capacity (byte-preserves fixed-count runs).
    "n_red_min":              None,
    "n_obstacles_min":        None,

    # ----- Moving obstacles (backlog §4, simplest kinematics) -----------
    # A fraction of obstacles patrol back-and-forth along one axis,
    # bouncing off the arena walls (deterministic, no response to blues).
    # Crashing into one uses the same per-agent crash penalty; blues are
    # never destroyed.  Defaults OFF (all static).  obstacle_belief_decay
    # (<1) fades the stale "comet trail" a moving obstacle leaves on the
    # belief map; enable it (e.g. 0.9) when obstacles move.
    "moving_obstacle_fraction": 0.0,   # 0..1 of placed obstacles that move
    "obstacle_speed":           0.0,   # m/step along the patrol axis
    "obstacle_belief_decay":    1.0,   # 1.0 = off (static-obstacle behaviour)

    # ----- Crash avoidance (per-agent penalties) -------------------------
    # Individual (NOT shared) penalties added to each UAV's own reward:
    #   r_i = r_team + r_crash_i
    # A crash rolls the offender back + zeroes its velocity (soft-stop);
    # the episode does NOT terminate.  Defaults 0.0 = feature off, which
    # byte-preserves the pre-crash shared-reward behaviour.  The crash-
    # avoidance run turns these on via CLI (obstacle ~2.0, ally ~1.0).
    "crash_obstacle_penalty":  0.0,   # per-step penalty for hitting an obstacle
    "crash_blue_penalty":      0.0,   # per-step penalty for a blue-blue collision
    "blue_collision_radius":   2.0,   # m; < 3 m capture radius so it's a true crash
    # Dense clearance / barrier shaping (backlog §16 #1): keeps blues off
    # obstacle surfaces (and off each other) with a smooth outward gradient.
    # OPT-IN (both weights 0 = off) — like every other shaping knob here.
    # Enable per run via --clearance-weight / --clearance-ally-weight, and
    # set BOTH together: obstacle clearance alone migrates crashes
    # blue->blue, and leaving the ally weight unset would silently inherit
    # whatever default this file carried.  The margins below only bite when
    # their weight is > 0.
    "clearance_weight":        0.0,   # obstacle barrier magnitude (0 = off)
    "clearance_margin":        4.0,   # m band beyond an obstacle surface
                                      # (blue stops in ~1.6 m at v_max, so 4 m
                                      # is ample reaction room without a huge
                                      # no-go zone around big obstacles)
    "clearance_ally_weight":   0.0,   # blue<->blue barrier magnitude (0 = off)
    "clearance_ally_margin":   3.0,   # m band beyond the collision radius

    # ----- Ally comms -----------------------------------------------------
    # bb_edge_visible IS present in the Stage 4 obs dict and is gated by
    # ``sensor_radius`` (bb_edge_visible[e] = 1 iff the two blues are
    # within sensor_radius of each other, per _compute_edge_visibility).
    # The "TDL always on, bb_edge_visible == 1" narrative in earlier
    # design notes is aspirational: the code has no separate comms
    # range knob.  See backlog §7 for the open item that would
    # decouple comms_radius from sensor_radius (default None ==
    # unbounded within the arena, matching the TDL doctrine).

    # ----- Warm start ----------------------------------------------------
    # v6.3: warm-start the CRITIC from the Stage 1 single-encoder GNN,
    # exactly as Stage 3 did (its encoder MLPs + critic trunk/head map
    # onto our critic for n_obstacles=0).  A pre-trained critic gives
    # meaningful advantages from rollout 0 -- the stabiliser Stage 3
    # relied on and the Stage 4 cold-start dropped.  Set None to
    # cold-start.  With obstacles the trunk widens, so only the encoder
    # warm-starts (shape-matched copy).
    "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",

    # ----- PPO (pinned to Stage 3's WINNING recipe) ----------------------
    # The Stage 4 v6 runs regressed because they dropped three coupled
    # Stage 3 stabilisers at once: 2 message rounds, a warm-started
    # critic, and aux 0.2.  Pinning the winning values here so a bare
    # run reproduces the Stage 3 setup (base STAGE1 defaults were
    # lr 3e-4 / ent 0.018, which Stage 3 overrode on the CLI).
    "lr":                1e-4,
    "ent_coef":          0.008,
    # aux 0.2 ONLY helps with the warm-started critic: the target is
    # critic_h_blue.detach(), so a cold/random critic makes aux a
    # garbage target (this is why aux 0.02 hurt in the cold-start
    # experiment).  Warm start + aux 0.2 is the proven Stage 3 combo.
    "aux_hidden_coef":   0.2,
    "freeze_critic":     False,     # not supported in v6 (cold start)
    "use_hidden_in_gnn": True,      # Stage 3 opt-1 hidden-in-GNN kept on
    # n_msg_rounds inherited = 2 (Stage 3); do NOT run with 1.
}
