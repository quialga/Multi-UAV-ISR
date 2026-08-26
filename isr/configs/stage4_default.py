"""
isr/configs/stage4_default.py — the single hyperparameter config.

Formerly the tail of a stage1 -> stage3 -> stage4 inheritance chain.  The
chain was flattened in the 2026-08 cleanup: Stage 3 had no runnable trainer
and Stage 1's was removed, so the two upstream files were a config layer for
stages that no longer exist.  Every key is now declared here explicitly —
one file, no inheritance, nothing shadowed.

Override individual fields via CLI flags in scripts/train_stage4.py rather
than editing this file.
"""
from __future__ import annotations


STAGE4_DEFAULTS = {
    # ===== Base knobs =====================================================
    # These came from the old STAGE1/STAGE3 layers and are unchanged in
    # value; only their home moved.

    # ----- Environment ----------------------------------------------------
    "capture_radius": 3.0,
    "dt":             1.0,
    "sensor_radius":  40.0,    # ~20% of the L=200 arena side

    # ----- Reward shape (env-level) ---------------------------------------
    #   r_i    = r_team + r_crash_i + r_clear_i - action_cost_i
    #   r_team = catch_reward * n_caught - step_cost
    #            [- uncaught_penalty * n_uncaught at episode end]
    # SHARED terms are mission-level; control effort is INDIVIDUAL.
    # (step_cost is overridden below with the arena scale-up.)
    "catch_reward":     10.0,   # per red caught (shared: a team success)
    "uncaught_penalty": 5.0,    # per red still alive at the end (shared)
    "action_cost_coef": 0.01,   # * |a_i|^2, INDIVIDUAL: own control effort

    # Gaussian speed prior for the command-layer velocity fusion (ridge
    # term).  ~ the fastest entity's v_max (BLUE_UAV 1.5, RED_TARGET 1.0).
    "vel_prior_std":    1.0,

    # ----- Vectorisation --------------------------------------------------
    # Rollout collection runs n_envs copies of the env in parallel; a
    # rollout yields n_envs * rollout_steps * n_agents samples.
    "n_envs":         16,

    # ----- PPO ------------------------------------------------------------
    "n_rollouts":     1000,        # total PPO updates
    "n_epochs":       10,          # PPO epochs per rollout
    "mb_size":        512,         # minibatch size
    "clip_eps":       0.2,
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "value_clip":     True,        # clip value updates by clip_eps too
    "normalize_adv":  True,        # standardise advantages per-minibatch
    "target_kl":      0.03,        # per-epoch early stop (None disables)

    # LR schedule — with a constant LR at low entropy, gradient steps have
    # disproportionate behavioural effect; linear decay dampens that.
    "lr_schedule":    "linear",    # or "constant"
    "lr_min_frac":    0.1,         # anneals lr -> lr * lr_min_frac

    # ----- Network --------------------------------------------------------
    "d_hidden":       64,
    "n_msg_rounds":   2,           # do NOT run with 1 (see PPO note below)
    "init_log_std":   0.0,         # std = e^0 = 1 at init; the action box
                                   # is [-1,1], so clipping is heavy early
                                   # and the policy shrinks std rapidly.

    # ----- Logging / eval / checkpointing ---------------------------------
    "log_interval":   1,
    "eval_interval":  50,
    "eval_episodes":  20,
    "save_interval":  100,
    # Save best.pt whenever the tracked metric beats the previous best by
    # min_delta.  See the det_composite note in scripts/train_stage4.py:
    # mean_return MIS-SELECTS on crash-penalty runs.
    "best_ckpt_metric":       "mean_return",
    "best_ckpt_min_delta":    0.05,
    "best_ckpt_min_episodes": 32,

    # ===== Stage 4 =========================================================

    # ----- Arena scale-up + team size (2026-08) ---------------------------
    # Motivation: at L=130 with R=40 and 5 blues the team's SENSOR COVERAGE
    #     C = n_blue * pi * R^2 / L^2 = 1.49
    # exceeds the arena — the team can blanket 149% of it, so search is
    # trivial whenever they spread out and the belief map barely matters.
    # (The measured 34% in-range fraction was a COORDINATION shortfall, not
    # a sensor-reach limit.)  L=200 with 7 blues puts coverage at 0.88 and
    # the measured memory-track share at ~92%, so the belief/memory path
    # becomes the dominant information source.
    #
    # Why not just shrink sensor_radius (free, no extra compute)?  Because
    # velocity fusion (§17) needs REDUNDANT coverage — two blues seeing the
    # same target — and the fusion rate tracks blue-spacing/2R closely:
    #     L=130 nb=5  spacing/2R 0.73 -> 8.1% of observations fused
    #     L=200 nb=5  spacing/2R 1.12 -> 0.2%   (fusion effectively dead)
    #     L=200 nb=7  spacing/2R 0.94 -> 1.7%
    #     L=200 nb=8  spacing/2R 0.88 -> 5.1%
    # Search difficulty and fusion are THE SAME QUANTITY with opposite
    # signs: you cannot have low coverage (hard search) and high overlap
    # (fusion) at once.  nb=7 is the chosen compromise — search stays hard
    # (~92% memory) while fusion stays off the floor.  A trained team can do
    # better than these random-blue numbers by deliberately pairing up when
    # it wants a velocity fix; that choice is now a real strategic option.
    #
    # Speeds are deliberately UNCHANGED (blue 1.5 / red 1.0).  The RATIO is
    # the task: pursuit/episode = L/(v_b - v_r)/max_steps stays ~1.3, i.e. a
    # stern chase from max separation still does NOT fit in an episode, so
    # cornering remains mandatory.  What does shift is the endgame share
    # (37% -> 25% of the episode), which is exactly the intent: more of each
    # episode spent searching and tracking.
    #
    # Compute scales ~O(L^3) (grid ops L^2, episode length L): ~3.6x.
    #
    # These four override STAGE1/STAGE3 for Stage 4 ONLY — the earlier
    # stages keep their validated 130 m / 200-step setup.
    "arena_size":     200.0,   # was 130; coverage 1.49 -> 0.88 at nb=7
    # A bigger arena needs more UAVs to patrol it.  nb=7 also keeps
    # blues-per-red at 1.75 with nr=4 (better than the old 5/3 = 1.67),
    # which matters because cornering one fleeing red takes 2-3 blues.
    "n_blue":         7,       # was 5 (STAGE3)
    # nr=4 buys a richer early SEARCH phase and the most fusion events
    # per episode (2.2 vs 1.8 at nr=3).  Note it does NOT make the
    # belief map progressively more important as reds are caught: the
    # memory share measured flat at ~90% for nr = 4/3/2/1, because it is
    # set by blue coverage geometry, not by how many targets remain.
    "n_red":          4,       # was 3 (STAGE3)
    "max_steps":      300,     # ~2.3 arena crossings, as at L=130
    "rollout_steps":  320,     # >= max_steps: whole episodes per rollout
    # step_cost * max_steps was 0.05*200 = 10 against a catch reward of 30.
    # Holding that ratio at max_steps=300 needs 0.033; leaving it at 0.05
    # would silently raise time pressure by half.
    "step_cost":      0.033,

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
    # No CNN.  Grid holds 5 m cells (40x40 at L=200): the earlier 2.5 m
    # refinement was chasing a convergence problem that turned out to be
    # architectural, not resolution.
    "use_belief_maps":     True,
    # H = W = arena_size / cell_size = 200 / 5.  Scaling this WITH the
    # arena is essential: leaving it at 26 would give 7.7 m cells and make
    # the motion-vs-resolution mismatch in backlog §20 materially worse.
    "belief_grid_size":    40,
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
    # go mushy within 3-4 steps.  (NOTE: p_move 0.2 injects 6x the
    # physically admissible variance per step and grows as sqrt(t) rather
    # than t -- see backlog 20; it is kept because no fixed kernel can do
    # better at this resolution.)
    # Half-life ln2/-ln(gamma): 0.99 gives ~69 steps, tuned against the
    # old 87-step arena crossing.  At L=200 the crossing is 133 steps, so
    # the half-life should be ~106 -> gamma = 0.5^(1/106).  Left at 0.99,
    # tracks would fade faster than the arena timescale, making the belief
    # map LESS useful exactly where the scale-up is meant to make it more.
    "enemy_belief_decay":     0.9935,
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
    # Held at the previous ~7% areal density: n*pi*r_bar^2/L^2 with
    # r_bar = 10 m gives 9 obstacles at L=200 (was 4 at L=130).  Radii
    # are physical and unchanged.
    "n_obstacles":            9,
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
    # v6.3 warm-started the CRITIC from the Stage 1 single-encoder GNN,
    # which historically mattered: dropping it was one of three coupled
    # stabilisers whose loss caused the v6 regression.
    #
    # MEASURE IT BEFORE TRUSTING IT (2026-08).  The Stage 1 checkpoint now
    # transfers only 28 of 38 critic tensors into the current policy, and
    # the ones that FAIL are every INPUT layer:
    #     red_input_mlp.0.weight  (64,4)   red node feature grew 1 -> 4
    #                                      (velocity covariance, §17)
    #     obs_input_mlp.* (4 tensors)      obstacle features 1 -> 5, and
    #                                      Stage 1 had no obstacles at all
    #     critic_trunk.0.weight   (64,194) masked-mean pool changed the width
    # What still transfers is the INTERIOR (message/update/edge MLPs, trunk
    # layer 2, head) — i.e. weights that now receive randomly-projected
    # inputs.  So the "meaningful advantages from rollout 0" claim no longer
    # holds at this transfer level; treat the warm start as neutral rather
    # than as a stabiliser.
    #
    # KNOCK-ON RISK: aux_hidden_coef below is explicitly coupled to having a
    # MEANINGFUL critic (its target is critic_h_blue.detach()).  With the
    # critic's input layers random, that target is noisy early — exactly the
    # regime where aux 0.2 was previously found to HURT.  If the from-scratch
    # baseline stalls in the first ~100 rollouts, drop aux_hidden_coef toward
    # 0 (or set this to None for an honest cold start) before suspecting
    # anything else.
    # BASELINE DEFAULT = None (honest cold start).  The Stage 1 checkpoint is
    # structurally incompatible (see above) AND lives under a gitignored
    # runs/ dir, so a fresh clone silently cold-starts anyway with only a
    # WARN line.  Depending on it made behaviour differ between laptop and
    # cloud.  The right warm start for run 2 is run 1's OWN checkpoint via
    # --warm-start-full, which transfers 77/77 tensors (verified) because it
    # is the same architecture.
    "warm_start_critic": None,

    # ----- PPO (pinned to Stage 3's WINNING recipe) ----------------------
    # The Stage 4 v6 runs regressed because they dropped three coupled
    # Stage 3 stabilisers at once: 2 message rounds, a warm-started
    # critic, and aux 0.2.  Pinning the winning values here so a bare
    # run reproduces the Stage 3 setup (base STAGE1 defaults were
    # lr 3e-4 / ent 0.018, which Stage 3 overrode on the CLI).
    "lr":                1e-4,
    "ent_coef":          0.008,
    # aux 0.2 ONLY helps when critic_h_blue (its .detach()ed target) is
    # MEANINGFUL.  A cold critic gives a target that is arbitrary AND
    # MOVING -- worse than a frozen random projection, because the actor
    # chases a representation drifting for reasons unrelated to its task.
    # (Hence aux 0.02 hurt in the cold-start experiment.)
    # BASELINE DEFAULT = 0.0 to match warm_start_critic=None above.
    # Turn it back to 0.2 for run 2, warm-started --warm-start-full from
    # run 1: that critic is converged, same-architecture and same-
    # distribution, which is the condition aux 0.2 was validated under and
    # the first time it is genuinely protected in this architecture.
    "aux_hidden_coef":   0.0,
    "use_hidden_in_gnn": True,      # Stage 3 opt-1 hidden-in-GNN kept on
    # n_msg_rounds inherited = 2 (Stage 3); do NOT run with 1.
}
