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

    # ----- Belief map (env-internal Bayesian tracker) --------------------
    # v6.1: ONE global fused belief map shared by the whole blue team
    # (common operational picture over TDL — Bayesian fusion of
    # independent sensors is log-odds addition).  The POLICY never sees
    # the raw grid: the env extracts top-K peaks per channel and
    # reconstructs the typed GNN graph (rb_edges from channel 0 =
    # P(enemy); ob_edges from channel 1 = P(obstacle)).  No CNN.
    # Grid back at 26x26 (5 m cells): the earlier 2.5 m refinement was
    # chasing a convergence problem that turned out to be architectural,
    # not resolution.
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

    # ----- Sensor model --------------------------------------------------
    "p_TP":                0.85,
    "p_FP":                0.15,
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

    # ----- Ally comms (choice B in design discussion) --------------------
    # bb_edge_visible is not present in the Stage 4 obs dict (allies
    # always share GPS in the baseline).  Kept as a config record only.
    "bb_edge_visible_always_on": True,

    # ----- Warm start ----------------------------------------------------
    # v6.3: warm-start the CRITIC from the Stage 1 single-encoder GNN,
    # exactly as Stage 3 did (its encoder MLPs + critic trunk/head map
    # onto our critic for n_obstacles=0).  A pre-trained critic gives
    # meaningful advantages from rollout 0 -- the stabiliser Stage 3
    # relied on and the Stage 4 cold-start dropped.  Set None to
    # cold-start.  With obstacles the trunk widens, so only the encoder
    # warm-starts (shape-matched copy).
    "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",

    # ----- Aux loss ------------------------------------------------------
    # v6.2: aux hidden loss ported from Stage 3 (opt-C, live critic
    # target).  MSE(actor_h_blue, critic_h_blue.detach()) encourages the
    # actor's GNN embedding (from noisy belief-derived positions) to
    # match the critic's (from ground truth).  Stage 3 landed on 0.1
    # for live-critic; 0.2 needed freeze-critic (which requires warm
    # start, not available in v6).
    "aux_hidden_coef":   0.0,
    "freeze_critic":     False,     # not supported in v6 (cold start)
    "use_hidden_in_gnn": True,      # Stage 3 opt-1 hidden-in-GNN kept on
}
