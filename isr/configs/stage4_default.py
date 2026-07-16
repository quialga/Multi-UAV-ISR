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

    # ----- Belief map ----------------------------------------------------
    "use_belief_maps":     True,
    "belief_grid_size":    26,     # H = W (arena_size / cell_size)
    # 4 channels: {P(enemy) Bayesian, P(obstacle) Bayesian,
    #              ally_positions deterministic, self_position deterministic}
    # Channels 0-1 use log-odds; channels 2-3 are direct binary
    # overlays computed from ground truth (perfect self-GPS + TDL uplink
    # for allies).  Only channels 0-1 pass through sigmoid at the CNN.
    "belief_channels":     4,
    "belief_clip":         10.0,   # log-odds clip (channels 0-1 only)

    # ----- Sensor model --------------------------------------------------
    "p_TP":                0.85,
    "p_FP":                0.15,
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

    # ----- Policy --------------------------------------------------------
    "belief_encoder_out_dim": 128,

    # ----- Warm start ----------------------------------------------------
    # Stage 3 checkpoints have red-side tensors + different critic input
    # shapes.  Cold-start.
    "warm_start_critic": None,

    # ----- Aux loss ------------------------------------------------------
    # Off for the Stage 4 baseline (design §6).  Diagnostic BCE logged
    # to TensorBoard but no gradient flows through the actor path.
    "aux_hidden_coef":   0.0,
    "freeze_critic":     False,
    "use_hidden_in_gnn": True,     # Stage 3 opt-1 hidden-in-GNN kept on
}
