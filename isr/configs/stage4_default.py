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
    # 3 channels: {P(enemy) Bayesian, P(obstacle) Bayesian,
    #              ally_positions deterministic}
    # Channels 0-1 use log-odds (sigmoid at CNN input); channel 2 is
    # a direct binary overlay from ground truth (TDL uplink model).
    # The self-position channel from v2 is DROPPED because the actor
    # now consumes ego-centric windows (see ``belief_window_size``),
    # so the window is inherently centred on the UAV -- redundant to
    # also mark it with a "1" channel.
    "belief_channels":     3,
    "belief_clip":         10.0,   # log-odds clip (channels 0-1 only)

    # ----- Ego-centric window (Phase 4 v3) -------------------------------
    # For each UAV, extract a KxK crop of the belief map centred on
    # its own cell (zero-padded at arena edges), and feed THAT to the
    # actor's belief CNN instead of the global (26x26) map.  Fixes
    # the "CNN can't produce ego-centric features" problem that stalled
    # v1/v2 at the random-policy baseline.  See docs/stage4_backlog.md
    # discussion.
    # K = 17 at cell_size 5m covers 42.5m radius, encompassing the
    # sensor_radius=40m disk with one cell margin.
    "belief_window_size":  17,

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
