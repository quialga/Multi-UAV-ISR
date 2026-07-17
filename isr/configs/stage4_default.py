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
    "belief_grid_size":    52,     # H = W (arena_size / cell_size = 130 / 2.5)
    # 3 channels: {P(enemy) Bayesian, P(obstacle) Bayesian,
    #              ally_positions deterministic}.
    # Channels 0-1 are log-odds; channel 2 is a direct binary overlay
    # from ground truth (TDL uplink model).  Channels 0-1 feed the
    # env's peak extractor (belief_peaks_enemy / belief_peaks_obstacle);
    # channel 2 is only used by the (default-off) actor CNN path.
    "belief_channels":     3,
    "belief_clip":         10.0,   # log-odds clip (channels 0-1 only)

    # ----- Belief peaks (v5.3, primary position signal) ------------------
    # The actor's enemy/obstacle POSITION signal.  Top-K (dx, dy, conf)
    # detections extracted from belief-map channels 0/1 -- the noisy
    # tracker output a real ISR operator sees.  Replaced the actor CNN,
    # which an ablation showed could not recover position from the
    # log-odds tensor.  n_enemy_peaks tracks n_red; n_obstacle_peaks
    # tracks n_obstacles (both set from the env at train time).
    "peak_msg_dim":        32,

    # ----- Actor belief CNN (v5.3: OFF; ablation path only) --------------
    # Set use_actor_belief_cnn=True to additionally feed an ego-centric
    # KxK belief crop through a per-UAV CNN.  K = 33 at cell_size 2.5m
    # covers ~41m radius (the sensor_radius=40m disk with margin).
    # These knobs are inert while use_actor_belief_cnn is False.
    "use_actor_belief_cnn": False,
    "belief_window_size":   33,
    "belief_encoder_out_dim": 128,

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
