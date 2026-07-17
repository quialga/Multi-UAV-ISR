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
