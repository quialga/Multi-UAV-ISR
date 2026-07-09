"""
isr/configs/stage3_default.py — Stage 3 hyperparameters.

Mirrors docs/stage3_design.md §2 (locked knobs).  Inherits from
STAGE1_DEFAULTS and overrides only what Stage 3 changes:

- Scenario scaled up to 5 blue vs 3 red on a 130x130 arena — the
  regime where Stage 2's GNN beat the MLP and where partial obs
  actually matters.
- ``sensor_radius = 40`` (about 30 % of arena side length).
- ``d_hidden`` unchanged (64) so the actor's GNN encoder and the
  CTDE critic can warm-start from a Stage 2 checkpoint.
- ``ent_coef = 0.018`` — the value that prevented policy collapse
  in the Stage 2 scaling GNN run (see docs/stage2_results.md §2.3).
- ``target_kl = 0.03`` — Stage 3 adds an optional per-epoch early
  stop on Schulman-k3 KL for the recurrent actor.

Override any field via CLI flags in scripts/train_stage3.py rather
than editing this file.
"""
from __future__ import annotations

from isr.configs.stage1_default import STAGE1_DEFAULTS


STAGE3_DEFAULTS = {
    **STAGE1_DEFAULTS,

    # ----- Environment (Stage 3 scale — see design §2) ------------------
    "n_blue":         5,
    "n_red":          3,
    "arena_size":     130.0,
    "sensor_radius":  40.0,

    # ----- Network ------------------------------------------------------
    # Locked at 64 for warm-start compatibility with the Stage 2 GNN
    # checkpoint (see docs/stage3_design.md §4-5).
    "d_hidden":       64,
    "n_msg_rounds":   2,

    # ----- PPO tweaks ---------------------------------------------------
    "ent_coef":       0.018,   # matches scaling_gnn from Stage 2 §2.3
    "target_kl":      0.03,    # None disables — Stage 3 default is on

    # ----- Belief-state distillation (optional) -------------------------
    # Auxiliary MSE loss regressing the actor's partial-obs GRU hidden
    # state toward the hidden state that would have been produced under
    # full observability.  0.0 disables the aux loss — the baseline
    # Stage 3 run trains without it, and it becomes a controlled
    # experiment (Phase 3.5).  See docs/stage3_design.md follow-up.
    "aux_hidden_coef": 0.0,

    # ----- Warm start ---------------------------------------------------
    # Path to the Stage 2 checkpoint whose GNN + critic head we copy
    # into the CTDE critic path at init.  Set to None to skip warm-start.
    "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",
}
