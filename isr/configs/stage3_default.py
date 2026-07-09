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
    # Auxiliary MSE loss regressing the actor's partial-obs GNN encoder
    # output (per-blue node embeddings) toward the CRITIC encoder's
    # full-obs output.  0.0 disables the aux loss — the baseline
    # Stage 3 run trains without it, and it becomes a controlled
    # experiment (Phase 3.5).  See docs/stage3_design.md follow-up.
    "aux_hidden_coef": 0.0,

    # When True, freezes the entire CTDE critic (encoder + trunk + head)
    # at the warm-started Stage 2 weights.  Aux target becomes a stable
    # Stage 2 oracle rather than a drifting one.  Set via --freeze-critic
    # on the CLI.  Requires warm_start_critic to be a valid path.
    #
    # Empirical note (Phase 3.5, 2026-07): option A (freeze_critic=True)
    # converged rapidly to ~2.9 catches then DEGRADED as the policy
    # drifted OOD from the frozen critic's V estimates.  Option C
    # (freeze_critic=False, aux on live critic) is the current
    # recommended config.  See docs/stage3_results.md for full
    # comparison and diagnosis.
    "freeze_critic": False,

    # ----- Phase 3.6 option 1: cross-blue hidden state sharing ---------
    # When True, prepend the previous per-blue GRU hidden state to the
    # blue node features fed into the actor's GNN encoder.  The GNN's
    # bb messages then carry belief state across blues.  Only the actor
    # path changes; the CTDE critic is unaffected.  See
    # docs/stage3_design.md §13.
    "use_hidden_in_gnn": False,

    # ----- Warm start ---------------------------------------------------
    # Path to the Stage 2 checkpoint whose GNN + critic head we copy
    # into the CTDE critic path at init.  Set to None to skip warm-start.
    "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",
}
