"""
isr/configs/stage1_default.py — Stage 1 hyperparameters.

Mirrors docs/design.md §3.9.  Override individual fields via CLI flags
in scripts/train_stage1.py rather than editing this file.
"""
from __future__ import annotations


STAGE1_DEFAULTS = {
    # ----- Environment ---------------------------------------------------
    "n_blue":         3,
    "n_red":          2,
    "arena_size":     100.0,
    "max_steps":      200,
    "capture_radius": 3.0,
    "dt":             1.0,

    # ----- Reward shape (env-level; inherited by every stage) ------------
    # Previously hard-coded in PursuitEnv._step.  Surfaced here so the
    # reward can be tuned/recorded without editing the env.  Defaults are
    # exactly the historical values, so every earlier run is reproducible.
    #   r_i = r_team + r_crash_i + r_clear_i - action_cost_i
    #   r_team = catch_reward * n_caught - step_cost   [- uncaught_penalty
    #                                        * n_uncaught at episode end]
    "catch_reward":     10.0,   # per red caught (shared: a team success)
    "step_cost":        0.05,   # per step (shared: mission time pressure)
    "uncaught_penalty": 5.0,    # per red still alive at episode end (shared)
    "action_cost_coef": 0.01,   # * |a_i|^2, INDIVIDUAL: own control effort

    # ----- Vectorisation -------------------------------------------------
    # Rollout collection runs n_envs copies of the env in parallel.
    # At rollout time we see (n_envs * rollout_steps * n_agents) samples,
    # which feeds the PPO buffer.  16 * 256 * 3 = 12 288 transitions per
    # rollout in the default config.
    "n_envs":         16,
    "rollout_steps":  300,

    # ----- PPO -----------------------------------------------------------
    "n_rollouts":     1000,        # total PPO updates
    "n_epochs":       10,          # PPO epochs per rollout
    "mb_size":        512,         # minibatch size
    "lr":             3e-4,
    "clip_eps":       0.2,
    "ent_coef":       0.01,
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "value_clip":     True,        # clip value updates by clip_eps too
    "normalize_adv":  True,        # standardise advantages per-minibatch

    # ----- Network -------------------------------------------------------
    "hidden":         [256, 256],
    "init_log_std":   0.0,         # log std at init; std = e^0 = 1
                                    # action box is [-1,1] so clipping is
                                    # heavy at init — policy learns to
                                    # shrink std rapidly.

    # ----- Logging / eval ------------------------------------------------
    "log_interval":   1,
    "eval_interval":  50,
    "eval_episodes":  20,
    "save_interval":  100,
}
