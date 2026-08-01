"""
gazebo/brain.py — the closed-loop decision logic, with NO ROS.

This is the "brain" milestone 3 runs: given the drones' current
positions, it produces the velocity commands for the next tick.  It is
deliberately free of rclpy / Gazebo so that

  * the ROS node (gazebo/policy_bridge.py) is a thin wrapper —
    read odometry -> brain.tick(...) -> publish — and
  * the parity test (tests/test_bridge_parity.py) can drive the EXACT
    same code without Gazebo and check it reproduces the training
    rollout bit-for-bit.

One source of truth, no second implementation to drift out of sync.

WHAT ONE TICK DOES (see PolicyBridge's module docstring for the plain-
language version).  The order below is chosen to match PursuitEnv.step
+ the training eval loop EXACTLY, including the order in which random
numbers are consumed (sensor-noise draws in the observation, Bernoulli
draws in the belief update) — that RNG alignment is what makes the
deployment perceive the same world it trained in, and is what the
parity test verifies:

    1. SYNC     copy the given positions into the shadow env; velocities
                come from our own integrator state (the env's post-step
                velocities, not odometry twist).
    tick 1 only:  skip straight to observe+act on the reset-time belief
                (PursuitEnv.reset already ran one belief update before
                the first observation; a second here would double-fuse).
    2. REFEREE  the env's capture rule: red within capture_radius of any
                blue -> caught, frozen, its belief blob wiped.
    3. CLOCK    advance t; end the episode when all reds are caught or
                max_steps is reached.
    4. PERCEIVE the env's own belief update (noisy sweeps + occlusion).
    5. OBSERVE  build the typed-graph observation (draws sensor noise).
    6. THINK    the frozen policy, deterministically, carrying its GRU
                hidden state across ticks.
    7. ACT      integrate accel -> velocity with the env's exact motion
                contract (walls + obstacle rollback); return the exec
                velocities to command.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from isr.env.pursuit_env import PursuitEnv
from gazebo.kinematics import integrate_cmd

# The env's kinematic contract (isr/env/entities.py + PursuitEnv
# defaults).  BLUE_UAV.v_max = 1.5, RED_TARGET.v_max = 1.0, dt = 1.0.
DT         = 1.0
V_MAX_BLUE = 1.5
V_MAX_RED  = 1.0


def env_kwargs_from_checkpoint(a: dict) -> dict:
    """Rebuild the training env's configuration from the checkpoint's
    saved arguments, so the shadow env perceives EXACTLY like the env
    the policy was trained in (same noise rates, same belief decay,
    same grid...).  Mirrors scripts/train_stage4.py's env_kwargs."""
    return dict(
        n_blue                   = a["n_blue"],
        n_red                    = a["n_red"],
        arena_size               = a["arena_size"],
        max_steps                = a["max_steps"],
        capture_radius           = a["capture_radius"],
        sensor_radius            = a["sensor_radius"],
        n_obstacles              = a["n_obstacles"],
        obstacle_radius_min      = a["obstacle_radius_min"],
        obstacle_radius_max      = a["obstacle_radius_max"],
        obstacle_spawn_clearance = a["obstacle_spawn_clearance"],
        use_belief_maps          = True,
        belief_grid_size         = a["belief_grid_size"],
        belief_channels          = a["belief_channels"],
        belief_clip              = a["belief_clip"],
        p_TP                     = a["p_tp"],
        p_FP                     = a["p_fp"],
        ray_step_size            = a["ray_step_size"],
        enemy_belief_decay       = a.get("enemy_belief_decay", 1.0),
        enemy_belief_diffusion   = a.get("enemy_belief_diffusion", 0.0),
        sensor_pos_noise_std     = a.get("sensor_pos_noise_std", 1.0),
        # Crash knobs affect only rewards (unused at deployment) but
        # blue_collision_radius also feeds the referee's ally counter.
        crash_obstacle_penalty   = a.get("crash_obstacle_penalty", 0.0),
        crash_blue_penalty       = a.get("crash_blue_penalty", 0.0),
        blue_collision_radius    = a.get("blue_collision_radius", 2.0),
    )


@dataclass
class TickResult:
    """What one tick tells the caller to do.

    blue_exec / red_exec — (n, 2) velocities to command this tick, or
        None when the episode just ended (caller should stop all drones).
    blue_accel — (n_blue, 2) the raw policy action this tick, or None on
        the terminal tick.  Exposed for the parity test / diagnostics.
    captures — red indices caught THIS tick (for logging).
    done — True on the tick that ends the episode.
    """
    blue_exec:  Optional[np.ndarray]
    red_exec:   Optional[np.ndarray]
    blue_accel: Optional[np.ndarray]
    captures:   List[int]
    done:       bool


class ClosedLoopBrain:
    """Milestone-3 decision logic over an already-reset shadow env."""

    def __init__(self, env: PursuitEnv, policy: GNNStage4Policy,
                 device: Optional[torch.device] = None) -> None:
        self.env = env
        self.policy = policy
        self.device = device or torch.device("cpu")
        self.n_blue = env.n_blue
        self.n_red = env.n_red

        self.hidden = policy.initial_hidden(1, self.device)
        # Commanded velocities = the drones' kinematic state (the env's
        # post-step velocities); we integrate accel into these.
        self.blue_vel = np.zeros((self.n_blue, 2), dtype=np.float32)
        self.red_vel = np.zeros((self.n_red, 2), dtype=np.float32)

        # Rising-edge crash-event counters (a blue parked against a
        # pillar counts ONCE), matching the training eval's counting.
        self.prev_obs_mask = np.zeros(self.n_blue, dtype=bool)
        self.prev_ally_mask = np.zeros(self.n_blue, dtype=bool)
        self.obs_crash_events = 0
        self.ally_crash_events = 0

        self._ticked_once = False
        self.done = False

    # ------------------------------------------------------------------ #

    @property
    def n_caught(self) -> int:
        return self.n_red - int(self.env._red_active.sum())

    def tick(self, blue_pos: np.ndarray, red_pos: np.ndarray) -> TickResult:
        """One decision.  ``blue_pos`` / ``red_pos`` are the drones'
        current positions (from Gazebo odometry, or from the parity
        test's own env-faithful integration)."""
        env = self.env
        if self.done:
            return TickResult(None, None, None, [], True)

        # ---- 1. SYNC ---------------------------------------------------
        env._blue_pos = np.asarray(blue_pos, dtype=np.float32).copy()
        env._blue_vel = self.blue_vel.copy()
        env._red_pos = np.asarray(red_pos, dtype=np.float32).copy()
        red_v = self.red_vel.copy()
        red_v[~env._red_active] = 0.0
        env._red_vel = red_v

        # ---- tick 1 = the env's "reset observation" -------------------
        if not self._ticked_once:
            self._ticked_once = True
            return self._decide([])

        # ---- 2. REFEREE (PursuitEnv.step step-5) ----------------------
        captures: List[int] = []
        if env._red_active.any():
            dists = np.linalg.norm(
                env._red_pos[:, None, :] - env._blue_pos[None, :, :], axis=-1)
            newly = env._red_active & (dists.min(axis=1) <= env.capture_radius)
            if newly.any():
                env._red_active &= ~newly
                env._red_vel[newly] = 0.0
                if env._belief_maps is not None:
                    env._clear_enemy_belief_at(env._red_pos[newly])
                captures = [int(j) for j in np.flatnonzero(newly)]

        # Report card, ally half: env's rule (pairwise <= radius),
        # rising edges.  (Obstacle half is counted in _decide from the
        # integrate_cmd attempted-entry mask.)
        bb = np.linalg.norm(
            env._blue_pos[:, None, :] - env._blue_pos[None, :, :], axis=-1)
        np.fill_diagonal(bb, np.inf)
        ally_mask = (bb <= env.blue_collision_radius).any(axis=1)
        self.ally_crash_events += int(np.count_nonzero(
            ally_mask & ~self.prev_ally_mask))
        self.prev_ally_mask = ally_mask

        # ---- 3. CLOCK / termination (PursuitEnv.step step-7) ----------
        env._t += 1
        if (not env._red_active.any()) or env._t >= env.max_steps:
            self.done = True
            return TickResult(None, None, None, captures, True)

        # ---- 4. PERCEIVE ----------------------------------------------
        env._update_belief_maps()
        return self._decide(captures)

    def _decide(self, captures: List[int]) -> TickResult:
        """Observe -> policy -> env-faithful commands (steps 5-7)."""
        env = self.env

        # ---- 5. OBSERVE (draws sensor-position noise) -----------------
        obs = env.structured_belief_observation()
        obs_t = {k: torch.from_numpy(v).float().unsqueeze(0)
                 for k, v in obs.items()}
        partial_obs, _ = split_stage4_obs(obs_t)

        # ---- 6. THINK (deterministic, carries GRU hidden) -------------
        with torch.no_grad():
            mean, self.hidden = self.policy.act_deterministic(
                partial_obs, self.hidden)
        accel = np.clip(mean.squeeze(0).cpu().numpy().astype(np.float32),
                        -1.0, 1.0)

        # ---- 7. ACT: env's exact motion contract ----------------------
        exec_b, self.blue_vel, hit_b = integrate_cmd(
            env._blue_pos, self.blue_vel, accel, V_MAX_BLUE, DT,
            env.arena_size, env._obstacle_pos, env._obstacle_r)
        self.obs_crash_events += int(np.count_nonzero(
            hit_b & ~self.prev_obs_mask))
        self.prev_obs_mask = hit_b

        # Reds: the shadow env's own red policy, called with the EXACT
        # arg list PursuitEnv.step uses (incl. arena_size wall repulsion)
        # so the Gazebo adversary tracks any red-policy change.
        red_a = env.red_policy(
            env._blue_pos, env._red_pos, env._red_active,
            env._obstacle_pos, env._obstacle_r, env.arena_size)
        red_a = np.clip(red_a.astype(np.float32), -1.0, 1.0)
        exec_r, self.red_vel, _ = integrate_cmd(
            env._red_pos, self.red_vel, red_a, V_MAX_RED, DT,
            env.arena_size, env._obstacle_pos, env._obstacle_r)
        self.red_vel[~env._red_active] = 0.0
        exec_r[~env._red_active] = 0.0

        return TickResult(exec_b, exec_r, accel, captures, False)
