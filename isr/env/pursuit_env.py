"""
isr/env/pursuit_env.py — Stage 1 pursuit-evasion environment.

PettingZoo ParallelEnv.  ``N_blue`` UAVs (the agents) chase ``N_red``
targets (env-internal, scripted policy in Stage 1) inside a 2D
continuous arena.

This is the regression bedrock for the curriculum — the simplest
possible MARL pursuit task with continuous actions.  Every later
stage swaps one thing at a time while keeping the rest constant.

See ``docs/design.md §3`` for the full spec (arena, kinematics,
observation, action, reward, hyperparameters, acceptance criterion).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from isr.env.entities import BLUE_UAV, RED_TARGET


# ---------------------------------------------------------------------------
# Default Stage 1 red policy: every red flees its nearest blue UAV.
# ---------------------------------------------------------------------------

def run_from_nearest_uav(
    blue_pos:   np.ndarray,   # (N_blue, 2)
    red_pos:    np.ndarray,   # (N_red,  2)
    red_active: np.ndarray,   # (N_red,) bool
) -> np.ndarray:
    """
    Scripted red policy used by ``PursuitEnv`` in Stage 1.

    Each *active* red target computes the unit vector pointing **away**
    from its nearest blue UAV and uses that as its acceleration command
    (magnitude exactly 1.0 — the policy is "always run at max
    effort").  Caught reds (``active=False``) return zero acceleration.

    The function is pure (no hidden state) so it's safe to swap out for
    a learned policy in Stage 4 by passing a different callable to the
    env's ``red_policy`` constructor argument.

    Returns
    -------
    (N_red, 2) float32 — per-red acceleration commands.
    """
    n_red = red_pos.shape[0]
    out = np.zeros((n_red, 2), dtype=np.float32)
    for i in range(n_red):
        if not red_active[i]:
            continue
        diffs = red_pos[i] - blue_pos              # vectors blue -> this red
        dists = np.linalg.norm(diffs, axis=1)
        nearest = int(np.argmin(dists))
        d_vec = diffs[nearest]
        norm = float(np.linalg.norm(d_vec))
        if norm > 1e-8:
            out[i] = d_vec / norm                  # unit vector AWAY from nearest blue
    return out


# ---------------------------------------------------------------------------
# Stage 1 environment
# ---------------------------------------------------------------------------

class PursuitEnv(ParallelEnv):
    """
    Stage 1 pursuit-evasion environment (full observability, scripted red).

    Parameters
    ----------
    n_blue          Number of blue UAVs (PettingZoo agents).
    n_red           Number of red targets (env-internal in Stage 1).
    arena_size      Side length of the square arena (units).
    max_steps       Episode horizon — truncate after this many steps.
    capture_radius  A red is caught when within this Euclidean distance
                    of any blue UAV.
    dt              Simulation step size (units of time).
    red_policy      Callable ``(blue_pos, red_pos, red_active) -> (N_red, 2)``
                    returning per-red acceleration commands.  Defaults to
                    ``run_from_nearest_uav``.  Stage 4 will pass a learned
                    policy here without further env changes.
    seed            Initial RNG seed.

    Observation (per blue UAV)
    --------------------------
    Identical *global* observation for every blue agent in Stage 1
    (parameter sharing + ``self_idx_onehot`` lets a single network
    serve every agent while knowing which one it is).  Flattened
    vector laid out as:

        [blue_pos_normalised   (N_blue * 2),
         blue_vel_normalised   (N_blue * 2),
         red_pos_normalised    (N_red  * 2),    # zero for caught
         red_vel_normalised    (N_red  * 2),    # zero for caught
         red_active_mask       (N_red,),
         self_idx_onehot       (N_blue,),
         time_remaining        (1,)]

    Action (per blue UAV)
    ---------------------
    Continuous 2D acceleration command in [-1, 1]^2.

    Reward
    ------
    Shared team reward, computed once per step and given to every blue
    UAV.  See ``docs/design.md §3.7``.
    """

    metadata = {"render_modes": ["human"], "name": "pursuit_v0"}

    def __init__(
        self,
        n_blue:         int                = 3,
        n_red:          int                = 2,
        arena_size:     float              = 100.0,
        max_steps:      int                = 200,
        capture_radius: float              = 3.0,
        dt:             float              = 1.0,
        red_policy:     Optional[Callable] = None,
        seed:           Optional[int]      = None,
    ):
        super().__init__()
        self.n_blue         = int(n_blue)
        self.n_red          = int(n_red)
        self.arena_size     = float(arena_size)
        self.max_steps      = int(max_steps)
        self.capture_radius = float(capture_radius)
        self.dt             = float(dt)
        self.red_policy     = red_policy or run_from_nearest_uav

        self._rng = np.random.default_rng(seed)

        # PettingZoo agent set: blue UAVs only.  Red is scripted inside
        # the env in Stage 1; Stage 4 will refactor or override
        # ``red_policy`` without touching the agent set.
        self.possible_agents = [f"blue_{i}" for i in range(self.n_blue)]
        self.agents = list(self.possible_agents)

        # Observation dim — see class docstring layout above.
        self._obs_dim = (
            self.n_blue * 2 +    # blue pos
            self.n_blue * 2 +    # blue vel
            self.n_red  * 2 +    # red pos
            self.n_red  * 2 +    # red vel
            self.n_red       +   # red active mask
            self.n_blue      +   # self_idx_onehot
            1                    # time remaining
        )

        self._action_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,),
                          dtype=np.float32)
            for a in self.possible_agents
        }

        # Mutable state — initialised in reset().
        self._blue_pos:   Optional[np.ndarray] = None
        self._blue_vel:   Optional[np.ndarray] = None
        self._red_pos:    Optional[np.ndarray] = None
        self._red_vel:    Optional[np.ndarray] = None
        self._red_active: Optional[np.ndarray] = None
        self._t:          int                  = 0
        # Diagnostic captured from the last step (rendering / logging).
        self._last_n_caught: int = 0

    # ------------------------------------------------------------------ #
    #  PettingZoo API                                                     #
    # ------------------------------------------------------------------ #

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        L = self.arena_size
        # Uniform random initial positions, zero initial velocities.
        # No minimum-separation constraint between blue and red in
        # Stage 1 — caught-at-spawn is rare given typical L/r ratios
        # and the policy will learn to handle it.  We can add a
        # min-separation constraint later if it matters empirically.
        self._blue_pos = self._rng.uniform(0.0, L,
                                           size=(self.n_blue, 2)).astype(np.float32)
        self._blue_vel = np.zeros((self.n_blue, 2), dtype=np.float32)
        self._red_pos  = self._rng.uniform(0.0, L,
                                           size=(self.n_red, 2)).astype(np.float32)
        self._red_vel  = np.zeros((self.n_red, 2), dtype=np.float32)
        self._red_active = np.ones(self.n_red, dtype=bool)
        self._t = 0
        self._last_n_caught = 0
        self.agents = list(self.possible_agents)

        obs = {a: self._build_obs(i) for i, a in enumerate(self.possible_agents)}
        info = {a: {} for a in self.possible_agents}
        return obs, info

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],   # obs
        Dict[str, float],        # rewards
        Dict[str, bool],         # terminations
        Dict[str, bool],         # truncations
        Dict[str, Dict],         # infos
    ]:
        # 1. Pack blue actions into a (N_blue, 2) array, clipping to the
        #    action box.  Missing entries default to zeros (defensive —
        #    PettingZoo expects every alive agent to act each step).
        blue_a = np.zeros((self.n_blue, 2), dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                blue_a[i] = np.clip(
                    np.asarray(actions[agent], dtype=np.float32),
                    -1.0, 1.0,
                )

        # 2. Red scripted action (only active reds will produce non-zero).
        red_a = self.red_policy(self._blue_pos, self._red_pos, self._red_active)
        red_a = np.clip(red_a.astype(np.float32), -1.0, 1.0)
        red_a[~self._red_active] = 0.0  # defensive

        # 3. Integrate blue kinematics (double-integrator with axis-wise
        #    velocity cap and arena wall stop).
        self._blue_pos, self._blue_vel = self._integrate(
            self._blue_pos, self._blue_vel, blue_a, BLUE_UAV.v_max,
        )

        # 4. Integrate red kinematics.
        self._red_pos, self._red_vel = self._integrate(
            self._red_pos, self._red_vel, red_a, RED_TARGET.v_max,
        )
        # Caught reds stay frozen at their last position.
        self._red_vel[~self._red_active] = 0.0

        # 5. Capture check — vectorised pairwise distance.
        n_caught_this_step = 0
        if self._red_active.any():
            # (N_red, N_blue) pairwise distances
            dists = np.linalg.norm(
                self._red_pos[:, None, :] - self._blue_pos[None, :, :],
                axis=-1,
            )
            min_dists = dists.min(axis=1)                       # (N_red,)
            newly_caught = self._red_active & (min_dists <= self.capture_radius)
            n_caught_this_step = int(newly_caught.sum())
            self._red_active &= ~newly_caught
            # Zero velocities of newly caught reds (cosmetic, but
            # keeps the obs clean).
            self._red_vel[newly_caught] = 0.0
        self._last_n_caught = n_caught_this_step

        # 6. Reward — see docs/design.md §3.7.
        catch_bonus = 10.0 * n_caught_this_step
        action_cost = 0.01 * float(np.sum(blue_a ** 2))   # sum over UAVs and axes
        step_cost   = 0.05
        r = catch_bonus - action_cost - step_cost

        # 7. Termination check.
        self._t += 1
        all_caught = bool(not self._red_active.any())
        time_up    = self._t >= self.max_steps
        terminated = all_caught
        truncated  = time_up and not all_caught
        if terminated or truncated:
            n_uncaught = int(self._red_active.sum())
            r += -5.0 * n_uncaught                          # terminal penalty
            self.agents = []                                # PettingZoo convention

        # 8. Pack the per-agent dicts.  Shared team reward — every
        #    blue agent gets the same r.
        rewards     = {a: float(r) for a in self.possible_agents}
        terminateds = {a: terminated for a in self.possible_agents}
        truncateds  = {a: truncated  for a in self.possible_agents}
        infos = {
            a: {
                "n_caught_this_step": n_caught_this_step,
                "n_red_remaining":    int(self._red_active.sum()),
                "t":                  self._t,
            }
            for a in self.possible_agents
        }
        obs = {a: self._build_obs(i) for i, a in enumerate(self.possible_agents)}
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        """No built-in render — see ``isr.utils.render`` for matplotlib."""
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    #  Inspector / helpers                                                #
    # ------------------------------------------------------------------ #

    def state_snapshot(self) -> Dict[str, Any]:
        """
        Read-only snapshot of full env state.  Used by the renderer and
        by tests that need to assert on positions / velocities directly.
        """
        return {
            "t":           self._t,
            "blue_pos":    self._blue_pos.copy(),
            "blue_vel":    self._blue_vel.copy(),
            "red_pos":     self._red_pos.copy(),
            "red_vel":     self._red_vel.copy(),
            "red_active":  self._red_active.copy(),
            "n_caught_last_step": self._last_n_caught,
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _integrate(
        self,
        pos:    np.ndarray,
        vel:    np.ndarray,
        accel:  np.ndarray,
        v_max:  float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Double-integrator step with axis-wise velocity cap and arena
        wall stop.  Returns ``(new_pos, new_vel)``.

        Wall handling: when a position update would leave the arena,
        we clip the position to the wall and zero the velocity
        component on that axis.  No bounce — keeps reasoning simple
        and the agent learns to back off.
        """
        new_vel = np.clip(vel + accel * self.dt, -v_max, v_max).astype(np.float32)
        new_pos = pos + new_vel * self.dt
        for axis in (0, 1):
            below = new_pos[:, axis] < 0.0
            above = new_pos[:, axis] > self.arena_size
            new_pos[below, axis] = 0.0
            new_pos[above, axis] = self.arena_size
            new_vel[below | above, axis] = 0.0
        return new_pos.astype(np.float32), new_vel

    def _build_obs(self, blue_idx: int) -> np.ndarray:
        """
        Build the flattened global observation for blue UAV ``blue_idx``.

        Only the ``self_idx_onehot`` block differs between blue agents.
        Everything else (positions, velocities, masks) is identical —
        Stage 1 is fully observable from a shared centralised viewpoint.
        """
        L = self.arena_size
        blue_pos_n = (self._blue_pos / L).astype(np.float32)
        blue_vel_n = (self._blue_vel / BLUE_UAV.v_max).astype(np.float32)
        red_pos_n  = (self._red_pos / L).astype(np.float32)
        red_vel_n  = (self._red_vel / RED_TARGET.v_max).astype(np.float32)
        # Zero out caught reds' pos/vel so the network can't latch onto
        # stale positions.  The active_mask carries the "this red still
        # exists" signal.
        red_pos_n[~self._red_active] = 0.0
        red_vel_n[~self._red_active] = 0.0

        active_mask = self._red_active.astype(np.float32)
        self_onehot = np.zeros(self.n_blue, dtype=np.float32)
        self_onehot[blue_idx] = 1.0
        time_remaining = np.array(
            [(self.max_steps - self._t) / self.max_steps], dtype=np.float32
        )

        return np.concatenate([
            blue_pos_n.flatten(),
            blue_vel_n.flatten(),
            red_pos_n.flatten(),
            red_vel_n.flatten(),
            active_mask,
            self_onehot,
            time_remaining,
        ]).astype(np.float32)
