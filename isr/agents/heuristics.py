"""
isr/agents/heuristics.py — scripted policies for blue and red.

Two distinct interfaces, kept separate on purpose:

1. **Blue agents** (PettingZoo agent perspective).  Class-based, with
   the signature ``act(obs, env, agent) -> action``.  Used as the blue
   policy in tests, as baselines for the PPO policy to beat, and
   inside the rendering / smoke scripts.  Mirrors the eventual
   neural-policy interface so PPO and heuristics are drop-in
   interchangeable in the runner.

2. **Red policies** (env-internal scripted opposition).  Plain
   callables of shape
   ``(blue_pos, red_pos, red_active) -> (N_red, 2) actions``.
   Passed to ``PursuitEnv(red_policy=...)``.  Replaces the default
   ``run_from_nearest_uav``.  Stage 4 will swap one of these for a
   learned policy without env changes.

Why two interfaces?  Blue agents are first-class PettingZoo agents and
each one acts once per env step via the standard agent loop.  Red
"agents" in Stage 1 are env-internal — the env applies them all in
one batched call inside ``step()``, which doesn't fit the
agent-by-agent ``act`` pattern.  Forcing them into the same shape
would either complicate the env loop or stop us from vectorising the
red step.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Re-export the default red policy so callers can grab everything from
# one module: ``from isr.agents.heuristics import run_from_nearest_uav``.
from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav  # noqa: F401


# ===========================================================================
#  Blue agents — class-based act(obs, env, agent) -> action
# ===========================================================================

class HeuristicBlueAgent:
    """Base class for scripted blue policies.  Override ``act``."""

    def act(
        self,
        obs:   np.ndarray,
        env:   PursuitEnv,
        agent: str,
    ) -> np.ndarray:
        """
        Return a 2D acceleration action in [-1, 1]^2 for the named
        blue agent.  Heuristics may peek at ``env.state_snapshot()``
        for ground truth; the eventual neural policy must work from
        ``obs`` alone.
        """
        raise NotImplementedError


class RandomAgent(HeuristicBlueAgent):
    """
    Uniform random action in the env's action box.  Sanity check —
    expected to perform poorly (small negative reward dominated by
    step + action costs, few captures).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs, env, agent):
        return self._rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


class GreedyPursuer(HeuristicBlueAgent):
    """
    Pure greedy nearest-target pursuit.

    Each blue UAV independently picks the closest *active* red target
    and accelerates toward it with unit magnitude.  No coordination
    between blue agents — three UAVs may all converge on the same
    target while another is unattended.

    This is the **Stage 1 baseline the PPO blue policy must beat**
    (by >=20% mean episode return, per ``docs/design.md §3.10``).  The
    20% gap is meaningful: beating greedy requires the learned policy
    to do something a per-UAV nearest-target rule cannot — split
    coverage so multiple UAVs don't redundantly chase the same red,
    anticipate where reds will dodge to, etc.  Beating greedy is the
    headline check that Stage 1 actually learns *coordination*, not
    just "go to the nearest red".
    """

    def act(self, obs, env, agent):
        snap = env.state_snapshot()
        blue_pos   = snap["blue_pos"]
        red_pos    = snap["red_pos"]
        red_active = snap["red_active"]

        if not red_active.any():
            return np.zeros(2, dtype=np.float32)

        my_idx = env.possible_agents.index(agent)
        my_pos = blue_pos[my_idx]

        diffs = red_pos - my_pos                # (N_red, 2)
        dists = np.linalg.norm(diffs, axis=1)   # (N_red,)
        # Caught reds are not pursuit targets — mask their distance to inf
        # so argmin picks the closest *active* red.
        dists = np.where(red_active, dists, np.inf)
        nearest = int(np.argmin(dists))

        d_vec = diffs[nearest]
        norm = float(np.linalg.norm(d_vec))
        if norm < 1e-8:
            # Already on top of the target — capture should have fired
            # this step; output zero so we don't waste action cost.
            return np.zeros(2, dtype=np.float32)
        return (d_vec / norm).astype(np.float32)


# ===========================================================================
#  Red policies — (blue_pos, red_pos, red_active) -> (N_red, 2)
# ===========================================================================
#
# Functional API matching ``PursuitEnv(red_policy=...)``.  The env calls
# these once per step with the current state arrays; the callable
# returns one row of acceleration per red target.  Caught reds (active
# = False) must produce zero action.

def stationary_red(
    blue_pos:   np.ndarray,
    red_pos:    np.ndarray,
    red_active: np.ndarray,
) -> np.ndarray:
    """
    Red doesn't move.  Easiest possible adversary — given Stage 1's
    blue/red speed advantage, blue should reliably catch every red
    within max_steps.  Useful as the absolute floor in benchmarks
    ("if blue can't beat this, the training pipeline is broken").
    """
    return np.zeros_like(red_pos, dtype=np.float32)


def random_red(seed: Optional[int] = None):
    """
    Returns a closure: red picks uniform random acceleration each step.
    Slightly harder than stationary (red occasionally drifts in a useful
    direction), much easier than ``run_from_nearest_uav``.  Used to
    bracket the difficulty curve.
    """
    rng = np.random.default_rng(seed)

    def policy(
        blue_pos:   np.ndarray,
        red_pos:    np.ndarray,
        red_active: np.ndarray,
    ) -> np.ndarray:
        out = rng.uniform(-1.0, 1.0, size=red_pos.shape).astype(np.float32)
        out[~red_active] = 0.0
        return out

    return policy


# ``run_from_nearest_uav`` is the env default; re-exported above for
# uniform import location.

__all__ = [
    "HeuristicBlueAgent", "RandomAgent", "GreedyPursuer",
    "stationary_red", "random_red", "run_from_nearest_uav",
]
