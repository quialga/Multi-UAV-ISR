"""
isr/train/vec_env.py — Hand-rolled vector wrapper around PursuitEnv.

Runs N independent copies of ``PursuitEnv`` and exposes a batched
``step / reset`` interface that returns numpy arrays shaped
``(n_envs, n_agents, ...)``.  Auto-resets envs that finish an episode
so the rollout collector can pull continuous transitions without
having to handle done envs specially.

Design notes
------------
- We deliberately avoid SuperSuit / gym-vector-env wrappers in Stage 1
  to keep the code transparent.  This file is ~150 lines and you can
  trace every transition end-to-end.
- Auto-reset convention: when an env terminates / truncates at step t,
  we (a) record the done flag in the returned ``dones`` array so the
  trainer's GAE bootstrap is correct, and (b) immediately reset the
  env so the obs returned alongside the done is the FIRST obs of the
  next episode.  This is the same convention as gym's vector envs.
- Episode return tracking is built in: per-env cumulative reward sums
  are reset on done, and the value at done time is appended to a
  ring buffer of the last `episode_buffer_size` completed episodes.
- All envs share kinematic config but get distinct seeds (base_seed,
  base_seed+1, ..., base_seed+n_envs-1) for diversity.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from isr.env.pursuit_env import PursuitEnv


class VectorPursuitEnv:
    """
    Vectorised ``PursuitEnv``.  See module docstring.

    Parameters
    ----------
    n_envs                    Number of parallel envs.
    env_kwargs                Dict of constructor args forwarded to
                              ``PursuitEnv(**env_kwargs)``.  Must NOT
                              include ``seed`` — we set it per-env.
    base_seed                 Seed for env 0; env i gets base_seed + i.
    episode_buffer_size       Ring buffer length for completed-episode
                              statistics (return, length, n_caught).
                              Used by the trainer's logging.
    """

    def __init__(
        self,
        n_envs:              int,
        env_kwargs:          Dict[str, Any],
        base_seed:           int = 0,
        episode_buffer_size: int = 128,
    ) -> None:
        self.n_envs = int(n_envs)

        # Build envs with distinct seeds so rollouts cover varied
        # initial states from the very first step.
        self.envs: List[PursuitEnv] = []
        for i in range(self.n_envs):
            kwargs = dict(env_kwargs)
            kwargs["seed"] = int(base_seed + i)
            self.envs.append(PursuitEnv(**kwargs))

        e0 = self.envs[0]
        self.n_agents   = e0.n_blue
        self.action_dim = 2
        self.obs_dim    = e0._obs_dim
        self.possible_agents = list(e0.possible_agents)
        self.arena_size = e0.arena_size
        self.max_steps  = e0.max_steps

        # Per-env episode-tracking state (running totals; rotated on done).
        self._ep_return = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_length = np.zeros(self.n_envs, dtype=np.int32)
        # Completed episodes ring buffer (FIFO) for logging.
        self.episode_returns: Deque[float] = deque(maxlen=episode_buffer_size)
        self.episode_lengths: Deque[int]   = deque(maxlen=episode_buffer_size)
        self.episode_caught:  Deque[int]   = deque(maxlen=episode_buffer_size)

    # ------------------------------------------------------------------ #
    #  Reset / step                                                        #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset every env.  Returns obs shaped (n_envs, n_agents, obs_dim).

        If ``seed`` is given, re-seeds with base ``seed`` and per-env
        offsets — used at the very start of training for full
        determinism.
        """
        obs_batch = np.zeros(
            (self.n_envs, self.n_agents, self.obs_dim), dtype=np.float32
        )
        for i, env in enumerate(self.envs):
            s = None if seed is None else int(seed + i)
            obs_dict, _ = env.reset(seed=s)
            for j, name in enumerate(self.possible_agents):
                obs_batch[i, j] = obs_dict[name]
        self._ep_return[:] = 0.0
        self._ep_length[:] = 0
        return obs_batch

    def step(
        self,
        actions: np.ndarray,    # (n_envs, n_agents, action_dim)
    ) -> Tuple[
        np.ndarray,  # obs   (n_envs, n_agents, obs_dim)
        np.ndarray,  # rewards (n_envs, n_agents) — per-agent (= shared team r)
        np.ndarray,  # dones (n_envs,) — episode done flag (term OR trunc)
        List[Dict],  # per-env info dict (raw — first agent's view is fine)
    ]:
        assert actions.shape == (self.n_envs, self.n_agents, self.action_dim), \
            f"actions shape {actions.shape} != " \
            f"{(self.n_envs, self.n_agents, self.action_dim)}"

        obs_batch = np.zeros(
            (self.n_envs, self.n_agents, self.obs_dim), dtype=np.float32
        )
        reward_batch = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)
        done_batch   = np.zeros(self.n_envs, dtype=np.float32)
        infos: List[Dict] = []

        for i, env in enumerate(self.envs):
            action_dict = {
                name: actions[i, j]
                for j, name in enumerate(self.possible_agents)
            }
            obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_dict)

            # Stage 1: term/trunc are agent-uniform; just read first.
            first = self.possible_agents[0]
            done = bool(term_d[first] or trunc_d[first])

            # Episode tracking (use the shared team reward).
            self._ep_return[i] += float(rew_d[first])
            self._ep_length[i] += 1

            if done:
                # Capture the final state's "caught" count for logging
                # before we reset.
                snap = env.state_snapshot()
                n_caught = int((~snap["red_active"]).sum())
                self.episode_returns.append(float(self._ep_return[i]))
                self.episode_lengths.append(int(self._ep_length[i]))
                self.episode_caught.append(n_caught)
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0

                # Auto-reset.  Next-episode seed = some derived value;
                # use base + i + a step-counter is overkill — just let
                # the env's internal RNG advance naturally.
                obs_d, _ = env.reset()

            for j, name in enumerate(self.possible_agents):
                obs_batch[i, j]    = obs_d[name]
                reward_batch[i, j] = float(rew_d[name])
            done_batch[i] = 1.0 if done else 0.0
            infos.append(info_d)

        return obs_batch, reward_batch, done_batch, infos

    # ------------------------------------------------------------------ #
    #  Episode statistics                                                  #
    # ------------------------------------------------------------------ #

    def recent_episode_stats(self) -> Dict[str, float]:
        """Return summary stats over the last completed-episode window."""
        if not self.episode_returns:
            return {
                "n_completed": 0, "mean_return": 0.0, "std_return": 0.0,
                "mean_length": 0.0, "mean_caught": 0.0,
            }
        rs = np.array(self.episode_returns, dtype=np.float32)
        ls = np.array(self.episode_lengths, dtype=np.float32)
        cs = np.array(self.episode_caught,  dtype=np.float32)
        return {
            "n_completed": int(len(rs)),
            "mean_return": float(rs.mean()),
            "std_return":  float(rs.std()),
            "mean_length": float(ls.mean()),
            "mean_caught": float(cs.mean()),
        }

    def close(self) -> None:
        for env in self.envs:
            env.close()
