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
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import (
    stationary_red, random_red, run_from_nearest_uav,
)


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
        red_policy_mix:      Optional[Sequence[Tuple[str, float]]] = None,
        obs_format:          str = "flat",
    ) -> None:
        """
        red_policy_mix: optional list of (name, weight) tuples specifying
            a per-episode categorical distribution over red policies.
            names supported: 'stationary', 'random', 'run'.  Weights are
            normalised.  If None (default), every env uses whatever red
            policy was passed via ``env_kwargs['red_policy']`` (or the
            env default = run_from_nearest_uav) for every episode — the
            legacy v1 behaviour.

            v2 default in scripts/train_stage1.py is
            [('stationary', 1), ('random', 1), ('run', 1)] — uniform
            mix — to eliminate the OOD failure documented in
            docs/stage1_analysis.md §3 (Failure mode 1).
        """
        self.n_envs = int(n_envs)

        # Build envs with distinct seeds so rollouts cover varied
        # initial states from the very first step.
        self.envs: List[PursuitEnv] = []
        for i in range(self.n_envs):
            kwargs = dict(env_kwargs)
            kwargs["seed"] = int(base_seed + i)
            self.envs.append(PursuitEnv(**kwargs))

        # ---- Red-policy mixing setup --------------------------------
        # A single RNG for the mixing decision — independent from any
        # env's internal RNG so rollout determinism stays per-env-seed.
        self._mix_rng = np.random.default_rng(int(base_seed))
        self._red_policy_names: Optional[List[str]] = None
        self._red_policy_probs: Optional[np.ndarray] = None
        if red_policy_mix is not None:
            names   = [n for n, _ in red_policy_mix]
            weights = np.asarray([w for _, w in red_policy_mix],
                                 dtype=np.float64)
            total = float(weights.sum())
            if total <= 0.0:
                raise ValueError("red_policy_mix weights must sum > 0")
            self._red_policy_names = names
            self._red_policy_probs = weights / total

        if obs_format not in ("flat", "structured"):
            raise ValueError(
                f"obs_format must be 'flat' or 'structured', got {obs_format!r}"
            )
        self.obs_format = obs_format

        e0 = self.envs[0]
        self.n_agents   = e0.n_blue
        self.action_dim = 2
        self.obs_dim    = e0._obs_dim
        self.possible_agents = list(e0.possible_agents)
        self.arena_size = e0.arena_size
        self.max_steps  = e0.max_steps
        # Graph-obs sizing, exposed for GraphRolloutBuffer construction.
        self.blue_feat_dim = e0.blue_feat_dim
        self.red_feat_dim  = e0.red_feat_dim
        self.edge_feat_dim = e0.edge_feat_dim
        self.n_bb_edges    = e0.n_bb_edges
        self.n_rb_edges    = e0.n_rb_edges
        self.n_blue        = e0.n_blue
        self.n_red         = e0.n_red

        # Per-env episode-tracking state (running totals; rotated on done).
        self._ep_return = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_length = np.zeros(self.n_envs, dtype=np.int32)
        # Completed episodes ring buffer (FIFO) for logging.
        self.episode_returns: Deque[float] = deque(maxlen=episode_buffer_size)
        self.episode_lengths: Deque[int]   = deque(maxlen=episode_buffer_size)
        self.episode_caught:  Deque[int]   = deque(maxlen=episode_buffer_size)

    # ------------------------------------------------------------------ #
    #  Red-policy mixing                                                   #
    # ------------------------------------------------------------------ #

    def _resample_red_policy(self, env: PursuitEnv) -> None:
        """
        If ``red_policy_mix`` is configured, sample a fresh red policy
        for this env according to the categorical weights.  Called
        before every env reset (initial + auto-reset on episode end).

        No-op when mixing is disabled — the env keeps whatever policy
        it was constructed with.
        """
        if self._red_policy_names is None or self._red_policy_probs is None:
            return
        idx = int(self._mix_rng.choice(
            len(self._red_policy_names), p=self._red_policy_probs,
        ))
        name = self._red_policy_names[idx]
        if name == "stationary":
            env.red_policy = stationary_red
        elif name == "random":
            # Each sampled random_red gets its own seeded closure so
            # different episodes see different random red trajectories.
            env.red_policy = random_red(
                seed=int(self._mix_rng.integers(0, 2**31 - 1)),
            )
        elif name == "run":
            env.red_policy = run_from_nearest_uav
        else:
            raise ValueError(f"Unknown red policy name in mix: {name!r}")

    # ------------------------------------------------------------------ #
    #  Reset / step                                                        #
    # ------------------------------------------------------------------ #

    def _empty_structured_batch(self) -> Dict[str, np.ndarray]:
        """Pre-allocate an empty batched dict for structured obs mode."""
        return {
            "blue_features":     np.zeros((self.n_envs, self.n_blue,
                                           self.blue_feat_dim),
                                          dtype=np.float32),
            "red_features":      np.zeros((self.n_envs, self.n_red,
                                           self.red_feat_dim),
                                          dtype=np.float32),
            "bb_edge_features":  np.zeros((self.n_envs, self.n_bb_edges,
                                           self.edge_feat_dim),
                                          dtype=np.float32),
            "rb_edge_features":  np.zeros((self.n_envs, self.n_rb_edges,
                                           self.edge_feat_dim),
                                          dtype=np.float32),
        }

    def _fill_structured_row(
        self,
        batch: Dict[str, np.ndarray],
        idx:   int,
        env:   PursuitEnv,
    ) -> None:
        s = env.structured_observation()
        for k, arr in s.items():
            batch[k][idx] = arr

    def reset(self, seed: Optional[int] = None):
        """
        Reset every env.  Return format depends on ``self.obs_format``:

        - "flat" (default): obs is np.ndarray shaped
          ``(n_envs, n_agents, obs_dim)``.
        - "structured": obs is a dict of arrays, each shaped
          ``(n_envs, n_tokens, feat_dim)`` for the GNN policy.

        If ``seed`` is given, re-seeds with base ``seed`` and per-env
        offsets — used at the very start of training for full
        determinism.
        """
        if self.obs_format == "flat":
            obs_batch = np.zeros(
                (self.n_envs, self.n_agents, self.obs_dim), dtype=np.float32
            )
        else:
            obs_batch = self._empty_structured_batch()

        for i, env in enumerate(self.envs):
            self._resample_red_policy(env)
            s = None if seed is None else int(seed + i)
            obs_dict, _ = env.reset(seed=s)
            if self.obs_format == "flat":
                for j, name in enumerate(self.possible_agents):
                    obs_batch[i, j] = obs_dict[name]
            else:
                self._fill_structured_row(obs_batch, i, env)

        self._ep_return[:] = 0.0
        self._ep_length[:] = 0
        return obs_batch

    def step(self, actions: np.ndarray):
        """
        Step every env.  Return format depends on ``self.obs_format``:

        - "flat":
            obs:     np.ndarray (n_envs, n_agents, obs_dim)
            rewards: np.ndarray (n_envs, n_agents) — per-agent = team r
            dones:   np.ndarray (n_envs,)
            infos:   list[dict]
        - "structured":
            obs:     dict of arrays for the GNN policy
            rewards: np.ndarray (n_envs,) — team reward per env
            dones:   np.ndarray (n_envs,)
            infos:   list[dict]
        """
        assert actions.shape == (self.n_envs, self.n_agents, self.action_dim), \
            f"actions shape {actions.shape} != " \
            f"{(self.n_envs, self.n_agents, self.action_dim)}"

        if self.obs_format == "flat":
            obs_batch = np.zeros(
                (self.n_envs, self.n_agents, self.obs_dim), dtype=np.float32
            )
            reward_batch = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)
        else:
            obs_batch = self._empty_structured_batch()
            reward_batch = np.zeros(self.n_envs, dtype=np.float32)

        done_batch: np.ndarray = np.zeros(self.n_envs, dtype=np.float32)
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
                snap = env.state_snapshot()
                n_caught = int((~snap["red_active"]).sum())
                self.episode_returns.append(float(self._ep_return[i]))
                self.episode_lengths.append(int(self._ep_length[i]))
                self.episode_caught.append(n_caught)
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0

                # Auto-reset with fresh red policy.
                self._resample_red_policy(env)
                obs_d, _ = env.reset()

            # Reward: same shared value from any agent.
            if self.obs_format == "flat":
                for j, name in enumerate(self.possible_agents):
                    obs_batch[i, j]    = obs_d[name]
                    reward_batch[i, j] = float(rew_d[name])
            else:
                self._fill_structured_row(obs_batch, i, env)
                reward_batch[i] = float(rew_d[first])

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
