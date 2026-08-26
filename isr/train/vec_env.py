"""
isr/train/vec_env.py — Hand-rolled vector wrapper around PursuitEnv.

Runs N independent copies of ``PursuitEnv`` and exposes a batched
``step / reset`` interface that returns the structured (graph)
observation used by the GNN policy.  Auto-resets envs that finish an
episode so the rollout collector can pull continuous transitions
without having to handle done envs specially.

Design notes
------------
- SuperSuit / gym-vector-env wrappers deliberately avoided to keep
  the code transparent and traceable end-to-end.
- Auto-reset convention: when an env terminates / truncates at step t,
  we (a) record the done flag in the returned ``dones`` array so the
  trainer's GAE bootstrap is correct, and (b) immediately reset the
  env so the obs returned alongside the done is the FIRST obs of the
  next episode.  Standard gym-vector convention.
- All envs share kinematic config but get distinct seeds (base_seed,
  base_seed+1, ..., base_seed+n_envs-1) for diversity.
- Episode return / length / catches are tracked in a ring buffer for
  logging.

The flat-obs mode (used by the MLP path) was removed in the
July-2026 cleanup — see docs/stage2_results.md.  Only the structured
(graph) obs is exposed now.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import (
    stationary_red, random_red, run_from_nearest_uav,
)


class Stage4VectorPursuitEnv:
    """
    Vectorised ``PursuitEnv`` with per-episode red-policy mixing, emitting
    the v6 typed-graph obs from ``structured_belief_observation()``:
    belief-derived actor node/edge features plus ground-truth ``true_*``
    critic features (and obstacle variants when configured).

    The big ``belief_maps`` / ``true_occupancy`` diagnostic tensors are NOT
    batched -- the policy consumes only the reconstructed graph, and storing
    full belief grids per step would blow up memory.

    Requires ``env_kwargs`` to set ``sensor_radius`` (the sensor step needs a
    defined disk) and ``use_belief_maps=True`` (the log-odds tensor feeds the
    peak extractor).

    Was a ``Stage4VectorPursuitEnv(VectorPursuitEnv)`` pair until the 2026-08
    cleanup; the fully-observable base had no remaining users once
    train_stage1.py was removed, so the two were collapsed into one class.
    """

    #: obs keys that stay inside the env (not fed to the policy/buffer).
    _EXCLUDE = ("belief_maps", "true_occupancy")

    def __init__(
        self,
        n_envs:              int,
        env_kwargs:          Dict[str, Any],
        base_seed:           int = 0,
        episode_buffer_size: int = 128,
        red_policy_mix:      Optional[Sequence[Tuple[str, float]]] = None,
    ) -> None:
        """
        red_policy_mix: optional list of (name, weight) tuples specifying
            a per-episode categorical distribution over red policies.
            Names supported: 'stationary', 'random', 'run'.  Weights
            are normalised.  When None (default), every env uses
            whatever red policy was passed via ``env_kwargs['red_policy']``
            (or the env default = run_from_nearest_uav) for every
            episode.

            The trainer default (``--red-policy-mix`` in
            scripts/train_stage4.py) is
            ``[('stationary', 1), ('random', 1), ('run', 1)]`` — uniform
            mix — to eliminate the OOD failure documented in
            docs/stage1_analysis.md §3 (Failure mode 1).
        """
        if env_kwargs.get("sensor_radius", None) is None:
            raise ValueError(
                "Stage4VectorPursuitEnv requires env_kwargs['sensor_radius'] "
                "to be set (belief update needs a defined sensor disk)."
            )
        if not env_kwargs.get("use_belief_maps", False):
            raise ValueError(
                "Stage4VectorPursuitEnv requires "
                "env_kwargs['use_belief_maps']=True."
            )
        self.n_envs = int(n_envs)

        # Distinct seeds so rollouts cover varied initial states.
        self.envs: List[PursuitEnv] = []
        for i in range(self.n_envs):
            kwargs = dict(env_kwargs)
            kwargs["seed"] = int(base_seed + i)
            self.envs.append(PursuitEnv(**kwargs))

        # Red-policy mixing setup — see _resample_red_policy.  Dedicated
        # RNG so mixing decisions don't consume per-env RNG draws.
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

        e0 = self.envs[0]
        self.n_agents   = e0.n_blue
        self.action_dim = 2
        self.possible_agents = list(e0.possible_agents)
        self.arena_size = e0.arena_size
        self.max_steps  = e0.max_steps
        # Graph-obs sizing, read by the trainer when it builds the policy.
        self.blue_feat_dim = e0.blue_feat_dim
        self.red_feat_dim  = e0.red_feat_dim
        self.edge_feat_dim = e0.edge_feat_dim
        self.n_bb_edges    = e0.n_bb_edges
        self.n_rb_edges    = e0.n_rb_edges
        self.n_blue        = e0.n_blue
        self.n_red         = e0.n_red
        self.n_obstacles   = e0.n_obstacles

        # Per-env episode-tracking state.
        self._ep_return = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_length = np.zeros(self.n_envs, dtype=np.int32)
        # Per-episode crash accumulators (summed over the episode).
        self._ep_obs_crashes  = np.zeros(self.n_envs, dtype=np.int32)
        self._ep_blue_crashes = np.zeros(self.n_envs, dtype=np.int32)
        # Completed episodes ring buffer.
        self.episode_returns: Deque[float] = deque(maxlen=episode_buffer_size)
        self.episode_lengths: Deque[int]   = deque(maxlen=episode_buffer_size)
        self.episode_caught:  Deque[int]   = deque(maxlen=episode_buffer_size)
        self.episode_obs_crashes:  Deque[int] = deque(maxlen=episode_buffer_size)
        self.episode_blue_crashes: Deque[int] = deque(maxlen=episode_buffer_size)

        # Discover the graph-obs schema from one sample observation so the
        # obstacle keys appear only when obstacles are configured.
        e0.reset(seed=base_seed)
        sample = e0.structured_belief_observation()
        self._graph_keys = [k for k in sample if k not in self._EXCLUDE]
        self._graph_shapes = {k: tuple(sample[k].shape)
                              for k in self._graph_keys}

    # ------------------------------------------------------------------ #
    #  Red-policy mixing                                                   #
    # ------------------------------------------------------------------ #

    def _resample_red_policy(self, env: PursuitEnv) -> None:
        """Sample a fresh red policy per configured mix.  No-op if disabled."""
        if self._red_policy_names is None or self._red_policy_probs is None:
            return
        idx = int(self._mix_rng.choice(
            len(self._red_policy_names), p=self._red_policy_probs,
        ))
        name = self._red_policy_names[idx]
        if name == "stationary":
            env.red_policy = stationary_red
        elif name == "random":
            env.red_policy = random_red(
                seed=int(self._mix_rng.integers(0, 2**31 - 1)),
            )
        elif name == "run":
            env.red_policy = run_from_nearest_uav
        else:
            raise ValueError(f"Unknown red policy name in mix: {name!r}")

    # ------------------------------------------------------------------ #
    #  Structured-obs helpers                                              #
    # ------------------------------------------------------------------ #

    def _empty_batch(self) -> Dict[str, np.ndarray]:
        return {
            k: np.zeros((self.n_envs,) + shp, dtype=np.float32)
            for k, shp in self._graph_shapes.items()
        }

    def _fill_row(
        self, batch: Dict[str, np.ndarray], idx: int, env: PursuitEnv,
    ) -> None:
        obs = env.structured_belief_observation()
        for k in self._graph_keys:
            batch[k][idx] = obs[k]

    # ------------------------------------------------------------------ #
    #  Reset / step                                                        #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset every env.  Returns the structured (graph) batched obs
        dict: each key mapped to ``(n_envs, n_tokens, feat_dim)``.

        If ``seed`` is given, re-seeds with base ``seed`` and per-env
        offsets — used at the very start of training for full
        determinism.
        """
        obs_batch = self._empty_batch()
        for i, env in enumerate(self.envs):
            self._resample_red_policy(env)
            s = None if seed is None else int(seed + i)
            env.reset(seed=s)
            self._fill_row(obs_batch, i, env)
        self._ep_return[:] = 0.0
        self._ep_length[:] = 0
        return obs_batch

    def step(
        self, actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        """
        Step every env.  Returns:
          obs:     dict of arrays for the GNN policy
          rewards: (n_envs, n_agents) — PER-AGENT reward r_i.  With crash
                   penalties off, every agent's reward equals the shared
                   team reward (so a shared-reward consumer can just take
                   ``rewards[:, 0]``).
          dones:   (n_envs,)
          infos:   list[dict]
        """
        assert actions.shape == (self.n_envs, self.n_agents, self.action_dim), \
            f"actions shape {actions.shape} != " \
            f"{(self.n_envs, self.n_agents, self.action_dim)}"

        A = self.n_agents
        obs_batch = self._empty_batch()
        reward_batch = np.zeros((self.n_envs, A), dtype=np.float32)
        done_batch   = np.zeros(self.n_envs, dtype=np.float32)
        infos: List[Dict] = []

        for i, env in enumerate(self.envs):
            action_dict = {
                name: actions[i, j]
                for j, name in enumerate(self.possible_agents)
            }
            _, rew_d, term_d, trunc_d, info_d = env.step(action_dict)

            first = self.possible_agents[0]
            done = bool(term_d[first] or trunc_d[first])
            r_vec = np.array([rew_d[name] for name in self.possible_agents],
                             dtype=np.float32)                  # (A,)

            # Logging return = mean per-agent return (== team when shared).
            self._ep_return[i] += float(r_vec.mean())
            self._ep_length[i] += 1
            self._ep_obs_crashes[i]  += int(info_d[first]["obstacle_crashes"])
            self._ep_blue_crashes[i] += int(info_d[first]["blue_crashes"])

            if done:
                snap = env.state_snapshot()
                # caught = active-at-start - active-now (padded reds under
                # variable-count never counted; snap default = full n_red).
                n_caught = int(snap.get("n_red_start", len(snap["red_active"]))
                               - int(snap["red_active"].sum()))
                self.episode_returns.append(float(self._ep_return[i]))
                self.episode_lengths.append(int(self._ep_length[i]))
                self.episode_caught.append(n_caught)
                self.episode_obs_crashes.append(int(self._ep_obs_crashes[i]))
                self.episode_blue_crashes.append(int(self._ep_blue_crashes[i]))
                self._ep_return[i] = 0.0
                self._ep_length[i] = 0
                self._ep_obs_crashes[i]  = 0
                self._ep_blue_crashes[i] = 0

                # Auto-reset with fresh red policy.
                self._resample_red_policy(env)
                env.reset()

            self._fill_row(obs_batch, i, env)
            reward_batch[i] = r_vec
            done_batch[i]   = 1.0 if done else 0.0
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
                "mean_obstacle_crashes": 0.0, "mean_blue_crashes": 0.0,
            }
        rs = np.array(self.episode_returns, dtype=np.float32)
        ls = np.array(self.episode_lengths, dtype=np.float32)
        cs = np.array(self.episode_caught,  dtype=np.float32)
        ocs = (np.array(self.episode_obs_crashes, dtype=np.float32)
               if self.episode_obs_crashes else np.zeros(1, dtype=np.float32))
        bcs = (np.array(self.episode_blue_crashes, dtype=np.float32)
               if self.episode_blue_crashes else np.zeros(1, dtype=np.float32))
        return {
            "n_completed": int(len(rs)),
            "mean_return": float(rs.mean()),
            "std_return":  float(rs.std()),
            "mean_length": float(ls.mean()),
            "mean_caught": float(cs.mean()),
            "mean_obstacle_crashes": float(ocs.mean()),
            "mean_blue_crashes":     float(bcs.mean()),
        }

    def belief_track_errors(self) -> List[float]:
        """Per-env belief peak-to-true-red distances (m).  Method (not a
        direct ``.envs`` walk) so the subprocess wrapper can serve the
        same diagnostic remotely."""
        return [env.belief_track_error() for env in self.envs]

    def close(self) -> None:
        for env in self.envs:
            env.close()
