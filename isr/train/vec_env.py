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


class VectorPursuitEnv:
    """
    Vectorised ``PursuitEnv`` with per-episode red-policy mixing and
    structured (graph) obs.  See module docstring.
    """

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

            v2 default in scripts/train_stage1.py is
            ``[('stationary', 1), ('random', 1), ('run', 1)]`` — uniform
            mix — to eliminate the OOD failure documented in
            docs/stage1_analysis.md §3 (Failure mode 1).
        """
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
        # Graph-obs sizing, exposed for GraphRolloutBuffer construction.
        self.blue_feat_dim = e0.blue_feat_dim
        self.red_feat_dim  = e0.red_feat_dim
        self.edge_feat_dim = e0.edge_feat_dim
        self.n_bb_edges    = e0.n_bb_edges
        self.n_rb_edges    = e0.n_rb_edges
        self.n_blue        = e0.n_blue
        self.n_red         = e0.n_red

        # Per-env episode-tracking state.
        self._ep_return = np.zeros(self.n_envs, dtype=np.float32)
        self._ep_length = np.zeros(self.n_envs, dtype=np.int32)
        # Completed episodes ring buffer.
        self.episode_returns: Deque[float] = deque(maxlen=episode_buffer_size)
        self.episode_lengths: Deque[int]   = deque(maxlen=episode_buffer_size)
        self.episode_caught:  Deque[int]   = deque(maxlen=episode_buffer_size)

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

    def _fill_row(
        self, batch: Dict[str, np.ndarray], idx: int, env: PursuitEnv,
    ) -> None:
        for k, arr in env.structured_observation().items():
            batch[k][idx] = arr

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
          rewards: (n_envs,) — team reward per env
          dones:   (n_envs,)
          infos:   list[dict]
        """
        assert actions.shape == (self.n_envs, self.n_agents, self.action_dim), \
            f"actions shape {actions.shape} != " \
            f"{(self.n_envs, self.n_agents, self.action_dim)}"

        obs_batch = self._empty_batch()
        reward_batch = np.zeros(self.n_envs, dtype=np.float32)
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
                env.reset()

            self._fill_row(obs_batch, i, env)
            reward_batch[i] = float(rew_d[first])
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


# ---------------------------------------------------------------------------
# Stage 3: partial-obs variant
# ---------------------------------------------------------------------------

class RecurrentVectorPursuitEnv(VectorPursuitEnv):
    """
    Stage 3 vectorised env.

    Same reset/step signatures as ``VectorPursuitEnv`` but the obs dict
    now includes two extra keys populated from
    ``PursuitEnv.structured_partial_observation()``:

    - ``bb_edge_visible`` : (n_envs, n_bb_edges) float32
    - ``rb_edge_visible`` : (n_envs, n_rb_edges) float32

    The four base feature tensors are identical between partial-obs
    and full-state views (design §2.2), so the trainer builds the
    full-state view for the CTDE critic by dropping the two mask
    keys — no extra work in the vec env.

    Hidden state is *not* tracked here: it belongs to the trainer
    (see stage3_design.md §7 stateless-at-update convention).  The
    ``dones`` array returned by ``step`` is enough for the trainer
    to know which env's hidden state to zero.
    """

    def __init__(
        self,
        n_envs:              int,
        env_kwargs:          Dict[str, Any],
        base_seed:           int = 0,
        episode_buffer_size: int = 128,
        red_policy_mix:      Optional[Sequence[Tuple[str, float]]] = None,
    ) -> None:
        # Enforce that env_kwargs actually sets a sensor_radius — this
        # class exists specifically to test partial observability.
        if env_kwargs.get("sensor_radius", None) is None:
            raise ValueError(
                "RecurrentVectorPursuitEnv requires env_kwargs['sensor_radius'] "
                "to be set (Stage 3 uses partial observability).  If you want "
                "the fully-observable env, use VectorPursuitEnv."
            )
        super().__init__(
            n_envs               = n_envs,
            env_kwargs           = env_kwargs,
            base_seed            = base_seed,
            episode_buffer_size  = episode_buffer_size,
            red_policy_mix       = red_policy_mix,
        )

    # ------------------------------------------------------------------ #
    #  Structured-obs override                                             #
    # ------------------------------------------------------------------ #

    def _empty_batch(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        batch = super()._empty_batch()
        batch["bb_edge_visible"] = np.zeros(
            (self.n_envs, self.n_bb_edges), dtype=np.float32,
        )
        batch["rb_edge_visible"] = np.zeros(
            (self.n_envs, self.n_rb_edges), dtype=np.float32,
        )
        return batch

    def _fill_row(  # type: ignore[override]
        self, batch: Dict[str, np.ndarray], idx: int, env: PursuitEnv,
    ) -> None:
        for k, arr in env.structured_partial_observation().items():
            batch[k][idx] = arr


# ---------------------------------------------------------------------------
# Stage 4: belief-map variant
# ---------------------------------------------------------------------------

class Stage4VectorPursuitEnv(VectorPursuitEnv):
    """
    Stage 4 vectorised env.  Same reset/step signatures as the Stage 3
    variant, but the obs dict is the Stage 4 schema
    (``structured_belief_observation()``): drops the red-side and
    visibility keys, adds ``belief_maps``, ``obstacle_positions``,
    ``true_occupancy``.

    Requires ``env_kwargs`` to have both ``sensor_radius`` set (so the
    sensor step has a defined disk) and ``use_belief_maps=True`` (so
    the log-odds tensor is actually maintained).
    """

    def __init__(
        self,
        n_envs:              int,
        env_kwargs:          Dict[str, Any],
        base_seed:           int = 0,
        episode_buffer_size: int = 128,
        red_policy_mix:      Optional[Sequence[Tuple[str, float]]] = None,
    ) -> None:
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
        super().__init__(
            n_envs               = n_envs,
            env_kwargs           = env_kwargs,
            base_seed            = base_seed,
            episode_buffer_size  = episode_buffer_size,
            red_policy_mix       = red_policy_mix,
        )
        # Pull Stage 4 sizing from the first env for convenience.
        e0 = self.envs[0]
        self.belief_channels    = e0.belief_channels
        self.belief_grid_size   = e0.belief_grid_size
        self.belief_window_size = getattr(e0, "belief_window_size", 0)
        self.n_obstacles        = e0.n_obstacles
        # Reserve batch space for the requested obstacle count.  A
        # rejection-sampled reset may drop below this if placement
        # fails; ``_fill_row`` pads with zeros in that case.
        self._obs_n_obstacles = int(e0.n_obstacles)

    # ------------------------------------------------------------------ #
    #  Structured-obs override                                             #
    # ------------------------------------------------------------------ #

    def _empty_batch(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        H = W = self.belief_grid_size
        C   = self.belief_channels
        nob = self._obs_n_obstacles
        K   = self.belief_window_size
        batch = {
            "blue_features":    np.zeros(
                (self.n_envs, self.n_blue, self.blue_feat_dim),
                dtype=np.float32),
            "bb_edge_features": np.zeros(
                (self.n_envs, self.n_bb_edges, self.edge_feat_dim),
                dtype=np.float32),
            "belief_maps":      np.zeros(
                (self.n_envs, self.n_blue, C, H, W),
                dtype=np.float32),
            "obstacle_positions": np.zeros(
                (self.n_envs, nob, 3), dtype=np.float32),
            # true_occupancy always 2 channels: enemy + obstacle.
            "true_occupancy":   np.zeros(
                (self.n_envs, 2, H, W), dtype=np.float32),
        }
        # Ego-centric windows (Stage 4 v3).  Allocated only when the
        # underlying env exposes them.
        if K > 0:
            batch["belief_windows"] = np.zeros(
                (self.n_envs, self.n_blue, C, K, K), dtype=np.float32,
            )
        return batch

    def _fill_row(  # type: ignore[override]
        self, batch: Dict[str, np.ndarray], idx: int, env: PursuitEnv,
    ) -> None:
        obs = env.structured_belief_observation()
        for k in ("blue_features", "bb_edge_features",
                  "belief_maps", "true_occupancy"):
            batch[k][idx] = obs[k]
        if "belief_windows" in batch and "belief_windows" in obs:
            batch["belief_windows"][idx] = obs["belief_windows"]
        # obstacle_positions might have differed per-env at reset (if
        # rejection sampling gave fewer than requested).  Pad or clip to
        # the batch's declared size.
        op = obs["obstacle_positions"]
        nob_batch = batch["obstacle_positions"].shape[1]
        if op.shape[0] >= nob_batch:
            batch["obstacle_positions"][idx] = op[:nob_batch]
        else:
            batch["obstacle_positions"][idx, :op.shape[0]] = op
