"""
isr/train/subproc_vec_env.py — multiprocess wrapper around
``Stage4VectorPursuitEnv``.

Motivation (profiled 2026-07): env stepping is pure-Python/numpy and
runs on ONE core while the other 7 idle.  This wrapper shards the env
pool across worker processes: each worker hosts a contiguous shard of
envs inside a regular in-process ``Stage4VectorPursuitEnv``, so every
env semantic (auto-reset, red-policy resampling, per-agent rewards,
crash statistics) is reused verbatim — the parent only scatters
actions and gathers/concatenates results.

The parent exposes the SAME contact surface the trainer uses on the
in-process class:

    reset(seed) / step(actions) / recent_episode_stats()
    belief_track_errors() / close()
    n_agents, action_dim, n_blue, n_red, n_obstacles,
    blue_feat_dim, red_feat_dim, edge_feat_dim

so ``--n-workers`` is a pure drop-in in ``train_stage4.py``.

Windows notes: uses the ``spawn`` start context (the only one on
Windows).  Workers import numpy + the env stack only — torch is NOT
imported in workers, keeping them light and avoiding thread-pool
oversubscription against the parent's torch threads.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker(
    remote,
    n_envs:              int,
    env_kwargs:          Dict[str, Any],
    base_seed:           int,
    episode_buffer_size: int,
    red_policy_mix:      Optional[Sequence[Tuple[str, float]]],
) -> None:
    """Host one env shard; serve commands over the pipe until 'close'.

    Every reply is a ``(status, payload)`` tuple: ``("ok", result)`` or
    ``("error", traceback_str)`` so parent-side failures carry the real
    worker stack trace instead of a bare EOFError.
    """
    # Import inside the worker: with the 'spawn' context this runs in a
    # fresh interpreter, and we deliberately keep torch out of it.
    from isr.train.vec_env import Stage4VectorPursuitEnv

    try:
        ve = Stage4VectorPursuitEnv(
            n_envs              = n_envs,
            env_kwargs          = env_kwargs,
            base_seed           = base_seed,
            episode_buffer_size = episode_buffer_size,
            red_policy_mix      = red_policy_mix,
        )
    except Exception:
        remote.send(("error", traceback.format_exc()))
        remote.close()
        return

    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if cmd == "reset":
                remote.send(("ok", ve.reset(seed=data)))
            elif cmd == "step":
                remote.send(("ok", ve.step(data)))
            elif cmd == "meta":
                remote.send(("ok", {
                    "n_agents":      ve.n_agents,
                    "action_dim":    ve.action_dim,
                    "n_blue":        ve.n_blue,
                    "n_red":         ve.n_red,
                    "n_obstacles":   ve.n_obstacles,
                    "blue_feat_dim": ve.blue_feat_dim,
                    "red_feat_dim":  ve.red_feat_dim,
                    "edge_feat_dim": ve.edge_feat_dim,
                }))
            elif cmd == "stats_raw":
                remote.send(("ok", {
                    "returns":      list(ve.episode_returns),
                    "lengths":      list(ve.episode_lengths),
                    "caught":       list(ve.episode_caught),
                    "obs_crashes":  list(ve.episode_obs_crashes),
                    "blue_crashes": list(ve.episode_blue_crashes),
                }))
            elif cmd == "track_err":
                remote.send(("ok",
                             [e.belief_track_error() for e in ve.envs]))
            elif cmd == "close":
                remote.send(("ok", None))
                break
            else:
                remote.send(("error", f"unknown command {cmd!r}"))
        except Exception:
            remote.send(("error", traceback.format_exc()))
    remote.close()


# ---------------------------------------------------------------------------
# Parent-side wrapper
# ---------------------------------------------------------------------------

class SubprocStage4VecEnv:
    """
    Drop-in multiprocess replacement for ``Stage4VectorPursuitEnv``.

    Envs are split into ``n_workers`` contiguous shards.  Seeding is
    parity-preserving: worker ``w`` (starting at global env index
    ``s_w``) receives ``reset(seed + s_w)``, and the in-worker vec env
    seeds its env ``j`` with ``(seed + s_w) + j`` — i.e. global env
    ``i`` gets ``seed + i`` exactly as in the single-process class.
    """

    def __init__(
        self,
        n_envs:              int,
        env_kwargs:          Dict[str, Any],
        base_seed:           int = 0,
        episode_buffer_size: int = 128,
        red_policy_mix:      Optional[Sequence[Tuple[str, float]]] = None,
        n_workers:           int = 4,
    ) -> None:
        n_workers = max(1, min(int(n_workers), int(n_envs)))
        self.n_envs    = int(n_envs)
        self.n_workers = n_workers

        # Contiguous shards, sizes differing by at most 1.
        base, extra = divmod(self.n_envs, n_workers)
        sizes  = [base + (1 if w < extra else 0) for w in range(n_workers)]
        starts = list(np.cumsum([0] + sizes[:-1]))
        self._shard_sizes  = sizes
        self._shard_starts = starts

        ctx = mp.get_context("spawn")
        self._remotes: List[Any] = []
        self._procs:   List[Any] = []
        for w in range(n_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_worker,
                args=(child_conn, sizes[w], dict(env_kwargs),
                      base_seed + starts[w], episode_buffer_size,
                      red_policy_mix),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._remotes.append(parent_conn)
            self._procs.append(proc)

        meta = self._ask(0, "meta", None)
        self.n_agents      = meta["n_agents"]
        self.action_dim    = meta["action_dim"]
        self.n_blue        = meta["n_blue"]
        self.n_red         = meta["n_red"]
        self.n_obstacles   = meta["n_obstacles"]
        self.blue_feat_dim = meta["blue_feat_dim"]
        self.red_feat_dim  = meta["red_feat_dim"]
        self.edge_feat_dim = meta["edge_feat_dim"]

        self._closed = False

    # -- plumbing -------------------------------------------------------- #

    def _ask(self, w: int, cmd: str, data) -> Any:
        self._remotes[w].send((cmd, data))
        return self._recv(w)

    def _recv(self, w: int) -> Any:
        status, payload = self._remotes[w].recv()
        if status != "ok":
            raise RuntimeError(
                f"SubprocStage4VecEnv worker {w} failed:\n{payload}")
        return payload

    def _broadcast_gather(self, cmd: str, per_worker_data) -> List[Any]:
        """Send to ALL workers first, then gather — the send/recv split
        is what lets the shards actually run in parallel."""
        for w, remote in enumerate(self._remotes):
            remote.send((cmd, per_worker_data[w]))
        return [self._recv(w) for w in range(self.n_workers)]

    # -- API ------------------------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        seeds = [
            None if seed is None else int(seed + self._shard_starts[w])
            for w in range(self.n_workers)
        ]
        shards = self._broadcast_gather("reset", seeds)
        return {
            k: np.concatenate([s[k] for s in shards], axis=0)
            for k in shards[0]
        }

    def step(
        self, actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        assert actions.shape[0] == self.n_envs, actions.shape
        chunks = [
            actions[s:s + n]
            for s, n in zip(self._shard_starts, self._shard_sizes)
        ]
        results = self._broadcast_gather("step", chunks)

        obs = {
            k: np.concatenate([r[0][k] for r in results], axis=0)
            for k in results[0][0]
        }
        rewards = np.concatenate([r[1] for r in results], axis=0)
        dones   = np.concatenate([r[2] for r in results], axis=0)
        infos: List[Dict] = []
        for r in results:
            infos.extend(r[3])
        return obs, rewards, dones, infos

    def recent_episode_stats(self) -> Dict[str, float]:
        """Aggregate the per-worker episode windows; same keys/semantics
        as ``VectorPursuitEnv.recent_episode_stats``."""
        raws = self._broadcast_gather(
            "stats_raw", [None] * self.n_workers)
        returns      = [x for r in raws for x in r["returns"]]
        lengths      = [x for r in raws for x in r["lengths"]]
        caught       = [x for r in raws for x in r["caught"]]
        obs_crashes  = [x for r in raws for x in r["obs_crashes"]]
        blue_crashes = [x for r in raws for x in r["blue_crashes"]]
        if not returns:
            return {
                "n_completed": 0, "mean_return": 0.0, "std_return": 0.0,
                "mean_length": 0.0, "mean_caught": 0.0,
                "mean_obstacle_crashes": 0.0, "mean_blue_crashes": 0.0,
            }
        rs = np.array(returns, dtype=np.float32)
        ocs = (np.array(obs_crashes, dtype=np.float32)
               if obs_crashes else np.zeros(1, dtype=np.float32))
        bcs = (np.array(blue_crashes, dtype=np.float32)
               if blue_crashes else np.zeros(1, dtype=np.float32))
        return {
            "n_completed": int(len(rs)),
            "mean_return": float(rs.mean()),
            "std_return":  float(rs.std()),
            "mean_length": float(np.mean(lengths)),
            "mean_caught": float(np.mean(caught)),
            "mean_obstacle_crashes": float(ocs.mean()),
            "mean_blue_crashes":     float(bcs.mean()),
        }

    def belief_track_errors(self) -> List[float]:
        raws = self._broadcast_gather("track_err", [None] * self.n_workers)
        return [x for r in raws for x in r]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for w, remote in enumerate(self._remotes):
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        for remote in self._remotes:
            remote.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
