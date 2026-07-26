"""
tests/test_subproc_vec_env.py — contract tests for the multiprocess
env wrapper (SubprocStage4VecEnv).

The wrapper must be a drop-in for Stage4VectorPursuitEnv: same obs
schema, same (E, A) per-agent rewards, same stats keys — and, with a
deterministic red-policy mix, bitwise-identical trajectories (the
seeding is parity-preserving: global env i gets seed + i in both).

Marked module-level with a longer timeout tolerance in mind: each test
spawns real subprocesses (Windows 'spawn' start method).
"""
from __future__ import annotations

import numpy as np

from isr.train.vec_env import Stage4VectorPursuitEnv
from isr.train.subproc_vec_env import SubprocStage4VecEnv


EK = dict(
    n_blue=3, n_red=2, n_obstacles=4, sensor_radius=40.0,
    use_belief_maps=True, max_steps=25, arena_size=80.0,
)
# Single-entry mix -> resampling always lands on the same policy, so
# in-proc and subproc trajectories are comparable step for step.
MIX = [("stationary", 1.0)]


def test_subproc_matches_inproc_bitwise():
    n_envs = 4
    ref = Stage4VectorPursuitEnv(n_envs=n_envs, env_kwargs=EK,
                                 base_seed=0, red_policy_mix=MIX)
    sub = SubprocStage4VecEnv(n_envs=n_envs, env_kwargs=EK,
                              base_seed=0, red_policy_mix=MIX,
                              n_workers=2)
    try:
        obs_r = ref.reset(seed=123)
        obs_s = sub.reset(seed=123)
        assert set(obs_r) == set(obs_s)
        for k in obs_r:
            np.testing.assert_array_equal(obs_r[k], obs_s[k], err_msg=k)

        rng = np.random.default_rng(7)
        for t in range(30):   # crosses the max_steps=25 auto-reset
            acts = rng.uniform(-1, 1, size=(n_envs, 3, 2)).astype(np.float32)
            obs_r, rew_r, done_r, _ = ref.step(acts)
            obs_s, rew_s, done_s, _ = sub.step(acts)
            np.testing.assert_array_equal(rew_r, rew_s, err_msg=f"t={t}")
            np.testing.assert_array_equal(done_r, done_s, err_msg=f"t={t}")
            for k in obs_r:
                np.testing.assert_array_equal(obs_r[k], obs_s[k],
                                              err_msg=f"{k} t={t}")

        sr, ss = ref.recent_episode_stats(), sub.recent_episode_stats()
        assert set(sr) == set(ss)
        assert sr["n_completed"] == ss["n_completed"] > 0
        assert abs(sr["mean_return"] - ss["mean_return"]) < 1e-5
    finally:
        sub.close()


def test_subproc_meta_and_shapes():
    sub = SubprocStage4VecEnv(n_envs=5, env_kwargs=EK, base_seed=0,
                              red_policy_mix=MIX, n_workers=3)
    try:
        ref = Stage4VectorPursuitEnv(n_envs=1, env_kwargs=EK,
                                     base_seed=0, red_policy_mix=MIX)
        assert sub.n_agents == ref.n_agents
        assert sub.action_dim == ref.action_dim
        assert (sub.n_blue, sub.n_red, sub.n_obstacles) == \
               (ref.n_blue, ref.n_red, ref.n_obstacles)
        assert sub.blue_feat_dim == ref.blue_feat_dim
        assert sub.edge_feat_dim == ref.edge_feat_dim

        obs = sub.reset(seed=0)
        for k, v in obs.items():
            assert v.shape[0] == 5, (k, v.shape)

        acts = np.zeros((5, 3, 2), dtype=np.float32)
        obs, rew, done, infos = sub.step(acts)
        assert rew.shape == (5, 3)          # per-agent (E, A)
        assert done.shape == (5,)
        assert len(infos) == 5

        errs = sub.belief_track_errors()
        assert len(errs) == 5
    finally:
        sub.close()


def test_subproc_uneven_shards_and_close_idempotent():
    # 5 envs across 3 workers -> shards 2/2/1.
    sub = SubprocStage4VecEnv(n_envs=5, env_kwargs=EK, base_seed=0,
                              red_policy_mix=MIX, n_workers=3)
    assert sub._shard_sizes == [2, 2, 1]
    assert sub._shard_starts == [0, 2, 4]
    sub.reset(seed=0)
    sub.close()
    sub.close()   # second close must be a no-op, not an error
