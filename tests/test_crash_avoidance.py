"""
tests/test_crash_avoidance.py — per-agent crash-penalty + per-agent
value/GAE contract tests (Stage 4 crash-avoidance extension).

Covers the crash-avoidance design:
  * r_i = r_team + r_crash_i  (crash penalty is INDIVIDUAL, catch/step
    reward stays SHARED)
  * crashes DON'T terminate the episode (soft-stop model)
  * the critic estimates a per-agent V(s, i)
  * GAE / advantages / returns are per-agent
  * a converged Stage 4 checkpoint warm-starts BOTH actor and critic

Run:
    pytest tests/test_crash_avoidance.py -v
"""
from __future__ import annotations

import numpy as np
import torch

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from isr.train.vec_env import Stage4VectorPursuitEnv
from isr.train.graph_buffer import Stage4RolloutBuffer


def _env(**overrides):
    kw = dict(
        n_blue=3, n_red=2, n_obstacles=4, arena_size=80.0, max_steps=50,
        red_policy=stationary_red,
    )
    kw.update(overrides)
    return PursuitEnv(**kw)


# ---------------------------------------------------------------------------
# Reward decomposition:  r_i = r_team + r_crash_i
# ---------------------------------------------------------------------------

def test_penalties_off_by_default_gives_shared_reward():
    """With penalties at 0 the reward is byte-identical across agents
    (the pre-crash shared-team-reward behaviour is preserved)."""
    env = _env()  # crash penalties default to 0.0
    env.reset(seed=3)
    acts = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    _, rew, _, _, _ = env.step(acts)
    vals = list(rew.values())
    assert all(abs(v - vals[0]) < 1e-9 for v in vals), rew


def test_blue_blue_crash_penalises_only_offenders():
    """Two blues placed on top of each other each eat the ally penalty;
    the third (clear) blue keeps the shared reward only."""
    env = _env(crash_blue_penalty=1.0, blue_collision_radius=2.0)
    env.reset(seed=1)
    # Force blue_0 and blue_1 into a collision; keep blue_2 far away.
    env._blue_pos[0] = np.array([40.0, 40.0])
    env._blue_pos[1] = np.array([40.5, 40.0])
    env._blue_pos[2] = np.array([10.0, 10.0])
    acts = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    _, rew, _, _, info = env.step(acts)

    # r_team is shared, so the crash penalty shows up as the difference.
    assert abs((rew["blue_0"] - rew["blue_2"]) - (-1.0)) < 1e-6
    assert abs((rew["blue_1"] - rew["blue_2"]) - (-1.0)) < 1e-6
    assert info["blue_0"]["blue_crashes"] == 2


def test_obstacle_crash_penalises_and_does_not_terminate():
    """Driving a blue into an obstacle applies the individual obstacle
    penalty (soft-stop) and the episode does NOT terminate."""
    env = _env(crash_obstacle_penalty=2.0, crash_blue_penalty=1.0)
    env.reset(seed=7)
    # Place blue_0 inside an obstacle so any move keeps it inside ->
    # the soft-crash rollback fires deterministically.
    env._blue_pos[0] = env._obstacle_pos[0].copy()
    # Keep the other blues well clear of each other and the obstacle.
    env._blue_pos[1] = np.array([5.0, 5.0])
    env._blue_pos[2] = np.array([5.0, 70.0])
    acts = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    acts["blue_0"] = np.array([1.0, 1.0], dtype=np.float32)  # try to move
    _, rew, term, trunc, info = env.step(acts)

    assert info["blue_0"]["obstacle_crashes"] >= 1
    # blue_0 penalised relative to a clear blue by ~ the obstacle penalty.
    assert (rew["blue_0"] - rew["blue_1"]) <= -2.0 + 1e-6
    # Soft-stop: no termination on crash.
    assert not any(term.values())
    assert not any(trunc.values())


# ---------------------------------------------------------------------------
# Per-agent critic V(s, i)
# ---------------------------------------------------------------------------

def test_critic_forward_is_per_agent():
    ek = dict(n_blue=3, n_red=2, n_obstacles=4, sensor_radius=40.0,
              use_belief_maps=True, max_steps=30)
    ve = Stage4VectorPursuitEnv(n_envs=4, env_kwargs=ek,
                                red_policy_mix=[("stationary", 1.0)])
    obs = ve.reset(seed=0)
    ot = {k: torch.from_numpy(v).float() for k, v in obs.items()}
    _, full = split_stage4_obs(ot)
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=4)
    v, h_blue = pol.critic_forward(full)
    assert v.shape == (4, 3), v.shape           # (n_envs, n_blue) = V(s, i)
    assert h_blue.shape[:2] == (4, 3)


def test_critic_per_agent_values_can_differ():
    """A shared-reward V(s) would give identical values to all agents;
    the agent-conditioned V(s, i) reads each blue's own node embedding,
    so the per-agent values are genuinely distinct."""
    ek = dict(n_blue=3, n_red=2, n_obstacles=0, sensor_radius=40.0,
              use_belief_maps=True, max_steps=30)
    ve = Stage4VectorPursuitEnv(n_envs=2, env_kwargs=ek,
                                red_policy_mix=[("stationary", 1.0)])
    obs = ve.reset(seed=5)
    ot = {k: torch.from_numpy(v).float() for k, v in obs.items()}
    _, full = split_stage4_obs(ot)
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=0)
    v, _ = pol.critic_forward(full)
    # Not all three per-agent values are equal within an env.
    spread = (v.max(dim=1).values - v.min(dim=1).values)
    assert (spread.abs() > 1e-6).any()


# ---------------------------------------------------------------------------
# Per-agent GAE in the buffer
# ---------------------------------------------------------------------------

def test_buffer_per_agent_gae_shapes_and_decomposition():
    T, E, A = 4, 2, 3
    buf = Stage4RolloutBuffer(rollout_steps=T, n_envs=E, n_agents=A,
                              action_dim=2, d_hidden=8,
                              device=torch.device("cpu"))
    for t in range(T):
        buf.add(
            obs={"blue_features": torch.zeros(E, A, 8)},
            actions=torch.zeros(E, A, 2),
            log_probs=torch.zeros(E, A),
            values=torch.zeros(E, A),
            rewards=torch.ones(E, A),          # per-agent reward
            dones=torch.zeros(E),              # per-env done
            hidden=torch.zeros(E, A, 8),
        )
    buf.compute_gae(last_value=torch.zeros(E, A), gamma=0.99, gae_lambda=0.95)
    assert buf.advantages.shape == (T, E, A)
    assert buf.returns.shape == (T, E, A)
    # returns == advantages + values (values are 0 here).
    assert torch.allclose(buf.returns, buf.advantages)


def test_buffer_gae_broadcasts_per_env_done_over_agents():
    """A done at the last env step must zero the bootstrap for BOTH
    agents in that env (per-env done broadcast over the agent dim)."""
    T, E, A = 2, 1, 2
    buf = Stage4RolloutBuffer(rollout_steps=T, n_envs=E, n_agents=A,
                              action_dim=2, d_hidden=4,
                              device=torch.device("cpu"))
    buf.add(obs={"x": torch.zeros(E, A, 1)},
            actions=torch.zeros(E, A, 2), log_probs=torch.zeros(E, A),
            values=torch.zeros(E, A), rewards=torch.tensor([[1.0, 2.0]]),
            dones=torch.zeros(E), hidden=torch.zeros(E, A, 4))
    buf.add(obs={"x": torch.zeros(E, A, 1)},
            actions=torch.zeros(E, A, 2), log_probs=torch.zeros(E, A),
            values=torch.zeros(E, A), rewards=torch.tensor([[3.0, 4.0]]),
            dones=torch.ones(E), hidden=torch.zeros(E, A, 4))
    buf.compute_gae(last_value=torch.tensor([[100.0, 100.0]]),
                    gamma=1.0, gae_lambda=1.0)
    # Last step is done -> bootstrap (100) must NOT leak in; adv == reward.
    assert torch.allclose(buf.advantages[1], torch.tensor([[3.0, 4.0]]))


# ---------------------------------------------------------------------------
# Full (actor + critic) Stage 4 -> Stage 4 warm-start
# ---------------------------------------------------------------------------

def test_load_full_stage4_copies_all_tensors(tmp_path):
    src = GNNStage4Policy(n_blue=3, n_red=2, n_obs=4)
    # Perturb src so it differs from a fresh init.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.1)
    ckpt = tmp_path / "src.pt"
    torch.save({"policy_state": src.state_dict()}, ckpt)

    dst = GNNStage4Policy(n_blue=3, n_red=2, n_obs=4)
    logs = []
    n_copied = dst.load_full_stage4(str(ckpt), log=logs.append)
    assert n_copied == len(dst.state_dict())
    for k, v in src.state_dict().items():
        assert torch.allclose(dst.state_dict()[k], v), k
    # Identical architecture -> nothing left at init -> nothing logged.
    assert logs == []


def test_load_full_stage4_reports_reinitialised_tensors(tmp_path):
    """Warm-start across an architecture change must copy everything
    shape-matching and NAME the tensors it leaves at init.

    Dropping obstacles entirely (n_obs 4 -> 0) removes the obstacle term
    from the masked-pool global context, so the critic-trunk input
    projection narrows and cannot transfer.  (Note 4 -> 2 would NOT skip
    anything -- the pool is count-invariant -- which is exactly the
    variable-entities property.)"""
    src = GNNStage4Policy(n_blue=3, n_red=2, n_obs=4)
    ckpt = tmp_path / "src.pt"
    torch.save({"policy_state": src.state_dict()}, ckpt)

    dst = GNNStage4Policy(n_blue=3, n_red=2, n_obs=0)
    logs = []
    n_copied = dst.load_full_stage4(str(ckpt), log=logs.append)

    assert n_copied < len(dst.state_dict())        # something skipped
    joined = "\n".join(logs)
    assert "critic_trunk.0.weight" in joined       # named explicitly
    assert "reinitialised" in joined


# ---------------------------------------------------------------------------
# Stage-1 back-compat: base vec env is per-agent; Stage 1 collapses col 0
# ---------------------------------------------------------------------------

def test_vec_env_reward_is_per_agent_and_stage1_collapse_matches():
    ek = dict(n_blue=3, n_red=2, n_obstacles=4, sensor_radius=40.0,
              use_belief_maps=True, max_steps=30)
    ve = Stage4VectorPursuitEnv(n_envs=4, env_kwargs=ek,
                                red_policy_mix=[("stationary", 1.0)])
    ve.reset(seed=0)
    _, reward_np, _, _ = ve.step(np.zeros((4, 3, 2), dtype=np.float32))
    assert reward_np.shape == (4, 3)      # (n_envs, n_agents)
    # Stage 1 (shared reward) collapses to per-env via column 0; since
    # this env has crash penalties OFF, every agent's reward is equal,
    # so the collapse loses nothing.
    collapsed = reward_np[:, 0]
    assert collapsed.shape == (4,)
    assert np.allclose(reward_np, reward_np[:, [0]])
