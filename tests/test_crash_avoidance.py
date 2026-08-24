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
    """With penalties at 0 AND zero actions, every agent sees the same
    reward — only the shared team terms (catch bonus, step cost) remain.
    Control effort is individual, so this holds only for equal actions;
    see test_action_cost_is_individual."""
    env = _env()  # crash penalties default to 0.0
    env.reset(seed=3)
    acts = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    _, rew, _, _, _ = env.step(acts)
    vals = list(rew.values())
    assert all(abs(v - vals[0]) < 1e-9 for v in vals), rew


def test_action_cost_is_individual_not_shared():
    """Control effort is first-person: a blue that manoeuvres hard must pay
    for it ALONE.  Previously the cost was summed over the whole team and
    charged to everyone, so ~81% of what an agent paid was its teammates'
    effort — pure gradient noise."""
    env = _env()                       # penalties off: isolate action cost
    env.reset(seed=3)
    agents = env.possible_agents
    acts = {a: np.zeros(2, dtype=np.float32) for a in agents}
    acts[agents[0]] = np.array([1.0, 1.0], dtype=np.float32)   # full effort
    _, rew, _, _, _ = env.step(acts)

    # The mover pays 0.01 * |a|^2 = 0.01 * 2 = 0.02; the others pay nothing.
    assert np.isclose(rew[agents[1]] - rew[agents[0]], 0.02, atol=1e-6)
    idle = [rew[a] for a in agents[1:]]
    assert all(abs(v - idle[0]) < 1e-9 for v in idle), (
        "idle agents must be unaffected by a teammate's manoeuvring")


def test_action_cost_gradient_wrt_own_action_is_unchanged():
    """The whole justification for the split: d/da_i of the teammates'
    terms is 0, so moving to a per-agent cost is variance reduction, NOT a
    change of objective.  Numerically, d(reward_i)/d(a_i) must still equal
    the analytic -0.02 * a_i."""
    env = _env()
    env.reset(seed=3)
    agents = env.possible_agents
    a0, h = 0.6, 1e-4

    def r0(x):
        e = _env()
        e.reset(seed=3)
        acts = {a: np.zeros(2, dtype=np.float32) for a in agents}
        acts[agents[0]] = np.array([x, 0.0], dtype=np.float32)
        return e.step(acts)[1][agents[0]]

    numeric = (r0(a0 + h) - r0(a0 - h)) / (2 * h)
    assert np.isclose(numeric, -0.02 * a0, atol=1e-3), (
        f"own-action gradient changed: {numeric} vs {-0.02 * a0}")


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
    # Stage 1 is a shared-reward learner and collapses to a per-env scalar
    # with the MEAN over agents (column 0 would be one agent's reward, not
    # the team's, now that control effort is individual).  Actions are zero
    # here and penalties are off, so every agent is still equal and the
    # collapse is exact either way.
    collapsed = reward_np.mean(axis=1)
    assert collapsed.shape == (4,)
    assert np.allclose(reward_np, reward_np[:, [0]])


# ---------------------------------------------------------------------------
#  Reward shape is configurable (was hard-coded in _step)
# ---------------------------------------------------------------------------

def test_reward_constants_default_to_historical_values():
    """Defaults must reproduce the old hard-coded reward exactly, so every
    earlier run stays reproducible."""
    env = _env()
    assert env.catch_reward == 10.0
    assert env.step_cost == 0.05
    assert env.uncaught_penalty == 5.0
    assert env.action_cost_coef == 0.01


def test_step_cost_and_action_cost_coef_take_effect():
    """Both the shared step cost and the individual effort coefficient are
    read from the constructor, not baked in."""
    agents_a = None
    env = PursuitEnv(n_blue=2, n_red=1, n_obstacles=0, arena_size=100.0,
                     max_steps=50, capture_radius=3.0,
                     step_cost=0.5, action_cost_coef=0.1,
                     red_policy=stationary_red, seed=0)
    env.reset(seed=0)
    agents_a = env.possible_agents
    acts = {a: np.zeros(2, dtype=np.float32) for a in agents_a}
    acts[agents_a[0]] = np.array([1.0, 0.0], dtype=np.float32)
    _, rew, _, _, _ = env.step(acts)
    # idle agent pays only the shared step cost
    assert np.isclose(rew[agents_a[1]], -0.5, atol=1e-6)
    # mover additionally pays 0.1 * |a|^2 = 0.1
    assert np.isclose(rew[agents_a[0]], -0.6, atol=1e-6)


def test_catch_reward_and_uncaught_penalty_take_effect():
    """catch_reward is paid on a capture; uncaught_penalty is charged per
    surviving red when the episode ends."""
    # Capture: blue starts on top of the red -> caught on the first step.
    env = PursuitEnv(n_blue=1, n_red=1, n_obstacles=0, arena_size=100.0,
                     max_steps=50, capture_radius=5.0, catch_reward=3.0,
                     step_cost=0.0, action_cost_coef=0.0,
                     red_policy=stationary_red, seed=0)
    env.reset(seed=0)
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._red_pos[0] = np.array([51.0, 50.0], dtype=np.float32)
    _, rew, term, _, _ = env.step({"blue_0": np.zeros(2, dtype=np.float32)})
    assert term["blue_0"] and np.isclose(rew["blue_0"], 3.0, atol=1e-6)

    # Truncation with the red alive -> -uncaught_penalty.
    env = PursuitEnv(n_blue=1, n_red=1, n_obstacles=0, arena_size=100.0,
                     max_steps=1, capture_radius=0.5, uncaught_penalty=2.0,
                     step_cost=0.0, action_cost_coef=0.0,
                     red_policy=stationary_red, seed=0)
    env.reset(seed=0)
    env._blue_pos[0] = np.array([1.0, 1.0], dtype=np.float32)
    env._red_pos[0] = np.array([99.0, 99.0], dtype=np.float32)
    _, rew, _, trunc, _ = env.step({"blue_0": np.zeros(2, dtype=np.float32)})
    assert trunc["blue_0"] and np.isclose(rew["blue_0"], -2.0, atol=1e-6)


def test_config_exposes_reward_shape():
    """The four constants live in the config chain (stage1 -> 3 -> 4)."""
    from isr.configs.stage1_default import STAGE1_DEFAULTS
    from isr.configs.stage4_default import STAGE4_DEFAULTS
    for k, v in (("catch_reward", 10.0), ("step_cost", 0.05),
                 ("uncaught_penalty", 5.0), ("action_cost_coef", 0.01)):
        assert STAGE1_DEFAULTS[k] == v
        assert STAGE4_DEFAULTS[k] == v      # inherited
