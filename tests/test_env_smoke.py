"""
tests/test_env_smoke.py — API-contract + sanity tests for PursuitEnv.

These are NOT learning tests — they catch regressions in the env's
public interface (dict shapes, dtypes, termination semantics, capture
geometry, determinism) when we extend the env at later stages.

Run:
    pytest tests/test_env_smoke.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.env.pursuit_env import PursuitEnv
from isr.env.entities import BLUE_UAV, RED_TARGET
from isr.agents.heuristics import GreedyPursuer, stationary_red


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------

def test_reset_returns_proper_dicts():
    env = PursuitEnv(seed=42)
    obs, info = env.reset(seed=42)

    assert set(obs.keys())  == set(env.possible_agents)
    assert set(info.keys()) == set(env.possible_agents)
    for a in env.possible_agents:
        assert obs[a].shape == (env._obs_dim,)
        assert obs[a].dtype == np.float32


def test_observation_and_action_spaces_match_outputs():
    env = PursuitEnv()
    obs, _ = env.reset(seed=0)
    for a in env.possible_agents:
        assert env.observation_space(a).shape == obs[a].shape
        assert env.action_space(a).shape       == (2,)
        assert env.action_space(a).low.min()   == -1.0
        assert env.action_space(a).high.max()  == +1.0


def test_step_returns_5_dicts_with_matching_keys():
    env = PursuitEnv(seed=0)
    obs, _ = env.reset(seed=0)
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    expected = set(env.possible_agents)
    for d in (obs, rew, term, trunc, info):
        assert set(d.keys()) == expected, f"key set mismatch: {d.keys()} vs {expected}"


def test_self_idx_onehot_is_identity_block():
    """Each blue agent's obs differs only in the self_idx_onehot block."""
    env = PursuitEnv(seed=0)
    obs, _ = env.reset(seed=0)
    n = env.n_blue
    # The onehot lives at obs[-(n_blue+1):-1] — last (n_blue+1) entries
    # are [self_idx_onehot (n_blue floats), time_remaining (1 float)].
    onehots = np.stack([obs[a][-(n + 1):-1] for a in env.possible_agents])
    assert np.allclose(onehots, np.eye(n)), \
        f"self_idx_onehot block should be identity, got:\n{onehots}"


# ---------------------------------------------------------------------------
# Termination semantics
# ---------------------------------------------------------------------------

def test_capture_inside_radius_triggers_termination_for_n1():
    """
    With one blue and one red placed 2.8 units apart and capture_radius=5,
    a single step should catch the red and end the episode for all agents.
    """
    env = PursuitEnv(
        n_blue=1, n_red=1, capture_radius=5.0,
        red_policy=stationary_red, seed=0,
    )
    env.reset(seed=0)
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._red_pos[0]  = np.array([52.0, 52.0], dtype=np.float32)  # dist ≈ 2.83

    obs, rew, term, trunc, info = env.step({"blue_0": np.zeros(2, dtype=np.float32)})

    assert term["blue_0"] is True
    assert trunc["blue_0"] is False
    assert info["blue_0"]["n_caught_this_step"] == 1
    assert info["blue_0"]["n_red_remaining"]    == 0
    assert env.agents == [], "PettingZoo: agents list empties at terminal step"


def test_outside_capture_radius_does_not_trigger():
    """A red far from any blue should not be caught."""
    env = PursuitEnv(
        n_blue=1, n_red=1, capture_radius=1.0,
        red_policy=stationary_red, seed=0,
    )
    env.reset(seed=0)
    env._blue_pos[0] = np.array([10.0, 10.0], dtype=np.float32)
    env._red_pos[0]  = np.array([90.0, 90.0], dtype=np.float32)

    obs, rew, term, trunc, info = env.step({"blue_0": np.zeros(2, dtype=np.float32)})

    assert term["blue_0"] is False
    assert info["blue_0"]["n_caught_this_step"] == 0


def test_episode_truncates_at_max_steps_with_uncaught_red():
    """
    Constrain so blue cannot reach red within max_steps; episode should
    truncate (not terminate) and reward should include the terminal
    penalty for the uncaught red.
    """
    env = PursuitEnv(
        n_blue=1, n_red=1, max_steps=5, capture_radius=0.5,
        red_policy=stationary_red, seed=0,
    )
    env.reset(seed=0)
    env._blue_pos[0] = np.array([1.0, 1.0], dtype=np.float32)
    env._red_pos[0]  = np.array([99.0, 99.0], dtype=np.float32)

    rewards_seen = []
    trunc_seen = False
    for _ in range(10):
        obs, rew, term, trunc, info = env.step({"blue_0": np.zeros(2, dtype=np.float32)})
        rewards_seen.append(rew["blue_0"])
        if trunc["blue_0"]:
            trunc_seen = True
            break
    assert trunc_seen, "Episode should truncate after max_steps"
    # Last reward should include the -5 terminal penalty for 1 uncaught red.
    # The per-step component is just -step_cost = -0.05 (zero action).
    assert rewards_seen[-1] <= -5.0, \
        f"Expected terminal reward ~-5.05 (penalty + step cost), got {rewards_seen[-1]}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_seed_determinism_same_initial_obs():
    """Two envs constructed with the same seed produce identical resets."""
    obs_a, _ = PursuitEnv(seed=123).reset(seed=123)
    obs_b, _ = PursuitEnv(seed=123).reset(seed=123)
    for a in obs_a:
        assert np.array_equal(obs_a[a], obs_b[a])


def test_seed_determinism_full_episode():
    """Same seed + same actions => same trajectory."""
    def run_one():
        env = PursuitEnv(red_policy=stationary_red, seed=7)
        obs, _ = env.reset(seed=7)
        blue = GreedyPursuer()
        trajectory = []
        while env.agents:
            actions = {a: blue.act(obs[a], env, a) for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            trajectory.append((float(rew["blue_0"]),
                               env.state_snapshot()["t"],
                               int(env.state_snapshot()["red_active"].sum())))
        return trajectory
    traj1 = run_one()
    traj2 = run_one()
    assert traj1 == traj2, "Episode trajectory should be deterministic for fixed seed"


# ---------------------------------------------------------------------------
# Calibration sanity (heuristics)
# ---------------------------------------------------------------------------

def test_greedy_catches_stationary_reds():
    """
    GreedyPursuer vs stationary reds — across multiple seeds, all reds
    should be caught and the episode return should be positive.
    """
    for seed in (0, 1, 2):
        env = PursuitEnv(red_policy=stationary_red, seed=seed)
        obs, _ = env.reset(seed=seed)
        blue = GreedyPursuer()
        total = 0.0
        while env.agents:
            actions = {a: blue.act(obs[a], env, a) for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            total += rew["blue_0"]
        snap = env.state_snapshot()
        assert int((~snap["red_active"]).sum()) == env.n_red, \
            f"seed={seed}: GreedyPursuer should catch all stationary reds"
        assert total > 5.0, \
            f"seed={seed}: Expected return > +5, got {total}"


# ---------------------------------------------------------------------------
# Kinematic invariants
# ---------------------------------------------------------------------------

def test_positions_stay_inside_arena():
    """No matter the actions, positions must stay in [0, arena_size]."""
    env = PursuitEnv(seed=0)
    obs, _ = env.reset(seed=0)
    # Maximum-magnitude actions in opposite directions — try to escape.
    actions = {a: np.array([1.0, 1.0], dtype=np.float32) for a in env.agents}
    for _ in range(env.max_steps):
        if not env.agents:
            break
        obs, rew, term, trunc, info = env.step(actions)
        snap = env.state_snapshot()
        for arr_name in ("blue_pos", "red_pos"):
            arr = snap[arr_name]
            assert (arr >= 0.0).all() and (arr <= env.arena_size).all(), \
                f"{arr_name} escaped arena at step {snap['t']}: {arr}"


def test_velocities_respect_class_cap():
    """Each entity's per-axis velocity must respect the class v_max."""
    env = PursuitEnv(seed=0)
    obs, _ = env.reset(seed=0)
    actions = {a: np.array([1.0, 1.0], dtype=np.float32) for a in env.agents}
    for _ in range(10):
        if not env.agents:
            break
        obs, rew, term, trunc, info = env.step(actions)
        snap = env.state_snapshot()
        assert (np.abs(snap["blue_vel"]) <= BLUE_UAV.v_max + 1e-6).all()
        assert (np.abs(snap["red_vel"])  <= RED_TARGET.v_max + 1e-6).all()
