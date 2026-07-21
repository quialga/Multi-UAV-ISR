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


def test_obs_dim_matches_v2_formula():
    """v2 obs dim = 8*N_red + 7*(N_blue - 1) + 8."""
    for n_blue, n_red in [(3, 2), (2, 3), (4, 1), (1, 2)]:
        env = PursuitEnv(n_blue=n_blue, n_red=n_red)
        obs, _ = env.reset(seed=0)
        expected = 8 * n_red + 7 * (n_blue - 1) + 8
        assert env._obs_dim == expected, (
            f"n_blue={n_blue} n_red={n_red}: env._obs_dim={env._obs_dim} "
            f"!= expected {expected}"
        )
        assert obs["blue_0"].shape == (expected,)


def test_obs_is_ego_centric_translation_invariant():
    """
    Translating the entire scene (all blues + reds by the same vector)
    should leave every agent's obs bit-exact — ego-centric relative obs
    is translation invariant.  The only field affected by translation is
    ``wall_distances``, which changes as expected; every other feature
    is a relative quantity and must match.
    """
    env_a = PursuitEnv(n_blue=3, n_red=2,
                       red_policy=stationary_red, seed=0)
    env_b = PursuitEnv(n_blue=3, n_red=2,
                       red_policy=stationary_red, seed=0)
    env_a.reset(seed=0)
    env_b.reset(seed=0)

    # Copy positions from env_a, translate everyone by (+5, +7) into env_b.
    dx, dy = 5.0, 7.0
    env_b._blue_pos = env_a._blue_pos + np.array([dx, dy], dtype=np.float32)
    env_b._blue_vel = env_a._blue_vel.copy()
    env_b._red_pos  = env_a._red_pos  + np.array([dx, dy], dtype=np.float32)
    env_b._red_vel  = env_a._red_vel.copy()
    env_b._red_active = env_a._red_active.copy()
    env_b._t = env_a._t

    obs_a = env_a._build_obs(0)
    obs_b = env_b._build_obs(0)

    # Relative blocks (everything except wall_distances) must match.
    # Wall block is the 4 entries starting 5 back from the end:
    # ... [self_vel (2), self_speed (1), walls (4), time (1)]
    wall_start = -1 - 4   # exclusive end of walls = -1, start = -5
    wall_end   = -1
    # Compare everything except the wall block:
    a_pre  = obs_a[:wall_start]; b_pre  = obs_b[:wall_start]
    a_post = obs_a[wall_end:];   b_post = obs_b[wall_end:]
    assert np.allclose(a_pre, b_pre, atol=1e-6), \
        "pre-walls block should be translation-invariant"
    assert np.allclose(a_post, b_post, atol=1e-6), \
        "time_remaining should be identical (both envs at same step)"
    # Wall block should differ predictably (shifted by dx/dy /L).
    walls_a = obs_a[wall_start:wall_end]
    walls_b = obs_b[wall_start:wall_end]
    L = env_a.arena_size
    expected_delta = np.array([dx/L, -dx/L, dy/L, -dy/L], dtype=np.float32)
    assert np.allclose(walls_b - walls_a, expected_delta, atol=1e-6), \
        f"wall_distances delta mismatch: {walls_b - walls_a} vs {expected_delta}"


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
# Stage 3 partial-observability (sensor_radius)
# ---------------------------------------------------------------------------

def test_full_obs_when_sensor_radius_none():
    """Default env is fully observable — visibility masks all ones."""
    env = PursuitEnv(n_blue=3, n_red=2, seed=0)   # sensor_radius=None (default)
    env.reset(seed=0)
    bb_v, rb_v = env._compute_edge_visibility()
    assert np.all(bb_v == 1.0)
    assert np.all(rb_v == 1.0)


def test_visibility_masks_reflect_within_radius():
    """
    Place agents at known positions and verify the visibility masks
    (``_compute_edge_visibility``, shared by the Stage 4 bb_edge_visible)
    match hand-computed distances vs sensor_radius.
    """
    env = PursuitEnv(n_blue=3, n_red=2, arena_size=100.0,
                     sensor_radius=20.0, red_policy=stationary_red, seed=0)
    env.reset(seed=0)
    # Overwrite positions to a controlled configuration:
    #   blue_0 at (50, 50), blue_1 at (55, 50), blue_2 at (80, 80)
    #   red_0  at (60, 50), red_1  at (95, 95)
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._blue_pos[1] = np.array([55.0, 50.0], dtype=np.float32)
    env._blue_pos[2] = np.array([80.0, 80.0], dtype=np.float32)
    env._red_pos[0]  = np.array([60.0, 50.0], dtype=np.float32)
    env._red_pos[1]  = np.array([95.0, 95.0], dtype=np.float32)

    bb_v, rb_v = env._compute_edge_visibility()

    # For each bb edge, verify visibility by manual distance check.
    for e in range(env.n_bb_edges):
        s, d = int(env.bb_edge_src[e]), int(env.bb_edge_dst[e])
        dist = np.linalg.norm(env._blue_pos[d] - env._blue_pos[s])
        expected = 1.0 if dist <= 20.0 else 0.0
        assert bb_v[e] == expected, (
            f"bb edge {e} ({s}->{d}) dist={dist:.2f} — mask {bb_v[e]} "
            f"expected {expected}"
        )

    # For each rb edge (r -> b), verify visibility by manual check.
    for e in range(env.n_rb_edges):
        r, b = int(env.rb_edge_src[e]), int(env.rb_edge_dst[e])
        dist = np.linalg.norm(env._blue_pos[b] - env._red_pos[r])
        expected = 1.0 if dist <= 20.0 else 0.0
        assert rb_v[e] == expected, (
            f"rb edge {e} (r={r}->b={b}) dist={dist:.2f} — mask "
            f"{rb_v[e]} expected {expected}"
        )


def test_bb_visibility_is_symmetric():
    """
    Euclidean distance is symmetric, so bb edges (i->j) and (j->i)
    should always be visible or hidden together.
    """
    env = PursuitEnv(n_blue=5, n_red=3, arena_size=130.0,
                     sensor_radius=40.0, seed=42)
    env.reset(seed=42)
    bb_v, _ = env._compute_edge_visibility()
    for e in range(env.n_bb_edges):
        s, d = int(env.bb_edge_src[e]), int(env.bb_edge_dst[e])
        # Find the reverse-direction edge.
        rev = None
        for e2 in range(env.n_bb_edges):
            if int(env.bb_edge_src[e2]) == d and int(env.bb_edge_dst[e2]) == s:
                rev = e2
                break
        assert rev is not None
        assert bb_v[e] == bb_v[rev], (
            f"bb visibility asymmetric: edge {e} ({s}->{d}) mask={bb_v[e]} "
            f"vs edge {rev} ({d}->{s}) mask={bb_v[rev]}"
        )


def test_caught_reds_are_hidden_regardless_of_range():
    """
    Once a red is caught, all rb edges from it must be zeroed even if
    physically inside sensor_radius.  Prevents the actor from
    treating a caught red as a live target.
    """
    env = PursuitEnv(n_blue=1, n_red=2, capture_radius=5.0,
                     sensor_radius=200.0,  # everything in range
                     red_policy=stationary_red, seed=0)
    env.reset(seed=0)
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._red_pos[0]  = np.array([52.0, 52.0], dtype=np.float32)  # will be caught
    env._red_pos[1]  = np.array([70.0, 70.0], dtype=np.float32)  # still alive
    env.step({"blue_0": np.zeros(2, dtype=np.float32)})
    assert not env._red_active[0]
    _, rb_v = env._compute_edge_visibility()
    for e in range(env.n_rb_edges):
        r = int(env.rb_edge_src[e])
        if r == 0:
            assert rb_v[e] == 0.0, (
                f"rb edge {e} from caught red {r} should be hidden"
            )
        else:
            assert rb_v[e] == 1.0, (
                f"rb edge {e} from alive red {r} should be visible"
            )


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


# ---------------------------------------------------------------------------
# Stage 2 structured / graph observation
# ---------------------------------------------------------------------------

def test_structured_obs_shapes():
    env = PursuitEnv(n_blue=3, n_red=2, seed=0)
    env.reset(seed=0)
    obs = env.structured_observation()
    assert set(obs.keys()) == {"blue_features", "red_features",
                               "bb_edge_features", "rb_edge_features"}
    assert obs["blue_features"].shape    == (3, 8)
    assert obs["red_features"].shape     == (2, 1)
    assert obs["bb_edge_features"].shape == (6, 7)   # 3 * (3 - 1)
    assert obs["rb_edge_features"].shape == (6, 7)   # 2 * 3
    # Dims consistent with env-exposed constants
    assert env.blue_feat_dim == obs["blue_features"].shape[1]
    assert env.red_feat_dim  == obs["red_features"].shape[1]
    assert env.edge_feat_dim == obs["bb_edge_features"].shape[1]
    assert env.n_bb_edges    == obs["bb_edge_features"].shape[0]
    assert env.n_rb_edges    == obs["rb_edge_features"].shape[0]


def test_bb_edges_are_bidirectional_with_negated_rel_pos():
    """
    For every blue-blue pair (i, j), the directed edge i->j and its
    reverse j->i must carry rel_pos with opposite sign and identical
    range.  Bearing_cs will *differ* (it's in sender frame) so we
    do not assert on it here.
    """
    env = PursuitEnv(n_blue=3, n_red=2, seed=1)
    env.reset(seed=1)
    obs = env.structured_observation()
    bb_src, bb_dst = env.bb_edge_src, env.bb_edge_dst
    bb_feats = obs["bb_edge_features"]      # (6, 7)

    for a in range(len(bb_src)):
        s_a, d_a = int(bb_src[a]), int(bb_dst[a])
        # find the reverse-direction edge
        rev = None
        for b in range(len(bb_src)):
            if int(bb_src[b]) == d_a and int(bb_dst[b]) == s_a:
                rev = b
                break
        assert rev is not None
        rel_pos_a = bb_feats[a,   0:2]
        rel_pos_b = bb_feats[rev, 0:2]
        assert np.allclose(rel_pos_a, -rel_pos_b, atol=1e-6), \
            f"edge {a} ({s_a}->{d_a}) rel_pos should negate edge {rev}"
        range_a = bb_feats[a,   4]
        range_b = bb_feats[rev, 4]
        assert np.isclose(range_a, range_b, atol=1e-6), \
            "range must be symmetric across direction"


def test_rb_edges_are_directed_and_carry_correct_rel_pos():
    """rb edge features go red -> blue: rel_pos = blue_pos - red_pos."""
    env = PursuitEnv(n_blue=3, n_red=2, red_policy=stationary_red, seed=2)
    env.reset(seed=2)
    obs = env.structured_observation()
    rb_feats = obs["rb_edge_features"]
    L = env.arena_size
    for i in range(len(env.rb_edge_src)):
        r = int(env.rb_edge_src[i])
        b = int(env.rb_edge_dst[i])
        expected_rel_pos_n = (env._blue_pos[b] - env._red_pos[r]) / L
        got = rb_feats[i, 0:2]
        assert np.allclose(got, expected_rel_pos_n, atol=1e-6), \
            f"rb edge {i} (r={r} -> b={b}) rel_pos mismatch"


def test_rb_edges_zeroed_for_caught_reds():
    """
    After a red is caught, all rb edges *from* that red must be zero
    (the network can still see active_flag on the node itself).
    """
    env = PursuitEnv(n_blue=1, n_red=2, capture_radius=5.0,
                     red_policy=stationary_red, seed=3)
    env.reset(seed=3)
    # Force positions so red_0 gets caught, red_1 stays alive far away.
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._red_pos[0]  = np.array([52.0, 52.0], dtype=np.float32)
    env._red_pos[1]  = np.array([90.0, 90.0], dtype=np.float32)
    # step to trigger capture of red_0 only
    env.step({"blue_0": np.zeros(2, dtype=np.float32)})
    assert not env._red_active[0]      # red_0 caught
    assert env._red_active[1]           # red_1 alive
    obs = env.structured_observation()
    rb_feats = obs["rb_edge_features"]
    for i in range(len(env.rb_edge_src)):
        r = int(env.rb_edge_src[i])
        if not env._red_active[r]:
            assert np.allclose(rb_feats[i], 0.0), \
                f"rb edge {i} from caught red {r} should be zeroed"
    # red_1's edges must be non-zero (assuming non-degenerate geometry)
    for i in range(len(env.rb_edge_src)):
        r = int(env.rb_edge_src[i])
        if r == 1:
            assert not np.allclose(rb_feats[i], 0.0), \
                f"rb edge {i} from alive red {r} should carry data"


def test_blue_node_wall_distances():
    """wall_distances at cols 3..7 should reproduce (x, L-x, y, L-y)/L."""
    env = PursuitEnv(n_blue=3, n_red=2, seed=4)
    env.reset(seed=4)
    env._blue_pos[0] = np.array([5.0, 5.0], dtype=np.float32)   # near lower-left
    obs = env.structured_observation()
    walls = obs["blue_features"][0, 3:7]
    L = env.arena_size
    expected = np.array([5.0 / L, (L - 5.0) / L, 5.0 / L, (L - 5.0) / L],
                        dtype=np.float32)
    assert np.allclose(walls, expected, atol=1e-6), \
        f"wall_distances {walls} should equal {expected}"


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


# ---------------------------------------------------------------------------
# Stage 4: obstacles + belief maps
# ---------------------------------------------------------------------------

def _stage4_env(**overrides):
    """Convenience factory for Stage 4 configs used by several tests."""
    kwargs = dict(
        n_blue=5, n_red=3, arena_size=130.0, capture_radius=3.0,
        sensor_radius=40.0,
        n_obstacles=4, obstacle_radius_min=5.0, obstacle_radius_max=15.0,
        # v6.1: ONE global fused belief map (C, H, W); 2 Bayesian
        # channels (enemy, obstacle); grid back at 26x26 (5 m cells).
        use_belief_maps=True, belief_grid_size=26, belief_channels=2,
        red_policy=stationary_red, seed=0,
    )
    kwargs.update(overrides)
    return PursuitEnv(**kwargs)


def test_stage4_obstacles_are_placed_and_non_overlapping():
    env = _stage4_env(n_obstacles=4)
    env.reset(seed=0)
    assert env._obstacle_pos.shape == (4, 2)
    assert env._obstacle_r.shape == (4,)
    # Non-overlapping (allow a small overlap tolerance since placement
    # uses "≥ r1+r2+1.0" so the actual minimum gap is 1 m).
    for i in range(4):
        for j in range(i + 1, 4):
            dist = float(np.linalg.norm(
                env._obstacle_pos[i] - env._obstacle_pos[j]))
            assert dist >= env._obstacle_r[i] + env._obstacle_r[j], (
                f"Obstacles {i}, {j} overlap: dist={dist:.2f}, "
                f"r_sum={env._obstacle_r[i] + env._obstacle_r[j]:.2f}"
            )
    # Obstacles are inside the arena (with the requested wall clearance).
    for i in range(4):
        pos = env._obstacle_pos[i]
        assert env.obstacle_radius_max <= pos[0] <= env.arena_size - env.obstacle_radius_max
        assert env.obstacle_radius_max <= pos[1] <= env.arena_size - env.obstacle_radius_max
    # Radii within the requested range.
    assert (env._obstacle_r >= env.obstacle_radius_min).all()
    assert (env._obstacle_r <= env.obstacle_radius_max).all()


def test_stage4_spawn_positions_clear_of_obstacles():
    env = _stage4_env()
    env.reset(seed=0)
    # Every blue and red position is outside every obstacle by at least
    # the requested spawn clearance.
    for pos in np.concatenate([env._blue_pos, env._red_pos], axis=0):
        for op, orad in zip(env._obstacle_pos, env._obstacle_r):
            assert float(np.linalg.norm(pos - op)) >= orad + env.obstacle_spawn_clearance, (
                f"Entity at {pos} too close to obstacle at {op} (r={orad})"
            )


def test_stage4_blue_cannot_move_into_obstacle():
    env = _stage4_env()
    env.reset(seed=0)
    # Force blue_0 to sit near an obstacle and try to move straight
    # into it.  Position should be rolled back and velocity zeroed.
    obs_pos = env._obstacle_pos[0]
    obs_r   = env._obstacle_r[0]
    # Place blue_0 just outside the obstacle boundary along +x.
    env._blue_pos[0] = obs_pos + np.array([obs_r + 1.0, 0.0], dtype=np.float32)
    env._blue_vel[0] = np.zeros(2, dtype=np.float32)
    prev_pos = env._blue_pos[0].copy()
    # Aggressive left action → would step into the obstacle.
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    actions["blue_0"] = np.array([-1.0, 0.0], dtype=np.float32)
    for _ in range(5):
        env.step(actions)
        # Blue should never end up inside an obstacle.
        for op, orad in zip(env._obstacle_pos, env._obstacle_r):
            d = float(np.linalg.norm(env._blue_pos[0] - op))
            assert d >= orad - 1e-3, (
                f"blue_0 penetrated obstacle: dist={d:.3f}, r={orad:.3f}"
            )


def test_stage4_belief_maps_shape_and_no_nans_after_200_steps():
    env = _stage4_env(max_steps=250)
    env.reset(seed=0)
    # v6.1: ONE global fused map (C, H, W).
    assert env._belief_maps.shape == (2, 26, 26)
    for _ in range(200):
        if not env.agents:
            env.reset(seed=1)
        actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
        env.step(actions)
    assert not np.isnan(env._belief_maps).any()
    assert not np.isinf(env._belief_maps).any()
    assert (env._belief_maps <= env.belief_clip + 1e-3).all()
    assert (env._belief_maps >= -env.belief_clip - 1e-3).all()


def test_stage4_occluded_cells_never_updated():
    """
    Place a blue with an obstacle directly between it and a candidate
    cell.  The occluded cell's log-odds MUST stay at 0 regardless of
    how many sensor updates fire.
    """
    env = _stage4_env()
    env.reset(seed=0)
    cs = env.belief_cell_size
    # Force a known obstacle geometry: single disk at (65, 65) r=10.
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r   = np.array([10.0],        dtype=np.float32)
    # Blue at (45, 65).  Pick an occluded cell behind the obstacle at
    # (82, 65) — directly through the obstacle centre.
    env._blue_pos[0] = np.array([45.0, 65.0], dtype=np.float32)
    occ_x, occ_y = 82.0, 65.0
    cxi_occ = int(occ_x / cs)
    cyi_occ = int(occ_y / cs)
    # A near cell at (48, 65) — in front of the obstacle, unoccluded.
    free_x, free_y = 48.0, 65.0
    cxi_free = int(free_x / cs)
    cyi_free = int(free_y / cs)

    # Move the OTHER blues far away so only blue_0's sensor is in play
    # (the map is global — any blue with line-of-sight would add
    # evidence to the "occluded" cell and defeat the test).
    for b in range(1, env.n_blue):
        env._blue_pos[b] = np.array([2.0, 2.0], dtype=np.float32)

    env._belief_maps[:] = 0.0
    for _ in range(20):
        env._update_belief_maps()

    assert env._belief_maps[0, cxi_occ, cyi_occ] == 0.0, (
        f"Occluded cell got updated: log-odds = "
        f"{env._belief_maps[0, cxi_occ, cyi_occ]}"
    )
    assert env._belief_maps[0, cxi_free, cyi_free] != 0.0, (
        "Un-occluded cell should have accumulated evidence"
    )


def test_stage4_caught_red_decays_from_belief_map():
    """
    Once a red is caught, the true_occupancy channel drops it to 0.
    Subsequent observations then add negative log-odds evidence, so
    P(enemy) decays.
    """
    env = _stage4_env()
    env.reset(seed=0)
    # Place blue_0 and red_0 close so capture fires this step.
    env._blue_pos[0] = np.array([50.0, 50.0], dtype=np.float32)
    env._red_pos[0]  = np.array([52.0, 50.0], dtype=np.float32)
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    env.step(actions)
    assert not env._red_active[0], "red_0 should be caught after step"
    # Record the log-odds of the cell centred near red_0's last known
    # position.
    cs = env.belief_cell_size
    cxi = int(52.0 / cs)   # 10 for cs=5
    cyi = int(50.0 / cs)   # 10
    initial_L = float(env._belief_maps[0, cxi, cyi])
    # Run ~15 more steps; because the red is gone, the sensor now sees
    # "no enemy" at that cell most of the time -> log-odds should
    # decrease from ``initial_L``.  Enough noise means we assert
    # monotone-with-slack rather than strict monotone.
    final_L = initial_L
    for _ in range(15):
        env.step(actions)
        final_L = float(env._belief_maps[0, cxi, cyi])
    assert final_L < initial_L, (
        f"Caught red's cell log-odds should decay: initial={initial_L:.3f} "
        f"final={final_L:.3f}"
    )


def test_stage4_true_occupancy_matches_ground_truth():
    env = _stage4_env()
    env.reset(seed=0)
    truth = env._true_occupancy()
    assert truth.shape == (2, 26, 26)
    # Enemy channel: every active red maps to exactly one cell = 1.
    n_ones_enemy = int(truth[0].sum())
    assert n_ones_enemy == int(env._red_active.sum())
    # Obstacle channel: at least a few cells are marked (obstacle
    # centres and near-centres).
    assert int(truth[1].sum()) > 0


def test_stage4_v61_global_map_fuses_multi_uav_evidence():
    """
    v6.1: the belief map is GLOBAL — evidence from different UAVs
    accumulates in the same tensor.  A cell visible to two UAVs should
    accumulate roughly twice the evidence magnitude per step of a cell
    visible to one (log-odds fusion is additive).
    """
    # Low sensor noise (0.99/0.01 -> |L| ~ 4.6, still under the ±10
    # clip for a double observation... actually 2x4.6=9.2 < 10) so a
    # single false positive is very unlikely and magnitudes stay
    # distinguishable.
    env = _stage4_env(n_obstacles=0, p_TP=0.99, p_FP=0.01)
    env.reset(seed=0)
    # Two blues staring at the same empty cell; the rest far away.
    env._blue_pos[0] = np.array([40.0, 65.0], dtype=np.float32)
    env._blue_pos[1] = np.array([90.0, 65.0], dtype=np.float32)
    for b in range(2, env.n_blue):
        env._blue_pos[b] = np.array([2.0, 2.0], dtype=np.float32)
    # Move reds away so the watched cell is empty (evidence: no enemy).
    env._red_pos[:] = np.array([5.0, 5.0], dtype=np.float32)

    cs = env.belief_cell_size
    cxi = int(65.0 / cs)
    cyi = int(65.0 / cs)     # cell (13, 13) at (65+2.5, 65+2.5) centre
    env._belief_maps[:] = 0.0
    env._update_belief_maps()
    # Cell (65, 65) is ~25m from both watchers -> both contribute.
    both = float(env._belief_maps[0, cxi, cyi])
    # A cell only ONE watcher can see: near (40, 65), ~50m from blue 1.
    cxi_one = int(38.0 / cs)
    one = float(env._belief_maps[0, cxi_one, cyi])
    assert both < 0.0 and one < 0.0, "empty cells should get negative evidence"
    assert abs(both) > 1.5 * abs(one), (
        f"doubly-observed cell should have ~2x evidence: both={both:.2f} "
        f"one={one:.2f}"
    )


def test_stage4_v6_obs_schema_with_obstacles():
    """
    v6 typed-graph schema: belief-derived actor node/edge features +
    ground-truth ``true_*`` critic features + obstacle variants.
    """
    env = _stage4_env(n_obstacles=4)
    env.reset(seed=0)
    for _ in range(10):
        env._update_belief_maps()
    obs = env.structured_belief_observation()
    n_rb = env.n_rb_edges
    n_ob = env.n_obstacles * env.n_blue
    expected = {
        "blue_features", "bb_edge_features", "bb_edge_visible",
        "red_features", "rb_edge_features", "rb_edge_visible",
        "true_red_features", "true_rb_edge_features",
        "obstacle_features", "ob_edge_features", "ob_edge_visible",
        "true_obstacle_features", "true_ob_edge_features",
        "belief_maps", "true_occupancy",
    }
    assert set(obs.keys()) == expected
    assert obs["blue_features"].shape         == (5, 8)
    assert obs["red_features"].shape          == (env.n_red, 1)
    assert obs["rb_edge_features"].shape      == (n_rb, 7)
    assert obs["rb_edge_visible"].shape       == (n_rb,)
    assert obs["true_rb_edge_features"].shape == (n_rb, 7)
    assert obs["obstacle_features"].shape     == (env.n_obstacles, 1)
    assert obs["ob_edge_features"].shape      == (n_ob, 7)
    assert obs["ob_edge_visible"].shape       == (n_ob,)
    assert obs["true_ob_edge_features"].shape == (n_ob, 7)


def test_stage4_v6_no_obstacle_keys_when_disabled():
    """No obstacle node/edge keys when n_obstacles == 0."""
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    obs = env.structured_belief_observation()
    for k in ("obstacle_features", "ob_edge_features", "ob_edge_visible",
              "true_obstacle_features", "true_ob_edge_features"):
        assert k not in obs


def test_stage4_v6_rb_position_comes_from_belief_when_unseen():
    """
    For a target NO blue can currently see, the rb_edge geometry is
    reconstructed from the belief-map PEAK (memory path).  Edge layout:
    [rel_pos(2), rel_vel(2), range(1), bearing(2)];
    rel_pos = blue_pos - peak_pos (normalised by L).
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    cs = env.belief_cell_size
    L = env.arena_size
    # Exactly one active red, and EVERY blue far from it (> 40 m) so it
    # is a MEMORY track (belief peak), not a live detection.
    env._red_active[:] = False
    env._red_active[0] = True
    env._red_pos[0] = np.array([120.0, 120.0], dtype=np.float32)
    for b in range(env.n_blue):
        env._blue_pos[b] = np.array([10.0, 10.0 + 3.0 * b], dtype=np.float32)
    env._belief_maps[:] = -5.0
    env._belief_maps[0, 20, 20] = 8.0   # dominant peak, cell centre (102.5, 102.5)
    obs = env.structured_belief_observation()
    rb = obs["rb_edge_features"]             # (n_rb, 7)
    peak = np.array([(20 + 0.5) * cs, (20 + 0.5) * cs], dtype=np.float32)
    expected_relpos = (env._blue_pos[0] - peak) / L
    # Edge (s=0, b=0) is index 0 in the (s outer, b inner) ordering.
    assert np.allclose(rb[0, :2], expected_relpos, atol=1e-3), (
        f"rb rel_pos {rb[0, :2]} != belief-peak-derived {expected_relpos}"
    )


def test_stage4_shared_position_own_doppler_velocity():
    """
    For a LIVE (detection-seeded) track, POSITION is the ONE shared/
    fused track position for every blue; the only per-blue difference is
    VELOCITY (own-sensor Doppler when the red is visible, else 0).
    - the blue that SEES the red gets the shared position + Doppler;
    - a blue OUT of range gets the SAME shared position (over TDL) but
      ZERO velocity.
    With sigma=0 the shared position equals the true red position
    exactly, at sub-cell precision; neither blue uses the cell peak.
    """
    env = _stage4_env(n_obstacles=0, sensor_pos_noise_std=0.0)
    env.reset(seed=0)
    L = env.arena_size
    env._red_active[:] = False
    env._red_active[0] = True
    env._red_pos[0] = np.array([69.3, 67.5], dtype=np.float32)   # off cell centre
    env._red_vel[0] = np.array([0.3, -0.2], dtype=np.float32)
    env._blue_pos[0] = np.array([60.0, 60.0], dtype=np.float32)  # 10 m -> sees it
    env._blue_pos[1] = np.array([5.0, 5.0], dtype=np.float32)    # ~90 m -> TDL only
    env._blue_vel[:] = 0.0
    env._belief_maps[:] = -5.0
    env._belief_maps[0, 13, 13] = 8.0    # residual blob (should be ignored)

    obs = env.structured_belief_observation()
    rb = obs["rb_edge_features"]
    N = env.n_blue
    v_max = 1.5   # BLUE_UAV.v_max (edge rel_vel normaliser)
    e0 = 0 * N + 0    # track 0 -> blue 0 (sees red 0)
    e1 = 0 * N + 1    # track 0 -> blue 1 (TDL only)

    # Both blues get the SHARED position (= true red pos at sigma=0).
    assert np.allclose(rb[e0, :2], (env._blue_pos[0] - env._red_pos[0]) / L, atol=1e-4)
    assert np.allclose(rb[e1, :2], (env._blue_pos[1] - env._red_pos[0]) / L, atol=1e-4), (
        "out-of-range blue gets the same shared track position over TDL"
    )
    # Velocity is the ONLY difference: seeing blue = Doppler, other = 0.
    # rel_vel = blue_vel - red_vel = -red_vel (blues stationary).
    assert np.allclose(rb[e0, 2:4], (-env._red_vel[0]) / v_max, atol=1e-4), (
        "seeing blue should get own-sensor Doppler velocity"
    )
    assert np.allclose(rb[e1, 2:4], 0.0, atol=1e-6), (
        "out-of-range blue has no own Doppler -> zero velocity"
    )


def test_stage4_v6_belief_rb_differs_from_true_rb():
    """
    Belief-derived rb edges (noisy) should generally differ from the
    ground-truth ``true_rb_edge_features`` the critic sees.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    for _ in range(5):
        env._update_belief_maps()
    obs = env.structured_belief_observation()
    assert not np.allclose(
        obs["rb_edge_features"], obs["true_rb_edge_features"],
    ), "belief rb should not exactly match ground-truth rb"


def test_stage4_v6_obstacle_edges_static_velocity():
    """
    Obstacle edges are rb-equivalent with the object velocity zeroed:
    rel_vel = blue_vel - 0.  With blues stationary, ob rel_vel == 0.
    """
    env = _stage4_env(n_obstacles=4)
    env.reset(seed=0)
    for _ in range(5):
        env._update_belief_maps()
    env._blue_vel[:] = 0.0
    obs = env.structured_belief_observation()
    ob = obs["ob_edge_features"]             # (n_ob, 7)
    assert np.allclose(ob[:, 2:4], 0.0), (
        "obstacle rel_vel should be 0 when blues are stationary "
        "(object velocity is 0)"
    )


def test_stage4_phaseA_off_by_default():
    """
    With the env defaults (decay=1.0, diffusion=0.0) the prediction
    step must be a byte-exact no-op — pre-Phase-A behaviour preserved.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    env._belief_maps[0, 10, 10] = 6.0
    env._belief_maps[0, 5, 20]  = -4.0
    before = env._belief_maps.copy()
    env._predict_enemy_belief()
    assert np.array_equal(env._belief_maps, before)


def test_stage4_phaseA_decay_fades_stale_evidence():
    """
    Pure decay (no diffusion): unobserved evidence of BOTH signs must
    fade geometrically toward log-odds 0 ("unknown").
    """
    env = _stage4_env(n_obstacles=0, enemy_belief_decay=0.97,
                      enemy_belief_diffusion=0.0)
    env.reset(seed=0)
    # Park every blue in the far corner so the planted cells are
    # outside all sensor disks (no evidence updates on them).
    for b in range(env.n_blue):
        env._blue_pos[b] = np.array([3.0, 3.0], dtype=np.float32)
    env._belief_maps[:] = 0.0
    env._belief_maps[0, 25, 25] = 6.0     # stale ghost, ~(127.5, 127.5)
    env._belief_maps[0, 20, 25] = -8.0    # stale "cleared" region

    for _ in range(10):
        env._update_belief_maps()

    ghost   = float(env._belief_maps[0, 25, 25])
    cleared = float(env._belief_maps[0, 20, 25])
    assert ghost == pytest.approx(6.0 * 0.97 ** 10, rel=1e-4), (
        f"ghost decayed to {ghost}, expected {6.0 * 0.97**10:.3f}"
    )
    assert cleared == pytest.approx(-8.0 * 0.97 ** 10, rel=1e-4)
    # Direction sanity: both moved TOWARD 0, neither crossed it.
    assert 0.0 < ghost < 6.0
    assert -8.0 < cleared < 0.0


def test_stage4_phaseA_diffusion_spreads_peak():
    """
    Pure diffusion: one prediction pass must lower an isolated peak
    and raise its neighbours above the background (mass spreads),
    conserving total probability for an interior peak on a uniform
    background.
    """
    env = _stage4_env(n_obstacles=0, enemy_belief_decay=1.0,
                      enemy_belief_diffusion=0.4)
    env.reset(seed=0)
    env._belief_maps[:] = 0.0             # uniform background P = 0.5
    env._belief_maps[0, 13, 13] = 6.0     # isolated interior peak

    P_before = 1.0 / (1.0 + np.exp(-env._belief_maps[0]))
    env._predict_enemy_belief()
    P_after = 1.0 / (1.0 + np.exp(-env._belief_maps[0]))

    # Peak fell, orthogonal neighbour rose above background.
    assert env._belief_maps[0, 13, 13] < 6.0
    assert env._belief_maps[0, 12, 13] > 0.0
    # Approximate mass conservation (interior peak, uniform borders).
    assert abs(P_before.sum() - P_after.sum()) < 1e-3


def test_stage4_belief_track_error_small_when_peak_on_red():
    """
    Plant a dominant peak at a red's true cell -> track error must be
    sub-cell (< 5 m at 5 m cells).
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    cs = env.belief_cell_size
    # Space the reds well apart so the extractor's NMS (2-cell radius)
    # keeps one track per target.
    env._red_pos = np.array(
        [[20.0, 20.0], [65.0, 110.0], [110.0, 30.0]], dtype=np.float32,
    )
    env._belief_maps[0][:] = -5.0
    # One dominant peak per red — every slot must correspond to a real
    # track for the error to be sub-cell.
    for r in range(env.n_red):
        pos = env._red_pos[r]
        env._belief_maps[0, int(pos[0] / cs), int(pos[1] / cs)] = 8.0
    err = env.belief_track_error()
    assert not np.isnan(err)
    assert err < 5.0, f"track error {err:.2f} m should be sub-cell"


def test_stage4_peak_nms_separates_blobs():
    """
    One strong DIFFUSE blob must yield exactly ONE track; the second
    slot must go to a distant weaker peak, not to the strong blob's
    neighbour cell.  This was the "clustering" bug: without NMS the
    top-K cells all came from the same blob and masked other targets.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    env._belief_maps[0][:] = -5.0
    # Strong blob around (10, 10): centre + neighbours all higher than
    # the distant weak peak.
    env._belief_maps[0, 10, 10] = 8.0
    env._belief_maps[0, 10, 11] = 7.5
    env._belief_maps[0, 11, 10] = 7.4
    # Distant weaker target.
    env._belief_maps[0, 20, 20] = 4.0

    peak_pos, conf = env._extract_belief_peaks(
        2, channel_idx=0, k_extract=2, nms_radius_cells=2,
    )
    cs = env.belief_cell_size
    cells = {(int(p[0] / cs), int(p[1] / cs)) for p in peak_pos}
    assert (10, 10) in cells, "strongest blob centre must be track 0"
    assert (20, 20) in cells, (
        f"2nd track must be the distant target, got cells {cells} — "
        "NMS failed to suppress the strong blob's neighbours"
    )


def test_stage4_dead_reds_free_no_track_slots():
    """
    After a capture, only n_active tracks are extracted; the dead
    slot must be conf-0 padded with zeroed edges and zero visibility.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    env._red_active[0] = False           # simulate one captured red
    obs = env.structured_belief_observation()
    conf = obs["red_features"][:, 0]     # (n_red,) node feat = conf
    n_real = int((conf > 0.0).sum())
    assert n_real == 2, f"expected 2 live tracks, got {n_real}"
    # The padded slot is the LAST one (real peaks sorted desc first).
    N = env.n_blue
    dead = 2                              # slot index of the padded track
    assert np.all(obs["rb_edge_visible"][dead * N:(dead + 1) * N] == 0.0)
    assert np.all(obs["rb_edge_features"][dead * N:(dead + 1) * N] == 0.0)


def test_stage4_capture_wipes_dead_targets_belief():
    """
    A capture must reset the enemy belief around the capture point to
    log-odds 0 — the dead target's stale blob cannot re-capture a
    track slot.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    cs = env.belief_cell_size
    # Engineer a certain capture: blue 0 on top of red 0.
    env._red_pos[0]  = np.array([65.0, 65.0], dtype=np.float32)
    env._blue_pos[0] = np.array([65.5, 65.0], dtype=np.float32)
    cxi = int(65.0 / cs)
    cyi = int(65.0 / cs)
    env._belief_maps[0, cxi, cyi] = 9.0   # strong stale blob at the kill
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    env.step(actions)
    assert not env._red_active[0], "red 0 should have been captured"
    # The blob is gone (0 before this step's sensor pass could only
    # have re-added bounded evidence; assert well below the planted 9).
    assert env._belief_maps[0, cxi, cyi] < 3.0, (
        f"stale blob survived the capture wipe: "
        f"L={env._belief_maps[0, cxi, cyi]:.2f}"
    )


def test_stage4_detection_seeded_two_close_visible_reds_split():
    """
    Two visible reds within one NMS radius (~one cell apart) must get
    TWO distinct live track slots — the belief-map NMS collapse that
    would have merged them is bypassed because live tracks are seeded
    directly from the sensor, not from belief peaks.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    env._red_active[:] = False
    env._red_active[:2] = True
    # ~4 m apart (< one 5 m cell) and both within blue 0's 40 m sensor.
    env._red_pos[0] = np.array([64.0, 65.0], dtype=np.float32)
    env._red_pos[1] = np.array([68.0, 65.0], dtype=np.float32)
    env._blue_pos[0] = np.array([65.0, 60.0], dtype=np.float32)
    track_pos, track_conf, track_red = env._build_enemy_tracks()
    live = track_red[track_red >= 0]
    assert set(live.tolist()) == {0, 1}, (
        f"two close visible reds must be two live tracks, got {track_red}"
    )
    assert (track_conf[:2] == 1.0).all()


def test_stage4_detection_seed_beats_belief_lag():
    """
    A visible red with NO belief peak yet (belief map all-negative)
    still gets a LIVE precise track — detection does not wait on the
    memory layer.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    env._red_active[:] = False
    env._red_active[0] = True
    env._red_pos[0]  = np.array([65.0, 65.0], dtype=np.float32)
    env._blue_pos[0] = np.array([60.0, 60.0], dtype=np.float32)   # in range
    env._belief_maps[:] = -8.0        # belief has NOT tracked red 0 yet
    track_pos, track_conf, track_red = env._build_enemy_tracks()
    assert track_red[0] == 0, "visible red should be a live track despite belief lag"
    assert track_conf[0] == 1.0


def test_stage4_unseen_red_is_memory_track():
    """A red no blue can see -> memory slot (track_red -1) from a belief peak."""
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    cs = env.belief_cell_size
    env._red_active[:] = False
    env._red_active[0] = True
    # Red far from every blue (> 40 m).
    env._red_pos[0] = np.array([120.0, 120.0], dtype=np.float32)
    for b in range(env.n_blue):
        env._blue_pos[b] = np.array([10.0, 10.0], dtype=np.float32)
    env._belief_maps[:] = -5.0
    env._belief_maps[0, 22, 22] = 6.0    # memory peak
    track_pos, track_conf, track_red = env._build_enemy_tracks()
    assert track_red[0] == -1, "unseen red must be a memory track"
    assert track_conf[0] > 0.5
    peak = np.array([(22 + 0.5) * cs, (22 + 0.5) * cs], dtype=np.float32)
    assert np.allclose(track_pos[0], peak, atol=1e-3)


def test_stage4_live_track_excludes_memory_duplicate():
    """
    A visible red's residual belief blob must NOT also spawn a memory
    track: with 1 visible + 1 unseen red, the visible red's cell is
    excluded from memory-peak extraction.
    """
    env = _stage4_env(n_obstacles=0)
    env.reset(seed=0)
    cs = env.belief_cell_size
    env._red_active[:] = False
    env._red_active[:2] = True
    env._red_pos[0]  = np.array([65.0, 65.0], dtype=np.float32)   # visible
    env._blue_pos[0] = np.array([64.0, 64.0], dtype=np.float32)
    env._red_pos[1]  = np.array([120.0, 120.0], dtype=np.float32) # unseen
    for b in range(1, env.n_blue):
        env._blue_pos[b] = np.array([10.0, 10.0], dtype=np.float32)
    env._belief_maps[:] = -5.0
    env._belief_maps[0, 13, 13] = 9.0    # strong blob at the VISIBLE red's cell
    env._belief_maps[0, 22, 22] = 5.0    # weaker blob at the unseen red
    track_pos, track_conf, track_red = env._build_enemy_tracks()
    # Slot 0: live red 0.  Slot 1: memory, and it must be the (22,22)
    # peak, NOT the excluded (13,13) blob.
    assert track_red[0] == 0
    assert track_red[1] == -1
    mem_cell = (int(track_pos[1, 0] / cs), int(track_pos[1, 1] / cs))
    assert mem_cell == (22, 22), (
        f"memory track landed on {mem_cell}; the live red's blob at "
        "(13,13) was not excluded"
    )


def test_stage4_v61_occlusion_exact_catches_grazing_chord():
    """
    The analytic segment-disk test must flag a GRAZING ray whose chord
    through the obstacle (~2.0 m here) is SHORTER than the old 2.5 m
    sample spacing — the case the sampled ray-march could miss.
    """
    env = _stage4_env()
    env.reset(seed=0)
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r   = np.array([10.0],        dtype=np.float32)
    uav = np.array([45.0, 55.05], dtype=np.float32)
    cells = np.array([
        [85.0, 55.05],   # grazing: perp 9.95 m < r=10 -> chord ~2.0 m
        [85.0, 50.0],    # passes ~12.4 m from centre -> clear
    ], dtype=np.float32)
    occ = env._rays_occluded_by_obstacles(uav, cells)
    assert bool(occ[0]), "grazing chord shorter than 2.5 m must occlude"
    assert not bool(occ[1]), "ray well clear of the disk must not occlude"


def test_stage4_backward_compat_when_flags_off():
    """
    Stage 1/2/3 behaviour must be byte-preserved when
    ``n_obstacles=0`` and ``use_belief_maps=False``.
    """
    env = PursuitEnv(n_blue=5, n_red=3, arena_size=130.0,
                     sensor_radius=40.0, seed=0)
    env.reset(seed=0)
    # No obstacles, no belief maps.
    assert env._obstacle_pos is not None   # populated as empty array
    assert env._obstacle_pos.shape == (0, 2)
    assert env._belief_maps is None
    # The base structured obs still works and carries no Stage 4 keys.
    obs = env.structured_observation()
    assert "belief_maps" not in obs
    assert "obstacle_positions" not in obs
    assert "true_occupancy" not in obs
