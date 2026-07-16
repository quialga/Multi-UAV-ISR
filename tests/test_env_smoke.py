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
    """Default env is fully observable — masks all ones."""
    env = PursuitEnv(n_blue=3, n_red=2, seed=0)   # sensor_radius=None (default)
    env.reset(seed=0)
    obs = env.structured_partial_observation()
    assert set(obs.keys()) >= {"bb_edge_visible", "rb_edge_visible"}
    assert np.all(obs["bb_edge_visible"] == 1.0)
    assert np.all(obs["rb_edge_visible"] == 1.0)
    # bb_edge_features and rb_edge_features must still be identical to
    # the Stage 2 full obs (unchanged in partial mode; only the mask
    # is separate).
    full = env.structured_observation()
    assert np.allclose(obs["bb_edge_features"], full["bb_edge_features"])
    assert np.allclose(obs["rb_edge_features"], full["rb_edge_features"])


def test_visibility_masks_reflect_within_radius():
    """
    Place agents at known positions and verify the visibility masks
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

    obs = env.structured_partial_observation()
    bb_v = obs["bb_edge_visible"]
    rb_v = obs["rb_edge_visible"]

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
    obs = env.structured_partial_observation()
    bb_v = obs["bb_edge_visible"]
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
    obs = env.structured_partial_observation()
    for e in range(env.n_rb_edges):
        r = int(env.rb_edge_src[e])
        if r == 0:
            assert obs["rb_edge_visible"][e] == 0.0, (
                f"rb edge {e} from caught red {r} should be hidden"
            )
        else:
            assert obs["rb_edge_visible"][e] == 1.0, (
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
        use_belief_maps=True, belief_grid_size=26, belief_channels=4,
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
    assert env._belief_maps.shape == (5, 4, 26, 26)
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
    # Force a known obstacle geometry: single disk at (65, 65) r=10.
    env._obstacle_pos = np.array([[65.0, 65.0]], dtype=np.float32)
    env._obstacle_r   = np.array([10.0],        dtype=np.float32)
    # Blue at (45, 65).  Cell (16, 13) centres at (82.5, 67.5), dist
    # ~37.5 m < R=40.  The straight ray (45,65) → (82.5,67.5) passes
    # near (65, 66) which is inside the obstacle → occluded.
    env._blue_pos[0] = np.array([45.0, 65.0], dtype=np.float32)
    cxi_occ, cyi_occ = 16, 13
    # A near cell (10, 13) centred at (52.5, 67.5) is unobstructed.
    cxi_free, cyi_free = 10, 13

    env._belief_maps[:] = 0.0
    for _ in range(20):
        env._update_belief_maps()

    # Occluded cell: must be exactly 0.
    assert env._belief_maps[0, 0, cxi_occ, cyi_occ] == 0.0, (
        f"Occluded cell got updated: log-odds = "
        f"{env._belief_maps[0, 0, cxi_occ, cyi_occ]}"
    )
    # Un-occluded cell: after 20 updates, log-odds should be
    # measurably non-zero with high probability.
    assert env._belief_maps[0, 0, cxi_free, cyi_free] != 0.0, (
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
    initial_L = float(env._belief_maps[0, 0, cxi, cyi])
    # Run ~15 more steps; because the red is gone, the sensor now sees
    # "no enemy" at that cell most of the time -> log-odds should
    # decrease from ``initial_L``.  Enough noise means we assert
    # monotone-with-slack rather than strict monotone.
    final_L = initial_L
    for _ in range(15):
        env.step(actions)
        final_L = float(env._belief_maps[0, 0, cxi, cyi])
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


def test_stage4_ally_channel_is_deterministic_and_shared():
    """
    Channel 2 (ally positions) must equal 1.0 at every active blue's
    cell and be IDENTICAL across all UAVs' belief maps (all-blues
    view via TDL uplink).
    """
    env = _stage4_env()
    env.reset(seed=0)
    cs = env.belief_cell_size
    ally_ch = env._belief_maps[:, 2]                    # (N_blue, H, W)
    # All UAVs see the same ally channel.
    for i in range(1, env.n_blue):
        assert np.allclose(ally_ch[i], ally_ch[0]), (
            f"Ally channel differs for UAV {i} vs UAV 0"
        )
    # Every blue's cell is marked = 1.
    for i in range(env.n_blue):
        cx = int(env._blue_pos[i, 0] / cs)
        cy = int(env._blue_pos[i, 1] / cs)
        assert ally_ch[0, cx, cy] == 1.0, (
            f"Blue {i}'s cell ({cx},{cy}) not marked in ally channel"
        )


def test_stage4_self_channel_is_per_uav():
    """
    Channel 3 (self position) is DIFFERENT per UAV: only THIS UAV's
    cell is marked in its own belief map.
    """
    env = _stage4_env()
    env.reset(seed=0)
    cs = env.belief_cell_size
    self_ch = env._belief_maps[:, 3]                    # (N_blue, H, W)
    for i in range(env.n_blue):
        cx = int(env._blue_pos[i, 0] / cs)
        cy = int(env._blue_pos[i, 1] / cs)
        # UAV i sees a 1 at its own cell...
        assert self_ch[i, cx, cy] == 1.0, (
            f"UAV {i}'s own cell not marked in self channel"
        )
        # ...and no more than a single 1 in its self channel.
        assert self_ch[i].sum() == pytest.approx(1.0), (
            f"UAV {i}'s self channel has multiple non-zero cells"
        )


def test_stage4_deterministic_channels_unaffected_by_clip():
    """
    ``belief_clip`` should only apply to log-odds channels (0-1);
    the deterministic channels stay in {0, 1}.  Run 300 steps with
    all blues moving toward the same corner and confirm channels
    2-3 are still {0, 1}, not clipped weirdly.
    """
    env = _stage4_env(max_steps=350)
    env.reset(seed=0)
    actions = {a: np.array([1.0, 1.0], dtype=np.float32) for a in env.agents}
    for _ in range(300):
        if not env.agents:
            env.reset(seed=1)
        env.step(actions)
    ally_ch = env._belief_maps[:, 2]
    self_ch = env._belief_maps[:, 3]
    unique_ally = np.unique(ally_ch)
    unique_self = np.unique(self_ch)
    assert set(unique_ally.tolist()).issubset({0.0, 1.0}), (
        f"Ally channel has non-binary values: {unique_ally}"
    )
    assert set(unique_self.tolist()).issubset({0.0, 1.0}), (
        f"Self channel has non-binary values: {unique_self}"
    )


def test_stage4_obs_dict_schema():
    env = _stage4_env()
    env.reset(seed=0)
    obs = env.structured_belief_observation()
    expected_keys = {
        "blue_features", "bb_edge_features",
        "belief_maps", "obstacle_positions", "true_occupancy",
    }
    assert set(obs.keys()) == expected_keys
    assert obs["blue_features"].shape    == (5, 8)
    assert obs["bb_edge_features"].shape == (env.n_bb_edges, 7)
    assert obs["belief_maps"].shape      == (5, 4, 26, 26)
    assert obs["obstacle_positions"].shape[1] == 3
    assert obs["true_occupancy"].shape   == (2, 26, 26)
    # Ensure we did NOT accidentally leak Stage 3 keys.
    for stage3_key in ("red_features", "rb_edge_features",
                       "bb_edge_visible", "rb_edge_visible"):
        assert stage3_key not in obs


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
    # Stage 3 obs dict still works.
    obs = env.structured_partial_observation()
    assert "belief_maps" not in obs
    assert "obstacle_positions" not in obs
    assert "true_occupancy" not in obs
