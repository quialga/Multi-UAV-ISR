"""
tests/test_assignment_greedy.py — ``AssignmentGreedyPursuer``.

The upper bound `docs/stage4_results.md` never had.  §10 measured an
uncoordinated rule at 2.90/3 and §12 measured PPO buying ~+0.20 past it
by de-clustering the team, but nothing measured how much coordination is
*available* — that was inferred, not observed.  This baseline is
`ObsGreedy` differing in exactly one respect (the team divides the
targets), so what it beats `ObsGreedy` by is what assignment alone is
worth.

Its value as a bound depends entirely on that "exactly one respect", so
these tests pin it:

* it really does split, where `ObsGreedy` really does pile on — the
  whole measurement is void if the two behave the same;
* it uses no information `ObsGreedy` lacks;
* it still holds the other tiers: confidence first, search last;
* one observation, one assignment, per timestep — a per-blue rebuild
  would re-draw the sensor and give each blue a different problem.

Run:
    pytest tests/test_assignment_greedy.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.heuristics import (AssignmentGreedyPursuer,
                                   ObservationGreedyPursuer, stationary_red)
from isr.env.pursuit_env import PursuitEnv

BASE = dict(n_blue=2, n_red=2, n_obstacles=0, arena_size=130.0, max_steps=60,
            capture_radius=3.0, sensor_radius=40.0, red_policy=stationary_red,
            track_detection=False, track_occlusion=False,
            sensor_pos_noise_std=0.0, sensor_vel_noise_std=0.0)


def _env(mode="belief", seed=0, **kw):
    cfg = dict(BASE, actor_obs=mode, use_belief_maps=(mode == "belief"),
               seed=seed)
    cfg.update(kw)
    e = PursuitEnv(**cfg)
    e.reset(seed=seed)
    return e


def _place(env, blue_pos, red_pos, wipe_belief=True):
    env._blue_pos = np.asarray(blue_pos, dtype=np.float32)
    env._blue_vel = np.zeros_like(env._blue_pos)
    env._red_pos = np.asarray(red_pos, dtype=np.float32)
    env._red_vel = np.zeros_like(env._red_pos)
    env._red_active[:] = True
    if wipe_belief and getattr(env, "_belief_maps", None) is not None:
        env._belief_maps[:] = 0.0


def test_it_splits_where_obsgreedy_piles_on():
    """The measurement is void unless the two differ.  Two blues close
    together, two reds at different distances: the uncoordinated rule
    sends BOTH at the nearer red; assignment sends one at each."""
    env = _env()
    # blues near the origin; reds 15 m east and 45 m east.
    _place(env, [[0.0, 0.0], [2.0, 0.0]], [[15.0, 0.0], [45.0, 0.0]])

    naive = ObservationGreedyPursuer()
    a_naive = [naive.act(None, env, a) for a in env.possible_agents]

    smart = AssignmentGreedyPursuer()
    a_smart = [smart.act(None, env, a) for a in env.possible_agents]

    # Both rules steer east here; what differs is the ASSIGNMENT, so
    # compare the targets chosen rather than the headings.
    assert len(set(smart._assign.values())) == 2, (
        f"assignment put both blues on one slot: {smart._assign}")
    naive_slots = set()
    feats = naive._obs["rb_edge_features"]
    vis = np.asarray(naive._obs["rb_edge_visible"])
    dst = np.arange(vis.shape[0]) % env.n_blue
    for b in range(env.n_blue):
        mine = np.nonzero((dst == b) & (vis > 0))[0]
        naive_slots.add(int(mine[int(np.argmin(feats[mine, 4]))]) // env.n_blue)
    assert len(naive_slots) == 1, (
        "test geometry failed to make the naive rule pile on")
    assert all(np.isfinite(a).all() for a in a_naive + a_smart)


def test_every_visible_target_gets_a_pursuer():
    """With at least as many blues as targets, nothing visible is
    ignored — the property assignment exists to provide."""
    env = _env(n_blue=3, n_red=3)
    _place(env, [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
           [[20.0, 0.0], [25.0, 5.0], [30.0, 10.0]])
    agent = AssignmentGreedyPursuer()
    for a in env.possible_agents:
        agent.act(None, env, a)
    slots = {e // env.n_blue for e in agent._assign.values()}
    assert len(slots) == 3, (agent._assign, slots)


def test_load_is_balanced_when_blues_outnumber_targets():
    """5 blues on 2 targets must come out 3/2, not 5/0."""
    env = _env(n_blue=5, n_red=2)
    _place(env, [[0.0, float(i) * 3.0] for i in range(5)],
           [[20.0, 0.0], [20.0, 30.0]])
    agent = AssignmentGreedyPursuer()
    for a in env.possible_agents:
        agent.act(None, env, a)
    slots = [e // env.n_blue for e in agent._assign.values()]
    counts = sorted(np.bincount(np.array(slots), minlength=2))
    assert counts == [2, 3], (agent._assign, counts)


def test_it_searches_when_nothing_is_trackable():
    """The third tier is inherited, not lost."""
    env = _env(mode="tracker", use_staleness=True, staleness_regions=5)
    _place(env, [[0.0, 0.0], [5.0, 0.0]], [[125.0, 125.0], [125.0, 120.0]])
    a = AssignmentGreedyPursuer().act(None, env, env.possible_agents[0])
    assert not np.allclose(a, 0.0), "should search, not hold"
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5


def test_it_holds_when_there_is_nothing_at_all():
    env = _env(mode="tracker")
    _place(env, [[0.0, 0.0], [5.0, 0.0]], [[125.0, 125.0], [125.0, 120.0]])
    a = AssignmentGreedyPursuer().act(None, env, env.possible_agents[0])
    assert np.allclose(a, 0.0), a


def test_one_observation_and_one_assignment_per_timestep():
    """Rebuilding per blue would re-draw the sensor, so each blue would be
    solving a different assignment problem — which is not an assignment."""
    env = _env()
    _place(env, [[0.0, 0.0], [10.0, 0.0]], [[20.0, 0.0], [40.0, 10.0]])
    calls = {"n": 0}
    real = env.structured_belief_observation

    def counting():
        calls["n"] += 1
        return real()

    env.structured_belief_observation = counting  # type: ignore[assignment]
    agent = AssignmentGreedyPursuer()
    first = None
    for who in env.possible_agents:
        agent.act(None, env, who)
        if first is None:
            first = dict(agent._assign)
    assert calls["n"] == 1, calls
    assert agent._assign == first, "the assignment changed mid-timestep"


def test_it_reads_only_what_obsgreedy_reads():
    """Its standing as an upper bound depends on using no extra
    information: the same two arrays, plus the coverage keys for the
    search tier."""
    env = _env()
    _place(env, [[0.0, 0.0], [10.0, 0.0]], [[20.0, 0.0], [40.0, 10.0]])
    obs = env.structured_belief_observation()
    trimmed = {k: obs[k] for k in ("rb_edge_features", "rb_edge_visible")}
    assign = AssignmentGreedyPursuer._assign_team(trimmed, env.n_blue)
    assert len(assign) == env.n_blue, assign
