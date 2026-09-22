"""
tests/test_observation_greedy.py — ``ObservationGreedyPursuer``.

``GreedyPursuer`` reads ``env.state_snapshot()``: TRUE red positions,
gated only by range.  It never misses a detection, never sees a false one
and its positions carry no error — the near-oracle in-range regime §4b
removed from the policy's inputs — so comparing a trained policy against
it mixes policy quality with an information gap.

``ObservationGreedyPursuer`` runs the same rule on the ACTOR's red nodes.
What must hold:

* it steers TOWARD the target, i.e. the rel_pos sign convention
  (rel_pos = dst - src, edges run red -> blue) is not inverted;
* it picks the NEAREST visible slot, not just any;
* it holds position when no slot is visible;
* it uses memory slots, which is the whole point of the fallback: a red
  the sensors cannot see right now still gets chased from belief memory;
* it builds the observation ONCE per timestep — calling it per blue would
  advance the belief map N_blue times per step;
* it works in both actor_obs modes, since the per-arm comparison is one
  of its two uses.

Run:
    pytest tests/test_observation_greedy.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.heuristics import ObservationGreedyPursuer, stationary_red
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
    """Put blues and reds where a test needs them.

    ``wipe_belief`` clears the belief map: reset() spawns the entities at
    random and the map already holds peaks from THOSE positions, so
    without wiping, the nearest 'track' can be a ghost of a spawn point
    rather than the red the test placed.
    """
    env._blue_pos = np.asarray(blue_pos, dtype=np.float32)
    env._blue_vel = np.zeros_like(env._blue_pos)
    env._red_pos = np.asarray(red_pos, dtype=np.float32)
    env._red_vel = np.zeros_like(env._red_pos)
    env._red_active[:] = True
    if wipe_belief and getattr(env, "_belief_maps", None) is not None:
        env._belief_maps[:] = 0.0


def test_steers_toward_the_target_not_away():
    """The sign trap: rel_pos is (blue - red), so the heading is its
    negation.  Getting it backwards would flee, and would still produce a
    plausible-looking unit vector."""
    env = _env()
    # blue_0 at the origin, a red 20 m due EAST (inside the 40 m sensor).
    _place(env, [[0.0, 0.0], [60.0, 60.0]], [[20.0, 0.0], [-60.0, -60.0]])

    a = ObservationGreedyPursuer().act(None, env, env.possible_agents[0])
    assert a[0] > 0.9, f"should accelerate east toward the red, got {a}"
    assert abs(a[1]) < 0.2, a
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5, "action should be a unit heading"


def test_picks_the_nearest_visible_slot():
    env = _env()
    # Two reds east of blue_0: one at 10 m, one at 30 m.  Both in range.
    _place(env, [[0.0, 0.0], [60.0, 60.0]], [[30.0, 0.0], [0.0, 10.0]])

    a = ObservationGreedyPursuer().act(None, env, env.possible_agents[0])
    # The 10 m red is due NORTH; the 30 m one due EAST.  Nearest wins.
    assert a[1] > 0.9, f"should chase the 10 m red to the north, got {a}"


def test_a_live_track_outranks_a_nearer_memory_peak():
    """Regression.  Ranking by distance alone made the rule chase ghosts:
    with the belief map wiped, the argmax peak lands in the corner cell
    (~2 m from a blue at the origin) and beat a real red 30 m away.  The
    mask is the confidence — 1.0 live, lower for a belief peak — so the
    most confident slots are picked first and distance only breaks ties
    among them."""
    env = _env()
    _place(env, [[0.0, 0.0], [60.0, 60.0]], [[30.0, 0.0], [-60.0, -60.0]])

    a = ObservationGreedyPursuer().act(None, env, env.possible_agents[0])
    assert a[0] > 0.9, f"should chase the live red east, not the corner ghost: {a}"
    assert abs(a[1]) < 0.2, a


def test_holds_position_when_nothing_is_visible():
    """Zero slots -> zero action, matching GreedyPursuer with nothing in
    range.  Tested in TRACKER mode, where a fresh env genuinely has no
    confirmed tracks: in belief mode this state is unreachable, because
    the belief map always offers memory peaks — which is the fallback the
    next test pins."""
    env = _env(mode="tracker")
    _place(env, [[0.0, 0.0], [5.0, 0.0]], [[125.0, 125.0], [125.0, 120.0]])

    a = ObservationGreedyPursuer().act(None, env, env.possible_agents[0])
    assert np.allclose(a, 0.0), a


def test_uses_memory_when_the_red_is_no_longer_visible():
    """The fallback that makes this more than a blind greedy: see a red,
    let it leave sensor range, and the belief-map memory slot still gives
    a heading."""
    env = _env()
    # Step 1: red well inside range, so the belief map registers it.
    _place(env, [[0.0, 0.0], [60.0, 60.0]], [[20.0, 0.0], [-60.0, -60.0]])
    env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})

    # Now move that red far out of every sensor radius.  Its memory peak
    # should survive and still point roughly east.
    _place(env, [[0.0, 0.0], [60.0, 60.0]], [[120.0, 0.0], [-60.0, -60.0]])
    a = ObservationGreedyPursuer().act(None, env, env.possible_agents[0])

    assert not np.allclose(a, 0.0), "memory slot should still give a heading"
    assert a[0] > 0.0, f"memory of an eastward red should point east, got {a}"


def test_observation_is_built_once_per_timestep():
    """Calling structured_belief_observation() per blue would advance the
    belief map N_blue times per step.  One agent instance, two blues, one
    build."""
    env = _env()
    _place(env, [[0.0, 0.0], [10.0, 0.0]], [[20.0, 0.0], [-60.0, -60.0]])

    calls = {"n": 0}
    real = env.structured_belief_observation

    def counting():
        calls["n"] += 1
        return real()

    env.structured_belief_observation = counting  # type: ignore[assignment]

    agent = ObservationGreedyPursuer()
    for who in env.possible_agents:
        agent.act(None, env, who)
    assert calls["n"] == 1, calls

    # A new timestep must rebuild it.
    env.structured_belief_observation = real      # type: ignore[assignment]
    env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
    env.structured_belief_observation = counting  # type: ignore[assignment]
    agent.act(None, env, env.possible_agents[0])
    assert calls["n"] == 2, calls


def test_runs_in_tracker_mode_too():
    """The per-arm comparison needs it to work on the tracker's confirmed
    tracks, not just the belief map."""
    env = _env(mode="tracker")
    _place(env, [[0.0, 0.0], [60.0, 60.0]], [[20.0, 0.0], [-60.0, -60.0]])
    # The tracker needs a few scans to CONFIRM a track (M-of-N).
    agent = ObservationGreedyPursuer()
    a = None
    for _ in range(6):
        env.step({x: np.zeros(2, dtype=np.float32) for x in env.agents})
        _place(env, [[0.0, 0.0], [60.0, 60.0]], [[20.0, 0.0], [-60.0, -60.0]])
        a = agent.act(None, env, env.possible_agents[0])

    assert a is not None and np.isfinite(a).all()
    assert a[0] > 0.5, f"confirmed track east of blue_0 should pull east, {a}"
