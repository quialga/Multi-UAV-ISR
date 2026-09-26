"""
tests/test_comms_radius.py — the blue-to-blue datalink range as its own knob.

``bb_edge_visible`` used to be gated by ``sensor_radius``, the same number
that decides whether a blue can DETECT A TARGET.  That conflated two
physically different channels: a bb edge is a radio message between our own
drones, and a datalink outranging an onboard sensor by an order of magnitude
is the normal case in real UAV teams.  ``comms_radius`` separates them
(docs/stage4_backlog.md §7).

What must hold:

* the default reproduces the old behaviour EXACTLY, or every result in
  docs/stage4_results.md silently moves;
* it gates bb edges and ONLY bb edges — rb visibility is sensing and must
  not move with the radio;
* ``inf`` really means every bb edge, at any separation;
* it survives the round trip through a checkpoint, since a policy trained
  with unlimited comms evaluated under gated comms is a silent regression.

Run:
    pytest tests/test_comms_radius.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.policy_loader import env_kwargs_from_checkpoint
from isr.env.pursuit_env import PursuitEnv
from scripts.train_stage4 import saved_args_from_flags

BASE = dict(n_blue=4, n_red=2, n_obstacles=0, arena_size=130.0,
            max_steps=50, capture_radius=3.0, sensor_radius=25.0)


def _masks(env: PursuitEnv):
    obs = env.structured_belief_observation()
    return obs["bb_edge_visible"], obs["rb_edge_visible"]


def _spread_out(env: PursuitEnv) -> None:
    """Park the blues at the four corners, so every pair is far apart.

    The arena diagonal is ~184 m against a sensor_radius of 25, so no pair
    is within sensing range and a sensor-gated mask must be all zeros.
    """
    L = env.arena_size
    env._blue_pos = np.array([[5.0, 5.0], [L - 5.0, 5.0],
                              [5.0, L - 5.0], [L - 5.0, L - 5.0]],
                             dtype=np.float32)


def test_default_reproduces_sensor_radius_gating():
    env = PursuitEnv(**BASE, seed=0)
    env.reset(seed=0)
    assert env.comms_radius == env.sensor_radius
    _spread_out(env)
    bb, _rb = _masks(env)
    assert bb.sum() == 0.0, "corners are 90+ m apart under a 25 m radius"


def test_inf_means_blues_always_talk():
    env = PursuitEnv(**BASE, comms_radius=float("inf"), seed=0)
    env.reset(seed=0)
    _spread_out(env)
    bb, _rb = _masks(env)
    assert bb.sum() == env.n_bb_edges, "every bb edge must be open"


def test_it_does_not_touch_rb_visibility():
    """Seeing a target is sensing; talking to a wingman is radio.

    Asserted against ``_compute_edge_visibility``, which owns the gating
    contract, NOT against the observation's ``rb_edge_visible``: in the
    belief path that key is derived from detections and reads all-zero at
    reset, so comparing it across two envs passes trivially and proves
    nothing.
    """
    gated = PursuitEnv(**BASE, seed=0)
    gated.reset(seed=0)
    open_ = PursuitEnv(**BASE, comms_radius=float("inf"), seed=0)
    open_.reset(seed=0)
    # Same geometry in both, so only the masks may differ.  One red sits
    # next to blue 0 and one across the arena, so rb is a genuine MIX of
    # ones and zeros -- an all-equal assertion on two constant arrays would
    # be the same trivial pass this test was written to avoid.
    for e in (gated, open_):
        _spread_out(e)
        e._red_pos = np.array([[8.0, 8.0], [120.0, 120.0]], dtype=np.float32)
    bb_g, rb_g = gated._compute_edge_visibility()
    bb_o, rb_o = open_._compute_edge_visibility()
    assert not np.array_equal(bb_g, bb_o), "bb must respond to the knob"
    assert 0.0 < rb_g.sum() < rb_g.size, f"rb must be a mix, got {rb_g}"
    np.testing.assert_array_equal(rb_g, rb_o)


def test_an_intermediate_radius_gates_by_distance():
    env = PursuitEnv(**BASE, comms_radius=60.0, seed=0)
    env.reset(seed=0)
    L = env.arena_size
    # Two pairs: one 40 m apart (inside 60), one across the arena (outside).
    env._blue_pos = np.array([[10.0, 10.0], [50.0, 10.0],
                              [L - 10.0, L - 10.0], [L - 50.0, L - 10.0]],
                             dtype=np.float32)
    bb, _rb = _masks(env)
    # 4 blues -> 12 directed edges; the two close pairs give 4 visible.
    assert bb.sum() == 4.0, f"expected the two near pairs, got {bb.sum()}"


def test_a_wider_radio_never_hides_an_edge_the_narrow_one_showed():
    """Monotonicity: opening the radio can only add edges."""
    narrow = PursuitEnv(**BASE, comms_radius=30.0, seed=1)
    wide = PursuitEnv(**BASE, comms_radius=90.0, seed=1)
    for e in (narrow, wide):
        e.reset(seed=1)
    np.testing.assert_array_equal(narrow._blue_pos, wide._blue_pos)
    bb_n, _ = _masks(narrow)
    bb_w, _ = _masks(wide)
    assert np.all(bb_w >= bb_n)


def test_it_round_trips_through_a_checkpoints_args():
    saved = saved_args_from_flags(
        "--n-blue 4 --n-red 2 --sensor-radius 25 --comms-radius inf "
        "--actor-obs tracker --use-staleness")
    assert np.isinf(saved["comms_radius"])
    kw = env_kwargs_from_checkpoint(saved)
    assert np.isinf(kw["comms_radius"])
    env = PursuitEnv(**kw, seed=0)
    assert np.isinf(env.comms_radius)


def test_pre_flag_checkpoints_keep_the_old_gating():
    """A checkpoint written before the flag existed must evaluate as trained."""
    saved = saved_args_from_flags(
        "--n-blue 4 --n-red 2 --sensor-radius 25 "
        "--actor-obs tracker --use-staleness")
    del saved["comms_radius"]                      # simulate an old checkpoint
    kw = env_kwargs_from_checkpoint(saved)
    env = PursuitEnv(**kw, seed=0)
    assert env.comms_radius == env.sensor_radius == 25.0


def test_full_observability_still_opens_everything():
    """sensor_radius None means fully observable; comms cannot narrow that.

    ``_compute_edge_visibility`` short-circuits to all-ones before it ever
    looks at ``comms_radius``, so this pins that the short-circuit stayed
    ahead of the new branch.
    """
    env = PursuitEnv(**dict(BASE, sensor_radius=None), seed=0)
    env.reset(seed=0)
    assert np.isinf(env.comms_radius)
    _spread_out(env)
    bb, rb = env._compute_edge_visibility()
    assert bb.sum() == env.n_bb_edges
    assert rb.sum() == env.n_rb_edges
