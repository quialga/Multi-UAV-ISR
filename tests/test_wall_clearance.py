"""
tests/test_wall_clearance.py — the arena walls get a barrier too.

Obstacles and allies were shaped and walls were not, which taught the policy
that touching a wall is free -- and mechanically it is: ``_integrate`` clips
the position to the wall and zeroes that velocity component, so contact costs
only momentum.  Measured on ``cnt_attention_ppo``: 0.53% of agent-steps in
contact against the uncoordinated heuristic's 0.08%, i.e. ~5 agent-steps per
episode.  Noise for score; a failure rate for an indoor certification trial.

ONE term rather than the obstacle path's crash + clearance pair, because a
blue cannot be INSIDE a wall: the depth saturates at 1 exactly at contact, so
the penalty there is the weight itself and a separate flat term would be
redundant.

What must hold:

* off by default, or every result in docs/stage4_results.md moves;
* the penalty at contact is EXACTLY the weight -- that is the contract that
  makes one knob enough;
* it ramps from zero at the margin and does not grow past contact;
* it is per-agent, and keyed on the NEAREST wall, not summed over four;
* it survives the checkpoint round trip.

Run:
    pytest tests/test_wall_clearance.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.policy_loader import env_kwargs_from_checkpoint
from isr.env.pursuit_env import PursuitEnv
from scripts.train_stage4 import saved_args_from_flags

L = 100.0
BASE = dict(n_blue=2, n_red=1, n_obstacles=0, arena_size=L, max_steps=10,
            capture_radius=3.0, sensor_radius=40.0)
W, M = 0.5, 2.0


def _reward_at(positions, weight=W, margin=M):
    """Step once with the blues parked at `positions`; return their rewards."""
    env = PursuitEnv(**BASE, wall_clearance_weight=weight,
                     wall_clearance_margin=margin, seed=0)
    env.reset(seed=0)
    env._blue_pos = np.asarray(positions, dtype=np.float32)
    _o, rew, _t, _tr, _i = env.step(
        {a: np.zeros(2, dtype=np.float32) for a in env.agents})
    return np.array([rew[a] for a in env.possible_agents])


MID = [L / 2, L / 2]        # far from every wall: the reference


def test_off_by_default():
    env = PursuitEnv(**BASE, seed=0)
    assert env.wall_clearance_weight == 0.0
    r = _reward_at([[0.0, L / 2], MID], weight=0.0)
    assert np.allclose(r[0], r[1]), "a wall cost something with the term off"


def test_the_penalty_at_contact_is_exactly_the_weight():
    """The contract that makes a single knob sufficient."""
    r = _reward_at([[0.0, L / 2], MID])
    assert np.isclose(r[1] - r[0], W, atol=1e-5), f"got {r[1] - r[0]}, want {W}"


def test_it_ramps_linearly_and_is_zero_at_the_margin():
    ref = _reward_at([MID, MID])[0]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = M * (1.0 - frac)                     # frac = depth into the band
        got = ref - _reward_at([[d, L / 2], MID])[0]
        assert np.isclose(got, W * frac, atol=1e-5), (
            f"at {d:.2f} m from the wall expected {W * frac:.3f}, got {got:.3f}")


def test_it_does_not_grow_past_contact():
    """Unlike the obstacle term, there is no 'inside' to keep penalising."""
    at_wall = _reward_at([[0.0, L / 2], MID])[0]
    # The integrator clips, so a blue cannot actually be outside; the depth
    # clamp is what guarantees the penalty cannot exceed the weight.
    assert np.isclose(_reward_at([MID, MID])[0] - at_wall, W, atol=1e-5)


def test_it_is_per_agent():
    r = _reward_at([[0.0, L / 2], MID])
    assert r[0] < r[1], "the blue in the middle was penalised too"


def test_it_uses_the_NEAREST_wall_not_a_sum_over_four():
    """A blue in a corner touches two walls.  Summing would double the
    penalty; the nearest-wall rule caps it at the weight."""
    corner = _reward_at([[0.0, 0.0], MID])
    edge = _reward_at([[0.0, L / 2], MID])
    assert np.isclose(corner[0], edge[0], atol=1e-5), (
        f"corner {corner[0]:.3f} differs from edge {edge[0]:.3f} -- "
        f"the penalty is being summed over walls")


def test_all_four_walls_are_covered():
    ref = _reward_at([MID, MID])[0]
    for p in ([0.0, L / 2], [L, L / 2], [L / 2, 0.0], [L / 2, L]):
        assert np.isclose(ref - _reward_at([p, MID])[0], W, atol=1e-5), (
            f"wall at {p} not penalised")


def test_a_tighter_margin_penalises_a_smaller_band():
    """The reason the margin is tighter than the obstacle one: three bands at
    once leave little admissible space."""
    d = 3.0                                   # inside a 4 m band, outside a 2 m
    ref = _reward_at([MID, MID], margin=4.0)[0]
    wide = ref - _reward_at([[d, L / 2], MID], margin=4.0)[0]
    tight = ref - _reward_at([[d, L / 2], MID], margin=2.0)[0]
    assert wide > 0.0 and np.isclose(tight, 0.0, atol=1e-6)


def test_it_round_trips_through_a_checkpoint():
    saved = saved_args_from_flags(
        "--n-blue 2 --n-red 1 --n-obstacles 0 --sensor-radius 40 "
        "--arena-size 100 --max-steps 10 "
        "--wall-clearance-weight 0.5 --wall-clearance-margin 2.0")
    assert saved["wall_clearance_weight"] == 0.5
    kw = env_kwargs_from_checkpoint(saved)
    assert kw["wall_clearance_weight"] == 0.5
    assert kw["wall_clearance_margin"] == 2.0


def test_a_pre_flag_checkpoint_keeps_it_off():
    saved = saved_args_from_flags(
        "--n-blue 2 --n-red 1 --n-obstacles 0 --sensor-radius 40 "
        "--arena-size 100 --max-steps 10")
    del saved["wall_clearance_weight"]
    assert env_kwargs_from_checkpoint(saved)["wall_clearance_weight"] == 0.0
