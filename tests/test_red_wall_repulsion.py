"""
tests/test_red_wall_repulsion.py — verifies the wall-repulsion fix for
the scripted red evader ``run_from_nearest_uav``.

Problem it fixes: a red fleeing straight away from its nearest blue runs
into the arena wall, has its perpendicular velocity clipped, and slides
ALONG the wall — pinning itself against the boundary.  Blue then learns a
degenerate wall-trapping counter (observed in the simulator: both reds and
blues hugging walls).  The red heuristic already repels off OBSTACLES for
exactly this reason ("trivially cornerable — a strawman adversary"); the
fix extends the same treatment to the four arena walls, gated on the new
``arena_size`` argument (None ⇒ off ⇒ byte-identical to the old behaviour).

Run:
    pytest tests/test_red_wall_repulsion.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.agents.heuristics import stationary_red, random_red


A = 130.0                    # arena side used across the function-level tests
INFLUENCE = 12.0             # must match the constant inside the heuristic


def _one(red_xy, blue_xy, arena_size=None):
    """Single active red, single blue; return the red's unit command."""
    red_pos    = np.array([red_xy],  dtype=np.float32)
    blue_pos   = np.array([blue_xy], dtype=np.float32)
    red_active = np.array([True])
    return run_from_nearest_uav(
        blue_pos, red_pos, red_active, None, None, arena_size,
    )[0]


def test_without_arena_size_flees_straight_into_wall():
    """Baseline / back-compat: with no arena_size the red near the right
    wall still flees straight at it (+x) — the pinning behaviour we are
    about to fix, preserved byte-for-byte when the arg is omitted."""
    out = _one(red_xy=[A - 3.0, A / 2], blue_xy=[A - 40.0, A / 2],
               arena_size=None)
    assert out[0] > 0.99          # essentially (+1, 0): straight into the wall
    assert abs(out[1]) < 1e-3


def test_wall_repulsion_redirects_away_from_wall():
    """Same geometry, but WITH arena_size: the right-wall inward push
    dominates the flee-into-wall vector, so the commanded x-component
    flips to point back toward open space."""
    out = _one(red_xy=[A - 3.0, A / 2], blue_xy=[A - 40.0, A / 2],
               arena_size=A)
    assert out[0] < 0.0           # now steered AWAY from the wall (−x)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)   # still max-effort


def test_far_from_walls_is_byte_identical():
    """A red far from every wall (all distances > influence) must get the
    exact same command with or without arena_size — wall repulsion only
    acts inside the influence band."""
    red = [A / 2, A / 2]          # dist to each wall = 65 >> influence
    blue = [A / 2 - 30, A / 2]
    a_off = _one(red, blue, arena_size=None)
    a_on  = _one(red, blue, arena_size=A)
    assert np.allclose(a_off, a_on, atol=1e-6)


def test_corner_pushes_diagonally_inward():
    """A red cornered against two walls gets a push from both, so the
    command points diagonally back toward the arena centre."""
    out = _one(red_xy=[A - 3.0, A - 3.0], blue_xy=[A - 30.0, A - 30.0],
               arena_size=A)
    assert out[0] < 0.0 and out[1] < 0.0                      # both inward
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_output_is_unit_and_caught_reds_zero():
    """Renormalisation invariant holds with wall repulsion on, and a
    caught (inactive) red still returns zero."""
    red_pos = np.array([[A - 2.0, A / 2], [A / 2, A / 2]], dtype=np.float32)
    blue_pos = np.array([[A - 30.0, A / 2]], dtype=np.float32)
    red_active = np.array([True, False])          # second red is "caught"
    out = run_from_nearest_uav(blue_pos, red_pos, red_active,
                               None, None, arena_size=A)
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)
    assert np.allclose(out[1], 0.0)               # caught red does not move


def test_wall_and_obstacle_repulsion_compose():
    """With both an obstacle and a wall in range the red still returns a
    single finite unit vector (the two pushes add into one term)."""
    red_pos = np.array([[A - 4.0, A / 2]], dtype=np.float32)
    blue_pos = np.array([[A - 40.0, A / 2]], dtype=np.float32)
    red_active = np.array([True])
    obstacle_pos = np.array([[A - 4.0, A / 2 - 10.0]], dtype=np.float32)
    obstacle_r = np.array([5.0], dtype=np.float32)
    out = run_from_nearest_uav(blue_pos, red_pos, red_active,
                               obstacle_pos, obstacle_r, arena_size=A)[0]
    assert np.all(np.isfinite(out))
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_env_steps_with_new_signature_and_red_does_not_pin():
    """End-to-end: the env now passes arena_size to the red policy.  A red
    parked next to the right wall, blue on the inner side, must NOT be
    driven further into the wall over a step (old behaviour pinned it)."""
    env = PursuitEnv(n_blue=1, n_red=1, n_obstacles=0, arena_size=A,
                     max_steps=50, capture_radius=3.0,
                     red_policy=run_from_nearest_uav)
    env.reset(seed=0)
    env._red_pos[0]  = np.array([A - 3.0, A / 2], dtype=np.float32)
    env._red_vel[0]  = np.zeros(2, dtype=np.float32)
    env._blue_pos[0] = np.array([A - 40.0, A / 2], dtype=np.float32)
    x_before = float(env._red_pos[0, 0])
    env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
    x_after = float(env._red_pos[0, 0])
    # With wall repulsion the command is inward (−x): the red moves away
    # from the wall, so its x must not increase toward the boundary.
    assert x_after <= x_before + 1e-4
    assert x_after < A                              # not pinned at the wall


def test_stationary_and_random_accept_arena_size():
    """The other two red policies must accept the new 6th positional arg
    (the env calls every red policy with the same signature)."""
    red_pos = np.array([[10.0, 10.0]], dtype=np.float32)
    blue_pos = np.array([[20.0, 20.0]], dtype=np.float32)
    red_active = np.array([True])
    s = stationary_red(blue_pos, red_pos, red_active, None, None, A)
    assert np.allclose(s, 0.0)
    r = random_red(seed=0)(blue_pos, red_pos, red_active, None, None, A)
    assert r.shape == (1, 2)
