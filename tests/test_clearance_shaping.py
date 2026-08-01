r"""
tests/test_clearance_shaping.py — verifies the clearance / barrier
shaping reward (backlog §16 #1).

Goal: near-ZERO crashes.  A flat crash penalty only fires at contact and
gives no positional gradient — it says "you're being punished" but not
which way to escape.  The clearance term is a dense per-agent penalty for
being within ``clearance_margin`` of an obstacle SURFACE, growing as the
blue approaches and continuing to grow INSIDE the disk, so its gradient
points continuously toward open space:

    penalty(d) shape, d = distance to surface (─ = 0, band = [0, margin]):

      reward 0 ─────────────────────────  (d >= margin: no penalty)
             \
              \                            (approaching: ramps down)
      -w ······\···························  (d = 0: surface)
                \
      (< -w) inside the disk (d < 0): keeps dropping -> gradient points OUT

Both the "keep clear" (outside) and "get out, this way" (inside) signals
come from the same monotone function.  Off (byte-identical) when
``clearance_weight = 0``.

Run:
    pytest tests/test_clearance_shaping.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red


def _env(weight=0.5, margin=8.0, seed=0):
    env = PursuitEnv(
        n_blue=1, n_red=1, n_obstacles=1, arena_size=130.0, max_steps=50,
        capture_radius=3.0, obstacle_radius_min=10.0, obstacle_radius_max=10.0,
        clearance_weight=weight, clearance_margin=margin,
        red_policy=stationary_red, seed=seed,
    )
    env.reset(seed=seed)
    return env


def _clearance_at(env, blue_xy):
    """Step once with the blue teleported to blue_xy (zero action) and
    return the clearance-penalty magnitude reported in info (>= 0)."""
    env._blue_pos[0] = np.array(blue_xy, dtype=np.float32)
    env._blue_vel[0] = np.zeros(2, dtype=np.float32)
    _, _, _, _, info = env.step({env.agents[0]: np.zeros(2, dtype=np.float32)})
    return float(info[env.agents[0]]["clearance_penalty"])


def test_off_by_default_is_byte_identical():
    """weight=0 (default) => no clearance term => reward unchanged, penalty
    reported as 0 even for a blue sitting on an obstacle."""
    env = _env(weight=0.0)
    o = env._obstacle_pos[0].copy()
    pen = _clearance_at(env, o)          # right on the centre
    assert pen == 0.0


def test_zero_beyond_the_margin():
    """A blue farther than clearance_margin from the surface gets no
    penalty."""
    env = _env(weight=0.5, margin=8.0)
    o = env._obstacle_pos[0].copy()
    r = env._obstacle_r[0]
    far = o + np.array([r + 8.0 + 5.0, 0.0])   # 5 m beyond the band
    assert _clearance_at(env, far) == 0.0


def test_penalty_grows_as_blue_approaches_surface():
    """Monotone: closer to the surface (from outside) => larger penalty,
    and exactly 0 at the margin edge."""
    env = _env(weight=0.5, margin=8.0)
    o = env._obstacle_pos[0].copy()
    r = env._obstacle_r[0]
    edge   = _clearance_at(env, o + np.array([r + 8.0, 0.0]))   # margin edge
    mid    = _clearance_at(env, o + np.array([r + 4.0, 0.0]))   # halfway in
    surface= _clearance_at(env, o + np.array([r + 0.0, 0.0]))   # at surface
    assert np.isclose(edge, 0.0, atol=1e-6)
    assert 0.0 < mid < surface


def test_penalty_keeps_growing_inside_the_disk():
    """The 'inverse'/get-out signal: INSIDE the disk the penalty is even
    larger than at the surface and grows toward the centre — so its
    gradient (penalty decreasing outward) always points OUT."""
    env = _env(weight=0.5, margin=8.0)
    o = env._obstacle_pos[0].copy()
    r = env._obstacle_r[0]
    surface = _clearance_at(env, o + np.array([r,       0.0]))  # d = 0
    inside  = _clearance_at(env, o + np.array([r / 2.0, 0.0]))  # d < 0
    centre  = _clearance_at(env, o)                             # deepest
    assert surface < inside < centre


def test_gradient_points_outward_everywhere_in_range():
    """Finite-difference check: moving a small step AWAY from the obstacle
    centre reduces the penalty, both when outside-within-band and when
    inside — i.e. the reward gradient pushes the blue out."""
    env = _env(weight=0.5, margin=8.0)
    o = env._obstacle_pos[0].copy()
    r = env._obstacle_r[0]
    for base_d in (-4.0, -1.0, 2.0, 6.0):        # inside and outside the surface
        p_here = _clearance_at(env, o + np.array([r + base_d,       0.0]))
        p_out  = _clearance_at(env, o + np.array([r + base_d + 1.0, 0.0]))
        assert p_out <= p_here + 1e-6, f"not outward at d={base_d}"


def test_sums_over_multiple_obstacles():
    """A blue squeezed between two obstacles feels both pushes (sum), so
    its penalty exceeds either obstacle alone."""
    env = PursuitEnv(
        n_blue=1, n_red=1, n_obstacles=2, arena_size=130.0, max_steps=50,
        obstacle_radius_min=8.0, obstacle_radius_max=8.0,
        clearance_weight=0.5, clearance_margin=10.0,
        red_policy=stationary_red, seed=1,
    )
    env.reset(seed=1)
    # Place two obstacles 22 m apart and the blue in the gap between them.
    env._obstacle_pos[0] = np.array([50.0, 65.0], dtype=np.float32)
    env._obstacle_pos[1] = np.array([72.0, 65.0], dtype=np.float32)
    env._obstacle_r[:]   = 8.0
    gap = np.array([61.0, 65.0], dtype=np.float32)      # within both bands
    both = _clearance_at(env, gap)
    # Far from obstacle 1 only: penalty from a single obstacle at the same
    # surface distance is strictly smaller.
    d_each = np.linalg.norm(gap - env._obstacle_pos[0]) - 8.0
    single = _clearance_at(env, np.array([50.0, 65.0 + 8.0 + d_each],
                                         dtype=np.float32))
    assert both > single > 0.0


def test_clearance_reduces_reward_but_not_when_off():
    """End-to-end: with the term on, a blue parked next to an obstacle gets
    a strictly lower per-agent reward than the same state with it off."""
    o_xy = None
    rewards = {}
    for w in (0.0, 0.5):
        env = _env(weight=w, margin=8.0, seed=3)
        o = env._obstacle_pos[0].copy(); o_xy = o
        r = env._obstacle_r[0]
        env._blue_pos[0] = o + np.array([r + 1.0, 0.0], dtype=np.float32)
        env._blue_vel[0] = np.zeros(2, dtype=np.float32)
        _, rew, _, _, _ = env.step({env.agents[0]: np.zeros(2, np.float32)})
        rewards[w] = rew[env.agents[0]]
    assert rewards[0.5] < rewards[0.0]
