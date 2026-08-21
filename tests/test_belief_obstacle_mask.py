"""
tests/test_belief_obstacle_mask.py — obstacle-aware enemy belief channel.

Two coupled bugs this covers:

1. **Obstacle interiors read "unknown".**  Cells inside an obstacle are
   permanently occluded, so the sensor update never reaches them and they
   sat near log-odds 0 (= P 0.5, "unknown").  But reds are kinematically
   clipped OUT of obstacle disks, so P(enemy | inside obstacle) = 0.
   Measured before the fix: interiors averaged -1.87 vs -7.64 for properly
   cleared open space — ~5.8 log-odds HIGHER — turning them into
   probability sinks that attracted phantom peaks (~6.6% of extracted
   peaks landed inside an obstacle, tens of metres from any real red).

2. **Diffusion leaked mass into obstacles.**  The isotropic 3x3 motion
   kernel spread probability into masked cells, continuously re-filling
   the interiors that decay was pulling toward 0 (which is *why* they sat
   at -1.87 rather than a clean 0).  The kernel now renormalises each
   source cell's outgoing weight over its VALID neighbours — a reflecting
   boundary at the obstacle wall, which also conserves probability instead
   of silently destroying it.

Run:
    pytest tests/test_belief_obstacle_mask.py -v
"""
from __future__ import annotations

import numpy as np

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav


def _env(n_obstacles=4, seed=0, diffusion=0.2, decay=0.99):
    env = PursuitEnv(
        n_blue=5, n_red=3, n_obstacles=n_obstacles, arena_size=130.0,
        max_steps=200, capture_radius=3.0, sensor_radius=40.0,
        use_belief_maps=True, enemy_belief_decay=decay,
        enemy_belief_diffusion=diffusion, sensor_pos_noise_std=1.0,
        red_policy=run_from_nearest_uav, seed=seed,
    )
    env.reset(seed=seed)
    return env


def _run(env, steps=60, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if not env.agents:
            break
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                  for a in env.agents})
    return env


def _inside_mask(env):
    """(W, H) bool — cell centre lies inside some obstacle disk."""
    cen = env._cell_centres
    d = np.linalg.norm(
        cen[:, :, None, :] - env._obstacle_pos[None, None, :, :], axis=-1)
    return np.any(d <= env._obstacle_r[None, None, :], axis=-1)


def test_enemy_channel_is_strongly_negative_inside_obstacles():
    """Interiors are KNOWN-EMPTY, so they must be pinned to the negative
    clip — not left near 0 ("unknown")."""
    env = _run(_env())
    inside = _inside_mask(env)
    assert inside.any(), "test needs at least one obstacle-interior cell"
    L = env._belief_maps[0]
    assert np.allclose(L[inside], -env.belief_clip, atol=1e-5)


def test_interiors_are_not_higher_than_cleared_open_space():
    """The actual bias that caused phantom peaks: interiors used to sit
    ~5.8 log-odds ABOVE properly cleared cells.  Now they are at the floor,
    so they can never out-rank observed-empty space."""
    env = _run(_env())
    inside = _inside_mask(env)
    L = env._belief_maps[0]
    assert L[inside].max() <= L[~inside].min() + 1e-6


def test_no_extracted_peak_lands_inside_an_obstacle():
    """The user-visible symptom: phantom enemy tracks inside obstacles."""
    env = _env()
    rng = np.random.default_rng(0)
    checked = 0
    for t in range(150):
        if not env.agents:
            break
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                  for a in env.agents})
        if t < 20 or t % 5:
            continue
        act = np.where(env._red_active)[0]
        if len(act) == 0:
            continue
        pk, cf = env._extract_belief_peaks(
            len(act), channel_idx=0, k_extract=len(act), nms_radius_cells=2)
        for p, c in zip(pk, cf):
            if c <= 0:
                continue
            checked += 1
            surf = np.linalg.norm(p - env._obstacle_pos, axis=1) - env._obstacle_r
            assert not bool((surf <= 0).any()), (
                f"peak {p} landed inside an obstacle at t={t}")
    assert checked > 0, "no peaks were actually checked"


def test_diffusion_does_not_leak_mass_into_obstacles():
    """After a pure prediction step (no sensor update), obstacle cells must
    hold no diffused probability."""
    env = _env()
    # Seed a bright blob adjacent to an obstacle, then predict repeatedly.
    env._belief_maps[0][:] = 0.0
    o = env._obstacle_pos[0]
    cs = env.belief_cell_size
    cx = int(np.clip(o[0] / cs, 0, env.belief_grid_size - 1))
    cy = int(np.clip(o[1] / cs, 0, env.belief_grid_size - 1))
    env._belief_maps[0][cx, cy] = env.belief_clip     # inside/at the obstacle
    inside = _inside_mask(env)
    for _ in range(5):
        env._predict_enemy_belief()
    L = env._belief_maps[0]
    assert np.allclose(L[inside], -env.belief_clip, atol=1e-5)


def test_no_obstacles_diffusion_is_unchanged():
    """With zero obstacles every cell is valid, so the renormalised kernel
    must reduce exactly to the plain convolution (no behaviour change for
    obstacle-free runs)."""
    env = PursuitEnv(
        n_blue=3, n_red=2, n_obstacles=0, arena_size=130.0, max_steps=100,
        sensor_radius=40.0, use_belief_maps=True,
        enemy_belief_decay=0.99, enemy_belief_diffusion=0.2,
        red_policy=run_from_nearest_uav, seed=1,
    )
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    env._belief_maps[0][:] = rng.normal(0, 2, env._belief_maps[0].shape)
    before = env._belief_maps[0].copy()

    # Reference: the plain (unmasked) predict computed by hand.
    L = before * env.enemy_belief_decay
    P = 1.0 / (1.0 + np.exp(-L))
    p_move = env.enemy_belief_diffusion
    Pp = np.pad(P, 1, mode="edge")
    ref = ((1.0 - p_move) * Pp[1:-1, 1:-1]
           + p_move * 0.20 * (Pp[:-2, 1:-1] + Pp[2:, 1:-1]
                              + Pp[1:-1, :-2] + Pp[1:-1, 2:])
           + p_move * 0.05 * (Pp[:-2, :-2] + Pp[:-2, 2:]
                              + Pp[2:, :-2] + Pp[2:, 2:]))
    ref = np.clip(ref, 1e-6, 1.0 - 1e-6)
    ref = np.clip(np.log(ref / (1.0 - ref)), -env.belief_clip, env.belief_clip)

    env._predict_enemy_belief()
    assert np.allclose(env._belief_maps[0], ref, atol=1e-5)


def test_obstacle_channel_is_untouched():
    """Only the ENEMY channel is pinned — inside an obstacle is exactly
    where the OBSTACLE channel should be high."""
    env = _run(_env())
    inside = _inside_mask(env)
    obstacle_channel = env._belief_maps[1]
    # Not forced negative: the obstacle channel still carries its own
    # (positive, sensor-driven) evidence where obstacles actually are.
    assert obstacle_channel[inside].max() > -env.belief_clip + 1e-6


def test_belief_window_size_attribute_removed():
    """Dead attribute (never read anywhere) is gone from the env."""
    env = _env(n_obstacles=1)
    assert not hasattr(env, "belief_window_size")
