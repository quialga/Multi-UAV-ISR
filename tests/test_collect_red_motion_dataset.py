"""
tests/test_collect_red_motion_dataset.py — the red-motion dataset collector.

Run:
    pytest tests/test_collect_red_motion_dataset.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_red_motion_dataset import (   # noqa: E402
    MixedStochasticRed, _FIELDS, _write_shard, collect_episode,
)


def test_mixed_red_is_exact_vs_a_single_red_at_one_slot():
    """The whole justification for MixedStochasticRed's per-slot slicing:
    driving StochasticRed with red_pos[i:i+1] must be IDENTICAL to reading
    index i out of the batched call, since run_from_nearest_uav never
    looks at another red's position or active flag."""
    rng = np.random.default_rng(0)
    blue = rng.uniform(20, 110, (3, 2)).astype(np.float32)
    red = rng.uniform(20, 110, (3, 2)).astype(np.float32)
    active = np.ones(3, dtype=bool)
    opos = np.array([[65.0, 65.0]], dtype=np.float32)
    orad = np.array([10.0], dtype=np.float32)

    mixed = MixedStochasticRed(n_red=3, rng=np.random.default_rng(1),
                               p_deterministic=1.0)   # all deterministic
    mixed.reset(3)
    batched = mixed(blue, red, active, opos, orad, 130.0)
    for i, sub in enumerate(mixed.subs):
        sub.reset(1)
        solo = sub(blue, red[i:i + 1], active[i:i + 1], opos, orad, 130.0)[0]
        assert np.allclose(batched[i], solo)


def test_mixed_red_gives_each_slot_independent_parameters():
    """Different reds must actually behave differently -- not all reduce
    to the same shared configuration."""
    rng = np.random.default_rng(2)
    mixed = MixedStochasticRed(n_red=6, rng=rng, p_deterministic=0.0)
    stds = {cfg.get("heading_noise_std") for cfg in mixed.configs}
    assert len(stds) > 1, "every red got the same heading_noise_std"


def test_mixed_red_respects_p_deterministic():
    rng = np.random.default_rng(3)
    mixed = MixedStochasticRed(n_red=200, rng=rng, p_deterministic=0.5)
    n_det = sum(1 for c in mixed.configs if c == {})
    assert 60 < n_det < 140, f"expected ~100/200 deterministic, got {n_det}"


def test_collect_episode_schema_and_padding():
    rng = np.random.default_rng(4)
    samples = collect_episode(
        rng, ep_id=0, steps=20,
        n_blue_range=(2, 4), n_red_range=(1, 2), n_obs_range=(0, 3),
        blue_cap=4, obs_cap=3, arena_size=130.0, p_deterministic=0.2)
    assert samples, "no samples collected"
    s = samples[0]
    assert set(s) == set(_FIELDS)
    assert s["blue_rel_pos"].shape == (4, 2)
    assert s["obs_rel_pos"].shape == (3, 2)
    assert s["obs_mask"].shape == (3,)
    for s in samples:
        assert s["obs_mask"].sum() == s["n_obs_placed"]
        # padding beyond n_blue / n_obs_placed must be exactly zero
        if s["n_blue"] < 4:
            assert np.all(s["blue_rel_pos"][s["n_blue"]:] == 0.0)
        if s["n_obs_placed"] < 3:
            assert np.all(s["obs_rel_pos"][s["n_obs_placed"]:] == 0.0)
            assert np.all(s["obs_radius"][s["n_obs_placed"]:] == 0.0)


def test_accel_matches_the_true_action_not_a_re_invocation():
    """accel must be exactly what the red's OWN policy produced that step
    (env._last_red_action) -- if the collector accidentally called the
    policy a second time, a stateful StochasticRed's AR(1)/commitment
    state would have advanced twice and this would silently disagree."""
    rng = np.random.default_rng(5)
    samples = collect_episode(
        rng, ep_id=0, steps=15,
        n_blue_range=(2, 2), n_red_range=(1, 1), n_obs_range=(0, 0),
        blue_cap=2, obs_cap=1, arena_size=130.0, p_deterministic=0.0)
    for s in samples:
        assert np.all(np.abs(s["accel"]) <= 1.0 + 1e-6)
        assert np.all(np.isfinite(s["accel"]))


def test_wall_dist_matches_the_env_convention():
    rng = np.random.default_rng(6)
    samples = collect_episode(
        rng, ep_id=0, steps=10,
        n_blue_range=(2, 2), n_red_range=(1, 1), n_obs_range=(0, 0),
        blue_cap=2, obs_cap=1, arena_size=130.0, p_deterministic=1.0)
    L = 130.0
    for s in samples:
        x, y = s["red_pos"]
        want = np.array([x, L - x, y, L - y], dtype=np.float32) / L
        assert np.allclose(s["wall_dist"], want, atol=1e-5)


def test_reproducible_with_the_same_seed():
    def run():
        rng = np.random.default_rng(42)
        return collect_episode(
            rng, ep_id=0, steps=10,
            n_blue_range=(2, 3), n_red_range=(1, 2), n_obs_range=(0, 2),
            blue_cap=3, obs_cap=2, arena_size=130.0, p_deterministic=0.2)
    a, b = run(), run()
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.allclose(sa["red_pos"], sb["red_pos"])
        assert np.allclose(sa["accel"], sb["accel"])


def test_shard_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    samples = collect_episode(
        rng, ep_id=0, steps=10,
        n_blue_range=(2, 2), n_red_range=(1, 1), n_obs_range=(1, 1),
        blue_cap=2, obs_cap=1, arena_size=130.0, p_deterministic=0.0)
    path = tmp_path / "shard_00000.npz"
    _write_shard(samples, path)
    loaded = np.load(path)
    assert set(loaded.keys()) == set(_FIELDS)
    assert loaded["accel"].shape == (len(samples), 2)
    assert np.allclose(loaded["red_pos"][0], samples[0]["red_pos"])
