"""
tests/test_red_motion_features.py — featuriser and action discretiser.

This module is shared by BOTH the training path and the eventual
inference-time ``motion_model`` adapter, precisely so the two cannot drift
apart.  These tests pin the invariants that matter for that: the
discretisation round-trips within bin resolution, the ZERO class is
handled rather than silently producing a meaningless ``arctan2(0, 0)``,
soft labels are proper distributions with CIRCULAR heading smoothing, and
the edge features match the env's 7-D convention.

Run:
    pytest tests/test_red_motion_features.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.red_motion_features import (
    N_BINS, N_HEADING_BINS, N_MAGNITUDE_BINS, V_NORM, ZERO_CLASS,
    accel_to_bin, bin_to_accel, edge_features, featurize_shard, soft_targets,
)


# --------------------------------------------------------------------- #
#  Discretisation
# --------------------------------------------------------------------- #

def test_zero_acceleration_gets_its_own_class():
    """|a| ~ 0 has NO heading -- arctan2(0,0) is meaningless.  1.78% of
    collected samples are exactly this, so it needs a real class, not a
    fabricated direction."""
    assert accel_to_bin(np.array([0.0, 0.0])) == ZERO_CLASS
    assert accel_to_bin(np.array([1e-9, -1e-9])) == ZERO_CLASS
    assert np.allclose(bin_to_accel(np.array(ZERO_CLASS)), 0.0)


def test_nonzero_acceleration_never_lands_in_the_zero_class():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5000, 2))
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    a *= rng.uniform(0.05, 1.0, (5000, 1))
    assert np.all(accel_to_bin(a) != ZERO_CLASS)


def test_round_trip_is_within_bin_resolution():
    """bin_to_accel(accel_to_bin(a)) must land in the same cell as a."""
    rng = np.random.default_rng(1)
    ang = rng.uniform(-np.pi, np.pi, 20000)
    mag = rng.uniform(0.02, 1.0, 20000)
    a = np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=-1)
    rec = bin_to_accel(accel_to_bin(a))
    # worst case: half a heading bin at radius 1 plus half a magnitude bin
    bound = np.pi / N_HEADING_BINS + 0.5 / N_MAGNITUDE_BINS
    assert np.max(np.linalg.norm(a - rec, axis=1)) < bound


def test_full_magnitude_lands_in_the_top_bin():
    """|a| == 1.0 exactly is 18.7% of the data (deterministic reds run at
    full effort) -- it must not fall off the end of the binning."""
    for ang in np.linspace(-np.pi, np.pi, 17):
        a = np.array([np.cos(ang), np.sin(ang)])
        idx = int(accel_to_bin(a))
        assert idx != ZERO_CLASS
        assert idx % N_MAGNITUDE_BINS == N_MAGNITUDE_BINS - 1


def test_headings_spread_over_all_bins():
    ang = np.linspace(-np.pi, np.pi, N_HEADING_BINS * 40, endpoint=False)
    a = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    h = accel_to_bin(a) // N_MAGNITUDE_BINS
    assert len(np.unique(h)) == N_HEADING_BINS


# --------------------------------------------------------------------- #
#  Soft labels
# --------------------------------------------------------------------- #

def test_soft_targets_are_proper_distributions():
    idx = np.array([0, 7, 123, ZERO_CLASS, N_BINS - 2])
    st = soft_targets(idx)
    assert st.shape == (5, N_BINS)
    assert np.allclose(st.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(st >= 0.0)


def test_zero_class_label_is_not_smoothed():
    """ZERO is discrete and unambiguous -- smoothing it into neighbouring
    grid cells would assert a heading it does not have."""
    st = soft_targets(np.array([ZERO_CLASS]))[0]
    assert st[ZERO_CLASS] == 1.0
    assert np.all(st[:ZERO_CLASS] == 0.0)


def test_heading_smoothing_wraps_circularly():
    """Heading bin 0 and bin 35 are 10 degrees apart, not unrelated
    classes."""
    # heading bin 0, magnitude bin 2
    idx = 0 * N_MAGNITUDE_BINS + 2
    st = soft_targets(np.array([idx]))[0]
    wrapped = (N_HEADING_BINS - 1) * N_MAGNITUDE_BINS + 2
    assert st[wrapped] > 0.0, "no mass on the wrap-around heading neighbour"
    assert st[1 * N_MAGNITUDE_BINS + 2] > 0.0


def test_magnitude_smoothing_does_not_wrap():
    """Magnitude is ordinal, not circular: the bottom bin must not bleed
    into the top one."""
    idx = 10 * N_MAGNITUDE_BINS + 0            # magnitude bin 0
    st = soft_targets(np.array([idx]))[0]
    top = 10 * N_MAGNITUDE_BINS + (N_MAGNITUDE_BINS - 1)
    assert st[top] == 0.0, "magnitude smoothing wrapped around"


def test_soft_target_mass_is_concentrated_on_the_true_bin():
    idx = 12 * N_MAGNITUDE_BINS + 2
    st = soft_targets(np.array([idx]))[0]
    assert st.argmax() == idx
    assert st[idx] > 0.6


# --------------------------------------------------------------------- #
#  Edge features
# --------------------------------------------------------------------- #

def test_edge_features_match_the_env_convention():
    """rel_pos(2) + rel_vel(2) + range(1) + bearing_cos_sin(2) = 7."""
    rel_pos = np.array([[3.0, 4.0]])
    rel_vel = np.array([[0.1, -0.2]])
    recv_vel = np.array([[1.0, 0.0]])
    e = edge_features(rel_pos, rel_vel, recv_vel)
    assert e.shape == (1, 7)
    assert np.allclose(e[0, :2], rel_pos[0])
    assert np.allclose(e[0, 2:4], rel_vel[0])
    assert np.isclose(e[0, 4], 5.0)                 # range
    assert np.isclose(e[0, 5], 3.0 / 5.0)           # cos: heading is +x
    assert np.isclose(e[0, 6], 4.0 / 5.0)           # sin: sender is to my left


def test_bearing_is_signed_left_versus_right():
    """An unsigned angle could not tell left from right, which is exactly
    the distinction an evader turning away needs."""
    recv_vel = np.array([[1.0, 0.0]])
    left = edge_features(np.array([[0.0, 5.0]]), np.zeros((1, 2)), recv_vel)
    right = edge_features(np.array([[0.0, -5.0]]), np.zeros((1, 2)), recv_vel)
    assert left[0, 6] > 0.0 and right[0, 6] < 0.0
    assert np.isclose(left[0, 5], right[0, 5])      # same cos, opposite sin


def test_degenerate_geometry_gives_zero_bearing():
    """Zero range or a stationary receiver -> bearing is undefined; zeros
    are the well-defined signal for that, matching _bearing_features."""
    e = edge_features(np.zeros((1, 2)), np.zeros((1, 2)),
                     np.array([[1.0, 0.0]]))
    assert np.allclose(e[0, 5:7], 0.0)
    e2 = edge_features(np.array([[3.0, 4.0]]), np.zeros((1, 2)),
                      np.zeros((1, 2)))
    assert np.allclose(e2[0, 5:7], 0.0)


# --------------------------------------------------------------------- #
#  Shard featurisation
# --------------------------------------------------------------------- #

def _fake_shard(n=8, blue_cap=4, obs_cap=3, seed=0):
    rng = np.random.default_rng(seed)
    n_blue = rng.integers(1, blue_cap + 1, n)
    placed = rng.integers(0, obs_cap + 1, n)
    obs_mask = np.arange(obs_cap)[None, :] < placed[:, None]
    a = rng.normal(size=(n, 2)).astype(np.float32)
    return dict(
        episode_id=np.arange(n), step=np.zeros(n, int), red_id=np.zeros(n, int),
        red_pos=rng.uniform(0, 130, (n, 2)).astype(np.float32),
        red_vel=rng.uniform(-1, 1, (n, 2)).astype(np.float32),
        accel=a,
        blue_rel_pos=rng.normal(size=(n, blue_cap, 2)).astype(np.float32),
        blue_rel_vel=rng.normal(size=(n, blue_cap, 2)).astype(np.float32),
        obs_rel_pos=rng.normal(size=(n, obs_cap, 2)).astype(np.float32),
        obs_rel_vel=np.zeros((n, obs_cap, 2), np.float32),
        obs_radius=rng.uniform(0, 0.1, (n, obs_cap)).astype(np.float32),
        obs_mask=obs_mask,
        wall_dist=rng.uniform(0, 1, (n, 4)).astype(np.float32),
        n_blue=n_blue, n_obs_placed=placed,
    )


def test_featurize_shard_shapes_and_dtypes():
    d = _fake_shard()
    f = featurize_shard(d)
    assert f["red_feats"].shape == (8, 1, 7)      # one red node per sample
    assert f["blue_feats"].shape == (8, 4, 1)
    assert f["obs_feats"].shape == (8, 3, 2)
    assert f["b2r_edge_feats"].shape == (8, 4, 7)
    assert f["o2r_edge_feats"].shape == (8, 3, 7)
    assert f["target"].shape == (8,)
    for k in ("red_feats", "blue_feats", "b2r_edge_feats", "b2r_active"):
        assert f[k].dtype == np.float32


def test_featurize_shard_active_masks_track_the_counts():
    d = _fake_shard()
    f = featurize_shard(d)
    assert np.all(f["b2r_active"].sum(axis=1) == d["n_blue"])
    assert np.all(f["o2r_active"].sum(axis=1) == d["n_obs_placed"])


def test_featurize_shard_normalises_the_raw_red_velocity():
    """red_vel is the one field the collector stores RAW -- forgetting to
    normalise it here is exactly the kind of train/serve skew this shared
    module exists to prevent."""
    d = _fake_shard()
    f = featurize_shard(d)
    assert np.allclose(f["red_feats"][:, 0, :2],
                      d["red_vel"] / V_NORM, atol=1e-6)
    speed = np.linalg.norm(d["red_vel"] / V_NORM, axis=-1)
    assert np.allclose(f["red_feats"][:, 0, 2], speed, atol=1e-6)


def test_featurize_shard_uses_relative_not_absolute_velocity():
    """Edge rel_vel must be sender MINUS receiver, per the env's
    convention -- not the sender's absolute velocity the collector
    stored."""
    d = _fake_shard()
    f = featurize_shard(d)
    want = d["blue_rel_vel"] - (d["red_vel"] / V_NORM)[:, None, :]
    assert np.allclose(f["b2r_edge_feats"][..., 2:4], want, atol=1e-5)


def test_featurize_shard_carries_wall_distances_through():
    d = _fake_shard()
    f = featurize_shard(d)
    assert np.allclose(f["red_feats"][:, 0, 3:7], d["wall_dist"], atol=1e-6)
