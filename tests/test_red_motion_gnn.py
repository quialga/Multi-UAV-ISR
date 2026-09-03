"""
tests/test_red_motion_gnn.py — the learned red-motion model's network.

The properties that matter structurally (not "does it train", which needs
the dataset): one categorical per RED node, a genuinely joint grid,
permutation invariance over senders, padding masks that actually mask, and
a near-uniform initialisation so an untrained model commits to nothing.

Run:
    pytest tests/test_red_motion_gnn.py -v
"""
from __future__ import annotations

import numpy as np
import torch

from isr.agents.red_motion_gnn import RedMotionGNN, _build_x2r_edges


def _inputs(B=4, n_blue=5, n_red=3, n_obs=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        red_feats=torch.randn(B, n_red, 7, generator=g),
        blue_feats=torch.randn(B, n_blue, 1, generator=g),
        b2r_edge_feats=torch.randn(B, n_blue * n_red, 7, generator=g),
        obs_feats=torch.randn(B, n_obs, 2, generator=g),
        o2r_edge_feats=torch.randn(B, n_obs * n_red, 7, generator=g),
        b2r_active=torch.ones(B, n_blue * n_red),
        o2r_active=torch.ones(B, n_obs * n_red),
    )


def test_one_categorical_per_red():
    """A step with N active reds must yield N independent predictions --
    matching the dataset's one-sample-per-(step, red) layout."""
    net = RedMotionGNN(n_blue=5, n_red=3, n_obs=4)
    out = net(**_inputs())
    assert out.shape == (4, 3, 36 * 5)


def test_grid_view_is_a_proper_joint_distribution():
    net = RedMotionGNN(n_blue=5, n_red=3, n_obs=4)
    p = net.grid_probs(**_inputs())
    assert p.shape == (4, 3, 36, 5)
    # sums to 1 over the JOINT grid, not per-axis
    assert torch.allclose(p.sum(dim=(-2, -1)), torch.ones(4, 3), atol=1e-5)


def test_initialises_near_uniform():
    """An untrained model must not commit to a heading before seeing data:
    entropy should start at ~log(n_bins)."""
    net = RedMotionGNN(n_blue=5, n_red=3, n_obs=4)
    with torch.no_grad():
        lp = net.log_probs(**_inputs())
    ent = float(-(lp.exp() * lp).sum(-1).mean())
    assert abs(ent - np.log(36 * 5)) < 0.05


def test_permutation_invariant_over_blues():
    """Only the SET of blues matters -- the network must not key on which
    slot a blue happens to occupy.  This is the property a flattened,
    padded MLP input would NOT have."""
    n_blue, n_red, n_obs = 4, 2, 0
    net = RedMotionGNN(n_blue=n_blue, n_red=n_red, n_obs=n_obs).eval()
    g = torch.Generator().manual_seed(1)
    red = torch.randn(1, n_red, 7, generator=g)
    blue = torch.randn(1, n_blue, 1, generator=g)
    # b2r edges are ordered (for s in senders: for r in reds)
    e = torch.randn(1, n_blue * n_red, 7, generator=g)

    with torch.no_grad():
        out_a = net(red, blue, e)

    perm = [2, 0, 3, 1]
    blue_p = blue[:, perm]
    e_p = e.reshape(1, n_blue, n_red, 7)[:, perm].reshape(1, -1, 7)
    with torch.no_grad():
        out_b = net(red, blue_p, e_p)

    assert torch.allclose(out_a, out_b, atol=1e-5), (
        "permuting the blue set changed the prediction")


def test_inactive_edges_are_masked_out():
    """Padding masks must genuinely remove a sender's influence: zeroing a
    blue's edge mask must give the same answer as if that blue's edge
    features were anything else entirely."""
    n_blue, n_red = 4, 2
    net = RedMotionGNN(n_blue=n_blue, n_red=n_red, n_obs=0).eval()
    g = torch.Generator().manual_seed(2)
    red = torch.randn(1, n_red, 7, generator=g)
    blue = torch.randn(1, n_blue, 1, generator=g)
    e = torch.randn(1, n_blue * n_red, 7, generator=g)

    mask = torch.ones(1, n_blue * n_red)
    mask.reshape(1, n_blue, n_red)[:, 3] = 0.0        # blue 3 is padding

    e_alt = e.clone().reshape(1, n_blue, n_red, 7)
    e_alt[:, 3] = torch.randn(n_red, 7, generator=g)  # garbage in that slot
    e_alt = e_alt.reshape(1, -1, 7)

    with torch.no_grad():
        a = net(red, blue, e, b2r_active=mask)
        b = net(red, blue, e_alt, b2r_active=mask)
    assert torch.allclose(a, b, atol=1e-6), "masked edge still leaked through"


def test_works_without_obstacles():
    net = RedMotionGNN(n_blue=3, n_red=2, n_obs=0)
    x = _inputs(B=2, n_blue=3, n_red=2, n_obs=1)
    out = net(x["red_feats"], x["blue_feats"], x["b2r_edge_feats"])
    assert out.shape == (2, 2, 36 * 5)


def test_edge_index_ordering_matches_the_env_convention():
    """(for s in senders: for r in reds) -- the same ordering the env uses
    for its x->blue edges, so one featuriser serves either graph."""
    src, dst = _build_x2r_edges(n_src=3, n_red=2)
    assert src.tolist() == [0, 0, 1, 1, 2, 2]
    assert dst.tolist() == [0, 1, 0, 1, 0, 1]


def test_gradients_flow_to_every_input_branch():
    net = RedMotionGNN(n_blue=3, n_red=2, n_obs=2)
    x = _inputs(B=2, n_blue=3, n_red=2, n_obs=2)
    for k in ("red_feats", "blue_feats", "b2r_edge_feats", "obs_feats",
             "o2r_edge_feats"):
        x[k].requires_grad_(True)
    net(**x).sum().backward()
    for k in ("red_feats", "blue_feats", "b2r_edge_feats", "obs_feats",
             "o2r_edge_feats"):
        assert x[k].grad is not None and torch.any(x[k].grad != 0), (
            f"no gradient reached {k}")


def test_parameter_count_is_modest():
    net = RedMotionGNN(n_blue=5, n_red=3, n_obs=4)
    n = sum(p.numel() for p in net.parameters())
    assert n < 250_000, f"unexpectedly large: {n}"
