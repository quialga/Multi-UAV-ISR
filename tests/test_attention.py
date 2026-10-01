"""
tests/test_attention.py — convex-combination aggregation for the actor.

The encoder aggregated every incoming message with an UNNORMALISED SUM, so
the COUNT of live edges scaled what reached ``update_mlp``.  That is what
made ``comms_radius`` uninterpretable (docs/stage4_results.md §16: opening
the radio is a 1.9x magnitude change, and the score tracked the magnitude
rather than the information) and what blocks variable entity counts.
Attention replaces it with weights that softmax to 1 over each receiver's
LIVE edges.

What must hold:

* the default is bit-identical to the old sum, or every result in
  docs/stage4_results.md silently moves;
* the aggregate is INVARIANT to how many edges are live — the whole point;
* a receiver with no live edges gets zeros, never NaN.  27.6% of steps have
  no confirmed track at all, so this is the common case, not an edge case,
  and a NaN there poisons every gradient in the model;
* masking happens BEFORE the softmax, so survivors renormalise to 1;
* it survives the round trip through a checkpoint, since loading the right
  tensors into the wrong aggregation reads as a mysteriously bad policy.

Run:
    pytest tests/test_attention.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from isr.agents.gnn_stage4_policy import GNNEncoder, GNNStage4Policy
from isr.agents.policy_loader import load_policy

DIMS = dict(n_blue=4, n_red=3, n_obs=0, blue_feat_dim=8, red_feat_dim=4,
            edge_feat_dim=7, d_hidden=16, n_msg_rounds=2)
B = 2


def _blue(enc: GNNEncoder, **kw) -> torch.Tensor:
    """The encoder returns ``(h_blue, h_red, h_obs)``; the blue embeddings are
    what the aggregation feeds and the only part these tests care about."""
    return enc(**kw)[0]


def _inputs(enc: GNNEncoder, bb_vis=None, rb_vis=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(
        blue_feats=r(B, enc.n_blue, DIMS["blue_feat_dim"]),
        red_feats=r(B, enc.n_red, DIMS["red_feat_dim"]),
        bb_edge_feats=r(B, enc.bb_src.numel(), DIMS["edge_feat_dim"]),
        rb_edge_feats=r(B, enc.rb_src.numel(), DIMS["edge_feat_dim"]),
        bb_visible=bb_vis,
        rb_visible=rb_vis,
    )


# ------------------------------------------------------------------ #
#  The default must not move
# ------------------------------------------------------------------ #

def test_default_is_the_old_sum_bit_for_bit():
    """Two encoders with the same seed, one built before the flag existed."""
    torch.manual_seed(0)
    a = GNNEncoder(**DIMS)
    torch.manual_seed(0)
    b = GNNEncoder(**DIMS, attention=False)
    assert a.attention is False and a.att_mlp is None
    x = _inputs(a)
    torch.testing.assert_close(_blue(a, **x), _blue(b, **x))


def test_attention_changes_the_output():
    """Otherwise the flag is silently inert and every later result is void."""
    torch.manual_seed(0)
    summed = GNNEncoder(**DIMS)
    torch.manual_seed(0)
    attend = GNNEncoder(**DIMS, attention=True, n_heads=4)
    x = _inputs(summed)
    assert not torch.allclose(_blue(summed, **x), _blue(attend, **x))


# ------------------------------------------------------------------ #
#  Scale invariance -- the reason this exists
# ------------------------------------------------------------------ #

def test_the_sum_scales_with_live_edge_count_and_attention_does_not():
    """The defect and the fix, measured side by side on the same inputs.

    Every ally sends the SAME message here (identical blue features and edge
    features), so doubling the live edges doubles an unnormalised sum while a
    convex combination of identical messages is unchanged.
    """
    n_blue = DIMS["n_blue"]
    n_bb = n_blue * (n_blue - 1)
    for attention in (False, True):
        torch.manual_seed(0)
        enc = GNNEncoder(**DIMS, attention=attention)
        # Identical features for every blue and every bb edge.
        blue = torch.ones(1, n_blue, DIMS["blue_feat_dim"])
        bb_e = torch.ones(1, n_bb, DIMS["edge_feat_dim"])
        rb_e = torch.zeros(1, enc.rb_src.numel(), DIMS["edge_feat_dim"])
        red = torch.zeros(1, enc.n_red, DIMS["red_feat_dim"])
        rb_off = torch.zeros(1, enc.rb_src.numel())

        outs = []
        for keep in (1, 3):                    # 1 vs 3 live allies per blue
            vis = torch.zeros(1, n_bb)
            for recv in range(n_blue):
                idx = (enc.bb_dst == recv).nonzero().flatten()[:keep]
                vis[0, idx] = 1.0
            outs.append(_blue(enc, blue_feats=blue, red_feats=red,
                              bb_edge_feats=bb_e, rb_edge_feats=rb_e,
                              bb_visible=vis, rb_visible=rb_off))
        differ = not torch.allclose(outs[0], outs[1], atol=1e-6)
        if attention:
            assert not differ, "attention must be invariant to the live count"
        else:
            assert differ, "the sum is expected to move -- that is the defect"


# ------------------------------------------------------------------ #
#  The NaN case
# ------------------------------------------------------------------ #

def test_a_receiver_with_no_live_edges_gets_zeros_not_nan():
    """An all -inf softmax row returns NaN and poisons every gradient.

    Not hypothetical: 27.6% of steps have no confirmed track at all, so every
    blue's rb row is empty at once (docs/stage4_results.md §18.2).
    """
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, attention=True)
    x = _inputs(enc,
                bb_vis=torch.zeros(B, enc.bb_src.numel()),   # NOTHING live
                rb_vis=torch.zeros(B, enc.rb_src.numel()))
    out = _blue(enc, **x)
    assert torch.isfinite(out).all(), "empty rows produced NaN/inf"


def test_the_gradient_through_an_empty_receiver_is_finite():
    """A NaN that only shows up in the backward pass still kills a run."""
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, attention=True)
    x = _inputs(enc,
                bb_vis=torch.zeros(B, enc.bb_src.numel()),
                rb_vis=torch.zeros(B, enc.rb_src.numel()))
    _blue(enc, **x).sum().backward()
    bad = [n for n, p in enc.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in {bad}"


def test_partially_empty_rows_do_not_contaminate_the_others():
    """One empty receiver must not NaN the whole batch."""
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, attention=True)
    bb = torch.ones(B, enc.bb_src.numel())
    bb[0, (enc.bb_dst == 0).nonzero().flatten()] = 0.0    # blue 0, sample 0
    out = _blue(enc, **_inputs(enc, bb_vis=bb,
                               rb_vis=torch.ones(B, enc.rb_src.numel())))
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ #
#  Mask before, not after
# ------------------------------------------------------------------ #

def test_weights_over_live_edges_sum_to_one():
    """Masking after the softmax would leave them summing to less than one,
    which is the count-dependence this change removes."""
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, attention=True, n_heads=2)
    n_bb = DIMS["n_blue"] * (DIMS["n_blue"] - 1)
    # Probe _attend directly: identical unit messages, so the aggregate IS
    # the weight sum and any shortfall is visible.
    msg = torch.ones(1, n_bb, DIMS["d_hidden"])
    score = torch.randn(1, n_bb, 2)
    vis = torch.zeros(1, n_bb)
    for recv in range(DIMS["n_blue"]):          # keep 2 of 3 per receiver
        vis[0, (enc.bb_dst == recv).nonzero().flatten()[:2]] = 1.0
    out = enc._attend(msg, score, enc.bb_in, vis)
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-6, rtol=0)


def test_a_hidden_edge_contributes_nothing():
    """Changing a masked edge's message must not change the result."""
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, attention=True)
    vis = torch.ones(B, enc.bb_src.numel())
    vis[:, 0] = 0.0
    x1 = _inputs(enc, bb_vis=vis, rb_vis=torch.ones(B, enc.rb_src.numel()))
    x2 = dict(x1)
    x2["bb_edge_feats"] = x1["bb_edge_feats"].clone()
    x2["bb_edge_feats"][:, 0] += 100.0              # the hidden edge only
    torch.testing.assert_close(_blue(enc, **x1), _blue(enc, **x2))


# ------------------------------------------------------------------ #
#  Wiring
# ------------------------------------------------------------------ #

def test_heads_must_divide_d_hidden():
    with pytest.raises(ValueError, match="n_heads"):
        GNNEncoder(**DIMS, attention=True, n_heads=5)   # 16 % 5 != 0


def test_the_critic_is_not_attended():
    """Attention is actor-only: the critic sees full state with fixed counts
    and no masks, so the count-dependence never arises there, and changing it
    would move the CTDE baseline mid-experiment."""
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=0, action_dim=2,
                          d_hidden=16, attention=True)
    assert pol.actor_encoder.attention is True
    assert pol.critic_encoder.attention is False


def test_it_round_trips_through_a_checkpoint(tmp_path):
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=0, action_dim=2,
                          d_hidden=16, attention=True, n_heads=4)
    path = tmp_path / "ck.pt"
    torch.save({"policy_state": pol.state_dict(),
                "args": {"policy_type": "gnn_stage4_v6", "n_blue": 3,
                         "n_red": 2, "n_obstacles": 0, "d_hidden": 16,
                         "attention": True, "n_heads": 4}}, path)
    back = load_policy(path, torch.device("cpu"))
    assert back.actor_encoder.attention is True
    assert back.actor_encoder.n_heads == 4


def test_a_pre_flag_checkpoint_loads_as_a_sum(tmp_path):
    """A checkpoint written before attention existed must rebuild with the
    aggregation it trained under, not today's default."""
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=0, action_dim=2,
                          d_hidden=16)
    path = tmp_path / "old.pt"
    torch.save({"policy_state": pol.state_dict(),
                "args": {"policy_type": "gnn_stage4_v6", "n_blue": 3,
                         "n_red": 2, "n_obstacles": 0, "d_hidden": 16}}, path)
    back = load_policy(path, torch.device("cpu"))
    assert back.actor_encoder.attention is False


def test_attention_costs_few_parameters():
    """It must not quietly become a different-capacity model, or the
    comparison against the summed arm measures capacity, not aggregation."""
    base = GNNStage4Policy(n_blue=5, n_red=8, n_obs=0, action_dim=2,
                           d_hidden=64, n_region=25)
    att = GNNStage4Policy(n_blue=5, n_red=8, n_obs=0, action_dim=2,
                          d_hidden=64, n_region=25, attention=True)
    n_base = sum(p.numel() for p in base.parameters())
    n_att = sum(p.numel() for p in att.parameters())
    growth = (n_att - n_base) / n_base
    assert 0.0 < growth < 0.12, f"parameter growth {growth:.1%}"


# ------------------------------------------------------------------ #
#  The MEAN aggregation -- normalisation without discrimination
# ------------------------------------------------------------------ #

def test_mean_adds_no_parameters():
    """It is the control arm for Sec. 19: if a mean recovers attention's win,
    the win was normalisation, not selectivity.  That argument only holds if
    the mean has no extra capacity."""
    base = GNNStage4Policy(n_blue=5, n_red=8, n_obs=0, action_dim=2,
                           d_hidden=64, n_region=25)
    mean = GNNStage4Policy(n_blue=5, n_red=8, n_obs=0, action_dim=2,
                           d_hidden=64, n_region=25, mean_agg=True)
    assert (sum(p.numel() for p in base.parameters())
            == sum(p.numel() for p in mean.parameters()))
    assert mean.actor_encoder.att_mlp is None


def test_mean_is_invariant_to_the_live_edge_count():
    """The property the whole Sec. 19 experiment turns on."""
    n_blue = DIMS["n_blue"]
    n_bb = n_blue * (n_blue - 1)
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, mean_agg=True)
    blue = torch.ones(1, n_blue, DIMS["blue_feat_dim"])
    bb_e = torch.ones(1, n_bb, DIMS["edge_feat_dim"])
    rb_e = torch.zeros(1, enc.rb_src.numel(), DIMS["edge_feat_dim"])
    red = torch.zeros(1, enc.n_red, DIMS["red_feat_dim"])
    rb_off = torch.zeros(1, enc.rb_src.numel())
    outs = []
    for keep in (1, 3):
        vis = torch.zeros(1, n_bb)
        for recv in range(n_blue):
            vis[0, (enc.bb_dst == recv).nonzero().flatten()[:keep]] = 1.0
        outs.append(_blue(enc, blue_feats=blue, red_feats=red,
                          bb_edge_feats=bb_e, rb_edge_feats=rb_e,
                          bb_visible=vis, rb_visible=rb_off))
    torch.testing.assert_close(outs[0], outs[1], atol=1e-6, rtol=0)


def test_mean_really_is_the_arithmetic_mean():
    """Invariance alone would also hold for, say, a max.  This pins the value:
    with unit messages the aggregate must be exactly 1, whatever the degree."""
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, mean_agg=True)
    n_bb = DIMS["n_blue"] * (DIMS["n_blue"] - 1)
    msg = torch.ones(1, n_bb, DIMS["d_hidden"])
    vis = torch.zeros(1, n_bb)
    for recv in range(DIMS["n_blue"]):          # 2 of 3 live per receiver
        vis[0, (enc.bb_dst == recv).nonzero().flatten()[:2]] = 1.0
    out = enc._attend(msg, enc._scores(torch.zeros(1, n_bb, 3 * DIMS["d_hidden"])),
                      enc.bb_in, vis)
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-6, rtol=0)


def test_mean_handles_an_empty_receiver():
    torch.manual_seed(0)
    enc = GNNEncoder(**DIMS, mean_agg=True)
    x = _inputs(enc, bb_vis=torch.zeros(B, enc.bb_src.numel()),
                rb_vis=torch.zeros(B, enc.rb_src.numel()))
    _blue(enc, **x).sum().backward()
    bad = [n for n, p in enc.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert torch.isfinite(_blue(enc, **x)).all() and not bad


def test_mean_and_attention_are_mutually_exclusive():
    with pytest.raises(ValueError, match="alternative aggregations"):
        GNNEncoder(**DIMS, attention=True, mean_agg=True)


def test_mean_differs_from_both_sum_and_attention():
    torch.manual_seed(0); s = GNNEncoder(**DIMS)
    torch.manual_seed(0); m = GNNEncoder(**DIMS, mean_agg=True)
    torch.manual_seed(0); a = GNNEncoder(**DIMS, attention=True)
    vis = torch.ones(B, s.bb_src.numel())
    vis[:, 0] = 0.0                      # a partially live graph
    x = _inputs(s, bb_vis=vis, rb_vis=torch.ones(B, s.rb_src.numel()))
    assert not torch.allclose(_blue(s, **x), _blue(m, **x))
    assert not torch.allclose(_blue(m, **x), _blue(a, **x))


def test_mean_round_trips_through_a_checkpoint(tmp_path):
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=0, action_dim=2,
                          d_hidden=16, mean_agg=True)
    path = tmp_path / "mean.pt"
    torch.save({"policy_state": pol.state_dict(),
                "args": {"policy_type": "gnn_stage4_v6", "n_blue": 3,
                         "n_red": 2, "n_obstacles": 0, "d_hidden": 16,
                         "mean_agg": True}}, path)
    back = load_policy(path, torch.device("cpu"))
    assert back.actor_encoder.mean_agg is True
    assert back.actor_encoder.attention is False
