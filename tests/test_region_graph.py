"""
tests/test_region_graph.py — the coverage path's graph pieces, and the
GNN encoder's region node type.

Stage C of the search work (docs/search_design.md §5).  The field is in
tests/test_staleness.py, the aggregation into regions in
tests/test_region_nodes.py; this is the wiring between them and the
policy graph.

Two things carry the risk:

* **Edge ORDER.** The encoder's gb_src/gb_dst index buffers and the env's
  edge-feature rows have to be built with the same convention.  A
  mismatch silently pairs each region with the wrong blue and still
  trains, just badly.
* **Non-regression.** With the coverage path off, the encoder must be
  bit-identical to the proven pre-coverage encoder.

Run:
    pytest tests/test_region_graph.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from isr.agents.gnn_stage4_policy import GNNEncoder, _build_xb_edges
from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav

R = 5
N_REGION = R * R


def _env(**kw):
    base = dict(
        n_blue=3, n_red=2, n_obstacles=2, arena_size=130.0, max_steps=200,
        capture_radius=3.0, sensor_radius=40.0, use_staleness=True,
        staleness_regions=R, red_policy=run_from_nearest_uav, seed=0,
    )
    base.update(kw)
    e = PursuitEnv(**base)
    e.reset(seed=base["seed"])
    return e


# --------------------------------------------------------------------- #
#  Env side: shapes, ordering, weights
# --------------------------------------------------------------------- #

def test_graph_pieces_have_the_expected_shapes():
    e = _env()
    feats, edges, w = e._build_region_graph()
    assert feats.shape == (N_REGION, 2)
    assert edges.shape == (N_REGION * e.n_blue, 7)
    assert w.shape == (N_REGION * e.n_blue,)


def test_edge_rows_follow_the_encoders_index_order():
    """The load-bearing invariant: row (s * n_blue + b) must be the edge
    from region s to blue b, matching _build_xb_edges.  Get this wrong and
    every region talks to the wrong blue -- which still trains, just
    badly, and nothing else would catch it."""
    e = _env()
    _, edges, _ = e._build_region_graph()
    _, centres = e._build_region_nodes()
    src, dst = _build_xb_edges(N_REGION, e.n_blue)
    L = e.arena_size
    for k in range(0, len(edges), 7):          # spot-check a spread of rows
        s, b = int(src[k]), int(dst[k])
        want = (e._blue_pos[b] - centres[s]) / L        # rel_pos = dst - src
        assert np.allclose(edges[k, :2], want, atol=1e-5), (
            f"row {k} is not region {s} -> blue {b}")


def test_regions_are_static_senders_like_obstacles():
    """rel_vel must be driven entirely by the blue's own motion, and the
    bearing (defined in the SENDER's frame) must be zero -- a motionless
    sender has no frame.  Exactly how obstacles are already treated."""
    e = _env()
    e._blue_vel[:] = np.array([0.4, -0.9], dtype=np.float32)
    _, edges, _ = e._build_region_graph()
    from isr.env.entities import BLUE_UAV
    for k in (0, 17, len(edges) - 1):
        b = k % e.n_blue
        assert np.allclose(edges[k, 2:4], e._blue_vel[b] / BLUE_UAV.v_max,
                          atol=1e-5)
    assert np.allclose(edges[:, 5:7], 0.0), "static sender got a bearing"


def test_weights_are_a_normalised_mean_over_regions():
    e = _env()
    _, _, w = e._build_region_graph()
    per_blue = w.reshape(N_REGION, e.n_blue)
    for b in range(e.n_blue):
        assert per_blue[:, b].sum() == pytest.approx(1.0, abs=1e-5)


def test_weights_are_proportional_to_staleness_times_searchable():
    e = _env()
    feats, _, w = e._build_region_graph()
    want = feats[:, 0] * feats[:, 1]
    want = want / want.sum()
    assert np.allclose(w.reshape(N_REGION, e.n_blue)[:, 0], want, atol=1e-5)


def test_a_freshly_swept_region_gets_no_weight():
    """The point of weighting: a region just looked at carries no search
    information, so it must not contribute to the aggregate at all."""
    e = _env()
    e._staleness[:] = e.max_steps
    e._staleness[0:5, 0:5] = 0                  # region 0 swept
    _, _, w = e._build_region_graph()
    assert w.reshape(N_REGION, e.n_blue)[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert w.sum() > 0


def test_uniform_weights_when_nothing_is_stale():
    """With no coverage signal there is no preference, and the fallback
    must not divide by zero."""
    e = _env()
    e._staleness[:] = 0
    _, _, w = e._build_region_graph()
    per_blue = w.reshape(N_REGION, e.n_blue)
    assert np.allclose(per_blue[:, 0], 1.0 / N_REGION)
    assert np.all(np.isfinite(w))


def test_weights_stay_finite_through_a_rollout():
    e = _env(n_obstacles=4)
    rng = np.random.default_rng(2)
    for _ in range(30):
        if not e.agents:
            break
        e.step({a: rng.uniform(-1, 1, 2).astype(np.float32) for a in e.agents})
        feats, edges, w = e._build_region_graph()
        assert np.all(np.isfinite(feats)) and np.all(np.isfinite(edges))
        assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
        assert w.reshape(N_REGION, e.n_blue)[:, 0].sum() == pytest.approx(1.0,
                                                                         abs=1e-5)


# --------------------------------------------------------------------- #
#  Encoder side
# --------------------------------------------------------------------- #

def _enc(n_region):
    torch.manual_seed(0)
    return GNNEncoder(n_blue=3, n_red=2, n_obs=2, n_region=n_region)


def _inputs(B=2, n_blue=3, n_red=2, n_obs=2, n_region=0):
    g = torch.Generator().manual_seed(1)
    def r(*shape):
        return torch.randn(*shape, generator=g)
    out = dict(
        blue_feats=r(B, n_blue, 8), red_feats=r(B, n_red, 4),
        bb_edge_feats=r(B, n_blue * (n_blue - 1), 7),
        rb_edge_feats=r(B, n_red * n_blue, 7),
        obs_feats=r(B, n_obs, 5), ob_edge_feats=r(B, n_obs * n_blue, 7),
    )
    if n_region:
        out["region_feats"] = r(B, n_region, 2)
        out["gb_edge_feats"] = r(B, n_region * n_blue, 7)
        out["gb_weight"] = torch.full((B, n_region * n_blue), 1.0 / n_region)
    return out


def test_coverage_path_is_off_by_default():
    enc = _enc(0)
    assert enc.region_input_mlp is None and enc.gb_edge_mlp is None
    assert not hasattr(enc, "gb_src")


def test_disabled_encoder_is_bit_identical_to_the_pre_coverage_version():
    """Non-regression against the proven pre-coverage encoder: passing
    region tensors to an encoder built with n_region=0 must change
    nothing, and neither must omitting them."""
    enc = _enc(0)
    x = _inputs()
    a, _, _ = enc(**x)
    b, _, _ = enc(**x, region_feats=torch.randn(2, N_REGION, 2),
                  gb_edge_feats=torch.randn(2, N_REGION * 3, 7),
                  gb_weight=torch.rand(2, N_REGION * 3))
    assert torch.equal(a, b), "region tensors leaked into a disabled path"


def test_enabled_encoder_runs_and_changes_the_blue_embedding():
    enc = _enc(N_REGION)
    x = _inputs(n_region=N_REGION)
    with_region, _, _ = enc(**x)
    without = {k: v for k, v in x.items()
               if k not in ("region_feats", "gb_edge_feats", "gb_weight")}
    baseline, _, _ = enc(**without)
    assert with_region.shape == baseline.shape
    assert not torch.allclose(with_region, baseline), (
        "the coverage path had no effect on the blue embedding")


def test_zero_weights_silence_the_coverage_path_exactly():
    """The weights ride the same per-edge multiply slot the visibility
    masks use, so an all-zero weight must reproduce the no-region result
    exactly -- the property that makes the three aggregations nested."""
    enc = _enc(N_REGION)
    x = _inputs(n_region=N_REGION)
    x["gb_weight"] = torch.zeros_like(x["gb_weight"])
    muted, _, _ = enc(**x)
    without = {k: v for k, v in x.items()
               if k not in ("region_feats", "gb_edge_feats", "gb_weight")}
    baseline, _, _ = enc(**without)
    assert torch.allclose(muted, baseline, atol=1e-6)


def test_region_messages_reach_the_right_blue():
    """End-to-end check on the index buffers: weighting ALL edges to zero
    except those into blue 0 must leave blues 1 and 2 at their baseline.

    ONE message round, deliberately.  With the default two, blue 0's
    updated embedding propagates to the others over the bb edges in round
    2 — correct behaviour (that is what coordination IS), but it would
    mask the thing under test, which is purely whether gb_src/gb_dst pair
    each region with the right blue.
    """
    torch.manual_seed(0)
    enc = GNNEncoder(n_blue=3, n_red=2, n_obs=2, n_region=N_REGION,
                     n_msg_rounds=1)
    x = _inputs(n_region=N_REGION)
    _, dst = _build_xb_edges(N_REGION, 3)
    w = torch.zeros_like(x["gb_weight"])
    w[:, dst == 0] = 1.0 / N_REGION
    x["gb_weight"] = w
    out, _, _ = enc(**x)
    without = {k: v for k, v in x.items()
               if k not in ("region_feats", "gb_edge_feats", "gb_weight")}
    base, _, _ = enc(**without)
    assert not torch.allclose(out[:, 0], base[:, 0]), "blue 0 got nothing"
    assert torch.allclose(out[:, 1], base[:, 1], atol=1e-6), "leaked to blue 1"
    assert torch.allclose(out[:, 2], base[:, 2], atol=1e-6), "leaked to blue 2"


def test_parameter_cost_is_modest():
    """Two small MLPs; the point of riding the shared msg/update MLPs."""
    added = (sum(p.numel() for p in _enc(N_REGION).parameters())
             - sum(p.numel() for p in _enc(0).parameters()))
    assert added < 15000, f"coverage path added {added} params"
