"""
isr/agents/red_motion_gnn.py — learned red-motion model (docs Sec. 8).

Predicts ``P(a_t | s_t, c_t)`` as a JOINT CATEGORICAL over a discretised
(heading, magnitude) grid, from a typed GNN embedding of the red's
situation.  Feeds the Gaussian-Sum tracker's ``motion_model`` plug point:
each significant grid cell becomes one predicted branch (docs Sec. 8.5).

Architecture: the SAME typed message-passing GNN as the blue RL policy
(``gnn_stage4_policy.py::GNNEncoder``), replicated SYMMETRICALLY with the
receiver role swapped.  That policy has blue nodes as the only receivers,
aggregating from bb / rb / ob edges; here RED nodes are the only
receivers, aggregating from:

    b2r   blue     -> red      (the dominant term: reds flee blues)
    o2r   obstacle -> red      (obstacle avoidance)

Everything else is deliberately identical to the proven version: separate
input MLP per node type, separate edge MLP per edge type, a SHARED message
MLP over ``[h_send, h_recv, e_edge]``, a residual node update on receivers
only, complete-bipartite static edge index buffers, and the same 7-D edge
feature convention as the env builds
(``rel_pos(2) + rel_vel(2) + range(1) + bearing_cos_sin(2)``, the bearing
measured between the RECEIVER's own velocity and the direction to the
sender).

One red node in, one categorical out — so a step with N active reds yields
N independent predictions, exactly matching the dataset's one-sample-per-
(step, red) layout.

Two deliberate departures from the blue policy, both because this models a
DIFFERENT thing:

* **No red->red edges.**  ``run_from_nearest_uav`` provably never reads
  another red's position or active flag (verified when building
  ``MixedStochasticRed``), so for the adversary we actually have they
  would carry zero signal and only cost parameters.  If a future red
  policy coordinates (a swarm evasion tactic, or self-play reds), adding
  an ``r2r`` edge type is the same pattern as the two above.
* **Masks mean PADDING, not visibility.**  The blue policy's
  ``*_visible`` masks encode its own sensors' partial observability.  The
  red heuristic reads every blue position with no sensor model, and
  training uses privileged ground truth (docs Sec. 8.1) — so masks here
  exist only to zero out padded, non-existent entities when team sizes
  vary across episodes.  Named ``*_active`` to keep that distinction
  visible.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# Single source of truth for the action discretisation.  Defining the bin
# counts here as well would let the head's width silently drift from the
# featuriser's labels -- a bug hit exactly once during development, when
# the ZERO class was added on one side only.
from isr.agents.red_motion_features import (
    N_BINS, N_HEADING_BINS, N_MAGNITUDE_BINS, ZERO_CLASS,
)


def _layer_init(layer: nn.Linear, std: float = 1.4142135623730951,
                bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden, out_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers.append(_layer_init(nn.Linear(prev, h)))
        layers.append(activation())
        prev = h
    layers.append(_layer_init(nn.Linear(prev, out_dim)))
    return nn.Sequential(*layers)


def _build_x2r_edges(n_src: int, n_red: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complete bipartite edges from ``n_src`` sender nodes to red nodes.

    Ordering matches the env's own convention for its x->blue edges
    (``for s in range(n_src): for r in range(n_red)``), so a featuriser
    can build edge tensors the same way for either graph.
    """
    src, dst = [], []
    for s in range(n_src):
        for r in range(n_red):
            src.append(s)
            dst.append(r)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))


class RedMotionGNN(nn.Module):
    """Typed GNN (reds as receivers) -> shared MLP -> joint categorical.

    Node features (all normalised by the collector's conventions):
      red      [vel(2), speed(1), wall_dist(4)]      = 7
        the red's OWN state.  No absolute position: the env's own
        ``test_obs_is_ego_centric_translation_invariant`` fixes
        translation invariance as a design property, and wall_dist already
        carries everything position-dependent the red policy actually
        reads.  No time-remaining either — the red heuristic never reads
        the step counter, so it would be a spurious input.
      blue     [active(1)]                            = 1
      obstacle [active(1), radius/L(1)]               = 2
        senders carry only what is NOT relational; their position and
        velocity live in the edge features, exactly as in the blue policy.
    """

    def __init__(
        self,
        n_blue:           int,
        n_red:            int,
        n_obs:            int = 0,
        red_feat_dim:     int = 7,
        blue_feat_dim:    int = 1,
        obs_feat_dim:     int = 2,
        edge_feat_dim:    int = 7,
        d_hidden:         int = 64,
        n_msg_rounds:     int = 2,
        trunk_hidden:     int = 128,
    ) -> None:
        super().__init__()
        self.n_blue = n_blue
        self.n_red = n_red
        self.n_obs = n_obs
        self.d_hidden = d_hidden
        self.n_msg_rounds = n_msg_rounds
        # Taken from red_motion_features, never redeclared here.
        self.n_heading_bins = N_HEADING_BINS
        self.n_magnitude_bins = N_MAGNITUDE_BINS
        self.n_bins = N_BINS
        self.zero_class = ZERO_CLASS

        self.red_input_mlp = _mlp(red_feat_dim, [d_hidden], d_hidden)
        self.blue_input_mlp = _mlp(blue_feat_dim, [d_hidden], d_hidden)
        self.b2r_edge_mlp = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.msg_mlp = _mlp(3 * d_hidden, [d_hidden], d_hidden)
        self.update_mlp = _mlp(2 * d_hidden, [d_hidden], d_hidden)

        b2r_src, b2r_dst = _build_x2r_edges(n_blue, n_red)
        self.register_buffer("b2r_src", b2r_src, persistent=False)
        self.register_buffer("b2r_dst", b2r_dst, persistent=False)

        if n_obs > 0:
            self.obs_input_mlp = _mlp(obs_feat_dim, [d_hidden], d_hidden)
            self.o2r_edge_mlp = _mlp(edge_feat_dim, [d_hidden], d_hidden)
            o2r_src, o2r_dst = _build_x2r_edges(n_obs, n_red)
            self.register_buffer("o2r_src", o2r_src, persistent=False)
            self.register_buffer("o2r_dst", o2r_dst, persistent=False)
        else:
            self.obs_input_mlp = None
            self.o2r_edge_mlp = None

        # Shared head: one red embedding -> logits over the joint
        # (heading, magnitude) grid PLUS the ZERO class, per docs Sec. 8.4.
        # A single flat softmax: each cell IS a possible mode, so
        # bimodality is native and needs no mixture machinery.  The extra
        # ZERO class covers |a| ~ 0, where the heading is undefined and
        # "do not accelerate" is a genuine discrete outcome (1.78% of
        # collected samples).  Small head gain: logits start near-uniform
        # rather than committing hard before any training.
        self.head = _mlp(d_hidden, [trunk_hidden, trunk_hidden], self.n_bins)
        _layer_init(self.head[-1], std=0.01)

    def forward(
        self,
        red_feats:     torch.Tensor,                    # (B, n_red, red_feat_dim)
        blue_feats:    torch.Tensor,                    # (B, n_blue, blue_feat_dim)
        b2r_edge_feats: torch.Tensor,                   # (B, n_b2r, edge_feat_dim)
        obs_feats:     Optional[torch.Tensor] = None,   # (B, n_obs, obs_feat_dim)
        o2r_edge_feats: Optional[torch.Tensor] = None,  # (B, n_o2r, edge_feat_dim)
        b2r_active:    Optional[torch.Tensor] = None,   # (B, n_b2r)
        o2r_active:    Optional[torch.Tensor] = None,   # (B, n_o2r)
    ) -> torch.Tensor:
        """Returns logits ``(B, n_red, n_heading_bins * n_magnitude_bins)``.

        One categorical per RED node — a step with N active reds gives N
        independent predictions.  Padded reds still produce logits; the
        caller selects the active ones (the dataset stores one sample per
        active red, so this is a non-issue there).
        """
        B = red_feats.shape[0]
        d = self.d_hidden

        h_red = self.red_input_mlp(red_feats)
        h_blue = self.blue_input_mlp(blue_feats)
        e_b2r = self.b2r_edge_mlp(b2r_edge_feats)

        has_obs = self.n_obs > 0 and obs_feats is not None
        if has_obs:
            h_obs = self.obs_input_mlp(obs_feats)
            e_o2r = self.o2r_edge_mlp(o2r_edge_feats)

        for _ in range(self.n_msg_rounds):
            h_send = h_blue.index_select(1, self.b2r_src)
            h_recv = h_red.index_select(1, self.b2r_dst)
            msg_b2r = self.msg_mlp(torch.cat([h_send, h_recv, e_b2r], dim=-1))
            if b2r_active is not None:
                msg_b2r = msg_b2r * b2r_active.unsqueeze(-1)

            agg = torch.zeros(B, self.n_red, d,
                             device=h_red.device, dtype=h_red.dtype)
            agg.index_add_(1, self.b2r_dst, msg_b2r)

            if has_obs:
                h_send_o = h_obs.index_select(1, self.o2r_src)
                h_recv_o = h_red.index_select(1, self.o2r_dst)
                msg_o2r = self.msg_mlp(
                    torch.cat([h_send_o, h_recv_o, e_o2r], dim=-1))
                if o2r_active is not None:
                    msg_o2r = msg_o2r * o2r_active.unsqueeze(-1)
                agg.index_add_(1, self.o2r_dst, msg_o2r)

            # Residual update on RED receivers only — reds are the sole
            # receivers here exactly as blues are in the policy's version.
            h_red = h_red + self.update_mlp(torch.cat([h_red, agg], dim=-1))

        return self.head(h_red)

    # ---------------- readout helpers ---------------- #

    def log_probs(self, *args, **kwargs) -> torch.Tensor:
        """Log-softmax over the joint grid, shape (B, n_red, n_bins)."""
        return torch.log_softmax(self.forward(*args, **kwargs), dim=-1)

    def grid_probs(self, *args, **kwargs
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(grid, p_zero)`` — the (heading, magnitude) probabilities
        reshaped to (B, n_red, n_heading, n_magnitude), and the ZERO
        class's probability (B, n_red) alongside it.

        For inspection and marginals only; the model itself is one flat
        joint softmax over grid + ZERO.  They are returned separately
        because the ZERO class has no heading and so no place in the grid
        — folding it in would invent a direction it does not have.
        """
        p = torch.softmax(self.forward(*args, **kwargs), dim=-1)
        grid = p[..., :self.zero_class].reshape(
            *p.shape[:-1], self.n_heading_bins, self.n_magnitude_bins)
        return grid, p[..., self.zero_class]
