"""
isr/agents/gnn_policy.py — Stage 2 GNN actor-critic.

Graph neural network policy over a typed entity graph (blue UAVs,
red targets).  See ``docs/stage2_gnn_design.md`` for the full spec.

Design summary:
- Two node types (blue: 8-D intrinsic features; red: 1-D active flag).
- Two edge types (blue-blue bidirectional 7-D; red-blue directed 7-D).
- Per-type input embedding MLPs (blue -> d_hidden, red -> d_hidden).
- Per-type edge embedding MLPs.
- 2 rounds of message passing with a shared message MLP and shared
  residual update MLP.
- Decentralised policy head: one shared actor MLP applied to each
  blue node's final embedding -> action mean per blue.  Log_std is a
  state-independent learned parameter (same convention as the v2 MLP
  policy).
- Centralised critic head (CTDE): sum of final blue embeddings
  concatenated with final red embeddings -> single V per graph state.

The forward pass consumes ONE dict per env-timestep and produces
action distributions for every blue simultaneously.  Rollout
collection therefore calls the policy once per env step instead of
n_blue times.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers (mirror the v2 MLP init conventions so parameter counts are matched)
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Linear, std: float = 1.4142135623730951,
                bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden: List[int], out_dim: int,
         activation=nn.Tanh, head_gain: float = 1.4142135623730951,
         hidden_gain: float = 1.4142135623730951) -> nn.Sequential:
    """
    Small MLP with a final linear projection.  All hidden layers use
    the standard tanh activation + orthogonal init at gain sqrt(2);
    the head gets its own gain (head_gain arg).
    """
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(_layer_init(nn.Linear(prev, h), std=hidden_gain))
        layers.append(activation())
        prev = h
    layers.append(_layer_init(nn.Linear(prev, out_dim), std=head_gain))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Graph edge index tables (built once at construction, registered as buffers)
# ---------------------------------------------------------------------------

def _build_bb_edges(n_blue: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst = [], []
    for s in range(n_blue):
        for d in range(n_blue):
            if s != d:
                src.append(s); dst.append(d)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))


def _build_rb_edges(n_red: int, n_blue: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst = [], []
    for r in range(n_red):
        for b in range(n_blue):
            src.append(r); dst.append(b)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))


# ---------------------------------------------------------------------------
# GNN Actor-Critic
# ---------------------------------------------------------------------------

class GNNActorCritic(nn.Module):
    """
    GNN policy + CTDE critic operating on the ISR entity graph.

    Interface mirrors ``ActorCritic`` (the v2 MLP) so the same PPO
    update code works: ``get_action_and_value(obs_dict, action=None)``
    returns a (action, log_prob, entropy, value) 4-tuple.  Shapes:
        obs_dict has batched tensors:
            blue_features:    (B, N_blue, blue_feat_dim)
            red_features:     (B, N_red,  red_feat_dim)
            bb_edge_features: (B, n_bb,   edge_feat_dim)
            rb_edge_features: (B, n_rb,   edge_feat_dim)
        action:   (B, N_blue, action_dim)  or None to sample
        returns:
            action:   (B, N_blue, action_dim)
            log_prob: (B, N_blue)   — per-agent (summed over action dim)
            entropy:  (B, N_blue)   — per-agent
            value:    (B,)          — one centralised V per state
    """

    def __init__(
        self,
        n_blue:         int,
        n_red:          int,
        blue_feat_dim:  int = 8,
        red_feat_dim:   int = 1,
        edge_feat_dim:  int = 7,
        action_dim:     int = 2,
        d_hidden:       int = 64,
        n_msg_rounds:   int = 2,
        init_log_std:   float = 0.0,
    ) -> None:
        super().__init__()
        self.n_blue         = n_blue
        self.n_red          = n_red
        self.d_hidden       = d_hidden
        self.n_msg_rounds   = n_msg_rounds
        self.action_dim     = action_dim

        # ---- Per-type input embedding MLPs -------------------------------
        # 2 hidden layers -> d_hidden output.  Head gain sqrt(2) since these
        # are trunk-style modules whose outputs feed further layers.
        self.blue_input_mlp = _mlp(blue_feat_dim, [d_hidden], d_hidden)
        self.red_input_mlp  = _mlp(red_feat_dim,  [d_hidden], d_hidden)
        self.bb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.rb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)

        # ---- Shared message + update MLPs (used every round) -------------
        # msg([h_sender, h_receiver, e_ij]) -> d_hidden
        self.msg_mlp    = _mlp(3 * d_hidden, [d_hidden], d_hidden)
        # update([h_old, aggregated_msg]) -> d_hidden (added residually)
        self.update_mlp = _mlp(2 * d_hidden, [d_hidden], d_hidden)

        # ---- Fixed graph structure (registered so it moves w/ .to(device)) --
        bb_src, bb_dst = _build_bb_edges(n_blue)
        rb_src, rb_dst = _build_rb_edges(n_red, n_blue)
        self.register_buffer("bb_src", bb_src, persistent=False)
        self.register_buffer("bb_dst", bb_dst, persistent=False)
        self.register_buffer("rb_src", rb_src, persistent=False)
        self.register_buffer("rb_dst", rb_dst, persistent=False)

        # ---- Actor head — decentralised execution ------------------------
        # Small-gain final projection (v2 convention: 0.01) so early
        # actions are near zero and exploration is reasonable at step 1.
        self.actor_mean = _layer_init(nn.Linear(d_hidden, action_dim), std=0.01)
        # State-independent per-dim log_std, learned.
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32)
        )

        # ---- Critic head — CTDE (sum blue + concat red) ------------------
        critic_in = d_hidden + n_red * d_hidden       # sum_blue + concat_red
        self.critic_trunk = _mlp(critic_in, [d_hidden], d_hidden)
        self.critic_head  = _layer_init(nn.Linear(d_hidden, 1), std=1.0)

    # ------------------------------------------------------------------ #
    #  Graph forward                                                       #
    # ------------------------------------------------------------------ #

    def _graph_forward(
        self,
        blue_feats:  torch.Tensor,   # (B, N_blue, blue_feat_dim)
        red_feats:   torch.Tensor,   # (B, N_red,  red_feat_dim)
        bb_edge_feats: torch.Tensor, # (B, n_bb,   edge_feat_dim)
        rb_edge_feats: torch.Tensor, # (B, n_rb,   edge_feat_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the full GNN forward (input embed + n_msg_rounds message
        passing) and returns the final blue and red node embeddings.

        Returns:
            h_blue: (B, N_blue, d_hidden)
            h_red:  (B, N_red,  d_hidden)
        """
        B = blue_feats.shape[0]
        d = self.d_hidden

        # ---- Input embedding --------------------------------------------
        h_blue = self.blue_input_mlp(blue_feats)     # (B, N_blue, d)
        h_red  = self.red_input_mlp(red_feats)       # (B, N_red,  d)
        e_bb   = self.bb_edge_mlp(bb_edge_feats)     # (B, n_bb,   d)
        e_rb   = self.rb_edge_mlp(rb_edge_feats)     # (B, n_rb,   d)

        # ---- Message passing rounds -------------------------------------
        for _ in range(self.n_msg_rounds):
            # Blue-Blue directed messages
            h_sender_bb   = h_blue.index_select(1, self.bb_src)   # (B, n_bb, d)
            h_receiver_bb = h_blue.index_select(1, self.bb_dst)   # (B, n_bb, d)
            msg_bb = self.msg_mlp(
                torch.cat([h_sender_bb, h_receiver_bb, e_bb], dim=-1)
            )                                                     # (B, n_bb, d)

            # Red-Blue directed messages (red senders, blue receivers)
            h_sender_rb   = h_red.index_select(1, self.rb_src)    # (B, n_rb, d)
            h_receiver_rb = h_blue.index_select(1, self.rb_dst)   # (B, n_rb, d)
            msg_rb = self.msg_mlp(
                torch.cat([h_sender_rb, h_receiver_rb, e_rb], dim=-1)
            )                                                     # (B, n_rb, d)

            # Aggregate SUM messages by blue receiver
            agg = torch.zeros(B, self.n_blue, d, device=h_blue.device,
                              dtype=h_blue.dtype)
            # index_add expects (dim, index, source); index over the node
            # dimension = dim 1 for our (B, N_blue, d) tensor.
            agg.index_add_(1, self.bb_dst, msg_bb)
            agg.index_add_(1, self.rb_dst, msg_rb)

            # Residual node update on blues (reds don't receive edges).
            h_blue = h_blue + self.update_mlp(
                torch.cat([h_blue, agg], dim=-1)
            )

        return h_blue, h_red

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (mean, log_std_expanded, value).

        mean:    (B, N_blue, action_dim)
        log_std: (B, N_blue, action_dim) — broadcast from state-independent
                                            learned parameter
        value:   (B,)                     — one centralised V per state
        """
        h_blue, h_red = self._graph_forward(
            obs["blue_features"], obs["red_features"],
            obs["bb_edge_features"], obs["rb_edge_features"],
        )
        B = h_blue.shape[0]

        # Actor: per-blue action mean from own node embedding.
        mean = self.actor_mean(h_blue)                         # (B, N_blue, action_dim)
        # Broadcast the state-independent log_std to (B, N_blue, action_dim).
        log_std = self.actor_log_std.view(1, 1, -1).expand_as(mean)

        # Critic: sum blue embeddings + concat red embeddings.
        blue_summary = h_blue.sum(dim=1)                       # (B, d)
        red_summary  = h_red.reshape(B, -1)                    # (B, N_red * d)
        state_repr   = torch.cat([blue_summary, red_summary], dim=-1)
        value = self.critic_head(self.critic_trunk(state_repr)).squeeze(-1)  # (B,)

        return mean, log_std, value

    def get_action_and_value(
        self,
        obs:    Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (if action is None) or evaluate the given action.

        Returns (action, log_prob, entropy, value):
            action:   (B, N_blue, action_dim)
            log_prob: (B, N_blue) — per-agent, sum over action_dim
            entropy:  (B, N_blue) — per-agent, sum over action_dim
            value:    (B,)        — centralised, one per graph state
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)   # (B, N_blue)
        entropy  = dist.entropy().sum(-1)          # (B, N_blue)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Greedy: return action mean (shape (B, N_blue, action_dim))."""
        mean, _, _ = self.forward(obs)
        return mean
