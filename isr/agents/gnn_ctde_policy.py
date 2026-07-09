"""
isr/agents/gnn_ctde_policy.py — Stage 3 GNN + GRU + CTDE actor-critic.

Two-view architecture per docs/stage3_design.md:

- **Actor path** (decentralised execution).  Consumes the
  partial-obs graph (visibility-masked edges) → shared GNN encoder →
  per-blue GRUCell for belief tracking → shared actor MLP → action
  mean.  ``log_std`` is a state-independent shared parameter (same
  convention as Stage 2, see stage2_gnn_design.md §3.3).

- **Critic path** (centralised training).  Consumes the full-state
  graph → separate GNN encoder (byte-identical architecture to
  Stage 2) → sum-of-blue + concat-of-red → critic MLP → V(state).
  No memory: the full state is Markov by construction.

The two GNN encoders have the **same architecture but separate
weights**.  Rationale (see design §4):
- They solve slightly different problems (encoding a masked
  subgraph vs a full graph).
- Warm-starting the critic path from a Stage 2 GNN checkpoint is
  clean when the weights are not tied — no accidental leakage into
  the actor path.

Interface mirrors ``GNNActorCritic`` but ``get_action_and_value``
now takes and returns a hidden state:

    (action, log_prob, entropy, value, new_hidden) = policy.get_action_and_value(
        partial_obs, full_state, hidden, action=None
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers (same init conventions as the Stage 2 GNN so warm-start is byte-safe)
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Linear, std: float = 1.4142135623730951,
                bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden: List[int], out_dim: int,
         activation=nn.Tanh, head_gain: float = 1.4142135623730951,
         hidden_gain: float = 1.4142135623730951) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(_layer_init(nn.Linear(prev, h), std=hidden_gain))
        layers.append(activation())
        prev = h
    layers.append(_layer_init(nn.Linear(prev, out_dim), std=head_gain))
    return nn.Sequential(*layers)


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
# GNN encoder submodule (reused for actor + critic paths)
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """
    2-round message-passing GNN over the typed entity graph, matching
    the Stage 2 architecture.  Weights are byte-compatible with
    Stage 2's inline encoder so warm-start via ``load_stage2_critic``
    is direct.

    ``forward`` accepts optional edge visibility masks; multiplied
    into the messages before ``index_add_`` aggregation so hidden
    edges contribute zero.  When both masks are None (default), the
    encoder is fully observable — identical semantics to Stage 2.
    """

    def __init__(
        self,
        n_blue:         int,
        n_red:          int,
        blue_feat_dim:  int = 8,
        red_feat_dim:   int = 1,
        edge_feat_dim:  int = 7,
        d_hidden:       int = 64,
        n_msg_rounds:   int = 2,
    ) -> None:
        super().__init__()
        self.n_blue       = n_blue
        self.n_red        = n_red
        self.d_hidden     = d_hidden
        self.n_msg_rounds = n_msg_rounds

        self.blue_input_mlp = _mlp(blue_feat_dim, [d_hidden], d_hidden)
        self.red_input_mlp  = _mlp(red_feat_dim,  [d_hidden], d_hidden)
        self.bb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.rb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.msg_mlp        = _mlp(3 * d_hidden, [d_hidden], d_hidden)
        self.update_mlp     = _mlp(2 * d_hidden, [d_hidden], d_hidden)

        bb_src, bb_dst = _build_bb_edges(n_blue)
        rb_src, rb_dst = _build_rb_edges(n_red, n_blue)
        self.register_buffer("bb_src", bb_src, persistent=False)
        self.register_buffer("bb_dst", bb_dst, persistent=False)
        self.register_buffer("rb_src", rb_src, persistent=False)
        self.register_buffer("rb_dst", rb_dst, persistent=False)

    def forward(
        self,
        blue_feats:    torch.Tensor,      # (B, N_blue, blue_feat_dim)
        red_feats:     torch.Tensor,      # (B, N_red,  red_feat_dim)
        bb_edge_feats: torch.Tensor,      # (B, n_bb,   edge_feat_dim)
        rb_edge_feats: torch.Tensor,      # (B, n_rb,   edge_feat_dim)
        bb_visible:    Optional[torch.Tensor] = None,   # (B, n_bb)
        rb_visible:    Optional[torch.Tensor] = None,   # (B, n_rb)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns final blue and red node embeddings after
        ``n_msg_rounds`` rounds of message passing.
        """
        B = blue_feats.shape[0]
        d = self.d_hidden

        h_blue = self.blue_input_mlp(blue_feats)
        h_red  = self.red_input_mlp(red_feats)
        e_bb   = self.bb_edge_mlp(bb_edge_feats)
        e_rb   = self.rb_edge_mlp(rb_edge_feats)

        for _ in range(self.n_msg_rounds):
            # Blue-Blue messages
            h_sender_bb   = h_blue.index_select(1, self.bb_src)
            h_receiver_bb = h_blue.index_select(1, self.bb_dst)
            msg_bb = self.msg_mlp(
                torch.cat([h_sender_bb, h_receiver_bb, e_bb], dim=-1)
            )
            if bb_visible is not None:
                msg_bb = msg_bb * bb_visible.unsqueeze(-1)

            # Red-Blue messages
            h_sender_rb   = h_red.index_select(1, self.rb_src)
            h_receiver_rb = h_blue.index_select(1, self.rb_dst)
            msg_rb = self.msg_mlp(
                torch.cat([h_sender_rb, h_receiver_rb, e_rb], dim=-1)
            )
            if rb_visible is not None:
                msg_rb = msg_rb * rb_visible.unsqueeze(-1)

            # Aggregate by receiver (blue nodes only receive).
            agg = torch.zeros(B, self.n_blue, d,
                              device=h_blue.device, dtype=h_blue.dtype)
            agg.index_add_(1, self.bb_dst, msg_bb)
            agg.index_add_(1, self.rb_dst, msg_rb)

            # Residual node update on blues.
            h_blue = h_blue + self.update_mlp(
                torch.cat([h_blue, agg], dim=-1)
            )
            # Reds don't update (no incoming edges for reds in this env).

        return h_blue, h_red


# ---------------------------------------------------------------------------
# Stage 3 CTDE actor-critic
# ---------------------------------------------------------------------------

class GNNCTDEPolicy(nn.Module):
    """
    Stage 3 policy: GNN + GRU actor + centralised (CTDE) GNN critic.

    Shapes:
        partial_obs (dict):
            blue_features    (B, N_blue, blue_feat_dim)
            red_features     (B, N_red,  red_feat_dim)
            bb_edge_features (B, n_bb,   edge_feat_dim)
            rb_edge_features (B, n_rb,   edge_feat_dim)
            bb_edge_visible  (B, n_bb)
            rb_edge_visible  (B, n_rb)
        full_state (dict):
            same 4 feature fields; no visibility masks (full obs).
        hidden       (B, N_blue, d_hidden)   — actor GRU state at time t
        action       (B, N_blue, action_dim) — optional; sample if None
    Returns from get_action_and_value:
        action       (B, N_blue, action_dim)
        log_prob     (B, N_blue)        — per-agent, sum over action_dim
        entropy      (B, N_blue)        — per-agent
        value        (B,)                — centralised, one per state
        new_hidden   (B, N_blue, d_hidden) — post-GRU state
    """

    def __init__(
        self,
        n_blue:            int,
        n_red:             int,
        blue_feat_dim:     int   = 8,
        red_feat_dim:      int   = 1,
        edge_feat_dim:     int   = 7,
        action_dim:        int   = 2,
        d_hidden:          int   = 64,
        n_msg_rounds:      int   = 2,
        init_log_std:      float = 0.0,
        use_hidden_in_gnn: bool  = False,
    ) -> None:
        """
        ``use_hidden_in_gnn`` (Phase 3.6, docs/stage3_design.md §13):
        when True, the previous-step per-blue GRU hidden state is
        concatenated with ``blue_features`` before the actor's GNN
        encoder is called.  The GNN's bb messages then carry belief
        state across blues (option 1 for cross-blue coordination).
        Only the actor path is affected — the CTDE critic still
        consumes plain blue features on full state.
        """
        super().__init__()
        self.n_blue            = n_blue
        self.n_red             = n_red
        self.d_hidden          = d_hidden
        self.action_dim        = action_dim
        self.use_hidden_in_gnn = use_hidden_in_gnn

        # Actor encoder blue-input dim depends on whether we prepend
        # the previous hidden state to the blue node features.
        actor_blue_feat_dim = (
            blue_feat_dim + d_hidden if use_hidden_in_gnn else blue_feat_dim
        )

        # ---- Actor path ----------------------------------------------
        self.actor_encoder = GNNEncoder(
            n_blue        = n_blue,
            n_red         = n_red,
            blue_feat_dim = actor_blue_feat_dim,
            red_feat_dim  = red_feat_dim,
            edge_feat_dim = edge_feat_dim,
            d_hidden      = d_hidden,
            n_msg_rounds  = n_msg_rounds,
        )
        # Per-blue GRU (parameter-shared across UAVs — the same GRUCell
        # is applied to each blue's node embedding independently).
        self.actor_gru = nn.GRUCell(input_size=d_hidden, hidden_size=d_hidden)
        # Actor head (per-blue action mean from post-GRU hidden state).
        self.actor_mean = _layer_init(nn.Linear(d_hidden, action_dim), std=0.01)
        # State-independent per-dim learned log_std, shared across all
        # blues and all batch elements.  See stage2_gnn_design.md §3.3.
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32)
        )

        # ---- Critic path (CTDE — same architecture as Stage 2 critic) ----
        self.critic_encoder = GNNEncoder(
            n_blue        = n_blue,
            n_red         = n_red,
            blue_feat_dim = blue_feat_dim,
            red_feat_dim  = red_feat_dim,
            edge_feat_dim = edge_feat_dim,
            d_hidden      = d_hidden,
            n_msg_rounds  = n_msg_rounds,
        )
        critic_in = d_hidden + n_red * d_hidden       # sum_blue + concat_red
        self.critic_trunk = _mlp(critic_in, [d_hidden], d_hidden)
        self.critic_head  = _layer_init(nn.Linear(d_hidden, 1), std=1.0)

    # ------------------------------------------------------------------ #
    #  Sub-forwards                                                        #
    # ------------------------------------------------------------------ #

    def actor_encode(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,       # (B, N_blue, d_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the actor's GNN encoder on ``partial_obs``, optionally
        with the previous per-blue hidden state prepended to the
        blue node features (Phase 3.6 cross-blue belief sharing).

        Returns (h_blue, h_red) — the pre-GRU node embeddings.  Shared
        between ``actor_forward`` and the aux belief-state loss so
        both paths see the same encoder input.
        """
        if self.use_hidden_in_gnn:
            blue_input = torch.cat(
                [partial_obs["blue_features"], hidden], dim=-1,
            )
        else:
            blue_input = partial_obs["blue_features"]
        return self.actor_encoder(
            blue_input,
            partial_obs["red_features"],
            partial_obs["bb_edge_features"],
            partial_obs["rb_edge_features"],
            bb_visible=partial_obs["bb_edge_visible"],
            rb_visible=partial_obs["rb_edge_visible"],
        )

    def actor_forward(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,       # (B, N_blue, d_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actor forward pass: partial-obs GNN -> GRU per blue -> mean head.

        Returns (mean, log_std_expanded, new_hidden).
        """
        h_blue, _ = self.actor_encode(partial_obs, hidden)   # (B, N_blue, d)
        B = h_blue.shape[0]
        d = self.d_hidden

        # Apply GRUCell per blue.  Flatten (B, N_blue) to a single batch
        # dim so a single shared GRUCell handles all agents.
        h_flat      = h_blue.reshape(B * self.n_blue, d)
        hidden_flat = hidden.reshape(B * self.n_blue, d)
        new_hidden_flat = self.actor_gru(h_flat, hidden_flat)
        new_hidden      = new_hidden_flat.reshape(B, self.n_blue, d)

        mean    = self.actor_mean(new_hidden)                       # (B, N_blue, action_dim)
        log_std = self.actor_log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std, new_hidden

    def critic_forward(
        self,
        full_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Critic forward pass: full-state GNN -> sum-blue + concat-red ->
        MLP -> V.  Returns value tensor of shape (B,).
        """
        h_blue, h_red = self.critic_encoder(
            full_state["blue_features"],
            full_state["red_features"],
            full_state["bb_edge_features"],
            full_state["rb_edge_features"],
            # No masks — critic sees full state.
        )
        B = h_blue.shape[0]
        blue_summary = h_blue.sum(dim=1)                            # (B, d)
        red_summary  = h_red.reshape(B, -1)                         # (B, N_red * d)
        state_repr   = torch.cat([blue_summary, red_summary], dim=-1)
        value = self.critic_head(self.critic_trunk(state_repr)).squeeze(-1)
        return value

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_action_and_value(
        self,
        partial_obs: Dict[str, torch.Tensor],
        full_state:  Dict[str, torch.Tensor],
        hidden:      torch.Tensor,
        action:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (if action is None) or evaluate the given action.

        Returns (action, log_prob, entropy, value, new_hidden).
        """
        mean, log_std, new_hidden = self.actor_forward(partial_obs, hidden)
        value = self.critic_forward(full_state)

        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)   # (B, N_blue)
        entropy  = dist.entropy().sum(-1)          # (B, N_blue)
        return action, log_prob, entropy, value, new_hidden

    @torch.no_grad()
    def act_deterministic(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy action = distribution mean.  Used at evaluation time.
        Returns (mean, new_hidden).
        """
        mean, _, new_hidden = self.actor_forward(partial_obs, hidden)
        return mean, new_hidden

    def initial_hidden(
        self,
        n_envs:  int,
        device:  torch.device,
    ) -> torch.Tensor:
        """Return a zero hidden tensor of shape ``(n_envs, N_blue, d_hidden)``."""
        return torch.zeros(n_envs, self.n_blue, self.d_hidden, device=device)

    # ------------------------------------------------------------------ #
    #  Stage 2 warm-start                                                  #
    # ------------------------------------------------------------------ #

    def load_stage2_critic(self, path: Union[str, Path]) -> int:
        """
        Copy the Stage 2 GNNActorCritic checkpoint's GNN encoder + critic
        head weights into *this* policy's critic sub-modules.  Actor
        weights (partial-obs GNN, GRU, actor head) are left as-is
        (random init).

        Returns the number of tensors successfully copied.

        Layout mapping (Stage 2 attribute -> Stage 3 attribute):
            blue_input_mlp   -> critic_encoder.blue_input_mlp
            red_input_mlp    -> critic_encoder.red_input_mlp
            bb_edge_mlp      -> critic_encoder.bb_edge_mlp
            rb_edge_mlp      -> critic_encoder.rb_edge_mlp
            msg_mlp          -> critic_encoder.msg_mlp
            update_mlp       -> critic_encoder.update_mlp
            critic_trunk     -> critic_trunk
            critic_head      -> critic_head
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        stage2_sd = ckpt["policy_state"]

        encoder_names = {
            "blue_input_mlp", "red_input_mlp",
            "bb_edge_mlp", "rb_edge_mlp",
            "msg_mlp", "update_mlp",
        }
        head_names = {"critic_trunk", "critic_head"}

        my_sd = self.state_dict()
        n_copied = 0
        for k, v in stage2_sd.items():
            root = k.split(".", 1)[0]
            if root in encoder_names:
                # Prepend "critic_encoder." for the Stage 3 name.
                target = f"critic_encoder.{k}"
            elif root in head_names:
                target = k
            else:
                # Actor / log_std / edge index buffers — skip.
                continue
            if target in my_sd and my_sd[target].shape == v.shape:
                my_sd[target].copy_(v)
                n_copied += 1
        self.load_state_dict(my_sd)
        return n_copied
