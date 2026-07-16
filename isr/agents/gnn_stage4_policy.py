"""
isr/agents/gnn_stage4_policy.py — Stage 4 belief-map CTDE policy.

Extends the Stage 3 CTDE + recurrent architecture with:

- A per-UAV CNN encoder over the (C=2, H=26, W=26) log-odds belief
  tensor.  Its 64-dim output is concatenated with the existing
  ``blue_features`` (8) and the Stage 3-opt-1 hidden state (64) to
  form a 136-dim actor node feature.
- A separate CNN encoder in the critic path that reads the
  ``true_occupancy`` (available at training time — CTDE) and produces
  a 64-dim global belief embedding.
- No red nodes in the GNN — enemies live only in the belief map.
- No `bb_edge_visible` mask — ally comms via GPS uplink always on.

See ``docs/stage4_design.md`` §5 for the full architecture rationale
and §6 for the (aux-off) PPO integration.

Cold-started: the critic is randomly initialised; the Stage 3
checkpoint's critic has red-side tensors and a differently-shaped
input, so byte-safe copy is not possible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Init helpers (byte-compatible with the Stage 3 policy)
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


# ---------------------------------------------------------------------------
# Belief CNN encoder (shared per-UAV weights)
# ---------------------------------------------------------------------------

class BeliefEncoder(nn.Module):
    """
    Small conv net that reduces a (C, H, W) log-odds belief tensor to
    a fixed ``out_dim`` embedding.

    Default architecture matches docs/stage4_design.md §5.1:
        Conv2d(2 → 16, k=3, s=2, p=1)   → (16, 13, 13)
        Conv2d(16 → 32, k=3, s=2, p=1)  → (32, 7, 7)
        Flatten                          → (1568,)
        Linear(1568, 64)                 → (64,)
    ~110 k params.

    Shared weights are applied per UAV: the caller reshapes
    (B, N_blue, C, H, W) to (B * N_blue, C, H, W), runs the encoder,
    and reshapes back.

    ``num_logit_channels`` controls per-channel input pre-processing:
    the first ``num_logit_channels`` channels are treated as raw log-
    odds and passed through ``torch.sigmoid`` at the input (so the CNN
    sees probabilities in [0, 1]); the remaining channels are passed
    through unchanged (they're already in [0, 1] -- e.g. deterministic
    ally / self position overlays, or the critic's ``true_occupancy``
    which is binary).

    Rationale for the split: cells with log-odds 8 vs 10 both mean
    "essentially certain" -- sigmoid collapses them so the CNN devotes
    capacity to distinguishing "unknown" vs "possibly" vs "definitely"
    instead of wasting it near saturation.  But deterministic channels
    (values in {0, 1}) would be uselessly compressed to {0.5, 0.73} by
    sigmoid, so they bypass it.

    Special values:
    - ``num_logit_channels = 0``: no sigmoid at all (all channels
      pass through -- use this for the critic on true_occupancy).
    - ``num_logit_channels = in_channels``: sigmoid every channel.
    """

    def __init__(
        self,
        in_channels:       int = 2,
        grid_size:         int = 33,
        out_dim:           int = 64,
        num_logit_channels:int = None,   # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.in_channels    = in_channels
        self.grid_size      = grid_size
        self.out_dim        = out_dim
        self.num_logit_channels = (
            in_channels if num_logit_channels is None
            else int(num_logit_channels)
        )
        assert 0 <= self.num_logit_channels <= in_channels

        # Build a conv stack that progressively halves the spatial dims.
        # For grid_size <= 17: 2 convs  (C->16 s1, 16->32 s2)  -> ~9x9
        # For grid_size >  17: 3 convs  (C->16 s1, 16->32 s2, 32->32 s2) -> ~9x9
        # This keeps the flatten dim reasonable (~2500-3200) for any
        # grid size in the [17, 52] range.
        layers = []
        layers.append(nn.Conv2d(in_channels, 16, kernel_size=3,
                                stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(16, 32, kernel_size=3,
                                stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if grid_size > 17:
            layers.append(nn.Conv2d(32, 32, kernel_size=3,
                                    stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size, grid_size)
            h = self.convs(dummy)
            flat_dim = int(h.reshape(1, -1).shape[1])
        self.flat_dim = flat_dim
        self.proj = _layer_init(nn.Linear(flat_dim, out_dim))

    def _apply_input_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to the first ``num_logit_channels`` only."""
        k = self.num_logit_channels
        if k == 0:
            return x
        if k == self.in_channels:
            return torch.sigmoid(x)
        # Split, sigmoid the first k, concat.  Channel dim is -3.
        logit_part = torch.sigmoid(x[..., :k, :, :])
        pass_part  = x[..., k:, :, :]
        return torch.cat([logit_part, pass_part], dim=-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W) or (B, N_blue, C, H, W)
        Returns (B, out_dim) or (B, N_blue, out_dim).
        """
        x = self._apply_input_transform(x)

        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            h = self.convs(x)
            h = h.reshape(B * N, self.flat_dim)
            emb = self.proj(h)
            return emb.reshape(B, N, self.out_dim)

        h = self.convs(x)
        h = h.reshape(x.shape[0], self.flat_dim)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Blue-only GNN encoder (no red nodes, no rb edges)
# ---------------------------------------------------------------------------

class BlueGNNEncoder(nn.Module):
    """
    2-round message passing over the blue-only graph.  Same edge-
    conditioned message + residual-update pattern as the Stage 3
    encoder, but without the red-blue path (reds live in the belief
    map, not the graph).
    """

    def __init__(
        self,
        n_blue:         int,
        blue_feat_dim:  int,
        edge_feat_dim:  int = 7,
        d_hidden:       int = 64,
        n_msg_rounds:   int = 2,
    ) -> None:
        super().__init__()
        self.n_blue       = n_blue
        self.d_hidden     = d_hidden
        self.n_msg_rounds = n_msg_rounds

        self.blue_input_mlp = _mlp(blue_feat_dim, [d_hidden], d_hidden)
        self.bb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.msg_mlp        = _mlp(3 * d_hidden, [d_hidden], d_hidden)
        self.update_mlp     = _mlp(2 * d_hidden, [d_hidden], d_hidden)

        bb_src, bb_dst = _build_bb_edges(n_blue)
        self.register_buffer("bb_src", bb_src, persistent=False)
        self.register_buffer("bb_dst", bb_dst, persistent=False)

    def forward(
        self,
        blue_feats:    torch.Tensor,   # (B, N, blue_feat_dim)
        bb_edge_feats: torch.Tensor,   # (B, n_bb, edge_feat_dim)
    ) -> torch.Tensor:
        B = blue_feats.shape[0]
        d = self.d_hidden
        h_blue = self.blue_input_mlp(blue_feats)
        e_bb   = self.bb_edge_mlp(bb_edge_feats)

        for _ in range(self.n_msg_rounds):
            h_sender   = h_blue.index_select(1, self.bb_src)
            h_receiver = h_blue.index_select(1, self.bb_dst)
            msg = self.msg_mlp(
                torch.cat([h_sender, h_receiver, e_bb], dim=-1),
            )
            agg = torch.zeros(B, self.n_blue, d,
                              device=h_blue.device, dtype=h_blue.dtype)
            agg.index_add_(1, self.bb_dst, msg)
            h_blue = h_blue + self.update_mlp(
                torch.cat([h_blue, agg], dim=-1),
            )
        return h_blue


# ---------------------------------------------------------------------------
# Stage 4 CTDE policy
# ---------------------------------------------------------------------------

class GNNStage4Policy(nn.Module):
    """
    Stage 4 CTDE + recurrent actor-critic with belief-map encoders.

    Actor input (per UAV, concatenated into a 136-dim node feature):
      - blue_features   (8)   — position, velocity, wall dists
      - hidden_prev     (64)  — Stage 3 opt-1 cross-blue hidden channel
      - belief embed    (64)  — CNN(belief_maps[i])

    Actor path:
      node features → BlueGNNEncoder → GRUCell → actor_mean

    Critic path (CTDE, sees full state incl. true_occupancy):
      blue_features → BlueGNNEncoder (separate weights) → sum over blues
        + CNN(true_occupancy)  →  critic_trunk → V

    ``get_action_and_value`` returns
        (action, log_prob, entropy, value, new_hidden)
    matching the Stage 3 CTDE interface.
    """

    def __init__(
        self,
        n_blue:                int,
        blue_feat_dim:         int   = 8,
        edge_feat_dim:         int   = 7,
        action_dim:            int   = 2,
        d_hidden:              int   = 64,
        n_msg_rounds:          int   = 2,
        init_log_std:          float = 0.0,
        belief_channels:       int   = 3,
        belief_grid_size:      int   = 26,
        belief_encoder_out_dim:int   = 128,
        actor_bayesian_channels:int  = 2,
        belief_window_size:    int   = 17,
        use_hidden_in_gnn:     bool  = True,
    ) -> None:
        """
        Args:
          actor_bayesian_channels: number of belief-map channels that
            are noisy log-odds (defaults to 2 for Stage 4: enemy +
            obstacle).  These get ``sigmoid`` at the CNN input.  Any
            additional channels (e.g. deterministic ally overlay) are
            treated as already in [0, 1] and pass through unchanged.

          belief_window_size: side length K of the ego-centric window
            fed to the actor's belief CNN.  The actor consumes a
            (belief_channels, K, K) crop centred on the UAV's cell,
            NOT the global (belief_channels, H, W) map.  Fixes the
            "CNN can't produce ego-centric features" problem
            observed in Stage 4 v1/v2.
        """
        super().__init__()
        self.n_blue             = n_blue
        self.d_hidden           = d_hidden
        self.action_dim         = action_dim
        self.use_hidden_in_gnn  = use_hidden_in_gnn
        self.belief_channels    = belief_channels
        self.belief_grid_size   = belief_grid_size
        self.belief_window_size = belief_window_size
        self.belief_dim         = belief_encoder_out_dim

        # ---- Actor belief encoder ----------------------------------------
        # Actor's input: KxK ego-centric window per UAV.
        #   channels [0..actor_bayesian_channels): log-odds -> sigmoid.
        #   channels [actor_bayesian_channels..):  binary -> passthrough.
        self.actor_belief_encoder = BeliefEncoder(
            in_channels=belief_channels,
            grid_size=belief_window_size,
            out_dim=belief_encoder_out_dim,
            num_logit_channels=min(actor_bayesian_channels, belief_channels),
        )

        # Actor node feature dim: base blue + optional hidden + belief.
        actor_node_feat_dim = blue_feat_dim + belief_encoder_out_dim
        if use_hidden_in_gnn:
            actor_node_feat_dim += d_hidden

        # ---- Actor GNN + GRU + head --------------------------------------
        self.actor_encoder = BlueGNNEncoder(
            n_blue        = n_blue,
            blue_feat_dim = actor_node_feat_dim,
            edge_feat_dim = edge_feat_dim,
            d_hidden      = d_hidden,
            n_msg_rounds  = n_msg_rounds,
        )
        self.actor_gru  = nn.GRUCell(input_size=d_hidden, hidden_size=d_hidden)
        self.actor_mean = _layer_init(nn.Linear(d_hidden, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32),
        )

        # ---- Critic belief encoder + GNN + head --------------------------
        # Critic reads global true_occupancy (2 channels: enemy +
        # obstacle, grid_size × grid_size) -- already binary [0,1].
        self.critic_belief_encoder = BeliefEncoder(
            in_channels=2,
            grid_size=belief_grid_size,
            out_dim=belief_encoder_out_dim,
            num_logit_channels=0,
        )
        self.critic_encoder = BlueGNNEncoder(
            n_blue        = n_blue,
            blue_feat_dim = blue_feat_dim,        # plain blue features here
            edge_feat_dim = edge_feat_dim,
            d_hidden      = d_hidden,
            n_msg_rounds  = n_msg_rounds,
        )
        # sum_blue (d_hidden) + true_occupancy embedding (belief_dim)
        critic_in = d_hidden + belief_encoder_out_dim
        self.critic_trunk = _mlp(critic_in, [d_hidden], d_hidden)
        self.critic_head  = _layer_init(nn.Linear(d_hidden, 1), std=1.0)

    # ------------------------------------------------------------------ #
    #  Sub-forwards                                                        #
    # ------------------------------------------------------------------ #

    def actor_encode(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,   # (B, N_blue, d_hidden)
    ) -> torch.Tensor:
        """
        Build the actor's per-blue node embedding by concatenating
        blue_features, hidden (if enabled), and belief-encoded features,
        then running the actor GNN.  Returns h_blue (B, N, d_hidden).

        Belief input: reads ``belief_windows`` (ego-centric, per-UAV KxK
        crops) if present in ``partial_obs``, else falls back to the
        global ``belief_maps`` for backward compatibility.
        """
        belief_input = partial_obs.get(
            "belief_windows", partial_obs.get("belief_maps"),
        )
        belief_emb = self.actor_belief_encoder(belief_input)
        parts = [partial_obs["blue_features"]]
        if self.use_hidden_in_gnn:
            parts.append(hidden)
        parts.append(belief_emb)
        blue_input = torch.cat(parts, dim=-1)
        return self.actor_encoder(blue_input, partial_obs["bb_edge_features"])

    def actor_forward(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actor forward pass.

        Returns (mean, log_std_expanded, new_hidden).
        """
        h_blue = self.actor_encode(partial_obs, hidden)
        B = h_blue.shape[0]
        d = self.d_hidden

        h_flat      = h_blue.reshape(B * self.n_blue, d)
        hidden_flat = hidden.reshape(B * self.n_blue, d)
        new_hidden_flat = self.actor_gru(h_flat, hidden_flat)
        new_hidden      = new_hidden_flat.reshape(B, self.n_blue, d)

        mean    = self.actor_mean(new_hidden)
        log_std = self.actor_log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std, new_hidden

    def critic_forward(
        self,
        full_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Full-state critic.  Consumes plain blue_features + bb_edge_features
        + true_occupancy (broadcast to a single team-level embedding).

        Returns V (B,).
        """
        # GNN over blue nodes only.
        h_blue = self.critic_encoder(
            full_state["blue_features"],
            full_state["bb_edge_features"],
        )                                       # (B, N, d_hidden)
        blue_sum = h_blue.sum(dim=1)            # (B, d_hidden)

        # True-occupancy embedding.  true_occupancy shape (B, C, H, W).
        true_emb = self.critic_belief_encoder(full_state["true_occupancy"])
        # (B, belief_dim)

        state_repr = torch.cat([blue_sum, true_emb], dim=-1)
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
        mean, _, new_hidden = self.actor_forward(partial_obs, hidden)
        return mean, new_hidden

    def initial_hidden(
        self,
        n_envs: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(n_envs, self.n_blue, self.d_hidden, device=device)
