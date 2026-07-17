"""
isr/agents/gnn_stage4_policy.py — Stage 4 belief-map CTDE policy.

Architecture as of v5.3 (see the version log in docs/stage4_design.md
for how it got here — the belief-map paradigm went through several
revisions before landing on explicit peak detections).

Actor (partial-obs, decentralised) per-UAV node feature is a
concatenation of:
- ``blue_features``       (8)   — own kinematics (precise, GPS/IMU)
- hidden state            (64)  — Stage 3-opt-1 cross-blue channel
- red context             (64)  — aggregated messages from VISIBLE
  rb_edges, which carry precise enemy VELOCITY only (Doppler radar);
  enemy position is deliberately excluded from the edges.
- enemy peak context      (32)  — encoded top-K (dx, dy, conf)
  detections from belief-map channel 0 (noisy P(enemy)).
- obstacle peak context   (32)  — encoded top-K detections from
  belief-map channel 1 (noisy P(obstacle)).
= 200-dim actor node feature (with hidden-in-GNN on).

The per-UAV belief CNN present in v1–v5.2 is DROPPED by default
(``use_actor_belief_cnn=False``): an ablation showed the CNN never
learned to recover position from the log-odds tensor.  Explicit
peak detections — the tracker output a real ISR operator sees — are
what the policy consumes instead.  The CNN can be re-enabled for
ablation via the constructor flag; ``BeliefEncoder`` is retained
for that path and for the critic.

Critic (full-obs, centralised, CTDE) keeps a ``BeliefEncoder`` CNN
over ``true_occupancy`` (available only at training time) plus a
blue-only GNN.  It is cold-started — the Stage 3 checkpoint's critic
has red-side tensors and a differently-shaped input.

Sensor-physics split maintained throughout: precise self/ally state
and enemy velocity flow through the graph; noisy enemy/obstacle
positions flow only through the Bayesian belief map's peak output.
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
# Belief CNN encoder
# ---------------------------------------------------------------------------

class BeliefEncoder(nn.Module):
    """
    Small conv net that reduces a (C, H, W) belief/occupancy tensor to
    a fixed ``out_dim`` embedding.

    As of v5.3 this is used only by the CRITIC (over the full-state
    ``true_occupancy`` grid).  The actor's per-UAV belief CNN is
    dropped by default in favour of explicit peak detections; the
    encoder remains available for the ``use_actor_belief_cnn=True``
    ablation path.

    Conv stack auto-adapts to the input grid so the flatten dim stays
    bounded across grid sizes:
        grid_size <= 17:  Conv(C→16, s1) → Conv(16→32, s2)         → ~9×9
        grid_size >  17:  … + Conv(32→32, s2)                       → ~9×9
    Followed by Flatten → Linear(flat_dim, out_dim).  ``flat_dim`` is
    computed dynamically at construction from a dummy forward pass.

    When applied per UAV, the caller reshapes (B, N_blue, C, H, W) to
    (B * N_blue, C, H, W), runs the encoder, and reshapes back.

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
        # Keeps flat dim reasonable (~2500-5500) for grid_size in [17, 52].
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
    encoder, but with no red NODES in the graph — red context
    (velocity from rb_edges, position from belief-map peaks) is
    aggregated per UAV and folded into each blue's INPUT node feature
    upstream, rather than as a separate node type.
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
    Stage 4 CTDE + recurrent actor-critic (v5.3).

    Actor input (per UAV, concatenated into a 200-dim node feature with
    hidden-in-GNN on and the actor CNN off — the default):
      - blue_features       (8)   — own kinematics + wall dists
      - hidden_prev         (64)  — Stage 3 opt-1 cross-blue channel
      - red context         (64)  — sum of encoded VISIBLE rb_edges
        (precise enemy velocity only; position excluded)
      - enemy peak context  (32)  — sum of encoded top-K belief-map
        channel-0 detections (noisy P(enemy) positions)
      - obstacle peak ctx   (32)  — sum of encoded top-K belief-map
        channel-1 detections (noisy P(obstacle) positions)
      (+ belief CNN embed (belief_encoder_out_dim) only if
       ``use_actor_belief_cnn=True`` — off by default)

    Actor path:
      node features → BlueGNNEncoder → GRUCell → actor_mean

    Critic path (CTDE, sees full state incl. true_occupancy):
      blue_features → BlueGNNEncoder (separate weights) → sum over blues
        + BeliefEncoder(true_occupancy)  →  critic_trunk → V

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
        belief_grid_size:      int   = 52,
        belief_encoder_out_dim:int   = 128,
        actor_bayesian_channels:int  = 2,
        belief_window_size:    int   = 33,
        use_hidden_in_gnn:     bool  = True,
        # ----- Red-side branch (Stage 4 v5.1) -----------------------
        # rb_edge is velocity-only (4-D): [rel_vel (2), src_vel (2)].
        # Position info stays in the noisy belief map.
        n_red:                 int   = 3,
        red_feat_dim:          int   = 1,
        rb_edge_feat_dim:      int   = 4,
        red_msg_dim:           int   = 64,
        # ----- Belief peak branch (Stage 4 v5.3) --------------------
        # Explicit peak detections from the belief map.  Two channels:
        #   - enemy peaks    : top-K from log-odds channel 0
        #     (K = n_red)
        #   - obstacle peaks : top-K from log-odds channel 1
        #     (K = n_obstacles)
        # Both come from the noisy Bayesian belief map -- consistent
        # sensor-physics story.  Replaces the actor's belief CNN
        # (dropped in v5.3 -- the CNN wasn't extracting usable info
        # anyway; peaks are the tracker output the policy should
        # actually see, as in real ISR systems).
        n_enemy_peaks:         int   = 3,
        n_obstacle_peaks:      int   = 4,
        peak_msg_dim:          int   = 32,
        use_actor_belief_cnn:  bool  = False,  # v5.3: default off
    ) -> None:
        """
        Args:
          n_enemy_peaks / n_obstacle_peaks: number of top-K detections
            extracted from belief-map channel 0 / channel 1 and fed to
            the actor as (dx, dy, conf) triples.  These are the primary
            enemy/obstacle POSITION signal (the actor CNN is off by
            default).  Defaults track n_red / n_obstacles.

          peak_msg_dim: output width of each per-peak encoder MLP.

          use_actor_belief_cnn: if True, additionally run a per-UAV CNN
            over an ego-centric belief window and concatenate its
            embedding to the node feature.  Off by default (v5.3): the
            CNN was ablation-shown not to recover position from the
            belief tensor, and peaks carry that signal directly.

          actor_bayesian_channels / belief_window_size: only relevant
            when ``use_actor_belief_cnn=True``.  The former marks how
            many leading belief channels are log-odds (sigmoid'd at the
            CNN input); the latter is the side length K of the
            ego-centric crop fed to the CNN.
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

        # ---- Actor belief encoder (v5.3: OPTIONAL) -----------------------
        # v5.3: dropped by default -- the CNN wasn't extracting usable
        # positional info from the belief map (ablation-confirmed by
        # the user).  Belief peaks + rb_edges + bb_edges together
        # carry all the actionable info.  Set use_actor_belief_cnn=True
        # to re-enable for ablation purposes.
        self.use_actor_belief_cnn = use_actor_belief_cnn
        if use_actor_belief_cnn:
            self.actor_belief_encoder = BeliefEncoder(
                in_channels=belief_channels,
                grid_size=belief_window_size,
                out_dim=belief_encoder_out_dim,
                num_logit_channels=min(actor_bayesian_channels, belief_channels),
            )
        else:
            self.actor_belief_encoder = None

        # ---- Red-to-blue message branch (Stage 4 v5) ---------------------
        # Restores the intercept signal that v1-v4 lacked.  For each
        # currently-visible red-blue edge (rb_edge_visible == 1), compute
        # a message from the red features + edge features (which carry
        # relative position AND VELOCITY -- critical for intercepting
        # evading reds), then sum-aggregate per destination blue.
        self.n_red             = n_red
        self.red_msg_dim       = red_msg_dim
        self.red_encoder       = _mlp(red_feat_dim, [red_msg_dim], red_msg_dim)
        self.rb_edge_encoder   = _mlp(rb_edge_feat_dim, [red_msg_dim], red_msg_dim)
        self.rb_msg_mlp        = _mlp(2 * red_msg_dim, [red_msg_dim], red_msg_dim)

        # rb edge index buffers -- convention:
        # edges are enumerated (red_i, blue_j) with i in [0..n_red),
        # j in [0..n_blue).  src = red index, dst = blue index.
        rb_src = torch.arange(n_red).repeat_interleave(n_blue)
        rb_dst = torch.arange(n_blue).repeat(n_red)
        self.register_buffer("rb_src", rb_src.long(), persistent=False)
        self.register_buffer("rb_dst", rb_dst.long(), persistent=False)

        # ---- Belief peak encoders (Stage 4 v5.3) -------------------------
        # Two separate small MLPs -- enemies mean "pursue", obstacles
        # mean "avoid", so distinct encoders let each learn its own
        # semantics.  Aggregated per-UAV via sum-pool.
        self.n_enemy_peaks    = n_enemy_peaks
        self.n_obstacle_peaks = n_obstacle_peaks
        self.peak_msg_dim     = peak_msg_dim
        self.enemy_peak_encoder    = _mlp(3, [peak_msg_dim], peak_msg_dim)
        self.obstacle_peak_encoder = _mlp(3, [peak_msg_dim], peak_msg_dim)

        # Actor node feature dim: blue + optional hidden + optional
        # belief_cnn + red_context + enemy_peaks + obstacle_peaks.
        actor_node_feat_dim = (
            blue_feat_dim
            + red_msg_dim
            + 2 * peak_msg_dim              # enemy + obstacle peaks
        )
        if use_actor_belief_cnn:
            actor_node_feat_dim += belief_encoder_out_dim
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

    def _aggregate_red_context(
        self,
        red_features:      torch.Tensor,  # (B, n_red, red_feat_dim)
        rb_edge_features:  torch.Tensor,  # (B, n_rb, rb_edge_feat_dim)
        rb_edge_visible:   torch.Tensor,  # (B, n_rb) or (B, n_rb, 1)
    ) -> torch.Tensor:
        """
        For each blue UAV, gather messages from currently-visible reds
        via rb_edge_features (which carry relative position + velocity)
        and sum-aggregate.  Non-visible edges contribute zero.

        Returns (B, n_blue, red_msg_dim).
        """
        B, n_rb, _ = rb_edge_features.shape
        # Encode reds and edges.
        red_emb  = self.red_encoder(red_features)            # (B, n_red, d)
        edge_emb = self.rb_edge_encoder(rb_edge_features)    # (B, n_rb, d)

        # Broadcast red embeddings to per-edge (using rb_src).
        red_per_edge = red_emb.index_select(1, self.rb_src)  # (B, n_rb, d)

        # Per-edge message: MLP(concat(red_emb, edge_emb)).
        msg = self.rb_msg_mlp(torch.cat([red_per_edge, edge_emb], dim=-1))

        # Mask by visibility.
        if rb_edge_visible.dim() == 2:
            vis = rb_edge_visible.unsqueeze(-1)              # (B, n_rb, 1)
        else:
            vis = rb_edge_visible
        msg = msg * vis

        # Scatter-sum to destination blue.
        agg = torch.zeros(
            B, self.n_blue, self.red_msg_dim,
            device=msg.device, dtype=msg.dtype,
        )
        agg.index_add_(1, self.rb_dst, msg)
        return agg

    def actor_encode(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,   # (B, N_blue, d_hidden)
    ) -> torch.Tensor:
        """
        Build the actor's per-blue node embedding by concatenating
        blue_features, hidden (if enabled), belief-peak contexts
        (enemy + obstacle), aggregated red context (velocity from
        visible rb_edges), and optionally CNN-encoded belief windows.
        Runs the actor GNN.  Returns h_blue (B, N, d_hidden).

        Red context: from ``red_features`` + ``rb_edge_features`` gated
        by ``rb_edge_visible``.  Zero fallback for missing keys.
        """
        B      = partial_obs["blue_features"].shape[0]
        device = partial_obs["blue_features"].device
        dtype  = partial_obs["blue_features"].dtype

        # Red context aggregation (v5).
        if all(k in partial_obs for k in (
            "red_features", "rb_edge_features", "rb_edge_visible",
        )):
            red_ctx = self._aggregate_red_context(
                partial_obs["red_features"],
                partial_obs["rb_edge_features"],
                partial_obs["rb_edge_visible"],
            )
        else:
            red_ctx = torch.zeros(
                B, self.n_blue, self.red_msg_dim,
                device=device, dtype=dtype,
            )

        # Enemy peak context (v5.2).
        if "belief_peaks_enemy" in partial_obs:
            peaks_e   = partial_obs["belief_peaks_enemy"]  # (B, N, Ke, 3)
            enemy_ctx = self.enemy_peak_encoder(peaks_e).sum(dim=2)
        else:
            enemy_ctx = torch.zeros(
                B, self.n_blue, self.peak_msg_dim,
                device=device, dtype=dtype,
            )

        # Obstacle peak context (v5.3).
        if "belief_peaks_obstacle" in partial_obs:
            peaks_o  = partial_obs["belief_peaks_obstacle"] # (B, N, Ko, 3)
            obst_ctx = self.obstacle_peak_encoder(peaks_o).sum(dim=2)
        else:
            obst_ctx = torch.zeros(
                B, self.n_blue, self.peak_msg_dim,
                device=device, dtype=dtype,
            )

        parts = [partial_obs["blue_features"]]
        if self.use_hidden_in_gnn:
            parts.append(hidden)
        if self.use_actor_belief_cnn:
            belief_input = partial_obs.get(
                "belief_windows", partial_obs.get("belief_maps"),
            )
            belief_emb = self.actor_belief_encoder(belief_input)
            parts.append(belief_emb)
        parts.append(red_ctx)
        parts.append(enemy_ctx)
        parts.append(obst_ctx)
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
