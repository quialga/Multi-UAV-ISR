"""
isr/agents/gnn_stage4_policy.py — Stage 4 (v6) GNN + GRU + CTDE policy.

v6 rationale: revert to the proven Stage 3 architecture
(``GNNCTDEPolicy`` at commit 8ec72f9) and extend it with a third
typed entity — OBSTACLES — as GNN nodes.  No CNN anywhere.  The
belief map is still maintained in the env exactly as before; it is
used only to RECONSTRUCT the graph features (enemy/obstacle relative
position + bearing + range) that the actor consumes, replacing the
ground-truth positions Stage 3 used.

Typed graph (both actor and critic):
- Nodes:   blue (N_blue), red (N_red), obstacle (N_obs)
- Edges:   bb (blue->blue), rb (red->blue), ob (obstacle->blue)
- Blue nodes are the only receivers; reds/obstacles never update.

Actor (partial obs, decentralised):
- rb / ob edge geometry (rel_pos, range, bearing) comes from each
  blue's belief-map PEAK detections (noisy).  Enemy edge velocity
  comes from radar (precise, when visible); obstacle velocity is 0
  (static) — the ob edge is an rb edge with the object velocity
  zeroed, so ``rel_vel`` is driven entirely by the blue's own motion.
- Per-edge visibility masks (belief confidence) gate the messages,
  exactly like Stage 3's rb_visible / bb_visible masks.
- GRUCell per blue for belief tracking; ``use_hidden_in_gnn`` (Stage 3
  opt-1) prepends the previous hidden state to blue node features.

Critic (CTDE, full obs — same philosophy as Stage 3, NO CNN):
- Same typed GNN on GROUND-TRUTH node/edge features (the critic is
  allowed full observability at training time).
- sum_blue + concat_red + concat_obstacle -> critic MLP -> V.

``get_action_and_value`` keeps the Stage 3 CTDE signature:
    (action, log_prob, entropy, value, new_hidden)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Init / MLP helpers (same conventions as Stage 2/3 GNN)
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


# Actor (belief-derived) and critic (ground-truth) obs key sets.
_ACTOR_KEYS = (
    "blue_features", "bb_edge_features", "bb_edge_visible",
    "red_features", "rb_edge_features", "rb_edge_visible",
    "obstacle_features", "ob_edge_features", "ob_edge_visible",
)
# Critic: shared precise keys + true_* remapped to the plain names the
# GNNEncoder expects.
_CRITIC_SHARED = ("blue_features", "bb_edge_features")
_CRITIC_TRUE_MAP = {
    "true_red_features":      "red_features",
    "true_rb_edge_features":  "rb_edge_features",
    "true_obstacle_features": "obstacle_features",
    "true_ob_edge_features":  "ob_edge_features",
}


def split_stage4_obs(obs: Dict[str, object]) -> Tuple[Dict, Dict]:
    """
    Split a combined Stage 4 v6 obs dict into ``(partial_obs, full_state)``
    for the actor and critic respectively.

    - partial_obs: the belief-derived actor keys that are present.
    - full_state:  shared precise keys + the ``true_*`` critic keys
      remapped to the plain node/edge names the GNNEncoder expects.
    """
    partial_obs = {k: obs[k] for k in _ACTOR_KEYS if k in obs}
    full_state = {k: obs[k] for k in _CRITIC_SHARED if k in obs}
    for src, dst in _CRITIC_TRUE_MAP.items():
        if src in obs:
            full_state[dst] = obs[src]
    return partial_obs, full_state


def _build_bb_edges(n_blue: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst = [], []
    for s in range(n_blue):
        for d in range(n_blue):
            if s != d:
                src.append(s); dst.append(d)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))


def _build_xb_edges(n_src: int, n_blue: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complete bipartite edges from ``n_src`` entity nodes to blues.

    Ordering matches the env: for s in range(n_src): for b in range(n_blue).
    Used for both rb (reds) and ob (obstacles).
    """
    src, dst = [], []
    for s in range(n_src):
        for b in range(n_blue):
            src.append(s); dst.append(b)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))


# ---------------------------------------------------------------------------
# Typed GNN encoder (blue + red + obstacle nodes; bb + rb + ob edges)
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """
    2-round message-passing GNN over the typed entity graph.  Same
    edge-conditioned message + residual-update pattern as the Stage 3
    encoder, extended with an obstacle node type and ob (obstacle->blue)
    edges.

    ``forward`` accepts optional per-edge-type visibility masks;
    multiplied into the messages before ``index_add_`` aggregation so
    hidden edges contribute zero.  When masks are None (default), the
    encoder is fully observable — identical semantics to Stage 2/3.

    When ``n_obs == 0`` the obstacle path is disabled (no obstacle
    nodes, no ob edges).
    """

    def __init__(
        self,
        n_blue:         int,
        n_red:          int,
        n_obs:          int,
        blue_feat_dim:  int = 8,
        red_feat_dim:   int = 1,
        obs_feat_dim:   int = 1,
        edge_feat_dim:  int = 7,
        d_hidden:       int = 64,
        n_msg_rounds:   int = 2,
    ) -> None:
        super().__init__()
        self.n_blue       = n_blue
        self.n_red        = n_red
        self.n_obs        = n_obs
        self.d_hidden     = d_hidden
        self.n_msg_rounds = n_msg_rounds

        self.blue_input_mlp = _mlp(blue_feat_dim, [d_hidden], d_hidden)
        self.red_input_mlp  = _mlp(red_feat_dim,  [d_hidden], d_hidden)
        self.bb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.rb_edge_mlp    = _mlp(edge_feat_dim, [d_hidden], d_hidden)
        self.msg_mlp        = _mlp(3 * d_hidden, [d_hidden], d_hidden)
        self.update_mlp     = _mlp(2 * d_hidden, [d_hidden], d_hidden)

        bb_src, bb_dst = _build_bb_edges(n_blue)
        rb_src, rb_dst = _build_xb_edges(n_red, n_blue)
        self.register_buffer("bb_src", bb_src, persistent=False)
        self.register_buffer("bb_dst", bb_dst, persistent=False)
        self.register_buffer("rb_src", rb_src, persistent=False)
        self.register_buffer("rb_dst", rb_dst, persistent=False)

        if n_obs > 0:
            self.obs_input_mlp = _mlp(obs_feat_dim,  [d_hidden], d_hidden)
            self.ob_edge_mlp   = _mlp(edge_feat_dim, [d_hidden], d_hidden)
            ob_src, ob_dst = _build_xb_edges(n_obs, n_blue)
            self.register_buffer("ob_src", ob_src, persistent=False)
            self.register_buffer("ob_dst", ob_dst, persistent=False)
        else:
            self.obs_input_mlp = None
            self.ob_edge_mlp   = None

    def forward(
        self,
        blue_feats:    torch.Tensor,      # (B, N_blue, blue_feat_dim)
        red_feats:     torch.Tensor,      # (B, N_red,  red_feat_dim)
        bb_edge_feats: torch.Tensor,      # (B, n_bb,   edge_feat_dim)
        rb_edge_feats: torch.Tensor,      # (B, n_rb,   edge_feat_dim)
        obs_feats:     Optional[torch.Tensor] = None,  # (B, N_obs, obs_feat_dim)
        ob_edge_feats: Optional[torch.Tensor] = None,  # (B, n_ob, edge_feat_dim)
        bb_visible:    Optional[torch.Tensor] = None,  # (B, n_bb)
        rb_visible:    Optional[torch.Tensor] = None,  # (B, n_rb)
        ob_visible:    Optional[torch.Tensor] = None,  # (B, n_ob)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns final (blue, red, obstacle) node embeddings after
        ``n_msg_rounds`` rounds.  ``h_obs`` is None when n_obs == 0.
        """
        B = blue_feats.shape[0]
        d = self.d_hidden

        h_blue = self.blue_input_mlp(blue_feats)
        h_red  = self.red_input_mlp(red_feats)
        e_bb   = self.bb_edge_mlp(bb_edge_feats)
        e_rb   = self.rb_edge_mlp(rb_edge_feats)

        has_obs = self.n_obs > 0 and obs_feats is not None
        if has_obs:
            h_obs = self.obs_input_mlp(obs_feats)
            e_ob  = self.ob_edge_mlp(ob_edge_feats)
        else:
            h_obs = None

        for _ in range(self.n_msg_rounds):
            # Blue-Blue messages.
            h_send_bb = h_blue.index_select(1, self.bb_src)
            h_recv_bb = h_blue.index_select(1, self.bb_dst)
            msg_bb = self.msg_mlp(torch.cat([h_send_bb, h_recv_bb, e_bb], dim=-1))
            if bb_visible is not None:
                msg_bb = msg_bb * bb_visible.unsqueeze(-1)

            # Red-Blue messages.
            h_send_rb = h_red.index_select(1, self.rb_src)
            h_recv_rb = h_blue.index_select(1, self.rb_dst)
            msg_rb = self.msg_mlp(torch.cat([h_send_rb, h_recv_rb, e_rb], dim=-1))
            if rb_visible is not None:
                msg_rb = msg_rb * rb_visible.unsqueeze(-1)

            # Aggregate onto blue receivers.
            agg = torch.zeros(B, self.n_blue, d,
                              device=h_blue.device, dtype=h_blue.dtype)
            agg.index_add_(1, self.bb_dst, msg_bb)
            agg.index_add_(1, self.rb_dst, msg_rb)

            # Obstacle-Blue messages.
            if has_obs:
                h_send_ob = h_obs.index_select(1, self.ob_src)
                h_recv_ob = h_blue.index_select(1, self.ob_dst)
                msg_ob = self.msg_mlp(
                    torch.cat([h_send_ob, h_recv_ob, e_ob], dim=-1),
                )
                if ob_visible is not None:
                    msg_ob = msg_ob * ob_visible.unsqueeze(-1)
                agg.index_add_(1, self.ob_dst, msg_ob)

            # Residual node update on blues only.
            h_blue = h_blue + self.update_mlp(torch.cat([h_blue, agg], dim=-1))

        return h_blue, h_red, h_obs


# ---------------------------------------------------------------------------
# Stage 4 v6 CTDE actor-critic
# ---------------------------------------------------------------------------

class GNNStage4Policy(nn.Module):
    """
    Stage 4 (v6) policy: GNN + GRU actor + centralised GNN critic, with
    obstacle nodes added.  Mirrors Stage 3's ``GNNCTDEPolicy`` exactly
    except for (a) the obstacle node/edge type and (b) the actor's
    node/edge features coming from belief-map peaks instead of ground
    truth.  No CNN.

    partial_obs (dict, ACTOR — belief-derived):
        blue_features    (B, N_blue, blue_feat_dim)
        red_features     (B, N_red,  red_feat_dim)
        obstacle_features(B, N_obs,  obs_feat_dim)      [if N_obs > 0]
        bb_edge_features (B, n_bb,   edge_feat_dim)
        rb_edge_features (B, n_rb,   edge_feat_dim)
        ob_edge_features (B, n_ob,   edge_feat_dim)     [if N_obs > 0]
        bb_edge_visible  (B, n_bb)
        rb_edge_visible  (B, n_rb)
        ob_edge_visible  (B, n_ob)                      [if N_obs > 0]
    full_state (dict, CRITIC — ground truth): same fields, no masks.
    hidden       (B, N_blue, d_hidden)
    """

    def __init__(
        self,
        n_blue:            int,
        n_red:             int,
        n_obs:             int   = 0,
        blue_feat_dim:     int   = 8,
        red_feat_dim:      int   = 1,
        obs_feat_dim:      int   = 1,
        edge_feat_dim:     int   = 7,
        action_dim:        int   = 2,
        d_hidden:          int   = 64,
        n_msg_rounds:      int   = 2,
        init_log_std:      float = 0.0,
        use_hidden_in_gnn: bool  = True,
    ) -> None:
        super().__init__()
        self.n_blue            = n_blue
        self.n_red             = n_red
        self.n_obs             = n_obs
        self.d_hidden          = d_hidden
        self.action_dim        = action_dim
        self.use_hidden_in_gnn = use_hidden_in_gnn

        actor_blue_feat_dim = (
            blue_feat_dim + d_hidden if use_hidden_in_gnn else blue_feat_dim
        )

        # ---- Actor path ----------------------------------------------
        self.actor_encoder = GNNEncoder(
            n_blue        = n_blue,
            n_red         = n_red,
            n_obs         = n_obs,
            blue_feat_dim = actor_blue_feat_dim,
            red_feat_dim  = red_feat_dim,
            obs_feat_dim  = obs_feat_dim,
            edge_feat_dim = edge_feat_dim,
            d_hidden      = d_hidden,
            n_msg_rounds  = n_msg_rounds,
        )
        self.actor_gru  = nn.GRUCell(input_size=d_hidden, hidden_size=d_hidden)
        self.actor_mean = _layer_init(nn.Linear(d_hidden, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32),
        )

        # ---- Critic path (CTDE, no CNN) ------------------------------
        self.critic_encoder = GNNEncoder(
            n_blue        = n_blue,
            n_red         = n_red,
            n_obs         = n_obs,
            blue_feat_dim = blue_feat_dim,
            red_feat_dim  = red_feat_dim,
            obs_feat_dim  = obs_feat_dim,
            edge_feat_dim = edge_feat_dim,
            d_hidden      = d_hidden,
            n_msg_rounds  = n_msg_rounds,
        )
        # Per-agent value V(s, i) input:
        #   own blue embedding (d)
        # + masked-MEAN-pooled red context (d) + active-red count (1)
        # + masked-MEAN-pooled obstacle context (d) + count (1)  [if n_obs>0]
        # Mean pooling (not the old concat/flatten) makes the width
        # INDEPENDENT of the red/obstacle counts, so a single critic
        # handles a variable number of entities and generalises to
        # counts unseen in training.  The count scalar restores the
        # "how many threats remain" signal that the mean discards.
        critic_in = d_hidden + (d_hidden + 1)
        if n_obs > 0:
            critic_in += (d_hidden + 1)
        self.critic_trunk = _mlp(critic_in, [d_hidden], d_hidden)
        self.critic_head  = _layer_init(nn.Linear(d_hidden, 1), std=1.0)

    # ------------------------------------------------------------------ #
    #  Sub-forwards                                                        #
    # ------------------------------------------------------------------ #

    def actor_encode(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,       # (B, N_blue, d_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the actor GNN on belief-derived features.  Returns the
        pre-GRU (h_blue, h_red, h_obs) node embeddings."""
        if self.use_hidden_in_gnn:
            blue_input = torch.cat([partial_obs["blue_features"], hidden], dim=-1)
        else:
            blue_input = partial_obs["blue_features"]

        obs_feats     = partial_obs.get("obstacle_features")
        ob_edge_feats = partial_obs.get("ob_edge_features")
        ob_visible    = partial_obs.get("ob_edge_visible")

        return self.actor_encoder(
            blue_input,
            partial_obs["red_features"],
            partial_obs["bb_edge_features"],
            partial_obs["rb_edge_features"],
            obs_feats     = obs_feats,
            ob_edge_feats = ob_edge_feats,
            bb_visible    = partial_obs["bb_edge_visible"],
            rb_visible    = partial_obs["rb_edge_visible"],
            ob_visible    = ob_visible,
        )

    def actor_forward(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std_expanded, new_hidden, h_blue_pre_gru).

        ``h_blue_pre_gru`` is the actor GNN's per-blue node embedding
        BEFORE the GRUCell — the target for the aux hidden loss
        (Stage 3 pattern, ported into v6).
        """
        h_blue, _, _ = self.actor_encode(partial_obs, hidden)
        B = h_blue.shape[0]
        d = self.d_hidden

        h_flat      = h_blue.reshape(B * self.n_blue, d)
        hidden_flat = hidden.reshape(B * self.n_blue, d)
        new_hidden_flat = self.actor_gru(h_flat, hidden_flat)
        new_hidden      = new_hidden_flat.reshape(B, self.n_blue, d)

        mean    = self.actor_mean(new_hidden)
        log_std = self.actor_log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std, new_hidden, h_blue

    def critic_encode(
        self,
        full_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the critic GNN on ground-truth features (no visibility
        masks — CTDE full state).  Returns (h_blue, h_red, h_obs)."""
        return self.critic_encoder(
            full_state["blue_features"],
            full_state["red_features"],
            full_state["bb_edge_features"],
            full_state["rb_edge_features"],
            obs_feats     = full_state.get("obstacle_features"),
            ob_edge_feats = full_state.get("ob_edge_features"),
        )

    def critic_forward(
        self,
        full_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Agent-conditioned centralised value V(s, i), one per blue.
        Returns (V, h_blue) where V is (B, N_blue) and h_blue is the
        target for the aux hidden loss.

        V(s, i) reads off agent i's OWN critic node embedding
        ``h_blue[i]`` (which has already absorbed the whole graph via
        2 rounds of message passing -- teammates, reds, obstacles) in
        place of the pooled ``sum(h_blue)`` a shared-reward V(s) would
        use.  The graph node IS the agent identity, so no one-hot id is
        needed.

        The red / obstacle global context is a masked MEAN pool over the
        ACTIVE nodes (plus a normalised active-count scalar), NOT the old
        concat-flatten.  Mean pooling is permutation-invariant AND
        count-independent, so the same critic handles a variable number
        of reds / obstacles and generalises to counts unseen in training.
        The active mask comes straight from the ground-truth node
        features the critic already receives: ``red_features[..., 0]`` is
        the active flag (1/0, i.e. ``_red_active``) and
        ``obstacle_features[..., 0]`` is placed (1) / padded (0).
        """
        h_blue, h_red, h_obs = self.critic_encode(full_state)
        B, N, d = h_blue.shape

        red_mask = (full_state["red_features"][..., 0] > 0.5).float()  # (B, N_red)
        red_ctx, red_cnt = self._masked_pool(h_red, red_mask)
        global_parts = [red_ctx, red_cnt]

        if h_obs is not None:
            obs_mask = (full_state["obstacle_features"][..., 0] > 0.5).float()
            obs_ctx, obs_cnt = self._masked_pool(h_obs, obs_mask)
            global_parts += [obs_ctx, obs_cnt]

        global_ctx = torch.cat(global_parts, dim=-1)      # (B, d(+1)[+d+1])
        # Broadcast the global (red/obstacle) context to every blue and
        # concat with that blue's own embedding -> per-agent input.
        global_exp = global_ctx.unsqueeze(1).expand(B, N, -1)
        per_agent = torch.cat([h_blue, global_exp], dim=-1)   # (B, N, trunk_in)
        value = self.critic_head(self.critic_trunk(per_agent)).squeeze(-1)  # (B, N)
        return value, h_blue

    @staticmethod
    def _masked_pool(
        h:    torch.Tensor,   # (B, K, d) node embeddings
        mask: torch.Tensor,   # (B, K) in {0, 1}: active nodes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked MEAN pool over active nodes + a normalised active-count
        scalar.  Returns ``(mean (B, d), frac (B, 1))``.  Mean is 0 when
        no node is active (denominator clamped); ``frac`` is the active
        fraction of the padded capacity ``K`` (0..1)."""
        m    = mask.unsqueeze(-1)                       # (B, K, 1)
        cnt  = mask.sum(dim=1, keepdim=True)            # (B, 1) active count
        mean = (h * m).sum(dim=1) / cnt.clamp(min=1.0)  # (B, d)
        frac = cnt / float(h.shape[1])                  # (B, 1) count / capacity
        return mean, frac

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_action_and_value(
        self,
        partial_obs: Dict[str, torch.Tensor],
        full_state:  Dict[str, torch.Tensor],
        hidden:      torch.Tensor,
        action:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns
            (action, log_prob, entropy, value, new_hidden,
             actor_h_blue, critic_h_blue)
        The last two are pre-GRU / post-GNN per-blue embeddings, used
        by the aux hidden loss (Stage 3 opt-C).  Consumers that don't
        need aux can simply discard them.
        """
        mean, log_std, new_hidden, actor_h_blue = self.actor_forward(partial_obs, hidden)
        value, critic_h_blue = self.critic_forward(full_state)

        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)   # (B, N_blue)
        entropy  = dist.entropy().sum(-1)          # (B, N_blue)
        return (action, log_prob, entropy, value, new_hidden,
                actor_h_blue, critic_h_blue)

    @torch.no_grad()
    def act_deterministic(
        self,
        partial_obs: Dict[str, torch.Tensor],
        hidden:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, _, new_hidden, _ = self.actor_forward(partial_obs, hidden)
        return mean, new_hidden

    def initial_hidden(self, n_envs: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n_envs, self.n_blue, self.d_hidden, device=device)

    # ------------------------------------------------------------------ #
    #  Critic warm-start (ported from Stage 3's load_stage2_critic)        #
    # ------------------------------------------------------------------ #

    def load_pretrained_critic(self, path) -> int:
        """
        Copy a pre-trained single-encoder GNN actor-critic (Stage 1/2
        scaling_gnn) into THIS policy's CRITIC sub-modules, leaving the
        actor random.  A pre-trained critic gives meaningful value
        estimates (and therefore meaningful PPO advantages) from
        rollout 0 — the ingredient Stage 3 relied on, dropped in the
        Stage 4 cold-start.

        Mapping (source root -> target):
          {blue,red}_input_mlp, {bb,rb}_edge_mlp, msg_mlp, update_mlp
                                        -> critic_encoder.<same>
          critic_trunk, critic_head     -> <same>
        Only tensors whose shapes match are copied (so the obstacle
        config, where critic_trunk widens, safely skips trunk/head and
        warm-starts just the encoder).  Returns the count copied.
        """
        import torch as _torch
        ckpt = _torch.load(path, map_location="cpu", weights_only=False)
        src = ckpt["policy_state"] if "policy_state" in ckpt else ckpt

        encoder_roots = {
            "blue_input_mlp", "red_input_mlp",
            "bb_edge_mlp", "rb_edge_mlp", "msg_mlp", "update_mlp",
        }
        head_roots = {"critic_trunk", "critic_head"}

        my_sd = self.state_dict()
        n_copied = 0
        for k, v in src.items():
            root = k.split(".", 1)[0]
            if root in encoder_roots:
                target = f"critic_encoder.{k}"
            elif root in head_roots:
                target = k
            else:
                continue
            if target in my_sd and my_sd[target].shape == v.shape:
                my_sd[target].copy_(v)
                n_copied += 1
        self.load_state_dict(my_sd)
        return n_copied

    def load_full_stage4(self, path) -> int:
        """
        Soft-load ALL shape-matching tensors (actor AND critic) from
        another GNNStage4Policy checkpoint.  Used to warm-start the
        crash-avoidance run from a converged obstacle policy
        (e.g. obstacles_v1): the architecture is identical, and the
        per-agent V(s,i) critic keeps the same tensor SHAPES as the
        old pooled V(s), so every actor + critic + head tensor copies
        in.  Returns the count copied.
        """
        import torch as _torch
        ckpt = _torch.load(path, map_location="cpu", weights_only=False)
        src = ckpt["policy_state"] if "policy_state" in ckpt else ckpt
        my_sd = self.state_dict()
        n_copied = 0
        for k, v in src.items():
            if k in my_sd and my_sd[k].shape == v.shape:
                my_sd[k].copy_(v)
                n_copied += 1
        self.load_state_dict(my_sd)
        return n_copied
