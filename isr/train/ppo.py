"""
isr/train/ppo.py — PPO clipped-objective update for the GNN + CTDE
critic path.

Standard PPO formulation (Schulman et al., 2017):

    L_policy   = -E[ min(r * A, clip(r, 1-eps, 1+eps) * A) ]
    L_value    =  E[ (V_pred - R)^2 ]   (optionally clipped)
    L_entropy = -E[ H(pi) ]
    L_total    = L_policy + vf_coef * L_value + ent_coef * L_entropy

where ``r`` is the importance ratio ``exp(new_log_prob - old_log_prob)``,
``A`` are GAE advantages, ``R`` are the bootstrapped GAE returns.

Shape notes for the GNN + CTDE setup:
- ``obs`` is a dict of tensors (not a single tensor).
- ``actions`` / ``log_probs`` / ``entropy`` carry a per-agent dim
  ``(mb, N_blue, ...)`` — we broadcast the ``(mb,)`` advantages
  against them.
- ``values`` are ``(mb,)`` — one centralised V per graph state.

The MLP + flat-obs PPO update was removed in the July-2026 cleanup
after the Stage 2 scaling experiment (see docs/stage2_results.md).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from isr.train.graph_buffer import GraphRolloutBuffer, RecurrentGraphRolloutBuffer


def ppo_update(
    policy,
    optimizer:     torch.optim.Optimizer,
    buffer:        GraphRolloutBuffer,
    *,
    clip_eps:      float,
    ent_coef:      float,
    vf_coef:       float,
    max_grad_norm: float,
    n_epochs:      int,
    mb_size:       int,
    value_clip:    bool = True,
    normalize_adv: bool = True,
) -> Dict[str, float]:
    """
    Run ``n_epochs`` passes of minibatch SGD over the buffer.  Returns
    averaged diagnostics for logging.
    """
    metrics = {
        "policy_loss":  0.0,
        "value_loss":   0.0,
        "entropy":      0.0,
        "approx_kl":    0.0,
        "clip_frac":    0.0,
        "n_minibatches": 0,
    }

    for _ in range(n_epochs):
        for batch in buffer.iter_minibatches(mb_size):
            obs           = batch["obs"]            # dict of (mb, ...) tensors
            old_actions   = batch["actions"]        # (mb, N_blue, action_dim)
            old_log_probs = batch["log_probs"]      # (mb, N_blue)
            old_values    = batch["values"]         # (mb,)
            advantages    = batch["advantages"]     # (mb,)
            returns       = batch["returns"]        # (mb,)

            if normalize_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # New forward pass through the GNN.
            _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                obs, action=old_actions,
            )
            # new_log_probs: (mb, N_blue); entropy: (mb, N_blue); new_values: (mb,)

            # Broadcast advantages against per-agent log-prob ratio.
            adv_bcast = advantages.unsqueeze(-1)                  # (mb, 1)
            log_ratio = new_log_probs - old_log_probs             # (mb, N_blue)
            ratio     = log_ratio.exp()
            surr1     = ratio * adv_bcast
            surr2     = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_bcast
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss on the (mb,)-shaped centralised value.
            if value_clip:
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -clip_eps, clip_eps,
                )
                v_loss_unclipped = (new_values - returns).pow(2)
                v_loss_clipped   = (v_clipped  - returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * (new_values - returns).pow(2).mean()

            entropy_loss = -entropy.mean()   # mean over (mb, N_blue)
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # Schulman's "k3" approximate KL — unbiased and non-neg on avg.
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                metrics["policy_loss"]   += policy_loss.item()
                metrics["value_loss"]    += value_loss.item()
                metrics["entropy"]       += (-entropy_loss).item()
                metrics["approx_kl"]     += approx_kl
                metrics["clip_frac"]     += clip_frac
                metrics["n_minibatches"] += 1

    n = max(metrics["n_minibatches"], 1)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        metrics[k] /= n
    return metrics


# ---------------------------------------------------------------------------
# Stage 3: CTDE + recurrent PPO update (with optional aux hidden loss)
# ---------------------------------------------------------------------------

def ppo_update_ctde(
    policy,
    optimizer:      torch.optim.Optimizer,
    buffer:         RecurrentGraphRolloutBuffer,
    *,
    clip_eps:       float,
    ent_coef:       float,
    vf_coef:        float,
    max_grad_norm:  float,
    n_epochs:       int,
    mb_size:        int,
    value_clip:     bool  = True,
    normalize_adv:  bool  = True,
    target_kl:      float = None,   # optional early stop on epoch mean KL
    aux_hidden_coef:float = 0.0,    # 0.0 disables belief-state distillation
) -> Dict[str, float]:
    """
    Stage 3 PPO update: recurrent actor + centralised (CTDE) critic.

    Same clipped objective + optionally clipped value loss as
    ``ppo_update``, but for each minibatch we rebuild the
    (partial_obs, full_state, hidden) triple that
    ``GNNCTDEPolicy.get_action_and_value`` expects.  The stored
    ``hidden`` is treated as a constant (stateless-at-update, see
    stage3_design.md §7) — no back-prop through time.

    Adds an optional early-stop knob ``target_kl``: if the mean
    Schulman-k3 KL over an epoch exceeds ``target_kl``, subsequent
    epochs in this call are skipped.  Motivated by the higher
    variance of the recurrent actor at scale.

    Optional belief-state distillation via ``aux_hidden_coef > 0``.
    When on, we regress the actor's partial-obs GNN encoder output
    (per-blue node embeddings, pre-GRU) toward the CRITIC encoder's
    full-obs output.  The critic's encoder has separate weights,
    is trained by value_loss under full observability, and is
    warm-started from a Stage 2 GNN checkpoint — so its embeddings
    encode informative state features (target positions, relative
    geometries).  The aux loss teaches the actor to approximate
    those embeddings from the masked view.

    Design note (v2 — see docs/stage3_design.md follow-up): the
    v1 formulation regressed the actor's post-GRU hidden state
    against a same-weights full-obs forward.  That trivially
    degenerated to "encoder ignores edge messages" (aux → 0, no
    learning) because both forwards shared weights.  The critic-
    encoder-as-oracle formulation has separate weights, so no
    such degenerate solution exists; the aux loss has an
    irreducible floor > 0 whenever partial obs genuinely lacks
    information the critic sees.

    When ``sensor_radius`` is effectively infinite the actor sees
    the same input as the critic and the aux loss reduces to the
    inherent difference between the two encoders' weights — small
    but non-zero.

    Returns averaged diagnostics for logging (aux_hidden_loss
    included as 0.0 when the aux path is disabled).
    """
    metrics = {
        "policy_loss":     0.0,
        "value_loss":      0.0,
        "entropy":         0.0,
        "approx_kl":       0.0,
        "clip_frac":       0.0,
        "aux_hidden_loss": 0.0,
        "n_minibatches":   0,
        "n_epochs_run":    0,
    }

    base_keys = ("blue_features", "red_features",
                 "bb_edge_features", "rb_edge_features")

    for epoch in range(n_epochs):
        epoch_kl_sum = 0.0
        epoch_mb_count = 0
        for batch in buffer.iter_minibatches(mb_size):
            obs           = batch["obs"]            # dict (mb, ...)
            old_actions   = batch["actions"]        # (mb, N_blue, action_dim)
            old_log_probs = batch["log_probs"]      # (mb, N_blue)
            old_values    = batch["values"]         # (mb,)
            advantages    = batch["advantages"]     # (mb,)
            returns       = batch["returns"]        # (mb,)
            bb_visible    = batch["bb_visible"]     # (mb, n_bb)
            rb_visible    = batch["rb_visible"]     # (mb, n_rb)
            hidden        = batch["hidden"]         # (mb, N_blue, d_hidden)

            if normalize_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Build the two views the CTDE policy consumes.
            partial_obs = {
                **{k: obs[k] for k in base_keys},
                "bb_edge_visible": bb_visible,
                "rb_edge_visible": rb_visible,
            }
            full_state = {k: obs[k] for k in base_keys}

            # New forward pass.  Keep new_hidden_partial for the aux
            # belief-state MSE (below); otherwise it's discarded per
            # the stateless-at-update convention.
            _, new_log_probs, entropy, new_values, new_hidden_partial = \
                policy.get_action_and_value(
                    partial_obs, full_state, hidden, action=old_actions,
                )

            adv_bcast = advantages.unsqueeze(-1)                # (mb, 1)
            log_ratio = new_log_probs - old_log_probs           # (mb, N_blue)
            ratio     = log_ratio.exp()
            surr1     = ratio * adv_bcast
            surr2     = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_bcast
            policy_loss = -torch.min(surr1, surr2).mean()

            if value_clip:
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -clip_eps, clip_eps,
                )
                v_loss_unclipped = (new_values - returns).pow(2)
                v_loss_clipped   = (v_clipped  - returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * (new_values - returns).pow(2).mean()

            entropy_loss = -entropy.mean()

            # Auxiliary belief-state distillation, critic-encoder-as-oracle
            # formulation (see docstring for the v1-vs-v2 design note).
            # Off when aux_hidden_coef == 0 — the tensor is still constructed
            # so the metric schema is consistent.
            if aux_hidden_coef > 0.0:
                # Actor's partial-obs pre-GRU node embeddings (trainable).
                h_blue_actor, _ = policy.actor_encoder(
                    partial_obs["blue_features"],
                    partial_obs["red_features"],
                    partial_obs["bb_edge_features"],
                    partial_obs["rb_edge_features"],
                    bb_visible=partial_obs["bb_edge_visible"],
                    rb_visible=partial_obs["rb_edge_visible"],
                )   # (mb, N_blue, d_hidden)
                # Critic's full-obs pre-aggregation node embeddings.
                # Detached — only the actor's encoder is trained by this loss.
                with torch.no_grad():
                    h_blue_critic, _ = policy.critic_encoder(
                        full_state["blue_features"],
                        full_state["red_features"],
                        full_state["bb_edge_features"],
                        full_state["rb_edge_features"],
                        # No visibility masks — critic sees the full graph.
                    )
                aux_hidden_loss = (h_blue_actor - h_blue_critic).pow(2).mean()
            else:
                aux_hidden_loss = torch.zeros((), device=new_values.device)

            loss = (
                policy_loss
                + vf_coef  * value_loss
                + ent_coef * entropy_loss
                + aux_hidden_coef * aux_hidden_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                metrics["policy_loss"]     += policy_loss.item()
                metrics["value_loss"]      += value_loss.item()
                metrics["entropy"]         += (-entropy_loss).item()
                metrics["approx_kl"]       += approx_kl
                metrics["clip_frac"]       += clip_frac
                metrics["aux_hidden_loss"] += aux_hidden_loss.item()
                metrics["n_minibatches"]   += 1
                epoch_kl_sum   += approx_kl
                epoch_mb_count += 1

        metrics["n_epochs_run"] = epoch + 1
        if target_kl is not None and epoch_mb_count > 0:
            if (epoch_kl_sum / epoch_mb_count) > target_kl:
                break

    n = max(metrics["n_minibatches"], 1)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl",
              "clip_frac", "aux_hidden_loss"):
        metrics[k] /= n
    return metrics
