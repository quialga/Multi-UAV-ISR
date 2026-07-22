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

The shapes above describe ``ppo_update`` (the shared-reward Stage 1/2
path).  ``ppo_update_stage4`` is now PER-AGENT: values / advantages /
returns are ``(mb, A)`` (agent-conditioned ``V(s, i)`` +
``r_i = r_team + r_crash_i``) and line up with ``new_log_probs``
directly, no ``(mb,)``→``(mb, A)`` broadcast.  See that function.

The MLP + flat-obs PPO update was removed in the July-2026 cleanup
after the Stage 2 scaling experiment (see docs/stage2_results.md).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from isr.train.graph_buffer import (
    GraphRolloutBuffer, Stage4RolloutBuffer,
)


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
# Stage 4: belief-map + CTDE + recurrent PPO update
# ---------------------------------------------------------------------------

def ppo_update_stage4(
    policy,
    optimizer:       torch.optim.Optimizer,
    buffer:          Stage4RolloutBuffer,
    *,
    clip_eps:        float,
    ent_coef:        float,
    vf_coef:         float,
    max_grad_norm:   float,
    n_epochs:        int,
    mb_size:         int,
    value_clip:      bool  = True,
    normalize_adv:   bool  = True,
    target_kl:       float = None,
    aux_hidden_coef: float = 0.0,
) -> Dict[str, float]:
    """
    Stage 4 (v6) PPO update.  Standard clipped-objective recurrent CTDE
    PPO.  Each minibatch carries the full v6 obs schema
    (belief-derived actor keys + ``true_*`` critic keys); it is split
    into ``partial_obs`` / ``full_state`` via ``split_stage4_obs`` and
    forwarded to ``policy.get_action_and_value``.

    Optional aux hidden loss (Stage 3 opt-C, ported):
        L_aux = MSE(actor_h_blue, critic_h_blue.detach())
    encourages the actor's per-blue GNN embedding (from belief-derived
    noisy positions) to match the critic's (from ground truth).  Only
    active when ``aux_hidden_coef > 0``; live-critic target (no freeze).
    """
    from isr.agents.gnn_stage4_policy import split_stage4_obs

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

    for epoch in range(n_epochs):
        epoch_kl_sum = 0.0
        epoch_mb_count = 0
        for batch in buffer.iter_minibatches(mb_size):
            partial_obs, full_state = split_stage4_obs(batch)
            old_actions   = batch["actions"]
            old_log_probs = batch["log_probs"]
            old_values    = batch["values"]
            advantages    = batch["advantages"]
            returns       = batch["returns"]
            hidden        = batch["hidden"]

            if normalize_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            (_, new_log_probs, entropy, new_values, _,
             actor_h_blue, critic_h_blue) = \
                policy.get_action_and_value(
                    partial_obs, full_state, hidden, action=old_actions,
                )

            # advantages / values / returns are now PER-AGENT (mb, A),
            # matching new_log_probs / new_values (mb, A) directly -- no
            # per-env -> per-agent broadcast needed anymore.
            log_ratio = new_log_probs - old_log_probs
            ratio     = log_ratio.exp()
            surr1     = ratio * advantages
            surr2     = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            if value_clip:
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -clip_eps, clip_eps,
                )
                v_loss_u = (new_values - returns).pow(2)
                v_loss_c = (v_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_u, v_loss_c).mean()
            else:
                value_loss = 0.5 * (new_values - returns).pow(2).mean()

            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            if aux_hidden_coef > 0.0:
                aux_loss = ((actor_h_blue - critic_h_blue.detach()) ** 2).mean()
                loss = loss + aux_hidden_coef * aux_loss
                metrics["aux_hidden_loss"] += aux_loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                metrics["policy_loss"]  += policy_loss.item()
                metrics["value_loss"]   += value_loss.item()
                metrics["entropy"]      += (-entropy_loss).item()
                metrics["approx_kl"]    += approx_kl
                metrics["clip_frac"]    += clip_frac
                metrics["n_minibatches"] += 1
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
