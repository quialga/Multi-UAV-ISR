"""
isr/train/ppo.py — PPO clipped-objective update.

One function, ``ppo_update``, that does N epochs of minibatch SGD over
a filled rollout buffer.  The policy / value loss + entropy bonus is
the standard formulation (Schulman et al., 2017):

    L_policy   = -E[ min(r * A, clip(r, 1-eps, 1+eps) * A) ]
    L_value    =  E[ (V_pred - R)^2 ]   (optionally clipped)
    L_entropy = -E[ H(pi) ]
    L_total    = L_policy + vf_coef * L_value + ent_coef * L_entropy

where ``r`` is the importance ratio ``exp(new_log_prob - old_log_prob)``,
``A`` are GAE advantages, ``R`` are the bootstrapped GAE returns.

The value clipping (``value_clip=True``) is a small stabiliser used in
openai/baselines and most PPO implementations: clip the *change* in
the value prediction to within ``clip_eps`` of the old value, then take
the max of (clipped, unclipped)^2.  Helps when value estimates have
heavy gradients in the early phase.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import Normal

from isr.train.buffer import RolloutBuffer


def ppo_update(
    policy:        nn.Module,
    optimizer:     torch.optim.Optimizer,
    buffer:        RolloutBuffer,
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
    an averaged set of diagnostics for logging.
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
            obs            = batch["obs"]
            old_actions    = batch["actions"]
            old_log_probs  = batch["log_probs"]
            old_values     = batch["values"]
            advantages     = batch["advantages"]
            returns        = batch["returns"]

            if normalize_adv:
                adv_mean = advantages.mean()
                adv_std  = advantages.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            # New forward pass.
            _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                obs, action=old_actions,
            )

            # Policy loss with clipped surrogate.
            log_ratio = new_log_probs - old_log_probs
            ratio = log_ratio.exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss.
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
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            # Diagnostics
            with torch.no_grad():
                # Approximate KL — Schulman's "k3" estimator from the
                # PPO implementation details note; unbiased and
                # non-negative on average.
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
