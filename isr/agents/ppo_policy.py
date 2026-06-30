"""
isr/agents/ppo_policy.py — Actor-critic network for Stage 1 PPO.

Shared MLP trunk -> separate actor and critic heads.  Continuous
diagonal Gaussian actor with state-independent (per-dimension learned)
log_std.  Standard PPO continuous setup.

Parameter sharing across blue UAVs is enforced by *only ever
constructing one ActorCritic* and calling it once per blue agent per
env step (the trainer flattens (n_envs, n_agents, obs_dim) into one
batch dimension).  The agents differ only in their input obs vector,
which carries a self_idx_onehot block.

For Stage 1 we keep the actor mean *unbounded* (no tanh squash) and
rely on env-side clipping to [-1,1].  This is the simplest correct
choice; the alternative (tanh-Gaussian with change-of-variables
log-prob correction) is what SAC uses but adds noise that PPO doesn't
benefit from at this scale.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Linear, std: float = 1.4142135623730951,
                bias_const: float = 0.0) -> nn.Linear:
    """
    Orthogonal init with gain sqrt(2) (the openai/baselines default for
    hidden layers).  Heads override the gain.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(in_dim: int, hidden: List[int], activation=nn.Tanh) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(_layer_init(nn.Linear(prev, h)))
        layers.append(activation())
        prev = h
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor-critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared MLP trunk + two heads (policy mean, value).

    Parameters
    ----------
    obs_dim       Dimension of the observation vector for one agent.
    action_dim    Dimension of the action vector for one agent (2 for
                  Stage 1 — 2D acceleration).
    hidden        List of hidden layer sizes for the trunk.
    init_log_std  Initial value for the per-dimension learned log_std
                  parameter.  log_std=0 -> std=1.
    """

    def __init__(
        self,
        obs_dim:      int,
        action_dim:   int,
        hidden:       List[int] = (256, 256),
        init_log_std: float     = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.trunk = _build_mlp(obs_dim, list(hidden))

        # Heads use small init for the policy and standard for the value
        # — the small policy init (gain 0.01) is a common PPO trick to
        # keep initial actions near-zero so exploration is reasonable
        # from step 1.
        self.actor_mean = _layer_init(nn.Linear(hidden[-1], action_dim), std=0.01)
        self.critic     = _layer_init(nn.Linear(hidden[-1], 1),          std=1.0)

        # State-independent learned log_std (per action dim).
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32)
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (mean, log_std_expanded, value).

        ``obs``: (B, obs_dim) where B = n_envs * n_agents during rollout
                 collection, or whatever batch size the trainer hands in.
        """
        h     = self.trunk(obs)
        mean  = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        log_std = self.actor_log_std.expand_as(mean)
        return mean, log_std, value

    # ------------------------------------------------------------------ #
    #  Action sampling / evaluation helpers                               #
    # ------------------------------------------------------------------ #

    def get_action_and_value(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action (if ``action`` is None) or evaluate a given action.

        Returns (action, log_prob, entropy, value).  log_prob and
        entropy are summed over the action dimension so they're scalar
        per sample.

        The unbounded Gaussian sample may fall outside [-1,1]; the env
        clips on the way in.  The log_prob is computed on the unclipped
        sample, which is the standard convention.
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy action = distribution mean.  Used at evaluation time."""
        mean, _, _ = self.forward(obs)
        return mean
