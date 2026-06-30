"""
isr/train/buffer.py — PPO rollout buffer with GAE.

Stores one rollout's worth of transitions, then computes GAE
advantages and returns and yields shuffled minibatches for the PPO
update.

Layout
------
All arrays are shaped ``(T, E, A, ...)`` where:
- T = rollout_steps
- E = n_envs
- A = n_agents (blue UAVs that share parameters)

The minibatch flatten collapses T*E*A into one dimension — every blue
agent's transition is a separate training sample.  Because blue UAVs
share parameters AND share the team reward, the value function
gradient signal from all agents at the same env step is highly
correlated, but the differing self_idx_onehot obs feature keeps the
samples non-degenerate.

GAE
---
Generalised Advantage Estimation:

    delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    R_t     = A_t + V_t          (the GAE return target for V_t)

Done mask zeroes future contributions across episode boundaries.
``last_value`` (V_{T}) is the bootstrap from the obs *after* the last
stored step.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        n_envs:        int,
        n_agents:      int,
        obs_dim:       int,
        action_dim:    int,
        device:        torch.device,
    ) -> None:
        self.T = rollout_steps
        self.E = n_envs
        self.A = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Per-step storage.  Pre-allocate and fill by index — faster
        # than appending Python lists.
        self.obs       = torch.zeros((self.T, self.E, self.A, obs_dim),
                                     dtype=torch.float32, device=device)
        self.actions   = torch.zeros((self.T, self.E, self.A, action_dim),
                                     dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((self.T, self.E, self.A),
                                     dtype=torch.float32, device=device)
        self.values    = torch.zeros((self.T, self.E, self.A),
                                     dtype=torch.float32, device=device)
        self.rewards   = torch.zeros((self.T, self.E, self.A),
                                     dtype=torch.float32, device=device)
        # ``dones`` is per-env (the episode either ends for all agents
        # or none — Stage 1 has fully synchronous termination).
        self.dones     = torch.zeros((self.T, self.E),
                                     dtype=torch.float32, device=device)

        # Filled by compute_gae.
        self.advantages = torch.zeros_like(self.rewards)
        self.returns    = torch.zeros_like(self.rewards)

        self.ptr = 0

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs:       torch.Tensor,    # (E, A, obs_dim)
        actions:   torch.Tensor,    # (E, A, action_dim)
        log_probs: torch.Tensor,    # (E, A)
        values:    torch.Tensor,    # (E, A)
        rewards:   torch.Tensor,    # (E, A)
        dones:     torch.Tensor,    # (E,)
    ) -> None:
        """Store one timestep across all envs."""
        assert self.ptr < self.T, "Buffer is full — call reset() between rollouts."
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr]    = values
        self.rewards[self.ptr]   = rewards
        self.dones[self.ptr]     = dones
        self.ptr += 1

    # ------------------------------------------------------------------ #
    #  GAE                                                                 #
    # ------------------------------------------------------------------ #

    def compute_gae(
        self,
        last_value: torch.Tensor,    # (E, A) — V(obs_T)
        gamma:      float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages + returns in place."""
        assert self.ptr == self.T, "Call compute_gae only after a full rollout."
        adv = torch.zeros(self.E, self.A, dtype=torch.float32, device=self.device)
        # ``dones`` is (T, E); broadcast over agents.
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_value   = last_value
            else:
                next_value   = self.values[t + 1]
            # not_done shape: (E,) -> (E, 1) for broadcast over agents
            not_done = (1.0 - self.dones[t]).unsqueeze(-1)
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            adv   = delta + gamma * gae_lambda * not_done * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------ #
    #  Minibatch iterator                                                  #
    # ------------------------------------------------------------------ #

    def iter_minibatches(self, mb_size: int) -> Iterator[dict]:
        """
        Flatten (T, E, A) into (T*E*A,) and yield shuffled minibatches.
        Each call shuffles a fresh permutation.
        """
        N = self.T * self.E * self.A
        flat_obs        = self.obs.reshape(N, self.obs_dim)
        flat_actions    = self.actions.reshape(N, self.action_dim)
        flat_log_probs  = self.log_probs.reshape(N)
        flat_values     = self.values.reshape(N)
        flat_advantages = self.advantages.reshape(N)
        flat_returns    = self.returns.reshape(N)

        perm = torch.randperm(N, device=self.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            yield {
                "obs":         flat_obs[idx],
                "actions":     flat_actions[idx],
                "log_probs":   flat_log_probs[idx],
                "values":      flat_values[idx],
                "advantages":  flat_advantages[idx],
                "returns":     flat_returns[idx],
            }
