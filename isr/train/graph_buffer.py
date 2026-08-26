"""
isr/train/graph_buffer.py — PPO rollout buffer for the Stage 2 GNN.

Similar layout to ``RolloutBuffer`` but adapted for two facts about
the GNN + CTDE setup:

1. **Observation is a dict of tensors, not a flat tensor.**  Each key
   stores its own (T, E, ...)-shaped array.
2. **The critic produces one centralised value per env-timestep**, not
   one per agent.  Rewards under shared team reward are also per env
   (all agents get identical values; we don't need to store them
   per-agent).  Actions and log_probs remain per (env, agent).

GAE is computed per env from the single V trace, yielding per-env
advantages that broadcast to all agents when we compute the policy
loss.
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch


# ---------------------------------------------------------------------------
# Stage 4: belief-map buffer
# ---------------------------------------------------------------------------

class Stage4RolloutBuffer:
    """
    Buffer for the Stage 4 (v6) typed-GNN CTDE recurrent policy.

    Generic dict-of-tensors design: the set of observation keys is
    discovered from the first ``add`` call and each key gets its own
    (T, E, ...) tensor.  This keeps the buffer agnostic to the exact
    v6 obs schema (blue/red/obstacle node features + bb/rb/ob edge
    features + visibility masks, both belief-derived actor keys and
    ``true_*`` critic keys).

    RL bookkeeping (actions / log_probs / values / rewards / dones /
    hidden / GAE) is PER-AGENT for the crash-avoidance extension:
    individual rewards ``r_i = r_team + r_crash_i`` and an agent-
    conditioned value ``V(s, i)`` make values / rewards / advantages /
    returns all ``(T, E, A)``.  Only ``dones`` stay per-env ``(T, E)``
    (a crash does NOT end the episode; it ends for all agents together
    on timeout / all-caught) and broadcast over the agent dim in GAE.
    Actions / log_probs / hidden are per-agent as before.

    Minibatch item = one env-timestep transition (stateless-at-update).
    """

    def __init__(
        self,
        rollout_steps:  int,
        n_envs:         int,
        n_agents:       int,
        action_dim:     int,
        d_hidden:       int,
        device:         torch.device,
    ) -> None:
        self.T = int(rollout_steps)
        self.E = int(n_envs)
        self.A = int(n_agents)
        self.action_dim = int(action_dim)
        self.d_hidden   = int(d_hidden)
        self.device     = device

        # Obs tensors are lazily allocated on the first add() from the
        # observed key -> shape schema.
        self._obs: Dict[str, torch.Tensor] = {}

        self.actions   = torch.zeros((self.T, self.E, self.A, self.action_dim),
                                     dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((self.T, self.E, self.A),
                                     dtype=torch.float32, device=device)
        self.hidden    = torch.zeros((self.T, self.E, self.A, self.d_hidden),
                                     dtype=torch.float32, device=device)
        # PER-AGENT RL tensors: individual rewards r_i = r_team + r_crash_i,
        # per-agent value V(s, i), so GAE/advantage/return are per (E, A).
        self.values     = torch.zeros((self.T, self.E, self.A), dtype=torch.float32, device=device)
        self.rewards    = torch.zeros((self.T, self.E, self.A), dtype=torch.float32, device=device)
        self.advantages = torch.zeros_like(self.rewards)
        self.returns    = torch.zeros_like(self.rewards)
        # dones are per-ENV (episode ends for all agents together — we
        # do NOT terminate on crash), broadcast over agents in GAE.
        self.dones      = torch.zeros((self.T, self.E), dtype=torch.float32, device=device)

        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs:       Dict[str, torch.Tensor],   # per-key (E, ...) tensors
        actions:   torch.Tensor,              # (E, A, action_dim)
        log_probs: torch.Tensor,              # (E, A)
        values:    torch.Tensor,              # (E, A)  per-agent V(s, i)
        rewards:   torch.Tensor,              # (E, A)  per-agent r_i
        dones:     torch.Tensor,              # (E,)    per-env
        hidden:    torch.Tensor,              # (E, A, d_hidden)
    ) -> None:
        assert self.ptr < self.T, "Buffer full — call reset() between rollouts."
        if not self._obs:
            for k, v in obs.items():
                self._obs[k] = torch.zeros(
                    (self.T,) + tuple(v.shape), dtype=torch.float32,
                    device=self.device,
                )
        for k, v in obs.items():
            self._obs[k][self.ptr] = v
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr]    = values
        self.rewards[self.ptr]   = rewards
        self.dones[self.ptr]     = dones
        self.hidden[self.ptr]    = hidden
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,   # (E, A)  per-agent bootstrap
        gamma:      float,
        gae_lambda: float,
    ) -> None:
        """Per-agent GAE.  ``dones`` are per-env and broadcast over the
        agent dim (an episode ends for all agents at once)."""
        assert self.ptr == self.T
        adv = torch.zeros((self.E, self.A), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            not_done   = (1.0 - self.dones[t]).unsqueeze(-1)   # (E, 1) -> broadcast
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            adv   = delta + gamma * gae_lambda * not_done * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def iter_minibatches(
        self, mb_size: int,
    ) -> Iterator[Dict[str, object]]:
        """
        Shuffle over (env-timestep) transitions.  Each minibatch dict
        carries every stored obs key plus the RL tensors; the PPO
        update splits obs keys into the actor / critic dicts by name.
        """
        N = self.T * self.E
        flat_obs = {
            k: v.reshape((N,) + tuple(v.shape[2:]))
            for k, v in self._obs.items()
        }
        flat_act = self.actions.reshape(N, self.A, self.action_dim)
        flat_lp  = self.log_probs.reshape(N, self.A)
        flat_val = self.values.reshape(N, self.A)       # per-agent
        flat_adv = self.advantages.reshape(N, self.A)   # per-agent
        flat_ret = self.returns.reshape(N, self.A)      # per-agent
        flat_hid = self.hidden.reshape(N, self.A, self.d_hidden)

        perm = torch.randperm(N, device=self.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            out: Dict[str, object] = {k: v[idx] for k, v in flat_obs.items()}
            out["actions"]    = flat_act[idx]
            out["log_probs"]  = flat_lp[idx]
            out["values"]     = flat_val[idx]
            out["advantages"] = flat_adv[idx]
            out["returns"]    = flat_ret[idx]
            out["hidden"]     = flat_hid[idx]
            yield out
