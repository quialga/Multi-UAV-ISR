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


class GraphRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        n_envs:        int,
        n_agents:      int,
        action_dim:    int,
        # Dict of key -> per-token feature dim (last dim of the field)
        obs_feature_dims: Dict[str, int],
        # Dict of key -> per-timestep token count (dim between env and feat)
        obs_token_counts: Dict[str, int],
        device:        torch.device,
    ) -> None:
        self.T = rollout_steps
        self.E = n_envs
        self.A = n_agents
        self.action_dim = action_dim
        self.device = device
        self._obs_feature_dims = dict(obs_feature_dims)
        self._obs_token_counts = dict(obs_token_counts)

        # Dict obs: pre-allocated per key.
        self.obs: Dict[str, torch.Tensor] = {}
        for key, feat_dim in obs_feature_dims.items():
            n_tokens = obs_token_counts[key]
            self.obs[key] = torch.zeros(
                (self.T, self.E, n_tokens, feat_dim),
                dtype=torch.float32, device=device,
            )

        # Per (T, E, A) tensors — action-level.
        self.actions   = torch.zeros((self.T, self.E, self.A, action_dim),
                                     dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((self.T, self.E, self.A),
                                     dtype=torch.float32, device=device)

        # Per (T, E) tensors — env-level (centralised).
        self.values    = torch.zeros((self.T, self.E),
                                     dtype=torch.float32, device=device)
        self.rewards   = torch.zeros((self.T, self.E),
                                     dtype=torch.float32, device=device)
        self.dones     = torch.zeros((self.T, self.E),
                                     dtype=torch.float32, device=device)

        # GAE outputs.
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
        obs:       Dict[str, torch.Tensor],   # each (E, n_tokens, feat_dim)
        actions:   torch.Tensor,              # (E, A, action_dim)
        log_probs: torch.Tensor,              # (E, A)
        values:    torch.Tensor,              # (E,)  — centralised
        rewards:   torch.Tensor,              # (E,)  — shared team reward
        dones:     torch.Tensor,              # (E,)
    ) -> None:
        assert self.ptr < self.T, "Buffer full — call reset() between rollouts."
        for key, tensor in obs.items():
            self.obs[key][self.ptr] = tensor
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
        last_value: torch.Tensor,   # (E,) — V(obs_T)
        gamma:      float,
        gae_lambda: float,
    ) -> None:
        assert self.ptr == self.T, "Call compute_gae only after a full rollout."
        adv = torch.zeros(self.E, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            not_done   = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            adv   = delta + gamma * gae_lambda * not_done * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------ #
    #  Minibatch iteration                                                 #
    # ------------------------------------------------------------------ #

    def iter_minibatches(self, mb_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Flatten (T, E) into a batch dim of size T*E and yield shuffled
        minibatches.  Each minibatch item is one graph state (with all
        n_blue agents' actions attached).  Advantages / returns / values
        are broadcast-ready: they carry the env-level dim but the loss
        computation typically broadcasts across the agent dim.
        """
        N = self.T * self.E
        # Flatten obs per key: (T, E, n_tokens, feat) -> (N, n_tokens, feat)
        flat_obs = {
            key: tensor.reshape(N, tensor.shape[2], tensor.shape[3])
            for key, tensor in self.obs.items()
        }
        # Flatten agent-level per (T, E, A, ...) -> (N, A, ...)
        flat_actions   = self.actions.reshape(N, self.A, self.action_dim)
        flat_log_probs = self.log_probs.reshape(N, self.A)
        # Flatten env-level per (T, E) -> (N,)
        flat_values     = self.values.reshape(N)
        flat_advantages = self.advantages.reshape(N)
        flat_returns    = self.returns.reshape(N)

        perm = torch.randperm(N, device=self.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            yield {
                "obs":        {k: v[idx] for k, v in flat_obs.items()},
                "actions":    flat_actions[idx],
                "log_probs":  flat_log_probs[idx],
                "values":     flat_values[idx],
                "advantages": flat_advantages[idx],
                "returns":    flat_returns[idx],
            }


# ---------------------------------------------------------------------------
# Stage 3: recurrent CTDE variant
# ---------------------------------------------------------------------------

class RecurrentGraphRolloutBuffer(GraphRolloutBuffer):
    """
    Extends ``GraphRolloutBuffer`` for the Stage 3 CTDE policy with:

    - Per-edge visibility masks (``bb_visible`` (T, E, n_bb),
      ``rb_visible`` (T, E, n_rb)) — extras that the actor path needs
      but the critic path ignores.  The base feature tensors stored in
      ``self.obs`` are identical between partial-obs and full-state
      views by construction (see stage3_design.md §2.2), so a single
      base dict is enough.
    - Per-(env, blue) hidden state ``hidden`` (T, E, A, d_hidden)
      — the GRU state that was the input at the given step.  Used by
      PPO update to re-condition the actor without back-propagating
      through time (stateless-at-update convention, design §7).

    Minibatch item = one env-timestep transition.  Advantages /
    returns / values remain env-level (broadcast over agents in the
    policy loss).  The stored ``hidden`` is treated as a constant
    during the PPO update — no BPTT.
    """

    def __init__(
        self,
        rollout_steps: int,
        n_envs:        int,
        n_agents:      int,
        action_dim:    int,
        obs_feature_dims: Dict[str, int],
        obs_token_counts: Dict[str, int],
        n_bb_edges:    int,
        n_rb_edges:    int,
        d_hidden:      int,
        device:        torch.device,
    ) -> None:
        super().__init__(
            rollout_steps    = rollout_steps,
            n_envs           = n_envs,
            n_agents         = n_agents,
            action_dim       = action_dim,
            obs_feature_dims = obs_feature_dims,
            obs_token_counts = obs_token_counts,
            device           = device,
        )
        self.n_bb_edges = n_bb_edges
        self.n_rb_edges = n_rb_edges
        self.d_hidden   = d_hidden

        self.bb_visible = torch.zeros((self.T, self.E, n_bb_edges),
                                      dtype=torch.float32, device=device)
        self.rb_visible = torch.zeros((self.T, self.E, n_rb_edges),
                                      dtype=torch.float32, device=device)
        # Hidden state at step t (input to GRU at that step, before the
        # actor forward pass — trainer-owned and reset on episode done).
        self.hidden = torch.zeros((self.T, self.E, self.A, d_hidden),
                                  dtype=torch.float32, device=device)

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def add(  # type: ignore[override]
        self,
        obs:        Dict[str, torch.Tensor],
        actions:    torch.Tensor,
        log_probs:  torch.Tensor,
        values:     torch.Tensor,
        rewards:    torch.Tensor,
        dones:      torch.Tensor,
        bb_visible: torch.Tensor,   # (E, n_bb)
        rb_visible: torch.Tensor,   # (E, n_rb)
        hidden:     torch.Tensor,   # (E, A, d_hidden) — input to GRU at this step
    ) -> None:
        assert self.ptr < self.T, "Buffer full — call reset() between rollouts."
        # Fill visibility + hidden at the current pointer BEFORE super()
        # increments it.
        self.bb_visible[self.ptr] = bb_visible
        self.rb_visible[self.ptr] = rb_visible
        self.hidden[self.ptr]     = hidden
        super().add(obs=obs, actions=actions, log_probs=log_probs,
                    values=values, rewards=rewards, dones=dones)

    # ------------------------------------------------------------------ #
    #  Minibatch iteration                                                 #
    # ------------------------------------------------------------------ #

    def iter_minibatches(  # type: ignore[override]
        self, mb_size: int,
    ) -> Iterator[Dict[str, object]]:
        """
        Same shuffling as the base buffer, but each minibatch also
        includes the visibility masks and hidden states so the PPO
        update can rebuild the (partial_obs, full_state, hidden)
        triple.  Callers construct:

            partial_obs = {**mb["obs"], "bb_edge_visible": mb["bb_visible"],
                                        "rb_edge_visible": mb["rb_visible"]}
            full_state  = mb["obs"]
            hidden      = mb["hidden"]
        """
        N = self.T * self.E
        flat_obs = {
            key: tensor.reshape(N, tensor.shape[2], tensor.shape[3])
            for key, tensor in self.obs.items()
        }
        flat_actions   = self.actions.reshape(N, self.A, self.action_dim)
        flat_log_probs = self.log_probs.reshape(N, self.A)
        flat_values     = self.values.reshape(N)
        flat_advantages = self.advantages.reshape(N)
        flat_returns    = self.returns.reshape(N)

        flat_bb_vis = self.bb_visible.reshape(N, self.n_bb_edges)
        flat_rb_vis = self.rb_visible.reshape(N, self.n_rb_edges)
        flat_hidden = self.hidden.reshape(N, self.A, self.d_hidden)

        perm = torch.randperm(N, device=self.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            yield {
                "obs":        {k: v[idx] for k, v in flat_obs.items()},
                "actions":    flat_actions[idx],
                "log_probs":  flat_log_probs[idx],
                "values":     flat_values[idx],
                "advantages": flat_advantages[idx],
                "returns":    flat_returns[idx],
                "bb_visible": flat_bb_vis[idx],
                "rb_visible": flat_rb_vis[idx],
                "hidden":     flat_hidden[idx],
            }
