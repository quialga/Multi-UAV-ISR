"""
isr/agents/policy_loader.py — Load a trained GNN checkpoint + adapt it
to the HeuristicBlueAgent interface so it can be dropped into the
rendering / evaluation scripts that already accept heuristic policies.

Two entry points:

- ``load_policy(path, device)``: load a checkpoint dict written by
  ``scripts/train_stage1.py`` and return a constructed
  ``GNNActorCritic`` on the requested device.

- ``TrainedBlueAgent(policy, device, deterministic=True)``: wraps a
  loaded policy as a blue agent with ``act(obs, env, agent) -> action``,
  matching ``HeuristicBlueAgent``.  Use ``deterministic=True`` (default)
  for evaluation — sample distribution mean, no exploration noise.

The MLP-loading path was removed in the July-2026 cleanup after the
Stage 2 scaling experiment (see docs/stage2_results.md).  Pre-Stage-2
MLP checkpoints raise a clear error at load time — their eval numbers
are preserved in docs/stage1_results.md and docs/stage1_analysis.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from isr.agents.heuristics import HeuristicBlueAgent
from isr.agents.gnn_policy import GNNActorCritic
from isr.agents.gnn_ctde_policy import GNNCTDEPolicy
from isr.configs.stage1_default import STAGE1_DEFAULTS
from isr.env.pursuit_env import PursuitEnv


_BASE_OBS_KEYS = (
    "blue_features", "red_features",
    "bb_edge_features", "rb_edge_features",
)


def load_policy(
    path:   Union[str, Path],
    device: torch.device,
):
    """
    Load a checkpoint and return a constructed + loaded policy.

    Dispatches on ``args['policy_type']``:
      - ``'gnn'``       -> ``GNNActorCritic`` (Stage 1/2 fully-obs).
      - ``'gnn_ctde'``  -> ``GNNCTDEPolicy``  (Stage 3 partial-obs +
                            GRU + CTDE critic).

    Pre-Stage-2 MLP checkpoints raise a clear error at load time.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    policy_type = args.get("policy_type", "mlp")

    if policy_type == "gnn":
        policy = GNNActorCritic(
            n_blue        = int(args["n_blue"]),
            n_red         = int(args["n_red"]),
            blue_feat_dim = 8,
            red_feat_dim  = 1,
            edge_feat_dim = 7,
            action_dim    = 2,
            d_hidden      = int(args.get("d_hidden", 64)),
            n_msg_rounds  = int(args.get("n_msg_rounds", 2)),
            init_log_std  = STAGE1_DEFAULTS["init_log_std"],
        ).to(device)
    elif policy_type == "gnn_ctde":
        policy = GNNCTDEPolicy(
            n_blue            = int(args["n_blue"]),
            n_red             = int(args["n_red"]),
            blue_feat_dim     = 8,
            red_feat_dim      = 1,
            edge_feat_dim     = 7,
            action_dim        = 2,
            d_hidden          = int(args.get("d_hidden", 64)),
            n_msg_rounds      = int(args.get("n_msg_rounds", 2)),
            init_log_std      = STAGE1_DEFAULTS["init_log_std"],
            use_hidden_in_gnn = bool(args.get("share_hidden_via_gnn",
                                              args.get("use_hidden_in_gnn",
                                                       False))),
        ).to(device)
    else:
        raise RuntimeError(
            f"Checkpoint at {path} has policy_type={policy_type!r}; only "
            f"'gnn' (Stage 1/2) and 'gnn_ctde' (Stage 3) are supported "
            f"after the July-2026 MLP cleanup.  See docs/stage2_results.md."
        )

    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    return policy


def build_trained_agent(
    policy,
    device:        torch.device,
    deterministic: bool = True,
):
    """
    Return the right ``HeuristicBlueAgent`` adapter for the given
    loaded policy — dispatches on the policy's class rather than
    forcing the caller to know.
    """
    if isinstance(policy, GNNCTDEPolicy):
        return TrainedCTDEBlueAgent(policy, device, deterministic)
    if isinstance(policy, GNNActorCritic):
        return TrainedBlueAgent(policy, device, deterministic)
    raise TypeError(f"Unsupported policy class: {type(policy).__name__}")


class TrainedBlueAgent(HeuristicBlueAgent):
    """
    Adapter that lets a trained ``GNNActorCritic`` be used wherever a
    ``HeuristicBlueAgent`` is expected (renderer, eval table, etc.).

    Per-agent calls run a tiny single-sample forward pass on the graph
    obs — fine for evaluation but not for vectorised rollouts (the
    trainer uses batched forwards directly).
    """

    def __init__(
        self,
        policy:        GNNActorCritic,
        device:        torch.device,
        deterministic: bool = True,
    ) -> None:
        self.policy        = policy
        self.device        = device
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, env: PursuitEnv, agent: str) -> np.ndarray:
        # Grab the whole graph and identify the acting agent's index.
        s = env.structured_observation()
        obs_dict = {
            k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
            for k, v in s.items()
        }
        with torch.no_grad():
            if self.deterministic:
                action_t = self.policy.act_deterministic(obs_dict)
            else:
                action_t, _, _, _ = self.policy.get_action_and_value(obs_dict)
        # action_t: (1, N_blue, 2).  Pick out the acting agent.
        idx = env.possible_agents.index(agent)
        a = action_t[0, idx].cpu().numpy().astype(np.float32)
        return np.clip(a, -1.0, 1.0)


class TrainedCTDEBlueAgent(HeuristicBlueAgent):
    """
    Adapter for the Stage 3 CTDE + recurrent policy (``GNNCTDEPolicy``).

    Two differences vs ``TrainedBlueAgent`` matter at eval time:

    1. **Hidden state.**  The actor's per-blue GRU carries a hidden
       state across timesteps.  It must be initialised to zeros at
       episode start (a fresh instance of this class per episode is
       enough — that's what ``eval_matrix`` does) and advanced once
       per env-step.
    2. **Partial obs.**  The actor consumes
       ``structured_partial_observation()`` (base features + edge
       visibility masks), not ``structured_observation()``.  This is
       the OOD-safe path — the policy was never trained under masks-
       all-ones and would behave weirdly under full obs.

    Because ``run_episode`` calls ``.act(obs, env, agent)`` once per
    (agent, step) — i.e. ``N_blue`` calls per env-step — we run the
    actor forward once per new ``env._t`` and cache the joint action
    tensor across the per-agent slot-ins.
    """

    def __init__(
        self,
        policy:        GNNCTDEPolicy,
        device:        torch.device,
        deterministic: bool = True,
    ) -> None:
        self.policy        = policy
        self.device        = device
        self.deterministic = deterministic
        # Initial hidden = zeros; shape (1, N_blue, d_hidden).
        self.hidden = policy.initial_hidden(n_envs=1, device=device)
        # Per-step cache: valid iff self._cached_t == env._t.
        self._cached_t:           int   = -1
        self._cached_actions           = None    # (1, N_blue, action_dim) tensor
        self._pending_new_hidden       = None    # rolled into self.hidden on next new step

    def act(self, obs: np.ndarray, env: PursuitEnv, agent: str) -> np.ndarray:
        if int(env._t) != self._cached_t:
            # New env-step.  Roll the pending hidden into the current one.
            if self._pending_new_hidden is not None:
                self.hidden = self._pending_new_hidden

            partial = env.structured_partial_observation()
            obs_dict = {
                k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
                for k, v in partial.items()
            }
            with torch.no_grad():
                if self.deterministic:
                    action_t, new_hidden = self.policy.act_deterministic(
                        obs_dict, self.hidden,
                    )
                else:
                    full_state = {k: obs_dict[k] for k in _BASE_OBS_KEYS}
                    action_t, _, _, _, new_hidden = self.policy.get_action_and_value(
                        obs_dict, full_state, self.hidden,
                    )
            self._cached_actions     = action_t          # (1, N_blue, 2)
            self._pending_new_hidden = new_hidden
            self._cached_t           = int(env._t)

        idx = env.possible_agents.index(agent)
        a = self._cached_actions[0, idx].cpu().numpy().astype(np.float32)
        return np.clip(a, -1.0, 1.0)
