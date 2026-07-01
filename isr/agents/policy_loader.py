"""
isr/agents/policy_loader.py — Load trained PPO checkpoints + adapt them
to the HeuristicBlueAgent interface so they can be dropped into the
rendering / evaluation scripts that already accept heuristic policies.

Two entry points:

- ``load_policy(path, device)``: load a checkpoint dict written by
  ``scripts/train_stage1.py`` and return a constructed ``ActorCritic``
  whose obs_dim / action_dim / hidden are reconstructed from the saved
  ``args`` block.

- ``TrainedBlueAgent(policy, device, deterministic=True)``: wraps a
  loaded policy as a blue agent with ``act(obs, env, agent) -> action``,
  matching ``HeuristicBlueAgent``.  Use ``deterministic=True`` (default)
  for evaluation — sample distribution mean, no exploration noise.
  Use ``False`` to roll out from the stochastic policy (e.g. for
  ensembling or noise-diversity sanity checks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from isr.agents.heuristics import HeuristicBlueAgent
from isr.agents.ppo_policy import ActorCritic
from isr.agents.gnn_policy import GNNActorCritic
from isr.configs.stage1_default import STAGE1_DEFAULTS
from isr.env.pursuit_env import PursuitEnv


def _infer_obs_dim(args_block: Dict) -> int:
    """
    Reconstruct the obs_dim from the saved training args.

    v2 checkpoints save ``obs_dim`` explicitly in ``args`` — we prefer
    that when present.  If missing (legacy v1 checkpoints), fall back
    to the v1 layout formula:

        v1: obs_dim = 4*N_blue + 4*N_red + N_red + N_blue + 1

    Note: v2 checkpoints cannot be loaded with the v1 formula because
    the obs_dim differs (v1=26 vs v2=38 for N=3, M=2).
    """
    if "obs_dim" in args_block:
        return int(args_block["obs_dim"])
    n_blue = args_block["n_blue"]
    n_red  = args_block["n_red"]
    return (4 * n_blue) + (4 * n_red) + n_red + n_blue + 1


def load_policy(
    path:   Union[str, Path],
    device: torch.device,
):
    """
    Load a checkpoint and return a constructed + loaded policy on the
    requested device.  Returns ``ActorCritic`` for MLP policies and
    ``GNNActorCritic`` for Stage 2 GNN policies — dispatch is by the
    ``policy_type`` key in the saved args ("mlp" or "gnn"; defaults to
    "mlp" for pre-Stage-2 checkpoints).
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    action_dim = 2
    policy_type = args.get("policy_type", "mlp")

    if policy_type == "gnn":
        # GNN checkpoint — reconstruct from graph dims saved in args.
        policy = GNNActorCritic(
            n_blue        = int(args["n_blue"]),
            n_red         = int(args["n_red"]),
            blue_feat_dim = 8,
            red_feat_dim  = 1,
            edge_feat_dim = 7,
            action_dim    = action_dim,
            d_hidden      = int(args.get("d_hidden", 64)),
            n_msg_rounds  = int(args.get("n_msg_rounds", 2)),
            init_log_std  = STAGE1_DEFAULTS["init_log_std"],
        ).to(device)
    else:
        obs_dim = _infer_obs_dim(args)
        policy = ActorCritic(
            obs_dim      = obs_dim,
            action_dim   = action_dim,
            hidden       = STAGE1_DEFAULTS["hidden"],
            init_log_std = STAGE1_DEFAULTS["init_log_std"],
        ).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    return policy


class TrainedBlueAgent(HeuristicBlueAgent):
    """
    Adapter that lets a trained MLP or GNN policy be used wherever a
    ``HeuristicBlueAgent`` is expected (renderer, eval table, etc.).

    - MLP policies (``ActorCritic``) accept the flat per-agent obs
      that the eval script already builds.
    - GNN policies (``GNNActorCritic``) require the graph obs, so we
      pull it directly from ``env.structured_observation()`` and read
      out the acting agent's action from the (N_blue, 2) batched
      output.

    Detected automatically by isinstance.  Per-agent calls run a tiny
    single-sample forward pass — fine for evaluation but not for
    vectorised rollouts (the trainer uses batched forwards directly).
    """

    def __init__(
        self,
        policy,
        device:        torch.device,
        deterministic: bool = True,
    ) -> None:
        self.policy        = policy
        self.device        = device
        self.deterministic = deterministic
        self._is_gnn       = isinstance(policy, GNNActorCritic)

    def act(self, obs: np.ndarray, env: PursuitEnv, agent: str) -> np.ndarray:
        if self._is_gnn:
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
        # MLP path (unchanged).
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.deterministic:
                action_t = self.policy.act_deterministic(obs_t)
            else:
                action_t, _, _, _ = self.policy.get_action_and_value(obs_t)
        a = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        return np.clip(a, -1.0, 1.0)
