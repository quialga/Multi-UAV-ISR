"""
isr/agents/policy_loader.py — Load a trained Stage 4 checkpoint + adapt it
to the HeuristicBlueAgent interface so it can be dropped into the
rendering / evaluation scripts that already accept heuristic policies.

Three entry points:

- ``load_policy(path, device)``: load a checkpoint written by
  ``scripts/train_stage4.py`` and return a constructed
  ``GNNStage4Policy`` on the requested device.

- ``env_kwargs_from_checkpoint(train_args)``: rebuild the env config the
  checkpoint was trained under, so an evaluation reproduces that regime.

- ``build_trained_agent(policy, device, deterministic=True)``: wrap it as
  a blue agent with ``act(obs, env, agent) -> action``, matching
  ``HeuristicBlueAgent``.  ``deterministic=True`` (default) takes the
  distribution mean — no exploration noise.

The Stage 1/2 (``GNNActorCritic``) and pre-Stage-2 MLP read paths were
removed in the 2026-08 cleanup along with their trainer; their results are
preserved in docs/stage1_results.md, docs/stage1_analysis.md and
docs/stage2_results.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from isr.agents.heuristics import HeuristicBlueAgent
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from isr.configs.stage4_default import STAGE4_DEFAULTS
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

    Only ``policy_type == 'gnn_stage4_v6'`` is supported.  Stage 1/2
    (``'gnn'``), Stage 3 (``'gnn_ctde'``) and pre-Stage-2 MLP checkpoints
    raise a clear error — their trainers no longer exist.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    policy_type = args.get("policy_type", "mlp")

    if policy_type == "gnn_stage4_v6":
        policy = GNNStage4Policy(
            n_blue            = int(args["n_blue"]),
            n_red             = int(args["n_red"]),
            n_obs             = int(args.get("n_obstacles", 0)),
            action_dim        = 2,
            d_hidden          = int(args.get("d_hidden", 64)),
            n_msg_rounds      = int(args.get("n_msg_rounds", 2)),
            init_log_std      = STAGE4_DEFAULTS["init_log_std"],
            use_hidden_in_gnn = bool(args.get("share_hidden_via_gnn", True)),
        ).to(device)
    else:
        raise RuntimeError(
            f"Checkpoint at {path} has policy_type={policy_type!r}; only "
            f"'gnn_stage4_v6' is supported.  The Stage 1/2 ('gnn'), Stage 3 "
            f"('gnn_ctde') and MLP read paths were removed with their "
            f"trainers in the 2026-08 cleanup."
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
    if isinstance(policy, GNNStage4Policy):
        return TrainedStage4BlueAgent(policy, device, deterministic)
    raise TypeError(f"Unsupported policy class: {type(policy).__name__}")


def env_kwargs_from_checkpoint(train_args: dict) -> dict:
    """
    Rebuild the env config a checkpoint was trained under, so a renderer or
    evaluator reproduces the same regime.

    Stage 1/2 checkpoints carry only the basic keys; Stage 4 adds belief
    maps, obstacles and the whole sensor model.  Every Stage 4 key uses a
    ``.get`` default that reproduces PRE-feature behaviour, so a checkpoint
    written before a knob existed evaluates as it did when it was trained.
    """
    g = train_args.get
    kw = dict(
        n_blue         = train_args["n_blue"],
        n_red          = train_args["n_red"],
        arena_size     = train_args["arena_size"],
        max_steps      = train_args["max_steps"],
        capture_radius = train_args["capture_radius"],
    )
    if g("sensor_radius") is not None:
        kw["sensor_radius"] = g("sensor_radius")

    if g("policy_type") != "gnn_stage4_v6":
        return kw

    kw.update(
        n_obstacles              = g("n_obstacles", 0),
        obstacle_radius_min      = g("obstacle_radius_min", 5.0),
        obstacle_radius_max      = g("obstacle_radius_max", 15.0),
        obstacle_spawn_clearance = g("obstacle_spawn_clearance", 10.0),
        moving_obstacle_fraction = g("moving_obstacle_fraction", 0.0),
        obstacle_speed           = g("obstacle_speed", 0.0),
        obstacle_belief_decay    = g("obstacle_belief_decay", 1.0),
        use_belief_maps          = True,
        belief_grid_size         = g("belief_grid_size", 26),
        belief_channels          = g("belief_channels", 2),
        belief_clip              = g("belief_clip", 10.0),
        p_TP                     = g("p_tp", 0.85),
        p_FP                     = g("p_fp", 0.15),
        p_TP_obstacle            = g("p_TP_obstacle", None),
        p_FP_obstacle            = g("p_FP_obstacle", None),
        ray_step_size            = g("ray_step_size", 2.5),
        enemy_belief_decay       = g("enemy_belief_decay", 0.99),
        enemy_belief_diffusion   = g("enemy_belief_diffusion", 0.2),
        sensor_pos_noise_std     = g("sensor_pos_noise_std", 1.0),
        # Pre-fix defaults: an older checkpoint evaluates in the sensor
        # regime it trained under, not today's.
        track_occlusion          = g("track_occlusion", False),
        track_detection          = g("track_detection", False),
        sensor_vel_noise_std     = g("sensor_vel_noise_std", 0.0),
        track_conf_min           = g("track_conf_min", 1.0),
        sensor_noise_range_growth= g("sensor_noise_range_growth", 0.0),
        vel_prior_std            = g("vel_prior_std", 1.0),
        crash_obstacle_penalty   = g("crash_obstacle_penalty", 0.0),
        crash_blue_penalty       = g("crash_blue_penalty", 0.0),
        blue_collision_radius    = g("blue_collision_radius", 2.0),
        clearance_weight         = g("clearance_weight", 0.0),
        clearance_margin         = g("clearance_margin", 4.0),
        clearance_ally_weight    = g("clearance_ally_weight", 0.0),
        clearance_ally_margin    = g("clearance_ally_margin", 3.0),
        catch_reward             = g("catch_reward", 10.0),
        step_cost                = g("step_cost", 0.05),
        uncaught_penalty         = g("uncaught_penalty", 5.0),
        action_cost_coef         = g("action_cost_coef", 0.01),
    )
    return kw


class TrainedStage4BlueAgent(HeuristicBlueAgent):
    """
    Adapter for a trained ``GNNStage4Policy``, matching the same
    ``act(obs, env, agent)`` interface the renderer already uses.

    Two differences from the Stage 1/2 adapter make this non-trivial:

    * the policy is RECURRENT (a GRU per blue), so the hidden state must
      advance exactly ONCE per env step -- not once per agent query;
    * it computes ALL blues in a single forward pass.

    So the forward is cached per ``env._t``: the first agent queried on a
    step runs it, the rest read their slice out.  A step index that goes
    backwards means a new episode, which resets the hidden state.
    """

    def __init__(
        self,
        policy:        GNNStage4Policy,
        device:        torch.device,
        deterministic: bool = True,
    ) -> None:
        self.policy        = policy
        self.device        = device
        self.deterministic = deterministic
        self.reset()

    def reset(self) -> None:
        """Drop the recurrent state (call between episodes)."""
        self._hidden    = None
        self._cached_t  = None
        self._actions   = None

    def act(self, obs: np.ndarray, env: PursuitEnv, agent: str) -> np.ndarray:
        t = int(env._t)
        if self._actions is None or t != self._cached_t:
            # New episode (or first call) -> fresh recurrent state.
            if self._hidden is None or (self._cached_t is not None
                                        and t < self._cached_t):
                self._hidden = self.policy.initial_hidden(1, self.device)
            s = env.structured_belief_observation()
            obs_t = {
                k: torch.from_numpy(np.asarray(v)).float()
                        .unsqueeze(0).to(self.device)
                for k, v in s.items() if isinstance(v, np.ndarray)
            }
            partial_obs, _ = split_stage4_obs(obs_t)
            with torch.no_grad():
                if self.deterministic:
                    mean, self._hidden = self.policy.act_deterministic(
                        partial_obs, self._hidden)
                else:
                    # (action, log_prob, entropy, value, new_hidden,
                    #  actor_h_blue, critic_h_blue) -- the CTDE critic path
                    # is unused here, so the actor obs stands in for it.
                    out = self.policy.get_action_and_value(
                        partial_obs, partial_obs, self._hidden)
                    mean, self._hidden = out[0], out[4]
            self._actions  = mean.squeeze(0).cpu().numpy().astype(np.float32)
            self._cached_t = t
        idx = env.possible_agents.index(agent)
        return np.clip(self._actions[idx], -1.0, 1.0)
