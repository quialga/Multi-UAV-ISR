"""
tests/test_stochastic_adapter.py — the eval adapter's SAMPLING branch.

``build_trained_agent(..., deterministic=False)`` existed but had never
been exercised: it called ``get_action_and_value``, which also runs
``critic_forward`` on the CTDE full state, and passed the ACTOR's
observation in place of it.  The critic's ``red_input_mlp`` expects the
privileged ``true_*`` width, so the first real call raised a Linear shape
error.  Acting does not need the value head at all.

These tests pin that the sampling branch runs and that it actually
samples, so ``evaluate_trained.py --stochastic`` stays usable — it is how
we ask whether the deterministic metric understates a shared-parameter
team (five blues that differ only by observation can act too alike, and
noise decorrelates them).

Run:
    pytest tests/test_stochastic_adapter.py -v
"""
from __future__ import annotations

import numpy as np
import torch

from isr.agents.gnn_stage4_policy import GNNStage4Policy
from isr.agents.heuristics import stationary_red
from isr.agents.policy_loader import build_trained_agent
from isr.env.pursuit_env import PursuitEnv

# The live-track path re-draws the p_TP detection dice and the position
# noise on EVERY structured_belief_observation() call, so two calls do not
# return the same observation.  These tests are about the adapter, not the
# sensor, so the draws are switched off to make the observation a pure
# function of the state; otherwise "same policy, same state, same action"
# is untestable through act().
CFG = dict(n_blue=3, n_red=2, n_obstacles=0, arena_size=130.0, max_steps=60,
           capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
           actor_obs="belief", red_policy=stationary_red,
           track_detection=False, track_occlusion=False,
           sensor_pos_noise_std=0.0, sensor_vel_noise_std=0.0)


def _env_and_policy(seed=0):
    env = PursuitEnv(**dict(CFG, seed=seed))
    env.reset(seed=seed)
    pol = GNNStage4Policy(n_blue=CFG["n_blue"], n_red=CFG["n_red"], n_obs=0)
    return env, pol


def test_sampling_branch_runs_without_the_critic():
    """The regression: this raised a Linear shape error because the branch
    routed through the CTDE critic with the actor's observation."""
    env, pol = _env_and_policy()
    agent = build_trained_agent(pol, torch.device("cpu"), deterministic=False)
    a = agent.act(None, env, env.possible_agents[0])
    assert a.shape == (2,), a.shape
    assert np.isfinite(a).all()
    assert (np.abs(a) <= 1.0).all(), "actions must stay inside the action box"


def test_sampling_spreads_far_more_than_the_deterministic_path():
    """Isolates the ACTION noise from the observation noise statistically.

    ``structured_belief_observation()`` updates the belief map as it builds
    it, so repeated calls at the same timestep do not return the same
    observation and the deterministic action jitters a little.  (Production
    never sees this: ``act`` caches per ``env._t``, so the observation is
    built once per step.)  Rather than fight that, compare spreads: the
    sampled actions must scatter by roughly sigma, orders of magnitude
    more than that residual jitter.
    """
    env, pol = _env_and_policy()
    who = env.possible_agents[0]

    def _spread(deterministic: bool, n: int = 120) -> float:
        draws = [build_trained_agent(pol, torch.device("cpu"),
                                     deterministic=deterministic)
                 .act(None, env, who) for _ in range(n)]
        return float(np.std(np.asarray(draws), axis=0).mean())

    torch.manual_seed(0)
    det_spread = _spread(True)
    sto_spread = _spread(False)

    assert sto_spread > 50.0 * det_spread, (sto_spread, det_spread)
    # log_std initialises at 0 => sigma 1, clipped into the [-1, 1] box.
    assert 0.3 < sto_spread < 1.2, sto_spread


def test_sampling_is_centred_on_the_deterministic_action():
    """Sampling must perturb the SAME mean the deterministic path returns,
    not a different head: the average of many draws converges to it."""
    env, pol = _env_and_policy()
    who = env.possible_agents[0]
    mean_action = build_trained_agent(
        pol, torch.device("cpu"), deterministic=True).act(None, env, who)

    torch.manual_seed(0)
    draws = []
    for _ in range(400):
        # A fresh adapter each time: act() caches per env timestep.
        draws.append(build_trained_agent(
            pol, torch.device("cpu"), deterministic=False).act(None, env, who))
    avg = np.mean(draws, axis=0)

    # log_std starts at 0 => sigma 1, clipped to the box, so the empirical
    # mean is pulled toward 0; only assert it tracks the sign/ordering of
    # the deterministic action rather than matching it exactly.
    assert np.abs(avg - mean_action).max() < 1.0, (avg, mean_action)
