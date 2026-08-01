"""
tests/test_bridge_parity.py — the Gazebo bridge's decision code is
BIT-IDENTICAL to the training rollout (milestone 4, exact code parity).

This is the airtight half of the milestone-4 fidelity check.  It proves
that ``gazebo.brain.ClosedLoopBrain`` — the exact code the live ROS node
runs — reproduces the training-time deterministic rollout step for step,
ISOLATING any Gazebo difference to the one seam Gazebo actually owns:
continuous physics vs the env's teleport, and control latency (measured
separately, in the live sim).

HOW IT CAN BE EXACT
-------------------
Two rollouts on the same seed and the same policy:

  * REFERENCE — the training eval loop: PursuitEnv steps itself
    (obs -> act_deterministic -> env.step), exactly as
    scripts/train_stage4.py::evaluate_policy_deterministic does.
  * BRIDGE — ClosedLoopBrain driven WITHOUT Gazebo: fed the reference's
    own per-step positions, so the only thing under test is the brain's
    orchestration (referee/belief/observe/act ordering) and its RNG
    consumption, not motion (motion is covered bit-for-bit by the fuzz
    test in tests/test_gazebo_kinematics.py).

The two match to the last bit because the brain's tick interleaves its
random draws — sensor-position noise in the observation, Bernoulli
draws in the belief update — into the SAME global sequence as
env.step + the eval loop (the ``_ticked_once`` first-tick special-case
is exactly what keeps reset's pre-observation belief update aligned).
Feeding identical positions guarantees identical draw *shapes* (same
visible reds/obstacles, same in-sensor cells), so identically-seeded
RNG streams produce identical values -> identical observations ->
identical actions.  If any of that ordering ever breaks, the actions
diverge and this test fails loudly.

A randomly-initialised policy is used deliberately: parity is about
code equivalence, not policy quality, so no checkpoint is needed and
the test is fast.
"""
from __future__ import annotations

import numpy as np
import torch

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from gazebo.brain import ClosedLoopBrain

CPU = torch.device("cpu")


def _small_env(seed: int) -> PursuitEnv:
    """A small but fully-featured Stage 4 env (belief + occlusion +
    noise + obstacles + walls) so parity exercises every RNG path."""
    kw = dict(
        n_blue=3, n_red=2, arena_size=60.0, max_steps=25,
        capture_radius=3.0, sensor_radius=25.0, n_obstacles=2,
        obstacle_radius_min=4.0, obstacle_radius_max=8.0,
        obstacle_spawn_clearance=6.0, use_belief_maps=True,
        belief_grid_size=12, belief_channels=2, belief_clip=10.0,
        p_TP=0.85, p_FP=0.15, ray_step_size=2.5,
        enemy_belief_decay=0.99, enemy_belief_diffusion=0.2,
        sensor_pos_noise_std=1.0, blue_collision_radius=2.0,
    )
    return PursuitEnv(**kw, red_policy=run_from_nearest_uav, seed=seed)


def _make_policy() -> GNNStage4Policy:
    torch.manual_seed(0)
    pol = GNNStage4Policy(n_blue=3, n_red=2, n_obs=2,
                          d_hidden=32, n_msg_rounds=2)
    pol.eval()
    return pol


def _reference_rollout(env: PursuitEnv, policy, seed: int):
    """The training deterministic-eval loop.  Records, per step, the
    pre-action positions (what the observation is built from) and the
    action, plus the total number of reds caught."""
    env.reset(seed=seed)
    hidden = policy.initial_hidden(1, CPU)
    agents = env.possible_agents
    actions, states = [], []
    while env.agents:
        states.append((env._blue_pos.copy(), env._red_pos.copy()))
        obs = env.structured_belief_observation()
        obs_t = {k: torch.from_numpy(v).float().unsqueeze(0)
                 for k, v in obs.items()}
        partial, _ = split_stage4_obs(obs_t)
        with torch.no_grad():
            mean, hidden = policy.act_deterministic(partial, hidden)
        a = np.clip(mean.squeeze(0).numpy().astype(np.float32), -1.0, 1.0)
        actions.append(a)
        env.step({agents[i]: a[i] for i in range(len(agents))})
    caught = env.n_red - int(env._red_active.sum())
    return actions, states, caught


def _brain_rollout(env: PursuitEnv, policy, seed: int, ref_states):
    """Drive ClosedLoopBrain with the reference's positions (no Gazebo)."""
    env.reset(seed=seed)
    brain = ClosedLoopBrain(env, policy, device=CPU)
    actions = []
    for blue_pos, red_pos in ref_states:
        res = brain.tick(blue_pos, red_pos)
        if res.blue_accel is not None:
            actions.append(res.blue_accel)
        if res.done:
            break
    return actions, brain


def test_brain_matches_training_rollout_bit_exact():
    for seed in (0, 1, 7, 13):
        policy = _make_policy()
        ref_actions, ref_states, ref_caught = _reference_rollout(
            _small_env(seed), policy, seed)
        brain_actions, brain = _brain_rollout(
            _small_env(seed), policy, seed, ref_states)

        # Episode length parity.
        assert len(brain_actions) == len(ref_actions), (
            f"seed {seed}: brain {len(brain_actions)} steps vs "
            f"reference {len(ref_actions)}")

        # Bit-exact action parity every step.
        for k, (ba, ra) in enumerate(zip(brain_actions, ref_actions)):
            np.testing.assert_array_equal(
                ba, ra, err_msg=f"seed {seed}: action mismatch at step {k}")

        # Same reds caught (transitively implied by action parity, but
        # asserted directly as an independent check on the referee).
        assert brain.n_caught == ref_caught, (
            f"seed {seed}: brain caught {brain.n_caught} vs "
            f"reference {ref_caught}")


def test_parity_is_a_real_test_not_vacuous():
    """Guard against a silently-empty rollout: the reference must
    actually take several steps and exercise the pipeline."""
    policy = _make_policy()
    ref_actions, ref_states, _ = _reference_rollout(
        _small_env(0), policy, 0)
    assert len(ref_actions) >= 5
    assert len(ref_states) == len(ref_actions)
