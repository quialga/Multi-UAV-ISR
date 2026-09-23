"""
tests/test_bc_from_obsgreedy.py — the behaviour-cloning path.

Cloning only works if three joins are right, and each has already bitten
this code once in a form that produced plausible-looking numbers rather
than a crash:

* **The refactor.**  ``act_from_observation`` is the rule ``act`` used to
  inline.  If they diverge, the expert that LABELS is not the expert that
  scored 2.91/3 in docs/stage4_results.md §9.
* **The batch indexing.**  Labelling reads ``(E, n_rb, 7)`` and must place
  each blue's action at the right ``(env, blue)``.  The equivalent slip —
  indexing red-slot edges as if they were true reds — is what
  ``test_observation_greedy.py`` was written for.
* **The handover to PPO.**  A BC checkpoint that ``load_full_stage4``
  only partially copies would warm-start a few tensors, leave the rest
  random, and look like "cloning didn't help" rather than like a bug.

Plus the one that is just arithmetic: the actor must actually move toward
the expert when the loss is stepped.

Run:
    pytest tests/test_bc_from_obsgreedy.py -v
"""
from __future__ import annotations

import numpy as np
import torch

from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from isr.agents.heuristics import ObservationGreedyPursuer, stationary_red
from isr.agents.policy_loader import env_kwargs_from_checkpoint, load_policy
from isr.env.pursuit_env import PursuitEnv
from isr.train.vec_env import Stage4VectorPursuitEnv
from scripts.train_bc import expert_actions
from scripts.train_stage4 import saved_args_from_flags

FLAGS = ("--arena-size 130 --n-blue 3 --n-red 2 --n-obstacles 0 "
         "--belief-grid-size 26 --max-steps 40 --actor-obs belief")


def _env_kwargs():
    return env_kwargs_from_checkpoint(saved_args_from_flags(FLAGS))


def _vec(n_envs=2, seed=0):
    v = Stage4VectorPursuitEnv(
        n_envs=n_envs, env_kwargs=_env_kwargs(), base_seed=seed,
        episode_buffer_size=16, red_policy_mix=[("stationary", 1.0)])
    return v, v.reset(seed=seed)


def _policy(vec_env):
    return GNNStage4Policy(
        n_blue=vec_env.n_blue, n_red=vec_env.n_red,
        n_obs=vec_env.n_obstacles, blue_feat_dim=vec_env.blue_feat_dim,
        red_feat_dim=4, obs_feat_dim=5,
        actor_n_red=vec_env.actor_n_red, actor_n_obs=vec_env.actor_n_obstacles,
        actor_red_feat_dim=vec_env.actor_red_feat_dim,
        actor_obs_feat_dim=vec_env.actor_obs_feat_dim,
        edge_feat_dim=vec_env.edge_feat_dim, action_dim=vec_env.action_dim)


# --------------------------------------------------------------------------
#  The expert survives being split apart
# --------------------------------------------------------------------------

def test_act_from_observation_matches_act():
    """``act`` must be ``act_from_observation`` on the cached observation
    and nothing else — otherwise the labeller and the scored baseline are
    two different policies."""
    env = PursuitEnv(**_env_kwargs(), red_policy=stationary_red, seed=3)
    env.reset(seed=3)
    for _ in range(5):
        env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})

    agent = ObservationGreedyPursuer()
    via_act = [agent.act(None, env, a) for a in env.possible_agents]
    # Same cached observation, called directly.
    direct = [agent.act_from_observation(agent._obs, i, env.n_blue)
              for i in range(env.n_blue)]
    for a, b in zip(via_act, direct):
        assert np.array_equal(a, b), (a, b)


def test_batch_labelling_matches_per_env_labelling():
    """The (E, n_blue) placement.  Row i of the batch must be labelled from
    env i's slice, and blue b's action must land at [i, b]."""
    vec_env, obs_np = _vec(n_envs=3, seed=11)
    for _ in range(4):
        obs_np, _, _, _ = vec_env.step(
            np.zeros((3, vec_env.n_blue, 2), dtype=np.float32))

    experts = [ObservationGreedyPursuer() for _ in range(3)]
    batch = expert_actions(experts, obs_np, vec_env.n_blue)
    assert batch.shape == (3, vec_env.n_blue, 2), batch.shape

    for i in range(3):
        row = {"rb_edge_features": obs_np["rb_edge_features"][i],
               "rb_edge_visible":  obs_np["rb_edge_visible"][i]}
        for b in range(vec_env.n_blue):
            want = ObservationGreedyPursuer().act_from_observation(
                row, b, vec_env.n_blue)
            assert np.array_equal(batch[i, b], want), (i, b, batch[i, b], want)
    vec_env.close()


def test_labels_are_unit_or_zero():
    """Every label is a heading or a hold — nothing in between, so an MSE
    against them has a well-defined scale."""
    vec_env, obs_np = _vec(n_envs=4, seed=5)
    for _ in range(6):
        obs_np, _, _, _ = vec_env.step(
            np.zeros((4, vec_env.n_blue, 2), dtype=np.float32))
    experts = [ObservationGreedyPursuer() for _ in range(4)]
    a = expert_actions(experts, obs_np, vec_env.n_blue).reshape(-1, 2)
    norms = np.linalg.norm(a, axis=1)
    assert np.all((np.abs(norms - 1.0) < 1e-5) | (norms < 1e-8)), norms
    vec_env.close()


# --------------------------------------------------------------------------
#  The fit
# --------------------------------------------------------------------------

def test_cloning_step_moves_the_actor_toward_the_expert():
    """Gradients reach the actor and the MSE falls.  Cheap, but it is the
    difference between 'cloning does not help here' and 'the optimiser was
    never wired to the actor'."""
    vec_env, obs_np = _vec(n_envs=4, seed=1)
    for _ in range(3):
        obs_np, _, _, _ = vec_env.step(
            np.zeros((4, vec_env.n_blue, 2), dtype=np.float32))
    policy = _policy(vec_env)

    obs_t = {k: torch.from_numpy(v).float() for k, v in obs_np.items()}
    partial_obs, _ = split_stage4_obs(obs_t)
    hidden = policy.initial_hidden(4, torch.device("cpu"))
    experts = [ObservationGreedyPursuer() for _ in range(4)]
    target = torch.from_numpy(
        expert_actions(experts, obs_np, vec_env.n_blue)).float()

    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        mean, _, _, _ = policy.actor_forward(partial_obs, hidden)
        loss = torch.nn.functional.mse_loss(mean, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
    vec_env.close()


def test_log_std_is_untouched_by_the_clone_loss():
    """PPO needs exploration width on arrival.  An MSE on the mean must not
    collapse sigma — which an NLL objective on a deterministic expert
    would."""
    vec_env, obs_np = _vec(n_envs=2, seed=2)
    policy = _policy(vec_env)
    before = policy.actor_log_std.detach().clone()

    obs_t = {k: torch.from_numpy(v).float() for k, v in obs_np.items()}
    partial_obs, _ = split_stage4_obs(obs_t)
    hidden = policy.initial_hidden(2, torch.device("cpu"))
    experts = [ObservationGreedyPursuer() for _ in range(2)]
    target = torch.from_numpy(
        expert_actions(experts, obs_np, vec_env.n_blue)).float()

    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(5):
        mean, _, _, _ = policy.actor_forward(partial_obs, hidden)
        loss = torch.nn.functional.mse_loss(mean, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert torch.equal(policy.actor_log_std, before), (
        policy.actor_log_std, before)
    vec_env.close()


# --------------------------------------------------------------------------
#  The handover to PPO
# --------------------------------------------------------------------------

def test_bc_checkpoint_round_trips_into_a_full_warm_start(tmp_path):
    """A BC checkpoint must be what ``--warm-start-full`` expects: readable
    by ``load_policy`` and copied ENTIRELY by ``load_full_stage4``.  A
    partial copy would leave most of the policy random and read as
    'cloning did not transfer'."""
    vec_env, _ = _vec(n_envs=2, seed=0)
    policy = _policy(vec_env)
    ckpt = tmp_path / "final.pt"
    torch.save({"policy_state": policy.state_dict(), "rollout": 1,
                "global_step": 10,
                "args": saved_args_from_flags(FLAGS)}, ckpt)

    reloaded = load_policy(ckpt, torch.device("cpu"))
    for k, v in policy.state_dict().items():
        assert torch.equal(reloaded.state_dict()[k], v), k

    fresh = _policy(vec_env)
    n_copied = fresh.load_full_stage4(ckpt)
    assert n_copied == len(policy.state_dict()), (
        n_copied, len(policy.state_dict()))
    for k, v in policy.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v), k
    vec_env.close()


def test_saved_args_pin_the_configs_reward_not_the_legacy_default():
    """``env_kwargs_from_checkpoint`` defaults ``step_cost`` to the
    PRE-feature 0.05 for old checkpoints.  Building an env config from the
    parser alone therefore silently evaluates under a reward no run trains
    with; ``saved_args_from_flags`` is what stops that."""
    kw = _env_kwargs()
    assert kw["step_cost"] == 0.033, kw["step_cost"]
    assert kw["catch_reward"] == 10.0
    assert kw["uncaught_penalty"] == 5.0
