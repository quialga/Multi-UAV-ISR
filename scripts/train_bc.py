"""
scripts/train_bc.py — behaviour cloning / DAgger from ``ObsGreedy``.

Why this exists.  `docs/stage4_results.md §8` measured that
``ObservationGreedyPursuer`` extracts 2.91/3 from the belief observation
with no learning at all, while PPO from scratch plateaus at 1.2/3 on the
same input (§6.5).  The observation is not the bottleneck; the policy is.
An expert that consumes **the policy's own observation space** is exactly
the precondition for cloning: it can label any state the student visits,
with no privileged information to strip out.

What it does.  Rolls the vectorised env, labels every state with the
expert, and regresses the actor's mean onto the expert action.  Then
hands PPO a checkpoint ``train_stage4.py --warm-start-full`` loads
directly.

Three design points that are not free choices:

* **The expert must label the STORED observation.**
  ``structured_belief_observation()`` re-draws the sensor on every call,
  so asking the expert to look at the env again would pair the stored
  observation with an action taken on a different realisation of it —
  labelling noise that the student cannot fit and would learn to average
  over.  ``ObservationGreedyPursuer.act_from_observation`` exists for
  this.

* **Hidden states are the student's own, stored per transition.**
  ``ppo_update_stage4`` does not back-propagate through time: it replays
  each transition from the hidden state recorded when it was collected
  (`isr/train/ppo.py`, ``hidden = batch["hidden"]``).  Cloning does the
  same, so the actor is trained under exactly the conditions PPO will
  fine-tune it under.  The student is forwarded at every step even when
  the expert is driving, purely to advance that hidden state.

* **The critic is fitted here too** (``--no-critic`` opts out).  A warm
  actor handed to PPO with a COLD critic gets destroyed in the first
  rollouts by large, wrong advantages — §6.5 records the neighbouring
  failure, where an aux loss against an ill-matched critic walked a
  1.29 policy down to 1.01.  Fitting V on the same rollouts (MC returns
  via the buffer's own GAE) means ``--warm-start-full`` restores a
  matched pair.

DAgger.  ``--beta`` is the probability the EXPERT drives; it decays
linearly to ``--beta-final`` across rounds.  beta=1 throughout is plain
behaviour cloning on expert trajectories; decaying it is DAgger, which
costs nothing extra here because the expert labels every state whoever
is driving.  The default decays, because the failure mode of plain BC is
precisely the states the expert never visits.

``log_std`` is deliberately untouched: the MSE has no gradient into it,
so the policy reaches PPO with its initial exploration width intact.

Run (defaults reproduce the §9 belief-mode expert, 2.91/3):
    python scripts/train_bc.py --n-rounds 40 --run-name bc_belief_v1

Then fine-tune:
    python scripts/train_stage4.py --warm-start-full \
        runs/stage4/bc_belief_v1/final.pt  <the §6.5 recipe> ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train_stage4 as train_stage4                       # noqa: E402
from isr.agents.gnn_stage4_policy import (                        # noqa: E402
    GNNStage4Policy, split_stage4_obs,
)
from isr.agents.heuristics import (                               # noqa: E402
    ObservationGreedyPursuer, run_from_nearest_uav, stationary_red,
    random_red,
)
from isr.agents.policy_loader import env_kwargs_from_checkpoint   # noqa: E402
from isr.configs.stage4_default import STAGE4_DEFAULTS            # noqa: E402
from isr.train.graph_buffer import Stage4RolloutBuffer            # noqa: E402
from isr.train.vec_env import Stage4VectorPursuitEnv              # noqa: E402
from scripts.train_stage4 import (                                # noqa: E402
    _to_device, evaluate_policy_deterministic,
)

# §7 / §10 geometry, TRACKER mode with the coverage path on.
#
# Not belief, although its expert scores a hair higher (2.913 vs 2.900):
# the belief path carries five privileged information leaks
# (docs/tracker_observation.md), and a policy cloned on it would learn to
# depend on knowledge no sensor reported.  Since §10 gave both arms a
# search tier they match on captures anyway, and the tracker gets there
# 14-31% faster.  Equal captures, better efficiency, less information.
DEFAULT_ENV_ARGS = (
    "--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 "
    "--belief-grid-size 26 --max-steps 200 --actor-obs tracker "
    "--use-staleness --staleness-regions 5"
)


def expert_actions(
    experts: List[ObservationGreedyPursuer],
    obs_np:  Dict[str, np.ndarray],
    n_blue:  int,
) -> np.ndarray:
    """Label a BATCH of observations.  Returns (E, n_blue, 2).

    Reads the stored arrays, never the envs — see the module docstring on
    why re-deriving from the env would label a different sensor draw.
    """
    # Every key the expert reads has to be sliced through, INCLUDING the
    # coverage path.  Omitting region_feats / gb_edge_feats silently
    # disables ObsGreedy's search tier, so the labels come from an expert
    # that holds position with nothing trackable -- 58.4% of agent-steps
    # on the tracker observation (§10.2).  Cloning that teaches the policy
    # to freeze, and reads afterwards as "cloning did not transfer".
    keys = [k for k in ("rb_edge_features", "rb_edge_visible",
                        "region_feats", "gb_edge_feats", "gb_weight")
            if k in obs_np]
    E = obs_np["rb_edge_features"].shape[0]
    out = np.zeros((E, n_blue, 2), dtype=np.float32)
    for i in range(E):
        row = {k: obs_np[k][i] for k in keys}
        for b in range(n_blue):
            out[i, b] = experts[i].act_from_observation(row, b, n_blue)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-args", default=DEFAULT_ENV_ARGS,
                   help="trainer flags describing the env to clone in")
    p.add_argument("--n-rounds", type=int, default=40,
                   help="collect/fit rounds (a round = one buffer fill)")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--rollout-steps", type=int, default=200)
    p.add_argument("--n-epochs", type=int, default=4,
                   help="passes over each round's data")
    p.add_argument("--mb-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=1.0,
                   help="P(expert drives) at round 0")
    p.add_argument("--beta-final", type=float, default=0.0,
                   help="P(expert drives) at the last round; the decay "
                        "between the two is what makes this DAgger rather "
                        "than plain behaviour cloning")
    p.add_argument("--no-critic", action="store_true",
                   help="skip the value-function fit.  Leaves PPO a cold "
                        "critic under a warm actor — see the module "
                        "docstring for why that is a bad trade")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--eval-interval", type=int, default=5,
                   help="rounds between deterministic evals (0 = never)")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Exactly the args dict a training run with these flags would save, so
    # the checkpoint is loadable by load_policy / --warm-start-full and
    # carries the same reward shape.
    saved_args = train_stage4.saved_args_from_flags(args.env_args)
    # The BC hyperparameters travel with it too, so a run is reproducible
    # from the file alone.
    saved_args["bc"] = dict(vars(args))
    env_kwargs = env_kwargs_from_checkpoint(saved_args)

    run_name = args.run_name or f"bc_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir  = Path("runs/stage4") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"BC from ObsGreedy -> {run_dir}")
    log(f"env_kwargs={env_kwargs}")

    # Same parsing the trainer does inline (scripts/train_stage4.py:478).
    red_mix = [(n.strip(), float(w))
               for n, w in (c.split(":") for c in
                            saved_args["red_policy_mix"].split(",") if c.strip())]
    log(f"red_policy_mix: {red_mix}")
    vec_env = Stage4VectorPursuitEnv(
        n_envs              = args.n_envs,
        env_kwargs          = env_kwargs,
        base_seed           = args.seed,
        episode_buffer_size = 256,
        red_policy_mix      = red_mix,
    )
    n_agents   = vec_env.n_agents
    action_dim = vec_env.action_dim

    policy = GNNStage4Policy(
        n_blue             = vec_env.n_blue,
        n_red              = vec_env.n_red,
        n_obs              = vec_env.n_obstacles,
        blue_feat_dim      = vec_env.blue_feat_dim,
        red_feat_dim       = 4,
        obs_feat_dim       = 5,
        actor_n_red        = vec_env.actor_n_red,
        actor_n_obs        = vec_env.actor_n_obstacles,
        actor_red_feat_dim = vec_env.actor_red_feat_dim,
        actor_obs_feat_dim = vec_env.actor_obs_feat_dim,
        edge_feat_dim      = vec_env.edge_feat_dim,
        action_dim         = action_dim,
        d_hidden           = saved_args["d_hidden"],
        n_msg_rounds       = saved_args["n_msg_rounds"],
        init_log_std       = STAGE4_DEFAULTS.get("init_log_std", 0.0),
        use_hidden_in_gnn  = saved_args["share_hidden_via_gnn"],
        # The student must SEE what the expert acts on.  Without this the
        # expert searches on region nodes the actor never receives, and the
        # clone is being asked to fit a function of inputs it does not have.
        n_region           = (saved_args["staleness_regions"] ** 2
                              if saved_args["use_staleness"] else 0),
    ).to(device)
    log(f"Policy: {sum(p_.numel() for p_ in policy.parameters())} params")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    buffer = Stage4RolloutBuffer(
        rollout_steps = args.rollout_steps,
        n_envs        = args.n_envs,
        n_agents      = n_agents,
        action_dim    = action_dim,
        d_hidden      = saved_args["d_hidden"],
        device        = device,
    )
    experts = [ObservationGreedyPursuer() for _ in range(args.n_envs)]

    obs_np = vec_env.reset(seed=args.seed)
    hidden = policy.initial_hidden(n_envs=args.n_envs, device=device)
    global_step = 0
    best_caught = float("-inf")
    t_start = time.time()

    for rnd in range(args.n_rounds):
        beta = (args.beta if args.n_rounds == 1 else
                args.beta + (args.beta_final - args.beta)
                * (rnd / (args.n_rounds - 1)))
        buffer.reset()
        expert_frac_sum = 0.0

        # ---- Collect -----------------------------------------------------
        for _step in range(args.rollout_steps):
            obs_t = {k: _to_device(v, device) for k, v in obs_np.items()}
            partial_obs, full_state = split_stage4_obs(obs_t)

            a_expert = expert_actions(experts, obs_np, vec_env.n_blue)
            a_expert_t = torch.from_numpy(a_expert).to(device)

            with torch.no_grad():
                # The student is forwarded every step even when the expert
                # drives: its hidden state has to keep advancing on the
                # states actually visited, because that is the state the
                # update replays each transition from.
                mean, _log_std, new_hidden, _ = policy.actor_forward(
                    partial_obs, hidden)
                value, _ = policy.critic_forward(full_state)

            # DAgger: who DRIVES is a coin flip per env; the label is the
            # expert's either way.
            drive_expert = rng.random(args.n_envs) < beta
            expert_frac_sum += float(drive_expert.mean())
            a_student = mean.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
            action_np = np.where(drive_expert[:, None, None],
                                 a_expert, a_student).astype(np.float32)

            next_obs_np, reward_np, done_np, _ = vec_env.step(action_np)

            buffer.add(
                obs       = obs_t,
                actions   = a_expert_t,      # the TARGET, not what was played
                log_probs = torch.zeros((args.n_envs, n_agents), device=device),
                values    = value,
                rewards   = torch.from_numpy(reward_np).to(device),
                dones     = torch.from_numpy(done_np).to(device),
                hidden    = hidden,
            )

            done_t = torch.from_numpy(done_np).to(device).view(-1, 1, 1)
            hidden = new_hidden * (1.0 - done_t)
            for i in np.nonzero(done_np)[0]:
                experts[int(i)].reset()

            obs_np = next_obs_np
            global_step += args.n_envs

        with torch.no_grad():
            last_obs_t = {k: _to_device(v, device) for k, v in obs_np.items()}
            _, last_full = split_stage4_obs(last_obs_t)
            last_value, _ = policy.critic_forward(last_full)
        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        # ---- Fit ---------------------------------------------------------
        bc_sum = v_sum = 0.0
        n_mb = 0
        for _epoch in range(args.n_epochs):
            for batch in buffer.iter_minibatches(args.mb_size):
                b_partial, b_full = split_stage4_obs(batch)
                target  = batch["actions"]
                b_hidden = batch["hidden"]

                mean, _log_std, _h, _ = policy.actor_forward(b_partial, b_hidden)
                bc_loss = nn.functional.mse_loss(mean, target)
                loss = bc_loss

                if not args.no_critic:
                    v_pred, _ = policy.critic_forward(b_full)
                    v_loss = 0.5 * (v_pred - batch["returns"]).pow(2).mean()
                    loss = loss + args.vf_coef * v_loss
                    v_sum += float(v_loss.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),
                                         args.max_grad_norm)
                optimizer.step()
                bc_sum += float(bc_loss.item())
                n_mb += 1

        ep = vec_env.recent_episode_stats()
        n_mb = max(n_mb, 1)
        log(f"[round {rnd+1:>4d}/{args.n_rounds}]  steps={global_step:>9d}  "
            f"beta={beta:.2f}  expert_drove={expert_frac_sum/args.rollout_steps:.2f}  "
            f"bc={bc_sum/n_mb:.4f}  vf={v_sum/n_mb:.3f}  "
            f"epR={ep['mean_return']:+7.2f}  "
            f"caught={ep['mean_caught']:.2f}/{vec_env.n_red}  "
            f"({(time.time()-t_start)/60:.1f} min)")

        # ---- Eval --------------------------------------------------------
        if args.eval_interval > 0 and (rnd + 1) % args.eval_interval == 0:
            evs = {}
            for name, red in (("stat", stationary_red),
                              ("rand", random_red(seed=0)),
                              ("run",  run_from_nearest_uav)):
                evs[name] = evaluate_policy_deterministic(
                    policy, env_kwargs, red, args.eval_episodes, device,
                )["mean_caught"]
            mean_caught = float(np.mean(list(evs.values())))
            log(f"    [det eval] stat={evs['stat']:.2f} rand={evs['rand']:.2f} "
                f"run={evs['run']:.2f}  mean={mean_caught:.2f}/{vec_env.n_red}")
            if mean_caught > best_caught:
                best_caught = mean_caught
                torch.save({"policy_state": policy.state_dict(),
                            "rollout": rnd + 1, "global_step": global_step,
                            "args": saved_args,
                            "best_metric": {"name": "det_caught",
                                            "value": mean_caught}},
                           run_dir / "best.pt")
                log(f"    saved best.pt (det_caught={mean_caught:.2f})")

    torch.save({"policy_state": policy.state_dict(),
                "rollout": args.n_rounds, "global_step": global_step,
                "args": saved_args}, run_dir / "final.pt")
    log(f"\nBC done in {(time.time()-t_start)/60:.1f} min.  "
        f"Checkpoints under {run_dir}")
    vec_env.close()


if __name__ == "__main__":
    main()
