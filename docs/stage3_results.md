# Stage 1 — Results

**Acceptance verdict: FAIL**
(trained policy mean return vs `run_from_nearest_uav`: **+13.93**;
bar = 1.20 × GreedyPursuer = **+28.74**; margin = **-14.81**)

Generated: 2026-07-08 23:14:54
Checkpoint: `C:\Users\quial\sources\Multi-UAV-ISR\runs\stage3\stage3_kl005-lr1e-4\final.pt`
Training: rollout 1000, global_step 4096000
Eval episodes per cell: 50 (deterministic, fixed seeds shared across blue policies)

## Evaluation table

Rows: blue policy.  Columns: red policy.
Cell shows ``mean_return ± std_return  (mean caught of 3 — mean episode length in steps)`` — averaged
over 50 episodes with matched seeds.

**Note on ``mean episode length``:** under the step-cost-dominated
reward (``-0.05`` per step), two policies with different coordination
quality can have very similar mean_return but noticeably different
episode lengths.  Fewer steps to catch all reds = more efficient
pursuit / better coordination.  Compare across the Trained row to see
whether a structured architecture is *behaviourally* more coordinated
even when reward moves only a little.

| Blue \\ Red | Stationary | Random | RunFromNearest |
|---|---|---|---|
| **Random** |  -17.53 ± 12.87  (0.94 caught — 198.4 steps) |  -13.71 ± 10.97  (1.20 caught — 199.3 steps) |  -29.66 ± 5.20  (0.14 caught — 200.0 steps) |
| **Greedy** |  +26.66 ± 1.25  (3.00 caught — 33.4 steps) |  +26.32 ± 1.44  (3.00 caught — 36.8 steps) |  +23.74 ± 2.37  (3.00 caught — 62.6 steps) |
| **Trained** |  +20.02 ± 7.51  (2.96 caught — 79.7 steps) |  +19.30 ± 6.27  (2.98 caught — 87.5 steps) |  +13.93 ± 10.63  (2.86 caught — 115.9 steps) |

## Visualisations

- [Trained vs Stationary](runs/stage3/stage3_kl005-lr1e-4/eval_gifs/trained_vs_stationary.gif)
- [Trained vs Random](runs/stage3/stage3_kl005-lr1e-4/eval_gifs/trained_vs_random.gif)
- [Trained vs RunFromNearest](runs/stage3/stage3_kl005-lr1e-4/eval_gifs/trained_vs_runfromnearest.gif)
- [Greedy vs RunFromNearest (reference)](runs/stage3/stage3_kl005-lr1e-4/eval_gifs/greedy_vs_runfromnearest.gif)

## Training arguments

```json
{
  "n_blue": 5,
  "n_red": 3,
  "arena_size": 130.0,
  "max_steps": 200,
  "capture_radius": 3.0,
  "sensor_radius": 40.0,
  "n_envs": 16,
  "rollout_steps": 256,
  "n_rollouts": 1000,
  "n_epochs": 10,
  "mb_size": 512,
  "lr": 0.0001,
  "clip_eps": 0.2,
  "ent_coef": 0.0075,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "target_kl": 0.05,
  "log_interval": 1,
  "save_interval": 100,
  "no_eval": true,
  "red_policy_mix": "stationary:1,random:1,run:1",
  "d_hidden": 64,
  "n_msg_rounds": 2,
  "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",
  "seed": 0,
  "device": "cpu",
  "run_name": "stage3_kl005-lr1e-4",
  "policy_type": "gnn_ctde"
}
```

## Interpretation notes

- The acceptance criterion (1.2× GreedyPursuer against the
  ``run_from_nearest_uav`` red) tests whether the learned policy does
  something a per-UAV nearest-target rule cannot — typically:
  splitting coverage across reds, anticipating where the red will
  dodge, or arriving from the direction red is fleeing.
- Reward components: +10 per catch, -0.01·||a||² (action cost), -0.05
  step cost, -5 terminal penalty per uncaught red.  A trained policy
  beating Greedy can do so by **catching the same number of reds
  faster** (less step cost), **catching with less effort** (lower
  action cost), or **catching more reds in time-up episodes** (less
  terminal penalty).
- Eval is deterministic (distribution mean, no exploration noise).
  Returns will be lower-variance than during training.
