# Stage 1 — Results

**Acceptance verdict: FAIL**
(trained policy mean return vs `run_from_nearest_uav`: **-4.43**;
bar = 1.20 × GreedyPursuer = **+20.42**; margin = **-24.84**)

Generated: 2026-07-01 18:59:00
Checkpoint: `C:\Users\quial\sources\Multi-UAV-ISR\runs\stage1\smoke_gnn\final.pt`
Training: rollout 5, global_step 20480
Eval episodes per cell: 3 (deterministic, fixed seeds shared across blue policies)

## Evaluation table

Rows: blue policy.  Columns: red policy.
Cell shows ``mean_return ± std_return  (mean caught of 2)`` — averaged
over 3 episodes with matched seeds.

| Blue \\ Red | Stationary | Random | RunFromNearest |
|---|---|---|---|
| **Random** |  -13.95 ± 7.07  (0.67 caught) |  -18.95 ± 7.07  (0.33 caught) |  -18.95 ± 7.07  (0.33 caught) |
| **Greedy** |  +18.48 ± 0.60  (2.00 caught) |  +18.19 ± 0.93  (2.00 caught) |  +17.01 ± 1.17  (2.00 caught) |
| **Trained** |   -7.41 ± 18.27  (0.67 caught) |  -10.39 ± 7.12  (0.67 caught) |   -4.43 ± 13.46  (1.00 caught) |

## Visualisations

- [Trained vs Stationary](runs/stage1/smoke_gnn/eval_gifs/trained_vs_stationary.gif)
- [Trained vs Random](runs/stage1/smoke_gnn/eval_gifs/trained_vs_random.gif)
- [Trained vs RunFromNearest](runs/stage1/smoke_gnn/eval_gifs/trained_vs_runfromnearest.gif)
- [Greedy vs RunFromNearest (reference)](runs/stage1/smoke_gnn/eval_gifs/greedy_vs_runfromnearest.gif)

## Training arguments

```json
{
  "n_blue": 3,
  "n_red": 2,
  "arena_size": 100.0,
  "max_steps": 200,
  "capture_radius": 3.0,
  "n_envs": 16,
  "rollout_steps": 256,
  "n_rollouts": 5,
  "n_epochs": 10,
  "mb_size": 512,
  "lr": 0.0003,
  "clip_eps": 0.2,
  "ent_coef": 0.01,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "log_interval": 1,
  "eval_interval": 50,
  "eval_episodes": 20,
  "save_interval": 100,
  "no_eval": true,
  "red_policy_mix": "stationary:1,random:1,run:1",
  "policy_type": "gnn",
  "d_hidden": 64,
  "n_msg_rounds": 2,
  "seed": 0,
  "device": "cpu",
  "run_name": "smoke_gnn",
  "obs_dim": 38
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
