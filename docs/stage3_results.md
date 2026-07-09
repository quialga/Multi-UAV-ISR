# Stage 3 — Results

**Acceptance verdict: FAIL**
(trained policy mean return vs `run_from_nearest_uav`: **+15.87**;
bar = GreedyPursuer + 5 (Stage 3 §9 return criterion) = **+16.47**; margin = **-0.59**)

Generated: 2026-07-09 14:59:09
Checkpoint: `C:\Users\quial\sources\Multi-UAV-ISR\runs\stage3\stage3_hiddencoef-01\checkpoint_00200.pt`
Training: rollout 200, global_step 819200
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
| **Greedy** |  +18.84 ± 13.22  (2.60 caught — 93.7 steps) |  +22.60 ± 9.54  (2.80 caught — 76.2 steps) |  +11.47 ± 14.16  (2.28 caught — 136.4 steps) |
| **Trained** |  +18.87 ± 8.85  (2.94 caught — 81.3 steps) |  +20.98 ± 6.68  (2.98 caught — 67.9 steps) |  +15.87 ± 9.59  (2.92 caught — 97.9 steps) |

## Visualisations

- [Trained vs Stationary](runs/stage3/stage3_hiddencoef-01/eval_gifs/trained_vs_stationary.gif)
- [Trained vs Random](runs/stage3/stage3_hiddencoef-01/eval_gifs/trained_vs_random.gif)
- [Trained vs RunFromNearest](runs/stage3/stage3_hiddencoef-01/eval_gifs/trained_vs_runfromnearest.gif)
- [Greedy vs RunFromNearest (reference)](runs/stage3/stage3_hiddencoef-01/eval_gifs/greedy_vs_runfromnearest.gif)

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
  "n_rollouts": 1500,
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
  "aux_hidden_coef": 0.1,
  "log_interval": 1,
  "save_interval": 100,
  "no_eval": true,
  "red_policy_mix": "stationary:1,random:1,run:1",
  "d_hidden": 64,
  "n_msg_rounds": 2,
  "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",
  "seed": 0,
  "device": "cpu",
  "run_name": "stage3_hiddencoef-01",
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
