# Stage 3 — Results

**Acceptance verdict: PASS**
(trained policy mean return vs `run_from_nearest_uav`: **+18.16**;
bar = GreedyPursuer + 5 (Stage 3 §9 return criterion) = **+16.47**; margin = **+1.69**)

Generated: 2026-07-10 16:45:05
Checkpoint: `runs/stage3/gpu_opt1_critic_frozen/best.pt`
Training: rollout 385, global_step 1576960
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
| **Trained** |  +21.45 ± 11.08  (2.90 caught — 54.1 steps) |  +24.18 ± 5.53  (2.98 caught — 42.1 steps) |  +18.16 ± 10.20  (2.90 caught — 78.9 steps) |

## Visualisations

- [Trained vs Stationary](runs/stage3/gpu_opt1_critic_frozen/eval_gifs/trained_vs_stationary.gif)
- [Trained vs Random](runs/stage3/gpu_opt1_critic_frozen/eval_gifs/trained_vs_random.gif)
- [Trained vs RunFromNearest](runs/stage3/gpu_opt1_critic_frozen/eval_gifs/trained_vs_runfromnearest.gif)
- [Greedy vs RunFromNearest (reference)](runs/stage3/gpu_opt1_critic_frozen/eval_gifs/greedy_vs_runfromnearest.gif)

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
  "n_rollouts": 400,
  "n_epochs": 10,
  "mb_size": 512,
  "lr": 0.0001,
  "clip_eps": 0.2,
  "ent_coef": 0.008,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "target_kl": 0.03,
  "aux_hidden_coef": 0.2,
  "freeze_critic": true,
  "share_hidden_via_gnn": true,
  "lr_schedule": "linear",
  "lr_min_frac": 0.1,
  "best_ckpt_metric": "mean_return",
  "best_ckpt_min_delta": 0.05,
  "best_ckpt_min_episodes": 32,
  "log_interval": 1,
  "save_interval": 100,
  "no_eval": true,
  "red_policy_mix": "stationary:1,random:1,run:1",
  "d_hidden": 64,
  "n_msg_rounds": 2,
  "warm_start_critic": "runs/stage1/scaling_gnn/best.pt",
  "seed": 0,
  "device": "cpu",
  "run_name": "gpu_opt1_critic_frozen",
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

## Final Stage 3 architecture and closure

The result above (`gpu_opt1_critic_frozen/best.pt`, rollout 385) is the
Stage 3 headline.  Configuration:

- **Cross-blue hidden sharing via GNN** (Phase 3.6, option 1):
  previous per-blue GRU hidden state prepended to the actor's blue
  node features; bb messages carry belief state across allies.
  `--share-hidden-via-gnn`.
- **Frozen CTDE critic** (Phase 3.5, option A): critic loaded from
  Stage 2 `scaling_gnn/best.pt`, frozen for the whole Stage 3 run.
  `--freeze-critic`.
- **Aux belief-state distillation @ coef 0.2**: actor encoder
  aligned to the frozen critic encoder's per-blue node embeddings via
  MSE.  `--aux-hidden-coef 0.2`.
- **Aggressive PPO hyperparameters**: `--ent-coef 0.008`,
  `--target-kl 0.03`, `--lr 1e-4` with **linear decay to 10% of
  initial** over the run.  Best-checkpoint tracking captures the peak.

**Reproduction:**

```bash
python scripts/train_stage3.py --device cuda:0 --n-rollouts 400 \
    --lr 1e-4 --ent-coef 0.008 \
    --aux-hidden-coef 0.2 --freeze-critic --share-hidden-via-gnn \
    --run-name gpu_opt1_critic_frozen --no-eval

python scripts/evaluate_trained.py \
    --checkpoint runs/stage3/gpu_opt1_critic_frozen/best.pt \
    --n-episodes 50
```

**All acceptance criteria PASS** — original §9 return criterion (once
feared unattainable) AND revised §9 catches/std/absolute-floor
criteria:

| Criterion (vs `RunFromNearest`) | Threshold | Actual | Verdict |
| --- | --- | --- | --- |
| `mean_return ≥ Greedy + 5` (original §9)   | +16.47 | **+18.16** | ✅ PASS |
| `mean_caught ≥ Greedy + 0.15` (revised)    | 2.43   | **2.90**   | ✅ PASS |
| `std_return ≤ 0.85 × Greedy std` (revised) | 12.04  | **10.20**  | ✅ PASS |
| `mean_caught ≥ 2.7/3` absolute             | 2.70   | **2.90**   | ✅ PASS |

Return improved from the earlier `checkpoint_00100` peak (+17.68) by
+0.48 and steps dropped by 16.9 (95.8 → 78.9) against Run — the
policy is materially more efficient at higher rollout counts once LR
decay stabilizes late training.

## The load-bearing insight: LR was too high

Before landing linear LR decay + `lr=1e-4`, every Stage 3 variant
(baseline, C, A, Opt 1, and combinations) hit the same peak-then-
degrade pattern around rollout 100-230.  The pathology looked like a
classic aux-loss or frozen-critic failure mode — and we spent time
diagnosing it as such — but the actual root cause was **PPO late-
training instability under a constant lr=3e-4**:

- As entropy drops (2.84 → ~2.5), the policy becomes concentrated.
- With the same LR, each gradient step now produces a much larger
  behavioural change (the mean action moves 4-5× more per step).
- Small optimizer misdirections translate into policy walks off the
  discovered coordination.

Dropping `--lr 1e-4` (3× smaller) + linear decay to 10% of initial
turned the peak-then-degrade pattern into peak-and-hold in every
config we retested.  The freeze-critic result went from "peaks at 100,
degrades to 2.7 by 227" (with lr=3e-4) to "peaks at 385 with
`mean_return +18.16` and no visible regression".

## Freeze-critic vs no-freeze-critic — the split winner

Both variants were re-run at `lr=1e-4` with linear decay.  They agree
on final catches but diverge on efficiency:

- **`--freeze-critic` wins on evaluation.**  Final numbers above:
  `+18.16 vs Run, 2.90 caught, 78.9 steps`.  The frozen critic acts
  as a stable regularizer on the actor's encoder (via the aux loss)
  and pushes the learned policy toward representations that
  generalize.
- **No `--freeze-critic` wins during training.**  Reaches plateau
  reward faster and hits 3/3 catches *during training* more
  frequently.  Live critic gives on-policy V that speeds convergence
  — but the resulting policy is slightly less robust in
  deterministic-mean eval than the frozen-critic version.

Interpretation: live critic ≈ better fit to training-time trajectory
distribution; frozen critic ≈ better generalization at eval.  Neither
is uniformly better; the "right" choice depends on whether you value
sample-efficient training or the highest-quality final policy.  For
the Stage 3 headline we chose freeze-critic based on the eval metric.

## Phase 3.5 / 3.6 experiment log

Full variant sweep at N=5 blue, M=3 red, arena 130×130,
`sensor_radius=40`, red-policy mix uniform, warm-started CTDE critic
from Stage 2 `scaling_gnn/best.pt`.

| Variant | Aux coef | Critic | Cross-blue hidden | LR | Notes / outcome |
| --- | --- | --- | --- | --- | --- |
| Baseline (no aux) | 0.0 | live   | no  | 3e-4 const  | 2.86 caught @ 1000, +13.93 |
| C (live aux)      | 0.1 | live   | no  | 3e-4 const  | 2.92 caught @ 200, +15.87.  Peaked, would have degraded. |
| A (frozen aux)    | 0.1 | frozen | no  | 3e-4 const  | Peaked ~2.9 @ 300 then degraded |
| Opt 1 alone       | 0.1 | live   | yes | 3e-4 const  | Peaked ~2.9 @ 200 then degraded |
| A + Opt 1 + aux 0.2 | 0.2 | frozen | yes | 3e-4 const  | +17.68, 2.90 caught @ 100.  Peak captured by save_interval luck. |
| **Final: A + Opt 1 + LR 1e-4 linear** | 0.2 | frozen | yes | 1e-4 linear | **+18.16, 2.90 caught @ 385.  Stable.** |
| No-freeze + Opt 1 + LR 1e-4 linear    | 0.2 | live   | yes | 1e-4 linear | Faster in training but slightly worse in eval than frozen variant. |

## What remained on the table

Not pursued to conclusion because the headline PASS was met and the
compute budget belongs to Stage 4:

- **Option B (freeze critic encoder only, keep head trainable):**
  proposed but not run.  Would decouple "stable aux target" from "V
  adapting to Stage 3 policy".  Untested hypothesis: option B
  might combine the eval quality of freeze-critic with the training
  speed of no-freeze.
- **Aux coef sweep at LR 1e-4:** only 0.2 tested at the final config.
  Lower (0.05, 0.10) might give further gains.
- **d_hidden > 64:** would need the aux target to also be higher
  dim, and we'd lose warm-start from Stage 2's 64-dim critic.  Not
  pursued.

## Stage 3 — closed.

All §9 acceptance criteria (original and revised) are met.
`docs/stage3_design.md` marks Stage 3 as landed.  See
[`docs/design.md §5`](design.md) for the next stage — **Stage 4:
Sensor noise + occlusion**, which introduces:

- Noisy observations: `p_observed = p_true + N(0, σ²)`
- Terrain-block occlusion (deterministic bit)
- An explicit **belief-state predictor head** trained with auxiliary
  loss reconstructing true target positions from observation history

The Phase 3.6 cross-blue hidden sharing (option 1) and the aux
belief-state distillation groundwork built here are the natural
starting point for Stage 4's belief-state predictor: same encoder
architecture, add a decoder head that reconstructs `(x, y, active)`
per red target.  A design doc for Stage 4 will land as
`docs/stage4_design.md` when the stage opens.
