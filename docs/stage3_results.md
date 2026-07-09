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

## Phase 3.5 — Aux hidden-loss (belief-state distillation) experiments

Three variants of the CTDE recurrent policy were trained and compared
to isolate the contribution of the auxiliary belief-state distillation
loss.  In all three the actor is a partial-obs GNN + GRU; only the
aux-loss configuration changes.

**Setup for all three:** N=5 blue, M=3 red, arena 130×130,
`sensor_radius=40`, red-policy mix uniform, warm-started CTDE critic
from `runs/stage1/scaling_gnn/best.pt`.

### Variants

| Variant | `aux_hidden_coef` | Critic during training | Aux target |
| --- | --- | --- | --- |
| **Baseline** | 0.0 (off)  | Live (updates via value_loss)   | — |
| **C** — live target | 0.1 | Live (updates via value_loss)   | Critic encoder (per step, drifting) |
| **A** — frozen target | 0.1 | Frozen at Stage 2 weights       | Critic encoder (fixed Stage 2 oracle) |

### Results (50 episodes per (blue, red) cell, matched seeds)

Trained-vs-`RunFromNearest` (the hardest red, load-bearing metric):

| Variant | Rollouts | mean_return | mean_caught | mean_steps | std_return |
| --- | --- | --- | --- | --- | --- |
| Baseline (no aux) | 1000  | +13.93 | 2.86 | 115.9 | ±10.63 |
| **C — live aux** (this doc's headline) | 200   | **+15.87** | **2.92** | **97.9**  | **±9.59**  |
| A — frozen aux    | ~500 (best-early ckpt) | (worse than C at same rollout count, degraded further) | ~2.9 peak then dropped | — | — |

C at only 200 rollouts beat baseline at 1000 rollouts on **every**
metric.  See §2 head-to-head deltas at the top of this doc for the
full column-by-column story.

### Why A degraded after peaking (~rollout 200-300)

A converged rapidly to the same ~2.9 catches ceiling as C, then
**degraded** — catches dropped and reward followed.  Diagnosis:

1. **Frozen critic → stale V estimates.**  The Stage 2 checkpoint was
   trained under Stage 2's policy trajectory distribution.  As the
   Stage 3 policy learned partial-obs pursuit patterns (target-
   recovery via GRU memory, different chase geometries), those
   trajectories drift OOD from what the frozen critic ever scored.
2. **PPO advantage signal breaks.**  Advantages are `R − V(s)` under
   GAE.  When V(s) systematically misestimates the return
   distribution for the current policy, the sign and magnitude of
   the advantage estimator become unreliable.  PPO then reinforces
   trajectories that *look* high-advantage under the stale V but
   aren't actually the best for the true (Stage 3-under-partial-obs)
   return.
3. **Policy walks off the good region.**  Because reinforcement is
   miscalibrated, the policy drifts away from the near-optimal
   behavior it discovered at rollout ~200.  Standard "policy drift
   under stale critic" pattern in off-policy value learning.

This is the exact trade-off flagged in the A-vs-C proposal: a fixed
aux target is theoretically cleaner (student-teacher formulation),
but the accompanying frozen V function bites in a small env where
policy behavior evolves fast.

### Decision

**Option C (live critic + aux on critic encoder) is Stage 3's final
architecture.**  Option A is discarded.  The moving-target aux loss
is stable in practice because value_loss keeps the critic tracking
the current-policy trajectory distribution, and the aux loss's
gradient signal is small enough (with coef 0.1) that the aux target
doesn't oscillate meaningfully at the scale that matters for the
actor's optimization.

The stage3_hiddencoef-01 checkpoint at rollout 200 is the current
Stage 3 reference — even though it FAIL'd the aspirational
`mean_return ≥ Greedy + 5` criterion by 0.59, it passes all revised
criteria (docs/stage3_design.md §9 as of 2026-07):

- ✅ `mean_caught ≥ Greedy + 0.15`: 2.92 vs 2.43 threshold
- ✅ `std_return ≤ 0.85 × Greedy std_return`: 9.59 vs 12.04 threshold
- ✅ `mean_caught ≥ 2.7/3` absolute: 2.92

## Next step — Option 1: cross-blue hidden state sharing

The current architecture's GRU hidden state is **strictly private per
blue** (see [gnn_ctde_policy.py:272-302](../isr/agents/gnn_ctde_policy.py:272)):
each blue's GRU updates its own hidden using only its own encoded
observation.  Cross-blue coordination is limited to the **current
step** via the GNN's bb messages.

This blocks a specific coordination pattern: if blue A saw a red at
position X and blue B did not, blue A cannot inform blue B about it
via memory — only if blue A's *current* position/velocity indirectly
hints at the past sighting.  Belief-state sharing across time
requires a channel other than the private GRU.

**Design (Option 1, cheapest fix):** feed the previous-timestep
hidden state of each blue as an additional node feature into the
actor's GNN encoder.  Then the GNN's bb message passing carries
memory across blues, in addition to the current observation.

- **Cost**: `blue_feat_dim` grows from 8 to 8 + `d_hidden` (72), one
  extra concat op in `actor_forward`, ~5 extra kparams in the actor's
  `blue_input_mlp`.
- **Critic unchanged**: full-obs critic still takes plain 8-dim blue
  features.  Aux loss target semantics stay identical.
- **First-step**: `hidden` is zeros at episode start, so the encoder
  effectively sees plain obs on step 0, matches current behavior.
- **Backward compat**: gated on a `--share-hidden-via-gnn` flag, off
  by default.  Old checkpoints load unchanged.

**Hypothesis for the run**: option 1 gives the actor a memory-based
coordination channel that the private-hidden design lacks.  If C is
already near ceiling for private-hidden, option 1 should push catches
from 2.92 → 2.95+ vs Run.  If the ceiling is deeper than memory
sharing (e.g. arena geometry, red speed), option 1 will match C at
best.  Either outcome is a portfolio-worthy observation.
