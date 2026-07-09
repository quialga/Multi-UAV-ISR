# Stage 3 — GPU run instructions

Prerequisite for reproducing the Stage 3 result claimed in
`docs/stage3_design.md §9`.  Assumes the Stage 2 GNN checkpoint at
`runs/stage1/scaling_gnn/best.pt` is present (used as the CTDE
critic's warm start).

## 1. What the run does

Trains the Stage 3 CTDE recurrent GNN policy from
`isr/agents/gnn_ctde_policy.py`:

- **Actor**: partial-obs GNN (`sensor_radius = 40`) → shared
  `nn.GRUCell` per blue → shared MLP → action mean.
- **Critic**: full-state GNN (byte-identical to Stage 2) → sum-blue +
  concat-red → MLP → V(s).  Warm-started from Stage 2's
  `scaling_gnn/best.pt` — the encoder + critic head copy directly
  because the scale matches (5 blue vs 3 red, 130×130).
- Env is `RecurrentVectorPursuitEnv` (partial obs + visibility
  masks), 16 parallel envs, `red_policy_mix = stationary:1,random:1,run:1`.

Trained on ~5 M transitions (1500 rollouts × 256 steps × 16 envs).

## 2. The exact command

```bash
python scripts/train_stage3.py \
    --device cuda:0 \
    --n-rollouts 1500 \
    --run-name gpu_1500 \
    --no-eval
```

Every other flag defaults come from `isr/configs/stage3_default.py`:

| Flag              | Value                             |
| ----------------- | --------------------------------- |
| `--n-blue`        | 5                                 |
| `--n-red`         | 3                                 |
| `--arena-size`    | 130                               |
| `--sensor-radius` | 40                                |
| `--n-envs`        | 16                                |
| `--rollout-steps` | 256                               |
| `--n-epochs`      | 10                                |
| `--mb-size`       | 512                               |
| `--lr`            | 3e-4                              |
| `--ent-coef`      | 0.018                             |
| `--target-kl`     | 0.03                              |
| `--d-hidden`      | 64                                |
| `--n-msg-rounds`  | 2                                 |
| `--warm-start-critic` | `runs/stage1/scaling_gnn/best.pt` |

Pass `--warm-start-critic none` to skip warm-start (useful for the
ablation quoted in design §5).

Pass `--target-kl 0` to disable the per-epoch KL early stop.

## 3. What to expect

CPU wall time is ~1.7 min per 5 rollouts (see the smoke run at the
end of Stage 3 step 7), so extrapolated CPU cost is ~8.5 h.  On a
consumer GPU expect **~30 min end-to-end** (Stage 2's `scaling_gnn`
took ~25 min at the same rollout count).

At init:

- `params ≈ 140,357`.
- Warm-start message: `copied 30 tensors` — this is the full critic
  path (encoder × 6 MLPs = 24 tensors + critic_trunk (4) +
  critic_head (2)).  If you see a smaller number, `d_hidden`,
  `n_red`, or the checkpoint scale probably don't match.

During training:

- Watch `ppo/approx_kl`: if it's consistently pegged at 0.03 with
  `n_epochs_run < 10`, the early stop is firing too aggressively —
  consider `--target-kl 0.05` or `--target-kl 0`.
- Watch `ppo/entropy`: a collapse to below ~1.5 is the same failure
  mode Stage 2's MLP hit at scale.  If it happens, raise
  `--ent-coef` to 0.03 as a first response.
- Watch `rollout/mean_caught`: baseline is 0.6/3 with a random
  policy (the smoke number).  Learning should push this well past
  1.5 by rollout ~500 and toward Greedy's ceiling (~2.7/3 depending
  on step penalty).

## 4. Acceptance (from design §9, revised 2026-07)

Stage 3 vs `run_from_nearest_uav` red at 50 episodes:

- `mean_caught` ≥ sensor-aware Greedy baseline **+0.15**, **and**
- `std_return` ≤ **0.85 ×** sensor-aware Greedy `std_return`
  (coordination-quality gate), **and**
- `mean_caught` ≥ **2.7 / 3** absolute (task actually solved).

Compare against the sensor-aware `GreedyPursuer` (commit ab947ee,
respects `env.sensor_radius`) evaluated in the same partial-obs env.
The Stage 2 GNN is NOT the H1 baseline for Stage 3 — it was trained
under full obs and is OOD on the sensor masks.

See `docs/stage3_design.md §9` for the full revised criterion and
`docs/stage3_results.md` for the criterion's history and rationale.

## 5. Deliverables to produce after the run

1. `runs/stage3/gpu_1500/final.pt` — the checkpoint.
2. `runs/stage3/gpu_1500/train_log.txt` — copy relevant lines into
   a new `docs/stage3_results.md` for the writeup.
3. Eval GIFs (Stage 3 vs Run / Stationary / Random) — reuse the
   Stage 2 eval script, swapping the policy import.
4. Update `docs/stage3_design.md §9` with the PASS/FAIL verdict.

## 6. Known limitations

- The warm-start assumes the Stage 2 checkpoint used the same
  `(n_blue, n_red, d_hidden)`.  Warm-start with mismatched scale
  silently degrades (only shape-matching tensors are copied); this
  is a known trade-off of the byte-safe copy.
- Hidden state is reset to zeros on every env-done via
  `hidden = new_hidden * (1 - done)`.  This is the standard
  stateless-at-episode-boundary convention (see
  `docs/stage3_design.md §7`).
