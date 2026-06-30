# Stage 1 — Results

_Pending: full training + evaluation._

This file is auto-generated and overwritten by
[`scripts/evaluate_trained.py`](../scripts/evaluate_trained.py) once a
trained checkpoint exists.  The committed version is the empty
template; run the evaluation locally / on GPU to populate it.

## Reproduce

```bash
# Train (≈75 min on a small GPU at 1000 rollouts, ≈2.5h on CPU)
python scripts/train_stage1.py --device cuda --n-rollouts 1000

# Evaluate + render GIFs + write this file
python scripts/evaluate_trained.py \
    --checkpoint runs/stage1/<timestamp>/best.pt \
    --device cuda \
    --n-episodes 50
```

## Spec + acceptance criterion

- [`docs/design.md §3`](design.md) — full Stage 1 spec (arena,
  kinematics, observation, action, reward).
- [`docs/design.md §3.10`](design.md) — acceptance criterion: trained
  blue must achieve ≥ 1.20× the mean episode return of `GreedyPursuer`
  against the `run_from_nearest_uav` red.

Heuristic calibration baseline (from the commit history, seeds 0/1/2,
20 episodes each):

| Blue | Red | mean reward | mean caught (of 2) |
|---|---|---|---|
| Random | RunFromNearest | -12.00 | 0.67 |
| **Greedy** | **RunFromNearest** | **+16.21** | **2.00** |

Acceptance bar: **≥ +19.45** trained mean return vs `run_from_nearest_uav`.
