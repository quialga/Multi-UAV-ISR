# Stage 1 — Results

_Pending v2 training + evaluation on GPU._

This file is auto-generated and overwritten by
[`scripts/evaluate_trained.py`](../scripts/evaluate_trained.py) once a
trained checkpoint exists.  The committed version is a placeholder.

## v1 result (documented for reference)

See [`docs/stage1_analysis.md`](stage1_analysis.md) for the full
analysis of the v1 result and the motivation for the v2 redesign
(ego-centric world-frame observation + red-policy mixing during
training, motivated by TERL, arXiv:2503.12395).

## Reproduce v2

```bash
# Train v2 (≈45 min on a small GPU at 1000 rollouts)
python scripts/train_stage1.py --device cuda --n-rollouts 1000

# Evaluate v2 + render GIFs + overwrite this file
python scripts/evaluate_trained.py \
    --checkpoint runs/stage1/<timestamp>/best.pt \
    --device cuda \
    --n-episodes 50
```

## Spec + acceptance criterion

- [`docs/design.md §3`](design.md) — full Stage 1 spec.
- Heuristic baseline (GreedyPursuer vs `run_from_nearest_uav`): +16.15.
- Acceptance bar: **≥ +19.38** trained mean episode return.
