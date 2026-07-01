# Stage 2 — Results

_Pending scaling experiment._

This file will be populated once a matched MLP-vs-GNN head-to-head has
been run at a larger team size (N=5 vs M=3 or N=6 vs M=4 — see
[`stage2_gnn_design.md §8b`](stage2_gnn_design.md) for the exact
commands and the hypothesis this experiment tests).

## Why we're not publishing the N=3 vs M=2 numbers here

An initial 700-rollout GNN training run at N=3 vs M=2 produced small
reward improvements over the Stage 1 v2 MLP baseline:

| vs Red | v2 MLP | Stage 2 GNN | Δ |
|---|---|---|---|
| Stationary | +17.17 | +17.41 | +0.24 |
| Random | +17.27 | +17.29 | +0.02 |
| RunFromNearest | +15.37 | +15.92 | +0.55 |

The much more striking finding was **qualitative**: side-by-side GIFs
show the GNN adopts an emergent **target-assignment** coordination
strategy (each blue picks a different red and pursues it directly),
while the MLP has all blues converge on the same red.  The
`mean_steps` metric confirmed the flat scale limitation: at N=3 vs
M=2, target assignment splits 3 UAVs across 2 targets → at least one
UAV is redundant → episode length is not shorter than parameter-shared
"pile on the same red".

The Stage 2 verdict is therefore held pending the scaling experiment.
If GNN's `mean_steps` beats MLP's by 15-25% at N=5-6, we have a clean
quantitative confirmation of coordination-via-attention.  If not, we
falsify the hypothesis and reframe.

## Reproduce (after scaling run lands)

```bash
python scripts/evaluate_trained.py \
    --checkpoint runs/stage1/<scaling-mlp>/best.pt \
    --device cuda --n-episodes 50
python scripts/evaluate_trained.py \
    --checkpoint runs/stage1/<scaling-gnn>/best.pt \
    --device cuda --n-episodes 50
```

Both eval calls auto-write to this file (the second overwrites the
first — plan to keep the GNN eval as the final content and paste the
MLP numbers into the head-to-head table by hand, or use
`--results-md` explicitly).
