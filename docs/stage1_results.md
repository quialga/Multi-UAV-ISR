# Stage 1 — Results

**Verdict: SOFT PASS** — see [`stage1_analysis.md §6b`](stage1_analysis.md)
for the strict-vs-soft reasoning.

- **Strict acceptance** (`≥ 1.20 × GreedyPursuer` vs
  `run_from_nearest_uav` = ≥ +19.38): trained **+15.37**, misses by 4.01.
- **Qualitative acceptance** (matches Greedy across the red
  distribution, no OOD collapse, 2.00/2 caught on every red type):
  passes cleanly. Trained sits within 0.78 of Greedy on the hardest
  red, matches Greedy on the other two.

The remaining gap to the strict bar is an architectural ceiling of the
flat-MLP shared-parameter policy (no coordination mechanism). Stage 1
was officially closed as a soft pass in July 2026; Stage 2 (GNN over
entities with CTDE critic — see
[`docs/stage2_gnn_design.md`](stage2_gnn_design.md)) directly tests
that architectural-ceiling hypothesis on the *same* env.

**Full v2 eval matrix** (50 episodes/cell, deterministic, matched
seeds; MLP policy from `runs/stage1/20260701_110225/`):

| Blue \ Red | Stationary | Random | RunFromNearest |
|---|---|---|---|
| Random  | −13.17 ± 9.76  (0.70 caught) | −13.88 ± 10.52 (0.64) | −20.24 ± 7.51 (0.24) |
| Greedy  | +17.65 ± 1.02  (2.00 caught) | +17.38 ± 1.21  (2.00) | +16.15 ± 1.30 (2.00) |
| **v2 Trained MLP** | **+17.17 ± 1.15 (2.00)** | **+17.27 ± 1.32 (2.00)** | **+15.37 ± 1.43 (2.00)** |

## Caveat

`scripts/evaluate_trained.py` **overwrites this file on every run**
with the auto-generated evaluation report of whichever checkpoint is
being evaluated.  If the file has been overwritten (e.g. after a
Stage 2 or smoke run), this hand-written SOFT-PASS verdict lives in
git history at commit [0fbee36](https://github.com/quialga/Multi-UAV-ISR/commit/0fbee36)
and can be restored from there.  Stage 2 results will land in
`docs/stage2_results.md` (separate file) once the GNN training run
completes.
