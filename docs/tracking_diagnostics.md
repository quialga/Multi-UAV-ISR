# Tracking diagnostics (feature/target-tracking)

Measurements behind the multi-target tracker, recorded so the conclusions
are not re-derived. All are CPU, offline, and reproducible with
`scripts/eval_tracking.py` plus the scratch scripts noted below.

Scenario throughout: 5 blue / 3 red / 4 obstacles, `run_from_nearest_uav`,
`sensor_radius = 40` in a 130 m arena, the realistic sensor model
(`p_TP`, occlusion, range-scaled noise). Random-walk blue policy for the
measurements (worst case for coverage).

## 1. Headline: tracker vs belief map vs references

8 episodes × 150 steps, 5 m match gate:

| configuration | MOTA | MOTP | IDF1 | recall | IDSW | Frag | FP | MT | ML |
|---|---|---|---|---|---|---|---|---|---|
| raw detections (floor) | 0.01 | 1.73 | 0.19 | 0.27 | 404 | 103 | 510 | 0.04 | 0.50 |
| belief peaks (today) | −0.39 | 2.14 | 0.24 | 0.33 | 178 | 138 | 2405 | 0.04 | 0.33 |
| KF + oracle assoc | 0.29 | 1.10 | 0.39 | 0.31 | 36 | 32 | 25 | 0.08 | 0.42 |
| KF + real assoc | 0.28 | 1.18 | 0.34 | 0.31 | 38 | 31 | 73 | 0.08 | 0.42 |

**Two references make these interpretable:**

- **Detectability ceiling = 0.30** — the fraction of active reds inside
  `sensor_radius` at any instant. Recall cannot exceed this without
  PREDICTING through the gaps, so it is the number to judge recall against,
  not 1.0.
- **Naive floor** — every raw return as a hypothesis, no filter, no
  association, no memory.

**What the tracker buys: precision and identity, not coverage.** 2× better
localisation (MOTP 1.18 vs 2.14, and MOTP is CONDITIONAL on matching, so
the belief map's 19–40 m peaks never even enter its 2.14 — the true gap is
wider), 30× fewer false positives (73 vs 2405), 4.7× fewer ID switches
(38 vs 178 — and the belief map's identity is HANDED OVER by the simulator,
yet still unstable because slot order shuffles with visibility).

**MOTA is ~degenerate here**: FN is >95% of its loss, so MOTA ≈ recall and
adds nothing. **IDF1 0.34** is ~74% of the ceiling that recall 0.31 allows
(perfect-identity max ≈ 0.46), so the identity deficit is mostly a recall
deficit. **Association is NOT the bottleneck**: oracle 0.29 vs real 0.28
MOTA; the FILTER limits quality.

## 2. Where recall is lost (`scratch/diag_gap.py`)

`recall = 1 − FN/n_gt`; per frame `n_gt` = active reds, hypotheses matched
to GT by Hungarian within the 5 m gate, `FN` = unmatched GT.

| mechanism | measured |
|---|---|
| gate rejects a re-detection | **0.8%** of returns (12 / 1442) |
| coasting drift vs own target | **~0.9 m per step** (1.7 m at 1, 5.5 m at 5, 9.2 m at 10) |
| FN because **no live track** for that target | **96%** |
| FN because a live track **drifted > 5 m** | 4% |

The gate almost never rejects, because `P` grows with `Q` while coasting, so
the Mahalanobis gate widens in proportion to the uncertainty — the filter
self-regulates. The drift is real, but its damage is indirect: a target
goes long unseen → the track dies at `max_misses` → no track → FN. Raising
`max_misses` does not help, because at 0.9 m/step the surviving track
becomes a false positive rather than a match (recall flat, MOTA collapses —
§3 of the coasting sweep in the commit history).

## 3. Tuning Q does not move recall (`scratch/sweep_q.py`)

`sigma_a` swept over 16×:

| sigma_a | recall | MOTP | FP | MOTA | NEES | NIS |
|---|---|---|---|---|---|---|
| 0.18 | 0.31 | 1.19 | 327 | 0.17 | 10.18 | 2.41 |
| 0.707 | 0.31 | 1.18 | 91 | 0.25 | 5.29 | 1.18 |
| **1.41** | 0.30 | 1.18 | 105 | 0.25 | **4.91** | 1.02 |
| 2.83 | 0.30 | 1.30 | 71 | 0.26 | 5.38 | 1.04 |

**Recall is flat.** `Q` scales the covariance, not the predicted MEAN, and
recall is measured on the mean. So the motion model, not filter tuning, is
the lever.

## 4. sigma_a: white-noise value × correlation correction

Two steps, agreeing between theory and measurement:

1. **White-noise value.** The red normalises acceleration to unit
   magnitude, so only direction varies: `a_i = a_max·cos θ`,
   `Var[a_i] = a_max²/2`, `sigma_white = a_max/√2 = 0.707` (measured 0.704).

2. **Correlation correction.** DWNA assumes white noise; this acceleration
   is not. Per-track lag-1 autocorrelation `rho = 0.546` (`scratch/autocorr.py`;
   a naive estimate that concatenates tracks is meaningless — it was
   ~0.04, an artefact of crossing track boundaries). For AR(1)-like
   acceleration the variance of the sum of N increments is
   `N s² (1 + 2 Σ_{k=1..N-1} (1−k/N) rho^k)`, an inflation of **2.4–3.1**
   over 5–20 step horizons, so `sigma_a` should be scaled **×1.55–1.77**.
   An independent NEES sweep put the optimum at **×2**. Within ~15%.

Adopted `sigma_a = a_max·√2` (×2 the white-noise value). NEES 5.29 → 4.91
against a target of 4.0. This buys calibration and fewer false positives;
it does NOT change recall (§3).

## 5. Why F cannot be fixed, and the recall budget

`F` is a LINEAR, state-INDEPENDENT transition. The red's motion is a
NON-LINEAR function of a state `F` does not even see (nearest blue,
obstacles, walls). No constant 4×4 matrix can represent "flee the nearest
blue". Evidence: swapping in the TRUE policy as the motion model
(`motion_model` plug point) roughly DOUBLES recall.

Recall budget:

| stage | recall | limited by |
|---|---|---|
| today (constant velocity) | 0.30 | at the detectability ceiling; bridges no gaps |
| + learned motion model (step b) | **0.64** (measured with the true policy) | still capped by detection |
| + directed search (step c) | > 0.64 | targets never yet detected |

The remaining 0.36 at stage 2 are targets **never detected**, so no track
is even born — unfixable by any filter; it needs SEARCH (the grid as birth
intensity, and the planner).

## 6. Clutter (`clutter_rate`) — the real cost of association

Added `clutter_rate` to `raw_detections()`: Poisson-mean false plots per
blue per scan, uniform in the sensor disk, occlusion-gated like a real
return, with a meaningless (random) Doppler. Modelled at the PLOT level,
not the belief map's per-cell `p_FP` — a per-cell rate would inject ~one
false return per resolution cell (~200 in a 40 m disk), which is not what
a real plot extractor (CFAR + clustering) leaves behind.

Why it matters: without clutter every return belongs to some real target,
so the matcher barely errs and the gate / M-of-N / birth logic look better
than they are. Oracle association rejects clutter by construction
(`truth_id < 0` is never assigned or used to birth a track), so the gap
between oracle and real IS the cost of associating under clutter:

| clutter / blue / scan | assoc | MOTA | recall | FP | tracks |
|---|---|---|---|---|---|
| 0.0 | oracle | 0.27 | 0.30 | 55 | 0.97 |
| 0.0 | real | 0.25 | 0.30 | 105 | 1.02 |
| 0.3 | oracle | 0.27 | 0.30 | 58 | 0.97 |
| 0.3 | real | **−0.22** | 0.30 | 1356 | 2.41 |
| 1.0 | oracle | 0.28 | 0.31 | 45 | 0.97 |
| 1.0 | real | **−2.52** | 0.29 | 7548 | 9.27 |

The oracle line is flat regardless of clutter (as it must be); the real
line collapses because clutter that survives one scan's gate becomes a
tentative track, and enough clutter births more tracks than there are
targets. This is the honest number the §1 headline table did not have
(it was measured at `clutter_rate = 0`); a follow-up should re-run it at a
non-zero rate and, if this collapse matters in practice, tighten
`birth_cluster_dist` / `confirm_hits` against it specifically.

## Reproduce

```
python scripts/eval_tracking.py --episodes 8 --steps 150
```

Scratch scripts (`scratch/`, not committed): `diag_gap.py` (§2),
`sweep_q.py` (§3), `autocorr.py` (§4), `clutter_impact.py` (§6). NEES/NIS
are in the eval output.
