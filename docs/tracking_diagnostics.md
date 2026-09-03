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

> **Table staleness note.** This table predates the `sigma_a` correlation
> correction (§4, `sigma_a: a_max/√2 → a_max·√2`) and the Gaussian-Sum
> refactor (§8). Re-measured with both in place, KF + oracle / KF + real
> now read `MOTA 0.28/0.27`, `MOTP 1.17/1.22`, `IDF1 0.38/0.37`,
> `recall 0.30/0.30`, `IDSW 46/46`, `FP 42/60`, `NEES 3.83/4.02`,
> `NIS 1.07/1.04` — all within measurement noise of this table, confirming
> (again) that §8's refactor is bit-exact for the default motion model.
> Left as originally recorded rather than silently overwritten.

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

## 7. The red is now STOCHASTIC (a scope decision)

Everything from here assumes a stochastic evader — a deterministic one is
convenient but not what a real adversary does, and it distorts the whole
downstream design (a predictor trained against it has no aleatoric
uncertainty to represent, so a mixture model has nothing to earn).

`isr/agents/stochastic_red.py::StochasticRed` wraps the deterministic
heuristic with three independent, separately-tunable sources. All default
to off, so with no arguments it is byte-identical to
`run_from_nearest_uav`.

Measured (same start, same blue actions, re-rolled noise; `scratch/check_stoch_red.py`):

| config | divergence @30 steps | rho lag-1 | aleatoric bimodality |
|---|---|---|---|
| deterministic (before) | 0.0 m | 0.662 | 0% |
| iid (`rho = 0`) | **1.6 m** (0.3 cells) | **0.536** ↓ | 0% |
| AR(1) correlated | 2.5 m | 0.669 | 0% |
| AR(1) + committed side | **9.0 m** | 0.639 | **20%** (80° apart) |

Three conclusions, each of which shaped the design:

1. **iid noise is nearly invisible.** 1.6 m over 30 steps is 0.3 belief
   cells — below the grid's resolution, and any predictor averages it out.
   It adds variance to the training target while changing no behaviour
   worth predicting.
2. **iid noise also WHITENS the natural correlation** (rho 0.662 → 0.536),
   destroying the very structure that §4's `sigma_a` correction is built
   on. Correlated (AR(1)) noise preserves it (0.669).
3. **Only the committed side choice produces ALEATORIC bimodality.**
   Measured with the state held EXACTLY fixed and only the policy's own
   noise re-rolled, so the 20% is not our belief uncertainty leaking in.
   This is the justification for a mixture/particle predictor rather than
   a single Gaussian.

The AR(1) innovation is scaled by `sqrt(1 - rho^2)` so the marginal std
stays `heading_noise_std` whatever `rho` is — otherwise turning correlation
up would silently turn the noise amount down, confounding two knobs.

**Consequence for step (b).** There are now TWO sources of multimodality:

* **epistemic** — our uncertainty over `s` pushed through the policy's
  "nearest blue" discontinuity. Measured at **45% bimodal even at
  sigma = 1 m**, rising to 78% at sigma = 20 m, modes 115-142° apart. This
  one needs no mixture *output*: sampling `s ~ N(x, P)` and evaluating a
  DETERMINISTIC regressor reproduces it for free.
* **aleatoric** — the evader's own committed choices (20%, above). This
  one does need a mixture output, or explicit noise sampling.

Either way, **collapsing the predicted mixture to moments is wrong**: with
modes ~115° apart the mean lands between them, which is the one heading the
evader will not fly. The `motion_model` plug point's `(x, P) -> (x-, P-)`
signature must therefore widen to carry particles or mixture components;
that is a real change to the tracker, not just a swapped function.

## 8. From a single Kalman filter to a Gaussian Sum

**Design discussion, 2026-09.** Once the red is stochastic (§7), the
question becomes how a LEARNED transition model plugs into the tracker,
and why a single Gaussian (what `motion_model` currently supports) is the
wrong target to plug it into. Recorded here because the derivation and
the measurements that settled it are easy to re-litigate otherwise.

### 8.1 The formal target, and what is already solved

The predictive distribution is

```
P(a_t | O_t) = sum_S  P(a_t | S_t) · P(S_t | O_t)
```

`a_t` = the evader's acceleration, `O_t` = our observations, `S_t` = the
true state (evader position/velocity + context: blue positions, obstacles,
walls). Given `a_t`, position and velocity at `t+1` follow from exact
kinematics.

**`P(S_t | O_t)` must NOT be learned — the tracker already computes it,
and it is measurably well calibrated.** It is the Bayesian posterior under
the sensor model (§1-§6): NEES 4.91 against a target of 4.0. Learning it
would re-derive, with a network, something already available in closed
form and independently verified. The split is:

| piece | source | why |
|---|---|---|
| `P(S_t \| O_t)` | **the tracker** (Bayes + sensor model) | known exactly, calibrated |
| `P(a_t \| S_t)` | **learned** | this is where the adversary's policy is genuinely unknown |

If the concern is that the tracker's Gaussian cannot represent a
multimodal posterior — correct, but the fix is the Gaussian Sum below,
not learning the posterior from scratch.

### 8.2 The missing term: G·mu_a, not just a bigger Q

Writing the propagation with a control input `a ~ (mu_a, Sigma_a)` and `G`
the standard constant-acceleration input matrix
(`G = [[dt^2/2, 0], [0, dt^2/2], [dt, 0], [0, dt]]`):

```
x_{t+1} = F x_t + G a_t
  =>  E[x-] = F x + G mu_a          P- = F P F^T + G Sigma_a G^T
```

`F` is unchanged (kinematics); the interesting terms are the two boxed
ones. Today `mu_a = 0` (DWNA assumes zero-mean acceleration), which is why
a coasting track drifts in a straight line and diverges ~0.9 m/step (§2).
**Nearly all the value of a learned model is `G·mu_a != 0`** — Q is the
second-order term.

Confirmed by construction: with `Sigma_a = sigma_a^2 I` (isotropic,
zero-mean), `G Sigma_a G^T` is EXACTLY the DWNA `Q` this tracker already
uses (`isr/tracking/tracker.py::_dwna_Q`) — checked numerically to
machine precision. **The current filter is the zero-knowledge limit of
this formulation**, not a different model: a learned motion model
generalises it rather than replacing it.

### 8.3 Where the multimodality comes from, and that it is small

The discontinuity in `run_from_nearest_uav` ("flee the NEAREST blue")
does not make `P(S_t | O_t)` multimodal directly — the sensor likelihood is
smooth (Gaussian). It enters through PREDICTION: our belief over `S_t`,
pushed through the policy's discontinuity, makes `P(a_t | O_t)` multimodal,
and that multimodal prior is what contaminates the next posterior. So this
is squarely a PREDICT-step problem, which is exactly where a Gaussian Sum
branches.

Measured mode count of `P(a | belief)`, sampling `s ~ N(x, P)` through the
(possibly stochastic) policy (`scratch/mode_count.py`):

| sigma_pos | mean modes | p90 | unimodal | >=5 modes |
|---|---|---|---|---|
| 1 m (deterministic red) | 1.75 | 3 | 57% | 4% |
| 5 m | 2.24 | 3 | 19% | 3% |
| 20 m | 3.13 | 4 | 3% | 6% |
| 1 m (StochasticRed) | 1.76 | 3 | 54% | 1% |
| 20 m (StochasticRed) | 2.65 | 4 | 18% | 6% |

Three conclusions:

1. **Branching factor is small** (mean 2-3, p90 3-4). No combinatorial
   explosion; a cap of 8 components gives generous headroom.
2. **The mode count is driven by OUR uncertainty, not the adversary.** At
   sigma=1 m (a live track) 54-57% of cases are already UNIMODAL — the
   Gaussian Sum degenerates to a plain Kalman filter exactly when the
   track is well localised, i.e. exactly when the extra machinery is not
   needed. It costs nothing in the easy case.
3. **The two branching sources do not compose multiplicatively.**
   `StochasticRed` at sigma=20 has FEWER modes (2.65) than the
   deterministic policy (3.13) — heading noise blurs epistemic modes
   together rather than multiplying them. Better than feared.

### 8.4 The action space is 1-D, so use a categorical, not a 2-D mixture

The red's acceleration has CONSTANT magnitude (`|a| = 1.000`, measured std
`0.000`; `StochasticRed`'s rotations preserve this). **The action lives on
the unit circle — it is one angle, not a 2-vector.** A learned model should
therefore output a categorical distribution over discretised heading bins
(e.g. 36 bins of 10 deg), not a 2-D (logistic) mixture density:

- Bimodality is native to a categorical (two peaks in a softmax) — no
  component count `K` to choose, no mode collapse, no `logsumexp` mixture
  machinery.
- WaveNet-style discretised logistic mixtures earn their keep on a LARGE
  output alphabet (65536 for 16-bit audio) where a plain categorical would
  have too many classes. With 36 heading bins, plain categorical (with
  neighbour label-smoothing, since a heading is circular and 10 deg away
  is not really a different class) is the simpler, sufficient choice.
- Discretise the ACTION (1-D), not the STATE (4-D) — a 4-D state grid is
  the cost we rejected in §5/§20 (the belief-map kernel investigation).

### 8.5 Multiple hypotheses per track: Gaussian Sum, not particles

§8.3's point 3 (`P(S) x P(a|S)` mixing epistemic and aleatoric branching)
argues for maintaining several `(x, P)` hypotheses per track with prune and
merge — a Gaussian Sum Filter (GSF) / multi-hypothesis tracker — rather
than a full particle filter: the branching factor is small (8.3), and each
branch has an analytic Gaussian update (Kalman), so GSF is the cheaper
sufficient tool.

**Implemented in `isr/tracking/tracker.py`.** A `Track` is now a mixture
of weighted `_Component`s (`w, x, P`), not a single `(x, P)`:

- **PREDICT**: each component is propagated through `motion_model(x, P) ->
  [(rel_weight, x_pred, P_pred), ...]` (a learned model would emit one
  branch per significant heading bin, per §8.4); branch weight = parent
  weight x `rel_weight`.
- **REDUCE (mandatory after every predict, not only after update)**: drop
  negligible weights, merge near-duplicate components (moment-matched,
  gated by Mahalanobis distance between means), cap the count. Mandatory
  because a coasting track (no detection to Bayes-reweight it) branches
  every PREDICT step; unchecked, the hypothesis count is unbounded after a
  long gap.
- **GATE**: Mahalanobis cost of a (track, detection) pair uses the
  BEST-matching component — a multimodal track is gated in if ANY
  hypothesis is consistent.
- **UPDATE**: every component is updated with the assigned detection, then
  components are Bayes-reweighted by their measurement log-likelihood
  (log-space softmax — necessary once components have diverged enough for
  raw likelihood ratios to under/overflow). This is "the update prunes
  automatically": a branch the detection contradicts is down-weighted, not
  merely left alone; the explicit REDUCE step only has to clean up what an
  ambiguous single observation could not resolve.
- **READOUT is always the DOMINANT (highest-weight) component** — `.pos`,
  `.vel`, `.x`, `.P` never average across components. This is the entire
  point: with modes ~90-140 deg apart (§8.3, §7), the mixture MEAN lands on
  a heading the evader will never fly. Anything downstream that reads a
  track (the MOT harness, a future policy) gets the dominant mode, not an
  average.
- **BIRTH is always unimodal** — a new track has no basis for multiple
  hypotheses yet; multimodality emerges only from subsequent PREDICT
  branching.

### 8.6 Non-regression: bit-exact for the default motion model

With `motion_model=None`, PREDICT always emits exactly ONE branch per
existing component, so a track never exceeds 1 component, REDUCE is a
no-op every time (nothing to prune/merge/cap), and the log-space Bayes
reweight of a single component is `exp(0)/exp(0) = 1` — inert. Every
number the tracker produces is then bit-for-bit identical to the
single-Kalman-filter implementation it replaces.

Verified three ways:

1. `tests/test_gaussian_sum.py` runs IDENTICAL detection streams (recorded
   from real `PursuitEnv` episodes, including clutter and oracle
   association) through the new tracker and a frozen copy of the
   pre-refactor one (`tests/_frozen_pre_gsf_tracker.py`, pinned at commit
   `b18bf66`), asserting exact equality of `x`, `P`, `confirmed`, `misses`,
   `history` and `last_nis` at every step, across several seeds.
2. `scripts/eval_tracking.py --episodes 8 --steps 150` produces IDENTICAL
   output with the refactor stashed vs applied (`MOTA 0.28/0.27`,
   `MOTP 1.17/1.22`, `recall 0.30/0.30` — see the §1 staleness note above).
3. The new mechanic is tested in isolation with a stand-in branching
   `motion_model` (rotates velocity two ways — a placeholder for the not-
   yet-built learned model): branching, dominant-mode readout vs mixture
   mean, Bayes down-weighting of a contradicted branch, merge, prune,
   component-count capping through a long coast, and weights summing to 1.

### 8.7 What is NOT yet built

The learned transition model itself (the categorical-over-heading network
of §8.4) does not exist yet — `motion_model` has no real implementation to
plug in, only the stand-in test model. `merge_gate` (squared Mahalanobis
merge threshold, default 4.0) and `max_components` (default 8) are
placeholders, sized from §8.3's measurements but not yet tuned against a
real branching model's actual mode separations. Both are inert until a
branching `motion_model` is supplied.

## Reproduce

```
python scripts/eval_tracking.py --episodes 8 --steps 150
```

Scratch scripts (`scratch/`, not committed): `diag_gap.py` (§2),
`sweep_q.py` (§3), `autocorr.py` (§4), `clutter_impact.py` (§6),
`check_stoch_red.py` (§7), `mode_count.py` (§8.3). NEES/NIS are in the
eval output.
