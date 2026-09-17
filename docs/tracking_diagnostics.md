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

> **PARTLY CONFIRMED, with corrected numbers — see §11.** The original
> text below claimed the true policy "roughly DOUBLES recall" and a budget
> of 0.30 → 0.64; neither 0.64 nor a doubling was reproduced. What was
> re-measured (§11.1, stochastic red, `max_misses=80`): at a 5 m gate,
> constant velocity 0.346 → learned model 0.509 (true-policy oracle
> 0.481); at a 20 m gate, 0.441 → 0.609. A real gain, but ~1.4× rather than
> 2×. It exists **only once `max_misses` is raised**: at the default coast
> budget of 5 steps every motion model scores ~0.33. The paragraph and
> table below are kept as originally written except where marked.

`F` is a LINEAR, state-INDEPENDENT transition. The red's motion is a
NON-LINEAR function of a state `F` does not even see (nearest blue,
obstacles, walls). No constant 4×4 matrix can represent "flee the nearest
blue". ~~Evidence: swapping in the TRUE policy as the motion model
(`motion_model` plug point) roughly DOUBLES recall.~~ *(Not reproduced;
see the banner.)*

Recall budget — note the rows were measured under DIFFERENT settings and
are not directly comparable:

| stage | recall | settings | limited by |
|---|---|---|---|
| today (constant velocity) | 0.30 | 5 m gate, `max_misses=5` | the detectability ceiling |
| + learned motion model (step b) | **0.58** (0.61 with oracle association) | 20 m gate, `max_misses=80` | still capped by detection |
| + directed search (step c) | > 0.61 *(expected, not measured)* | — | targets never yet detected |

Like-for-like, the motion-model gain is the §11.1 table: 0.346 → 0.509 at
5 m and 0.441 → 0.609 at 20 m, both with `max_misses=80`. The remainder
at stage 2 are largely targets never detected, so no track is born — that
needs SEARCH (`docs/search_design.md`).

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

### 6.1 Hardening for the long coast budget

The tracker that will feed the policy coasts long (§11: `max_misses` 80),
and that makes clutter far worse than the table above, which used 5: every
phantom that manages to confirm now lives up to 80 steps. Operating point
**0.2 plots / blue / scan** (a CFAR false-alarm rate of ~10⁻³ per
resolution cell, with ~200 cells in a 40 m disk); **0.5** as a stress
point.

Measured (`scratch/clutter_sweep.py`): 8 episodes × 150 steps, stochastic
red, random blues, **constant-velocity motion model**, 20 m gate, coasting
bounded at 80 steps. Every configuration within a clutter level steps on
the same detections.

| config (clutter 0.2) | MOTA | recall | FP | IDSW | Frag | confirmed / step |
|---|---|---|---|---|---|---|
| one shared budget, 2-of-3 | −0.64 | 0.380 | 3604 | 56 | — | 4.14 |
| tentative budget split, 2-of-3 | −0.56 | 0.369 | 3288 | 61 | 41 | 3.85 |
| split, 3-of-4 | −0.27 | 0.348 | 2186 | 54 | 38 | 2.87 |
| **coverage-aware misses, 3-of-4, 3 in view** | **0.17** | 0.339 | **559** | **47** | **32** | **1.48** |
| coverage-aware, 2-of-3, 3 in view | 0.10 | 0.349 | 851 | 53 | 31 | 1.76 |
| coverage-aware, 3-of-4, 6 in view | −0.24 | 0.346 | 2042 | 53 | 37 | 2.74 |
| oracle association, split 2-of-3 | 0.17 | 0.414 | 819 | 45 | 32 | 1.92 |

What each lever does:

* **Separate tentative miss budget** (the "split" rows; a
  `max_misses_tentative` parameter since removed — superseded by the
  confirmation deadline below): alive tentatives fall 5× (4.37 → 0.80 per
  step) but false positives only 9%. It fixed the cost and the
  phantom-absorption failure, not the confirmation rate.
* **3-of-4 confirmation**: FP −34% against 2-of-3, at one step of latency
  (recall 0.369 → 0.348). 3-of-5 was slightly worse — a longer window gives
  clutter more scans to collect its hits.
* **Doppler gating**: not in the table because it provably cannot help
  here. With the CV model's calibrated process noise the predicted velocity
  has sd ~1.41 m/s, while clutter Doppler lies in [−1, 1], so the Doppler
  term adds at most ~2.9 to d² against a threshold of 11.3. The red really
  can change radial speed that much per scan. It could help only with a
  motion model whose velocity prediction is confident
  (`tests/test_doppler_gating.py`).
* **Coverage-aware miss counting** (`step(..., coverage=...)`): a confirmed
  track's miss counts only if some blue would certainly have had it in view
  — in range with a 2σ margin, with line of sight. A confirmed phantom sits
  where its plot appeared, so sensors keep looking at it and it dies in a
  few steps; a real target that fled out of range accrues no counted misses
  and coasts on, bounded by `max_coast_steps`. This is the dominant lever:
  **FP ÷3.9 and MOTA −0.27 → 0.17** against split 3-of-4. Under stress
  (0.5): MOTA −0.41 → 0.06, FP 2620 → 925.

Checks that it does no harm: without clutter the coverage rows are
identical to their counterparts on every metric, and with clutter IDSW and
Frag go *down* — no sign of real tracks being killed on `p_TP` bad luck.

**Confirmation deadline vs a tentative miss budget.** The two rules for
tentatives overlap: M-of-N only PROMOTES, the miss budget only DELETES,
but the budget's sensible value is fixed by the window (after N−1
consecutive misses every earlier hit has left it), and set independently
they can clash — a budget of 0 silently turns 3-of-4 into 3-of-3. They
also leave a gap: a tentative that ALTERNATES hit and miss never confirms
and never chains enough misses to die. `confirm_deadline` replaces the
budget with "one full window to confirm, otherwise delete". Same sweep,
coverage-aware, 3 in view:

| | clutter | MOTA | FP | IDSW | tentatives / step | tentatives past their window / step |
|---|---|---|---|---|---|---|
| 3-of-4 + miss budget 3 | 0.2 | 0.17 | 559 | 47 | 2.05 | 0.227 |
| 3-of-4 + deadline | 0.2 | 0.16 | 602 | 47 | 1.49 | 0.000 |
| 3-of-4 + miss budget 3 | 0.5 | 0.06 | 925 | 51 | 5.26 | 1.224 |
| 3-of-4 + deadline | 0.5 | 0.10 | 780 | 43 | 3.74 | 0.000 |

The gap was real (0.23 lingering tentatives per step at 0.2, 1.2 at 0.5)
and the deadline removes it entirely. At the operating point the two are
close, the deadline slightly behind (FP +8%, one seed set — not
distinguishable from noise here); under stress the deadline is clearly
better. Without clutter they are identical.

The miss budget was therefore **removed**. With `confirm_deadline` on, the
deadline is the only rule that deletes a tentative, and `max_misses`
governs confirmed tracks alone — otherwise a small `max_misses` (such as
the in-view budget of 3) would cut tentatives before their window ends and
reopen the same clash. Without the deadline, tentatives fall under
`max_misses` as originally.

Caveats. The in-view budget is sharply sensitive: 6 gives back almost all
of the gain, far more than doubling a phantom's lifetime would explain; a
plausible but unverified reason is that a longer-lived phantom gets more
chances to absorb another clutter plot, resetting its counter. The
coverage row showing fewer FP than the oracle row is **not** it beating
perfect association — that oracle row lacks the coverage rule and keeps
drifting CV tracks for 80 steps; nor is it established why the coverage
configuration shows fewer FP with clutter (559) than the same tracker
without it (709). Occlusion used true obstacle geometry. Not yet
re-checked with the learned motion model.

### 6.2 The chosen configuration with the learned model, and its cost

Chosen configuration: 3-of-4 with `confirm_deadline`, coverage-aware
misses with an in-view budget of 3, `max_coast_steps` 80, clutter 0.2.
Same 8 episodes × 150 steps (`scratch/tracker_timing.py`), both trackers
stepping on the same detections:

| motion model | MOTA | recall | FP | IDSW | Frag |
|---|---|---|---|---|---|
| constant velocity | 0.16 | 0.338 | 602 | 47 | 32 |
| **learned (v3)** | **0.28** | **0.387** | **332** | 47 | 31 |

The learned model holds up under clutter: MOTA +0.12, recall +15%, false
positives −45%.

Cost per env step, single torch thread (the relevant setting for env
worker processes):

| | mean | p95 |
|---|---|---|
| `env.step` | 2.30 ms | 3.29 ms |
| `raw_detections` + `track_coverage` | 0.34 ms | — |
| tracker, constant velocity | 0.69 ms | 1.76 ms |
| tracker, learned | 6.13 ms | 11.89 ms |

The learned tracker makes 3.1 motion-model calls per step (one per
component, ~3.1 tracks alive), 1.6 ms each, and **81% of its time is inside
those calls**. For a ~100k-parameter network that per-call cost is mostly
per-sample Python overhead (building a feature dict, `featurize_shard` on
one sample, tensor creation), so batching every call of a step into one
featurise-and-forward is the obvious optimisation — expected to be
several-fold, not measured.

At the Stage 4 training configuration (16 envs × 320 steps × 1000
rollouts = 5.12 M env steps), the learned tracker adds ~31 s of CPU per
rollout — **~8.7 h over a run if envs step in-process** (`--n-workers 0`,
the default), or roughly a sixteenth of that with 16 env workers. The CV
tracker adds ~3.5 s per rollout. Policy inference and PPO update time were
not measured, so these are added costs, not shares of the total.

Caveats: a Windows laptop CPU, and noisy — the same run with torch's
default 4 threads showed `env.step` at 4.4 ms, although `env.step` does not
use torch, so run-to-run variance is large and the benefit of one thread
cannot be cleanly separated from it. Random blue actions throughout.

**Batching the motion-model calls.** The tracker now calls
`predict_batch(xs, Ps)` once per step for every component of every track
when the model provides it (`MultiTargetTracker._predict_all`); the
adapter packs each sample exactly as before, concatenates, and runs one
featurise and one forward. Same run, same detections, one torch thread
(`scratch/batching_timing.py`):

| learned tracker | mean | vs `env.step` |
|---|---|---|
| one call per component | 12.80 ms | 3.0× |
| one batched call per step | 7.17 ms | 1.7× |

**1.79× faster**, with identical tracking (MOTA 0.28, recall 0.387, FP 332,
IDSW 47 either way; equivalence pinned in `tests/test_motion_batching.py`).
That is less than the several-fold expected above, and the breakdown says
why — the batched cost still grows with the number of components
predicted:

| components / step | one call each | batched |
|---|---|---|
| 1–2 | 6.12 ms | 4.40 ms |
| 3–5 | 15.61 ms | 8.05 ms |
| 6+ | 27.83 ms | 16.02 ms |

A single batched forward would be nearly flat in that count, so most of
what remains is per-component work outside the network: packing each
sample, basin splitting, branch construction, and the tracker's own
reduce. Not profiled. Absolute timings in this run were slower than the
previous one (`env.step` 4.3 ms against 2.3 ms), again machine noise, so
the within-run ratios are what to rely on: applied to the estimate above,
the learned tracker's added cost over a 1000-rollout run falls from ~8.7 h
to roughly 5 h in-process, or on the order of 20 minutes with 16 env
workers.

### 6.3 At the training configuration: clutter-captured tracks

**Setting** (`scratch/obs_k_sigma.py`). Everything above was measured at
L=130 with 5 blues. This is the Stage 4 training geometry
(`STAGE4_DEFAULTS`: L=200, 7 blues, 4 reds, 9 obstacles, 300 steps), clutter
0.2, obstacle radius noise 2 m, random blues, and the TRAINING red mix
(stationary / random / run, cycled per episode), 15 episodes. Chosen
configuration from §6.2. The learned model (v3) takes its obstacle context
from the obstacle tracker (confirmed tracks, estimated radius), as the
actor path will. Evaluation labels: at every hit a track takes the nearest
active red within 6 m, and keeps that label while it coasts.

The purpose was sizing the graph slots K and a position-σ cut-off for the
tracker-fed actor observation. It found this first.

**A confirmed track can be confidently wrong.** Confirmed tracks with
σ ≤ 10 m whose labelled red is more than 40 m away, per step (misleading /
all σ ≤ 10 m):

| | clutter 0 | clutter 0.2 |
|---|---|---|
| constant velocity | 0.00 / 1.17 | **0.24 / 1.46** (16%) |
| learned | 0.00 / 1.27 | **0.07 / 1.37** (5%) |

p90 error of confirmed tracks with σ in 2–5 m: 3.9 m → **109 m** (CV),
6.3 m → **61 m** (learned). Without clutter σ tracks the error well — the
learned model's labelled red is within 40 m in ≥ 99% of track-steps up to
σ = 40 m — so the failure is clutter-specific. Mechanism: a coasting track's
chi² gate grows with its predicted covariance; a false plot inside it counts
as a hit, which resets the misses, shrinks the covariance and moves the
estimate onto the plot, and the track lives on. Such a track reports a
SMALL σ, so no σ cut-off can remove it from the actor's graph.

**The kinematic reach gate** (`max_target_speed`; removed from the code after
§6.4 — last present in commit 8d026ab) bounded association by physics: a return joins a track only within
`max_target_speed · scans_since_hit + 3·sqrt(σ_z² + σ_hit²)` of where the
track stood at its last hit, with `max_target_speed = √2` (the env caps red
velocity at 1 m/s per axis). Same 15 episodes, clutter 0.2:

| | misleading / σ ≤ 10 m | p90 err, σ 2–5 m | MOTA stationary | MOTA random |
|---|---|---|---|---|
| CV | 0.24 / 1.46 | 109 m | 0.27 (FP 930) | 0.33 |
| CV + reach gate | 0.10 / 1.33 | 61 m | 0.20 (FP 1736) | 0.39 |
| learned | 0.07 / 1.37 | 61 m | 0.40 | 0.50 |
| learned + reach gate | 0.06 / 1.38 | 52 m | 0.39 | 0.52 |

It halves the problem for CV and barely moves it for the learned model.
(The `run` red is seen too rarely by random blues — recall ≤ 0.07 — to say
anything about it.) At clutter 0 it changes little (MOTA within 0.01 for
both models). For CV the false positives rise, consistent with tracks that
clutter no longer resets coasting on with large σ and drifting estimates:
confirmed CV track-steps with σ > 40 m go from 148 to 2077.

**Why a kinematic bound is not enough** (`scratch/reach_leak.py`, 8 episodes
of stationary and random reds, reach gate on). Hits on confirmed tracks after
which no red lies within 6 m of the track, taken as clutter hits, by the
number of scans the track had coasted:

| scans since last hit | CV clutter / real hits | clutter share | learned clutter / real | clutter share |
|---|---|---|---|---|
| 1 | 2 / 2949 | 0.001 | 0 / 2894 | 0.000 |
| 2–3 | 9 / 344 | 0.025 | 3 / 353 | 0.008 |
| 4–10 | 29 / 19 | **0.60** | 12 / 25 | **0.32** |
| 11–30 | 74 / 11 | **0.87** | 35 / 9 | **0.80** |
| 31–80 | 21 / 0 | 1.00 | 12 / 0 | 1.00 |

After about four coasted scans a single return on a confirmed track is more
likely clutter than the target, and by then the physically admissible reach
(√2 m per scan) is legitimately large, so no bound on speed rejects it. The
evidence that is missing is not WHERE the return is but HOW MUCH one return
is worth against the clutter density when the track could be anywhere in a
wide region. §6.4 solves this with re-acquisition; the reach gate adds
nothing on top of it and was removed.

Two more results from the same run:

* **Slots.** Confirmed red tracks per step: max 7, p99 6 (CV); max 6, p99 5
  (learned), against at most 4 real reds. The excess is duplicates and the
  clutter-captured tracks above, so K is to be re-measured once those are
  fixed.
* **The learned model out of its distribution.** v3 was trained at L=130 on
  stochastic reds; here it sees L=200 and stationary / random reds. It still
  beats CV at clutter 0.2 (MOTA 0.40 vs 0.27 stationary, 0.50 vs 0.33
  random); at clutter 0 the two are close (0.40 vs 0.46 stationary, 0.49 vs
  0.44 random).

### 6.4 Re-acquisition with confirmation, and the observation's K and σ cut-off

The two halves of track existence are now handled separately. The ABSENCE
of a return counts toward deleting a confirmed track only where it should
have been seen (coverage-aware misses, §6.1). The PRESENCE of a return after
a long coast is no longer trusted on its own:

`reacquire_after` (`tests/test_reacquisition.py`): a track that has gone
this many scans without a hit is LOST for association, so no single return
re-attaches to it. Returns near it start a tentative, which must pass the
normal 3-of-4 with deadline; when it confirms, it absorbs the closest lost
confirmed track consistent with it (χ² on both covariances) and takes over
its id. Clutter rarely passes 3-of-4, so the
single-plot capture of §6.3 disappears; a real re-acquisition reappears in
the graph a few scans later. `reacquire_after = 4` is read off §6.3's
clutter-share table (clutter < 3% of hits up to 3 scans, 32–60% at 4–10);
other values were not swept.

Same 15 episodes and seeds as §6.3; obstacle tracker in its static + merge
configuration (§9.6) for every row, so the learned model's obstacle context
is the same across rows.

| clutter | variant | model | misleading / σ ≤ 10 m | p90 err, σ 2–5 m | MOTA stat. | MOTA random | FP stat. | IDSW stat. | confirmed p99 / max |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | chosen (§6.2) | CV | 0.24 / 1.46 | 109 m | 0.27 | 0.33 | 930 | 38 | 6 / 7 |
| 0.2 | **+ reacquire 4** | CV | **0.00** / 1.13 | **4.0 m** | **0.44** | **0.44** | 569 | 25 | **4 / 5** |
| 0.2 | + reacquire 4 + reach gate | CV | 0.00 / 1.13 | 4.0 m | 0.43 | 0.43 | 724 | 23 | 5 / 5 |
| 0 | + reacquire 4 | CV | 0.00 / 1.13 | 3.7 m | 0.47 | 0.47 | 405 | 26 | 4 / 4 |
| 0.2 | chosen (§6.2) | learned | 0.07 / 1.37 | 60 m | 0.40 | 0.50 | 462 | 39 | 5 / 6 |
| 0.2 | **+ reacquire 4** | learned | **0.00** / 1.21 | **5.9 m** | **0.43** | **0.51** | 587 | 33 | 5 / 6 |
| 0.2 | + reacquire 4 + reach gate | learned | 0.00 / 1.21 | 5.9 m | 0.41 | 0.50 | 692 | 30 | 5 / 6 |
| 0 | + reacquire 4 | learned | 0.00 / 1.23 | 6.0 m | 0.43 | 0.52 | 606 | 41 | 4 / 5 |

* **The clutter damage is essentially gone.** With re-acquisition, clutter
  0.2 lands within 0.03 MOTA of clutter 0 for both models, and misleading
  confirmed tracks with σ ≤ 10 m fall to 0.00 per step (at most one on a
  rare step, for CV).
* **The reach gate adds nothing on top** (same misleading count, slightly
  lower MOTA, more FP), so it was removed from the code. Without it, a
  re-acquired track can inherit the id of a lost track further away than the
  red could have travelled; that only affects ids (the MOT identity
  metrics), never what the actor sees.
* **The learned model's FP rise** (462 → 587) is consistent with lost
  tracks now coasting on with a large, honest σ instead of being reset by
  clutter: its track-steps with σ > 40 m go from 2 to 1634. MOT counts them
  whatever their σ; the actor observation drops them by σ (below).
* The `run` red remains too rarely seen by random blues (recall ≤ 0.16) to
  judge; its IDSW are volatile across runs (learned: 9 to 46).

**σ is honest again, which fixes the cut-off.** Share of confirmed
track-steps whose labelled red is within 40 m (the sensor radius) of the
readout, clutter 0.2 with re-acquisition:

| σ (m) | 0–30 | 30–40 | > 40 |
|---|---|---|---|
| CV | ≥ 0.99 | 0.98 | 0.63 |
| learned | 1.00 | 0.99 | 0.78 |

**Adopted for the tracker-fed actor observation:**

* **σ cut-off τ = 40 m** (largest eigen-sd of the moment-matched mixture's
  position covariance): below it a blue flying to the readout has the red
  within its sensor radius in ≥ 98% of track-steps; above it that drops to
  63–78%. (Measured with `sigma_a_model` 0.35. After calibrating it to 0.20
  the 30–40 m band drops to 0.91 for the learned model; §11.6 explains why
  and keeps the cut-off at 40 m.)
* **K = 8 red slots** (2 × `n_red`). After the cut-off: p99 4–5, max 5–6
  confirmed tracks per step against at most 4 reds. Random blues; a trained
  team that converges several blues on one red could raise duplicates, hence
  the margin. Overflow keeps the smallest σ.
* **K = 12 obstacle slots** (`n_obstacles` + 3). Confirmed obstacle tracks:
  p99 10, max 10 for 9 obstacles. No σ cut-off is needed for static
  obstacles: their covariance does not grow while unobserved.

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

**Magnitude was assumed constant here, and that assumption did not
survive.** `run_from_nearest_uav` always renormalises to `|a| = 1.0` — a
scripted-heuristic artefact (the same class as the missing turn-rate
limit), not a physical law. The first cut of magnitude noise repeated
§7's own iid mistake: a plain multiplicative `speed_jitter` measured
**0.58 m divergence over 30 steps and lag-1 rho = 0.02** — indistinguishable
from white noise, i.e. exactly as invisible as iid heading noise was.
Retired and replaced with the same construction as heading: a
**deterministic threat ramp** (`min_effort` at range `>= threat_range`,
rising linearly to full effort as the nearest blue closes to zero range —
`threat_range` is the evader's OWN danger perception, deliberately
independent of our `sensor_radius`) **plus correlated AR(1) noise** on top
(`magnitude_noise_std` / `magnitude_rho`). Measured:

| magnitude source | divergence @30 steps | lag-1 rho of \|a\| |
|---|---|---|
| old `speed_jitter` (iid, retired) | 0.58 m | 0.02 |
| threat ramp alone (no noise) | 0.00 m | — |
| threat ramp + AR(1) noise | **2.71 m** | **0.765** |

The ramp alone gives exactly 0 divergence under re-rolled noise, as it
must — it is a deterministic function of state, so two seeds with
identical blue actions produce identical magnitude trajectories. The
combined rho (0.765) exceeds heading's natural 0.546 because the
deterministic ramp is ITSELF smooth in time (distance to the nearest blue
changes gradually), stacking with the AR(1) noise's own correlation.

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

### 8.4 The action space: discretised (heading, magnitude), not heading-only

An earlier pass at this section argued the action is 1-D — heading only —
because the deterministic `run_from_nearest_uav` always has `|a| = 1.000`.
**That assumption does not survive §7's magnitude fix**: constant-max
magnitude was itself a scripted-heuristic artefact (§7), not a property to
bake into the model's architecture. A realistic evader modulates effort
by assessed threat, so the learned model's action space should be the
full discretised grid: **heading bins x magnitude bins** (e.g. 36 headings
of 10 deg x 5 magnitudes at 0/25/50/75/100%), a single softmax over the
joint grid — matching the proposal made much earlier for the belief-map
reachability kernel (a sweep over discretised acceleration directions and
magnitudes). That kernel did not work as a FIXED, non-learned belief-map
update (§5/§20: it degenerates to ballistic extrapolation with a frozen
`v`, which loses badly once the evader turns at the measured 0.315 rad/step
median). Applied instead as the OUTPUT of a per-step LEARNED classifier —
where a real turn is a new categorical draw, not a frozen prediction
carried forward — it is exactly the right discretisation. The original
idea was not wrong; it was aimed at the wrong layer.

A single joint categorical over the `H x M` grid (still only ~180 classes
for 36x5) is preferred over a factored `P(heading) x P(magnitude)`:

- It represents ANY joint distribution over the discretised grid —
  including "sharp turn implies reduced forward effort" — without an
  explicit mixture-of-components structure: each grid cell IS a possible
  mode, and bimodality (§8.3) is just two grid cells holding mass, natively,
  with no component count `K` to choose, no mode collapse, no `logsumexp`
  mixture machinery.
- If the true behaviour turns out to be constant-max-magnitude after all,
  the network simply learns to put ~all magnitude-mass on the top bin, at
  negligible extra cost over the heading-only design.
- Discretise the ACTION (2-D: heading x magnitude), not the STATE (4-D) —
  a 4-D state grid is the cost already rejected in §5/§20.
- WaveNet-style discretised logistic mixtures earn their keep on a LARGE
  output alphabet (65536 for 16-bit audio); 180 classes is nowhere near
  that scale, so a plain categorical (with neighbour label-smoothing on
  the circular heading axis) is the simpler, sufficient choice.

This changes nothing about the Gaussian-Sum tracker (§8.5): `motion_model`
already accepts arbitrary `(weight, x_pred, P_pred)` branches, and does not
care whether they came from a 1-D heading-only categorical or a 2-D
heading x magnitude one — the extra dimension is confined entirely to the
not-yet-built learned-model layer.

**The magnitude axis needs a scale, and there are two** (`ACCEL_SCALE`,
`ACCEL_SCALE_BOX` in `red_motion_features`). The grid's units are the
env's: `PursuitEnv.step` clips the red action PER AXIS to [-1, 1] and feeds
it straight to `_integrate`.

* The **unit disk** (1.0) fits `run_from_nearest_uav` and `StochasticRed`,
  which normalise to `|a| <= 1`. Every checkpoint up to v3 used it.
* The env's **whole box** (√2) is what a red using the corners needs —
  `random_red`, one of the three policies the Stage 4 trainer runs, draws
  uniformly in the box and exceeds 1 in 21% of its steps, and a learned or
  self-play red would too. The collector fails loudly rather than folding
  those into the top bin (`test_collector_rejects_a_red_that_breaches_the_grid_bound`).

The scale travels WITH the model (`RedMotionGNN.accel_scale`, saved in the
checkpoint and read by `LearnedRedMotion`), so a consumer cannot get it
wrong and old checkpoints keep working. Keeping 5 magnitude bins at the box
scale also puts a bin CENTRE at 0.99 — on the flee response's `|a| = 1`,
which the unit-disk grid represented as 0.9.

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
  branch per significant (heading, magnitude) grid cell, per §8.4); branch
  weight = parent
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

## 9. Obstacle tracking: a Kalman filter, deliberately simpler than the red's

**Design discussion, 2026-09.** Prompted by two threads meeting: (1) the
red's learned motion model needs `p(c_t^obs)` — our uncertainty over
obstacle context — to marginalise over (§8.1), and today that uncertainty
is only ~14% resolved when it actually matters (the red is inside an
obstacle's influence band but no blue currently senses it: measured 17% of
instants in-band, only 14% of THOSE sensed — see the "two questions" scratch
investigation); and (2) a future FAST, SMALL interceptor obstacle
("simulate a missile launched at an ally" — explicitly deferred, not built
here) will need the same identity-preserving, occlusion-robust tracking the
red already has. Both point at the same missing piece: obstacles have no
persistent tracker, only a per-step belief-map peak or (when seen) an exact
instantaneous measurement.

**Not needed for collision avoidance.** Checked first, since it would
undercut the whole exercise if a prior obstacle map were required to avoid
crashing: `sensor_radius=40` against obstacle radii 5-15 m gives ~25 steps
of own-radar warning before the clearance margin, and the crash-avoidance
reward already operates on the SENSED track, not a prior. Reactive-only
avoidance was already correct; this is about PREDICTING the red and about
future fast-moving obstacles, not about blues bumping into static ones.

### 9.1 Why a separate tracker, not a reuse of MultiTargetTracker

* **State differs.** Obstacles carry a RADIUS the red's state does not:
  `[px, py, vx, vy, r]` (5-D) vs `[px, py, vx, vy]` (4-D).
* **No measured reason for a Gaussian Sum.** The red's multimodality comes
  from a policy discontinuity ("flee the NEAREST blue") and its own
  committed manoeuvre choices (§8.3, §7) — an obstacle does neither. Adding
  the Gaussian-Sum machinery here would be speculative generality with no
  measurement behind it, which is exactly the discipline this document
  exists to avoid. A single Gaussian per track is implemented; IF the
  future interceptor's guidance law (proportional navigation or similar)
  turns out to have its own discontinuities, that should be MEASURED
  first, the same way it was for the red, before retrofitting a mixture.

What IS shared, not re-derived: the Joseph-form Kalman update and the
Hungarian solver. `MultiTargetTracker._update`'s Joseph-form math was
dimension-agnostic already except for one hardcoded `np.eye(4)`; extracted
to `isr/tracking/kalman.py::joseph_update` with `np.eye(len(x))` instead —
bit-identical for the red's 4-D case (`len(x) == 4`), and now directly
reusable at 5-D for obstacles. Verified: the full non-regression suite
(`tests/test_gaussian_sum.py`, bit-exact against the frozen pre-GSF
tracker) still passes unchanged after the extraction.

### 9.2 RADIUS is the cleanest part of this design

A rigid obstacle's TRUE radius does not change during an episode, so
`F_r = 1`, `Q_r = 0` is not an approximation — unlike DWNA's white-noise
acceleration assumption for the red, there is no modelling error here at
all. The filter's radius variance shrinks monotonically with every
sighting and never diverges. `_build_obstacle_tracks` (what the policy
observes) reports the exact radius when seen — no measurement noise — so a
new, additive, default-off knob (`obstacle_radius_noise_std`, feeding a new
`PursuitEnv.raw_obstacle_detections()`, mirroring `raw_detections()`) gives
the tracker's radius filter something to average out; at the 0.0 default,
`raw_obstacle_detections` stays exact and `_fuse_birth` takes the first
exact return as ground truth outright rather than blending it.

RADIUS also needs no ridge prior, unlike velocity: it is a scalar measured
directly, never rank-deficient the way a single line-of-sight velocity
measurement is (§17's whole reason for a ridge). `vel_prior_std` (renamed
from an earlier, misleadingly-named `radius_prior_std` placeholder that
turned out to be used only for the velocity ridge) plays the same role as
the red tracker's parameter of the same name.

### 9.3 sigma_a: measured from the EXISTING patrol physics, not assumed

The obstacle's own acceleration statistics are nothing like the red's.
Measured directly from `_move_obstacles` (bounce-off-walls patrol, already
in the codebase, `moving_obstacle_fraction` / `obstacle_speed`, typically 0
in existing configs):

| regime | fraction of steps | \|a\| |
|---|---|---|
| between bounces | 99.1% | exactly 0 |
| at a bounce | 0.9% | exactly `2 * obstacle_speed` |

This is a discrete regime change (a velocity-sign flip), not a small
continuous perturbation — nothing like the red's continuously-manoeuvring,
always-max-magnitude acceleration that motivated DWNA in the first place.
A NEES sweep (oracle association, isolating the filter):

| sigma_a | NEES (static) | NEES (patrolling) |
|---|---|---|
| 0.005 | 3.02 | **13,305,336** |
| 0.02 | 2.87 | 52.11 |
| 0.05 | 2.87 | 11.43 |
| **0.1** | 2.91 | **4.96** |
| 0.3 | 3.03 | 3.17 |

`sigma_a = 0.1` calibrates the patrolling case well (NEES 4.96 vs a target
of 5.0) at negligible cost to the static one (2.91 vs 2.87) — adopted as
the default. Two things worth recording from this sweep:

1. **Static obstacles are INSENSITIVE to sigma_a** across two orders of
   magnitude (NEES flat ~2.9-3.0). Position/radius converge from
   measurement fusion regardless; sigma_a mainly matters once there is
   real motion to get wrong.
2. **NEES ~3, not ~5, even at good calibration** — a measurement artefact,
   not miscalibration: under the default exact-radius sensor model, the
   FIRST sighting already equals the truth exactly (§9.2), so the radius
   dimension contributes ~0 to NEES forever (zero residual against a
   correctly near-zero `P_r`) rather than its "fair share" of ~1. The
   effective target under default settings is closer to 4, not 5.
3. **Too-small sigma_a is not merely "cautious" here — it is
   catastrophic.** At sigma_a=0.005 a single bounce, hit with a P the
   filter believes is near-certain, drives NEES into the millions. This is
   sharper than the red's analogous finding (§4): the red's sigma_a choice
   only affects calibration quality; here an under-estimated sigma_a can
   blow up numerically the moment the ONE real discontinuity in this
   motion model (a bounce) actually occurs.

### 9.4 The un-modelled bounce: measured cost, not just a caveat

A discrete velocity-sign flip is not something ANY continuous-noise KF can
predict through — it is corrected only by subsequent measurements
disagreeing with a temporarily-wrong prediction. Measured (oracle
association, `sigma_a=0.1`, position error vs. steps since the most recent
bounce):

| steps since bounce | 0 | +1 | +2 | +3 | +4 | +5 | +6 | +7 |
|---|---|---|---|---|---|---|---|---|
| mean position error (m) | 0.54 | 0.70 | **0.88** | 0.85 | 0.77 | 0.68 | 0.48 | 0.47 |

A modest, bounded, self-healing bump (+0.3-0.4 m peaking ~2 steps after the
bounce, recovering below baseline by step 6-7) — not a catastrophic loss of
track. The `motion_model` plug point (carried here for interface symmetry
with the red tracker, unused today) is the natural place to close this gap
later — e.g. a model that predicts a bounce when a wall is within the next
step's reach — but the measured cost does not currently justify building
it; same "measure before building" discipline as everywhere else in this
document.

### 9.5 A real advantage single-instant fusion did not have

`_fuse_radial_velocity`'s one-shot WLS needs >= 2 non-collinear blues
SIMULTANEOUSLY to recover full 2-D velocity. A PERSISTENT filter does
better for an obstacle specifically: a single blue that circles it (or
just flies past) over several steps sweeps out DIFFERENT line-of-sight
angles over TIME, and the Kalman recursion accumulates each radial
equation exactly as the momentary multi-blue case would, one at a time.
Verified (`tests/test_obstacle_tracker.py`
`test_single_blue_recovers_velocity_by_circling_over_time`): one blue due
south of a moving obstacle (LOS ~ perpendicular to its motion, Doppler
sees almost none of it) then the SAME blue observed from due east 10 steps
later (LOS ~ aligned) converges the full velocity, matching a scenario no
single instantaneous fusion could resolve.

The reverse also holds and is a genuine, not merely theoretical,
limitation: a LONE observer whose LOS stays perpendicular to the true
motion for many consecutive steps has no Doppler correction on the other
axis, so process noise alone can random-walk that component far enough to
push the predicted position outside the gate — measured directly
(`test_single_blue_static_obstacle_can_drift_off_gate`): a static
obstacle's position, tracked from a single blue due south of it, drifts
from 65 m to 62 m over 3 steps on `vx` noise alone and fragments the
track; a second, non-collinear blue prevents it. Identical failure mode,
and identical fix, to the red tracker's
`test_single_blue_crossing_geometry_is_unobservable` (§ tracker.py).

### 9.6 At the training configuration: drift, duplicates, and the static model

Measured with real association at the Stage 4 training geometry (L=200,
7 random blues, 9 obstacles, all static as in every stage4 config to date),
radius noise 2 m, 3-of-4 confirmation with deadline, and a "never forget"
policy for confirmed obstacles (they do not move, so there is no reason to
delete one). §9.3's insensitivity to `sigma_a` was measured with ORACLE
association and does not cover this regime.

With the default `sigma_a = 0.1` the tracker held up to 13–14 confirmed
tracks for 9 obstacles, with a centre error p90 of 11 m. Following the
duplicate events (`scratch/debug_obstacle_dups.py`) shows two separate
mechanisms.

**Drift.** An obstacle track's state includes a velocity. Measurements are
noisy, so the estimated velocity is never exactly 0 — it reached 1.7 m/s
for obstacles that do not move. While nobody is looking, the filter
predicts `position + velocity · dt` every step, so the track slides away.
On top of that, with `sigma_a = 0.1` its uncertainty keeps growing. Because
confirmed obstacles are never forgotten, the drifting track stays alive
(centre error p90 69 m, position sd p90 29 m, 51 scans unobserved at p90).
When a blue sees the obstacle again, the return falls outside the drifted
track's gate, and a second track is born.

**Splits.** The 99% gate rejects about 1% of genuine returns by
construction. A rejected return opens a tentative track next to the good
one, and because several blues often see the same obstacle at once, both
tracks keep receiving returns and the tentative confirms. (Likely
mechanism, consistent with the diagnostic — new duplicates sit a median
1.7 m from the centre — but not traced return by return.)

Variants on identical detections (`scratch/obstacle_variants.py`,
10 episodes; "misleading" = a confirmed track more than 8 m from the centre
of its obstacle):

| variant | confirmed mean / p99 / max | duplicates mean / max | misleading / step | centre err p50 / p90 / max | radius err p90 |
|---|---|---|---|---|---|
| current (`sigma_a` 0.1) | 7.78 / 13 / 14 | 1.46 / 5 | 1.04 | 1.19 / 11.2 / 373 m | 2.19 m |
| static (`sigma_a` 0, `vel_prior_std` 0) | 7.31 / 10 / 11 | 0.34 / 2 | **0** | 0.21 / 0.96 / 4.1 m | 0.84 m |
| **static + merge** | 7.14 / 10 / 10 | **0.18 / 1** | **0** | 0.20 / 0.75 / 3.6 m | 0.84 m |
| current + merge | 6.14 / 10 / 11 | 0.20 / 2 | 0.27 | 0.86 / 4.19 / 186 m | 1.17 m |

**The static model** (`sigma_a = 0`, `vel_prior_std = 0`) is configuration
only, no code. With no process noise the covariance does not grow, and with
the velocity prior at zero the filter pins the velocity at 0 with a tiny
variance, so no update can move it. No velocity, no drift: a track nobody
observes keeps exactly its estimate and its covariance.

**The duplicate merge** (`merge_chi2`, `tests/test_obstacle_merge.py`)
compares pairs of tracks on position and radius with a 3-degree-of-freedom
χ² test, and merges the pairs that are statistically the same obstacle
(the survivor is the confirmed / older track, keeping the more certain
estimate). Two real obstacles can never pass it, because the env places
them without overlap: their centres are at least `r_a + r_b + 1 m` apart,
i.e. 11 m or more. It halves the splits that remain with the static model.

With both, the maximum is 10 confirmed tracks for 9 obstacles; the actor
observation uses **K = 12** obstacle slots (`n_obstacles + 3`) for margin.

#### What changes with moving obstacles

The env already switches automatically: with `moving_obstacle_fraction` and
`obstacle_speed` both above zero, `PursuitEnv` builds the obstacle tracker
with `sigma_a = 0.1` instead of the static model. It has to — with the
velocity pinned at 0 the filter cannot represent motion at all and would
lose every moving obstacle.

But then the drift comes back, and it is measured: with `sigma_a = 0.1`
plus the merge (last row) there are still 0.27 misleading tracks per step
and the worst centre error reaches 186 m. Two things are needed before
training with moving obstacles (neither is built or measured yet):

1. **A drifted track must be able to die.** Today it never does. The
   natural fix is the logic the red tracker already uses (§6.1): count a
   miss only where the obstacle should have been seen, with a finite
   budget. If the track has drifted to a place the blues are looking at and
   nothing is there, it dies.
2. **The filter must tell a still obstacle from a patrolling one.** With a
   mixed fraction the same assumption cannot hold for every obstacle. The
   standard answer is an IMM (Interacting Multiple Model) filter: each
   track runs a static model and a constant-velocity model in parallel, and
   weighs which one explains the measurements better.

One more risk to note (reasoning, not measured): the merge is safe today
because, with the static model, covariances are small and the χ² test is
strict. With `sigma_a = 0.1` and very uncertain tracks the test becomes
permissive, and two different real obstacles could end up merged. The
merge would then need a physical distance bound on top of the statistical
test.

Cost: 2.8 ms per step in a quiet run before the gating was batched; see
docs/tracker_observation.md §5 for the current per-step cost.

## 10. Red-motion dataset collector

**Implementation, 2026-09.** `scripts/collect_red_motion_dataset.py`
collects the one-step transitions `(s_t, c_t) -> a_t` the learned motion
model (§8) will be trained on. Full design reasoning lives in the script's
own module docstring (kept there, not duplicated here, since it is what an
implementer reads first); summarised:

**Red policies — per-red domain randomisation, not one fixed adversary.**
"The red" is a proxy for an unknown real evader, so each red gets an
independently sampled `StochasticRed` configuration
(`MixedStochasticRed`, since one `StochasticRed` instance applies the same
parameters to every red it drives): 15% fully deterministic (every knob
off, a clean baseline + regression check), the rest sampled from ranges
CENTRED ON §7's already-validated values (not picked arbitrarily), with
per-episode jitter so the dataset does not key on one exact repeated
value. Verified exact (not an approximation): `run_from_nearest_uav` never
reads another red's position or active flag, so driving each
`StochasticRed` sub-instance with a single-red slice is bit-identical to
the batched call restricted to that index.

**Blue policies — scripted, no trained checkpoint needed.** `GreedyPursuer`
(already in the codebase, no training) mixed per-blue with `RandomAgent`
(independent per blue, per-episode `p_greedy`), plus team-size variation
(`n_blue`/`n_red`/`n_obstacles` sampled per episode). The dataset only
needs a good SPREAD of approach geometries, not the exact deployed blue
policy — flagged for later, not a blocker now: once the paused from-scratch
checkpoint exists, check whether trained-blue rollouts induce a
meaningfully different geometry distribution (cheap to check by comparing
held-out prediction error), rather than assuming either way.

**Getting a_t without corrupting stateful red policies.** `red_policy` is
a pure function from the tracker's perspective, but `StochasticRed` is
NOT stateless — it advances an AR(1) phase and commitment counters on
every call. Re-invoking it a second time just to log its own output would
double-advance that state (and backing the acceleration out from the
velocity DELTA instead is biased wherever v clips at `v_max` — exactly the
aggressive-manoeuvre samples that matter most). Fixed with a one-line,
additive hook: `PursuitEnv.step()` now stores `self._last_red_action`
right after computing it — the same pattern already used for
`_last_red_detect`.

**Schema and scope.** Ego-centric (relative to the red), arena/speed-
normalised context — blue and obstacle relative position/velocity (+
obstacle radius), wall distances — matching `_build_obs`'s existing
convention (`wall_distances = (x, L-x, y, L-y) / L`). Fixed-capacity
padding + boolean masks for the variable obstacle count (mirrors the env's
own `_red_active`-style convention); blues need no mask since their count
is fixed within an episode, only varied across episodes. `.npz` shards, no
new dependency. Training uses PRIVILEGED (ground-truth) state throughout
— matching §8.1's CTDE-style argument that the belief-derived,
uncertainty-marginalised context is an INFERENCE-time concern, not a
data-collection one. Explicitly deferred to training time, not the
collector's job: input augmentation (perturbing `s_t`/`c_t` by belief-
scale noise) and discretising `a_t` into the (heading, magnitude) grid
(§8.4).

**Verified**: `MixedStochasticRed`'s per-slot slicing is bit-identical to
a direct single-red call; different reds in one episode get genuinely
different parameters; padding/masks are exact; `accel` matches
`_last_red_action` (not a re-derived or re-invoked value); same-seed runs
are reproducible. Throughput: ~1600 samples/s, ~7.7 episodes/s on CPU (a
150-step episode with ~3 active reds yields ~450 samples) — a
million-sample dataset costs on the order of ten minutes, no GPU.

### 10.1 Collecting for the TRAINING distribution (`--red-mix training`)

v1–v3 were trained on `MixedStochasticRed` at L=130 with up to 6 blues and
5 obstacles. The Stage 4 trainer runs something else entirely: L=200, 7
blues, 9 obstacles, and a red policy drawn per episode from
**stationary / random / run** (`--red-policy-mix`, uniform). A model for
that distribution has to be collected in it — §6.3 measured v3 running out
of its own distribution.

`--red-mix training` reproduces the trainer's mix exactly: one policy per
episode for every red, sampled uniformly from the three, the same way
`Stage4VectorPursuitEnv._resample_red_policy` does. Nothing in the state
says WHICH of the three a red is following, so the model has to learn the
mixture — a still red, a random walk and a flee response all look like
plausible continuations of the same position and velocity, weighted by how
often each occurs and by whatever the geometry hints at.

Two more collection changes, both for parallel runs into one directory:
`--episode-offset` (the train/val split is BY episode, so ids must stay
distinct across processes) and `--shard-prefix`. The collector also no
longer maintains a belief map — nothing in the labels or in any policy here
reads it, and it was most of the env's per-step cost at L=200.

Throughput at the training geometry: ~400 samples and ~3 s per 300-step
episode, so a ~1.5 M-sample dataset is about 45 minutes across 4 processes.

## 11. The learned motion model, measured downstream

The model from §10 was trained, wired into the `motion_model` plug point
(`isr/agents/learned_red_motion.py`) and measured. It works — but three
bugs in the adapter had to be found first, and the middle two of them
produced a confident, thoroughly documented negative result that was
entirely an artefact of the measuring apparatus. That sequence is recorded
here because the failure mode is the interesting part: every one of the
three was invisible in a single prediction step and ruinous over a coast.

### 11.1 The headline

STOCHASTIC red, 6 episodes × 120 steps, `max_misses=80`, 20 m match gate.
Same episodes for every row, so this is attribution rather than a
comparison across runs.

| configuration | MOTA | MOTP | IDF1 | recall | FP | MT |
|---|---|---|---|---|---|---|
| KF + real assoc (constant velocity) | 0.19 | 3.13 | 0.45 | 0.41 | 460 | 0.17 |
| **LEARNED + real assoc** | **0.52** | 3.07 | **0.61** | **0.58** | **96** | **0.44** |
| LEARNED + ORACLE assoc | 0.60 | 3.16 | 0.66 | 0.61 | 1 | 0.50 |

Better on every column. MOTA nearly triples, and false positives drop
**5×** — because a coasted track that stays near the truth is a match,
while one that drifts away is counted twice over, as both a miss and a
false positive.

Recall by match gate, at three coast budgets:

| motion model | coast | 5 m | 10 m | 20 m | 40 m |
|---|---|---|---|---|---|
| constant velocity | 5 | 0.328 | 0.331 | 0.331 | 0.331 |
| TRUE policy | 5 | 0.330 | 0.331 | 0.331 | 0.331 |
| LEARNED | 5 | 0.331 | 0.331 | 0.331 | 0.331 |
| constant velocity | 20 | 0.345 | 0.373 | 0.390 | 0.400 |
| LEARNED | 20 | 0.383 | 0.398 | 0.401 | 0.401 |
| constant velocity | 80 | 0.346 | 0.380 | 0.441 | 0.509 |
| TRUE policy | 80 | 0.481 | 0.584 | 0.597 | 0.597 |
| **LEARNED** | 80 | **0.509** | 0.568 | **0.609** | 0.609 |

Two factors, which play different roles:

* **The coast budget is necessary.** At `max_misses=5` every motion model
  scores 0.331, including a perfect one. Measured invisibility gaps last a
  median **74 steps** (deterministic) / **88** (stochastic), and only
  14% / 0% are within 5 steps — so a 5-step budget can cover at most a few
  percent of a gap no matter how good the prediction is. This is a
  CONFIGURATION change, and without it nothing else registers.
* **The match gate changes what is being measured, not whether the model
  helps.** The learned model gains at the tight 5 m gate too (0.346 →
  0.509). 5 m is a MOT evaluation convention; what a blue needs is closer
  to "put the target inside my sensor disk" (40 m radius). Relaxing the
  gate lifts constant velocity as well (0.346 → 0.509 at 40 m).

The learned model reaches, and at the tight gate slightly exceeds, the
TRUE-policy oracle (0.509 vs 0.481 at 5 m, one 6-episode run). A possible
reason, not verified: the oracle queries the policy at the red's TRUE
position but applies the answer to a component that may already have
drifted, whereas the learned model queries its own estimated state.

Drift while coasting, stochastic red, `max_misses=80`:

| motion model | 6–20 misses | 21–50 | 51–80 |
|---|---|---|---|
| constant velocity | 8.8 m | 22.9 m | 53.3 m |
| TRUE policy | 3.2 m | 5.2 m | 4.2 m |
| LEARNED | 2.8 m | 4.2 m | 5.0 m |

The error SATURATES rather than growing. This is worth stating because an
earlier argument here was that bridging an N-step gap within tolerance D
requires velocity accuracy better than `D/N` — 0.068 m/s for N = 74,
against the tracker's 0.35–0.40 — and therefore that no model could
bridge. The measured drift contradicts that argument. The likely reason
is that it assumed the error integrates from a fixed initial velocity
error, whereas a motion model that tracks the policy keeps pulling the
velocity back toward the right one. Caveat on the TRUE-policy row: the
oracle reads the red's true position every step, so its bounded drift is
partly because it is fed truth; the LEARNED row has no such access and
still stays around 5 m.

### 11.2 Three adapter bugs, and why each was invisible in one step

**(a) Top-k cells are not modes.** Training uses SOFT LABELS that smear
mass onto adjacent bins, so the k most probable cells tend to be one mode
sampled k times. And with 181 classes and a ~7% peak, truncating to the
top k (with `max_branches` binding before the 0.9 mass threshold) discards
most of the distribution — an estimate, the top-k mass was never measured
directly. Branch covariances then described a narrower belief than the
model predicted: NEES 6.6 against a target of 4.0.

Fixed by splitting the categorical into BASINS around local maxima of the
heading marginal and assigning every heading bin to its nearest kept peak.
Nothing is discarded; `max_branches` now controls how finely modes are
resolved, not how much of the distribution survives. Measured, with
everything else unchanged: NEES 6.6 → 5.0. Heading defines the modes
because the multimodality seen for this adversary is a split in direction
(nearest-blue ties, §11.4; commitment side); magnitude is ordinal, so
splitting on it would manufacture near-duplicates.

*Correction:* an earlier version of this paragraph blamed top-k for the
**1.01 components per track** measured at the time. That was wrong — with
basins it was still 1.02. The collapse has a different cause, found later
(§11.4): branches are only ~0.3 m apart after one step, inside each
component's own spread, so they are merged whatever the branching rule.

**(b) The wrong integration gain.** `PursuitEnv._integrate` advances the
position with the NEW velocity:

```
v' = clip(v + a*dt, -v_max, v_max)      p' = p + v'*dt
```

so `p' = p + v*dt + a*dt^2`. The adapter used the textbook `dt^2/2`,
predicting half a metre short per step at full acceleration — nothing in
one step, cumulative over a coast.

**(c) No velocity cap.** The env clips red speed AXIS-WISE at
`RED_TARGET.v_max = 1.0`. The adapter did not, so a unit acceleration
applied every step of an 80-step coast reached 80 m/s and put the estimate
**over a thousand metres outside a 130 m arena** (measured: 1172 m median
drift at 51–80 misses). Constant velocity never accelerates and so was
never affected, which is exactly why it appeared to win.

(b) and (c) were in the evaluation oracle too, which is how they produced
a *coherent* false story: a "perfect" motion model that lost to constant
velocity, and a plausible-sounding arithmetic explanation for why. The
tell was in the diagnostics rather than the conclusion — a drift of 1172 m
in a 130 m arena is not a model degrading, it is divergence.

`test_prediction_matches_the_env_integration_exactly` and
`test_velocity_never_exceeds_the_red_speed_cap` now pin both against the
env's own formula.

### 11.3 Model quality, isolated from input quality

Querying the network three ways and scoring against `env._last_red_action`
(deterministic red, where the noise floor is exactly 0° — no randomness
excuses anything):

| query state | median | mean | p90 |
|---|---|---|---|
| TRUTH (exact position and velocity) | 8.0° | 26.1° | 100.3° |
| exact velocity, tracker position | 9.6° | 25.8° | 76.4° |
| tracker estimate (what really happens) | 14.5° | 32.4° | 113.7° |

The tracker's state costs +6.5°, of which ~4.9° comes from VELOCITY error
(0.35–0.40 m/s) and only ~1.6° from position (measured with model v2).
The red policy itself does not read velocity, so whatever the network
extracts from it is presumably a trace of the adversary's hidden state
(AR(1) phase, commitment) — plausible, not verified.

**And the model supplies that itself.** A track is born with velocity from
DOPPLER fusion — weighted least squares on the radial components, so with
≥2 non-collinear lines of sight the velocity is determined at birth
(measured median error 0.27 m/s from one return, 0.14 from two, 0.08 from
three or more). Tracking that error by track age:

| track age | velocity error, CV | velocity error, LEARNED |
|---|---|---|
| 0 (birth) | 0.20 | 0.22 |
| 1–2 | 0.46 | **0.31** |
| 3–5 | 0.47 | **0.24** |
| 6–15 | 0.50 | **0.27** |
| > 15 | 0.48 | **0.39** |

Under constant velocity the error gets WORSE with age — 0.20 at birth to
~0.48 — which a correctly specified filter does not do. It is model
misspecification, not convergence: the red manoeuvres every step and CV
converges to a biased velocity. The learned model holds the estimate near
its birth value instead.

So there is a loop here: velocity error is the largest part of the
model's own input penalty, and the model roughly halves it. Better model →
better velocity estimate → better input → better prediction.

The heavy tail is real and not explained by inputs: **p90 = 100° with
perfect inputs on a perfectly predictable target**.

Training 3× longer (120 epochs, val loss 3.928 → 3.736, converged — train
3.62 vs val 3.75, plateaued from epoch ~104) sharpened the median 8.0° →
7.1° and made the p90 **worse**, 100° → 121°. So the tail is not simply
underfitting: on those states the model is confidently wrong, and more
training made it more so. Downstream, v3 buys tighter localisation (MOTP
3.07 → 2.65) and no extra recall.

### 11.4 The tail is the policy's discontinuity, and the mixture already holds the answer

`run_from_nearest_uav` is DISCONTINUOUS in the identity of the nearest
blue: near a tie, an arbitrarily small displacement flips the flee
direction by a large angle. Bucketing the error by the distance ratio of
the two nearest blues (`d2/d1`, so 1.0 is a perfect tie):

| d2/d1 | n | top mode median | p90 | **best branch** median | p90 |
|---|---|---|---|---|---|
| 1.00–1.05 | 1411 | 19.4° | **134.4°** | 11.4° | **37.4°** |
| 1.05–1.15 | 1104 | 16.1° | 79.3° | 13.1° | 42.2° |
| 1.15–1.35 | 893 | 12.2° | 71.5° | 11.3° | 54.1° |
| 1.35–2.00 | 702 | 16.2° | 89.3° | 14.6° | 67.9° |
| > 2.00 | 210 | 8.4° | 30.8° | 8.0° | 25.7° |
| ALL | 4320 | 15.7° | 93.6° | 11.9° | 46.1° |

(Measured with model v3 at TRUE positions, stochastic and deterministic
reds pooled.) The tail concentrates at near-ties, and that bucket is a
THIRD of all samples. What this does and does not show:

* It is **hard to learn, not impossible**. For a deterministic red the
  nearest blue is a deterministic function of exact positions, so the
  information *is* in the state; what makes it hard is that the answer
  jumps discontinuously across the tie, which a smooth network fits
  poorly. *(An earlier version said "the information is not in the state"
  and "cannot be trained away" — both wrong for exact inputs; the
  120-epoch run shows only that more of the same training did not help.)*
  Where it genuinely becomes ambiguous is at inference: with ~1 m of
  tracker position error, a near-tie can fall either way — epistemic
  uncertainty in the §8.3 sense.
* The predicted categorical **contains** the right direction: at a
  near-tie the best branch is p90 37° against the top mode's 134°. That is
  the kind of multimodality the joint categorical can represent — but, as
  the rest of this section shows, the tracker's mixture does not currently
  exploit it.

Which raises the question of why the mixture is not visibly doing this
work:

```
components per track: mean 1.01   p90 1   max 2
```

The branches collapse to one. The obvious suspect was `merge_gate = 4.0`,
which the tracker's own docstring flags as "a placeholder default, not yet
tuned against a real branching model". **It is not the cause**, and no
value of it would be:

| coast age (misses) | branch separation | branch sd | d² |
|---|---|---|---|
| 0 (just updated) | 0.34 m | 1.01 m | 0.441 |
| 1–5 | 0.28 m | 4.60 m | 0.038 |
| 6–20 | 0.19 m | 47.25 m | 0.005 |
| 21–80 | 0.23 m | 305.72 m | 0.002 |

| d2/d1 (tie ratio) | branch separation | branch sd | d² | top-2 heading sep |
|---|---|---|---|---|
| 1.00–1.05 | 0.31 m | 8.34 m | 0.018 | **48°** |
| 1.05–1.15 | 0.25 m | 15.21 m | 0.013 | 39° |
| > 1.50 | 0.23 m | 215.58 m | 0.004 | 49° |

At a near-tie the branches genuinely point 48° apart — the model produces
the multimodality. But ONE STEP of that acceleration difference separates
the resulting states by only **0.31 m**, against a track sd of 1 m in the
very best case (freshly updated) and 305 m mid-coast. So d² is 0.002–0.44,
already below the gate everywhere. Lowering `sigma_a_model` to 0.15 moves
the p90 of d² from 1.09 to 2.41 — still under 4.

**The multimodality is real in ACCELERATION and invisible in one-step
STATE space.** The modes live in the trajectory, not in the next state:
branches would need many steps to separate enough to be distinguishable,
and `_reduce` runs after every PREDICT, merging them at step 1 before they
can diverge. No gate value fixes that; it is structural to reducing a
Gaussian Sum at a one-step horizon.

The classical remedy would be an **IMM**, where each mode is a LABELLED
filter with a mode-transition matrix and modes are never merged by
proximity — identity persists by construction instead of being
rediscovered by distance.

**Tested directly instead of argued.** `merge_gate < 0` disables merging
outright (a squared Mahalanobis distance is never negative), so the
hypotheses simply persist and diverge across steps. `max_components` then
binds, since `_predict_track` calls the motion model once per EXISTING
component and each call returns up to `max_branches`, so the count
multiplies ×4 every step (1 → 4 → 16 → 64 …) — which is why `_reduce` is
mandatory rather than optional.

| merge | max_comp | MOTA | MOTP | IDF1 | recall | FP | IDSW | comps |
|---|---|---|---|---|---|---|---|---|
| — (constant velocity) | — | 0.19 | 3.13 | 0.45 | 0.406 | 460 | 11 | — |
| 4.0 (today) | 8 | 0.52 | 2.65 | 0.60 | 0.580 | 98 | 27 | 1.01 |
| never | 8 | 0.52 | 2.59 | 0.62 | 0.581 | 97 | 24 | 7.69 |
| never | 16 | 0.52 | 2.67 | 0.62 | 0.581 | 97 | 24 | 15.02 |
| never | 32 | 0.52 | 2.84 | 0.61 | 0.581 | 97 | 28 | 28.75 |
| 0.25 | 16 | **0.53** | 2.61 | 0.61 | 0.581 | 97 | **21** | 1.54 |

The hypotheses now genuinely survive — 1.01 → 7.7 → 28.8 components,
saturating the cap — and the metrics do not move: MOTA 0.52 → 0.52, recall
0.580 → 0.581. Only IDF1 (0.60 → 0.62), IDSW (27 → 24) and MOTP (2.65 →
2.59) shift, all marginally.

So, in this setup, **merging was not destroying useful information**:
keeping up to ~30 near-identical Gaussians multiplies the per-step work
and buys essentially nothing, consistent with the branches being
indistinguishable at a one-step horizon. `max_components=32` gave a worse
MOTP (2.84 vs 2.59); one possible reason, not verified, is that the
readout takes the dominant component, which among many near-identical
ones can be a slightly worse branch.

`merge_gate = 0.25` scored marginally best (MOTA 0.53, IDSW 21 at 1.54
components) — in a single 6-episode run, within noise of the others.

An uncomfortable corollary: with 1.01 components per track the mixture has
been essentially **inert**, so the measured gain (MOTA 0.19 → 0.52) is
attributable to the better MEAN prediction rather than to the Gaussian
Sum. The gain therefore does not depend on the branching machinery — which
currently earns nothing measurable.

### 11.5 Tuning: sigma_a_model, and the coast budget

8 episodes × 150 steps, stochastic red, 20 m gate, `merge_gate = 0.25`.

**`sigma_a_model` against NEES** (target 4.0 for a 4-D state), coast 80:

| sigma | NEES | NIS | MOTA | MOTP | IDF1 | recall | FP | IDSW |
|---|---|---|---|---|---|---|---|---|
| **0.35** | 3.25 | 1.24 | **0.56** | 3.05 | 0.65 | 0.626 | **160** | 65 |
| 0.25 | 3.88 | 1.29 | 0.54 | 2.78 | 0.65 | 0.622 | 220 | 82 |
| 0.18 | 4.50 | 1.37 | 0.53 | 2.87 | 0.64 | 0.626 | 266 | 91 |
| 0.12 | 4.91 | 1.43 | 0.53 | 3.00 | 0.64 | 0.634 | 291 | 90 |
| 0.06 | 5.38 | 1.44 | 0.52 | 2.94 | 0.63 | 0.631 | 298 | 86 |
| 0.00 | 5.58 | 1.46 | 0.50 | 2.93 | 0.58 | 0.629 | 371 | 101 |

**This is the first place in this project where NEES and the MOT metrics
disagree.** NEES-closest is 0.25; every MOT column prefers 0.35. NEES 3.25
means slightly CONSERVATIVE — covariance a little larger than the actual
error — which is the safe direction, and it carries no observed cost: FP
is *lowest* there. Chasing NEES = 4.0 exactly costs MOTA and doubles FP.

**Adopted `sigma_a_model = 0.35`** — the placeholder turns out to have
been a good value, now measured rather than assumed.

**Coast budget** at `sigma_a_model = 0.35`:

| coast | model | MOTA | MOTP | IDF1 | recall | FP | FN | IDSW | NEES |
|---|---|---|---|---|---|---|---|---|---|
| 20 | CV | 0.35 | 2.92 | 0.47 | 0.386 | 101 | 2211 | 40 | 3.85 |
| 20 | LEARNED | 0.38 | 1.85 | 0.49 | 0.409 | 48 | 2128 | 39 | 4.64 |
| 40 | CV | 0.29 | 3.16 | 0.47 | 0.399 | 361 | 2162 | 39 | 3.73 |
| 40 | LEARNED | 0.47 | 2.49 | 0.55 | 0.502 | 68 | 1794 | 39 | 4.01 |
| 80 | CV | 0.13 | 3.30 | 0.42 | 0.405 | 931 | 2142 | 43 | 3.57 |
| 80 | LEARNED | **0.56** | 3.05 | 0.65 | 0.626 | 160 | 1348 | 65 | 3.25 |
| 150 | CV | 0.05 | 3.35 | 0.40 | 0.407 | 1235 | 2135 | 47 | 3.59 |
| 150 | LEARNED | **0.60** | 3.17 | 0.68 | **0.689** | 227 | 1119 | 81 | 3.04 |

The two models move in OPPOSITE directions. Constant velocity peaks around
coast 20 and then degrades badly — MOTA 0.35 → 0.05, FP 101 → 1235 — because
a longer budget only keeps drifting tracks alive. The learned model climbs
monotonically, MOTA 0.31 → 0.60 and recall 0.320 → 0.689, with FP an order
of magnitude lower than CV's at the same budget.

At coast 5 the two are identical (0.31 each). At coast 150 it is 0.05
against 0.60. The coast budget is not one parameter among others: it is
the condition that separates a model that can predict from one that
cannot, and raising it is actively HARMFUL without one.

Two honest caveats. Recall 0.689 sits at **2.2× the detectability
ceiling** (0.31), so the model is genuinely bridging gaps rather than
accumulating luck. But `max_misses = 150` with 150-step episodes means a
track never dies — that row is the limiting "never kill anything" case,
not a tuned value, and the cost shows in IDSW (39 → 81).

**Recommended `max_misses = 80`**: it captures most of the gain (MOTA 0.56
of 0.60) with 30% fewer false positives and 20% fewer ID switches, and it
is a real timeout rather than the degenerate case.

One coupling to note: the learned model's NEES FALLS as the coast budget
rises (4.64 at coast 20 → 3.04 at 150), so the NEES-optimal sigma depends
on the budget. At a long budget the current setting is conservative.

### 11.6 sigma_a_model calibrated at the training configuration (frozen)

`sigma_a_model` stands for the learned model's own error, on top of the
within-bin quantisation spread. It had never been tuned — 0.35 was a
placeholder. Calibrated now against NEES
(`scratch/calibrate_sigma_a.py`) at the Stage 4 training geometry, clutter
0.2, obstacle radius noise 2 m, the training red mix, through the ACTOR
path (obstacle context and coverage occlusion from the obstacle tracker),
with the chosen tracker configuration including re-acquisition.

An honest filter's NEES averages the state dimension, 4. ORACLE
association isolates the filter from association errors; the real-
association means are inflated by a handful of mis-associated tracks (9.30
at 0.35) while their medians match the oracle's, which is what "outliers,
not the filter" looks like.

| `sigma_a_model` | NEES mean | NEES median | coasting median | MOTA stat / rand |
|---|---|---|---|---|
| 0.10 | 5.36 | 3.90 | 3.24 | 0.47 / 0.50 |
| 0.15 | 4.68 | 3.47 | 2.78 | 0.49 / 0.53 |
| **0.20** | **4.08** | **3.06** | 2.28 | 0.47 / 0.53 |
| 0.25 | 3.60 | 2.71 | 1.86 | 0.46 / 0.54 |
| 0.35 (old) | 2.97 | 2.24 | 1.30 | 0.47 / 0.53 |
| 0.50 | 2.49 | 1.75 | 0.80 | 0.51 / 0.53 |

(Median target for χ²₄ is 3.36. "Coasting" = steps with no hit, where the
process noise actually acts; it is consistently LOWER than the overall
median, i.e. the coasted covariance is the over-cautious part while the
post-update one is slightly overconfident.)

**0.20 adopted**: mean 4.08, median 3.06. Tracking quality is flat across
the whole sweep (MOTA within 0.05), so this is about honest uncertainty,
not performance — which matters because the actor now consumes that
uncertainty directly.

**Why the 40 m cut-off protects a little less with the calibrated value.**
The actor is not shown red tracks whose position sd is above 40 m
(`--tracker-sigma-cutoff`, docs/tracker_observation.md §2). That cut-off
looks at the uncertainty the tracker DECLARES, not at the error it actually
makes. Lowering `sigma_a_model` barely changes where the tracker thinks the
red is; it changes how much uncertainty it admits to. Measured at both
values (`scratch/obs_k_sigma.py`, medians over labelled tracks):

| scans without a hit | declared σ at 0.35 | declared σ at 0.20 | actual error (either) |
|---|---|---|---|
| 20 | 26 m | 20 m | ~18 m |
| 40 | 68 m | 51 m | ~26–28 m |

With 0.35 the filter exaggerated: a track that had gone a long time without
a detection declared a large σ and the 40 m cut-off removed it. After
calibration, the same track with the same error declares a smaller σ and
passes. The data show it: in the 30–40 m σ band, shown tracks had gone a
median of 25 scans without a hit at 0.35, and 30 scans at 0.20 — older,
and therefore more wrong, tracks carrying the same σ label.

In other words, with the inflated σ the 40 m cut-off was in practice a
stricter one. Now σ means what it says, and 40 cuts where it says. Share of
shown track-steps whose red is within 40 m of the readout:

| σ band | 0.35 | 0.20 | 0.15 |
|---|---|---|---|
| ≤ 26 m | 1.00 | 0.99 | 0.98 |
| 26–30 m | 1.00 | 0.97 | 0.94 |
| 30–40 m | 1.00 | 0.91 | 0.87 |

Misleading tracks with σ ≤ 10 m stay at 0.00 per step and the slot counts
do not change (p99 4–5, max 6), so 0.20 keeps the observation's guarantees
while 0.15 starts to erode them. The old protection can be had back by
lowering the flag (e.g. to 30 m); the cut-off stays at 40 because lowering
it to 30 removes only 0.07 nodes per step on average, and those nodes are
right 91% of the time.

**CV's own `sigma_a` is a separate, open question.** The same sweep says
`a_max·√2 = 1.41` is over-cautious under the TRAINING red mix (NEES median
1.05, coasting 0.14) and that smaller values track better there (MOTA 0.62
vs 0.45 with stationary reds at σ_a 0.35). That value was calibrated
against a stochastic evader fleeing at full acceleration (§4), which two of
the three training policies are not. It is left unchanged: lowering it
would help stationary/random reds and could hurt fleeing ones, and with
random blues the fleeing case is seen too rarely (recall ≤ 0.15) to
measure.

### 11.7 What is still open

* **The mixture earns nothing, and that is now settled** (§11.4).
  Disabling merging entirely lets the hypotheses persist (1.01 → 28.8
  components) and changes recall by 0.001. The branches are not
  distinguishable at a one-step horizon, so neither `merge_gate` tuning
  nor an IMM has a measured case behind it. The measured gain is
  attributable to the mean prediction. `merge_gate = 0.25` was marginally
  best in one 6-episode run, within noise.
* ~~`sigma_a_model`~~ and ~~`max_misses`~~ — measured in §11.5. Adopted
  0.35 and 80. Neither is applied as a DEFAULT yet: `max_misses` is a
  tracker-wide default that the belief-map and policy paths also consume,
  so changing it is a production decision, not a diagnostics one.
* **Velocity estimation** is worth attention: it contributes ~4.9° of the
  +6.5° input penalty (§11.3), even though the red policy itself does not
  read velocity.
* More of the same training is not indicated: the model converged (§11.3).
  Whether more capacity or a representation better suited to the
  nearest-blue discontinuity would reduce the tail was not tested (§11.4).
* Obstacle geometry is ground truth in these rows; deployment reads the
  obstacle tracker.
* **The remaining budget is SEARCH**, and it is a separate line of work:
  see `docs/search_design.md`. Measured there: 57% of the "lost" mass is
  not lost at all — a track holds it to within 3.1 m and no blue goes
  there — and only 43% needs genuine search.

## Reproduce

```
python scripts/eval_tracking.py --episodes 8 --steps 150
python scripts/eval_tracking.py --learned runs/red_motion/model_v2.pt \
    --stochastic-red --max-misses 20
```

Scratch scripts (`scratch/`, not committed): `diag_gap.py` (§2),
`sweep_q.py` (§3), `autocorr.py` (§4), `clutter_impact.py` (§6),
`check_stoch_red.py` (§7), `mode_count.py` (§8.3),
`sweep_obstacle_sigma_a.py` (§9.3, and the bounce-cost measurement in
§9.4), `noise_floor.py` (§10), `input_quality.py` (§11.2),
`verify_ceiling.py` and `ceiling_x_coast.py` (§11.3). NEES/NIS are in the
eval output.
