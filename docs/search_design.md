# Search design (feature/target-tracking)

Where the recall budget lives, and the representation being built for it.
Continues from `tracking_diagnostics.md` §11, which ends with the
motion-model work cashed in and the remaining budget being SEARCH.

Throughout, **measured** means a number from a script named alongside it;
anything else is labelled as reasoning or as not established. Several
earlier claims in this document were wrong and have been corrected; the
record of what was falsified is kept in §5.4.

## 1. "The target is not on radar" is two different problems

Measured over 6 episodes × 120 steps, stochastic red, **random blue
actions**, with the tracker and the learned motion model running
(`scratch/search_budget.py`):

| | count | share | of unseen |
|---|---|---|---|
| on radar now | 670 | 31.0% | — |
| **NOT on radar** | 1490 | **69.0%** | — |
| RE-ACQUISITION: a confirmed track within 20 m of it | 849 | 39.3% | **57.0%** |
| DISCOVERY: seen before, no track within 20 m | 195 | 9.0% | 13.1% |
| DISCOVERY: never detected | 446 | 20.6% | 29.9% |

When a track holds an unseen target it sits **3.1 m** from truth (median,
p90 12.9 m); the nearest blue to an unseen target is **73 m** away (median,
p90 103 m) against a 40 m sensor radius.

So for 57% of the unseen mass the system already knows roughly where the
target is — what is missing is that knowledge reaching the policy (the
tracker wired into the observation), not discovery. The other 43% has no
track, and the tracker structurally cannot help with it. That is what the
coverage representation below is for. These shares were measured with
random blues and will differ under a real policy.

## 2. Two questions, two representations

| question | answered by | shape |
|---|---|---|
| where is *this target*? | tracker / obstacle tracker | a few objects with identity |
| where have I *not looked*? | staleness field | a dense field |

Today the belief map is asked to do both. As the policy's source for
unseen targets it measured MOTP 6.64 with 829 false positives, against
2.65 and 98 for the tracker with the learned motion model (20 m gate,
`max_misses=80`; `tracking_diagnostics.md` §11.1).

The split is *intended* to avoid double-counting — a target a track
already explains should not also light up a region as "something here".
That only holds once the tracker is actually wired in (stage D, not built).

## 3. The staleness field

**Steps since each cell was last observed.** One integer per cell, capped
at `max_steps`, no decay factor, no diffusion kernel, no log-odds clip.

It is a **proxy**, not a derived optimum. The intuition: a target that can
move into a cell is more likely to be there the longer since anyone
looked. This red FLEES blues rather than wandering, so the relation need
not be monotone. The measured support is limited to one scripted-policy
test (§5.3): under a sweep, hidden reds sat in never-observed cells.

What it does separate cleanly, by construction: the belief map's log-odds
decay toward zero, so "never looked here" becomes indistinguishable from
"looked long ago and forgot". Staleness keeps them apart.

### Resets on LOOKING, not on FINDING

`p_TP = 0.85`, so a look can miss a real target. Resetting only where
something was detected would turn the field back into a belief map, and an
empty swept region would stay maximally stale and be swept again.

### Occlusion is exact

A cell inside the sensor disk but behind an obstacle was **not** looked at
and keeps ageing. The field uses the same analytic segment-disk test
(`_rays_occluded_by_obstacles`) as the belief update.

`_observed_cells_mask` duplicates that geometry rather than being
refactored out of `_update_belief_maps`: that per-blue loop interleaves the
occlusion test with RNG draws, and restructuring it risks reordering the
random stream (not verified either way). The duplication is pinned by
`test_observed_mask_matches_the_cells_the_belief_update_touches`.

### Known limitation, not measured

A single look resets to zero although `p_TP < 1` makes it inconclusive.
Plausibly minor — invisibility gaps run ~70+ steps, and three independent
looks would leave 0.15³ = 0.3% — but never quantified. If it matters,
count CONSECUTIVE CLEAN LOOKS instead: still one integer per cell.

## 4. Region nodes

The 26 × 26 field does not enter the graph raw. It is aggregated to R × R
**region nodes**, a new typed node — the same extension pattern Stage 4
used to add obstacles.

**Resolution.** With a 40 m sensor radius in a 130 m arena, R = 5 gives
26 m regions — small enough that a blue at a region's centre sweeps all
of it (half-diagonal 18 m < 40 m). The same criterion is satisfied by any
region up to ~56 m, i.e. R ≥ 3, so **R = 5 is a choice within that range,
not a derived value**. (An earlier version claimed 26 m was "about one
footprint across"; the footprint is 80 m across, so that was wrong.)

Why not the raw grid: 676 cell-nodes against 5 blues is 3,380 edges,
against 55 in the whole current graph (5 blues, 3 reds, 4 obstacles).

26 has no divisor near 5, so regions are split as evenly as the grid
allows — 5, 5, 5, 5, 6. Both features are intensive (a mean and a
fraction), so the uneven region introduces no bias, and each centre is
computed from its own cells.

### Features

| feature | meaning |
|---|---|
| `staleness` | mean steps since observed over the region's SEARCHABLE cells, ÷ `max_steps` → [0, 1] |
| `searchable` | fraction of cells outside every obstacle |

Obstacle interiors are excluded from the mean because they cannot hide a
target. A fully blocked region reports staleness **0** (nothing to find
there) and `searchable` 0 — also the guard against a NaN from averaging an
empty set.

## 5. Aggregation, and whether coverage helps at all

### 5.1 What is implemented

Region → blue messages are multiplied by per-edge weights
`staleness × searchable`, normalised to sum to 1 over regions, then added
into the blue's aggregate with the other edge types. The weights ride the
same per-edge multiply slot the visibility masks use, so the aggregation
mechanism is unchanged, and uniform weights recover a plain mean.

Why normalise at all: there are K = R² region edges per blue against 11
other edges (4 bb + 3 rb + 4 ob), so an unnormalised sum would grow with
K and make R a scale knob on this pathway.

Two consequences that were **not deliberate**, both measured or read off
the code:

* **Cross-type balance.** Normalised weights make the coverage term a
  convex combination (about one message's worth) while bb/rb/ob are raw
  sums of 3–4 messages. At initialisation coverage is only **~9%** of the
  aggregate's norm (`scratch/type_balance.py`).
* **No per-blue conditioning.** Weights depend only on the region, so
  every blue receives the same mixture, differing only through message
  geometry.

The staleness weight is a deterministic function of the region's own two
features. It adds **no information**; it can only change how easily the
network uses information already present.

### 5.2 What the weighting changes, at initialisation

With equal weights the aggregate is, to first order,
`A·mean(h_region) + B·mean(e_gb)`: it depends on the staleness field only
through its mean, and "which region is stale" can enter only via the MLP's
non-linearity. Measured with an **untrained** encoder
(`scratch/region_agg_mechanism.py`), relative change in the aggregate:

| probe | what changes | uniform | weighted |
|---|---|---|---|
| PERMUTE | which regions are stale (mean held) | 0.0059 | 0.2131 |
| SHIFT | the overall level | 0.1767 | 0.1555 |

The uniform column is the informative one: a uniform mean barely responds
to the arrangement of staleness at initialisation. The weighted column's
larger PERMUTE response **follows by construction** — the weights are the
staleness — and is not evidence that weighting is useful.

### 5.3 Does coverage help find reds? Not established

Decodability probes: ridge regression predicting the unit direction from a
blue to the nearest hidden red, split by episode, held-out mean cosine.
Two blue policies — RANDOM, and SWEEP (blues 0,1 hold a line at x = L/4 and
3L/4 sweeping up and down; blues 2–4 chase visible reds). Untrained
encoder. (`scratch/staleness_vs_search.py`, `scratch/coverage_gain_recheck.py`)

| features | RANDOM | SWEEP |
|---|---|---|
| unfitted "point toward arena centre" | 0.594 | **0.489** |
| position only (wall distances) | 0.680 | 0.334 |
| own features | 0.669 | 0.368 |
| position + weighted coverage | 0.670 | 0.458 |
| own + uniform coverage | 0.646 | 0.362 |
| own + weighted coverage | 0.670 | 0.456 |

Under RANDOM blues coverage adds nothing. Under SWEEP, weighted coverage
lifts the fitted probes by ~0.09–0.12 — but **no fitted probe, with or
without coverage, beats the unfitted centre rule (0.489)**, although a
linear map on wall distances can represent that rule exactly. The probe
baseline is therefore unreliable under the sweep, and the coverage gain
cannot be read as evidence. Why the probe fails is not known.

What does survive, because it involves no fitting: under the sweep, the
cell of each hidden red had median staleness at the cap (never observed),
while the unobserved free cells it could have been hiding in had a median
of **66 steps** (AUC 0.680). Under random blues both medians sat at the
cap (AUC 0.638, in a saturated map). So staleness does indicate *which
cells* reds hide in under systematic search. Whether a network can turn
that into better search decisions was not shown, and only training can
settle it — judged on captures and time-to-capture, not cosines.

Context for those runs: the sweep policy captured 18 of 60 reds over 20
episodes × 120 steps, against 1 for random blues. Hidden reds were within
15 m of a wall 86% (random) and 97% (sweep) of the time.

### 5.4 Explanations that were wrong

Kept because the measurements that killed them are the useful part.

* *"A uniform mean is biased toward where more region nodes lie."* Wrong
  framing — every message gets weight 1/K. An early proxy averaged raw
  `rel_pos` vectors (`scratch/region_mean_bias.py`), which says nothing
  about a mean of MLP messages since `mean(f(x)) ≠ f(mean(x))`.
* *"Geometric contributions cancel near the arena centre."* Falsified: the
  uniform response is flat across distance from the centre
  (`scratch/region_agg_signal.py`).
* *"A mean over K nodes attenuates differences by 1/√K."* Falsified: the
  response does not fall with R (`scratch/region_agg_vs_R.py`).
* *"Random blues pile up near walls, so position predicts reds better."*
  Falsified: blues were within 15 m of a wall 49% (random) vs 41% (sweep)
  of the time (`scratch/own_prior_mechanism.py`).
* *"Velocity features make the sweep probe overfit."* Falsified: position
  alone scores lower still, 0.334 (`scratch/coverage_gain_recheck.py`).
* *"Coverage adds +0.088 under the sweep."* Retracted: measured against a
  probe baseline that underperforms an unfitted rule (§5.3).

## 6. What is built, and what is not

| stage | status |
|---|---|
| A — staleness field in the env (`use_staleness`) | **done**, `tests/test_staleness.py` |
| B — aggregation to region nodes (`staleness_regions`) | **done**, `tests/test_region_nodes.py` |
| C — region node type + gb edges in `GNNEncoder` | **done**, `tests/test_region_graph.py` |
| C2 — region features into `_build_obs` / the policy | not started |
| D — tracker as the source for red/obstacle nodes | not started — next |
| E — learned attention over regions | not started |

Nothing in A–C reaches the policy yet (C2). All of it is off by default:
the env with `use_staleness=False` is unchanged step for step
(`test_disabled_env_is_unchanged_step_for_step`), and the encoder with
`n_region=0` is bit-identical to the pre-coverage version. C2 and D change
the observation semantics and need GPU retraining; the current checkpoint
consumes belief-map features and is the control, so the belief map should
stay switchable rather than be deleted in the same change.

## 7. The open risk: there is no reward for searching

```python
r_team = catch_reward * n_caught - step_cost
```

Nothing pays for detecting or for reducing uncertainty; credit for
sweeping a region arrives many steps later through a capture that may
never happen. The observation is necessary, not sufficient.

§5.3 adds a reason to take this seriously: the only regime where staleness
was measured to point at hidden reds was under *systematic* search. A
freshly initialised policy behaves close to random, the regime where it
did not. That argues — as reasoning, not a result — for an **imitation
warm-start** from a scripted sweeper, then PPO fine-tuning: it would put
the policy where coverage is informative from the start, and it sidesteps
reward shaping. The supervised pipeline built for the red-motion model is
a direct template. Reward shaping for coverage remains the fallback;
rewarding "detect" invites orbiting easy targets instead of capturing.

On the sweeper as a teacher: with a 40 m sensor radius, two blues 65 m
apart cover the 130 m width, and the per-axis speed cap (`BLUE_UAV` 1.5 vs
`RED_TARGET` 1.0) lets a line advance faster than an evader retreats along
the sweep axis. In principle that allows a gap-free sweep. In practice the
one scripted version captured 18 of 60 reds in 120 steps — far better
than random, far from clearing the arena. Pursuit is not coverage: what
matters is sweeping with no gap to slip back through.

## Reproduce

```
pytest tests/test_staleness.py tests/test_region_nodes.py tests/test_region_graph.py -v
python scratch/search_budget.py runs/red_motion/model_v3.pt
python scratch/staleness_vs_search.py
python scratch/coverage_gain_recheck.py
```
