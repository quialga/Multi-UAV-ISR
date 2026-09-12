# Search design (feature/target-tracking)

Where the recall budget actually lives, and the representation being built
for it. Continues from `tracking_diagnostics.md` §11, which ends by
showing that the motion-model work has been fully cashed in and that what
remains is a SEARCH problem.

## 1. "The target is not on radar" is two different problems

Measured over 6 episodes × 120 steps, stochastic red, with the tracker and
the learned motion model running (`scratch/search_budget.py`):

| | count | share | of unseen |
|---|---|---|---|
| on radar now | 670 | 31.0% | — |
| **NOT on radar** | 1490 | **69.0%** | — |
| RE-ACQUISITION: a track already holds it (< 20 m) | 849 | 39.3% | **57.0%** |
| DISCOVERY: seen before, track lost | 195 | 9.0% | 13.1% |
| DISCOVERY: never detected | 446 | 20.6% | 29.9% |

Supporting numbers: when a track holds an unseen target it sits **3.1 m**
from truth (median, p90 12.9 m), and the nearest blue to an unseen target
is **73 m** away (median; p90 103 m) against a 40 m sensor radius.

So **57% of the "lost" mass is not lost at all** — we know where those
targets are to within a few metres, and nobody goes there. That part needs
no search; it needs the tracker wired into the observation.

The other 43% is genuine discovery, and the tracker structurally cannot
help: a target never detected has no track. That is what this document is
about.

## 2. Two questions, two representations, no overlap

| question | answered by | shape |
|---|---|---|
| where is *this target*? | tracker / obstacle tracker | a few objects with identity |
| where have I *not looked*? | staleness field | a dense field |

The belief map is currently asked to do both, and does the first badly —
as the policy's source for unseen targets it scores MOTP 6.64 with 829
false positives, against the tracker's 2.65 and 98
(`tracking_diagnostics.md` §11.1).

Splitting the roles also removes a double-counting risk: a target that a
track already explains must not ALSO light up a region as "probably
something here", or the policy sees it twice.

## 3. Why staleness, and not a Bayesian coverage map

The quantity is **steps since each cell was last observed**. Plain integer
per cell, no decay factor, no diffusion kernel, no log-odds clip.

This is not merely the simpler choice, it is the correct one. For a target
never yet detected the prior is uniform, so "probability it is in this
region AND we have not seen it" is monotone in the time since we last
looked — **staleness is the sufficient statistic**. A Bayesian term only
adds information where there is POSITIVE evidence, and positive evidence
is exactly what the tracker owns.

The belief map confounds the two: as its log-odds decay toward zero,
"never looked here" becomes indistinguishable from "looked long ago and
forgot". Staleness separates them by construction.

### Resets on LOOKING, not on FINDING

`p_TP = 0.85`, so a look can miss a real target. Resetting only where
something was detected would quietly turn the field back into a (bad)
belief map, and an empty swept region would stay maximally stale forever
and be swept again and again.

### Occlusion is exact, not approximated

A cell inside the sensor disk but behind an obstacle was **not** looked
at, and keeps ageing. The field uses the same analytic segment-disk test
(`_rays_occluded_by_obstacles`) the belief update uses.

`_observed_cells_mask` duplicates that geometry rather than being
refactored out of `_update_belief_maps`, deliberately: that per-blue loop
interleaves the occlusion test with RNG draws, so restructuring it would
reorder the random stream and break the bit-exactness its own tests rely
on. A duplicate that can drift silently is worse than no duplicate, so
`test_observed_mask_matches_the_cells_the_belief_update_touches` pins the
agreement — with decay and diffusion off, the only cells the belief update
can change are the ones it observed.

### Known limitation, recorded rather than fixed

A single look resets to zero even though `p_TP < 1` makes it
inconclusive. Judged second-order: the dominant failure is "nobody has
been near this region for ~70 steps" (§1), not "we looked and were
unlucky" — three looks leave a 0.15³ = 0.3% miss. If it ever matters, the
fix is to count CONSECUTIVE CLEAN LOOKS instead: still one integer per
cell, still no Bayes.

## 4. Region nodes: resolution set by physics

The 26 × 26 field does not enter the graph raw. It is aggregated to
R × R **region nodes**, a new typed node — the same extension pattern
Stage 4 used to add obstacles.

Resolution is not a matter of taste. A region finer than the sensor
footprint is a distinction the policy cannot act on, because arriving
anywhere inside it sweeps the whole thing. With a 40 m sensor radius in a
130 m arena, **R = 5 gives 26 m regions**, about one footprint across.

This is also why the raw grid is the wrong input: 676 cell-nodes against 5
blues is 3,380 edges (the current graph has ~70), and worse, it would
re-diffuse exactly what the tracker resolved into objects.

26 has no divisor near 5, so regions are split as evenly as the grid
allows — 5, 5, 5, 5, 6 — rather than forcing a resolution that divides.
Both features are INTENSIVE (a mean and a fraction), so the uneven region
introduces no bias, and each centre is computed from its own cells rather
than assumed at a regular spacing.

### Features

| feature | meaning |
|---|---|
| `staleness` | mean steps since observed over the region's SEARCHABLE cells, ÷ `max_steps` → [0, 1] |
| `searchable` | fraction of cells outside every obstacle |

Averaging obstacle interiors into the staleness would dilute the signal
with area that cannot hide a target. And a fully blocked region reports
staleness **0**, not max: there is nothing to find there, so reporting max
would send the policy to search solid rock.

`searchable` earns its place separately — without it a stale but
unsearchable region is indistinguishable from a stale open one.

## 5. Aggregation: weighted, not averaged

There are ~25 region nodes against 14 other edges into a blue, so the
encoder's `index_add_` SUM would let coverage dominate the aggregate by
sheer count, with a magnitude scaling as R² — turning a RESOLUTION knob
into a gradient-scale one.

A uniform MEAN fixes the scale but not the content, and the reason is
worth stating precisely because the obvious intuition is backwards. Each
message already carries its own direction (`gb_edge_mlp` embeds
`rel_pos`), so the mean is **not** direction-blind — it is
direction-DOMINATED. The 25 messages differ mostly because their geometry
differs, while staleness only modulates each one mildly through
`h_region`.

Measured on the aggregate itself (`scratch/region_agg_signal.py`), by
replacing the real staleness field with a flat one and asking how much of
the aggregate moves:

| weighting | share of the aggregate carrying coverage |
|---|---|
| uniform mean | **0.016** |
| staleness × searchable | **0.309** (19.7×) |

So under a uniform mean, 98.4% of what the coverage path delivers is
positional baseline — and that part is **redundant**, since the blue
already has its position in its own wall-distance features. The pathway
would deliver almost nothing new.

The weights are `staleness × searchable`, normalised to sum to 1 over
regions. Zero parameters, and the limits are right: a uniform field gives
uniform weights (correctly — there is no preference), a structured one
concentrates on what is actually unexplored. This does not hand-code
where to go; it states that a freshly swept region carries no SEARCH
information, which is true by the definition of the node type.

Mechanically the weights ride the same per-edge multiply slot the
visibility masks already use, so the aggregation mechanism is untouched —
and the three options are nested (uniform ⊂ weighted ⊂ learned
attention), so the attention step ablates as a single variable.

*Caveat, stated rather than buried:* measured with an UNTRAINED encoder,
so this is accessibility at initialisation — what decides whether the
pathway can start learning, not what training converges to.

*Method note:* an earlier version of this measurement
(`scratch/region_mean_bias.py`) used the mean of the raw `rel_pos`
vectors as a proxy. It pointed the same way but could not establish the
claim, since the aggregate is a mean of MESSAGES through a non-linear MLP
and `mean(f(x)) ≠ f(mean(x))`. It is kept only as the input-side
diagnostic it actually is.

## 6. What is built, and what is not

| stage | status |
|---|---|
| A — staleness field in the env (`use_staleness`) | **done**, `tests/test_staleness.py` |
| B — aggregation to region nodes (`staleness_regions`) | **done**, `tests/test_region_nodes.py` |
| C — region node type + gb edges in `GNNEncoder` | **done**, `tests/test_region_graph.py` |
| C2 — region features into `_build_obs` / the policy | not started |
| D — tracker as the source for red/obstacle nodes | not started |
| E — learned attention over regions (ablate against C) | not started |

A and B are off by default and change nothing when disabled — pinned by
`test_disabled_env_is_unchanged_step_for_step`. C and D change the
observation semantics and therefore need GPU retraining; the current
checkpoint consumes belief-map features and is the control to compare
against, which is why the belief map should stay switchable rather than
being deleted in the same change.

## 7. The open risk: there is no reward for searching

```python
r_team = catch_reward * n_caught - step_cost
```

Nothing pays for detecting or for reducing uncertainty. Credit for
sweeping a region arrives ~70 steps later through a capture that may never
happen. **The observation is necessary, not sufficient**: the policy can
be given perfect region nodes and still not learn to use them.

Two ways out, in preference order:

1. **Imitation warm-start.** Pre-train against a principled sweeper, then
   fine-tune with PPO. This sidesteps reward shaping entirely, and the
   supervised pipeline built for the red-motion model (collector,
   featuriser, training loop) is a direct template.
2. **Reward shaping** for coverage. Kept as the fallback because rewarding
   "detect" invites orbiting easy targets instead of capturing them.

A structural fact that makes (1) promising: sensor radius 40 m means **two
blues spaced 65 m apart cover the full 130 m width**, and
`BLUE_UAV.v_max = 1.5` against `RED_TARGET.v_max = 1.0` means a sweeping
line advances faster than the evader retreats — the condition for a
clearing sweep to terminate. Two blues can clear while three pursue, so a
systematic sweep is not a weak baseline here but a demanding one.

Note that pursuit is not coverage: a sweep that does not CLOSE lets the
evader slip back into cleared ground and sweeps forever. "I have looked at
every cell" is not the criterion; "I have looked with no gap to slip back
through" is.

## Reproduce

```
pytest tests/test_staleness.py tests/test_region_nodes.py -v
python scratch/search_budget.py runs/red_motion/model_v3.pt
```
