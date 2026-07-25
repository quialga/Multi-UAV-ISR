# Stage 4 — entity-count generalisation eval

`scripts/eval_stage4_counts.py` measures how a trained Stage 4 policy
performs at **entity counts that differ from training** — the
"train on 2–4, evaluate zero-shot on 6 (or 10)" test that the
variable-entities work (`stage4_results.md` §"Post-close extensions")
unlocks. It sweeps the blue-team size, the number of reds, and the
number of obstacles independently, on a single checkpoint, without any
retraining.

See also: `stage4_results.md` (the variable-entities extension and the
count-agnostic critic), `stage4_backlog.md` (variable entity counts,
LANDED).

---

## The idea it rests on: capacity ≠ active count

Two things are easy to conflate:

| | What it is | When set | Does the model "know" it? |
|---|---|---|---|
| **Capacity** (`n_blue`, `n_red`, `n_obstacles`) | The padded tensor dimension — an *upper bound* on entities of that type | Policy/env construction | Structural, not learned |
| **Active count** | How many actually exist this episode | Sampled/spawned at `reset` | **Observed** from the graph, never told |

The policy is **never handed "there are 5 enemies" as an input.** It
reads the active count from the observation: inactive/absent entities
have their `active`-flag node feature at 0 and their edges zeroed, so
they drop out of message passing (actor) and the mean-pool (critic).

Every **learned tensor is count-agnostic**:

- The per-node / per-edge MLPs are shared across nodes — their shapes
  depend on feature dims, not counts.
- The critic's global context is a **masked-mean pool + count scalar**
  (`d + (d+1)[+(d+1)]`), independent of how many reds/obstacles exist —
  this is the variable-entities change that replaced the old
  count-hard-coded flatten.
- The edge-index buffers (`bb_src`, `rb_src`, …) are
  `register_buffer(..., persistent=False)` — rebuilt at construction,
  never saved.

**Consequence:** a policy trained at one capacity can be *re-instantiated
at a different capacity and load the exact same weights.* That is how
the eval reaches counts outside the trained range — and it even extends
to `n_blue`, whose weights are per-UAV (only the edge buffers and
`initial_hidden` depend on team size).

Because deterministic eval runs **only the actor**, a checkpoint whose
critic trunk can't transfer (e.g. a pre-pool checkpoint, `512 → 194`)
still evaluates fine — the single skipped critic tensor is unused, and
the script reports it once up front.

---

## Usage

```bash
# sweep red counts on a policy trained at some capacity
python scripts/eval_stage4_counts.py \
    --checkpoint runs/stage4/<run>/best.pt --n-red 2 4 6 8

# vary obstacle count too, more episodes for a tighter estimate
python scripts/eval_stage4_counts.py --checkpoint <ckpt> \
    --n-red 3 6 --n-obstacles 4 8 --n-episodes 40

# team-size (n_blue) generalisation — count-agnostic too
python scripts/eval_stage4_counts.py --checkpoint <ckpt> \
    --n-blue 3 5 8 --n-red 4
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | (required) | Stage 4 `.pt` (`best.pt` / `final.pt`) |
| `--n-blue` | trained `n_blue` | blue team sizes to sweep |
| `--n-red` | trained `n_red` | red counts to sweep |
| `--n-obstacles` | trained `n_obstacles` | obstacle counts to sweep |
| `--n-episodes` | 20 | deterministic episodes per (count, red-policy) cell |
| `--seed-base` | 30000 | eval seed base (fixed → reproducible) |
| `--device` | cpu | torch device |

Each cell is evaluated **deterministically** (action = policy mean, no
exploration noise) against three scripted reds — `stationary`, `random`,
`run` (`run_from_nearest_uav`). All perception / crash / moving-obstacle
knobs are carried over from the checkpoint's saved `args`, so the eval
env matches the training regime; only the entity counts are overridden
(and fixed — no `*_min` sampling — so every episode has exactly the swept
count).

---

## Reading the output

```
n_blue  n_red  n_obs |  vs stationary    vs random      vs run     mean frac
     5      3      4 |   2.80/3 (0.93)  3.00/3 (1.00) 3.00/3 (1.00)   0.98
     5      6      4 |   3.60/6 (0.60)  4.00/6 (0.67) 3.40/6 (0.57)   0.61
```

- Each cell is `mean_caught/N (fraction)`. **`frac = mean reds caught /
  n_red`** is the count-comparable metric — a raw catch count isn't
  comparable across different red totals.
- `mean frac` averages the fraction over the three red policies.
- Caught is computed as `n_red_start − active_now` (padding-aware), so it
  never miscounts padded reds as caught.

**The two axes move in opposite directions — read them accordingly:**

- **More reds ⇒ harder** (same team must corner more evaders): the
  fraction *should* fall with `n_red`. The generalisation question is how
  *gracefully* — a variable-N-trained policy should hold a much flatter
  curve than a fixed-count one.
- **More blues ⇒ easier** (more pursuers): the fraction rises, or
  saturates near 1.0. Here the question is whether the policy actually
  *coordinates the extra UAVs* rather than ignoring them — a flat-high
  blue curve is the "it uses the team it's given" signal.

---

## Baseline: a fixed-count policy degrades off-distribution

Illustrative run on `obstacles_v1` (trained at `n_blue=5, n_red=3,
n_obstacles=4`; **5 episodes/cell**, so indicative, not a final number):

Red-count sweep (`n_blue=5, n_obs=4`):

| n_red | mean frac |
|---|---|
| 2 | 1.00 |
| 3 (trained) | 0.98 |
| 5 | 0.72 |
| 6 | 0.61 |

Team-size sweep (`n_red=3, n_obs=4`):

| n_blue | mean frac |
|---|---|
| 3 | 0.87 |
| 5 (trained) | 0.98 |
| 8 | 0.98 |

`obstacles_v1` was trained at a **fixed** count, so it degrades on the
red axis away from 3 (0.98 → 0.61 at 6). That drop is exactly the
**baseline a variable-N-trained policy is meant to beat** — the headline
"train on 2–4, zero-shot to 6" result is a much flatter red-count curve
than this. (It already generalises reasonably on the blue axis — 8 UAVs
hold at 0.98 — because more pursuers only makes the task easier.)

---

## Caveats

- **Scripted reds only.** This measures generalisation against the
  stationary/random/`run` heuristics, not a learned evader. The
  self-play adversary (roadmap capstone) would be a separate, harder
  eval.
- **Absolute difficulty scales with the red:blue ratio.** Compare a
  policy against the *fixed-count baseline at each count*, not against a
  flat line — some fraction drop at high `n_red` is the task getting
  harder, not the policy failing to generalise.
- **Episode count.** Use `--n-episodes 40`+ for a reported number; the
  table above used 5 for a quick smoke.
- **Actor-only.** The eval never runs the critic, so it says nothing
  about value-estimate quality — only about the deployed policy's
  behaviour.
