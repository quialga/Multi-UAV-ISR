# Attention in the typed GNN — the mathematics

What the aggregation computes, per channel, written as equations rather than
as code.  Results and experiments live in `docs/stage4_results.md` §16-§21;
this file is the specification those sections measure.

---

## 1. Notation

| symbol | meaning |
|---|---|
| `h_i^(t)` | embedding of node *i* after message round *t*, in ℝ^d (d = 64) |
| `τ` | edge type, `τ ∈ {bb, rb, ob, gb}` |
| `N_τ(i)` | senders of type τ into receiver *i* |
| `e_ji^τ` | encoded edge feature for the edge *j → i* |
| `v_ji^τ ∈ {0,1}` | visibility mask |
| `n_τ(i)` | number of LIVE in-edges of type τ at *i*, i.e. `Σ_j v_ji^τ` |
| `H` | attention heads (4) |

Blue nodes are the only receivers; reds, obstacles and regions never update.
The four channels are blue→blue (allies), red→blue (tracks), obstacle→blue
and region→blue (coverage).

---

## 2. What is unchanged

Per-type encoders, a **shared** message function φ, and a residual update on
receivers only:

```
h_i^(0) = MLP_type(i)(x_i)
e_ji^τ  = MLP_edge^τ(f_ji),        f_ji ∈ ℝ⁷ = [rel_pos(2), rel_vel(2), range, bearing(2)]

m_ji^τ  = φ( [ h_j^(t) ‖ h_i^(t) ‖ e_ji^τ ] )          φ : ℝ^{3d} → ℝ^d

h_i^(t+1) = h_i^(t) + υ( [ h_i^(t) ‖ a_i ] )           υ : ℝ^{2d} → ℝ^d
```

φ and υ are shared across all four channels; only the input and edge
encoders are per-type.  Everything above is untouched by this work.

**The only thing that changed is how `a_i` is formed from the messages.**

---

## 3. The original aggregation, and its defect

```
a_i^τ = Σ_{j ∈ N_τ(i)}  v_ji^τ · m_ji^τ
a_i   = Σ_τ  a_i^τ
```

If messages of a type have comparable norms ≈ μ and are correlated:

```
‖a_i^τ‖ ≈ μ · n_τ(i)
```

**The live-edge count enters the magnitude, with no upper bound.**  Three
consequences, all measured:

1. `comms_radius` cannot be varied as an information knob, because changing
   it changes `n_bb` and therefore the scale (§16).
2. A variable entity count is not representable: 1 → 3 live tracks triples
   `‖a_i^rb‖` (§21).
3. Worse than either, the *shares* of the aggregate move.  Since all four
   channels sum into the same d dimensions, what matters is their relative
   size:

| channel | sum @ comms 40 | sum @ comms inf |
|---|---|---|
| bb (allies) | 2.11 msgs → 53% | **4.0 → 68%** |
| rb (targets) | 0.86 → 22% | **0.86 → 15%** |
| gb (regions) | ≈1.0 → 25% | ≈1.0 → 17% |

Opening the radio squeezes the **target** channel from 22% to 15% while ally
traffic grows to 68%.  The policy loses track of what it is chasing — which
is why it failed against a *stationary* red (2.62/3), the prediction that
distinguished this mechanism from the saturation story first proposed (§19).

---

## 4. Attention, per channel

**Score** each edge, per head, with a **shared** ψ : ℝ^{3d} → ℝ^H:

```
s_ji^τ,k = ψ_k( [ h_j^(t) ‖ h_i^(t) ‖ e_ji^τ ] )
```

ψ is shared across channels for the same reason φ is: the type information
already arrives through `e_ji^τ`, so a per-type scorer would duplicate it.

**Masked softmax** over the receiver's live in-neighbours:

```
                exp(s_ji^τ,k) · v_ji^τ
α_ji^τ,k  =  ──────────────────────────────────
             Σ_{l ∈ N_τ(i)} exp(s_li^τ,k) · v_li^τ
```

Setting `s = −∞` where `v = 0` and applying a plain softmax is the same
thing; the `−∞` form is how it is computed stably.  **Masking after the
softmax would be a different and wrong operation** — the survivors would then
sum to less than 1, and that deficit depends on how many survived, which
re-introduces precisely the count-dependence being removed.  With scores
`[2,1,1,0]` and the last two hidden: masking after gives
`[0.51, 0.19, 0, 0]`, summing to 0.70; masking before gives
`[0.73, 0.27, 0, 0]`, summing to 1.

**Heads** partition the message rather than duplicating it.  Split
`m ∈ ℝ^d` into `H` blocks `m^[k] ∈ ℝ^{d/H}`; head *k* weights block *k*:

```
a_i^τ,[k] = Σ_{j ∈ N_τ(i)}  α_ji^τ,k · m_ji^τ,[k]

a_i^τ = [ a_i^τ,[1] ‖ … ‖ a_i^τ,[H] ] ∈ ℝ^d
```

So the concatenation is **within** a channel, across heads.  One head would
impose a single ranking of neighbours, and "the nearest ally" and "the ally
best placed to intercept" are different rankings that a single weight vector
must compromise between.

### 3 properties that matter

**(i) Normalisation.**  `Σ_j α_ji^τ,k = 1` whenever `n_τ(i) > 0`.

**(ii) A norm bound, by convexity.**  The α are non-negative and sum to one,
so `a_i^τ,[k]` lies in the convex hull of the blocks `m_ji^τ,[k]`, and by
Jensen:

```
‖a_i^τ‖  ≤  max_j ‖m_ji^τ‖
```

**Each channel contributes at most one message's worth, whatever the live
count.**  This single inequality is the whole fix; the sum admits no
analogue.

**(iii) The empty case is defined, not indefinite.**

```
α_ji^τ,k ≡ 0  ∀j      when  n_τ(i) = 0      ⟹   a_i^τ = 0
```

Without this definition an all-`−∞` row gives `0/0`.  It is defined as zero
because "no neighbour contributed" is what should be meant, and it is not an
edge case: 27.6% of steps have no confirmed track at all, and since the
tracker is command-layer fusion, when that happens every blue's rb row is
empty simultaneously.

### The mean is a restriction, not a rival

Set `H = 1` and `s_ji ≡ 0`:

```
α_ji^τ = v_ji^τ / Σ_l v_li^τ = 1 / n_τ(i)
```

which is exactly the arithmetic mean over live edges.  **mean ⊂ attention.**

That containment is what makes the mean a clean ablation: the control is a
restriction of the treatment, and it isolates the two effects at the point
where each lives.

| effect | where it acts |
|---|---|
| **normalisation** | the denominator `Σ_l exp(s_li)·v_li` |
| **discrimination** | the numerator's dependence on `s` |

Measured split: ≈ 2/3 normalisation, 1/3 discrimination (§19.4), consistently
at both red speeds.

---

## 5. The region channel

The coverage path is the one channel that was *already* normalised, by a
hand-designed weighting:

```
a_i^gb = Σ_g  w_g · m_gi^gb ,        w_g = (σ_g · c_g) / Σ_g' (σ_g' · c_g')
```

with σ_g the staleness of region *g* and c_g its searchable flag.  Two facts
about `w_g` drive the change.

**It carries no receiver index.**  `w_g` is a function of the region alone, so
all blues receive the *same* mixture of regions.

**It adds no information.**  σ_g and c_g are the region's own two node
features, so the weighting is a function of data the network already has; it
supplies an inductive bias, not knowledge.  The bias it supplies is real
though: with uniform `w = 1/K` the aggregate is, to first order,

```
a_i ≈ A · mean_g(h_g) + B · mean_g(e_gi)
```

so it depends on the staleness field only through its **mean**, and *which*
region is stale enters only through the MLP's non-linearity.  Measured at
initialisation: permuting which regions are stale moves a uniformly-averaged
aggregate by **0.6%**, while shifting the overall level moves it by **17.7%**.
Multiplying by σ_g puts the arrangement in at first order.

Replacing `w_g` with attention:

```
α_gi^k = softmax_g ( ψ_k( [ h_g ‖ h_i ‖ e_gi ] ) )
```

gains both properties at once: the weighting is learned *and* it depends on
the receiver.  There is no mask on this channel — coverage is a command-layer
quantity, so no region is hidden from any blue — hence `v_gi ≡ 1`, the softmax
runs over all `R²` regions, and the empty case cannot arise here.

### Why the receiver dependence is the point

Aggregation is a **lossy projection**: it collapses `R²` messages into one
vector in ℝ^d, and the information about *which* region received weight is
destroyed there.  With a receiver-independent `w`, that loss is **identical
for every blue**, so no blue can reconstruct downstream an assignment the
aggregate already averaged away.

To express "blue 1 searches region A while blue 2 searches region B" the
*weights* must differ, not merely the messages.  A shared mixture can only
differentiate blues through `m_gi`, which is the same φ applied to
`(h_g, h_i, e_gi)` — downstream of an average already taken with
blue-independent weights.  Division of search is therefore not merely unlearnt
under `w_g`; it is **not representable**.

---

## 6. Combining the channels: the open question

All of §4 and §5 normalise **within** a channel.  Across channels the
aggregate is still a plain sum:

```
a_i = a_i^bb + a_i^rb + a_i^ob + a_i^gb
```

By property (ii) each term is bounded by one message's worth, so the
magnitude problem is gone.  But the *weighting between channels* is now
implicitly uniform and constant: each channel enters with weight 1.  Its
content varies with the state; its **budget** does not.

This is a real gap, and there is a second reason it bites.  All four channels
pass through the same φ and sum into the **same d dimensions**, so nothing
makes them occupy distinct subspaces.  The channels are **superimposed**, and
υ — which receives `[h_i ‖ a_i]` — therefore cannot in general separate "this
came from a target" from "this came from a region".  The type identity
survives only in the message content, not in the position.

So a tactical decision that plainly exists — *with a confirmed track, chase;
without one, search; with an obstacle close, avoid it* — is something the
current architecture **cannot express as a reallocation of attention between
channels**.

### Option A — concatenate instead of summing

```
a_i = [ a_i^bb ‖ a_i^rb ‖ a_i^ob ‖ a_i^gb ] ∈ ℝ^{4d}
υ : ℝ^{5d} → ℝ^d
```

The channels become **positionally separable**, so υ can learn a per-channel
weighting — fixed, or state-dependent through its own non-linearity.

* **No new mechanism**, only capacity.
* Costs ≈ 3× υ's parameters (input 2d → 5d).
* Changes υ's shape, so a warm start from a summing checkpoint
  **re-initialises** one of the two shared MLPs — a real transplant cost.
* The per-channel weighting stays **implicit**: there is no quantity to read.

### Option B — a second level of attention over channels

Score each channel's aggregate with a shared g, then softmax **across
channels**:

```
β_i^τ,k = softmax_τ ( g_k( [ h_i ‖ a_i^τ ] ) )

a_i^[k] = Σ_τ  β_i^τ,k · a_i^τ,[k]
a_i     = [ a_i^[1] ‖ … ‖ a_i^[H] ]
```

This is the same construction as §4 with channels in place of edges: score
each candidate independently with a shared scorer, normalise across
candidates.  Scoring each channel separately — rather than feeding all four
aggregates into one MLP — matters for two reasons: it keeps the number of
channels out of the architecture (turning obstacles on does not change an
input width), and it shares "how to score an aggregate" across channels
instead of learning it four times.

* **Keeps per-channel normalisation**, so the count-invariance of §4 and the
  norm bound (ii) survive intact.
* Makes the channel budget **state-dependent**: `β` depends on `h_i` and on
  what each channel actually delivered this step.
* Adds one small MLP, `g : ℝ^{2d} → ℝ^H`.
* Transplants cleanly: it *adds* parameters rather than reshaping υ.

**Not Option C.**  A single flat softmax over all edges of all channels at
once would force a blue to trade attention on a target against attention on a
wingman, and would discard the per-channel normalisation that §19 and §21
measured to be the main benefit.

### Why B is worth having even if it ties A on score

The comparison against the summing baseline is weak on its own: the sum is
just "uniform channel weights", so any added capacity should edge it, and a
score difference would not say *why*.

**B's advantage is that `β` is a readable quantity.**  It converts the
question from *"does it score higher?"* — which is hard to interpret — into
*"does it gate as the mechanism predicts?"*, which is falsifiable by
conditioning on state:

| condition on the receiving blue | prediction |
|---|---|
| ≥ 1 live track | **β_rb rises** |
| 0 live tracks | **β_gb rises** |
| obstacle within *k* radii | **β_ob rises** |
| allies beyond `comms_radius` | β_bb falls |

If `β` is approximately constant across those conditions, the gate learned
nothing and Option A suffices — a clean negative result rather than a failure.
If it tracks them, the architecture is demonstrably doing the thing it was
designed to do, which is far stronger evidence than ±0.1 on captures.

Option A offers no equivalent: with a concatenation the per-channel weighting
is distributed through υ's weights and is not a quantity one can condition on
and read off.  **That is a legitimate reason to prefer a slightly more complex
architecture**, independent of which one scores higher.

### Order

A first, because it is cheaper and adds capacity without a mechanism: if it
recovers the benefit, B is justified only by explainability rather than by
performance, which is still a decision but a different one.  Then B, measured
primarily by the conditional β table above and only secondarily by score.
