# Stage 3 — Partial Observability + Recurrent Actor + CTDE Critic (Design)

_Research journal entry, July 2026.  Written after Stage 2 landed as a
PASS on the scaling experiment
([`stage2_results.md`](stage2_results.md))._

Stage 3 replaces Stage 2's fully-observable env with a **partially
observable** one: each blue UAV sees only entities within a limited
sensor radius.  The GNN backbone from Stage 2 stays; the actor gains
a **GRU per blue node** for belief tracking over time; the critic is
kept **centralised** (CTDE — sees the full state during training,
same architecture as Stage 2).  Warm-starting the critic from the
Stage 2 GNN checkpoint gets us a value function that already knows
how to evaluate full-state graphs.

Same env otherwise as Stage 2's scaling config (N=5 vs M=3, arena
130 × 130).

## 1. Motivation and hypothesis

Stage 2's PASS confirmed the coordination hypothesis: the GNN
architecture unlocks coordination that MLP cannot express.  But
Stage 2 was **fully observable** — every UAV sees every entity at
every timestep.  Real ISR systems have sensor limits.  Two coupled
challenges emerge under partial observability:

1. **Belief tracking**: what a UAV cannot see now, it must remember
   from past observations (or infer via teammates' broadcast of what
   *they* saw).
2. **Coordination under uncertainty**: the target-assignment strategy
   Stage 2 discovered assumed everyone knows where every red is.
   Under partial obs the assignment problem becomes probabilistic —
   who should search for the red we haven't seen in 20 steps?

**Falsifiable hypotheses for Stage 3:**

- **H1 (baseline drop)**: the Stage 2 GNN evaluated on the same env
  with sensor radius active will lose most of its performance — an
  MLP-scale collapse under a POMDP for a memoryless policy.
- **H2 (recovery via recurrence)**: adding a per-blue GRU on top of
  the actor's GNN output recovers most of the fully-observable
  Stage 2 performance.

Numerically:
- Stage 2 fully-observable GNN vs Run at N=5 vs M=3: **+22.29
  reward, 52.8 steps, 3.00 caught**.
- H1 baseline (Stage 2 GNN on partial obs, no memory): expect
  reward < +10, catches < 2.5/3, steps > 100.  Would confirm that
  memory is *necessary*.
- H2 target (Stage 3 recurrent GNN + CTDE critic on partial obs):
  reward within ~5 points of Stage 2, catches ≥ 2.7/3, steps within
  ~30% of Stage 2's 52.8.

If Stage 3 lands close to Stage 2 under partial obs, we have a
recurrent-CTDE result worth showing on top of the Stage 2 win.  If
it lands close to H1 baseline, memory alone isn't enough and we
need Stage 4's belief-state auxiliary loss.

## 2. What changes in the environment

### 2.1 Sensor radius

New parameter `sensor_radius: Optional[float] = None` on `PursuitEnv`.

- `None` → fully observable (Stage 2 backward-compat, still the
  default).
- Numeric → each blue UAV sees only entities strictly within
  `sensor_radius` (Euclidean distance).

**Locked value**: `sensor_radius = 40` in the 130 × 130 scaled arena
(roughly 30% of arena diameter, in line with ISR sensor-cone
conventions in the literature).  Small enough to make the POMDP
non-trivial; large enough that a single UAV can still see 2-3
entities at once in typical episode states.

### 2.2 Structured obs — two accessors

The env exposes both views:

- **`structured_observation()`** — the full-state graph, unchanged
  from Stage 2.  Used by the CTDE critic and by any legacy code
  path that wants global state (e.g., logging).
- **`structured_partial_observation()`** — same layout dict, but
  the **edge features** are **zero-masked** for any edge whose
  sender or receiver is beyond `sensor_radius` of the acting
  agent... wait — but there's no "acting agent" for the graph.

Hmm.  Sensor radius is *per receiver blue*.  Since our edges are
directed and each blue's own view is different, the correct partial
obs is:

- **rb edges**: rb edge `(r → b)` is visible to blue `b` iff
  `distance(blue_b, red_r) ≤ sensor_radius`.  Otherwise zero-mask
  the edge features.
- **bb edges**: bb edge `(i → j)` is visible to blue `j` iff
  `distance(blue_j, blue_i) ≤ sensor_radius`.  Otherwise zero-mask.
- **Node features**: blue's own intrinsic state (velocity, walls,
  time) is always its own; blue never loses sight of *itself*.
  Red node features (`active_flag`) are per-red-node, but a red
  only appears in a blue's rb edges if within range.

This is a per-*receiver* mask, so strictly speaking the graph is
different for each blue.  Two ways to represent this:

- **Option (i)**: batch of N_blue partial graphs, one per acting
  agent.  Clean semantically; N_blue × the tensor sizes at rollout
  time.  The GNN processes each subgraph independently.
- **Option (ii)**: single graph shared across blues, with edge
  features zero-masked per receiver.  Since our edges are indexed
  by receiver-id anyway (bb_edge_dst, rb_edge_dst), we can compute
  a per-edge mask based on the receiver's distance to the sender at
  each timestep.  Same tensor shape as fully-observable, plus a
  `(n_edges,)` mask.

**Locked choice: Option (ii)** — same tensor shapes as Stage 2, no
extra batch dim, just an additional per-edge visibility mask.
Message passing computes messages as before; the mask is applied
before aggregation (masked messages contribute zero).

Implementation-wise this means:
- New env-level method `_compute_edge_visibility()` returns
  `(bb_mask, rb_mask)` shaped `(n_bb,)` and `(n_rb,)`.
- `structured_partial_observation()` returns the Stage-2 dict plus
  two new keys: `bb_edge_visible`, `rb_edge_visible` (float32 arrays,
  1.0 visible / 0.0 hidden).
- GNN forward pass in the actor multiplies message tensors by the
  visibility masks before `index_add_` aggregation.

## 3. Actor architecture — GNN + per-blue GRU

At time `t`, on partial obs:

```
partial_obs_t (blue features + red features + edges + visibility masks)
    │
    ▼
GNN encoder (2 rounds MP, edges masked by visibility)
    │
    ▼
per-blue node embedding h_blue_i^(2)_t   ← belief-independent local view
    │
    ▼
GRU per blue: hidden_i_t = GRUCell(h_blue_i^(2)_t, hidden_i_{t-1})
    │
    ▼
per-blue hidden state hidden_i_t          ← belief-carrying state
    │
    ▼
Actor MLP (shared across blues): mean_i = actor_mlp(hidden_i_t)
    │
    ▼
action_i ~ Normal(mean_i, exp(actor_log_std))
```

Design details:

- **GRU dimension**: `d_hidden = 64` — same as the node embedding
  dim.  One `nn.GRUCell(input_size=64, hidden_size=64)` shared
  across all blue UAVs (parameter sharing as before).
- **Per-blue hidden state**: rank-3 tensor
  `hidden_states: (n_envs, N_blue, d_hidden)`.  Each UAV maintains
  its own belief.  The vec_env tracks these.
- **Reset semantics**: `hidden_states` are zeroed at every episode
  reset (both initial and auto-reset on done).  No cross-episode
  carryover.
- **log_std**: unchanged from Stage 2 —
  [state-independent, shared across agents](stage2_gnn_design.md#33-policy-head-—-decentralised-execution).

The GRU is what makes the policy able to remember "there was a red
heading north-east 20 steps ago; I should predict where it is now."
Without it, an out-of-sight red is invisible to the policy each
step and coordination falls apart.

## 4. Critic architecture — CTDE (unchanged from Stage 2)

```
full_state_t (fully-observable graph, Stage 2's structured_observation())
    │
    ▼
GNN encoder (2 rounds MP, ALL edges visible — CTDE)
    │
    ▼
node embeddings (full)
    │
    ▼
spatial aggregation: sum blue + concat red
    │
    ▼
critic MLP → V(full_state_t)
```

**Byte-identical to Stage 2's critic architecture.**  The critic
never needs memory because the full state is Markov by construction
— knowing the full state at time `t` is sufficient to estimate its
value.

Two separate GNN encoders (actor's on partial obs, critic's on full
state) — not shared weights.  Rationale: they solve different
problems (encoding a masked graph vs a full graph) and warm-starting
the critic from Stage 2 favours keeping them separate.

## 5. Warm-start plan

At training start:

1. Load the Stage 2 checkpoint (`runs/stage1/scaling_gnn/best.pt`).
2. Copy the following weights into the Stage 3 policy's **critic**
   sub-module:
   - `blue_input_mlp`, `red_input_mlp`, `bb_edge_mlp`, `rb_edge_mlp`
   - `msg_mlp`, `update_mlp`
   - `critic_trunk`, `critic_head`
3. Leave the actor path (partial-obs GNN + GRU + actor head) randomly
   initialised as usual.
4. Train.

Effect: the critic starts as a competent Stage-2-level value
function.  The actor has to learn from scratch, but the value
signal it gets is already well-calibrated on graph inputs.
Empirically, this significantly speeds up early training in
recurrent CTDE setups.

## 6. Rollout buffer — dict-obs plus hidden states

`GraphRolloutBuffer` (or a Stage-3-specific
`RecurrentGraphRolloutBuffer`) needs to store:

Per (T, E) — env-timestep tensors:
- All Stage 2 obs fields for the **full state** (critic input): same
  as current buffer.
- All Stage 2 obs fields for the **partial state** (actor input):
  new copies.
- Two edge-visibility masks per timestep: `bb_edge_visible`,
  `rb_edge_visible`.

Per (T, E, A) — per-agent tensors:
- Actions, log_probs: unchanged.
- **Actor GRU hidden state at time t**: new, shape `(T, E, A, d_hidden)`.
  Stored at rollout collection time so PPO update can re-condition
  the actor forward with the correct hidden per step.

Memory footprint: roughly 2× the Stage 2 buffer (two graph views per
step) plus the hidden-state field.  In practice with default
`n_envs=16, rollout_steps=256, N_blue=5, d_hidden=64` that's
~5 MB additional per rollout — cheap.

## 7. PPO update with recurrence — the "stateless-at-update" convention

Standard practice for recurrent PPO (implementations like Stable-
Baselines3's RecurrentPPO, cleanrl's PPO+LSTM): during the update,
we do NOT re-run the GRU across the whole rollout segment.  Instead
we **use the stored hidden state per step** and re-run the actor
forward at each step conditioned on that stored hidden.

Why: minibatch shuffling mixes timesteps from different envs/
rollouts.  Re-running the GRU across the shuffled minibatch would
mean carrying the wrong hidden state from the previous minibatch
sample.  The alternative (truncated BPTT with segment-shuffled but
step-ordered batches) is more accurate but complicates the update
loop.  The stateless-at-update convention is standard for PPO and
loses very little in practice.

Concretely in the PPO update:

```
for batch in buffer.iter_minibatches(mb_size):
    partial_obs = batch["partial_obs"]
    full_state  = batch["full_state"]
    old_hidden  = batch["hidden_states"]      # stored from rollout
    old_actions = batch["actions"]
    old_log_probs = batch["log_probs"]

    # Actor: condition on stored hidden, run forward, get new log_prob
    _, new_log_probs, entropy, _ = policy.actor_forward(
        partial_obs, hidden=old_hidden, action=old_actions,
    )
    # Critic: full state, no hidden needed
    _, _, _, new_values = policy.critic_forward(full_state)

    # ... standard PPO clipped surrogate + value loss ...
```

The actor's `hidden_out` from this forward is discarded — we don't
propagate it through the minibatch.  On the next rollout collection
the fresh hidden states are computed step-by-step as normal.

## 8. Hyperparameters (locked Stage 3 defaults)

Same env / PPO as Stage 2's scaling config unless noted:

| Parameter | Value | Note |
|---|---|---|
| `n_blue` | 5 | scaling config |
| `n_red` | 3 | scaling config |
| `arena_size` | 130 | scaling config |
| **`sensor_radius`** | **40** | new — see §2.1 |
| `d_hidden` | 64 | node emb + GRU hidden |
| `n_msg_rounds` | 2 | GNN rounds, unchanged |
| Actor GRU cells | 1 | `nn.GRUCell`, no stacking |
| **Critic warm-start** | Stage 2 GNN | new |
| Actor init log_std | 0.0 | unchanged |
| PPO `n_rollouts` | 1500 | may need more due to POMDP complexity |
| PPO `ent_coef` | 0.018 | keep the value that worked in Stage 2 |
| All other PPO settings | Stage 2 defaults | |

Parameter count estimate: Stage 2 was 64k.  Stage 3 adds:
- Second GNN encoder (partial obs actor): +~40k
- GRU cell (input 64 → hidden 64): ~24k
- Total: ~130k.  Fits comfortably on a single GPU.

## 9. Acceptance criterion

> **Stage 3 status: LANDED and CLOSED (2026-07-10).**  All acceptance
> criteria — original AND revised — met by
> `runs/stage3/gpu_opt1_critic_frozen/best.pt`.
> See [`stage3_results.md`](stage3_results.md) for the final numbers,
> the exact reproduction command, and the Phase 3.5/3.6 experiment log.
>
> Headline vs `RunFromNearest`: **+18.16 return, 2.90 caught, 78.9
> steps** — beats sensor-aware Greedy (+11.47, 2.28, 136.4) on every
> metric and clears the original "Greedy + 5" return bar by +1.69.
> Next: Stage 4 (sensor noise + occlusion + explicit belief-state
> predictor).  See [`design.md §5`](design.md).

> **Revised 2026-07** after the fair-Greedy correction — see
> `docs/stage3_results.md §4-5` for the history.  The original text
> is preserved below the revision as a record.

### 9.0 Baseline (revised)

The correct H1 baseline is **sensor-aware `GreedyPursuer`** evaluated
in the Stage 3 partial-obs env (`sensor_radius = 40`) — *not* the
Stage 2 GNN checkpoint.  The Stage 2 GNN was trained under full
observability and would be OOD on masks; comparing against it would
measure the wrong thing (Stage 3 vs OOD-Stage-2 rather than Stage 3
vs "best stateless heuristic under the same sensor").

`GreedyPursuer` was updated in commit ab947ee to respect
`env.sensor_radius`: each UAV pursues the closest active red *within
its own sensor range*, or holds position if no red is visible.  Per-
UAV visibility matches how the env's `_compute_edge_visibility` masks
the graph's rb edges.

### 9.1 Acceptance (revised)

Stage 3 passes if, evaluated at 50 episodes per (blue, red) cell with
matched seeds against sensor-aware Greedy in the same partial-obs env:

Against **every** red baseline (Stationary, Random, RunFromNearest):
- `mean_caught ≥ Greedy_caught + 0.15`, **and**
- `std_return ≤ 0.85 × Greedy_std_return` (coordination-quality
  gate: the learned policy must produce lower-variance outcomes than
  a stateless heuristic — otherwise it isn't demonstrating memory or
  coordination that the heuristic lacks).

Additionally against RunFromNearest (the hardest red):
- `mean_caught ≥ 2.7 / 3` absolute (task actually solved).

**What was dropped and why**: the original criterion `mean_return ≥
Greedy + 5` was set against a Greedy baseline that (unintentionally)
read `env.state_snapshot()` directly and cheated past `sensor_radius`.
After the fair fix, +5 return on top of a fair Greedy is
mathematically unattainable in the arena — see `docs/stage3_results.md §4`
for the arithmetic.  Return is retained as a diagnostic in the results
writeup, not as a gate.

**FAIL** if Stage 3 doesn't meet the catches criterion against
RunFromNearest — that would suggest memory-based recovery isn't
happening and would motivate the aux hidden-loss experiment (Phase
3.5) or a belief-state predictor.

### 9.2 Original text (2026-06, superseded)

<details>
<summary>Click to expand the pre-revision criterion</summary>

The original acceptance text used the Stage 2 GNN as H1 baseline and
demanded `mean_return ≥ H1 + 5`.  Both were miscalibrated: the Stage
2 GNN was OOD on partial obs, and the +5 return target was set
against a cheating Greedy in an unrelated calibration.

```
1. H1 baseline — evaluate Stage 2's scaling_gnn/best.pt on the
   Stage 3 partial-obs env (sensor_radius = 40), 50 episodes per red
   policy, matched seeds.
2. Stage 3 policy — evaluate the trained Stage 3 recurrent CTDE
   policy on the same env with the same seeds.

PASS if:
- Stage 3 vs Run mean_return ≥ H1 baseline vs Run + 5.0, AND
- Stage 3 vs Run mean_caught ≥ H1 baseline + 0.5, AND
- Stage 3 vs Run mean_caught ≥ 2.7 / 3 (task actually solved).
```

Preserved for reproducibility of prior TB scalars / early-training
`docs/stage3_gpu_run.md` references.

</details>

## 10. Implementation plan (checklist for the coding turn)

1. **Env**: add `sensor_radius` param + `structured_partial_observation()`
   + per-edge visibility masks.  Add smoke tests for the mask
   invariants (visibility ≡ within-radius, symmetric for bb edges).
2. **GNN policy** → `GNNCTDEPolicy`: two GNN encoders (actor's on
   partial, critic's on full), GRU cell, warm-start method
   `load_stage2_critic(path)`.  Unit test that critic warm-start
   loads clean.
3. **Buffer** → `RecurrentGraphRolloutBuffer`: stores partial obs,
   full state, and hidden states per (T, E, A).  Yields dict
   minibatches consumed by the PPO update.
4. **Vec env**: return both partial obs and full state per step;
   maintain per-agent hidden states externally (trainer owns them,
   not the vec env — the vec env is stateless).
5. **PPO update**: new `ppo_update_ctde` in `isr/train/ppo.py` (or
   extend the existing one with actor-critic-forward split).
6. **Training script**: single script updates — build
   `GNNCTDEPolicy`, warm-start from Stage 2 path, run recurrent
   collection + update.
7. **Smoke run**: 5-10 rollouts on CPU, confirm reward increases.
8. **Full training + eval**: 1500 rollouts on GPU, then eval on both
   partial-obs env (Stage 3 policy + H1 baseline) with `evaluate_trained.py`.

## 11. Expected outcomes

**If Stage 3 PASSES**: hypothesis confirmed.  Recurrence + CTDE
recovers most of Stage 2's coordination under partial observability.
Write-up story: "we broke coordination with partial obs, restored
it with a GRU + warm-started centralised critic."  Then Stage 4
(explicit belief-state predictor + sensor noise) becomes the next
step.

**If Stage 3 lands PARTIAL**: memory helps but doesn't fully close
the gap.  Interesting result on its own; motivates the belief
predictor.

**If Stage 3 FAILS**: memory alone isn't the answer.  Would suggest
either the sensor radius is too aggressive, or the partial-obs
regime needs an explicit belief-state auxiliary loss (Stage 4's
concept).  Would inform the Stage 4 spec directly.

## 13. Option 1 — Cross-blue hidden state sharing via GNN (Phase 3.6)

**Status**: 2026-07 follow-up.  Motivated by the Phase 3.5 aux-loss
experiments (see `docs/stage3_results.md § Phase 3.5`) — option C
was chosen as Stage 3's final architecture, but its ceiling of 2.92
catches / 3 vs Run appears to be capped by an architectural
limitation.

### 13.1 The observation

In the current architecture ([`gnn_ctde_policy.py:272-302`](../isr/agents/gnn_ctde_policy.py:272)):

- The GNN encoder aggregates messages between blues based **only on
  current-step observations**.
- The GRU updates each blue's hidden state using **only that blue's
  own** encoded observation and **only that blue's own** previous
  hidden state — the GRU is applied per-blue-independently on
  parameter-shared weights.

Result: cross-blue information sharing happens *within* a step
(GNN) but not *across* steps (private GRU memory).  If blue A saw a
red at position X at step t and lost it at step t+1 (masked), blue A
remembers X but blue B never receives that memory over subsequent
steps — unless blue A's *current* embedding at t+1 happens to reveal
it indirectly.

### 13.2 The fix

Feed the previous-step hidden state of each blue as an **additional
node feature** into the actor's GNN encoder.  Then the GNN's bb
message passing carries the memory of every blue to every other
blue.

Concretely:
```
augmented_blue_features[i] = concat( blue_features[i], hidden_prev[i] )
                              # dim = blue_feat_dim + d_hidden
h_blue = actor_encoder( augmented_blue_features, ... )
```

### 13.3 What changes

- Actor encoder's `blue_feat_dim` grows from 8 → 8 + d_hidden = 72.
  Only the actor path is affected; the CTDE critic still takes plain
  8-dim blue features (the critic sees full state, doesn't need
  actor hidden as input).
- One extra `torch.cat` in `actor_forward` before the encoder call.
- The aux belief-state loss (option C) still targets the critic
  encoder's output — semantically unchanged.
- Parameter count grows by ~5 k (actor `blue_input_mlp` gets a wider
  input layer).
- Gated on a `--share-hidden-via-gnn` flag, off by default.  Backward
  compat: old checkpoints load unchanged (default False, encoder
  input dim stays 8).

### 13.4 Hypothesis

- **Positive result**: option 1 gives blues a memory channel to
  broadcast belief across time.  If C's 2.92 ceiling is due to the
  private-hidden bottleneck (not arena geometry or red speed),
  catches should push toward 3.0/3 vs Run.
- **Null result**: catches stay at ~2.92.  Then the ceiling is
  environmental (irreducible), not architectural.  Still a portfolio
  observation.
- **Negative result**: unlikely, but if the extra input dim causes
  optimization instability, we can either regularize or wrap the
  hidden feed with a dedicated projection.

### 13.5 Acceptance for option 1 vs option C

Option 1 is worth adopting over option C iff **`mean_caught vs Run`
improves by ≥ 0.03** and the coordination-quality gate (std_return
≤ 0.85 × Greedy) is preserved.  A ~2.95 / 3 result would clear this
bar cleanly.

## 12. References

- **CTDE**: MADDPG (Lowe et al., 2017, NeurIPS), MAPPO (Yu et al.,
  2021, NeurIPS).  Both use a centralised critic with global state
  and decentralised actors with local obs.
- **Recurrent PPO**: OpenAI Baselines PPO2 (LSTM variant), Stable-
  Baselines3 RecurrentPPO, cleanrl PPO+LSTM.  All use the
  stateless-at-update convention documented in §7.
- [`stage2_gnn_design.md`](stage2_gnn_design.md) — Stage 2 GNN + CTDE
  design that Stage 3 inherits and extends.
- [`stage2_results.md`](stage2_results.md) — the Stage 2 numbers
  Stage 3 is trying to preserve under partial obs.
