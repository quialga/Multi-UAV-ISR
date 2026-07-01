# Stage 2 — GNN over Entities + CTDE Critic (Design)

_Research journal entry, July 2026.  Written after Stage 1 v2 landed
as a soft pass ([`stage1_analysis.md §6b`](stage1_analysis.md))._

Stage 2 is a **controlled architecture-only experiment on the
Stage 1 v2 environment**.  Everything about the world stays fixed
(full observability, mixed red-policy training, 3 blue vs 2 red,
double-integrator kinematics, shared team reward).  The only variable
is the policy/critic architecture: we swap the flat MLP for a GNN
over an entity-typed graph.

This document is the exhaustive spec.  It defines the graph, node
and edge features, message passing rules, both heads, hyperparameters,
and the acceptance criterion.

## 1. Motivation and hypothesis

Stage 1 v2 verdict (see [`stage1_analysis.md §6b`](stage1_analysis.md)):

- Trained MLP catches 2.00/2 reds on every red type — behaviour is
  qualitatively correct.
- Trained MLP sits within 0.78 of GreedyPursuer on the hardest red
  (RunFromNearest: v2 +15.37, Greedy +16.15).
- Misses the strict 1.20× bar by 4.01 points.
- Interpreted as an **architectural ceiling**: shared-parameter MLP
  has no coordination mechanism; three UAVs each execute an
  independent pursuit skill that on average matches Greedy but never
  beats it.

**Hypothesis for Stage 2:** structured architectures that expose
inter-agent relational reasoning (message passing between blue nodes
across a graph) will close the coordination gap and beat the strict
acceptance bar on the *same env*.

**Falsification test:** if the GNN also lands at ~+15 vs Run, the
"architectural ceiling" story is wrong and we need a different
diagnosis (e.g. reward structure, action space granularity, PPO
hyperparameters).  That would be a strong finding worth publishing
too — negative result on a specific hypothesis.

## 2. Graph specification

### 2.1 Node types (2)

| Type | Count | Feature vector (dim) |
|---|---|---|
| **Blue** | `N_blue` (default 3) | `[vel_xy_normalised (2), speed_normalised (1), wall_distances (4), time_remaining (1)]` — **8 D** |
| **Red** | `N_red` (default 2) | `[active_flag (1)]` — **1 D** |

**Rationale for the asymmetry:** node features are *intrinsic* — properties
of the entity that don't depend on who's looking.  Blue's intrinsic
state (own velocity, speed, proximity to walls, remaining episode time)
is meaningful in the world frame.  Red's absolute position and velocity
don't exist as a concept under ego-centric semantics; a red's state is
only meaningful relative to some blue, which lives on the *edges*.
The one bit of intrinsic red state that still matters — "am I still
counted as an active target?" — goes on the node.

### 2.2 Edge types (2)

| Type | Direction | Feature vector (dim) |
|---|---|---|
| **Blue-Blue** | Bidirectional (two directed edges per pair, one each way) | `[rel_pos (2), rel_vel (2), range (1), bearing_cs (2)]` — **7 D** |
| **Red-Blue** | Directed, Red → Blue only | Same 7 D layout |

Where:
- `rel_pos = other_pos − sender_pos` (from sender's frame),
- `rel_vel = other_vel − sender_vel`,
- `range = ||rel_pos||`,
- `bearing_cs = (cos θ, sin θ)` between `sender_vel` and `rel_pos`,
  zeroed when `|sender_vel| < 1e-6` (same convention as
  `PursuitEnv._bearing_features`).

**Rationale for asymmetric bearing across direction:** the bearing
`(cos θ, sin θ)` is defined in the *sender's* body frame — the angle
between the sender's velocity and the vector to the receiver.  For a
blue-blue pair `(i, j)`, the edge `i → j` and the edge `j → i` carry
*different* bearing values because `blue_i` and `blue_j` have different
velocities.  `rel_pos` is also negated between the two directions.
Both edges are needed; treating them as one undirected edge would
lose the ego-centric interpretation.

**Rationale for no Blue → Red edges in Stage 1:** in Stage 1 red is
scripted; it does not process observations and does not compute a
policy.  Adding Blue → Red edges would allocate parameters to a
message pathway that has no consumer.  When red becomes a learned
agent (Stage 5 self-play), we add the reverse direction trivially.

**No Red-Red edges:** red targets don't coordinate in Stage 1.  Same
reasoning as above.

### 2.3 Total graph size

For default `N_blue=3, N_red=2`:
- **5 nodes** (3 blue + 2 red)
- **Blue-Blue edges:** 3 × 2 = 6 directed edges
- **Red-Blue edges:** 2 × 3 = 6 directed edges
- **Total edges:** 12

Trivially small — hand-rolled message passing in pure PyTorch, no
`pytorch-geometric` dependency needed.

## 3. GNN architecture

### 3.1 Input embedding (per-type)

Two small MLPs, one per node type, project raw features into a
shared `d_hidden`-dimensional space.

```
h_blue^(0) = MLP_blue(blue_features)     # (N_blue, d_hidden)
h_red^(0)  = MLP_red(red_features)       # (N_red,  d_hidden)
```

Each MLP is 2 layers with `tanh` activation and orthogonal init (same
convention as the Stage 1 policy).  `d_hidden = 64` — smaller than the
Stage 1 MLP's 256 because per-token compute is amortised across many
tokens and we do 2 rounds of it.

Edge features are also projected once:

```
e_bb = MLP_edge_bb(blue_blue_edge_features)   # (n_bb_edges, d_hidden)
e_rb = MLP_edge_rb(red_blue_edge_features)    # (n_rb_edges, d_hidden)
```

Edges do not update across rounds — only nodes do.

### 3.2 Message passing (2 rounds)

Each round computes messages, aggregates them by receiver, and
updates node embeddings with a residual step.

**Message computation** (per directed edge, per round `k`):

```
m_ij^(k) = MLP_msg([h_sender^(k), h_receiver^(k), e_ij])
```

`MLP_msg` is a 2-layer MLP with `d_hidden` inputs (concatenated) and
`d_hidden` outputs.  A single shared `MLP_msg` is used across edge
types — the type-specific edge embedding at input carries the type
signal.

**Aggregation** (per receiver, **sum** — locked design choice):

```
m_i^(k) = Σ_{j : (j,i) is an edge}  m_ji^(k)
```

Sum preserves count of contributing senders (useful for permutation
invariance while still communicating team size).

**Node update** (residual, per node):

```
h_i^(k+1) = h_i^(k) + MLP_update([h_i^(k), m_i^(k)])
```

Residual keeps early-training gradients well-behaved and lets the
network learn "identity" as a fallback if a round of messaging is
uninformative.

Two rounds by design (locked design choice).  Round 1 gives each
blue a 1-hop view (all teammates + all reds).  Round 2 uses *updated*
embeddings for a second-order view ("my teammate has now processed
that red X is being pursued").

### 3.3 Policy head — decentralised execution

Each blue UAV `i` acts based on **its own post-round-2 node embedding**:

```
mean_i    = MLP_actor(h_blue_i^(2))         # (2,)  action mean
log_std   = learned parameter               # (2,), state-independent
value_i   = ignored for the policy path
```

The policy is a diagonal Gaussian; action = mean at eval, sampled
during rollout collection.  Same output format as Stage 1 v2, so the
existing PPO update code applies with no changes.

Parameter sharing across the three blue UAVs is enforced by the same
`MLP_actor` being applied to all three `h_blue_i^(2)`.  Symmetry is
now provided by the node-type separation — no `self_idx_onehot`
needed (the acting agent is identified by *which node's embedding
we read out*).

### 3.4 Critic head — centralised training

The critic uses global information the policy doesn't have access to.
Locked design choice (A = **sum aggregation over blue nodes**):

```
blue_summary = Σ_i h_blue_i^(2)              # (d_hidden,)
red_summary  = concat(h_red_j^(2) for j in fixed order)
                                             # (N_red * d_hidden,)
state_repr   = concat(blue_summary, red_summary)
                                             # ((1 + N_red) * d_hidden,)
V(state)     = MLP_critic(state_repr)        # (1,)
```

- **Sum** over blue: preserves count of active blues, permutation-
  invariant, matches shared team reward semantics.
- **Concat** over red: red identities matter (which specific red is
  where), and their count is fixed for Stage 1.  Permutation
  invariance across reds is not a Stage 1 concern; a later stage
  with variable red counts would swap this to a sum too.

One centralised V shared across all agents — under shared team
reward this is exactly the right baseline for GAE advantage
computation.  Each blue agent's advantage uses the *same* V(state)
value; only the policy gradient differs per agent (each queries its
own node's action).

## 4. Observation restructure

`PursuitEnv._build_obs` currently returns a flat 38-D vector.  Stage 2
needs a structured obs.  Approach:

- **Add `_build_structured_obs(blue_idx)`** returning a dict:
  ```
  {
      "blue_features":   np.ndarray (N_blue, 8),   # node feats
      "red_features":    np.ndarray (N_red,  1),
      "bb_edges":        np.ndarray (n_bb, 4),     # (sender_idx, receiver_idx, ...)
                                                    # or split arrays; TBD in impl
      "bb_edge_features": np.ndarray (n_bb, 7),
      "rb_edges":        np.ndarray (n_rb, 4),
      "rb_edge_features": np.ndarray (n_rb, 7),
      "acting_agent_idx": int,                      # index into blue nodes
  }
  ```
- **Keep `_build_obs` (flat) unchanged** as the v2 baseline path so the
  Stage 1 v2 MLP policy stays loadable and the head-to-head comparison
  is fair.
- The `VectorPursuitEnv` and PPO buffer need to handle dict obs at
  Stage 2 (dict-of-tensors), not just a flat tensor.  The rollout
  storage becomes one buffer per obs key; minibatches yield dicts too.

Exact obs plumbing decision comes with the implementation turn.

## 5. Hyperparameters (default Stage 2 config)

Same env + PPO as Stage 1 v2 unless noted:

| Parameter | Value | Note |
|---|---|---|
| `d_hidden` | 64 | node embedding dim |
| `n_msg_rounds` | 2 | locked |
| `blue_aggregation` (critic) | sum | locked |
| MLP layer counts (blue_emb, red_emb, edge_emb, msg, update, actor, critic) | 2 each | matched-compute discipline |
| Activation | Tanh | consistent with Stage 1 |
| Init | Orthogonal, gain √2 hidden, 0.01 policy head, 1.0 value head | matches Stage 1 |
| PPO clip_eps | 0.2 | unchanged |
| PPO ent_coef | 0.01 | unchanged |
| PPO n_rollouts | 1000 | unchanged (adjustable if convergence timing differs) |
| red_policy_mix | `stationary:1,random:1,run:1` | unchanged |

Approximate parameter count target: **60-80k** (matched to Stage 1
v2's 76.5k so the architecture comparison is compute-fair).

## 6. Acceptance criterion

**Same env, same red-policy mix, same eval protocol as Stage 1 v2**:
50 episodes per (blue, red) cell, deterministic actions, matched
seeds across blue policies.

Two thresholds:

1. **Strict:** trained blue vs `run_from_nearest_uav` ≥ +19.38
   (1.20 × GreedyPursuer's +16.15).  This is the Stage 1 bar that
   the v2 MLP missed by 4.01.
2. **Soft:** trained blue vs `run_from_nearest_uav` ≥ +18.00 (beats
   Greedy by ≥ 1.85 points).  Would demonstrate coordination without
   quite clearing the 20% margin.

**Additional required checks:**
- OOD generalisation preserved: vs Stationary ≥ Greedy − 1.0
  (i.e. ≥ +16.65).  Regression here would suggest the GNN overfits
  to the harder red types.
- 2.00/2 caught across every red type (no dropped-catch regression).

## 7. Ablations to run

Recorded here so we don't skip them in the write-up:

1. **1 round vs 2 rounds** of message passing (test whether
   second-order reasoning matters).
2. **Sum vs mean vs attention** aggregation of blue nodes in the
   critic.
3. **With vs without red_features on node** (does the active_flag
   need its own embedding, or is edge-feature-only enough?).
4. **d_hidden sweep** (32, 64, 128) — compute-vs-quality trade-off.

Only ablation 1 is a required part of the Stage 2 write-up; the
others are optional depending on how much compute we spend.

## 8. Implementation plan + status

1. **[x] Doc first** (this file + `design.md` re-numbering).
2. **[x] Env obs restructure:** `PursuitEnv._build_structured_obs()`
   and public `structured_observation()` accessor
   ([`isr/env/pursuit_env.py`](../isr/env/pursuit_env.py)); five
   graph-invariant smoke tests all passing (shapes, negation on
   bidirectional bb edges, rb rel_pos convention, zeroed edges out
   of caught reds, wall-distance layout).
3. **[x] GNN policy:** [`isr/agents/gnn_policy.py`](../isr/agents/gnn_policy.py)
   ~220 lines pure PyTorch.  Per-type input embedding MLPs, per-type
   edge embedding MLPs, shared message MLP, shared residual update
   MLP, decentralised actor head reading acting agent's blue node
   embedding, CTDE critic head (sum-of-blue + concat-of-red).
   `get_action_and_value(obs_dict, action=None)` returns
   `(action, log_prob, entropy, value)` with shapes
   `(B, N_blue, 2)`, `(B, N_blue)`, `(B, N_blue)`, `(B,)`.  **64k
   params** vs MLP's 76.5k — matched-compute discipline holds.
4. **[x] Buffer + vec_env changes:**
   - New [`isr/train/graph_buffer.py`](../isr/train/graph_buffer.py)
     `GraphRolloutBuffer` stores dict obs per key
     `(T, E, n_tokens, feat_dim)`, per-agent actions/log_probs, per-env
     values/rewards/dones.  GAE computed per env from single V trace;
     advantages broadcast per-agent in the loss.
   - [`isr/train/vec_env.py`](../isr/train/vec_env.py) got an
     `obs_format="flat"|"structured"` parameter.  Structured mode's
     reset/step return the batched graph dict + per-env reward
     `(n_envs,)`.  Flat mode unchanged.
   - New `ppo_update_graph` in
     [`isr/train/ppo.py`](../isr/train/ppo.py) — same clipped
     surrogate math as flat `ppo_update` with per-agent
     broadcasting of advantages against the per-agent ratio.
5. **[x] Training script:**
   [`scripts/train_stage1.py`](../scripts/train_stage1.py) gained
   `--policy-type {mlp,gnn}` (default mlp), `--d-hidden`,
   `--n-msg-rounds`.  A `_run_gnn_training` helper packages the
   full GNN loop (structured vec_env, GNNActorCritic,
   GraphRolloutBuffer, ppo_update_graph).  `main()` dispatches
   right after vec_env setup; MLP path unchanged.
6. **[x] Smoke test:** 5-rollout CPU smoke run passed —
   `Policy: GNNActorCritic d_hidden=64 rounds=2 params=64005`,
   KL ~0.003, clip_frac ~3%, entropy stable ~2.85, no crashes.
   Full evaluation pipeline (load → dispatch → eval matrix → GIFs
   → markdown writeup) also verified against the smoke checkpoint.
7. **[ ] Full training + eval:** 1000-rollout GNN training on GPU
   + head-to-head evaluation matrix (v2 MLP vs GNN, 50 episodes/
   cell, same seeds).  Auto-writes `docs/stage2_results.md`.  This
   is the user's turn on the pod; see README §
   *"Training on GPU (RunPod or similar)"* for the exact commands.

Two subsequent cleanups already landed as part of the
implementation:
- [`isr/agents/policy_loader.py`](../isr/agents/policy_loader.py)
  now auto-dispatches to `GNNActorCritic` or `ActorCritic` based on
  the checkpoint's saved `policy_type`; `TrainedBlueAgent` detects
  the GNN case via `isinstance` and pulls graph obs from
  `env.structured_observation()`.
- [`scripts/evaluate_trained.py`](../scripts/evaluate_trained.py)'s
  `--results-md` default is now policy-type-aware: MLP checkpoints
  overwrite `docs/stage1_results.md`, GNN checkpoints overwrite
  `docs/stage2_results.md`.  Explicit override still supported.

## 9. Expected outcomes

Two directions are plausible; both are informative.

**If GNN clears the strict bar:** hypothesis confirmed.  The Stage 1
gap was architectural.  Write-up story: "identified the ceiling,
predicted the fix from literature, measured the improvement in a
controlled comparison."  Then Stage 3 (POMDP + GRU on the GNN)
becomes the natural next step.

**If GNN lands within ±1 of the MLP (~+15):** hypothesis falsified.
The ceiling isn't architectural — something else limits Stage 1 v2's
performance.  Candidate diagnoses to test next:
- Reward shaping (catch bonus too coarse to distinguish fast vs slow
  catches?);
- Action space granularity (2D continuous is fine but maybe too
  reactive?);
- PPO hyperparameters (entropy too low so policy stops exploring
  coordination strategies?).

Either outcome is a real result and worth writing up.

## 10. References

- [TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced RL (arXiv:2503.12395)](https://arxiv.org/abs/2503.12395) — the transformer sibling of our GNN; same
  entity-typed structured-obs philosophy at scale.
- MADDPG / MAPPO literature — reference for the CTDE (Centralised
  Training, Decentralised Execution) pattern used in our critic
  design.
- `docs/stage1_analysis.md` — Stage 1 v1 → v2 upgrade story and the
  soft-pass verdict this stage aims to test.
