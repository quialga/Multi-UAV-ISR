# Multi-UAV ISR — Design Document

Specification for the multi-stage curriculum.  Stage 1 is detailed
exhaustively; Stages 2-7 are sketched and will be detailed when their turn
arrives so we don't over-design upfront.

## 1. Conceptual setup

A **blue team** of `N_blue` UAVs operates over a 2D arena and must locate,
track, and maintain custody of a **red team** of `N_red` mobile ground
targets within a bounded time budget.

Across stages we add adversarial complexity, partial observability, and
coordination friction.  The mathematical problem stays the same; what
changes is which information is hidden and which adversary is learned.

### Vocabulary

- **Blue (UAV / predator):** the team we usually train.
- **Red (target / evader):** the team being tracked.  Stage 1: fixed
  scripted policy.  Stage 4 onwards: learned.
- **Capture / custody:** a red target is "in custody" if at least one blue
  UAV is within `capture_radius` of it.  In Stages 1-3 we model this as a
  one-shot event ("caught"); from Stage 6 onwards custody must be
  maintained, not just achieved once.
- **Sensor radius:** the distance a blue UAV can see (introduced Stage 2).
  Stage 1 = infinite.
- **Comms radius:** the distance between two blue UAVs over which they can
  exchange messages (introduced Stage 5).  Before Stage 5, blue agents
  share a centralised observation.

## 2. Curriculum overview

Each stage has:
- A **concept added** (the learning objective).
- A **new mechanic in the env** (the surface that exercises the concept).
- An **acceptance criterion** (a measurable check that the concept works).

| Stage | Concept | Env mechanic | Acceptance |
|---:|---|---|---|
| 1 | Actor-critic on a continuous-action multi-agent task; vectorised rollouts | Pursuit-evasion, full observability, fixed-policy red, flat-MLP shared policy | Trained blue beats GreedyPursuer baseline by ≥20% mean episode reward |
| **2** | **Structured architecture — GNN over entities with CTDE critic** | **Same env as Stage 1 (full obs, fixed red mix).  Isolates architecture as the sole variable vs Stage 1 baseline** | **GNN blue closes the Stage 1 coordination gap: trained_vs_run ≥ +19.38 (the strict 1.20× Greedy bar that Stage 1 v2 MLP missed by 4.01)** |
| 3 | POMDP + recurrent policies | Sensor radius (finite, no noise yet); GRU on top of the Stage 2 GNN | GRU policy beats non-recurrent Stage 2 GNN under matched compute at fixed sensor radius |
| 4 | State estimation under sensor noise; auxiliary belief loss | Gaussian sensor noise + occlusion by terrain; predictor head | Belief loss correlates with policy performance; visualised belief tracks ground truth |
| 5 | Self-play, non-stationarity, exploitability | Red team learned; fictitious self-play / league training | Trained blue beats *every* historic red policy in the league, not just the latest |
| 6 | Differentiable communication | Range-gated message passing; learned messages | Learned-comm blue beats no-comm blue under information-limited regimes |
| 7 | Non-stationary objectives, custody-over-time | Moving targets, appearing/disappearing, priority weights | Mean custody-time-ratio improves vs Stage 6 baseline |
| 8 | Robust evaluation | Held-out red policies, distribution-shift sweeps | Worst-case episode reward across held-out opponents ≥ X% of in-distribution reward |

Curriculum is 8 stages (up from 7).  Stage 2 was originally "POMDP +
recurrent"; after Stage 1's soft-pass verdict (see
[`stage1_analysis.md §6b`](stage1_analysis.md)), the curriculum was
resequenced to test the "architectural ceiling" hypothesis as a
standalone step *before* introducing partial observability.  This
isolates architecture as the sole variable in the Stage 1 → Stage 2
comparison and gives us two separable results instead of one lumped
one.  Stages 3-8 (previously 2-7) shift by one; concepts unchanged.

## 3. Stage 1 — Pursuit-Evasion baseline (detailed spec)

### 3.1 Goal

Get a clean, fast, vectorisable MARL training pipeline working on the
**simplest possible version** of the problem: 2D continuous, fully observable,
fixed-policy red team, parameter-shared PPO blue team.

Once this works, every subsequent stage swaps one element while keeping the
rest constant.

### 3.2 Arena and kinematics

- **Arena:** axis-aligned square `[0, L] × [0, L]`, `L = 100.0` (arbitrary
  units).  Agents are confined to the box; positions clipped at the edges and
  velocity component reflected on contact (simple wall bounce).
- **Time:** discrete steps, `dt = 1.0`.  Episode length cap: `max_steps = 200`.
- **Entity state:** each entity has position `p ∈ R²` and velocity `v ∈ R²`.
- **Action:** continuous 2D acceleration command `a ∈ [-1, 1]²`.
- **Dynamics (per step):**
  ```
  v_next = clip(v + a · dt, -v_max, v_max)
  p_next = clip(p + v_next · dt, 0, L)
  ```
  with `v_max` per-entity-class (blue UAVs slightly faster than red targets so
  pursuit is actually possible — see §3.6).
- **Wall handling:** when `p_next` hits `0` or `L` on an axis, the velocity
  component on that axis is set to zero (no bounce — simpler to reason about
  than reflection, and the agent learns to back off).

### 3.3 Capture

- A red target is **caught** the moment it enters within `capture_radius` of
  any blue UAV.
- Caught targets are removed from the simulation (deactivated).
- Episode terminates when **all** red targets are caught, or `max_steps` is
  reached (whichever first).

### 3.4 Observation (Stage 1 = full observability)

Each blue UAV receives the **same global observation** (parameter sharing,
centralised observation).  Flattened vector:

```
[blue_positions   : N_blue * 2 floats, normalised to [0,1] via /L
 blue_velocities  : N_blue * 2 floats, normalised by /v_max_blue
 red_positions    : N_red  * 2 floats, normalised /L  (zero-padded for caught)
 red_velocities   : N_red  * 2 floats, normalised /v_max_red (zero, caught)
 red_active_mask  : N_red  * 1 floats, 1.0 if active, 0.0 if caught
 self_idx_onehot  : N_blue * 1 floats, 1.0 at this agent's index, 0 elsewhere
 time_remaining   : 1 float, (max_steps - t) / max_steps]
```

`self_idx_onehot` is the trick that lets us **share parameters** across blue
UAVs while still letting each agent know which one of N_blue it is.  In
Stage 2 we'll switch to per-agent local observations and this trick goes
away.

### 3.5 Action

Per blue UAV: continuous `Box(-1, 1, shape=(2,))`.  Red team is scripted in
Stage 1 (see §3.6).

### 3.6 Red team policy (fixed in Stage 1)

`RunFromNearestUAV`:
```
For each red target:
    find nearest active blue UAV (in Euclidean distance)
    action = unit_vector(red_pos - nearest_blue_pos)
    (capped at action magnitude 1.0)
```

Red `v_max_red = 1.0`.  Blue `v_max_blue = 1.5`.  This ratio guarantees pursuit
is possible in principle but not trivial (red can dodge for a long time
against a single UAV; coordination among blue UAVs matters).

### 3.7 Reward (Stage 1: shared team reward)

Per step, the **whole blue team** receives:

```
r_step = sum_{red caught this step} (+10.0)
       + sum_{blue UAVs}  (-0.01 * ||action||²)    [action cost]
       - 0.05                                      [step cost]
```

Plus at terminal step:
```
r_terminal = -5.0 * (number of red targets still uncaught)
```

The same `r_step` is given to every blue UAV in the same step (parameter-
shared agents see the same return).  We move to per-agent / role-asymmetric
rewards in later stages.

### 3.8 Episode termination

- All red caught → terminate, episode-success.
- `t >= max_steps` → truncate, with terminal penalty for uncaught targets.

### 3.9 Default Stage 1 hyperparameters

| Parameter | Value |
|---|---|
| `N_blue` | 3 |
| `N_red` | 2 |
| `L` (arena side) | 100.0 |
| `dt` | 1.0 |
| `max_steps` | 200 |
| `v_max_blue` | 1.5 |
| `v_max_red` | 1.0 |
| `capture_radius` | 3.0 |

PPO:

| Parameter | Value |
|---|---|
| Policy net | MLP, hidden = [256, 256], tanh |
| Value net | MLP, hidden = [256, 256], tanh (separate head) |
| Optimiser | Adam, lr = 3e-4 |
| Clip ratio | 0.2 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| Batch size | 4096 (env steps per rollout) |
| Minibatch size | 256 |
| Epochs / rollout | 10 |
| Entropy coef | 0.01 |
| Value coef | 0.5 |
| Max grad norm | 0.5 |
| Total rollouts | 100 (smoke); 1000 (Stage 1 acceptance) |

### 3.10 Acceptance criterion (Stage 1)

After training: `mean_episode_return(trained_blue) ≥ 1.2 × mean_episode_return(GreedyPursuer)`.

`GreedyPursuer` is the obvious non-learned baseline: each UAV always
accelerates toward the closest active red target.  Beating it by 20%
demonstrates that the policy learns **coordination** (e.g., splitting
targets, herding), not just "go to the nearest red".

### 3.11 Non-goals for Stage 1

- No sensor radius limits (Stage 2).
- No sensor noise or occlusion (Stage 3).
- No learned red (Stage 4).
- No inter-agent comms (Stage 5).
- No moving objectives that change priority mid-episode (Stage 6).
- No formal exploitability evaluation (Stage 7).

## 4. Tech stack decisions

- **Python 3.10+, PyTorch ≥ 2.0** — standard.
- **Env API: PettingZoo `ParallelEnv`** — covers multi-agent.  We don't use
  the AEC API because step semantics for synchronous multi-agent are
  cleaner with parallel.
- **`gymnasium.spaces`** for action / obs space typing.
- **NumPy** for env kinematics — vectorise over entities, avoid Python
  per-entity loops.
- **Vectorisation:** for training, wrap N copies of the env in a vector
  wrapper that returns batched obs / takes batched actions.  Run on CPU
  with N=8-16; GPU for the policy net only.
- **Logging:** stdout per-rollout summary + TensorBoard (`tensorboard --logdir runs/`).
- **Checkpoints:** `runs/{stage}/{timestamp}/checkpoint_{step}.pt` plus a
  `best.pt` symlink updated on improvement.
- **Rendering:** matplotlib in Stage 1.  We don't need anything fancier until
  Stage 4+ when self-play episodes become genuinely interesting to watch.

## 5. Stages 2-7 (sketches, expand later)

- **Stage 2 — Structured architecture (GNN + CTDE critic):** same
  environment as Stage 1 (full observability, mixed-red training
  distribution).  Replace the flat MLP with a GNN over an
  entity-typed graph (blue and red nodes; blue-blue bidirectional
  edges + red-blue directed edges), 2 rounds of message passing,
  residual node updates.  Policy head reads the acting agent's blue
  node embedding.  Critic head sums blue embeddings and concatenates
  red embeddings — a centralised value estimate consistent with the
  shared team reward (CTDE).  Full detailed spec in
  [`docs/stage2_gnn_design.md`](stage2_gnn_design.md).  The natural
  falsification test of the Stage 1 v2 verdict: if the architectural-
  ceiling story is right, GNN should close the +4.01 gap.
- **Stage 3 — Sensor radius:** observation becomes per-UAV local
  (relative-position features of entities within `sensor_radius`, zero
  padded).  Recurrent policies (GRU) integrate observations over time.
  Reference architecture is the Stage 2 GNN with GRU on the node
  embeddings.  Compare MLP vs GRU under matched compute.
- **Stage 4 — Sensor noise + occlusion:** sensor reports
  `p_observed = p_true + N(0, σ²)`, occluded by simple terrain blocks.
  Add an explicit belief-state predictor head trained with auxiliary
  loss reconstructing true target positions from observation history.
- **Stage 5 — Self-play:** swap scripted red for a learned policy.
  Naive self-play (latest vs latest) is unstable; use **fictitious
  self-play** (sample opponents from history) or **PSRO / league
  training** (maintain a population of past opponents, train against
  the meta-game).  Track **exploitability** as the headline metric.
- **Stage 6 — Comms:** introduce a discrete message vocabulary or
  continuous message vector that blue UAVs can broadcast within a
  `comms_radius`.  Learn the message protocol differentiably.  Compare
  no-comm, fixed-protocol comm (designed by hand), and learned comm.
- **Stage 7 — Dynamic objectives:** targets move with learned policies,
  new targets spawn at random rates, custody must be **maintained**
  (not just achieved once).  Priority weighting (VIP targets worth
  more).  Reward structure shifts to integral-of-custody.
- **Stage 8 — Robustness eval:** train a "blue under test", evaluate
  against a held-out population of red policies (different scripted
  strategies, different historic self-play checkpoints).  Report mean
  and **worst-case** reward across the held-out set.  Implement a
  simple regret / exploitability estimator.

## 5b. Note on MLP removal (July 2026)

The flat-MLP + parameter-sharing baseline that Stage 1 was built on
was **removed from the codebase** after the Stage 2 scaling
experiment.  Justification:

- At N=3 vs M=2 the MLP was a soft pass (matches Greedy across the
  red distribution — see [`stage1_analysis.md §6b`](stage1_analysis.md)).
- At N=5 vs M=3 the MLP **structurally fails**: catches only 2.12 of
  3 reds on average against `run_from_nearest_uav`, policy collapse
  visible in the training log (entropy negative, KL blown past 0.2,
  clip_frac 0.63).  See [`stage2_results.md §2.3`](stage2_results.md).
- The MLP is not competitive at the scales the curriculum runs on
  from Stage 2 onwards.  Maintaining two policy paths (MLP + GNN)
  in the trainer / vec_env / policy loader added complexity for
  code that was never going to be used.

Historical eval numbers are preserved in `docs/stage1_results.md`,
`docs/stage1_analysis.md`, and `docs/stage2_results.md`.
Pre-Stage-2 MLP checkpoints raise a clear error at load time; the
`.pt` files themselves are still under `runs/stage1/`.

## 6. Cross-cutting decisions

- **Parameter sharing among blue UAVs:** YES in Stage 1.  We may relax in
  later stages if role specialisation matters.  When parameter-shared, we
  feed a `self_idx_onehot` feature to let the policy know which agent it
  is.
- **Reward sharing:** shared team reward in Stage 1.  Per-agent rewards
  (credit assignment) are a Stage 4+ concern.
- **Centralised training, decentralised execution (CTDE):** the value
  function in PPO can use centralised obs while the policy uses local obs.
  In Stage 1 obs is fully centralised so this doesn't kick in; from Stage 2
  the value function gets the centralised view, the policy gets the
  agent's local view.

## 7. Open questions (revisit as we go)

- **Reward shaping intensity:** Stage 1 has a tiny step cost.  If it
  dominates, agents learn to do nothing.  If absent, episodes drift.  May
  need a knob; defer until Stage 1 training results land.
- **Self-play algorithm choice:** fictitious self-play vs PSRO vs PFSP.
  Decide at Stage 4 start based on compute budget.
- **Belief representation in Stage 3:** explicit `(mean, var)` over each
  target's position vs implicit recurrent state.  Explicit is better for
  interpretability; implicit is sometimes better-performing.  A/B at
  Stage 3.
