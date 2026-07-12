# Stage 4 — Belief maps + sensor noise + obstacles + occlusion

Design spec for the fourth stage of the Multi-UAV ISR curriculum.
Extends the Stage 3 CTDE recurrent policy with a per-UAV occupancy-
grid belief map maintained via classical Bayesian log-odds updates
(Moravec & Elfes, 1985/1989), sensor noise, static circular obstacles,
and ray-cast occlusion.

See [`stage3_results.md`](stage3_results.md) for the Stage 3 baseline
this stage extends, and [`stage4_backlog.md`](stage4_backlog.md) for
deferred items (moving obstacles, cone sensor, crash penalties,
learnable sensor models, etc.).

## 1. Motivation and hypothesis

Stage 3 established that a recurrent CTDE actor with cross-blue hidden
sharing can coordinate under partial observability when the observation
is a clean per-edge visibility mask.  Real ISR platforms don't get
clean masks; they get **noisy, occluded observations from a physical
sensor**, and the drone must fuse these into a coherent world model
before it can act.

**Hypothesis (H4):** the classical Bayesian occupancy grid + a small
CNN encoder is sufficient state representation for a GNN-based
multi-UAV policy to solve pursuit under sensor noise, occlusion, and
obstacles.  No learned belief-state predictor, no normalizing flow, no
end-to-end map learning — just principled Bayesian sensor fusion and a
learned policy on top.

If H4 holds, later stages can extend the map (learned dynamics, sensor
model calibration) with a clear "does the extension actually help?"
ablation against this baseline.  If H4 fails, we know the classical
occupancy grid is insufficient and we need learned belief mechanisms
(FORBES, MCOP-style fusion, etc.).

## 2. What changes vs Stage 3

| Aspect | Stage 3 | Stage 4 |
|---|---|---|
| Env observability | Per-edge visibility masks based on `sensor_radius` | Full physical sensor: noise + ray-cast occlusion |
| Red representation | Red nodes in GNN + `rb_edges` | Reds only in the belief map's enemy channel |
| Obstacles | None | 4 (default) static circular obstacles, radii `[5, 15] m` |
| Belief mechanism | Implicit via GRU hidden | Explicit 2-channel occupancy grid per UAV |
| Sensor shape | Disk | Disk (cone deferred to backlog) |
| Ally comms | Sensor-gated `bb_edge_visible` | Unconditional GPS (`bb_edge_visible ≡ 1`) |
| Warm-start critic | From Stage 2 `scaling_gnn/best.pt` | Cold-start (Stage 3 checkpoint incompatible with new obs dict + obstacle geometry) |
| Aux loss | Belief-state distillation via critic encoder | Diagnostic-only BCE(belief, truth), no gradient |

## 3. Environment additions

### 3.1 Static obstacles

Circular obstacles placed at env reset via rejection sampling.

- `n_obstacles`: default 4, CLI overridable.
- `obstacle_radius_min = 5 m`, `obstacle_radius_max = 15 m`.  Each
  obstacle draws its radius uniformly from `[min, max]`.
- Placement: uniform position in the arena, rejected if any of:
  * Overlaps an existing obstacle (touching allowed within 1 m).
  * Overlaps a blue or red spawn zone (10 m clearance).
  * Center closer than `obstacle_radius_max` to any arena wall.
- Positions and radii stored as `_obstacle_pos: (n_obs, 2)` and
  `_obstacle_r: (n_obs,)` on the env.  Fixed for the episode.
- Kinematics: blues and reds cannot move into an obstacle.  Attempted
  motion into an obstacle is *clipped* to the obstacle boundary (soft
  reflection; crash penalty is a backlog item).

### 3.2 Sensor model — noisy Bernoulli detection per cell

For each channel `c ∈ {enemy, obstacle}` and each cell `(x, y)` inside
the UAV's sensor disk that is NOT occluded, the sensor returns a
detection with probability:

$$
P(\text{detect} \mid c, x, y) = \begin{cases}
p_{TP} = 0.85 & \text{if channel } c \text{ actually present in cell } (x, y) \\
p_{FP} = 0.15 & \text{otherwise}
\end{cases}
$$

Same values for both channels in v1; per-channel differentiation is
in the backlog.

### 3.3 Occlusion — ray-cast from UAV to cell

For each candidate cell `(cx, cy)` in the sensor disk of UAV `i` at
position `(ux, uy)`:

```
n_steps = ceil(dist / step_size)     # step_size = 2.5 m
for k in 1..n_steps - 1:
    p = (ux, uy) + (k/n_steps) * ((cx, cy) - (ux, uy))
    if p is inside any obstacle disk:
        cell is occluded — skip observation this step
        break
```

Rays passing THROUGH an obstacle stop at the obstacle boundary.  The
first obstacle cell along the ray is still observable (it produces a
positive observation on the obstacle channel via the log-odds update
below), but cells past it are not.

### 3.4 Bayesian log-odds belief update — the whole rule

Per-UAV per-channel per-cell log-odds tensor `L[i, c, x, y]` initialised
to 0 (uniform prior, P=0.5).

For each `(c, x, y)` observed by UAV `i` this step:

$$
\Delta L_{\text{detect}} = \log\frac{p_{TP}}{p_{FP}} = \log\frac{0.85}{0.15} \approx +1.735
$$
$$
\Delta L_{\text{no detect}} = \log\frac{1 - p_{TP}}{1 - p_{FP}} = \log\frac{0.15}{0.85} \approx -1.735
$$

```
if sensor reported "detect" on cell (c, x, y):
    L[i, c, x, y] += 1.735
else:
    L[i, c, x, y] -= 1.735
L[i, c, x, y] = clip(L[i, c, x, y], -10.0, +10.0)
```

Cells not observed this step retain their previous log-odds — natural
memory of past evidence with no decay.  A future extension may add an
optional forgetting factor (`L *= 0.99`) for concept-drift robustness.

At consumption time, the policy either reads log-odds directly (they
form a well-conditioned input for a CNN) or converts to probability via
`P = sigmoid(L)` for the diagnostic loss.

### 3.5 True occupancy grid (training-side only)

For the CTDE critic and the diagnostic BCE metric, the env computes the
current true occupancy:

```
true_occupancy[0, x, y] = 1  iff  any active red is inside cell (x, y)
true_occupancy[1, x, y] = 1  iff  cell (x, y) is inside any obstacle
```

Shape `(2, H, W) = (2, 26, 26)`.  Provided as an obs-dict key exposed
only for training and eval logging; the actor never reads it.

### 3.6 Grid resolution locked

Arena `130 × 130 m`, cell size `5 m` ⇒ `H = W = 26`.  Rationale:
`capture_radius = 3 m` so each red typically fits in one cell, and
the 26×26 grid keeps belief tensors under 40 KB per UAV.

## 4. Observation dict schema

### 4.1 Removed from Stage 3

- `red_features`
- `rb_edge_features`
- `rb_edge_visible`
- `bb_edge_visible`  (choice B — always 1 for GPS ally comms)

### 4.2 Retained from Stage 3

- `blue_features` — `(N_blue, 8)`.  Unchanged: `(x, y, vx, vy, d_wall_N,
  d_wall_S, d_wall_E, d_wall_W)`.
- `bb_edge_features` — `(n_bb, 7)`.  Unchanged edge geometry between
  every pair of blues.  Interpreted as continuous GPS-style comms
  (radio range treated as unbounded within the arena — see §7 of
  `stage4_backlog.md` for the deferred `comms_radius` option).

### 4.3 New for Stage 4

- `belief_maps` — `(N_blue, 2, 26, 26)` float32.  Per-UAV log-odds
  clipped to `[-10, +10]`.
- `obstacle_positions` — `(N_obstacles, 3)` float32.  `(x, y, radius)`.
  Static within an episode; blues can read it directly (assumed known
  a priori — think "pre-mission satellite briefing").  Reds do not
  see it (they use `run_from_nearest_uav` which doesn't care).
- `true_occupancy` — `(2, 26, 26)` float32 binary.  Training + eval
  logging only.  Actor MUST NOT read this key (asserted in policy
  forward).

## 5. Policy architecture

### 5.1 Belief CNN encoder — shared per-UAV

Small ConvNet that reduces each UAV's `(2, 26, 26)` log-odds tensor to
a 64-dim embedding:

```
Conv2d(2  → 16, kernel=3, stride=2, padding=1)   → (16, 13, 13)
ReLU
Conv2d(16 → 32, kernel=3, stride=2, padding=1)   → (32, 7, 7)
ReLU
Flatten                                          → (1568,)
Linear(1568, 64)                                 → (64,)
```

~110 k params.  Shared weights across UAVs (invariant to UAV
identity).  Called `belief_encoder`.

### 5.2 Actor — GNN + GRU (extends Stage 3 option 1)

Per-UAV node feature construction:

```
h_blue_input[i] = concat(
    blue_features[i],       # 8
    hidden_prev[i],         # 64 (Stage 3 option 1 hidden-in-GNN)
    belief_encoder(belief_maps[i]),   # 64
)                           # → 136 dim per node
```

The GNN encoder then runs as in Stage 3 (2 rounds, shared MLPs) over
these 136-dim node features + the 7-dim `bb_edge_features`.  Output
per UAV → GRUCell → shared actor MLP → mean action.  `log_std` still
a state-independent per-dim parameter.

### 5.3 Critic — cold start, sees true occupancy

Fully centralised critic, cold-started (random init).  Consumes the
same 8-dim `blue_features` per UAV plus a CNN encoding of
`true_occupancy` (not the belief maps):

```
h_blue_critic[i] = concat(
    blue_features[i],             # 8
    critic_belief_encoder(true_occupancy),   # 64  ← global, one per env-step
)
```

Node features → GNN encoder (separate weights from actor) → sum over
blue nodes + concat with the global true-occupancy embedding →
critic_trunk → critic_head → V(s).

**Why no warm-start:** Stage 3 checkpoints have `red_input_mlp`,
`rb_edge_mlp`, and `critic_trunk` sized for the Stage 3 obs dict.  The
Stage 4 critic's input dimensionality, parameter names, and semantics
differ enough that byte-safe copy would net less than a third of
tensors even at best.  Empirically cold-start converges within ~200
rollouts on this scale (Stage 3 headline was 385).

### 5.4 What the policy MUST NOT see

Actor path assertions at construction:
- `belief_encoder` is called with `belief_maps` only, never
  `true_occupancy`.
- `critic_belief_encoder` is a separate instance of the same
  architecture with different weights, and it's called only on
  `true_occupancy`.
- Unit test verifies gradient flow: `true_occupancy` in the obs dict
  produces zero gradient into the actor.

## 6. PPO update changes

Same clipped objective as `ppo_update_ctde` from Stage 3.  Two
differences:

- New diagnostic scalar `belief_bce` logged per rollout (see §7).
- Aux loss removed — the empirical winner in Stage 3 was
  `--aux-hidden-coef 0.2 --freeze-critic`, but freeze-critic requires
  a warm-started critic (which we're not doing here).  Aux loss can
  come back in Stage 4 v2 once the baseline is characterised.

## 7. Diagnostic BCE metric

At each rollout, compute two BCEs and log to TensorBoard:

```
per_uav_bce = mean_over_i BCE(sigmoid(belief_maps[i]), true_occupancy)
fused_bce   = BCE(sigmoid(sum_i belief_maps[i]), true_occupancy)
```

- `per_uav_bce` measures the quality of each UAV's local belief.
- `fused_bce` measures the quality of the log-odds-summed shared
  belief — the "team belief" that would emerge from Bayesian fusion.

Both are **diagnostic only** — no backward pass, no gradient into any
network parameter.  Interpretation:
- If `per_uav_bce` stays high, the sensor model is miscalibrated OR
  UAVs aren't covering the arena.
- If `fused_bce` << `per_uav_bce`, the multi-UAV Bayesian fusion is
  doing what it should — combining independent observations sharpens
  the shared belief.

## 8. Locked Stage 4 hyperparameters

Inherits Stage 3 defaults for shared knobs.  New keys:

```python
STAGE4_DEFAULTS = {
    **STAGE3_DEFAULTS,   # inherits scale, PPO knobs, GNN dims, etc.

    # ----- Belief map ----------------------------------------------------
    "belief_grid_size":  26,       # H = W (locked to arena / cell_size)
    "belief_channels":   2,        # {enemy, obstacle}
    "belief_clip":       10.0,     # log-odds clip range

    # ----- Sensor model --------------------------------------------------
    "p_TP":              0.85,
    "p_FP":              0.15,
    "sensor_shape":      "disk",   # cone is a backlog item
    "sensor_radius":     40.0,     # unchanged from Stage 3
    "ray_step_size":     2.5,      # metres

    # ----- Obstacles -----------------------------------------------------
    "n_obstacles":       4,
    "obstacle_radius_min": 5.0,
    "obstacle_radius_max": 15.0,

    # ----- Ally comms ----------------------------------------------------
    "bb_edge_visible_always_on": True,   # choice B — GPS uplink

    # ----- Policy --------------------------------------------------------
    "belief_encoder_out_dim": 64,

    # ----- Warm start ----------------------------------------------------
    # Stage 3 checkpoints are incompatible with the Stage 4 obs dict +
    # critic input shape.  Cold-start the critic.
    "warm_start_critic": None,

    # ----- Aux loss ------------------------------------------------------
    # Off for the Stage 4 baseline; may return in a follow-up.
    "aux_hidden_coef": 0.0,
    "freeze_critic":   False,
}
```

## 9. Implementation plan (checklist for the coding turn)

1. **Env — obstacles**: add `n_obstacles`, `obstacle_radius_min/max`
   params.  Rejection-sampling placement at reset.  Kinematic
   clipping in `_step`.  Store `_obstacle_pos`, `_obstacle_r`.
2. **Env — belief maps**: allocate `_belief_maps: (N_blue, 2, H, W)`.
   Zero at reset.  New method `_update_belief_maps()` called at end
   of `_step`.
3. **Env — ray-cast occlusion**: helper `_cell_occluded(uav_pos, cell,
   obstacles) -> bool`.  Used by belief update to skip occluded cells.
4. **Env — sensor step**: for each blue, for each cell in sensor disk,
   compute occlusion, roll Bernoulli detection with `p_TP` / `p_FP`
   depending on ground truth, apply log-odds evidence.
5. **Env — obs dict**: drop `red_features`, `rb_edge_*`,
   `bb_edge_visible`.  Add `belief_maps`, `obstacle_positions`,
   `true_occupancy`.  Update `structured_partial_observation`.
6. **Env — smoke tests**:
   - Belief map has no NaNs after 200 steps.
   - Occluded cells never updated even with UAV within `sensor_radius`.
   - `per_uav_bce` monotonically decreases (or stabilises) over an
     episode for a random walker.
   - After a red is caught, its cells' `P(enemy)` decays within
     ~10 steps.
7. **Policy — belief CNN encoder**: new module in
   `gnn_ctde_policy.py`.  Independently applied per UAV.
8. **Policy — actor**: extend node feature concat to include belief
   embedding (136 dim).
9. **Policy — critic**: cold-start.  New `critic_belief_encoder`
   consumes `true_occupancy` for the centralised state.
10. **Vec env**: propagate the new obs keys through
    `RecurrentVectorPursuitEnv`.
11. **Buffer**: `RecurrentGraphRolloutBuffer` gains fields for
    `belief_maps`, `obstacle_positions`, `true_occupancy`.
12. **Training script**: assemble `GNNCTDEPolicyV2` (or update the
    existing class), diagnostic BCE, cold-start critic init.
13. **Smoke run**: 5 rollouts CPU — confirm no crashes, belief BCE
    decreasing, mean_return non-degenerate.
14. **Full training**: 400 rollouts on GPU.  Log
    `ppo/per_uav_bce`, `ppo/fused_bce` alongside standard PPO
    metrics.

## 10. Acceptance criterion

Stage 4 passes if evaluated against a sensor-aware Greedy baseline
(same fair-Greedy modification as Stage 3, extended to respect
occlusion) in the same partial-obs env, at 50 episodes per (blue,
red) cell with matched seeds:

Against every red baseline (Stationary, Random, RunFromNearest):
- `mean_caught ≥ Greedy_caught + 0.15`, **and**
- `std_return ≤ 0.85 × Greedy std_return`.

Against RunFromNearest additionally:
- `mean_caught ≥ 2.5 / 3` absolute (slightly relaxed from Stage 3's
  2.7 floor since Stage 4 is materially harder — full-cone sensor noise
  and occlusion by 4 obstacles).

Diagnostic:
- `fused_bce` should stabilise **below** `per_uav_bce` within the
  first 100 rollouts.  If not, the multi-UAV Bayesian fusion is not
  benefiting the policy and we would need to reconsider the log-odds
  fusion approach.

## 11. Expected outcomes

- **If Stage 4 PASSES with these settings**: the classical Bayesian
  occupancy grid is sufficient state representation for multi-agent
  ISR under noise + occlusion.  This is a strong, defensible result:
  no fancy belief mechanism needed for the pursuit task at this scale.
  Follow-up stages target increasingly realistic sensors (cone,
  radar, false alarms concentrated near obstacles).
- **If Stage 4 FAILS on catches**: baseline occupancy grid is
  insufficient.  Follow-ups:
  1. Add a learned ConvGRU update layer on the belief map (Hoermann
     et al.).
  2. Replace log-odds fusion with learned attention (MCOP-style).
  3. Try normalising-flow belief posteriors (FORBES).
- **If Stage 4 FAILS on stability (late-training degradation like
  Stage 3)**: the LR schedule and best-checkpoint tracking from
  Stage 3 already ship in `train_stage3.py`; port them to Stage 4's
  training script.

## 12. References

- Moravec & Elfes (1985), *High Resolution Maps from Wide Angle Sonar*.
- Elfes (1989), *Using Occupancy Grids for Mobile Robot Perception
  and Navigation*.
- Thrun, Burgard, Fox (2005), *Probabilistic Robotics* — chapter 9
  (occupancy grid mapping) is the canonical treatment.
- [`stage3_results.md`](stage3_results.md) — the Stage 3 baseline.
- [`stage4_backlog.md`](stage4_backlog.md) — deferred items.
- Web research from the Stage 4 planning session (2026-07):
  MCOP arxiv 2510.12679, FORBES arxiv 2205.11051, Belief States for
  Coop MARL arxiv 2504.08417.  None used directly in the v1 design —
  reserved for follow-ups if the classical approach falls short.
