# Stage 4 — Backlog

Deferred items surfaced during Stage 4 design (2026-07).  Not required
to pass the baseline acceptance criterion in
[`stage4_design.md §10`](stage4_design.md); revisited after Stage 4
ships or when specific weaknesses appear in the results.

Ordered roughly by expected impact × implementation ease.

## 1. Blue ↔ obstacle crash penalty

**Motivation.** Currently, when a blue attempts to move into an
obstacle, kinematics clip its position to the obstacle boundary
(soft reflection).  This trains the policy to happily graze
obstacles.  A real UAV crashes.

**Sketch.**
- Detection: after kinematic clipping, check whether the pre-clip
  position was inside any obstacle disk.
- Penalty: `reward -= λ_crash_obstacle` (default 5.0 — half a red
  catch).  Applied per (blue, step); accumulated in the shared
  team reward.
- No episode termination for v1 — the blue is still on the arena
  boundary, can continue.  A `--terminate-on-crash` flag can add hard
  termination later.

**Blocking**: none.  ~15 lines in `PursuitEnv._step`.

## 2. Blue ↔ blue crash penalty

**Motivation.** Same argument — real UAVs collide.  Without this,
the policy has no reason to route around allies.

**Sketch.**
- New env param `blue_collision_radius` (default 2.0 m — smaller than
  `capture_radius` so a normal capture doesn't trigger this).
- After per-step kinematics: if `dist(blue_i, blue_j) <
  blue_collision_radius`, apply `-λ_crash_blue` to each.
- Optionally clip positions so they don't overlap.

**Blocking**: none.  ~10 lines in `PursuitEnv._step`.  Design
question: is the penalty symmetric?  For v1 recommend yes — both
blues are equally responsible.

## 3. Cone-shaped sensor with heading

**Motivation.** Radar / gimbal-camera sensors are physically directional.
A cone-of-view + heading models this correctly.  Also opens up
interesting emergent behaviours (blues learn to point their cones at
predicted target locations).

**Sketch.**
- New blue node feature: `heading ∈ (cos θ, sin θ)` (2-dim, avoids
  the discontinuity of raw angle).  Bumps `blue_feat_dim` from 8 to
  10.
- New env kinematics: heading updates from action (either action_dim
  grows to 3 with a heading change, or heading rotates toward
  velocity direction each step).
- Belief update: instead of "cell in sensor disk", condition on both
  distance AND angle within `sensor_fov / 2` of heading.
- Occlusion still ray-cast (same code).

**Blocking**: the action dimension change breaks buffer/policy
constructor signatures.  Handled by making `sensor_fov = 360°` the
default (equivalent to Stage 4's disk) and gating cone behaviour on
`sensor_fov < 360`.

## 4. Moving obstacles

**Motivation.** Real ISR environments have moving vehicles, other
UAVs (non-hostile), etc.  Static obstacles are a simplification.

**Sketch.**
- Each obstacle gets a scripted velocity at reset (constant during
  the episode, small magnitude — order of red_speed / 2).
- Belief map for obstacles now has **time-varying truth** — the
  `P(obstacle at cell)` should decay when the obstacle moves away.
  Options:
  1. Add explicit forgetting: `L_obstacle *= 0.98` per step.  Cheap,
     principled but coarse.
  2. Learned ConvGRU update (Hoermann et al.) — much more powerful
     but requires a training loss.
- Kinematics: obstacles move deterministically, don't respond to
  blues.

**Blocking**: the belief update needs to track when the last
observation was.  Adds a third channel (`age of last update`).  Or
just goes to option 2 (learned update) directly.

## 5. Learnable sensor model

**Motivation.** `p_TP`, `p_FP` are currently fixed hyperparameters.
If they're miscalibrated relative to the actual sensor noise, the
belief map is biased.  Making them learnable lets the framework
self-calibrate.

**Sketch.**
- Wrap `p_TP`, `p_FP` in learnable `nn.Parameter` (via sigmoid so they
  stay in `[0, 1]`).
- New aux loss: `L_calibration = BCE(sigmoid(fused_belief),
  true_occupancy)`.  This one gets a real gradient because the
  belief maps now depend on trainable parameters.
- Careful: the belief maps are computed in the env (via NumPy).  Either
  move the log-odds arithmetic into torch (small code change), or
  train just the sensor-model params in a separate calibration phase.

**Blocking**: mixing env-side numpy state with torch differentiability.
Moving belief updates into the policy (as a first "layer" of the
actor) is the cleanest solution.

## 6. Contrastive / teacher-student aux loss

**Motivation.** Stage 3 showed that a well-designed aux loss can help
(the `freeze-critic + high-aux + Opt 1` combo).  Stage 4 baseline
skips aux entirely (`aux_hidden_coef=0`) because we can't warm-start
the critic.  A different aux formulation can re-introduce this
regularisation without the warm-start dependency.

**Sketch — three variants:**
1. Frozen random projection: `frozen_proj = Linear(true_occupancy.flatten()
   → 64)` init and never trained.  Aux: `MSE(belief_encoder(belief_maps),
   frozen_proj(true_occupancy).detach())`.
2. EMA critic encoder: maintain a slowly-updated copy of the critic
   encoder (`τ = 0.005` per step).  Use its output as the stable
   aux target.  Same MSE loss shape as Stage 3.
3. Contrastive: pull encoder outputs of neighbouring timesteps
   together, push away from far-timesteps.  No target network needed.
   Larger implementation cost.

**Blocking**: only worth pursuing if the baseline shows the encoder
is failing to preserve belief information (measured by the
diagnostic BCE staying high).

## 7. Decoupled `comms_radius` for ally GPS uplink

**Motivation.** In Stage 4 baseline, `bb_edge_visible ≡ 1` — allies
always share GPS.  Realistic for LOS radio in an open arena; wrong
for jamming, terrain shadowing, or urban environments.

**Sketch.**
- New env param `comms_radius`.  Default = `None` (unbounded, same
  as v1).  When set, `bb_edge_visible[e] = 1 iff dist ≤ comms_radius`.
- Blue node features gain a `n_comms_lost` count (how many allies
  are out of comms this step) — helps the policy know when it's in
  a degraded-comms situation.

**Blocking**: none — trivial extension of the Stage 3 visibility mask.

## 8. Per-channel sensor noise

**Motivation.** In reality, obstacles are much easier to detect
than moving targets (they're bigger, more predictable, don't try to
hide).  A single `(p_TP, p_FP)` pair for both channels overstates
enemy-detection quality if calibrated to obstacles.

**Sketch.**
- Config becomes `p_TP_enemy`, `p_TP_obstacle`, `p_FP_enemy`,
  `p_FP_obstacle`.
- Belief update precomputes one `(ΔL_detect, ΔL_no_detect)` pair per
  channel.
- Suggested realistic defaults: `enemy: p_TP=0.75, p_FP=0.20`,
  `obstacle: p_TP=0.95, p_FP=0.05`.

**Blocking**: none.

## 9. False-alarm concentration near obstacles

**Motivation.** Real sensors get confused by clutter — an obstacle
edge often produces spurious target detections.  Currently `p_FP` is
uniform; making it position-dependent would model radar clutter.

**Sketch.**
- For each cell, compute distance to the nearest obstacle boundary.
- `p_FP_effective(cell) = p_FP_base + β * exp(-dist / obstacle_boundary_scale)`.
- Only affects the sensor step; belief update is unchanged.

**Blocking**: only interesting after the baseline works.  Would
justify a follow-up "robustness under clutter" experiment.

## 10. Learned ConvGRU belief update (Hoermann-style)

**Motivation.** Classical log-odds treats cells independently.  A
ConvGRU can capture spatial correlations (a red at cell X makes reds
at neighbouring cells more likely, which the log-odds update
completely ignores).

**Sketch.**
- Replace log-odds addition with a ConvGRU cell:
  `new_belief_map[i] = ConvGRU(cell_evidence, prev_belief_map[i])`.
- Trained with the diagnostic BCE loss (which now DOES get a
  gradient because the update is learned).
- Loses closed-form multi-UAV fusion — needs a learned fusion module
  too (options 2b / 3 in the Stage 4 design's fusion table).

**Blocking**: only worth the complexity if the Stage 4 baseline
plateaus below acceptance and the fused BCE stays high.

## 11. Belief-map decay / forgetting

**Motivation.** Stage 4 baseline keeps log-odds indefinitely.  If a
red was seen at cell X but has moved to cell Y in the intervening
steps, the belief map still says P(enemy at X) is high — stale.

**Sketch.**
- Per-step decay `L *= exp(-dt / τ_forget)` where `τ_forget ~ 50`
  steps.  Belief "half-life".
- Simple, closed-form, no learning needed.
- Downside: also forgets obstacle beliefs (which are stationary).  Fix
  by applying decay only to the enemy channel.

**Blocking**: none.  Add if the policy shows "chasing ghost targets"
behaviour.

## 12. Time-since-last-update channel

**Motivation.** Related to §11.  If we don't want to decay the belief,
we can still tell the policy how stale each cell is by adding an "age"
channel.

**Sketch.**
- New third channel: `age[cell] = steps since last observation`.
- Incremented at each step for un-observed cells; reset to 0 on
  observation.
- Policy learns to trust fresh observations more than stale ones.

**Blocking**: none.  Adds one channel (bumps CNN input to 3
channels).

---

## Design questions still open

Items where the "right" choice depends on empirical results:

- **`p_TP` / `p_FP` defaults for Stage 4 acceptance.**  Currently 0.85
  / 0.15 (medium).  If the baseline crushes acceptance with big
  margin, tighten to 0.75 / 0.25 for the "real" reported result.
- **`n_msg_rounds` in the GNN.**  Inherited Stage 3's value of 2.  The
  richer 136-dim node features from belief embeddings might benefit
  from a third round.  Ablation candidate.
- **Grid resolution.**  Locked at 5m for v1.  Ablations: 3m (finer
  localisation, 4× compute) vs 10m (much lighter, cell often contains
  multiple entities).
