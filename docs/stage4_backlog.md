# Stage 4 — Backlog

Deferred items surfaced during Stage 4 design (2026-07).  Not required
to pass the baseline acceptance criterion in
[`stage4_design.md §10`](stage4_design.md); revisited after Stage 4
ships or when specific weaknesses appear in the results.

Ordered roughly by expected impact × implementation ease.

## LANDED (during Stage 4 execution — no longer backlog)

Items originally deferred here that ended up shipping as part of the
v6.x architecture.  Kept as short pointers so the reader can trace
"why is this in the code if it says backlog?":

- **§6 aux hidden loss** — landed as `aux_hidden_coef=0.2` (Stage 3
  opt-C ported: `MSE(actor_h_blue, critic_h_blue.detach())`, live-
  critic target).  See `isr/train/ppo.py::ppo_update_stage4` and
  `stage4_results.md` for why it was necessary (dropping it was one
  of the three coupled Stage 3 stabilisers whose absence caused the
  1.4/3 plateau).  The frozen-random-projection / EMA / contrastive
  variants originally proposed here remain open, but only worth
  revisiting if the live-critic version proves insufficient.
- **§7 ally comms range** — implemented, but as a `sensor_radius`-
  gated `bb_edge_visible` rather than a separate `comms_radius`
  knob (see §7 note below).
- **§11 belief-map decay** — landed as Phase A's `enemy_belief_decay`
  (default `0.99`, `L ← γ·L` per step, enemy channel only).  See
  `_predict_enemy_belief` and `stage4_results.md`.
- **§12 time-since-last-update channel** — subsumed by §11.  A cell
  not observed for many steps naturally loses confidence via the
  Phase A decay; the "age" channel would only add value if the
  policy needed to distinguish stale-high-confidence from
  fresh-medium-confidence, which decay already handles.  Not needed.
- **§1 blue↔obstacle crash penalty + §2 blue↔blue crash penalty** —
  landed together as the crash-avoidance extension (branch
  `feature/crash-avoidance`).  Per-agent (INDIVIDUAL, not shared)
  penalties: `r_i = r_team + r_crash_i`.  This required moving the
  whole Stage 4 RL path from shared to per-agent: the critic now
  estimates an agent-conditioned `V(s, i)` (reads blue `i`'s own
  post-message-passing node embedding instead of the pooled
  `sum(h_blue)` — same trunk width, so a shared-reward checkpoint
  still warm-starts cleanly), GAE / advantages / returns are per
  `(env, agent)`, and dones stay per-env (crashes do NOT terminate —
  soft-stop rollback).  Warm-started BOTH actor and critic from the
  converged obstacle policy via `--warm-start-full`
  (`load_full_stage4`).  Penalties are 0.0 by default (byte-preserves
  the pre-crash shared-reward path); the run enables them via
  `--crash-obstacle-penalty` / `--crash-blue-penalty` /
  `--blue-collision-radius`.  TB scalars `crash/obstacle_rate`,
  `crash/ally_rate`.  See `isr/env/pursuit_env.py`,
  `isr/agents/gnn_stage4_policy.py::critic_forward`,
  `isr/train/graph_buffer.py::Stage4RolloutBuffer`,
  `tests/test_crash_avoidance.py`.  Deferred to a v2/v3 escalation:
  the `--terminate-on-crash` hard-termination variant (rejected for
  v1 — an aborted episode discards the catches already earned and
  injects a warm-start distribution shift).  *Measured:* capture held
  ~2.95/3 while obstacle+ally crashes fell ~40 → ~3–5 per episode
  (1000-epoch run in progress to push lower).
- **Obstacle live-sensor refinement** (not a numbered item; shipped
  with crash avoidance) — the actor's obstacle node position was always
  the belief-map peak (grid-quantised ~½ cell).  Now a sensed obstacle
  supplies its precise own-radar position (true centre + noise), the
  peak used only when out of sensor range — mirrors the enemy-track
  treatment so blues hug boundaries safely under the crash penalty.
  See `_build_obstacle_tracks` / `_obstacle_graph_from_tracks`,
  `tests/test_obstacle_tracks.py`.
- **Variable entity counts** (not a numbered item; branch
  `feature/variable-entities`) — `n_red` / `n_obstacles` are now a
  padded CAPACITY; `n_red_min` / `n_obstacles_min` sample the active
  count per episode (unused slots padded inactive, reusing the
  caught-red machinery).  The critic's global context switched from a
  count-hard-coded FLATTEN (`d + N_red·d + N_obs·d`) to a masked-MEAN
  pool + count scalar (`d + (d+1)[+(d+1)]`), making `V(s, i)`
  count-agnostic → one policy generalises across (and beyond) trained
  counts.  Actor was already count-native; buffer/vec_env unchanged.
  See `gnn_stage4_policy.py::critic_forward` / `_masked_pool`,
  `tests/test_variable_entities.py`.  *Implemented + tested; training
  pending.*  (The pool narrows `critic_trunk.0.weight`, the only tensor
  a pre-pool checkpoint can't warm-start; `load_full_stage4` transfers
  the other 76/77 and now NAMES what it leaves at init.)
- **§4 moving obstacles** — LANDED as reciprocating patrol (simplest
  kinematics; branch `feature/moving-obstacles`).  See the annotated §4
  below for what shipped vs the original sketch.  *Implemented + tested;
  training pending.*
- **§13a doctrine reframe** — belief map is now consistently
  described as a mission-command layer environment latent (sibling
  of `true_occupancy`, not an emergent property of TDL messaging)
  across every doctrine paragraph in `pursuit_env.py`,
  `stage4_default.py`, `stage4_results.md`, and the relevant test.
  Dissolved the residual tension with sensor-gated
  `bb_edge_visible`.  §13b (per-UAV command-link gating on the
  peaks) remains open — see below.

## 1. Blue ↔ obstacle crash penalty  — ✅ LANDED (see LANDED section)

> Shipped as the per-agent crash-avoidance extension.  The v1 sketch
> below is kept for provenance, with two deltas from what landed:
> (a) the penalty is INDIVIDUAL (`r_crash_i` on the offender), not
> accumulated in the shared team reward; (b) the default magnitude is
> 0.0 (off) with the run setting ~2.0 via CLI, not a hard-coded 5.0.

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

## 2. Blue ↔ blue crash penalty  — ✅ LANDED (see LANDED section)

> Shipped alongside §1.  Landed as `blue_collision_radius` (default
> 2.0 m) with a symmetric per-agent penalty (both blues in a colliding
> pair eat it), matching the v1 recommendation below.  Positions are
> NOT clipped apart (the "optionally clip" note was dropped — the
> penalty alone teaches routing around allies).

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

## 4. Moving obstacles  — ✅ LANDED (see LANDED section)

> Shipped as **reciprocating patrol** (branch `feature/moving-obstacles`)
> — the simplest kinematics, chosen deliberately over the missiles-that-
> destroy-blues idea (which would force a variable *active-blue* count
> mid-episode + new termination/reward machinery).  Deltas from the v1
> sketch below:
> - **Kinematics:** a random subset (`moving_obstacle_fraction`) patrols
>   back-and-forth along ONE axis at `obstacle_speed`, bouncing off the
>   arena walls (disk clamped to `[r, L-r]`, velocity component flips).
>   No response to blues, as sketched.
> - **Time-varying truth:** the obstacle occupancy grid (cached-once for
>   static obstacles) is recomputed each step when obstacles move, so the
>   belief-truth channel + occlusion track them.
> - **Belief forgetting:** took option 1 (explicit decay) as
>   `obstacle_belief_decay` (default 1.0 = off; enable ~0.9 with motion)
>   — decay-only, NO diffusion (reciprocating motion isn't a random
>   walk).  Did **not** add the "age of last update" third channel (§12
>   reasoning: decay already fades stale cells; the live-sensor
>   refinement handles near-field accuracy).
> - **Velocity in the graph:** obstacle velocity now feeds the edge
>   features — true for the CTDE critic, own-radar Doppler for the actor
>   when a moving obstacle is a live (seen) track — so blues anticipate
>   the sweep.  Crashing a moving obstacle uses the SAME per-agent crash
>   penalty (§1); blues are never destroyed.
>
> See `_move_obstacles` / `_recompute_obstacle_grid`,
> `tests/test_moving_obstacles.py`.

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

## 6. Contrastive / teacher-student aux loss  **[SUPERSEDED — see LANDED §6]**

The Stage-3-style live-critic aux (`MSE(actor_h_blue,
critic_h_blue.detach())`, `aux_hidden_coef=0.2`) shipped in v6.x and
was one of the recipe pieces that unblocked convergence.  The three
alternative formulations sketched below (frozen random projection,
EMA critic encoder, contrastive) remain OPEN as *upgrades* — worth
trying only if the live-critic version turns out to be a limiter in
some future stage.

**Original sketch — three variants:**
1. Frozen random projection: `frozen_proj = Linear(true_occupancy.flatten()
   → 64)` init and never trained.  Aux: `MSE(belief_encoder(belief_maps),
   frozen_proj(true_occupancy).detach())`.
2. EMA critic encoder: maintain a slowly-updated copy of the critic
   encoder (`τ = 0.005` per step).  Use its output as the stable
   aux target.  Same MSE loss shape as Stage 3.
3. Contrastive: pull encoder outputs of neighbouring timesteps
   together, push away from far-timesteps.  No target network needed.
   Larger implementation cost.

**Blocking**: only revisit if aux 0.2 (live critic) plateaus somewhere
we care about.  Current runs (`belief_v3` 3.00/3, `obstacles_v1`
2.95/3) do not motivate the extra cost.

## 7. Decoupled `comms_radius` for ally GPS uplink  **[PARTIALLY LANDED]**

**Current state (v6.x — different from the sketch below).**
`bb_edge_visible` is NOT identically 1 as originally planned — it is
gated by `sensor_radius`:

```
bb_edge_visible[e]  =  1 iff  dist(blue_i, blue_j)  <=  sensor_radius
```

(See `PursuitEnv._compute_edge_visibility` and
`structured_belief_observation`.)  This means the "always-on TDL"
narrative in some docstrings is aspirational — the actual code
gates ally comms at sensor range.  This is *close enough* to a
realistic ISR system that we shipped with it, but the sensor range
(40 m) is well below a realistic TDL range (kilometres), so blues
lose ally comms more aggressively than they should.

**Still-open upgrade — decouple the two ranges.**
- New env param `comms_radius`.  Default: `None` (unbounded within
  the arena; matches the aspirational TDL doctrine).  When set,
  `bb_edge_visible[e] = 1 iff dist ≤ comms_radius` -- **independent
  of `sensor_radius`**.
- Optional: blue node features gain a `n_comms_lost` count (how many
  allies are out of comms this step) — helps the policy know when
  it's in a degraded-comms situation.

(The doctrine-vs-code drift the docstrings previously showed --
docstrings claiming "always-on TDL" while the code sensor-gated
`bb_edge_visible` -- was cleaned up in commit `718b7bd`.  Every
mention now describes the actual sensor-gated behaviour or is
explicitly marked as "design predicted / never landed".)

**Blocking**: none.  ~5 lines in `_compute_edge_visibility` to plumb
the new knob.  Priority is low because `sensor_radius`-gated bb
visibility already gives an interesting partial-comms behaviour;
would matter for a scenario deliberately studying TDL loss (jamming,
urban shadowing).

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

## 11. Belief-map decay / forgetting  **[LANDED as Phase A decay]**

Shipped as `enemy_belief_decay=0.99` (default) in
`_predict_enemy_belief` — each step, before the sensor update, the
enemy-channel log-odds are pulled toward 0 by `L ← γ·L`.  Obstacle
channel untouched (static → prediction is identity), matching the
"downside fix" the original sketch called for.

The paired diffusion step (`enemy_belief_diffusion=0.2`, isotropic
random-walk motion model applied in probability space via a 3×3
convolution) was added alongside as the second half of the Bayesian
predict step; that was NOT in this backlog item, it emerged during
Phase A design.  See `stage4_results.md` for both.

## 12. Time-since-last-update channel  **[SUBSUMED by §11]**

The "age" channel is no longer needed: Phase A's decay already
provides the same functional signal in compressed form — an
unobserved cell's log-odds shrink geometrically toward 0
("unknown"), so stale-high-confidence cells naturally become
lower-confidence over time.

The one thing an explicit age channel would give that decay does not
is *bi-directional* stale detection — distinguishing "stale +positive"
from "stale -negative" (both decay to 0 in log-odds, so from the
network's point of view they become indistinguishable from unknown).
If a future stage needs to tell "cell was recently observed as empty"
apart from "cell has never been observed," this item may reopen.
Not motivated by the current results.

## 13. Reframe belief map as MISSION-COMMAND state (doctrine + code)

**Motivation.**  Earlier Stage 4 docstrings justified the global
fused belief map as an emergent property of the blue team's TDL
messaging ("Bayesian fusion of independent sensors is log-odds
addition, and the always-on TDL comms assumption means sharing
detections over the same link is realistic").  That reading was
coherent but had a residual tension with the code: `bb_edge_visible`
is sensor-radius-gated per step, so a UAV out of comms with its
neighbours nonetheless kept contributing fresh evidence to the
"shared" map -- which shouldn't be possible on a strict per-UAV view.

The cleaner doctrinal position (now the one used throughout the code
and docs):

> The global belief map represents the shared operational picture
> maintained by the mission command layer.  It is an environment-level
> latent state used for target tracking and evaluation, rather than
> the instantaneous knowledge available to each UAV.

This matches real ISR C2 (CAOC / TOC / AWACS run a persistent fused
track picture; UAVs receive updates via the C2 downlink), and it puts
the belief map in the same category as `true_occupancy` -- both are
environment-level latents (belief noisy; `true_occupancy` ground truth
for the CTDE critic).

### 13a — Docstring reframe  **[LANDED — commit `b0b7e6b`]**

Rewrote the doctrine paragraphs in
`PursuitEnv._update_belief_maps`,
`_enemy_graph_from_tracks`,
and `structured_belief_observation` (plus the constructor param,
the attribute comment, the reset comment, `configs/stage4_default.py`,
and `docs/stage4_results.md`'s architecture section) to describe the
map as command-layer fused state instead of "what the team sees over
TDL".  Dissolves the "always-on TDL assumption" hand-wave and the
residual tension with `bb_edge_visible`'s sensor gating.  No code
change; 52/52 tests unchanged.

### 13b — Command-link gating on peaks  **[OPEN]**

Under the strict reading, `belief_peaks_enemy` are also command-layer
outputs and should reach each UAV only if that UAV has a command
link.  New env param `command_link_visible[b]` (per-blue, could
default to always-on for backward compat; could become a function of
distance to a "base station" or of a jamming mask); when 0, that
UAV's rb edges fall back to the memory / no-info path (peak_conf 0,
zero velocity) regardless of what command has fused.  Enables a
clean study of degraded C2 -- a scenario that separates realistic
ISR sims from toy trackers.

**Blocking.**  Needs the env knob plus a plumbing pass through
`_build_enemy_tracks` / vec_env / obs schema; only worth building
when we actually want to study C2 loss (so probably alongside the
`comms_radius` decoupling in §7 -- they're the "degraded network"
pair).

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
