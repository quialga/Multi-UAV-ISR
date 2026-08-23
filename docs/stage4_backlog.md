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
- **Obstacle-aware enemy belief channel** (not a numbered item; belief-map
  correctness fix) — two coupled bugs.  (a) Cells inside an obstacle are
  permanently occluded, so the sensor update never reached them and they
  sat near log-odds 0 (= P 0.5, "unknown") — but reds are kinematically
  clipped OUT of obstacle disks, so P(enemy | inside obstacle) = 0.
  Measured: interiors averaged **−1.87** vs **−7.64** for properly cleared
  open space (~5.8 log-odds HIGHER), making them probability sinks that
  attracted phantom peaks (**6.6%** of extracted peaks landed inside an
  obstacle, tens of metres from any red).  (b) The isotropic diffusion
  kernel leaked mass INTO those masked cells, continuously re-filling what
  decay pulled toward 0 — which is *why* they sat at −1.87 rather than a
  clean 0.  Fix: `_pin_enemy_belief_in_obstacles` pins the enemy channel to
  `−belief_clip` inside the current obstacle footprint (channel 1 untouched
  — that is exactly where the obstacle channel *should* be high), and the
  diffusion kernel now renormalises each source cell's outgoing weight over
  its VALID neighbours (a reflecting boundary at the obstacle wall, which
  conserves probability instead of silently destroying it; with no
  obstacles it reduces exactly to the plain convolution).  *Measured after:*
  interiors **−10.00**, phantom peaks **0.0%** (was 6.6%).  **Note:** mean
  peak error is unchanged (**32.2 m** vs 31.0 m) — the ~30 m
  `belief_track_error` is dominated by the *unseen-red fallback* (only ~34%
  of active reds are in sensor range at any instant), not by these phantom
  peaks; this fix removes a correctness bug and phantom tracks, not the
  headline error.  Directional (velocity-aware) prediction remains open —
  see §15b.  See `_predict_enemy_belief`,
  `tests/test_belief_obstacle_mask.py`.
- **`belief_window_size` removed** — dead constructor param/attribute, set
  but never read anywhere since the CNN/ego-window design was dropped in
  the v5.x redesign.
- **Obstacle radius as a node feature** (not a numbered item; observability
  fix) — the actor received the obstacle CENTRE (refined track) but never
  its RADIUS, while the crash penalty and clearance barrier are defined on
  the SURFACE (`centre_dist − radius`).  With radii 5–15 m the actor could
  not locate the danger boundary from a centre alone.  Obstacle node
  feature is now `[conf, radius/arena_size]` (`obs_feat_dim` 1→2, actor +
  critic): true measured radius for a seen/live track, the surveyed field's
  mean (command prior) for an unseen/memory track.  Warm-start soft-loads
  75/77 tensors; the two `obs_input_mlp.0.weight` (actor+critic) reinit
  (shape (d,1)→(d,2)).  The detection gate is still centre-distance only
  (radius omission there is harmless — `sensor_radius=40` gives ample
  warning).  See `_build_obstacle_tracks` / `_obstacle_graph_from_tracks`
  / `_true_obstacle_graph`, `tests/test_obstacle_tracks.py`.
- **Red wall-repulsion** (not a numbered item; scripted-adversary fix)
  — `run_from_nearest_uav` fled straight away from the nearest blue,
  which ran a cornered red into the arena wall; the perpendicular
  velocity clipped and the red slid ALONG the boundary, pinning itself.
  Blue then learned a degenerate wall-trapping counter (observed in the
  simulator: reds AND blues hugging walls).  The heuristic already
  repelled off OBSTACLES for this exact reason ("strawman adversary");
  extended the same falloff repulsion to the four arena walls, gated on
  a new `arena_size` arg (None ⇒ off ⇒ byte-identical; the env now
  passes it).  Corners get a diagonal push from two walls.  *Measured:*
  under a greedy pursuer, red time within 8 m of a wall fell 0.63 → 0.42.
  Makes the scripted red a less exploitable benchmark and a better base
  for self-play.  Retrain the curriculum against it (the v1 warm-start
  learned wall-camping).  See `run_from_nearest_uav`,
  `tests/test_red_wall_repulsion.py`.
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
  `tests/test_variable_entities.py`.  *Fixed-count baseline VALIDATED
  (`pool_fixed_v1`, 2.97/3 det eval — matches the pre-pool flatten
  ~2.95/3, so the pool is capture-neutral at fixed count; see
  `stage4_results.md`); variable-N training still pending.*  (The pool
  narrows `critic_trunk.0.weight`, the only tensor a pre-pool checkpoint
  can't warm-start; `load_full_stage4` transfers the other 76/77 and now
  NAMES what it leaves at init.)
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

## 3. Cone-shaped sensor with heading  — ❌ DROPPED (not worth it)

> **Decision (2026-07): won't do.**  Modern combat/ISR UAV radar is
> AESA — effectively wide-FOV / electronically steered / ~360°
> coverage — so the circular sensor disk is a *defensible physical
> abstraction*, not a shortcut.  A directional cone + heading would add
> an action-dimension change that breaks the buffer/policy constructor
> signatures (see "Blocking") for modest research novelty.  Only worth
> reopening for a deliberate **sensor-management** study (directional
> EO/IR gimbal, "point your camera at the predicted intercept"), which
> is not on the roadmap.  Sketch kept for the design record.

**Motivation (original).** Radar / gimbal-camera sensors are physically
directional.  A cone-of-view + heading models this correctly.  Also
opens up interesting emergent behaviours (blues learn to point their
cones at predicted target locations).

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

**In plain terms (what this item is).**  The actor acts on *noisy*
belief-derived positions; the critic (CTDE) sees *ground truth*.  A
"teacher–student" auxiliary loss nudges the actor's internal per-blue
representation to match the critic's ground-truth one — i.e. it teaches
the actor to squeeze out of noisy inputs the same useful features the
critic gets for free from perfect information.  ("Teacher–student" =
one network produces a target the other learns to copy; the critic is
the teacher, the actor the student.  "Contrastive" is a related
self-supervised idea that instead pulls together representations that
*should* be alike and pushes apart ones that shouldn't, with no explicit
teacher target.)

**Why it's marked SUPERSEDED.**  The *simplest* version of this idea
already shipped in v6.x and was one of the pieces that unblocked
convergence: `aux_hidden_coef=0.2` adds
`MSE(actor_h_blue, critic_h_blue.detach())` — the actor's per-blue
embedding is pulled toward the *live* critic's (Stage-3 opt-C, see the
LANDED §6 note at the top of this file).  So the core idea is DONE.
What stays OPEN here are three *fancier variants* of the same
teacher–student trick (below) — worth trying ONLY if the simple
live-critic version turns out to be a limiter in some future stage.

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

## 11. Belief-map decay / forgetting  — ✅ **[LANDED as Phase A decay]**

Shipped as `enemy_belief_decay=0.99` (default) in
`_predict_enemy_belief` — each step, before the sensor update, the
enemy-channel log-odds are pulled toward 0 by `L ← γ·L`.  Obstacle
channel untouched *for static obstacles* (prediction is identity),
matching the "downside fix" the original sketch called for.  (Later, the
moving-obstacles work — §4 — added an OPTIONAL decay on the obstacle
channel too, `obstacle_belief_decay` (default 1.0 = off), to fade a
moving obstacle's stale trail; it's decay-only, no diffusion.)

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

## 14. 3D environment

**Motivation.**  The current arena is a 2D plane — UAVs move in
`(x, y)` with no altitude.  Real ISR operations are fundamentally 3D:
altitude affects sensor footprint (higher ⇒ wider coverage but weaker
signal), obstacle avoidance geometry (fly over vs fly around), and
inter-UAV separation (vertical stacking).  A 3D extension brings the
sim closer to real mission planning.

**Sketch.**
- Arena becomes a 3D box `[0, L] × [0, L] × [0, H]` with a
  configurable ceiling `H` (e.g. 50 m).
- Blue actions grow from 2D `(vx, vy)` to 3D `(vx, vy, vz)` — bumps
  `action_dim` from 2 to 3.  Buffer, PPO ratio, log-prob shapes follow.
- Node features gain altitude: blue `(x, y)` → `(x, y, z)`; red /
  obstacle positions likewise.  Edge features (`dx, dy, dist`) become
  `(dx, dy, dz, dist)`.
- Sensor model: disk → sphere (or altitude-dependent cone — higher
  altitude = wider ground footprint but reduced detection probability,
  modelling real SAR/EO trade-offs).
- Obstacles: 3D cylinders or spheres; a UAV above/below can overfly.
  This changes the crash-avoidance geometry (2D disk intersection →
  3D sphere/cylinder intersection).
- Belief maps: remain 2D ground-plane grids (the "operational picture"
  is projected to the ground, matching real C2 track displays).  Altitude
  of the observing UAV affects `p_TP` / `p_FP` if altitude-dependent
  sensors are modelled.
- Red policies: `run_from_nearest_uav` stays 2D (ground targets); or
  generalise to 3D if modelling aerial adversaries.

**Blocking**: action-dim change propagates through buffer, policy
constructor, PPO update (same pattern as §3's heading, but this time
worth doing).  Recommend starting with a `z_enabled=False` default
that collapses to the current 2D arena, so existing checkpoints and
tests stay valid.

## 15. Learned trajectory prediction (belief-kernel / aux head)

**Motivation.**  The current belief-map diffusion uses a fixed
isotropic 3×3 kernel (`enemy_belief_diffusion=0.2`) — effectively
a random-walk motion model.  This is a poor fit for enemies that
flee deterministically ("run" policy: belief should shift in the
escape direction, not smear symmetrically) or for moving obstacles
on a predictable patrol.  A learned prediction step would let the
policy anticipate enemy / obstacle positions K steps ahead, enabling
intercept courses rather than pure pursuit.

**Two implementation paths (from lighter to heavier):**

### 15a — Trajectory prediction auxiliary head (recommended first)

Add a small MLP that reads the GNN per-node embeddings (which
already encode belief + temporal info via the GRU hidden state) and
predicts enemy/obstacle positions K steps ahead.  Trained with a
supervised MSE loss against ground-truth future positions (available
in the CTDE critic path).

- Stays entirely inside PyTorch — NO changes to the env-side NumPy
  belief update.
- The actor benefits because the shared encoder backbone is forced
  to learn motion-predictive features (the grad flows back through
  the shared GNN layers).
- Adds ~one small MLP + one aux loss term; same pattern as the
  existing `aux_hidden_coef`.
- K = 3–5 steps is a natural starting point (one capture-radius
  worth of horizon at typical speeds).

### 15b — Learned belief-map kernel (heavier; see also §10)

Replace the fixed 3×3 isotropic convolution in
`_predict_enemy_belief` with a learnable kernel (potentially larger,
directional, conditioned on local velocity estimates).  This is a
subset of §10 (full ConvGRU) but scoped to just the predict step,
not the full update rule.

- **Fundamental obstacle:** the belief update runs in NumPy on the
  env side, outside the differentiable graph.  To get gradients you
  must either (a) move the predict step into PyTorch inside the
  policy forward pass, or (b) train the kernel with a separate
  supervised phase (predict next-step belief from current belief +
  observations, compare against ground truth).
- Path (b) is feasible but decouples the kernel from policy
  performance — the kernel optimises reconstruction, not task reward.
- Path (a) merges with §10 and is the right long-term answer if
  belief-map quality becomes the bottleneck.

*Graph-embedding-conditioned variant (raised 2026-08).*  Rather than a
kernel conditioned only on local velocity, make the predict-step kernel a
function of a **global graph embedding** (a hyper-network / FiLM-style
conditioning: the pooled GNN feature parameterises the belief-update
kernel).  Appealing because the global embedding already fuses every
UAV's evidence, so the kernel could learn *scene-specific* motion (e.g.
"this obstacle is mid-sweep, shift its mass along +x").  Same NumPy↔torch
blocker as above, plus the extra machinery of generating kernel weights
from an embedding.

**Honest relevance to the current crash problem (important).**  §15 is a
*perception / anticipation* improvement — it sharpens *where* a moving
obstacle will be.  The `moving_v1` failure (see `stage4_results.md` §5)
was **not** perception; it was a **reward-structure / optimisation**
collapse (per-step crash penalty exploding under sweeps and dominating
the return).  Better prediction does not fix a diverging reward, so §15
is **not** the first-order lever here — §16 (clearance shaping / crash as
a constraint) is.  Two further caveats specific to *this* setup: (i) the
obstacle **velocity is already a graph feature**, and reciprocating
motion is piecewise-constant-velocity, so a learned kernel mostly adds
*wall-bounce* anticipation — a narrow gain until the motion model gets
richer/stochastic; (ii) §15a (the cheap aux head) should run *first* to
*measure* whether the encoder is even motion-limited before paying for
15b's numpy↔torch surgery.

**Recommendation:** start with 15a (aux head).  The GRU hidden state
already tracks temporal patterns implicitly; the aux head makes that
explicit and measures how much the encoder actually learns about
motion.  If the aux head's prediction error stays high (the encoder
doesn't learn motion), that motivates 15b / §10 to give it a better
input signal.  But sequence all of §15 **after** §16 — fix the reward
structure before sharpening perception.

**Blocking**: 15a — none (same pattern as `aux_hidden_coef`).
15b — the NumPy↔PyTorch boundary (same blocker as §5 and §10).

## 16. Crash-avoidance as a CONSTRAINT (clearance shaping / safe-RL)

**Motivation (from the `moving_v1` collapse — `stage4_results.md` §5).**
Operating stance: *a crash = losing a drone*, so near-zero crashes
matters more than a marginal catch.  A scalar crash **penalty** in the
reward is the wrong tool for that: it is a *soft trade-off* (the policy
accepts crashes whenever catches outweigh them), and cranking it just
reproduces the moving-obstacle divergence (penalty dominates the return,
training degrades on both capture and crashes).  Near-zero crashes needs
a **constraint-shaped** treatment, not a bigger negative number.

**Three pieces (roughly in priority order):**

1. **Dense clearance / barrier shaping (the workhorse).**  ✅ **LANDED.**
   `clearance_weight` (0 = off, default) / `clearance_margin` (8 m) on
   `PursuitEnv`; CLI `--clearance-weight` / `--clearance-margin`.  Per-agent
   reward `r_clear_i = −w · Σ_o clip((margin − surface_dist)/margin, 0, ∞)`
   summed over obstacles: 0 beyond the band, ramps to `w` at the surface,
   and keeps growing INSIDE the disk, so the gradient points OUT everywhere
   in range.  Reported as `info["clearance_penalty"]`.  See `_step` §6d and
   `tests/test_clearance_shaping.py`.  **Ally extension:**
   `clearance_ally_weight` / `clearance_ally_margin` (3 m) add the same
   barrier between blues ("surface" = `blue_collision_radius`) — needed
   because obstacle clearance ALONE bunches blues into the same clear
   space and just migrates crashes obstacle→ally (measured on
   `clearance_fixed_v1`: obstacle events ~1.25→~0.9 but ally events
   1.17→~1.7).  **Both weights are OPT-IN (default 0.0)** — set them
   together per run (`--clearance-weight` / `--clearance-ally-weight`);
   an earlier 0.6/0.6 default was reverted because a partial override
   (obstacle set, ally left default) silently ran a *stronger* ally
   barrier than intended and degraded capture (`clearance_fixed_v2`).
   13 tests total (obstacle + ally: off,
   zero-beyond-margin, monotone, grows-inside, symmetric, gradient-apart,
   compose).  *Design spec kept below.*  A smooth
   per-step reward term that is **deepest inside an obstacle and decays
   outward** over a margin band.  Unlike the current *flat* occupancy
   penalty (same value anywhere inside → zero positional gradient → tells
   the drone it is being punished but **not which way to escape**), the
   barrier's gradient points continuously toward clear space: a proactive
   "keep your distance" *and* a real "get out, THIS direction" signal.
   This is the same idea as the red heuristic's obstacle repulsion, moved
   into the **blue reward**.  It also cures the divergence: the policy
   learns to keep a buffer, so occupancy (and thus the crash term) never
   explodes.  Flag-gated, small magnitude, off by default.
   - *Design note (why not pure event-based):* penalising only crash
     *entry* (rising edge) removes the incentive to *leave* a crash under
     the soft-stop physics — the barrier's inside-gradient is what
     supplies that incentive instead.
2. **Discrete crash-event = drone-loss semantics.**  A crash is a
   one-time catastrophic event (you lose the drone), not per-step
   occupancy.  Ultimately model it literally as **destroy-on-crash**
   (remove the blue mid-episode — now feasible via the variable-entity
   machinery: a crashed blue goes inactive like a caught red, and the
   masked-mean critic / count-agnostic actor already handle a shrinking
   team).  Introduce *after* clearance shaping already makes crashes rare,
   so destruction events don't destabilise training (the reason the old
   `--terminate-on-crash` sketch was deferred).
3. **Constrained-RL / Lagrangian (most principled, most work).**  Treat
   `E[crash_events] ≤ ε` as a hard constraint with a learned Lagrange
   multiplier (PPO-Lagrangian / CPO style) instead of a fixed penalty
   weight.  The multiplier self-tunes the trade-off, avoiding the manual
   penalty-magnitude search that produced v1/v3/moving_v1.

**Also:** keep `lr` at the default (`1e-4`) and entropy **low** (a safety
task wants precise, not erratic, control near obstacles — do NOT raise
entropy; consider annealing it down), and **curriculum** the motion
(`fraction 0.1 / speed 0.5` first).

**Blocking:** #1 is ~a dozen lines in `_step` + a flag + a test (no
numpy↔torch issue — it reads the same obstacle geometry the crash check
already uses).  #2 reuses the caught-red inactivation path.  #3 is a
larger PPO change.  **Start with #1.**

## 17. Radial / tangential Doppler decomposition

**Motivation.**  The live-track sensor model now applies `p_TP` detection,
line-of-sight gating, range-scaled position/Doppler noise, and range-based
confidence (see `stage4_results.md`).  One realism gap remains: the actor
receives the **full 2-D velocity vector**, but a real radar measures only
the **radial** component (along the line of sight) from Doppler phase.  The
**tangential** component is not directly observable — a tracker must infer
it over several scans from angle rate, far more noisily.

**Why it matters for THIS task.**  The asymmetry is task-relevant, not
cosmetic:

- A red **fleeing directly away** from a blue is almost pure radial →
  measured beautifully.
- A red **crossing** a blue's line of sight is almost pure tangential →
  nearly invisible to Doppler — and that is exactly the geometry where
  intercept prediction needs velocity most.

The current model erases that distinction, so the policy gets crossing-target
velocity for free.

**Sketch.**

```
los       = (target_pos − blue_pos) / ‖·‖
v_radial  = (v_true · los) los         # measured well (small noise)
v_tangent = v_true − v_radial          # poorly observed
v_meas    = v_radial + n_r·los + tangential_observability · v_tangent + n_t
```

New knob `tangential_observability ∈ [0, 1]`: `1.0` = today's behaviour
(full vector), `0.0` = pure Doppler (radial only).  Applies per (blue,
target) edge, since the line of sight is per-blue — which also makes the
existing "own-sensor Doppler" doctrine sharper.

**Deferred deliberately.**  Unlike the detection/occlusion fixes (physics
violations), this one *changes what the policy can infer* about crossing
targets, so it deserves its own before/after comparison rather than being
bundled with the sensor-coherence work.  Expect intercept quality against
the `run` evader to drop when it lands — that would be an honest
degradation, not a regression.

**Blocking**: none — ~15 lines in `_measured_vel` plus threading the
line-of-sight vector through the two graph builders.

## 18. Track continuity / coasting on a missed detection

**Motivation.**  Now that live tracks obey the real detection chain
(`track_detection`, see `stage4_results.md` §4b), a target is missed on
~`1 − p_TP` of scans per observing blue.  Today a single miss drops the
track **straight back to the belief-map peak** — a hard switch from a
*measured* position (precise, `conf = 1`) to a *grid-quantised* one
(~½ cell error).  Real radar does not behave that way.

**What real trackers do.**  A **track** is a persistent object maintained
across scans, distinct from the per-scan **detections** that feed it.  On a
miss the tracker does not delete it, it **coasts**:

1. propagate the track forward on its motion model (constant velocity),
2. inflate the uncertainty (no measurement to correct the covariance),
3. drop the track only after several consecutive misses (classic **M-of-N**
   logic, e.g. 3 straight misses).

So a blink degrades the estimate gracefully instead of collapsing it.

**Why this is the right fix (vs exempting obstacles from `p_TP`).**  The
tempting shortcut — "obstacles are big and static, just always detect them"
— re-opens exactly the incoherence §4b closed: two different detection
models fed by one sensor.  Coasting keeps the **sensor** honest (it still
misses) and puts the memory where it physically belongs, in the
**tracker**.

**Sketch.**
- Per target (red and obstacle) store: last-good measured position, last
  measured velocity, and a `staleness` counter (scans since last hit).
- On a **hit**: refresh all three, `staleness = 0`.
- On a **miss**: emit `last_pos + last_vel · staleness · dt` with `conf`
  decayed by staleness (and, if `sensor_noise_range_growth` is on, an
  uncertainty that grows the same way).
- Past `max_coast_scans`, fall through to the belief peak as today.
- For a **static obstacle** coasting is exactly correct — it has not moved,
  so the last measurement is still true and `conf` should decay slowly (or
  not at all).  This is where the current flicker is most visible, because
  clearance shaping (§16) is defined on the obstacle **surface**.
- It helps **enemy** tracks too: a fleeing red would coast along its last
  Doppler instead of jumping to a stale blob on every blink.

**Blocking**: none technically — but deliberately **deferred until measured**.
Add tracker state only if the precise↔quantised flicker actually shows up in
the clearance / crash-event numbers; there is no point carrying per-target
history to fix a problem that may not bite.  Check
`eval_det/{obstacle,ally}_crashes` on a run with the realistic sensor model
before building this.

## 19. Information-seeking reward (potential-based, NOT hand-coded bonuses)

**The idea as originally raised (2026-08).**  Two per-agent shaping terms:
(a) a **reward** for a UAV that detects an enemy no other UAV has found so
far, and (b) a **penalty** when a UAV loses an enemy that no other UAV is
tracking.  Intent: make information-gathering — the actual point of ISR —
a first-class objective instead of something rewarded only indirectly
through captures.

**Why NOT to ship those two terms as stated.**

1. *They target a bottleneck we do not have.*  Against `pool_fixed_v4`:
   stationary 2.95 / random 2.95 / **run 2.45**.  A STATIONARY red never
   moves, so catching it is purely find-then-reach — and those score
   2.95/3.  If search/detection were limiting, stationary would suffer
   too.  The whole gap is the FLEEING red, i.e. a pursuit / coordination
   problem, not a detection problem.
2. *(a) is non-Markovian.*  "not found by any other UAV **so far**" depends
   on episode history, so it needs a per-red `discovered` flag added to the
   state; otherwise the policy receives rewards it cannot explain from what
   it observes → high-variance gradients.
3. *(b) has two concrete perverse incentives.*
   - **Shadowing beats closing.**  Holding a track is easier than
     capturing.  To capture you must close, which makes the red flee
     harder and raises the chance of losing it — so a custody penalty
     makes the final approach the riskiest moment and can teach the policy
     to loiter at max sensor range instead of committing.  Symptom to
     watch for: detection stats UP, capture DOWN.
   - **Acquisition avoidance.**  If being the sole tracker is a liability,
     never becoming the sole tracker is a valid way to avoid the penalty.

**The principled form — potential-based shaping.**  Ng, Harada & Russell
(1999), *Policy invariance under reward transformations*: any shaping of
the form

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

provably leaves the OPTIMAL POLICY UNCHANGED, whatever `Phi` is.  Define a
team-knowledge potential over the belief state, e.g.

```
Phi(s) = - sum_r  uncertainty(track r)      # or -sum_r entropy / cov trace
```

Both original ideas then fall out **automatically and safely**:

- discovering a previously-unknown red collapses its uncertainty → large
  positive shaping (idea (a), without the history bookkeeping);
- losing sole custody inflates uncertainty → negative shaping (idea (b),
  without the shadowing incentive — because the shaping cannot change which
  policy is optimal, it can only speed up finding it).

It is also **Markovian by construction**: uncertainty is a function of the
current belief state, not of who saw what earlier.

**Sequencing / blocking.**  Deliberately deferred.  `moving_v1` collapsed
from adding ONE reward term; obstacle clearance, ally clearance and the
realistic sensor model (§4b) are all still unvalidated, so adding two more
shaped terms now would make attribution impossible.  **Train first, then
diagnose:**

- uncaught reds mostly **undetected** for their lifetime → search IS the
  bottleneck → implement the potential-based version above;
- uncaught reds mostly **detected but uncaught** (what the stationary
  result predicts) → the fix is pursuit / coordination, and detection
  shaping will not help.

*Measurement caveat when re-running this diagnostic:* `pool_fixed_v4`
predates the obstacle-radius node feature, so loading it now reinitialises
`obs_input_mlp.0.weight` in both encoders (75/77 tensors) — the obstacle
encoder is RANDOM and the policy scores far below its logged 2.45/3.  Use
the training-log det evals, or a checkpoint trained after the radius
change, rather than re-evaluating that file.

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
- **Crash-penalty sweet spot.**  `pool_fixed_v1` used 2.0/1.0;
  `pool_fixed_v3` raised to 5.0/5.0, which ~halved ally-crash events but
  cost a little capture + mean return (caution → longer detours).  Crash
  events are near a floor (~1/episode), so returns diminish past this.
  **~3.0/3.0 is the likely sweet spot** — an ablation candidate before
  the moving-obstacle runs, where the right balance may shift (a moving
  hazard makes crashes both more likely and more costly).  See
  `stage4_results.md` §3 "Crash-penalty ablation".

- **Best-checkpoint selector on crash-penalty runs** — ✅ **FIXED.**
  `best_ckpt_metric='mean_return'` mis-selected when crash penalties are
  on: caution lowers return over training, so `mean_return` peaked at
  ~rollout 1 and `best.pt` was saved as ≈ the warm-start (observed on
  `pool_fixed_v3`).  Added two DETERMINISTIC selectors to
  `--best-ckpt-metric`: `det_caught` (deterministic mean caught) and
  `det_composite` (`det_caught − λ·(obstacle+ally crash events)`, λ via
  `--best-ckpt-crash-lambda`, default 0.5).  These update best-ckpt at
  eval time (require `--eval-interval > 0`; a startup guard errors
  otherwise).  **Use `--best-ckpt-metric det_composite` for the curriculum
  runs.**  The `mean_return` / `mean_caught` stochastic selectors are kept
  for back-compat and remain the default.  See the best-ckpt block +
  `_maybe_save_best` in `scripts/train_stage4.py`.
