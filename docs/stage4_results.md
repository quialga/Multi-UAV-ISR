# Stage 4 — Results

Stage 4 adds **partial, noisy perception** on top of the Stage 3
pursuit task: a Bayesian belief map (log-odds occupancy grid), static
**obstacles** with line-of-sight **occlusion**, and **sensor noise**.
The blues no longer receive ground-truth enemy/obstacle positions —
they must act on a fused, uncertain picture of the world, which is the
first real step toward what a fielded ISR UAV actually sees.

The headline result: **the belief-driven policy reaches the same
capture performance as the fully-observable oracle** (3/3 catches in
training), paying only a small, honest cost in convergence speed and
final reward for acting under uncertainty instead of ground truth.

---

## Final architecture (v6.x)

The Stage 4 architecture went through a long redesign (see the version
log below); what landed is the **proven Stage 3 typed-GNN CTDE policy,
unchanged, fed by a model-based perception front-end**.  No CNN
anywhere.

### Perception front-end (env-internal, model-based)

- **Mission-command belief map** `(2, 26, 26)` — a log-odds grid
  maintained at the mission command layer, not by the UAVs
  individually.  It is an **environment-level latent** used for
  target tracking and evaluation, in the same category as
  `true_occupancy`, with the crucial difference that the belief map
  is *noisy* (command fuses raw sensor returns with the sensor
  model's error) whereas `true_occupancy` is ground truth used only
  by the CTDE critic.  Bayesian fusion of independent sensor returns
  is log-odds addition, so all UAVs' observations accumulate into
  the same grid.  Channel 0 = P(enemy), channel 1 = P(obstacle).
  5 m cells.  See `docs/stage4_backlog.md §13` for the doctrine
  discussion and the follow-on that would gate track delivery per
  UAV via a command link.
- **Bayesian update** each step: predict → update.
  - *Predict* (enemy channel only, Phase A): decay `L ← 0.99·L`
    (forgetting → stale tracks fade, cleared regions re-acquirable) +
    isotropic diffusion `p_move=0.2` in probability space (random-walk
    motion model, calibrated to the red's ≤1 m/s speed). Obstacles are
    static → prediction = identity.
  - *Update*: log-odds evidence from the noisy sensor
    (`p_TP=0.85`, `p_FP=0.15`), gated by an **exact analytic
    segment–disk occlusion test** (no sampling).
- **Detection-seeded track extraction**: the K = n_red enemy track
  slots are filled *detection-first* — one live track per red any blue
  currently sees (seeded directly from the sensor, continuous measured
  position + Doppler velocity), remaining slots filled from belief-map
  peaks (NMS-deduplicated) for unseen reds. Dead slots (captured reds)
  are conf-0 padded. This bypasses belief-map lag / NMS-collapse /
  association fragility for anything under direct observation.

### Policy (typed GNN + GRU + CTDE — identical to Stage 3)

- Typed graph: **blue + red + obstacle** nodes; **bb / rb / ob** edges
  (7-D each: `rel_pos, rel_vel, range, bearing_cs`). Separate input and
  edge MLPs per entity type (no one-hot labels needed — type is
  architecturally encoded).
- **Actor** consumes the belief-derived graph (noisy positions,
  visibility-gated); **critic** (CTDE) consumes the ground-truth graph.
- GRUCell per blue for belief tracking; Stage 3 opt-1 hidden-in-GNN.
- **~144 k params** (n_obstacles=0) / ~178 k (with obstacles) — Stage 3
  scale.

---

## Results

Two runs, both under the Stage 3 winning recipe (see *Training
arguments*), 400 rollouts, `stationary:1,random:1,run:1` red mix.
Numbers are the **deterministic** `[det eval]` metric (greedy actions,
20 episodes per red type, seed base 30000) at rollout 400.

| Run | stationary | random | **run** (evader) | **mean** |
|---|---|---|---|---|
| `belief_v3` — no obstacles | **3.00/3** | **3.00/3** | **3.00/3** | **3.00/3** |
| `obstacles_v1` — 4 obstacles + occlusion + collision-aware evader | 2.95/3 | 3.00/3 | 2.90/3 | **2.95/3** |

**Headline: −0.05 catches** for the full realistic difficulty stack
(obstacles + line-of-sight occlusion + a collision-avoiding evader
that steers around obstacles instead of pinning itself on them) —
and this on top of an intentional handicap: `obstacles_v1` warm-started
from the Stage 1 checkpoint with the **obstacle-branch critic cold**
(29 tensors copy, 30 for the no-obstacle run). A tiny gap under stacked
difficulty is a strong result for the perception layer.

**Convergence speed** (deterministic mean over rollouts):

| Milestone | `belief_v3` | `obstacles_v1` |
|---|---|---|
| mean ≥ 2.5 | rollout 75 | rollout 150 (~2× slower) |
| mean ≥ 2.9 | rollout 150 | rollout ~400 (only at the end) |
| first mean = 3.00 | rollout **275**, held for 6 evals | never (peaks 2.95) |

Two things worth noting in the trajectories:

- **The `run` column becomes the hardest metric under obstacles**
  (2.90) — reversing `belief_v3`'s ordering where all three red types
  hit 3.00 equally. That's exactly the signature of the collision-
  aware evader working as intended: the difficulty now shows up on
  the harder adversary, not on the strawman.
- `obstacles_v1` plateaus around 2.7-2.85 from rollout 175-375 and
  only reaches 2.95 on the final eval — it *may* not be fully
  converged. A longer run (600 rollouts) or the two-phase curriculum
  (warm the obstacle-critic branch via a full-obs pretrain) are the
  natural escalations if the last 0.05 matters.

---

## Training arguments (Stage 3 winning recipe, now the Stage 4 default)

```
lr                 1e-4  (linear decay to 0.1×)
ent_coef           0.008
target_kl          0.03
n_msg_rounds       2
aux_hidden_coef    0.2          # MSE(actor_h_blue, critic_h_blue.detach())
warm_start_critic  runs/stage1/scaling_gnn/best.pt
use_hidden_in_gnn  True         # Stage 3 opt-1
n_envs 16, rollout_steps 300, n_epochs 10, mb_size 512
gamma 0.99, gae_lambda 0.95, clip_eps 0.2, vf_coef 0.5
```

Belief-map / perception knobs:
```
belief_grid_size 26, belief_channels 2, belief_clip 10
p_TP 0.85, p_FP 0.15, ray_step_size 2.5 (occlusion margin)
enemy_belief_decay 0.99, enemy_belief_diffusion 0.2
sensor_pos_noise_std 1.0
```

Reproduce (both runs):
```
# belief_v3 — no obstacles
python scripts/train_stage4.py --n-rollouts 400 --n-obstacles 0 \
  --red-policy-mix stationary:1,random:1,run:1 \
  --eval-interval 25 --run-name belief_v3

# obstacles_v1 — 4 obstacles + occlusion + collision-aware evader
# (uses default n_obstacles=4; the run heuristic auto-avoids
# obstacles when they are present)
python scripts/train_stage4.py --n-rollouts 400 \
  --red-policy-mix stationary:1,random:1,run:1 \
  --eval-interval 25 --run-name obstacles_v1
```

---

## The redesign journey (what didn't work, and why)

Stage 4's first design fed the raw belief grid to a **CNN**. It never
learned — capture ratio stuck at ~1.0–1.4 across every variant. The
debugging trail is the scientifically valuable part:

- **v1–v2 (CNN over global map)** — no convergence. The CNN produced a
  global summary vector; the policy couldn't recover *where* a target
  was relative to a UAV.
- **v3 (ego-centric window CNN)** — still stuck. Translation-
  equivariance wasn't the (only) issue.
- **v4 (2.5 m grid, 52×52)** — resolution wasn't the issue either.
- **v5 (rb_edges restored)** — the key realization: **the belief map is
  a static snapshot with no velocity; intercepting an evader is
  impossible without it.** Restoring Stage 3's rb edges (position from
  belief, velocity from radar) started real learning.
- **v5.1–5.3 (sensor-physics split, obstacle peaks, drop CNN)** — an
  ablation (user-run) confirmed the CNN could not extract position from
  the belief tensor; explicit peak detections + graph edges could.
- **v6 (back to the Stage 3 typed GNN, no CNN)** — the decisive
  restructure: reconstruct the *exact* Stage 3 graph from belief peaks,
  add obstacles as a third typed node. Params dropped 860k → 144k.
- **v6.1 (global fused map, exact occlusion, grid 26)** — team-shared
  belief; analytic segment–disk occlusion (the sampled ray-march could
  miss grazing chords).
- **Phase A + NMS + detection-seeding** — Bayesian prediction step
  (forgetting + motion diffusion), non-maximum suppression so one blob
  = one track, and detection-first slot filling so direct sensing
  never waits on the memory layer.

### The load-bearing insight: it was a config regression, not the map

Even with the perception fixed, both oracle and belief plateaued
(~1.4–2.36). The cause was **not** the belief map — the v6 runs had
silently dropped **three coupled Stage 3 stabilisers at once**:

| Knob | Stage 3 | v6 (broken) |
|---|---|---|
| `n_msg_rounds` | 2 | 1 (half the coordination depth) |
| `warm_start_critic` | Stage 1 ckpt | off (cold → garbage early advantages) |
| `aux_hidden_coef` | 0.2 | 0 |

Critically, these are **coupled**: aux 0.2's target is
`critic_h_blue.detach()`, so it only helps once the critic is warm-
started — which is exactly why an earlier "aux 0.02" experiment (cold
critic) *hurt*. Reading Stage 3's `train_log.txt` (stochastic 2.94/3 on
the harder mix) settled that this was a real regression, not a
stochastic-vs-deterministic artifact. Restoring all three at once (and
pinning them as the config default) recovered 3/3 on both paths.

Lesson: when porting a proven RL setup, the stabilisers are a package —
change one variable at a time, and diff against the working config's
actual logged args before concluding an architectural fault.

---

## Diagnostics added

- `--eval-interval` / `evaluate_policy_deterministic` — the greedy
  (deterministic) capture metric, comparable to Stage 3's eval. The
  per-rollout `caught` stat is stochastic and always lower.
- `belief_track_error()` (`trk=` in the log) — mean distance from each
  extracted enemy peak to the nearest true red; measures whether the
  belief map is tracking or lagging, independent of the policy.
- *Scripted-pursuit sufficiency check* (was
  `scripts/diag_scripted_pursuit.py`, removed 2026-08) — a hand-coded
  controller driven only by the real `rb_edge_features` captured 3/3 on the
  L=130 / 5v3 / no-obstacle config, proving the observation was
  *sufficient* to solve that task and that any policy shortfall there was a
  learning problem, not an observability one.  The script was deleted
  because it monkey-patched the private `_build_enemy_tracks` (broken by the
  velocity-fusion signature change), hardcoded the retired config, and
  A/B'd against a cell-centre-peak path that no longer exists.  **The
  question is worth re-asking on the L=200 / 7v4 baseline**, where ~92% of
  observations are memory tracks — but that needs a fresh check against the
  current obs, not a repair of this one.

---

## What remains on the table

- **Close the last 0.05 on `obstacles_v1`.** `obstacles_v1` still hit
  its peak (2.95) at the final eval and was warm-started with the
  **obstacle-critic branch cold** (29/30 tensors). Two natural
  escalations: (a) a longer single-phase run (400 → 600 rollouts,
  same command) since the trajectory was still trending up; or (b)
  the **two-phase curriculum** — Phase 1 pretrains with a
  full-obs actor on obstacles to produce an obstacle-aware critic,
  Phase 2 warms Stage-4-with-belief from that Phase 1 checkpoint.
- **Crash penalty.** ✅ LANDED post-close — per-agent obstacle + ally
  crash penalties (see "Post-close extensions" below and
  `docs/stage4_backlog.md §1/§2`).
- **Occlusion-seeking evader.** The evader currently avoids obstacle
  *collisions* but does not deliberately hide behind them to break
  line-of-sight — the "boss" adversary that weaponizes occlusion,
  held in reserve as a harder stress test.
- **Phase B — per-target Bayesian filter.** Phase A is an isotropic
  approximation on the shared log-odds grid. Phase B (per-target
  normalised distributions, velocity-directed anisotropic prediction
  using the radar velocity, data association) is the principled next
  step for tracking evaders through occlusion; deferred until a
  concrete weakness in the memory layer motivates the cost. See
  `docs/stage4_backlog.md`.
- **Sensor-noise robustness sweep.** Report performance vs `p_TP/p_FP`
  and `sensor_pos_noise_std` to characterise graceful degradation.

---

## Stage 4 — closed.

Both regimes land above 2.9/3 captures under deterministic evaluation
against a mixed red distribution (stationary + random + collision-
avoiding evader):

- **No-obstacle regime** (`belief_v3`): **3.00/3** — the belief-
  driven policy matches the fully-observable oracle exactly.
- **Full-difficulty regime** (`obstacles_v1`, 4 obstacles + occlusion
  + collision-aware evader, obstacle-critic branch cold-started):
  **2.95/3** — a 0.05-catch honest cost for the entire realistic
  perception stack.

The perception front-end (global Bayesian belief map → prediction
step → detection-seeded tracks → typed GNN with obstacle nodes) is
validated. The model-based / learning-based split the design argued
for holds up empirically. The remaining open items (closing the last
0.05 under obstacles via longer training or the two-phase critic
curriculum, occlusion-seeking evader, Phase B per-target filter, noise
sweep) are extensions and stress tests, not blockers on the core stage.

---

## Post-close extensions (2026-07)

Four capability extensions landed on top of the closed Stage 4 baseline,
each on its own branch, each fully unit-tested (**84 tests green**), and
each **byte-preserving the v6.x behaviour when its knobs are at their
off defaults**. Two are trained + measured; two are implemented and
awaiting a training run.

### 1. Crash avoidance — per-agent penalties (MEASURED) · `feature/crash-avoidance`

Real UAVs crash; the v6.x policy learned to *graze* obstacles because
hitting one was a free soft-stop. Added an **individual** (not shared)
crash penalty so each UAV owns its own mistakes:

- Reward decomposition **`r_i = r_team + r_crash_i`** — the catch/step
  reward stays team-shared; the crash penalty is charged only to the
  offending blue. Covers both blue↔obstacle and blue↔blue collisions
  (symmetric; `blue_collision_radius = 2 m < 3 m` capture radius). A
  crash is a soft-stop (rollback + zeroed velocity); the episode is
  **not** terminated.
- This required moving the whole Stage 4 RL path from shared to
  **per-agent**: the critic now estimates an **agent-conditioned
  `V(s, i)`** (it reads blue *i*'s own post-message-passing node
  embedding instead of the pooled `sum(h_blue)` — same trunk width, so
  a shared-reward checkpoint still warm-starts), GAE / advantages /
  returns are per `(env, agent)`, and dones stay per-env.
- **Warm-start both actor and critic** from the converged obstacle
  policy (`--warm-start-full`, `load_full_stage4`): the run starts
  already flying + catching and only has to learn to avoid crashes.

**Result (observed, warm-started run + longer run in progress):**
capture held at **~2.95/3** while obstacle+ally crashes per episode fell
roughly an order of magnitude — from **~40** (unpenalised baseline) to
**~3–5**. A 1000-epoch run is underway to push the residual lower; the
trend suggests more epochs help (the crash objective is a small
correction on an already-good policy).

### 2. Obstacle live-sensor refinement (MEASURED) · `feature/crash-avoidance`

The actor's obstacle node position was always the belief-map **peak**
(grid-quantised to ~half a cell). Under the crash penalty that forced a
conservative safety margin the policy couldn't resolve. Mirroring the
enemy-track treatment, an obstacle a blue currently senses now supplies
its **precise own-radar position** (true centre + `sensor_pos_noise_std`
noise) — the belief peak is used only for out-of-sensor obstacles. This
lets blues hug boundaries tightly and safely, and contributed to the
crash reduction above (seen-obstacle position error → ~0 vs ~half a cell
for the peak).

### 3. Variable entity counts + count-agnostic critic (fixed-count baseline validated; variable-N training pending) · `feature/variable-entities`

One policy that trains on — and generalises across — a **variable number
of reds and obstacles**. The actor was already count-native (per-node
GNN); the change was in the **critic's global context**, swapping the
old flatten (`h_red.reshape` → width `d + N_red·d + N_obs·d`, which
hard-coded the counts) for a **masked-MEAN pool over active nodes + a
normalised count scalar** (width `d + (d+1)[+(d+1)]`, count-independent).
`n_red`/`n_obstacles` become a padded **capacity**; `n_red_min` /
`n_obstacles_min` make each reset sample the active count in
`[min, capacity]`, with unused slots padded inactive (reusing the
caught-red machinery, so they're invisible to detection/capture/edges).
Buffer + vec_env are unchanged (shapes stay at capacity). Enables the
"train on 2–4, evaluate zero-shot on 6" result once trained — measured
with the count-sweep harness `scripts/eval_stage4_counts.py` (blue / red
/ obstacle axes), documented in
[`stage4_generalization_eval.md`](stage4_generalization_eval.md).

*Architecture note:* this pool change narrows `critic_trunk.0.weight`
(512 → 194 for `n_red=3,n_obs=4`), the **only** tensor that can't warm-
start from a pre-pool checkpoint. `load_full_stage4` transfers the other
**76/77** tensors (whole actor + both GNN encoders + value head) and now
**names** any tensor it leaves at fresh init.

*Fixed-count baseline — the pool costs nothing* (`pool_fixed_v1`, GPU
run, 2026-07-27). To rule out that the masked-mean pool degrades the
critic at a fixed count, we retrained the **fixed** task from scratch
with the pool — `n_blue=5, n_red=3, n_obstacles=4` static, crash
penalties `2.0 / 1.0`, no warm start (`warm_start_full=None`), default
recipe (`lr 1e-4`, `ent 0.008`, 1000 rollouts, 128 envs):

```
python scripts/train_stage4.py --device cuda --n-envs 128 --mb-size 2048 \
    --n-workers 8 --n-epochs 4 --n-rollouts 1000 \
    --crash-obstacle-penalty 2.0 --crash-blue-penalty 1.0 \
    --eval-interval 25 --run-name pool_fixed_v1
```

Final deterministic eval **`stat=3.00  rand=3.00  run=2.90  mean=2.97/3`**.
This **matches / slightly beats** the pre-pool flatten baseline
(`obstacles_v1` ≈ 2.95/3), confirming the flatten→pool swap is
capture-neutral at fixed count. The slight dip seen in an earlier pool
run was a **warm-start confound** — a `--warm-start-full` from a *flatten*
checkpoint silently dropped the 512-wide trunk into the 194-wide pool
slot, compounded by throttled fine-tune `lr/ent`; a clean from-scratch
run (Stage-1 encoder warm-start + normal cold trunk, identical to how the
flatten baselines started) closes the gap. `pool_fixed_v1/best.pt` is now
the correct **pool** warm-start base for the variable-N / moving-obstacle
curriculum (pool→pool ⇒ the critic trunk transfers cleanly).

*Crash accounting is now in the deterministic eval too* (TB
`eval_det/{obstacle,ally}_crashes`, appended to the `[det eval]` log line
when penalties are on). It counts **distinct crash EVENTS** — rising
edges per blue, so a UAV that lingers inside an obstacle for many steps
counts **once** (until it leaves and re-enters) — *not* the per-step
occupancy the training-loop `crash(o/a)` stat sums. Measured on
`pool_fixed_v1/best.pt` (20 eps/red, `max_steps=200`): **≈0.9–1.1
obstacle events and ≈1.1–2.0 ally events per episode** (lowest vs
stationary reds, highest vs `run`) — a reassuring deployment number,
distinct crashes are rare. Two things to keep straight when reading it
against the training stat:

- **They measure different things and are not directly comparable.** The
  training `crash(o/a)` stat is per-step *occupancy* — a blue camped on a
  boundary counts every step (≈3.1/2.8) — whereas the event count is
  *incidents* (≈1). The deployed policy genuinely **hugs** boundaries
  (high occupancy) but rarely enters them *anew*, so by incident count it
  is safe; a hard "any-contact-destroys" model would judge it more
  harshly than this soft-stop one does.
- **It is not post-capture camping.** The post-capture crash bucket is
  ~0; the events occur on-mission while cornering the last surviving red
  (episodes run to `max_steps` because ≈1 red typically escapes).

*Crash-penalty ablation* (`pool_fixed_v3`, warm-started full from
`pool_fixed_v1`). Raised both penalties **2.0/1.0 → 5.0/5.0** and
lengthened episodes (`max_steps 200 → 400`), fine-tune recipe
(`lr 4e-5, ent 0.002`), on the same fixed 5/3/4 task (obstacle radii
5–15 m):

| | caught/3 | obstacle events | ally events |
|---|---|---|---|
| `pool_fixed_v1` (2.0/1.0) | ~2.95–2.97 | ~0.9–1.1 | ~1.1–2.0 |
| `pool_fixed_v3` (5.0/5.0) | 2.88–2.95 | ~0.8–1.1 | **~0.7–0.9** (late) |

The higher penalty **roughly halves ally-crash events** (~1.3 → ~0.9
avg) and leaves obstacle events flat, at a small capture cost. Two
findings worth recording:

- **Diminishing returns + a caution cost.** Crash events were already
  near a floor (~1/episode); pushing the penalty to 5/5 traded a slight
  capture dip and *lower mean return* (the policy takes wider, longer
  detours around obstacles — caution costs steps) for the ally-crash
  gain. The residual ~1 crash is inherent to the task: cornering a red
  hiding beside a large obstacle forces a gap-threading approach. **The
  4-obstacle env is not "too hard" — at ~2.95/3 with ~1 crash it is
  effectively solved.** A ~3.0/3.0 penalty is the likely sweet spot; see
  the design-questions note.
- **`best_ckpt_metric='mean_return'` mis-selects on crash-penalty runs.**
  Because caution *lowers* return over training, `mean_return` peaks
  early — `pool_fixed_v3/best.pt` was saved at **rollout 1** (≈ the
  warm-start), not the improved-crash policy. The only v3 checkpoint that
  reflects the 5/5 training is `checkpoint_00100`. Every crash-penalty
  run hits this; fix the selector (track `mean_caught`, or a
  `caught − λ·crash` composite) before the curriculum runs.

**Warm-start base chosen for the variable-N / moving-obstacle curriculum:
`pool_fixed_v1/best.pt` (rollout 922, fully converged, 2.97/3).** v3 has
no converged best-checkpoint (its `best.pt` is rollout 1; `checkpoint_00100`
is a mid-training, lower-capture snapshot), and the static-obstacle
caution v3 added is re-learned during the curriculum anyway (moving
obstacles need *anticipatory* caution, learned fresh). Carry v3's
**5.0/5.0** penalties into the curriculum runs, and fix the checkpoint
selector first.

### 4. Moving obstacles — reciprocating patrol (implemented; first run diverged — see §5) · `feature/moving-obstacles`

Backlog §4 with the simplest kinematics: a fraction of obstacles patrol
back-and-forth along one axis, bouncing off the arena walls
(`moving_obstacle_fraction`, `obstacle_speed`). Chosen deliberately over
missiles-that-destroy-blues — it reuses the per-agent crash penalty (no
attrition ⇒ no variable *active-blue* count, no new termination/reward
machinery) while still adding the real new content: **time-varying
belief truth** the policy must track and anticipate.

- Obstacle **velocity** now flows into the graph (true for the CTDE
  critic; own-radar Doppler for the actor when a moving obstacle is a
  live/seen track) so blues can anticipate the sweep.
- The obstacle occupancy grid, previously cached-once (static
  assumption), is recomputed each step when obstacles move so
  belief-truth + occlusion track them.
- `obstacle_belief_decay` (default `1.0` = off) fades the stale "comet
  trail" a moving obstacle leaves on belief channel 1 (decay-only — no
  diffusion; reciprocating motion isn't a random walk). It is
  knob-gated, not auto-linked to motion, so enable it together with
  `obstacle_speed`.

**Recommended next run:** warm-start moving-obstacles from the crash-
avoidance policy (`--warm-start-full`), ramping `obstacle_speed` up from
a low value (curriculum knob), optionally combined with `--n-red-min` /
`--n-obstacles-min` for variable counts in the same run.

### 4b. Live tracks now obey the same sensor model as the belief map

**The inconsistency.**  The live-track gate was **range-only**, so a target
inside `sensor_radius` produced a `conf = 1.0` measurement **every step**,
even **through obstacles** — while the belief map, fed by the *same* radar,
applied `p_TP = 0.85` / `p_FP = 0.15` and an exact line-of-sight test.  A
real radar is one chain: **detect (P_d) → measure (position + Doppler)**.
You cannot miss the detection and still report an accurate position.  The
practical consequence was that any red within 40 m in a 130 m arena was
effectively **oracle information**, which makes the Stage 4 "partial, noisy
perception" framing weaker than it reads — **results above this section
were obtained under that near-oracle in-range regime.**

**What landed** (all with escape hatches to reproduce pre-fix runs):

| knob | default | effect |
|---|---|---|
| `track_occlusion` | `True` | a live track needs clear line of sight |
| `track_detection` | `True` | the `p_TP` draw must fire; a miss yields **no measurement** and the track coasts on the belief/memory path |
| `sensor_vel_noise_std` | `0.1` | Doppler is measured, not exact (small vs the 1.0 m position noise — radar gets velocity from phase, not by differencing positions) |
| `track_conf_min` | `0.5` | confidence falls with range (SNR proxy): `conf(r) = c_min + (1−c_min)(1−(r/R)²)` |
| `sensor_noise_range_growth` | `1.0` | **accuracy** falls with range too: `σ(r) = σ_base(1 + g(r/R)²)` |

The last two are deliberately coupled: confidence and accuracy are two
consequences of the *same* SNR falloff, so degrading one while holding the
other constant would be internally inconsistent.  Confidence depends on
**range only** — never on true-vs-false status, which would leak ground
truth into the actor (a false alarm is indistinguishable from a real
detection at detection time; real systems separate them over *time* via
track score, not instantaneously).

**Measured** (5 seeds, random-walk blues, 4 obstacles):

| | pre-fix | realistic |
|---|---|---|
| live tracks / in-range events | 100.0% | **94.2%** |
| mean live position error | 1.25 m | **2.11 m** |
| mean live confidence | 1.00 | **0.71** |

94.2% (rather than 85%) is correct and worth understanding: a red in range
of several blues gets an **independent detection draw per blue**, so fusion
across platforms genuinely raises detection probability — `1 − (1−p_TP)ⁿ`.

Occlusion-through-walls was rare in the current 4-obstacle map (**0.9%** of
in-range events had *every* seeing blue blocked), so this is a correctness
fix rather than an explanation of past training results; it would matter far
more with fewer blues, larger obstacles, or an urban map.

Still open: **radial/tangential Doppler** — the actor still gets the full
2-D velocity vector, whereas a real radar measures only the radial
component well.  See `stage4_backlog.md` §17.

### 5. Curriculum training results — wall-repelling reds + the crash problem

Two runs on top of the wall-repulsion fix (§backlog "Red wall-repulsion")
and the crash-aware selector: a fixed-obstacle baseline, then the first
moving-obstacle attempt.

**`pool_fixed_v4` — the honest fixed baseline (wall-repelling reds).**
From scratch, `n_blue=5 n_red=3 n_obstacles=4` static, radii 5–15 m,
`max_steps=200`, crash `2.0/2.0`, default recipe (`lr 1e-4, ent 0.008`,
1000 rollouts, 512 envs), `--best-ckpt-metric det_composite`.

| run | mean | stat | rand | **run** | crash events (o / a) |
|---|---|---|---|---|---|
| `pool_fixed_v1` (old pinning reds) | 2.97 | 3.00 | 3.00 | 2.90 | ~1 / ~1–2 |
| **`pool_fixed_v4` (wall-repelling reds)** | **2.78** | 2.95 | 2.95 | **2.45** | 1.25 / 1.17 |

The entire drop is on the **fleeing `run` red (2.90 → 2.45)**; `stat`/`rand`
are unchanged (those reds don't flee). That ~0.45 is the **wall-camping
gift being removed** — 2.78/3 is the real capture number against a
non-degenerate evader, not a regression. Two plumbing confirmations: the
selector fix works (`best.pt` at **rollout 950**, no more rollout-1
mis-save), and crash *events* stay ~1.2. This is a solid base; more
epochs might lift `run` slightly but with diminishing returns.

**`moving_v1` — first moving-obstacle attempt (diverged; stopped ~225).**
Warm-started from `pool_fixed_v4`; `moving_obstacle_fraction=0.25`,
`obstacle_speed=1.0`, `obstacle_belief_decay=0.9`, `max_steps=320`, crash
`3.0/3.0`, `lr 4e-5, ent 0.002`.

Symptom: return slid **`+5.76 → +2.14 → … → −1.82`** (went negative),
capture bled `2.93 → 2.70`, and ally-crash events *rose* (`~1.4 → ~2.1`)
while obstacle occupancy only halved (`crash(o) 10.2 → 6`). KL (~0.002)
and clip (~0.01) stayed tiny — **not** a numerical/PPO blow-up, a steady
*degradation on the objective*.

Diagnosis — a **reward-conflict collapse**, not perception:

- **The per-step crash penalty is mis-specified for *moving* obstacles.**
  A blue *swept over* by an obstacle eats the penalty every step it's
  pinned (`crash(o)=10` occupancy at rollout 1), even though it couldn't
  avoid the sweep. At `3.0` over 320 steps that term comes to **dominate
  the return** — larger than the whole catch reward — so the optimiser is
  driven by an avoidance signal the warm-started (static-obstacle) policy
  can't yet satisfy. It degrades capture trying to cut crashes, can't cut
  them enough, and slides on both.
- **Too-gentle recipe.** `lr 4e-5 / ent 0.002` (fine-tune values) can't
  adapt to genuinely new dynamics fast enough to escape the high-crash
  basin. (Higher *entropy* is **not** the fix — for a safety task more
  exploration means more erratic actions near obstacles; keep it low,
  consider annealing down.)
- **`max_steps` 200 → 320 warm-start mismatch** miscalibrates the
  warm-started critic (`val ~12–15` early) to the longer horizon.

**Direction (crash-avoidance as a hard goal).** The operating stance
shifted: *a crash = losing a drone*, so near-zero crashes matters more
than a marginal catch. Key realisation — **a scalar penalty term can't
reliably reach ~0 crashes** (it's a soft trade-off; the policy accepts
crashes when catches outweigh them, and cranking it just reproduces this
collapse). The intended fix is a **dense clearance/barrier shaping**
reward (deepest inside an obstacle, decaying outward, so its *gradient*
points continuously toward clear space — a proactive "keep distance" plus
a real "get out, this way" signal that the flat step-penalty never gave),
paired with a **discrete crash-event = drone-loss** semantics (ultimately
destroy-on-crash, now feasible via the variable-entity machinery), under
a **motion curriculum** (`fraction 0.1 / speed 0.5` first) with the
**default `lr 1e-4`, low entropy**. See `stage4_backlog.md` §5 / §10 /
§15 for the perception-side options and the safe-RL framing.

---

### 6. The big-batch recipe does not train, and the task grew (MEASURED, 2026-09-19/21) · `feature/target-tracking`

Six runs on a rented RTX 3090 (32 vCPU), while preparing the
belief-map-vs-tracker comparison. The comparison never started: the
control arm would not learn.

#### 6.1 The symptom

| run | task | recipe | outcome |
|---|---|---|---|
| `belief_ctrl_s0` | 7/4/9, crash 2.0/2.0 | `n_envs 1024, rollout_steps 128, mb 16384, epochs 4` | det eval **0.52 → 0.43 → 0.40** at rollouts 75/100/125, entropy RISING 2.84 → 3.25, `kl ≈ 0.0003`. Stopped at 125. |
| `nopen_probe` | same, crash **0/0** | same | det eval 0.28 @50, 0.35 @75; `caught` flat at 0.57–0.58 for 75 rollouts; `kl = 0.0000`. |

Removing the crash penalties changed nothing, which killed the first
hypothesis (that avoidance was drowning the capture signal). The reward
decomposition says why it was wrong: `r_team = 10·n_caught − 0.05/step
[− 5·n_uncaught at the end]`, so each extra red caught is worth **+15**
and the terminal term already points hard at capture. Measured, the
crash penalties were worth only ~5.2 of ~25 return points (epR −24.96
with them, −19.77 without, same rollout).

#### 6.2 The cause: minibatch size, not episode truncation

A 2×2 on the smallest possible task (1 blue, 1 stationary red, no
obstacles, 60 rollouts, det eval at the end):

| | `mb 4096` | `mb 512` |
|---|---|---|
| **truncated** (`rollout_steps 128`) | `tiny_b1` **0.03/1** | `abl_smallmb` **0.47/1** |
| **whole episodes** (`rollout_steps 200`) | `abl_fullep` **0.05/1** | `tiny_oldrecipe` **0.42/1** |

Sample counts per rollout were matched (~32k) so only the axis under test
varies. The column decides and the row does not: **`mb_size` is the
lever**; collecting whole episodes is not required. `abl_fullep` ended at
`kl = 0.0000  clip = 0.000` — the policy never moved.

With one agent there is no credit-assignment confound, which also rules
out "7 agents sharing a team reward" as the cause.

**Health check, cheap and reliable:** `kl ≈ 0.000` with `clip = 0.000`
rollout after rollout means the run is dead, whatever else looks
plausible. Healthy is ~0.002–0.03; above that `target_kl` starts binding
and `eps` drops below `n_epochs`.

**Consequence:** `n_envs` can go back up for throughput as long as
`mb_size` stays small — but updates per rollout scale with `n_envs` at
fixed `mb_size`, so that needs dimensioning rather than copying.

#### 6.3 The corrected recipe trains, and still falls far short

`n_envs 64, rollout_steps 200, mb 512, epochs 10`, 600 rollouts each:

| run | task | result |
|---|---|---|
| `b5r3_long` | 5/3, no obstacles, **arena 200**, `max_steps 200` | **1.02/3** |
| `real749_long` | 7/4/9, crash 2.0/2.0, **arena 200**, `max_steps 200` | **0.57/4** (166.6 min) |

`real749_long` in detail: det eval 0.57 @225, 0.60 @475, 0.57 @600, peak
0.68 @125 — flat. Learning was real (entropy 2.84 → 1.57, crash
occupancy `o=11.4/a=10.5` @25 → eval `o=3.68/a=2.03` @600) but the `run`
(fleeing) column sat at **0.05 throughout** and never moved.

`b5r3_long` spent **7.7 M env steps** against `belief_v3`'s 1.9 M — 4×
the data for a third of the result. So it is not a data-budget problem.

#### 6.4 Why: the arena grew and the command undid the compensation

Commit `2dc0528` (2026-08-25, *"scale the Stage 4 arena to L=200 so
search and the belief map matter"*) changed, together: `arena_size`
130 → 200, `max_steps` → **300**, `rollout_steps` → **320**,
`belief_grid_size` 26 → 40, `n_obstacles` 4 → 9 (and `n_blue` 5 → 7,
`n_red` 3 → 4). Before it, Stage 4 inherited `arena_size 130` from
`stage3_default` and `max_steps 200` from `stage1_default` — those are
`belief_v3`'s conditions.

Sensor coverage `C = n_blue·πR²/L²` and episode length in arena
crossings (`v_max 1.5`):

| | arena | `max_steps` | coverage | crossings |
|---|---|---|---|---|
| `belief_v3` (3.00/3) | 130 | 200 | **1.49** | **2.3** |
| config as written today | 200 | 300 | 0.88 | 2.25 |
| `b5r3_long` (1.02/3) | 200 | 200 | **0.63** | **1.5** |
| `real749_long` (0.57/4) | 200 | 200 | 0.88 | 1.5 |

Two compounding handicaps against the run that scored 3.00/3: **42% of
the sensor coverage and 65% of the episode length**. The second one was
self-inflicted — every run in this batch passed `--max-steps 200`,
carried over from a command line written before `2dc0528`, overriding the
300 the config had raised precisely to preserve the crossing count.

**NOT ESTABLISHED:** that this fully explains the gap. It is a sufficient
explanation on paper; `bv3_repro` (arena 130, 5/3, no obstacles,
`belief_grid_size 26`, `max_steps 200`, corrected recipe, 600 rollouts)
is the run that settles it.

It cannot settle it alone, though, because the arena is not the only
thing that changed since `belief_v3`. The full delta is:

1. **§4b, the live-track sensor model** — the big one. `belief_v3` ran
   when a red inside `sensor_radius` yielded `conf = 1.0` and an exact
   position every step, through walls; §4b itself flags that "results
   above this section were obtained under that near-oracle in-range
   regime". Today detection must pass the `p_TP` draw, needs line of
   sight, and position error grows with range. The 94.2% detection
   figure §4b reports is for a red seen by SEVERAL blues (fusion gives
   `1 − (1−p_TP)ⁿ`); a red seen by ONE blue — the normal case for a
   dispersed search team — is 85% per step, and a 2.11 m position error
   against a 3 m capture radius makes the final closing phase materially
   harder.
2. `enemy_belief_decay` 0.99 → 0.9935.
3. No warm-started critic and `aux_hidden_coef 0`, the two stabilisers
   `belief_v3` had.
4. A different PPO geometry (16×300 then, 64×200 now; `mb 512` and 10
   epochs in both).

So three outcomes are possible, not two: back near 3/3 (difficulty
explained it all), stalled near 1/3 (something else is wrong), or
somewhere between (the arena explains part and §4b the rest, with no
regression). §4b left escape hatches, so it isolates in one run:
`--no-track-occlusion --no-track-detection --track-conf-min 1.0
--sensor-noise-range-growth 0.0 --sensor-vel-noise-std 0.0` at arena 130
reproduces `belief_v3`'s perception exactly, warm start aside.

#### 6.5 The honest baseline: 1.2/3, and why 3.00/3 is not the target

`bv3_repro` and a warm-started replicate both plateau at the same place,
so this is a reproducible ceiling rather than one noisy number:

| run | what | result |
|---|---|---|
| `bv3_repro` | cold, 600 rollouts | **1.17/3** (`stat 1.65 / rand 1.75 / run 0.10` at rollout 600) |
| `bv3_warm_noaux` | warm-started from `bv3_repro/best.pt` | **1.19/3** (7 evals to rollout 175) |

`bv3_repro` reaches ~1.2 by **rollout 50** and then oscillates 0.92–1.45
for the remaining 550 rollouts with no trend; the mean over evals 50–200
(1.18) and over evals 425–600 (1.14) is the same number. Entropy falls
from 2.84 to −0.67 (σ 1.0 → 0.17) across that stretch and changes
nothing, so the ceiling is not set by exploration breadth. The warm
replicate sits at the same 1.2 with σ ≈ 0.21 throughout — same ceiling
from opposite ends of the exploration range.

**Exact configuration of the baseline** (`bv3_repro`):

```
--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0
--belief-grid-size 26 --max-steps 200 --sensor-radius 40 (default)
--red-policy-mix stationary:1,random:1,run:1
--n-envs 64 --rollout-steps 200 --mb-size 512 --n-epochs 10
--n-rollouts 600 --eval-interval 25 --best-ckpt-metric det_caught --seed 0
lr 1e-4 (linear decay), ent_coef 0.008, aux_hidden_coef 0.0,
cold start (no warm_start_critic, no warm_start_full),
§4b live-track sensor model ON (occlusion, p_TP draw, conf and accuracy
falling with range), actor_obs = belief.
```

**Why `belief_v3`'s 3.00/3 is not a recoverable target.** Three things
changed since it was recorded (2026-07-21), and two of them are
deliberate realism improvements nobody wants to undo:

1. **The evader stopped cornering itself.** `f9bc512` (2026-07-31) added
   wall repulsion to `run_from_nearest_uav`: before it, a fleeing red ran
   into the arena wall, had its perpendicular velocity clipped and slid
   along the boundary, pinning itself — and the commit message records
   that blue "learned a degenerate wall-trapping counter", with red time
   within 8 m of a wall measured at 0.63 before and 0.42 after. That is
   exactly the `run` third of the eval, which sits at ~0.18 in every run
   above while `stat` and `rand` sit near 1.7.
2. **Live tracks stopped being near-oracle in range** (§4b).
3. The arena grew 130 → 200 — measured as the *smallest* of the three:
   returning to 130 bought only +0.15/3 (1.02 → 1.17).

**So the number the tracker observation has to beat is 1.2/3 under these
conditions**, not 3.00/3. Two caveats to state whenever that comparison
is reported: with `n_obstacles 0` only the RED half of the tracker path
is exercised (the obstacle tracker is inert), and the tracker path's K
slots and σ cut-off were calibrated at L=200 with 7 blues / 4 reds /
9 obstacles, so their adequacy at this geometry is an assumption rather
than a measurement.

**`aux_hidden_coef 0.2` is harmful here, against what the config assumed.**
`stage4_default.py` says aux 0.2 is safe once the critic is "MEANINGFUL"
— converged, same architecture, same distribution. `bv3_warm` met all
three conditions (warm-started full from `bv3_repro/best.pt`, 61 tensors
copied, identical task) and degraded **monotonically from 1.29 to 1.01
over 95 rollouts** while the `aux` term itself fell from 8.05 to 1.44.
At rollout 5 the aux term contributes ~1.6 to the loss against a policy
loss of ~0.02, and `kl` overshoots `target_kl` in the first epoch every
rollout (`eps=1`). The actor learns to imitate the critic's hidden state
and pays for it in captures. Removing it (`bv3_warm_noaux`) restores the
1.2 plateau.

#### 6.6 Two traps worth not repeating

- **`best_ckpt_metric`.** The default `mean_return` mis-selects on any
  crash-penalty run (§3 already records this); these runs used
  `det_caught` / `det_composite`. Consider changing the default.
- **Never change a RunPod pod's `args` while it is running.** It recreates
  the container immediately: it killed `bv3_repro`'s first attempt at
  rollout 5 and left the pod billing overnight with nothing running. Set
  the command before starting, guard each run with a `.done_` marker on
  the persistent volume so a container recreate skips finished work, and
  watch for shutdown via log inactivity read from the API rather than a
  marker line in the container log (a recreate wipes that log).

---

### 7. Belief map vs tracker observation — first head-to-head (MEASURED, 2026-09-22) · `feature/target-tracking`

Both arms identical except `--actor-obs`: arena 130, 5 blue / 3 red, no
obstacles, `max_steps 200`, `n_envs 64 / rollout_steps 200 / mb 512 /
n_epochs 10`, cold, seed 0, 600 rollouts, eval every 25. The tracker arm
runs **constant velocity** (no `--red-motion-ckpt`).

#### 7.1 Headline: a tie

Mean of the **last eight** deterministic evals (rollouts 425–600), which
is the robust comparison — single points swing ±0.25 in both arms:

| | belief map | tracker (CV) |
|---|---|---|
| **mean / 3** | **1.144** | **1.155** |
| `stat` | 1.61 | 1.56 |
| `rand` | 1.63 | 1.67 |
| `run` | 0.19 | 0.24 |

A 0.011 difference against within-arm swings of ±0.15 is a tie. The
`run` column favours the tracker but not meaningfully: its eight values
range 0.05–0.45 and the belief's 0.00–0.25, so 0.24 vs 0.19 is noise.
(The final eval alone reads `run` 0.40 vs 0.10 — a 4x gap that does
**not** survive averaging. Do not quote it.)

#### 7.2 The real finding: the two learning curves have different shapes

Splitting both 24-eval series into thirds:

| rollouts | belief map | tracker (CV) |
|---|---|---|
| 25–200 | **1.146** | 0.869 |
| 225–400 | **1.221** | 0.986 |
| 425–600 | 1.144 | **1.155** |

The belief map is **at its ceiling by rollout 25** and flat for the
remaining 575 — no trend across 24 evals, with a `lr` that traverses the
whole 1e-4 → 1e-5 range along the way. The tracker climbs monotonically
across all three thirds and is **still climbing at 600**: its last four
are 1.15, 1.20, 1.13, 1.28, with `kl` down at 0.011–0.014 against a
`target_kl` of 0.03 and `eps=10`, i.e. no longer limited by the trust
region but by the decayed `lr`. It did not converge; it ran out of
budget.

So the tracker observation **matches the belief map while giving up the
five privileged-information leaks** listed in
`docs/tracker_observation.md` (true red count, memory slots, slot
identity, true obstacle radius, obstacle count), with no coverage
information at all, and without having converged.

#### 7.3 The tracker itself is healthy — the deficit was never perception

Throughout training: `nees` 3.8–5.5 around its 4.0 target (so the
uncertainty the actor receives is honest, not inflated), `nis` 1.1–1.3
(mildly conservative innovation covariance), and `trk` 2.2–5.0 m against
the belief map's 25–50 m. The shown tracks are metre-accurate and
well-calibrated even as the policy changes the geometry the filter sees.
Whatever limits this task, it is not track quality.

Note `trk` is **not comparable across arms**: belief measures the
distance from each extracted peak to the nearest red, tracker measures
only confirmed tracks under the σ cut-off.

#### 7.4 Caveats to state whenever this is reported

* **Only the red half of the tracker path is exercised.** With
  `n_obstacles 0` the obstacle tracker is inert, so this says nothing
  about the obstacle-estimation half.
* **The tracker path has no coverage information.** The belief map does
  double duty — with `enemy_belief_decay 0.9935` and
  `enemy_belief_diffusion 0.2`, unobserved cells drift back toward the
  prior, which *is* a where-have-I-looked signal. The tracker arm gets K
  discrete confirmed tracks and nothing else. Region / staleness nodes
  (docs/search_design.md) would close that asymmetry. **This run is not
  evidence that they would help**: the two arms tie, so nothing here
  attributes the shared ceiling to missing coverage. An earlier draft of
  this section claimed it did, on a reading of rollouts ≤200 where the
  tracker looked behind on `stat`/`rand` — it had simply not converged
  yet.
* **K slots and the σ cut-off were calibrated elsewhere** (L=200, 7 blue /
  4 red / 9 obstacles). At this geometry 8 slots comfortably exceed
  2 × n_red and the 40 m cut-off is still the sensor radius, but that is
  an assumption, not a measurement.
* **The two arms carry different false-alarm models**, not none vs some:
  the tracker path has `clutter_rate 0.2` false plots, the belief map has
  per-cell `p_FP 0.15`. Neither is privileged; they are not identical.

#### 7.5 Next

The tracker at **1000 rollouts**, same everything else. The belief arm is
deliberately **not** re-run at 1000: it reached ~1.15 at rollout 25 with
`lr` barely decayed and never exceeded its 1.221 third-mean across the
full `lr` range, so extra budget is measured — not assumed — to do
nothing for it. If the tracker at 1000 lands clearly above ~1.3, a
matching 1000-rollout belief run becomes worth its 1.6 h to close the
compute-parity objection; if it ties again, it is not.

Note that `--n-rollouts 1000` is **not** "600 and then more": the `lr`
decays linearly over the budget, so a 1000-rollout run holds a higher
`lr` for longer and is a different recipe.

---

### 8. The observation is not the bottleneck — the policy is (MEASURED, 2026-09-22)

#### 8.1 Why the old bar was not a bar

`GreedyPursuer` reads `env.state_snapshot()`: **true** red positions,
gated only by range. It never misses a detection, never sees a false one,
and its positions carry no error — precisely the near-oracle in-range
regime §4b removed from the policy's inputs. Reading "greedy 2.63 vs
trained 1.27" as *a trivial baseline beats us* is therefore wrong: the
comparison mixes policy quality with an information gap we created on
purpose.

#### 8.2 `ObservationGreedyPursuer`, the same-information baseline

Same rule, run on what the **actor** is shown.

**Inputs.** Exactly the arrays the policy receives for its enemy graph —
`rb_edge_features` (K track slots × N blues, 7-D) and `rb_edge_visible`,
from `structured_belief_observation()`. In belief mode those come from
`_build_enemy_tracks` (live tracks + belief-map memory peaks); in tracker
mode from the tracker's confirmed tracks. Same producer as the policy's,
whichever arm is running.

It uses a **strict subset** of them: `rel_pos` (indices 0:2) and `range`
(index 4), plus the mask. It ignores relative velocity (2:4), bearing
(5:7), the `red_features` nodes (confidence + velocity covariance), its
teammates (`blue_features`, `bb_*`), obstacles, and has no memory of its
own — only what the observation already carries. The policy gets all of
that **plus** a GRU.

**Rule.** Among this blue's slots with a non-zero mask, keep the most
confident ones and steer at the nearest of those; hold position if there
are none. The mask *is* the confidence (1.0 live, the belief peak's value
for memory, 0 padding), so a live target always outranks a remembered one
and memory is used exactly when nothing is live. Ranking by distance
alone chased ghosts — a test caught it steering at a 2 m belief artefact
over a real red 30 m away.

#### 8.3 Results (50 episodes per cell, matched seeds)

Trained row = the tracker-CV arm's checkpoint.

| Blue | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| Random | 0.94 (198 steps) | 1.20 (199) | 0.12 (200) | **0.75** |
| Trained | 1.72 (185) | 1.84 (186) | **0.26** (200) | **1.27** |
| Greedy (true-in-range) | 2.60 (94) | 2.80 (76) | 2.50 (121) | **2.63** |
| **ObsGreedy** (same obs) | **2.86 (66)** | **2.92 (63)** | **2.62 (114)** | **2.80** |

#### 8.4 What this establishes

* **Perception memory beats exact-in-range truth.** `ObsGreedy` beats
  `GreedyPursuer` on *every* column and finishes episodes ~30% faster.
  The mechanism is in `GreedyPursuer`'s own code: with nothing in sensor
  range it returns a zero action and sits still, while `ObsGreedy` still
  gets a heading from a belief-map peak or a coasting track. This is a
  result **for** the perception stack — the memory it carries is worth
  more than the position error it adds.
* **The observation is not the bottleneck.** 2.80/3 is extractable from
  it by a rule with no learning, no velocity, no teammates and no
  recurrence. Neither the belief map nor the tracker is what limits the
  trained policy.
* **The policy is.** It sits at 1.27/3 — **25% of the way from random
  (0.75) to `ObsGreedy` (2.80)** — on an input from which a five-line
  rule extracts 2.80. Against the evader it is barely above chance:
  **0.26 against random's 0.12**, where `ObsGreedy` gets 2.62. Episode
  length says the same thing: `ObsGreedy` closes passive reds in 63–66
  steps, the policy burns 185–200 and times out.
* **So §7's belief-vs-tracker comparison is premature.** It measures
  which of two policies, both far below what their own inputs allow, is
  marginally less bad. The tie there stands as a fact; it should not be
  reported as a finding about the two representations until a policy
  exists that approaches what either observation supports.

#### 8.5 What it unlocks

`ObsGreedy` is an **expert that consumes the policy's own observation
space**, which is exactly the precondition for behaviour cloning: its
actions can be generated for any state the policy visits, with no
privileged information to strip out. Pre-training the actor to imitate it
and then fine-tuning with PPO is the standard treatment for this
symptom — an agent that fails to learn a behaviour expressible in a few
lines — and it was not available before this baseline existed.

It also replaces `GreedyPursuer` as the honest acceptance bar. The
repo-wide criterion of "1.2 × GreedyPursuer" was calibrated against an
oracle-in-range opponent and is the wrong target under §4b perception.
