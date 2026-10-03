# Stage 4 — Results

Stage 4 adds **partial, noisy perception** on top of the Stage 3
pursuit task: a Bayesian belief map (log-odds occupancy grid), static
**obstacles** with line-of-sight **occlusion**, and **sensor noise**.
The blues no longer receive ground-truth enemy/obstacle positions —
they must act on a fused, uncertain picture of the world, which is the
first real step toward what a fielded ISR UAV actually sees.

> **Read §6.5 onward before quoting anything above it.** This file is
> kept as a running record, oldest first, and its early headline —
> *the belief-driven policy matches the fully-observable oracle at 3/3*
> — **no longer holds**. It was recorded in 2026-07 at arena 130 with a
> near-oracle in-range sensor and an evader that cornered itself. Two
> deliberate realism changes and one geometry change since then put the
> honest figure at **1.2–1.28/3** (§6.5, §8.3), and §6.5 explains why
> 3.00/3 is not a recoverable target.
>
> The current state of play, in three lines:
> * a trivial rule extracts **2.90/3** from the same observation the
>   policy gets, so the observation is not the bottleneck — the policy
>   is (§8, §10);
> * the belief map and the tracker are **equivalent on captures** once
>   both have a search signal, and the tracker is 14–31% faster while
>   giving up five privileged information leaks (§10);
> * behaviour cloning from that rule **reaches it** — 2.940/3 in 768k
>   steps against PPO-from-scratch's 1.280 in 12.8M, and the repo-wide
>   acceptance criterion passes for the first time under §4b
>   perception (§11).
>
> * PPO fine-tuning from that clone reaches **2.967/3** and converts 9 of
>   the expert's 12 failures on the evader into wins (§12.1);
> * but **§6–§12 were all measured where coordination is worth nothing** —
>   dividing the targets between blues buys −0.03 at red 1.0 and only
>   starts paying at red ≈ 1.4 (§12.2);
> * moved to red 1.4, where it does pay, the policy beats explicit target
>   assignment by **4.7 standard errors** and is 28% faster (§13).
>
> Two things to carry into anything that follows. The repo-wide
> "1.2 × Greedy" acceptance bar is **degenerate above red 1.2** — it falls
> as the task hardens (§13.2). And §13's level is depressed by a tracker
> left un-recalibrated for the faster red (`nees` 8–9 against a target of
> 4), so it is a floor, not a ceiling.

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
* **~~The belief map does coverage double duty.~~ NEITHER path had
  coverage information — corrected in §9.3.** This bullet used to argue
  that `enemy_belief_decay 0.9935` plus `enemy_belief_diffusion 0.2`
  made unobserved cells drift back toward the prior, so the belief map
  carried a where-have-I-looked signal the tracker lacked. The drift is
  real *in the grid*; what the actor gets is K peaks, and
  `_extract_belief_peaks` is a global argmax, so it lands on whatever
  false alarm is freshest **inside the current sensor footprint**.
  Measured: the cell it steers to has staleness 0 in 579 of 579 cases
  against a map mean of 36.5 steps (§9.3). So the asymmetry this bullet
  described did not exist — both arms were blind, one of them noisily.
  §10 gives both a real one via region / staleness nodes
  (`docs/search_design.md`), and the belief/tracker gap closes to 0.013.
* **This run is not evidence that region nodes would help.** The two
  arms tie, so nothing here attributes the shared ceiling to missing
  coverage. An earlier draft claimed it did, on a reading of rollouts
  ≤200 where the tracker looked behind on `stat`/`rand` — it had simply
  not converged yet. §10 measures the answer at the expert level
  instead; whether it moves a *trained policy* is still unmeasured.
* **K slots and the σ cut-off were calibrated elsewhere** (L=200, 7 blue /
  4 red / 9 obstacles). At this geometry 8 slots comfortably exceed
  2 × n_red and the 40 m cut-off is still the sensor radius, but that is
  an assumption, not a measurement.
* **The two arms carry different false-alarm models**, not none vs some:
  the tracker path has `clutter_rate 0.2` false plots, the belief map has
  per-cell `p_FP 0.15`. Neither is privileged; they are not identical.

#### 7.5 The tracker at 1000 rollouts: it plateaus (MEASURED, 2026-09-23)

The plan here was: run the tracker to 1000, and *if it lands clearly
above ~1.3, a matching 1000-rollout belief run becomes worth its 1.6 h to
close the compute-parity objection; if it ties again, it is not.* The
belief arm was deliberately not re-run, since it reached ~1.15 by rollout
25 and never exceeded its 1.221 third-mean across the full `lr` range.

`bv3_tracker_cv_1k`, 40 deterministic evals:

| rollouts | mean /3 | `stat` | `rand` | `run` |
|---|---|---|---|---|
| 25–325 | 1.058 | 1.41 | 1.51 | 0.26 |
| 350–650 | 1.172 | 1.60 | 1.68 | 0.23 |
| 675–1000 | 1.226 | 1.67 | 1.73 | 0.28 |
| **last 8 (825–1000)** | **1.223** | 1.69 | 1.72 | 0.26 |

**1.223, not "clearly above 1.3". So the matching belief run is not
worth running** — the condition was written in advance and it was not
met. Best single eval was 1.33 at rollout 700 and did not hold.

Two corrections this forces on §7.2:

* **"Still climbing at 600" was noise.** The thirds gain +0.114 then
  +0.054, and the last eight equal the final third to 0.003. It is
  flat by ~700. The run's own `best.pt` (by `det_caught`) is from
  **rollout 325**, not 1000.
* **The `run` column is flat across all three thirds** (0.26 / 0.23 /
  0.28). 975 rollouts bought *nothing* against the evader — which is
  §8's finding arriving from a second direction.

Note that `--n-rollouts 1000` is **not** "600 and then more": the `lr`
decays linearly over the budget, so a 1000-rollout run holds a higher
`lr` for longer and is a different recipe.

---

### 8. The observation is not the bottleneck — the policy is (MEASURED, 2026-09-22)

#### 8.1 Why the old bar was not a bar

`GreedyPursuer` reads `env.state_snapshot()`: **true** red positions,
gated only by range. It never misses a detection, never sees a false one,
and its positions carry no error — precisely the near-oracle in-range
regime §4b removed from the policy's inputs. Reading "greedy 2.65 vs
trained 1.28" as *a trivial baseline beats us* is therefore wrong: the
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

Both trained rows are `bv3_tracker_cv_1k` (§7.5), so the whole table runs
in **tracker** mode: `ObsGreedy` here reads the tracker's confirmed
tracks, not the belief map. The belief-mode counterpart is §9.

Re-measured 2026-09-23 with the seeding fix of §9.1 and the checkpoints
pulled off the pod, which is also what settled *which* checkpoint — the
run's `best.pt` is from **rollout 325**, a third of the way in, not the
1000-rollout endpoint. Both are shown, since the gap between them is
itself the point.

| Blue | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| Random | 0.94 (198 steps) | 1.08 (195) | 0.12 (200) | **0.713** |
| Trained — `best.pt` @ **325** | 1.76 (187) | 1.86 (183) | 0.22 (200) | **1.280** |
| Trained — `final.pt` @ 1000 | 1.70 (190) | 1.78 (182) | 0.30 (200) | **1.260** |
| Greedy (true-in-range) | 2.60 (94) | 2.84 (75) | 2.50 (121) | **2.647** |
| **ObsGreedy** (same obs) | **2.86 (66)** | **2.86 (69)** | **2.62 (114)** | **2.780** |

675 extra rollouts are worth **−0.02**, which is §7.5's plateau seen from
the evaluation side.

**On the superseded row.** The version first published here read
`1.72 / 1.84 / 0.26 = 1.27` and reproduces *neither* checkpoint — it sits
between them in all three columns. `Stationary` and `RunFromNearest` face
deterministic reds, so §9.1's seeding bug cannot account for it; it must
have come from a third checkpoint. Its provenance is not recoverable:
the pod's own `eval_results.json` was overwritten by the re-measurement
(the script always writes that filename next to the checkpoint), so the
row is replaced rather than explained. The copy on the pod's volume
survives if it is ever worth recovering.

#### 8.4 What this establishes

* **Perception memory beats exact-in-range truth.** `ObsGreedy` beats
  `GreedyPursuer` on every column and finishes episodes ~30% faster. The
  mechanism is in `GreedyPursuer`'s own code: with nothing in sensor
  range it returns a zero action and sits still, while `ObsGreedy` still
  gets a heading from a belief-map peak or a coasting track. This is a
  result **for** the perception stack — the memory it carries is worth
  more than the position error it adds. State the margin honestly,
  though: it is +0.26 on `Stationary` and +0.12 on `RunFromNearest` but
  only **+0.02 on the random red**, which is a tie on that column.
* **The observation is not the bottleneck.** 2.78/3 is extractable from
  it by a rule with no learning, no velocity, no teammates and no
  recurrence — **2.90/3 once §10 gives that rule a search tier**.
  Neither the belief map nor the tracker is what limits the trained
  policy.
* **The policy is.** Its best checkpoint sits at 1.280/3 — **27% of the
  way from random (0.713) to `ObsGreedy` (2.780), and 24% of the way to
  §10's 2.900** — on an input from which a five-line rule extracts that
  much. Against the evader it is barely above chance: **0.22–0.30
  against random's 0.12**, where `ObsGreedy` gets 2.62. Episode length
  says the same thing: `ObsGreedy` closes passive reds in 66–69 steps
  (47–50 with the search tier), the policy burns 182–200 and times out.
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

**Which `ObsGreedy`, settled in §10.** The expert to clone is the one on
the **tracker** observation: it is the one a fielded system could run,
the belief path carrying five privileged leaks
(`docs/tracker_observation.md`) that a cloned policy would come to depend
on. Since §10 gave both arms a search tier they match on captures anyway,
and the tracker is 14–31% faster. **The bar is its 2.900/3** — not the
belief arm's 2.913, which is reached with privileged information, and not
the 2.780 in §8.3, which was reached while idle half the time.

Agreed in advance, so the result is not read after the fact: **≥ 2.0/3
means cloning transfers** and PPO fine-tuning earns its GPU time;
**< 1.5/3 is the more interesting outcome** — it would say a policy with
a GRU and the full typed graph cannot represent a five-line rule over its
own observation, which is a finding about the architecture, not the
algorithm.

---

### 9. Belief map vs tracker, with the policy removed (MEASURED, 2026-09-23) · `feature/target-tracking`

§7 compared the two observations through two trained policies; §8 then
showed both policies sit at ~27% of what their own input supports. A
comparison of representations read off two policies that far below their
ceiling measures the policies, not the representations.

`ObsGreedy` consumes the actor's enemy graph and nothing else, so running
it under both `--actor-obs` values asks the question directly: **how much
pursuit is extractable from each observation, with no learning
involved?** Same env, same seeds, 50 episodes per cell:

```
python scripts/compare_observation_quality.py --n-episodes 50
```

#### 9.1 The control, and a metric bug it exposed

`Random` and `Greedy` read no actor observation — `Greedy` reads
`state_snapshot()`, `Random` reads nothing — so their rows must be
identical in both modes. They are, **to every digit** (mean of the three
reds: 0.713 and 2.647 in both). Any ObsGreedy difference is therefore the
observation and not the env.

Getting that clean took a fix. `evaluate_trained.eval_matrix` used to take
red *policies*, not factories, so `random_red(seed=0)`'s closure held one
RNG stream shared across every episode **and every blue row**: its draws
depended on how many steps the preceding rows consumed, and episode length
is exactly what this table measures. It showed up as the only column
failing the control — `Stationary` and `RunFromNearest` reproduced
digit-for-digit across two independent sweeps while `Random` did not.

`eval_matrix` now takes `(seed) -> policy` and rebuilds the red per
episode, so every cell of every table faces identical opposition.
`tests/test_eval_matrix_seeding.py` pins the property that states it
without jargon: **the matrix must not depend on the order of its blue
rows**. Under the old code it did — swapping the `Greedy` and `Random`
rows moved Greedy's own `Random`-column score from 1.50 to 1.25, a 0.25
swing bought with nothing but row order.

Tables produced before the fix were re-measured rather than annotated:
§8.3 in full (which is also where the checkpoint ambiguity got settled),
and the §9.2 table below was already clean.

#### 9.2 Results — the belief map wins by 0.13/3 (SUPERSEDED by §10)

**The 0.13 is not a property of the two representations.** §10 wires the
coverage path both observations were missing, gives `ObsGreedy` the
matching search tier, and the gap collapses to **0.013** — the belief
map's whole advantage here was the tracker arm freezing for want of a
search rule. The table stands as a correct measurement *of the
configuration without region nodes*; the heading's conclusion does not.

Mean reds caught of 3, 50 matched-seed episodes per cell:

| Blue | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| Random *(control)* | 0.94 | 1.08 | 0.12 | **0.713** in both modes |
| Greedy *(control)* | 2.60 | 2.84 | 2.50 | **2.647** in both modes |
| ObsGreedy — **belief** | **3.00** (68.6 steps) | **3.00** (64.6) | **2.74** (125.2) | **2.913** |
| ObsGreedy — **tracker** | 2.86 (65.5) | 2.86 (68.5) | 2.62 (114.2) | **2.780** |

The belief map wins every column by 0.12–0.14. Consistent in sign and
size across three different reds at n=50 matched seeds, so unlike §7's
0.011 this is a difference rather than noise.

#### 9.3 The mechanism: the tracker is sharper but goes blind

The means hide two opposite effects. Taking the `Stationary` column
per-episode:

| | clean sweeps (3/3) | steps on those | failures |
|---|---|---|---|
| belief | **50 / 50** | 68.6 | — |
| tracker | 43 / 50 | **43.6** | 7 episodes, all 2/3 at the 200-step cap |

The tracker is **36% faster when it works** — its confirmed tracks are
metre-accurate (§7.3: `trk` 2.2–5.0 m against the belief map's 25–50 m),
so pursuit is direct instead of drifting toward a smeared peak. It then
loses 7 episodes outright, never acquiring the third red at all.

Why: counting the agent-steps where `ObsGreedy` returns the zero action
because no slot is visible,

| | idle agent-steps, all episodes | idle in the timed-out episodes |
|---|---|---|
| belief | **0.0%** | — |
| tracker | **49.9%** | **85.7%** |

In belief mode a blue is *never* without something to chase. In tracker
mode a blue with no confirmed track has nothing, holds position, and
stops contributing — half the time overall, and almost always in the
episodes that fail.

**What the belief map falls back to is not what it looks like**
(MEASURED 2026-09-24, correcting the first version of this section,
which said the fallback made idle blues "wander and incidentally sweep
the arena"). It does not sweep. `_extract_belief_peaks` is a global
argmax of `sigmoid(log_odds)` with no threshold, and the log-odds
arithmetic decides where that lands:

| cell state | log-odds | posterior | measured, at step 60 |
|---|---|---|---|
| observed, false-alarm evidence | `L > 0` | `p > 0.5` | max **0.943** |
| never observed | `L = 0` | `p = 0.5` | max **0.499**, median 0.354 |
| observed, empty | `L < 0` | `p ≈ 0` | median **0.000** |

An unobserved cell starts at the prior and can only fall — decay pulls
it toward 0 and diffusion mixes it with searched neighbours — so it can
never outrank a cell that just took a `p_FP = 0.15` false alarm. **The
argmax is therefore always inside the current sensor footprint.**

Directly: switching on the env's `_staleness` field (steps since each
cell was last observed — implemented, never shown to any policy, §9.5)
and reading off the cell `ObsGreedy` steers to when nothing is live:

| | staleness of the cell it steers to | staleness of the map |
|---|---|---|
| 579 belief-driven headings, 20 eps | **0.0 steps, every single one** | 36.5 steps |

It steers at a clutter echo under its own radar, not at unexplored
ground. The coverage information *is* in the grid — "never looked" and
"looked, empty" are genuinely different values — and **the peak
extraction destroys it** before the actor sees anything.

#### 9.4 What this establishes

*(§10 acted on every item here and measured the result; read the two
together.)*

* **The belief map's advantage is motion, not coverage and not
  accuracy.** On accuracy the tracker wins outright (43.6 vs 68.6
  steps). What it loses on is never having a reason to move: it freezes
  half the time. The belief map always hands back *a* heading, so its
  blues keep moving, and a UAV in motion eventually blunders into a red
  while a stationary one only finds what walks into its radar. That is
  the whole 0.13/3. It is a duller mechanism than "the belief map knows
  where to search", which is what the first version of §9.4 claimed.
  **§10 confirms it by removing the cause**: give the tracker a search
  rule and the 0.13 becomes 0.013.
* **Neither observation carries a search signal — retracting the claim
  that this one does.** §7.4 said the §7 tie was not evidence for region
  nodes; the first version of this section said removing the policy
  *made* it evidence, on the reading that the belief arm had coverage
  and the tracker did not. §9.3's staleness measurement kills that: the
  belief fallback points at staleness-0 cells, i.e. ground being swept
  right now. So the position is **neither arm has a where-should-I-look
  signal** — one is blind and the other chases its own clutter. That is
  a stronger reason to wire region / staleness nodes
  (`docs/search_design.md`) than the one withdrawn, because it applies
  to both arms rather than closing a gap between them. **True as
  written on 2026-09-23 and no longer true of the code**: §10 wires
  them, and both arms now have one.
* **The §7 tie now has an explanation.** Two observations that differ by
  0.13/3 for a greedy rule produced policies differing by 0.011 — further
  evidence that at 1.15/3 neither policy is limited by its input.
* **`ObsGreedy` is not yet a same-information expert for a searching
  policy.** Its rule bottoms out at "hold position", which is only the
  honest action while the observation offers nothing better. The moment
  region nodes enter the policy's observation, an expert that ignores
  them stops being the same-information baseline this section rests on
  and becomes a handicapped one. The fallback has to become "go to the
  stalest searchable region" in the same commit that wires them.
  **Done in §10.1**, in that commit.
* **Scope.** `n_obstacles 0`, so only the red half of the tracker path is
  exercised; and `ObsGreedy` ignores velocity, bearing and teammates, so
  this measures what the *enemy graph* supports for a memoryless
  single-target rule, not what a coordinating policy could extract.

#### 9.5 Staleness was implemented at both ends and wired at neither (DONE in §10)

*This section is the state on 2026-09-23 and the work order §10 carried
out. Kept because "add region nodes" sounded like a feature and was in
fact three edits plus a leak — worth knowing next time something looks
absent and is merely disconnected.*

| piece | where | state then | state now |
|---|---|---|---|
| `_staleness` field, steps since each cell was observed | `pursuit_env.py`, flag `use_staleness` | implemented | reachable via `--use-staleness` |
| `R×R` region nodes `[staleness, searchable]` + weighted region→blue edges | `pursuit_env.py` | implemented | emitted by `structured_belief_observation()` |
| coverage path in the encoder (`n_region`, `region_feat_dim=2`, gb messages) | `gnn_stage4_policy.py` | implemented | fed, actor-side, via `n_region` |
| tests | `test_staleness.py`, `test_region_nodes.py`, `test_region_graph.py` | 50 passing | 51 — three rewritten, one added (see the leak) |

**One leak to close before wiring it — closed in §10.1.**
`_build_region_nodes`'s `searchable` feature was computed from
`_obstacle_grid`, which is built from the **true** obstacle positions and
radii — the ground-truth occupancy the CTDE critic is allowed to see. At
`n_obstacles 0` it is 1.0 everywhere and harmless, which is why the path
could be wired before fixing it; the moment obstacles return it would
hand the tracker arm a true obstacle map and become the **sixth** entry
in `docs/tracker_observation.md`'s leak table, of exactly the kind the
tracker path exists to remove. It now comes from the obstacle belief
channel in belief mode and the obstacle tracker in tracker mode.

Three tests in `test_region_nodes.py` **failed on that fix, because they
were pinning the leak** — they drove `_obstacle_grid` and asserted
`searchable` followed. They now drive the estimate, and a new test moves
truth alone and asserts nothing changes.

The `staleness` feature itself is clean: it is derived from the blues'
own positions and sensor geometry, so it is self-knowledge, not
knowledge of the enemy. A real C2 layer knows where its own sensors have
swept.

Not connected: `structured_belief_observation()` emits no
`region_feats` / `gb_edge_feats` / `gb_weight`; `GNNStage4Policy` builds
both encoders without `n_region` (`:412`, `:432`); `train_stage4.py` has
no flag. The only callers of `_build_region_graph` in the repo are tests.
**No training run has ever seen a region node** — so §7.4's asymmetry is
not a gap in the design, it is an unconnected wire.

This also means the comparison above is clean: neither mode had region
nodes, and `ObsGreedy` would not read them anyway, so §9.2 is a
comparison of the **enemy graph alone**. The coverage difference it
measures is the one living *inside* that graph — `enemy_belief_decay
0.9935` giving memory peaks the tracker has no equivalent for.

---

### 10. Wiring the coverage path closes the belief/tracker gap entirely (MEASURED, 2026-09-24) · `feature/target-tracking`

§9.3 found that **neither** observation carries a where-have-we-not-looked
signal: the tracker is blind without a confirmed track, and the belief
map's fallback peak lands on cells being swept right now, so it chases
its own clutter. §9.5 recorded that the field, the region nodes and the
encoder path all existed, fully tested, connected by nothing. This wires
them.

#### 10.1 What was connected

`structured_belief_observation()` emits `region_feats` / `gb_edge_feats` /
`gb_weight` under `--use-staleness`; `split_stage4_obs` routes them to the
actor; `GNNStage4Policy` takes `n_region` (0 = off, and the policy stays
byte-identical); `train_stage4.py` gains `--use-staleness` /
`--staleness-regions`. At R=5 that is 25 region nodes and +9 024
parameters (136 965 → 145 989).

**Actor only.** The critic already reads `true_occupancy`, so "where have
we looked" is not news to it, and adding nodes there would move the CTDE
baseline for no information gain.

**`ObsGreedy` had to grow the same tier**, or it would have stopped being
a same-information baseline the moment the policy could see regions. With
nothing on the enemy graph it now heads for the region maximising
`staleness × searchable / (1 + range)` (`range` is already normalised by
`arena_size`). Distance-discounted, because a plain staleness argmax
crosses the arena for a marginally older region; and **uncoordinated**,
like its other tiers — splitting the arena between five blues is the
learned policy's job, and a baseline that coordinates is no longer the
trivial rule that makes it a credible bar.

**A leak closed on the way.** `searchable` was computed from
`_obstacle_grid` — the *true* occupancy the CTDE critic may read. Harmless
at `n_obstacles 0`, but these nodes go to the **actor**, so once obstacles
return it would have handed the tracker arm a true obstacle map: a sixth
entry in `docs/tracker_observation.md`'s leak table, of exactly the kind
the tracker path exists to remove. It now reads the obstacle belief
channel in belief mode and the obstacle tracker in tracker mode. Three
tests were pinning the leaky behaviour by driving the truth grid; they now
drive the estimate, and a new one asserts that moving truth alone changes
nothing.

#### 10.2 The gap was the paralysis, and it is gone

50 matched-seed episodes per cell, same geometry as §9:

| ObsGreedy on | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| belief | 3.00 (68.6 steps) | 3.00 (64.6) | 2.74 (125.2) | **2.913** |
| tracker | 2.98 (**47.3**) | 3.00 (**49.5**) | 2.72 (**107.1**) | **2.900** |

**−0.013, against §9.2's −0.133.** The belief map's entire advantage was
the tracker arm freezing; give both a principled search fallback and it
vanishes. On the stationary red alone the tracker expert goes from 58.4%
of agent-steps idle at 2.80/3 to **0.0% idle at 2.95/3**.

The belief row is unchanged to the digit, including step counts — the
control that says region nodes did nothing there. Correct: its enemy
graph always offers a memory peak, so the search tier never fires.

#### 10.3 What this establishes

* **On captures the two observations are now equivalent**, and the
  tracker gets there **14–31% faster** (47.3 vs 68.6 steps on
  `Stationary`, 49.5 vs 64.6 on `Random`, 107.1 vs 125.2 on `run`) —
  §7.3's metre-accurate tracks finally showing up in behaviour instead of
  being spent recovering from paralysis.
* **And it does so having given up all five privileged leaks.** Equal
  captures, better efficiency, less information: on this evidence the
  tracker is the path to build on, not the fallback.
* **§9.2's 0.133 is explained and retired.** It measured a missing
  search rule, not a difference between representations.
* **The clone target follows.** The expert to imitate is `ObsGreedy` on
  the **tracker** observation — it is the one a fielded system could
  actually run. The acceptance bar is its own **2.900**, not the belief
  arm's 2.913 (which is reached with privileged information) and not
  §9.2's 2.780 (which was reached while idle half the time).
* **Scope, unchanged.** `n_obstacles 0`, so the obstacle half of the
  tracker path is still inert, and `searchable` is still trivially 1.0
  everywhere — the leak fix above is untested against real obstacles
  because there are none to test against yet.

---

### 11. Behaviour cloning from `ObsGreedy` reaches the expert (MEASURED, 2026-09-24) · `feature/target-tracking`

§8 established that a five-line rule extracts far more from the actor's
observation than PPO ever did, and §10 made the tracker version of that
rule a competent expert with no privileged information. This clones it.

`bc_tracker_v1`: tracker observation, region nodes on, 60 rounds ×
64 envs × 200 steps = **768 000 steps**, 121 min of laptop CPU, β decayed
1.0 → 0.3, seed 0. Config in `scripts/train_bc.py`; the checkpoint carries
its own hyperparameters under `args["bc"]`.

#### 11.1 Result: the clone matches its expert, and acceptance PASSES

50 matched-seed episodes per cell, the same protocol and seeds as §10:

| Blue | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| Random | 0.94 (198.4 steps) | 1.08 (194.6) | 0.12 (200.0) | **0.713** |
| Greedy (true-in-range) | 2.60 (93.7) | 2.84 (75.0) | 2.50 (121.0) | **2.647** |
| `ObsGreedy` (the expert) | 2.98 (47.3) | 3.00 (49.5) | 2.72 (107.1) | **2.900** |
| **Trained (the clone)** | **3.00 (53.5)** | **3.00 (56.1)** | **2.82 (100.7)** | **2.940** |

**Acceptance: PASS** — `+23.13` against the `1.20 × Greedy` bar of
`+21.93`. The repo-wide Stage 1 criterion, which had never passed under
§4b perception (§8.3 recorded `-20.27` → FAIL).

**The clone matches the expert; it does not beat it.** 2.940 vs 2.900 is
+0.04, and the column carrying it is `RunFromNearest` at +0.10 — **0.8
standard errors** on n=50. Stated as a win it would be the same
over-claim this document has had to retract twice.

For scale, against the from-scratch baseline:

| | steps | mean /3 | `run` |
|---|---|---|---|
| PPO from scratch, best of 1000 rollouts (§8.3) | 12 800 000 | 1.280 | 0.22 |
| **behaviour cloning, 60 rounds** | **768 000** | **2.940** | **2.82** |

The evader column is the one to look at: it sat at ~0.25 for every
trained policy in this document, barely above random's 0.12, and is now
2.82.

#### 11.2 The learning curve, and what the β floor was for

Deterministic eval every 5 rounds (20 episodes at seed base 30 000 — the
in-training metric, not the 50-episode table above):

| round | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean /3 | 1.63 | 1.77 | 1.83 | 1.87 | 2.10 | 2.52 | 2.62 | 2.72 | 2.80 | 2.87 | 2.78 | **2.88** |

Monotone but for round 55, and crucially **no collapse as β fell to
0.3**, with the clone loss still falling (0.309 → 0.082). An earlier
belief-mode pilot that decayed β to **0** over only 12 rounds did
collapse — 1.80 at round 8 down to 1.33 by round 12, clone loss *rising*
0.19 → 0.38 — because with no expert left driving, the data comes
entirely from a student still too weak to generate it. The floor is what
prevents that, and it is the one hyperparameter here worth defending.

#### 11.3 The training log's `caught` is not the policy — read `[det eval]`

Worth stating because it looks alarming: the per-round line reads
`caught ≈ 2.9` while the deterministic eval reads 1.87. They measure
different things. The round line comes from the episodes collected for
training, in which **the expert drives with probability β**, redrawn per
step per env — so it is a mixture, dominated by the expert. At round 1
(β = 1.0) it read 2.94, which is just the expert's own score.

Decomposed at the round-20 checkpoint, stationary red, 20 matched
episodes:

| β (P the expert drives) | caught |
|---|---|
| 1.0 — expert alone | 2.85 |
| 0.73 — the mixture at the time | 2.90 |
| 0.30 — the final mixture | 2.95 |
| 0.0 — **the student alone** | **1.80** |

The student alone reproduces that round's `[det eval] stat=1.80`
exactly. Note the mixture does **not** degrade as β falls — it rises
slightly — so "watch `caught` drop as β decays" is not a progress signal.
Only `[det eval]` is.

#### 11.4 A trap in the handover to PPO, and the flag that defuses it

Cloning regresses the **mean** and deliberately leaves `log_std` alone —
the MSE has no gradient into it, which is the point, since an NLL
objective against a deterministic expert would drive σ to zero and leave
PPO no exploration. The consequence is that the clone arrives with
`log_std` still at its init **0.0, i.e. σ = 1.0**, which on a `[-1,1]²`
action box is close to uniform noise.

PPO **samples** its rollout actions. Measured on this checkpoint:

| | Stationary | Random | RunFromNearest | mean |
|---|---|---|---|---|
| deterministic | 2.65 | 2.70 | 2.20 | **2.52** |
| sampled (σ = 1.0) | 2.20 | 2.15 | **0.75** | **1.70** |

Fine-tuning from here unmodified would have PPO collecting rollouts from
a policy performing 0.8/3 below the one being evaluated, computing
advantages on trajectories that do not reflect it, with `ent_coef 0.008`
pushing to keep the width — and it would have read as *PPO destroyed the
clone*.

`train_stage4.py --reset-log-std` fixes it, applied **after** the
warm-start copy, which is the only place it works: `load_full_stage4`
copies every shape-matching tensor and `actor_log_std` matches, so a
checkpoint's own σ otherwise silently wins. §6.5 records entropy settling
near σ 0.17–0.2 on this task, so **−1.2 (σ 0.30)** is the value to start
from.

#### 11.5 Why the clone matches an expert it only imitated — unexplained

Two mechanisms were proposed, **both tested, neither held**. Recorded
because the negative results are the useful part:

* **Decorrelation — falsified.** `ObsGreedy` is uncoordinated (five blues
  chase the same nearest track), and the β mixture above *improves* as
  more student enters, which suggested the clone's imperfections spread
  the team. Measured, the opposite is true: the clone is **more**
  clustered (52.0 m mean pairwise blue distance against the expert's
  56.5 on stationary, 41.4 vs 45.7 on the evader) and has **more**
  redundant pairs (17.4% vs 13.6%, 28.8% vs 24.9%). It matches the
  expert while coordinating *worse*.
* **Interception — not supported.** The clone sees relative velocity,
  bearing, its teammates and has a GRU, all of which `ObsGreedy` ignores,
  so leading a fleeing target was the obvious candidate. The expert's
  lead angle measures exactly 0.00° (pure pursuit by construction — a
  check that the measurement works) and the clone's is −7.4° against
  stationary reds and −6.6° against the evader. **Nearly the same offset
  whether or not the target moves**, which argues against a
  target-motion explanation; and the sign convention does not cleanly
  separate "leading the target" from "following its own momentum" when
  the target is still.

So the clone deviates systematically from pure pursuit, it is neither
interception nor decorrelation, and what it *is* remains unmeasured. The
GRU's memory is the untested candidate.

#### 11.6 Caveats

* **One seed, one run.** n=1 on the BC run itself; the table is n=50
  episodes within it.
* **Deterministic only.** Every headline number here is the distribution
  mean. §11.4 is what the same policy does when sampled.
* **It inherits the expert's ceiling and its uncoordination**, measurably
  (§11.5): 2.90 is a ceiling reached, not a floor to build from. The
  headroom above it is coordination, which neither the expert nor the
  clone has, and which is what PPO fine-tuning would have to add.
* **`n_obstacles 0`**, so the obstacle half of the tracker path is still
  inert, as everywhere above.

---

### 12. PPO past the clone, and the speed at which coordination starts to pay (MEASURED, 2026-09-24/25) · `feature/target-tracking`

Two experiments that only make sense together. The first fine-tunes PPO
from §11's clone. The second asks whether this task rewards coordination
at all — because if it does not, the first has nowhere to go for reasons
that have nothing to do with the algorithm.

#### 12.1 Fine-tuning holds, and wins on the evader

`ppo_from_bc_v1`: warm-started full from `bc_tracker_v1/best.pt`,
`--reset-log-std -1.2`, `lr 3e-5`, `aux_hidden_coef 0`, 60 rollouts,
154 min CPU. Same env as §11.

50 matched-seed episodes, comparable with §10 and §11:

| policy | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| `ObsGreedy` (the expert) | 2.98 (47.3) | 3.00 (49.5) | 2.72 (107.1) | **2.900** |
| Clone (BC) | 3.00 (53.5) | 3.00 (56.1) | 2.82 (100.7) | **2.940** |
| PPO `best.pt` @ rollout 5 | 2.94 (59.3) | 3.00 (54.1) | 2.76 (106.9) | **2.900** |
| **PPO `final.pt` @ rollout 60** | 2.98 (53.2) | 3.00 (51.7) | **2.92 (94.0)** | **2.967** |

The fine-tune **does not destroy the clone** — the outcome §6.5's
`bv3_warm` made worth fearing — and it wins on the evader. Per-episode
on that column, 50 matched seeds:

| | caught | steps *on clean 3/3* | clean sweeps |
|---|---|---|---|
| `ObsGreedy` | 2.72 | 77.7 | 38 / 50 |
| Clone | 2.82 | 81.7 | 42 / 50 |
| **PPO final** | **2.92** | 87.3 | **47 / 50** |

**38 → 47 clean sweeps is 2.6 standard errors** on matched seeds: PPO
converts 9 of the expert's 12 failures into wins. Note it is *slower* on
the episodes that were already won (87.3 vs 77.7), so "fewer mean steps"
in the table above is entirely timeout conversion, not speed — the two
are mechanically coupled and only the conditional number separates them.

Two things the run also settles:

* **`--reset-log-std` earns its keep.** 154 minutes of PPO sampling with
  σ 0.30 and not one degraded eval. At the σ 1.0 the clone ships with,
  the behaviour policy scores 1.70 (§11.4) and this run would have begun
  from there.
* **`best_ckpt_metric` mis-selected again**, in a new way. §6.6 recorded
  it picking badly on crash runs; here the metric was the right one
  (`det_caught`) and it still chose **rollout 5**, whose 20-episode eval
  read 2.97 by luck, over 55 later rollouts. `best.pt` (2.900) is worse
  than `final.pt` (2.967). On a nearly flat curve, 20-episode evals
  select noise.

#### 12.2 Coordination is worth nothing below red 1.2, and pays from 1.4

`docs/design.md §3.6` set blue 1.5 against red 1.0 so "coordination among
blue agents has to provide the extra edge". That did not come true, and
§10 already implied it: `ObsGreedy` has no coordination at all and scores
2.90/3. The arithmetic says why — a stern chase closes 0.5/step, covering
100 units of a 130 m arena in an episode, so nobody ever has to cut
anybody off.

`AssignmentGreedyPursuer` measures the gap instead of inferring it: it is
`ObsGreedy` differing in **exactly one respect**, the team dividing the
targets (balanced capacity, greedy matching on the same observation). It
reads nothing extra — every (blue, slot) range is already in the enemy
graph, since track positions are the shared command-layer fusion. What it
beats `ObsGreedy` by is what target assignment alone is worth.

Red = `run_from_nearest_uav`, 30 matched-seed episodes per cell:

| red v_max | closing | `ObsGreedy` | `AssignGreedy` | **coordination** |
|---|---|---|---|---|
| 1.00 | 0.50 | 2.73 (108.9) | 2.70 (109.9) | −0.03 ± 0.13 |
| 1.10 | 0.40 | 2.77 (111.9) | 2.70 (113.0) | −0.07 ± 0.11 |
| 1.20 | 0.30 | 2.67 (127.8) | 2.63 (124.1) | −0.03 ± 0.14 |
| 1.30 | 0.20 | 2.50 (152.2) | 2.60 (135.8) | +0.10 ± 0.14 |
| **1.40** | 0.10 | 2.07 (183.0) | **2.43 (166.7)** | **+0.37 ± 0.17** |
| 1.50 | 0.00 | 0.80 (190.8) | 1.20 (184.2) | +0.40 ± 0.25 |

Three measurements in a row at **zero** (−0.03, −0.07, −0.03) and then
+0.37 beyond two standard errors. **Every result in §6–§12 was measured
in a regime where dividing the targets buys nothing**, which is worth
knowing before reading any of them as a statement about coordination.

Deliberately *not* reported as a gap to 3.00: that would assume 3.00 is
reachable at every speed, which is unknown and at red 1.5 almost
certainly false. Two rules identical but for the assignment is a measured
bound; a ceiling nobody has demonstrated is not.

#### 12.3 Whatever PPO learned, it is not target assignment

These two experiments contradict the obvious story, and the contradiction
is the useful part.

At red 1.0, `AssignGreedy` beats `ObsGreedy` by **−0.03** — target
division is worthless there. Yet at that same speed PPO reached **2.92**
on the evader against the expert's 2.72 and the assignment rule's 2.70.
**PPO is above the assignment bound at a speed where assignment buys
nothing.**

So `AssignGreedy` is not a bound on what is achievable; it is a bound on
what *one kind* of coordination achieves. And §11.5's open question
narrows rather than closes: three candidate mechanisms have now been
proposed and measured, and all three are dead.

| mechanism | test | verdict |
|---|---|---|
| decorrelating an over-correlated team | team spread, redundant pairs (§11.5) | **falsified** — the clone is *more* clustered than the expert and matches it |
| interception / leading the target | lead angle vs the expert's 0.00° | **unsupported** — same −7° offset whether or not the target moves |
| dividing the targets | `AssignGreedy` at red 1.0 | **worthless there** (−0.03), yet PPO gains +0.20 |

PPO *does* reduce redundant pursuit (28.9% → 23.3%) and raise team spread
(37.9 → 41.6 m) relative to the clone — that much is measured. But the
assignment sweep says de-conflicting targets cannot be *why* it scores
higher at this speed. The GRU's memory remains untested, and so does
whatever positional behaviour the eval GIFs might show.

#### 12.4 What this sets up

The interesting problem starts at **red ≈ 1.4**, where coordination is
measurably worth +0.37 to a rule that does nothing else. Below 1.3 it is
worth nothing, so a policy trained there cannot demonstrate coordination
however good it is — there is none to demonstrate.

Note what that means for a curriculum: its early rungs teach pure
pursuit, and **§11 already has pure pursuit for free** (the clone is at
2.94 at red 1.0 after 768k supervised steps). A ramp starting at 1.0
would spend its budget re-learning what cloning already gave. Starting
from the clone at ~1.3 and ramping up skips that.

**Caveats.** One seed and one run on the fine-tune. The redundancy and
spread metrics are this project's own constructions, not standard ones.
`AssignGreedy` uses greedy matching rather than Hungarian (no scipy
dependency); at 5×3 the two agree almost always and the difference is far
below the effects measured. And `n_obstacles 0` throughout, so the
obstacle half of the tracker path is still inert.

---

### 13. Coordination, finally measured where it pays (MEASURED, 2026-09-25) · `feature/target-tracking`

§12.2 found that at the configured speed ratio, dividing the targets
between blues is worth **−0.03 ± 0.13** — nothing — and first pays at red
**1.4** (+0.37 ± 0.17). Every result in §6–§12 was therefore recorded in
a regime where coordination could not be demonstrated, however good a
policy was. This is the first run in the regime where it can.

`ppo_red14_v1`: warm-started full from `ppo_from_bc_v1/final.pt`,
`--red-v-max 1.4`, `lr 3e-5`, `aux_hidden_coef 0`, 100 rollouts, 279 min
CPU. No σ reset needed — the parent checkpoint already carried σ 0.304,
which `ent_coef 0.008` had barely moved over the previous 60 rollouts.

**No speed ramp, and none was needed.** The starting checkpoint already
scored 2.56 on the evader at red 1.4 *zero-shot*, above `AssignGreedy`'s
2.36, so there was no cliff to protect it from. (There is also no ramp
mechanism: `red_v_max` is fixed at env construction and the vec env
builds its envs once.)

#### 13.1 It beats the coordination bound by 4.7 standard errors

50 matched-seed episodes per cell, **all rows at red v_max 1.4** — not
comparable with any table above, which are all at 1.0:

| Blue | Stationary | Random | RunFromNearest | **mean /3** |
|---|---|---|---|---|
| Random | 0.94 (198.4) | 1.20 (194.4) | 0.06 (200.0) | **0.733** |
| Greedy (true-in-range) | 2.60 (93.7) | 2.90 (72.8) | 2.06 (180.3) | **2.520** |
| `ObsGreedy` | 2.98 (47.3) | 3.00 (53.8) | 2.04 (182.8) | **2.673** |
| `AssignGreedy` (coordination bound) | 2.98 (45.0) | 3.00 (50.1) | 2.36 (168.9) | **2.780** |
| **Trained `best.pt` @ 80** | **3.00 (52.1)** | **3.00 (46.8)** | **2.92 (122.4)** | **2.973** |
| Trained `final.pt` @ 100 | 2.98 (53.1) | 3.00 (47.1) | 2.90 (126.6) | **2.960** |

The evader column is the whole story. **2.92 against `AssignGreedy`'s
2.36 is +0.56, or 4.7 standard errors** on n=50 matched seeds — and in
**122 steps against 169**, 28% faster. Unlike §12.1's speed figure this
is not timeout conversion: it catches *more* and takes *less* time.

Against the lineage: +0.36 on the starting checkpoint (2.56 zero-shot at
this speed), +0.88 on `ObsGreedy`, +1.00 on the clone's 1.92.

So the policy does something that dividing the targets one-to-one does
not. §12.3 predicted exactly this and could not demonstrate it, because
at red 1.0 there was nothing to demonstrate.

#### 13.2 Two things this table does *not* say

**The acceptance criterion has stopped meaning anything here.** It reads
PASS at +23.62 against a bar of +10.95, but the bar is `1.20 × Greedy`
and Greedy collapses at this speed (2.06 on the evader, down from 2.50 at
red 1.0). A bar that *falls* as the task gets harder measures nothing.
`AssignGreedy` is the reference to quote; the repo-wide criterion should
be retired for any run above red 1.2.

**2.97 is not the task's ceiling — the tracker is over-confident, and
it cannot be tuned out.** `nees` ran **8.2–9.6** against its 4.0 target
for most of this run (and ~12 against the evader alone, which is where a
constant-velocity model hurts most): the covariance understates the real
error. Every row above shares the defect, so the *comparison* holds and
the *level* is depressed.

**The first version of this paragraph blamed `vel_prior_std` not having
been raised with `red_v_max`, as `stage4_backlog.md §21.3` had asked.
That was wrong, and `scripts/sweep_tracker_calibration.py` is what
measured it wrong.** Neither available knob fixes it:

| `a_max` (→ `sigma_a = a_max·√2`) | `nees` | `trk` | caught |
|---|---|---|---|
| **1.00 (shipped)** | 12.31 | 3.81 m | **2.92** |
| 1.50 | **11.58** | 5.76 m | 2.80 |
| 3.00 | 14.30 | 9.48 m | 2.92 |
| 6.00 | 20.37 | 15.01 m | 2.56 |

`vel_prior_std` is only the BIRTH prior and washes out within a few
updates — moving it 1.0 → 2.8 changed `nees` by ~1 out of ~24 of error.
`a_max` *is* the steady-state lever (it sets the process noise and is used
for nothing else), and swept over sigma_a 1.41 → 8.49 the best reachable
`nees` is **11.6**, rising in both directions while track error nearly
quadruples and captures fall.

Which is the finding: **this is not a mis-calibration, it is the motion
model.** `sigma_a` admits more *white* acceleration, while the error
against `run_from_nearest_uav` is *systematic* — the evader turns away
from the nearest blue, every time, and constant velocity predicts
"straight on", every time. Isotropic noise cannot correct a directional
bias; it only blurs the track. The fix is `stage4_backlog.md §15`, the
trained red-motion model that is built and has never been switched on.
Meanwhile the shipped `a_max=1.0` is already the best operating point, so
nothing should be changed on this evidence.

#### 13.3 `best_ckpt_metric` behaved, and why that is informative

§12.1 recorded the selector choosing rollout 5 on a flat curve, leaving
`best.pt` worse than `final.pt`. Here it chose **rollout 80**, and
`best.pt` (2.973) is indeed above `final.pt` (2.960). The difference is
not the metric — it was `det_caught` both times — but the signal: a curve
that actually moves gives 20-episode evals something to select above
their own noise. The §12.1 fix stands (require persistence, or a running
mean) but the diagnosis is now sharper: it fails on flat curves, not in
general.

#### 13.4 Still unexplained, now five hypotheses deep

The mechanism remains unidentified. Every candidate proposed so far has
been measured and none survived:

| mechanism | test | verdict |
|---|---|---|
| decorrelating an over-correlated team | spread, redundant pairs (§11.5) | falsified — the clone is *more* clustered yet matches |
| interception / leading the target | lead angle vs the expert's 0.00° | unsupported — same −7° whether or not the target moves |
| dividing the targets | `AssignGreedy` at red 1.0 | worthless there (−0.03), yet PPO gained |
| cutting the red off | capture geometry, 50 episodes | **15.9% vs the clone's 15.7%** — indistinguishable |
| spreading before the red commits | mean spread over steps 0–25 | **58.6 m vs the clone's 56.6** — indistinguishable |

The last two came from watching the evaluation GIFs, where one episode
showed the fine-tuned team opening early and taking the red head-on. That
episode is real; it does not generalise. Both metrics are flat across 50
matched seeds while the score is not.

What is established is narrower and more robust than any of them: the
policy **transfers to a regime it never trained in** (2.56 at red 1.4
from a red-1.0 run, where the clone it came from drops to 1.92) and
**beats explicit target assignment by 4.7 SE once trained there**. The
GRU's memory is the one proposed mechanism never tested.

#### 13.5 Caveats

* **One seed, one run**, as with §11 and §12.
* **Constant-velocity tracker throughout.** The filter predicts "it will
  keep going straight" about a *reactive* evader that turns whenever the
  nearest blue moves. What kept that honest at red 1.0 was calibration,
  and §13.2 is that calibration failing at 1.4. **§14 switched the learned
  motion model on and measured it**: it ties CV on captures (both at the
  3.00/3 ceiling under the training mix) and is not worth a retrain, so CV
  stays. §14.6 also corrects the NEES figures quoted in §13.2 — they were
  measured against the pure evader, not the training mix.
* **`n_obstacles 0`**, so the obstacle half of the tracker path is still
  inert, as everywhere above.
* **The 3.00/3 at rollout 85 was the peak of the noise.** The last eight
  evals mean 2.94 overall and 2.85 on the evader; quote those.

---

### 14. The learned red-motion model, switched on and measured (MEASURED, 2026-09-25) · `feature/target-tracking`

§13.5 listed the constant-velocity tracker as a caveat: `--red-motion-ckpt`
was built, trained and never used, so the filter predicts "it keeps going
straight" about an evader that turns whenever the nearest blue moves.  This
section switches it on and measures it.  **Verdict: it does not earn a
retrain.**  The reasoning matters more than the verdict, because two of the
three things found along the way are corrections to earlier claims here.

#### 14.1 What the network actually emits — and why `sigma_a_model` exists

The natural expectation is that a learned motion model replaces both halves
of the constant-velocity assumption: the mean (`a = 0`) *and* the
covariance (`Q = G (sigma_a^2 I) G^T`).  It does not.  `RedMotionGNN` emits
a **categorical over a 181-cell (heading, magnitude) grid** — one flat
softmax, no variance head.  So `LearnedRedMotion._basins` assembles each
branch's covariance from three separate sources:

| term | what it is | free? |
|---|---|---|
| spread of cell means across the basin, weighted by the network's own probabilities | the model's **self-reported** uncertainty | no |
| within-cell quantisation spread | fixed property of the grid | no |
| `sigma_a_model^2 * I` | the network's **error against the world** | **yes** |

Only the third is tunable, and it exists precisely because the first cannot
replace it: **the categorical says how concentrated the prediction is, never
how wrong it is.**  A cross-entropy fit reports confidence on its training
distribution and says nothing about generalisation at serve time.

This also dissolves what looked like a paradox in the first measurement —
track error improving while NEES got worse.  NEES is a *ratio*: error
divided by declared covariance.  The learned model cut the numerator ~10%
(5.28 -> 4.78 m) while `sigma_a_model = 0.10` cut the denominator far more
(0.10 against CV's 1.414 is 14x in sd, ~200x in variance).  Numerator down
a little, denominator down a lot, ratio up: 12.4 -> 21.3.  That is the
signature of a better predictor with badly declared confidence, not of a
worse predictor.

#### 14.2 Which term dominates flips between regimes

`docs/tracking_diagnostics.md §11.7` calibrated the shipped `0.10` and found
`sigma_a_model` **nearly inert**: 0.35 -> 0.0 moved the NEES median
1.64 -> 1.67, because v4 has to hedge across stationary/random/run and the
basin spread carries the uncertainty.  The first sweep here found the
opposite — 0.10 -> 0.70 moving NEES 19.8 -> 10.8 — because it was run
against **pure `run_from_nearest_uav`**, where the model stops hedging, the
basin term collapses and the additive term is all that is left.

Both are right about their own regime.  The value does not transfer, and
the sweep below therefore measures **both** the pure evader and the
trainer's actual `stationary:1,random:1,run:1` mix.

#### 14.3 A bug that invalidated the first sweep

`PursuitEnv` constructed `LearnedRedMotion` without passing `red_v_max`, so
the adapter kept its `RED_TARGET.v_max` default of **1.0** while the env ran
reds at **1.4**.  `_advance()` clips every predicted branch to that cap:
a systematic under-prediction of speed by up to 40%.

That is **bias, not spread**, so no covariance term can absorb it — and it
is the most likely explanation for the NEES floor around 10.8 that the
first sweep could not get past.  Fixed at `pursuit_env.py:736`
(`v_max=self.red_v_max`), with the reasoning in a comment so it is not
re-introduced.

**The first sweep's red-1.4 rows measured this bug**, and its headline
result did not survive the fix:

| red 1.4, `sigma_a_model` | before (capped at 1.0) | after (cap fixed) |
|---|---|---|
| 0.10 (shipped) | nees 19.77, caught 2.80 | nees 15.04, caught 2.84 |
| **0.70** | nees 10.79, caught **3.00** | nees 14.29, caught **2.88** |
| CV baseline | nees 12.31, caught 2.92 | unchanged (does not use the model) |

The 3.00/3 that made the model look like a clear win was an artefact.

#### 14.4 The sweep, cap fixed

`ppo_red14_v1/best.pt`, deterministic, 25 matched seeds per row, NEES
target 4.0.  `mix` resamples one policy per episode exactly as the trainer
does.

**Pure `run_from_nearest_uav`** (1/3 of the training distribution):

| red | `sigma_a_model` | nees | trk (m) | caught | steps |
|---|---|---|---|---|---|
| 1.0 | CV | 12.40 | 5.28 | 3.00 | 80.4 |
| 1.0 | 0.70 | 15.53 | 4.93 | 3.00 | **69.8** |
| 1.4 | CV | 12.31 | 3.81 | **2.92** | 120.4 |
| 1.4 | 0.85 | 10.52 | 3.99 | 2.92 | 111.1 |

**The trainer's mix** — the regime that actually matters:

| red | `sigma_a_model` | nees | trk (m) | caught | steps |
|---|---|---|---|---|---|
| 1.0 | CV | **5.92** | 6.35 | 3.00 | 59.6 |
| 1.0 | 0.40 | 8.39 | 5.82 | 3.00 | 54.0 |
| 1.0 | 0.85 | 7.70 | 6.27 | 3.00 | 55.3 |
| 1.4 | CV | 10.48 | 5.53 | 3.00 | 68.1 |
| 1.4 | 0.85 | **8.94** | 5.43 | 3.00 | **66.6** |

#### 14.5 Why the gate fails

**Captures are at ceiling.**  Under the training mix CV already takes 3.00/3
at both speeds.  There is no headroom for the motion model to demonstrate
value on the metric the project is judged by, and every learned row ties it.

The remaining axes are close and noisy:

* **NEES** — CV wins at red 1.0 (5.92 against the best learned 7.70),
  the model wins at 1.4 (8.94 against 10.48).  A wash.  And across the
  sweep NEES has no monotone trend at all (at red 1.4, pure evader:
  15.0, 17.2, 11.9, 14.3, 10.5, 12.6, 15.3), so at n=25 the estimator's
  own variance exceeds the effect being swept.  **`sigma_a_model` is not
  a strong lever here**, which is §11.7's finding reproduced.
* **Track error** — the model is marginally better everywhere
  (5.43–6.27 m against 5.53–6.35 m).  Within noise.
* **Time to capture** — the one real gain: 59.6 -> 51.2–55.3 steps under
  the mix at red 1.0, about 12%, and the pure-evader block agrees
  (80.4 -> 69.8).  But at red **1.4** it shrinks to 68.1 -> 66.6, ~2%.

So the only solid improvement is time-to-capture **in the regime we are
leaving**.  Against that: re-collecting the dataset, retraining the motion
model, then retraining the clone and PPO on top.  Not worth it.

**Decision: keep the constant-velocity tracker.**  The learned path stays
built, wired and now measured rather than assumed.

#### 14.6 Two corrections to earlier claims

* **"NEES bottoms out at 11.6 against a 4.0 target, the error is
  systematic"** (§13.2 and the `sweep_tracker_calibration` work) was
  measured against **pure `run_from_nearest_uav`**.  Under the actual
  training mix the CV tracker reads **5.92 at red 1.0** — 1.9 from target,
  not 8.4.  The tracker is far better calibrated in the regime it trains
  in than that line implied.  The degradation at red 1.4 is real
  (5.92 -> 10.48), and that part of §13.2 stands.
* **§13.5's caveat** that `--red-motion-ckpt` is "built, trained and never
  used" is superseded by this section; `stage4_backlog.md §15` likewise.

#### 14.7 What v4 was trained on, and what retraining would need

Recorded because the question recurs and the answer is not in any one file:

| | v4 dataset | current training config |
|---|---|---|
| reds | `stationary / random / run` | same |
| blues | `GreedyPursuer` + `RandomAgent`, p_greedy ~ U(0.3, 1.0) | trained PPO policy |
| red `v_max` | **1.0** (the collector has no flag) | **1.4** |
| arena | **200** | **130** |
| capacities | 7 blue / 9 obstacle | 5 blue / 0 obstacle |

So v4 is out of distribution on speed, arena and blue policy simultaneously.
If it is ever revisited, `collect_red_motion_dataset.py` needs a
`--red-v-max` that **samples** rather than fixes the speed — otherwise the
same gap reopens the moment red speed becomes variable within an episode.

#### 14.8 Caveats

* **25 episodes per row**, which the NEES non-monotonicity shows is not
  enough to resolve `sigma_a_model` — but is enough for the capture-ceiling
  argument, which is what the decision rests on.
* **`LEARNED_MOTION_CONFIG` is left at 0.10.**  Under this geometry 0.85
  looks better, but that rests on the same 25-episode noise, and 0.10 was
  calibrated correctly for §11.7's geometry (arena 200).  Changing a
  global default to fit one noisy sweep in one geometry would be
  overfitting; the regime-dependence is documented in the adapter's
  docstring instead.
* **`n_obstacles 0`**, so `o2r` edges — a third of what the model
  conditions on — were inert in every row above.

---

### 15. Red 1.5: where pure pursuit stops working (MEASURED, 2026-09-26) · `feature/target-tracking`

§13 measured coordination at red 1.4 and found it worth +0.37 — real, but
thin next to a Greedy baseline already scoring 2.50/3.  The reason to go to
1.5 is geometric rather than incremental: `BLUE_UAV.v_max` is **1.5**, so at
red 1.5 the evader is exactly as fast as its pursuers and chasing from
behind can never close.  Capture has to come from interception, which means
**coordination stops being an advantage and becomes a requirement**.

That prediction is visible before any training, in the heuristic baselines
the trainer prints at startup:

| Greedy vs the evader | caught | steps |
|---|---|---|
| red 1.4 | 2.50 / 3 | 169.2 |
| **red 1.5** | **1.00 / 3** | 191.0 |

Pure chase falls off a cliff between 1.4 and 1.5.  That is what makes 1.5
the regime with measurable headroom — and §14 had just shown how easily a
saturated metric wastes an experiment.

#### 15.1 Two training cycles

Both warm-started, `n_rollouts 100`, otherwise identical to `ppo_red14_v1`
(verified by diffing the reconstructed args dict against the checkpoint's,
which caught a `belief_grid_size` default that had drifted 26 -> 40 since
that run and would have silently changed the comparison).

| run | warm start | `lr` | elapsed |
|---|---|---|---|
| `ppo_red15_v1` | `ppo_red14_v1/best.pt` | 3e-05 | 322.7 min |
| `ppo_red15_v2` | `ppo_red15_v1/final.pt` | **6e-05** | 341.3 min |

The second cycle exists because v1's optimizer diagnostics said its plateau
was partly the learning-rate schedule and not convergence: at rollout 100
`kl` was **0.0011** against a `target_kl` of 0.03 — 27x below, so the trust
region *never once* cut an update short — with `clip` down to 0.006 and
`lr` decayed to 3e-06.  Entropy had not collapsed (0.510 -> 0.564), so
exploration was still alive.  The binding constraint was the schedule.

Raising `lr` to 6e-05 confirmed that reading and then hit the same wall:
v2's `kl` peaked at **0.0095**, still a third of target, and v2 was flat
from rollout 5 to 100 across a 4x range of `lr`.  **So ~2.7 is a real
ceiling for this configuration at red 1.5, and the bottleneck is not the
optimizer.**  A third cycle at a higher `lr` is not the lever; what remains
is the reward shape, the observation, network capacity, or the fraction of
episodes at red 1.5 that are simply unwinnable.

#### 15.2 The reference table

50 matched seeds, `run_from_nearest_uav`, deterministic, tracker
observation.  The heuristics run in the **same partial-observability env as
the policy** — unlike the trainer's startup baselines, which use
`sensor_radius=None` and so are not comparable.

| blue | red | caught | steps | return |
|---|---|---|---|---|
| `ObsGreedy` | 1.5 | 0.66 ± 0.11 | 197.2 | -13.58 |
| `AssignGreedy` | 1.5 | 1.06 ± 0.14 | 192.7 | -7.38 |
| `red14/best`, no retraining | 1.5 | 2.16 ± 0.13 | 170.1 | 10.04 |
| `red15_v1/final` | 1.5 | 2.72 ± 0.08 | 148.4 | 19.20 |
| `red15_v2/best` | 1.5 | 2.72 ± 0.09 | 149.2 | 19.09 |
| **`red15_v2/final`** | 1.5 | **2.72 ± 0.08** | **143.8** | **19.29** |
| `red15_v2/final` | 1.4 | 2.96 ± 0.03 | 109.9 | 24.47 |
| `red14/best` | 1.4 | 2.92 ± 0.04 | 118.2 | 23.79 |

Table produced by `scripts/eval_red15_reference.py`.

**`red15_v2/final` is the reference checkpoint** for everything downstream:
tied on captures, fewest steps, best return.

#### 15.3 What the table says

* **The coordination claim gets its cleanest evidence yet.**  Against the
  best heuristic in the same observation regime, 2.72 against **1.06** —
  a margin of **+1.66, about 10 SE**.  At red 1.4 the comparable margin was
  +0.34.  Whatever the policy is doing, it is not pursuit: pursuit is on the
  table at 1.06 and 0.66.
* **Coordination's value keeps growing with red speed.**  The
  `AssignGreedy` − `ObsGreedy` gap — coordination by construction, target
  division, nothing learned — reads **−0.03 at red 1.0** (§13), **+0.37 at
  1.4** (§13), **+0.40 at 1.5**.  Monotone, and the sign flip between 1.0
  and 1.4 is the whole story of why §13 had to move off 1.0.
* **The curriculum step was worth +0.56.**  `red14/best` transplanted to
  1.5 with no retraining scores 2.16; trained at 1.5 it reaches 2.72.  At
  these SEs that is ~3.6 SE — solid.  Note the training log *understated*
  this: its rollout-5 eval already read 2.32, because five rollouts of
  training had happened by then.
* **No catastrophic forgetting.**  Trained at 1.5, the policy scores
  **2.96 ± 0.03 back at red 1.4**, against `red14/best`'s 2.92 ± 0.04 — a
  tie at worst, and 7% fewer steps.  The harder regime cost nothing on the
  easier one it came from.

#### 15.4 A correction

**The second cycle did not improve captures.**  Reading the training logs,
v2's mean over its last eight evals was 2.750 against v1's 2.590, and that
was reported here as "+0.16, the `lr` bump paid off".  Measured properly —
both final checkpoints on the same 50 seeds — they are **identical at
2.72 ± 0.08**.  The 25-episode evals in the log were noisier than the
difference, and v1's last-eight window happened to include its 2.28 dip at
rollout 90.

What v2 did buy is ~3% in time-to-capture (143.8 against 148.4 steps) and
the ceiling argument in §15.1, which was the methodological reason for
running it: an under-trained 1.5 reference would confound any later
`comms_radius` result with "more training".  That reason still holds.  The
capture gain does not exist.

#### 15.5 Caveats

* **One seed per run**, as with §11–§14.  Two cycles at different `lr` are
  not two seeds.
* **`n_obstacles 0`**, as everywhere above.
* **Constant-velocity tracker**, per §14's decision.
* `stat` and `rand` sit at 3.00 throughout, so every number that moves here
  is the evader's.
* The unwinnable-episode fraction at red 1.5 is **not** measured.  Until it
  is, "2.72 is the ceiling" is a statement about this configuration, not
  about the task.

---

### 16. `comms_radius`, and why the free ablation cannot answer the question (MEASURED, 2026-09-26) · `feature/target-tracking`

`bb_edge_visible` was gated by `sensor_radius` — the same number that decides
whether a blue can DETECT A TARGET.  That conflated two physically different
channels: a bb edge is a radio message between our own drones, and a datalink
outranging an onboard sensor by an order of magnitude is the normal case in
real UAV teams.  Backlog §7 tracked decoupling them; `comms_radius` does it,
defaulting to `sensor_radius` so every earlier result is untouched.  It does
**not** touch `rb_edge_visible`: seeing a target is sensing.

Then, before spending a retrain on it, an eval-time ablation — free, because
`comms_radius` changes no tensor shapes (bb edges are already complete,
`n_blue*(n_blue-1)`; only the mask moves).

#### 16.1 The ablation, and the confound that invalidates it

`ppo_red15_v2/final.pt`, red 1.5, 50 matched seeds, deterministic.  The
policy trained with comms gated at `sensor_radius` 40.

The second column is measured separately, on the **same fixed states** (59
sampled from trajectories the policy visits under its trained mask), so the
input perturbation is isolated from the trajectory divergence:

| `comms_radius` | rel. bb magnitude | caught | steps |
|---|---|---|---|
| 20 | **0.55x** | **2.88 ± 0.05** | 141.2 |
| 40 (as trained) | 1.00x | 2.72 ± 0.08 | 143.8 |
| 60 | 1.56x | 2.70 ± 0.08 | 151.5 |
| 90 | 2.23x | 2.26 ± 0.11 | 177.6 |
| 130 | 2.48x | 1.70 ± 0.15 | 187.5 |
| inf | 2.50x | **1.26 ± 0.16** | 190.4 |

> **The mechanism below is SUPERSEDED by §19.2.**  This section attributes the
> collapse to magnitude saturating the `tanh` activations.  §19 trained both
> aggregations from scratch and found a better explanation: the four pathways
> share one aggregate, so opening the radio does not merely inflate a scale,
> it squeezes the TARGET channel from 22% to 15% of that aggregate while ally
> chatter grows to 68%.  That also predicts the stationary-red failure §19.1
> measured, which saturation does not.  The numbers here stand; the
> interpretation does not.  Table produced by `scripts/eval_comms_radius.py`.

**The damage tracks the magnitude, not the information.**  The policy
aggregates ally messages with an UNNORMALISED SUM — `agg.index_add_(1,
self.bb_dst, msg_bb)`, with `bb_visible` multiplied into `msg_bb` first — so
the count of open bb edges scales what reaches `update_mlp`.  Opening the
radio to `inf` multiplies that term by 1.9x against what the policy trained
on (the 2.5x in the table above is the small-sample figure, revised below);
the score collapses to 1.26, barely above `AssignGreedy`'s 1.06 (§15.2).
Narrowing to 20 m is a *smaller* perturbation and the score goes slightly UP.

`gnn_stage4_policy.py` already knows this failure mode.  Its region path is
normalised for exactly this reason — *"an unnormalised `index_add_` SUM would
grow with K, making the resolution R also a scale knob on this pathway"* —
and `bb`/`rb`/`ob` were left as raw sums.

**So the ablation cannot answer the question in EITHER direction.**  The
pre-registered reading was: a drop proves nothing (out-of-distribution inputs
degrading a policy is the expected null), a rise would be strong evidence, and
a flat response to *narrowing* would say the policy barely uses ally edges.
What was not anticipated is that **narrowing helping is equally
uninformative** — it is just a gentler perturbation.  Neither the 1.26 nor
the 2.88 says anything about ally information.

One crack in the scale story, left standing rather than argued away: 130 and
`inf` differ by almost nothing in magnitude (2.48x against 2.50x) yet score
1.70 against 1.26, about 2.4 SE apart.  Scale alone does not explain all of
it; once trajectories diverge, everything compounds.

**The magnitude column above is revised downward by a later, larger
measurement.**  It came from 59 sampled states and put comms 40 at 40.0% of
bb edges open.  Measured on full trajectories — 8,185 (blue, step) samples —
the figure is **52.8%**, i.e. 2.11 live edges of 4, so opening the radio is a
**1.9x** jump rather than 2.5x.  The mechanism and the conclusion are
unchanged; the number was over-stated.  The same measurement gives the
per-receiver distribution that §18.4 turns on: 0 live ally edges in 12% of
samples, 1 in 26%, 2 in 23%, 3 in 14%, 4 in 24%.

#### 16.2 What can answer it

A retrain.  That comparison is **not** confounded: both policies train to
convergence under their own aggregate scale, so comparing `red15_v2` against
a run with `--comms-radius inf` is an end-to-end comparison of two trained
policies, which is the question actually being asked.

#### 16.3 A latent confound worth recording

The same unnormalised sum makes **blue team size** a scale knob on the bb
pathway, since `n_blue` sets the edge count.  It does not bite today —
`n_blue` is fixed at 5, and there are `--n-red-min` and `--n-obstacles-min`
flags but no `--n-blue-min` — but any future variable-blue-count work
inherits it.  Normalising bb/rb/ob the way the region path already is would
remove both this and the `comms_radius` scale coupling, at the cost of
invalidating every existing checkpoint.

#### 16.4 Caveats

* The ablation is **one checkpoint**.  A different policy might be more or
  less scale-sensitive.
* The magnitude column is a proxy: it counts open edges rather than measuring
  `||agg||` through the network, so it captures the scaling mechanism but not
  the non-linear response to it.
* `n_obstacles 0`, so the `ob` pathway was inert and its own sum-scaling
  never exercised.

---

### 17. The comms experiment, and two things it found instead (MEASURED, 2026-09-28) · `feature/target-tracking`

§16 established that flipping `comms_radius` on a trained checkpoint cannot
answer whether unlimited comms helps, and that every arm therefore has to be
trained from scratch.  Three arms — `comms_radius` 20, 40 and `inf` — each
clone (60 rounds) then PPO (150 rollouts) at red 1.5, 31 h of unattended
compute.

**It did not answer the question.**  It found two other things, one of which
is more useful than the answer would have been.

Produced by `scripts/run_comms_experiment.sh` (three arms, unattended) with
`scripts/eval_checkpoint.py` for every number below.

#### 17.1 The comms result: null, and premature

50 matched seeds against the evader, deterministic:

| arm | `comms_radius` | clone | PPO best | PPO final |
|---|---|---|---|---|
| `inf` | inf | 1.34 ± 0.14 | 1.74 ± 0.14 | **1.86 ± 0.12** |
| `narrow` | 20 | 1.10 ± 0.14 | 1.68 ± 0.14 | 1.56 ± 0.15 |
| `incumbent` | 40 | 1.00 ± 0.12 | 1.62 ± 0.12 | 1.56 ± 0.14 |
| *reference, curriculum* | 40 | — | — | *2.72 ± 0.08* |

`inf` over `incumbent` is 0.30 with a combined SE of 0.18 — **1.6 SE**, not
significant.  20 and 40 are *identical* at 1.56, so there is no monotone
trend either.

And the two reasonable ways of reading the data **disagree on the ordering**.
By final checkpoint `inf` leads (1.86 vs 1.56).  By the mean of the last
eval block in the training log, `narrow` leads (1.83 vs 1.78).  When the
ranking inverts depending on which summary you take, the differences are
noise.

The reason is visible in the eval trajectories: **all three arms were still
rising at rollout 150.**

| eval block | `inf` | `narrow` | `incumbent` |
|---|---|---|---|
| rollouts 5–30 | 1.39 | 1.17 | 1.14 |
| 95–120 | 1.75 | 1.72 | 1.51 |
| 125–150 | 1.78 ↗ | 1.83 ↗ | 1.63 ↗ |

None plateaued, and `kl` finished at 0.0009–0.0013 against a `target_kl` of
0.03 — 20-30x under, with `lr` decayed to near zero.  **The arms are
under-trained, not converged**, so this compares three unfinished policies.

#### 17.2 The incumbent control is what saved it

`comms_radius` 40 **from scratch** reaches 1.56.  The same `comms_radius` 40
**via curriculum** reaches 2.72.  Identical comms setting, a gap of ~1.2 —
about 7 SE.

So the entire distance from the new arms to the reference is **training path
and budget, nothing to do with comms**.  Without this arm the comparison
would have been `inf`'s 1.86 against the reference's 2.72, and the natural
reading — "unlimited comms hurts" — would have been flatly wrong.

Recorded as a method note, because it generalises: **when an experiment
changes the training path, it needs an incumbent arm on that same path, or a
path effect gets attributed to the treatment.**

It is also direct evidence that the curriculum does real work rather than
merely saving time.  §15 priced the 1.4 → 1.5 step at +0.56; this prices the
whole curriculum at ~1.2.

#### 17.3 A design error, owned

150 rollouts was chosen by scaling from the curriculum runs, where 100
sufficed — but those started from an already-good policy.  From scratch at
red 1.5, beginning from a clone scoring 1.00–1.34, it is not enough.  The
evidence was available beforehand and was not used.  The 31 h produced three
under-trained policies and one useful control.

#### 17.4 The task almost never requires exploration

Prompted by the question of whether the 2.72 policy ever learned to search,
measured on `ppo_red15_v2/final.pt` at red 1.5, 30 episodes:

| | |
|---|---|
| arena | 130 × 130 = 16,900 m² |
| sensor disc × 5 blues | 25,133 m² |
| **team sensor coverage** | **1.49x the whole arena** |
| **reds already inside a blue's sensor at t = 0** | **78.9%** |
| steps until all three reds have been seen at once | **median 6** (of 200) |

Four reds in five are visible before anything moves, and the team's sensors
cover the arena one and a half times over.  **There is essentially nothing to
explore**, which is consistent with §9.3's correction that the policy chases
clutter rather than stale cells: the coverage pathway has had almost no work
to do in any result recorded here.

Nothing above is invalidated by this — captures are captures — but any claim
that the policy *learned to search* is unsupported, and the staleness/region
machinery has never been tested under conditions that need it.

Forcing exploration needs the analogue of raising red speed to 1.5: break the
coverage ratio.  Two knobs, neither clean.  Shrinking `sensor_radius`
(40 → 20 gives 0.37x) is one number and keeps the geometry, but the sensor
also feeds the tracker and the rb edges, so it confounds exploration with
tracking difficulty.  Growing the arena (130 → 260 gives 0.37x) keeps sensor
quality fixed but changes travel times and interacts with red speed.  The
arena is preferred: perception quality is the one thing not to move when
perception is the object of study.

#### 17.5 What would actually answer the comms question

Not more sweeps.  §16.1 showed the aggregation is an unnormalised sum, so
`comms_radius` is a scale knob as well as an information knob, and §17.1
showed the from-scratch path is too weak to separate anything.  The next step
is **attention** (§18 design), which makes the aggregation scale-invariant and
turns `comms_radius` into a pure information knob — and the decisive
experiment becomes sum vs attention, both at `comms_radius inf`, both
continuing the curriculum from `ppo_red14_v1/best.pt`.

#### 17.6 Caveats

* **One seed per arm.**  Three arms are not three seeds.
* The non-evader reds sit near ceiling throughout (2.72–2.94), so only the
  evader column carries information.  `narrow` is notably faster there
  (100.8 steps at its best checkpoint against `inf`'s 140.2), which is a real
  difference but on a saturated metric.
* `n_obstacles 0`, as everywhere above.
* The exploration figures are for `ppo_red15_v2` at red 1.5 only; a different
  arena or sensor radius would change them entirely, which is the point.

---

### 18. Attention over the typed edges (DESIGN, 2026-09-28) · not yet built

Not a measurement — the agreed design, written down before building so the
experiment it enables is fixed in advance rather than chosen after seeing
results.

#### 18.1 Why, in one line each

The encoder aggregates every incoming message with an **unnormalised sum**:
`agg.index_add_(1, self.bb_dst, msg_bb)`, with the visibility mask multiplied
into `msg_bb` first.  So the *count* of live edges scales what reaches
`update_mlp`.  That single fact is behind three separate blocked items:

| blocked item | how the sum blocks it |
|---|---|
| `comms_radius` (§16, §17) | opening the radio is a 1.9x magnitude change, so the knob cannot be varied without also perturbing scale — the experiment is uninterpretable |
| variable red counts | 1 → 3 active reds triples the rb term; a policy trained at one count is out of distribution at another |
| heterogeneous red speeds | no way to weight the fast red over the slow one; every sender contributes equally |

Attention replaces the sum with a **convex combination**: weights are a
softmax over the incoming edges of each receiver, so they sum to 1 regardless
of how many edges are live.  Magnitude stops depending on count, and the
weights themselves become the selectivity the three items above need.

The operational framing that motivated this: *the right coordination is not
the same against 1, 2 or 3 evaders*.  A blue should be able to read the reds'
positions, count and estimated velocities and conclude it needs one ally on a
target, or two, or all of them.  A sum cannot express that; an attention
weight is exactly that quantity.

#### 18.2 Where it goes

Per receiver, per edge type, per message round.  The current path is

```
msg   = msg_mlp([h_send, h_recv, e_edge])      # (B, n_edges, d)
msg   = msg * visible                          # mask -> zero
agg  += index_add(msg)                         # UNNORMALISED SUM
h_recv = h_recv + update_mlp([h_recv, agg])
```

and becomes

```
msg   = msg_mlp([h_send, h_recv, e_edge])      # unchanged
score = att_mlp([h_send, h_recv, e_edge])      # (B, n_edges, n_heads)
score = score.masked_fill(~visible, -inf)      # masked BEFORE softmax
alpha = softmax(score, over each receiver's incoming edges)
agg  += index_add(alpha * msg)                 # CONVEX COMBINATION
```

Four decisions, each with its reason:

**Masking before the softmax, not after.**  Zeroing a weight after softmax
leaves the remaining weights summing to less than 1, which reintroduces the
scale dependence the change exists to remove.  `-inf` before the softmax
removes the edge from the normalisation entirely.

**Per edge TYPE, not one softmax over everything.**  bb, rb, ob and gb are
different questions ("which ally", "which target", "which obstacle", "where to
look").  One softmax across all of them would force a blue to trade attention
on a target against attention on a wingman, and would make the *relative*
weight of the four pathways a learned quantity that varies per state — a much
larger change than intended.  Each type normalises within itself and the four
results are summed, exactly as today.

**A receiver with NO live edges of a type must get a zero vector**, not
`NaN`.  An all-`-inf` row makes softmax produce `NaN`, which then poisons
every downstream gradient.  This is the single most likely way to get a run
that trains for an hour and emits garbage, so it needs an explicit guard and
a test.  It is not an edge case: measured on `ppo_red15_v2` at red 1.5,
**27.6% of steps have no confirmed track at all**, and on the bb side 12% of
(blue, step) samples have no live ally edge.

A correction on what the rb mask *is*, since an earlier draft of this section
had it wrong.  In tracker mode the actor's "red" nodes are the **8 tracker
slots**, not the 3 true reds, and `rb_edge_visible` is 8x5 = 40 edges, not
the Stage-3 path's 3x5 = 15.  More importantly the mask is **identical across
all five blues in 100% of steps that have any track**: the tracker is
command-layer fusion, so it encodes "which slot holds a confirmed track", a
GLOBAL fact, not "which red can this blue see".  When it is empty it is empty
for the whole team at once.

That makes rb attention a different and more interesting thing than per-blue
visibility gating: since every blue reads the same tracks, attention over rb
is not "what can I see" but **"which target do I commit to"** — learned
target assignment, and the natural place for the 1-vs-2-vs-3 allocation that
motivates this work.  §13 looked for exactly that mechanism with five
hypotheses and found none of them.

**Multi-head, heads = 4 at `d_hidden` 64.**  One head forces a single
ranking; "the nearest ally" and "the ally nearest my target" are different
questions and a single softmax has to pick one.  Four heads at 16 dims each
is the conventional split and adds little.

#### 18.3 What it costs

`att_mlp` is one extra MLP of the same input width as `msg_mlp`
(`3 * d_hidden`) with `n_heads` outputs instead of `d_hidden` — so roughly
`d_hidden / n_heads` times *fewer* output parameters than `msg_mlp`, about a
5-8% parameter increase on 145,989.  The forward cost is one more MLP per edge
type per round plus a scatter-softmax.

**It invalidates every existing checkpoint.**  The `state_dict` gains tensors
and the aggregation semantics change, so `policy_loader` cannot load an old
checkpoint into the new class and must not silently try.  The knob therefore
ships as `--attention` defaulting **off**, with the loader reading it from the
checkpoint's args the way `use_staleness` and `comms_radius` already are, so
every result in this document keeps evaluating exactly as it was trained.

#### 18.4 The experiment, fixed in advance

> **Outcome in §19, and two claims below were wrong.**  (1) Parity was
> pre-registered as the expected result; attention won by +0.88 on the evader,
> about 10 SE.  (2) The table below says the normalisation benefit at comms
> inf is "nil — constant scale, absorbable by the weights".  §19.2 falsifies
> that: the reasoning treated bb in isolation, and the pathways share one
> aggregate.  None of the four pre-registered readings matched the outcome
> (§19.3).  The design below is what ran; the predictions are kept as written
> rather than edited into agreement with the result.

An earlier draft warm-started both arms from `ppo_red14_v1/best.pt` and
called the resulting asymmetry "the hypothesis".  That was too convenient and
is withdrawn: **neither arm warm-starts cleanly.**  Arm A eats the magnitude
jump from opening comms, but arm B is displaced too, in the opposite
direction — the old `msg_mlp`/`update_mlp` were trained consuming a sum of
~2.11 messages, and a convex combination is worth ~1, so arm B starts ~0.47x
displaced with a randomly initialised `att_mlp` on top.  Two differently
broken warm starts do not make a controlled comparison.

**The two comms settings also test different halves of the hypothesis**,
which decides where to run it.  Measured live bb edges per receiver at comms
40: 0 in 12% of samples, 1 in 26%, 2 in 23%, 3 in 14%, 4 in 24%.

| | comms 40 | comms inf |
|---|---|---|
| live allies | swings 0–4 | **always 4** |
| **normalisation** benefit | **maximal** — scale varies 4x | nil — constant scale, absorbable by the weights |
| **discrimination** benefit | **diluted** — softmax over 1 element is 1.0, so attention is a literal no-op in the 38% of samples with 0 or 1 live edge | **maximal** — always four allies to rank |

The question this work exists to answer — can a blue decide it needs one ally
on a target, or two, or three — is the **discrimination** half.  So it is run
at `comms_radius inf`, and testing at 40 would have measured the wrong half.

Both arms therefore train **from birth at comms inf**, each with its own
clone (they cannot share one, the architectures differ), through a
1.4 → 1.5 curriculum:

| | sum arm | attention arm |
|---|---|---|
| clone at comms inf, red 1.4 | ~2 h | ~2.5 h |
| PPO @ red 1.4 | ~8 h | ~10 h |
| PPO @ red 1.5 | ~8 h | ~10 h |
| | **~18 h** | **~22.5 h** |

so **~41 h** for both.  Attention's measured throughput cost is **1.25x**
(90 against 72 steps/s at the production configuration), and the encoder
forward alone is 1.0-1.4x — it is the aggregation that changes, not the
model's size, and the parameter count grows 8.6% (145,989 -> 158,601).

A first measurement of that cost read 47x and would have put the experiment
at 22 days.  It was an artefact: an orphaned process left by a malformed
command was holding a core throughout.  Worth the note because the lesson
generalises — check for stray processes before believing any timing on this
machine.

Curriculum rather than a single from-scratch run at 1.5, because §17
measured that path as ~1.2 weaker and still rising at 150 rollouts.  Starting
the clone at 1.4 rather than 1.0 is what keeps this at 36 h instead of ~52 h.
Neither arm is ever displaced: each sees its own aggregation and unlimited
comms from its first step.

**Expect nothing from rb attention in this experiment.**  Tracks are global
and only **1.04 of 8 slots are live on average** (max 3), so there is
usually nothing to discriminate between.  The target-assignment payoff needs
more reds or variable counts — the roadmap item this unblocks.  What this
experiment can show is bb attention: breaking the symmetry between blues that
are all reading the same global picture.

Read it against two reference points, not one:
* `ppo_red15_v2/final.pt` = **2.72 ± 0.08**, the curriculum's best at
  comms 40, which is what "did attention beat the incumbent" means;
* arm A itself, which is what "did attention beat the sum, all else equal"
  means.

Pre-registered readings, so the result cannot be reinterpreted after the fact:
* **B > A and B ≥ 2.72** — attention helps and unlimited comms is usable.
  Proceed to variable red counts, where the sum is a hard blocker anyway.
* **B ≈ A, both < 2.72** — the handicap of opening comms dominates; rerun both
  at comms 40 before concluding anything about attention.
* **B ≈ A ≈ 2.72** — aggregation is not the bottleneck at this scale.
  Attention still earns its place for variable counts, but not on this metric,
  and that should be said plainly rather than hunted for in subgroups.
* **B < A** — attention costs more than it buys here; keep it off by default
  and record it.

#### 18.5 Deliberately NOT in this change

* **The region path keeps `gb_weight`.**  It is already a hand-designed
  normalised weighting applied per-edge at exactly the site attention weights
  would occupy, so it is the natural *second* place to apply this — but
  changing it in the same step would confound the two.  It is the obvious
  follow-up once §18.4 has an answer.
* **`rb`, `ob` attention is built but the experiment does not isolate it.**
  All four types get the mechanism (leaving some as sums would be a strange
  hybrid), but §18.4 only asks about the aggregate effect.
* **No exploration changes.**  §17.4 showed the task barely requires search,
  and fixing that means changing the arena, which would invalidate the 2.72
  reference this experiment is measured against.  Architecture first, geometry
  second.

---

### 19. Attention beats the sum decisively, for a reason §18 got wrong (MEASURED, 2026-10-01) · `feature/target-tracking`

§18.4 pre-registered **parity** as the expected outcome and as a pass.  That
was wrong.  Two arms, identical but for the aggregation, both at
`comms_radius inf`, both clone -> PPO@1.4 -> PPO@1.5, 150 rollouts per stage:

| | evader | stationary | random | steps (evader) |
|---|---|---|---|---|
| **sum** | 1.60 ± 0.07 | **2.62 ± 0.04** | 2.68 ± 0.04 | 189.0 |
| **attention** | **2.48 ± 0.05** | **2.98 ± 0.01** | 3.00 ± 0.01 | 156.9 |
| *reference* (curriculum, comms 40) | 2.75 ± 0.04 | 3.00 ± 0.00 | 3.00 ± 0.00 | 141.4 |

200 episodes, matched seeds, deterministic.  The clones started **level** —
1.78 ± 0.12 against 1.86 ± 0.11, half an SE apart — so the comparison is fair.
`best` and `final` agree in both arms (1.60/1.59, 2.48/2.49), so this is not a
checkpoint-selection artefact.  The evader gap is **+0.88, about 10 SE**.

Neither arm had an optimiser pathology: `kl` 0.0012-0.0016 against a 0.03
target, `clip` 0.005-0.011, entropy 0.44-0.59 in both.  The sum arm did not
break, it learned worse.

Produced by `scripts/run_attention_experiment.sh`; all evaluations via
`scripts/eval_checkpoint.py`.

#### 19.1 The decisive column is the STATIONARY red

A stationary red does not move.  Scoring **2.62/3** against one is not "worse
pursuit" — it is failing to act reliably on where the target is.  Every policy
recorded in this document sat at 3.00 there.  The summed arm at
`comms_radius inf` is broken in a basic way, and attention restores it to
2.98/3.00.

#### 19.2 The mechanism — and what §18 got wrong

§18.4 argued that at comms inf the normalisation benefit would be **nil**,
because the live bb count is then constant at 4 and "a constant scale factor
is absorbable by the weights".  That reasoning treated bb **in isolation**.
It is wrong because the four pathways **share one aggregate**, so what matters
is their RELATIVE magnitudes:

| pathway | sum @ comms 40 | sum @ comms inf | mean or attention |
|---|---|---|---|
| bb (allies) | 2.11 msgs, 53% | **4.0 msgs, 68%** | ~1, 33% |
| **rb (targets)** | 0.86, 22% | **0.86, 15%** | ~1, 33% |
| gb (regions) | ~1.0, 25% | ~1.0, 17% | ~1, 33% |

Opening the radio squeezes the **target channel** from 22% to 15% of the
aggregate while ally chatter grows to 68%.  The policy loses track of what it
is chasing — which is exactly why it fails against a target that is sitting
still.

The asymmetry was **already measured in this codebase** and recorded as an
unintended side effect, in the comment that explains why the region path is
normalised: *"with weights summing to 1 the coverage term is a convex
combination (about one message's worth) while bb/rb/ob are raw sums of 3-4
messages each, so at initialisation it is only ~9% of the aggregate's norm"*.
Nobody connected it to `comms_radius`.

This also explains §16's ablation better than §16 did.  That section
attributed the 2.72 -> 1.26 collapse to magnitude saturating the tanh
activations.  The likelier cause is rb being drowned out — same evidence, a
mechanism that also predicts the stationary-red failure, which saturation does
not.

#### 19.3 What this does NOT establish

* **None of §18.4's four pre-registered readings matches.**  They were: B
  wins and reaches the reference; B ≈ A below it; B ≈ A ≈ reference; B < A.
  What happened is B ≫ A but **0.27 below the reference** (4.2 SE).  Recorded
  as a miss rather than retrofitted.
* **Attention did not reach the reference**, and that gap is confounded:
  attention was still rising at the end (eval blocks 2.10 -> 2.24 -> 2.39,
  final 2.48) with 17% fewer PPO steps, from a clone worth 1.86 against the
  reference chain's 2.88.  2.48 is a floor here, not a ceiling.
* **One seed per arm.**  A 10 SE gap in a final evaluation is not 10 SE across
  seeds.
* **Measured at `comms_radius inf` only.**  This may be "attention rescues
  unlimited comms" rather than "attention is better".  At comms 40 the
  imbalance is milder (53/22 against 68/15), so the benefit could be much
  smaller — untested.

#### 19.4 Normalisation or discrimination? The control arm

Attention does two separable things: it **normalises** (each type contributes
~1 message regardless of live count) and it **discriminates** (weights differ
per neighbour).  §19.2's mechanism is entirely about the first.  If that is
the whole story, a plain MEAN over live edges — no parameters, no selectivity
— should recover most of the 0.88.

So a third arm ran with `--mean-agg`, identical in every other respect.
**The answer is both, with normalisation about two thirds of it** — which is
neither of the two readings pre-registered above, the second
over-dichotomous pre-registration in a row (see also §18.4).

Red 1.5, 200 episodes, the same matched seeds:

| arm | evader | stationary | random | steps |
|---|---|---|---|---|
| sum | 1.60 ± 0.07 | 2.62 ± 0.04 | 2.68 ± 0.04 | 189.0 |
| **mean** | **2.18 ± 0.06** | **2.94 ± 0.03** | 2.96 ± 0.01 | 172.7 |
| attention | 2.48 ± 0.05 | 2.98 ± 0.01 | 3.00 ± 0.01 | 156.9 |

| | red 1.4 (50 eps) | red 1.5 (200 eps) |
|---|---|---|
| **normalisation** (mean − sum) | +0.48, 3.2 SE | **+0.58, 6.3 SE** |
| **discrimination** (attention − mean) | +0.30, 2.3 SE | **+0.30, 3.8 SE** |
| share | 62% / 38% | 66% / 34% |

Three things this settles:

* **The §19.2 mechanism is confirmed.**  A single division takes the
  stationary red from 2.62 to **2.94**, against attention's 2.98.  What broke
  the policy was the between-type imbalance, not saturation — and the fix
  needs no parameters.
* **Discrimination is real**, not marginal: +0.30 at 3.8 SE on 200 episodes.
  It carries one confound — the mean's clone started 0.16 below attention's
  (1.70 against 1.86), so the architecture-attributable part is somewhere
  between ~0.14 and 0.30.
* **It does not grow with difficulty.**  +0.30 at red 1.4 and +0.30 at red
  1.5, identical.  A provisional reading from the mean arm's mid-training
  evals predicted the opposite (that discrimination's share would rise at the
  harder speed); it did not, and the arm kept climbing after that reading was
  taken.

So the engineering conclusion for the current three-red configuration is that
the **mean is the better trade** — a conclusion §21.4 then shows does NOT
generalise, because at five targets with a varying count the mean retains
65% of its single-target performance against attention's 90%: two thirds of the benefit for zero
parameters and zero wall clock, against attention's +8.6% parameters and 1.25x
time.  Attention earns its place on what it unlocks rather than on this
measurement — variable entity counts and heterogeneous speeds, where
selectivity is the whole point and the summed aggregation is a hard blocker.
§20 is the experiment that tests that.

Implementation note: a mean IS attention with uniform weights, so it is
implemented by passing a single channel of zero scores through the same
`_attend` path.  That is not a shortcut for its own sake — it means the mean
inherits the mask-before-softmax ordering and the empty-row guard rather than
reimplementing the two things that were hard to get right.  A test pins that
the mean adds exactly zero parameters, which is what makes it a clean control.

---

### 20. Does attention's advantage grow with the target count? (DESIGN, 2026-10-01) · RUN, see §21

> **Outcome in §21: the hypothesis holds.**  The gap grows monotonically
> with the count, 0.010 at one target to 0.256 at five, which is the first
> of the three readings pre-registered in §20.6.  The design below is what
> ran; §21.3 records the one way the arms were not level.

§19 showed attention beats the summed aggregation by +0.88 on the evader, and
that **most of it is normalisation**: a plain mean over live edges recovers
+0.48 of that at red 1.4, leaving +0.30 for discrimination (§19.4, and at 2.3
SE on 50 episodes with part of it inherited from a clone that started 0.16
lower).  So the question attention was actually built for is still open: the
operational claim is that a blue should decide, from the targets' positions,
count and estimated velocities, whether it needs one ally on a target or
three — and that is **selectivity**, which only pays when there is something
to select between.

This section is the experiment that tests it, designed before running and with
every parameter decided by a cheap measurement rather than by argument.  Four
design intuitions were tested and **three of them were wrong**, which is the
reason for the measurements.

#### 20.1 The hypothesis, stated so it can fail

Not "attention scores higher" — §19 already established that.  The claim is
that **attention degrades more slowly than a mean as the number of live
targets varies**, because a mean normalises the magnitude while attention can
additionally reorder which neighbours matter.

The comparison is therefore **mean against attention**.  The summed
aggregation is excluded: §19 established it is broken at `comms_radius inf`
(2.62/3 against a *stationary* red), so including it would re-measure a known
breakage and crowd out the question.

#### 20.2 The variable the hypothesis is about is TRACKS, not reds

The actor's "red" nodes are the fixed `tracker_red_slots` (8), not the active
reds, so the actor graph is identical at every count — which is what makes one
policy evaluable across a sweep at all (pinned by
`tests/test_variable_counts.py`).  What varies is how many slots hold a
confirmed track, and raising `n_red` widens that distribution substantially.
Measured with a policy driving:

| `n_red` | live tracks/blue, mean | **sd** | max | spread over 0..5 |
|---|---|---|---|---|
| 1 | 0.80 | **0.50** | 2 | 24 / 72 / 4 / 0 / 0 / 0 % |
| 3 | 1.20 | **0.94** | 5 | 22 / 47 / 20 / 8 / 2 / 0 % |
| 5 | 2.34 | **1.51** | 7 | 13 / 18 / 24 / 23 / 13 / 7 % |

The standard deviation **triples**.  That is the quantity an unnormalised sum
conflates with magnitude and a mean or attention neutralises, so the
experiment has signal to find.

#### 20.3 Four design questions, decided by measurement

**Tracker slots: 8, unchanged.**  The concern was that 5 reds would saturate
the 8 slots and degrade the input before the aggregation ever sees it,
compressing the effect.  Measured: occupancy is *identical* at 8, 12 and 16
slots (mean 2.38, max 7), and the limit is hit in **0.0%** of steps.  No
saturation, so no architecture change — which also keeps A2 comparable with
§19's red-1.4 arms.

**Team size: 5 blues against 5 reds.**  Not presentation: with surplus blues
allocation has slack, and without them it binds.  The value of coordination —
`AssignGreedy` minus `ObsGreedy`, allocation by construction against none —
**triples**:

| | uncoordinated | coordinated | advantage |
|---|---|---|---|
| 3v3, red 1.4 | 2.18 | 2.32 | +0.14 |
| **5v5, red 1.4** | 3.02 | 3.45 | **+0.43** |

**`max_steps`: 200, unchanged** — and this one inverted the expectation.  The
heuristics spend 188-197 of 200 steps at 5v5, so the metric looked
time-bound, and raising the budget looked necessary to measure allocation
rather than a race.  Measured, the coordination advantage **shrinks** as the
budget grows:

| `max_steps` | ObsGreedy | AssignGreedy | advantage |
|---|---|---|---|
| **200** | 3.02 | 3.45 | **+0.43** |
| 300 | 3.92 | 4.33 | +0.40 |
| 400 | 4.38 | 4.70 | +0.33 |

Given enough time, uncoordinated pursuit catches everything anyway
(`ObsGreedy` reaches 4.38/5 at 400 steps).  **Time pressure is what makes
allocation matter**, so the budget stays where coordination is worth most —
which also preserves comparability with every other result here.

**Red speed: 1.4.**  A ceiling worry — attention scores 1.000 at `n_red` 1
and 2 there — turned out not to bite, because a ceiling only kills the
comparison if BOTH arms sit at it, and even the broken summed arm reads 0.867
at `n_red` 1.  The deciding reason is convergence: §19's red-1.4 arms were
flattening (sum 1.66 -> 1.99 -> 2.05) while the red-1.5 arms were still
climbing at the end (attention 2.10 -> 2.24 -> 2.39, finishing 2.48).
Comparing degradation between two *unconverged* policies measures where each
happened to be when the budget ran out.

#### 20.4 The metric: not a slope

The natural summary — fit a line to captured fraction against count and
compare slopes — was tried on §19's existing arms and **does not work**.  The
curves are not linear, so the fit is fragile: the summed arm at red 1.5 reads
0.900, 0.483, 0.444, 0.450, 0.553, falling then rising, and a linear slope on
that carries an SE of 0.056 and flips sign between red speeds.

So the primary metric is the **captured fraction at each count, compared
point-by-point on matched seeds**, with a secondary robust summary: the
fraction at `n_red` 5 relative to `n_red` 1.  `scripts/eval_red_count_sweep.py`
reports both (it still prints a slope, with its SE, as a diagnostic — not as
the verdict).

Fraction rather than raw captures, because raw captures rise with the count
simply because there is more to catch: a flat raw line already means
degradation.

#### 20.5 What already exists, and what it does and does not say

The zero-shot curves of §19's arms are measured — both trained at `n_red` 3
fixed, evaluated at 1..5.  Attention dominates the sum at every count, at both
speeds.  But the gap **narrows** at high counts rather than widening (red 1.4:
0.133, 0.317, 0.334, 0.275, 0.254 across counts 1-5), plausibly because with
many targets the task is dominated by stumbling onto the nearest one, which
needs no allocation.

That is **weak evidence about this hypothesis, in both directions**, for two
reasons.  It compares the wrong pair — sum against attention is dominated by
§19's normalisation breakage, not by selectivity.  And a policy trained at a
fixed count never had a reason to develop selectivity, so zero-shot
generalisation is not what the claim is about.

#### 20.6 The run

| | |
|---|---|
| arms | **mean** (`--mean-agg`) against **attention** (`--attention --n-heads 4`) |
| blues / reds | 5 / 5, capacity |
| `n_red_min` | **1** — samples 1-5 active per episode *during training* |
| red `v_max` | 1.4 |
| `comms_radius` | inf |
| `tracker_red_slots` / `max_steps` | 8 / 200 |
| stages | clone (60 rounds) then one PPO stage, 150 rollouts |
| evaluation | `n_red` fixed at 1,2,3,4,5, 100 episodes each, matched seeds |
| baselines | `ObsGreedy` 3.02 ± 0.17, `AssignGreedy` 3.45 ± 0.18 (measured) |
| cost | ~11 h per arm, ~22 h |

Training samples the count and evaluation fixes it: the policy must learn to
handle variability, and the measurement isolates each level.

Pre-registered readings, **three** this time rather than a binary pair,
because the last two pre-registrations were both too dichotomous for results
that came out mixed (§18.4, §19.4):

* **Attention's advantage grows with count** — the hypothesis holds, and the
  case for selectivity (and for extending it to the region path) is made.
* **Advantage flat in count** — attention is a better aggregation but not for
  the stated reason; the honest claim becomes "normalisation plus a constant
  offset", and the selectivity argument needs a different test.
* **Advantage narrows with count**, as the zero-shot curves hint — then
  coordination genuinely matters less when targets are dense, and the
  interesting regime for this architecture is *few* targets under time
  pressure, not many.  That would redirect the roadmap rather than close it.

---

### 21. Attention's advantage grows with the target count (MEASURED, 2026-10-03) · `feature/target-tracking`

§20's hypothesis, confirmed.  Two arms differing **only** in the aggregation —
a mean over live edges against learned attention — each trained from scratch
at 5 blues against 5 reds with the active count **sampled 1-5 per episode**,
then evaluated at each count held fixed.  Produced by
`scripts/run_count_experiment.sh`; sweeps by
`scripts/eval_red_count_sweep.py`.

Captured fraction, 100 episodes per point, matched seeds, red 1.4,
`comms_radius inf`:

| `n_red` | `ObsGreedy` | `AssignGreedy` | mean | **attention** | gap |
|---|---|---|---|---|---|
| 1 | 0.890 | 0.900 | 0.980 | **0.990** | +0.010 |
| 2 | 0.760 | 0.820 | 0.860 | **0.965** | +0.105 |
| 3 | 0.713 | 0.773 | 0.743 | **0.943** | +0.200 |
| 4 | 0.627 | 0.750 | 0.680 | **0.917** | +0.237 |
| 5 | 0.592 | 0.706 | 0.640 | **0.896** | **+0.256** |
| fitted slope | −0.0728 | −0.0458 | −0.0860 | **−0.0235** | |

**The gap grows monotonically with the count**, from 0.010 at one target to
0.256 at five.  That is §20.6's first pre-registered reading, and the first
pre-registration in this series to land cleanly rather than between the
options offered.

Three summaries:

* **Retention** (fraction at `n_red` 5 over `n_red` 1): attention **90.5%**,
  `AssignGreedy` 78.4%, `ObsGreedy` 66.5%, mean **65.3%**.
* **Absolute captures at `n_red` 5**: 4.48 ± 0.08 against the mean's
  3.20 ± 0.12 — **+1.28, about 8.9 SE**.
* **Time**: 161.6 steps at five targets against the mean's 195.9, which is
  nearly the 200-step limit.  The mean arm runs out of clock; attention does
  not.

#### 21.1 It beats the hand-coded allocator, which the mean does not

`AssignGreedy` does explicit target division with balanced capacity — the
allocation this architecture is supposed to learn.  Two things stand out.

**The mean arm loses to it at every count above two**: 0.743 against 0.773 at
three targets, 0.640 against 0.706 at five (3.20 against 3.53 captures, about
2 SE).  A learned policy beaten by a hand-written rule, at precisely the
counts where allocation matters.  It wins at one and two targets (0.980
against 0.900), so the curves cross — which is invisible at any single
operating point, and is why §20.4 made the curve the metric.

**Attention beats it everywhere and degrades half as fast** (−0.0235 against
−0.0458).  It learned the allocation the rule encodes, and kept learning past
it.

#### 21.2 This reverses what the zero-shot curves suggested

§20.5 measured the same sweep on §19's arms — trained at `n_red` 3 **fixed** —
and found the gap *narrowing* with count (0.133, 0.317, 0.334, 0.275, 0.254).
That was reported as weak evidence against this hypothesis.  It was weak for
the reason recorded there, and the correction is now measured: a policy
trained at a fixed count never had a reason to develop selectivity, so its
zero-shot behaviour at other counts says nothing about what the mechanism is
worth when training exercises it.  Training both arms with the count varying
was the decisive design choice, and it came from the user's insistence on it
over this document's reading of the zero-shot result.

#### 21.3 Caveats, and the second one matters

* **One seed per arm**, as everywhere in §15-§21.
* **The clones did NOT start level.**  On the evader: attention 2.10 against
  the mean's 1.25, a 0.85 head start — far larger than §19's 0.16.  So the
  comparison of LEVEL is not clean.
  The comparison of SHAPE survives it better: a uniform head start shifts a
  curve up without changing its slope, and the slope differs by more than a
  factor of two.  And the clone gap is itself evidence rather than only a
  confound — cloning is supervised fitting with no RL involved, so attention
  fitting the same expert better at a varying count is the mechanism showing
  up in the simplest possible setting.
* **The fitted slope's SE (0.0004) is understated.**  It is a residual-based
  fit SE over five points, so it measures how straight the line is, not the
  sampling uncertainty of the points it is fitted to (0.010-0.016 each).  The
  robust statistics are the point comparison at `n_red` 5 (8.9 SE) and the
  retention ratio; the slope is a shape summary, not a significance test.
* `n_obstacles 0`, so the `ob` pathway remains inert and its own attention is
  built but unexercised.

#### 21.4 What this settles, and what it opens

The engineering conclusion of §19.4 — that the mean was the better trade for
three reds, at two thirds of the benefit for zero cost — **does not
generalise**.  At five targets with a varying count, the mean retains 65% of
its single-target performance and attention retains 90%.  The +8.6%
parameters and 1.25x wall clock buy something that a division cannot.

Open, and now with measured rather than theoretical justification:

* **Attention on the region path** (§18.5).  `gb_weight` is a hand-designed
  normalised weighting sitting at exactly the site attention weights occupy,
  and the region count `R*R` is a far larger and more variable fan-in than the
  eight tracker slots — so if selectivity pays anywhere else, it pays there.
* **Heterogeneous red speeds**, the other half of the original motivation: a
  blue weighting the fast evader over the slow one is the same mechanism
  applied to a different feature.
