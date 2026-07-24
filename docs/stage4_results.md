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
- `scripts/diag_scripted_pursuit.py` — a hand-coded controller on the
  real obs (captures 3/3), proving the observation is *sufficient* to
  solve the task, so any policy shortfall is a learning problem.

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

### 3. Variable entity counts + count-agnostic critic (implemented + tested; training pending) · `feature/variable-entities`

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
"train on 2–4, evaluate zero-shot on 6" result once trained.

*Architecture note:* this pool change narrows `critic_trunk.0.weight`
(512 → 194 for `n_red=3,n_obs=4`), the **only** tensor that can't warm-
start from a pre-pool checkpoint. `load_full_stage4` transfers the other
**76/77** tensors (whole actor + both GNN encoders + value head) and now
**names** any tensor it leaves at fresh init.

### 4. Moving obstacles — reciprocating patrol (implemented + tested; training pending) · `feature/moving-obstacles`

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
