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

- **Global fused belief map** `(2, 26, 26)` — ONE shared log-odds grid
  for the whole blue team (common operational picture over TDL;
  Bayesian fusion of independent sensors is log-odds addition).
  Channel 0 = P(enemy), channel 1 = P(obstacle). 5 m cells.
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

Both runs use the Stage 3 winning recipe (see *Training arguments*),
400 rollouts, `stationary:1,random:1,run:1` red mix.

| Actor input | Train caught (stochastic) | Reward | Convergence |
|---|---|---|---|
| **Oracle** (ground-truth graph) | 3/3 | ~22–23 | faster |
| **Belief** (noisy, detection-seeded) | 3/3 | ~20–21 | slower |

- Oracle recovers Stage 3's level (Stage 3 was 2.94/3 stochastic),
  confirming the CTDE setup is sound.
- The **belief–oracle gap is small** (~1–2 reward, slightly slower
  convergence) — the genuine, irreducible cost of acting on partial,
  noisy perception rather than ground truth. A small gap means the
  perception layer is doing its job.

> Record the deterministic `[det eval]` mean (and the `run=` /
> evader column) for both runs as the clean headline metric.

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

Reproduce:
```
python scripts/train_stage4.py --n-rollouts 400 --n-obstacles 0 \
  --red-policy-mix stationary:1,random:1,run:1 \
  --eval-interval 25 --run-name belief_v3
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

- **Obstacles end-to-end.** The obstacle graph path (nodes + ob edges +
  occlusion) is wired and unit-tested, but a full `--n-obstacles 4`
  training run against the belief path is not yet reported.
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

## Stage 4 — closed (no-obstacle regime).

The belief-driven policy matches the fully-observable oracle at 3/3
captures, at a small honest cost for acting under uncertainty. The
perception front-end (global Bayesian belief map → prediction step →
detection-seeded tracks → typed GNN) is validated; the model-based /
learning-based split the design argued for holds up empirically. The
remaining open items (full obstacle run, Phase B filter, noise sweep)
are extensions, not blockers.
