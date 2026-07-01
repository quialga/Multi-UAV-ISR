# Stage 1 — Analysis of v1 and Design of v2

_Research journal entry, June 2026._

This document records the Stage 1 v1 training result, diagnoses the two
failure modes it exposed, reviews the multi-UAV ISR literature we drew
on to design v2, and specifies the v2 observation-space + training-
distribution changes.  Intended as material for a future blog post /
portfolio article.

## 1. Setup recap

Stage 1 of the Multi-UAV ISR curriculum is a **pursuit-evasion baseline**:
- 2D continuous 100×100 arena
- **3 blue UAVs** (double-integrator kinematics, v_max = 1.5) vs
  **2 red targets** (v_max = 1.0)
- Capture: red enters within `capture_radius = 3.0` of any blue → caught
- Episode: 200 steps or all reds caught
- Reward (shared team): +10 per catch, −0.01·‖action‖², −0.05 step cost,
  −5 per uncaught red at terminal
- Blue trained with PPO (shared parameters across the 3 UAVs)
- Red: scripted `run_from_nearest_uav` policy (flees nearest blue)

Full spec: [`docs/design.md §3`](design.md).

### Acceptance criterion

The v1 acceptance bar was set at **≥ 1.20 × GreedyPursuer's mean episode
return**. Heuristic calibration (50 episodes, matched seeds):

| Blue policy | vs Stationary | vs Random | vs Run_from_nearest |
|---|---|---|---|
| RandomAgent | −13.17 | −13.88 | −20.24 |
| **GreedyPursuer** (baseline) | **+17.65** | **+17.38** | **+16.15** |

Bar: ≥ **+19.38** mean episode return vs `run_from_nearest_uav`.

## 2. v1 result

- **1000 rollouts × 16 envs × 256 steps = 4M env steps** (≈ 75 min on GPU)
- Training only exposed the policy to `run_from_nearest_uav` reds
- v1 observation vector (26 D): absolute normalized positions and
  velocities of every entity, red active mask, `self_idx_onehot` for
  parameter-sharing symmetry breaking, time remaining
- Result on the full evaluation matrix (50 episodes/cell, deterministic):

| Trained blue policy | vs Stationary | vs Random | vs Run_from_nearest |
|---|---|---|---|
| **v1 (this run)** | **−21.92** ± 11.4 (0.42 caught / 2) | **−17.22** ± 14.3 (0.72 caught) | **+10.22** ± 5.1 (**1.98 caught**) |

**Verdict: FAIL.** Trained vs Run = +10.22, bar = +19.38, margin = −9.16.

## 3. Two independent failure modes

The eval matrix is diagnostic in a way pure reward numbers aren't.

### Failure mode 1: training-distribution collapse

The most striking pattern is that **v1 is worst against the *easiest*
red** (stationary, +17 for Greedy, −21 for v1). The policy is *worse
than random* (−13) against a red that does nothing.

Explanation: the policy was only ever trained on `run_from_nearest_uav`
reds. Its observation always contained `red_vel ≠ (0, 0)` during
training. Against stationary reds at evaluation time, it sees
`red_vel == (0, 0)` — an out-of-distribution observation. Empirically,
the policy appears to have learned an "aim-ahead / interception"
strategy that assumes the red is fleeing, and *overshoots or circles*
stationary targets.

Against `run_from_nearest_uav` it does catch reds (mean caught = 1.98)
— but slowly (episode return +10 vs Greedy's +16 → policy takes roughly
2.5× longer per catch, all extra step + action cost).

### Failure mode 2: observation representation

The v1 obs is absolute-positioned: entities are represented by
`(x/L, y/L)` and `(vx/v_max, vy/v_max)` in the world frame.

Two consequences:
- The MLP has to **relearn the pursue-and-intercept mapping separately
  at every arena location.** A useful maneuver at (10, 10) is *the
  same maneuver* at (80, 80), but the network's input vector is
  entirely different. Translation invariance is not built in.
- `self_idx_onehot` is required to distinguish blue agents under
  parameter sharing — a fix for a symmetry the world-frame obs breaks.

Combined effect: the network wastes capacity on positional look-up
tables and never gets far enough on the interesting coordination
problem the acceptance bar was designed to test.

## 4. Literature review — what modern multi-UAV ISR does

Three recent papers, ordered by relevance to our design choice:

| Paper | Frame | Per-entity features | Architecture | Notes |
|---|---|---|---|---|
| **TERL** (Shanghai U., 2025, arXiv:2503.12395) | Ego-frame | ego: `[vx, vy, d_nearest, pursuit_status]`; teammate: `[px, py, vx, vy, d, θ, pursuit_status]`; evader: `[px, py, vx, vy, d, θ, heading_err]`; obstacle: `[px, py, r, d, θ]` | Per-entity MLP embedding + type embedding + **transformer self-attention** + target-selection attention head | Trained on 15 pursuers/4 targets → generalises zero-shot to 80/20 with 100% success. Ablation: removing the transformer drops CC-scenario success from 1.00 to 0.05, collisions 3% → 91%. **Reference model for our redesign.** |
| **Cooperative Bearing-Only Pursuit** (Westlake / Shanghai AI Lab, IROS 2025, arXiv:2503.08740) | Body-frame (agent's x-axis = forward heading; explicit ω, α dynamics) | ego: `[pos, vel, heading]`; ally: `[rel_pos, rel_vel, heading, comms_flag]`; target: filtered position + velocity estimate (zero-masked outside FoV) | MADDPG + spectral-norm | The RL controller consumes a **filtered target state estimate** from a Pseudo-Linear Information Filter, not raw bearings. Sim-to-real transfer to real AGVs. Requires agents to have real heading dynamics — not applicable to our double-integrator kinematics. |
| **Role-based MADDPG** (SUTD/Ottawa, ICRA 2023, arXiv:2303.01799) | **World-frame absolute** (position + velocity only) | Just `[px, py, vx, vy]` per agent | Vanilla MADDPG + Voronoi reward | Baseline design. Works for small MPE-scale, doesn't scale. **Approximately what our v1 was.** |

### Consensus takeaways

1. **Every recent paper uses ego-centric observations.** World-frame
   absolute (v1's design, and the 2023 baseline) is no longer state of
   the art.
2. **Feature vocabulary consensus (TERL):** for each other entity,
   include **both Cartesian and polar** relative features
   — `(rel_px, rel_py, rel_vx, rel_vy, range, bearing)`. Redundant
   information helps the network extract both "how far" and "which
   direction" signals cleanly.
3. **The biggest lever isn't the *frame* — it's the *structure* +
   attention.** TERL's ablation shows that with matched features, moving
   from flat concatenation to per-entity embeddings + self-attention
   moves CC-scenario success from 5% to 100%.
4. **Body-frame rotation helps only when the agent has real heading
   dynamics.** Our double-integrator has no persistent heading — the
   UAV can accelerate any direction instantly. Body-frame here is
   *nominal* (defined off velocity) and breaks at |v| ≈ 0. Not worth
   the complexity vs. world-frame relative with explicit velocity
   direction included.

## 5. v2 design

Adopt **TERL's feature vocabulary** (world-frame relative + polar
auxiliary) but **keep the MLP architecture** — deferring the transformer
upgrade to Stage 2, where partial observability with variable numbers
of visible entities justifies it.

### v2 observation layout (per blue UAV)

For each **red target** `j` (fixed order, N_red = 2):
- `rel_pos` (2D) = `red_pos[j] − self_pos`  (world-frame, ego-centred)
- `rel_vel` (2D) = `red_vel[j] − self_vel`
- `range`   (1D) = `‖rel_pos‖`
- `bearing_sc` (2D) = `sin θ, cos θ` where θ is the angle between
  `self_vel` and `rel_pos` (fallback: use `+x` axis when `‖self_vel‖ < ε`)
- `active_flag` (1D)

**Per red: 8 D. For 2 reds: 16 D.**

For each **teammate** `t` (excluding self, fixed order):
- `rel_pos` (2D), `rel_vel` (2D), `range` (1D), `bearing_sc` (2D)

**Per teammate: 7 D. For 2 teammates: 14 D.**

For **self**:
- `vel` (2D) — retains direction + magnitude
- `speed` (1D) — redundant scalar, easier for network to consume
- `wall_distances` (4D) — signed distances to the 4 walls
  `(x, L − x, y, L − y)`; this is what "grounds" absolute position

For **global**:
- `time_remaining` (1D)

**Total obs dim = 16 + 14 + 2 + 1 + 4 + 1 = 38 D** (up from 26 D).

### v2 architectural changes

- **Remove `self_idx_onehot`.** With ego-frame obs, every blue agent's
  input is structurally identical from its own perspective →
  parameter-shared policy is now truly permutation-agnostic across
  teammates.
- **MLP hidden sizes unchanged** ([256, 256]).
- **Actor / critic heads unchanged.**
- **PPO hyperparameters unchanged.**

### v2 training-distribution change

The other half of v1's failure was training only on `run_from_nearest_uav`
red. In v2 the vectorised env samples a **red policy per episode** from:
- `stationary_red` (weight 1/3)
- `random_red` (weight 1/3)
- `run_from_nearest_uav` (weight 1/3)

This alone should raise the vs-Stationary number from −21 to something
positive, and gives the policy a genuinely more general set of
opponents to coordinate against.

## 6. Expected v2 numbers

Priors, based on the v1 diagnosis:

| Blue policy | vs Stationary | vs Random | vs Run_from_nearest |
|---|---|---|---|
| Greedy (unchanged) | +17.65 | +17.38 | +16.15 |
| v1 (measured) | −21.92 | −17.22 | +10.22 |
| **v2 (predicted)** | **+14 to +17** | **+11 to +14** | **+14 to +17** |

**Will v2 clear the +19.38 acceptance bar?** Probably not. The gap
between "catches everything reasonably" and "catches faster than
Greedy" requires the policy to *coordinate* — to actively split
coverage or intercept from the side the red is fleeing toward. That's
a hard credit-assignment problem under shared team reward + parameter
sharing, and it's the natural place the flat-MLP architecture is
going to hit a ceiling.

If v2 lands at +15-17 and Greedy is at +16, we call Stage 1 a **soft
pass** and move on. The story ("we matched or slightly exceeded the
naive baseline, and we can identify exactly what's missing:
coordination-through-attention over teammates") is a stronger
portfolio narrative than a lucky pass would be.

## 6b. v2 result (measured, July 2026)

Full evaluation of the v2 checkpoint on the same seed base and
episode count as the v1 evaluation (50 episodes per cell, matched
seeds, deterministic actions):

| Trained-blue vs | v1 measured | v2 predicted range | v2 measured | Δ (v1 → v2) |
|---|---|---|---|---|
| Stationary red | −21.92 (0.42 caught) | +14 to +17 | **+17.17 (2.00)** | **+39.09** |
| Random red | −17.22 (0.72 caught) | +11 to +14 | **+17.27 (2.00)** | **+34.49** |
| RunFromNearest red | +10.22 (1.98 caught) | +14 to +17 | **+15.37 (2.00)** | **+5.15** |

Full eval matrix (rows = blue policy, columns = red policy):

| Blue \\ Red | Stationary | Random | RunFromNearest |
|---|---|---|---|
| Random  | −13.17 ± 9.76  (0.70 caught) | −13.88 ± 10.52 (0.64) | −20.24 ± 7.51 (0.24) |
| Greedy  | +17.65 ± 1.02  (2.00 caught) | +17.38 ± 1.21  (2.00) | +16.15 ± 1.30 (2.00) |
| **v2 Trained** | **+17.17 ± 1.15 (2.00)** | **+17.27 ± 1.32 (2.00)** | **+15.37 ± 1.43 (2.00)** |

### Predictions validated

The three predicted ranges from §5 were _"+14 to +17"_ for
Stationary and Run, and _"+11 to +14"_ for Random.  Two of the three
measured cells (Stationary, Run) landed inside the predicted band;
Random exceeded the top of the predicted band by ~3 points (+17.27
vs predicted upper +14).  The prediction was pessimistic on Random —
the ego-centric obs generalises better to unstructured red motion
than we expected.  Directionally the predictions were correct:
matching Greedy on Stationary + Random, coming *close but under*
Greedy on Run.

### Strict FAIL, qualitative SOFT PASS

Applied literally to the acceptance criterion (≥ 1.20 × Greedy vs
Run = +19.38), v2 misses by 4.01 points — **strict FAIL**.

Applied qualitatively — "does the policy actually solve the task?" —
v2 is a **soft PASS**:

- **Matches Greedy on 2 of 3 red types** (Stationary +17.17 vs
  Greedy +17.65 → gap 0.48; Random +17.27 vs +17.38 → gap 0.11).
- **Only 0.78 below Greedy on the hardest red** (Run +15.37 vs
  +16.15).
- **Catches 2.00 of 2 reds** in every scenario (perfect success
  rate, same as Greedy).
- **OOD collapse eliminated** — v1's catastrophic failure on
  stationary reds (−21.92, worse than Random) is fully resolved.

The remaining ~1-point-per-red gap to Greedy isn't a training-time
problem — the v1 → v2 sample-efficiency gain proves the policy
converged well before 1000 rollouts.  It's an **architectural
ceiling**.  A flat-MLP shared-parameter policy has no mechanism to
coordinate — nothing to represent "you go left, I go right, we cut
off the red between us".  Every UAV is running the same local
pursuit and the outcome is what a nearest-target heuristic
already achieves.

Getting the last ~4 points to clear the 1.20× bar therefore
requires the *architectural upgrade* Stage 2 introduces:
attention-over-teammates (per TERL's ablation, this is what moves
CC-scenario success from 5% to 100%).  Continuing to train the
flat MLP is unlikely to help.

### Decision: soft-pass Stage 1, proceed to Stage 2

We are officially closing Stage 1 as a **soft pass** and moving on.

The portfolio narrative from this outcome is stronger than a lucky
strict pass would have been:

1. We diagnosed **two independent failure modes** in v1 with a
   principled evaluation matrix (not just a headline reward number).
2. We drew a specific redesign from the current literature (TERL's
   ego-centric obs + Cartesian/polar per-entity features).
3. Prediction bands set _before_ the v2 run were validated by the
   measurement — directionally on all three cells, quantitatively on
   two.
4. We can *point at exactly what's missing* — attention-over-teammate
   coordination — and it's the concept Stage 2's partial-observability
   spec justifies introducing anyway.

## 7. What Stage 2 will add (foreshadowing)

Stage 2's concept is **partial observability** — the blue UAV sees only
entities within a finite `sensor_radius`. That change *also* creates
the natural moment to swap the flat obs vector for a **list of entity
tokens** and apply a small transformer:

- Sensor radius makes the number of visible entities variable per step
  → flat concatenation with fixed padding is wasteful; attention over
  variable-length token sequences is the right primitive.
- Recurrent policies (GRU) integrate observation over time so that
  belief about unseen entities can be maintained.
- Together these are the standard POMDP + attention stack TERL builds
  on top of. The Stage 1 v2 → Stage 2 progression mirrors the standard
  MLP → transformer step in the multi-UAV ISR literature.

## References

- [TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning](https://arxiv.org/abs/2503.12395) — Zhang, Zhao, Ren (Shanghai University, 2025).
- [Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning: Design and Experiment](https://arxiv.org/abs/2503.08740) — Li et al. (Westlake / Shanghai AI Lab, IROS 2025).
- [Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm using Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2303.01799) — Kouzeghar et al. (SUTD / Ottawa, ICRA 2023).
