# Multi-UAV ISR — Adversarial Multi-Agent RL in POMDPs

> **🔬 Research project — under active development.**
> This repository contains an **ongoing personal research project** exploring
> multi-agent reinforcement learning for cooperative UAV search and tracking in
> partially observable environments. It is an **experimental platform** for
> evaluating different perception, communication, and coordination
> architectures — APIs, results, and structure may change as the work
> progresses.

A research-oriented Multi-Agent Reinforcement Learning project focused on the
**core problem of autonomous-systems engineering**: a team of agents (UAVs)
performing **Intelligence, Surveillance, and Reconnaissance** in an
adversarial, partially observable environment.

The setup is intentionally generic — the same mathematical problem appears in
**search-and-rescue**, **wildlife anti-poaching**, **disaster response**, and
**ISR for defense**. Code and concepts transfer cleanly between framings.

## Why this project

Multi-agent RL most papers train on are either:

- **Fully observable, cooperative** (overcooked, MPE simple-spread): tractable
  but doesn't exercise the hard parts.
- **Single-agent POMDP** (Atari with frame stack): partial observability
  without multi-agent coordination.
- **Production-scale benchmarks** (StarCraft II): genuinely interesting but
  require massive compute and a lot of domain wrapping.

This project hits the **sweet spot in the middle**: a small enough setup to
train on a single GPU, large enough to surface every concept you need for
real autonomous-systems work:

- **Partial observability (POMDP)** — sensors have limited range and noise.
- **Self-play and non-stationarity** — both teams learn; the equilibrium is
  the interesting object.
- **Limited communication** — comms are range-gated; teams must coordinate
  with sparse information.
- **Dynamic objectives** — targets appear, move, and re-prioritise during an
  episode.
- **Belief-state estimation** — agents must maintain a belief over unseen
  parts of the world.
- **Adversarial robustness** — evaluated against held-out opponents, not
  just the policies seen during training.

## The eight-stage curriculum

The project is structured as a **learning curriculum**: each stage adds one
concept on top of the last, with measurable acceptance criteria.  See
[`docs/design.md`](docs/design.md) for full detail.

| Stage | Concept added | Status |
|------:|---|---|
| 1 | Pursuit-evasion baseline (continuous 2D, fully observable, fixed red, MLP shared policy) | **soft pass** (v2: +15.37 vs Run, Greedy = +16.15; 2.00/2 caught on every red — see [`docs/stage1_analysis.md`](docs/stage1_analysis.md)) |
| 2 | **Structured architecture — GNN over entities + CTDE critic** (same env, MLP → GNN as sole variable) | **PASS** (GNN 15.7% faster than Greedy on `mean_steps` vs Run; MLP at scale collapses; see [`docs/stage2_results.md`](docs/stage2_results.md)) |
| 3 | **Partial observability** — sensor radius + GRU on top of the Stage 2 GNN | **PASS** (+18.16 vs Run; bar +16.47; margin +1.69 — see [`docs/stage3_results.md`](docs/stage3_results.md)) |
| 4 | **Sensor noise + occlusion** — Bayesian belief map (log-odds occupancy), obstacles, line-of-sight occlusion, typed-GNN perception front-end | **landed** (belief-driven policy matches the fully-observable oracle at 3/3 catches — see [`docs/stage4_results.md`](docs/stage4_results.md)) |
| 5 | Self-play — red team becomes learned; fictitious self-play, league training | **next (roadmap capstone)** |
| 6 | Communication — range-gated message passing between teammates | |
| 7 | Dynamic objectives — moving / appearing / re-prioritised targets | |
| 8 | Adversarial robustness evaluation — exploitability metrics, held-out opponents | |

Curriculum resequenced from 7 stages to 8 in July 2026 to isolate
architecture as a standalone Stage 2 variable *before* introducing
partial observability (previously Stage 2, now Stage 3).  The change
was motivated by the Stage 1 v2 soft-pass verdict — see
[`docs/stage1_analysis.md §6b`](docs/stage1_analysis.md) for the
rationale.

Each stage produces a self-contained mini-result and a reusable training
artifact.

### Stage 4 post-close extensions

Four capabilities shipped on top of the Stage 4 baseline, each on its own
branch, each with tests.  See [`docs/stage4_backlog.md`](docs/stage4_backlog.md)
for the full annotated record.

- **Crash avoidance** — per-agent reward decomposition `r_i = r_team + r_crash_i`
  for blue↔obstacle and blue↔blue collisions, an agent-conditioned critic
  `V(s, i)`, and per-agent GAE. *Measured:* capture held ~2.95/3 while crashes
  fell ~40 → ~3–5 per episode.
- **Variable entity counts** — `n_red` / `n_obstacles` become a padded
  **capacity** with a per-episode active count sampled from `[*_min, capacity]`.
  The critic's global context switched from a count-hard-coded flatten to a
  **masked-mean pool + count scalar**, making every learned tensor
  count-agnostic → one policy generalises across (and beyond) trained counts.
  *Implemented + tested; training in progress.*
- **Moving obstacles** — reciprocating patrol (wall-to-wall bounce) with
  obstacle velocity flowing into the GNN edge features and an optional
  `obstacle_belief_decay` to fade a moving obstacle's stale belief trail.
  *Implemented + tested; training in progress.*
- **Count-generalisation eval** — `scripts/eval_stage4_counts.py` sweeps
  `n_blue` / `n_red` / `n_obstacles` on a single checkpoint to measure
  zero-shot generalisation. See
  [`docs/stage4_generalization_eval.md`](docs/stage4_generalization_eval.md).

## Architecture

```
Multi-UAV-ISR/
├── isr/                            # main Python package
│   ├── env/
│   │   ├── entities.py             # UAV / Target dataclasses + kinematics
│   │   └── pursuit_env.py          # PettingZoo ParallelEnv (Stages 1–4:
│   │                               #   sensors, belief maps, obstacles,
│   │                               #   occlusion, crash model, moving obstacles)
│   ├── agents/
│   │   ├── heuristics.py           # random / greedy / scripted red & blue baselines
│   │   ├── gnn_policy.py           # Stage 2–3 typed-GNN policy (+ GRU)
│   │   ├── gnn_stage4_policy.py    # Stage 4 typed-GNN + CTDE critic + belief front-end
│   │   └── policy_loader.py        # checkpoint → policy dispatch
│   ├── train/
│   │   ├── ppo.py                  # PPO-clip (shared + per-agent updates)
│   │   ├── graph_buffer.py         # graph rollout buffer (per-agent GAE)
│   │   ├── vec_env.py              # vectorised env wrapper
│   │   └── subproc_vec_env.py      # subprocess parallel envs
│   ├── utils/
│   │   └── render.py               # matplotlib episode renderer
│   └── configs/
│       ├── stage1_default.py       # Stage 1 hyperparameters
│       ├── stage3_default.py       # Stage 3 (partial observability)
│       └── stage4_default.py       # Stage 4 (belief / obstacles / crash / moving)
├── scripts/
│   ├── train_stage1.py             # Stage 1 (MLP) / Stage 2 (GNN) training entry
│   ├── train_stage4.py             # Stage 3–4 training entry (belief GNN)
│   ├── eval_stage4_counts.py       # entity-count generalisation sweep
│   ├── evaluate_trained.py         # evaluate a checkpoint → docs/stageN_results.md
│   ├── render_demo.py              # render one episode (random or trained)
│   └── diag_scripted_pursuit.py    # scripted-baseline diagnostics
├── tests/                          # env contract + per-extension test suites
└── docs/
    ├── design.md                   # full curriculum + stage-by-stage spec
    ├── stageN_design.md            # per-stage design specs
    ├── stageN_results.md           # per-stage acceptance results
    ├── stage4_backlog.md           # deferred items + annotated extension record
    └── stage4_generalization_eval.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate.bat         # Windows cmd
pip install -r requirements.txt
```

## Run

Random-policy episode (sanity check):
```bash
python scripts/render_demo.py
```

Render Greedy heuristic blue vs Run-from-nearest red:
```bash
python scripts/render_demo.py --blue greedy --red run --save out/demo.gif
```

Run the test suite:
```bash
pytest -q
```

## Training

### Stage 1 (MLP) / Stage 2 (GNN)

Both live in `scripts/train_stage1.py`, selected by `--policy-type`
(everything else — env config, red-policy mix, PPO hyperparameters —
is shared so the head-to-head stays fair):

```bash
# Stage 1 MLP baseline
python scripts/train_stage1.py --device cuda --n-rollouts 1000

# Stage 2 GNN (same script, one flag)
python scripts/train_stage1.py --device cuda --policy-type gnn --n-rollouts 1000
```

### Stage 3–4 (belief-map GNN)

`scripts/train_stage4.py` trains the typed-GNN + CTDE policy on the
partial-observability / belief-map task. A bare run reproduces the
proven recipe (2 message rounds, Stage-1 critic-encoder warm-start,
`aux_hidden_coef=0.2`):

```bash
python scripts/train_stage4.py --run-name stage4_baseline --n-rollouts 1000
```

Post-Stage-4 knobs (all off by default, so a bare run byte-preserves the
baseline):

```bash
# crash avoidance (per-agent penalties)
--crash-obstacle-penalty 2.0 --crash-blue-penalty 1.0 --blue-collision-radius 2.0

# variable entity counts (sample active count in [min, capacity] each episode)
--n-red-min 2 --n-obstacles-min 0

# moving obstacles (fraction that patrol + belief decay to fade stale trails)
--moving-obstacle-fraction 0.5 --obstacle-speed 1.0 --obstacle-belief-decay 0.9

# warm-start BOTH actor + critic from a converged Stage 4 checkpoint
--warm-start-full runs/stage4/<run>/best.pt
```

### Evaluate

```bash
# per-stage acceptance eval → writes docs/stageN_results.md
python scripts/evaluate_trained.py --checkpoint runs/stage1/<run>/best.pt --device cuda --n-episodes 50

# entity-count generalisation sweep (Stage 4)
python scripts/eval_stage4_counts.py --checkpoint runs/stage4/<run>/best.pt --n-red 2 4 6 8
```

Acceptance criteria per stage are documented in the corresponding
`docs/stageN_design.md` / `docs/stageN_results.md`.

## Training on GPU (RunPod or similar)

The belief-map GNN trains on a single small GPU (RTX 3060 / T4 class) in
well under an hour for 1000 rollouts; CPU works but is markedly slower.

```bash
git clone https://github.com/quialga/Multi-UAV-ISR.git
cd Multi-UAV-ISR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

TensorBoard scalars stream to `runs/stage{N}/<run>/tb/` — one directory
per run.

## Status

- **Stage 1 — soft pass** (July 2026): v2 MLP matches GreedyPursuer across the
  red distribution, catches 2.00/2 on every red type, misses the strict 1.20×
  bar by 4.01 points. See [`docs/stage1_analysis.md`](docs/stage1_analysis.md).
- **Stage 2 — PASS** (July 2026): at N=5 vs M=3 scaling, the GNN + CTDE-critic
  architecture catches all 3 reds every episode and runs 15.7% faster than
  GreedyPursuer on episode length vs `run_from_nearest_uav`. The MLP baseline
  structurally fails at the same scale. See
  [`docs/stage2_results.md`](docs/stage2_results.md).
- **Stage 3 — PASS** (July 2026): sensor-range partial observability + a GRU on
  the Stage 2 GNN clears the return bar (+18.16 vs Run, margin +1.69). See
  [`docs/stage3_results.md`](docs/stage3_results.md).
- **Stage 4 — landed** (July 2026): a Bayesian belief-map perception front-end
  (log-odds occupancy, obstacles, line-of-sight occlusion, sensor noise) feeds
  the unchanged Stage 3 typed-GNN CTDE policy. The belief-driven policy reaches
  the same 3/3 capture performance as the fully-observable oracle, paying only a
  small honest cost in convergence for acting under uncertainty. Four extensions
  (crash avoidance, variable entity counts, moving obstacles, count-generalisation
  eval) have shipped on top. See [`docs/stage4_results.md`](docs/stage4_results.md)
  and [`docs/stage4_backlog.md`](docs/stage4_backlog.md).
- **Stage 5 — next**: self-play (the red team becomes a learned evader) is the
  remaining roadmap capstone.

Per-stage result writeups land in `docs/stage{N}_results.md` as each stage
completes; broader analysis notes accumulate in `docs/` alongside them.

## License

Licensed under the **Apache License 2.0** — see the [`LICENSE`](LICENSE) file
for the full text. Apache 2.0 is permissive (use, modify, and redistribute
freely, including commercially) while adding an explicit patent grant and
requiring attribution and a statement of changes. If you build on this work, a
citation or link back is appreciated.
