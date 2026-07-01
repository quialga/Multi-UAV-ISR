# Multi-UAV ISR — Adversarial Multi-Agent RL in POMDPs

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

## The seven-stage curriculum

The project is structured as a **learning curriculum**: each stage adds one
concept on top of the last, with measurable acceptance criteria.  See
[`docs/design.md`](docs/design.md) for full detail.

| Stage | Concept added | Status |
|------:|---|---|
| 1 | Pursuit-evasion baseline (continuous 2D, fully observable, fixed red, MLP shared policy) | **soft pass** (v2: +15.37 vs Run, Greedy = +16.15; strict bar +19.38 not cleared but 2.00/2 caught on every red — see [`docs/stage1_analysis.md`](docs/stage1_analysis.md)) |
| 2 | **Structured architecture — GNN over entities + CTDE critic** (same env, MLP → GNN as sole variable; falsification test of the Stage 1 "architectural ceiling" verdict) | **next** — spec: [`docs/stage2_gnn_design.md`](docs/stage2_gnn_design.md) |
| 3 | Partial observability — sensor radius, GRU on top of the Stage 2 GNN | |
| 4 | Sensor noise + occlusion — explicit belief-state predictor (aux loss) | |
| 5 | Self-play — red team becomes learned; fictitious self-play, league training | |
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

## Architecture

```
Multi-UAV-ISR/
├── isr/                        # main Python package
│   ├── env/
│   │   ├── entities.py         # UAV, Target dataclasses + kinematics
│   │   ├── pursuit_env.py      # PettingZoo ParallelEnv (Stage 1)
│   │   └── spaces.py           # observation / action space builders
│   ├── agents/
│   │   ├── heuristics.py       # random / greedy / scripted baselines
│   │   └── ppo_policy.py       # neural policy (MLP, later GRU)
│   ├── train/
│   │   ├── ppo.py              # PPO clip implementation
│   │   ├── buffer.py           # rollout buffer
│   │   └── normalizer.py       # running obs normalizer
│   ├── utils/
│   │   └── render.py           # matplotlib episode renderer
│   └── configs/
│       └── stage1_default.py   # Stage 1 hyperparameters
├── scripts/
│   ├── train_stage1.py         # main training entry point
│   └── render_demo.py          # render one episode (random or trained)
├── tests/
│   └── test_env_smoke.py       # env API contract test
└── docs/
    ├── design.md               # full curriculum + stage-by-stage spec
    └── stageN_results.md       # per-stage acceptance results (added as stages land)
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

Train Stage 1 on CPU:
```bash
python scripts/train_stage1.py
```

Evaluate a trained checkpoint (writes `docs/stage1_results.md`):
```bash
python scripts/evaluate_trained.py --checkpoint runs/stage1/<run>/best.pt
```

## Training on GPU (RunPod or similar)

Stage 1 trains on CPU in ~2.5h but is faster on a single small GPU
(RTX 3060 / T4 class) at ~30-45 min.

### Setup on a fresh pod

```bash
git clone https://github.com/quialga/Multi-UAV-ISR.git
cd Multi-UAV-ISR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Verify CUDA is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Train — Stage 1 (MLP)

```bash
python scripts/train_stage1.py --device cuda --n-rollouts 1000
```

Default `--policy-type mlp`.  Reproduces the Stage 1 v2 soft-pass
baseline.

### Train — Stage 2 (GNN)

Same script, `--policy-type gnn` flag:

```bash
python scripts/train_stage1.py \
    --device cuda \
    --policy-type gnn \
    --n-rollouts 1000
```

Optional knobs (all shown at default values): `--d-hidden 64`,
`--n-msg-rounds 2`.  Everything else (env config, red-policy mix,
PPO hyperparameters, `n_envs`, `rollout_steps`) is shared with the
MLP path so the head-to-head comparison stays fair.

Live stdout shows the same fields for both policies:
per-rollout reward, catch rate, `pol / val / ent / kl / clip`
diagnostics.  TensorBoard scalars stream to
`runs/stage1/<timestamp>/tb/` — one directory per run,
policy-type-agnostic.

**Expected timing (Stage 2 GNN, single T4-class GPU, 1000
rollouts):** ≈ 30-60 min.  GNN forward is heavier per step than MLP
(2 rounds × ~50 vector ops each vs one MLP forward) but the
message-passing tensors are small and parallelise well on GPU.
CPU is much slower (~200 sps observed on a laptop) — GPU is
strongly recommended.

### Evaluate + sync results back

```bash
python scripts/evaluate_trained.py \
    --checkpoint runs/stage1/<timestamp>/best.pt \
    --device cuda \
    --n-episodes 50
```

Auto-dispatches based on the checkpoint's `policy_type`:
- **MLP checkpoint** → writes `docs/stage1_results.md`
- **GNN checkpoint** → writes `docs/stage2_results.md`
- **Both** → produce `runs/stage1/<timestamp>/eval_results.json` +
  4 GIFs under `runs/stage1/<timestamp>/eval_gifs/`

Acceptance criteria per stage are documented in
[`docs/stage1_analysis.md §6b`](docs/stage1_analysis.md) and
[`docs/stage2_gnn_design.md §6`](docs/stage2_gnn_design.md).

Pack everything for download:
```bash
tar czf stage1_artifacts.tar.gz runs/stage1/<timestamp>/ docs/stage1_results.md
```

## Status

Stage 1 landed as a **soft pass** (July 2026) — v2 matches
GreedyPursuer across the red distribution, catches 2.00/2 on every red
type, misses the strict 1.20× bar by 4.01 points.  Full analysis and
verdict reasoning in [`docs/stage1_analysis.md`](docs/stage1_analysis.md);
the red-policy mixing that closed v1's OOD failure is documented in
[`docs/red_policy_mixing.md`](docs/red_policy_mixing.md).  Stage 2 (partial
observability + attention-over-entities) is next.

Per-stage result writeups land in `docs/stage{N}_results.md` as each
stage completes; broader analysis notes accumulate in `docs/` alongside
them.
