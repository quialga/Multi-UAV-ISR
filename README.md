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
| 1 | Pursuit-evasion baseline (continuous 2D, fully observable, fixed red, PPO blue) | in progress |
| 2 | Partial observability — sensor radius, GRU/LSTM recurrent policies | |
| 3 | Sensor noise + occlusion — explicit belief-state predictor (aux loss) | |
| 4 | Self-play — red team becomes learned; fictitious self-play, league training | |
| 5 | Communication — range-gated message passing between teammates | |
| 6 | Dynamic objectives — moving / appearing / re-prioritised targets | |
| 7 | Adversarial robustness evaluation — exploitability metrics, held-out opponents | |

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

Train Stage 1:
```bash
python scripts/train_stage1.py
```

## Status

Stage 1 in progress.  This README will track which stages are landed; per-stage
result writeups land in `docs/stage{N}_results.md` as each one passes its
acceptance criterion.
