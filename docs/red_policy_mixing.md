# Red-policy mixing during training

_Design + implementation note.  Part of the Stage 1 v1 → v2 upgrade
(see [`stage1_analysis.md`](stage1_analysis.md) for context)._

This document explains the small but load-bearing change that fixed
v1's out-of-distribution (OOD) failure mode: **sampling the red
adversary policy per episode from a categorical mix during training**,
rather than training against a single scripted red for the whole run.

## 1. Motivation

v1 trained blue only against `run_from_nearest_uav`.  At eval time
the trained policy collapsed against reds it had never seen:

| Trained-blue mean episode return | v1 |
|---|---|
| vs Stationary red | **−21.92** (worse than `RandomAgent`, −13.17) |
| vs Random red | −17.22 |
| vs RunFromNearest red | +10.22 |

The pattern — best on the *hardest* red, worst on the *easiest* — is
the classic signature of a policy that has learned a specialised
skill (aim-ahead / interception for a fleeing target) and generalises
badly outside that distribution.

Fix: train on a **mixture** of red policies so the network sees the
full spread of adversary behaviours.

**v2 measured, same eval seeds:**

| Trained-blue vs | v1 | v2 | Δ |
|---|---|---|---|
| Stationary | −21.92 | **+17.17** | +39.09 |
| Random | −17.22 | **+17.27** | +34.49 |
| RunFromNearest | +10.22 | **+15.37** | +5.15 |

Mean caught per episode against all three reds is now 2.00/2.  The
OOD failure is gone.

## 2. Design choices

Four decisions that keep the implementation minimal and well-behaved:

1. **Mixing happens at episode boundaries, not step boundaries.**  Each
   env samples a red policy at reset time and keeps it for the whole
   episode.  Mixing per-step would create a non-Markov transition
   distribution and break the credit-assignment story for shaping-
   free PPO.
2. **The distribution is a plain categorical with configurable
   weights.**  No annealing, no curriculum.  A uniform mix
   (`stationary:1, random:1, run:1`) worked out of the box; the API
   supports arbitrary weights in case we want to weight harder reds
   more later.
3. **The mixing RNG is separate from every env's internal RNG.**
   Otherwise mixing decisions would consume env RNG draws and break
   the per-env-seed reproducibility guarantee.  We use one master mix
   RNG seeded from `base_seed`.
4. **`random_red` gets a fresh seed on every resample.**  It's a
   closure — the same closure would give the same random red-policy
   trajectory across episodes.  Resampling with an integer drawn from
   the mixing RNG gives independent trajectories.

Explicit non-goals (things we intentionally did **not** do):

- **No self-play / learned adversary.**  Red is scripted in Stage 1.
  Learned red comes in at Stage 4 with a proper league / fictitious-
  self-play mechanism.
- **No curriculum / staged difficulty.**  We could plausibly ramp
  `run` weight up over training, but the flat mix already solved the
  OOD problem, so complexity wasn't warranted.
- **No per-env policy assignment.**  All 16 parallel envs sample
  independently from the same distribution.  Sticky per-env
  assignments would reduce diversity within a rollout without
  obvious upside.

## 3. Implementation

Three files touched: `isr/train/vec_env.py` (core), `scripts/train_stage1.py`
(CLI), and the imports in `vec_env.py`.

### 3.1 The mix definition

At `VectorPursuitEnv` construction time:

```python
def __init__(
    self,
    n_envs:              int,
    env_kwargs:          Dict[str, Any],
    base_seed:           int = 0,
    episode_buffer_size: int = 128,
    red_policy_mix:      Optional[Sequence[Tuple[str, float]]] = None,
) -> None:
    ...
    # One dedicated RNG for the mixing decision — kept independent
    # from every env's internal RNG so per-env-seed reproducibility
    # of rollouts is preserved.
    self._mix_rng = np.random.default_rng(int(base_seed))

    self._red_policy_names: Optional[List[str]] = None
    self._red_policy_probs: Optional[np.ndarray] = None
    if red_policy_mix is not None:
        names   = [n for n, _ in red_policy_mix]
        weights = np.asarray([w for _, w in red_policy_mix],
                             dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("red_policy_mix weights must sum > 0")
        self._red_policy_names = names
        self._red_policy_probs = weights / total
```

When `red_policy_mix` is `None`, the mechanism is fully disabled and
each env keeps whatever `red_policy` was passed via `env_kwargs`
(legacy v1 behaviour, useful for ablations).

### 3.2 Resampling helper

A single point of dispatch that maps a sampled category name to a red
policy callable and installs it on the env:

```python
def _resample_red_policy(self, env: PursuitEnv) -> None:
    """Sample a fresh red policy for this env; no-op if mixing disabled."""
    if self._red_policy_names is None or self._red_policy_probs is None:
        return
    idx = int(self._mix_rng.choice(
        len(self._red_policy_names), p=self._red_policy_probs,
    ))
    name = self._red_policy_names[idx]
    if name == "stationary":
        env.red_policy = stationary_red
    elif name == "random":
        # New seeded closure per episode so consecutive 'random' picks
        # produce independent red trajectories.
        env.red_policy = random_red(
            seed=int(self._mix_rng.integers(0, 2**31 - 1)),
        )
    elif name == "run":
        env.red_policy = run_from_nearest_uav
    else:
        raise ValueError(f"Unknown red policy name in mix: {name!r}")
```

The `env.red_policy` attribute is the same callable the env consumes
inside `PursuitEnv.step()`, so swapping it in place has no downstream
plumbing cost.

### 3.3 Reset hooks

Two places call `_resample_red_policy`:

**Initial reset (train-start):**
```python
def reset(self, seed: Optional[int] = None) -> np.ndarray:
    ...
    for i, env in enumerate(self.envs):
        self._resample_red_policy(env)
        s = None if seed is None else int(seed + i)
        obs_dict, _ = env.reset(seed=s)
        ...
```

**Auto-reset (mid-rollout, when an episode ends):**
```python
if done:
    ...  # log completed-episode stats
    # Resample red policy first (no-op if mixing disabled), then
    # let the env's internal RNG advance naturally.
    self._resample_red_policy(env)
    obs_d, _ = env.reset()
```

Every episode boundary triggers a fresh sample.  There is no need for
a manual "flush" or curriculum tick.

### 3.4 CLI + config wiring

The training script exposes the mix as a comma-separated string that
gets parsed into the `(name, weight)` list expected by
`VectorPursuitEnv`:

```python
p.add_argument("--red-policy-mix",
               default="stationary:1,random:1,run:1",
               help="Comma-separated 'name:weight' pairs specifying the "
                    "per-episode red-policy distribution.  Set to 'run:1' "
                    "to reproduce the legacy v1 training distribution.")
```

Parsed once at run start:

```python
red_policy_mix = []
for chunk in args.red_policy_mix.split(","):
    chunk = chunk.strip()
    if not chunk:
        continue
    name, w = chunk.split(":")
    red_policy_mix.append((name.strip(), float(w)))
```

And forwarded into `VectorPursuitEnv(red_policy_mix=red_policy_mix,
...)`.

## 4. Usage recipes

**Default v2 training** (uniform mix — recommended):
```bash
python scripts/train_stage1.py --device cuda --n-rollouts 1000
```

**Reproduce v1 training distribution** (single red for ablation):
```bash
python scripts/train_stage1.py --red-policy-mix "run:1"
```

**Weight the harder red more heavily** (once basic capability is there):
```bash
python scripts/train_stage1.py --red-policy-mix "stationary:1,random:1,run:3"
```

**Programmatic use** (from a notebook / script that skips the CLI):
```python
from isr.train.vec_env import VectorPursuitEnv

vec_env = VectorPursuitEnv(
    n_envs=16,
    env_kwargs={"n_blue": 3, "n_red": 2, "arena_size": 100.0,
                "max_steps": 200, "capture_radius": 3.0},
    base_seed=0,
    red_policy_mix=[("stationary", 1.0),
                    ("random",     1.0),
                    ("run",        1.0)],
)
```

## 5. What we could add later

Concrete extensions that were ruled out for v2 but are cheap to add
when they become useful:

- **Curriculum-style mixing.**  Ramp `run` weight up over training —
  start with all `stationary` (easy), transition to a uniform mix,
  then bias toward `run`.  Would let us report a "learning curve
  under increasing difficulty" plot, but is only useful if the flat
  mix stops working, which it didn't.
- **Sticky per-env mixing.**  Fix each env's red policy for a
  block of K episodes rather than resampling every episode.  Might
  stabilise value estimates when the value network is small.
- **Learned mix weights** (PSRO-lite).  Use exploitability metrics
  from evaluation to bias the training mix toward reds the current
  policy handles worst.  Natural fit for Stage 4.
- **Broader red policy zoo.**  Add e.g. `random_wall_hugger`,
  `patrol_around_center`, `predictable_zigzag`.  Cheap and adds
  distribution coverage.

Any of these can be dropped in with only additions to
`_resample_red_policy`'s dispatch — the surrounding infrastructure is
policy-agnostic.
