# Gazebo bridge (Phase 1 — sim-to-sim deployment)

Deploys the **frozen** Stage 4 policy in Gazebo Sim (Harmonic).
Training stays in the fast Python env; Gazebo replaces **only the
kinematics/rendering layer** ("Layer 1"). Perception (belief maps,
occlusion, noisy tracks) keeps running in the Python env, driven by
Gazebo poses — the *shadow-env pattern*. Rationale: the actor was
trained on belief-derived observations, so feeding it Gazebo ground
truth would be out-of-distribution; reusing the env's perception code
byte-for-byte keeps the policy in-distribution with zero
reimplementation.

## Environment (what runs where)

| Piece | Where | Why |
|---|---|---|
| Training / tests / this generator | Windows venv | unchanged workflow |
| Gazebo Sim 8 (Harmonic) + ROS 2 Jazzy | WSL2 Ubuntu 24.04 | Linux-first tooling |
| Bridge node (next milestone) | WSL2 Python | must speak gz-transport |

Gazebo is installed via the **ROS Jazzy vendor packages**, so the `gz`
CLI only exists after sourcing ROS in each WSL terminal:

```bash
source /opt/ros/jazzy/setup.bash
```

**Gotcha (verified):** `gz sim` silently ignores world files on the
Windows mount (`/mnt/c/...`) and loads its empty default world instead
— while `gz sdf --check` accepts the same path fine. Always copy world
files into the WSL filesystem (e.g. `~/`) before loading them.

## Milestone 1 — world generation (this directory)

`world_gen.py` runs a headless `PursuitEnv.reset(seed)` — the same
scenario sampling training uses — and writes an SDF world mirroring it
1:1 (1 env unit = 1 m, identity (x, y) mapping):

- `blue_i` / `red_j`: single-link kinematic drones. `<gravity>false>`
  holds altitude (blue 5 m, red 1 m); a `VelocityControl` plugin makes
  `/model/<name>/cmd_vel` (`gz.msgs.Twist`) set the model's velocity
  directly each step — matching the env's velocity-level contract (the
  bridge integrates the policy's acceleration → velocity itself, with
  the env's exact `dt`/`v_max` component-wise clip).
- `obstacle_k`: static cylinders at the env's exact centres/radii,
  12 m tall (blues fly at 5 m: no flying over, same as the 2D model).
- 4 translucent boundary walls (the env stops blues at the walls).

### Generate + view

```bash
# Windows (repo root) — generate:
.venv\Scripts\python.exe gazebo\world_gen.py --seed 0 --out gazebo\worlds\arena_seed0.sdf
```

```bash
# WSL — copy to native filesystem, then open with the GUI (WSLg):
source /opt/ros/jazzy/setup.bash
cp /mnt/c/Users/quial/sources/Multi-UAV-ISR/gazebo/worlds/arena_seed0.sdf ~/
gz sim ~/arena_seed0.sdf
```

In the GUI press ▶ (bottom-left) to run the simulation. Useful CLI
(each in a sourced WSL terminal, with the sim running):

```bash
gz model --list                          # all models in the world
gz model -m blue_0 -p                    # one model's pose
gz topic -l                              # all live topics
# drive blue_0 at 1.5 m/s in +x (the env's v_max):
gz topic -t /model/blue_0/cmd_vel -m gz.msgs.Twist -p "linear: {x: 1.5}"
```

`-s` runs the server headless (no GUI), `-r` starts unpaused:
`gz sim ~/arena_seed0.sdf -s -r`.

### Verified (2026-07-27)

Headless load in WSL: all 16 models present at the env's coordinates;
`cmd_vel` with `vx = 1.5` moved `blue_0` ~7.9 m in ~5.2 s with zero
y-drift and altitude locked at 5.0 m — the kinematic contract holds.

Contract tests: `tests/test_world_gen.py` (pure Python, no Gazebo
needed — parses the SDF and checks it mirrors the env layout).

## Roadmap (remaining Phase 1 milestones)

2. **Scripted reds open-loop**: bridge drives red drones with the env's
   heuristic evader policies; blues hover.
3. **Closed loop**: shadow env consumes Gazebo poses →
   `_update_belief_maps()` → `structured_belief_observation()` → frozen
   policy (GRU hidden state carried across steps) → accel→vel
   integration → `cmd_vel`; referee (captures at 3 m, crash counts)
   from shadow-env logic.
4. **Parity + eval**: lockstep same-seed comparison vs the pure-Python
   rollout; then a 20-episode deterministic eval (target: catch rate
   within noise of `runs/stage4/crash_penalty_v3`'s ~2.95/3).
