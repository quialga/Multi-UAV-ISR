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

## Milestone 2 — scripted red evaders (DONE, verified 2026-07-28)

The reds get their first brain: `scripted_reds.py` (see its docstring
for a plain-language walkthrough) runs the env's own
`run_from_nearest_uav` heuristic — imported, not re-implemented — in a
1 Hz look→think→act loop against live Gazebo poses, integrating
acceleration→velocity with the env's exact `DT`/`V_MAX_RED` and
publishing per-red `cmd_vel`.

Plumbing decisions that matter:

- **Per-drone odometry, not the fused pose topic.** Each drone carries
  an `OdometryPublisher` plugin (`/model/<name>/odometry`, 20 Hz).
  The world-level `dynamic_pose/info` firehose loses model names in
  the ros_gz `Pose_V→TFMessage` translation (verified: names present
  in the raw gz message, empty `child_frame_id` after the bridge), so
  identity rides on the topic name instead. Bonus: odometry carries
  velocity, which milestone 3's shadow env needs.
- **`ros_gz_bridge parameter_bridge`** translates the 8 odometry
  topics GZ→ROS and the 3 red `cmd_vel` topics ROS→GZ
  (`[` = into ROS, `]` = into Gazebo in the mapping syntax).
- WSL Python setup: system `python3` + ROS's numpy + user-level
  `pettingzoo` (`pip install --user --break-system-packages`), repo
  imported straight from `/mnt/c` via `PYTHONPATH` — **prepend**, never
  overwrite, or `rclpy` disappears.

Run it (two WSL terminals):

```bash
# T1: the sim (press play!)
source /opt/ros/jazzy/setup.bash && gz sim ~/arena_seed0.sdf
```

```bash
# T2: translator + brain (Ctrl+C to stop)
bash /mnt/c/Users/quial/sources/Multi-UAV-ISR/gazebo/milestone2.sh
```

Verified headless: reds fled their nearest blues at the correct
headings (~20 m in 15 s, per-axis v_max saturation).

### The stuck-at-the-wall bug (found by watching, fixed 2026-07-31)

Symptom: any drone touching a wall froze there permanently. Two
compounding mismatches with the env's wall contract
(`PursuitEnv._integrate`: clip position, zero ONLY the into-wall
velocity component, sliding stays free, no friction anywhere):

1. **Command side** — the bridge nodes integrated acceleration into a
   velocity that never learned about walls, so a drone kept being
   commanded at full speed INTO the wall. Fix:
   `gazebo/kinematics.integrate_cmd` mirrors the env's rule
   (fuzz-tested bit-identical in `tests/test_gazebo_kinematics.py`).
2. **Physics side** — Gazebo's default contact friction gripped a
   pressed drone so hard it couldn't even slide sideways. The env has
   no friction concept. Fix: `mu = 0` on every touchable surface
   (drones, walls, obstacles) in the world file.

Verified: pressed full-speed into the west wall, a drone still slides
tangentially at commanded speed and escapes on demand.

## Milestone 3 — closed loop: the trained policy flies (DONE, verified 2026-07-30)

`policy_bridge.py` (read its docstring — it is the plain-language
walkthrough of the whole architecture) closes the loop:

    Gazebo odometry → shadow env (sync kinematic state) → referee
    (env's capture rule + crash event counters) → env's own belief/
    occlusion/track perception → frozen policy act_deterministic with
    GRU hidden state carried across ticks → accel→vel integration
    (env's exact DT / per-axis v_max) → cmd_vel for 5 blues;
    reds keep the milestone-2 flee heuristic, honouring the referee.

Decisions that matter:

- **Sim-time ticking**: `/clock` is bridged and the node sets
  `use_sim_time`, so the 1 Hz decision timer counts GAZEBO seconds —
  pausing the sim pauses the brain; faster-than-realtime batch eval
  (milestone 4) keeps 1 decision per sim-second.
- **Checkpoint loading**: `crash_penalty_v3/best.pt` predates the
  count-agnostic critic pool, so exactly one CRITIC tensor is stale
  (`critic_trunk.0.weight` 512→194). Loaded via `load_full_stage4`
  (76/77 tensors); deployment is actor-only, and the bridge asserts
  the actor copied completely.
- Torch in WSL: `pip install --user --break-system-packages torch
  --index-url https://download.pytorch.org/whl/cpu`.

First verified episode (headless, seed 0): blues hunted from 16 m
standoff down to **capture of red_2 at t=167 s**; episode report:
1/3 caught in 200 s, 1 obstacle-contact event, 0 ally-proximity
events. Note the gap vs the pure-Python deterministic eval
(~2.95/3): quantifying and attributing it (control latency, odometry
staleness, single-episode noise) is precisely milestone 4.

Run it (two WSL terminals):

```bash
# T1: source ROS, then:  gz sim ~/arena_seed0.sdf   (press play)
# T2:
bash /mnt/c/Users/quial/sources/Multi-UAV-ISR/gazebo/milestone3.sh
```

## Faithfulness audit (2026-07-31)

Goal: the deployed policy must experience the SAME world it trained
in — no drift. Status of every seam between bridge and training env:

| Seam | Status | Mechanism |
|---|---|---|
| Scenario (spawns, obstacle layout) | exact | same `reset(seed)`; startup check warns on mismatch |
| Perception (belief, occlusion, noise, tracks) | exact | env's own code runs in the shadow env, untouched |
| Policy invocation | exact | `act_deterministic` + GRU hidden, identical to training eval |
| Velocity clip / wall / obstacle motion rules | exact | `kinematics.integrate_cmd`; fuzz-tested trajectory-identical to `_integrate` + `_clip_positions_from_obstacles` (positions, velocities, crash masks) |
| Observation velocities | exact | shadow env fed the integrator's `state_vel` (the env's post-step velocity), not odometry twist |
| First observation | exact | tick 1 consumes the reset-time belief update (no double fuse) |
| Referee (captures, belief wipe) | exact | env's step-5 block mirrored; per-tick ordering matches step |
| Crash report card | exact rule | obstacle = env's attempted-entry mask; ally = env's radius rule; rising-edge events like the training eval |
| Physics interference | designed out | walls stand off 0.5 m outside the line; obstacle collision shells thinner than the true radius — contact physics is a failsafe that never engages in normal flight; frictionless everywhere |
| Red adversary | exact | bridge drives reds via `env.red_policy(...)` with `arena_size` — the exact call `PursuitEnv.step` uses, so wall repulsion (and any future red-policy change) tracks automatically. Guarded by `tests/test_bridge_red_faithfulness.py` |
| Executed motion between ticks | approximate | Gazebo moves continuously where the env teleports once per second; endpoints coincide by construction (`exec_vel`), mid-second positions have no env counterpart |
| Control latency | residual gap | the brain thinks ~0.1–0.2 s of sim time per tick while Gazebo keeps moving; the env applies actions instantly. Quantified in milestone 4 (lockstep pause-step eval) |

Post-fix closed-loop episode (seed 0, `pool_fixed_v1`): **2/3 caught
in 200 s** (captures t=42 s, t=145 s), 2 obstacle-contact events, 0
ally events — up from 1/3 with first capture at t=167 s before the
wall/faithfulness fixes.

After merging main's red wall-repulsion (`env.red_policy` fix above),
a re-run caught **3/3 in 186 s** (captures t=43/182/185 s), 0 obstacle
and 0 ally events — now against the harder, non-pinning reds. (Single
stochastic episode, not a controlled comparison — milestone 4 does the
20-episode eval.)

## Roadmap (remaining Phase 1 milestones)

4. **Parity + eval**: lockstep same-seed comparison vs the pure-Python
   rollout (localises any bug to the one seam Gazebo owns:
   kinematics); then a 20-episode deterministic eval at
   faster-than-realtime, attributing any residual gap (control
   latency is the prime suspect).
