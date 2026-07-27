"""
gazebo/world_gen.py — generate a Gazebo world from a PursuitEnv reset.

THE SHADOW-ENV PATTERN (Phase 1, Layer 1)
-----------------------------------------
The scenario source of truth is ``PursuitEnv.reset(seed)``: it samples
the obstacle layout and every spawn position exactly as training did.
This script runs that reset headlessly and emits an SDF world that
MIRRORS it — one Gazebo model per env entity, at the env's coordinates.
Gazebo replaces only the kinematics/rendering layer; perception (belief
maps, occlusion, tracks) stays in the Python env, driven by Gazebo
poses (that wiring is the bridge node, next milestone).

GAZEBO CONCEPTS USED HERE (beginner notes)
------------------------------------------
- SDF ("Simulation Description Format"): the XML file format Gazebo
  loads.  One ``<world>`` contains many ``<model>`` elements.
- model: a named simulation object (a drone, a pillar, the ground).
  ``<static>true</static>`` models never move and cost no physics.
- link: a rigid body inside a model.  Our models are single-link.
  A link holds ``<collision>`` (shape the physics engine collides),
  ``<visual>`` (shape the renderer draws — they need not match) and
  ``<inertial>`` (mass/inertia).
- pose: ``x y z roll pitch yaw`` (metres / radians) relative to the
  parent frame.  We use env coordinates directly: 1 env unit = 1 m,
  env (x, y) -> Gazebo (x, y), z = a fixed per-team altitude.
- plugin ("system"): a shared library attached to the world or to a
  model that runs code every physics step.  World-level boilerplate:
  Physics (steps the engine), SceneBroadcaster (publishes world state
  so GUIs/tools can see it), UserCommands (lets tools spawn/move
  things at runtime).
- VelocityControl plugin (per drone): listens on the Gazebo topic
  ``/model/<name>/cmd_vel`` (Twist message) and applies the commanded
  linear/angular velocity to the model KINEMATICALLY each step — no
  forces, no motor model.  This matches our double-integrator
  contract: the bridge integrates the policy's acceleration into a
  velocity (same dt / v_max clip) and publishes it; Gazebo executes.
- gravity: real physics would drop a hovering box.  Drone links set
  ``<gravity>false</gravity>`` so altitude holds without a controller
  — the honest equivalent of the 2D env's "altitude doesn't exist".

ENTITY MAPPING
--------------
  env blue i   -> model "blue_i"  : 0.6 m blue box,  z = ALT_BLUE (5 m)
  env red j    -> model "red_j"   : 0.6 m red box,   z = ALT_RED  (1 m)
  env obstacle -> model "obstacle_k": static grey cylinder, exact env
                  (x, y) centre and radius, OBSTACLE_H tall — visual
                  reminder that the 2D model has no "fly over".
  arena bounds -> 4 static translucent walls (blues stop at walls in
                  the env; the walls make that visible)

Run (from repo root, any Python with the repo's deps):
    python gazebo/world_gen.py --seed 0 --out gazebo/worlds/arena_seed0.sdf

Then in a WSL terminal (sourcing makes the `gz` CLI visible):
    source /opt/ros/jazzy/setup.bash
    gz sim gazebo/worlds/arena_seed0.sdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALT_BLUE   = 5.0   # m — blue cruise altitude (cosmetic; 2D model)
ALT_RED    = 1.0   # m — red targets fly low
OBSTACLE_H = 12.0  # m — obstacle cylinder height (taller than ALT_BLUE)
WALL_H     = 12.0
WALL_T     = 0.5

# World-level boilerplate systems every gz-sim world needs:
# Physics steps the engine, UserCommands accepts runtime service calls
# (spawn/set-pose), SceneBroadcaster publishes the scene graph + poses.
_WORLD_PLUGINS = """\
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
"""

_GROUND_AND_SUN = """\
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 50 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
          <material><ambient>0.55 0.6 0.55 1</ambient><diffuse>0.55 0.6 0.55 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _drone_model(name: str, x: float, y: float, z: float,
                 rgba: str) -> str:
    """One single-link kinematic drone.  <gravity>false</gravity> keeps
    it hovering; the VelocityControl plugin makes /model/<name>/cmd_vel
    (gz.msgs.Twist) set its velocity directly each physics step."""
    return f"""\
    <model name="{name}">
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="body">
        <gravity>false</gravity>
        <inertial>
          <mass>1.0</mass>
          <inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>0.6 0.6 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.6 0.6 0.2</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
      <plugin filename="gz-sim-velocity-control-system"
              name="gz::sim::systems::VelocityControl">
        <topic>/model/{name}/cmd_vel</topic>
      </plugin>
    </model>
"""


def _obstacle_model(k: int, x: float, y: float, r: float) -> str:
    return f"""\
    <model name="obstacle_{k}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {OBSTACLE_H / 2:.3f} 0 0 0</pose>
      <link name="body">
        <collision name="collision">
          <geometry><cylinder><radius>{r:.3f}</radius><length>{OBSTACLE_H}</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>{r:.3f}</radius><length>{OBSTACLE_H}</length></cylinder></geometry>
          <material><ambient>0.4 0.4 0.45 1</ambient><diffuse>0.4 0.4 0.45 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _wall_models(arena: float) -> str:
    """Four static translucent slabs on the arena boundary."""
    half = arena / 2.0
    specs = [
        ("wall_south", half, -WALL_T / 2, arena + WALL_T, WALL_T),
        ("wall_north", half, arena + WALL_T / 2, arena + WALL_T, WALL_T),
        ("wall_west", -WALL_T / 2, half, WALL_T, arena + WALL_T),
        ("wall_east", arena + WALL_T / 2, half, WALL_T, arena + WALL_T),
    ]
    out = []
    for name, x, y, sx, sy in specs:
        out.append(f"""\
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {WALL_H / 2:.3f} 0 0 0</pose>
      <link name="body">
        <collision name="collision">
          <geometry><box><size>{sx:.3f} {sy:.3f} {WALL_H}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.3f} {sy:.3f} {WALL_H}</size></box></geometry>
          <material><ambient>0.7 0.7 0.75 0.35</ambient><diffuse>0.7 0.7 0.75 0.35</diffuse></material>
          <transparency>0.65</transparency>
        </visual>
      </link>
    </model>
""")
    return "".join(out)


def env_to_sdf(env, world_name: str = "arena") -> str:
    """Serialise a freshly-reset PursuitEnv into an SDF world string."""
    parts = [f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="default" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
{_WORLD_PLUGINS}{_GROUND_AND_SUN}"""]
    parts.append(_wall_models(float(env.arena_size)))
    for k in range(env.n_obstacles):
        x, y = env._obstacle_pos[k]
        parts.append(_obstacle_model(k, float(x), float(y),
                                     float(env._obstacle_r[k])))
    for i in range(env.n_blue):
        x, y = env._blue_pos[i]
        parts.append(_drone_model(f"blue_{i}", float(x), float(y),
                                  ALT_BLUE, "0.1 0.3 0.9 1"))
    for j in range(env.n_red):
        x, y = env._red_pos[j]
        parts.append(_drone_model(f"red_{j}", float(x), float(y),
                                  ALT_RED, "0.9 0.15 0.1 1"))
    parts.append("  </world>\n</sdf>\n")
    return "".join(parts)


def make_env(seed: int, n_blue: int, n_red: int, n_obstacles: int,
             arena_size: float):
    """Reset a headless PursuitEnv purely for its sampled layout.
    Belief machinery is off — world gen only needs geometry; the
    bridge node (next milestone) runs the full training config."""
    from isr.env.pursuit_env import PursuitEnv
    env = PursuitEnv(n_blue=n_blue, n_red=n_red, arena_size=arena_size,
                     n_obstacles=n_obstacles, seed=seed)
    env.reset(seed=seed)
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-blue", type=int, default=5)
    p.add_argument("--n-red", type=int, default=3)
    p.add_argument("--n-obstacles", type=int, default=4)
    p.add_argument("--arena-size", type=float, default=130.0)
    p.add_argument("--out", type=Path,
                   default=Path("gazebo/worlds/arena_seed0.sdf"))
    args = p.parse_args()

    env = make_env(args.seed, args.n_blue, args.n_red, args.n_obstacles,
                   args.arena_size)
    sdf = env_to_sdf(env)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(sdf, encoding="utf-8")
    print(f"Wrote {args.out}  "
          f"({args.n_blue} blue, {args.n_red} red, "
          f"{env.n_obstacles} obstacles, seed={args.seed})")


if __name__ == "__main__":
    main()
