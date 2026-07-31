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
- OdometryPublisher plugin (per drone): the mirror of VelocityControl.
  It broadcasts the drone's own position AND velocity on
  ``/model/<name>/odometry`` 20x/second.  One private topic per drone
  means a reader always knows WHO a measurement belongs to from the
  topic name alone.  (The world-level ``dynamic_pose/info`` firehose
  also exists, but the ros_gz bridge drops the model names when
  translating it — verified 2026-07-27 — so per-model odometry is our
  pose source.)
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

# Drone COLLISION stays a 0.6 m box (the env treats agents as points;
# keeping the physical footprint small minimises contact-geometry
# mismatch against the env's point-vs-disk crash model).  The VISUAL
# is drawn much larger + emissive so a 130 m arena view still shows
# the drones clearly — renderer-only, physics never sees it.
DRONE_COL_XY = 0.6
DRONE_COL_Z  = 0.2
DRONE_VIS_XY = 3.0
DRONE_VIS_Z  = 0.8

# The COMMAND layer (gazebo/kinematics.py) enforces the env's exact
# wall/obstacle motion rules, so Gazebo's contact physics must be a
# FAILSAFE that never engages in normal flight — if physics blocked a
# drone before the command rule did, positions would drift from the
# env (e.g. a wall-hugger stopping 0.3 m short of coordinate 0 because
# of its own body width).  Hence:
#   - walls sit OUTSIDE the arena line by WALL_STANDOFF, so a drone
#     CENTRE can reach exactly 0 / arena_size like in the env;
#   - obstacle COLLISION cylinders are slightly thinner than the true
#     radius (visual stays exact), so the command-side stop at the
#     true radius fires before any physical contact.
WALL_STANDOFF     = 0.5
OBSTACLE_COL_SHRINK = 0.35

# Frictionless contact for every surface a drone can touch.  The
# training env has NO friction concept: at a wall it zeroes only the
# into-the-wall velocity component and lets the agent slide freely
# along it (PursuitEnv._integrate).  Gazebo's default friction instead
# GRIPS a drone pressed against a wall/pillar so hard it cannot slide
# at all (the "stuck at the wall" bug).  mu = tangential friction
# coefficient; 0 = ice.  (The command-side half of this fix is
# gazebo/kinematics.py.)
_FRICTIONLESS = ("<surface><friction><ode>"
                 "<mu>0</mu><mu2>0</mu2>"
                 "</ode></friction></surface>")

def _gui_config(arena: float) -> str:
    """GUI layout with a start camera that FRAMES THE ARENA.

    Without this, gz-sim's default camera sits near the world origin —
    which is the arena's (0, 0) CORNER — and 130 m of scene is out of
    frame.  Declaring any <gui> plugin replaces the default layout, so
    the full standard set must be listed: MinimalScene (the 3D
    viewport; carries <camera_pose> = x y z roll pitch yaw),
    GzSceneManager (syncs world -> viewport), InteractiveViewControl
    (mouse orbit/zoom), CameraTracking (right-click "Move To"/"Follow"),
    WorldControl (play/pause/step bar), WorldStats (sim-time readout),
    EntityTree (model list panel).

    Camera: above the south edge at ~0.65*arena altitude, pitched down
    0.65 rad, yawed +90 deg (facing +y) -> the whole arena in view.
    """
    cx = arena / 2.0
    cy = -0.45 * arena
    cz = 0.65 * arena
    return f"""\
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.8 0.85 0.95</background_color>
        <camera_pose>{cx:.1f} {cy:.1f} {cz:.1f} 0 0.65 1.5708</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui>
          <property key="state" type="string">floating</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <gz-gui>
          <property key="state" type="string">floating</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="CameraTracking" name="Camera tracking">
        <gz-gui>
          <property key="state" type="string">floating</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <title>World control</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="double" key="width">121</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="left" target="left"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <start_paused>true</start_paused>
        <use_event>true</use_event>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <title>World stats</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="double" key="width">290</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="right" target="right"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <sim_time>true</sim_time>
        <real_time>true</real_time>
        <real_time_factor>true</real_time_factor>
        <iterations>true</iterations>
      </plugin>
      <plugin filename="EntityTree" name="Entity tree"/>
    </gui>
"""


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
          <geometry><box><size>{DRONE_COL_XY} {DRONE_COL_XY} {DRONE_COL_Z}</size></box></geometry>
          {_FRICTIONLESS}
        </collision>
        <visual name="visual">
          <geometry><box><size>{DRONE_VIS_XY} {DRONE_VIS_XY} {DRONE_VIS_Z}</size></box></geometry>
          <material>
            <ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>
            <emissive>{rgba}</emissive>
          </material>
        </visual>
      </link>
      <plugin filename="gz-sim-velocity-control-system"
              name="gz::sim::systems::VelocityControl">
        <topic>/model/{name}/cmd_vel</topic>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system"
              name="gz::sim::systems::OdometryPublisher">
        <odom_topic>/model/{name}/odometry</odom_topic>
        <odom_publish_frequency>20</odom_publish_frequency>
      </plugin>
    </model>
"""


def _obstacle_model(k: int, x: float, y: float, r: float) -> str:
    r_col = max(r - OBSTACLE_COL_SHRINK, 0.1)
    return f"""\
    <model name="obstacle_{k}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {OBSTACLE_H / 2:.3f} 0 0 0</pose>
      <link name="body">
        <collision name="collision">
          <geometry><cylinder><radius>{r_col:.3f}</radius><length>{OBSTACLE_H}</length></cylinder></geometry>
          {_FRICTIONLESS}
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
    off = WALL_STANDOFF + WALL_T / 2      # wall centre, outside the line
    span = arena + 2 * WALL_STANDOFF + WALL_T
    specs = [
        ("wall_south", half, -off, span, WALL_T),
        ("wall_north", half, arena + off, span, WALL_T),
        ("wall_west", -off, half, WALL_T, span),
        ("wall_east", arena + off, half, WALL_T, span),
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
          {_FRICTIONLESS}
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
{_WORLD_PLUGINS}{_gui_config(float(env.arena_size))}{_GROUND_AND_SUN}"""]
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
