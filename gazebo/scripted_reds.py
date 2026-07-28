"""
gazebo/scripted_reds.py — Milestone 2: give the red drones a brain.

WHAT THIS PROGRAM IS, IN PLAIN LANGUAGE
---------------------------------------
Right now the drones in Gazebo are puppets: they hang in the air
waiting for someone to tell them a velocity.  This program is the
first puppeteer.  Once per second it:

    1. LOOKS   — reads where every drone currently is in Gazebo,
    2. THINKS  — asks the training env's own evader logic
                 ("run away from the nearest blue UAV, steer around
                 obstacles") what each red should do about it,
    3. ACTS    — sends each red drone a new velocity command.

That look → think → act loop, repeated forever, is the entire idea of
robot control.  Milestone 3 will reuse this exact skeleton and only
swap step 2: instead of a hand-written flee rule for the reds, the
trained neural network will decide for the blues.

HOW THE PROGRAM TALKS TO GAZEBO (the plumbing, explained)
---------------------------------------------------------
Programs never call Gazebo directly.  Everything goes over "topics" —
named mailboxes.  A program can PUBLISH (drop a letter in a mailbox)
or SUBSCRIBE (get every letter that lands in one).  Gazebo constantly
publishes drone positions into one mailbox and listens for velocity
commands in others.  This decoupling is the core design idea of both
Gazebo and ROS: programs only agree on mailbox names and letter
formats, never on each other's internals.

There is one complication: Gazebo has ITS OWN mailbox system, separate
from ROS's, and this Python program only speaks ROS (via ``rclpy``,
which came with your ROS install).  The fix is a translator process
called ``ros_gz_bridge`` — the launcher script starts it for us.  For
each mailbox we list, the bridge copies letters between the two
systems, converting formats as it goes.  Data flows like this:

    Gazebo sim                       this program (ROS side)
    ----------                       ------------------------
    drone positions ──► bridge ──►   "poses" mailbox   (we SUBSCRIBE)
    red velocities  ◄── bridge ◄──   "cmd_vel" mailboxes (we PUBLISH)

Mailboxes used (one line each):
- ``/model/<name>/odometry`` (one per drone, we SUBSCRIBE to all 8):
  each drone's OdometryPublisher plugin broadcasts its own position
  and velocity here 20x/second.  Because every drone has its own
  private mailbox, the topic name itself tells us who the measurement
  belongs to.  (Gazebo also offers one firehose topic with every
  model's pose in it, but the ROS translation of that message loses
  the model names — we found this out the hard way — so per-drone
  odometry it is.)
- ``/model/red_<j>/cmd_vel``: each drone's VelocityControl plugin
  (declared in the world file) listens here.  We publish a ROS
  ``Twist`` — a velocity request with linear x/y/z parts — and the
  bridge hands it to Gazebo, which moves the drone at exactly that
  velocity until told otherwise.

WHY THE HEURISTIC IS IMPORTED, NOT REWRITTEN
--------------------------------------------
``run_from_nearest_uav`` is imported from ``isr`` — the same function
the training env calls every step.  Re-typing it here would invite
tiny differences (a sign flip, a different tie-break) that would make
Gazebo behaviour quietly diverge from training.  The whole Phase 1
philosophy is: Gazebo replaces ONLY physics; every decision-making
line of code is the original, imported.

WHY WE INTEGRATE ACCELERATION OURSELVES
---------------------------------------
The heuristic (like the trained policy) outputs an ACCELERATION —
"push this way".  But Gazebo's VelocityControl plugin wants a
VELOCITY — "move this fast".  So we do the same one-line integration
the env does each step:

    velocity <- clip(velocity + accel * DT, -V_MAX_RED, +V_MAX_RED)

with the env's exact numbers (DT = 1 s, V_MAX_RED = 1 m/s, clipped
per-axis).  Keeping this arithmetic identical to the env's
``_integrate`` is what makes Gazebo motion comparable to training.

WHAT IS DELIBERATELY MISSING (comes in Milestone 3)
---------------------------------------------------
- No referee: reds are never "caught" here, so they flee forever.
- Blues stay parked (nothing publishes to their cmd_vel yet).
- No belief maps / sensors: the heuristic legitimately uses true
  positions, because that is exactly what it sees during training too.

RUN IT (two WSL terminals)
--------------------------
  Terminal 1:  source /opt/ros/jazzy/setup.bash
               gz sim ~/arena_seed0.sdf        # press play!
  Terminal 2:  bash /mnt/c/Users/quial/sources/Multi-UAV-ISR/gazebo/milestone2.sh

Watch the red squares run from the blues and swerve around pillars.
Ctrl+C in terminal 2 stops them (they will coast with their last
commanded velocity; that is Gazebo faithfully doing what it was told).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy                                    # ROS 2 Python client
from rclpy.node import Node
from geometry_msgs.msg import Twist             # velocity-command letter
from nav_msgs.msg import Odometry               # position+velocity letter

from isr.env.pursuit_env import run_from_nearest_uav
from gazebo.world_gen import make_env

# The env's exact kinematic constants (isr/env/entities.py: RED_TARGET
# has v_max = 1.0; PursuitEnv default dt = 1.0).  Change nothing here
# without changing the env — they must match or motion diverges.
DT        = 1.0
V_MAX_RED = 1.0


class ScriptedReds(Node):
    """A ROS "node" = one running program with mailboxes attached.

    rclpy calls our functions for us when things happen (a letter
    arrives, a timer fires) — we never write a while-loop ourselves.
    """

    def __init__(self, n_blue: int, n_red: int, obstacle_pos, obstacle_r):
        super().__init__("scripted_reds")
        self.n_blue = n_blue
        self.n_red = n_red
        # Obstacle geometry is STATIC scenario knowledge (sampled at
        # reset, mirrored into the world file) — the flee heuristic
        # uses it to steer around pillars instead of hugging them.
        self.obstacle_pos = obstacle_pos
        self.obstacle_r = obstacle_r

        # Latest known (x, y) of every drone, filled by pose letters.
        self.pos: dict[str, np.ndarray] = {}

        # Each red drone's current commanded velocity.  We integrate
        # acceleration into THIS array — it is the reds' "momentum".
        self.red_vel = np.zeros((n_red, 2), dtype=np.float32)

        # SUBSCRIBE: one odometry mailbox per drone (blues too — the
        # heuristic needs to know where the pursuers are).  Whenever a
        # letter arrives, _on_odom runs with the drone's name attached;
        # we simply remember the newest position per drone.
        names = ([f"blue_{i}" for i in range(n_blue)]
                 + [f"red_{j}" for j in range(n_red)])
        for name in names:
            self.create_subscription(
                Odometry, f"/model/{name}/odometry",
                # name=name freezes the loop variable into the callback.
                lambda msg, name=name: self._on_odom(name, msg), 10)

        # PUBLISH: one velocity mailbox per red drone.
        self.cmd_pubs = [
            self.create_publisher(Twist, f"/model/red_{j}/cmd_vel", 10)
            for j in range(n_red)
        ]

        # TIMER: run the look->think->act tick every DT seconds — the
        # same decision rate the policy trained at (1 step = 1 s).
        self.create_timer(DT, self._tick)
        self.get_logger().info(
            f"scripted_reds up: {n_red} reds fleeing {n_blue} blues, "
            f"tick every {DT}s")

    def _on_odom(self, name: str, msg: Odometry) -> None:
        """An odometry letter arrived: remember where this drone is.

        We only need x and y — altitude is cosmetic in our 2D
        scenario.  (Odometry also carries velocity; milestone 3's
        bridge will use it to fill the shadow env's velocity state.)"""
        p = msg.pose.pose.position
        self.pos[name] = np.array([p.x, p.y], dtype=np.float32)

    def _tick(self) -> None:
        """The 1 Hz look->think->act cycle."""
        # LOOK: assemble position arrays in index order.  If Gazebo
        # hasn't reported everyone yet (first seconds, or sim paused
        # before play), skip this tick rather than act on stale data.
        try:
            blue_pos = np.stack([self.pos[f"blue_{i}"]
                                 for i in range(self.n_blue)])
            red_pos = np.stack([self.pos[f"red_{j}"]
                                for j in range(self.n_red)])
        except KeyError:
            self.get_logger().warning(
                "waiting for poses... (is the sim running? press play)")
            return

        # THINK: the training env's own evader logic, byte-identical.
        # (No referee yet, so every red counts as active/uncaught.)
        red_active = np.ones(self.n_red, dtype=bool)
        accel = run_from_nearest_uav(
            blue_pos, red_pos, red_active,
            self.obstacle_pos, self.obstacle_r,
        )
        accel = np.clip(accel.astype(np.float32), -1.0, 1.0)

        # INTEGRATE: acceleration -> velocity, the env's exact rule
        # (per-axis clip at v_max, not a diagonal-length clip).
        self.red_vel = np.clip(self.red_vel + accel * DT,
                               -V_MAX_RED, V_MAX_RED)

        # ACT: one velocity letter per red.  z stays 0 (hold altitude).
        for j, pub in enumerate(self.cmd_pubs):
            cmd = Twist()
            cmd.linear.x = float(self.red_vel[j, 0])
            cmd.linear.y = float(self.red_vel[j, 1])
            pub.publish(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Scripted red evaders (M2).")
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match the seed the world file was "
                        "generated with — it re-samples the same "
                        "obstacle layout for the flee heuristic.")
    p.add_argument("--n-blue", type=int, default=5)
    p.add_argument("--n-red", type=int, default=3)
    p.add_argument("--n-obstacles", type=int, default=4)
    p.add_argument("--arena-size", type=float, default=130.0)
    args = p.parse_args()

    # Re-run the same env reset that generated the world file: same
    # seed -> same obstacle layout (verified deterministic by
    # tests/test_world_gen.py::test_same_seed_same_world).
    env = make_env(args.seed, args.n_blue, args.n_red,
                   args.n_obstacles, args.arena_size)

    rclpy.init()
    node = ScriptedReds(args.n_blue, args.n_red,
                        env._obstacle_pos, env._obstacle_r)
    try:
        rclpy.spin(node)          # hand control to ROS; timers/mail run
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException):
        pass                      # Ctrl+C / kill: exit quietly
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
