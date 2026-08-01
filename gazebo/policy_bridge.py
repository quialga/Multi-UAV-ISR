"""
gazebo/policy_bridge.py — Milestone 3: the trained brain flies the blues.

WHAT THIS PROGRAM IS, IN PLAIN LANGUAGE
---------------------------------------
Milestone 2 gave the red drones a hand-written brain.  This program
gives the blue drones the REAL one: the neural network you trained
(the crash-avoidance policy), loaded from its checkpoint file and run
exactly the way the training code's evaluation runs it.  Both teams
are driven from this single program:

    blues:  odometry -> shadow env perception -> neural network -> cmd_vel
    reds:   odometry -> flee heuristic (milestone 2 logic)       -> cmd_vel

THE SHADOW ENV — THE ONE IDEA THAT MAKES THIS CORRECT
-----------------------------------------------------
During training, the policy NEVER saw true enemy positions.  It saw a
processed picture: a Bayesian "belief map" accumulated from noisy
sensor sweeps (85% hit rate, 15% false alarms), blocked by obstacles
(occlusion), decaying over time, with enemy "tracks" extracted from
its peaks.  All of that machinery lives inside ``PursuitEnv`` — it is
part of the WORLD the policy grew up in, not part of the policy.

So feeding the network raw Gazebo positions would be like asking a
radar operator to suddenly read the battlefield through a window —
it has never seen that format.  Instead we keep a headless
``PursuitEnv`` instance — the SHADOW ENV — running alongside Gazebo:

    Gazebo's job   : physics.  Where drones actually are after moving.
    shadow env job : everything else.  Sensors, noise, occlusion,
                     belief map, track extraction, referee.

Each second we copy Gazebo's drone positions INTO the shadow env's
state arrays, then let the env's own perception code (untouched,
imported, byte-identical to training) turn them into the observation
the network expects.  The env is called "shadow" because it mirrors
Gazebo's reality without simulating any motion itself.

ONE TICK, STEP BY STEP
----------------------
The per-tick decision logic (sync -> referee -> belief -> observe ->
think -> integrate) lives in ``gazebo/brain.py`` (``ClosedLoopBrain``),
deliberately free of ROS so the parity test can run the EXACT same code
without Gazebo (tests/test_bridge_parity.py).  THIS file is the thin
ROS wrapper around it: each simulated second it reads the drones'
odometry, calls ``brain.tick(blue_pos, red_pos)``, and publishes the
velocity commands the brain returns (stopping every drone and printing
the episode report when the brain signals the episode is over).

WHY THE LOOP RUNS ON SIMULATION TIME, NOT WALL-CLOCK TIME
---------------------------------------------------------
The policy was trained at "one decision per simulated second".  If we
used a wall-clock timer, pausing Gazebo (or running it faster than
real time later, for batch evaluation) would silently desynchronise
the brain from the world.  So the launcher bridges Gazebo's /clock
into ROS and this node sets ``use_sim_time``: its 1 Hz timer counts
GAZEBO's seconds.  Pause the sim and the brain pauses with it —
try it.

RUN IT (two WSL terminals)
--------------------------
  Terminal 1:  source /opt/ros/jazzy/setup.bash
               gz sim ~/arena_seed0.sdf        # press play!
  Terminal 2:  bash /mnt/c/Users/quial/sources/Multi-UAV-ISR/gazebo/milestone3.sh

Watch for: blues spreading into a sweep, converging on reds as belief
peaks form, and (crash-avoidance policy!) giving pillars a wide berth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.agents.gnn_stage4_policy import GNNStage4Policy
from gazebo.brain import ClosedLoopBrain, env_kwargs_from_checkpoint, DT


class PolicyBridge(Node):
    """Thin ROS wrapper: odometry -> ClosedLoopBrain.tick -> cmd_vel.

    All decision logic lives in gazebo/brain.py; this class only does
    ROS I/O (subscribe odometry, publish velocities, log)."""

    def __init__(self, brain: ClosedLoopBrain):
        # use_sim_time: our timer ticks on GAZEBO's clock (see module
        # docstring).  Must be set before the timer is created.
        super().__init__(
            "policy_bridge",
            parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.brain = brain
        self.n_blue = brain.n_blue
        self.n_red = brain.n_red

        # Latest odometry position per drone: name -> pos(2,).
        self.pos: dict[str, np.ndarray] = {}
        self.spawn_checked = False
        self._spawn_blue = brain.env._blue_pos.copy()

        names = ([f"blue_{i}" for i in range(self.n_blue)]
                 + [f"red_{j}" for j in range(self.n_red)])
        for name in names:
            self.create_subscription(
                Odometry, f"/model/{name}/odometry",
                lambda msg, name=name: self._on_odom(name, msg), 10)
        self.pubs = {
            name: self.create_publisher(Twist, f"/model/{name}/cmd_vel", 10)
            for name in names
        }
        self.create_timer(DT, self._tick)
        self.get_logger().info(
            f"policy_bridge up: {self.n_blue} blues on the trained policy, "
            f"{self.n_red} reds fleeing; "
            f"episode <= {brain.env.max_steps} sim-s")

    # ------------------------------------------------------------------ #

    def _on_odom(self, name: str, msg: Odometry) -> None:
        """Remember this drone's newest position (x, y).  Odometry also
        carries velocity, but the brain uses its own integrator state
        for velocities (the env's post-step values), so we ignore it."""
        p = msg.pose.pose.position
        self.pos[name] = np.array([p.x, p.y], dtype=np.float32)

    def _publish(self, name: str, vel_xy) -> None:
        cmd = Twist()
        cmd.linear.x = float(vel_xy[0])
        cmd.linear.y = float(vel_xy[1])
        self.pubs[name].publish(cmd)

    def _stop_all(self) -> None:
        for name in self.pubs:
            self._publish(name, (0.0, 0.0))

    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        if self.brain.done:
            return
        try:
            blue_pos = np.stack([self.pos[f"blue_{i}"]
                                 for i in range(self.n_blue)])
            red_pos = np.stack([self.pos[f"red_{j}"]
                                for j in range(self.n_red)])
        except KeyError:
            self.get_logger().warning(
                "waiting for odometry... (is the sim running? press play)")
            return

        # One-time sanity check: Gazebo world and shadow env must be the
        # SAME scenario (same --seed as world_gen).
        if not self.spawn_checked:
            self.spawn_checked = True
            drift = float(np.abs(blue_pos - self._spawn_blue).max())
            if drift > 1.0:
                self.get_logger().error(
                    f"spawn mismatch up to {drift:.1f} m — was the world "
                    f"file generated with a different --seed?  Perception "
                    f"(obstacle layout!) will be wrong.")

        result = self.brain.tick(blue_pos, red_pos)
        env = self.brain.env

        for j in result.captures:
            self.get_logger().info(
                f"*** CAPTURE: red_{j} at t={env._t}s "
                f"({int(env._red_active.sum())} remaining)")

        if result.done:
            self._stop_all()
            self.get_logger().info(
                f"=== EPISODE OVER: caught {self.brain.n_caught}/{self.n_red} "
                f"in {env._t}s | obstacle-contact events "
                f"{self.brain.obs_crash_events} | ally-proximity events "
                f"{self.brain.ally_crash_events} ===")
            rclpy.shutdown()
            return

        for i in range(self.n_blue):
            self._publish(f"blue_{i}", result.blue_exec[i])
        for j in range(self.n_red):
            self._publish(f"red_{j}", result.red_exec[j])

        if env._t and env._t % 20 == 0:
            nearest = float(np.min(np.linalg.norm(
                env._red_pos[env._red_active][:, None, :]
                - env._blue_pos[None, :, :], axis=-1))) \
                if env._red_active.any() else 0.0
            self.get_logger().info(
                f"t={env._t:3d}s  active reds="
                f"{int(env._red_active.sum())}/{self.n_red}  "
                f"nearest blue-red dist={nearest:5.1f} m")


def load_policy(ckpt_path: str):
    """Load a Stage 4 checkpoint's ACTOR for deployment.  Uses the
    shape-matched soft loader: post-pool checkpoints (pool_fixed_*) copy
    77/77 tensors; older ones (crash_penalty_v3) copy 76/77 (the stale
    count-agnostic critic tensor stays at init).  Deployment runs the
    actor only, so we assert every actor tensor copied and ignore the
    critic.  Returns (policy, checkpoint_dict)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    policy = GNNStage4Policy(
        n_blue=a["n_blue"], n_red=a["n_red"], n_obs=a["n_obstacles"],
        d_hidden=a["d_hidden"], n_msg_rounds=a["n_msg_rounds"],
        use_hidden_in_gnn=a.get("share_hidden_via_gnn", True),
    )
    n_copied = policy.load_full_stage4(ckpt_path)
    n_total = len(policy.state_dict())
    print(f"Loaded {n_copied}/{n_total} tensors "
          f"(any missing are stale-shape CRITIC tensors; actor complete).")
    actor_keys = [k for k in policy.state_dict() if k.startswith("actor")]
    src = ck["policy_state"]
    n_actor_ok = sum(1 for k in actor_keys
                     if k in src and src[k].shape
                     == policy.state_dict()[k].shape)
    assert n_actor_ok == len(actor_keys), "actor tensors failed to load!"
    policy.eval()
    return policy, ck


def main() -> None:
    p = argparse.ArgumentParser(
        description="Milestone 3: trained policy drives the blues.")
    p.add_argument("--ckpt",
                   default=str(REPO / "runs/stage4/pool_fixed_v4/best.pt"),
                   help="Stage 4 checkpoint to fly.")
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match world_gen's --seed (same scenario).")
    args = p.parse_args()

    print(f"Loading checkpoint {args.ckpt} ...")
    policy, ck = load_policy(args.ckpt)
    print(f"Policy loaded: rollout {ck.get('rollout')}, "
          f"best_metric {ck.get('best_metric')}")

    # Shadow env: the training env's exact perception configuration.
    env = PursuitEnv(**env_kwargs_from_checkpoint(ck["args"]),
                     red_policy=run_from_nearest_uav, seed=args.seed)
    env.reset(seed=args.seed)
    brain = ClosedLoopBrain(env, policy)

    rclpy.init()
    node = PolicyBridge(brain)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
