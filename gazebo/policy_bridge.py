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

ONE TICK, STEP BY STEP (the loop below, in order)
-------------------------------------------------
    1. SYNC     copy positions+velocities from Gazebo odometry into
                the shadow env (Gazebo replaced the env's kinematics,
                so this is the env's "movement" phase).
    2. REFEREE  the env's own capture rule: any red within 3 m of any
                blue is caught -> marked inactive, its belief blob is
                wiped (exactly what env.step does).  Also counts
                obstacle / ally near-collisions for the report card.
    3. PERCEIVE run the env's belief-map update (noisy sweeps,
                occlusion, decay) and build the graph observation.
    4. THINK    the frozen network, deterministically (no exploration
                noise) — same call as training's evaluation:
                act_deterministic(observation, memory).
    5. MEMORY   the network is recurrent: it carries a small hidden
                state ("what I remember from previous seconds") that
                we pass back in every tick and reset only at episode
                start.  Forgetting to carry this would silently
                lobotomise the policy.
    6. ACT      the network outputs accelerations; integrate them into
                velocities with the env's exact rule (dt = 1 s, per-
                axis clip at 1.5 m/s) and publish one cmd_vel per blue.
                Reds get their milestone-2 flee commands (caught reds
                get zero — they freeze, like the env's frozen reds).
    7. CLOCK    when all reds are caught (or 200 s pass), stop every
                drone, print the episode report, and exit.

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
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from gazebo.kinematics import integrate_cmd

# The env's kinematic contract (isr/env/entities.py + PursuitEnv
# defaults).  BLUE_UAV.v_max = 1.5, RED_TARGET.v_max = 1.0, dt = 1.0.
DT         = 1.0
V_MAX_BLUE = 1.5
V_MAX_RED  = 1.0


def env_kwargs_from_checkpoint(a: dict) -> dict:
    """Rebuild the training env's configuration from the checkpoint's
    saved arguments, so the shadow env perceives EXACTLY like the env
    the policy was trained in (same noise rates, same belief decay,
    same grid...).  Mirrors scripts/train_stage4.py's env_kwargs."""
    return dict(
        n_blue                   = a["n_blue"],
        n_red                    = a["n_red"],
        arena_size               = a["arena_size"],
        max_steps                = a["max_steps"],
        capture_radius           = a["capture_radius"],
        sensor_radius            = a["sensor_radius"],
        n_obstacles              = a["n_obstacles"],
        obstacle_radius_min      = a["obstacle_radius_min"],
        obstacle_radius_max      = a["obstacle_radius_max"],
        obstacle_spawn_clearance = a["obstacle_spawn_clearance"],
        use_belief_maps          = True,
        belief_grid_size         = a["belief_grid_size"],
        belief_channels          = a["belief_channels"],
        belief_clip              = a["belief_clip"],
        p_TP                     = a["p_tp"],
        p_FP                     = a["p_fp"],
        ray_step_size            = a["ray_step_size"],
        enemy_belief_decay       = a.get("enemy_belief_decay", 1.0),
        enemy_belief_diffusion   = a.get("enemy_belief_diffusion", 0.0),
        sensor_pos_noise_std     = a.get("sensor_pos_noise_std", 1.0),
    )


class PolicyBridge(Node):
    """One ROS node closing the whole loop at 1 (simulated) Hz."""

    def __init__(self, env: PursuitEnv, policy: GNNStage4Policy):
        # use_sim_time: our timer ticks on GAZEBO's clock (see module
        # docstring).  Must be set before the timer is created.
        super().__init__(
            "policy_bridge",
            parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.env = env
        self.policy = policy
        self.n_blue = env.n_blue
        self.n_red = env.n_red

        # The network's recurrent memory (batch of 1 "env" = Gazebo).
        self.hidden = policy.initial_hidden(1, torch.device("cpu"))

        # Commanded velocities = the drones' kinematic state.  The env
        # integrates acceleration into velocity; we do it here (Gazebo
        # then executes the velocity, closing the double-integrator).
        self.blue_vel = np.zeros((self.n_blue, 2), dtype=np.float32)
        self.red_vel = np.zeros((self.n_red, 2), dtype=np.float32)

        # Latest odometry per drone: name -> (pos(2,), vel(2,)).
        self.odom: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Report-card counters (rising-edge events, like the training
        # eval): a blue parked against a pillar counts one crash, not
        # one per second.
        self.prev_obs_mask = np.zeros(self.n_blue, dtype=bool)
        self.prev_ally_mask = np.zeros(self.n_blue, dtype=bool)
        self.obs_crash_events = 0
        self.ally_crash_events = 0
        self.spawn_checked = False
        self.done = False

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
            f"{self.n_red} reds fleeing; episode <= {env.max_steps} sim-s")

    # ------------------------------------------------------------------ #

    def _on_odom(self, name: str, msg: Odometry) -> None:
        """Remember this drone's newest position and velocity.

        Note on frames: odometry velocity is expressed in the drone's
        OWN body frame.  Our drones never rotate (we command no yaw),
        so body frame == world frame and no conversion is needed."""
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.odom[name] = (
            np.array([p.x, p.y], dtype=np.float32),
            np.array([v.x, v.y], dtype=np.float32),
        )

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
        if self.done:
            return
        env = self.env

        # ---- 1. SYNC: Gazebo's reality -> shadow env state ------------
        try:
            blue = [self.odom[f"blue_{i}"] for i in range(self.n_blue)]
            red = [self.odom[f"red_{j}"] for j in range(self.n_red)]
        except KeyError:
            self.get_logger().warning(
                "waiting for odometry... (is the sim running? press play)")
            return
        env._blue_pos = np.stack([p for p, _ in blue]).astype(np.float32)
        env._blue_vel = np.stack([v for _, v in blue]).astype(np.float32)
        env._red_pos = np.stack([p for p, _ in red]).astype(np.float32)
        # Caught reds are frozen in the env; keep their velocity zero.
        red_v = np.stack([v for _, v in red]).astype(np.float32)
        red_v[~env._red_active] = 0.0
        env._red_vel = red_v

        # One-time sanity check: the Gazebo world and the shadow env
        # must describe the SAME scenario (same --seed as world_gen).
        if not self.spawn_checked:
            self.spawn_checked = True
            drift = float(np.abs(env._blue_pos
                                 - self._spawn_blue).max())
            if drift > 1.0:
                self.get_logger().error(
                    f"spawn mismatch up to {drift:.1f} m — was the world "
                    f"file generated with a different --seed?  Perception "
                    f"(obstacle layout!) will be wrong.")

        # ---- 2. REFEREE: the env's own capture rule --------------------
        # (Mirrors PursuitEnv.step step-5 exactly: nearest-blue distance
        # <= capture_radius -> caught, freeze, wipe the belief blob.)
        if env._red_active.any():
            dists = np.linalg.norm(
                env._red_pos[:, None, :] - env._blue_pos[None, :, :],
                axis=-1)
            newly = env._red_active & (dists.min(axis=1)
                                       <= env.capture_radius)
            if newly.any():
                env._red_active &= ~newly
                env._red_vel[newly] = 0.0
                if env._belief_maps is not None:
                    env._clear_enemy_belief_at(env._red_pos[newly])
                for j in np.flatnonzero(newly):
                    self.get_logger().info(
                        f"*** CAPTURE: red_{j} at t={env._t}s "
                        f"({int(env._red_active.sum())} remaining)")

        # Report card: obstacle / ally proximity events (approximate —
        # Gazebo blocks entry by contact where the env rolls back, so
        # we count "touching" instead; rising edges = events).
        d_obs = np.linalg.norm(
            env._blue_pos[:, None, :] - env._obstacle_pos[None, :, :],
            axis=-1) - env._obstacle_r[None, :]
        obs_mask = (d_obs <= 0.6).any(axis=1)
        bb = np.linalg.norm(
            env._blue_pos[:, None, :] - env._blue_pos[None, :, :], axis=-1)
        np.fill_diagonal(bb, np.inf)
        ally_mask = (bb <= 2.0).any(axis=1)
        self.obs_crash_events += int(np.count_nonzero(
            obs_mask & ~self.prev_obs_mask))
        self.ally_crash_events += int(np.count_nonzero(
            ally_mask & ~self.prev_ally_mask))
        self.prev_obs_mask, self.prev_ally_mask = obs_mask, ally_mask

        # ---- Episode clock / termination (env.step step-7) -------------
        env._t += 1
        all_caught = not env._red_active.any()
        if all_caught or env._t >= env.max_steps:
            self._stop_all()
            self.done = True
            n = self.n_red - int(env._red_active.sum())
            self.get_logger().info(
                f"=== EPISODE OVER: caught {n}/{self.n_red} in {env._t}s | "
                f"obstacle-contact events {self.obs_crash_events} | "
                f"ally-proximity events {self.ally_crash_events} ===")
            rclpy.shutdown()
            return

        # ---- 3. PERCEIVE: the env's own sensor + belief pipeline -------
        env._update_belief_maps()
        obs = env.structured_belief_observation()

        # ---- 4-5. THINK with MEMORY: same call as training's eval ------
        obs_t = {k: torch.from_numpy(v).float().unsqueeze(0)
                 for k, v in obs.items()}
        partial_obs, _ = split_stage4_obs(obs_t)
        with torch.no_grad():
            mean, self.hidden = self.policy.act_deterministic(
                partial_obs, self.hidden)
        accel = mean.squeeze(0).numpy().astype(np.float32)
        accel = np.clip(accel, -1.0, 1.0)          # env clips actions too

        # ---- 6. ACT: integrate accel -> vel (env's exact rule, incl.
        # the wall-stop: zero the component that would push past a
        # wall — see gazebo/kinematics.py) ------------------------------
        self.blue_vel = integrate_cmd(env._blue_pos, self.blue_vel,
                                      accel, V_MAX_BLUE, DT,
                                      env.arena_size)
        for i in range(self.n_blue):
            self._publish(f"blue_{i}", self.blue_vel[i])

        # Reds: milestone-2 flee logic, now honouring the referee.
        red_a = run_from_nearest_uav(
            env._blue_pos, env._red_pos, env._red_active,
            env._obstacle_pos, env._obstacle_r)
        red_a = np.clip(red_a.astype(np.float32), -1.0, 1.0)
        self.red_vel = integrate_cmd(env._red_pos, self.red_vel,
                                     red_a, V_MAX_RED, DT,
                                     env.arena_size)
        self.red_vel[~env._red_active] = 0.0       # caught reds freeze
        for j in range(self.n_red):
            self._publish(f"red_{j}", self.red_vel[j])

        if env._t % 20 == 0:
            nearest = float(np.min(np.linalg.norm(
                env._red_pos[env._red_active][:, None, :]
                - env._blue_pos[None, :, :], axis=-1))) \
                if env._red_active.any() else 0.0
            self.get_logger().info(
                f"t={env._t:3d}s  active reds="
                f"{int(env._red_active.sum())}/{self.n_red}  "
                f"nearest blue-red dist={nearest:5.1f} m")

    # Stashed at construction for the spawn sanity check.
    _spawn_blue: np.ndarray = None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Milestone 3: trained policy drives the blues.")
    p.add_argument("--ckpt",
                   default=str(REPO / "runs/stage4/pool_fixed_v1/best.pt"),
                   help="Stage 4 checkpoint to fly.")
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match world_gen's --seed (same scenario).")
    args = p.parse_args()

    print(f"Loading checkpoint {args.ckpt} ...")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]

    # Shadow env: the training env's exact perception configuration.
    env = PursuitEnv(**env_kwargs_from_checkpoint(a),
                     red_policy=run_from_nearest_uav, seed=args.seed)
    env.reset(seed=args.seed)

    # The frozen brain, rebuilt with the trained architecture.  Loaded
    # via the repo's shape-matched soft loader rather than a strict
    # load: the crash_penalty_v3 checkpoint predates the count-agnostic
    # critic pool (critic_trunk.0.weight 512 -> 194), so that ONE critic
    # tensor can't copy.  Deployment only runs the ACTOR
    # (act_deterministic), so the critic is irrelevant here — but we
    # still verify the actor copied completely.
    policy = GNNStage4Policy(
        n_blue=a["n_blue"], n_red=a["n_red"], n_obs=a["n_obstacles"],
        d_hidden=a["d_hidden"], n_msg_rounds=a["n_msg_rounds"],
        use_hidden_in_gnn=a.get("share_hidden_via_gnn", True),
    )
    n_copied = policy.load_full_stage4(args.ckpt)
    n_total = len(policy.state_dict())
    print(f"Loaded {n_copied}/{n_total} tensors "
          f"(the missing ones are stale-shape CRITIC tensors; "
          f"the actor is complete).")
    actor_keys = [k for k in policy.state_dict() if k.startswith("actor")]
    src = ck["policy_state"]
    n_actor_ok = sum(1 for k in actor_keys
                     if k in src and src[k].shape
                     == policy.state_dict()[k].shape)
    assert n_actor_ok == len(actor_keys), "actor tensors failed to load!"
    policy.eval()
    print(f"Policy loaded: rollout {ck.get('rollout')}, "
          f"best_metric {ck.get('best_metric')}")

    rclpy.init()
    node = PolicyBridge(env, policy)
    node._spawn_blue = env._blue_pos.copy()
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
