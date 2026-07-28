#!/usr/bin/env bash
# gazebo/milestone2.sh — start the translator + the red drones' brain.
#
# WHAT THIS SCRIPT STARTS (two programs):
#
# 1. ros_gz_bridge "parameter_bridge" — the TRANSLATOR.
#    Gazebo and ROS each have their own topic (mailbox) system; this
#    process copies messages between them for exactly the mailboxes we
#    name below.  The odd @-syntax reads:
#        <topic>@<ROS type>[<Gazebo type>    copy Gazebo -> ROS only
#        <topic>@<ROS type>]<Gazebo type>    copy ROS -> Gazebo only
#    We translate drone positions INTO ROS, and red velocity commands
#    OUT to Gazebo.
#
# 2. scripted_reds.py — the BRAIN (see its docstring).
#
# The bridge runs in the background; the brain in the foreground so
# its log lines are visible.  Ctrl+C stops the brain, and the trap
# below then also stops the bridge — no stray processes left behind.
#
# PREREQUISITE: the sim is already up in another terminal and PLAYING:
#     source /opt/ros/jazzy/setup.bash
#     gz sim ~/arena_seed0.sdf          # press the play button!
set -e
source /opt/ros/jazzy/setup.bash

REPO=/mnt/c/Users/quial/sources/Multi-UAV-ISR

# Per-drone odometry out of Gazebo; red velocity commands into Gazebo.
# (Why not the single all-models pose topic?  Its ROS translation
# loses the model names — verified — so we use one odometry topic per
# drone and let the topic NAME carry the identity.)
BRIDGE_TOPICS=()
for i in 0 1 2 3 4; do
  BRIDGE_TOPICS+=("/model/blue_${i}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry")
done
for j in 0 1 2; do
  BRIDGE_TOPICS+=("/model/red_${j}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry")
  BRIDGE_TOPICS+=("/model/red_${j}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist")
done

ros2 run ros_gz_bridge parameter_bridge "${BRIDGE_TOPICS[@]}" &
BRIDGE_PID=$!
trap "kill $BRIDGE_PID 2>/dev/null" EXIT

# Give the bridge a moment to connect both sides before the brain
# starts expecting pose messages.
sleep 2

# PREPEND the repo to PYTHONPATH — ROS already put rclpy's modules on
# it, and plain PYTHONPATH=$REPO would wipe them out.
PYTHONPATH="$REPO:${PYTHONPATH:-}" python3 "$REPO/gazebo/scripted_reds.py" "$@"
