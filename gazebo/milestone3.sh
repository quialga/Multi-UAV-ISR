#!/usr/bin/env bash
# gazebo/milestone3.sh — translator + the FULL closed loop:
# trained policy on the blues, flee heuristic on the reds, referee.
#
# Same wiring pattern as milestone2.sh, three additions:
#   - the 5 blue cmd_vel channels (the trained policy now drives them),
#   - Gazebo's /clock into ROS, so the brain ticks on SIMULATED time
#     (pause the sim and the brain pauses; run the sim faster and the
#     brain keeps perfect 1-decision-per-sim-second step),
#   - policy_bridge.py instead of scripted_reds.py (see its docstring).
#
# PREREQUISITE: the sim is up in another terminal and PLAYING:
#     source /opt/ros/jazzy/setup.bash
#     gz sim ~/arena_seed0.sdf          # press the play button!
set -e
source /opt/ros/jazzy/setup.bash

REPO=/mnt/c/Users/quial/sources/Multi-UAV-ISR

BRIDGE_TOPICS=("/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock")
for i in 0 1 2 3 4; do
  BRIDGE_TOPICS+=("/model/blue_${i}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry")
  BRIDGE_TOPICS+=("/model/blue_${i}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist")
done
for j in 0 1 2; do
  BRIDGE_TOPICS+=("/model/red_${j}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry")
  BRIDGE_TOPICS+=("/model/red_${j}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist")
done

ros2 run ros_gz_bridge parameter_bridge "${BRIDGE_TOPICS[@]}" &
BRIDGE_PID=$!
trap "kill $BRIDGE_PID 2>/dev/null" EXIT
sleep 2

# Prepend the repo (ROS already put rclpy on PYTHONPATH).
PYTHONPATH="$REPO:${PYTHONPATH:-}" python3 "$REPO/gazebo/policy_bridge.py" "$@"
