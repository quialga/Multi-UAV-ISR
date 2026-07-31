"""
gazebo/kinematics.py — the env's exact motion contract, shared by the
bridge nodes.

WHY THIS FILE EXISTS
--------------------
The policy was trained inside ``PursuitEnv``, whose per-step motion
rule is (``_integrate`` + ``_clip_positions_from_obstacles``):

    1. velocity <- clip(velocity + accel * dt, ±v_max)   per axis
    2. position <- position + velocity * dt
    3. WALLS: if that position leaves the arena, clip it exactly ONTO
       the wall and zero the velocity component on that axis only —
       sliding along the wall stays free, no friction, no bounce.
    4. OBSTACLES: if the (wall-clipped) position lands inside any
       pillar disk, the move is CANCELLED entirely: position rolls
       back, the WHOLE velocity is zeroed (the env's soft-crash).

In the bridge, Gazebo executes motion — but the commands we send must
make Gazebo trace the SAME trajectory the env would have produced,
or the policy flies in a subtly different world than it trained in
(the user-visible symptom was drones wedged at walls; subtler drift
would silently degrade behaviour without any visible bug).

TWO VELOCITIES PER TICK (the non-obvious part)
----------------------------------------------
The env teleports: an agent 1.0 m from the wall moving at 1.5 m/s
lands ON the wall this step, with velocity recorded as 0.  Gazebo
moves continuously, so to arrive exactly on the wall in one tick we
must command the PARTIAL speed (1.0 m/s for one second) — while the
env-faithful velocity to remember (and to show the policy in its
observation) is 0.  Hence every tick yields:

    exec_vel  — what we PUBLISH so Gazebo lands on the env's
                next position (partial approach speeds at walls,
                full stop at obstacle-entry attempts);
    state_vel — the env's post-step velocity (what ``_integrate``
                would have stored): fed back into the next
                integration AND into the shadow env's observation.

The returned ``obstacle_hit`` mask is the env's exact
"attempted to enter a pillar" crash signal (same ``dist <= r`` test),
used for the episode report card.

``tests/test_gazebo_kinematics.py`` fuzzes multi-step trajectories of
this function against the env's own ``_integrate`` +
``_clip_positions_from_obstacles`` and requires identical positions
and velocities at every step.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def integrate_cmd(
    pos:          np.ndarray,             # (K, 2) current positions
    vel:          np.ndarray,             # (K, 2) env-state velocities
    accel:        np.ndarray,             # (K, 2) accelerations [-1, 1]
    v_max:        float,
    dt:           float,
    arena:        float,
    obstacle_pos: Optional[np.ndarray] = None,   # (N, 2)
    obstacle_r:   Optional[np.ndarray] = None,   # (N,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One env-identical step.  Returns (exec_vel, state_vel,
    obstacle_hit) — see module docstring."""
    # 1-2. per-axis velocity clip; provisional landing point.
    state = np.clip(vel + accel * dt, -v_max, v_max).astype(np.float32)
    exec_ = state.copy()
    nxt = pos + state * dt

    # 3. Walls: land exactly ON the boundary (exec = partial speed),
    #    remember velocity 0 on that axis (state), like the env.
    for axis in (0, 1):
        below = nxt[:, axis] < 0.0
        above = nxt[:, axis] > arena
        exec_[below, axis] = (0.0 - pos[below, axis]) / dt
        exec_[above, axis] = (arena - pos[above, axis]) / dt
        state[below | above, axis] = 0.0

    # 4. Obstacles: the env tests the WALL-CLIPPED landing point
    #    (dist <= r, PursuitEnv._positions_in_any_obstacle) and, on a
    #    hit, cancels the whole move and zeroes the whole velocity.
    hit = np.zeros(pos.shape[0], dtype=bool)
    if obstacle_pos is not None and len(obstacle_pos) > 0:
        land = pos + exec_ * dt
        dists = np.linalg.norm(
            land[:, None, :] - obstacle_pos[None, :, :], axis=-1)
        hit = np.any(dists <= obstacle_r[None, :], axis=1)
        exec_[hit] = 0.0
        state[hit] = 0.0

    return exec_, state, hit
