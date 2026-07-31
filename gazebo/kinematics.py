"""
gazebo/kinematics.py — the env's exact motion arithmetic, shared by
the bridge nodes.

WHY THIS FILE EXISTS
--------------------
The training env moves agents with one rule
(``PursuitEnv._integrate``):

    velocity <- clip(velocity + accel * dt, -v_max, +v_max)   per axis
    position <- position + velocity * dt
    WALLS: if the position update would leave the arena, clip the
    position to the wall and ZERO THE VELOCITY COMPONENT ON THAT AXIS
    ONLY.  The other component survives — an agent pressed against a
    wall can still slide along it, and can accelerate away next tick.

In Gazebo, the wall itself stops the drone (it is a physical object),
but our COMMANDED velocity must still follow the env's rule.  Without
it, a drone told to flee into a wall keeps being commanded at full
speed INTO the wall forever: the env would have zeroed that component
at first contact.  Pressing into the wall also caused a second,
Gazebo-only failure — contact friction gripping the drone so hard it
could not even slide sideways (the "stuck at the wall" bug).  The
friction half of the fix lives in world_gen.py (frictionless contact
surfaces, matching the env, which has no friction concept at all);
the command half lives here.
"""
from __future__ import annotations

import numpy as np


def integrate_cmd(
    pos:   np.ndarray,   # (K, 2) current positions (from odometry)
    vel:   np.ndarray,   # (K, 2) current COMMANDED velocities
    accel: np.ndarray,   # (K, 2) accelerations in [-1, 1]
    v_max: float,
    dt:    float,
    arena: float,
) -> np.ndarray:
    """One env-identical integration step for the commanded velocity:
    per-axis v_max clip, then the env's wall-stop rule (zero the
    offending component where the next position would cross a wall).
    Returns the new commanded velocity (K, 2)."""
    new_vel = np.clip(vel + accel * dt, -v_max, v_max).astype(np.float32)
    next_pos = pos + new_vel * dt
    for axis in (0, 1):
        out = (next_pos[:, axis] < 0.0) | (next_pos[:, axis] > arena)
        new_vel[out, axis] = 0.0
    return new_vel
