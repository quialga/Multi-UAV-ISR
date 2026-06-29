"""
isr/env/entities.py — Entity *class* metadata.

The mutable kinematic state of each entity (position, velocity, active
flag) lives inside the environment as flat numpy arrays so we can step
the whole arena in vectorised calls.  This module only holds the
immutable per-class properties — speed cap, render color, marker — that
distinguish "what kind of thing" each entity is.

Adding a new entity class (e.g. "stealth_target" with lower v_max) is
a one-line addition here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityClass:
    """
    Per-class metadata.  Frozen so callers can't accidentally mutate it
    (and so it's hashable, in case we ever index by class).

    name   -- short identifier used in logs / render legends.
    v_max  -- maximum component-wise speed (arena units / time-step).
              The same cap applies to vx and vy independently — we clip
              the velocity vector component-wise, not by L2 norm, which
              is the simpler / more common choice for grid-aligned 2D
              kinematic agents.
    color  -- matplotlib color string for rendering.
    marker -- matplotlib marker string for rendering.
    """
    name:   str
    v_max:  float
    color:  str
    marker: str


# Stage 1 entity classes.  Blue UAVs are slightly faster than red
# targets (1.5 vs 1.0) so pure pursuit is winnable in principle but
# not trivial — coordination among blue agents has to provide the
# extra edge.  See docs/design.md §3.6 for rationale.
BLUE_UAV = EntityClass(
    name="blue_uav",
    v_max=1.5,
    color="tab:blue",
    marker="^",
)

RED_TARGET = EntityClass(
    name="red_target",
    v_max=1.0,
    color="tab:red",
    marker="o",
)
