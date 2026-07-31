"""
tests/test_bridge_red_faithfulness.py — guard against red-adversary
drift between the training env and the Gazebo bridge.

The bug this locks out: the bridge nodes used to call the scripted red
evader with 5 args (no ``arena_size``), so the Gazebo reds ran WITHOUT
the wall repulsion the training env applies — they fled straight into
walls and pinned, diverging from the adversary the policy trained
against.  See docs/stage4_backlog.md "Red wall-repulsion".

The bridge module imports rclpy/torch (WSL-only), so we can't import
and call it here.  Instead we assert two things that together
guarantee faithfulness:

1. The behavioural contract: with ``arena_size`` the scripted red is
   steered AWAY from a nearby wall; without it, straight into the wall.
   (If this ever flips, the "pass arena_size" requirement is moot.)
2. The source contract: the bridge's red-command call passes an
   ``arena_size`` argument.  A cheap, formatting-tolerant check that
   fails loudly if someone reverts the call to the 5-arg form.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from isr.env.pursuit_env import run_from_nearest_uav

GAZEBO = Path(__file__).resolve().parent.parent / "gazebo"
A = 130.0


def test_arena_size_is_what_prevents_wall_pinning():
    """The behavioural reason the bridge must pass arena_size: near the
    right wall, the flee command flips from +x (into wall) to -x (away)
    once arena_size is supplied."""
    red_pos = np.array([[A - 3.0, A / 2]], dtype=np.float32)
    blue_pos = np.array([[A - 40.0, A / 2]], dtype=np.float32)
    active = np.array([True])
    into = run_from_nearest_uav(blue_pos, red_pos, active, None, None, None)
    away = run_from_nearest_uav(blue_pos, red_pos, active, None, None, A)
    assert into[0, 0] > 0.99      # 5-arg form: straight into the wall
    assert away[0, 0] < 0.0       # 6-arg form: steered back inward


def _red_policy_call_passes_arena_size(path: Path) -> bool:
    """True iff the file contains a call to ``run_from_nearest_uav`` or
    ``<x>.red_policy`` that includes an ``arena_size`` argument (by name
    or as an ``*.arena_size`` attribute in its args)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        is_red = (
            (isinstance(f, ast.Name) and f.id == "run_from_nearest_uav")
            or (isinstance(f, ast.Attribute) and f.attr == "red_policy")
        )
        if not is_red:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Attribute) and arg.attr == "arena_size":
                return True
        for kw in node.keywords:
            if kw.arg == "arena_size":
                return True
    return False


def test_policy_bridge_red_call_passes_arena_size():
    assert _red_policy_call_passes_arena_size(GAZEBO / "policy_bridge.py")
