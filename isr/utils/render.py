"""
isr/utils/render.py — Matplotlib renderer for ``PursuitEnv``.

Two entry points:

- ``render_frame(ax, snap, arena_size, ...)`` — draws a single frame
  into a matplotlib axis.  The smallest unit we can reuse from tests,
  notebooks, or a custom animation loop.
- ``animate_episode(snaps, arena_size, save_path=None, ...)`` — wraps a
  list of state snapshots in a ``FuncAnimation`` and either displays
  interactively (``save_path=None``) or saves to GIF via Pillow.

Snapshots are the dicts returned by ``PursuitEnv.state_snapshot()``.
We don't render straight from the env — collect snapshots during the
rollout and pass the list here.  That keeps rendering decoupled from
the env step loop (so the same renderer works for the smoke test, the
demo script, and post-training visualisations).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Matplotlib is imported lazily inside the functions so unit tests that
# don't render can run without a display.

from isr.env.entities import BLUE_UAV, RED_TARGET


def render_frame(
    ax,
    snap:          Dict[str, Any],
    arena_size:    float,
    trail_snaps:   Optional[List[Dict[str, Any]]] = None,
    title:         str = "",
    velocity_scale: float = 5.0,
) -> None:
    """
    Draw one frame.

    Parameters
    ----------
    ax           Matplotlib axis to draw into.  The function clears it
                 before drawing, so the same axis can be reused across
                 frames (this is how ``FuncAnimation`` calls us).
    snap         State snapshot dict from ``PursuitEnv.state_snapshot()``.
    arena_size   Side length of the arena box (sets axis limits).
    trail_snaps  Optional list of previous snapshots; the most recent
                 are drawn faded behind the current frame to show recent
                 motion.  Pass an empty list or None to disable trails.
    title        Title text rendered above the frame.
    velocity_scale  Multiplier on velocity arrows for visibility.
                    Default 5x (a v=1 vector renders as a 5-unit arrow).
    """
    ax.clear()
    ax.set_xlim(0.0, arena_size)
    ax.set_ylim(0.0, arena_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_title(title, fontsize=10)

    # ---- Trails (oldest faintest, recent stronger) ------------------------
    if trail_snaps:
        n_trail = len(trail_snaps)
        for i, ts in enumerate(trail_snaps):
            # alpha ramps from ~0.1 (oldest) to ~0.4 (most recent in trail)
            alpha = 0.1 + 0.3 * (i / max(n_trail - 1, 1))
            ax.scatter(
                ts["blue_pos"][:, 0], ts["blue_pos"][:, 1],
                c=BLUE_UAV.color, marker=BLUE_UAV.marker,
                s=15, alpha=alpha, edgecolor="none",
            )
            active_then = ts["red_active"]
            if active_then.any():
                ax.scatter(
                    ts["red_pos"][active_then, 0],
                    ts["red_pos"][active_then, 1],
                    c=RED_TARGET.color, marker=RED_TARGET.marker,
                    s=15, alpha=alpha, edgecolor="none",
                )

    # ---- Current positions ------------------------------------------------
    blue_pos = snap["blue_pos"]
    red_pos  = snap["red_pos"]
    active   = snap["red_active"]

    ax.scatter(
        blue_pos[:, 0], blue_pos[:, 1],
        c=BLUE_UAV.color, marker=BLUE_UAV.marker, s=160,
        edgecolor="black", linewidth=1.2,
        label=f"Blue UAV (×{len(blue_pos)})", zorder=10,
    )
    if active.any():
        ax.scatter(
            red_pos[active, 0], red_pos[active, 1],
            c=RED_TARGET.color, marker=RED_TARGET.marker, s=120,
            edgecolor="black", linewidth=1.2,
            label=f"Red active (×{int(active.sum())})", zorder=10,
        )
    if (~active).any():
        ax.scatter(
            red_pos[~active, 0], red_pos[~active, 1],
            c="gray", marker="x", s=120, alpha=0.6,
            label=f"Caught (×{int((~active).sum())})", zorder=5,
        )

    # ---- Velocity arrows on blue UAVs ------------------------------------
    blue_vel = snap["blue_vel"]
    for i in range(len(blue_pos)):
        v = blue_vel[i]
        if float(np.linalg.norm(v)) > 1e-3:
            ax.arrow(
                blue_pos[i, 0], blue_pos[i, 1],
                v[0] * velocity_scale, v[1] * velocity_scale,
                head_width=1.2, head_length=1.6, length_includes_head=True,
                fc=BLUE_UAV.color, ec=BLUE_UAV.color, alpha=0.7, zorder=8,
            )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def animate_episode(
    snaps:       List[Dict[str, Any]],
    arena_size:  float,
    save_path:   Optional[str] = None,
    fps:         int           = 10,
    trail_len:   int           = 5,
    figsize:     tuple         = (8, 8),
):
    """
    Render a list of snapshots as an animation.

    Parameters
    ----------
    snaps        Episode trajectory: one snapshot per simulation step
                 (typically ``[env.state_snapshot()] + [env.state_snapshot()
                 after each step]``).
    arena_size   Side length of the arena (passed to render_frame).
    save_path    If given, write a GIF to this path (Pillow writer).  If
                 None, display interactively via ``plt.show()`` — blocks
                 until the window is closed.
    fps          Animation frame rate.  GIF size scales linearly with
                 episode length and fps.
    trail_len    How many previous frames to draw as a fading trail
                 behind the current frame.
    figsize      Matplotlib figure size in inches.

    Returns the FuncAnimation object (useful in notebook contexts).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=figsize)
    n_red_total = len(snaps[0]["red_active"])

    def update(frame_idx: int):
        snap = snaps[frame_idx]
        n_caught = int((~snap["red_active"]).sum())
        title = (
            f"t = {snap['t']:>3d}   "
            f"caught {n_caught}/{n_red_total}   "
            f"(frame {frame_idx + 1}/{len(snaps)})"
        )
        trail_start = max(0, frame_idx - trail_len)
        trail = snaps[trail_start:frame_idx]
        render_frame(ax, snap, arena_size, trail_snaps=trail, title=title)

    anim = FuncAnimation(
        fig, update, frames=len(snaps),
        interval=int(1000 / fps), repeat=False,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
    else:
        plt.show()

    return anim
