"""Empirical demo: can a simple scripted pursuer capture static reds
using ONLY the Stage 4 obs (rb edges)?

Compares:
  A) OLD behaviour (cell-centre peaks only, no live measurement)
  B) NEW behaviour (continuous measurement when visible, sigma=1 m)

Controller: each blue flies toward the source of its highest-visibility
rb edge (rel_pos = blue - src, so accelerate along -rel_pos).
"""
import sys
sys.path.insert(0, r"C:\Users\quial\sources\Multi-UAV-ISR")
import numpy as np
from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red


def run(disable_measurement: bool, n_eps: int = 10, max_steps: int = 350):
    caught_all = []
    for ep in range(n_eps):
        env = PursuitEnv(
            n_blue=5, n_red=3, arena_size=130.0, capture_radius=3.0,
            sensor_radius=40.0, n_obstacles=0, use_belief_maps=True,
            belief_grid_size=26, belief_channels=2, max_steps=max_steps,
            enemy_belief_decay=0.99, enemy_belief_diffusion=0.2,
            sensor_pos_noise_std=1.0,
            red_policy=stationary_red, seed=100 + ep,
        )
        if disable_measurement:
            # Force MEMORY-ONLY tracks: every slot is a belief-map peak
            # (cell-centre position, no velocity, track_red = -1) -- the
            # pre-detection-seeded, quantised endgame.
            def _mem_only(_e=env):
                n_act = int(_e._red_active.sum())
                pos, conf = _e._extract_belief_peaks(
                    _e.n_red, channel_idx=0, k_extract=n_act,
                    nms_radius_cells=2,
                )
                return pos, conf, np.full(_e.n_red, -1, dtype=np.int32)
            env._build_enemy_tracks = _mem_only
        env.reset(seed=100 + ep)
        N = env.n_blue
        L = env.arena_size
        while env.agents:
            obs = env.structured_belief_observation()
            rb = obs["rb_edge_features"]      # (n_rb, 7)
            vis = obs["rb_edge_visible"]      # (n_rb,)
            actions = {}
            for b, agent in enumerate(env.possible_agents):
                # Edges for this blue: e = s*N + b for each track s.
                idxs = [s * N + b for s in range(env.n_red)]
                vs = vis[idxs]
                if vs.max() <= 0:
                    a = np.zeros(2, dtype=np.float32)
                else:
                    s_best = int(np.argmax(vs))
                    rel = rb[idxs[s_best], :2] * L    # blue - src
                    d = np.linalg.norm(rel)
                    a = (-rel / max(d, 1e-6)).astype(np.float32)
                actions[agent] = a
            env.step(actions)
        snap = env.state_snapshot()
        caught_all.append(int((~snap["red_active"]).sum()))
    return np.mean(caught_all), caught_all


mean_old, eps_old = run(disable_measurement=True)
mean_new, eps_new = run(disable_measurement=False)
print(f"OLD (cell-centre only): caught {mean_old:.2f}/3   per-ep {eps_old}")
print(f"NEW (measurement fix):  caught {mean_new:.2f}/3   per-ep {eps_new}")
