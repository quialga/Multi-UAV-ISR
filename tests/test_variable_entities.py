"""
tests/test_variable_entities.py — variable number of reds / obstacles
with a count-agnostic (masked-mean-pool) critic.

Design under test:
  * n_red / n_obstacles are a PADDED CAPACITY; each reset samples the
    active count in [min, capacity].  Padded reds start inactive and are
    treated exactly like caught reds everywhere (no detections, no
    capture, zeroed edges).
  * The critic pools red / obstacle node embeddings with a MASKED MEAN
    (+ active-count scalar), so V(s, i) width is fixed regardless of
    count and is INVARIANT to padded (inactive) entities — the property
    that lets one policy generalise across entity counts.

Run:
    pytest tests/test_variable_entities.py -v
"""
from __future__ import annotations

import numpy as np
import torch

from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import stationary_red
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from isr.train.vec_env import Stage4VectorPursuitEnv


def _env(**kw):
    base = dict(
        n_blue=3, n_red=6, n_obstacles=4, arena_size=130.0, max_steps=30,
        sensor_radius=40.0, use_belief_maps=True, red_policy=stationary_red,
    )
    base.update(kw)
    return PursuitEnv(**base)


# ---------------------------------------------------------------------------
# Env: per-episode count sampling + padding semantics
# ---------------------------------------------------------------------------

def test_active_count_varies_within_bounds():
    env = _env(n_red=6, n_red_min=2, n_obstacles=4, n_obstacles_min=1)
    reds, obs = set(), set()
    for s in range(20):
        env.reset(seed=s)
        na = int(env._red_active.sum())
        no = len(env._obstacle_pos)
        assert 2 <= na <= 6 and 1 <= no <= 4
        # padding contiguous: first na active, rest inactive.
        assert env._red_active[:na].all() and not env._red_active[na:].any()
        assert env._n_red_start == na
        reds.add(na)
        obs.add(no)
    assert len(reds) > 1 and len(obs) > 1   # genuinely varying


def test_fixed_count_when_min_unset_is_backward_compatible():
    env = _env(n_red=6, n_obstacles=4)   # no *_min
    for s in range(5):
        env.reset(seed=s)
        assert int(env._red_active.sum()) == 6
        assert env._n_red_start == 6


def test_padding_reds_are_invisible_to_belief_map():
    """Only 2 of 6 reds active -> the enemy belief channel must not light
    up from the 4 padded reds (they aren't in true_occupancy)."""
    env = _env(n_red=6, n_red_min=2, n_obstacles=0)
    env.reset(seed=1)
    # Force exactly 2 active for determinism.
    env._red_active[:] = False
    env._red_active[:2] = True
    env._n_red_start = 2
    truth = env._true_occupancy()          # (2, W, H)
    # Enemy channel counts at most the 2 active reds' cells.
    assert truth[0].sum() <= 2 + 1e-6


def test_caught_metric_ignores_padding():
    env = _env(n_red=6, n_red_min=3, n_obstacles=0)
    env.reset(seed=4)
    snap = env.state_snapshot()
    caught = snap["n_red_start"] - int(snap["red_active"].sum())
    assert caught == 0                     # nothing caught yet, padding excluded


# ---------------------------------------------------------------------------
# Critic: count-agnostic masked pooling
# ---------------------------------------------------------------------------

def _full_state(n_red=6, n_obs=4, n_envs=2, seed=0):
    ek = dict(n_blue=3, n_red=n_red, n_obstacles=n_obs, sensor_radius=40.0,
              use_belief_maps=True, max_steps=30)
    ve = Stage4VectorPursuitEnv(n_envs=n_envs, env_kwargs=ek,
                                red_policy_mix=[("stationary", 1.0)])
    obs = ve.reset(seed=seed)
    ot = {k: torch.from_numpy(v).float() for k, v in obs.items()}
    _, full = split_stage4_obs(ot)
    return full


def test_critic_value_width_is_count_independent():
    pol = GNNStage4Policy(n_blue=3, n_red=6, n_obs=4).eval()
    with torch.no_grad():
        v, _ = pol.critic_forward(_full_state(6, 4))
    assert v.shape == (2, 3)               # (n_envs, n_blue), independent of counts


def test_env_zeroes_inactive_red_channels():
    """The basis of padding-safety: an inactive (padded) red carries NO
    information into the critic graph beyond its 0 active-flag — its rb
    edge features are zeroed, so its position/velocity never leak in.
    Padded reds at any position therefore produce the same graph."""
    env = _env(n_red=6, n_red_min=2, n_obstacles=0)
    env.reset(seed=3)
    env._red_active[:] = False
    env._red_active[:2] = True
    env._n_red_start = 2
    base = env._build_structured_obs()
    # Node feature of inactive reds is exactly the 0 active-flag.
    assert np.allclose(base["red_features"][2:, 0], 0.0)
    # rb edges from any inactive red are fully zeroed.
    for e, r in enumerate(env.rb_edge_src):
        if not env._red_active[r]:
            assert np.allclose(base["rb_edge_features"][e], 0.0)


def test_critic_invariant_to_padded_red_positions():
    """Value must not depend on WHERE the padded reds are — the env
    zeroes their edges, so moving them cannot change V.  This is the
    invariance that makes count generalisation sound."""
    pol = GNNStage4Policy(n_blue=3, n_red=6, n_obs=4).eval()
    env = _env(n_red=6, n_red_min=2, n_obstacles=4)
    env.reset(seed=5)
    env._red_active[:] = False
    env._red_active[:2] = True
    env._n_red_start = 2

    def value_now():
        ot = {k: torch.from_numpy(v).float()[None]
              for k, v in env.structured_belief_observation().items()
              if isinstance(v, np.ndarray) and v.ndim >= 1}
        _, full = split_stage4_obs(ot)
        with torch.no_grad():
            v, _ = pol.critic_forward(full)
        return v

    v_ref = value_now()
    # Teleport the 4 inactive reds elsewhere; V must not move.
    env._red_pos[2:] = env._rng.uniform(0, env.arena_size,
                                        size=env._red_pos[2:].shape).astype(np.float32)
    v_moved = value_now()
    assert torch.allclose(v_ref, v_moved, atol=1e-5)


def test_critic_value_responds_to_active_count():
    """Fewer active reds is a genuinely different state -> the count
    scalar + masked mean let the value reflect it."""
    pol = GNNStage4Policy(n_blue=3, n_red=6, n_obs=4).eval()
    full = _full_state(6, 4)
    with torch.no_grad():
        v_all, _ = pol.critic_forward(full)
        fewer = {k: v.clone() for k, v in full.items()}
        fewer["red_features"][:, 3:, 0] = 0.0      # drop 3 reds
        v_few, _ = pol.critic_forward(fewer)
    assert (v_all - v_few).abs().max() > 1e-6


def test_critic_handles_zero_active_reds():
    pol = GNNStage4Policy(n_blue=3, n_red=6, n_obs=4).eval()
    full = _full_state(6, 4)
    full = {k: v.clone() for k, v in full.items()}
    full["red_features"][:, :, 0] = 0.0            # all reds inactive
    with torch.no_grad():
        v, _ = pol.critic_forward(full)
    assert torch.isfinite(v).all()                 # no div-by-zero blow-up
