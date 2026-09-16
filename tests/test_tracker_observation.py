"""
tests/test_tracker_observation.py — the trackers as the actor's
observation, end to end (PursuitEnv actor_obs="tracker", the vector env,
the policy and the checkpoint loader).

What must hold:
* belief mode is the untouched control (no trackers, same sizing);
* tracker mode has fixed actor slots, and the critic's ground-truth keys
  are exactly those of belief mode;
* nothing the sensors did not report reaches the actor — an undetectable
  red changes nothing, a seen obstacle's radius is the tracker's estimate;
* the actor path never touches ground-truth obstacles: the learned motion
  model's context and the coverage occlusion test use the obstacle
  tracker's estimates;
* trackers advance once per env step, not once per observation call.

Run:
    pytest tests/test_tracker_observation.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import isr.env.pursuit_env as pe
from isr.agents.gnn_stage4_policy import GNNStage4Policy, split_stage4_obs
from isr.agents.heuristics import stationary_red
from isr.agents.policy_loader import _actor_sizing, env_kwargs_from_checkpoint
from isr.env.pursuit_env import PursuitEnv
from isr.tracking import actor_graph
from isr.train.vec_env import Stage4VectorPursuitEnv

BASE = dict(n_blue=3, n_red=2, n_obstacles=3, arena_size=200.0, max_steps=60,
            capture_radius=3.0, sensor_radius=40.0, p_TP=0.85, p_FP=0.15,
            sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
            sensor_noise_range_growth=1.0, track_conf_min=0.5,
            track_occlusion=True, track_detection=True,
            red_policy=stationary_red)
ACTOR_KEYS = ("red_features", "rb_edge_features", "rb_edge_visible",
              "obstacle_features", "ob_edge_features", "ob_edge_visible")
CRITIC_KEYS = ("blue_features", "bb_edge_features", "bb_edge_visible",
               "true_red_features", "true_rb_edge_features",
               "true_obstacle_features", "true_ob_edge_features")


def _env(mode="tracker", seed=3, **kw):
    cfg = dict(BASE, actor_obs=mode, use_belief_maps=(mode == "belief"),
               seed=seed)
    cfg.update(kw)
    e = PursuitEnv(**cfg)
    e.reset(seed=seed)
    return e


def _place(env, blue_pos, red_pos=None):
    """Put blues (still) and reds where a test needs them, and restart the
    trackers so nothing seen from the random spawn carries over."""
    env._blue_pos = np.asarray(blue_pos, dtype=np.float32)
    env._blue_vel = np.zeros_like(env._blue_pos)
    if red_pos is not None:
        env._red_pos = np.asarray(red_pos, dtype=np.float32)
        env._red_vel = np.zeros_like(env._red_pos)
    env._red_tracker = actor_graph.make_red_tracker(env._red_motion)
    env._obstacle_tracker = actor_graph.make_obstacle_tracker(static=True)


def _still(env):
    return {a: np.zeros(2, dtype=np.float32) for a in env.agents}


# --------------------------------------------------------------------- #
#  Modes and sizing
# --------------------------------------------------------------------- #

def test_belief_mode_is_the_default_and_builds_no_trackers():
    e = PursuitEnv(**dict(BASE, use_belief_maps=True, seed=0))
    e.reset(seed=0)
    assert e.actor_obs == "belief" and e._red_tracker is None
    assert (e.actor_n_red, e.actor_n_obstacles) == (e.n_red, e.n_obstacles)
    assert (e.actor_red_feat_dim, e.actor_obs_feat_dim) == (4, 5)


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError):
        PursuitEnv(**dict(BASE, actor_obs="map"))
    with pytest.raises(ValueError):
        PursuitEnv(**dict(BASE, actor_obs="belief", red_motion_ckpt="x.pt"))
    with pytest.raises(ValueError):
        PursuitEnv(**dict(BASE, sensor_radius=None, actor_obs="tracker"))


def test_tracker_mode_shapes_and_unchanged_critic_keys():
    t = _env("tracker", tracker_red_slots=5, tracker_obstacle_slots=6)
    b = _env("belief")
    ot, ob = t.structured_belief_observation(), b.structured_belief_observation()
    N = t.n_blue
    assert ot["red_features"].shape == (5, actor_graph.RED_FEAT_DIM)
    assert ot["rb_edge_features"].shape == (5 * N, 7)
    assert ot["rb_edge_visible"].shape == (5 * N,)
    assert ot["obstacle_features"].shape == (6, actor_graph.OBS_FEAT_DIM)
    assert ot["ob_edge_features"].shape == (6 * N, 7)
    assert set(np.unique(ot["rb_edge_visible"])) <= {0.0, 1.0}
    # Reset draws positions before any detection, so the ground truth is
    # the same state in both modes.
    for k in CRITIC_KEYS:
        np.testing.assert_array_equal(ot[k], ob[k], err_msg=k)


def test_no_obstacles_means_no_obstacle_keys():
    e = _env("tracker", n_obstacles=0)
    o = e.structured_belief_observation()
    assert "obstacle_features" not in o and e.actor_n_obstacles == 0


# --------------------------------------------------------------------- #
#  Nothing unreported reaches the actor
# --------------------------------------------------------------------- #

def test_an_undetectable_red_changes_nothing_the_actor_sees():
    obs = []
    for far in ((185.0, 185.0), (150.0, 190.0)):
        e = _env("tracker", n_obstacles=0, clutter_rate=0.0)
        _place(e, [[20, 20], [30, 25], [25, 35]], [[40.0, 20.0], far])
        for _ in range(8):
            e.step(_still(e))
        obs.append(e.structured_belief_observation())
    for k in ("red_features", "rb_edge_features", "rb_edge_visible"):
        np.testing.assert_array_equal(obs[0][k], obs[1][k], err_msg=k)
    assert obs[0]["red_features"][:, 0].sum() == 1.0, "only the seen red"


def test_a_seen_obstacle_shows_its_estimated_radius_not_the_true_one():
    e = _env("tracker", n_obstacles=1, obstacle_radius_noise_std=2.0,
             clutter_rate=0.0)
    c, r = e._obstacle_pos[0], float(e._obstacle_r[0])
    _place(e, [c + (r + 12, 0), c + (0, r + 12), c - (r + 12, 0)],
           [[5.0, 5.0], [195.0, 5.0]])
    for _ in range(10):
        e.step(_still(e))
    o = e.structured_belief_observation()
    (tr,) = e._obstacle_tracker.confirmed_tracks()
    assert o["obstacle_features"][:, 0].sum() == 1.0
    assert o["obstacle_features"][0, 1] == pytest.approx(tr.radius / e.arena_size)
    assert tr.radius != pytest.approx(r, abs=1e-3), "noise must reach the estimate"


# --------------------------------------------------------------------- #
#  The actor path uses estimated obstacles
# --------------------------------------------------------------------- #

class _StubMotion:
    """A constant-velocity motion model that records its context."""

    def __init__(self):
        self.obs_r = []

    def set_context(self, blue_pos, blue_vel, obs_pos=None, obs_vel=None,
                    obs_r=None):
        self.obs_r.append(None if obs_r is None else np.array(obs_r))

    def __call__(self, x, P):
        F = np.eye(4); F[0, 2] = F[1, 3] = 1.0
        return [(1.0, F @ x, F @ P @ F.T + 0.5 * np.eye(4))]


def test_motion_model_context_is_the_obstacle_tracker_estimate():
    e = _env("tracker", n_obstacles=1, obstacle_radius_noise_std=2.0,
             clutter_rate=0.0)
    e._red_motion = _StubMotion()
    c, r = e._obstacle_pos[0], float(e._obstacle_r[0])
    _place(e, [c + (r + 12, 0), c + (0, r + 12), c - (r + 12, 0)])
    for _ in range(10):
        e.step(_still(e))
    _, _, est_r = actor_graph.confirmed_obstacle_geometry(e._obstacle_tracker.tracks)
    assert len(est_r) == 1
    np.testing.assert_array_equal(e._red_motion.obs_r[-1], est_r)
    assert e._red_motion.obs_r[0] is None, "nothing confirmed on the first scan"


def test_coverage_occlusion_uses_the_obstacle_tracker_estimate(monkeypatch):
    seen = []
    real = pe.disk_occluder

    def spy(centres, radii, margin):
        seen.append((np.array(centres), np.array(radii)))
        return real(centres, radii, margin)

    monkeypatch.setattr(pe, "disk_occluder", spy)
    e = _env("tracker", n_obstacles=1, obstacle_radius_noise_std=2.0)
    c, r = e._obstacle_pos[0], float(e._obstacle_r[0])
    _place(e, [c + (r + 12, 0), c + (0, r + 12), c - (r + 12, 0)])
    for _ in range(10):
        e.step(_still(e))
    est_c, _, est_r = actor_graph.confirmed_obstacle_geometry(
        e._obstacle_tracker.tracks)
    assert seen, "occlusion from estimates was never built"
    np.testing.assert_array_equal(seen[-1][0], est_c)
    np.testing.assert_array_equal(seen[-1][1], est_r)


# --------------------------------------------------------------------- #
#  Trackers advance in step() only
# --------------------------------------------------------------------- #

def test_observation_calls_do_not_advance_the_trackers():
    e = _env("tracker")
    for _ in range(5):
        e.step(_still(e))
    t_red, t_obs = e._red_tracker.t, e._obstacle_tracker.t
    o1 = e.structured_belief_observation()
    o2 = e.structured_belief_observation()
    assert (e._red_tracker.t, e._obstacle_tracker.t) == (t_red, t_obs) == (6, 6)
    for k in ACTOR_KEYS:
        np.testing.assert_array_equal(o1[k], o2[k], err_msg=k)


def test_reset_starts_fresh_trackers_with_the_first_scan():
    e = _env("tracker")
    for _ in range(5):
        e.step(_still(e))
    old = e._red_tracker
    e.reset(seed=9)
    assert e._red_tracker is not old
    assert e._red_tracker.t == 1 and e._obstacle_tracker.t == 1


# --------------------------------------------------------------------- #
#  Vector env, policy, checkpoint loader
# --------------------------------------------------------------------- #

def _vec(mode, n_envs=2):
    kw = {k: v for k, v in BASE.items() if k != "red_policy"}
    kw.update(actor_obs=mode, use_belief_maps=(mode == "belief"))
    return Stage4VectorPursuitEnv(n_envs=n_envs, env_kwargs=kw, base_seed=0,
                                  red_policy_mix=[("stationary", 1.0)])


def _policy(ve, **override):
    kw = dict(n_blue=ve.n_blue, n_red=ve.n_red, n_obs=ve.n_obstacles,
              red_feat_dim=4, obs_feat_dim=5, d_hidden=16,
              actor_n_red=ve.actor_n_red, actor_n_obs=ve.actor_n_obstacles,
              actor_red_feat_dim=ve.actor_red_feat_dim,
              actor_obs_feat_dim=ve.actor_obs_feat_dim)
    kw.update(override)
    return GNNStage4Policy(**kw)


def test_policy_runs_on_the_tracker_observation():
    ve = _vec("tracker")
    torch.manual_seed(0)
    pol = _policy(ve)
    obs = ve.reset(seed=0)
    hidden = pol.initial_hidden(ve.n_envs, torch.device("cpu"))
    for _ in range(3):
        t = {k: torch.from_numpy(v) for k, v in obs.items()}
        partial, full = split_stage4_obs(t)
        action, logp, _ent, value, hidden, _a, _c = pol.get_action_and_value(
            partial, full, hidden)
        assert action.shape == (ve.n_envs, ve.n_blue, 2)
        assert value.shape == (ve.n_envs, ve.n_blue)
        obs, _r, _d, _i = ve.step(action.clamp(-1, 1).numpy())
    ve.close()


def test_default_actor_sizing_leaves_the_policy_unchanged():
    torch.manual_seed(0)
    a = GNNStage4Policy(n_blue=3, n_red=2, n_obs=3, d_hidden=16)
    torch.manual_seed(0)
    b = GNNStage4Policy(n_blue=3, n_red=2, n_obs=3, d_hidden=16,
                        actor_n_red=None, actor_n_obs=None)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_only_the_actor_input_layers_change_shape():
    ve = _vec("tracker")
    torch.manual_seed(0)
    belief_like = GNNStage4Policy(n_blue=3, n_red=2, n_obs=3, d_hidden=16)
    tracker = _policy(ve)
    diff = [k for k, v in tracker.state_dict().items()
            if belief_like.state_dict()[k].shape != v.shape]
    assert sorted(diff) == ["actor_encoder.obs_input_mlp.0.weight",
                            "actor_encoder.red_input_mlp.0.weight"]
    ve.close()


def test_checkpoint_loader_reproduces_either_mode():
    old = dict(policy_type="gnn_stage4_v6", n_blue=3, n_red=2, n_obstacles=3,
               arena_size=200.0, max_steps=60, capture_radius=3.0,
               sensor_radius=40.0)
    kw = env_kwargs_from_checkpoint(old)
    assert kw["actor_obs"] == "belief" and kw["use_belief_maps"] is True
    assert kw["clutter_rate"] == 0.0 and _actor_sizing(old) == {}

    new = dict(old, actor_obs="tracker", tracker_red_slots=8,
               tracker_obstacle_slots=12, tracker_sigma_cutoff=40.0,
               red_motion_ckpt=None, clutter_rate=0.2,
               obstacle_radius_noise_std=2.0)
    kw = env_kwargs_from_checkpoint(new)
    assert kw["actor_obs"] == "tracker" and kw["use_belief_maps"] is False
    assert kw["tracker_red_slots"] == 8 and kw["clutter_rate"] == 0.2
    assert _actor_sizing(new) == dict(
        actor_n_red=8, actor_n_obs=12,
        actor_red_feat_dim=actor_graph.RED_FEAT_DIM,
        actor_obs_feat_dim=actor_graph.OBS_FEAT_DIM)
    PursuitEnv(**kw).reset(seed=0)
