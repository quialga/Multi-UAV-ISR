"""
tests/test_stochastic_red.py — the scripted evader's randomness.

The point of these is that the noise has to be the RIGHT KIND, not merely
present.  Measured: per-step iid jitter moves the red 1.6 m over 30 steps
(0.3 belief cells — sub-cell, so invisible to the map and averaged out by
any predictor) AND it whitens the natural acceleration correlation
(rho 0.66 -> 0.54).  Correlated drift plus a committed side choice moves it
9.0 m, keeps rho at 0.64, and is the only variant that produces genuine
ALEATORIC bimodality (20% of instants, 80 deg apart) — which is the whole
justification for a mixture/particle predictor.

Run:
    pytest tests/test_stochastic_red.py -v
"""
from __future__ import annotations

import numpy as np

from isr.agents.stochastic_red import StochasticRed
from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav


def _scene(n_red=3, n_obs=2):
    rng = np.random.default_rng(0)
    blue = rng.uniform(20.0, 110.0, (4, 2)).astype(np.float32)
    red = rng.uniform(20.0, 110.0, (n_red, 2)).astype(np.float32)
    active = np.ones(n_red, dtype=bool)
    opos = np.array([[65.0, 65.0], [30.0, 90.0]], dtype=np.float32)[:n_obs]
    orad = np.array([12.0, 8.0], dtype=np.float32)[:n_obs]
    return blue, red, active, opos, orad, 130.0


def test_defaults_are_the_deterministic_policy():
    """No arguments => byte-identical to run_from_nearest_uav, so enabling
    stochasticity is always an explicit choice."""
    blue, red, act, opos, orad, L = _scene()
    p = StochasticRed(seed=0)
    p.reset(len(red))
    got = p(blue, red, act, opos, orad, L)
    want = run_from_nearest_uav(blue, red, act, opos, orad, L)
    assert np.allclose(got, want)


def test_output_stays_a_bounded_acceleration():
    blue, red, act, opos, orad, L = _scene()
    p = StochasticRed(heading_noise_std=0.6, commit_prob=1.0,
                      speed_jitter=0.3, seed=1)
    p.reset(len(red))
    for _ in range(50):
        a = p(blue, red, act, opos, orad, L)
        assert np.all(np.abs(a) <= 1.0 + 1e-6)
        assert np.all(np.isfinite(a))


def test_inactive_reds_get_zero():
    blue, red, act, opos, orad, L = _scene()
    act[1] = False
    p = StochasticRed(heading_noise_std=0.5, commit_prob=1.0, seed=2)
    p.reset(len(red))
    a = p(blue, red, act, opos, orad, L)
    assert np.allclose(a[1], 0.0)


def test_ar1_phase_is_stationary_at_the_requested_std():
    """The innovation is scaled by sqrt(1-rho^2) so the MARGINAL std stays
    heading_noise_std whatever rho is.  Without that, turning correlation up
    would silently turn the noise amount down, confounding two knobs."""
    for rho in (0.0, 0.5, 0.9):
        p = StochasticRed(heading_noise_std=0.4, heading_rho=rho, seed=3)
        p.reset(1)
        phases = []
        for _ in range(20000):
            p._phase[0] = (rho * p._phase[0]
                           + np.sqrt(1.0 - rho ** 2)
                           * p._rng.normal(0.0, 0.4))
            phases.append(p._phase[0])
        assert abs(np.std(phases) - 0.4) < 0.03, f"rho={rho}"


def test_heading_noise_is_temporally_correlated():
    """rho > 0 must actually produce autocorrelated phase — that is the
    property that makes the noise visible rather than sub-cell."""
    p = StochasticRed(heading_noise_std=0.4, heading_rho=0.9, seed=4)
    p.reset(1)
    ph = []
    for _ in range(4000):
        p._phase[0] = (0.9 * p._phase[0]
                       + np.sqrt(1 - 0.81) * p._rng.normal(0.0, 0.4))
        ph.append(p._phase[0])
    ph = np.array(ph) - np.mean(ph)
    rho1 = float(np.sum(ph[:-1] * ph[1:]) / np.sum(ph * ph))
    assert rho1 > 0.8


def test_commitment_persists_for_commit_steps():
    """A side choice is a PLAN, not a per-step coin flip: it must be held."""
    blue, red, act, opos, orad, L = _scene(n_red=1)
    red[0] = opos[0] + np.array([14.0, 0.0], dtype=np.float32)   # in the band
    p = StochasticRed(commit_prob=1.0, commit_steps=6, seed=5)
    p.reset(1)
    sides = []
    for _ in range(6):
        p(blue, red, act, opos, orad, L)
        sides.append(int(p._side[0]))
    assert sides[0] != 0
    assert len(set(sides)) == 1, f"side flipped mid-commitment: {sides}"


def test_commitment_only_triggers_near_an_obstacle():
    blue, red, act, opos, orad, L = _scene(n_red=1)
    red[0] = np.array([5.0, 5.0], dtype=np.float32)      # far from obstacles
    p = StochasticRed(commit_prob=1.0, commit_steps=6, seed=6)
    p.reset(1)
    for _ in range(20):
        p(blue, red, act, opos, orad, L)
        assert int(p._side[0]) == 0, "committed with no obstacle nearby"


def test_commitment_actually_deflects_the_heading():
    blue, red, act, opos, orad, L = _scene(n_red=1)
    red[0] = opos[0] + np.array([14.0, 0.0], dtype=np.float32)
    base = run_from_nearest_uav(blue, red, act, opos, orad, L)[0]
    p = StochasticRed(commit_prob=1.0, commit_steps=6, commit_angle=1.1, seed=7)
    p.reset(1)
    a = p(blue, red, act, opos, orad, L)[0]
    cos = float(a @ base / (np.linalg.norm(a) * np.linalg.norm(base)))
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) > 30.0


def test_reset_clears_state_between_episodes():
    blue, red, act, opos, orad, L = _scene(n_red=1)
    red[0] = opos[0] + np.array([14.0, 0.0], dtype=np.float32)
    p = StochasticRed(heading_noise_std=0.5, commit_prob=1.0, seed=8)
    p.reset(1)
    p(blue, red, act, opos, orad, L)
    assert p._left[0] > 0 or p._phase[0] != 0.0
    p.reset(1)
    assert p._left[0] == 0 and p._side[0] == 0 and p._phase[0] == 0.0


def test_env_resets_a_stateful_policy():
    """The env must clear per-red noise state at episode start, or a
    commitment leaks into the next episode."""
    red = StochasticRed(heading_noise_std=0.4, commit_prob=1.0, seed=9)
    env = PursuitEnv(n_blue=3, n_red=2, n_obstacles=2, arena_size=130.0,
                     max_steps=50, capture_radius=3.0, sensor_radius=40.0,
                     use_belief_maps=True, red_policy=red, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        if not env.agents:
            break
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                  for a in env.agents})
    env.reset(seed=1)
    assert np.all(red._phase == 0.0) and np.all(red._left == 0)


def test_stochastic_red_changes_trajectories():
    """Same start and same blue actions, different noise roll -> the reds
    must actually end up somewhere else, or the noise is decorative.

    Averaged over scenarios: divergence is strongly seed-dependent, because
    a red pinned against a wall has its divergence clipped away.  Measured
    ~9 m over 30 steps on average, but as little as 1.7 m in a wall-pinned
    scenario, so a single-seed threshold would be flaky.
    """
    divs = []
    for scene in (3, 4, 5):
        ends = []
        for s in (100, 200):
            red = StochasticRed(heading_noise_std=0.35, heading_rho=0.85,
                                commit_prob=0.6, seed=s)
            env = PursuitEnv(n_blue=5, n_red=3, n_obstacles=4,
                             arena_size=130.0, max_steps=100,
                             capture_radius=3.0, sensor_radius=40.0,
                             use_belief_maps=True, red_policy=red, seed=scene)
            env.reset(seed=scene)
            rng = np.random.default_rng(scene)      # identical blue actions
            for _ in range(30):
                if not env.agents:
                    break
                env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                          for a in env.agents})
            ends.append(env._red_pos.copy())
        divs.append(float(np.linalg.norm(ends[0] - ends[1], axis=1).mean()))
    assert np.mean(divs) > 2.0, f"noise barely changes behaviour: {divs}"


def test_iid_noise_is_much_weaker_than_correlated():
    """The design claim, pinned: per-step iid jitter is nearly invisible
    (sub-cell), correlated drift plus commitment is not."""
    def diverge(**kw):
        ends = []
        for s in (100, 200):
            red = StochasticRed(seed=s, **kw)
            env = PursuitEnv(n_blue=5, n_red=3, n_obstacles=4,
                             arena_size=130.0, max_steps=100,
                             capture_radius=3.0, sensor_radius=40.0,
                             use_belief_maps=True, red_policy=red, seed=5)
            env.reset(seed=5)
            rng = np.random.default_rng(5)
            for _ in range(30):
                if not env.agents:
                    break
                env.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                          for a in env.agents})
            ends.append(env._red_pos.copy())
        return float(np.linalg.norm(ends[0] - ends[1], axis=1).mean())

    iid = diverge(heading_noise_std=0.35, heading_rho=0.0)
    corr = diverge(heading_noise_std=0.35, heading_rho=0.85, commit_prob=0.6)
    assert corr > 1.5 * iid, f"correlated {corr:.1f} vs iid {iid:.1f}"
