"""
tests/test_staleness.py — the coverage field: steps since each grid cell
was last OBSERVED.

Two things carry real risk here and are pinned accordingly.

1. The geometry must be the SAME geometry the belief update uses.
   ``_observed_cells_mask`` deliberately duplicates it rather than
   refactoring the proven per-blue loop (which interleaves occlusion with
   RNG draws, so restructuring would reorder the random stream).  A
   duplicate that can drift silently is worse than no duplicate, so the
   agreement is asserted directly.

2. Staleness answers "when did I last LOOK", NOT "did I find something".
   Resetting only on a successful detection would silently turn it back
   into a (bad) belief map, since p_TP < 1.

Run:
    pytest tests/test_staleness.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav


def _env(**kw):
    base = dict(
        n_blue=3, n_red=2, n_obstacles=0, arena_size=130.0, max_steps=200,
        capture_radius=3.0, sensor_radius=40.0, use_staleness=True,
        red_policy=run_from_nearest_uav, seed=0,
    )
    base.update(kw)
    e = PursuitEnv(**base)
    e.reset(seed=base["seed"])
    return e


def _zeros(e):
    return {a: np.zeros(2, dtype=np.float32) for a in e.agents}


# --------------------------------------------------------------------- #
#  Allocation and the off switch
# --------------------------------------------------------------------- #

def test_off_by_default_and_allocates_when_enabled():
    assert _env(use_staleness=False)._staleness is None
    e = _env()
    g = e.belief_grid_size
    assert e._staleness.shape == (g, g)
    assert e._staleness.dtype == np.int32


def test_independent_of_the_belief_map():
    """The two are separate answers to separate questions, so neither may
    require the other to be switched on."""
    e = _env(use_belief_maps=False)
    assert e._belief_maps is None and e._staleness is not None
    e.step(_zeros(e))
    assert e._staleness is not None


# --------------------------------------------------------------------- #
#  The update rule
# --------------------------------------------------------------------- #

def test_starts_at_maximum_except_where_the_blues_can_already_see():
    """At reset nothing has been looked at yet.  Starting at zero would
    tell the policy the whole arena is freshly swept — the exact opposite
    of the truth."""
    e = _env()
    seen = e._observed_cells_mask()
    assert seen.any(), "blues should see something from their spawn"
    assert np.all(e._staleness[seen] == 0)
    assert np.all(e._staleness[~seen] == e.max_steps)


def test_ages_by_one_step_and_resets_where_observed():
    e = _env()
    before = e._staleness.copy()
    e.step(_zeros(e))
    seen = e._observed_cells_mask()
    assert np.all(e._staleness[seen] == 0)
    unseen = ~seen
    expected = np.minimum(before[unseen] + 1, e.max_steps)
    assert np.array_equal(e._staleness[unseen], expected)


def test_never_exceeds_max_steps():
    """The cap is what makes dividing by max_steps a [0, 1] normalisation,
    and what makes the initial 'never observed' value the natural maximum
    rather than an arbitrary sentinel."""
    e = _env(max_steps=20)
    for _ in range(40):
        if not e.agents:
            break
        e.step(_zeros(e))
    assert e._staleness.max() <= 20


def test_a_stationary_blue_lets_the_rest_of_the_arena_age_monotonically():
    e = _env(n_blue=1, n_red=1, sensor_radius=20.0)
    e._blue_pos[:] = np.array([[65.0, 65.0]], dtype=np.float32)
    e._staleness[:] = 0
    prev = e._staleness.copy()
    for _ in range(5):
        e.step(_zeros(e))
        seen = e._observed_cells_mask()
        # Nothing gets FRESHER without being looked at.
        assert np.all(e._staleness[~seen] > prev[~seen])
        prev = e._staleness.copy()


# --------------------------------------------------------------------- #
#  "Looked", not "found"
# --------------------------------------------------------------------- #

def test_resets_on_looking_even_when_no_target_is_there():
    """Empty cells inside the sensor disk must reset.  If staleness only
    reset where something was detected, an empty swept region would stay
    maximally stale forever and the policy would sweep it again and
    again."""
    e = _env(n_red=1)
    # Park the single red far from blue 0, so blue 0's disk is empty.
    e._blue_pos[0] = np.array([20.0, 20.0], dtype=np.float32)
    e._red_pos[0] = np.array([120.0, 120.0], dtype=np.float32)
    e._staleness[:] = e.max_steps
    e._update_staleness()
    near = np.linalg.norm(e._cell_centres - e._blue_pos[0], axis=-1) <= 40.0
    assert np.all(e._staleness[near] == 0), (
        "looking at empty space must still count as having looked")


def test_is_not_affected_by_sensor_false_negatives():
    """p_TP < 1 means a look can miss a real target.  Staleness asks only
    whether we LOOKED, so it must be identical regardless of the sensor's
    Bernoulli draws — otherwise it silently becomes a belief map again."""
    out = []
    for p_tp in (1.0, 0.3):
        e = _env(use_belief_maps=True, p_TP=p_tp, p_FP=0.0, seed=7)
        for _ in range(6):
            e.step(_zeros(e))
        out.append(e._staleness.copy())
    assert np.array_equal(out[0], out[1]), (
        "staleness changed with detection probability")


# --------------------------------------------------------------------- #
#  The duplicated geometry
# --------------------------------------------------------------------- #

def test_observed_mask_matches_an_independent_recomputation():
    e = _env(n_obstacles=0, sensor_radius=35.0)
    got = e._observed_cells_mask()
    d = np.linalg.norm(
        e._cell_centres[:, :, None, :] - e._blue_pos[None, None, :, :], axis=-1)
    want = (d <= 35.0).any(axis=-1)          # no obstacles -> no occlusion
    assert np.array_equal(got, want)


def test_observed_mask_matches_the_cells_the_belief_update_touches():
    """The pin on the duplicated geometry.  With decay and diffusion off,
    the ONLY cells the belief update can change are the ones it observed,
    so the two code paths must agree exactly."""
    e = _env(n_obstacles=3, use_belief_maps=True,
             enemy_belief_decay=1.0, enemy_belief_diffusion=0.0,
             p_TP=1.0, p_FP=1.0, seed=3)
    for _ in range(3):
        e.step(_zeros(e))
        before = e._belief_maps[0].copy()
        mask = e._observed_cells_mask()
        e._update_belief_maps()
        changed = e._belief_maps[0] != before
        # Cells already saturated at the clip cannot change further, so
        # "changed" is a SUBSET of "observed"; every unobserved cell must
        # be untouched, which is the direction that matters.
        assert not np.any(changed & ~mask), (
            "belief update touched cells the observed mask excludes")


def test_obstacles_occlude_the_same_way_for_both():
    """An obstacle between a blue and a cell blocks the look, so that cell
    must keep ageing even though it is inside the sensor disk."""
    e = _env(n_blue=1, n_red=1, n_obstacles=1, sensor_radius=60.0)
    e._blue_pos[:] = np.array([[20.0, 65.0]], dtype=np.float32)
    e._obstacle_pos = np.array([[50.0, 65.0]], dtype=np.float32)
    e._obstacle_r = np.array([12.0], dtype=np.float32)
    e._recompute_obstacle_grid()
    mask = e._observed_cells_mask()
    d = np.linalg.norm(e._cell_centres - e._blue_pos[0], axis=-1)
    in_disk = d <= 60.0
    assert in_disk.sum() > mask.sum(), "the obstacle occluded nothing"
    # A cell directly behind the obstacle must be excluded.
    behind = np.argmin(np.linalg.norm(
        e._cell_centres - np.array([75.0, 65.0]), axis=-1).reshape(-1))
    bx, by = np.unravel_index(behind, mask.shape)
    assert in_disk[bx, by] and not mask[bx, by]


# --------------------------------------------------------------------- #
#  Plumbing
# --------------------------------------------------------------------- #

def test_reset_clears_the_field_between_episodes():
    e = _env()
    for _ in range(10):
        e.step(_zeros(e))
    aged = e._staleness.copy()
    e.reset(seed=1)
    assert not np.array_equal(aged, e._staleness)
    seen = e._observed_cells_mask()
    assert np.all(e._staleness[~seen] == e.max_steps)


def test_exposed_in_the_state_snapshot_as_a_copy():
    e = _env()
    snap = e.state_snapshot()
    assert "staleness" in snap
    snap["staleness"][0, 0] = -999
    assert e._staleness[0, 0] != -999, "snapshot aliased the live array"


def test_disabled_env_is_unchanged_step_for_step():
    """Non-regression: with the flag off, nothing about the env may move."""
    def run(flag):
        e = _env(use_staleness=flag, use_belief_maps=True, seed=11)
        rng = np.random.default_rng(0)
        poss = []
        for _ in range(12):
            if not e.agents:
                break
            e.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                    for a in e.agents})
            poss.append(e._red_pos.copy())
        return np.array(poss)
    assert np.array_equal(run(False), run(True))
