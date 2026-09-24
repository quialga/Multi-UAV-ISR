"""
tests/test_region_nodes.py — aggregation of the staleness field into the
R x R REGION nodes the graph will consume.

Stage B of the search work.  Stage A (the field itself) is in
tests/test_staleness.py; wiring these nodes into the policy graph is a
separate change again.

The risks here are quiet ones — an aggregation that is subtly wrong still
returns plausible-looking numbers — so the tests check the arithmetic
against independently computed values rather than just shapes.

Run:
    pytest tests/test_region_nodes.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav


def _env(**kw):
    base = dict(
        n_blue=3, n_red=2, n_obstacles=0, arena_size=130.0, max_steps=200,
        capture_radius=3.0, sensor_radius=40.0, use_staleness=True,
        staleness_regions=5, red_policy=run_from_nearest_uav, seed=0,
    )
    base.update(kw)
    e = PursuitEnv(**base)
    e.reset(seed=base["seed"])
    return e


# --------------------------------------------------------------------- #
#  Partition
# --------------------------------------------------------------------- #

def test_regions_tile_the_grid_exactly_once():
    """Every cell belongs to exactly one region -- no gaps, no overlap.
    26 has no divisor near 5, so the split is uneven by construction and
    that is precisely where an off-by-one would hide."""
    e = _env(staleness_regions=5)
    bounds = e._region_bounds()
    assert bounds[0][0] == 0
    assert bounds[-1][1] == e.belief_grid_size
    for (a0, a1), (b0, b1) in zip(bounds, bounds[1:]):
        assert a1 == b0, "regions must be contiguous"
    assert sum(b - a for a, b in bounds) == e.belief_grid_size


def test_region_sizes_are_as_even_as_the_grid_allows():
    e = _env(staleness_regions=5)
    sizes = [b - a for a, b in e._region_bounds()]
    assert sorted(sizes) == [5, 5, 5, 5, 6]     # 26 cells over 5 regions
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("R", [1, 2, 4, 5, 7, 13, 26])
def test_partition_holds_at_every_resolution(R):
    e = _env(staleness_regions=R)
    bounds = e._region_bounds()
    assert len(bounds) == R
    assert sum(b - a for a, b in bounds) == e.belief_grid_size
    assert all(b > a for a, b in bounds), "empty region"
    feats, centres = e._build_region_nodes()
    assert feats.shape == (R * R, 2) and centres.shape == (R * R, 2)


# --------------------------------------------------------------------- #
#  Centres
# --------------------------------------------------------------------- #

def test_centres_are_the_centroid_of_each_regions_own_cells():
    """Computed from the cells, not assumed at a regular spacing -- which
    matters exactly because the last region is one cell wider."""
    e = _env(staleness_regions=5)
    _, centres = e._build_region_nodes()
    bounds = e._region_bounds()
    k = 0
    for x0, x1 in bounds:
        for y0, y1 in bounds:
            want = e._cell_centres[x0:x1, y0:y1].reshape(-1, 2).mean(0)
            assert np.allclose(centres[k], want, atol=1e-5)
            k += 1


def test_centres_lie_inside_the_arena():
    e = _env(staleness_regions=5)
    _, centres = e._build_region_nodes()
    assert np.all(centres >= 0) and np.all(centres <= e.arena_size)


def test_single_region_is_the_arena_centre():
    e = _env(staleness_regions=1)
    _, centres = e._build_region_nodes()
    assert centres.shape == (1, 2)
    assert np.allclose(centres[0], e.arena_size / 2.0, atol=e.belief_cell_size)


# --------------------------------------------------------------------- #
#  Staleness feature
# --------------------------------------------------------------------- #

def test_staleness_is_the_mean_over_the_region_normalised():
    e = _env(staleness_regions=5)
    rng = np.random.default_rng(0)
    e._staleness[:] = rng.integers(0, e.max_steps, e._staleness.shape)
    feats, _ = e._build_region_nodes()
    bounds = e._region_bounds()
    k = 0
    for x0, x1 in bounds:
        for y0, y1 in bounds:
            want = e._staleness[x0:x1, y0:y1].mean() / e.max_steps
            assert feats[k, 0] == pytest.approx(want, abs=1e-6)
            k += 1


def test_staleness_feature_is_normalised_to_the_unit_interval():
    """The [0, 1] range is what makes max_steps the right cap on the field
    and lets the same feature scale across episode lengths."""
    e = _env()
    e._staleness[:] = e.max_steps
    feats, _ = e._build_region_nodes()
    assert np.allclose(feats[:, 0], 1.0)
    e._staleness[:] = 0
    feats, _ = e._build_region_nodes()
    assert np.allclose(feats[:, 0], 0.0)


def test_a_freshly_swept_region_reads_lower_than_an_untouched_one():
    """The signal the policy is meant to act on, end to end."""
    e = _env(staleness_regions=5)
    e._staleness[:] = e.max_steps
    e._staleness[0:5, 0:5] = 0                # sweep the first region
    feats, _ = e._build_region_nodes()
    assert feats[0, 0] == 0.0
    assert np.all(feats[1:, 0] > feats[0, 0])


# --------------------------------------------------------------------- #
#  Searchable fraction
# --------------------------------------------------------------------- #

def test_searchable_is_one_with_no_obstacles():
    e = _env(n_obstacles=0)
    feats, _ = e._build_region_nodes()
    assert np.allclose(feats[:, 1], 1.0)


def _believe_obstacle(e, cx0, cx1, cy0, cy1):
    """Make the team BELIEVE those cells are blocked.

    Region nodes feed the actor, so ``searchable`` is sourced from the
    obstacle estimate, never from ``_obstacle_grid`` (which is built from
    true positions and radii and belongs to the CTDE critic).  Driving the
    truth grid therefore no longer moves the feature — and these tests
    used to do exactly that, which is to say they pinned the leak.
    """
    e._belief_maps[1][:] = -10.0
    e._belief_maps[1][cx0:cx1, cy0:cy1] = +10.0
    return e._estimated_obstacle_mask()


def test_searchable_counts_only_cells_outside_obstacles():
    e = _env(n_obstacles=1, staleness_regions=5, use_belief_maps=True)
    blocked = _believe_obstacle(e, 0, 6, 0, 6)
    feats, _ = e._build_region_nodes()
    bounds = e._region_bounds()
    free = ~blocked
    k = 0
    for x0, x1 in bounds:
        for y0, y1 in bounds:
            blk = free[x0:x1, y0:y1]
            assert feats[k, 1] == pytest.approx(blk.mean(), abs=1e-6)
            k += 1
    assert feats[0, 1] < 1.0, "the obstacle blocked nothing in region 0"


def test_staleness_averages_only_over_searchable_cells():
    """Obstacle interiors cannot hide a target, so averaging them in would
    dilute the very signal the feature exists to carry."""
    e = _env(n_obstacles=1, staleness_regions=5, use_belief_maps=True)
    free = ~_believe_obstacle(e, 0, 6, 0, 6)
    # Rock reads as freshly seen, open ground as never seen.  A naive mean
    # over ALL cells would be pulled down; the correct one stays at 1.0.
    e._staleness[:] = e.max_steps
    e._staleness[~free] = 0
    feats, _ = e._build_region_nodes()
    # PARTIALLY blocked only: a fully blocked region reports staleness 0 by
    # design (nothing to find there), which the next test pins.
    partial = (feats[:, 1] > 0.0) & (feats[:, 1] < 1.0)
    assert partial.any(), "test set up no partially blocked region"
    assert np.allclose(feats[partial, 0], 1.0), (
        "obstacle interiors leaked into the staleness mean")


def test_a_fully_blocked_region_reports_zero_staleness():
    """Nothing to find there, so sweeping it gains nothing -- reporting
    max would send the policy to search solid rock."""
    e = _env(n_obstacles=1, staleness_regions=2, use_belief_maps=True)
    _believe_obstacle(e, 0, 13, 0, 13)          # region 0 entirely blocked
    e._staleness[:] = e.max_steps
    feats, _ = e._build_region_nodes()
    assert feats[0, 1] == 0.0
    assert feats[0, 0] == 0.0


def test_searchable_reads_the_estimate_not_the_true_obstacle_grid():
    """The leak this file used to pin.  Region nodes reach the ACTOR, so a
    `searchable` sourced from `_obstacle_grid` would hand the tracker arm
    a true obstacle map — the sixth entry in the leak table of
    docs/tracker_observation.md.  Moving truth alone must change nothing.
    """
    e = _env(n_obstacles=1, staleness_regions=2, use_belief_maps=True)
    e._belief_maps[1][:] = -10.0                # believed entirely open
    before, _ = e._build_region_nodes()

    e._obstacle_pos = np.array([[13.0, 13.0]], dtype=np.float32)
    e._obstacle_r = np.array([40.0], dtype=np.float32)
    e._recompute_obstacle_grid()
    assert e._obstacle_grid.any(), "test failed to place a true obstacle"

    after, _ = e._build_region_nodes()
    assert np.allclose(before, after), (
        "the true obstacle grid leaked into the actor's region features")


# --------------------------------------------------------------------- #
#  Plumbing
# --------------------------------------------------------------------- #

def test_returns_zeros_when_staleness_is_disabled():
    e = _env(use_staleness=False, staleness_regions=5)
    feats, centres = e._build_region_nodes()
    assert feats.shape == (25, 2) and np.all(feats == 0.0)
    assert centres.shape == (25, 2)


def test_features_stay_finite_and_bounded_through_a_rollout():
    e = _env(n_obstacles=4, staleness_regions=5)
    rng = np.random.default_rng(1)
    for _ in range(40):
        if not e.agents:
            break
        e.step({a: rng.uniform(-1, 1, 2).astype(np.float32) for a in e.agents})
        feats, centres = e._build_region_nodes()
        assert np.all(np.isfinite(feats)) and np.all(np.isfinite(centres))
        assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_rejects_a_resolution_finer_than_the_grid():
    with pytest.raises(AssertionError):
        _env(staleness_regions=27)
