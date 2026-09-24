"""
isr/agents/heuristics.py — scripted policies for blue and red.

Two distinct interfaces, kept separate on purpose:

1. **Blue agents** (PettingZoo agent perspective).  Class-based, with
   the signature ``act(obs, env, agent) -> action``.  Used as the blue
   policy in tests, as baselines for the PPO policy to beat, and
   inside the rendering / smoke scripts.  Mirrors the eventual
   neural-policy interface so PPO and heuristics are drop-in
   interchangeable in the runner.

2. **Red policies** (env-internal scripted opposition).  Plain
   callables of shape
   ``(blue_pos, red_pos, red_active) -> (N_red, 2) actions``.
   Passed to ``PursuitEnv(red_policy=...)``.  Replaces the default
   ``run_from_nearest_uav``.  Stage 4 will swap one of these for a
   learned policy without env changes.

Why two interfaces?  Blue agents are first-class PettingZoo agents and
each one acts once per env step via the standard agent loop.  Red
"agents" in Stage 1 are env-internal — the env applies them all in
one batched call inside ``step()``, which doesn't fit the
agent-by-agent ``act`` pattern.  Forcing them into the same shape
would either complicate the env loop or stop us from vectorising the
red step.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Re-export the default red policy so callers can grab everything from
# one module: ``from isr.agents.heuristics import run_from_nearest_uav``.
from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav  # noqa: F401


# ===========================================================================
#  Blue agents — class-based act(obs, env, agent) -> action
# ===========================================================================

class HeuristicBlueAgent:
    """Base class for scripted blue policies.  Override ``act``."""

    def act(
        self,
        obs:   np.ndarray,
        env:   PursuitEnv,
        agent: str,
    ) -> np.ndarray:
        """
        Return a 2D acceleration action in [-1, 1]^2 for the named
        blue agent.  Heuristics may peek at ``env.state_snapshot()``
        for ground truth; the eventual neural policy must work from
        ``obs`` alone.
        """
        raise NotImplementedError


class RandomAgent(HeuristicBlueAgent):
    """
    Uniform random action in the env's action box.  Sanity check —
    expected to perform poorly (small negative reward dominated by
    step + action costs, few captures).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs, env, agent):
        return self._rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)


class GreedyPursuer(HeuristicBlueAgent):
    """
    Pure greedy nearest-target pursuit.

    Each blue UAV independently picks the closest *active* red target
    and accelerates toward it with unit magnitude.  No coordination
    between blue agents — three UAVs may all converge on the same
    target while another is unattended.

    This is the **Stage 1 baseline the PPO blue policy must beat**
    (by >=20% mean episode return, per ``docs/design.md §3.10``).  The
    20% gap is meaningful: beating greedy requires the learned policy
    to do something a per-UAV nearest-target rule cannot — split
    coverage so multiple UAVs don't redundantly chase the same red,
    anticipate where reds will dodge to, etc.  Beating greedy is the
    headline check that Stage 1 actually learns *coordination*, not
    just "go to the nearest red".
    """

    def act(self, obs, env, agent):
        snap = env.state_snapshot()
        blue_pos   = snap["blue_pos"]
        red_pos    = snap["red_pos"]
        red_active = snap["red_active"]

        if not red_active.any():
            return np.zeros(2, dtype=np.float32)

        my_idx = env.possible_agents.index(agent)
        my_pos = blue_pos[my_idx]

        diffs = red_pos - my_pos                # (N_red, 2)
        dists = np.linalg.norm(diffs, axis=1)   # (N_red,)
        # Caught reds are not pursuit targets — mask their distance to inf
        # so argmin picks the closest *active* red.
        dists = np.where(red_active, dists, np.inf)

        # Sensor-radius honesty: if the env is partial-observable
        # (Stage 3), each UAV can only see reds within its own
        # sensor_radius — matching the per-receiver edge visibility the
        # trained CTDE policy sees.  Without this Greedy effectively
        # cheats past the sensor cap and no fair comparison is possible.
        # See docs/stage3_gpu_run.md §4 acceptance notes.
        R = getattr(env, "sensor_radius", None)
        if R is not None:
            dists = np.where(dists <= float(R), dists, np.inf)
            if not np.isfinite(dists).any():
                # No red visible from this UAV this step — the honest
                # stateless action is to hold position (a smarter policy
                # would recall the last-known red via memory; that is
                # exactly the GRU's job in the Stage 3 policy).
                return np.zeros(2, dtype=np.float32)

        nearest = int(np.argmin(dists))

        d_vec = diffs[nearest]
        norm = float(np.linalg.norm(d_vec))
        if norm < 1e-8:
            # Already on top of the target — capture should have fired
            # this step; output zero so we don't waste action cost.
            return np.zeros(2, dtype=np.float32)
        return (d_vec / norm).astype(np.float32)


class ObservationGreedyPursuer(HeuristicBlueAgent):
    """``GreedyPursuer``'s rule, run on the ACTOR's observation.

    ``GreedyPursuer`` reads ``env.state_snapshot()`` — TRUE red positions,
    gated only by range.  It therefore never misses a detection, never sees
    a false one, and its positions carry no error: exactly the near-oracle
    in-range regime §4b removed from the policy's inputs.  Comparing a
    trained policy against it mixes policy quality with an information gap.

    This variant chases the same targets the POLICY is shown — the actor's
    red nodes, with their detection misses, position error and clutter —
    so it answers "what does a trivial pursuit rule achieve on THIS
    observation?".  Two uses: a fair bar for the trained policy, and a
    learning-free measure of observation quality (run it under
    ``actor_obs='belief'`` and ``'tracker'`` to compare the two
    representations without a policy in the way).

    Rule: among this blue's red-slot edges with a non-zero visibility
    mask, steer at the nearest one; hold position if there are none
    (matching ``GreedyPursuer`` when nothing is in range).

    Memory is the fallback, not an equal: the mask IS the track confidence
    (1.0 for a live track, the belief peak's confidence for a memory one,
    0 for padding), so the rule takes the most confident slots on offer and
    picks the nearest of THOSE.  A live target therefore always outranks a
    remembered one, and memory is used exactly when nothing is live.

    Ranking by distance alone was tried first and is wrong: a low-confidence
    belief peak sitting closer than a live track would win, so the baseline
    chased ghosts.  In tracker mode the mask is binary for every confirmed
    track (coasting included), so this degenerates to "nearest confirmed
    track" — correct, because the tracker exposes no live/memory split and
    neither does the policy's observation.
    """

    def __init__(self) -> None:
        self._env = None
        self._t   = None
        self._obs = None

    def reset(self) -> None:
        self._env = None
        self._t   = None
        self._obs = None

    def act(self, obs, env, agent):
        # One observation per (env, timestep), shared by every blue.
        # structured_belief_observation() re-draws the sensor each call --
        # the p_TP detection dice and the position noise -- so calling it
        # per blue would give the five UAVs five different realisations of
        # the same instant, where the policy gets one.  (It does NOT mutate
        # the belief map: _update_belief_maps runs in step(), and an earlier
        # version of this comment claiming otherwise was wrong.)
        t = int(env._t)
        if self._obs is None or env is not self._env or t != self._t:
            self._obs = env.structured_belief_observation()
            self._env = env
            self._t   = t
        return self.act_from_observation(
            self._obs, env.possible_agents.index(agent), int(env.n_blue))

    def act_from_observation(self, obs, my_idx: int, n_blue: int):
        """The rule itself, on a GIVEN observation.

        Split out from ``act`` so the same expert can label a BATCH of
        observations that were built elsewhere — behaviour cloning has to
        pair each stored observation with the action the expert takes on
        *that* observation, and re-deriving it from the env would re-draw
        the sensor and label a different realisation.

        ``obs`` is anything exposing ``rb_edge_features`` (n_rb, 7) and
        ``rb_edge_visible`` (n_rb,) for a single env.
        """
        feats = np.asarray(obs["rb_edge_features"])    # (n_rb, 7)
        vis   = np.asarray(obs["rb_edge_visible"])     # (n_rb,)

        # NOT env.rb_edge_dst: those index the n_red TRUE reds, while the
        # actor's enemy graph has K track SLOTS (K != n_red in general --
        # tracker mode pads to --tracker-red-slots).  The documented edge
        # ordering is "for s in K, for b in N_blue", so edge e lands on
        # blue e % n_blue.
        assert vis.shape[0] % n_blue == 0, (vis.shape, n_blue)
        dst  = np.arange(vis.shape[0]) % n_blue
        mine = np.nonzero((dst == my_idx) & (vis > 0.0))[0]
        if mine.size == 0:
            return self._search(obs, my_idx, n_blue)

        # Live before remembered: keep only the most confident slots this
        # blue has, then take the nearest of those.
        best_conf = float(vis[mine].max())
        mine = mine[vis[mine] >= best_conf - 1e-6]

        # Layout is [rel_pos (2), rel_vel (2), range (1), bearing (2)] with
        # rel_pos = dst - src.  These edges run red -> blue, so rel_pos is
        # (blue - red) and the heading toward the target is its negation.
        best    = int(mine[int(np.argmin(feats[mine, 4]))])
        return self._heading(feats[best, 0:2])

    @staticmethod
    def _heading(rel_pos) -> np.ndarray:
        """Unit step toward a target given ``rel_pos = dst - src``.

        These edges run target -> blue, so rel_pos is (blue - target) and
        the heading is its negation.  Getting the sign backwards flees the
        target and still produces a plausible unit vector, which is why
        ``test_observation_greedy.py`` opens on it.
        """
        toward = -np.asarray(rel_pos, dtype=np.float32)
        norm   = float(np.linalg.norm(toward))
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (toward / norm).astype(np.float32)

    def _search(self, obs, my_idx: int, n_blue: int) -> np.ndarray:
        """Nothing on the enemy graph: go look somewhere.

        Holding position is only the honest action while the observation
        offers nothing better, and it is expensive — measured, `ObsGreedy`
        on the tracker observation is idle 49.9% of agent-steps and 85.7%
        of them in the episodes it loses (docs/stage4_results.md §9.3).

        When the observation carries the coverage path, head for the
        region maximising

            staleness x searchable / (1 + range)

        ``range`` (edge feature 4) is already normalised by ``arena_size``,
        so this is the ``staleness / (1 + d/L)`` trade agreed for the rule:
        a plain staleness argmax sends a blue across the whole arena for a
        marginally older cell.  ``searchable`` zeroes regions that are
        solid rock, matching the weighting ``_build_region_graph`` uses.

        Deliberately UNCOORDINATED, like every other tier: five blues may
        pick the same region, and the distance term is the only thing
        spreading them.  Dividing the arena between them is a job for the
        learned policy — the moment this baseline coordinates it stops
        being the trivial rule that makes it a credible bar.
        """
        region = obs.get("region_feats") if hasattr(obs, "get") else None
        gb     = obs.get("gb_edge_feats") if hasattr(obs, "get") else None
        if region is None or gb is None:
            return np.zeros(2, dtype=np.float32)

        region = np.asarray(region)          # (R*R, 2) [staleness, searchable]
        gb     = np.asarray(gb)              # (R*R*n_blue, 7)
        if region.size == 0 or gb.size == 0:
            return np.zeros(2, dtype=np.float32)

        # Same ordering as the enemy edges: "for s in regions, for b in
        # blues", so edge e runs region e // n_blue -> blue e % n_blue.
        mine  = np.arange(my_idx, gb.shape[0], n_blue)
        rng   = gb[mine, 4]
        score = region[:, 0] * region[:, 1] / (1.0 + rng)
        if not np.any(score > 0.0):
            return np.zeros(2, dtype=np.float32)
        return self._heading(gb[mine[int(np.argmax(score))], 0:2])


# ===========================================================================
#  Red policies — (blue_pos, red_pos, red_active) -> (N_red, 2)
# ===========================================================================
#
# Functional API matching ``PursuitEnv(red_policy=...)``.  The env calls
# these once per step with the current state arrays; the callable
# returns one row of acceleration per red target.  Caught reds (active
# = False) must produce zero action.

def stationary_red(
    blue_pos:   np.ndarray,
    red_pos:    np.ndarray,
    red_active: np.ndarray,
    obstacle_pos: Optional[np.ndarray] = None,   # accepted, unused
    obstacle_r:   Optional[np.ndarray] = None,   # accepted, unused
    arena_size:   Optional[float] = None,        # accepted, unused
) -> np.ndarray:
    """
    Red doesn't move.  Easiest possible adversary — given Stage 1's
    blue/red speed advantage, blue should reliably catch every red
    within max_steps.  Useful as the absolute floor in benchmarks
    ("if blue can't beat this, the training pipeline is broken").
    Obstacle args are accepted for a uniform red-policy signature and
    ignored (a stationary target needs no collision-avoidance).
    """
    return np.zeros_like(red_pos, dtype=np.float32)


def random_red(seed: Optional[int] = None):
    """
    Returns a closure: red picks uniform random acceleration each step.
    Slightly harder than stationary (red occasionally drifts in a useful
    direction), much easier than ``run_from_nearest_uav``.  Used to
    bracket the difficulty curve.  Obstacle args accepted and ignored
    (a random walk needs no collision-avoidance; obstacle kinematic
    clipping still stops it physically).
    """
    rng = np.random.default_rng(seed)

    def policy(
        blue_pos:   np.ndarray,
        red_pos:    np.ndarray,
        red_active: np.ndarray,
        obstacle_pos: Optional[np.ndarray] = None,
        obstacle_r:   Optional[np.ndarray] = None,
        arena_size:   Optional[float] = None,
    ) -> np.ndarray:
        out = rng.uniform(-1.0, 1.0, size=red_pos.shape).astype(np.float32)
        out[~red_active] = 0.0
        return out

    return policy


# ``run_from_nearest_uav`` is the env default; re-exported above for
# uniform import location.

__all__ = [
    "HeuristicBlueAgent", "RandomAgent", "GreedyPursuer",
    "stationary_red", "random_red", "run_from_nearest_uav",
]
