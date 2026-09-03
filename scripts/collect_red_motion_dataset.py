"""
scripts/collect_red_motion_dataset.py — dataset for the learned red-motion
model (docs/tracking_diagnostics.md Sec. 8).

Collects one-step transitions ``(s_t, c_t) -> a_t``:
  s_t = the red's own [pos, vel]           (privileged / ground truth)
  c_t = ego-centric context: blue and obstacle relative position/velocity
        (+ obstacle radius), wall distances (+ ground truth, per Sec. 8.1:
        training uses privileged state, exactly like CTDE; the belief-
        derived, uncertainty-marginalised context is an INFERENCE-time
        concern, not a data-collection one)
  a_t = the TRUE acceleration the red actually took that step

H = 1 (single-step): this is what the Gaussian-Sum tracker's PREDICT step
consumes one branch at a time (docs Sec. 8.5), and it is exactly the
quantity ``red_policy`` already emits per step -- no finite-differencing,
no multi-step target construction.

Explicitly OUT OF SCOPE here, deferred to training time:
  * INPUT AUGMENTATION (perturbing s_t/c_t by belief-scale noise, so the
    model is accurate where the tracker's samples actually live at
    inference) -- a data-loader transform, not a collection-time choice,
    so the augmentation noise scale can be tuned later without
    re-collecting.
  * DISCRETISING a_t into the (heading, magnitude) grid (Sec. 8.4) -- log
    the raw continuous acceleration; bin it however the eventual network
    wants at training time.

------------------------------------------------------------------------
Why THESE red policies (representative, not a single fixed adversary)
------------------------------------------------------------------------
"The red" here is whatever ``StochasticRed`` config a real evader might
run, and we do not know which one in advance -- the simulator is a proxy
for an unknown real adversary.  So each RED gets its OWN independently
sampled configuration (``MixedStochasticRed`` below; one plain
``StochasticRed`` instance applies ONE set of parameters to every red it
drives, so genuinely independent per-red profiles need a small wrapper),
drawn per episode:

  * ``P_DETERMINISTIC`` of reds get NO randomness at all (every knob off,
    i.e. plain ``run_from_nearest_uav``) -- keeps a clean baseline in the
    data and lets held-out evaluation isolate "did the model forget how
    to predict a textbook flee response" from "does it handle noise".
  * The rest sample heading/commitment/magnitude parameters from ranges
    CENTRED ON the values already measured to produce visible, non-
    pathological dynamics in docs Sec. 7 (heading_noise_std=0.35,
    heading_rho=0.85, commit_prob=0.6, min_effort=0.3,
    magnitude_noise_std=0.2, magnitude_rho=0.85) rather than picked
    arbitrarily -- with per-episode jitter around them so the dataset
    does not key on one exact, suspiciously-repeated parameter value.

------------------------------------------------------------------------
Why THESE blue policies, and why NO trained checkpoint is needed
------------------------------------------------------------------------
No trained blue policy exists yet (the from-scratch GPU run is paused on
budget).  That is not a blocker: the red-motion dataset only needs a good
SPREAD of how blues approach (relative geometry, speed, coordination) --
not the exact final blue policy.  ``GreedyPursuer`` (scripted, already in
``isr/agents/heuristics.py``, no training required) gives purposeful chase
geometry; mixed per-blue with ``RandomAgent`` (independent per blue, with a
per-episode ``p_greedy``) adds erratic/uncoordinated coverage a trained
policy might not produce as often.  Team sizes (``n_blue``, and red/
obstacle counts via the env's existing ``n_red_min`` / ``n_obstacles_min``)
are varied too, for the same reason.

FLAG FOR LATER, not a blocker now: once the paused from-scratch checkpoint
exists, it would be worth checking whether TRAINED-blue trajectories
induce a meaningfully different geometry distribution than
scripted+random blues do (e.g. tighter stand-off, different approach
angles) -- cheap to check by comparing held-out prediction error on
checkpoint-generated rollouts, not something to assume either way.

------------------------------------------------------------------------
Storage
------------------------------------------------------------------------
Fixed-capacity padding + boolean masks (mirrors the env's OWN convention
for variable obstacle/red counts, e.g. ``_red_active``) -- ``.npz`` shards,
no new dependency.  Every value is ego-centric (relative to the red) and
arena/speed-normalised, matching ``_build_obs``'s existing convention
(``wall_distances = (x, L-x, y, L-y) / L``).

Run:
    python scripts/collect_red_motion_dataset.py --episodes 200 --steps 150
    python scripts/collect_red_motion_dataset.py --episodes 2000 --out data/red_motion --shard-size 20000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.env.entities import BLUE_UAV, RED_TARGET
from isr.env.pursuit_env import PursuitEnv
from isr.agents.heuristics import GreedyPursuer, RandomAgent
from isr.agents.stochastic_red import StochasticRed

V_NORM = BLUE_UAV.v_max     # shared velocity normaliser (blue and obstacle)

# Ranges centred on docs/tracking_diagnostics.md Sec. 7's validated values.
RED_RANGES = dict(
    heading_noise_std=(0.20, 0.50),
    heading_rho=(0.70, 0.95),
    commit_prob=(0.20, 1.00),
    commit_steps=(4, 12),
    threat_range=(15.0, 35.0),
    min_effort=(0.20, 0.80),
    magnitude_noise_std=(0.05, 0.30),
    magnitude_rho=(0.70, 0.95),
)


# --------------------------------------------------------------------- #
#  Red: per-red independent domain randomisation
# --------------------------------------------------------------------- #

def _sample_red_config(rng: np.random.Generator, p_det: float) -> Dict:
    if rng.random() < p_det:
        return {}          # every knob at its off default
    cfg = {}
    for k, (lo, hi) in RED_RANGES.items():
        v = rng.uniform(lo, hi)
        cfg[k] = int(round(v)) if k == "commit_steps" else float(v)
    return cfg


class MixedStochasticRed:
    """N independently-parameterised ``StochasticRed`` instances combined
    into one ``red_policy`` callable.

    One ``StochasticRed`` applies the SAME parameters to every red it
    drives; genuinely representative data needs each red to plausibly be a
    different "kind" of evader, so this holds one sub-instance per red
    slot.  Exact, not an approximation: ``run_from_nearest_uav`` (and
    hence ``StochasticRed``) is already per-red independent internally --
    it never reads another red's position or a different red's active
    flag -- so driving each slot with ``red_pos[i:i+1]`` is identical to
    the batched call restricted to index ``i``.
    """

    def __init__(self, n_red: int, rng: np.random.Generator,
                p_deterministic: float = 0.15) -> None:
        self.configs: List[Dict] = [_sample_red_config(rng, p_deterministic)
                                    for _ in range(n_red)]
        self.subs = [StochasticRed(seed=int(rng.integers(1 << 31)), **cfg)
                    for cfg in self.configs]

    def reset(self, n_red: int, seed: Optional[int] = None) -> None:
        assert n_red == len(self.subs), (
            "MixedStochasticRed must be rebuilt (not resized) per episode")
        for s in self.subs:
            s.reset(1)

    def __call__(self, blue_pos, red_pos, red_active,
                 obstacle_pos=None, obstacle_r=None, arena_size=None):
        out = np.zeros_like(red_pos, dtype=np.float32)
        for i, sub in enumerate(self.subs):
            out[i] = sub(blue_pos, red_pos[i:i + 1], red_active[i:i + 1],
                        obstacle_pos, obstacle_r, arena_size)[0]
        return out


# --------------------------------------------------------------------- #
#  Blue: scripted, no trained checkpoint needed (see module docstring)
# --------------------------------------------------------------------- #

_SHARED_GREEDY = GreedyPursuer()          # stateless: one instance suffices


def _sample_blue_heuristics(agent_names, rng: np.random.Generator) -> Dict:
    p_greedy = float(rng.uniform(0.3, 1.0))
    out = {}
    for a in agent_names:
        if rng.random() < p_greedy:
            out[a] = _SHARED_GREEDY
        else:
            out[a] = RandomAgent(seed=int(rng.integers(1 << 31)))
    return out


# --------------------------------------------------------------------- #
#  One episode -> a list of sample dicts
# --------------------------------------------------------------------- #

def collect_episode(rng: np.random.Generator, ep_id: int, steps: int,
                    n_blue_range, n_red_range, n_obs_range,
                    blue_cap: int, obs_cap: int, arena_size: float,
                    p_deterministic: float) -> List[Dict]:
    n_blue = int(rng.integers(n_blue_range[0], n_blue_range[1] + 1))
    n_red = int(rng.integers(n_red_range[0], n_red_range[1] + 1))
    n_obs = int(rng.integers(n_obs_range[0], n_obs_range[1] + 1))
    assert n_blue <= blue_cap and n_obs <= obs_cap

    seed = int(rng.integers(1 << 31))
    red_policy = MixedStochasticRed(n_red, rng, p_deterministic)
    # No n_obstacles_min: team sizes are already varied ACROSS episodes
    # (n_blue/n_red/n_obs sampled above); adding the env's OWN within-
    # capacity placed-count randomness on top would be a redundant second
    # layer of the same kind of diversity, not a new one.
    env = PursuitEnv(
        n_blue=n_blue, n_red=n_red, n_obstacles=n_obs,
        arena_size=arena_size, max_steps=steps + 5, capture_radius=3.0,
        sensor_radius=40.0, use_belief_maps=True,
        red_policy=red_policy, seed=seed,
    )
    obs_d, _info = env.reset(seed=seed)
    blues = _sample_blue_heuristics(env.possible_agents, rng)

    samples: List[Dict] = []
    L = env.arena_size
    for t in range(steps):
        if not env.agents:
            break
        # Snapshot the state the red policy will actually SEE.  PursuitEnv
        # calls red_policy at step 2, BEFORE obstacles move (2.5) and
        # before blues are integrated (3), so this pre-step snapshot is
        # exactly its input -- no ordering guesswork.
        #
        # Reading env._red_pos AFTER step() instead pairs state_{t+1} with
        # action_t: an off-by-one that injected a median 8.4 degrees of
        # spurious, unlearnable error (measured with zero-noise reds, which
        # must reproduce the policy EXACTLY), inflating the apparent noise
        # floor and capping what any model could reach.  See
        # test_action_matches_the_state_the_policy_saw.
        pre = dict(
            red_pos=env._red_pos.copy(), red_vel=env._red_vel.copy(),
            blue_pos=env._blue_pos.copy(), blue_vel=env._blue_vel.copy(),
            red_active=env._red_active.copy(),
            obs_pos=None if env._obstacle_pos is None else env._obstacle_pos.copy(),
            obs_vel=None if env._obstacle_vel is None else env._obstacle_vel.copy(),
            obs_r=None if env._obstacle_r is None else env._obstacle_r.copy(),
        )
        actions = {a: blues[a].act(obs_d[a], env, a) for a in env.agents}
        obs_d, _rew, _term, _trunc, _info = env.step(actions)

        act_a = env._last_red_action
        placed = 0 if pre["obs_pos"] is None else len(pre["obs_pos"])
        for r in np.where(pre["red_active"])[0]:
            rp, rv = pre["red_pos"][r], pre["red_vel"][r]

            blue_rel_pos = np.zeros((blue_cap, 2), dtype=np.float32)
            blue_rel_vel = np.zeros((blue_cap, 2), dtype=np.float32)
            blue_rel_pos[:n_blue] = (pre["blue_pos"] - rp) / L
            blue_rel_vel[:n_blue] = pre["blue_vel"] / V_NORM

            obs_rel_pos = np.zeros((obs_cap, 2), dtype=np.float32)
            obs_rel_vel = np.zeros((obs_cap, 2), dtype=np.float32)
            obs_radius = np.zeros((obs_cap,), dtype=np.float32)
            obs_mask = np.zeros((obs_cap,), dtype=bool)
            if placed > 0:
                ovel = (pre["obs_vel"] if pre["obs_vel"] is not None
                       else np.zeros((placed, 2), dtype=np.float32))
                obs_rel_pos[:placed] = (pre["obs_pos"] - rp) / L
                obs_rel_vel[:placed] = ovel / V_NORM
                obs_radius[:placed] = pre["obs_r"] / L
                obs_mask[:placed] = True

            wall_dist = np.array(
                [rp[0], L - rp[0], rp[1], L - rp[1]], dtype=np.float32) / L

            samples.append(dict(
                episode_id=ep_id, step=t, red_id=int(r),
                red_pos=rp.astype(np.float32), red_vel=rv.astype(np.float32),
                accel=act_a[r].astype(np.float32),
                blue_rel_pos=blue_rel_pos, blue_rel_vel=blue_rel_vel,
                obs_rel_pos=obs_rel_pos, obs_rel_vel=obs_rel_vel,
                obs_radius=obs_radius, obs_mask=obs_mask,
                wall_dist=wall_dist,
                n_blue=n_blue, n_obs_placed=placed,
            ))
    return samples


# --------------------------------------------------------------------- #
#  Shard writer
# --------------------------------------------------------------------- #

_FIELDS = ("episode_id", "step", "red_id", "red_pos", "red_vel", "accel",
          "blue_rel_pos", "blue_rel_vel", "obs_rel_pos", "obs_rel_vel",
          "obs_radius", "obs_mask", "wall_dist", "n_blue", "n_obs_placed")


def _write_shard(samples: List[Dict], path: Path) -> None:
    arrays = {f: np.stack([s[f] for s in samples]) for f in _FIELDS}
    np.savez_compressed(path, **arrays)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--out", type=str, default="data/red_motion")
    p.add_argument("--shard-size", type=int, default=20000)
    p.add_argument("--n-blue-min", type=int, default=2)
    p.add_argument("--n-blue-max", type=int, default=6)
    p.add_argument("--n-red-min", type=int, default=1)
    p.add_argument("--n-red-max", type=int, default=4)
    p.add_argument("--n-obs-min", type=int, default=0)
    p.add_argument("--n-obs-max", type=int, default=5)
    p.add_argument("--arena-size", type=float, default=130.0)
    p.add_argument("--p-deterministic", type=float, default=0.15,
                   help="fraction of REDS with no stochasticity at all")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    buf: List[Dict] = []
    shard_idx = 0
    n_total = 0
    t0 = time.time()
    for ep in range(a.episodes):
        buf.extend(collect_episode(
            rng, ep, a.steps,
            (a.n_blue_min, a.n_blue_max), (a.n_red_min, a.n_red_max),
            (a.n_obs_min, a.n_obs_max), a.n_blue_max, a.n_obs_max,
            a.arena_size, a.p_deterministic))
        while len(buf) >= a.shard_size:
            shard, buf = buf[:a.shard_size], buf[a.shard_size:]
            _write_shard(shard, out_dir / f"shard_{shard_idx:05d}.npz")
            n_total += len(shard)
            shard_idx += 1
        if (ep + 1) % max(1, a.episodes // 20) == 0:
            elapsed = time.time() - t0
            print(f"[{ep + 1}/{a.episodes}] episodes, "
                  f"{n_total + len(buf)} samples so far, {elapsed:.0f}s")

    if buf:
        _write_shard(buf, out_dir / f"shard_{shard_idx:05d}.npz")
        n_total += len(buf)
        shard_idx += 1

    print(f"\nDone: {n_total} samples across {shard_idx} shard(s) in {out_dir}")
    print("Fields per sample:", ", ".join(_FIELDS))
    print(f"blue_cap={a.n_blue_max}  obs_cap={a.n_obs_max}  "
          f"(pad capacities baked into this run's shards)")


if __name__ == "__main__":
    main()
