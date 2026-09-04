"""
scripts/eval_tracking.py — MOT evaluation of the tracker vs the belief map.

Runs three configurations over the same episodes so the error can be
attributed rather than guessed:

  1. BELIEF PEAKS  — what the policy consumes today.  Its identity is
     handed over by the simulator (the track slot IS the red index), so its
     IDF1/IDSW are meaningless and are reported only for completeness.
  2. KF + ORACLE association — associates from the labels.  This is the
     CEILING of the filter: if this is already poor, the filter is at fault.
  3. KF + REAL association — the full system.  The gap against (2) is
     exactly what data association costs.

With ``--learned CKPT`` two more rows appear, using the trained
red-motion model as the tracker's ``motion_model`` instead of the
constant-velocity default.  The gap against rows 2-3 is exactly what
PREDICTING the adversary's next acceleration buys — the number the whole
learned-motion line of work is for.  Because every configuration runs on
the SAME episodes, that gap is attribution, not comparison across runs.

Two idealisations in the learned rows, both stated rather than hidden:
obstacle geometry is taken as ground truth (in deployment it comes from
the separate obstacle tracker), and blue positions are exact — the latter
is not an idealisation at all, since blues are our OWN drones.

Everything is CPU and needs no training.

Run:
    python scripts/eval_tracking.py
    python scripts/eval_tracking.py --episodes 12 --steps 150
    python scripts/eval_tracking.py --learned runs/red_motion/model_v2.pt
    python scripts/eval_tracking.py --learned ... --stochastic-red
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.agents.stochastic_red import StochasticRed
from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.tracking import MultiTargetTracker
from isr.tracking.metrics import ConsistencyAccumulator, MOTAccumulator

# Representative stochastic adversary — the middle of the domain
# randomisation the collector trained on, not a new set of magic numbers.
STOCH_RED = dict(heading_noise_std=0.25, heading_rho=0.85, commit_prob=0.15,
                commit_steps=8, commit_angle=1.1, magnitude_noise_std=0.15)


def _env(seed: int, stochastic_red: bool = False, **kw):
    policy = run_from_nearest_uav
    if stochastic_red:
        policy = StochasticRed(seed=seed, **STOCH_RED)
    base = dict(
        n_blue=5, n_red=3, n_obstacles=4, arena_size=130.0, max_steps=400,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        enemy_belief_decay=0.99, enemy_belief_diffusion=0.2,
        sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
        sensor_noise_range_growth=1.0, track_conf_min=0.5,
        red_policy=policy, seed=seed,
    )
    base.update(kw)
    e = PursuitEnv(**base)
    e.reset(seed=seed)
    return e


def run(episodes: int, steps: int, match_dist: float, seed_base: int = 900,
        learned_ckpt: str = "", sigma_a_model: float = 0.35,
        max_branches: int = 4, stochastic_red: bool = False,
        max_misses: int = 5, merge_gate: float = 4.0,
        max_components: int = 8):
    learned_keys = ("lrn_oracle", "lrn_real") if learned_ckpt else ()
    keys = ("raw", "belief", "oracle", "real") + learned_keys
    acc = {k: MOTAccumulator(match_dist) for k in keys}
    cons = {k: ConsistencyAccumulator() for k in ("oracle", "real") + learned_keys}
    n_tracks = {k: [] for k in keys}
    detectable = []          # the ceiling nothing can beat without PREDICTING
    branch_counts = []

    motion = None
    if learned_ckpt:
        from isr.agents.learned_red_motion import (
            LearnedRedMotion, load_red_motion_model,
        )
        model, blue_cap, obs_cap = load_red_motion_model(learned_ckpt)
        motion = LearnedRedMotion(
            model, blue_cap, obs_cap, dt=1.0, a_max=1.0, arena_size=130.0,
            max_branches=max_branches, sigma_a_model=sigma_a_model)

    for ep in range(episodes):
        e = _env(seed_base + ep, stochastic_red=stochastic_red)
        rng = np.random.default_rng(seed_base + ep)
        common = dict(dt=1.0, a_max=1.0, vel_prior_std=1.0,
                     max_misses=max_misses)
        trk = {
            "oracle": MultiTargetTracker(oracle_association=True, **common),
            "real": MultiTargetTracker(**common),
        }
        if motion is not None:
            gs = dict(motion_model=motion, max_components=max_components,
                     merge_gate=merge_gate)
            trk["lrn_oracle"] = MultiTargetTracker(
                oracle_association=True, **gs, **common)
            trk["lrn_real"] = MultiTargetTracker(**gs, **common)
        for _ in range(steps):
            if not e.agents:
                break
            e.step({a: rng.uniform(-1, 1, 2).astype(np.float32)
                    for a in e.agents})

            active = np.where(e._red_active)[0]
            # GT ids must be unique ACROSS episodes: the red index repeats
            # every episode, so using it directly merges 8 different
            # trajectories into one and corrupts IDF1 / MT / ML / Frag, and
            # counts a bogus ID switch at every episode boundary.
            gt_ids = [f"e{ep}_r{int(r)}" for r in active]
            gt_pos = e._red_pos[active] if len(active) else np.zeros((0, 2))

            # --- 0. references that make the numbers interpretable -------
            # (a) the detectability ceiling: no tracker can exceed the
            #     fraction of targets currently in sensor range unless it
            #     PREDICTS through the gaps.
            if len(active):
                dd = np.linalg.norm(e._red_pos[active][:, None, :]
                                    - e._blue_pos[None, :, :], axis=-1)
                detectable.append(float((dd.min(axis=1) <= e.sensor_radius).mean()))
            # (b) the naive floor: report every raw return as a hypothesis,
            #     no filtering, no association, no memory.
            raw = e.raw_detections()
            acc["raw"].update(gt_ids, gt_pos,
                              [f"e{ep}_d{i}" for i in range(len(raw))],
                              np.array([d["z_pos"] for d in raw]) if raw
                              else np.zeros((0, 2)))
            n_tracks["raw"].append(len(raw))

            # --- 1. what the policy sees today: the belief track slots ----
            tp, tc, tr_id, _tv, _tvc = e._build_enemy_tracks()
            real_slots = [s for s in range(len(tc)) if tc[s] > 0]
            acc["belief"].update(gt_ids, gt_pos,
                                 [f"e{ep}_slot{s}" for s in real_slots],
                                 tp[real_slots] if real_slots else np.zeros((0, 2)))
            n_tracks["belief"].append(len(real_slots))

            # --- 2 & 3. the tracker -------------------------------------
            dets = raw
            # The learned model is conditional on where the blues are RIGHT
            # NOW, so the context has to be refreshed every step, before any
            # tracker that uses it predicts.  Blue positions are exact
            # because blues are OUR drones; obstacle geometry is ground
            # truth here (see the module docstring).
            if motion is not None:
                motion.set_context(
                    blue_pos=e._blue_pos, blue_vel=e._blue_vel,
                    obs_pos=getattr(e, "_obstacle_pos", None),
                    obs_vel=getattr(e, "_obstacle_vel", None),
                    obs_r=getattr(e, "_obstacle_r", None))
            for key in trk:
                trk[key].step(dets)
                if key.startswith("lrn_"):
                    branch_counts.extend(
                        len(t.components) for t in trk[key].tracks)
                cons[key].add_nis(trk[key].last_nis)
                conf = trk[key].confirmed_tracks()
                acc[key].update(gt_ids, gt_pos,
                                [t.id for t in conf],
                                np.array([t.pos for t in conf])
                                if conf else np.zeros((0, 2)))
                n_tracks[key].append(len(conf))
                # NEES needs the matching truth; use the nearest active red
                # within the match gate so a mis-associated track is not
                # scored against a target it never claimed.
                for t in conf:
                    if not len(active):
                        continue
                    d = np.linalg.norm(e._red_pos[active] - t.pos, axis=1)
                    j = int(d.argmin())
                    if d[j] <= match_dist:
                        r = active[j]
                        x_true = np.concatenate([e._red_pos[r], e._red_vel[r]])
                        cons[key].add_nees(t.x, t.P, x_true)

    return (acc, cons, n_tracks, float(np.mean(detectable)),
            np.array(branch_counts) if branch_counts else np.zeros(0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--match-dist", type=float, default=5.0,
                   help="max distance for a hypothesis to count as a match")
    p.add_argument("--learned", type=str, default="",
                   help="checkpoint for the learned red-motion model; adds "
                        "two rows using it as the tracker's motion_model")
    p.add_argument("--sigma-a-model", type=float, default=0.35,
                   help="per-branch model-error std (NOT yet tuned; sweep "
                        "this against NEES before trusting the rows)")
    p.add_argument("--max-branches", type=int, default=4,
                   help="cap on MODES per prediction; no mass is discarded "
                        "at any setting, only resolved more or less finely")
    p.add_argument("--max-misses", type=int, default=5,
                   help="consecutive misses before a track dies -- this is "
                        "the COAST BUDGET, and it caps how much any motion "
                        "model can buy by predicting through sensor gaps")
    p.add_argument("--merge-gate", type=float, default=4.0,
                   help="squared Mahalanobis distance below which two "
                        "components are folded together; NEGATIVE disables "
                        "merging entirely (d^2 is never < 0)")
    p.add_argument("--max-components", type=int, default=8,
                   help="cap on Gaussian-Sum components per track -- the "
                        "binding constraint once merging is off, since "
                        "branches multiply every step")
    p.add_argument("--stochastic-red", action="store_true",
                   help="evaluate against the stochastic adversary the model "
                        "was trained on, instead of the deterministic one")
    a = p.parse_args()

    acc, cons, ntr, ceiling, branches = run(
        a.episodes, a.steps, a.match_dist, learned_ckpt=a.learned,
        sigma_a_model=a.sigma_a_model, max_branches=a.max_branches,
        stochastic_red=a.stochastic_red, max_misses=a.max_misses,
        merge_gate=a.merge_gate, max_components=a.max_components)

    labels = {"raw": "raw detections (floor)",
              "belief": "belief peaks (today)",
              "oracle": "KF + ORACLE assoc",
              "real":   "KF + real assoc",
              "lrn_oracle": "LEARNED + ORACLE assoc",
              "lrn_real":   "LEARNED + real assoc"}
    keys = ("raw", "belief", "oracle", "real")
    if a.learned:
        keys = keys + ("lrn_oracle", "lrn_real")
    cols = ("MOTA", "MOTP", "IDF1", "recall", "IDSW", "Frag", "FP", "FN", "MT", "ML")

    red_kind = "STOCHASTIC" if a.stochastic_red else "deterministic"
    print(f"\n{a.episodes} episodes x {a.steps} steps, match gate "
          f"{a.match_dist} m, {red_kind} red, coast budget "
          f"{a.max_misses} steps\n")
    hdr = f"{'configuration':<22}" + "".join(f"{c:>8}" for c in cols) + f"{'tracks':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        s = acc[k].summary()
        row = f"{labels[k]:<22}"
        for c in cols:
            v = s[c]
            row += f"{v:>8.2f}" if abs(v) < 1000 else f"{v:>8.0f}"
        row += f"{np.mean(ntr[k]):>8.2f}"
        print(row)

    print("\nfilter consistency (NEES target 4.0; NIS target ~2 for the "
          "2-D position update)")
    for k in keys:
        if k in ("raw", "belief"):
            continue
        c = cons[k].summary()
        if c:
            print(f"  {labels[k]:<22} NEES {c.get('NEES', float('nan')):6.2f}"
                  f"   NIS {c.get('NIS', float('nan')):6.2f}")

    if a.learned:
        print(f"\nLEARNED rows use {a.learned}")
        print(f"  sigma_a_model {a.sigma_a_model}  "
              f"max_branches {a.max_branches}")
        if len(branches):
            print(f"  components per track: mean {branches.mean():.2f}  "
                  f"p90 {np.percentile(branches, 90):.0f}  "
                  f"max {branches.max()}")
        print("  NEES above 4.0 means the branches are OVERCONFIDENT: raise")
        print("  sigma_a_model.  Overconfidence here is the dangerous")
        print("  direction -- a tight gate rejects true detections, tracks")
        print("  die, and recall falls BELOW the constant-velocity rows,")
        print("  which reads as 'the learned model is worse' when the real")
        print("  fault is an untuned covariance.")
        print("  Obstacle geometry is ground truth in these rows; in")
        print("  deployment it comes from the obstacle tracker.")

    print(f"\nDETECTABILITY CEILING = {ceiling:.2f} — the fraction of active")
    print("targets inside sensor_radius.  Recall cannot exceed this without")
    print("PREDICTING through the gaps, so judge recall against it, not 1.0.")

    print("\nMOTP is the localisation error in metres over MATCHED pairs, so")
    print("it is CONDITIONAL on matching: the belief map's peaks that sit")
    print("19-40 m out are counted as FP/FN and never enter its MOTP.")
    print("The belief row's IDF1/IDSW are not meaningful: its slot identity")
    print("comes from the simulator, not from association.")
    print("\nMOTA is ~degenerate here: FN is >95% of its loss, so MOTA ~ recall")
    print("and carries no information the recall column does not.")


if __name__ == "__main__":
    main()
