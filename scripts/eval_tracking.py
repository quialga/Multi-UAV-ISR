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

Everything is CPU and needs no training.

Run:
    python scripts/eval_tracking.py
    python scripts/eval_tracking.py --episodes 12 --steps 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr.env.pursuit_env import PursuitEnv, run_from_nearest_uav
from isr.tracking import MultiTargetTracker
from isr.tracking.metrics import ConsistencyAccumulator, MOTAccumulator


def _env(seed: int, **kw):
    base = dict(
        n_blue=5, n_red=3, n_obstacles=4, arena_size=130.0, max_steps=400,
        capture_radius=3.0, sensor_radius=40.0, use_belief_maps=True,
        enemy_belief_decay=0.99, enemy_belief_diffusion=0.2,
        sensor_pos_noise_std=1.0, sensor_vel_noise_std=0.1,
        sensor_noise_range_growth=1.0, track_conf_min=0.5,
        red_policy=run_from_nearest_uav, seed=seed,
    )
    base.update(kw)
    e = PursuitEnv(**base)
    e.reset(seed=seed)
    return e


def run(episodes: int, steps: int, match_dist: float, seed_base: int = 900):
    keys = ("raw", "belief", "oracle", "real")
    acc = {k: MOTAccumulator(match_dist) for k in keys}
    cons = {k: ConsistencyAccumulator() for k in ("oracle", "real")}
    n_tracks = {k: [] for k in keys}
    detectable = []          # the ceiling nothing can beat without PREDICTING

    for ep in range(episodes):
        e = _env(seed_base + ep)
        rng = np.random.default_rng(seed_base + ep)
        trk = {
            "oracle": MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0,
                                         oracle_association=True),
            "real": MultiTargetTracker(dt=1.0, a_max=1.0, vel_prior_std=1.0),
        }
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
            for key in ("oracle", "real"):
                trk[key].step(dets)
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

    return acc, cons, n_tracks, float(np.mean(detectable))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--match-dist", type=float, default=5.0,
                   help="max distance for a hypothesis to count as a match")
    a = p.parse_args()

    acc, cons, ntr, ceiling = run(a.episodes, a.steps, a.match_dist)

    labels = {"raw": "raw detections (floor)",
              "belief": "belief peaks (today)",
              "oracle": "KF + ORACLE assoc",
              "real":   "KF + real assoc"}
    keys = ("raw", "belief", "oracle", "real")
    cols = ("MOTA", "MOTP", "IDF1", "recall", "IDSW", "Frag", "FP", "FN", "MT", "ML")

    print(f"\n{a.episodes} episodes x {a.steps} steps, match gate "
          f"{a.match_dist} m\n")
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
    for k in ("oracle", "real"):
        c = cons[k].summary()
        if c:
            print(f"  {labels[k]:<22} NEES {c.get('NEES', float('nan')):6.2f}"
                  f"   NIS {c.get('NIS', float('nan')):6.2f}")

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
