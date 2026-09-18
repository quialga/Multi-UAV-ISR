# The tracker as the actor's observation (`actor_obs="tracker"`)

Status: built and tested, **not yet trained**. The belief-map path is
unchanged and remains the control (`actor_obs="belief"`, the default).

## 1. Why: the belief path hands the actor ground truth

`PursuitEnv._build_enemy_tracks` / `_build_obstacle_tracks` fill the
actor's red and obstacle nodes from the belief map plus direct sensor
reads. Five things in that path are knowledge no sensor reported:

| leak | belief path | tracker path |
|---|---|---|
| number of red nodes | `n_red` — the true count | confirmed tracks, capped at K |
| memory slots | one per red **nobody sees** | no such concept |
| slot identity | a live slot carries the red's true index | tracks have their own ids |
| obstacle radius | the **true** radius when seen | the tracker's estimate |
| obstacle count | one per placed obstacle | confirmed obstacle tracks |

The tracker path removes all five. The **critic keeps everything**: its
`true_*` keys are byte-identical in both modes (CTDE — privileged
information at training time is the point).

## 2. What the actor sees

Both node types come from `isr/tracking/actor_graph.py`.

* **Only confirmed tracks.** A tentative track is a hypothesis the tracker
  itself has not accepted; it never becomes a node.
* **A red track is dropped above a position-sd cut-off**
  (`--tracker-sigma-cutoff`, 40 m, the sensor radius). Below it, a blue
  flying to the readout very likely has the red inside its radar; above it,
  no longer (§2.1). The tracker keeps the track internally — it can still
  be re-acquired — and only deletes it after 80 scans without a hit.
* **Overflow keeps the smallest σ.** K = 8 red slots (2 × `n_red`) and
  K = 12 obstacle slots (`n_obstacles` + 3) hold the measured maxima with
  margin.
* **Readout = dominant component; uncertainty = the whole mixture.** The
  position and velocity are the tracker's own readout (the highest-weight
  Gaussian, never the average of separated modes), while the covariance is
  the mixture's moment-matched one, so a track that is split between two
  hypotheses reports the spread rather than hiding it.
* **Presence is binary.** Node feature 0 and the edge visibility mask are
  1 for a present slot and 0 for padding. In the belief path a single
  `conf` was both the mask and a quality measure; here quality travels in
  its own features.

Node features (all bounded; sds are `tanh(sd / scale)`, correlations in
[-1, 1]):

| red slot (8) | obstacle slot (9) |
|---|---|
| present | present |
| sd_x, sd_y, ρ of position (scale = sensor radius) | radius / arena |
| sd_vx, sd_vy, ρ of velocity (scale = `vel_prior_std`) | radius sd (scale 2 m) |
| staleness = scans since the last hit / `max_coast_steps` | position sd_x, sd_y, ρ |
| | velocity sd_vx, sd_vy, ρ |

Edges are the usual 7-D `[rel_pos, rel_vel, range, bearing]` from each slot
to each blue, zeroed on padding slots — the same layout and ordering as the
belief path, so the GNN itself is untouched.

### 2.1 What σ is, in metres, and what the cut-off does with it

A track's uncertainty is a covariance MATRIX, not a distance. The cut-off
turns it into one number:

1. The state is `[px, py, vx, vy]` with a 4×4 covariance. Keep the 2×2
   position block, in m²:

   ```
   P_pos = [ σx²       ρ·σx·σy ]
           [ ρ·σx·σy   σy²     ]
   ```

   For a track holding several hypotheses, take the covariance of the whole
   mixture first (weighted covariances plus the spread between the
   components' means).
2. That matrix describes an uncertainty ELLIPSE: its eigenvectors are the
   ellipse's axes, its eigenvalues the variances along them.
3. `σ = sqrt(largest eigenvalue)` — the standard deviation along the most
   uncertain direction, in metres (`actor_graph.max_sd`).

Why the largest eigenvalue and not σx or σy: the ellipse can be rotated,
and then the axes mislead. With `σx = σy = 20 m` and correlation 0.9, the
eigenvalues are 760 and 40 m², i.e. **27.6 m** along the diagonal and
6.3 m across it — an ellipse stretched diagonally, which is what a red
fleeing on a diagonal produces. The largest eigenvalue catches the worst
case whatever the orientation.

That single number is used for two things only: the cut-off, and choosing
which tracks keep a slot when there are more tracks than slots. **The actor
receives the full matrix** as three features (σx, σy, ρ), so the network
does see the ellipse's shape and orientation.

Two things worth keeping in mind:

* **σ is one standard deviation, not a radius that "surely" contains the
  red.** For an honest round 2-D Gaussian the red is within 1σ only ~39% of
  the time. Measured, shown tracks do much better than that — with the v4
  model, 100% of shown track-steps have their red within 40 m — because
  most shown tracks sit well below the cut-off and because the covariance
  of a coasting track is still conservative
  (docs/tracking_diagnostics.md §11.7).
* **The cut-off acts on the uncertainty the tracker DECLARES, not on its
  actual error.** That is why the same 40 m has meant different things as
  the model changed: an over-cautious filter declares a large σ and the
  cut-off removes tracks that were fine, an honest one lets them through.
  With the v4 model the filter is still ~33% over-cautious, so the cut-off
  errs safe — every shown track under it has its red within 40 m
  (docs/tracking_diagnostics.md §11.7).

## 3. No ground truth anywhere in the actor path

The detections are physics: they come from the true geometry, as a real
radar's would. Everything the trackers then *assume* is estimated:

* the red tracker's **learned motion model** takes its obstacle context
  (centres, velocities, radii) from the **obstacle tracker's** confirmed
  tracks, not from the env's obstacles;
* the **coverage** test that decides whether a miss counts (§6.1) tests
  occlusion against those same estimated disks (`coverage.disk_occluder`),
  not the true ones.

Order inside a step: obstacle tracker → its estimates → red tracker. The
trackers advance **once per `env.step`**; the observation builder only
reads them, so building the observation twice cannot advance them twice
(`tests/test_tracker_observation.py`).

## 4. Configuration

| flag / key | default | meaning |
|---|---|---|
| `--actor-obs` | `belief` | `tracker` switches the actor's source |
| `--red-motion-ckpt` | None | learned red motion model; None = constant velocity |
| `--tracker-red-slots` | 8 | K red slots |
| `--tracker-obstacle-slots` | 12 | K obstacle slots |
| `--tracker-sigma-cutoff` | 40.0 | red position-sd cut-off (m) |
| `--clutter-rate` | 0.2 | false plots per blue per scan (tracker path only) |
| `--obstacle-radius-noise-std` | 2.0 | radius measurement noise (tracker path only) |

The tracker configurations themselves are not CLI knobs: they are the
measured ones, in `isr/tracking/actor_graph.py` (red: 3-of-4 with deadline,
coverage-aware in-view budget 3, coast cap 80, re-acquisition after 4;
obstacles: 3-of-4 with deadline, never forgets, duplicate merge, static
model when the scenario has no moving obstacles).

In tracker mode the belief map is **not computed at all**
(`use_belief_maps` follows `--actor-obs`), so `belief_track_error` no longer
applies.  `PursuitEnv.tracker_diagnostics()` replaces it, drained once per
rollout by the trainer: mean distance from each SHOWN red node to the
nearest active red, plus NEES (honest uncertainty, target 4.0) and NIS.
The error is not comparable in LEVEL with the belief map's: that one always
has a peak per red, this one only counts confirmed tracks under the sd
cut-off.

## 5. Cost

Per env step (step + observation), training geometry (L=200, 7 blues,
4 reds, 9 obstacles), clutter 0.2, one torch thread:

| | step | observation | total |
|---|---|---|---|
| belief (control) | 4.19 ms | 3.64 ms | 7.84 ms |
| tracker, constant velocity | 6.74 ms | 1.23 ms | 7.97 ms |
| tracker, learned motion | 12.81 ms | 1.41 ms | 14.22 ms |

Two optimisations were needed to get there, both verified not to change
what they compute:

* **edge bearings vectorised** over edges instead of a per-edge Python
  loop (~20 ms per step at this geometry, in BOTH modes) — bit-identical on
  154k random edges including zero velocities and zero ranges;
* **obstacle gating batched** per observing blue with the 2×2 inverse in
  closed form — agrees with the per-pair version to 1e-15 relative, with no
  gate decision changed.

End to end, `scripts/train_stage4.py` with 4 envs in-process and one torch
thread: 34–36 steps/s in belief mode, 39–41 in tracker mode with constant
velocity. (With torch's default thread count both are far slower —
oversubscription, not the observation.)

## 6. What is not done

* **Nothing is trained yet.** Whether the tracker observation trains
  better than the belief map is exactly the open question.
* The learned motion model is now **v4**, retrained in the training
  distribution (§10.1), with `sigma_a_model` 0.10 and `merge_gate` 2.0
  (docs/tracking_diagnostics.md §11.7).  It tracks much better than v3
  (MOTA 0.70/0.63 against 0.43/0.51) and costs more, because keeping the
  mixture alive means more components to predict, gate and reduce.  The
  gate and the merge search have been batched (median 18.7 -> 11.6 ms per
  step); what dominates now is the adapter's per-component branch
  construction and the obstacle tracker.
* The filter stays over-cautious by design: NEES median 1.9 against 3.36,
  so the σ the actor sees is ~33% larger than honest.  That is the price of
  summarising a multimodal belief with one Gaussian; it errs safe for the
  cut-off (every shown track under 40 m has its red within 40 m).  Training
  logs NEES and NIS per rollout (`tracker/nees`, `tracker/nis`).
* **Moving obstacles are not ready.** The env switches the obstacle tracker
  away from the static model automatically, but then drifting tracks come
  back (0.27 misleading per step) and nothing can delete them yet; they
  also make the duplicate merge riskier. docs/tracking_diagnostics.md §9.6
  ("What changes with moving obstacles") lists what is needed first.
* Region / coverage nodes (docs/search_design.md) are a separate stage and
  are not part of this one.
