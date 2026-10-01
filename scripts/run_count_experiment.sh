#!/usr/bin/env bash
# Does attention's advantage GROW with the target count?  (Sec. 20)
#
# Sec. 19 showed attention beats the summed aggregation by +0.88, and that most
# of it is normalisation: a plain mean recovers +0.48 of that, leaving +0.30
# for discrimination at 2.3 SE.  So the claim attention was built for -- a blue
# deciding whether a target needs one ally or three -- is still open, because
# selectivity only pays when there is something to select between.
#
# Arms are MEAN against ATTENTION.  The sum is excluded: Sec. 19 established it
# is broken at comms inf (2.62/3 against a STATIONARY red), so including it
# would re-measure a known breakage.
#
# Every parameter below was decided by a cheap measurement, and three of four
# design intuitions were wrong (Sec. 20.3):
#   * 8 tracker slots: occupancy is identical at 8/12/16 and the limit is hit
#     in 0.0% of steps, so 5 reds do not saturate it.  No architecture change.
#   * 5v5: the value of coordination triples against 3v3 (+0.43 vs +0.14),
#     because without surplus blues allocation binds instead of having slack.
#   * max_steps 200: raising it SHRINKS the coordination advantage (+0.43 at
#     200, +0.40 at 300, +0.33 at 400) because given time, uncoordinated
#     pursuit catches everything anyway.  Time pressure is the point.
#   * red 1.4 not 1.5: Sec. 19's red-1.4 arms were converging while the
#     red-1.5 arms were still climbing, and comparing degradation between
#     unconverged policies measures where each stopped, not what it can do.
#
# Training SAMPLES the count (--n-red-min 1) and evaluation FIXES it, so the
# policy must learn to handle variability while the measurement isolates each
# level.  Note env_kwargs_from_checkpoint silently dropped n_red_min until
# 2026-10-01, which would have trained the clone at fixed capacity --
# tests/test_variable_counts.py now pins it.
#
# Run:
#     nohup bash scripts/run_count_experiment.sh > runs/count_experiment.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

OUT="runs/count_experiment"
SUMMARY="$OUT/SUMMARY.md"
mkdir -p "$OUT"

COUNTS="1,2,3,4,5"
SWEEP_EPISODES=100
N_RED=5
RED_V_MAX=1.4

ENV_BASE="--arena-size 130 --n-blue 5 --n-red $N_RED --n-red-min 1 \
--n-obstacles 0 --belief-grid-size 26 --max-steps 200 --actor-obs tracker \
--use-staleness --staleness-regions 5 --comms-radius inf \
--red-v-max $RED_V_MAX"

BC_ROUNDS=60
PPO_ROLLOUTS=150
PPO_LR=3e-05
RESET_LOG_STD=-1.2

ARMS=("mean:--mean-agg" "attention:--attention --n-heads 4")

started=$(date '+%Y-%m-%d %H:%M:%S')
declare -A STATUS

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
have_ckpt() { [ -s "runs/stage4/$1/best.pt" ] || [ -s "runs/stage4/$1/final.pt" ]; }

write_summary() {
  # `local` is mandatory -- bash functions share the caller's scope, and an
  # earlier version of this pattern clobbered the arm loop's variables, which
  # would have made arm 2 overwrite arm 1's outputs.
  local arm label f
  {
    echo "# Does attention's advantage grow with the target count?  (Sec. 20)"
    echo
    echo "Started $started, updated $(date '+%Y-%m-%d %H:%M:%S')."
    echo
    echo "Arms: MEAN vs ATTENTION.  5 blues / $N_RED reds capacity, 1-$N_RED"
    echo "sampled per episode during training, red v_max $RED_V_MAX,"
    echo "comms_radius inf, 8 tracker slots, max_steps 200.  Clone"
    echo "($BC_ROUNDS rounds) then PPO ($PPO_ROLLOUTS rollouts, lr $PPO_LR)."
    echo
    echo "Primary metric is the captured FRACTION at each fixed count on"
    echo "matched seeds -- NOT a fitted slope, which Sec. 20.4 showed is"
    echo "fragile on these non-linear curves (sign flips between red speeds)."
    echo
    echo "Measured baselines at 5v5 red 1.4: \`ObsGreedy\` 3.02 +- 0.17,"
    echo "\`AssignGreedy\` 3.45 +- 0.18."
    echo
    echo "| arm | stage | status |"
    echo "|---|---|---|"
    for arm in "${ARMS[@]}"; do
      label="${arm%%:*}"
      echo "| $label | clone | ${STATUS[$label/bc]:-pending} |"
      echo "| $label | ppo | ${STATUS[$label/ppo]:-pending} |"
      echo "| $label | sweep | ${STATUS[$label/sweep]:-pending} |"
    done
    echo
    for arm in "${ARMS[@]}"; do
      label="${arm%%:*}"
      f="$OUT/${label}_sweep.txt"
      if [ -f "$f" ]; then
        echo "## $label"
        echo '```'
        cat "$f"
        echo '```'
        echo
      fi
    done
    echo "Logs \`$OUT/<arm>_<stage>.log\`; checkpoints \`runs/stage4/cnt_<arm>_*/\`."
  } > "$SUMMARY"
}

write_summary
log "count experiment starting; ${#ARMS[@]} arms, summary at $SUMMARY"

# Heuristic baselines, once -- they do not depend on either arm, and having
# them in the same file and seed base as the arms is what makes the
# comparison paired rather than quoted from elsewhere.
if [ ! -f "$OUT/baseline_assign_sweep.txt" ]; then
  log "baselines: heuristic sweeps"
  for h in obs assign; do
    $PY scripts/eval_red_count_sweep.py \
        --ckpt runs/stage4/att_attention_ppo15/final.pt --heuristic "$h" \
        --red-v-max "$RED_V_MAX" --counts "$COUNTS" \
        --episodes "$SWEEP_EPISODES" \
        > "$OUT/baseline_${h}_sweep.txt" 2>&1 || true
  done
  write_summary
fi

for arm in "${ARMS[@]}"; do
  label="${arm%%:*}"
  flags="${arm#*:}"
  bc_name="cnt_${label}_bc"
  ppo_name="cnt_${label}_ppo"

  # ---- clone ---------------------------------------------------------- #
  if have_ckpt "$bc_name"; then
    log "$label: clone present, reusing"
    STATUS["$label/bc"]="reused"
  else
    log "$label: clone ($flags)"
    STATUS["$label/bc"]="running"; write_summary
    if $PY scripts/train_bc.py \
          --env-args "$ENV_BASE $flags" \
          --n-rounds "$BC_ROUNDS" --seed 0 --device cpu \
          --run-name "$bc_name" > "$OUT/${label}_bc.log" 2>&1; then
      STATUS["$label/bc"]="done"
    else
      STATUS["$label/bc"]="FAILED"
      STATUS["$label/ppo"]="skipped"; STATUS["$label/sweep"]="skipped"
      write_summary; log "$label: clone FAILED; skipping this arm"
      continue
    fi
  fi
  write_summary

  # ---- PPO ------------------------------------------------------------ #
  # Not resumed from a bare checkpoint: PPO writes best.pt early, so a
  # truncated run is indistinguishable from a finished one on disk.
  # Completion is judged by the sweep file, written only after training ends.
  if have_ckpt "$ppo_name" && [ -f "$OUT/${label}_sweep.txt" ]; then
    log "$label: PPO + sweep already complete, reusing"
    STATUS["$label/ppo"]="reused"; STATUS["$label/sweep"]="reused"
    write_summary; continue
  fi
  rm -rf "runs/stage4/$ppo_name"
  log "$label: PPO"
  STATUS["$label/ppo"]="running"; write_summary
  if $PY scripts/train_stage4.py \
        $ENV_BASE --sensor-radius 40.0 $flags \
        --n-envs 64 --rollout-steps 200 --n-rollouts "$PPO_ROLLOUTS" \
        --n-epochs 10 --mb-size 512 --lr "$PPO_LR" --ent-coef 0.008 \
        --target-kl 0.03 --n-workers 4 --torch-threads 4 \
        --eval-interval 5 --eval-episodes 25 \
        --best-ckpt-metric det_caught \
        --red-policy-mix stationary:1,random:1,run:1 \
        --warm-start-full "runs/stage4/$bc_name/best.pt" \
        --reset-log-std "$RESET_LOG_STD" \
        --share-hidden-via-gnn --seed 0 --device cpu \
        --run-name "$ppo_name" > "$OUT/${label}_ppo.log" 2>&1; then
    STATUS["$label/ppo"]="done"; write_summary
    log "$label: count sweep"
    STATUS["$label/sweep"]="running"; write_summary
    if $PY scripts/eval_red_count_sweep.py \
          --ckpt "runs/stage4/$ppo_name/final.pt" --counts "$COUNTS" \
          --episodes "$SWEEP_EPISODES" --json "$OUT/${label}_sweep.json" \
          > "$OUT/${label}_sweep.txt" 2>&1; then
      STATUS["$label/sweep"]="done"
    else
      STATUS["$label/sweep"]="FAILED"
    fi
  else
    STATUS["$label/ppo"]="FAILED"
    log "$label: PPO FAILED (see $OUT/${label}_ppo.log)"
  fi
  write_summary
done

write_summary
log "all arms finished; summary at $SUMMARY"
