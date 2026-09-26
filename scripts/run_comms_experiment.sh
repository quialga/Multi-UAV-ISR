#!/usr/bin/env bash
# Does an unlimited blue-to-blue datalink produce a better policy?
#
# docs/stage4_results.md Sec. 16 showed the cheap version of this experiment
# cannot answer it.  Flipping comms_radius on a trained checkpoint collapses
# it from 2.72 to 1.26, but the policy aggregates ally messages with an
# UNNORMALISED SUM, so opening the radio multiplies that term by 2.5x against
# what it trained on.  The damage tracked the magnitude, not the information:
# NARROWING to 20 m is a gentler perturbation and the score went slightly UP.
# Neither direction said anything about ally information.
#
# So every arm is trained FROM SCRATCH -- clone, then PPO -- and each converges
# under its own aggregate scale.  Comparing two such arms is an end-to-end
# comparison of two trained policies, which is the question actually asked.
#
# THREE arms, not two.  20 and inf answer "does more comms help".  40 is the
# incumbent every result in stage4_results.md was measured under, and without
# it neither new arm can be compared to the existing 2.72 +- 0.08 (Sec. 15.2),
# because those came through a 1.4 -> 1.5 curriculum rather than from scratch
# -- comparing across both changes at once would confound comms with the
# training path.  Three points also separate a monotone trend from a peak.
#
# Ordered inf, 20, 40 and the summary is rewritten after EVERY stage, so the
# decisive comparison is readable as soon as the first two arms land without
# waiting on the third.
#
# Unattended by construction: no prompts, no interactive git, stages are
# sequential (8 cores, and each run already uses 1 main + 4 workers, so
# running arms concurrently would only trade wall-clock for contention), and
# a failed stage is recorded and skipped rather than killing the batch.
#
# Run:
#     bash scripts/run_comms_experiment.sh
#     nohup bash scripts/run_comms_experiment.sh > runs/comms_experiment.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

OUT="runs/comms_experiment"
SUMMARY="$OUT/SUMMARY.md"
mkdir -p "$OUT"

# Red speed 1.5: blue v_max is also 1.5, so a stern chase can never close and
# capture requires interception.  Pure pursuit scores 0.66-1.06 there against
# the policy's 2.72, which is the dynamic range that makes an effect visible
# (Sec. 15).  At red 1.4 the metric is nearly saturated and Sec. 14 is the
# record of a saturated metric wasting an experiment.
RED_V_MAX=1.5

# Reproduces the reference env exactly: diffing this against
# ppo_red15_v2/final.pt's saved args shows no physics differences, only the
# run/optimiser knobs train_bc.py sets itself.
ENV_BASE="--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 \
--belief-grid-size 26 --max-steps 200 --actor-obs tracker \
--use-staleness --staleness-regions 5 --red-v-max $RED_V_MAX"

# Clone: 60 rounds is what reached the expert in runs/stage4/bc_tracker_v1
# (2.88/3 in 120.9 min); the script's own default of 40 is lower.
BC_ROUNDS=60

# PPO from a clone, exactly the recipe that worked in ppo_from_bc_v1:
#   * warm start from the clone's BEST, not final;
#   * --reset-log-std -1.2 (sigma ~ 0.30).  The clone is fit with a
#     cross-entropy-like loss that never trains log_std, so it arrives
#     meaningless; leaving it destroys the clone and resetting it to 1.0
#     destroys it faster (Sec. 12).  -1.2 is also where ppo_red15_v2's own
#     log_std settled after 1.28 M steps, so it is not an arbitrary pick.
#   * lr 3e-05.  Sec. 15.1 showed kl running 10x under target at 6e-05 on a
#     CONVERGED policy, but that is not evidence about a fresh clone, which is
#     the fragile case -- so this keeps the proven value.
PPO_ROLLOUTS=150
PPO_LR=3e-05
RESET_LOG_STD=-1.2

# label:comms_radius
ARMS=("inf:inf" "narrow:20" "incumbent:40")

started=$(date '+%Y-%m-%d %H:%M:%S')
declare -A STATUS

write_summary() {
  # `local` is NOT optional here.  Bash functions share the caller's scope, so
  # an earlier version of this loop reused the names `label` and `cr` and
  # clobbered the arm loop's own variables on the first call -- every log line,
  # STATUS key and output filename after that point carried the LAST arm's
  # label, which would have made arm 2 overwrite arm 1's results.  The
  # training itself was unaffected (bc_name/ppo_name/env_args are computed
  # before the first call and never reassigned), which is exactly what made it
  # quiet.
  local arm label cr stage key st res num
  {
    echo "# comms_radius experiment"
    echo
    echo "Started $started, updated $(date '+%Y-%m-%d %H:%M:%S')."
    echo
    echo "Red v_max $RED_V_MAX, clone $BC_ROUNDS rounds, PPO $PPO_ROLLOUTS"
    echo "rollouts at lr $PPO_LR warm-started from the clone's best.pt with"
    echo "--reset-log-std $RESET_LOG_STD.  Every arm trained from scratch, so"
    echo "each converges under its own aggregate scale (see Sec. 16)."
    echo
    echo "Reference, for context only -- NOT directly comparable, it came"
    echo "through a 1.4 -> 1.5 curriculum rather than from scratch:"
    echo "\`ppo_red15_v2/final.pt\` = 2.72 +- 0.08 vs the evader at red 1.5."
    echo
    echo "| arm | comms_radius | stage | status | caught vs evader |"
    echo "|---|---|---|---|---|"
    for arm in "${ARMS[@]}"; do
      label="${arm%%:*}"; cr="${arm##*:}"
      for stage in bc ppo; do
        key="$label/$stage"
        st="${STATUS[$key]:-pending}"
        res="$OUT/${label}_${stage}_eval.txt"
        num="--"
        if [ -f "$res" ]; then
          num=$(grep -E "^\s*run\s" "$res" 2>/dev/null \
                | awk '{print $2" "$3" "$4}')
          [ -z "$num" ] && num="--"
        fi
        echo "| $label | $cr | $stage | $st | $num |"
      done
    done
    echo
    echo "Logs: \`$OUT/<arm>_<stage>.log\`.  Checkpoints under"
    echo "\`runs/stage4/comms_<arm>_<stage>/\`."
  } > "$SUMMARY"
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Resume: a stage whose checkpoint already exists is not redone.  Cheap
# insurance on a ~31 h batch -- the first launch had to be killed five minutes
# into arm 1's PPO to fix a bookkeeping bug, and without this the 1.6 h clone
# would have been thrown away with it.  Delete a run's directory to force it.
have_ckpt() { [ -s "runs/stage4/$1/best.pt" ] || [ -s "runs/stage4/$1/final.pt" ]; }

write_summary
log "comms experiment starting; ${#ARMS[@]} arms, summary at $SUMMARY"

for arm in "${ARMS[@]}"; do
  label="${arm%%:*}"
  cr="${arm##*:}"
  bc_name="comms_${label}_bc"
  ppo_name="comms_${label}_ppo"
  env_args="$ENV_BASE --comms-radius $cr"

  # ---- clone -------------------------------------------------------- #
  if have_ckpt "$bc_name"; then
    log "$label: clone already present, reusing runs/stage4/$bc_name"
    STATUS["$label/bc"]="reused"
    $PY scripts/eval_checkpoint.py \
        --ckpt "runs/stage4/$bc_name/best.pt" --episodes 50 --reds all \
        > "$OUT/${label}_bc_eval.txt" 2>&1 || true
    write_summary
  else
  log "$label: clone (comms_radius=$cr)"
  STATUS["$label/bc"]="running"; write_summary
  if $PY scripts/train_bc.py \
        --env-args "$env_args" \
        --n-rounds "$BC_ROUNDS" \
        --seed 0 --device cpu \
        --run-name "$bc_name" > "$OUT/${label}_bc.log" 2>&1; then
    STATUS["$label/bc"]="done"
    $PY scripts/eval_checkpoint.py \
        --ckpt "runs/stage4/$bc_name/best.pt" --episodes 50 --reds all \
        > "$OUT/${label}_bc_eval.txt" 2>&1 || true
    log "$label: clone done"
  else
    STATUS["$label/bc"]="FAILED"
    STATUS["$label/ppo"]="skipped"
    write_summary
    log "$label: clone FAILED (see $OUT/${label}_bc.log); skipping its PPO"
    continue
  fi
  write_summary
  fi

  # ---- PPO ---------------------------------------------------------- #
  # NOT resumed on an existing checkpoint: PPO writes best.pt early and keeps
  # going, so a half-finished run looks identical to a finished one from the
  # filesystem.  Reusing one would silently report a truncated arm as a
  # result.  Delete runs/stage4/comms_<arm>_ppo to redo it.
  if have_ckpt "$ppo_name" && [ -f "$OUT/${label}_ppo_final_eval.txt" ]; then
    log "$label: PPO already complete, reusing"
    STATUS["$label/ppo"]="reused"; write_summary
    continue
  fi
  rm -rf "runs/stage4/$ppo_name"
  log "$label: PPO"
  STATUS["$label/ppo"]="running"; write_summary
  if $PY scripts/train_stage4.py \
        $env_args \
        --sensor-radius 40.0 \
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
    STATUS["$label/ppo"]="done"
    for ck in best final; do
      $PY scripts/eval_checkpoint.py \
          --ckpt "runs/stage4/$ppo_name/$ck.pt" --episodes 50 --reds all \
          > "$OUT/${label}_ppo_${ck}_eval.txt" 2>&1 || true
    done
    cp "$OUT/${label}_ppo_final_eval.txt" "$OUT/${label}_ppo_eval.txt" 2>/dev/null
    log "$label: PPO done"
  else
    STATUS["$label/ppo"]="FAILED"
    log "$label: PPO FAILED (see $OUT/${label}_ppo.log)"
  fi
  write_summary
done

write_summary
log "all arms finished; summary at $SUMMARY"
