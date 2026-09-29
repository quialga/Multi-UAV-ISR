#!/usr/bin/env bash
# Sum vs attention, the experiment docs/stage4_results.md Sec. 18.4 fixes.
#
# Sec. 16 showed the actor aggregates ally messages with an UNNORMALISED SUM,
# so the count of live edges scales what reaches update_mlp.  That is what made
# comms_radius uninterpretable and what blocks variable entity counts.  Sec. 17
# then failed to settle the comms question for a different reason: from-scratch
# arms land ~1.2 below the curriculum and were still rising at 150 rollouts.
#
# So: two arms differing ONLY in the aggregation, both at comms_radius inf,
# both trained from birth through a 1.4 -> 1.5 curriculum.  Neither is ever
# displaced -- an earlier plan warm-started both from a comms-40 checkpoint,
# which displaces BOTH (the sum arm upward by the magnitude jump, the attention
# arm downward because a convex combination is worth ~1 message against the
# ~2.11 its weights were trained on).  Two differently broken warm starts are
# not a controlled comparison.
#
# Run at comms inf, not 40, because the two settings test different halves:
# at 40 the live-ally count swings 0-4 (so NORMALISATION is exercised) but
# attention is a literal no-op in the 38% of samples with 0 or 1 live edge; at
# inf there are always four allies to rank, which is the DISCRIMINATION half --
# the one that matters for deciding whether a target needs one blue or three.
#
# EXPECT PARITY, not a win.  Stated up front so the result cannot be
# reinterpreted afterwards: with three reds and the team's sensors covering
# 1.49x the arena (Sec. 17.4), there may simply be nothing for discrimination
# to exploit.  Parity is a PASS -- it licenses attention for the regimes where
# it should pay (variable counts, heterogeneous speeds), which the sum blocks
# outright.  The margin is +-0.15, which is why the final eval is 200 episodes
# and not 50: at 50 the CI on a difference is +-0.22, too coarse to claim
# parity against a 2.72 reference.
#
# Unattended: sequential (8 cores, and --torch-threads above 4 measured SLOWER,
# so there is nothing to gain from overlapping), resumable, and a failed stage
# is recorded and skipped rather than killing the batch.
#
# Run:
#     nohup bash scripts/run_attention_experiment.sh > runs/attention_experiment.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

OUT="runs/attention_experiment"
SUMMARY="$OUT/SUMMARY.md"
mkdir -p "$OUT"

EVAL_EPISODES=200
REFERENCE="runs/stage4/ppo_red15_v2/final.pt"

ENV_BASE="--arena-size 130 --n-blue 5 --n-red 3 --n-obstacles 0 \
--belief-grid-size 26 --max-steps 200 --actor-obs tracker \
--use-staleness --staleness-regions 5 --comms-radius inf"

BC_ROUNDS=60
PPO_ROLLOUTS=150
PPO_LR=3e-05
RESET_LOG_STD=-1.2

# label:extra-flags
ARMS=("sum:" "attention:--attention --n-heads 4")

started=$(date '+%Y-%m-%d %H:%M:%S')
declare -A STATUS

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# A stage whose checkpoint exists is reused; a PPO stage is judged complete by
# its EVAL file, which is only written after training returns, because PPO
# writes best.pt early and a truncated run is otherwise indistinguishable from
# a finished one on disk.
have_ckpt() { [ -s "runs/stage4/$1/best.pt" ] || [ -s "runs/stage4/$1/final.pt" ]; }

score_of() {   # prints "caught +- se" for the evader row of an eval file
  local f="$1"
  [ -f "$f" ] || { echo "--"; return; }
  local n; n=$(grep -E "^\s*run\s" "$f" 2>/dev/null | awk '{print $2" "$3" "$4}')
  [ -z "$n" ] && echo "--" || echo "$n"
}

write_summary() {
  # `local` is mandatory: bash functions share the caller's scope, and an
  # earlier version of this pattern clobbered the arm loop's own variables,
  # which would have made arm 2 overwrite arm 1's outputs.
  local arm label stage key st
  {
    echo "# sum vs attention  (docs/stage4_results.md Sec. 18.4)"
    echo
    echo "Started $started, updated $(date '+%Y-%m-%d %H:%M:%S')."
    echo
    echo "Both arms: comms_radius inf from birth, clone ($BC_ROUNDS rounds) at"
    echo "red 1.4, then PPO at 1.4, then PPO at 1.5 -- $PPO_ROLLOUTS rollouts"
    echo "per stage at lr $PPO_LR.  The clone->PPO warm start resets log_std to"
    echo "$RESET_LOG_STD (the clone arrives at sigma 1.0, which Sec. 12 measured"
    echo "as destroying it); the 1.4->1.5 warm start does not, since by then"
    echo "log_std is trained."
    echo
    echo "**Pre-registered expectation: PARITY, not a win.**  Margin +-0.15 on"
    echo "the evader, measured over $EVAL_EPISODES episodes.  Parity is a PASS:"
    echo "it licenses attention for variable counts and heterogeneous speeds,"
    echo "which the summed aggregation blocks outright."
    echo
    echo "| what | stage | status | caught vs evader |"
    echo "|---|---|---|---|"
    echo "| *reference* \`ppo_red15_v2/final\` (curriculum, comms 40) | — | ${STATUS[ref]:-pending} | $(score_of "$OUT/reference_eval.txt") |"
    for arm in "${ARMS[@]}"; do
      label="${arm%%:*}"
      for stage in bc ppo14 ppo15; do
        key="$label/$stage"
        st="${STATUS[$key]:-pending}"
        echo "| $label | $stage | $st | $(score_of "$OUT/${label}_${stage}_eval.txt") |"
      done
    done
    echo
    echo "Logs \`$OUT/<arm>_<stage>.log\`; checkpoints \`runs/stage4/att_<arm>_<stage>/\`."
  } > "$SUMMARY"
}

write_summary
log "attention experiment starting; ${#ARMS[@]} arms, summary at $SUMMARY"

# ---- reference, re-measured at the same episode count ------------------ #
# The 2.72 +- 0.08 in Sec. 15.2 is 50 episodes.  Comparing it to a 200-episode
# arm would pair unequal precisions, so it is re-run here at 200 with the same
# seed_base, making every number in the table like for like.
if [ ! -f "$OUT/reference_eval.txt" ] && [ -f "$REFERENCE" ]; then
  log "reference: re-evaluating at $EVAL_EPISODES episodes"
  STATUS[ref]="running"; write_summary
  if $PY scripts/eval_checkpoint.py --ckpt "$REFERENCE" --red-v-max 1.5 \
        --episodes "$EVAL_EPISODES" --reds all \
        > "$OUT/reference_eval.txt" 2>&1; then
    STATUS[ref]="done"
  else
    STATUS[ref]="FAILED"
  fi
  write_summary
fi

for arm in "${ARMS[@]}"; do
  label="${arm%%:*}"
  flags="${arm#*:}"
  bc_name="att_${label}_bc"
  p14_name="att_${label}_ppo14"
  p15_name="att_${label}_ppo15"

  # ---- clone, red 1.4 ------------------------------------------------- #
  if have_ckpt "$bc_name"; then
    log "$label: clone present, reusing"
    STATUS["$label/bc"]="reused"
  else
    log "$label: clone (flags: ${flags:-none})"
    STATUS["$label/bc"]="running"; write_summary
    if $PY scripts/train_bc.py \
          --env-args "$ENV_BASE --red-v-max 1.4 $flags" \
          --n-rounds "$BC_ROUNDS" --seed 0 --device cpu \
          --run-name "$bc_name" > "$OUT/${label}_bc.log" 2>&1; then
      STATUS["$label/bc"]="done"
    else
      STATUS["$label/bc"]="FAILED"
      STATUS["$label/ppo14"]="skipped"; STATUS["$label/ppo15"]="skipped"
      write_summary; log "$label: clone FAILED; skipping the rest of this arm"
      continue
    fi
  fi
  $PY scripts/eval_checkpoint.py --ckpt "runs/stage4/$bc_name/best.pt" \
      --episodes 50 --reds run > "$OUT/${label}_bc_eval.txt" 2>&1 || true
  write_summary

  # ---- PPO at red 1.4, then at 1.5 ------------------------------------ #
  # stage : run name : red speed : warm start : reset flag
  prev="$bc_name"
  for stage in "ppo14:$p14_name:1.4:--reset-log-std $RESET_LOG_STD" \
               "ppo15:$p15_name:1.5:"; do
    tag="${stage%%:*}"; rest="${stage#*:}"
    name="${rest%%:*}"; rest="${rest#*:}"
    speed="${rest%%:*}"; reset="${rest#*:}"

    if have_ckpt "$name" && [ -f "$OUT/${label}_${tag}_eval.txt" ]; then
      log "$label: $tag already complete, reusing"
      STATUS["$label/$tag"]="reused"; prev="$name"; write_summary; continue
    fi
    rm -rf "runs/stage4/$name"
    log "$label: $tag (red $speed, from $prev)"
    STATUS["$label/$tag"]="running"; write_summary
    if $PY scripts/train_stage4.py \
          $ENV_BASE --red-v-max "$speed" --sensor-radius 40.0 $flags \
          --n-envs 64 --rollout-steps 200 --n-rollouts "$PPO_ROLLOUTS" \
          --n-epochs 10 --mb-size 512 --lr "$PPO_LR" --ent-coef 0.008 \
          --target-kl 0.03 --n-workers 4 --torch-threads 4 \
          --eval-interval 5 --eval-episodes 25 \
          --best-ckpt-metric det_caught \
          --red-policy-mix stationary:1,random:1,run:1 \
          --warm-start-full "runs/stage4/$prev/best.pt" $reset \
          --share-hidden-via-gnn --seed 0 --device cpu \
          --run-name "$name" > "$OUT/${label}_${tag}.log" 2>&1; then
      STATUS["$label/$tag"]="done"
      # The 1.5 stage is the one the verdict rests on, so it gets the full
      # episode count and both checkpoints; 1.4 is a curriculum waypoint.
      if [ "$tag" = "ppo15" ]; then
        $PY scripts/eval_checkpoint.py --ckpt "runs/stage4/$name/best.pt" \
            --episodes "$EVAL_EPISODES" --reds all \
            > "$OUT/${label}_ppo15_best_eval.txt" 2>&1 || true
        $PY scripts/eval_checkpoint.py --ckpt "runs/stage4/$name/final.pt" \
            --episodes "$EVAL_EPISODES" --reds all \
            > "$OUT/${label}_${tag}_eval.txt" 2>&1 || true
      else
        $PY scripts/eval_checkpoint.py --ckpt "runs/stage4/$name/final.pt" \
            --episodes 50 --reds run \
            > "$OUT/${label}_${tag}_eval.txt" 2>&1 || true
      fi
      prev="$name"
    else
      STATUS["$label/$tag"]="FAILED"
      write_summary; log "$label: $tag FAILED; skipping the rest of this arm"
      break
    fi
    write_summary
  done
done

write_summary
log "all arms finished; summary at $SUMMARY"
