#!/usr/bin/env bash
# Obstacles, finally.  Not one result from Sec. 7 to Sec. 21 ran with them:
# n_obstacles=0 and crash_obstacle_penalty=0.0 in every checkpoint, so the
# obstacle nodes, ob edges, obstacle tracker, crash penalties and occlusion
# are all built with zero results.  This is the largest unmeasured surface in
# the project, and it is a precondition for two other things: the ob channel
# is what gives the channel-gating experiment a third real channel, and cover
# is what lets a slow red survive in self-play.
#
# ONE STAGE at fixed CAPACITY 9 with the active count sampled 1-9 per episode,
# warm-started from cnt_attention_ppo.  A first attempt ramped the CAPACITY
# 4 -> 9 and failed; this is the redesign, and both changes come from that
# failure:
#
#   * Capacity is the TENSOR WIDTH; the active count is the difficulty.
#     Ramping the capacity re-initialises critic_trunk.0.weight (129 -> 194
#     the moment obstacles exist), so the first attempt began PPO with a warm
#     actor and a COLD CRITIC -- which Secs. 6.5 and 12 established gets
#     destroyed by large wrong advantages.  It started at 1.37 against the
#     blind heuristic's 1.73 and never caught up.  Sampling the active count
#     at a fixed capacity never touches the critic's width.
#   * Sampling also closes the 5-8 gap a two-point ramp leaves, the same
#     lesson Sec. 21 measured for red counts: a policy trained at a fixed
#     count has no reason to handle variation.
#
# Every coefficient below comes from a measurement, not a guess:
#
#   * Baseline crash rate with NO incentive, counted as the trainer counts it
#     (rising edges, so a sustained overlap is one event): 7.9 events/episode
#     at 4 obstacles, 13.6 at 9 -- with blues inside an obstacle for 17.4% and
#     30.5% of agent-steps respectively.  The historical clearance_fixed_v1
#     reached 1.20 at 4 obstacles, so ~6.6x better than blind, not zero.
#
#   * The coefficients are an ORDER OF MAGNITUDE below the first attempt, and
#     that attempt is why.  Decomposing its -83 return: catches +16, uncaught
#     -7, step cost -6, so crash + clearance was about -86 -- FIVE TIMES the
#     catch reward.  The earlier budget was computed as crash x exposure and
#     ignored that the clearance term covers a far larger band than the crash
#     one (23.5% of the arena against 17.4% exposure) AND is summed over
#     obstacles, so a blue between two feels both.  Hence crash 2.0 -> 0.3 and
#     clearance 0.6 -> 0.2, for about -11 against roughly +30 from catches.
#
#     The deeper point: a dominant penalty is self-extinguishing only if there
#     is budget to recover.  clearance_fixed_v1 absorbed 2.0 across 61.4M
#     steps; with 1.9M it just dominates.  Shaping magnitude should scale
#     INVERSELY with budget.
#
#   * Obstacle radii drop 5-15 to 4-10, halving their physical area (17.4% ->
#     8.9%) and lifting the admissible area at the HARDEST count from 46.4% to
#     59.7% while KEEPING the 8 m margin -- better than the 54.0% that cutting
#     the margin to 6 bought with the big radii, and it keeps the wide
#     gradient.  With the count sampled, 9 obstacles is only ~11% of episodes.
#
#   * (superseded) the per-step crash reasoning that set 2.0 -> 1.0:
#     cost 2.0 x 0.305 x 200 = -122 per episode against roughly +30 from
#     catches: four times the reward, which risks a policy that learns to
#     dodge and stops hunting.  So the curriculum holds the PENALTY BUDGET
#     roughly constant rather than the coefficient -- 2.0 at 4 obstacles
#     (-70) and 1.0 at 9 (-61).  Otherwise going 4 -> 9 silently doubles the
#     shaping pressure at the moment the task gets harder.
#
#   * The obstacle clearance margin drops 8 -> 6 at 9 obstacles, because the
#     admissible area (outside every penalised band) is 70.6% at 4/margin-8
#     but only 46.7% at 9/margin-8, against 54.3% at 9/margin-6.  A policy
#     with nowhere unpenalised to stand freezes, which Sec. 10 measured as a
#     real failure mode at 58.4% idle agent-steps.
#
#   * The ally clearance is NOT optional: with obstacle clearance alone, blues
#     avoiding obstacles crowd into the same clear space and crashes just
#     migrate obstacle -> ally (observed on clearance_fixed_v1).
#
#   * Wall clearance is new, 0.5 at a 2 m margin.  Walls were free while
#     obstacles and allies were shaped, and the trained policy touched them
#     0.53% of agent-steps against the heuristic's 0.08%.  Not an exploit --
#     mean wall distance 25.7 m against the 21.7 m a uniform point gives --
#     but ~5 agent-steps of contact per episode, which is noise for score and
#     a failure rate for an indoor certification trial.
#
# REGIONS ARE HIBERNATED (no --use-staleness).  Their directional signal
# measures ~zero against mobile evaders, and gb is 125 of 230 edges, so
# dropping it roughly DOUBLES throughput (105 against 46-56 sps measured) --
# which is the budget this harder task needs.  That is a second change
# alongside obstacles, so the forgetting check below compares against
# cnt_attention_ppo WITH ITS REGIONS ZEROED (4.78 +- 0.06, already measured),
# not against its intact score, so both sides are region-less.
#
# Run:
#     nohup bash scripts/run_obstacle_curriculum.sh > runs/obstacle_curriculum.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

OUT="runs/obstacle_curriculum"
SUMMARY="$OUT/SUMMARY.md"
mkdir -p "$OUT"

WARM_FROM="runs/stage4/cnt_attention_ppo/best.pt"
EVAL_EPISODES=100
PPO_ROLLOUTS=150
PPO_LR=3e-05

# Shared across both stages.  No --use-staleness: regions hibernated.
ENV_BASE="--arena-size 130 --n-blue 5 --n-red 5 --n-red-min 1 \
--belief-grid-size 26 --max-steps 200 --actor-obs tracker \
--comms-radius inf --red-v-max 1.4 --sensor-radius 40.0 \
--n-obstacles-min 1 --obstacle-radius-min 4.0 --obstacle-radius-max 10.0 \
--attention --n-heads 4 \
--clearance-ally-weight 0.6 --clearance-ally-margin 3.0 \
--wall-clearance-weight 0.5 --wall-clearance-margin 2.0"

# stage : n_obstacles : crash penalty : obstacle clearance margin
STAGES=("obs1to9:9:0.3:8.0")

started=$(date '+%Y-%m-%d %H:%M:%S')
declare -A STATUS

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
have_ckpt() { [ -s "runs/stage4/$1/best.pt" ] || [ -s "runs/stage4/$1/final.pt" ]; }

write_summary() {
  # `local` is mandatory: bash functions share the caller's scope, and an
  # earlier version of this pattern clobbered the loop's own variables.
  local st tag f
  {
    echo "# Obstacle curriculum, 4 -> 9"
    echo
    echo "Started $started, updated $(date '+%Y-%m-%d %H:%M:%S')."
    echo
    echo "Warm-started from \`$WARM_FROM\`.  Regions HIBERNATED (~2x faster;"
    echo "their directional signal measures ~zero against mobile evaders)."
    echo "Attention on, so the ob channel is normalised like bb and rb -- which"
    echo "is what makes 9 obstacles viable: under a sum ob would be 45 of 105"
    echo "edges and would swamp the target channel."
    echo
    echo "Penalty BUDGET held roughly constant across stages rather than the"
    echo "coefficient: 2.0 at 4 obstacles (-70/episode at the measured blind"
    echo "crash rate) and 1.0 at 9 (-61).  Obstacle clearance margin 8 -> 6 so"
    echo "the admissible area stays above ~55% (54.3% at 9/6; 46.7% at 9/8)."
    echo
    echo "Blind baselines, crashes as EVENTS (rising edges, as the trainer"
    echo "counts them):"
    echo
    echo "| n_obs | agent | caught | crash events/ep | % steps inside |"
    echo "|---|---|---|---|---|"
    echo "| 4 | ObsGreedy | 2.67 ± 0.20 | 7.9 ± 0.9 | 17.4% |"
    echo "| 4 | AssignGreedy | 3.08 ± 0.18 | 5.7 ± 0.7 | 16.4% |"
    echo "| 9 | ObsGreedy | 2.35 ± 0.17 | 13.6 ± 1.1 | 30.5% |"
    echo "| 9 | AssignGreedy | 2.90 ± 0.20 | 12.4 ± 1.0 | 27.8% |"
    echo
    echo "| stage | n_obs | status |"
    echo "|---|---|---|"
    for st in "${STAGES[@]}"; do
      tag="${st%%:*}"
      echo "| $tag | $(echo "$st" | cut -d: -f2) | ${STATUS[$tag]:-pending} |"
    done
    echo
    for st in "${STAGES[@]}"; do
      tag="${st%%:*}"
      for f in "$OUT/${tag}_eval.txt" "$OUT/${tag}_noobs_eval.txt"; do
        if [ -f "$f" ]; then
          echo "### $(basename "$f" .txt)"
          echo '```'
          cat "$f"
          echo '```'
          echo
        fi
      done
    done
    echo "FORGETTING CHECK: \`*_noobs_eval\` is the obstacle-trained policy"
    echo "evaluated back at n_obstacles=0.  Compare against"
    echo "cnt_attention_ppo with its regions ZEROED, 4.78 ± 0.06 -- not its"
    echo "intact 4.73, and not a region-ful number, so both sides are"
    echo "region-less.  A drop there is evidence that decoupling the shared"
    echo "MLPs would be worth its +69% parameters; no drop means it never was."
    echo
    echo "Logs \`$OUT/<stage>.log\`; checkpoints \`runs/stage4/obs_<stage>/\`."
  } > "$SUMMARY"
}

write_summary
log "obstacle curriculum starting; ${#STAGES[@]} stages, summary at $SUMMARY"

prev="$WARM_FROM"
for st in "${STAGES[@]}"; do
  tag="${st%%:*}"
  n_obs=$(echo "$st" | cut -d: -f2)
  crash=$(echo "$st" | cut -d: -f3)
  cmargin=$(echo "$st" | cut -d: -f4)
  name="obs_${tag}"

  if have_ckpt "$name" && [ -f "$OUT/${tag}_eval.txt" ]; then
    log "$tag: already complete, reusing"
    STATUS[$tag]="reused"; prev="runs/stage4/$name/best.pt"
    write_summary; continue
  fi
  rm -rf "runs/stage4/$name"
  log "$tag: n_obstacles=$n_obs crash=$crash clearance_margin=$cmargin from $prev"
  STATUS[$tag]="running"; write_summary

  # --reset-log-std is NOT passed: prev is a trained PPO policy whose log_std
  # is meaningful, unlike a clone's (Sec. 12).
  if $PY scripts/train_stage4.py \
        $ENV_BASE \
        --n-obstacles "$n_obs" \
        --crash-obstacle-penalty "$crash" \
        --crash-blue-penalty "$crash" \
        --clearance-weight 0.2 --clearance-margin "$cmargin" \
        --n-envs 64 --rollout-steps 200 --n-rollouts "$PPO_ROLLOUTS" \
        --n-epochs 10 --mb-size 512 --lr "$PPO_LR" --ent-coef 0.008 \
        --target-kl 0.03 --n-workers 4 --torch-threads 4 \
        --eval-interval 5 --eval-episodes 25 \
        --best-ckpt-metric det_composite \
        --red-policy-mix stationary:1,random:1,run:1 \
        --warm-start-full "$prev" \
        --share-hidden-via-gnn --seed 0 --device cpu \
        --run-name "$name" > "$OUT/${tag}.log" 2>&1; then
    STATUS[$tag]="done"
    $PY scripts/eval_checkpoint.py --ckpt "runs/stage4/$name/final.pt" \
        --episodes "$EVAL_EPISODES" --reds all \
        > "$OUT/${tag}_eval.txt" 2>&1 || true
    # Forgetting check: same policy, obstacles removed.
    $PY scripts/eval_red_count_sweep.py --ckpt "runs/stage4/$name/final.pt" \
        --counts 5 --episodes "$EVAL_EPISODES" \
        > "$OUT/${tag}_noobs_eval.txt" 2>&1 || true
    prev="runs/stage4/$name/best.pt"
  else
    STATUS[$tag]="FAILED"
    log "$tag: FAILED (see $OUT/${tag}.log); stopping the curriculum"
    write_summary
    break
  fi
  write_summary
done

write_summary
log "curriculum finished; summary at $SUMMARY"
