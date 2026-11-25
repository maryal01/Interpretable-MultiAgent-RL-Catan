#!/usr/bin/env bash
# RQ2 experiment grid: exchange conditions x SEEDS, JOBS runs at a time, then
# evaluation, behaviour profiles and figures.
#
#   ./run_rq2.sh                     # conditions enabled below x seeds 0 1 2
#   SEEDS="0" ./run_rq2.sh           # one seed per condition
#   JOBS=9 THREADS=1 ./run_rq2.sh    # more runs at once, fewer threads each
#   OUT=results/rq2_trading EXTRA="--trading" ONLY="none interleave" ./run_rq2.sh
#                                    # same grid with player trading on, two conditions only
#
# Everything goes under results/rq2/ (see results/rq2/README.md):
#   runs/<condition>_s<seed>/     checkpoints, log.jsonl, args.json
#   logs/<condition>_s<seed>.log  training stdout/stderr
#   eval.csv, curves.csv, summary.txt
#   profiles/<condition>_s<seed>.{json,txt}   behaviour profile of the final checkpoint
#   figures/*.png                 plot_rq2.py output
# Every step skips what is already done, so it is safe to re-run (e.g. after
# enabling more conditions).
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-../.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-6}          # parallel training runs (10 cores: 6 x 2 threads)
THREADS=${THREADS:-2}    # torch threads per run
ITERS=${ITERS:-200}
PROFILE_GAMES=${PROFILE_GAMES:-100}
OUT=${OUT:-results/rq2}
EXTRA=${EXTRA:-""}       # extra train_rq2.py arguments for every run, e.g. "--trading"
ONLY=${ONLY:-""}         # space-separated condition names to run (default: every enabled one)
mkdir -p "$OUT/runs" "$OUT/logs" "$OUT/profiles"
# Catanatron's move order depends on Python's string hashing; fix it so runs are reproducible.
export PYTHONHASHSEED=0

# name | train_rq2.py arguments. Every condition uses the same frames, iterations and
# PPO settings; only the exchange differs. Comment a line out to drop it from the grid.
CONDITIONS=(
  # Baseline: four independent policies, never exchange anything.
  "none|--exchange none"

  # Alternate centralised / decentralised policy training: 15 independent iterations,
  # then 5 in which every agent applies the same averaged update (pooled learning,
  # differences between agents preserved). Cycle = 20 iterations, 25% centralised.
  "interleave|--exchange interleave --central-iters 5 --decentral-iters 15 --central-weighting mean"

  # One network for all agents: maximum sharing, zero diversity (reference point).
  "shared|--exchange shared"

  # Every 10 iterations, move each agent's weights 30% toward the best agent's.
  "blend|--exchange blend --every 10 --alpha 0.3"

  # Every 10 iterations, train each agent for 2 epochs to imitate the best agent on its own states.
  "distill|--exchange distill --every 10 --beta 1.0 --distill-epochs 2"

  # interleave, but in central phases agents that won more in the previous phase count more.
  "interleave_wins|--exchange interleave --central-iters 5 --decentral-iters 15 --central-weighting wins"

  # Other C:D ratios, to trace the speed-vs-diversity trade-off.
  "interleave_c2|--exchange interleave --central-iters 2 --decentral-iters 18"
  "interleave_c10|--exchange interleave --central-iters 10 --decentral-iters 10"
)

# ---- 1. training
last=$(printf "iter_%04d.pt" $((ITERS - 1)))
jobs_file=$(mktemp)
for seed in $SEEDS; do
  for c in "${CONDITIONS[@]}"; do
    cond="${c%%|*}"; name="${cond}_s${seed}"; extra="${c#*|} $EXTRA"
    if [[ -n "$ONLY" && " $ONLY " != *" $cond "* ]]; then continue; fi
    if [[ -f "$OUT/runs/$name/$last" ]]; then echo "skip $name (done)"; continue; fi
    rm -rf "$OUT/runs/$name"   # an unfinished run restarts from scratch
    echo "$PY train_rq2.py $extra --seed $seed --iters $ITERS --threads $THREADS --out $OUT/runs/$name > $OUT/logs/$name.log 2>&1" >> "$jobs_file"
  done
done
echo "$(date '+%F %T') starting $(wc -l < "$jobs_file" | tr -d ' ') training runs, $JOBS at a time"
tr '\n' '\0' < "$jobs_file" | xargs -0 -n1 -P "$JOBS" sh -c
rm -f "$jobs_file"
echo "$(date '+%F %T') training done"

# ---- 2. win rate vs random bots per checkpoint
$PY rq2_eval.py eval "$OUT"/runs/* --every 20 --games 40 --workers 4 --csv "$OUT/eval.csv"
$PY rq2_eval.py summary --csv "$OUT/eval.csv" --runs-dir "$OUT/runs" --out "$OUT/curves.csv" | tee "$OUT/summary.txt"
echo "$(date '+%F %T') evaluation done"

# ---- 3. behaviour profile of every final checkpoint (self-play, all 4 agents)
jobs_file=$(mktemp)
for d in "$OUT"/runs/*_s*; do
  name=$(basename "$d")
  [[ -f "$d/$last" && ! -f "$OUT/profiles/$name.json" ]] || continue
  echo "$PY profile_policies.py $d/$last --games $PROFILE_GAMES --json $OUT/profiles/$name.json > $OUT/profiles/$name.txt 2>&1" >> "$jobs_file"
done
echo "$(date '+%F %T') profiling $(wc -l < "$jobs_file" | tr -d ' ') final checkpoints"
tr '\n' '\0' < "$jobs_file" | xargs -0 -n1 -P 4 sh -c
rm -f "$jobs_file"

# ---- 4. figures
$PY plot_rq2.py --dir "$OUT"
echo "$(date '+%F %T') all done"
