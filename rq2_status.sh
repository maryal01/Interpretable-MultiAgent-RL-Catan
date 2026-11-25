#!/usr/bin/env bash
# Status of the RQ2 experiment:   ./rq2_status.sh        (or: watch -n 60 ./rq2_status.sh)
cd "$(dirname "$0")"
OUT=${OUT:-results/rq2}
ITERS=${ITERS:-200}

echo "== $(date '+%F %T')"
for f in "$OUT"/run_rq2.log "$OUT"/run_rq2_batch2.log; do
  [[ -f $f ]] && { echo "-- $(basename "$f") (last steps)"; grep -av "^  \[\|frozen\|^skip" "$f" | grep -a "^20" | tail -4; }
done

echo "-- training (iterations finished / $ITERS)"
done_runs=0; total=0
for d in "$OUT"/runs/*_s*; do
  [[ -d $d ]] || continue
  total=$((total + 1))
  n=$( [[ -f $d/log.jsonl ]] && wc -l < "$d/log.jsonl" | tr -d ' ' || echo 0)
  [[ $n -ge $ITERS ]] && done_runs=$((done_runs + 1))
  err=$(grep -c Traceback "$OUT/logs/$(basename "$d").log" 2>/dev/null)
  printf "  %-22s %4s/%s %s\n" "$(basename "$d")" "$n" "$ITERS" "$([[ ${err:-0} -gt 0 ]] && echo '  ERROR - see logs/')"
done
echo "  $done_runs of $total started runs finished"

[[ -f $OUT/eval.csv ]] && echo "-- evaluated checkpoints: $(( ($(wc -l < "$OUT/eval.csv") - 1) / 4 ))"
echo "-- profiles: $(ls "$OUT"/profiles/*.json 2>/dev/null | wc -l | tr -d ' ')   figures: $(ls "$OUT"/figures/*.png 2>/dev/null | wc -l | tr -d ' ')"

echo "-- processes: $(ps -axo command | grep -c '^[^ ]*python[^ ]* train_rq2.py') training, $(ps -axo command | grep -cE '^[^ ]*python[^ ]* (rq2_eval|profile_policies|plot_rq2).py') eval/profile/plot"
pgrep -q caffeinate && echo "-- caffeinate: on (idle sleep prevented)" || echo "-- caffeinate: off"
pmset -g batt | head -1 | grep -q Battery && echo "-- WARNING: on battery; plug in and keep the lid open or the Mac may sleep"
