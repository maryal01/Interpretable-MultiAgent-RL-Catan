#!/usr/bin/env bash
# RQ2, 500 iterations: add shared, blend and distill (3 seeds) to the run_rq2_long.sh folders,
# trading on and off, then compare all five strategies (none + 4 exchange strategies).
#
#   caffeinate -i -s ./run_rq2_long_more.sh > results/rq2_long_more.log 2>&1 &
#   WAIT_PID=<pid> ...   waits for that process (e.g. run_rq2_long.sh) to exit first
#
# 9 runs per setting go in one wave (JOBS=9, 1 thread each on 10 cores). Re-running is safe:
# finished runs, evaluations and profiles are skipped.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
if [[ -n "${WAIT_PID:-}" ]]; then
  echo "$(date '+%F %T') waiting for pid $WAIT_PID to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
export ITERS=500 ONLY="shared blend distill" JOBS=${JOBS:-9} THREADS=${THREADS:-1} PY

echo "$(date '+%F %T') == trading on: shared, blend, distill"
OUT=results/rq2_long_trading EXTRA="--trading" ./run_rq2.sh > >(tee -a results/rq2_long_trading.log) 2>&1

echo "$(date '+%F %T') == trading off: shared, blend, distill"
OUT=results/rq2_long_notrade EXTRA="" ./run_rq2.sh > >(tee -a results/rq2_long_notrade.log) 2>&1

echo "$(date '+%F %T') == comparison (all five strategies)"
$PY compare_trading.py --off results/rq2_long_notrade --on results/rq2_long_trading \
    --strategies none shared blend distill interleave --out results/rq2_long_compare
echo "$(date '+%F %T') == all done"
