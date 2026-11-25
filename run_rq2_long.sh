#!/usr/bin/env bash
# RQ2, 500 iterations: none vs interleave (c5 d15 mean), 3 seeds, with trading on and off,
# then the trading on/off comparison.
#
#   caffeinate -i -s ./run_rq2_long.sh > results/rq2_long.log 2>&1 &
#
# Re-running is safe: run_rq2.sh skips finished runs, evaluations and profiles.
#   results/rq2_long_trading/   trading on   (run_rq2.sh output: runs, eval, summary, profiles, figures)
#   results/rq2_long_notrade/   trading off
#   results/rq2_long_compare/   compare_trading.py output (figures + summary.md)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
export ITERS=500 ONLY="none interleave" JOBS=${JOBS:-6} THREADS=${THREADS:-2} PY

echo "$(date '+%F %T') == set 1: trading on"
OUT=results/rq2_long_trading EXTRA="--trading" ./run_rq2.sh > >(tee -a results/rq2_long_trading.log) 2>&1

echo "$(date '+%F %T') == set 2: trading off"
OUT=results/rq2_long_notrade EXTRA="" ./run_rq2.sh > >(tee -a results/rq2_long_notrade.log) 2>&1

echo "$(date '+%F %T') == comparison"
$PY compare_trading.py --off results/rq2_long_notrade --on results/rq2_long_trading --out results/rq2_long_compare
echo "$(date '+%F %T') == all done"
