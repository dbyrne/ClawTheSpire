#!/usr/bin/env bash
# Wait for hploss-ucb-v1 to reach gen 60, then benchmark on 3 encounter sets.
set -u
LOG=C:/coding-projects/STS2/sts2-solver/ucb_benchmarks.log
PY=C:/Users/david/AppData/Local/Programs/Python/Python311/python.exe
UCB_PY=C:/coding-projects/sts2-hploss-ucb-v1/sts2-solver/.venv/Scripts/python.exe

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

progress_gen() {
  local pjson=$1
  "$PY" -c "
import json
try:
  with open(r'''$pjson''') as f: d = json.load(f)
  print(d.get('gen', 0))
except Exception:
  print(0)
"
}

UCB_PROG=C:/coding-projects/sts2-hploss-ucb-v1/sts2-solver/experiments/hploss-ucb-v1/betaone_progress.json
TARGET=60

log "monitor started. target gen=$TARGET"
while true; do
  G=$(progress_gen "$UCB_PROG")
  log "status: hploss-ucb-v1=$G"
  if [ "$G" -ge "$TARGET" ]; then
    log "reached gen $TARGET"
    break
  fi
  sleep 300
done

for ES in lean-decks-v1 draw-synergy-v1 base-es-v1; do
  log "=== hploss-ucb-v1 on $ES (gen 60) ==="
  cd C:/coding-projects/sts2-hploss-ucb-v1/sts2-solver && \
    "$UCB_PY" -m sts2_solver.betaone.experiment_cli benchmark hploss-ucb-v1 \
      --encounter-set "$ES" --sims 1000 --repeats 10 >> "$LOG" 2>&1
done

log "ALL BENCHMARKS COMPLETE"
