#!/usr/bin/env bash
# Overnight monitor: wait for hploss-aux-v1 + trunk-baseline-v1 to reach gen 60,
# then run benchmarks on both across 3 encounter sets (es-lean, es-draw, es-base).

set -u
LOG=C:/coding-projects/STS2/sts2-solver/overnight_benchmarks.log
PY=C:/Users/david/AppData/Local/Programs/Python/Python311/python.exe

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

progress_gen() {
  local pjson=$1
  "$PY" -c "
import json, sys
try:
  with open(r'''$pjson''') as f: d = json.load(f)
  print(d.get('gen', 0))
except Exception:
  print(0)
"
}

HP_PROG=C:/coding-projects/sts2-hploss-aux-v1/sts2-solver/experiments/hploss-aux-v1/betaone_progress.json
TB_PROG=C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver/experiments/trunk-baseline-v1/betaone_progress.json
TARGET_GEN=60

log "monitor started. target gen=$TARGET_GEN"

while true; do
  HP_GEN=$(progress_gen "$HP_PROG")
  TB_GEN=$(progress_gen "$TB_PROG")
  log "status: hploss=$HP_GEN tb=$TB_GEN"
  if [ "$HP_GEN" -ge "$TARGET_GEN" ] && [ "$TB_GEN" -ge "$TARGET_GEN" ]; then
    log "both experiments reached gen $TARGET_GEN"
    break
  fi
  sleep 300
done

HP_PY=C:/coding-projects/sts2-hploss-aux-v1/sts2-solver/.venv/Scripts/python.exe
TB_PY=C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver/.venv/Scripts/python.exe

ENCOUNTER_SETS=(lean-decks-v1 draw-synergy-v1 base-es-v1)

for ES in "${ENCOUNTER_SETS[@]}"; do
  log "=== benchmark trunk-baseline-v1 on $ES ==="
  cd C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver && \
    "$TB_PY" -m sts2_solver.betaone.experiment_cli benchmark trunk-baseline-v1 \
      --encounter-set "$ES" --sims 1000 --repeats 10 >> "$LOG" 2>&1
  log "=== benchmark hploss-aux-v1 on $ES ==="
  cd C:/coding-projects/sts2-hploss-aux-v1/sts2-solver && \
    "$HP_PY" -m sts2_solver.betaone.experiment_cli benchmark hploss-aux-v1 \
      --encounter-set "$ES" --sims 1000 --repeats 10 >> "$LOG" 2>&1
done

log "ALL BENCHMARKS COMPLETE"
