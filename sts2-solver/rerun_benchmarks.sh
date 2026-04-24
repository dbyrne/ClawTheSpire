#!/usr/bin/env bash
# Re-run benchmarks on concluded gen (50 for both). --checkpoint auto resolves
# to the finalized gen. 3 encounter sets x 2 experiments = 6 runs.

set -u
LOG=C:/coding-projects/STS2/sts2-solver/rerun_benchmarks.log
HP_PY=C:/coding-projects/sts2-hploss-aux-v1/sts2-solver/.venv/Scripts/python.exe
TB_PY=C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver/.venv/Scripts/python.exe

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "starting benchmark re-run on finalized gen 50"

for ES in lean-decks-v1 draw-synergy-v1 base-es-v1; do
  log "=== trunk-baseline-v1 on $ES (gen 50) ==="
  cd C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver && \
    "$TB_PY" -m sts2_solver.betaone.experiment_cli benchmark trunk-baseline-v1 \
      --encounter-set "$ES" --sims 1000 --repeats 10 >> "$LOG" 2>&1
  log "=== hploss-aux-v1 on $ES (gen 50) ==="
  cd C:/coding-projects/sts2-hploss-aux-v1/sts2-solver && \
    "$HP_PY" -m sts2_solver.betaone.experiment_cli benchmark hploss-aux-v1 \
      --encounter-set "$ES" --sims 1000 --repeats 10 >> "$LOG" 2>&1
done

log "ALL BENCHMARKS COMPLETE"
