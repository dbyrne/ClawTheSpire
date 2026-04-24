#!/usr/bin/env bash
# Final benchmark sweep for tb-v2 before finalize.
# gen 60 = baseline (already have one run, this is a same-day re-benchmark for
#          within-engine determinism check).
# gen 80 = best-balanced we have available (P=105 V=108, closest to both eval
#          peaks that got lost to the save_every bug).
# gen 90 = end state (both metrics decayed back to ~baseline).
set -u
LOG=C:/coding-projects/STS2/sts2-solver/tb_v2_final_sweep.log
PY=C:/coding-projects/sts2-trunk-baseline-v2/sts2-solver/.venv/Scripts/python.exe

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

for GEN in 60 80 90; do
  log "=== gen $GEN on lean-decks-v1 ==="
  cd C:/coding-projects/sts2-trunk-baseline-v2/sts2-solver && \
    "$PY" -m sts2_solver.betaone.experiment_cli benchmark trunk-baseline-v2 \
      --checkpoint "gen$GEN" --encounter-set lean-decks-v1 --sims 1000 --repeats 10 >> "$LOG" 2>&1
done

log "DONE"
