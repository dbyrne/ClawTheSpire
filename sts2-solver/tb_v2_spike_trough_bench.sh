#!/usr/bin/env bash
# Benchmark tb-v2 gens 60 (baseline), 61 (V-Eval peak), 70 (V-Eval trough) on
# es-lean. Tests whether V-Eval oscillation corresponds to narrow-set WR
# oscillation, with gen 60 as the anchor (the fork start-point).
set -u
LOG=C:/coding-projects/STS2/sts2-solver/tb_v2_spike_trough_bench.log
PY=C:/coding-projects/sts2-trunk-baseline-v2/sts2-solver/.venv/Scripts/python.exe

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

for GEN in 60 61 70; do
  log "=== gen $GEN on lean-decks-v1 ==="
  cd C:/coding-projects/sts2-trunk-baseline-v2/sts2-solver && \
    "$PY" -m sts2_solver.betaone.experiment_cli benchmark trunk-baseline-v2 \
      --checkpoint "gen$GEN" --encounter-set lean-decks-v1 --sims 1000 --repeats 10 >> "$LOG" 2>&1
done

log "DONE"
