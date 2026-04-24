#!/usr/bin/env bash
set -u
LOG=C:/coding-projects/STS2/sts2-solver/tb_gen50_vs_60_eval.log
PY=C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver/.venv/Scripts/python.exe

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== trunk-baseline-v1 gen 50 on expanded 109-scenario P-Eval ==="
cd C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver && \
  "$PY" -m sts2_solver.betaone.experiment_cli eval trunk-baseline-v1 --checkpoint gen50 >> "$LOG" 2>&1

log "=== trunk-baseline-v1 gen 60 on expanded 109-scenario P-Eval ==="
cd C:/coding-projects/sts2-trunk-baseline-v1/sts2-solver && \
  "$PY" -m sts2_solver.betaone.experiment_cli eval trunk-baseline-v1 --checkpoint gen60 >> "$LOG" 2>&1

log "DONE"
