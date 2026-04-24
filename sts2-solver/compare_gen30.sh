#!/usr/bin/env bash
# 3-way gen-30 comparison on the expanded 125-scenario P-Eval suite.
# Fair training-stage comparison: all three experiments were cold-started
# on the same post-sim-fix code, same lean-decks-v1 encounter set.

set -u
LOG=C:/coding-projects/STS2/sts2-solver/compare_gen30.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

TB=C:/coding-projects/sts2-trunk-baseline-v1
HP=C:/coding-projects/sts2-hploss-aux-v1
UCB=C:/coding-projects/sts2-hploss-ucb-v1

log "=== trunk-baseline-v1 gen 30 ==="
cd "$TB" && git checkout main -- sts2-solver/src/sts2_solver/betaone/eval.py
cd "$TB/sts2-solver" && \
  "$TB/sts2-solver/.venv/Scripts/python.exe" -m sts2_solver.betaone.experiment_cli eval trunk-baseline-v1 --checkpoint gen30 >> "$LOG" 2>&1

log "=== hploss-aux-v1 gen 30 ==="
cd "$HP" && git checkout main -- sts2-solver/src/sts2_solver/betaone/eval.py
cd "$HP/sts2-solver" && \
  "$HP/sts2-solver/.venv/Scripts/python.exe" -m sts2_solver.betaone.experiment_cli eval hploss-aux-v1 --checkpoint gen30 >> "$LOG" 2>&1

log "=== hploss-ucb-v1 gen 30 ==="
cd "$UCB/sts2-solver" && \
  "$UCB/sts2-solver/.venv/Scripts/python.exe" -m sts2_solver.betaone.experiment_cli eval hploss-ucb-v1 --checkpoint gen30 >> "$LOG" 2>&1

log "DONE"
