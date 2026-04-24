#!/usr/bin/env bash
# Fill expanded-suite eval gaps for trunk-baseline-v1 and hploss-aux-v1.
set -u
LOG=C:/coding-projects/STS2/sts2-solver/fill_eval_gaps.log
TB=C:/coding-projects/sts2-trunk-baseline-v1
HP=C:/coding-projects/sts2-hploss-aux-v1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_eval() {
  local worktree=$1 name=$2 gen=$3
  log "=== $name gen $gen ==="
  cd "$worktree/sts2-solver" && \
    "$worktree/sts2-solver/.venv/Scripts/python.exe" -m sts2_solver.betaone.experiment_cli eval "$name" --checkpoint "gen$gen" >> "$LOG" 2>&1
}

for gen in 10 20 40; do
  run_eval "$TB" trunk-baseline-v1 "$gen"
done

for gen in 10 20 40 50 60; do
  run_eval "$HP" hploss-aux-v1 "$gen"
done

log "DONE"
