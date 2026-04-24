"""Polling sync: reads distill_history.jsonl → writes betaone_progress.json.

Keeps in-flight distillation experiments' progress.json current so the TUI
shows live epoch progress. Run as a long-lived background process.

Usage:
    python -m scripts.sync_distill_progress  # polls all experiments/distill-* every 30s
"""
from __future__ import annotations

import json
import os
import sys
import time
import yaml
from pathlib import Path


POLL_SECONDS = 30


def _sync_one(exp_dir: Path):
    history_path = exp_dir / "distill_history.jsonl"
    if not history_path.exists():
        return False
    prog_path = exp_dir / "betaone_progress.json"
    cfg_path = exp_dir / "config.yaml"

    # Read last epoch from history
    try:
        with open(history_path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            return False
        last = json.loads(lines[-1])
    except (json.JSONDecodeError, OSError):
        return False

    epoch = last.get("epoch", 0)
    total_epochs = epoch  # fallback
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            total_epochs = cfg.get("training", {}).get("epochs", epoch)
        except Exception:
            pass

    # Detect if training still running: history modified in last 5 min
    age = time.time() - history_path.stat().st_mtime
    is_live = age < 300
    phase = "TRAINING" if is_live else "DONE"

    # Reuse 'gen' as 'epoch' for TUI display (TUI shows "Gen N/M")
    progress = {
        "gen": epoch,
        "win_rate": 0.0,
        "avg_hp": 0.0,
        "steps": 0,
        "buffer_size": 0,
        "episodes": 0,
        "policy_loss": last.get("train_pol_loss", 0),
        "value_loss": last.get("train_val_loss", 0),
        "num_sims": 0,
        "gen_time": last.get("time_s", 0),
        "timestamp": last.get("timestamp", time.time()),
        "num_generations": total_epochs,
        "best_win_rate": last.get("val_top1", 0),  # repurpose for val top1
        "phase": phase,
    }
    with open(prog_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exp_root = Path("experiments")
    while True:
        distill_dirs = sorted([d for d in exp_root.iterdir()
                               if d.is_dir() and d.name.startswith("distill-")])
        synced = 0
        for d in distill_dirs:
            if _sync_one(d):
                synced += 1
        print(f"[{time.strftime('%H:%M:%S')}] synced {synced}/{len(distill_dirs)} distill dirs", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
