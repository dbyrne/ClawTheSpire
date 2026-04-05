"""Combined validator — runs simulator, decision, and move table checks.

Usage:
    python -m sts2_solver.validate [logs_dir]

Defaults to the latest gen*/ directory under logs/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .validate_snapshots import (
    main as snapshot_main,
    print_report as snapshot_report,
)
from .validate_decisions import (
    main as decision_main,
    print_report as decision_report,
)
from .validate_move_tables import (
    main as move_table_main,
    print_report as move_table_report,
)

log = logging.getLogger(__name__)


def _resolve_logs_dir(dir_arg: Path | None) -> Path:
    """Resolve logs directory, defaulting to latest gen*/."""
    if dir_arg is not None:
        return dir_arg
    base = Path(__file__).resolve().parents[3] / "logs"
    gen_dirs = sorted(base.glob("gen*/"), key=lambda p: p.name)
    if gen_dirs:
        return gen_dirs[-1]
    return base


def main(logs_dir: Path | None = None) -> int:
    """Run all validators and return exit code (0 = all pass)."""
    logs_dir = _resolve_logs_dir(logs_dir)
    print(f"\nValidating logs in: {logs_dir}\n")

    # --- Simulator validation ---
    snap_report = snapshot_main(logs_dir)

    # --- Decision routing validation ---
    dec_report = decision_main(logs_dir)

    # --- Move table validation ---
    mt_report = move_table_main(logs_dir)

    # --- Combined summary ---
    snap_ok = snap_report.failed == 0
    dec_ok = dec_report.failed == 0
    mt_ok = mt_report.total_mismatches == 0

    print(f"{'='*60}")
    print(f"  COMBINED RESULTS")
    print(f"{'='*60}")
    print(f"  Simulator:    {'PASS' if snap_ok else 'FAIL'}"
          f"  ({snap_report.passed}/{snap_report.validated} turns)")
    print(f"  Decisions:    {'PASS' if dec_ok else 'FAIL'}"
          f"  ({dec_report.passed}/{dec_report.total} decisions)")
    print(f"  Move tables:  {'PASS' if mt_ok else 'FAIL'}"
          f"  ({mt_report.total_mismatches} mismatches"
          f", {len(mt_report.missing_enemies)} missing)")
    if dec_report.warnings:
        print(f"  Warnings:     {dec_report.warnings}")
    if dec_report.network_quality:
        print(f"  Net quality:  {len(dec_report.network_quality)} turns with issues")
    print(f"{'='*60}\n")

    return 0 if (snap_ok and dec_ok and mt_ok) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(dir_arg))
