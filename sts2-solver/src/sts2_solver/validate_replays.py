"""Backward-compatible alias for the consolidated validator.

Use validate_snapshots directly:
    python -m sts2_solver.validate_snapshots [logs_dir]
"""

from .validate_snapshots import main, print_report, SnapshotValidationReport

if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = main(dir_arg)
    sys.exit(0 if report.failed == 0 else 1)
