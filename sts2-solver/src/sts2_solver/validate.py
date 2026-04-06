"""Combined validator — rebuilds profiles, then runs all checks.

Usage:
    python -m sts2_solver.validate [logs_dir]

Defaults to the latest gen*/ directory under logs/.
Steps: rebuild enemy + event profiles from ALL logs, then validate
the specified gen's logs against simulator, decisions, and move tables.
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
from .cross_validate import (
    main as cross_validate_main,
    print_report as cross_validate_report,
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


def _rebuild_profiles(logs_dir: Path) -> None:
    """Rebuild enemy and event profiles from all available logs."""
    # Use the parent of logs_dir (e.g. logs/) so we get all gens, not just one
    all_logs = logs_dir.parent if logs_dir.name.startswith("gen") else logs_dir

    from .build_enemy_profiles import (
        build_all_profiles as build_enemy,
        load_profiles as load_enemy,
        save_profiles as save_enemy,
        _default_profile_path as enemy_path,
    )
    from .build_event_profiles import (
        build_all_profiles as build_event,
        load_profiles as load_event,
        save_profiles as save_event,
        _default_profile_path as event_path,
    )

    # Enemy profiles
    existing_enemy = load_enemy(enemy_path())
    enemy_profiles = build_enemy(all_logs, min_combats=2, existing=existing_enemy)
    save_enemy(enemy_profiles, enemy_path())
    n_new_enemy = len(enemy_profiles) - len(existing_enemy)
    print(f"  Enemy profiles: {len(enemy_profiles)} total"
          f"{f' (+{n_new_enemy} new)' if n_new_enemy else ''}")

    # Event profiles
    existing_event = load_event(event_path())
    event_profiles = build_event(all_logs, min_observations=1, existing=existing_event)
    save_event(event_profiles, event_path())
    n_new_event = len(event_profiles) - len(existing_event)
    print(f"  Event profiles: {len(event_profiles)} total"
          f"{f' (+{n_new_event} new)' if n_new_event else ''}")

    # Map pool
    from .build_map_pool import main as build_maps, _default_pool_path
    n_maps = build_maps(all_logs)
    print(f"  Map pool: {n_maps} total")

    # Encounter pool
    from .build_encounter_pool import main as build_encounters
    n_enc = build_encounters(all_logs)
    print(f"  Encounter pool: {n_enc} total")

    # Shop pool
    from .build_shop_pool import main as build_shops
    n_shops = build_shops(all_logs)
    print(f"  Shop pool: {n_shops} total")


def main(logs_dir: Path | None = None) -> int:
    """Run all validators and return exit code (0 = all pass)."""
    logs_dir = _resolve_logs_dir(logs_dir)
    print(f"\nValidating logs in: {logs_dir}\n")

    # --- Rebuild profiles + map pool from latest log data ---
    print("Rebuilding profiles and map pool...")
    _rebuild_profiles(logs_dir)
    print()

    # --- Simulator validation ---
    snap_report = snapshot_main(logs_dir)

    # --- Decision routing validation ---
    dec_report = decision_main(logs_dir)

    # --- Move table validation ---
    mt_report = move_table_main(logs_dir)

    # --- Cross-validation (self-play vs real game parity) ---
    print()
    xv_report = cross_validate_main(logs_dir)

    # --- Combined summary ---
    snap_ok = snap_report.failed == 0
    dec_ok = dec_report.failed == 0
    mt_ok = mt_report.total_mismatches == 0
    xv_ep_ok = len(xv_report.enemy_phase_diffs) == 0

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
    xv_enc_pass = xv_report.encoding_turns_checked - len(xv_report.encoding_diffs)
    xv_ep_pass = xv_report.enemy_combats_checked - len(xv_report.enemy_phase_diffs)
    print(f"  X-val encode: {xv_enc_pass}/{xv_report.encoding_turns_checked} turns")
    print(f"  X-val enemy:  {'PASS' if xv_ep_ok else 'FAIL'}"
          f"  ({xv_ep_pass}/{xv_report.enemy_combats_checked} turns)")
    if xv_report.decision_turns_checked > 0:
        print(f"  X-val decide: {xv_report.decision_matches}"
              f"/{xv_report.decision_turns_checked} match")
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
