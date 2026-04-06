"""Extract real shop offerings from logs for use in self-play training.

Parses shop_snapshot events to collect card offerings with real prices.
Self-play can sample from this pool instead of generating synthetic shops.

Usage:
    python -m sts2_solver.build_shop_pool [logs_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


def extract_shops(logs_dir: Path) -> list[dict]:
    """Extract shop snapshots from all logs.

    Returns list of {floor, gold, cards, relics, potions, remove_cost}.
    """
    shops: list[dict] = []

    for f in sorted(logs_dir.rglob("*.jsonl")):
        try:
            with open(f, encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh]
        except Exception:
            continue

        for e in events:
            if e.get("type") == "shop_snapshot":
                cards = e.get("cards", [])
                if not cards:
                    continue
                shops.append({
                    "floor": e.get("floor", 0),
                    "cards": cards,
                    "relics": e.get("relics", []),
                    "potions": e.get("potions", []),
                    "remove_cost": e.get("remove_cost"),
                })

    return shops


def _default_pool_path() -> Path:
    return Path(__file__).resolve().parent / "shop_pool.json"


def main(logs_dir: Path | None = None) -> int:
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"

    shops = extract_shops(logs_dir)
    pool_path = _default_pool_path()

    # Merge with existing pool
    existing: list[dict] = []
    if pool_path.exists():
        with open(pool_path, encoding="utf-8") as f:
            existing = json.load(f)

    # Deduplicate by card set (same cards at same prices = same shop)
    seen: set[tuple] = set()
    merged: list[dict] = []
    for shop in existing + shops:
        key = tuple(
            (c.get("card_id", ""), c.get("price", 0))
            for c in sorted(shop["cards"], key=lambda c: c.get("card_id", ""))
        )
        if key not in seen:
            seen.add(key)
            merged.append(shop)

    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=None, separators=(",", ":"))

    # Summary
    total = len(merged)
    floor_counts = Counter(s["floor"] for s in merged)
    avg_cards = sum(len(s["cards"]) for s in merged) / max(1, total)
    all_prices = [c["price"] for s in merged for c in s["cards"] if c.get("price")]
    avg_price = sum(all_prices) / max(1, len(all_prices))
    print(f"Shop pool: {total} unique shops (avg {avg_cards:.1f} cards, avg price {avg_price:.0f}g)")
    for floor in sorted(floor_counts):
        print(f"  Floor {floor:2d}: {floor_counts[floor]} shops")

    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dir_arg)
