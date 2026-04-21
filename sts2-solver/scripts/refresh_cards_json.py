"""Refresh STS2-Agent/mcp_server/data/eng/cards.json from the live game.

Queries the mod's /cards/all endpoint (added in Router.cs) which iterates
ModelDb.AllCards and serializes each card's id/name/cost/type/target/keywords/
tags/dynamic_vars/damage/block/hit_count.

The mod's dump is narrower than the existing cards.json schema. This script
performs a *merge*: fields the endpoint provides overwrite the existing
entry's matching fields; fields the endpoint doesn't provide (description,
powers_applied, upgrade deltas, cards_draw, energy_gain, hp_loss, vars,
image_url, etc.) are preserved from the existing cards.json.

Run:
    STS2_API_BASE_URL=http://127.0.0.1:8081 python scripts/refresh_cards_json.py

Default URL is 127.0.0.1:8081 (matches game_client.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_URL = os.environ.get("STS2_API_BASE_URL", "http://127.0.0.1:8081")
REPO_ROOT = Path(__file__).resolve().parents[2]
CARDS_JSON = REPO_ROOT / "STS2-Agent" / "mcp_server" / "data" / "eng" / "cards.json"


# Enum-name translation: the mod's endpoint returns enum.ToString() values
# (e.g. "AnyEnemy", "AllEnemies", "Attack"). cards.json has historically used
# those same strings for `target` and `type`, so no translation needed. But
# keyword lowercasing differs — leave as-is unless we see mismatches.


def fetch_cards(base_url: str) -> list[dict]:
    req = Request(base_url.rstrip("/") + "/cards/all")
    with urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    if not payload.get("ok"):
        raise RuntimeError(f"Endpoint returned error: {payload.get('error')}")
    return payload["data"]


def merge_one(existing: dict | None, fresh: dict) -> dict:
    """Merge one fresh card dump into the existing cards.json entry.

    Fresh fields win when present. Existing fields stay for everything the
    endpoint doesn't dump yet.
    """
    merged = dict(existing) if existing else {}
    merged["id"] = fresh["id"]
    for key in ("name", "type", "rarity", "target", "max_upgrade_level"):
        if fresh.get(key) is not None:
            merged[key] = fresh[key]
    if fresh.get("cost") is not None:
        merged["cost"] = fresh["cost"]
    if "costs_x" in fresh:
        merged["is_x_cost"] = fresh["costs_x"]
    # Damage/Block/HitCount: pull from dynamic_vars first, then top-level fallback.
    dyn = fresh.get("dynamic_vars") or {}
    for src, dst in (("Damage", "damage"), ("Block", "block"), ("HitCount", "hit_count")):
        if src in dyn:
            merged[dst] = dyn[src]
        elif fresh.get(dst) is not None:
            merged[dst] = fresh[dst]
    merged["keywords"] = fresh.get("keywords") or []
    merged["tags"] = fresh.get("tags") or []
    # Preserve existing description if the endpoint couldn't produce a clean one
    if fresh.get("description"):
        merged["description"] = fresh["description"]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(CARDS_JSON))
    parser.add_argument("--diff-only", action="store_true",
                        help="Print cards whose key fields changed; don't write.")
    args = parser.parse_args()

    print(f"Fetching {args.base_url}/cards/all ...")
    fresh_cards = fetch_cards(args.base_url)
    print(f"  got {len(fresh_cards)} cards")

    existing_cards: list[dict] = []
    existing_by_id: dict[str, dict] = {}
    out_path = Path(args.output)
    if out_path.exists():
        existing_cards = json.loads(out_path.read_text(encoding="utf-8"))
        existing_by_id = {c["id"]: c for c in existing_cards if "id" in c}
        print(f"  existing cards.json has {len(existing_cards)} entries")

    # Build merged list: one entry per fresh card (dedupe by id; keep first).
    seen: set[str] = set()
    merged_cards: list[dict] = []
    diffs: list[tuple[str, dict, dict]] = []
    for fresh in fresh_cards:
        cid = fresh.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        existing = existing_by_id.get(cid)
        merged = merge_one(existing, fresh)

        if existing is not None:
            changed = {
                k: (existing.get(k), merged.get(k))
                for k in ("damage", "block", "hit_count", "target", "cost", "type")
                if existing.get(k) != merged.get(k)
            }
            if changed:
                diffs.append((cid, {k: v[0] for k, v in changed.items()},
                                   {k: v[1] for k, v in changed.items()}))
        merged_cards.append(merged)

    # Carry over any existing cards the endpoint didn't include (defensive)
    for cid, existing in existing_by_id.items():
        if cid not in seen:
            merged_cards.append(existing)
            print(f"  [warn] {cid} not in mod dump; kept existing entry")

    if diffs:
        print(f"\n{len(diffs)} cards changed:")
        for cid, before, after in diffs[:40]:
            print(f"  {cid}: {before} -> {after}")
        if len(diffs) > 40:
            print(f"  ... and {len(diffs) - 40} more")
    else:
        print("\nNo changes detected.")

    if args.diff_only:
        return 0

    out_path.write_text(json.dumps(merged_cards, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nWrote {len(merged_cards)} cards -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
