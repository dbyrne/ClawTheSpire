"""Build lean-decks-v2 encounter set.

Design goals (per 2026-04-21 v3 g88 failure analysis):
  - Start from lean-decks-v1 (189 encounters), scale hp * 0.9 across the
    board to raise combat difficulty (WR target drops from ~75% on v1 to
    ~65%; more eval signal per combat).
  - Add encounters featuring the three underrepresented / load-bearing
    mechanics identified in v3's BAD cluster analysis:
      * DODGE_AND_ROLL  (0.5% -> ~15%): direct distribution gap
      * WRAITH_FORM     (3.7% -> ~10%): direct distribution gap
      * ACCELERANT      (32% in v1 but decision-state rarer): curate decks
        where Accelerant + existing poison carries the damage plan
  - Keep v1 curation style — dedup by (deck, enemies, relics).

Compared to uber-decks-v1:
  - Much tighter scope (3 gap-fillers vs 6)
  - No merge with draw-synergy-v1 (stay aligned to v1's deck distribution)
  - HP scaled 0.9 AFTER calibration so 50%-WR anchor becomes ~40% WR
    (explicit difficulty bump, not a re-calibration target)

Usage:
    PYTHONIOENCODING=utf-8 python scripts/build_lean_decks_v2.py \\
        --output experiments/_benchmark/encounter_sets/lean-decks-v2.jsonl \\
        --checkpoint C:/coding-projects/sts2-reanalyse-v3/sts2-solver/experiments/reanalyse-v3/betaone_gen88.pt

Outputs:
    {output}           — the encounter set in JSONL
    {output}.audit.txt — coverage audit report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import sts2_engine

from sts2_solver.betaone.deck_gen import build_random_deck
from sts2_solver.betaone.network import (
    BetaOneNetwork, export_onnx, network_kwargs_from_meta,
)

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "STS2-Agent" / "mcp_server" / "data" / "eng"
_SOLVER_ROOT = _REPO / "sts2-solver" / "src" / "sts2_solver"


# Eval coverage target for audit. Same set as build_uber_decks.
EVAL_REQUIRED_CARDS = set("""
ACCELERANT ACCURACY ACROBATICS ADRENALINE BACKFLIP BLADE_DANCE BLUR BULLET_TIME
BURST CALCULATED_GAMBLE CLOAK_AND_DAGGER DAGGER_SPRAY DAGGER_THROW DEADLY_POISON
DEFEND_SILENT DODGE_AND_ROLL ESCAPE_PLAN EXPOSE FINISHER FOOTWORK GRAND_FINALE
HIDDEN_DAGGERS INFECTION INFINITE_BLADES MALAISE NEUTRALIZE NOXIOUS_FUMES
OMNISLICE PIERCING_WAIL PREDATOR PREPARED REFLEX SKEWER STORM_OF_STEEL
STRIKE_SILENT SUCKER_PUNCH SURVIVOR TACTICIAN TOOLS_OF_THE_TRADE UNTOUCHABLE
WELL_LAID_PLANS WRAITH_FORM
""".split())
STATUS_CARDS = {"SHIV", "SLIMED", "WOUND"}


# Gap-filler specs — tighter than uber-decks.
# Accelerant encounters: core includes at least one poison-apply card so
# the Accelerant-poison combo is mechanically available during combat.
GAP_FILLER_SPECS = [
    (["DODGE_AND_ROLL"],              30, "dodge_and_roll"),
    (["WRAITH_FORM"],                 20, "wraith_form"),
    (["ACCELERANT", "DEADLY_POISON"], 20, "accelerant_poison"),
]

HP_SCALE = 0.81  # across-the-board difficulty bump (v1 baseline × 0.81)


# ---------------------------------------------------------------------------
# Helpers (mirror build_uber_decks.py)
# ---------------------------------------------------------------------------

def load_encounter_set(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encounter_dedup_key(enc: dict) -> tuple:
    deck = enc.get("deck", [])
    card_ids = tuple(sorted((c.get("id") if isinstance(c, dict) else c) for c in deck))
    enemies = tuple(enc.get("enemies", []))
    relics = tuple(sorted(enc.get("relics", [])))
    return (card_ids, enemies, relics)


def load_game_data() -> tuple[str, str]:
    monsters_raw = json.loads((_DATA / "monsters.json").read_text(encoding="utf-8"))
    monsters = {
        m["id"]: {
            "name": m.get("name", m["id"]),
            "min_hp": m.get("min_hp") or 20,
            "max_hp": m.get("max_hp") or 20,
        }
        for m in monsters_raw if m.get("id")
    }
    profiles = json.loads((_SOLVER_ROOT / "enemy_profiles.json").read_text(encoding="utf-8"))
    return json.dumps(monsters), json.dumps(profiles)


def load_card_vocab(ckpt_path: Path) -> tuple[dict, str]:
    cv = json.loads((ckpt_path.parent / "card_vocab.json").read_text(encoding="utf-8"))
    return cv, json.dumps(cv)


def export_ckpt_onnx(ckpt_path: Path, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cv = json.loads((ckpt_path.parent / "card_vocab.json").read_text(encoding="utf-8"))
    kwargs = network_kwargs_from_meta(ckpt.get("arch_meta"))
    net = BetaOneNetwork(num_cards=len(cv), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return export_onnx(net, str(out_dir))


def calibrate_hp(
    enemies: list[str], deck: list[dict], relics: list[str],
    monster_json: str, profiles_json: str, card_vocab_json: str,
    onnx_path: str, sims: int = 50, combats: int = 32,
) -> int | None:
    relics_list = [relics] * combats
    lo, hi = 15, 70
    best_hp = None
    best_diff = 1.0

    for _ in range(6):
        mid = (lo + hi) // 2
        r = sts2_engine.betaone_mcts_selfplay(
            encounters_json=json.dumps([enemies] * combats),
            decks_json=json.dumps([deck] * combats),
            player_hp=mid, player_max_hp=70, player_max_energy=3,
            relics_json=json.dumps(relics_list), potions_json="[]",
            monster_data_json=monster_json,
            enemy_profiles_json=profiles_json,
            onnx_path=onnx_path, card_vocab_json=card_vocab_json,
            num_sims=sims, temperature=0.0,
            seeds=list(range(combats)),
            add_noise=False,
        )
        wins = sum(1 for o in r["outcomes"] if o == "win")
        wr = wins / max(len(r["outcomes"]), 1)
        diff = abs(wr - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_hp = mid
        if wr > 0.55:
            hi = mid - 1
        elif wr < 0.45:
            lo = mid + 1
        else:
            break
        if lo > hi:
            break

    if best_hp is None or best_diff > 0.30:
        return None
    return best_hp


def scale_hp(enc: dict, scale: float) -> dict:
    """Return a copy with hp scaled, min 1."""
    out = dict(enc)
    out["hp"] = max(1, int(round(enc.get("hp", 50) * scale)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to v3 g88 .pt for HP calibration of new encounters")
    p.add_argument("--base-set", default="lean-decks-v1",
                   help="Base encounter set name (default: lean-decks-v1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--combats", type=int, default=32)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print("=" * 70)
    print("build_lean_decks_v2")
    print("=" * 70)

    # --- Load base set, scale HP ---
    es_dir = _REPO / "sts2-solver" / "experiments" / "_benchmark" / "encounter_sets"
    base_path = es_dir / f"{args.base_set}.jsonl"
    base_raw = load_encounter_set(base_path)
    print(f"\n  loaded {args.base_set}: {len(base_raw)} encounters")
    base_scaled = [scale_hp(e, HP_SCALE) for e in base_raw]

    # Collect pools for new-encounter sampling
    enemy_pool = [tuple(e.get("enemies", [])) for e in base_raw if e.get("enemies")]
    relic_pool = [tuple(sorted(e.get("relics", []))) for e in base_raw]

    # Dedup signature of base (to skip dup gap-fillers)
    seen = set()
    for e in base_scaled:
        seen.add(encounter_dedup_key(e))

    # --- Generate gap-filler candidates ---
    gap_candidates: list[dict] = []
    for core_cards, count, label in GAP_FILLER_SPECS:
        print(f"\n  generating gap-filler: {label} ({core_cards}) × {count}")
        attempts = 0
        for _ in range(count):
            while attempts < count * 10:
                attempts += 1
                deck = build_random_deck(rng=rng, core_cards=core_cards)
                enemies = list(rng.choice(enemy_pool))
                relics = list(rng.choice(relic_pool))
                cand = {
                    "enemies": enemies,
                    "deck": deck,
                    "hp": 50,  # overwritten by calibration
                    "relics": relics,
                    "_gap_label": label,
                }
                if encounter_dedup_key(cand) not in seen:
                    seen.add(encounter_dedup_key(cand))
                    gap_candidates.append(cand)
                    break

    print(f"\n  gap-filler candidates generated: {len(gap_candidates)}")

    # --- Calibrate new encounters ---
    print(f"\n  exporting ONNX from {args.checkpoint.name}...")
    onnx_dir = Path(tempfile.gettempdir()) / "lean_decks_v2_onnx"
    onnx_path = export_ckpt_onnx(args.checkpoint, onnx_dir)

    monster_json, profiles_json = load_game_data()
    card_vocab, card_vocab_json = load_card_vocab(args.checkpoint)

    print(f"\n  calibrating {len(gap_candidates)} new encounters against "
          f"{args.checkpoint.parent.name} (50%% WR target)...")
    t0 = time.time()
    calibrated_new: list[dict] = []
    for i, enc in enumerate(gap_candidates):
        hp = calibrate_hp(
            enemies=enc["enemies"],
            deck=enc["deck"],
            relics=enc.get("relics", []),
            monster_json=monster_json, profiles_json=profiles_json,
            card_vocab_json=card_vocab_json, onnx_path=onnx_path,
            sims=args.sims, combats=args.combats,
        )
        if hp is None:
            continue
        # Apply HP_SCALE *after* calibration — explicit difficulty bump.
        scaled_hp = max(1, int(round(hp * HP_SCALE)))
        out = {
            "enemies": enc["enemies"],
            "deck": enc["deck"],
            "hp": scaled_hp,
            "relics": enc.get("relics", []),
        }
        calibrated_new.append(out)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(gap_candidates) - (i + 1)) / max(rate, 0.001)
        print(f"    [{i+1}/{len(gap_candidates)}] {enc['_gap_label']} hp={hp}→{scaled_hp}  "
              f"{len(calibrated_new)}/{i+1} kept  "
              f"{elapsed/60:.1f}m elapsed  ETA {eta/60:.1f}m",
              flush=True)

    # --- Assemble final v2 ---
    v2 = base_scaled + calibrated_new
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for enc in v2:
            f.write(json.dumps(enc) + "\n")
    print(f"\n  wrote {len(v2)} encounters to {args.output}")

    # --- Coverage audit ---
    audit_path = args.output.with_suffix(".audit.txt")
    card_counts = Counter()
    for enc in v2:
        ids = {c.get("id") if isinstance(c, dict) else c for c in enc.get("deck", [])}
        for cid in ids:
            card_counts[cid] += 1

    n = len(v2)
    lines = [f"Coverage audit: {args.output.name}", f"n = {n} encounters",
             f"Base: {args.base_set} ({len(base_scaled)}), added: {len(calibrated_new)}",
             f"HP scale applied: ×{HP_SCALE}", ""]
    lines.append(f"{'card':25} | {'count':>6} | {'pct':>6}")
    lines.append("-" * 50)
    gap_count = 0
    for card in sorted(EVAL_REQUIRED_CARDS - STATUS_CARDS):
        c = card_counts.get(card, 0)
        pct = c / n * 100
        flag = " ← GAP" if pct < 20 else ""
        if pct < 20:
            gap_count += 1
        lines.append(f"{card:25} | {c:>6} | {pct:>5.1f}%{flag}")
    lines.append("")
    lines.append(f"gaps (<20% coverage): {gap_count} of {len(EVAL_REQUIRED_CARDS - STATUS_CARDS)}")
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  coverage audit → {audit_path}")


if __name__ == "__main__":
    main()
