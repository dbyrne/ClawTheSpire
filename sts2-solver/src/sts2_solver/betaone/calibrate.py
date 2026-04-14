"""Calibrate recorded encounters to ~50% MCTS win rate.

Reads recorded_encounters.jsonl, binary searches player HP for each
encounter, and writes calibrated_hp back to the records.

Usage:
    python -m sts2_solver.betaone.calibrate [--sims 50] [--combats 32]
"""

import argparse
import json
from pathlib import Path

import sts2_engine

from .deck_gen import lookup_card

_DATA_DIR = Path(__file__).resolve().parents[4] / "STS2-Agent" / "mcp_server" / "data" / "eng"
_SOLVER_DIR = Path(__file__).resolve().parents[1]
_CHECKPOINTS = Path(__file__).resolve().parents[4] / "betaone_checkpoints"


def _load_game_data():
    monsters_raw = json.loads((_DATA_DIR / "monsters.json").read_text(encoding="utf-8"))
    monsters = {m["id"]: {"name": m.get("name", m["id"]),
                "min_hp": m.get("min_hp") or 20,
                "max_hp": m.get("max_hp") or 20}
                for m in monsters_raw if m.get("id")}
    profiles = json.loads((_SOLVER_DIR / "enemy_profiles.json").read_text(encoding="utf-8"))
    card_vocab = json.loads((_CHECKPOINTS / "card_vocab.json").read_text(encoding="utf-8"))
    return json.dumps(monsters), json.dumps(profiles), json.dumps(card_vocab)


def calibrate_encounter(enc: dict, monster_json: str, profiles_json: str,
                        card_vocab_json: str, onnx_path: str,
                        num_sims: int = 50, combats: int = 32) -> int | None:
    """Binary search on player HP to find ~50% MCTS win rate."""
    enemy_ids = enc["enemy_ids"]
    deck_ids = enc.get("deck", [])

    deck_cards = []
    for cid in deck_ids:
        try:
            deck_cards.append(lookup_card(cid.rstrip("+")))
        except Exception:
            pass
    if not deck_cards:
        print(f"  Skipping: no valid deck cards")
        return None

    lo, hi = 15, 70
    best_hp = None
    best_diff = 1.0

    for _ in range(6):
        mid = (lo + hi) // 2
        r = sts2_engine.betaone_mcts_selfplay(
            encounters_json=json.dumps([enemy_ids] * combats),
            decks_json=json.dumps([deck_cards] * combats),
            player_hp=mid, player_max_hp=70, player_max_energy=3,
            relics=[], potions_json="[]",
            monster_data_json=monster_json,
            enemy_profiles_json=profiles_json,
            onnx_path=onnx_path,
            card_vocab_json=card_vocab_json,
            num_sims=num_sims, temperature=0.0,
            seeds=list(range(combats)),
            add_noise=False,
        )
        wins = sum(1 for o in r["outcomes"] if o == "win")
        wr = wins / max(len(r["outcomes"]), 1)
        diff = abs(wr - 0.5)
        print(f"  HP={mid:3d}: {wins}/{combats} = {wr:.0%}")

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

    return best_hp


def calibrate_all(records: list[dict], monster_json: str, profiles_json: str,
                  card_vocab_json: str, onnx_path: str,
                  encounters_path: str | Path | None = None,
                  num_sims: int = 50, combats: int = 32,
                  quiet: bool = False) -> tuple[list[dict], float | None]:
    """Calibrate all recorded encounters in-place.

    Returns (records, avg_calibrated_hp).  Writes updated records to disk
    if *encounters_path* is provided.
    """
    if not records:
        return records, None

    if not quiet:
        print(f"Calibrating {len(records)} encounters ({num_sims} sims, {combats} combats/HP)")

    updated = 0
    hp_values: list[int] = []
    for i, rec in enumerate(records):
        enemy_names = rec.get("enemy_names", rec.get("enemy_ids", ["?"]))
        old_hp = rec.get("calibrated_hp", "-")
        if not quiet:
            print(f"  [{i+1}/{len(records)}] {enemy_names} (current cal={old_hp})")

        hp = calibrate_encounter(rec, monster_json, profiles_json, card_vocab_json,
                                 onnx_path, num_sims=num_sims, combats=combats)
        if hp is not None:
            rec["calibrated_hp"] = hp
            hp_values.append(hp)
            if not quiet:
                print(f"    -> calibrated_hp = {hp}")
            updated += 1
        else:
            # Keep existing calibrated_hp if recalibration skipped
            if rec.get("calibrated_hp") is not None:
                hp_values.append(rec["calibrated_hp"])
            if not quiet:
                print(f"    -> skipped")

    if encounters_path is not None:
        with open(encounters_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    avg_hp = sum(hp_values) / len(hp_values) if hp_values else None
    if not quiet:
        print(f"  Calibration done: {updated}/{len(records)} updated, avg HP = {avg_hp:.1f}" if avg_hp else
              f"  Calibration done: no valid encounters")
    return records, avg_hp


def main():
    parser = argparse.ArgumentParser(description="Calibrate recorded encounters")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per decision")
    parser.add_argument("--combats", type=int, default=32, help="Combats per HP level")
    parser.add_argument("--path", default=str(_CHECKPOINTS / "recorded_encounters.jsonl"))
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No recorded encounters at {path}")
        return

    onnx_path = str(_CHECKPOINTS / "onnx" / "betaone.onnx")
    if not Path(onnx_path).exists():
        # Export from latest checkpoint
        import torch
        from .network import BetaOneNetwork, export_onnx
        card_vocab = json.loads((_CHECKPOINTS / "card_vocab.json").read_text(encoding="utf-8"))
        net = BetaOneNetwork(num_cards=len(card_vocab))
        ckpt = torch.load(_CHECKPOINTS / "betaone_latest.pt", weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        onnx_path = export_onnx(net, str(_CHECKPOINTS / "onnx"))
        print(f"Exported ONNX from gen {ckpt.get('gen', '?')}")

    monster_json, profiles_json, card_vocab_json = _load_game_data()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(l) for l in lines if l.strip()]

    calibrate_all(records, monster_json, profiles_json, card_vocab_json,
                  onnx_path, encounters_path=path,
                  num_sims=args.sims, combats=args.combats)


if __name__ == "__main__":
    main()
