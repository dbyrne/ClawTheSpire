"""DeckNet diagnostic: pick distribution and V-vs-deck-size correlation.

Loads checkpoints across training and checks:
  1. For each eval scenario: how often does the net prefer ADD vs SKIP?
  2. Does V systematically prefer larger decks (broadcast-credit bias)?
  3. What's the spread of V predictions across scenarios (mean collapse)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SOLVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOLVER / "src"))

from sts2_solver.decknet.encoder import encode_batch
from sts2_solver.decknet.eval import build_scenarios
from sts2_solver.decknet.network import DeckNet
from sts2_solver.decknet.state import ModKind, apply_mod


import sys as _sys
EXP_NAME = _sys.argv[1] if len(_sys.argv) > 1 else "decknet-phase0"
EXP_DIR = SOLVER / "experiments" / EXP_NAME
VOCAB_PATH = EXP_DIR / "card_vocab.json"


def load_model(checkpoint: str) -> DeckNet:
    vocab = json.loads(VOCAB_PATH.read_text())
    net = DeckNet(num_cards=len(vocab))
    ckpt_path = EXP_DIR / f"decknet_{checkpoint}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, vocab


def analyze(net: DeckNet, vocab: dict, label: str) -> dict:
    scenarios = build_scenarios()

    add_picks = 0
    skip_picks = 0
    v_by_deck_size: list[tuple[int, float]] = []  # (deck_size_after_mod, V)
    all_values: list[float] = []

    with torch.no_grad():
        for sc in scenarios:
            candidate_states = [apply_mod(sc.state, m) for m in sc.candidates]
            batch = encode_batch(candidate_states, vocab)
            values = net(
                batch["card_ids"], batch["card_stats"],
                batch["deck_mask"], batch["global_state"],
            )
            best = int(torch.argmax(values).item())
            chosen_mod = sc.candidates[best]
            if chosen_mod.kind == ModKind.IDENTITY:
                skip_picks += 1
            elif chosen_mod.kind == ModKind.ADD:
                add_picks += 1

            # Record V vs deck size for all candidates
            for st, v in zip(candidate_states, values.tolist()):
                v_by_deck_size.append((len(st.deck), v))
                all_values.append(v)

    sizes = np.array([s for s, _ in v_by_deck_size])
    vs = np.array([v for _, v in v_by_deck_size])
    correlation = float(np.corrcoef(sizes, vs)[0, 1]) if len(set(sizes)) > 1 else 0.0

    return {
        "label": label,
        "add_picks": add_picks,
        "skip_picks": skip_picks,
        "v_mean": float(np.mean(all_values)),
        "v_std": float(np.std(all_values)),
        "v_min": float(np.min(all_values)),
        "v_max": float(np.max(all_values)),
        "deck_size_v_corr": correlation,
    }


def main():
    checkpoints = _sys.argv[2:] or ["gen5", "gen10", "gen25", "gen50", "gen75", "gen100"]
    print(f"{'gen':>6} | ADD | SKIP | V mean  std   range     | corr(size,V)")
    print("-" * 70)
    for ckpt in checkpoints:
        net, vocab = load_model(ckpt)
        r = analyze(net, vocab, ckpt)
        print(f"{ckpt:>6} | {r['add_picks']:3d} | {r['skip_picks']:4d} | "
              f"{r['v_mean']:+.2f} {r['v_std']:.2f}  [{r['v_min']:+.2f},{r['v_max']:+.2f}] | "
              f"{r['deck_size_v_corr']:+.3f}")


if __name__ == "__main__":
    main()
