"""Export DeckNet to ONNX for the Rust simulator to consume.

The Rust DeckNetEvaluator (sts2-engine/src/decknet.rs) expects exactly
four named inputs and one scalar output:

    card_ids     (B, MAX_DECK)        int64
    card_stats   (B, MAX_DECK, 28)    float32
    deck_mask    (B, MAX_DECK)        bool
    global_state (B, GLOBAL_DIM)      float32
    -> value     (B,)                 float32  in [-1, 1]

We use a dynamic batch dim so Rust can send a variable number of
candidate decks in one call (card reward = 4 candidates, shop remove
= |deck|+1 candidates, etc.).
"""

from __future__ import annotations

import os
import torch

from .encoder import MAX_DECK, GLOBAL_DIM
from .network import DeckNet
from ..betaone.network import CARD_STATS_DIM


def export_onnx(net: DeckNet, out_dir: str, name: str = "decknet.onnx") -> str:
    """Export DeckNet to an ONNX file. Returns the absolute path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, name))

    net_eval = net.eval()

    # Dummy inputs with batch size 2 — dynamic axis lets the exported graph
    # accept any batch size at inference time.
    dummy_card_ids = torch.zeros((2, MAX_DECK), dtype=torch.long)
    dummy_card_stats = torch.zeros((2, MAX_DECK, CARD_STATS_DIM), dtype=torch.float32)
    dummy_deck_mask = torch.zeros((2, MAX_DECK), dtype=torch.bool)
    dummy_global = torch.zeros((2, GLOBAL_DIM), dtype=torch.float32)

    torch.onnx.export(
        net_eval,
        (dummy_card_ids, dummy_card_stats, dummy_deck_mask, dummy_global),
        out_path,
        input_names=["card_ids", "card_stats", "deck_mask", "global_state"],
        output_names=["value"],
        dynamic_axes={
            "card_ids": {0: "batch"},
            "card_stats": {0: "batch"},
            "deck_mask": {0: "batch"},
            "global_state": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return out_path
