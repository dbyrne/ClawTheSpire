"""DeckNet training loop.

Self-play workflow:
  1. Export DeckNet → ONNX
  2. Call Rust play_all_games_decknet → get N full-run results
  3. For each run, extract card-reward decisions (option_samples) and
     the raw DeckBuildingState snapshot embedded in each sample
  4. Broadcast the run outcome to each decision: terminal value is
     run_outcome_value(floor_reached, won)
  5. Train V with MSE loss on (state_after_chosen_mod, outcome) pairs
  6. Repeat generations until convergence

Why this is simpler than combat training:
  - No policy head, no cross-entropy, no MCTS visit counts
  - Single regression target (the run outcome)
  - Per-sample state already serialized from Rust — no simulator
    duplication in Python
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sts2_engine

from ..betaone.paths import SOLVER_PKG, BENCHMARK_DIR
from ..betaone.data_utils import load_solver_json, build_monster_data_json
from .encoder import encode_batch, MAX_DECK, GLOBAL_DIM
from .network import DeckNet
from .onnx_export import export_onnx
from .state import (
    CardRef, DeckBuildingState, DeckModification, ModKind,
    apply_mod, state_from_dict,
)


# ---------------------------------------------------------------------------
# Run outcome → scalar value target
# ---------------------------------------------------------------------------

ACT_FLOORS = 18  # rough floors-per-act; used to normalize floor_reached


def run_outcome_value(floor_reached: int, won_final_boss: bool) -> float:
    """Map a full-run outcome to a scalar in [-1, 1].

    Monotonic in progression: die on act 1 floor 3 < survive act 1 <
    reach act 2 boss < beat the Heart.

    A beat-the-run (boss-3 win) maps to +1.0. Reaching the last floor
    but losing maps to +0.6 (survived a long way). Dying on floor 1
    maps to ~-0.9.
    """
    if won_final_boss:
        return 1.0
    # Total potential floors ≈ 3 acts × ACT_FLOORS
    total = 3 * ACT_FLOORS
    frac = max(0.0, min(1.0, floor_reached / total))
    # Map [0, 1] → [-1, +0.6]. Losing deep is much better than losing early,
    # but still punished relative to winning.
    return frac * 1.6 - 1.0


# ---------------------------------------------------------------------------
# Training sample
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    """One deck-building decision's training tuple.

    `state_after_mod` is the state that would exist after the model's
    chosen action was applied. This is what V should predict the value
    of — the run outcome is the target.
    """
    state_after_mod: DeckBuildingState
    value_target: float                # the run outcome, broadcast


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """FIFO with win-reservoir oversampling.

    Pattern scavenged from legacy alphazero/self_play.py — keeps a
    dedicated reservoir of "winning" samples (value > threshold) so the
    network always sees positive examples even when wins are rare.
    """

    def __init__(self, capacity: int = 50_000, win_reservoir: int = 5_000,
                 win_threshold: float = 0.5):
        self.capacity = capacity
        self.win_threshold = win_threshold
        self.main: deque[TrainingSample] = deque(maxlen=capacity)
        self.wins: deque[TrainingSample] = deque(maxlen=win_reservoir)

    def add(self, sample: TrainingSample) -> None:
        self.main.append(sample)
        if sample.value_target >= self.win_threshold:
            self.wins.append(sample)

    def __len__(self) -> int:
        return len(self.main)

    def sample(self, batch_size: int, win_mix: float = 0.10) -> list[TrainingSample]:
        if len(self.main) == 0:
            return []
        n_wins = min(int(batch_size * win_mix), len(self.wins))
        n_main = batch_size - n_wins
        main_idx = np.random.choice(len(self.main), size=min(n_main, len(self.main)), replace=False)
        batch = [self.main[i] for i in main_idx]
        if n_wins > 0 and len(self.wins) > 0:
            win_idx = np.random.choice(len(self.wins), size=n_wins, replace=False)
            batch += [self.wins[i] for i in win_idx]
        return batch


# ---------------------------------------------------------------------------
# Sample extraction: Rust output → TrainingSamples
# ---------------------------------------------------------------------------

def extract_samples(
    run_result: dict, card_vocab_inv: dict[int, str],
) -> list[TrainingSample]:
    """Extract (state_after_mod, value_target) tuples from one Rust run result.

    Rust emits option_samples with:
      - raw_state_json: DeckBuildingState snapshot before decision
      - option_types, option_cards, chosen_idx: which option was picked
    We build the post-modification state (add the chosen card if any)
    and assign run_outcome_value as the target.
    """
    floor_reached = run_result.get("floor_reached", 0)
    outcome = run_result.get("outcome", "")
    won = outcome == "win"
    target = run_outcome_value(floor_reached, won)

    samples: list[TrainingSample] = []
    for os_sample in run_result.get("option_samples", []):
        raw_json = os_sample.get("raw_state_json", "")
        if not raw_json:
            continue
        try:
            state_dict = json.loads(raw_json)
            state_before = state_from_dict(state_dict)
        except Exception:
            continue

        chosen_idx = os_sample["chosen_idx"]
        option_types = os_sample["option_types"]
        option_cards = os_sample["option_cards"]
        if chosen_idx >= len(option_types):
            continue

        # Build the DeckModification that was chosen
        chosen_type = option_types[chosen_idx]
        OPTION_CARD_REWARD = 12
        OPTION_CARD_SKIP = 13
        if chosen_type == OPTION_CARD_REWARD:
            vocab_id = option_cards[chosen_idx]
            card_id = card_vocab_inv.get(vocab_id)
            if card_id is None or card_id in ("<PAD>", "<UNK>"):
                continue
            mod = DeckModification(kind=ModKind.ADD, card=CardRef(id=card_id, upgraded=False))
        elif chosen_type == OPTION_CARD_SKIP:
            mod = DeckModification(kind=ModKind.IDENTITY)
        else:
            continue  # Phase 0 scope — skip non-card decisions

        state_after = apply_mod(state_before, mod)
        samples.append(TrainingSample(state_after_mod=state_after, value_target=target))

    return samples


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_batch(
    net: DeckNet, optimizer: torch.optim.Optimizer,
    states: list[DeckBuildingState], targets: list[float],
    card_vocab: dict[str, int],
) -> float:
    """One gradient step on a batch. Returns MSE loss."""
    batch = encode_batch(states, card_vocab)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    predictions = net(
        batch["card_ids"], batch["card_stats"],
        batch["deck_mask"], batch["global_state"],
    )
    loss = F.mse_loss(predictions, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def load_card_vocab() -> tuple[dict[str, int], dict[int, str]]:
    path = BENCHMARK_DIR / "card_vocab.json"
    vocab = json.loads(path.read_text(encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}
    return vocab, inv


def build_full_vocabs_json() -> str:
    """Build the full Vocabs structure the Rust simulator requires.

    Rust's Vocabs struct has seven HashMap<String, i64> fields (cards,
    powers, relics, intent_types, acts, bosses, room_types). The DeckNet
    evaluator only reads cards — but the full-run simulator beyond it
    (events, map, enemy intents) needs all seven populated.

    We reuse legacy alphazero's builder since the vocab set is stable.
    """
    from ..data_loader import load_cards
    from ..alphazero.encoding import build_vocabs_from_card_db
    card_db = load_cards()
    vocabs = build_vocabs_from_card_db(card_db)
    data = {
        "cards": dict(vocabs.cards.token_to_idx),
        "powers": dict(vocabs.powers.token_to_idx),
        "relics": dict(vocabs.relics.token_to_idx),
        "intent_types": dict(vocabs.intent_types.token_to_idx),
        "acts": dict(vocabs.acts.token_to_idx),
        "bosses": dict(vocabs.bosses.token_to_idx),
        "room_types": dict(vocabs.room_types.token_to_idx),
    }
    return json.dumps(data)


def load_run_assets() -> dict:
    """Load the JSON bundles needed to drive the Rust full-run simulator.

    Uses legacy alphazero's data loaders because the Rust simulator expects
    specific object shapes (EncounterData as a keyed dict, not a list; card
    pool with rarity tags; etc.) — legacy self_play.py already produces
    exactly the right format, so we mirror its call sites.
    """
    from ..simulator import (
        _ensure_data_loaded, _MONSTERS_BY_ID,
        _load_enemy_profiles, _ENCOUNTERS_BY_ID,
        _load_event_profiles, _build_card_pool,
    )
    from ..alphazero.full_run import _card_to_dict
    from ..data_loader import CardDB, load_cards

    _ensure_data_loaded()

    monsters = {mid: {"name": m.get("name", mid),
                      "min_hp": m.get("min_hp") or 20,
                      "max_hp": m.get("max_hp") or m.get("min_hp") or 20}
                for mid, m in _MONSTERS_BY_ID.items()}
    encounters = {eid: {"id": eid, "monsters": enc.get("monsters", []),
                        "room_type": enc.get("room_type", "Normal"),
                        "is_weak": enc.get("is_weak", False)}
                  for eid, enc in _ENCOUNTERS_BY_ID.items()}

    card_db = load_cards()
    pools = _build_card_pool(card_db, "silent")
    pool_data = []
    for rarity, cards in pools.items():
        for card in cards:
            d = _card_to_dict(card)
            d["rarity"] = rarity
            pool_data.append(d)
    all_cards = [_card_to_dict(c) for c in card_db.all_cards()]

    from ..betaone.paths import SOLVER_PKG
    map_pool_file = SOLVER_PKG / "map_pool.json"
    shop_pool_file = SOLVER_PKG / "shop_pool.json"
    map_pool_json = map_pool_file.read_text(encoding="utf-8") if map_pool_file.exists() else "[]"
    shop_pool_json = shop_pool_file.read_text(encoding="utf-8") if shop_pool_file.exists() else "[]"

    return {
        "monster_data_json": json.dumps(monsters),
        "enemy_profiles_json": json.dumps(_load_enemy_profiles()),
        "encounter_pool_json": json.dumps(encounters),
        "event_profiles_json": json.dumps(_load_event_profiles()),
        "card_pool_json": json.dumps(pool_data),
        "card_db_json": json.dumps(all_cards),
        "map_pool_json": map_pool_json,
        "shop_pool_json": shop_pool_json,
    }


def train(
    output_dir: str,
    betaone_checkpoint: str,
    num_generations: int = 10,
    runs_per_gen: int = 100,
    mcts_sims: int = 200,
    temperature: float = 1.0,
    combat_replays: int = 1,
    lr: float = 3e-4,
    batch_size: int = 256,
    train_steps_per_gen: int = 50,
    option_epsilon: float = 0.15,
    replay_capacity: int = 50_000,
):
    """Train DeckNet via full-run self-play."""
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Resolve BetaOne checkpoint path. Configs may store it relative to the
    # sts2-solver repo dir (e.g., "experiments/foo/betaone_latest.pt"); resolve
    # against SOLVER_ROOT if not absolute and not found as-is.
    if not os.path.isabs(betaone_checkpoint) and not os.path.exists(betaone_checkpoint):
        from ..betaone.paths import SOLVER_ROOT
        resolved = str(SOLVER_ROOT / betaone_checkpoint)
        if os.path.exists(resolved):
            betaone_checkpoint = resolved

    # Load vocab + network
    card_vocab, card_vocab_inv = load_card_vocab()
    net = DeckNet(num_cards=len(card_vocab))
    n_copied = net.load_card_embed_from_betaone(betaone_checkpoint, freeze=True)
    print(f"DeckNet: {net.param_count():,} params, "
          f"{n_copied} card embeddings inherited (frozen) from BetaOne")

    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    # Load simulator assets (these don't change across gens)
    assets = load_run_assets()
    # Rust's Vocabs needs all seven sub-maps, not just cards — build once
    full_vocabs_json = build_full_vocabs_json()

    # BetaOne combat ONNXes — needed by the simulator to actually fight combats
    betaone_dir = os.path.dirname(betaone_checkpoint)
    betaone_onnx = os.path.join(betaone_dir, "onnx", "betaone.onnx")
    if not os.path.exists(betaone_onnx):
        # Export on the fly if missing
        from ..betaone.network import BetaOneNetwork, export_onnx as betaone_export
        bo_ckpt = torch.load(betaone_checkpoint, weights_only=False)
        bo_net = BetaOneNetwork(num_cards=len(card_vocab))
        bo_net.load_state_dict(bo_ckpt["model_state_dict"])
        betaone_onnx = betaone_export(bo_net, os.path.join(betaone_dir, "onnx"))

    replay = ReplayBuffer(capacity=replay_capacity)
    history_path = os.path.join(output_dir, "decknet_history.jsonl")
    progress_path = os.path.join(output_dir, "decknet_progress.json")

    for gen in range(1, num_generations + 1):
        t0 = time.time()

        # --- Export current DeckNet to ONNX ---
        decknet_onnx = export_onnx(net, onnx_dir)

        # --- Call Rust full-run simulator ---
        seeds = [gen * 100_000 + i for i in range(runs_per_gen)]
        results = sts2_engine.play_all_games_decknet(
            num_games=runs_per_gen,
            onnx_full_path=betaone_onnx,
            onnx_value_path=betaone_onnx,
            onnx_combat_path=betaone_onnx,
            onnx_decknet_path=decknet_onnx,
            vocab_json=full_vocabs_json,
            mcts_sims=mcts_sims,
            temperature=temperature,
            seeds=seeds,
            combat_replays=combat_replays,
            option_epsilon=option_epsilon,
            **assets,
        )

        # --- Extract training samples ---
        n_wins = 0
        n_samples_added = 0
        floors = []
        for run in results:
            floors.append(run.get("floor_reached", 0))
            if run.get("outcome") == "win":
                n_wins += 1
            for sample in extract_samples(run, card_vocab_inv):
                replay.add(sample)
                n_samples_added += 1

        if len(replay) < batch_size:
            print(f"Gen {gen}: only {len(replay)} samples buffered, skipping train")
            continue

        # --- Train ---
        net.train()
        total_loss = 0.0
        for _ in range(train_steps_per_gen):
            batch = replay.sample(batch_size)
            if not batch:
                break
            states = [s.state_after_mod for s in batch]
            targets = [s.value_target for s in batch]
            total_loss += train_batch(net, optimizer, states, targets, card_vocab)
        avg_loss = total_loss / max(train_steps_per_gen, 1)
        elapsed = time.time() - t0

        # --- Log ---
        avg_floor = float(np.mean(floors)) if floors else 0.0
        win_rate = n_wins / max(len(results), 1)
        record = {
            "gen": gen,
            "runs": len(results),
            "wins": n_wins,
            "win_rate": round(win_rate, 4),
            "avg_floor": round(avg_floor, 2),
            "samples_added": n_samples_added,
            "buffer_size": len(replay),
            "loss": round(avg_loss, 5),
            "gen_time": round(elapsed, 2),
            "timestamp": time.time(),
        }
        print(f"Gen {gen:3d} | wins {n_wins:3d}/{len(results):3d} ({win_rate:.1%}) "
              f"| avg_floor {avg_floor:.1f} | samples {n_samples_added:4d} "
              f"| buf {len(replay):5d} | loss {avg_loss:.4f} | {elapsed:.1f}s")
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(progress_path, "w") as f:
            record["num_generations"] = num_generations
            json.dump(record, f, indent=2)

        # --- Checkpoint ---
        torch.save(
            {
                "gen": gen, "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_cards": len(card_vocab),
            },
            os.path.join(output_dir, "decknet_latest.pt"),
        )
        if gen % 5 == 0:
            torch.save(
                {"gen": gen, "model_state_dict": net.state_dict(), "num_cards": len(card_vocab)},
                os.path.join(output_dir, f"decknet_gen{gen}.pt"),
            )

    print(f"\nTraining complete after {num_generations} generations.")
