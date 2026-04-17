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

    Monotonic in progression: die on floor 3 < die on floor 15 < win.
    A beat-the-run maps to +1.0. Reaching the last floor but losing
    maps to ~+0.5. Dying on floor 1 maps to ~-0.9.

    Normalizer is ACT_FLOORS because Phase 0 runs Act 1 only. The
    previous 3*ACT_FLOORS denominator compressed all loser targets into
    [-0.97, -0.50] (spread 0.47) which gave the net too little signal
    between early and late deaths. With ACT_FLOORS, spread is 1.42 across
    the same range of outcomes.
    """
    if won_final_boss:
        return 1.0
    frac = max(0.0, min(1.0, floor_reached / ACT_FLOORS))
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

# Phase 0 target modes:
#   "run_outcome"     — legacy broadcast credit (run_outcome_value)
#   "policy_bootstrap"— per-decision target from BetaOne MCTS policy weights
TARGET_MODE_OUTCOME = "run_outcome"
TARGET_MODE_BOOTSTRAP = "policy_bootstrap"

# Minimum MCTS searches where a card was in hand before we trust the signal.
# Cards drawn fewer times get no target (sample dropped).
MIN_OCCURRENCES = 3

# Max floor, used to normalize late-game weight. Phase 0 runs Act 1 (≤17).
MAX_FLOOR = 17


def _late_weight(floor: int, sample_floor: int) -> float:
    """Linear ramp from the drafting floor to the end of the act.

    A combat immediately after drafting carries near-zero weight (deck
    context hasn't matured); late-run combats carry full weight. Combo
    pieces drafted early get credited by their late-game value, not
    their tepid early contribution.
    """
    remaining = max(1, MAX_FLOOR - sample_floor)
    progress = max(0, floor - sample_floor)
    return min(1.0, progress / remaining)


def _bootstrap_target_for_card(
    card_id: str, sample_floor: int,
    combat_stats: list[dict],
) -> float | None:
    """Forward-looking, late-weighted mean of MCTS policy weight on card_id.

    Considers only combats at floor > sample_floor (forward-looking) where
    the card was actually in hand. Within that window, later combats weigh
    more (late-weighting). Returns None if there's no signal — the sample
    should be dropped rather than fall back to broadcast credit.
    """
    sum_w = 0.0
    sum_wp = 0.0
    total_occ = 0
    for combat in combat_stats:
        floor = combat.get("floor", 0)
        if floor <= sample_floor:
            continue
        stats = combat.get("card_stats", {}).get(card_id)
        if not stats:
            continue
        s, c = stats[0], stats[1]
        if c <= 0:
            continue
        combat_mean = s / c
        weight = _late_weight(floor, sample_floor)
        if weight <= 0:
            continue
        sum_w += weight * c
        sum_wp += weight * s  # equivalently weight*c*combat_mean
        total_occ += c
    if sum_w == 0.0 or total_occ < MIN_OCCURRENCES:
        return None
    return sum_wp / sum_w  # weighted mean policy ∈ [0, 1]


def _bootstrap_target_for_skip(
    deck: list[CardRef], sample_floor: int, combat_stats: list[dict],
) -> float | None:
    """Skip target: median per-card forward-looking late-weighted policy
    mean across cards currently in the deck. A skip is "at least as good
    as the median existing card" — if the deck is strong, skip is easy to
    justify; if weak, a marginal add is worth taking."""
    per_card = []
    for card_ref in deck:
        v = _bootstrap_target_for_card(card_ref.id, sample_floor, combat_stats)
        if v is not None:
            per_card.append(v)
    if not per_card:
        return None
    return float(np.median(per_card))


def extract_samples(
    run_result: dict, card_vocab_inv: dict[int, str],
    target_mode: str = TARGET_MODE_BOOTSTRAP,
) -> list[TrainingSample]:
    """Extract (state_after_mod, value_target) tuples from one Rust run result.

    Rust emits option_samples with:
      - raw_state_json: DeckBuildingState snapshot before decision
      - option_types, option_cards, chosen_idx: which option was picked
      - card_policy_stats_json: per-combat MCTS policy aggregates (shared
        across all samples in the run; filtered per-sample by floor)

    For TARGET_MODE_BOOTSTRAP, target is BetaOne's revealed preference on
    the chosen card across combats AFTER the decision, weighted toward the
    late run where deck context is mature. Samples with no signal (card
    never drawn, insufficient occurrences) are dropped — they'd otherwise
    fall back to broadcast credit which is exactly what we're replacing.

    For TARGET_MODE_OUTCOME, legacy broadcast of run_outcome_value.
    """
    floor_reached = run_result.get("floor_reached", 0)
    outcome = run_result.get("outcome", "")
    won = outcome == "win"
    run_outcome = run_outcome_value(floor_reached, won)

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

        chosen_card_id: str | None = None
        if chosen_type == OPTION_CARD_REWARD:
            vocab_id = option_cards[chosen_idx]
            chosen_card_id = card_vocab_inv.get(vocab_id)
            if chosen_card_id is None or chosen_card_id in ("<PAD>", "<UNK>"):
                continue
            mod = DeckModification(
                kind=ModKind.ADD,
                card=CardRef(id=chosen_card_id, upgraded=False),
            )
        elif chosen_type == OPTION_CARD_SKIP:
            mod = DeckModification(kind=ModKind.IDENTITY)
        else:
            continue  # Phase 0 scope — skip non-card decisions

        # Compute the target
        target: float | None = None
        if target_mode == TARGET_MODE_BOOTSTRAP:
            stats_json = os_sample.get("card_policy_stats_json", "")
            combat_stats = json.loads(stats_json) if stats_json else []
            sample_floor = int(os_sample.get("floor", 0))
            if chosen_type == OPTION_CARD_REWARD:
                raw = _bootstrap_target_for_card(
                    chosen_card_id, sample_floor, combat_stats,
                )
            else:
                raw = _bootstrap_target_for_skip(
                    state_before.deck, sample_floor, combat_stats,
                )
            if raw is not None:
                # Map [0, 1] policy weight → [-1, +1] V target
                target = 2.0 * raw - 1.0
        elif target_mode == TARGET_MODE_OUTCOME:
            target = run_outcome
        else:
            raise ValueError(f"Unknown target_mode: {target_mode!r}")

        if target is None:
            continue  # no signal — drop the sample rather than fall back

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
    known_monster_ids = set(monsters.keys())

    # Filter encounters whose monsters aren't all in the monster database —
    # otherwise the Rust spawn_enemy path panics at run time. Safer to drop
    # a few entries than to abort the whole training run.
    encounters = {}
    dropped = 0
    for eid, enc in _ENCOUNTERS_BY_ID.items():
        enc_monsters = enc.get("monsters", [])
        missing = [m.get("id") for m in enc_monsters if m.get("id") not in known_monster_ids]
        if missing:
            dropped += 1
            continue
        encounters[eid] = {
            "id": eid,
            "monsters": enc_monsters,
            "room_type": enc.get("room_type", "Normal"),
            "is_weak": enc.get("is_weak", False),
        }
    if dropped:
        print(f"  Dropped {dropped} encounters with unknown monsters (kept {len(encounters)}).")

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
    target_mode: str = TARGET_MODE_BOOTSTRAP,
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
        from ..betaone.network import (
            BetaOneNetwork, export_onnx as betaone_export,
            network_kwargs_from_meta,
        )
        bo_ckpt = torch.load(betaone_checkpoint, weights_only=False)
        bo_net = BetaOneNetwork(
            num_cards=len(card_vocab),
            **network_kwargs_from_meta(bo_ckpt.get("arch_meta")),
        )
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
        n_samples_dropped = 0  # target_mode=bootstrap drops cards with no signal
        floors = []
        for run in results:
            floors.append(run.get("floor_reached", 0))
            if run.get("outcome") == "win":
                n_wins += 1
            # For bootstrap mode, extract_samples may drop samples with no
            # signal. Compare to the raw sample count for visibility.
            raw_count = len(run.get("option_samples", []))
            extracted = extract_samples(run, card_vocab_inv, target_mode=target_mode)
            n_samples_dropped += max(0, raw_count - len(extracted))
            for sample in extracted:
                replay.add(sample)
                n_samples_added += 1

        # --- Train (only if the replay buffer has enough to draw a batch) ---
        trained = len(replay) >= batch_size
        total_loss = 0.0
        n_train_steps = 0
        if trained:
            net.train()
            for _ in range(train_steps_per_gen):
                batch = replay.sample(batch_size)
                if not batch:
                    break
                states = [s.state_after_mod for s in batch]
                targets = [s.value_target for s in batch]
                total_loss += train_batch(net, optimizer, states, targets, card_vocab)
                n_train_steps += 1
        avg_loss = (total_loss / n_train_steps) if n_train_steps else 0.0
        elapsed = time.time() - t0

        # --- Log every gen, whether or not a training step ran. Without this
        # the TUI sees no progress file and shows the experiment as "new"
        # during the warm-up phase where samples are still accumulating.
        avg_floor = float(np.mean(floors)) if floors else 0.0
        win_rate = n_wins / max(len(results), 1)
        record = {
            "gen": gen,
            "runs": len(results),
            "wins": n_wins,
            "win_rate": round(win_rate, 4),
            "avg_floor": round(avg_floor, 2),
            "samples_added": n_samples_added,
            "samples_dropped": n_samples_dropped,
            "steps": n_samples_added,  # canonical key for the TUI's window formula
            "buffer_size": len(replay),
            "loss": round(avg_loss, 5),
            "trained": trained,
            "gen_time": round(elapsed, 2),
            "target_mode": target_mode,
            "timestamp": time.time(),
        }
        status_tag = "" if trained else " [warmup]"
        drop_tag = f" dropped {n_samples_dropped}" if n_samples_dropped else ""
        print(f"Gen {gen:3d} | wins {n_wins:3d}/{len(results):3d} ({win_rate:.1%}) "
              f"| avg_floor {avg_floor:.1f} | samples {n_samples_added:4d}{drop_tag} "
              f"| buf {len(replay):5d} | loss {avg_loss:.4f}{status_tag} | {elapsed:.1f}s")
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
