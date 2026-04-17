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
# Training units
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    """Legacy regression tuple. Used by TARGET_MODE_OUTCOME."""
    state_after_mod: DeckBuildingState
    value_target: float


@dataclass
class TrainingPair:
    """One deck-building decision as a pairwise comparison.

    Every ADD-chosen decision yields a pair: the chosen post-mod state
    (deck+X) vs the counterfactual do-nothing state (deck unchanged),
    each annotated with its bootstrap policy-weight target. Training
    uses a pairwise margin loss that cares only about the DIRECTION of
    target_chosen - target_skip, never the absolute magnitudes. This
    principled decoupling from absolute scale kills the deck-size
    shortcut that absolute-regression variants kept learning.
    """
    state_chosen: DeckBuildingState   # deck+X after ADD
    state_skip: DeckBuildingState     # deck unchanged
    target_chosen: float              # bootstrap target for chosen state
    target_skip: float                # bootstrap target for skip state


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """FIFO replay buffer holding regression samples OR pairwise comparisons.

    Mode is set at construction time. The caller picks based on target_mode;
    the loss function picks based on what the buffer holds. No win
    reservoir for pair mode — the pair targets are signed, so under/over
    representation of positive outcomes is handled at the pair level
    (margin cares about direction, not magnitude).
    """

    def __init__(self, capacity: int = 50_000, pair_mode: bool = False):
        self.capacity = capacity
        self.pair_mode = pair_mode
        self.main: deque = deque(maxlen=capacity)
        # Win reservoir only used for regression mode (sparse wins).
        self.wins: deque = deque(maxlen=5_000)
        self.win_threshold = 0.5

    def add(self, item) -> None:
        self.main.append(item)
        if not self.pair_mode and item.value_target >= self.win_threshold:
            self.wins.append(item)

    def __len__(self) -> int:
        return len(self.main)

    def sample(self, batch_size: int, win_mix: float = 0.10) -> list:
        if len(self.main) == 0:
            return []
        if self.pair_mode:
            n = min(batch_size, len(self.main))
            idx = np.random.choice(len(self.main), size=n, replace=False)
            return [self.main[i] for i in idx]
        n_wins = min(int(batch_size * win_mix), len(self.wins))
        n_main = batch_size - n_wins
        main_idx = np.random.choice(
            len(self.main), size=min(n_main, len(self.main)), replace=False,
        )
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
#                       (flawed: policy weight is relative-to-hand, confounds
#                       with deck competition, biases toward early drafts)
#   "root_value"      — per-decision target from BetaOne MCTS turn-1 root_value
#                       of subsequent combats. Deck-quality signal; no
#                       policy-weight confound. Regression, not pairwise.
TARGET_MODE_OUTCOME = "run_outcome"
TARGET_MODE_BOOTSTRAP = "policy_bootstrap"
TARGET_MODE_ROOT_VALUE = "root_value"

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


def _rootval_target(sample_floor: int, combat_stats: list[dict]) -> float | None:
    """Forward-looking, late-weighted mean of BetaOne's turn-1 root_value
    across combats at floor > sample_floor.

    Root_value is BetaOne's estimate of "how winnable is this combat at its
    start with this deck" — a pure deck-quality signal. Unlike policy
    weight, it doesn't depend on which other cards are competing in the
    hand, so the target isn't confounded by deck composition.
    """
    sum_w = 0.0
    sum_wv = 0.0
    n = 0
    for combat in combat_stats:
        floor = combat.get("floor", 0)
        if floor <= sample_floor:
            continue
        rv = combat.get("root_value")
        if rv is None:
            continue
        weight = _late_weight(floor, sample_floor)
        if weight <= 0:
            continue
        # BetaOne's V can exceed [-1, 1] due to value-head calibration;
        # DeckNet's V is tanh-bounded. Clamp so the target is reachable.
        rv_clamped = max(-1.0, min(1.0, float(rv)))
        sum_w += weight
        sum_wv += weight * rv_clamped
        n += 1
    if n == 0 or sum_w == 0.0:
        return None
    return sum_wv / sum_w


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


def extract_pairs(
    run_result: dict, card_vocab_inv: dict[int, str],
) -> list[TrainingPair]:
    """Extract pairwise comparisons from ADD-chosen decisions.

    Rust emits option_samples with:
      - raw_state_json: DeckBuildingState snapshot before decision
      - option_types, option_cards, chosen_idx: which option was picked
      - card_policy_stats_json: per-combat MCTS policy aggregates

    Each ADD-chosen decision yields one TrainingPair: (deck+X, deck)
    with policy-weight targets for each side. The pair is usable only
    when both targets can be computed (card X was drawn at least
    MIN_OCCURRENCES times forward, and the deck has at least one other
    card with enough signal for the skip baseline). Decisions where
    the model chose SKIP produce no pair — we have no counterfactual
    data about what adding would have given us.
    """
    pairs: list[TrainingPair] = []
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

        # Only ADD-chosen decisions can form pairs (we have no bootstrap
        # signal for counterfactual ADDs on SKIP-chosen decisions).
        OPTION_CARD_REWARD = 12
        chosen_type = option_types[chosen_idx]
        if chosen_type != OPTION_CARD_REWARD:
            continue

        vocab_id = option_cards[chosen_idx]
        chosen_card_id = card_vocab_inv.get(vocab_id)
        if chosen_card_id is None or chosen_card_id in ("<PAD>", "<UNK>"):
            continue

        stats_json = os_sample.get("card_policy_stats_json", "")
        combat_stats = json.loads(stats_json) if stats_json else []
        sample_floor = int(os_sample.get("floor", 0))

        raw_chosen = _bootstrap_target_for_card(
            chosen_card_id, sample_floor, combat_stats,
        )
        raw_skip = _bootstrap_target_for_skip(
            state_before.deck, sample_floor, combat_stats,
        )
        if raw_chosen is None or raw_skip is None:
            continue

        # Map [0, 1] policy mean → [-1, +1] V target (margin loss cares
        # only about sign of (t_chosen - t_skip), but keeping the same
        # scale as V outputs keeps the margin parameter interpretable).
        target_chosen = 2.0 * raw_chosen - 1.0
        target_skip = 2.0 * raw_skip - 1.0

        state_chosen = apply_mod(
            state_before,
            DeckModification(
                kind=ModKind.ADD,
                card=CardRef(id=chosen_card_id, upgraded=False),
            ),
        )
        state_skip = apply_mod(
            state_before, DeckModification(kind=ModKind.IDENTITY),
        )
        pairs.append(TrainingPair(
            state_chosen=state_chosen, state_skip=state_skip,
            target_chosen=target_chosen, target_skip=target_skip,
        ))

    return pairs


def extract_rootval_samples(
    run_result: dict, card_vocab_inv: dict[int, str],
) -> list[TrainingSample]:
    """Extract regression samples using MCTS root_value as the target.

    One sample per decision (ADD or SKIP, whichever was chosen). Target is
    the forward-looking, late-weighted mean of BetaOne's turn-1 root_value
    over subsequent combats. Samples where no subsequent combats happened
    are dropped.
    """
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

        OPTION_CARD_REWARD = 12
        OPTION_CARD_SKIP = 13
        chosen_type = option_types[chosen_idx]
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
            continue

        stats_json = os_sample.get("card_policy_stats_json", "")
        combat_stats = json.loads(stats_json) if stats_json else []
        sample_floor = int(os_sample.get("floor", 0))
        target = _rootval_target(sample_floor, combat_stats)
        if target is None:
            continue  # no subsequent combats, no signal

        state_after = apply_mod(state_before, mod)
        samples.append(TrainingSample(
            state_after_mod=state_after, value_target=float(target),
        ))
    return samples


def extract_samples(
    run_result: dict, card_vocab_inv: dict[int, str],
) -> list[TrainingSample]:
    """Legacy TARGET_MODE_OUTCOME extractor: one sample per decision with
    run_outcome broadcast as the target. Kept for diagnostic comparisons
    against the bootstrap path."""
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

        OPTION_CARD_REWARD = 12
        OPTION_CARD_SKIP = 13
        chosen_type = option_types[chosen_idx]
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
            continue

        state_after = apply_mod(state_before, mod)
        samples.append(TrainingSample(
            state_after_mod=state_after, value_target=run_outcome,
        ))

    return samples


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

# Pairwise margin — gradient stops once V_chosen - V_skip exceeds this in the
# target-preferred direction. Tuned against the scale of V outputs ∈ [-1,+1].
PAIRWISE_MARGIN = 0.1


def train_pair_batch(
    net: DeckNet, optimizer: torch.optim.Optimizer,
    pairs: list[TrainingPair], card_vocab: dict[str, int],
    margin: float = PAIRWISE_MARGIN,
) -> float:
    """One pairwise-margin gradient step. Returns mean loss.

    For each pair, the loss pushes V(chosen) and V(skip) apart in the
    direction of sign(target_chosen - target_skip):

        L = max(0, margin - sign(Δt) · (V_chosen - V_skip))

    Only the ordering of targets matters — absolute scale falls out. The
    model cannot latch onto a "bigger deck = higher V" shortcut because
    within each pair the two states differ by exactly one card and the
    target ordering is data-driven (not size-driven).
    """
    # Encode chosen and skip states together so encoding cost amortizes
    all_states = [p.state_chosen for p in pairs] + [p.state_skip for p in pairs]
    batch = encode_batch(all_states, card_vocab)

    v_all = net(
        batch["card_ids"], batch["card_stats"],
        batch["deck_mask"], batch["global_state"],
    )
    n = len(pairs)
    v_chosen = v_all[:n]
    v_skip = v_all[n:]

    t_chosen = torch.tensor([p.target_chosen for p in pairs], dtype=torch.float32)
    t_skip = torch.tensor([p.target_skip for p in pairs], dtype=torch.float32)
    delta_t = t_chosen - t_skip
    # Drop ties (|Δt| < 1e-6) to avoid meaningless pushes
    mask = delta_t.abs() > 1e-6
    if not mask.any():
        return 0.0
    sign = torch.sign(delta_t[mask])
    diff = (v_chosen - v_skip)[mask]
    loss = F.relu(margin - sign * diff).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        [p for p in net.parameters() if p.requires_grad], 1.0,
    )
    optimizer.step()
    return loss.item()


def train_batch(
    net: DeckNet, optimizer: torch.optim.Optimizer,
    states: list[DeckBuildingState], targets: list[float],
    card_vocab: dict[str, int],
) -> float:
    """Legacy regression step (TARGET_MODE_OUTCOME)."""
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

    pair_mode = (target_mode == TARGET_MODE_BOOTSTRAP)
    if target_mode not in (
        TARGET_MODE_OUTCOME, TARGET_MODE_BOOTSTRAP, TARGET_MODE_ROOT_VALUE,
    ):
        raise ValueError(f"Unknown target_mode: {target_mode!r}")
    replay = ReplayBuffer(capacity=replay_capacity, pair_mode=pair_mode)
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

        # --- Extract training units (pairs or samples) ---
        n_wins = 0
        n_units_added = 0
        n_units_dropped = 0
        floors = []
        for run in results:
            floors.append(run.get("floor_reached", 0))
            if run.get("outcome") == "win":
                n_wins += 1
            raw_count = len(run.get("option_samples", []))
            if pair_mode:
                extracted = extract_pairs(run, card_vocab_inv)
            elif target_mode == TARGET_MODE_ROOT_VALUE:
                extracted = extract_rootval_samples(run, card_vocab_inv)
            else:
                extracted = extract_samples(run, card_vocab_inv)
            n_units_dropped += max(0, raw_count - len(extracted))
            for unit in extracted:
                replay.add(unit)
                n_units_added += 1

        # --- Train ---
        trained = len(replay) >= batch_size
        total_loss = 0.0
        n_train_steps = 0
        if trained:
            net.train()
            for _ in range(train_steps_per_gen):
                batch = replay.sample(batch_size)
                if not batch:
                    break
                if pair_mode:
                    total_loss += train_pair_batch(net, optimizer, batch, card_vocab)
                else:
                    states = [s.state_after_mod for s in batch]
                    targets = [s.value_target for s in batch]
                    total_loss += train_batch(
                        net, optimizer, states, targets, card_vocab,
                    )
                n_train_steps += 1
        avg_loss = (total_loss / n_train_steps) if n_train_steps else 0.0
        elapsed = time.time() - t0

        # --- Log every gen, whether or not a training step ran. Without this
        # the TUI sees no progress file and shows the experiment as "new"
        # during the warm-up phase where samples are still accumulating.
        avg_floor = float(np.mean(floors)) if floors else 0.0
        win_rate = n_wins / max(len(results), 1)
        unit_label = "pairs" if pair_mode else "samples"
        record = {
            "gen": gen,
            "runs": len(results),
            "wins": n_wins,
            "win_rate": round(win_rate, 4),
            "avg_floor": round(avg_floor, 2),
            "samples_added": n_units_added,
            "samples_dropped": n_units_dropped,
            "steps": n_units_added,  # canonical key for the TUI's window formula
            "buffer_size": len(replay),
            "loss": round(avg_loss, 5),
            "trained": trained,
            "gen_time": round(elapsed, 2),
            "target_mode": target_mode,
            "timestamp": time.time(),
        }
        status_tag = "" if trained else " [warmup]"
        drop_tag = f" dropped {n_units_dropped}" if n_units_dropped else ""
        print(f"Gen {gen:3d} | wins {n_wins:3d}/{len(results):3d} ({win_rate:.1%}) "
              f"| avg_floor {avg_floor:.1f} | {unit_label} {n_units_added:4d}{drop_tag} "
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
