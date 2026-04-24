"""Heuristic DeckNet replacement: score candidate decks via combat-net oracle.

At each card-reward decision point, for each candidate (add X or skip):
  1. Apply the modification to the deck.
  2. Select N "probe" encounters (next-K floors + 1 boss proxy).
  3. For each probe x K random initial draws:
       Build a turn-1 Scenario, encode, forward pass, read value head.
  4. Aggregate (mean) into a per-candidate score.
  5. Argmax picks the candidate.

No training. Uses a trained BetaOne checkpoint as a deck-quality oracle.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from sts2_solver.betaone.data_utils import build_monster_data_json
from sts2_solver.betaone.deck_gen import lookup_card
from sts2_solver.betaone.eval import (
    ActionSpec,
    Scenario,
    encode_state,
    enemy as make_enemy,
)
from sts2_solver.betaone.network import (
    ACTION_DIM,
    MAX_ACTIONS,
    MAX_HAND,
    BetaOneNetwork,
    network_kwargs_from_meta,
)

from .state import CardRef, DeckBuildingState, DeckModification, apply_mod


# ---------------------------------------------------------------------------
# Encounter pool loading + probe selection
# ---------------------------------------------------------------------------

def load_encounter_pool(path: Path | str | None = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent.parent / "encounter_pool.json"
    return json.loads(Path(path).read_text())


def select_probes(
    state: DeckBuildingState,
    pool: list[dict],
    *,
    k_next: int = 3,
    include_boss: bool = True,
    rng: random.Random | None = None,
    known_enemies: set[str] | None = None,
) -> list[dict]:
    """Next-K floors from pool, plus one highest-floor encounter (boss proxy).

    The pool doesn't tag monster/elite/boss, so floor-range is the proxy.
    When `known_enemies` is provided, filters to encounters whose enemies
    all appear in that set — needed because the Rust monster-data subset
    doesn't cover every enemy id in the live-game pool.
    """
    rng = rng or random.Random(0)

    if known_enemies is not None:
        pool = [e for e in pool
                if all(eid in known_enemies for eid in e.get("enemies", []))]
    if not pool:
        return []

    probes: list[dict] = []

    low = state.floor + 1
    high = state.floor + k_next
    near = [e for e in pool if low <= e.get("floor", 0) <= high]
    if near:
        probes.extend(rng.sample(near, k=min(k_next, len(near))))
    else:
        by_distance = sorted(pool, key=lambda e: abs(e.get("floor", 0) - (state.floor + 2)))
        probes.extend(by_distance[:k_next])

    if include_boss:
        top_floor = max(e.get("floor", 0) for e in pool)
        boss_candidates = [e for e in pool if e.get("floor", 0) == top_floor]
        if not boss_candidates:
            sorted_pool = sorted(pool, key=lambda e: -e.get("floor", 0))
            boss_candidates = sorted_pool[: max(1, len(sorted_pool) // 5)]
        probes.append(rng.choice(boss_candidates))

    return probes


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

_MONSTER_DATA: dict | None = None


def _monsters() -> dict:
    global _MONSTER_DATA
    if _MONSTER_DATA is None:
        _MONSTER_DATA = json.loads(build_monster_data_json())
    return _MONSTER_DATA


def build_probe_scenario(
    deck_card_refs: list[CardRef],
    probe: dict,
    *,
    player_hp: int,
    player_max_hp: int,
    relics: set[str],
    draw_seed: int = 0,
) -> Scenario:
    """Turn-1 Scenario with deck shuffled + 5-card initial draw.

    Same probe + seed across all candidates = apples-to-apples comparison.
    Enemy intents default to generic Attack since per-enemy intent accuracy
    doesn't matter for argmax (constant offset across candidates).
    """
    rng = random.Random(draw_seed)
    ids = [c.id for c in deck_card_refs]
    rng.shuffle(ids)
    hand_ids = ids[:5]
    hand = [lookup_card(cid) for cid in hand_ids]
    draw_size = max(0, len(ids) - 5)

    monsters = _monsters()
    enemies: list[dict] = []
    for eid in probe.get("enemies", []):
        m = monsters.get(eid, {})
        hp = m.get("max_hp", 30)
        if hp > 500:  # boss stub hp=9999 would nuke the encoder scale
            hp = 250
        enemies.append(make_enemy(hp=hp, max_hp=hp, intent="Attack", damage=10, hits=1))

    return Scenario(
        name="probe",
        category="decknet_heuristic",
        description="",
        player={
            "hp": player_hp,
            "max_hp": player_max_hp,
            "energy": 3,
            "max_energy": 3,
            "block": 0,
            "powers": {},
        },
        enemies=enemies,
        hand=hand,
        actions=[ActionSpec("end_turn", label="end")],
        best_actions=[0],
        relics=set(relics),
        turn=1,
        draw_size=draw_size,
        discard_size=0,
        exhaust_size=0,
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _card_id_to_vocab_idx(card: dict, card_vocab: dict) -> int:
    cid = card.get("id", "")
    if card.get("upgraded"):
        cid = f"{cid}+"
    return card_vocab.get(cid, 0)


def value_of_scenario(
    scenario: Scenario,
    net: BetaOneNetwork,
    card_vocab: dict,
    *,
    device: str = "cpu",
) -> float:
    state_v = encode_state(scenario)
    state_t = torch.tensor([state_v], dtype=torch.float32, device=device)

    action_feats = torch.zeros(1, MAX_ACTIONS, ACTION_DIM, device=device)
    action_mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool, device=device)
    action_mask[0, 0] = False
    hand_card_ids = torch.zeros(1, MAX_HAND, dtype=torch.long, device=device)
    for i, c in enumerate(scenario.hand[:MAX_HAND]):
        hand_card_ids[0, i] = _card_id_to_vocab_idx(c, card_vocab)
    action_card_ids = torch.zeros(1, MAX_ACTIONS, dtype=torch.long, device=device)

    with torch.no_grad():
        out = net(state_t, action_feats, action_mask, hand_card_ids, action_card_ids)
    value = out[1]
    return float(value.item())


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    k_next: int = 3
    include_boss: bool = True
    num_draws: int = 3
    probe_seed: int = 0
    use_max_hp: bool = True


def score_candidates(
    state: DeckBuildingState,
    candidates: list[DeckModification],
    *,
    net: BetaOneNetwork,
    card_vocab: dict,
    pool: list[dict],
    config: ProbeConfig = ProbeConfig(),
    device: str = "cpu",
    return_detail: bool = False,
) -> list[float] | tuple[list[float], list[dict]]:
    rng = random.Random(config.probe_seed)
    probes = select_probes(
        state, pool, k_next=config.k_next,
        include_boss=config.include_boss, rng=rng,
    )
    draw_seeds = [config.probe_seed + i for i in range(config.num_draws)]

    scores: list[float] = []
    details: list[dict] = []
    for mod in candidates:
        new_state = apply_mod(state, mod)
        hp = new_state.player.max_hp if config.use_max_hp else new_state.player.hp
        values: list[float] = []
        for probe in probes:
            for seed in draw_seeds:
                scenario = build_probe_scenario(
                    new_state.deck, probe,
                    player_hp=hp,
                    player_max_hp=new_state.player.max_hp,
                    relics=set(new_state.relics),
                    draw_seed=seed,
                )
                v = value_of_scenario(scenario, net, card_vocab, device=device)
                values.append(v)
        mean_v = sum(values) / len(values) if values else 0.0
        scores.append(mean_v)
        if return_detail:
            details.append({
                "deck_size": len(new_state.deck),
                "values": values,
                "mean": mean_v,
                "std": (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0,
            })
    if return_detail:
        return scores, details
    return scores


# ---------------------------------------------------------------------------
# Checkpoint + vocab loader
# ---------------------------------------------------------------------------

def load_combat_net(
    checkpoint_path: Path | str,
    device: str = "cpu",
    *,
    untrained: bool = False,
) -> tuple[BetaOneNetwork, dict]:
    """Load trained checkpoint, or (when untrained=True) return a randomly-
    initialized network matching the checkpoint's arch + vocab size.

    Used as a baseline reference: "what would this heuristic score with no
    learned weights?"
    """
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    cv_path = ckpt_path.parent / "card_vocab.json"
    if not cv_path.exists():
        raise FileNotFoundError(
            f"card_vocab.json not found next to {ckpt_path}; "
            f"this heuristic expects vocab in the experiment dir."
        )
    card_vocab = json.loads(cv_path.read_text())

    arch_meta = ckpt.get("arch_meta", {})
    kwargs = network_kwargs_from_meta(arch_meta)
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    if not untrained:
        net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    net.to(device)
    return net, card_vocab
