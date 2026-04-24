"""Rollout-based heuristic: score decks by running actual MCTS combats.

For each candidate deck + probe encounter, calls
`sts2_engine.betaone_mcts_fight_combat` to simulate a real combat via MCTS.
The score is HP-preserved (final_hp / max_hp) averaged across probes + seeds.

This is the expensive-but-grounded alternative to the forward-pass-value
heuristic in `heuristic.py`. The thesis: combat-net knows how to PLAY
combats (trained for it); the question is whether that knowledge
transfers to deck comparison when we use it as a rollout player rather
than a static evaluator.
"""

from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch

import sts2_engine

from sts2_solver.betaone.data_utils import build_monster_data_json, load_solver_json
from sts2_solver.betaone.deck_gen import lookup_card
from sts2_solver.betaone.network import (
    BetaOneNetwork,
    export_onnx,
    network_kwargs_from_meta,
)

from .heuristic import select_probes
from .state import CardRef, DeckBuildingState, DeckModification, apply_mod


# ---------------------------------------------------------------------------
# ONNX + vocab setup (one-shot per run)
# ---------------------------------------------------------------------------

@dataclass
class CombatOracle:
    """Everything the FFI needs bundled so we don't re-load per call."""
    onnx_path: str
    card_vocab_json: str
    monster_data_json: str
    enemy_profiles_json: str
    gen_id: int
    known_enemies: frozenset[str] = frozenset()


def prepare_oracle(
    checkpoint_path: Path | str,
    *,
    onnx_out_dir: Path | str | None = None,
    device: str = "cpu",
    gen_id: int = 0,
) -> CombatOracle:
    """Load checkpoint, export ONNX, load game data; return handle for FFI calls."""
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    cv_path = ckpt_path.parent / "card_vocab.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"card_vocab.json not next to {ckpt_path}")
    card_vocab = json.loads(cv_path.read_text())

    arch_meta = ckpt.get("arch_meta", {})
    kwargs = network_kwargs_from_meta(arch_meta)
    net = BetaOneNetwork(num_cards=len(card_vocab), **kwargs)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    if onnx_out_dir is None:
        onnx_out_dir = Path(tempfile.gettempdir()) / "decknet_heuristic_onnx"
    onnx_path = export_onnx(net, str(onnx_out_dir))

    md_str = build_monster_data_json()
    known = frozenset(json.loads(md_str).keys())
    return CombatOracle(
        onnx_path=onnx_path,
        card_vocab_json=json.dumps(card_vocab),
        monster_data_json=md_str,
        enemy_profiles_json=load_solver_json("enemy_profiles.json"),
        gen_id=gen_id,
        known_enemies=known,
    )


# ---------------------------------------------------------------------------
# Single-combat rollout via FFI
# ---------------------------------------------------------------------------

def _deck_to_cards_json(deck: list[CardRef]) -> str:
    cards = []
    for ref in deck:
        d = lookup_card(ref.id)
        if ref.upgraded:
            d = {**d, "upgraded": True}
        cards.append(d)
    return json.dumps(cards)


def run_probe_combat(
    deck: list[CardRef],
    enemy_ids: list[str],
    *,
    player_hp: int,
    player_max_hp: int,
    relics: list[str],
    oracle: CombatOracle,
    num_sims: int = 50,
    seed: int = 0,
) -> dict:
    """One MCTS combat via FFI. Returns {outcome, final_hp, decisions, num_sims}."""
    deck_json = _deck_to_cards_json(deck)
    return sts2_engine.betaone_mcts_fight_combat(
        deck_json=deck_json,
        player_hp=int(player_hp),
        player_max_hp=int(player_max_hp),
        player_max_energy=3,
        enemy_ids=list(enemy_ids),
        relics=list(relics),
        potions_json="[]",
        monster_data_json=oracle.monster_data_json,
        enemy_profiles_json=oracle.enemy_profiles_json,
        onnx_path=oracle.onnx_path,
        card_vocab_json=oracle.card_vocab_json,
        num_sims=int(num_sims),
        temperature=0.0,
        seed=int(seed),
        gen_id=int(oracle.gen_id),
    )


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

@dataclass
class RolloutConfig:
    k_next: int = 3
    include_boss: bool = True
    num_seeds: int = 1
    num_sims: int = 50
    probe_seed: int = 0
    # Score = win_rate * win_weight + hp_frac * hp_weight
    # Default weighted toward WR; HP acts as tiebreaker within same WR bucket.
    win_weight: float = 1.0
    hp_weight: float = 0.2


def score_candidates_rollout(
    state: DeckBuildingState,
    candidates: list[DeckModification],
    *,
    oracle: CombatOracle,
    pool: list[dict],
    config: RolloutConfig = RolloutConfig(),
    return_detail: bool = False,
) -> list[float] | tuple[list[float], list[dict]]:
    """For each candidate, run combat rollouts on probes, return mean HP-preserved.

    Same probes + same seed-offsets across all candidates → apples-to-apples
    (only variable is the resulting deck).
    """
    rng = random.Random(config.probe_seed)
    probes = select_probes(
        state, pool,
        k_next=config.k_next,
        include_boss=config.include_boss,
        rng=rng,
        known_enemies=set(oracle.known_enemies),
    )
    seeds = [config.probe_seed * 1000 + i for i in range(config.num_seeds)]

    scores: list[float] = []
    details: list[dict] = []
    max_hp = state.player.max_hp

    for mod in candidates:
        new_state = apply_mod(state, mod)
        hp_fracs: list[float] = []
        wins = 0
        trials = 0
        outcomes: list[dict] = []
        for probe in probes:
            for seed in seeds:
                result = run_probe_combat(
                    new_state.deck,
                    probe.get("enemies", []),
                    player_hp=max_hp,
                    player_max_hp=max_hp,
                    relics=list(new_state.relics),
                    oracle=oracle,
                    num_sims=config.num_sims,
                    seed=seed,
                )
                final_hp = result.get("final_hp", 0)
                hp_fracs.append(final_hp / max_hp)
                if result.get("outcome") == "win":
                    wins += 1
                trials += 1
                outcomes.append(result)

        mean_hp_frac = sum(hp_fracs) / len(hp_fracs) if hp_fracs else 0.0
        wr = wins / trials if trials else 0.0
        score = config.win_weight * wr + config.hp_weight * mean_hp_frac
        scores.append(score)
        if return_detail:
            details.append({
                "deck_size": len(new_state.deck),
                "mean_hp_frac": mean_hp_frac,
                "win_rate": wr,
                "wins": wins,
                "trials": trials,
                "score": score,
                "outcomes": outcomes,
            })

    if return_detail:
        return scores, details
    return scores
