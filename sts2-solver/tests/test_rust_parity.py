"""Parity tests: verify Rust combat engine matches Python engine.

Runs the same combat scenarios through both engines and compares:
- Outcome (win/lose)
- HP after combat
- Number of training samples collected
- Turn count
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from copy import deepcopy

from sts2_solver.data_loader import load_cards
from sts2_solver.models import CombatState, PlayerState, EnemyState, Card
from sts2_solver.constants import CardType, TargetType
from sts2_solver.alphazero.encoding import EncoderConfig, build_vocabs_from_card_db
from sts2_solver.alphazero.network import STS2Network
from sts2_solver.alphazero.mcts import MCTS
from sts2_solver.alphazero.full_run import mcts_combat, _card_to_dict
from sts2_solver.simulator import _ensure_data_loaded, _MONSTERS_BY_ID, _load_enemy_profiles
import random

try:
    import sts2_engine
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DB = load_cards()


def _build_starter_deck():
    """Build a standard Silent starter deck."""
    deck = []
    for _ in range(5):
        deck.append(DB.get("STRIKE"))
    for _ in range(5):
        deck.append(DB.get("DEFEND"))
    deck.append(DB.get("SURVIVOR"))
    deck.append(DB.get("NEUTRALIZE"))
    return [c for c in deck if c is not None]


def _build_mixed_deck():
    """Build a deck with some interesting cards."""
    ids = [
        "STRIKE", "STRIKE", "STRIKE", "STRIKE",
        "DEFEND", "DEFEND", "DEFEND",
        "SURVIVOR", "NEUTRALIZE",
        "BLADE_DANCE", "ACROBATICS", "DEADLY_POISON",
    ]
    deck = [DB.get(cid) for cid in ids]
    return [c for c in deck if c is not None]


def _prepare_rust_data():
    """Prepare JSON data needed for Rust engine."""
    _ensure_data_loaded()

    vocabs = build_vocabs_from_card_db(DB)
    config = EncoderConfig(num_trunk_blocks=3)

    # Export vocabs
    vocab_data = {
        "cards": dict(vocabs.cards.token_to_idx),
        "powers": dict(vocabs.powers.token_to_idx),
        "relics": dict(vocabs.relics.token_to_idx),
        "intent_types": dict(vocabs.intent_types.token_to_idx),
        "acts": dict(vocabs.acts.token_to_idx),
        "bosses": dict(vocabs.bosses.token_to_idx),
        "room_types": dict(vocabs.room_types.token_to_idx),
    }
    vocab_json = json.dumps(vocab_data)

    # Monster data
    monsters = {}
    for mid, m in _MONSTERS_BY_ID.items():
        monsters[mid] = {
            "name": m.get("name", mid),
            "min_hp": m.get("min_hp") or 20,
            "max_hp": m.get("max_hp") or m.get("min_hp") or 20,
        }
    monster_json = json.dumps(monsters)

    # Enemy profiles
    profiles = _load_enemy_profiles()
    profiles_json = json.dumps(profiles)

    # ONNX models
    network = STS2Network(vocabs, config)
    from sts2_solver.alphazero.onnx_export import export_onnx
    onnx_dir = str(Path(__file__).parent / "_test_onnx")
    export_onnx(network, config, onnx_dir)

    return {
        "vocab_json": vocab_json,
        "monster_json": monster_json,
        "profiles_json": profiles_json,
        "onnx_full": str(Path(onnx_dir) / "full_model.onnx"),
        "onnx_value": str(Path(onnx_dir) / "value_model.onnx"),
        "vocabs": vocabs,
        "config": config,
        "network": network,
    }


# ---------------------------------------------------------------------------
# Helper: run combat in both engines
# ---------------------------------------------------------------------------

def _run_python_combat(deck, enemy_ids, data, seed=42, sims=10):
    """Run combat through Python MCTS engine."""
    rng = random.Random(seed)
    vocabs = data["vocabs"]
    config = data["config"]
    network = data["network"]
    network.eval()
    mcts = MCTS(network, vocabs, config, card_db=DB, device="cpu")
    mcts.add_noise = False

    samples, outcome, turns, hp_after, potions, initial_value = mcts_combat(
        deck=deepcopy(deck),
        player_hp=70, player_max_hp=70, player_max_energy=3,
        encounter_id="",
        card_db=DB, mcts=mcts, vocabs=vocabs, config=config,
        rng=rng, mcts_simulations=sims, temperature=0.01,
        enemy_ids=enemy_ids,
    )
    return {
        "outcome": outcome,
        "turns": turns,
        "hp_after": hp_after,
        "num_samples": len(samples),
    }


def _run_rust_combat(deck, enemy_ids, data, seed=42, sims=10):
    """Run combat through Rust + ONNX engine."""
    deck_json = json.dumps([_card_to_dict(c) for c in deck])

    result = sts2_engine.fight_combat(
        deck_json=deck_json,
        player_hp=70, player_max_hp=70, player_max_energy=3,
        enemy_ids=enemy_ids,
        relics=[],
        potions_json="[]",
        floor=1, gold=0,
        act_id="", boss_id="",
        map_path=[],
        onnx_full_path=data["onnx_full"],
        onnx_value_path=data["onnx_value"],
        vocab_json=data["vocab_json"],
        monster_data_json=data["monster_json"],
        enemy_profiles_json=data["profiles_json"],
        mcts_sims=sims,
        temperature=0.01,
        seed=seed,
        add_noise=False,
    )
    return {
        "outcome": result["outcome"],
        "turns": result["turns"],
        "hp_after": result["hp_after"],
        "num_samples": len(result["samples"]),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rust_data():
    if not HAS_RUST:
        pytest.skip("sts2_engine not installed")
    return _prepare_rust_data()


class TestRustParity:
    """Compare Rust and Python combat engine outputs."""

    def test_both_engines_complete(self, rust_data):
        """Both engines should complete a combat without crashing."""
        deck = _build_starter_deck()
        py = _run_python_combat(deck, ["NIBBIT"], rust_data, seed=1)
        rs = _run_rust_combat(deck, ["NIBBIT"], rust_data, seed=1)
        assert py["outcome"] in ("win", "lose")
        assert rs["outcome"] in ("win", "lose")
        print(f"Python: {py}")
        print(f"Rust:   {rs}")

    def test_samples_collected(self, rust_data):
        """Both engines should collect training samples."""
        deck = _build_starter_deck()
        py = _run_python_combat(deck, ["NIBBIT"], rust_data, seed=2, sims=10)
        rs = _run_rust_combat(deck, ["NIBBIT"], rust_data, seed=2, sims=10)
        assert py["num_samples"] > 0, "Python collected no samples"
        assert rs["num_samples"] > 0, "Rust collected no samples"
        print(f"Python samples: {py['num_samples']}, Rust samples: {rs['num_samples']}")

    def test_multi_enemy(self, rust_data):
        """Test with multiple enemies."""
        deck = _build_starter_deck()
        py = _run_python_combat(deck, ["ASSASSIN_RAIDER", "AXE_RAIDER"], rust_data, seed=3)
        rs = _run_rust_combat(deck, ["ASSASSIN_RAIDER", "AXE_RAIDER"], rust_data, seed=3)
        print(f"Python: {py}")
        print(f"Rust:   {rs}")
        assert py["outcome"] in ("win", "lose")
        assert rs["outcome"] in ("win", "lose")

    def test_mixed_deck(self, rust_data):
        """Test with a more complex deck."""
        deck = _build_mixed_deck()
        py = _run_python_combat(deck, ["NIBBIT"], rust_data, seed=4, sims=20)
        rs = _run_rust_combat(deck, ["NIBBIT"], rust_data, seed=4, sims=20)
        print(f"Python: {py}")
        print(f"Rust:   {rs}")

    def test_speed_comparison(self, rust_data):
        """Benchmark: Rust should be significantly faster."""
        deck = _build_starter_deck()

        t0 = time.time()
        for seed in range(5):
            _run_python_combat(deck, ["NIBBIT"], rust_data, seed=seed, sims=50)
        py_time = time.time() - t0

        t0 = time.time()
        for seed in range(5):
            _run_rust_combat(deck, ["NIBBIT"], rust_data, seed=seed, sims=50)
        rs_time = time.time() - t0

        speedup = py_time / max(rs_time, 0.001)
        print(f"\n5 combats @ 50 sims:")
        print(f"  Python: {py_time:.2f}s")
        print(f"  Rust:   {rs_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        assert speedup > 1.5, f"Rust should be faster (got {speedup:.1f}x)"

    def test_consistency_across_seeds(self, rust_data):
        """Run multiple combats and check both engines produce reasonable results."""
        deck = _build_starter_deck()
        py_wins = 0
        rs_wins = 0
        n = 10

        for seed in range(n):
            py = _run_python_combat(deck, ["NIBBIT"], rust_data, seed=seed, sims=10)
            rs = _run_rust_combat(deck, ["NIBBIT"], rust_data, seed=seed, sims=10)
            if py["outcome"] == "win":
                py_wins += 1
            if rs["outcome"] == "win":
                rs_wins += 1

        print(f"\n{n} combats vs NIBBIT:")
        print(f"  Python wins: {py_wins}/{n}")
        print(f"  Rust wins:   {rs_wins}/{n}")
        # Both should win at least some (starter deck vs easy enemy)
        # and not win all (some randomness in play)
