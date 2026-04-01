"""Tests for the card data loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts2_solver.constants import CardType, TargetType
from sts2_solver.data_loader import load_cards


def _db():
    return load_cards()


def test_loads_all_cards():
    db = _db()
    # Should have base + upgraded variants
    assert len(db) > 500


def test_strike_ironclad():
    db = _db()
    card = db.get("STRIKE_IRONCLAD")
    assert card is not None
    assert card.name == "Strike"
    assert card.cost == 1
    assert card.damage == 6
    assert card.card_type == CardType.ATTACK
    assert card.target == TargetType.ANY_ENEMY
    assert card.upgraded is False


def test_strike_ironclad_upgraded():
    db = _db()
    card = db.get("STRIKE_IRONCLAD", upgraded=True)
    assert card is not None
    assert card.damage == 9
    assert card.upgraded is True
    assert card.cost == 1


def test_bash():
    db = _db()
    card = db.get("BASH")
    assert card.damage == 8
    assert card.cost == 2
    assert card.powers_applied == (("Vulnerable", 2),)


def test_bash_upgraded():
    db = _db()
    card = db.get("BASH", upgraded=True)
    assert card.damage == 10
    assert card.powers_applied == (("Vulnerable", 3),)


def test_blood_wall():
    db = _db()
    card = db.get("BLOOD_WALL")
    assert card.block == 16
    assert card.hp_loss == 2
    assert card.cost == 2
    assert card.card_type == CardType.SKILL


def test_defend_ironclad():
    db = _db()
    card = db.get("DEFEND_IRONCLAD")
    assert card.block == 5
    assert card.cost == 1


def test_defend_upgraded():
    db = _db()
    card = db.get("DEFEND_IRONCLAD", upgraded=True)
    assert card.block == 8


def test_card_with_no_upgrade():
    db = _db()
    # Armaments has no upgrade field
    card = db.get("ARMAMENTS")
    assert card is not None
    upgraded = db.get("ARMAMENTS", upgraded=True)
    assert upgraded is None


def test_keywords():
    db = _db()
    # Find a card with Exhaust keyword
    stoke = db.get("STOKE")
    if stoke is not None:
        assert "Exhaust" in stoke.keywords
