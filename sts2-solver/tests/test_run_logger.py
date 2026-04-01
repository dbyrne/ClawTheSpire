"""Tests for the event-sourced run logger."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sts2_solver.run_logger import RunLogger, _deck_counts, _potion_slots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_run_state(run_id="TEST_RUN_1", floor=1, hp=80, gold=99, **overrides):
    state = {
        "run_id": run_id,
        "screen": "MAP",
        "available_actions": ["choose_map_node"],
        "run": {
            "character_name": "The Ironclad",
            "character_id": "IRONCLAD",
            "floor": floor,
            "current_hp": hp,
            "max_hp": 80,
            "gold": gold,
            "max_energy": 3,
            "deck": [
                {"name": "Strike", "card_id": "STRIKE_IRONCLAD"},
                {"name": "Strike", "card_id": "STRIKE_IRONCLAD"},
                {"name": "Strike", "card_id": "STRIKE_IRONCLAD"},
                {"name": "Defend", "card_id": "DEFEND_IRONCLAD"},
                {"name": "Defend", "card_id": "DEFEND_IRONCLAD"},
                {"name": "Bash", "card_id": "BASH"},
            ],
            "relics": [{"relic_id": "BURNING_BLOOD", "name": "Burning Blood"}],
            "potions": [
                {"index": 0, "occupied": False},
                {"index": 1, "occupied": False},
            ],
        },
        "map": None,
        "combat": None,
    }
    state.update(overrides)
    return state


def _read_events(logs_dir: Path) -> list[dict]:
    """Read all events from all JSONL files in the dir."""
    events = []
    for path in sorted(logs_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_run_start_creates_file(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state = _base_run_state()
        logger.ensure_run(state)
        logger.close()

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        assert "TEST_RUN_1" in files[0].name

    def test_run_start_event(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state = _base_run_state()
        logger.ensure_run(state)
        logger.close()

        events = _read_events(tmp_path)
        assert len(events) == 1
        assert events[0]["type"] == "run_start"
        assert events[0]["run_id"] == "TEST_RUN_1"
        assert events[0]["character"] == "The Ironclad"
        assert "Bash" in events[0]["deck"]

    def test_run_start_includes_game_version(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.game_version = "0.7.0"
        state = _base_run_state()
        logger.ensure_run(state)
        logger.close()

        events = _read_events(tmp_path)
        assert events[0]["game_version"] == "0.7.0"

    def test_run_start_game_version_null_when_unknown(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state = _base_run_state()
        logger.ensure_run(state)
        logger.close()

        events = _read_events(tmp_path)
        assert events[0]["game_version"] is None

    def test_new_run_id_closes_old_and_starts_new(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.ensure_run(_base_run_state(run_id="RUN_A"))
        logger.ensure_run(_base_run_state(run_id="RUN_B"))
        logger.close()

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 2


class TestStateDiffs:
    def test_hp_change(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.ensure_run(_base_run_state(hp=80))
        # Simulate taking damage
        logger._emit_diffs(_base_run_state(hp=65))
        logger.close()

        events = _read_events(tmp_path)
        hp_events = [e for e in events if e["type"] == "hp_change"]
        assert len(hp_events) == 1
        assert hp_events[0]["hp"] == 65
        assert hp_events[0]["delta"] == -15

    def test_gold_change(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.ensure_run(_base_run_state(gold=99))
        logger._emit_diffs(_base_run_state(gold=179))
        logger.close()

        events = _read_events(tmp_path)
        gold_events = [e for e in events if e["type"] == "gold_change"]
        assert len(gold_events) == 1
        assert gold_events[0]["gold"] == 179
        assert gold_events[0]["delta"] == 80

    def test_deck_change_card_added(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state1 = _base_run_state()
        logger.ensure_run(state1)

        state2 = _base_run_state()
        state2["run"]["deck"].append({"name": "Inflame", "card_id": "INFLAME"})
        logger._emit_diffs(state2)
        logger.close()

        events = _read_events(tmp_path)
        deck_events = [e for e in events if e["type"] == "deck_change"]
        assert len(deck_events) == 1
        assert "Inflame" in deck_events[0]["added"]

    def test_deck_change_card_removed(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state1 = _base_run_state()
        logger.ensure_run(state1)

        state2 = _base_run_state()
        state2["run"]["deck"] = [d for d in state2["run"]["deck"] if d["name"] != "Strike"]
        logger._emit_diffs(state2)
        logger.close()

        events = _read_events(tmp_path)
        deck_events = [e for e in events if e["type"] == "deck_change"]
        assert len(deck_events) == 1
        assert "Strike" in deck_events[0]["removed"]

    def test_relic_gained(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.ensure_run(_base_run_state())

        state2 = _base_run_state()
        state2["run"]["relics"].append({"relic_id": "VAJRA", "name": "Vajra"})
        logger._emit_diffs(state2)
        logger.close()

        events = _read_events(tmp_path)
        relic_events = [e for e in events if e["type"] == "relic_gained"]
        assert len(relic_events) == 1
        assert relic_events[0]["relic_id"] == "VAJRA"

    def test_potion_change(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.ensure_run(_base_run_state())

        state2 = _base_run_state()
        state2["run"]["potions"][0] = {"index": 0, "occupied": True, "name": "Fire Potion"}
        logger._emit_diffs(state2)
        logger.close()

        events = _read_events(tmp_path)
        potion_events = [e for e in events if e["type"] == "potion_change"]
        assert len(potion_events) == 1
        assert potion_events[0]["potion"] == "Fire Potion"
        assert potion_events[0]["previous"] is None

    def test_map_revealed(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        logger.ensure_run(_base_run_state())

        state2 = _base_run_state()
        state2["map"] = {"nodes": [{"type": "monster"}, {"type": "elite"}]}
        logger._emit_diffs(state2)
        logger.close()

        events = _read_events(tmp_path)
        map_events = [e for e in events if e["type"] == "map_revealed"]
        assert len(map_events) == 1
        assert len(map_events[0]["map"]["nodes"]) == 2

    def test_no_change_no_events(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state = _base_run_state()
        logger.ensure_run(state)
        logger._emit_diffs(state)
        logger.close()

        events = _read_events(tmp_path)
        # Only run_start, no diffs
        assert len(events) == 1
        assert events[0]["type"] == "run_start"


class TestCombatLogging:
    def test_combat_start_and_end(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        combat_state = _base_run_state()
        combat_state["combat"] = {
            "player": {"current_hp": 80},
            "enemies": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "is_alive": True},
            ],
        }
        logger.ensure_run(combat_state)
        logger.log_combat_start(combat_state)
        logger.log_combat_turn(
            cards_played=["Strike", "Strike", "Defend"],
            score=45.0,
            states_evaluated=30,
            solve_ms=2.5,
        )
        logger.log_combat_end(combat_state, "win")
        logger.close()

        events = _read_events(tmp_path)
        types = [e["type"] for e in events]
        assert "combat_start" in types
        assert "combat_turn" in types
        assert "combat_end" in types

        start = next(e for e in events if e["type"] == "combat_start")
        assert start["enemies"][0]["name"] == "Jaw Worm"

        turn = next(e for e in events if e["type"] == "combat_turn")
        assert turn["cards_played"] == ["Strike", "Strike", "Defend"]
        assert turn["solve_ms"] == 2.5

        end = next(e for e in events if e["type"] == "combat_end")
        assert end["outcome"] == "win"


class TestDecisionLogging:
    def test_log_decision(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state = _base_run_state()
        logger.ensure_run(state)
        logger.log_decision(
            game_state=state,
            screen_type="card_reward",
            options=["choose_reward_card", "skip_reward_cards"],
            choice={"action": "choose_reward_card", "option_index": 1},
            source="advisor",
            latency_ms=350.0,
        )
        logger.close()

        events = _read_events(tmp_path)
        decisions = [e for e in events if e["type"] == "decision"]
        assert len(decisions) == 1
        assert decisions[0]["screen_type"] == "card_reward"
        assert decisions[0]["source"] == "advisor"
        assert decisions[0]["latency_ms"] == 350.0


class TestRunEnd:
    def test_run_end_captures_final_state(self, tmp_path):
        logger = RunLogger(logs_dir=tmp_path)
        state = _base_run_state(floor=50, hp=1, gold=500)
        logger.ensure_run(state)
        logger.log_run_end(state, "victory")

        events = _read_events(tmp_path)
        end = next(e for e in events if e["type"] == "run_end")
        assert end["outcome"] == "victory"
        assert end["floor"] == 50
        assert end["final_hp"] == 1


class TestHelpers:
    def test_deck_counts(self):
        deck = [
            {"name": "Strike"},
            {"name": "Strike"},
            {"name": "Defend"},
        ]
        counts = _deck_counts(deck)
        assert counts == {"Strike": 2, "Defend": 1}

    def test_deck_counts_upgraded(self):
        deck = [{"name": "Strike", "upgraded": True}]
        counts = _deck_counts(deck)
        assert counts == {"Strike+": 1}

    def test_potion_slots(self):
        potions = [
            {"index": 0, "occupied": True, "name": "Fire Potion"},
            {"index": 1, "occupied": False},
        ]
        slots = _potion_slots(potions)
        assert slots == {0: "Fire Potion", 1: None}
