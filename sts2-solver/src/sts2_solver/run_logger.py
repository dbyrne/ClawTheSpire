"""Event-sourced run logger.

Captures a full state snapshot at run start, then logs only deltas and
decisions.  Output is one JSONL file per run under ``logs/``.

To reconstruct state at any point, replay events from the initial snapshot.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGS_DIR = Path(os.environ.get(
    "STS2_LOGS_DIR",
    Path(__file__).resolve().parents[3] / "logs",
))


class RunLogger:
    """Tracks a single run, emitting events to a JSONL file."""

    def __init__(self, logs_dir: Path | None = None):
        self.logs_dir = logs_dir or LOGS_DIR
        self.game_version: str | None = None
        self.metadata: dict[str, Any] = {}  # Extra fields for run_start (model, etc.)
        self._file = None
        self._path: Path | None = None
        self._run_id: str | None = None
        self._prev_state: dict | None = None
        self._turn_start_hp: int | None = None
        self._combat_start_hp: int | None = None
        self._combat_enemies: list[dict] | None = None
        self._combat_turn: int = 0
        self._last_map_node: dict | None = None  # Track map position across screens

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_run(self, game_state: dict) -> None:
        """Start tracking a run if not already, or detect a new run."""
        run = game_state.get("run") or {}
        run_id = game_state.get("run_id") or run.get("run_id")

        if run_id and run_id != self._run_id:
            self._start_run(game_state, run_id)

    def close(self) -> None:
        """Flush and close the current log file."""
        if self._file:
            self._file.close()
            self._file = None
        self._run_id = None
        self._prev_state = None

    # ------------------------------------------------------------------
    # Event emitters — called by mcp_server
    # ------------------------------------------------------------------

    def log_decision(
        self,
        game_state: dict,
        screen_type: str,
        options: Any,
        choice: dict,
        source: str,
        latency_ms: float | None = None,
        user_prompt: str | None = None,
        network_value: float | None = None,
        head_scores: dict | None = None,
    ) -> None:
        """Log a strategic or auto decision.

        If user_prompt is provided, it's included for training data extraction.
        network_value is the network's win expectancy (-1 to +1) at decision time.
        head_scores contains option_eval or deck_eval head outputs for all candidates.
        """
        self.ensure_run(game_state)
        self._emit_diffs(game_state)

        event: dict[str, Any] = {
            "type": "decision",
            "screen_type": screen_type,
            "options": options,
            "choice": choice,
            "source": source,
        }
        if latency_ms is not None:
            event["latency_ms"] = round(latency_ms, 1)
        if user_prompt is not None:
            event["user_prompt"] = user_prompt
        if network_value is not None:
            event["network_value"] = round(network_value, 4)
        if head_scores is not None:
            event["head_scores"] = head_scores

        # Include map state on map decisions so we capture the position
        # and available nodes at each navigation choice
        if screen_type == "map":
            map_data = game_state.get("map") or (game_state.get("agent_view") or {}).get("map")
            if map_data:
                event["map_node"] = map_data.get("current_node")
                event["available_nodes"] = map_data.get("available_nodes")

        self._emit(event)

    def log_combat_start(self, game_state: dict) -> None:
        """Log the beginning of a combat encounter.

        Includes the full deck list so the validator can reconstruct
        exact pile contents (accounting for cards added mid-run).
        """
        self.ensure_run(game_state)
        combat = game_state.get("combat") or {}
        enemies = combat.get("enemies") or []
        player = combat.get("player") or {}

        self._combat_start_hp = player.get("current_hp")
        self._combat_turn = 0
        self._combat_enemies = [
            {"name": e.get("name", "?"), "hp": e.get("current_hp", 0), "max_hp": e.get("max_hp", 0)}
            for e in enemies if e.get("is_alive", True)
        ]

        run = game_state.get("run") or {}
        # Enhancement #4: deck contents at combat start
        deck_cards = []
        for c in run.get("deck", []):
            if isinstance(c, dict):
                name = c.get("name") or c.get("card_id", "?")
                if c.get("upgraded"):
                    name += "+"
                deck_cards.append(name)
            elif isinstance(c, str):
                deck_cards.append(c)

        self._emit({
            "type": "combat_start",
            "floor": run.get("floor"),
            "enemies": self._combat_enemies,
            "deck": deck_cards,
        })

    def log_combat_turn(
        self,
        cards_played: list[str],
        score: float,
        states_evaluated: int,
        solve_ms: float,
        game_state: dict | None = None,
        targets_chosen: list[int | None] | None = None,
        network_value: float | None = None,
        discards: list[str] | None = None,
        hand_after: list[str] | None = None,
    ) -> None:
        """Log a single combat turn's solver output.

        If game_state is provided, a ``combat_snapshot`` event is emitted
        first, capturing the full pre-action combat state for simulator
        validation and AlphaZero training data generation.

        targets_chosen is a parallel list to cards_played: for each card,
        the enemy index it targeted (None for untargeted cards).

        network_value is the MCTS root value (win expectancy, -1 to +1)
        from the start of the turn.

        discards is a list of card names discarded during the turn
        (from Survivor, Acrobatics, etc. pending choice resolution).

        hand_after is the list of card names remaining in hand after
        all cards are played but before end of turn.
        """
        self._combat_turn += 1

        # Emit snapshot BEFORE the turn event (pre-action state)
        if game_state is not None:
            self._emit_combat_snapshot(game_state, self._combat_turn)

        event: dict[str, Any] = {
            "type": "combat_turn",
            "turn": self._combat_turn,
            "cards_played": cards_played,
            "score": round(score, 1),
            "states_evaluated": states_evaluated,
            "solve_ms": round(solve_ms, 1),
        }
        if targets_chosen is not None:
            event["targets_chosen"] = targets_chosen
        if network_value is not None:
            event["network_value"] = round(network_value, 4)
        if discards is not None:
            event["discards"] = discards
        if hand_after is not None:
            event["hand_after"] = hand_after

        self._emit(event)

    def _emit_combat_snapshot(self, game_state: dict, turn: int) -> None:
        """Emit a full combat state snapshot for replay validation.

        Captures everything needed to reconstruct the CombatState at the
        start of the player's action phase: hand, draw pile, discard pile,
        player stats, and all enemy states including intents.
        """
        combat = game_state.get("combat") or {}
        player = combat.get("player") or {}
        hand_raw = combat.get("hand") or []
        enemies_raw = combat.get("enemies") or []
        run = game_state.get("run") or {}

        # Snapshot the hand (card names + upgrade status + cost + playability)
        # Merge playability info from agent_view.combat.hand if available
        av_combat = (game_state.get("agent_view") or {}).get("combat") or {}
        av_hand = av_combat.get("hand") or []
        # Build index → agent_view card mapping
        av_hand_by_idx: dict[int, dict] = {}
        for avc in av_hand:
            if isinstance(avc, dict) and "i" in avc:
                av_hand_by_idx[avc["i"]] = avc

        hand = []
        for idx, c in enumerate(hand_raw):
            av_card = av_hand_by_idx.get(c.get("index", idx), {})
            entry: dict[str, Any] = {
                "name": c.get("name") or c.get("card_id", "?"),
                "card_id": c.get("card_id", ""),
                "cost": c.get("energy_cost", c.get("cost")),
                "upgraded": bool(c.get("upgraded")),
                "playable": c.get("playable", av_card.get("playable")),
                "targets": list(c.get("valid_target_indices") or av_card.get("targets") or []),
                "unplayable_reason": c.get("unplayable_reason") or av_card.get("why"),
            }
            hand.append(entry)

        # Snapshot enemies (HP, block, powers, intents)
        enemies = []
        for e in enemies_raw:
            if not e.get("is_alive", True):
                continue
            from .bridge import parse_intents
            intents = e.get("intents") or []
            intent_type, intent_damage, intent_hits, intent_block = parse_intents(intents)

            entry = {
                "name": e.get("name", "?"),
                "id": e.get("id") or e.get("enemy_id", ""),
                "hp": e.get("current_hp", 0),
                "max_hp": e.get("max_hp", 0),
                "block": e.get("block", 0),
                "powers": [
                    {"name": p.get("name", ""), "amount": p.get("amount", 0)}
                    for p in (e.get("powers") or [])
                    if p.get("amount", 0) != 0
                ],
                "intent_type": intent_type,
                "intent_damage": intent_damage,
                "intent_hits": intent_hits,
                "intent_block": intent_block,
            }
            enemies.append(entry)

        self._emit({
            "type": "combat_snapshot",
            "turn": turn,
            "player": {
                "hp": player.get("current_hp"),
                "max_hp": player.get("max_hp"),
                "block": player.get("block", 0),
                "energy": player.get("energy"),
                "powers": [
                    {"name": p.get("name", ""), "amount": p.get("amount", 0)}
                    for p in (player.get("powers") or [])
                    if p.get("amount", 0) != 0
                ],
            },
            "hand": hand,
            "enemies": enemies,
            "draw_pile_size": _pile_size(game_state, "draw"),
            "discard_pile_size": _pile_size(game_state, "discard"),
            "exhaust_pile_size": _pile_size(game_state, "exhaust"),
            "draw_pile": _parse_pile(game_state, "draw")[1],
            "discard_pile": _parse_pile(game_state, "discard")[1],
            "exhaust_pile": _parse_pile(game_state, "exhaust")[1],
            "relics": [r.get("name") or r.get("relic_id", "?") for r in run.get("relics", [])],
            "available_actions": list(game_state.get("available_actions") or []),
        })

    def log_combat_end(self, game_state: dict, outcome: str) -> None:
        """Log end of combat. outcome is 'win', 'defeat', or 'loss'."""
        if self._combat_start_hp is None:
            return  # No active combat to close
        run = game_state.get("run") or {}
        hp_after = run.get("current_hp") or (game_state.get("combat", {}).get("player", {}).get("current_hp"))

        self._emit({
            "type": "combat_end",
            "outcome": outcome,
            "turns": self._combat_turn,
            "hp_before": self._combat_start_hp,
            "hp_after": hp_after,
        })
        self._combat_start_hp = None
        self._combat_enemies = None
        self._combat_turn = 0

    def log_run_end(self, game_state: dict, outcome: str) -> None:
        """Log the end of a run."""
        run = game_state.get("run") or {}
        deck = run.get("deck", [])
        relics = run.get("relics", [])

        self._emit({
            "type": "run_end",
            "outcome": outcome,
            "floor": run.get("floor"),
            "final_deck": _summarize_deck_list(deck),
            "final_relics": [r.get("name") or r.get("relic_id", "?") for r in relics],
            "final_hp": run.get("current_hp"),
            "final_max_hp": run.get("max_hp"),
            "final_gold": run.get("gold"),
        })
        self.close()

    # ------------------------------------------------------------------
    # State diffing
    # ------------------------------------------------------------------

    def _emit_diffs(self, game_state: dict) -> None:
        """Compare current state to previous and emit delta events."""
        if self._prev_state is None:
            self._prev_state = game_state
            return

        prev_run = self._prev_state.get("run") or {}
        curr_run = game_state.get("run") or {}

        # HP change
        prev_hp = prev_run.get("current_hp")
        curr_hp = curr_run.get("current_hp")
        prev_max = prev_run.get("max_hp")
        curr_max = curr_run.get("max_hp")
        if (prev_hp, prev_max) != (curr_hp, curr_max) and curr_hp is not None:
            self._emit({
                "type": "hp_change",
                "hp": curr_hp,
                "max_hp": curr_max,
                "delta": (curr_hp - prev_hp) if prev_hp is not None and curr_hp is not None else None,
            })

        # Gold change
        prev_gold = prev_run.get("gold")
        curr_gold = curr_run.get("gold")
        if prev_gold != curr_gold and curr_gold is not None:
            self._emit({
                "type": "gold_change",
                "gold": curr_gold,
                "delta": (curr_gold - prev_gold) if prev_gold is not None else None,
            })

        # Deck changes
        prev_deck = _deck_counts(prev_run.get("deck", []))
        curr_deck = _deck_counts(curr_run.get("deck", []))
        if prev_deck != curr_deck:
            added = {k: curr_deck[k] - prev_deck.get(k, 0) for k in curr_deck if curr_deck[k] > prev_deck.get(k, 0)}
            removed = {k: prev_deck[k] - curr_deck.get(k, 0) for k in prev_deck if prev_deck[k] > curr_deck.get(k, 0)}
            if added or removed:
                self._emit({
                    "type": "deck_change",
                    "added": added if added else None,
                    "removed": removed if removed else None,
                    "deck_size": sum(curr_deck.values()),
                })

        # Relic changes
        prev_relics = {r.get("relic_id") or r.get("id", "") for r in prev_run.get("relics", [])}
        curr_relics_list = curr_run.get("relics", [])
        curr_relics = {r.get("relic_id") or r.get("id", "") for r in curr_relics_list}
        new_relics = curr_relics - prev_relics
        for relic_id in new_relics:
            relic = next((r for r in curr_relics_list if (r.get("relic_id") or r.get("id", "")) == relic_id), {})
            self._emit({
                "type": "relic_gained",
                "relic_id": relic_id,
                "name": relic.get("name", relic_id),
            })

        # Potion changes
        prev_potions = _potion_slots(prev_run.get("potions", []))
        curr_potions = _potion_slots(curr_run.get("potions", []))
        if prev_potions != curr_potions:
            for slot, curr_pot in curr_potions.items():
                prev_pot = prev_potions.get(slot)
                if curr_pot != prev_pot:
                    self._emit({
                        "type": "potion_change",
                        "slot": slot,
                        "potion": curr_pot,
                        "previous": prev_pot,
                    })

        # Map — log on first reveal and on every navigation change.
        # Use a persistent tracker since map data disappears from game_state
        # on non-map screens (combat, rewards, etc.)
        curr_map = game_state.get("map") or (game_state.get("agent_view") or {}).get("map")
        if curr_map:
            curr_node = curr_map.get("current_node")
            if self._last_map_node is None:
                # First time map is available
                self._emit({
                    "type": "map_revealed",
                    "map": curr_map,
                })
            elif curr_node != self._last_map_node:
                self._emit({
                    "type": "map_updated",
                    "current_node": curr_node,
                    "available_nodes": curr_map.get("available_nodes"),
                    "map": curr_map,
                })
            self._last_map_node = curr_node

        self._prev_state = game_state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_run(self, game_state: dict, run_id: str) -> None:
        """Open a log file and emit the run_start snapshot.

        If an existing log file for this run_id exists, resume appending
        to it (with a run_resume event) instead of creating a new file.
        """
        self.close()
        self._run_id = run_id

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Look for an existing log file for this run_id
        existing = sorted(self.logs_dir.glob(f"run_{run_id}_*.jsonl"))
        if existing:
            path = existing[-1]  # Use the most recent one
            self._path = path
            self._file = open(path, "a", encoding="utf-8")
            # Emit a resume marker instead of full run_start
            run = game_state.get("run") or {}
            self._emit({
                "type": "run_resume",
                "run_id": run_id,
                "floor": run.get("floor"),
                "hp": run.get("current_hp"),
                "max_hp": run.get("max_hp"),
            })
            # If resuming mid-combat, restore combat tracking state
            combat = game_state.get("combat") or {}
            player = combat.get("player") or {}
            if game_state.get("in_combat") or player.get("current_hp") is not None:
                self._combat_start_hp = player.get("current_hp") or run.get("current_hp")
                self._combat_turn = 0
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self.logs_dir / f"run_{run_id}_{ts}.jsonl"
            self._path = path
            self._file = open(path, "a", encoding="utf-8")

            run = game_state.get("run") or {}
            deck = run.get("deck", [])
            relics = run.get("relics", [])
            potions = run.get("potions", [])

            event: dict[str, Any] = {
                "type": "run_start",
                "run_id": run_id,
                "game_version": self.game_version,
                **self.metadata,
                "character": run.get("character_name") or run.get("character_id"),
                "floor": run.get("floor"),
                "hp": run.get("current_hp"),
                "max_hp": run.get("max_hp"),
                "gold": run.get("gold"),
                "max_energy": run.get("max_energy"),
                "deck": _summarize_deck_list(deck),
                "relics": [r.get("name") or r.get("relic_id", "?") for r in relics],
                "potions": [
                    p.get("name") if p.get("occupied") else None
                    for p in potions
                ],
                "map": game_state.get("map"),
            }
            self._emit(event)

        self._prev_state = game_state

    def _emit(self, event: dict) -> None:
        """Write a single event line to the JSONL file."""
        if self._file is None:
            return
        event["ts"] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
        self._file.flush()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_pile(game_state: dict, pile: str) -> tuple[int, list[str]]:
    """Get card pile size and card names from agent_view or combat dict.

    The C# mod exposes piles as grouped card stacks. Entries can be:
    - dict with "line" field: "2x Strike (1) — Deal 6 damage."
    - string: "Strike*2 [1⚡]—Deal 6 damage." or "Defend [1⚡]—Gain 5 Block."

    The *N suffix means N copies of that card in the pile.

    Returns (total_count, card_names_list) with names expanded per copy.
    """
    import re

    av = game_state.get("agent_view") or {}
    candidates = [
        (av.get("combat") or {}).get(pile),
        ((av.get("run") or {}).get("piles") or {}).get(pile),
    ]
    for pile_data in candidates:
        if pile_data and isinstance(pile_data, list):
            total = 0
            names: list[str] = []
            for entry in pile_data:
                if isinstance(entry, dict):
                    line = entry.get("line", "")
                elif isinstance(entry, str):
                    line = entry
                else:
                    total += 1
                    names.append(str(entry))
                    continue

                # Parse formats:
                #   "CardName*N [cost...]—desc"  (N copies)
                #   "CardName [cost...]—desc"    (1 copy)
                #   "Nx CardName (cost) — desc"  (old format)
                name = None
                count = 1

                # Try *N suffix: "Strike*2 [1⚡]—Deal 6 damage."
                m = re.match(r"(.+?)\*(\d+)\s*[\[（]", line)
                if m:
                    name = m.group(1).strip()
                    count = int(m.group(2))
                else:
                    # Try no multiplier: "Defend [1⚡]—Gain 5 Block."
                    m = re.match(r"(.+?)\s*[\[（]", line)
                    if m:
                        name = m.group(1).strip()
                    else:
                        # Try Nx prefix: "2x Strike (1) — desc"
                        m = re.match(r"(?:(\d+)x\s)?(.+?)(?:\s*\(\d+\))?\s*[—–-]", line)
                        if m:
                            count = int(m.group(1)) if m.group(1) else 1
                            name = m.group(2).strip()

                if name:
                    total += count
                    names.extend([name] * count)
                else:
                    total += 1
                    names.append(line.split("[")[0].split("(")[0].strip())
            return total, names

    # Fallback: combat dict might have {pile}_pile as a list
    combat = game_state.get("combat") or {}
    fallback = combat.get(f"{pile}_pile") or []
    return len(fallback), [c.get("name", "?") if isinstance(c, dict) else str(c) for c in fallback]


def _pile_size(game_state: dict, pile: str) -> int:
    """Get card pile size. Convenience wrapper around _parse_pile."""
    size, _ = _parse_pile(game_state, pile)
    return size


def _card_display_name(card: dict) -> str:
    """Get display name (with + for upgraded) from a card dict."""
    name = card.get("name") or card.get("card_id", "?")
    if card.get("upgraded"):
        name += "+"
    return name


def _deck_counts(deck: list[dict]) -> dict[str, int]:
    """Build {card_name: count} from a deck list."""
    counts: dict[str, int] = {}
    for card in deck:
        name = _card_display_name(card)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _potion_slots(potions: list[dict]) -> dict[int, str | None]:
    """Build {slot_index: potion_name_or_none}."""
    result: dict[int, str | None] = {}
    for i, p in enumerate(potions):
        idx = p.get("index", i)
        result[idx] = p.get("name") if p.get("occupied") else None
    return result


def _summarize_deck_list(deck: list[dict]) -> list[str]:
    """Return a compact list of card names (with + for upgraded)."""
    return sorted(_card_display_name(card) for card in deck)
