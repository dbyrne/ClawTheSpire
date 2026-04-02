"""Load game data (relics, events, potions, cards) for the strategic advisor."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .data_loader import DEFAULT_DATA_DIR


_MARKUP_RE = re.compile(r"\[/?(?:gold|green|blue|red|white|gray|grey)\]")


def strip_markup(text: str) -> str:
    """Remove color markup tags like [gold], [/green], etc."""
    return _MARKUP_RE.sub("", text)


class GameDataDB:
    """Indexed game data for relics, events, potions, and cards."""

    def __init__(
        self,
        cards_raw: dict[str, dict],
        relics: dict[str, dict],
        events: dict[str, dict],
        potions: dict[str, dict],
    ):
        self.cards_raw = cards_raw
        self.relics = relics
        self.events = events
        self.potions = potions

    def card_description(self, card_id: str) -> str:
        """Compact card description for prompts."""
        card = self.cards_raw.get(card_id.upper())
        if not card:
            return card_id
        name = card.get("name", card_id)
        cost = card.get("cost", "?")
        ctype = card.get("type", "")
        desc = strip_markup(card.get("description", "")).replace("\n", " ")
        rarity = card.get("rarity", "")
        return f"{name} ({ctype}, {rarity}, {cost} energy): {desc}"

    def relic_description(self, relic_id: str) -> str:
        """Compact relic description for prompts."""
        relic = self.relics.get(relic_id.upper())
        if not relic:
            return relic_id
        name = relic.get("name", relic_id)
        desc = strip_markup(relic.get("description", ""))
        rarity = relic.get("rarity", "")
        return f"{name} ({rarity}): {desc}"

    def event_info(self, event_id: str) -> dict | None:
        """Get event data by ID."""
        return self.events.get(event_id.upper())

    def potion_description(self, potion_id: str) -> str:
        """Compact potion description for prompts."""
        potion = self.potions.get(potion_id.upper())
        if not potion:
            return potion_id
        name = potion.get("name", potion_id)
        desc = strip_markup(potion.get("description", ""))
        return f"{name}: {desc}"


def _load_json_index(path: Path) -> dict[str, dict]:
    """Load a JSON array and index by uppercase ID."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"].upper(): item for item in items if "id" in item}


def load_game_data(data_dir: Path | None = None) -> GameDataDB:
    """Load all game data files into an indexed database."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    return GameDataDB(
        cards_raw=_load_json_index(data_dir / "cards.json"),
        relics=_load_json_index(data_dir / "relics.json"),
        events=_load_json_index(data_dir / "events.json"),
        potions=_load_json_index(data_dir / "potions.json"),
    )
