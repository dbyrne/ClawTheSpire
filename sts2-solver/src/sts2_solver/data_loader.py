"""Load card and power data from STS2-Agent JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import CardType, TargetType
from .models import Card


# Default path to STS2-Agent data directory
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "STS2-Agent" / "mcp_server" / "data" / "eng"


def _parse_target(target: str | None) -> TargetType:
    if target is None:
        return TargetType.SELF
    try:
        return TargetType(target)
    except ValueError:
        return TargetType.SELF


def _parse_card_type(card_type: str | None) -> CardType:
    if card_type is None:
        return CardType.STATUS
    try:
        return CardType(card_type)
    except ValueError:
        return CardType.STATUS


def _parse_powers_applied(raw: list[dict] | None) -> tuple[tuple[str, int], ...]:
    if not raw:
        return ()
    return tuple((p["power"], p["amount"]) for p in raw)


def _parse_keywords(raw: list[str] | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(raw)


def _parse_tags(raw: list[str] | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(raw)


def _parse_spawns(raw: list[dict] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    # spawns_cards can be a list of card IDs or objects
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and "card_id" in item:
            result.append(item["card_id"])
        elif isinstance(item, dict) and "id" in item:
            result.append(item["id"])
    return tuple(result)


def _card_from_json(raw: dict) -> Card:
    """Parse a single card JSON object into a Card dataclass."""
    cost = raw.get("cost")
    keywords = raw.get("keywords") or []
    is_unplayable = any(
        k.lower() == "unplayable" if isinstance(k, str) else False
        for k in keywords
    )
    if is_unplayable or cost == -1:
        cost = -1  # Unplayable cards keep cost -1 (blocked by can_play_card)
    elif cost is None:
        cost = 0  # X-cost cards default to 0 base cost

    return Card(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        cost=cost,
        card_type=_parse_card_type(raw.get("type")),
        target=_parse_target(raw.get("target")),
        upgraded=False,
        damage=raw.get("damage"),
        block=raw.get("block"),
        hit_count=raw.get("hit_count") or 1,
        powers_applied=_parse_powers_applied(raw.get("powers_applied")),
        cards_draw=raw.get("cards_draw") or 0,
        energy_gain=raw.get("energy_gain") or 0,
        hp_loss=raw.get("hp_loss") or 0,
        keywords=_parse_keywords(raw.get("keywords")),
        tags=_parse_tags(raw.get("tags")),
        spawns_cards=_parse_spawns(raw.get("spawns_cards")),
        is_x_cost=bool(raw.get("is_x_cost")),
    )


def _parse_upgrade_delta(value: str | int | bool) -> int | bool:
    """Parse an upgrade value like '+2' into an integer delta, or bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("+"):
        return int(value[1:])
    if isinstance(value, str) and value.startswith("-"):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _make_upgraded(base: Card, upgrade: dict) -> Card:
    """Create an upgraded variant of a card by applying upgrade deltas."""
    damage = base.damage
    block = base.block
    cost = base.cost
    hit_count = base.hit_count
    powers_applied = list(base.powers_applied)
    cards_draw = base.cards_draw
    energy_gain = base.energy_gain
    hp_loss = base.hp_loss
    keywords = set(base.keywords)

    for key, value in upgrade.items():
        delta = _parse_upgrade_delta(value)
        key_lower = key.lower()

        if key_lower == "damage" and isinstance(delta, int) and damage is not None:
            damage += delta
        elif key_lower == "block" and isinstance(delta, int) and block is not None:
            block += delta
        elif key_lower == "cost" and isinstance(delta, int):
            cost = delta  # cost upgrades are absolute, not delta
        elif key_lower == "cards" and isinstance(delta, int):
            cards_draw += delta
        elif key_lower == "energy" and isinstance(delta, int):
            energy_gain += delta
        elif key_lower == "repeat" and isinstance(delta, int):
            hit_count += delta
        elif key_lower == "add_innate" and delta is True:
            keywords.add("Innate")
        elif key_lower == "remove_exhaust" and delta is True:
            keywords.discard("Exhaust")
        elif key_lower == "vulnerable" and isinstance(delta, int):
            powers_applied = _upgrade_power_amount(powers_applied, "Vulnerable", delta)
        elif key_lower in ("strength", "strengthpower") and isinstance(delta, int):
            powers_applied = _upgrade_power_amount(powers_applied, "Strength", delta)
        elif key_lower == "power" and isinstance(delta, int):
            # Generic: bump the first power's amount
            if powers_applied:
                name, amt = powers_applied[0]
                powers_applied[0] = (name, amt + delta)
        # Other upgrade keys (extradamage, calculationbase, etc.) are handled
        # by custom card effects that read the upgraded flag or card vars.

    return Card(
        id=base.id,
        name=base.name,
        cost=cost,
        card_type=base.card_type,
        target=base.target,
        upgraded=True,
        damage=damage,
        block=block,
        hit_count=hit_count,
        powers_applied=tuple(powers_applied),
        cards_draw=cards_draw,
        energy_gain=energy_gain,
        hp_loss=hp_loss,
        keywords=frozenset(keywords),
        tags=base.tags,
        spawns_cards=base.spawns_cards,
        is_x_cost=base.is_x_cost,
    )


def _upgrade_power_amount(
    powers: list[tuple[str, int]], power_name: str, delta: int
) -> list[tuple[str, int]]:
    """Find a power by name and increase its amount."""
    result = []
    found = False
    for name, amount in powers:
        if name == power_name and not found:
            result.append((name, amount + delta))
            found = True
        else:
            result.append((name, amount))
    return result


class CardDB:
    """Card database indexed by ID. Holds both base and upgraded variants."""

    def __init__(self, cards: dict[str, Card]):
        self._cards = cards

    def get(self, card_id: str, upgraded: bool = False) -> Card | None:
        key = f"{card_id}+" if upgraded else card_id
        return self._cards.get(key)

    def get_base(self, card_id: str) -> Card | None:
        return self._cards.get(card_id)

    def get_upgraded(self, card_id: str) -> Card | None:
        return self._cards.get(f"{card_id}+")

    def all_cards(self) -> list[Card]:
        return list(self._cards.values())

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card_id: str) -> bool:
        return card_id in self._cards


def load_cards(data_dir: Path | None = None) -> CardDB:
    """Load all cards from cards.json and return a CardDB."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    cards_path = data_dir / "cards.json"
    with open(cards_path, encoding="utf-8") as f:
        raw_cards = json.load(f)

    cards: dict[str, Card] = {}
    for raw in raw_cards:
        card = _card_from_json(raw)
        cards[card.id] = card

        # Generate upgraded variant if upgrade data exists
        upgrade = raw.get("upgrade")
        if upgrade:
            upgraded = _make_upgraded(card, upgrade)
            cards[f"{card.id}+"] = upgraded

    return CardDB(cards)


def load_powers(data_dir: Path | None = None) -> dict[str, dict]:
    """Load power definitions from powers.json.

    Returns a dict of {power_id: power_data} for reference.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    powers_path = data_dir / "powers.json"
    with open(powers_path, encoding="utf-8") as f:
        raw_powers = json.load(f)

    return {p["id"]: p for p in raw_powers}
