"""Bridge between MCP/HTTP game state and the combat simulator.

Converts raw game state JSON (from the STS2 Agent mod HTTP API) into
the simulator's CombatState, and converts solver Actions back into
MCP-compatible action parameters.
"""

from __future__ import annotations

from .actions import Action
from .data_loader import CardDB
from .models import Card, CombatState, EnemyState, PlayerState


def state_from_mcp(raw: dict, card_db: CardDB) -> CombatState:
    """Convert an MCP game state dict into a CombatState.

    Args:
        raw: The full game state dict from GET /state or get_game_state().
        card_db: Card database for resolving card definitions.

    Returns:
        A CombatState ready for the solver.
    """
    combat = raw.get("combat") or {}
    player_raw = combat.get("player") or {}
    enemies_raw = combat.get("enemies") or []
    hand_raw = combat.get("hand") or []

    # Build player state
    player = PlayerState(
        hp=player_raw.get("current_hp", 0),
        max_hp=player_raw.get("max_hp", 0),
        block=player_raw.get("block", 0),
        energy=player_raw.get("energy", 0),
        max_energy=raw.get("run", {}).get("max_energy", 3),
        powers=_parse_powers(player_raw.get("powers", [])),
    )

    # Build hand from runtime card data (has computed values)
    player.hand = [_card_from_runtime(c, card_db) for c in hand_raw]

    # Build draw/discard piles from run data if available
    run = raw.get("run") or {}
    # Note: runtime pile data may not have full card objects, just summaries.
    # For single-turn solving, we mainly need the hand. Piles matter for
    # draw effects and multi-turn lookahead.
    # The raw state has draw/discard as card objects in combat view sometimes.

    # Build enemies
    enemies = [_enemy_from_runtime(e) for e in enemies_raw]

    return CombatState(
        player=player,
        enemies=enemies,
        turn=raw.get("turn", 1),
    )


def action_to_mcp(action: Action) -> dict:
    """Convert a solver Action to MCP act() parameters.

    Returns:
        Dict with 'action', 'card_index', and optionally 'target_index'.
    """
    if action.action_type == "end_turn":
        return {"action": "end_turn"}

    result = {
        "action": "play_card",
        "card_index": action.card_idx,
    }
    if action.target_idx is not None:
        result["target_index"] = action.target_idx
    return result


def actions_to_mcp_sequence(actions: list[Action]) -> list[dict]:
    """Convert a list of solver Actions to MCP action dicts."""
    return [action_to_mcp(a) for a in actions]


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_powers(powers_raw: list[dict]) -> dict[str, int]:
    """Parse runtime powers list into {power_name: amount} dict."""
    result: dict[str, int] = {}
    for p in powers_raw:
        # Runtime format: {"power_id": "VULNERABLE_POWER", "name": "Vulnerable", "amount": 2}
        name = p.get("name", "")
        amount = p.get("amount", 0)
        if name and amount != 0:
            result[name] = amount
    return result


def _card_from_runtime(raw: dict, card_db: CardDB) -> Card:
    """Build a Card from runtime combat hand data.

    Prefers the card_db definition but uses runtime dynamic_values
    for the most accurate current values (accounting for relics, powers, etc.).
    Runtime values override base values for damage/block calculations.
    """
    card_id = raw.get("card_id", "")
    upgraded = raw.get("upgraded", False)

    # Try to get from card_db
    card = card_db.get(card_id, upgraded=upgraded)

    if card is not None:
        # Override with runtime dynamic values if present
        dynamic = {dv["name"]: dv["base_value"] for dv in raw.get("dynamic_values", [])}
        damage = dynamic.get("Damage", card.damage)
        block = dynamic.get("Block", card.block)

        # Return a copy with runtime values if they differ
        if damage != card.damage or block != card.block:
            return Card(
                id=card.id,
                name=card.name,
                cost=raw.get("energy_cost", card.cost),
                card_type=card.card_type,
                target=card.target,
                upgraded=upgraded,
                damage=damage,
                block=block,
                hit_count=card.hit_count,
                powers_applied=card.powers_applied,
                cards_draw=card.cards_draw,
                energy_gain=card.energy_gain,
                hp_loss=card.hp_loss,
                keywords=card.keywords,
                tags=card.tags,
                spawns_cards=card.spawns_cards,
                is_x_cost=card.is_x_cost,
            )
        return card

    # Card not in DB - build a minimal Card from runtime data
    from .constants import CardType, TargetType

    dynamic = {dv["name"]: dv["base_value"] for dv in raw.get("dynamic_values", [])}

    target_str = raw.get("target_type", "Self")
    try:
        target = TargetType(target_str)
    except ValueError:
        target = TargetType.SELF

    card_type_str = raw.get("card_type", "Skill")
    try:
        card_type = CardType(card_type_str)
    except ValueError:
        card_type = CardType.SKILL

    return Card(
        id=card_id,
        name=raw.get("name", card_id),
        cost=raw.get("energy_cost", 0),
        card_type=card_type,
        target=target,
        upgraded=upgraded,
        damage=dynamic.get("Damage"),
        block=dynamic.get("Block"),
    )


def _enemy_from_runtime(raw: dict) -> EnemyState:
    """Build an EnemyState from runtime combat enemy data."""
    # Parse intents
    intents = raw.get("intents", [])
    intent_type = None
    intent_damage = None
    intent_hits = 1
    intent_block = None

    for intent in intents:
        it = intent.get("intent_type", "")
        if it == "Attack":
            intent_type = "Attack"
            intent_damage = intent.get("damage")
            intent_hits = intent.get("hits", 1)
        elif it == "Defend":
            intent_type = intent_type or "Defend"
            intent_block = intent.get("block")
        elif it in ("Buff", "Debuff", "StatusCard"):
            intent_type = intent_type or it

    return EnemyState(
        id=raw.get("enemy_id", ""),
        name=raw.get("name", ""),
        hp=raw.get("current_hp", 0),
        max_hp=raw.get("max_hp", 0),
        block=raw.get("block", 0),
        powers=_parse_powers(raw.get("powers", [])),
        intent_type=intent_type,
        intent_damage=intent_damage,
        intent_hits=intent_hits,
        intent_block=intent_block,
    )
