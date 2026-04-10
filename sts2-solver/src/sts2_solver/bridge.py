"""Bridge between MCP/HTTP game state and the combat simulator.

Converts raw game state JSON (from the STS2 Agent mod HTTP API) into
the simulator's CombatState, and converts solver Actions back into
MCP-compatible action parameters.
"""

from __future__ import annotations

import random

from .actions import Action
from .data_loader import CardDB
from .enemy_predict import annotate_predictions
from .models import Card, CombatState, EnemyState, PlayerState


def state_from_mcp(raw: dict, card_db: CardDB,
                   move_indices: dict[tuple[int, str], int] | None = None) -> CombatState:
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

    # Build draw/discard/exhaust piles from structured data if available,
    # falling back to display-string parsing for older mod versions.
    av_combat = (raw.get("agent_view") or {}).get("combat") or {}
    for struct_key, attr in [("draw_cards", "draw_pile"),
                             ("discard_cards", "discard_pile"),
                             ("exhaust_cards", "exhaust_pile")]:
        struct_cards = av_combat.get(struct_key) or []
        if struct_cards:
            # Structured format: [{card_id, upgraded, card_type}]
            pile_cards: list[Card] = []
            for sc in struct_cards:
                card = card_db.get(sc["card_id"], upgraded=sc.get("upgraded", False))
                if card is not None:
                    pile_cards.append(card)
            if attr == "draw_pile":
                random.shuffle(pile_cards)
            setattr(player, attr, pile_cards)
        else:
            # Fallback: parse display strings (pre-fork mod)
            from .run_logger import _parse_pile
            pile_name = struct_key.replace("_cards", "")
            _, card_names = _parse_pile(raw, pile_name)
            pile_cards = []
            for name in card_names:
                base = name.rstrip("+")
                upgraded = name.endswith("+")
                card = card_db.get_by_name(base, upgraded=upgraded)
                if card is None:
                    card = card_db.get_by_name(base)
                if card is not None:
                    pile_cards.append(card)
            if attr == "draw_pile":
                random.shuffle(pile_cards)
            setattr(player, attr, pile_cards)

    # Build enemies
    enemies = [_enemy_from_runtime(e) for e in enemies_raw]

    # Extract relic IDs for evaluator awareness
    run = raw.get("run") or {}
    relics_raw = run.get("relics") or raw.get("relics") or []
    relic_ids = frozenset(
        r.get("relic_id", r.get("id", "")) if isinstance(r, dict) else str(r)
        for r in relics_raw
    )

    floor = run.get("floor", 0)

    # Predict future enemy intents from move tables
    annotate_predictions(enemies, turns=2, move_indices=move_indices)

    # Parse potions from run state.
    # Uses potion_id from the enriched mod API when available,
    # falling back to keyword-based classification for older mods.
    potions_raw = run.get("potions") or []
    potions = []
    for p in potions_raw:
        if not p.get("occupied"):
            potions.append({})
            continue
        pot: dict = {"name": p.get("name", "?")}
        potion_id = p.get("potion_id") or ""
        classified = _classify_potion(potion_id, p.get("name", ""))
        if classified:
            pot.update(classified)
            potions.append(pot)
        else:
            potions.append({})
    player.potions = potions

    gold = run.get("gold", 0)

    state = CombatState(
        player=player,
        enemies=enemies,
        turn=raw.get("turn", 1),
        relics=relic_ids,
        floor=floor,
        gold=gold,
    )

    # Set mid-turn counters from enriched mod API (Phase 1A).
    # These are maintained by the mod's GameActionService and exposed
    # on combat.player.  When present, the runner no longer needs to
    # reconstruct them from logged card names.
    cards_played = player_raw.get("cards_played_this_turn")
    if cards_played is not None:
        state.cards_played_this_turn = cards_played
        state.attacks_played_this_turn = player_raw.get("attacks_played_this_turn", 0)
        skills = player_raw.get("skills_played_this_turn", 0)
        if skills > 0:
            state.player.powers["_skills_played"] = skills

    _validate_bridge_state(state, raw)

    return state


# ---------------------------------------------------------------------------
# Bridge invariant checks
# ---------------------------------------------------------------------------

# Powers the simulator always stores as positive (debuff magnitude).
# If the bridge produces a negative value for any of these, the network
# will see an inverted feature vs training.
_POSITIVE_SIGN_POWERS = {"Shrink", "Weak", "Vulnerable", "Frail"}

# Known potion effect keys the simulator and encoding understand.
_KNOWN_POTION_EFFECTS = {"heal", "block", "strength", "damage_all", "enemy_weak"}

import logging as _logging
_bridge_log = _logging.getLogger("sts2_solver.bridge")


def _validate_bridge_state(state: "CombatState", raw: dict) -> None:
    """Check bridge output for divergences from simulator conventions.

    These are cheap assertions that catch the class of bugs where
    the game API represents data differently than the simulator.
    Warnings are logged, not raised, so the runner keeps going.
    """
    # 1. Power sign convention: debuffs should be positive in our model
    for name in _POSITIVE_SIGN_POWERS:
        val = state.player.powers.get(name, 0)
        if val < 0:
            _bridge_log.warning(
                "BRIDGE_DIVERGENCE: player power %s=%d (expected positive). "
                "Network sees inverted feature vs training.", name, val)

    # 2. Potion classification: occupied slots should have known effects
    run = raw.get("run") or {}
    potions_raw = run.get("potions") or []
    for i, (pot_raw, pot_state) in enumerate(
        zip(potions_raw, state.player.potions)
    ):
        if not pot_raw.get("occupied"):
            continue
        if not pot_state or not any(k in pot_state for k in _KNOWN_POTION_EFFECTS):
            name = pot_raw.get("name", "?")
            pid = pot_raw.get("potion_id", "?")
            _bridge_log.warning(
                "BRIDGE_DIVERGENCE: potion slot %d (%s / %s) has no "
                "classified effect — network sees empty slot.", i, name, pid)

    # 3. Hand cards: every card should resolve from card_db (not fallback)
    for card in state.player.hand:
        if card.id == "" or card.id.startswith("UNKNOWN"):
            _bridge_log.warning(
                "BRIDGE_DIVERGENCE: hand card %r not found in card_db. "
                "Network sees fallback stats.", card.name)

    # 4. Enemy powers: check for unexpected negative values
    for j, enemy in enumerate(state.enemies):
        for name in _POSITIVE_SIGN_POWERS:
            val = enemy.powers.get(name, 0)
            if val < 0:
                _bridge_log.warning(
                    "BRIDGE_DIVERGENCE: enemy %d (%s) power %s=%d "
                    "(expected positive).", j, enemy.name, name, val)


def action_to_mcp(action: Action) -> dict:
    """Convert a solver Action to MCP act() parameters.

    Returns:
        Dict with 'action' and relevant indices.
    """
    if action.action_type == "end_turn":
        return {"action": "end_turn"}

    if action.action_type == "use_potion":
        result = {"action": "use_potion", "option_index": action.potion_idx}
        if action.target_idx is not None:
            result["target_index"] = action.target_idx
        return result

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

# Potion ID → simulator effect mapping.  Keys are substrings matched
# against the potion_id from the mod API (case-insensitive).
_POTION_ID_MAP: dict[str, dict] = {
    "BLOOD": {"heal": 20},
    "HEAL": {"heal": 20},
    "FAIRY": {"heal": 20},
    "FRUIT": {"heal": 20},
    "REGEN": {"heal": 20},
    "BLOCK": {"block": 12},
    "GHOST": {"block": 12},
    "IRON": {"block": 12},
    "STRENGTH": {"strength": 2},
    "FLEX": {"strength": 2},
    "FIRE": {"damage_all": 20},
    "EXPLOSIVE": {"damage_all": 20},
    "ATTACK": {"damage_all": 20},
    "WEAK": {"enemy_weak": 3},
    "FEAR": {"enemy_weak": 3},
    "VULNERABLE": {"enemy_weak": 3},
}


def _classify_potion(potion_id: str, name: str) -> dict | None:
    """Classify a potion by ID or name into simulator effect dict."""
    key = (potion_id or name or "").upper()
    for pattern, effect in _POTION_ID_MAP.items():
        if pattern in key:
            return dict(effect)
    return None


def _parse_powers(powers_raw: list[dict]) -> dict[str, int]:
    """Parse runtime powers list into {power_name: amount} dict.

    The game stores some debuffs as negative amounts (e.g. Shrink=-1)
    while the simulator stores them as positive (Shrink=+1, matching
    Weak/Vulnerable convention).  Normalize here so the network sees
    consistent signs between training and real games.
    """
    # Powers the game stores negative but the simulator stores positive
    _FLIP_SIGN = {"Shrink"}

    result: dict[str, int] = {}
    for p in powers_raw:
        # Runtime format: {"power_id": "VULNERABLE_POWER", "name": "Vulnerable", "amount": 2}
        name = p.get("name", "")
        amount = p.get("amount", 0)
        if name and amount != 0:
            if name in _FLIP_SIGN and amount < 0:
                amount = -amount
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

        # Return a copy with runtime values if they differ from DB
        runtime_cost = raw.get("energy_cost", card.cost)
        if damage != card.damage or block != card.block or runtime_cost != card.cost:
            return Card(
                id=card.id,
                name=card.name,
                cost=runtime_cost,
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

    # Check for Unplayable keyword or -1 cost (Status/Curse cards)
    runtime_cost = raw.get("energy_cost", 0)
    runtime_keywords = raw.get("keywords") or []
    is_unplayable = runtime_cost == -1 or any(
        (k.lower() if isinstance(k, str) else "") == "unplayable"
        for k in runtime_keywords
    )

    return Card(
        id=card_id,
        name=raw.get("name", card_id),
        cost=-1 if is_unplayable else runtime_cost,
        card_type=card_type,
        target=target,
        upgraded=upgraded,
        damage=dynamic.get("Damage"),
        block=dynamic.get("Block"),
    )


def parse_intents(intents: list[dict]) -> tuple[str | None, int | None, int, int | None]:
    """Parse a list of intent dicts into (type, damage, hits, block).

    Used by bridge, run_logger, and runner to unify intent parsing.
    """
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

    return intent_type, intent_damage, intent_hits, intent_block


def _enemy_from_runtime(raw: dict) -> EnemyState:
    """Build an EnemyState from runtime combat enemy data."""
    intent_type, intent_damage, intent_hits, intent_block = parse_intents(
        raw.get("intents", []))

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
