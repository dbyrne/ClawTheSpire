"""MCP server exposing the combat solver and strategic advisor as tools.

This runs as a separate MCP server alongside the STS2-Agent MCP server.
Claude can call `solve_combat` during combat and `advise_strategy` for all
other decisions (card rewards, map, events, shop, rest, boss relics).
"""

from __future__ import annotations

import json
import os
import traceback

from fastmcp import FastMCP

from .advisor import StrategicAdvisor
from .bridge import state_from_mcp, actions_to_mcp_sequence
from .data_loader import load_cards, CardDB
from .game_client import GameClient
from .game_data import GameDataDB, load_game_data
from .run_logger import RunLogger
from .solver import solve_turn, format_solution

mcp = FastMCP(
    name="sts2-solver",
    instructions=(
        "Combat solver for Slay the Spire 2. "
        "Call solve_combat during combat to get optimal card play sequences. "
        "The solver evaluates all legal play orderings and returns the best one. "
        "Call advise_strategy for non-combat decisions (card rewards, map, events, "
        "shop, rest sites, boss relics). It uses GPT-4o-mini for strategic reasoning."
    ),
)

# Lazy-loaded singletons
_card_db: CardDB | None = None
_game_client: GameClient | None = None
_game_data: GameDataDB | None = None
_advisor: StrategicAdvisor | None = None
_logger: RunLogger | None = None


def _get_card_db() -> CardDB:
    global _card_db
    if _card_db is None:
        _card_db = load_cards()
    return _card_db


def _get_client() -> GameClient:
    global _game_client
    if _game_client is None:
        _game_client = GameClient()
    return _game_client


def _get_game_data() -> GameDataDB:
    global _game_data
    if _game_data is None:
        _game_data = load_game_data()
    return _game_data


def _get_advisor() -> StrategicAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = StrategicAdvisor(_get_game_data(), _get_client(), logger=_get_logger())
    return _advisor


def _get_logger() -> RunLogger:
    global _logger
    if _logger is None:
        _logger = RunLogger()
        try:
            health = _get_client().get_health()
            _logger.game_version = health.get("game_version")
        except Exception:
            pass  # Game may not be running yet
    return _logger


@mcp.tool()
def solve_combat(raw_state: str | None = None, execute: bool = True) -> str:
    """Find the optimal card play sequence for the current combat turn.

    If raw_state is not provided, fetches the current state from the game API.
    If raw_state is provided, it should be a JSON string of the game state
    (from get_raw_game_state or GET /state).

    Args:
        raw_state: Optional JSON game state. If omitted, fetches live state.
        execute: If True (default), plays the cards directly via the game API
                 and returns a summary. If False, returns the action sequence
                 without executing (dry-run mode).
    """
    card_db = _get_card_db()
    client = _get_client()

    try:
        if raw_state:
            game_state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        else:
            game_state = client.get_state()
    except ConnectionError as e:
        return f"Error: Cannot connect to game API. Is the game running? ({e})"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON state: {e}"

    # Check we're in combat
    screen = game_state.get("screen", "")
    if "COMBAT" not in screen.upper():
        return f"Not in combat (screen={screen}). Solver only works during combat."

    # Check there are actions available
    actions = game_state.get("available_actions", [])
    if "play_card" not in actions:
        return "No cards can be played right now (not player's turn or no playable cards)."

    logger = _get_logger()
    logger.ensure_run(game_state)

    # Log combat start on first turn
    turn = game_state.get("turn", 0)
    if turn <= 1:
        logger.log_combat_start(game_state)

    try:
        state = state_from_mcp(game_state, card_db)
    except Exception as e:
        return f"Error converting game state: {e}\n{traceback.format_exc()}"

    # Solve
    import time as _time
    _t0 = _time.perf_counter()
    from .config import detect_character
    result = solve_turn(state, card_db=card_db, character=detect_character(game_state))
    solve_ms = (_time.perf_counter() - _t0) * 1000
    mcp_actions = actions_to_mcp_sequence(result.actions)

    # Log solver turn — simulate hand mutations to resolve card names
    cards_played = []
    log_hand = list(state.player.hand)
    for a in result.actions:
        if a.card_idx is not None and a.card_idx < len(log_hand):
            cards_played.append(log_hand[a.card_idx].name)
            log_hand.pop(a.card_idx)
    logger.log_combat_turn(
        cards_played=cards_played,
        score=result.score,
        states_evaluated=result.states_evaluated,
        solve_ms=solve_ms,
        game_state=game_state,
    )

    # Format the solution header
    lines = [format_solution(result, state), ""]

    if not execute:
        # Dry-run: return action dicts for manual execution
        lines.append("MCP actions to execute:")
        for i, action in enumerate(mcp_actions, 1):
            lines.append(f"  {i}. {json.dumps(action)}")
        return "\n".join(lines)

    # Execute each action against the game API
    lines.append("Executing:")
    hand = list(state.player.hand)
    for i, (solver_action, mcp_action) in enumerate(
        zip(result.actions, mcp_actions), 1
    ):
        # Log what we're doing
        if solver_action.action_type == "end_turn":
            label = "End Turn"
        else:
            card = hand[solver_action.card_idx]
            target = f" -> enemy {solver_action.target_idx}" if solver_action.target_idx is not None else ""
            label = f"{card.name}{target}"
            hand.pop(solver_action.card_idx)

        try:
            client.execute_action(
                mcp_action["action"],
                card_index=mcp_action.get("card_index"),
                target_index=mcp_action.get("target_index"),
            )
            lines.append(f"  {i}. {label} ✓")
        except Exception as e:
            lines.append(f"  {i}. {label} ✗ {e}")
            lines.append("Stopped execution due to error.")
            break

    # Fetch post-combat state for summary
    try:
        post_state = client.get_state()
        post_combat = post_state.get("combat") or {}
        post_player = post_combat.get("player") or {}
        post_screen = post_state.get("screen", "")

        hp = post_player.get("current_hp", "?")
        block = post_player.get("block", 0)

        enemies_alive = []
        for e in post_combat.get("enemies", []):
            if e.get("current_hp", 0) > 0:
                enemies_alive.append(f"{e.get('name', '?')} {e['current_hp']}hp")

        lines.append("")
        if "COMBAT" not in post_screen.upper() or not enemies_alive:
            lines.append(f"Combat won! HP: {hp}")
            logger.log_combat_end(post_state, "win")
        else:
            lines.append(f"Turn ended. HP: {hp}, Block: {block}")
            lines.append(f"Enemies remaining: {', '.join(enemies_alive)}")
    except Exception:
        pass  # Non-critical, solver already did its job

    return "\n".join(lines)


@mcp.tool()
def advise_strategy(raw_state: str | None = None, execute: bool = True) -> str:
    """Get strategic advice for non-combat decisions (card rewards, map, events,
    shop, rest sites, boss relics, chests).

    If raw_state is not provided, fetches the current state from the game API.

    Args:
        raw_state: Optional JSON game state. If omitted, fetches live state.
        execute: If True (default), executes the recommended action via the game API.
                 If False, returns the recommendation without acting.
    """
    advisor = _get_advisor()
    client = _get_client()

    try:
        if raw_state:
            game_state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        else:
            game_state = client.get_state()
    except ConnectionError as e:
        return f"Error: Cannot connect to game API. Is the game running? ({e})"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON state: {e}"

    try:
        return advisor.advise(game_state, execute=execute)
    except Exception as e:
        return f"Error in advisor: {e}\n{traceback.format_exc()}"


@mcp.tool()
def solver_health() -> str:
    """Check solver status: card database loaded, game API reachable."""
    card_db = _get_card_db()
    lines = [
        f"Solver ready",
        f"Card database: {len(card_db)} entries loaded",
    ]

    try:
        client = _get_client()
        health = client.get_health()
        lines.append(f"Game API: connected ({health.get('game_version', '?')})")
    except Exception as e:
        lines.append(f"Game API: not reachable ({e})")

    return "\n".join(lines)


def main():
    mcp.run()
