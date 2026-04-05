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
from .data_loader import load_cards, CardDB
from .game_client import GameClient
from .game_data import GameDataDB, load_game_data
from .run_logger import RunLogger

mcp = FastMCP(
    name="sts2-solver",
    instructions=(
        "Strategic advisor for Slay the Spire 2. "
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
