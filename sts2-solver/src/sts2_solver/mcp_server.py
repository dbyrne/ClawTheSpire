"""MCP server exposing game state inspection and solver health tools.

This runs as a separate MCP server alongside the STS2-Agent MCP server.
Used for Claude Code debugging of live game state.
"""

from __future__ import annotations

import json
import os
import traceback

from fastmcp import FastMCP

from .data_loader import load_cards, CardDB
from .game_client import GameClient

mcp = FastMCP(
    name="sts2-solver",
    instructions=(
        "Debugging tools for Slay the Spire 2. "
        "Call solver_health to check connectivity and card database status."
    ),
)

# Lazy-loaded singletons
_card_db: CardDB | None = None
_game_client: GameClient | None = None


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
