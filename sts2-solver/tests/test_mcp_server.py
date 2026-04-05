"""Integration tests for the MCP server tools."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import sts2_solver.mcp_server as mcp_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons between tests."""
    mcp_mod._card_db = None
    mcp_mod._game_client = None
    yield
    mcp_mod._card_db = None
    mcp_mod._game_client = None


# ---------------------------------------------------------------------------
# solver_health
# ---------------------------------------------------------------------------


class TestSolverHealth:
    def test_reports_card_db_loaded(self):
        result = mcp_mod.solver_health()
        assert "Solver ready" in result
        assert "entries loaded" in result

    def test_reports_game_api_unreachable(self):
        mock_client = MagicMock()
        mock_client.get_health.side_effect = ConnectionError("refused")
        mcp_mod._game_client = mock_client

        result = mcp_mod.solver_health()
        assert "not reachable" in result

    def test_reports_game_api_connected(self):
        mock_client = MagicMock()
        mock_client.get_health.return_value = {"game_version": "0.7.0"}
        mcp_mod._game_client = mock_client

        result = mcp_mod.solver_health()
        assert "connected" in result
        assert "0.7.0" in result


# ---------------------------------------------------------------------------
# Singleton getters
# ---------------------------------------------------------------------------


class TestSingletons:
    def test_card_db_cached(self):
        db1 = mcp_mod._get_card_db()
        db2 = mcp_mod._get_card_db()
        assert db1 is db2

    def test_client_cached(self):
        c1 = mcp_mod._get_client()
        c2 = mcp_mod._get_client()
        assert c1 is c2
