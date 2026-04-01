"""Tests for the game client HTTP wrapper."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib import error as urllib_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from sts2_solver.game_client import GameClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict, ok: bool = True):
    """Create a mock urllib response with the API envelope."""
    body = json.dumps({"ok": ok, "data": data}).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _mock_raw_response(body: dict):
    """Create a mock response with raw body (no envelope)."""
    raw = json.dumps(body).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGameClientInit:
    def test_default_url(self):
        client = GameClient()
        assert "127.0.0.1" in client.base_url

    def test_custom_url(self):
        client = GameClient(base_url="http://localhost:9999")
        assert client.base_url == "http://localhost:9999"

    def test_trailing_slash_stripped(self):
        client = GameClient(base_url="http://localhost:9999/")
        assert client.base_url == "http://localhost:9999"


class TestGetState:
    @patch("sts2_solver.game_client.request.urlopen")
    def test_returns_unwrapped_data(self, mock_urlopen):
        state_data = {"screen": "COMBAT", "turn": 1}
        mock_urlopen.return_value = _mock_response(state_data)

        client = GameClient()
        result = client.get_state()
        assert result == state_data

    @patch("sts2_solver.game_client.request.urlopen")
    def test_raises_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib_error.URLError("Connection refused")

        client = GameClient()
        with pytest.raises(ConnectionError, match="Cannot reach game API"):
            client.get_state()


class TestGetHealth:
    @patch("sts2_solver.game_client.request.urlopen")
    def test_returns_health_data(self, mock_urlopen):
        health_data = {"game_version": "0.7.0", "status": "ok"}
        mock_urlopen.return_value = _mock_response(health_data)

        client = GameClient()
        result = client.get_health()
        assert result == health_data


class TestExecuteAction:
    @patch("sts2_solver.game_client.request.urlopen")
    def test_end_turn(self, mock_urlopen):
        result_data = {"status": "completed"}
        mock_urlopen.return_value = _mock_response(result_data)

        client = GameClient()
        result = client.execute_action("end_turn")
        assert result == result_data

        # Verify the request body
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["action"] == "end_turn"
        assert "client_context" in body

    @patch("sts2_solver.game_client.request.urlopen")
    def test_play_card_with_target(self, mock_urlopen):
        result_data = {"status": "completed"}
        mock_urlopen.return_value = _mock_response(result_data)

        client = GameClient()
        result = client.execute_action(
            "play_card", card_index=0, target_index=1
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["action"] == "play_card"
        assert body["card_index"] == 0
        assert body["target_index"] == 1

    @patch("sts2_solver.game_client.request.urlopen")
    def test_play_card_no_target(self, mock_urlopen):
        result_data = {"status": "completed"}
        mock_urlopen.return_value = _mock_response(result_data)

        client = GameClient()
        client.execute_action("play_card", card_index=2)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["card_index"] == 2
        assert "target_index" not in body

    @patch("sts2_solver.game_client.request.urlopen")
    def test_option_index(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"status": "completed"})

        client = GameClient()
        client.execute_action("choose_reward_card", option_index=1)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["option_index"] == 1

    @patch("sts2_solver.game_client.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib_error.URLError("Connection refused")

        client = GameClient()
        with pytest.raises(ConnectionError):
            client.execute_action("end_turn")


class TestUnwrap:
    def test_unwraps_envelope(self):
        body = {"ok": True, "data": {"screen": "MAP"}}
        assert GameClient._unwrap(body) == {"screen": "MAP"}

    def test_passthrough_no_envelope(self):
        body = {"screen": "MAP"}
        assert GameClient._unwrap(body) == {"screen": "MAP"}

    def test_non_dict(self):
        body = "just a string"
        assert GameClient._unwrap(body) == "just a string"
