"""Minimal HTTP client for the STS2 Agent mod API."""

from __future__ import annotations

import json
import os
from urllib import request, error
from typing import Any


DEFAULT_BASE_URL = os.environ.get("STS2_API_BASE_URL", "http://127.0.0.1:8081")


class GameClient:
    """Thin HTTP client for the STS2 game mod API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

    def get_state(self) -> dict[str, Any]:
        """GET /state — full game state snapshot."""
        return self._get("/state")

    def get_health(self) -> dict[str, Any]:
        """GET /health — mod health check."""
        return self._get("/health")

    def execute_action(
        self,
        action: str,
        *,
        card_index: int | None = None,
        target_index: int | None = None,
        option_index: int | None = None,
    ) -> dict[str, Any]:
        """POST /action — execute a game action."""
        payload = {"action": action}
        if card_index is not None:
            payload["card_index"] = card_index
        if target_index is not None:
            payload["target_index"] = target_index
        if option_index is not None:
            payload["option_index"] = option_index
        payload["client_context"] = {"source": "solver"}
        return self._post("/action", payload)

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        try:
            with request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.URLError as e:
            raise ConnectionError(f"Cannot reach game API at {url}: {e}") from e
        return self._unwrap(body)

    def _post(self, path: str, payload: dict) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, method="POST", data=data)
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.URLError as e:
            raise ConnectionError(f"Cannot reach game API at {url}: {e}") from e
        return self._unwrap(body)

    @staticmethod
    def _unwrap(body: dict) -> dict[str, Any]:
        """Unwrap the mod API's {ok, data} envelope."""
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body
