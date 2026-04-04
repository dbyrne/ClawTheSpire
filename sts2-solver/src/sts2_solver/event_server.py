"""WebSocket + HTTP server for live dashboard updates.

Runs in a background daemon thread. The runner calls broadcast methods
after each SQLite write; connected dashboards receive events instantly.

    WS  ws://localhost:8765/ws       → live event stream
    HTTP http://localhost:8765/api/* → historical data from SQLite
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

try:
    import websockets
    from websockets.asyncio.server import serve, ServerConnection
    from websockets.http11 import Request, Response
    HAS_WS = True
except ImportError:
    HAS_WS = False

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_SQLITE = _PROJECT_ROOT / "runs.db"


def _json_serial(obj: Any) -> str:
    """Fallback serializer for non-standard types."""
    return str(obj)


def _json(data: Any) -> str:
    return json.dumps(data, default=_json_serial)


def _cors_headers() -> list[tuple[str, str]]:
    return [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        ("Content-Type", "application/json"),
    ]


# ---------------------------------------------------------------------------
# SQLite reader (read-only, for HTTP API)
# ---------------------------------------------------------------------------

class _SqliteReader:
    """Read-only SQLite connection for serving historical data."""

    def __init__(self, path: Path):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")

    def get_runs(self, limit: int = 500) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM runs ORDER BY started_at ASC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_events(self, run_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY ts ASC",
            (run_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            # Parse detail JSON string back to dict
            if d.get("detail") and isinstance(d["detail"], str):
                try:
                    d["detail"] = json.loads(d["detail"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result


# ---------------------------------------------------------------------------
# EventServer
# ---------------------------------------------------------------------------

class EventServer:
    """WS + HTTP server for the dashboard, running in a background thread."""

    def __init__(
        self,
        sqlite_path: Path = _DEFAULT_SQLITE,
        host: str = "localhost",
        port: int = 8765,
    ):
        self._sqlite_path = sqlite_path
        self._host = host
        self._port = port

        # Live state (written by main thread via broadcast_*)
        self._current_run: dict | None = None
        self._current_events: list[dict] = []
        self._event_id = 0

        # Async internals (set in _run)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[ServerConnection] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    _started = False

    def start(self) -> None:
        """Start the server in a background daemon thread (idempotent)."""
        if self._started:
            return
        if not HAS_WS:
            log.warning("websockets not installed — EventServer disabled")
            return
        self._started = True
        t = threading.Thread(target=self._run, daemon=True, name="event-server")
        t.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._reader = _SqliteReader(self._sqlite_path)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with serve(
            self._ws_handler,
            self._host,
            self._port,
            process_request=self._http_handler,
        ):
            log.info("EventServer listening on %s:%d", self._host, self._port)
            print(f"[EventServer] listening on {self._host}:{self._port}")
            await asyncio.Future()  # run forever

    # ------------------------------------------------------------------
    # HTTP handler (process_request intercepts before WS upgrade)
    # ------------------------------------------------------------------

    async def _http_handler(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        """Handle HTTP requests. Return None to proceed with WS upgrade."""
        path = request.path

        # Let WebSocket connections through
        if path == "/ws":
            return None

        def _respond(body: bytes, status: HTTPStatus = HTTPStatus.OK) -> Response:
            hdrs = [*_cors_headers(), ("Content-Length", str(len(body)))]
            return Response(status.value, status.phrase, websockets.Headers(hdrs), body)

        def _empty(status: HTTPStatus = HTTPStatus.NO_CONTENT) -> Response:
            return Response(status.value, status.phrase, websockets.Headers(_cors_headers()))

        # Strip query string for matching
        clean_path = path.split("?")[0]

        # API routes
        if clean_path == "/api/runs":
            runs = self._reader.get_runs(500)
            return _respond(_json(runs).encode())

        if clean_path.startswith("/api/runs/") and clean_path.endswith("/events"):
            run_id = clean_path[len("/api/runs/"):-len("/events")]
            events = self._reader.get_events(run_id)
            return _respond(_json(events).encode())

        if clean_path.startswith("/api/runs/"):
            run_id = clean_path[len("/api/runs/"):].rstrip("/")
            if run_id:
                run = self._reader.get_run(run_id)
                if run is None:
                    return _empty(HTTPStatus.NOT_FOUND)
                return _respond(_json(run).encode())

        # Health check
        if clean_path == "/health":
            return _respond(b'{"ok":true}')

        return _empty(HTTPStatus.NOT_FOUND)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _ws_handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        try:
            # Send snapshot of current state
            snapshot = {
                "type": "snapshot",
                "run": self._current_run,
                "events": self._current_events,
            }
            await ws.send(_json(snapshot))

            # Keep connection alive; we don't expect client messages
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)

    # ------------------------------------------------------------------
    # Broadcast (called from main thread)
    # ------------------------------------------------------------------

    def _schedule_broadcast(self, msg: dict) -> None:
        """Thread-safe: schedule a broadcast on the server's event loop."""
        if self._loop is None:
            return
        data = _json(msg)
        asyncio.run_coroutine_threadsafe(self._send_all(data), self._loop)

    async def _send_all(self, data: str) -> None:
        dead: list[ServerConnection] = []
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def broadcast_run_start(self, run: dict) -> None:
        """Called when a new run begins."""
        self._current_run = run
        self._current_events = []
        self._event_id = 0
        self._schedule_broadcast({"type": "run_start", "data": run})

    def broadcast_run_update(self, run: dict) -> None:
        """Called when a run ends or is updated."""
        self._current_run = run
        self._schedule_broadcast({"type": "run_update", "data": run})

    def broadcast_event(self, event: dict) -> None:
        """Called when a new event is logged."""
        self._event_id += 1
        event = {**event, "id": self._event_id}
        self._current_events.append(event)
        self._schedule_broadcast({"type": "event", "data": event})


# ---------------------------------------------------------------------------
# Module-level singleton — reused across Runner instances
# ---------------------------------------------------------------------------

_instance: EventServer | None = None


def get_event_server(
    sqlite_path: Path = _DEFAULT_SQLITE,
    host: str = "localhost",
    port: int = 8765,
) -> EventServer:
    """Return the singleton EventServer, creating and starting it on first call."""
    global _instance
    if _instance is None:
        _instance = EventServer(sqlite_path=sqlite_path, host=host, port=port)
        _instance.start()
    return _instance
