"""Dual-write store: local SQLite for analysis + Supabase for live dashboard.

The runner calls RunStore methods to log events. Each write goes to:
  1. Local SQLite (always, synchronous, fast)
  2. Supabase (if configured, non-blocking background thread)

Local analysis:
    sqlite3 runs.db "SELECT floor, network_value FROM run_events
                     WHERE run_id='XYZ' AND network_value IS NOT NULL"

Live dashboard reads from Supabase via real-time subscriptions.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SQLITE_PATH = _PROJECT_ROOT / "runs.db"

# Load .env if present
_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


# ---------------------------------------------------------------------------
# Local SQLite
# ---------------------------------------------------------------------------

def _init_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            character TEXT NOT NULL,
            checkpoint TEXT,
            gen TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            outcome TEXT,
            final_floor INT,
            final_hp INT,
            max_hp INT
        );
        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            event_type TEXT NOT NULL,
            floor INT,
            hp INT,
            max_hp INT,
            network_value REAL,
            detail TEXT,  -- JSON
            ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_re_run_id ON run_events(run_id);
        CREATE INDEX IF NOT EXISTS idx_re_run_ts ON run_events(run_id, ts);
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Supabase background writer
# ---------------------------------------------------------------------------

class _SupabaseWriter:
    """Drains a queue of DB operations and pushes them to Supabase."""

    def __init__(self, url: str, key: str):
        from supabase import create_client
        self._client = create_client(url, key)
        self._queue: Queue[tuple[str, dict]] = Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, table: str, op: str, data: dict) -> None:
        self._queue.put((table, op, data))

    def _worker(self) -> None:
        while True:
            try:
                table, op, data = self._queue.get(timeout=5)
            except Empty:
                continue
            try:
                if op == "upsert":
                    self._client.table(table).upsert(data).execute()
                elif op == "insert":
                    self._client.table(table).insert(data).execute()
                elif op == "update":
                    pk = data.pop("_pk")
                    pk_val = data.pop("_pk_val")
                    self._client.table(table).update(data).eq(pk, pk_val).execute()
            except Exception as e:
                # Non-fatal: log and continue
                print(f"[RunStore] Supabase write error ({table}/{op}): {e}")

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for the queue to drain (best-effort)."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------------

class RunStore:
    """Dual-write event store for STS2 runs."""

    def __init__(
        self,
        sqlite_path: Path = SQLITE_PATH,
        supabase_url: str = SUPABASE_URL,
        supabase_key: str = SUPABASE_KEY,
        event_server: Any = None,
    ):
        self._db = _init_sqlite(sqlite_path)
        self._supa: _SupabaseWriter | None = None
        self._ws = event_server  # EventServer instance (optional)
        if supabase_url and supabase_key:
            try:
                self._supa = _SupabaseWriter(supabase_url, supabase_key)
            except Exception as e:
                print(f"[RunStore] Supabase init failed ({e}), continuing local-only")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        character: str,
        checkpoint: str | None = None,
        gen: str | None = None,
        hp: int | None = None,
        max_hp: int | None = None,
    ) -> None:
        ts = self._now()
        self._db.execute(
            "INSERT OR REPLACE INTO runs (run_id, character, checkpoint, gen, started_at, max_hp)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, character, checkpoint, gen, ts, max_hp),
        )
        self._db.commit()

        if self._supa:
            self._supa.enqueue("runs", "upsert", {
                "run_id": run_id,
                "character": character,
                "checkpoint": checkpoint,
                "gen": gen,
                "started_at": ts,
                "max_hp": max_hp,
            })

        if self._ws:
            self._ws.broadcast_run_start({
                "run_id": run_id, "character": character,
                "checkpoint": checkpoint, "gen": gen,
                "started_at": ts, "ended_at": None,
                "outcome": None, "final_floor": None,
                "final_hp": hp, "max_hp": max_hp,
            })

        # Also emit a run_start event
        self.log_event(run_id, "run_start", hp=hp, max_hp=max_hp)

    def end_run(
        self,
        run_id: str,
        outcome: str,
        floor: int,
        hp: int | None = None,
        max_hp: int | None = None,
    ) -> None:
        ts = self._now()
        self._db.execute(
            "UPDATE runs SET ended_at=?, outcome=?, final_floor=?, final_hp=?, max_hp=?"
            " WHERE run_id=?",
            (ts, outcome, floor, hp, max_hp, run_id),
        )
        self._db.commit()

        if self._supa:
            self._supa.enqueue("runs", "update", {
                "_pk": "run_id", "_pk_val": run_id,
                "ended_at": ts,
                "outcome": outcome,
                "final_floor": floor,
                "final_hp": hp,
                "max_hp": max_hp,
            })

        if self._ws:
            # Re-read the full row so the dashboard gets complete data
            row = self._db.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                cols = [d[0] for d in self._db.execute("SELECT * FROM runs LIMIT 0").description]
                self._ws.broadcast_run_update(dict(zip(cols, row)))

        self.log_event(run_id, "run_end", floor=floor, hp=hp, max_hp=max_hp,
                        detail={"outcome": outcome})

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log_event(
        self,
        run_id: str,
        event_type: str,
        floor: int | None = None,
        hp: int | None = None,
        max_hp: int | None = None,
        network_value: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        ts = self._now()
        detail_json = json.dumps(detail) if detail else None

        self._db.execute(
            "INSERT INTO run_events (run_id, event_type, floor, hp, max_hp, network_value, detail, ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, event_type, floor, hp, max_hp, network_value, detail_json, ts),
        )
        self._db.commit()

        if self._supa:
            self._supa.enqueue("run_events", "insert", {
                "run_id": run_id,
                "event_type": event_type,
                "floor": floor,
                "hp": hp,
                "max_hp": max_hp,
                "network_value": round(network_value, 4) if network_value is not None else None,
                "detail": detail,
                "ts": ts,
            })

        if self._ws:
            self._ws.broadcast_event({
                "run_id": run_id,
                "event_type": event_type,
                "floor": floor,
                "hp": hp,
                "max_hp": max_hp,
                "network_value": round(network_value, 4) if network_value is not None else None,
                "detail": detail,
                "ts": ts,
            })

    # Convenience methods

    def log_combat_start(
        self, run_id: str, floor: int, hp: int, max_hp: int,
        enemies: list[str],
    ) -> None:
        self.log_event(run_id, "combat_start", floor=floor, hp=hp, max_hp=max_hp,
                        detail={"enemies": enemies})

    def log_combat_turn(
        self, run_id: str, floor: int, turn: int, hp: int, max_hp: int,
        cards_played: list[str], network_value: float | None = None,
    ) -> None:
        self.log_event(run_id, "combat_turn", floor=floor, hp=hp, max_hp=max_hp,
                        network_value=network_value,
                        detail={"turn": turn, "cards_played": cards_played})

    def log_combat_end(
        self, run_id: str, floor: int, hp: int, max_hp: int,
        outcome: str, turns: int, enemies: list[str] | None = None,
    ) -> None:
        self.log_event(run_id, "combat_end", floor=floor, hp=hp, max_hp=max_hp,
                        detail={"outcome": outcome, "turns": turns, "enemies": enemies})

    def log_decision(
        self, run_id: str, floor: int, hp: int, max_hp: int,
        screen_type: str, choice: str,
        network_value: float | None = None,
        head_scores: dict | None = None,
    ) -> None:
        detail: dict = {"screen_type": screen_type, "choice": choice}
        if head_scores is not None:
            detail["head_scores"] = head_scores
        self.log_event(run_id, "decision", floor=floor, hp=hp, max_hp=max_hp,
                        network_value=network_value, detail=detail)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Wait for pending Supabase writes to complete."""
        if self._supa:
            self._supa.flush()

    def close(self) -> None:
        self.flush()
        self._db.close()
