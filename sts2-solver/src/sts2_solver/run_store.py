"""Local SQLite event store for STS2 runs.

The runner calls RunStore methods to log events. Each write goes to
local SQLite and optionally broadcasts via the EventServer WebSocket.

Local analysis:
    sqlite3 runs.db "SELECT floor, network_value FROM run_events
                     WHERE run_id='XYZ' AND network_value IS NOT NULL"
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SQLITE_PATH = _PROJECT_ROOT / "runs.db"


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
# RunStore
# ---------------------------------------------------------------------------

class RunStore:
    """SQLite event store for STS2 runs."""

    def __init__(
        self,
        sqlite_path: Path = SQLITE_PATH,
        event_server: Any = None,
    ):
        self._db = _init_sqlite(sqlite_path)
        self._ws = event_server  # EventServer instance (optional)

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

    def close(self) -> None:
        self._db.close()
