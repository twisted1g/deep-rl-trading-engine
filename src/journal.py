"""SQLite-журнал сделок и snapshot состояния трейдера.

Используется:
- `_open_position` / `_close_position` для записи сделок,
- `step()` для сохранения `last_processed_bar_time` / `paused`,
- TG-командами `/pnl` и `/trades` для чтения.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    side            TEXT    NOT NULL,
    qty             REAL    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    pnl_pct         REAL,
    pnl_abs         REAL,
    exit_reason     TEXT,
    opened_at       TEXT    NOT NULL,
    closed_at       TEXT,
    bar_time_open   TEXT,
    bar_time_close  TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);

CREATE TABLE IF NOT EXISTS state (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),
    last_processed_bar_time  TEXT,
    paused                   INTEGER NOT NULL DEFAULT 0,
    updated_at               TEXT    NOT NULL
);
"""


@dataclass
class TraderStateSnapshot:
    last_processed_bar_time: Optional[str]
    paused: bool


class TradeJournal:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as cx:
            cx.execute("PRAGMA journal_mode=WAL")
            cx.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            cx = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
            cx.row_factory = sqlite3.Row
            try:
                yield cx
            finally:
                cx.close()

    # ---------- trades ----------

    def open_trade(
        self,
        side: str,
        qty: float,
        entry_price: float,
        bar_time: Optional[str] = None,
    ) -> int:
        now = _utcnow()
        with self._conn() as cx:
            cur = cx.execute(
                "INSERT INTO trades (side, qty, entry_price, opened_at, bar_time_open) "
                "VALUES (?, ?, ?, ?, ?)",
                (side, qty, entry_price, now, bar_time),
            )
            return int(cur.lastrowid)

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl_pct: float,
        pnl_abs: Optional[float] = None,
        exit_reason: str = "model",
        bar_time: Optional[str] = None,
    ) -> None:
        with self._conn() as cx:
            cx.execute(
                "UPDATE trades SET exit_price=?, pnl_pct=?, pnl_abs=?, "
                "exit_reason=?, closed_at=?, bar_time_close=? WHERE id=?",
                (exit_price, pnl_pct, pnl_abs, exit_reason,
                 _utcnow(), bar_time, trade_id),
            )

    def last_open_trade(self) -> Optional[sqlite3.Row]:
        with self._conn() as cx:
            cur = cx.execute(
                "SELECT * FROM trades WHERE closed_at IS NULL "
                "ORDER BY id DESC LIMIT 1"
            )
            return cur.fetchone()

    def recent(self, n: int = 10) -> list[sqlite3.Row]:
        with self._conn() as cx:
            cur = cx.execute(
                "SELECT * FROM trades WHERE closed_at IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (n,),
            )
            return list(cur.fetchall())

    def stats(self) -> dict:
        with self._conn() as cx:
            cur = cx.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins, "
                "COALESCE(SUM(pnl_pct), 0) AS total_pnl, "
                "COALESCE(AVG(pnl_pct), 0) AS avg_pnl "
                "FROM trades WHERE closed_at IS NOT NULL"
            )
            row = cur.fetchone()
        n = int(row["n"] or 0)
        wins = int(row["wins"] or 0)
        return {
            "n_trades": n,
            "wins": wins,
            "win_rate": (wins / n) if n else 0.0,
            "total_pnl_pct": float(row["total_pnl"] or 0.0),
            "avg_pnl_pct": float(row["avg_pnl"] or 0.0),
        }

    # ---------- state ----------

    def save_state(self, last_processed_bar_time: Optional[str], paused: bool) -> None:
        with self._conn() as cx:
            cx.execute(
                "INSERT INTO state (id, last_processed_bar_time, paused, updated_at) "
                "VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  last_processed_bar_time=excluded.last_processed_bar_time, "
                "  paused=excluded.paused, "
                "  updated_at=excluded.updated_at",
                (last_processed_bar_time, int(paused), _utcnow()),
            )

    def load_state(self) -> Optional[TraderStateSnapshot]:
        with self._conn() as cx:
            cur = cx.execute(
                "SELECT last_processed_bar_time, paused FROM state WHERE id=1"
            )
            row = cur.fetchone()
        if row is None:
            return None
        return TraderStateSnapshot(
            last_processed_bar_time=row["last_processed_bar_time"],
            paused=bool(row["paused"]),
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
