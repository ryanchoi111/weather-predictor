import json
import aiosqlite
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    model TEXT NOT NULL,
    ensemble_temps TEXT NOT NULL,
    mean_temp REAL NOT NULL,
    std_temp REAL NOT NULL,
    lead_hours INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER REFERENCES forecasts(id),
    city TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    edge_cents REAL NOT NULL,
    model_prob REAL NOT NULL,
    market_price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pending_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    discord_message_id TEXT NOT NULL,
    city TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    event_ticker TEXT NOT NULL,
    order_cost_cents INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES signals(id),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    count INTEGER NOT NULL,
    order_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    filled_at TEXT
);
"""


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_forecast(self, city: str, model: str, temps: list[float], lead_hours: int) -> int:
        arr = np.array(temps)
        cursor = await self._db.execute(
            "INSERT INTO forecasts (city, model, ensemble_temps, mean_temp, std_temp, lead_hours) VALUES (?, ?, ?, ?, ?, ?)",
            (city, model, json.dumps(temps), float(arr.mean()), float(arr.std()), lead_hours),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def save_signal(self, forecast_id: int, city: str, ticker: str, side: str,
                          edge_cents: float, model_prob: float, market_price_cents: int, contracts: int) -> int:
        cursor = await self._db.execute(
            "INSERT INTO signals (forecast_id, city, ticker, side, edge_cents, model_prob, market_price_cents, contracts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (forecast_id, city, ticker, side, edge_cents, model_prob, market_price_cents, contracts),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def save_trade(self, signal_id: int, ticker: str, side: str, action: str,
                         price_cents: int, count: int, order_id: str, status: str = "pending") -> int:
        cursor = await self._db.execute(
            "INSERT INTO trades (signal_id, ticker, side, action, price_cents, count, order_id, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (signal_id, ticker, side, action, price_cents, count, order_id, status),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_todays_trades(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE date(created_at) = date('now')"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def save_pending_approval(
        self, signal_id: int, discord_message_id: str, city: str, ticker: str,
        side: str, price_cents: int, contracts: int, event_ticker: str, order_cost_cents: int,
    ) -> int:
        cursor = await self._db.execute(
            """INSERT INTO pending_approvals
            (signal_id, discord_message_id, city, ticker, side, price_cents, contracts, event_ticker, order_cost_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal_id, discord_message_id, city, ticker, side, price_cents, contracts, event_ticker, order_cost_cents),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_pending_approval(self, discord_message_id: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM pending_approvals WHERE discord_message_id = ?",
            (discord_message_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def resolve_approval(self, approval_id: int, status: str) -> None:
        await self._db.execute(
            "UPDATE pending_approvals SET status = ?, resolved_at = datetime('now') WHERE id = ?",
            (status, approval_id),
        )
        await self._db.commit()

    async def get_pending_approvals(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM pending_approvals WHERE status = 'pending'"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_latest_forecast(self, city: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM forecasts WHERE city = ? ORDER BY created_at DESC LIMIT 1",
            (city,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
