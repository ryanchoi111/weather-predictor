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

CREATE TABLE IF NOT EXISTS forecast_actuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
    city TEXT NOT NULL,
    forecast_date TEXT NOT NULL,
    model_mean_f REAL NOT NULL,
    actual_high_f REAL,
    source TEXT DEFAULT 'iem',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(forecast_id)
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

CREATE TABLE IF NOT EXISTS bucket_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
    city TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    ticker TEXT NOT NULL,
    title TEXT,
    low_temp REAL,
    high_temp REAL,
    model_prob REAL NOT NULL,
    market_price_cents INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(forecast_id, ticker)
);

CREATE TABLE IF NOT EXISTS settled_contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    event_ticker TEXT NOT NULL,
    city TEXT NOT NULL,
    title TEXT,
    low_temp REAL,
    high_temp REAL,
    result TEXT NOT NULL,
    model_prob REAL,
    market_price_cents INTEGER,
    traded INTEGER NOT NULL DEFAULT 0,
    trade_side TEXT,
    trade_price_cents INTEGER,
    trade_contracts INTEGER,
    pnl_cents INTEGER,
    settled_at TEXT NOT NULL DEFAULT (datetime('now'))
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

    async def save_forecast_actual(
        self, forecast_id: int, city: str, forecast_date: str, model_mean_f: float,
        actual_high_f: float | None = None,
    ) -> int:
        cursor = await self._db.execute(
            """INSERT OR IGNORE INTO forecast_actuals
            (forecast_id, city, forecast_date, model_mean_f, actual_high_f)
            VALUES (?, ?, ?, ?, ?)""",
            (forecast_id, city, forecast_date, model_mean_f, actual_high_f),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_forecast_actual(self, forecast_id: int, actual_high_f: float) -> None:
        await self._db.execute(
            "UPDATE forecast_actuals SET actual_high_f = ? WHERE forecast_id = ?",
            (actual_high_f, forecast_id),
        )
        await self._db.commit()

    async def get_unfilled_actuals(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM forecast_actuals WHERE actual_high_f IS NULL"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_forecast_actuals(self, city: str | None = None) -> list[dict]:
        if city:
            cursor = await self._db.execute(
                "SELECT * FROM forecast_actuals WHERE city = ? AND actual_high_f IS NOT NULL ORDER BY forecast_date",
                (city,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM forecast_actuals WHERE actual_high_f IS NOT NULL ORDER BY forecast_date"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def save_settled_contract(
        self, ticker: str, event_ticker: str, city: str, title: str,
        low_temp: float | None, high_temp: float | None, result: str,
        model_prob: float | None = None, market_price_cents: int | None = None,
        traded: bool = False, trade_side: str | None = None,
        trade_price_cents: int | None = None, trade_contracts: int | None = None,
        pnl_cents: int | None = None,
    ) -> int:
        cursor = await self._db.execute(
            """INSERT OR IGNORE INTO settled_contracts
            (ticker, event_ticker, city, title, low_temp, high_temp, result,
             model_prob, market_price_cents, traded, trade_side, trade_price_cents,
             trade_contracts, pnl_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, event_ticker, city, title, low_temp, high_temp, result,
             model_prob, market_price_cents, int(traded), trade_side,
             trade_price_cents, trade_contracts, pnl_cents),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_settled_contracts(self, city: str | None = None) -> list[dict]:
        if city:
            cursor = await self._db.execute(
                "SELECT * FROM settled_contracts WHERE city = ? ORDER BY settled_at",
                (city,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM settled_contracts ORDER BY settled_at"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_unsettled_tickers(self) -> set[str]:
        """Get tickers we already recorded so we don't re-fetch."""
        cursor = await self._db.execute("SELECT ticker FROM settled_contracts")
        rows = await cursor.fetchall()
        return {r["ticker"] for r in rows}

    async def get_calibration_data(self) -> list[dict]:
        """Get all settled contracts that have model_prob for calibration analysis."""
        cursor = await self._db.execute(
            "SELECT * FROM settled_contracts WHERE model_prob IS NOT NULL ORDER BY settled_at"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_trade_pnl_summary(self) -> dict:
        """Get aggregate P&L stats for traded contracts."""
        cursor = await self._db.execute(
            """SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_cents <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl_cents) as total_pnl_cents,
                AVG(pnl_cents) as avg_pnl_cents
            FROM settled_contracts WHERE traded = 1 AND pnl_cents IS NOT NULL"""
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def get_latest_forecast(self, city: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM forecasts WHERE city = ? ORDER BY created_at DESC LIMIT 1",
            (city,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
