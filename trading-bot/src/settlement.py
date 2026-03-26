"""Fetch all settled weather contracts from Kalshi and record outcomes.

Grabs every settled market for all weather series — not just trades we made.
This gives the calibration module a full picture of model accuracy.
"""

from .kalshi.client import KalshiClient
from .kalshi.markets import parse_temperature_buckets, _parse_range
from .data.db import Database
from .data.logger import get_logger

log = get_logger("settlement")

# Series tickers for all tracked cities
CITY_SERIES = {
    "KXHIGHNY": "New York",
    "KXHIGHCHI": "Chicago",
    "KXHIGHLA": "Los Angeles",
    "KXHIGHMIA": "Miami",
    "KXHIGHAUS": "Austin",
}


async def fetch_settled_events(client: KalshiClient, series_ticker: str) -> list[dict]:
    """Fetch settled events for a weather series."""
    try:
        resp = await client.get("/events", params={
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 50,
        })
        return resp.get("events", [])
    except Exception as e:
        log.warning("settled_events_fetch_failed", series=series_ticker, error=str(e))
        return []


async def fetch_settled_markets(client: KalshiClient, event_ticker: str) -> list[dict]:
    """Fetch all markets (buckets) for a settled event."""
    try:
        resp = await client.get("/markets", params={"event_ticker": event_ticker})
        return resp.get("markets", [])
    except Exception as e:
        log.warning("settled_markets_fetch_failed", event=event_ticker, error=str(e))
        return []


def compute_pnl(side: str, price_cents: int, contracts: int, result: str) -> int:
    """Compute P&L in cents for a settled trade."""
    won = (side == result)
    if won:
        return (100 - price_cents) * contracts
    return -price_cents * contracts


async def process_settlements(
    client: KalshiClient,
    db: Database,
) -> dict:
    """Fetch all settled weather contracts and record them.

    Returns summary dict with counts.
    """
    already_settled = await db.get_unsettled_tickers()
    # Get all signals and trades for matching
    all_signals = {}
    all_trades = {}
    try:
        cursor = await db._db.execute("SELECT * FROM signals")
        rows = await cursor.fetchall()
        for r in rows:
            r = dict(r)
            all_signals[r["ticker"]] = r
    except Exception:
        pass
    try:
        cursor = await db._db.execute("SELECT * FROM trades WHERE status != 'cancelled'")
        rows = await cursor.fetchall()
        for r in rows:
            r = dict(r)
            all_trades[r["ticker"]] = r
    except Exception:
        pass

    new_settlements = 0
    new_traded = 0
    total_pnl = 0

    for series_ticker, city in CITY_SERIES.items():
        events = await fetch_settled_events(client, series_ticker)
        for event in events:
            event_ticker = event["event_ticker"]
            markets = await fetch_settled_markets(client, event_ticker)

            for market in markets:
                ticker = market["ticker"]
                if ticker in already_settled:
                    continue

                result = market.get("result")
                if result not in ("yes", "no"):
                    continue

                title = (market.get("title") or "").replace("**", "")
                low, high = _parse_range(market.get("subtitle", "") or title)

                # Check if we had a model prediction for this contract
                signal = all_signals.get(ticker)
                model_prob = signal["model_prob"] if signal else None
                market_price = signal["market_price_cents"] if signal else None

                # Check if we traded this contract
                trade = all_trades.get(ticker)
                traded = trade is not None
                trade_side = trade["side"] if trade else None
                trade_price = trade["price_cents"] if trade else None
                trade_contracts = trade["count"] if trade else None
                pnl = None
                if trade:
                    pnl = compute_pnl(trade_side, trade_price, trade_contracts, result)
                    total_pnl += pnl
                    new_traded += 1

                await db.save_settled_contract(
                    ticker=ticker,
                    event_ticker=event_ticker,
                    city=city,
                    title=title,
                    low_temp=low,
                    high_temp=high,
                    result=result,
                    model_prob=model_prob,
                    market_price_cents=market_price,
                    traded=traded,
                    trade_side=trade_side,
                    trade_price_cents=trade_price,
                    trade_contracts=trade_contracts,
                    pnl_cents=pnl,
                )
                new_settlements += 1

    summary = {
        "new_settlements": new_settlements,
        "new_traded": new_traded,
        "total_pnl_cents": total_pnl,
    }
    if new_settlements > 0:
        log.info("settlements_processed", **summary)
    return summary
