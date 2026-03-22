import re
from dataclasses import dataclass

from .client import KalshiClient


@dataclass
class TemperatureBucket:
    ticker: str
    title: str
    low_temp: float | None
    high_temp: float | None
    yes_price: int
    yes_ask: int
    yes_bid: int
    volume: int


async def get_weather_events(client: KalshiClient, series_ticker: str) -> list[dict]:
    resp = await client.get("/events", params={
        "series_ticker": series_ticker,
        "status": "open",
    })
    return resp.get("events", [])


async def get_event_markets(client: KalshiClient, event_ticker: str) -> list[dict]:
    resp = await client.get("/markets", params={"event_ticker": event_ticker})
    return resp.get("markets", [])


def _dollars_to_cents(val) -> int:
    """Convert dollar string/float to cents, or return 0 if None."""
    if val is None:
        return 0
    return int(round(float(val) * 100))


def _parse_volume(val) -> int:
    """Parse volume from API — may be int, float string, or None."""
    if val is None:
        return 0
    return int(float(val))


def parse_temperature_buckets(markets: list[dict]) -> list[TemperatureBucket]:
    buckets = []
    for m in markets:
        low, high = _parse_range(m.get("subtitle", "") or m.get("title", ""))
        # New API uses dollar fields; fall back to legacy cent fields
        yes_price = _dollars_to_cents(m.get("last_price_dollars")) or (m.get("yes_price") or 0)
        yes_ask = _dollars_to_cents(m.get("yes_ask_dollars")) or (m.get("yes_ask") or 0)
        yes_bid = _dollars_to_cents(m.get("yes_bid_dollars")) or (m.get("yes_bid") or 0)
        volume = _parse_volume(m.get("volume_fp")) or _parse_volume(m.get("volume"))
        title = (m.get("title") or "").replace("**", "")
        buckets.append(TemperatureBucket(
            ticker=m["ticker"],
            title=title,
            low_temp=low,
            high_temp=high,
            yes_price=yes_price,
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            volume=volume,
        ))
    return sorted(buckets, key=lambda b: b.low_temp if b.low_temp is not None else -999)


def _parse_range(text: str) -> tuple[float | None, float | None]:
    text = text.replace("\u00b0F", "").replace("\u00b0", "")

    m = re.search(r"(\d+)\s+or\s+(above|higher|more)", text, re.IGNORECASE)
    if m:
        return float(m.group(1)), None

    m = re.search(r"(\d+)\s+or\s+(below|lower|less)", text, re.IGNORECASE)
    if m:
        return None, float(m.group(1))

    m = re.search(r"[Bb]etween\s+(\d+)\s+and\s+(\d+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    return None, None


async def get_orderbook(client: KalshiClient, ticker: str) -> dict:
    return await client.get(f"/orderbook/{ticker}")
