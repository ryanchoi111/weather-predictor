import re
from dataclasses import dataclass

from .client import KalshiClient


@dataclass
class TemperatureBucket:
    ticker: str
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


def parse_temperature_buckets(markets: list[dict]) -> list[TemperatureBucket]:
    buckets = []
    for m in markets:
        low, high = _parse_range(m.get("subtitle", "") or m.get("title", ""))
        buckets.append(TemperatureBucket(
            ticker=m["ticker"],
            low_temp=low,
            high_temp=high,
            yes_price=m.get("yes_price", 0),
            yes_ask=m.get("yes_ask", 0),
            yes_bid=m.get("yes_bid", 0),
            volume=m.get("volume", 0),
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
