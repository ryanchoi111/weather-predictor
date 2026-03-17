from dataclasses import dataclass

from .client import KalshiClient


@dataclass
class Position:
    ticker: str
    count: int
    avg_price: int
    side: str


class OrderValidationError(Exception):
    pass


def validate_order_params(side: str, price_cents: int, count: int) -> None:
    if side not in ("yes", "no"):
        raise OrderValidationError(f"Invalid side: {side}")
    if not 1 <= price_cents <= 99:
        raise OrderValidationError(f"Price must be 1-99 cents, got {price_cents}")
    if count < 1:
        raise OrderValidationError(f"Count must be >= 1, got {count}")


async def create_order(
    client: KalshiClient,
    ticker: str,
    side: str,
    price_cents: int,
    count: int,
    action: str = "buy",
) -> dict:
    validate_order_params(side, price_cents, count)
    return await client.post("/portfolio/orders", json={
        "ticker": ticker,
        "action": action,
        "side": side,
        "type": "limit",
        "yes_price": price_cents if side == "yes" else None,
        "no_price": price_cents if side == "no" else None,
        "count": count,
    })


async def cancel_order(client: KalshiClient, order_id: str) -> dict:
    return await client.delete(f"/portfolio/orders/{order_id}")


async def get_positions(client: KalshiClient) -> list[Position]:
    resp = await client.get("/portfolio/positions")
    positions = []
    for p in resp.get("market_positions", []):
        if p.get("position", 0) != 0:
            positions.append(Position(
                ticker=p["ticker"],
                count=abs(p["position"]),
                avg_price=p.get("market_average_price", 0),
                side="yes" if p["position"] > 0 else "no",
            ))
    return positions


async def get_balance(client: KalshiClient) -> int:
    resp = await client.get("/portfolio/balance")
    return resp.get("balance", 0)
