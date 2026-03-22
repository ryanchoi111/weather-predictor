from dataclasses import dataclass


@dataclass
class Signal:
    ticker: str
    title: str
    side: str  # "yes" or "no"
    edge_cents: float
    model_prob: float
    market_price_cents: int
    confidence: float  # ensemble std


def generate_signals(
    tickers: list[str],
    titles: list[str],
    model_probs: list[float],
    market_prices_cents: list[int],
    ensemble_std: float,
    edge_threshold: float = 5.0,
    max_std: float = 8.0,
    min_volume: list[int] | None = None,
    min_liquidity: int = 10,
) -> list[Signal]:
    """Generate trading signals where model disagrees with market.

    Edge = model_prob * 100 - market_price_cents (for YES)
    If edge > threshold -> buy YES
    If edge < -threshold -> buy NO (edge on NO side)
    """
    if ensemble_std > max_std:
        return []

    signals = []
    for i, (ticker, title, prob, price) in enumerate(zip(tickers, titles, model_probs, market_prices_cents)):
        if min_volume and min_volume[i] < min_liquidity:
            continue

        if price >= 95 or price <= 5:
            continue

        model_cents = prob * 100
        edge = model_cents - price

        if abs(edge) < edge_threshold:
            continue

        side = "yes" if edge > 0 else "no"
        signals.append(Signal(
            ticker=ticker,
            title=title,
            side=side,
            edge_cents=abs(edge),
            model_prob=prob,
            market_price_cents=price,
            confidence=ensemble_std,
        ))

    return sorted(signals, key=lambda s: s.edge_cents, reverse=True)
