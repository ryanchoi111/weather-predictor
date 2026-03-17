def kelly_size(
    model_prob: float,
    market_price_cents: int,
    side: str,
    bankroll_cents: int,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.05,
) -> int:
    """Fractional Kelly sizing -> number of contracts.

    Kelly formula for binary bet:
    f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = net odds (payout/risk)
    """
    if side == "yes":
        p = model_prob
        cost = market_price_cents
    else:
        p = 1 - model_prob
        cost = 100 - market_price_cents

    if cost <= 0 or cost >= 100:
        return 0

    b = (100 - cost) / cost
    q = 1 - p

    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0

    f_adj = f_star * kelly_fraction

    max_dollars = bankroll_cents * max_position_pct
    kelly_dollars = bankroll_cents * f_adj

    position_cents = min(kelly_dollars, max_dollars)
    contracts = int(position_cents / cost)

    return max(contracts, 0)
