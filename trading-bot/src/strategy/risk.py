from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class RiskLimits:
    max_single_position_pct: float = 0.05
    max_event_exposure_pct: float = 0.15
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.15
    max_open_orders: int = 20
    min_bankroll_dollars: int = 100


@dataclass
class RiskState:
    bankroll_cents: int = 0
    high_water_mark_cents: int = 0
    daily_pnl_cents: int = 0
    open_order_count: int = 0
    event_exposures: dict[str, int] = field(default_factory=dict)


class RiskBreachError(Exception):
    pass


def check_pre_trade(
    limits: RiskLimits,
    state: RiskState,
    order_cost_cents: int,
    event_ticker: str,
) -> None:
    """Run all pre-trade risk checks. Raises RiskBreachError on violation."""
    bankroll = state.bankroll_cents

    if bankroll < limits.min_bankroll_dollars * 100:
        raise RiskBreachError(f"Bankroll ${bankroll/100:.2f} below minimum ${limits.min_bankroll_dollars}")

    if order_cost_cents > bankroll * limits.max_single_position_pct:
        raise RiskBreachError(
            f"Order ${order_cost_cents/100:.2f} exceeds {limits.max_single_position_pct:.0%} single position limit"
        )

    current_event = state.event_exposures.get(event_ticker, 0)
    if (current_event + order_cost_cents) > bankroll * limits.max_event_exposure_pct:
        raise RiskBreachError(f"Event {event_ticker} exposure would exceed {limits.max_event_exposure_pct:.0%} limit")

    if state.daily_pnl_cents < -(bankroll * limits.max_daily_loss_pct):
        raise RiskBreachError(f"Daily loss limit breached: PnL ${state.daily_pnl_cents/100:.2f}")

    if state.high_water_mark_cents > 0:
        drawdown = (state.high_water_mark_cents - bankroll) / state.high_water_mark_cents
        if drawdown > limits.max_drawdown_pct:
            raise RiskBreachError(f"Drawdown {drawdown:.1%} exceeds {limits.max_drawdown_pct:.0%} limit")

    if state.open_order_count >= limits.max_open_orders:
        raise RiskBreachError(f"Open orders {state.open_order_count} at max {limits.max_open_orders}")


def update_high_water_mark(state: RiskState) -> None:
    if state.bankroll_cents > state.high_water_mark_cents:
        state.high_water_mark_cents = state.bankroll_cents
