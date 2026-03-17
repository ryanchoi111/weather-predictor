import asyncio
import subprocess
from pathlib import Path

import yaml
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .data.db import Database
from .data.logger import setup_logging, send_discord_alert, get_logger
from .forecast_client import ForecastClient, ForecastRequest
from .kalshi.client import KalshiClient, KalshiConfig
from .kalshi.markets import get_weather_events, get_event_markets, parse_temperature_buckets
from .kalshi.orders import create_order, get_balance, get_positions
from .strategy.probability import compute_bucket_probabilities
from .strategy.signals import generate_signals
from .strategy.sizing import kelly_size
from .strategy.risk import RiskLimits, RiskState, RiskBreachError, check_pre_trade, update_high_water_mark

log = get_logger("bot")


class WeatherBot:
    def __init__(self):
        self.config = self._load_config()
        self.cities = self._load_cities()
        self.kalshi = KalshiClient(KalshiConfig())
        self.db = Database()

        forecast_cfg = self.config.get("forecast", {})
        self.forecast = ForecastClient(
            provider=forecast_cfg.get("provider", "mock"),
            base_url=forecast_cfg.get("service_url", "http://localhost:8000"),
            runpod_endpoint_id=forecast_cfg.get("runpod_endpoint_id", ""),
            runpod_api_key=forecast_cfg.get("runpod_api_key", ""),
            timeout=forecast_cfg.get("timeout_seconds", 300),
            max_retries=forecast_cfg.get("max_retries", 3),
        )

        risk_cfg = self.config.get("risk", {})
        self.risk_limits = RiskLimits(**risk_cfg) if risk_cfg else RiskLimits()
        self.risk_state = RiskState()
        self.discord_url = self.config.get("discord_webhook_url", "")

    def _load_config(self) -> dict:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_cities(self) -> list[dict]:
        cities_path = Path(__file__).parent.parent / "config" / "cities.yaml"
        if cities_path.exists():
            with open(cities_path) as f:
                data = yaml.safe_load(f) or {}
                return data.get("cities", [])
        return []

    async def run_cycle(self):
        """Main trading cycle: forecast -> discover -> signal -> trade."""
        log.info("cycle_start")

        balance_cents = await get_balance(self.kalshi)
        self.risk_state.bankroll_cents = balance_cents
        update_high_water_mark(self.risk_state)

        strategy_cfg = self.config.get("strategy", {})

        for city in self.cities:
            try:
                await self._process_city(city, balance_cents, strategy_cfg)
            except Exception as e:
                log.error("city_error", city=city["name"], error=str(e))
                await send_discord_alert(self.discord_url, f"Error processing {city['name']}: {e}", "error")

        log.info("cycle_complete")

    async def _process_city(self, city: dict, balance_cents: int, strategy_cfg: dict):
        log.info("processing_city", city=city["name"])

        request = ForecastRequest(
            latitude=city["latitude"],
            longitude=city["longitude"],
        )
        forecast = await self.forecast.get_forecast(request)
        forecast_id = await self.db.save_forecast(
            city=city["name"],
            model=forecast.model_name,
            temps=forecast.temperatures_f,
            lead_hours=forecast.lead_hours,
        )

        events = await get_weather_events(self.kalshi, city["series_ticker"])
        if not events:
            log.info("no_events", city=city["name"])
            return

        for event in events:
            markets = await get_event_markets(self.kalshi, event["event_ticker"])
            buckets = parse_temperature_buckets(markets)
            if not buckets:
                continue

            bucket_ranges = [(b.low_temp, b.high_temp) for b in buckets]
            model_probs = compute_bucket_probabilities(
                forecast.temperatures_f,
                bucket_ranges,
                min_prob=strategy_cfg.get("min_probability_floor", 0.005),
            )

            signals = generate_signals(
                tickers=[b.ticker for b in buckets],
                model_probs=model_probs,
                market_prices_cents=[b.yes_price for b in buckets],
                ensemble_std=forecast.std_f,
                edge_threshold=strategy_cfg.get("edge_threshold_cents", 5),
                max_std=strategy_cfg.get("max_ensemble_std_f", 8.0),
                min_volume=[b.volume for b in buckets],
                min_liquidity=strategy_cfg.get("min_liquidity_contracts", 10),
            )

            for signal in signals:
                contracts = kelly_size(
                    model_prob=signal.model_prob,
                    market_price_cents=signal.market_price_cents,
                    side=signal.side,
                    bankroll_cents=balance_cents,
                    kelly_fraction=strategy_cfg.get("kelly_fraction", 0.25),
                    max_position_pct=self.risk_limits.max_single_position_pct,
                )
                if contracts == 0:
                    continue

                price = signal.market_price_cents if signal.side == "yes" else (100 - signal.market_price_cents)
                order_cost = price * contracts

                try:
                    check_pre_trade(self.risk_limits, self.risk_state, order_cost, event["event_ticker"])
                except RiskBreachError as e:
                    log.warning("risk_breach", signal=signal.ticker, error=str(e))
                    await send_discord_alert(self.discord_url, f"Risk breach: {e}", "warning")
                    continue

                signal_id = await self.db.save_signal(
                    forecast_id=forecast_id,
                    city=city["name"],
                    ticker=signal.ticker,
                    side=signal.side,
                    edge_cents=signal.edge_cents,
                    model_prob=signal.model_prob,
                    market_price_cents=signal.market_price_cents,
                    contracts=contracts,
                )

                try:
                    result = await create_order(
                        self.kalshi, signal.ticker, signal.side, price, contracts,
                    )
                    order_id = result.get("order", {}).get("order_id", "unknown")
                    await self.db.save_trade(
                        signal_id=signal_id,
                        ticker=signal.ticker,
                        side=signal.side,
                        action="buy",
                        price_cents=price,
                        count=contracts,
                        order_id=order_id,
                        status="placed",
                    )
                    self.risk_state.open_order_count += 1
                    event_ticker = event["event_ticker"]
                    self.risk_state.event_exposures[event_ticker] = (
                        self.risk_state.event_exposures.get(event_ticker, 0) + order_cost
                    )
                    log.info("order_placed", ticker=signal.ticker, side=signal.side,
                             price=price, contracts=contracts, edge=signal.edge_cents)
                    await send_discord_alert(
                        self.discord_url,
                        f"Order: {signal.side.upper()} {contracts}x {signal.ticker} @{price}¢ (edge {signal.edge_cents:.1f}¢)",
                    )
                except Exception as e:
                    log.error("order_failed", ticker=signal.ticker, error=str(e))

    async def start(self):
        setup_logging()
        await self.db.init()

        caffeinate = subprocess.Popen(
            ["caffeinate", "-di"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        scheduler = AsyncIOScheduler()
        trading_cfg = self.config.get("trading", {})
        tz = trading_cfg.get("timezone", "America/New_York")

        for time_str in trading_cfg.get("schedule_times", ["06:00", "12:00", "16:00", "20:00"]):
            hour, minute = time_str.split(":")
            scheduler.add_job(
                self.run_cycle,
                CronTrigger(hour=int(hour), minute=int(minute), timezone=tz),
            )

        scheduler.start()
        log.info("bot_started", schedule=trading_cfg.get("schedule_times"))
        await send_discord_alert(self.discord_url, "Weather bot started")

        try:
            await self.run_cycle()
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            log.info("bot_stopping")
        finally:
            scheduler.shutdown()
            caffeinate.terminate()
            await self.kalshi.close()
            await self.db.close()


def main():
    bot = WeatherBot()
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
