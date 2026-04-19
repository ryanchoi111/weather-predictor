import asyncio
import os
import subprocess
from pathlib import Path

import yaml

# Load .env into os.environ
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
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
from .discord_bot import ApprovalBot
from .nws import get_current_temp_f
from .calibration.pipeline import run_calibration_pipeline
from .calibration.feedback import load_calibration_model, apply_calibration
from .settlement import process_settlements

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
            base_url=forecast_cfg.get("service_url", os.getenv("FORECAST_SERVICE_URL", "http://localhost:8000")),
            runpod_endpoint_id=forecast_cfg.get("runpod_endpoint_id", "") or os.getenv("RUNPOD_ENDPOINT_ID", ""),
            runpod_api_key=forecast_cfg.get("runpod_api_key", "") or os.getenv("RUNPOD_API_KEY", ""),
            timeout=forecast_cfg.get("timeout_seconds", 300),
            max_retries=forecast_cfg.get("max_retries", 3),
        )

        risk_cfg = self.config.get("risk", {})
        self.risk_limits = RiskLimits(**risk_cfg) if risk_cfg else RiskLimits()
        self.risk_state = RiskState()
        self.discord_url = self.config.get("discord_webhook_url", "")

        self.calibration_cfg = self._load_calibration_config()

        discord_cfg = self.config.get("discord", {})
        bot_token = discord_cfg.get("bot_token", "")
        channel_id = discord_cfg.get("channel_id", "")
        self.approval_bot: ApprovalBot | None = None
        forecast_channel_id = discord_cfg.get("forecast_channel_id", "")
        if bot_token and channel_id:
            self.approval_bot = ApprovalBot(
                bot_token=bot_token,
                channel_id=int(channel_id),
                db=self.db,
                kalshi=self.kalshi,
                risk_limits=self.risk_limits,
                risk_state=self.risk_state,
                forecast_channel_id=int(forecast_channel_id) if forecast_channel_id else None,
            )

    def _load_config(self) -> dict:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_calibration_config(self) -> dict:
        config_path = Path(__file__).parent.parent / "config" / "calibration.yaml"
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
        """Main trading cycle: settlements -> forecast -> discover -> signal -> trade."""
        log.info("cycle_start")

        # Check for newly settled contracts (all weather markets, not just our trades)
        try:
            settlement_summary = await process_settlements(self.kalshi, self.db)
            if settlement_summary["new_settlements"] > 0:
                log.info("settlements_recorded", **settlement_summary)
                if self.approval_bot and settlement_summary["new_traded"] > 0:
                    pnl = settlement_summary["total_pnl_cents"]
                    pnl_str = f"+${pnl/100:.2f}" if pnl >= 0 else f"-${abs(pnl)/100:.2f}"
                    await self.approval_bot.send_settlement_summary(
                        new_settled=settlement_summary["new_settlements"],
                        new_traded=settlement_summary["new_traded"],
                        pnl_str=pnl_str,
                    )
        except Exception as e:
            log.warning("settlement_check_failed", error=str(e))

        balance_cents = await get_balance(self.kalshi)
        self.risk_state.bankroll_cents = balance_cents
        update_high_water_mark(self.risk_state)

        strategy_cfg = self.config.get("strategy", {})
        all_signals = []

        for city in self.cities:
            try:
                city_signals = await self._process_city(city, balance_cents, strategy_cfg)
                all_signals.extend(city_signals)
            except Exception as e:
                log.error("city_error", city=city["name"], error=str(e))
                await send_discord_alert(self.discord_url, f"Error processing {city['name']}: {e}", "error")

        if self.approval_bot and all_signals:
            try:
                await self.approval_bot.send_cycle_summary(all_signals)
            except Exception as e:
                log.error("cycle_summary_failed", error=str(e))

        log.info("cycle_complete")

    async def _process_city(self, city: dict, balance_cents: int, strategy_cfg: dict) -> list[dict]:
        log.info("processing_city", city=city["name"])
        collected_signals = []

        request = ForecastRequest(
            latitude=city["latitude"],
            longitude=city["longitude"],
        )
        forecast = await self.forecast.get_forecast(request)

        # Sanity check: compare model forecast vs current observed temp
        nws_station = city.get("nws_station", "")
        observed_f = await get_current_temp_f(nws_station) if nws_station else None
        max_deviation = strategy_cfg.get("max_forecast_deviation_f", 10.0)
        if observed_f is not None:
            deviation = abs(observed_f - forecast.mean_f)
            if deviation > max_deviation:
                log.warning(
                    "forecast_deviation",
                    city=city["name"],
                    model_mean=f"{forecast.mean_f:.1f}F",
                    observed=f"{observed_f:.1f}F",
                    deviation=f"{deviation:.1f}F",
                )
                if self.approval_bot:
                    await self.approval_bot.send_forecast_warning(
                        city=city["name"],
                        model_mean=forecast.mean_f,
                        observed=observed_f,
                        deviation=deviation,
                    )
                return collected_signals

        forecast_id = await self.db.save_forecast(
            city=city["name"],
            model=forecast.model_name,
            temps=forecast.temperatures_f,
            lead_hours=forecast.lead_hours,
        )

        from datetime import date as date_cls
        await self.db.save_forecast_actual(
            forecast_id=forecast_id,
            city=city["name"],
            forecast_date=date_cls.today().isoformat(),
            model_mean_f=forecast.mean_f,
        )

        # Run calibration (bias + METAR) once per city, reuse calibrated ensemble for all events
        cal_cfg = self.calibration_cfg
        bias_enabled = cal_cfg.get("bias_correction", {}).get("enabled", False)
        metar_enabled = cal_cfg.get("metar_fusion", {}).get("enabled", False)
        calibrated_temps = None
        if bias_enabled or metar_enabled:
            from datetime import datetime
            from .calibration.bias_correction import load_bias_model, correct_ensemble, get_season
            from .calibration.metar_client import fetch_todays_observations
            from .calibration.metar_fusion import fuse_metar_with_ensemble
            import numpy as np

            calibrated = np.array(forecast.temperatures_f)
            target_date = datetime.now()
            season = get_season(target_date.date())

            if bias_enabled:
                bias_model = load_bias_model(
                    city["name"], season,
                    model_dir=cal_cfg.get("bias_correction", {}).get("model_dir", "./data/bias_models/"),
                )
                if bias_model is not None:
                    calibrated = correct_ensemble(calibrated, bias_model)
                    log.info("bias_correction_applied", city=city["name"], season=season,
                             mae_before=bias_model.mae_before, mae_after=bias_model.mae_after)

            if metar_enabled:
                metar_station = city.get("metar_station")
                if metar_station:
                    try:
                        observations = await fetch_todays_observations(metar_station)
                        if observations:
                            fusion = fuse_metar_with_ensemble(
                                calibrated, observations,
                                forecast_hour=target_date.hour,
                                max_shift_f=cal_cfg.get("metar_fusion", {}).get("max_shift_f", 5.0),
                            )
                            calibrated = fusion.updated_ensemble
                    except Exception as e:
                        log.warning("metar_fusion_skipped", city=city["name"], error=str(e))

            if np.std(calibrated) < 0.1:
                log.warning("ensemble_degenerate", city=city["name"], std=f"{np.std(calibrated):.3f}F")
                calibrated = calibrated + np.random.normal(0, 0.5, len(calibrated))

            calibrated_temps = calibrated.tolist()
            log.info("calibration_complete", city=city["name"], bias=bias_enabled, metar=metar_enabled,
                     mean=f"{np.mean(calibrated):.1f}F", std=f"{np.std(calibrated):.1f}F")

        events = await get_weather_events(self.kalshi, city["series_ticker"])
        if not events:
            log.info("no_events", city=city["name"])
            return collected_signals

        for event in events:
            markets = await get_event_markets(self.kalshi, event["event_ticker"])
            buckets = parse_temperature_buckets(markets)
            if not buckets:
                continue

            bucket_ranges = [(b.low_temp, b.high_temp) for b in buckets]

            if calibrated_temps is not None:
                from .calibration.ensemble_calibration import compute_calibrated_bucket_probabilities
                model_probs = compute_calibrated_bucket_probabilities(
                    np.array(calibrated_temps),
                    bucket_ranges,
                    kde_bandwidth=cal_cfg.get("ensemble", {}).get("kde_bandwidth", 0.3),
                    min_prob=strategy_cfg.get("min_probability_floor", 0.005),
                )
            else:
                model_probs = compute_bucket_probabilities(
                    forecast.temperatures_f,
                    bucket_ranges,
                    min_prob=strategy_cfg.get("min_probability_floor", 0.005),
                )

            # Apply Platt scaling if calibration model exists (learned from settled contracts)
            platt_model = load_calibration_model(city=city["name"])
            if platt_model:
                model_probs = apply_calibration(model_probs, platt_model)
                log.info("platt_scaling_applied", city=city["name"],
                         brier_improvement=f"{platt_model.brier_before - platt_model.brier_after:.4f}")

            # Save model prediction for EVERY bucket (for calibration feedback)
            try:
                await self.db.save_bucket_predictions(
                    forecast_id=forecast_id,
                    city=city["name"],
                    event_ticker=event["event_ticker"],
                    tickers=[b.ticker for b in buckets],
                    titles=[b.title for b in buckets],
                    low_temps=[b.low_temp for b in buckets],
                    high_temps=[b.high_temp for b in buckets],
                    model_probs=model_probs,
                    market_prices=[b.yes_price for b in buckets],
                )
            except Exception as e:
                log.warning("bucket_predictions_save_failed", city=city["name"], error=str(e))

            if self.approval_bot:
                try:
                    await self.approval_bot.send_forecast_summary(
                        city=city["name"],
                        mean_temp=forecast.mean_f,
                        std_temp=forecast.std_f,
                        n_members=len(forecast.temperatures_f),
                        buckets=buckets,
                        model_probs=model_probs,
                    )
                except Exception as e:
                    log.error("forecast_summary_failed", city=city["name"], error=str(e))

            signals = generate_signals(
                tickers=[b.ticker for b in buckets],
                titles=[b.title for b in buckets],
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

                collected_signals.append({
                    "city": city["name"],
                    "ticker": signal.ticker,
                    "title": signal.title,
                    "side": signal.side,
                    "edge_cents": signal.edge_cents,
                    "model_prob": signal.model_prob,
                    "market_price_cents": signal.market_price_cents,
                    "contracts": contracts,
                    "confidence": signal.confidence,
                })

                if self.approval_bot:
                    try:
                        await self.approval_bot.send_approval_request(
                            signal_id=signal_id,
                            city=city["name"],
                            ticker=signal.ticker,
                            contract_title=signal.title,
                            side=signal.side,
                            contracts=contracts,
                            price_cents=price,
                            edge_cents=signal.edge_cents,
                            model_prob=signal.model_prob,
                            market_price_cents=signal.market_price_cents,
                            mean_temp=forecast.mean_f,
                            std_temp=forecast.std_f,
                            n_members=len(forecast.temperatures_f),
                            event_ticker=event["event_ticker"],
                            order_cost_cents=order_cost,
                        )
                    except Exception as e:
                        log.error("approval_request_failed", ticker=signal.ticker, error=str(e))
                else:
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

        return collected_signals

    async def start(self):
        setup_logging()
        await self.db.init()

        caffeinate = subprocess.Popen(
            ["caffeinate", "-di"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        scheduler = AsyncIOScheduler(job_defaults={"misfire_grace_time": 3600, "coalesce": True})
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

        if self.approval_bot:
            asyncio.create_task(self.approval_bot.start_bot())
            log.info("discord_approval_bot_starting")

        try:
            await self.run_cycle()
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            log.info("bot_stopping")
        finally:
            scheduler.shutdown()
            caffeinate.terminate()
            if self.approval_bot and not self.approval_bot.is_closed():
                await self.approval_bot.close()
            await self.kalshi.close()
            await self.db.close()


def main():
    bot = WeatherBot()
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
