import asyncio
import discord
from discord import ButtonStyle, Interaction, Embed, Color
from discord.ui import View, Button, button

from .data.db import Database
from .data.logger import get_logger
from .kalshi.client import KalshiClient
from .kalshi.orders import create_order, get_balance
from .strategy.risk import RiskLimits, RiskState, RiskBreachError, check_pre_trade

log = get_logger("discord_bot")


class ApprovalView(View):
    """Persistent view with Approve/Reject buttons for trade signals."""

    def __init__(self, bot: "ApprovalBot"):
        super().__init__(timeout=None)
        self.bot = bot

    @button(label="Approve", style=ButtonStyle.green, custom_id="trade_approve")
    async def approve(self, interaction: Interaction, btn: Button):
        await self.bot.handle_approval(interaction, approved=True)

    @button(label="Reject", style=ButtonStyle.red, custom_id="trade_reject")
    async def reject(self, interaction: Interaction, btn: Button):
        await self.bot.handle_approval(interaction, approved=False)


class ApprovalBot(discord.Client):
    """Discord bot that sends trade approval requests and handles responses."""

    def __init__(
        self,
        bot_token: str,
        channel_id: int,
        db: Database,
        kalshi: KalshiClient,
        risk_limits: RiskLimits,
        risk_state: RiskState,
        forecast_channel_id: int | None = None,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.bot_token = bot_token
        self.channel_id = channel_id
        self.forecast_channel_id = forecast_channel_id
        self.db = db
        self.kalshi = kalshi
        self.risk_limits = risk_limits
        self.risk_state = risk_state
        self._channel: discord.TextChannel | None = None
        self._forecast_channel: discord.TextChannel | None = None
        self._ready_event = asyncio.Event()

    async def on_ready(self):
        log.info("discord_bot_ready", user=str(self.user))
        self._channel = self.get_channel(self.channel_id)
        if not self._channel:
            log.error("discord_channel_not_found", channel_id=self.channel_id)
            return

        if self.forecast_channel_id:
            self._forecast_channel = self.get_channel(self.forecast_channel_id)
            if not self._forecast_channel:
                log.error("discord_forecast_channel_not_found", channel_id=self.forecast_channel_id)
        else:
            self._forecast_channel = self._channel

        self._ready_event.set()

        # Re-register persistent view for pending approvals surviving restart
        pending = await self.db.get_pending_approvals()
        if pending:
            self.add_view(ApprovalView(self))
            log.info("restored_pending_approvals", count=len(pending))

    async def start_bot(self):
        """Start the Discord bot (non-blocking, intended for asyncio.create_task)."""
        await self.start(self.bot_token)

    async def send_approval_request(
        self,
        signal_id: int,
        city: str,
        ticker: str,
        contract_title: str,
        side: str,
        contracts: int,
        price_cents: int,
        edge_cents: float,
        model_prob: float,
        market_price_cents: int,
        mean_temp: float,
        std_temp: float,
        n_members: int,
        event_ticker: str,
        order_cost_cents: int,
    ) -> str:
        """Send an approval embed to Discord. Returns the message ID."""
        await self._ready_event.wait()
        if not self._channel:
            self._channel = self.get_channel(self.channel_id)
        if not self._channel:
            raise RuntimeError(f"Discord channel {self.channel_id} not found")

        cost_dollars = order_cost_cents / 100

        embed = Embed(
            title=f"[TRADE SIGNAL] {city}",
            color=Color.blue(),
        )
        embed.add_field(name="Ticker", value=f"{ticker}\n{contract_title}", inline=False)
        embed.add_field(name="Side", value=side.upper(), inline=True)
        embed.add_field(name="Contracts", value=str(contracts), inline=True)
        embed.add_field(name="Price", value=f"{price_cents}c", inline=True)
        embed.add_field(
            name="Edge",
            value=f"+{edge_cents:.1f}c | Model: {model_prob*100:.1f}% vs Market: {market_price_cents}c",
            inline=False,
        )
        embed.add_field(
            name="Forecast",
            value=f"{mean_temp:.1f}F mean | {std_temp:.1f}F std ({n_members} members)",
            inline=False,
        )
        embed.add_field(name="Cost", value=f"${cost_dollars:.2f}", inline=False)

        view = ApprovalView(self)
        msg = await self._channel.send(embed=embed, view=view)

        await self.db.save_pending_approval(
            signal_id=signal_id,
            discord_message_id=str(msg.id),
            city=city,
            ticker=ticker,
            side=side,
            price_cents=price_cents,
            contracts=contracts,
            event_ticker=event_ticker,
            order_cost_cents=order_cost_cents,
        )
        log.info("approval_request_sent", ticker=ticker, message_id=msg.id)
        return str(msg.id)

    async def send_forecast_summary(
        self,
        city: str,
        mean_temp: float,
        std_temp: float,
        n_members: int,
        buckets: list,
        model_probs: list[float],
    ) -> None:
        """Send a forecast overview embed showing model prediction vs all contracts."""
        await self._ready_event.wait()
        channel = self._forecast_channel or self._channel
        if not channel:
            return

        embed = Embed(
            title=f"[FORECAST] {city}",
            description=f"**{mean_temp:.1f}F** mean | {std_temp:.1f}F std ({n_members} members)",
            color=Color.teal(),
        )

        lines = []
        for bucket, prob in zip(buckets, model_probs):
            market = bucket.yes_price
            edge = prob * 100 - market
            if edge >= 0:
                side = "YES"
                abs_edge = edge
            else:
                side = "NO"
                abs_edge = -edge
            lines.append(
                f"`{bucket.ticker}` {side} | Market: {market}c | Model: {prob*100:.0f}% | Edge: +{abs_edge:.0f}c\n  _{bucket.title}_\n"
            )

        # Discord embed field max is 1024 chars — split if needed
        chunk = ""
        field_num = 1
        for line in lines:
            if len(chunk) + len(line) + 1 > 1024:
                embed.add_field(name=f"Contracts ({field_num})", value=chunk, inline=False)
                chunk = ""
                field_num += 1
            chunk += line + "\n"
        if chunk:
            label = "Contracts" if field_num == 1 else f"Contracts ({field_num})"
            embed.add_field(name=label, value=chunk, inline=False)

        await channel.send(embed=embed)
        log.info("forecast_summary_sent", city=city)

    async def send_forecast_warning(
        self, city: str, model_mean: float, observed: float, deviation: float,
    ) -> None:
        """Send a warning when model forecast deviates significantly from observed temp."""
        await self._ready_event.wait()
        channel = self._forecast_channel or self._channel
        if not channel:
            return

        embed = Embed(
            title=f"[WARNING] {city} — Forecast Deviation",
            description=(
                f"Model: **{model_mean:.1f}F** | Observed: **{observed:.1f}F** | "
                f"Deviation: **{deviation:.1f}F**\n\n"
                f"Skipping signals for {city} — model may be unreliable."
            ),
            color=Color.orange(),
        )
        await channel.send(embed=embed)
        log.warning("forecast_warning_sent", city=city, deviation=deviation)

    async def send_cycle_summary(self, all_signals: list[dict]) -> None:
        """Send a ranked summary of best signals across all cities."""
        await self._ready_event.wait()
        channel = self._channel
        if not channel:
            return

        # Sort by edge descending
        ranked = sorted(all_signals, key=lambda s: s["edge_cents"], reverse=True)[:5]

        embed = Embed(
            title="[CYCLE SUMMARY] Top Signals",
            description=f"{len(all_signals)} signals across all cities",
            color=Color.gold(),
        )

        lines = []
        for i, s in enumerate(ranked, 1):
            lines.append(
                f"**{i}.** `{s['ticker']}` — {s['city']}\n"
                f"  {s['side'].upper()} | Edge: +{s['edge_cents']:.1f}c | "
                f"Model: {s['model_prob']*100:.0f}% vs Market: {s['market_price_cents']}c | "
                f"Std: {s['confidence']:.1f}F\n"
                f"  _{s['title']}_\n"
            )

        chunk = ""
        field_num = 1
        for line in lines:
            if len(chunk) + len(line) + 1 > 1024:
                embed.add_field(name=f"Rankings ({field_num})", value=chunk, inline=False)
                chunk = ""
                field_num += 1
            chunk += line
        if chunk:
            label = "Rankings" if field_num == 1 else f"Rankings ({field_num})"
            embed.add_field(name=label, value=chunk, inline=False)

        await channel.send(embed=embed)
        log.info("cycle_summary_sent", total_signals=len(all_signals))

    async def send_settlement_summary(
        self, new_settled: int, new_traded: int, pnl_str: str,
    ) -> None:
        """Send settlement P&L notification."""
        await self._ready_event.wait()
        channel = self.get_channel(self.channel_id)
        if not channel:
            return

        color = Color.green() if pnl_str.startswith("+") else Color.red()
        embed = Embed(
            title="Settlements",
            description=(
                f"**{new_settled}** contracts settled\n"
                f"**{new_traded}** were our trades\n"
                f"**P&L: {pnl_str}**"
            ),
            color=color,
        )

        # Get cumulative P&L
        try:
            summary = await self.db.get_trade_pnl_summary()
            if summary and summary.get("total_trades"):
                total_pnl = summary["total_pnl_cents"] or 0
                wins = summary["wins"] or 0
                total = summary["total_trades"] or 1
                cum_str = f"+${total_pnl/100:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl)/100:.2f}"
                embed.add_field(
                    name="Cumulative",
                    value=f"Record: {wins}/{total} ({wins/total:.0%})\nTotal P&L: {cum_str}",
                    inline=False,
                )
        except Exception:
            pass

        await channel.send(embed=embed)
        log.info("settlement_summary_sent", pnl=pnl_str)

    async def handle_approval(self, interaction: Interaction, approved: bool):
        """Handle Approve/Reject button press."""
        message_id = str(interaction.message.id)
        approval = await self.db.get_pending_approval(message_id)

        if not approval:
            await interaction.response.send_message("Unknown approval request.", ephemeral=True)
            return

        if approval["status"] != "pending":
            await interaction.response.send_message(
                f"Already handled: {approval['status'].upper()}", ephemeral=True
            )
            return

        if not approved:
            await self.db.resolve_approval(approval["id"], "rejected")
            embed = interaction.message.embeds[0].copy()
            embed.color = Color.light_grey()
            embed.set_footer(text="REJECTED")
            await interaction.response.edit_message(embed=embed, view=None)
            log.info("trade_rejected", ticker=approval["ticker"])
            return

        # Approved — re-check risk limits then execute
        try:
            balance = await get_balance(self.kalshi)
            self.risk_state.bankroll_cents = balance
            check_pre_trade(
                self.risk_limits,
                self.risk_state,
                approval["order_cost_cents"],
                approval["event_ticker"],
            )
        except RiskBreachError as e:
            await self.db.resolve_approval(approval["id"], "risk_blocked")
            embed = interaction.message.embeds[0].copy()
            embed.color = Color.orange()
            embed.set_footer(text=f"BLOCKED: {e}")
            await interaction.response.edit_message(embed=embed, view=None)
            log.warning("approval_risk_blocked", ticker=approval["ticker"], error=str(e))
            return

        try:
            result = await create_order(
                self.kalshi,
                approval["ticker"],
                approval["side"],
                approval["price_cents"],
                approval["contracts"],
            )
            order_id = result.get("order", {}).get("order_id", "unknown")
            await self.db.save_trade(
                signal_id=approval["signal_id"],
                ticker=approval["ticker"],
                side=approval["side"],
                action="buy",
                price_cents=approval["price_cents"],
                count=approval["contracts"],
                order_id=order_id,
                status="placed",
            )
            self.risk_state.open_order_count += 1
            self.risk_state.event_exposures[approval["event_ticker"]] = (
                self.risk_state.event_exposures.get(approval["event_ticker"], 0)
                + approval["order_cost_cents"]
            )
            await self.db.resolve_approval(approval["id"], "approved")

            embed = interaction.message.embeds[0].copy()
            embed.color = Color.green()
            embed.set_footer(text=f"APPROVED — Order {order_id}")
            await interaction.response.edit_message(embed=embed, view=None)
            log.info("trade_approved", ticker=approval["ticker"], order_id=order_id)

        except Exception as e:
            await self.db.resolve_approval(approval["id"], "error")
            embed = interaction.message.embeds[0].copy()
            embed.color = Color.red()
            embed.set_footer(text=f"ORDER FAILED: {e}")
            await interaction.response.edit_message(embed=embed, view=None)
            log.error("approval_order_failed", ticker=approval["ticker"], error=str(e))
