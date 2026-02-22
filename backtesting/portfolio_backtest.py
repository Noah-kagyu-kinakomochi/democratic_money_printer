"""
Portfolio Backtester.

Simulates the FULL MoneyPrinter pipeline over historical data:
- All 12 models (3 strategies × 4 timeframes) vote democratically
- Multi-symbol portfolio with position tracking
- Long + short positions
- Cash management with $1,000 starting capital

Usage:
    python main.py simulate
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from models.market import SignalType
from strategy.base import StrategyModel
from strategy.democracy import DemocraticEngine

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class SimTrade:
    """A single simulated trade."""
    day: int  # day index in simulation
    date: datetime
    symbol: str
    side: str  # "BUY", "SELL", "SHORT", "COVER"
    qty: float
    price: float
    pnl: float = 0.0


@dataclass
class PortfolioBacktestResult:
    """Full portfolio backtest results."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_cash: float = 1_000.0
    final_value: float = 1_000.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    # Benchmark-relative metrics (evaluation mode)
    benchmark_return_pct: float = 0.0  # buy-and-hold return
    alpha_pct: float = 0.0  # model return - benchmark return
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    trades: list[SimTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_dates: list[datetime] = field(default_factory=list)
    per_symbol: dict = field(default_factory=dict)  # symbol -> {pnl, trades, ...}


class PortfolioBacktester:
    """
    Full-system portfolio backtester.

    Walks through 1 year of daily bars, runs democratic voting
    across all symbols, and simulates trade execution.
    """

    def __init__(
        self,
        democracy: DemocraticEngine,
        initial_cash: float = 1_000.0,
        position_size_pct: float = 0.1,
        slippage_pct: float = 0.001,
        does_short: bool = True,
    ):
        self.democracy = democracy
        self.initial_cash = initial_cash
        self.position_size_pct = position_size_pct
        self.slippage_pct = slippage_pct
        self.does_short = does_short

    def run(
        self,
        data_by_symbol: dict[str, dict[str, pd.DataFrame]],
        lookback_bars: int = 60,
    ) -> PortfolioBacktestResult:
        """
        Run a full portfolio backtest.

        Args:
            data_by_symbol: {symbol: {timeframe: DataFrame}}
                            Each DataFrame has [timestamp, open, high, low, close, volume]
            lookback_bars: Minimum bars before first signal

        Returns:
            PortfolioBacktestResult with full P&L and trade log
        """
        # Build a unified trading calendar from 1Day data
        trading_days = self._build_trading_calendar(data_by_symbol)
        if len(trading_days) < lookback_bars + 5:
            console.print("[red]Not enough trading days for simulation[/red]")
            return PortfolioBacktestResult(initial_cash=self.initial_cash)

        symbols = list(data_by_symbol.keys())

        # State
        cash = self.initial_cash
        positions: dict[str, float] = {}  # symbol -> qty (negative = short)
        entry_prices: dict[str, float] = {}  # symbol -> avg entry price
        all_trades: list[SimTrade] = []
        equity_curve: list[float] = []
        daily_dates: list[datetime] = []
        per_symbol_pnl: dict[str, float] = {s: 0.0 for s in symbols}
        per_symbol_trades: dict[str, int] = {s: 0 for s in symbols}

        console.print(f"\n[bold cyan]Portfolio Simulation[/bold cyan]")
        console.print(f"  Start Cash: ${self.initial_cash:,.2f}")
        console.print(f"  Symbols: {', '.join(symbols)}")
        console.print(f"  Trading Days: {len(trading_days)}")
        console.print(f"  Models: {len(self.democracy.models)}")
        console.print(f"  Short Selling: {'Enabled' if self.does_short else 'Disabled'}")
        console.print()

        # Walk through each trading day
        for day_idx in range(lookback_bars, len(trading_days) - 1):
            current_date = trading_days[day_idx]
            next_date = trading_days[day_idx + 1]

            # Get prices for today and tomorrow's open
            prices_today: dict[str, float] = {}
            prices_next_open: dict[str, float] = {}

            for symbol in symbols:
                day_data = data_by_symbol[symbol].get("1Day", pd.DataFrame())
                if day_data.empty:
                    continue

                # Current close
                mask_today = day_data["timestamp"] <= pd.Timestamp(current_date)
                if mask_today.any():
                    prices_today[symbol] = float(day_data.loc[mask_today].iloc[-1]["close"])

                # Next day's open
                mask_next = day_data["timestamp"] <= pd.Timestamp(next_date)
                if mask_next.any():
                    prices_next_open[symbol] = float(day_data.loc[mask_next].iloc[-1]["open"])

            # Calculate approximate equity for sizing (using yesterday's close)
            # This mimics "Account Equity" available at the start of the day
            current_equity = cash
            for sym, qty in positions.items():
                p = prices_today.get(sym, 0)
                if qty > 0:
                     current_equity += qty * p
                elif qty < 0:
                     # Short equity approximation (margin + pnl)
                     entry = entry_prices.get(sym, p)
                     margin = entry * abs(qty)
                     unrealized_pnl = (entry - p) * abs(qty)
                     current_equity += margin + unrealized_pnl

            # Run democratic voting for each symbol
            for symbol in symbols:
                if symbol not in prices_today or symbol not in prices_next_open:
                    continue

                # 🛑 Hard Stop Loss Check 🛑
                # Check if current position is down > 2%
                current_qty = positions.get(symbol, 0.0)
                if current_qty != 0:
                    entry = entry_prices.get(symbol, 0.0)
                    current_price = prices_today.get(symbol, 0.0)
                    if entry > 0:
                        if current_qty > 0:
                             # Long Position
                             pnl_pct = (current_price - entry) / entry
                             if pnl_pct < -0.02: # -2% Stop Loss
                                 # Force Close
                                 exec_price = prices_next_open[symbol]
                                 exec_price_adj = exec_price * (1 - self.slippage_pct)
                                 pnl = (exec_price_adj - entry) * current_qty
                                 cash += current_qty * exec_price_adj
                                 per_symbol_pnl[symbol] += pnl
                                 per_symbol_trades[symbol] += 1
                                 all_trades.append(SimTrade(
                                     day=day_idx, date=next_date, symbol=symbol,
                                     side="SELL (Stop Loss)", qty=current_qty,
                                     price=exec_price_adj, pnl=pnl,
                                 ))
                                 positions.pop(symbol, None)
                                 entry_prices.pop(symbol, None)
                                 continue # Skip voting for this symbol

                        elif current_qty < 0:
                             # Short Position
                             pnl_pct = (entry - current_price) / entry
                             if pnl_pct < -0.02: # -2% Stop Loss
                                 # Force Cover
                                 exec_price = prices_next_open[symbol]
                                 exec_price_adj = exec_price * (1 + self.slippage_pct)
                                 pnl = (entry - exec_price_adj) * abs(current_qty)
                                 margin = entry * abs(current_qty)
                                 cash += margin + pnl
                                 per_symbol_pnl[symbol] += pnl
                                 per_symbol_trades[symbol] += 1
                                 all_trades.append(SimTrade(
                                     day=day_idx, date=next_date, symbol=symbol,
                                     side="COVER (Stop Loss)", qty=abs(current_qty),
                                     price=exec_price_adj, pnl=pnl,
                                 ))
                                 positions.pop(symbol, None)
                                 entry_prices.pop(symbol, None)
                                 continue # Skip voting for this symbol

                # Build data windows for each timeframe
                data_by_tf = {}
                for tf, tf_data in data_by_symbol[symbol].items():
                    if tf_data.empty:
                        continue
                    mask = tf_data["timestamp"] <= pd.Timestamp(current_date)
                    window = tf_data.loc[mask].tail(lookback_bars).reset_index(drop=True)
                    if len(window) >= 10:  # need minimum data
                        data_by_tf[tf] = window

                if not data_by_tf:
                    continue

                # Vote
                try:
                    consensus = self.democracy.vote_multi_tf(symbol, data_by_tf)
                except Exception:
                    continue

                signal = consensus.signal_type
                current_qty = positions.get(symbol, 0.0)
                exec_price = prices_next_open[symbol]
                
                # Dynamic Sizing: Equity * MaxPct * Confidence
                base_trade_value = current_equity * self.position_size_pct
                scaled_trade_value = base_trade_value * consensus.confidence

                # Execute based on signal
                if signal == SignalType.BUY:
                    exec_price_adj = exec_price * (1 + self.slippage_pct)

                    if current_qty < 0:
                        # Cover short position — return margin ± P&L
                        pnl = (entry_prices[symbol] - exec_price_adj) * abs(current_qty)
                        margin = entry_prices[symbol] * abs(current_qty)
                        cash += margin + pnl  # return margin + profit (or minus loss)
                        per_symbol_pnl[symbol] += pnl
                        per_symbol_trades[symbol] += 1
                        all_trades.append(SimTrade(
                            day=day_idx, date=next_date, symbol=symbol,
                            side="COVER", qty=abs(current_qty),
                            price=exec_price_adj, pnl=pnl,
                        ))
                        positions.pop(symbol, None)
                        entry_prices.pop(symbol, None)

                    elif current_qty == 0:
                        # Open long
                        trade_value = scaled_trade_value
                        if trade_value > 10 and cash >= trade_value:
                            qty = trade_value / exec_price_adj
                            cash -= trade_value
                            positions[symbol] = qty
                            entry_prices[symbol] = exec_price_adj
                            all_trades.append(SimTrade(
                                day=day_idx, date=next_date, symbol=symbol,
                                side="BUY", qty=qty, price=exec_price_adj,
                            ))
                            per_symbol_trades[symbol] += 1

                elif signal == SignalType.SELL:
                    exec_price_adj = exec_price * (1 - self.slippage_pct)

                    if current_qty > 0:
                        # Close long position
                        pnl = (exec_price_adj - entry_prices[symbol]) * current_qty
                        cash += current_qty * exec_price_adj
                        per_symbol_pnl[symbol] += pnl
                        per_symbol_trades[symbol] += 1
                        all_trades.append(SimTrade(
                            day=day_idx, date=next_date, symbol=symbol,
                            side="SELL", qty=current_qty,
                            price=exec_price_adj, pnl=pnl,
                        ))
                        positions.pop(symbol, None)
                        entry_prices.pop(symbol, None)

                    elif current_qty == 0 and self.does_short:
                        # Open short — set aside margin collateral
                        trade_value = scaled_trade_value
                        if trade_value > 10 and cash >= trade_value:
                            qty = trade_value / exec_price_adj
                            cash -= trade_value  # short margin collateral
                            positions[symbol] = -qty
                            entry_prices[symbol] = exec_price_adj
                            all_trades.append(SimTrade(
                                day=day_idx, date=next_date, symbol=symbol,
                                side="SHORT", qty=qty, price=exec_price_adj,
                            ))
                            per_symbol_trades[symbol] += 1

            # Calculate end-of-day portfolio equity
            equity = cash
            for sym, qty in positions.items():
                price = prices_today.get(sym, 0)
                if qty > 0:
                    equity += qty * price
                elif qty < 0:
                    # Short equity = margin + unrealized P&L
                    entry = entry_prices.get(sym, price)
                    margin = entry * abs(qty)
                    unrealized_pnl = (entry - price) * abs(qty)
                    equity += margin + unrealized_pnl
            equity_curve.append(equity)
            daily_dates.append(current_date)

        # Close all remaining positions at last known prices
        last_day = trading_days[-1]
        for symbol, qty in list(positions.items()):
            day_data = data_by_symbol[symbol].get("1Day", pd.DataFrame())
            if day_data.empty:
                continue
            last_close = float(day_data.iloc[-1]["close"])

            if qty > 0:
                pnl = (last_close - entry_prices[symbol]) * qty
                cash += qty * last_close
                per_symbol_pnl[symbol] += pnl
                all_trades.append(SimTrade(
                    day=len(trading_days) - 1, date=last_day, symbol=symbol,
                    side="SELL (close)", qty=qty, price=last_close, pnl=pnl,
                ))
            elif qty < 0:
                pnl = (entry_prices[symbol] - last_close) * abs(qty)
                cash += entry_prices[symbol] * abs(qty) + pnl
                per_symbol_pnl[symbol] += pnl
                all_trades.append(SimTrade(
                    day=len(trading_days) - 1, date=last_day, symbol=symbol,
                    side="COVER (close)", qty=abs(qty), price=last_close, pnl=pnl,
                ))
        positions.clear()

        final_value = cash
        total_return_pct = (final_value - self.initial_cash) / self.initial_cash * 100

        # Calculate metrics
        winning = sum(1 for t in all_trades if t.pnl > 0)
        losing = sum(1 for t in all_trades if t.pnl < 0)
        mdd = self._max_drawdown(equity_curve)
        sharpe = self._sharpe_ratio(equity_curve)

        # Benchmark: equal-weight buy-and-hold across all symbols
        benchmark_return_pct = self._compute_benchmark(
            data_by_symbol, trading_days, lookback_bars, symbols
        )
        alpha_pct = total_return_pct - benchmark_return_pct

        # Sortino ratio from equity curve
        sortino = self._sortino_ratio(equity_curve)

        # Calmar ratio
        annualized_return = (1 + total_return_pct / 100) ** (252 / max(len(equity_curve), 1)) - 1
        calmar = annualized_return / mdd if mdd > 0 else 0.0

        result = PortfolioBacktestResult(
            start_date=trading_days[lookback_bars] if trading_days else None,
            end_date=trading_days[-1] if trading_days else None,
            initial_cash=self.initial_cash,
            final_value=final_value,
            total_return_pct=total_return_pct,
            total_trades=len(all_trades),
            winning_trades=winning,
            losing_trades=losing,
            max_drawdown=mdd,
            sharpe_ratio=sharpe,
            benchmark_return_pct=benchmark_return_pct,
            alpha_pct=alpha_pct,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            trades=all_trades,
            equity_curve=equity_curve,
            daily_dates=daily_dates,
            per_symbol={
                s: {"pnl": per_symbol_pnl[s], "trades": per_symbol_trades[s]}
                for s in symbols
            },
        )

        return result

    def _build_trading_calendar(
        self, data_by_symbol: dict[str, dict[str, pd.DataFrame]]
    ) -> list[datetime]:
        """Extract unique sorted trading days from 1Day data."""
        all_dates = set()
        for symbol, tf_data in data_by_symbol.items():
            day_data = tf_data.get("1Day", pd.DataFrame())
            if not day_data.empty:
                for ts in day_data["timestamp"]:
                    if isinstance(ts, pd.Timestamp):
                        all_dates.add(ts.to_pydatetime())
                    else:
                        all_dates.add(ts)
        return sorted(all_dates)

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
        return float(np.max(dd))

    @staticmethod
    def _sharpe_ratio(equity_curve: list[float]) -> float:
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1]
        if np.std(returns) == 0:
            return 0.0
        return float((np.mean(returns) / np.std(returns)) * np.sqrt(252))

    @staticmethod
    def _sortino_ratio(equity_curve: list[float]) -> float:
        """Sortino ratio: only penalizes downside volatility."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1]
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float(np.mean(returns) * np.sqrt(252)) if np.mean(returns) > 0 else 0.0
        downside_std = float(np.std(downside))
        if downside_std == 0:
            return 0.0
        return float((np.mean(returns) / downside_std) * np.sqrt(252))

    @staticmethod
    def _compute_benchmark(
        data_by_symbol: dict, trading_days: list, lookback_bars: int, symbols: list
    ) -> float:
        """
        Compute equal-weight buy-and-hold benchmark return.
        Simulates buying equal dollar amounts of all symbols at start
        and holding to end.
        """
        if not trading_days or lookback_bars >= len(trading_days) - 1:
            return 0.0

        start_date = trading_days[lookback_bars]
        end_date = trading_days[-1]
        total_return = 0.0
        count = 0

        for symbol in symbols:
            day_data = data_by_symbol[symbol].get("1Day", pd.DataFrame())
            if day_data.empty:
                continue

            # Find start and end prices
            mask_start = day_data["timestamp"] <= pd.Timestamp(start_date)
            mask_end = day_data["timestamp"] <= pd.Timestamp(end_date)

            if not mask_start.any() or not mask_end.any():
                continue

            start_price = float(day_data.loc[mask_start].iloc[-1]["close"])
            end_price = float(day_data.loc[mask_end].iloc[-1]["close"])

            if start_price > 0:
                total_return += (end_price - start_price) / start_price * 100
                count += 1

        return total_return / count if count > 0 else 0.0


def print_backtest_report(result: PortfolioBacktestResult) -> None:
    """Pretty-print the backtest results."""
    console.rule("[bold cyan]Portfolio Backtest Results")

    if result.start_date is None or result.end_date is None:
        console.print("\n  ⚠️  No simulation data — not enough trading days")
        console.rule("[bold cyan]End of Report")
        return

    # Summary
    ret_color = "green" if result.total_return_pct >= 0 else "red"
    alpha_color = "green" if result.alpha_pct >= 0 else "red"
    console.print(f"\n  📅 Period: {result.start_date:%Y-%m-%d} → {result.end_date:%Y-%m-%d}")
    console.print(f"  💰 Start: ${result.initial_cash:,.2f}")
    console.print(f"  💵 End:   ${result.final_value:,.2f}")
    console.print(f"  📊 Return: [{ret_color}]{result.total_return_pct:+.2f}%[/{ret_color}]")
    console.print(f"  📈 Benchmark (Buy & Hold): {result.benchmark_return_pct:+.2f}%")
    console.print(f"  🎯 Alpha: [{alpha_color}]{result.alpha_pct:+.2f}%[/{alpha_color}]")
    console.print(f"  📉 Max Drawdown: {result.max_drawdown:.2%}")
    console.print(f"  📈 Sharpe: {result.sharpe_ratio:.3f}  |  Sortino: {result.sortino_ratio:.3f}  |  Calmar: {result.calmar_ratio:.3f}")
    console.print(f"  🔄 Total Trades: {result.total_trades}")
    if result.total_trades > 0:
        win_rate = result.winning_trades / result.total_trades * 100
        console.print(f"  ✅ Winning: {result.winning_trades} ({win_rate:.0f}%)")
        console.print(f"  ❌ Losing:  {result.losing_trades}")

    # Per-symbol breakdown
    if result.per_symbol:
        table = Table(title="\nPer-Symbol Breakdown")
        table.add_column("Symbol", style="cyan")
        table.add_column("P&L", justify="right")
        table.add_column("Trades", justify="right")

        for symbol, data in sorted(result.per_symbol.items()):
            pnl = data["pnl"]
            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                symbol,
                f"[{pnl_style}]${pnl:+,.2f}[/{pnl_style}]",
                str(data["trades"]),
            )
        console.print(table)

    # Recent trades
    if result.trades:
        trade_table = Table(title="\nTrade Log (last 20)")
        trade_table.add_column("Date", style="dim")
        trade_table.add_column("Symbol", style="cyan")
        trade_table.add_column("Side")
        trade_table.add_column("Qty", justify="right")
        trade_table.add_column("Price", justify="right")
        trade_table.add_column("P&L", justify="right")

        for t in result.trades[-20:]:
            pnl_str = f"${t.pnl:+,.2f}" if t.pnl != 0 else "—"
            pnl_style = "green" if t.pnl > 0 else ("red" if t.pnl < 0 else "dim")
            side_style = "green" if "BUY" in t.side else "red"
            date_str = t.date.strftime("%Y-%m-%d") if isinstance(t.date, datetime) else str(t.date)
            trade_table.add_row(
                date_str,
                t.symbol,
                f"[{side_style}]{t.side}[/{side_style}]",
                f"{t.qty:.4f}",
                f"${t.price:.2f}",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
            )
        console.print(trade_table)

    console.rule("[bold cyan]End of Report")
