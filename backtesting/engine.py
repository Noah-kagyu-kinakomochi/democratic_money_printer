"""
Backtesting Engine.

Simulates trading a single strategy model over historical data.
Walks through bars day-by-day, runs the model's analyze(),
simulates fills at next-bar open, and tracks portfolio performance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from models.market import SignalType
from strategy.base import StrategyModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.1  # risk 10% of capital per trade
    commission_pct: float = 0.0  # Alpaca is commission-free
    slippage_pct: float = 0.001  # 0.1% slippage estimate


@dataclass
class BacktestTrade:
    """Record of a trade during backtesting."""
    date: datetime
    signal: SignalType
    price: float
    qty: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    model_name: str
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100_000.0
    final_value: float = 100_000.0
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_value - self.initial_capital) / self.initial_capital

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.model_name} on {self.symbol}: "
            f"return={self.total_return:.2%}, trades={self.total_trades}, "
            f"win_rate={self.win_rate:.0%})"
        )


class BacktestEngine:
    """
    Simulates trading a strategy model over historical bars.
    
    Process:
    1. Walk through bars chronologically
    2. For each bar, feed the model the history up to that point
    3. Model produces a signal (BUY/SELL/HOLD)
    4. Simulate execution at next bar's open price
    5. Track P&L and portfolio equity
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        model: StrategyModel,
        symbol: str,
        data: pd.DataFrame,
        min_lookback: int = None,
    ) -> BacktestResult:
        """
        Run a backtest for a single model on historical data.
        
        Args:
            model: Strategy model to test
            symbol: Ticker symbol
            data: OHLCV DataFrame sorted by timestamp ascending
            min_lookback: Minimum bars before first signal (default: model.min_data_points)
        
        Returns:
            BacktestResult with full performance metrics
        """
        if data.empty or len(data) < 5:
            return BacktestResult(model_name=model.name, symbol=symbol)

        if min_lookback is None:
            min_lookback = model.min_data_points

        if len(data) <= min_lookback:
            logger.warning(f"Skipping {model.name}: insufficient data ({len(data)} < {min_lookback})")
            return BacktestResult(model_name=model.name, symbol=symbol)

        capital = self.config.initial_capital
        position_qty = 0.0
        position_entry_price = 0.0
        trades: list[BacktestTrade] = []
        equity_curve = []
        daily_returns = []
        prev_equity = capital

        result = BacktestResult(
            model_name=model.name,
            symbol=symbol,
            initial_capital=capital,
            start_date=data["timestamp"].iloc[min_lookback].to_pydatetime() if hasattr(data["timestamp"].iloc[min_lookback], 'to_pydatetime') else data["timestamp"].iloc[min_lookback],
            end_date=data["timestamp"].iloc[-1].to_pydatetime() if hasattr(data["timestamp"].iloc[-1], 'to_pydatetime') else data["timestamp"].iloc[-1],
        )

        for i in range(min_lookback, len(data) - 1):
            # Feed model history up to current bar (Optimization: Limit to last 2000 bars)
            # Most strategies don't need infinite history. This prevents O(N^2) copying.
            start_idx = max(0, i + 1 - 2000)
            history = data.iloc[start_idx : i + 1].copy()
            # history = history.reset_index(drop=True) # Avoid reset_index if possible, but strategies might rely on 0-based index?
            # Strategy logic usually looks at relative .iloc[-1], so index reset shouldn't strictly be required 
            # unless they access specific 0-based indices. 
            # Safe bet: verify strategy implementation. 
            # But let's keep it robust:
            if start_idx > 0:
                 history = history.reset_index(drop=True)
            current_close = float(data.iloc[i]["close"])
            next_open = float(data.iloc[i + 1]["open"])

            # Get model's signal
            try:
                signal = model.analyze(symbol, history)
            except Exception:
                signal_type = SignalType.HOLD
            else:
                signal_type = signal.signal_type

            # Execute at next bar's open
            execution_price = next_open * (1 + self.config.slippage_pct * (1 if signal_type == SignalType.BUY else -1))

            if signal_type == SignalType.BUY and position_qty == 0:
                # Open long position
                trade_value = capital * self.config.position_size_pct
                qty = trade_value / execution_price
                commission = trade_value * self.config.commission_pct
                capital -= trade_value + commission
                position_qty = qty
                position_entry_price = execution_price

            elif signal_type == SignalType.SELL and position_qty > 0:
                # Close position
                trade_value = position_qty * execution_price
                commission = trade_value * self.config.commission_pct
                pnl = (execution_price - position_entry_price) * position_qty - commission
                capital += trade_value - commission

                trades.append(BacktestTrade(
                    date=data.iloc[i + 1]["timestamp"],
                    signal=SignalType.SELL,
                    price=execution_price,
                    qty=position_qty,
                    pnl=pnl,
                ))

                position_qty = 0.0
                position_entry_price = 0.0

            # Calculate equity
            mark_to_market = position_qty * current_close if position_qty > 0 else 0
            equity = capital + mark_to_market
            equity_curve.append(equity)

            # Daily return
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            daily_returns.append(daily_ret)
            prev_equity = equity

        # Close any remaining position at last close
        if position_qty > 0:
            final_close = float(data.iloc[-1]["close"])
            pnl = (final_close - position_entry_price) * position_qty
            capital += position_qty * final_close
            trades.append(BacktestTrade(
                date=data.iloc[-1]["timestamp"],
                signal=SignalType.SELL,
                price=final_close,
                qty=position_qty,
                pnl=pnl,
            ))
            position_qty = 0.0

        result.final_value = capital
        result.trades = trades
        result.daily_returns = daily_returns
        result.equity_curve = equity_curve

        logger.info(
            f"Backtest {model.name}/{symbol}: "
            f"return={result.total_return:.2%}, "
            f"trades={result.total_trades}, "
            f"win_rate={result.win_rate:.0%}"
        )

        return result

    @staticmethod
    def compute_benchmark_returns(
        data: pd.DataFrame,
        min_lookback: int = 30,
    ) -> list[float]:
        """
        Compute daily buy-and-hold returns from the same data window
        used for backtesting. This is the benchmark that models are
        scored against — an always-buy model matches this exactly.

        Args:
            data: OHLCV DataFrame sorted by timestamp ascending
            min_lookback: Bars to skip (matches backtest start)

        Returns:
            List of daily returns for the benchmark period
        """
        if data.empty or len(data) < min_lookback + 2:
            return []

        close = data["close"].values.astype(float)
        benchmark_returns = []

        for i in range(min_lookback, len(data) - 1):
            prev_close = close[i]
            curr_close = close[i + 1]
            if prev_close > 0:
                benchmark_returns.append((curr_close - prev_close) / prev_close)
            else:
                benchmark_returns.append(0.0)

        return benchmark_returns

