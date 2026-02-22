"""
MoneyPrinter — Main Bot Engine.

Orchestrates the full pipeline:
1. Learn adaptive weights via backtesting
2. Ingest latest market data
3. Run the democratic strategy engine
4. Execute consensus trades
5. Log everything
"""

import logging
import sys
from datetime import datetime, timezone
import pandas as pd

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.markup import escape
from rich.text import Text

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.portfolio_backtest import PortfolioBacktester, print_backtest_report
from backtesting.scorer import ModelScorer
from backtesting.weight_learner import WeightLearner, WeightLearnerConfig
from backtesting.weight_store import save_weights, load_weights
from config.settings import Settings
from data.data_fetcher import DataFetcher
from data.ingestion import DataIngestionService
from data.sources.alpaca_source import AlpacaDataSource
from data.sources.alpha_vantage import AlphaVantageSource
from models.market import OrderSide, SignalType, ConsensusSignal, Signal, Vote
from storage.base import StorageAdapter
from storage.sqlite_store import SqliteStore
from storage.parquet_store import ParquetStore
from strategy.base import ModelConfig
from strategy.democracy import DemocraticEngine
from strategy.models.auto_regression import AutoRegressionStrategy
from strategy.models.correlation_regime import CorrelationRegimeStrategy
from strategy.models.macd_strategy import MACDStrategy
from strategy.models.moving_average import MovingAverageCrossover
from strategy.models.rsi_strategy import RSIStrategy
from trading.alpaca_broker import AlpacaBroker
from trading.base import Broker
from strategy.models.sentiment_strategy import SentimentStrategy
from strategy.models.dl_model import DeepLearningStrategy
from strategy.models.gb_strategy import GradientBoostingStrategy
from strategy.regime import RegimeDetector, MarketRegime
from data.loader import HybridDataLoader

console = Console()
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure rich logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


class MoneyPrinterEngine:
    """
    The main bot orchestrator.
    
    Wires together all components:
    - StorageAdapter (data persistence)
    - DataSource (market data)
    - DemocraticEngine (strategy decisions)
    - Broker (trade execution)
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # ─── Initialize Storage ──────────────────────────────────
        self.storage = self._create_storage(settings)
        self.storage.initialize()

        # ─── Initialize Data Source ──────────────────────────────
        self.data_source = AlpacaDataSource(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            base_url=settings.alpaca.base_url,
        )

        self.sentiment_source = AlphaVantageSource(
            api_key=settings.alphavantage.api_key
        )

        # ─── Initialize Ingestion ───────────────────────────────
        self.ingestion = DataIngestionService(
            source=self.data_source,
            storage=self.storage,
            sentiment_source=self.sentiment_source,
        )

        # ─── Initialize Data Fetcher (read-only) ────────────────
        self.data_fetcher = DataFetcher(self.storage)
        
        # ─── Initialize Hybrid Data Loader ──────────────────────
        self.data_loader = HybridDataLoader()

        # ─── Initialize Strategy Models (N × 4 timeframes) ─────
        models = self._create_models(
            settings.strategy.timeframes,
            self.data_fetcher,
            settings.strategy.active_models
        )
        logger.info(f"Created {len(models)} models ({len(models)//len(settings.strategy.timeframes)} strategies × {len(settings.strategy.timeframes)} timeframes)")

        # ─── Initialize Democratic Engine ───────────────────────
        self.democracy = DemocraticEngine(
            models=models,
            voting_method=settings.strategy.voting_method,
            min_confidence=settings.strategy.min_confidence,
            min_weight=settings.strategy.min_model_weight,
            regime_config=settings.regime,
        )
        
        # ─── Initialize Regime Detector ─────────────────────────
        self.regime_detector = RegimeDetector(settings.regime)

        # ─── Initialize Weight Learner ──────────────────────────
        self.weight_learner = WeightLearner(WeightLearnerConfig(
            lookback_days=settings.strategy.lookback_days,
        ))

        # ─── Initialize Broker ──────────────────────────────────
        self.broker = AlpacaBroker(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            paper=(settings.trading.mode == "paper"),
        )

        # ─── Initialize Exporter ────────────────────────────────
        from data.exporter import DatasetExporter
        self.exporter = DatasetExporter(self.storage)

        logger.info("MoneyPrinter Engine initialized ✅")

    def export_data(self, output_dir: str = "dataset"):
        """Export ML-ready dataset for default symbols."""
        self.exporter.export_dataset(
            symbols=self.settings.strategy.default_symbols,
            output_dir=output_dir,
        )

    @staticmethod
    def _create_storage(settings: Settings) -> StorageAdapter:
        """Factory method to create the configured storage backend."""
        backend = settings.storage.backend
        if backend == "sqlite":
            return SqliteStore(settings.storage.sqlite_path)
        elif backend == "parquet":
            return ParquetStore(settings.storage.parquet_dir)
        elif backend == "databricks":
            # Future: from storage.databricks_store import DatabricksStore
            raise NotImplementedError(
                "Databricks storage is planned for Phase 3. "
                "Use 'sqlite' or 'parquet' for now."
            )
        else:
            raise ValueError(f"Unknown storage backend: {backend}")

    @staticmethod
    def _create_models(timeframes: list[str], data_fetcher: DataFetcher, active_models: list[str]) -> list:
        """
        Factory: create one instance of each strategy model per timeframe.
        """
        strategy_defs = [
            (MovingAverageCrossover, "MA", 1.0, {"short_window": 5, "long_window": 20}),
            (RSIStrategy, "RSI", 1.0, {"period": 14, "oversold": 30, "overbought": 70}),
            (MACDStrategy, "MACD", 1.0, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            (AutoRegressionStrategy, "AutoReg", 1.0, {"lags": 30, "train_window": 120}),
            (CorrelationRegimeStrategy, "CorrRegime", 1.0, {"corr_window": 30, "shift_lookback": 5}),
            (SentimentStrategy, "Sentiment", 1.5, {"threshold": 0.15, "lookback_hours": 24}),
            (GradientBoostingStrategy, "GB", 2.0, {}),
            (DeepLearningStrategy, "DL", 1.5, {}),
        ]

        models = []
        active_list = active_models
        
        for cls, base_name, weight, params in strategy_defs:
            if base_name not in active_list:
                continue
                
            for tf in timeframes:
                model = cls(ModelConfig(
                    name=f"{base_name}_{tf}",
                    weight=weight,
                    params=params,
                    timeframe=tf,
                ))
                
                # Dependency Injection
                if hasattr(model, "set_fetcher"):
                    model.set_fetcher(data_fetcher)
                
                models.append(model)
        return models

    def train_ml_models(self, symbols: list[str] = None):
        """
        Train all models that support the `train` method (e.g. GB, DL).
        """
        if symbols is None:
            symbols = self.settings.strategy.default_symbols

        console.rule("[bold yellow]Training ML Models")
        
        # Gather data
        data_by_symbol = {}
        for symbol in symbols:
            # Need enough history for training
            n_bars = 2000 # Good amount for ML
            data = self.data_fetcher.get_latest_bars(symbol, n=n_bars, timeframe="1Min")
            if not data.empty:
                 # Enrich with Macro Data
                 data = self.data_loader.merge_macro_data(data)
                 data_by_symbol[symbol] = data
                 
        if not data_by_symbol:
            console.print("  ⚠️  No data available for training.")
            return

        # Train Trainable Models
        for model in self.democracy.models:
            if hasattr(model, "train"):
                # Filtering logic: if the model is a specific timeframe, does it need specific data?
                # The GB model handles its own feature extraction from provided DFs.
                # However, we have a list of DFs.
                # Let's pass the dictionary of {symbol: df}
                try:
                    console.print(f"  🚂 Training {model.name}...")
                    model.train(data_by_symbol)
                except Exception as e:
                    logger.error(f"Failed to train {model.name}: {e}")

    def learn_weights(self, symbols: list[str] = None) -> dict:
        """
        Learn adaptive weights by backtesting each model on recent data.
        Loads previous weights, blends with fresh scores, and saves.
        """
        if symbols is None:
            symbols = self.settings.strategy.default_symbols

        console.rule("[bold yellow]Step 0: Adaptive Weight Learning")

        # 0.5: Train Trainable Models (e.g. Deep Learning, Gradient Boosting)
        self.train_ml_models(symbols)

        # Load previous weights for blending
        stored = load_weights()
        previous_weights = stored.weights if stored else {}
        if previous_weights:
            console.print(f"  📂 Loaded previous weights ({stored.age_hours:.1f}h old, {len(previous_weights)} models)")
        else:
            console.print("  🆕 No previous weights — starting fresh")

        # Gather data for each symbol × timeframe
        data_by_symbol = {}
        timeframes = self.settings.strategy.timeframes
        for symbol in symbols:
            for tf in timeframes:
                # Convert lookback_days to N bars
                if tf == "1Min":
                    # Fetch extra buffer for calculation
                    n_bars = self.settings.strategy.lookback_days * 1440 
                elif tf == "5Min":
                    n_bars = self.settings.strategy.lookback_days * 288
                elif tf == "15Min":
                    n_bars = self.settings.strategy.lookback_days * 96
                else: # 1Day
                    n_bars = self.settings.strategy.lookback_days
                    
                # Ensure minimum for DL models (need ~100)
                if n_bars < 120:
                    n_bars = 120
                    
                data = self.data_fetcher.get_latest_bars(symbol, n=n_bars, timeframe=tf)
                if not data.empty:
                    # Enrich with Macro Data
                    data = self.data_loader.merge_macro_data(data)
                    data_by_symbol.setdefault(symbol, {})[tf] = data

        if not data_by_symbol:
            console.print("  ⚠️  No data available for weight learning — keeping current weights")
            return {}

        # Run weight learning (fresh scores)
        update = self.weight_learner.learn_weights(
            models=self.democracy.models,
            data_by_symbol=data_by_symbol,
        )

        # Blend with previous weights (exponential smoothing)
        blended = self.weight_learner.blend_with_previous(
            fresh_weights=update.weights,
            previous_weights=previous_weights,
        )

        # Apply blended weights
        self.democracy.update_weights(blended)

        # Save to JSON for reuse
        score_data = {
            s.model_name: {
                "training_score": s.training_score,
                "alpha": s.alpha,
                "sharpe_ratio": s.sharpe_ratio,
            }
            for s in update.scores
        }
        save_weights(blended, scores=score_data)

        # Print weight table
        weight_table = Table(title="Updated Model Weights (blended)")
        weight_table.add_column("Model", style="cyan")
        weight_table.add_column("Weight", justify="right")
        weight_table.add_column("Previous", justify="right", style="dim")
        weight_table.add_column("Train Score", justify="right")

        for score in update.scores:
            w = blended.get(score.model_name, 0)
            prev_w = previous_weights.get(score.model_name, None)
            prev_str = f"{prev_w:.3f}" if prev_w is not None else "new"
            weight_table.add_row(
                score.model_name,
                f"{w:.3f}",
                prev_str,
                f"{score.training_score:.3f}",
            )
        console.print(weight_table)

        return blended

    def run_backtest(self, symbols: list[str] = None) -> dict:
        """
        Run backtests for all models and display results.
        """
        if symbols is None:
            symbols = self.settings.strategy.default_symbols

        console.rule("[bold cyan]Backtesting All Models")
        engine = BacktestEngine()
        scorer = ModelScorer()

        results = {}
        for model in self.democracy.models:
            tf = model.config.timeframe
            for symbol in symbols:
                # Need enough data for DL model (24h * 60m = 1440 bars)
                data = self.data_fetcher.get_latest_bars(symbol, n=2000, timeframe=tf)
                if data.empty:
                    continue
                
                # Enrich with Macro Data
                data = self.data_loader.merge_macro_data(data)

                result = engine.run(model, symbol, data)
                score = scorer.score(result)
                results[f"{model.name}/{symbol}"] = score
                
                # Log why 0 trades occurred (debug info)
                if score.total_trades == 0:
                    # Run one analyze to see the reason
                    sig = model.analyze(symbol, data)
                    logger.info(f"  ℹ️  {model.name}/{symbol} [0 trades]: {sig.reason}")

        # Summary table
        table = Table(title="Backtest Results")
        table.add_column("Model/Symbol", style="cyan")
        table.add_column("Return", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Trades", justify="right")

        for key, score in results.items():
            ret_style = "green" if score.total_return >= 0 else "red"
            table.add_row(
                key,
                f"[{ret_style}]{score.total_return:.2%}[/{ret_style}]",
                f"{score.sharpe_ratio:.3f}",
                f"{score.win_rate:.0%}",
                f"{score.max_drawdown:.2%}",
                str(score.total_trades),
            )
        console.print(table)

        return results

    def run_portfolio_backtest(self, symbols: list[str] = None) -> None:
        """
        Run a full portfolio simulation over 1 year of historical data.
        Tests the entire pipeline: all models + democratic voting + execution.
        """
        if symbols is None:
            symbols = self.settings.strategy.default_symbols

        console.rule("[bold cyan]Full Portfolio Simulation — 1 Year")
        timeframes = self.settings.strategy.timeframes

        # Step 0: Load saved weights (if available)
        stored = load_weights()
        if stored and stored.weights:
            console.print(f"  📂 Loading saved weights ({stored.age_hours:.1f}h old)")
            self.democracy.update_weights(stored.weights)
        else:
            console.print("  ⚠️  No saved weights — using defaults (run 'weights' first for trained simulation)")

        # Step 1: Ensure we have 1 year of data
        console.rule("[yellow]Step 1: Loading Data (1 year)")
        self.ingestion.ingest_all_timeframes(
            symbols=symbols,
            timeframes=timeframes,
            lookback_days=365,
        )

        # Step 2: Load all data from storage
        data_by_symbol: dict[str, dict[str, 'pd.DataFrame']] = {}
        for symbol in symbols:
            data_by_symbol[symbol] = {}
            for tf in timeframes:
                data = self.data_fetcher.get_ohlcv(symbol, timeframe=tf)
                if not data.empty:
                    # Enrich with Macro Data
                    data = self.data_loader.merge_macro_data(data)
                    data_by_symbol[symbol][tf] = data
                    console.print(f"  {symbol}/{tf}: {len(data)} bars (macro enriched)")

        # Step 3: Run portfolio simulation
        # Use a lower confidence threshold for simulation — 12 models
        # across different timeframes naturally dilute confidence
        from strategy.democracy import DemocraticEngine as SimDemocracy
        sim_democracy = SimDemocracy(
            models=self.democracy.models,
            voting_method=self.settings.strategy.voting_method,
            min_confidence=0.05,  # lower threshold for 12-model ensemble
        )

        console.rule("[yellow]Step 2: Running Simulation")
        backtester = PortfolioBacktester(
            democracy=sim_democracy,
            initial_cash=1_000.0,
            position_size_pct=self.settings.trading.max_position_pct,
            slippage_pct=0.001,
            does_short=self.settings.trading.does_short,
        )

        result = backtester.run(
            data_by_symbol=data_by_symbol,
            lookback_bars=60,
        )

        # Step 4: Print report
        print_backtest_report(result)

    def run_cycle(self, symbols: list[str] = None) -> dict:
        """
        Run one full trading cycle:
        0. Learn adaptive weights from backtesting
        1. Ingest data
        2. Analyze with democratic engine
        3. Execute trades
        4. Return results
        """
        if symbols is None:
            symbols = self.settings.strategy.default_symbols

        console.rule("[bold cyan]MoneyPrinter — Trading Cycle")
        console.print(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        console.print(f"📊 Symbols: {', '.join(symbols)}")
        console.print(f"🗳️  Voting: {self.settings.strategy.voting_method}")
        console.print()

        # Step 1: Ingest data (all timeframes)
        console.rule("[yellow]Step 1: Data Ingestion")
        timeframes = self.settings.strategy.timeframes
        ingestion_results = self.ingestion.ingest_all_timeframes(
            symbols=symbols,
            timeframes=timeframes,
            lookback_days=self.settings.strategy.lookback_days,
        )

        for sym, tf_results in ingestion_results.items():
            total = sum(v for v in tf_results.values() if v >= 0)
            console.print(f"  {sym}: ✅ {total} bars across {len(tf_results)} timeframes")

        # Step 1.5: Learn adaptive weights
        self.learn_weights(symbols)

        # Step 2: Democratic voting (multi-timeframe)
        console.rule("[yellow]Step 2: Democratic Strategy Engine")
        console.print(f"  🗳️  {len(self.democracy.models)} models voting ({len(timeframes)} timeframes)")
        
        decisions = {}
        latest_prices = {}
        
        for symbol in symbols:
            console.print(f"\n[bold]{symbol}[/bold]")

            # Load data for each timeframe
            data_by_tf = {}
            for tf in timeframes:
                data = self.storage.load_ohlcv(symbol, timeframe=tf)
                if not data.empty:
                    # Enrich with Macro Data
                    data = self.data_loader.merge_macro_data(data)
                    data_by_tf[tf] = data
            
            if not data_by_tf:
                console.print(f"  ⚠️  No data available for {symbol}")
                continue
            
            # Capture latest price for sizing
            if "1Min" in data_by_tf and not data_by_tf["1Min"].empty:
                 latest_prices[symbol] = data_by_tf["1Min"]["close"].iloc[-1]
            elif "1Day" in data_by_tf and not data_by_tf["1Day"].empty:
                 latest_prices[symbol] = data_by_tf["1Day"]["close"].iloc[-1]
            else:
                 latest_prices[symbol] = 0.0

            # 2a. Detect Market Regime (using default 1Day data)
            regime = MarketRegime.UNKNOWN
            if "1Day" in data_by_tf:
                regime = self.regime_detector.detect_regime(data_by_tf["1Day"])
            elif "1Min" in data_by_tf:
                regime = MarketRegime.SIDEWAYS_LOW_VOL
            
            # Display regime using Text to avoid markup errors with emoji
            regime_text = Text()
            regime_text.append("  🌊 Regime: ")
            regime_text.append(str(regime.value), style="bold cyan")
            console.print(regime_text)

            # 🛑 Hard Stop Loss Check 🛑
            # If position is down > 2%, force exit immediately (bypass voting)
            position = self.broker.get_position(symbol)
            force_exit = False
            
            if position:
                try:
                    unrealized_plpct = float(position.unrealized_plpc)
                    if unrealized_plpct < -0.02:
                         console.print(f"  🛑 Stop Loss Triggered: P&L {unrealized_plpct:.2%} < -2.0%")
                         force_exit = True
                except (AttributeError, ValueError):
                    pass
            
            if force_exit:
                # Create a synthetic SELL/COVER signal
                # Determine side based on position
                pos_qty = float(position.qty)
                signal_type = SignalType.SELL if pos_qty > 0 else SignalType.BUY
                
                consensus = ConsensusSignal(
                    signal_type=signal_type,
                    confidence=1.0, # Max confidence to ensure close
                    symbol=symbol,
                    votes=[],
                )
                console.print(f"  → Stop Loss Override: {signal_type.value}")
            else:
                consensus = self.democracy.vote_multi_tf(symbol, data_by_tf, regime=regime)
            
            decisions[symbol] = consensus
            if not force_exit:
                 console.print(f"  → {consensus}")

        # Step 3: Execute trades
        console.rule("[yellow]Step 3: Trade Execution")
        
        # Fetch Account Equity for Sizing
        account = self.broker.get_account()
        equity = float(account.get("equity", 0.0))
        console.print(f"  💰 Account Equity: ${equity:,.2f}")
        
        trades = {}
        for symbol, consensus in decisions.items():
            if consensus.signal_type == SignalType.HOLD:
                console.print(f"  {symbol}: HOLD — no action")
                continue

            side = OrderSide.BUY if consensus.signal_type == SignalType.BUY else OrderSide.SELL

            # Check existing position
            position = self.broker.get_position(symbol)

            if side == OrderSide.SELL and position is None:
                if not self.settings.trading.does_short:
                    console.print(f"  {symbol}: SELL signal but no position (shorting disabled) — skipping")
                    continue
                # Short sell: open a short position
                console.print(f"  {symbol}: SELL signal, opening SHORT position")

            if side == OrderSide.BUY and position is not None:
                # Check if it's a short position (negative qty)
                try:
                    pos_qty = float(position.qty)
                except (AttributeError, TypeError):
                    pos_qty = 0
                if pos_qty < 0:
                    console.print(f"  {symbol}: BUY signal — closing short position")
                else:
                    console.print(f"  {symbol}: BUY signal but already holding long — skipping")
                    continue

            # Execute - Dynamic Sizing
            # Qty = (Equity * MaxPositionPct * Confidence) / Price
            price = latest_prices.get(symbol, 0.0)
            qty = self.settings.trading.default_qty
            
            if equity > 0 and price > 0:
                max_pos_value = equity * self.settings.trading.max_position_pct
                target_value = max_pos_value * consensus.confidence
                calc_qty = int(target_value / price)
                qty = max(1, calc_qty) # Ensure at least 1 share
                
                console.print(f"  ⚖️  Dynamic Sizing: ${target_value:,.2f} ({consensus.confidence:.0%} conf) -> {qty} shares")

            trade = self.broker.submit_order(symbol, side, qty)
            trade.consensus = consensus
            self.storage.save_trade(trade)
            trades[symbol] = trade

            emoji = "🟢" if side == OrderSide.BUY else "🔴"
            console.print(
                f"  {emoji} {symbol}: {side.value.upper()} {qty} shares "
                f"(status: {trade.status.value})"
            )

        # Summary
        console.rule("[bold green]Cycle Complete")
        self._print_summary(decisions, trades)

        return {
            "ingestion": ingestion_results,
            "decisions": decisions,
            "trades": trades,
        }

    def _print_summary(self, decisions: dict, trades: dict):
        """Print a pretty summary table."""
        table = Table(title="Decision Summary")
        table.add_column("Symbol", style="cyan")
        table.add_column("Signal", style="bold")
        table.add_column("Confidence", justify="right")
        table.add_column("Votes (B/S/H)", justify="center")
        table.add_column("Action", style="green")

        for symbol, consensus in decisions.items():
            signal_style = {
                SignalType.BUY: "[green]BUY[/green]",
                SignalType.SELL: "[red]SELL[/red]",
                SignalType.HOLD: "[yellow]HOLD[/yellow]",
            }
            counts = consensus.vote_summary
            vote_str = f"{counts['BUY']}/{counts['SELL']}/{counts['HOLD']}"

            action = "—"
            if symbol in trades:
                trade = trades[symbol]
                action = f"{trade.side.value.upper()} {trade.qty}"

            table.add_row(
                symbol,
                signal_style[consensus.signal_type],
                f"{consensus.confidence:.0%}",
                vote_str,
                action,
            )

        console.print(table)

    def show_account(self):
        """Display account information."""
        account = self.broker.get_account()
        console.rule("[bold cyan]Account Info")
        for k, v in account.items():
            if isinstance(v, float):
                console.print(f"  {k}: ${v:,.2f}")
            else:
                console.print(f"  {k}: {v}")

    def show_positions(self):
        """Display current positions."""
        positions = self.broker.get_all_positions()
        if not positions:
            console.print("  No open positions")
            return

        table = Table(title="Open Positions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Entry Price", justify="right")
        table.add_column("Market Value", justify="right")
        table.add_column("P&L", justify="right")

        for p in positions:
            pl = p["unrealized_pl"]
            pl_style = "green" if pl >= 0 else "red"
            table.add_row(
                p["symbol"],
                str(p["qty"]),
                f"${p['avg_entry_price']:,.2f}",
                f"${p['market_value']:,.2f}",
                f"[{pl_style}]${pl:,.2f}[/{pl_style}]",
            )

        console.print(table)

    def view_data(self, symbol: str = None, data_type: str = "ohlcv", limit: int = 10):
        """View stored data in the terminal."""
        if not symbol and data_type == "ohlcv":
            symbols = self.storage.list_symbols()
            console.print(f"[bold]Stored Symbols (OHLCV):[/bold] {', '.join(symbols)}")
            return

        console.rule(f"[bold cyan]View Data: {symbol or 'All'} ({data_type})")

        df = pd.DataFrame()
        if data_type.lower() == "ohlcv" and symbol:
            # Try 1Day first, then 1Min
            df = self.storage.load_ohlcv(symbol, timeframe="1Day")
            if df.empty:
                df = self.storage.load_ohlcv(symbol, timeframe="1Min")
                if not df.empty:
                    console.print("[dim]Showing 1Min data (1Day empty)[/dim]")
            else:
                console.print("[dim]Showing 1Day data[/dim]")

        elif data_type.lower() == "sentiment" and symbol:
            df = self.storage.load_sentiment(symbol)
            if not df.empty:
                table = Table(title=f"Sentiment Data: {symbol} (Latest {limit})")
                table.add_column("Timestamp", style="dim")
                table.add_column("Score", justify="right")
                table.add_column("Label")
                table.add_column("Title / Summary")
                table.add_column("Source", style="cyan")

                # Show latest news first
                subset = df.tail(limit).iloc[::-1]

                for _, row in subset.iterrows():
                    ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
                    score_val = float(row['score'])
                    score_str = f"{score_val:.2f}"
                    if score_val > 0.15:
                        score_str = f"[green]{score_str}[/green]"
                    elif score_val < -0.15:
                        score_str = f"[red]{score_str}[/red]"
                    
                    label = row.get("label", "")
                    
                    # Prefer title, fallback to summary
                    text = row.get("title", "") or row.get("summary", "")
                    if isinstance(text, str) and len(text) > 60:
                        text = text[:57] + "..."
                        
                    source = row.get("source", "")
                    
                    table.add_row(ts, score_str, label, str(text), str(source))
                
                console.print(table)
                console.print(f"\n[dim]Total items: {len(df)}[/dim]")
                return

        elif data_type.lower() == "trades":
            trades = self.storage.load_trades(symbol)
            if trades:
                df = pd.DataFrame([t.to_dict() for t in trades])

        elif data_type.lower() == "macro":
            # Ensure loaded
            if self.data_loader.macro_data.empty:
                self.data_loader.load_macro_data()
            
            df = self.data_loader.macro_data.copy()
            if not df.empty:
                if symbol:
                    # Filter columns by symbol (fuzzy match)
                    # e.g. symbol="VIX" -> matches "VIX_Close"
                    cols = [c for c in df.columns if symbol.upper() in c.upper()]
                    if cols:
                        df = df[cols]
                    else:
                        console.print(f"[yellow]No macro columns found matching '{symbol}'[/yellow]")
                        console.print(f"Available: {', '.join(df.columns)}")
                        return

        if df.empty:
            console.print("[red]No data found.[/red]")
            return

        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            console.print(df.head(limit))
            console.print(f"\n[dim]... ({len(df)} rows total)[/dim]")

    def shutdown(self):
        """Clean up resources."""
        self.storage.close()
        logger.info("MoneyPrinter Engine shut down 🛑")
