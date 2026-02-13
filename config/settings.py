"""
Configuration management for MoneyPrinter.
Loads settings from .env and provides typed access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (override=False means OS env vars take priority)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _env(key: str, default: str = "") -> str:
    """Get env var, treating empty strings as unset (falls back to default)."""
    value = os.getenv(key, "")
    return value if value else default


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"  # paper by default
    data_url: str = "https://data.alpaca.markets"

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "sqlite"  # "sqlite", "parquet", or "databricks"
    sqlite_path: str = str(_PROJECT_ROOT / "db" / "moneyprinter.db")
    parquet_dir: str = str(_PROJECT_ROOT / "data_store" / "parquet")
    # Databricks (future)
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_catalog: str = "main"
    databricks_schema: str = "trading"


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the democratic strategy engine."""
    voting_method: str = "weighted"  # "majority", "weighted", or "unanimous"
    min_confidence: float = 0.6  # minimum confidence threshold to act
    default_symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FXY"])
    # 5Min and 15Min are now generated on-the-fly from 1Min
    timeframes: list[str] = field(default_factory=lambda: ["1Min", "1Day"])
    lookback_days: int = 60  # how many days of history to analyze
    min_model_weight: float = 0.01  # Skip models with weight below this to save compute

    # Active models (can be set via ACTIVE_MODELS env var)
    #in active : DL
    active_models: list[str] = field(default_factory=lambda: [
        m.strip() for m in _env("ACTIVE_MODELS", "MA,RSI,MACD,AutoReg,CorrRegime,Sentiment").split(",") if m.strip()
    ])


@dataclass(frozen=True)
class TradingConfig:
    mode: str = "paper"  # "paper" or "live"
    max_position_pct: float = 0.1  # max 10% of portfolio per position
    default_qty: int = 1  # default shares per trade
    does_short: bool = True  # allow short selling


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for Market Regime Detection."""
    adx_threshold: float = 25.0
    volatility_threshold: float = 0.015  # 1.5% daily vol
    adx_period: int = 14
    smoothing_window: int = 3  # Hysteresis
    
    # Dynamic Weight Multipliers
    trend_boost: float = 1.5
    trend_penalty: float = 0.5
    chop_boost: float = 1.5
    chop_penalty: float = 0.5


@dataclass(frozen=True)
class AlphaVantageConfig:
    api_key: str = ""
    base_url: str = "https://www.alphavantage.co/query"


@dataclass(frozen=True)
class Settings:
    alpaca: AlpacaConfig
    alphavantage: AlphaVantageConfig
    storage: StorageConfig
    strategy: StrategyConfig
    trading: TradingConfig
    regime: RegimeConfig
    project_root: Path = _PROJECT_ROOT


def load_settings() -> Settings:
    """Load settings from environment variables."""
    trading_mode = _env("TRADING_MODE", "paper")

    if trading_mode == "live":
        alpaca_base_url = "https://api.alpaca.markets"
    else:
        alpaca_base_url = "https://paper-api.alpaca.markets"

    return Settings(
        alpaca=AlpacaConfig(
            api_key=_env("ALPACA_API_KEY"),
            secret_key=_env("ALPACA_SECRET_KEY"),
            base_url=alpaca_base_url,
        ),
        alphavantage=AlphaVantageConfig(
            api_key=_env("ALPHAVANTAGE_KEY"),
        ),
        storage=StorageConfig(
            backend=_env("STORAGE_BACKEND", "sqlite"),
            databricks_host=_env("DATABRICKS_HOST"),
            databricks_token=_env("DATABRICKS_TOKEN"),
            databricks_catalog=_env("DATABRICKS_CATALOG", "main"),
            databricks_schema=_env("DATABRICKS_SCHEMA", "trading"),
        ),
        strategy=StrategyConfig(),
        trading=TradingConfig(
            mode=trading_mode,
            does_short=_env("DOES_SHORT", "true").lower() == "true",
        ),
        regime=RegimeConfig(),
    )
