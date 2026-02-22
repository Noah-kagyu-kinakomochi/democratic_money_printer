"""
Configuration management for MoneyPrinter.
Loads settings from .env and provides typed access.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from data.gcs_utils import get_secret

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
    min_confidence: float = 0.3  # minimum confidence threshold to act (Lowered for FXY)
    default_symbols: list[str] = field(default_factory=lambda: ["FXY"])
    # 5Min and 15Min are now generated on-the-fly from 1Min
    timeframes: list[str] = field(default_factory=lambda: ["1Min", "1Day"])
    lookback_days: int = 7  # how many days of history to analyze (7 days for weight learning)
    min_model_weight: float = 0.01  # Skip models with weight below this to save compute

    # Active models (can be set via ACTIVE_MODELS env var)
    # The system now concentrates all resources exclusively on the Deep Learning Transformer
    active_models: list[str] = field(default_factory=lambda: [
        m.strip() for m in _env("ACTIVE_MODELS", "DL").split(",") if m.strip()
    ])


@dataclass(frozen=True)
class TradingConfig:
    mode: str = "paper"  # "paper" or "live"
    max_position_pct: float = 0.95  # max 95% of portfolio per position (Sniper Mode)
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
class GCSConfig:
    """Google Cloud Storage configuration for artifact management."""
    bucket_name: str = ""
    dataset_path: str = "datasets/data.parquet"   # GCS path for training data
    model_path: str = "models/best_model.pth"      # GCS path for trained model
    scaler_path: str = "models/scaler.pkl"          # GCS path for scaler


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
    gcs: GCSConfig
    project_root: Path = _PROJECT_ROOT


def load_settings() -> Settings:
    """Load settings from .env and fallback to GCP Secret Manager."""
    load_dotenv()

    def _env(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    gcp_project = _env("GOOGLE_CLOUD_PROJECT", "moneyprinter-prod") # Fallback to prod project

    # Determine Alpaca credentials, falling back to Secret Manager if missing
    alpaca_api_key = _env("ALPACA_API_KEY")
    if not alpaca_api_key:
        alpaca_api_key = get_secret("ALPACA_API_KEY", gcp_project) or ""
        
    alpaca_secret_key = _env("ALPACA_SECRET_KEY")
    if not alpaca_secret_key:
        alpaca_secret_key = get_secret("ALPACA_SECRET_KEY", gcp_project) or ""

    trading_mode_str = _env("TRADING_MODE", "paper").lower()

    if trading_mode_str == "live":
        alpaca_base_url = "https://api.alpaca.markets"
    else:
        alpaca_base_url = "https://paper-api.alpaca.markets"

    return Settings(
        alpaca=AlpacaConfig(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key,
            base_url="https://paper-api.alpaca.markets" if trading_mode_str == "paper" else "https://api.alpaca.markets"
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
            mode=trading_mode_str,
            does_short=_env("DOES_SHORT", "true").lower() == "true",
        ),
        regime=RegimeConfig(),
        gcs=GCSConfig(
            bucket_name=_env("GCS_BUCKET_NAME"),
        ),
    )
