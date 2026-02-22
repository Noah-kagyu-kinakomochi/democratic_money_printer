#!/usr/bin/env python3
"""
MoneyPrinter — Automated Trading Bot 🖨️💰

Usage:
    python main.py run           Run one trading cycle (with weight learning)
    python main.py account       Show account info
    python main.py positions     Show open positions
    python main.py ingest        Ingest market data only (no trading)
    python main.py analyze       Run analysis only (no trading)
    python main.py backtest      Run backtests for all models
    python main.py simulate      Full portfolio simulation (1 year, $1000)
    python main.py weights       Learn and display adaptive weights
    python main.py train-ml      Train ML models (Gradient Boosting, etc.)
    python main.py package       Bundle data for cloud training
"""

import sys
import os
import glob
import pandas as pd

from config.settings import load_settings
from core.engine import MoneyPrinterEngine, setup_logging, console
from data.gcs_utils import GCSManager


def main():
    setup_logging()

    console.print(r"""
[bold cyan]
  ███╗   ███╗ ██████╗ ███╗   ██╗███████╗██╗   ██╗
  ████╗ ████║██╔═══██╗████╗  ██║██╔════╝╚██╗ ██╔╝
  ██╔████╔██║██║   ██║██╔██╗ ██║█████╗   ╚████╔╝
  ██║╚██╔╝██║██║   ██║██║╚██╗██║██╔══╝    ╚██╔╝
  ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║███████╗   ██║
  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝
  [bold green]P R I N T E R[/bold green]   🖨️💰
[/bold cyan]
    """)

    settings = load_settings()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"

    # Validate API keys
    if not settings.alpaca.api_key or settings.alpaca.api_key == "your_api_key_here":
        console.print("[bold red]⚠️  Alpaca API key not configured![/bold red]")
        console.print("Copy .env.example to .env and add your Alpaca API keys.")
        console.print("Get keys at: https://app.alpaca.markets")
        sys.exit(1)

    engine = MoneyPrinterEngine(settings)
    gcs = GCSManager(bucket_name=settings.gcs.bucket_name or None)
    try:
        if command == "run":
            engine.run_cycle()
        elif command == "account":
            engine.show_account()
        elif command == "positions":
            engine.show_positions()
        elif command == "ingest":
            console.rule("[bold cyan]Data Ingestion Only")
            results = engine.ingestion.ingest_symbols(
                symbols=settings.strategy.default_symbols,
                lookback_days=settings.strategy.lookback_days,
            )
            for sym, rows in results.items():
                status = f"✅ {rows} bars" if rows >= 0 else "❌ failed"
                console.print(f"  {sym}: {status}")
        elif command == "ingest-sentiment":
            console.rule("[bold cyan]Sentiment Ingestion")
            results = engine.ingestion.ingest_sentiment(
                symbols=settings.strategy.default_symbols,
                limit=200
            )
            for sym, count in results.items():
                status = f"✅ {count} items" if count >= 0 else "❌ failed"
                console.print(f"  {sym}: {status}")
        elif command == "export":
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset"
            console.rule("[bold cyan]Export Data for ML")
            engine.export_data(output_dir=output_dir)
        elif command == "view":
            arg1 = sys.argv[2] if len(sys.argv) > 2 else None
            arg2 = sys.argv[3] if len(sys.argv) > 3 else "ohlcv"
            
            # Support "python main.py view macro" (no symbol)
            known_types = ["macro", "sentiment", "trades", "ohlcv"]
            if arg1 in known_types and arg2 == "ohlcv":
                symbol = None
                data_type = arg1
            else:
                symbol = arg1
                data_type = arg2
                
            engine.view_data(symbol, data_type)
        elif command == "analyze":
            console.rule("[bold cyan]Analysis Only (No Trading)")
            symbols = settings.strategy.default_symbols
            for symbol in symbols:
                # Load 1Day data for regime detection
                data_day = engine.storage.load_ohlcv(symbol, timeframe="1Day")
                
                # Load default data (likely 1Day or 1Min depending on what's available)
                # For analysis, let's load what the engine would use. 
                # But simple analyze just grabs one DF. 
                # Let's match run_cycle logic loosely.
                
                if data_day.empty:
                    console.print(f"  ⚠️  No 1Day data for {symbol} — run 'ingest' first")
                    continue

                regime = engine.regime_detector.detect_regime(data_day)
                console.print(f"\n[bold]{symbol}[/bold]")
                console.print(f"  🌊 Regime: [bold cyan]{regime.value}[/bold cyan]")
                
                consensus = engine.democracy.vote(symbol, data_day, regime=regime)
                console.print(f"  → {consensus}")
        elif command == "backtest":
            engine.run_backtest()
        elif command == "simulate":
            engine.run_portfolio_backtest()
        elif command == "weights":
            engine.learn_weights()
        elif command == "train-ml":
            engine.train_ml_models()
        elif command == "package":
            console.rule("[bold cyan]Packaging for Cloud Training 📦")
            
            # 1. Export Data
            output_dir = "dataset"
            engine.export_data(output_dir=output_dir)
            
            # 2. Consolidate
            console.print("  🔄 Consolidating parquet files...")
            files = glob.glob(f"{output_dir}/**/*.parquet", recursive=True)
            if not files:
                console.print("[red]❌ No exported data found![/red]")
            else:
                try:
                    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                    
                    target_path = "training_bundle/data.parquet"
                    df.to_parquet(target_path)
                    
                    console.print(f"  ✅ Bundled {len(df)} rows from {len(files)} files.")
                    console.print(f"  📁 Saved locally to: [bold green]{target_path}[/bold green]")
                    
                    # 3. Upload directly to GCS
                    if gcs.bucket_name:
                        console.print("  ☁️  Uploading to Google Cloud Storage...")
                        gcs_dest = settings.gcs.dataset_path
                        success = gcs.upload_file(target_path, gcs_dest)
                        if success:
                            console.print(f"  ✅ Uploaded to gs://{gcs.bucket_name}/{gcs_dest}")
                        else:
                            console.print("[red]❌ GCS Upload failed! Check credentials and bucket name.[/red]")
                    else:
                        console.print("[yellow]⚠️  GCS_BUCKET_NAME not set — skipping upload. Set it in .env to enable.[/yellow]")

                    console.print("\n[bold yellow]Next Steps (Automated):[/bold yellow]")
                    console.print("  1. Push to `main` to trigger GitHub Actions (gcp-gpu-training.yml).")
                    console.print("  2. The cloud workflow will pull `data.parquet` directly from GCS.")
                except Exception as e:
                    console.print(f"[red]❌ Consolidation failed: {e}[/red]")
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print(__doc__)
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

