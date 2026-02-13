"""
Databricks Job Entrypoint.

Usage:
    Set environment variables and run:
    python databricks_job.py

Environment Variables:
    JOB_TASK: "run_cycle", "weights", "package" (default: "run_cycle")
    LOG_LEVEL: "INFO" (default)
    STORAGE_BACKEND: "sqlite" or "parquet" (or "databricks" in future)
"""

import os
import sys
import logging
import traceback
from rich.logging import RichHandler

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_settings
from core.engine import MoneyPrinterEngine

# Configure Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("moneyprinter")

def main():
    logger.info("🚀 Starting Databricks Job")
    
    try:
        # Load Settings
        settings = load_settings()
        logger.info(f"🔧 Loaded Settings: mode={settings.trading.mode}, backend={settings.storage.backend}")
        
        # Initialize Engine
        engine = MoneyPrinterEngine(settings)
        logger.info("✅ Engine Initialized")
        
        # Determine Task
        task = os.getenv("JOB_TASK", "run_cycle")
        logger.info(f"📋 Executing Task: {task}")
        
        if task == "run_cycle":
            engine.run_cycle()
        elif task == "weights":
            engine.learn_weights()
            # If running in cloud, might want to upload weights?
            # For now, just save locally.
        elif task == "package":
            # Call the main.py logic or expose a method
            # Reimplementing simplified version here or importing from main?
            # Using engine.export_data is cleaner.
            output_dir = "dataset"
            engine.export_data(output_dir=output_dir)
            logger.info("Data exported. (Consolidation step skipped in job, assume packaging separately)")
        elif task == "ingest":
            engine.ingestion.ingest_symbols(
                symbols=settings.strategy.default_symbols,
                lookback_days=settings.strategy.lookback_days
            )
        else:
            logger.error(f"❌ Unknown task: {task}")
            sys.exit(1)
            
        logger.info("🎉 Job Completed Successfully")
        
    except Exception as e:
        logger.critical(f"🔥 Job Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.shutdown()

if __name__ == "__main__":
    main()
