"""
Clean project state.

Usage:
    python clean.py [--all]

Options:
    --all       Delete the database (requires re-ingestion)
"""

import os
import sys
import shutil
from pathlib import Path

def clean():
    root = Path(__file__).parent
    
    # 1. Clean compiled files
    print("🧹 Cleaning __pycache__...")
    for p in root.rglob("__pycache__"):
        shutil.rmtree(p)
    for p in root.rglob("*.pyc"):
        p.unlink()
        
    # 2. Clean weights
    weights_path = root / "data" / "weights.json"
    if weights_path.exists():
        print(f"🗑️  Deleting {weights_path.name}...")
        weights_path.unlink()
    else:
        print(f"   {weights_path.name} not found (fresh state)")
        
    # 3. Clean bundles
    bundle_data = root / "training_bundle" / "data.parquet"
    if bundle_data.exists():
         print(f"📦 Deleting training bundle data...")
         bundle_data.unlink()

    # 4. Clean logs
    log_file = root / "moneyprinter.log"
    if log_file.exists():
        print("📝 Clearing log file...")
        log_file.unlink()

    # 5. Full Reset (Database)
    if "--all" in sys.argv:
        db_path = root / "db" / "moneyprinter.db"
        if db_path.exists():
            print(f"🔥 DELETING DATABASE {db_path.name} (Re-ingestion required!)")
            db_path.unlink()
        
        # Also clean parquet store if exists
        parquet_dir = root / "data_store" / "parquet"
        if parquet_dir.exists():
             shutil.rmtree(parquet_dir)
             print("🔥 Deleted parquet data store")

    print("\n✨ Clean complete!")

if __name__ == "__main__":
    clean()
