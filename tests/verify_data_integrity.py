import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.append(os.getcwd())

from data.loader import HybridDataLoader

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("DataAudit")

def test_data_integrity():
    print("🕵️‍♀️ Starting Operation Data Heist Audit...")
    
    loader = HybridDataLoader()
    
    # 1. Fetch/Load Macro Data
    # For this test, we might need real data or we can mock it if yfinance fails.
    # Let's try to load what's there or fetch minimal.
    if loader.macro_data.empty:
        print("  ⏳ Fetching macro data (this might take a moment)...")
        # Fetch small window to be fast
        try:
             loader.fetch_macro_history(lookback_years=1)
        except Exception as e:
             print(f"  ❌ Failed to fetch macro data: {e}. Cannot proceed with real data audit.")
             return

    if loader.macro_data.empty:
         print("  ❌ Macro data is empty. Cannot audit.")
         return

    macro = loader.macro_data
    print(f"  ✅ Macro Data Loaded: {len(macro)} rows")
    
    # ---------------------------------------------------------
    # Check 1: Timezone Assertion
    # ---------------------------------------------------------
    print("\n[Check 1] Timezone Assertion 🕒")
    tz_issue = False
    if macro.index.tz is None:
        print("  ❌ FAIL: Macro index is timezone-naive!")
        tz_issue = True
    else:
        # Check if UTC
        tz_name = str(macro.index.tz)
        if "UTC" in tz_name or "00:00" in tz_name:
            print(f"  ✅ PASS: Index is {tz_name}")
        else:
            print(f"  ⚠️ WARNING: Index is {tz_name}, expected UTC.")
            tz_issue = True # Strict check
            
    # ---------------------------------------------------------
    # Check 2: Weekend Logic & Alignment
    # ---------------------------------------------------------
    print("\n[Check 2] Weekend Logic & Look-ahead Bias 🗓️")
    
    # Create a dummy "Crypto" dataframe (trades 24/7)
    # Spanning a known Friday-Monday
    # Let's find a recent Friday in the macro data to align with
    last_date = macro.index[-1]
    # Find a Friday
    days_to_subtract = (last_date.weekday() - 4) % 7
    friday = last_date - timedelta(days=days_to_subtract)
    
    # Create timestamps for Fri, Sat, Sun, Mon at 10:00 AM UTC
    # Note: Macro data usually has 00:00 UTC timestamp for the day.
    
    # Timestamps: Fri 10am, Sat 10am, Sun 10am, Mon 10am
    # If look-ahead bias exists: Fri 10am will match Fri 00:00 Macro (which holds Fri Close) ❌
    # Correct behavior: Fri 10am should match Thu 00:00 Macro (Thu Close) OR Fri 00:00 if it meant Open?
    # Usually yfinance Daily Candle = OHLC for that day. 
    # At 10am on Fri, we DO NOT know Fri Close. We only know Thu Close.
    # So Fri 10am matches -> Thu Close.
    
    timestamps = [
        friday.replace(hour=10, minute=0, second=0, microsecond=0),
        friday.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1), # Sat
        friday.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=2), # Sun
        friday.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=3), # Mon
    ]
    
    dummy_df = pd.DataFrame(index=timestamps)
    dummy_df["Crypto_Price"] = [100, 101, 102, 103]
    
    # Merge
    merged = loader.merge_macro_data(dummy_df)
    
    # Pick a macro column (e.g. SP500_Close)
    col = [c for c in merged.columns if "SP500" in c or "VIX" in c]
    if not col:
        print("  ❌ Could not find macro columns in merged data.")
        return
        
    target_col = col[0]
    
    print(f"  Inspecting column: {target_col}")
    print(merged[[target_col]])
    
    # Analyze Look-ahead
    # Fri 10am
    fri_ts = timestamps[0]
    merged_val = merged.loc[fri_ts, target_col]
    
    # Get actual macro value for Friday
    # Assuming macro index is 00:00 UTC for that Friday
    # If yfinance index is Fri 00:00, and it contains Fri Close.
    # Fri 10am (merged) should NOT equal Fri 00:00 (macro) value.
    # It should equal Thu 00:00 (macro) value.
    
    # Let's check what value we got
    # Find the row in macro data with Fri index
    try:
        macro_fri_val = macro.loc[macro.index.date == fri_ts.date()][target_col].iloc[0]
    except:
        macro_fri_val = None
        
    try:
        # Thurs value
        thurs_date = fri_ts.date() - timedelta(days=1)
        macro_thu_val = macro.loc[macro.index.date == thurs_date][target_col].iloc[0]
    except:
        macro_thu_val = None
        
    print(f"\n  🔍 Deep Dive on {fri_ts.date()} (Friday):")
    print(f"    - Macro File (Thu Close): {macro_thu_val}")
    print(f"    - Macro File (Fri Close): {macro_fri_val}")
    print(f"    - Merged Data (Fri 10am): {merged_val}")
    
    if merged_val == macro_fri_val:
        print("  ❌ FAIL: Look-ahead bias detected! Fri 10am has Fri Close data.")
    elif merged_val == macro_thu_val:
        print("  ✅ PASS: Fri 10am has Thu Close data (Correct lag).")
    else:
        print("  ⚠️ UNKNOWN: Merged value matches neither? Check alignment.")

    # Check Weekend Hold
    # Sat 10am should have same value as Fri 10am (which is Thu close) OR Fri Close?
    # Wait, on Saturday 10am, we DO know Friday Close.
    # So Sat 10am -> matches Fri Close.
    
    sat_ts = timestamps[1]
    sat_val = merged.loc[sat_ts, target_col]
    
    print(f"\n  🔍 Deep Dive on {sat_ts.date()} (Saturday):")
    print(f"    - Merged Data (Sat 10am): {sat_val}")
    
    if sat_val == macro_fri_val:
        print("  ✅ PASS: Sat 10am has Fri Close data (Correct knowledge).")
    elif sat_val == macro_thu_val:
        print("  ⚠️ OK-ISH: Sat 10am has Thu Close data key (Stale but safe).")
    else:
        print("  ⚠️ UNKNOWN: Weekend value misalignment.")

    # Check TZ
    if tz_issue:
        print("\n  ❌ Final Result: TZ FAIL")
    else:
        print("\n  ✅ Final Result: TZ PASS")

if __name__ == "__main__":
    test_data_integrity()
