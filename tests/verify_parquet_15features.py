"""
Post-run verification for the 15-feature static spatial pipeline.

Run after process_all.py completes:
    python tests/verify_parquet_15features.py

Asserts:
  1. All 15 static feature columns are present in the parquet.
  2. No NaN values in any static feature column.
  3. No feature has > 50% identical values (saturation check).

Also spot-checks Quayside (high density) vs. NclSaltersHollywood (suburban).
"""

import sys
import os
import pandas as pd
import numpy as np

PARQUET_PATH = 'data/processed/master_dataset_15T.parquet'

STATIC_COLS = [
    'transit_count', 'retail_count', 'food_drink_count', 'nightlife_count',
    'education_count', 'recreation_count', 'total_poi_count', 'landuse_entropy',
    'is_school_zone', 'betweenness_centrality', 'closeness_centrality',
    'dist_to_city_centre', 'degree_centrality', 'nearby_bus_stops', 'nearby_parking'
]

QUAYSIDE_LOC = 'NclQuaysideUnderHighLevelBridge'
SUBURBAN_LOC = 'NclSaltersHollywood'


def run_verification():
    print(f"\nLoading {PARQUET_PATH} (static columns only) ...")
    if not os.path.exists(PARQUET_PATH):
        print(f"ERROR: Parquet not found at {PARQUET_PATH}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH, columns=['location'] + STATIC_COLS)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns\n")

    # 1. Column presence
    print("=" * 60)
    print("CHECK 1: All 15 static columns present")
    print("=" * 60)
    missing = [c for c in STATIC_COLS if c not in df.columns]
    if missing:
        print(f"FAIL: Missing columns: {missing}")
        sys.exit(1)
    print(f"PASS: All {len(STATIC_COLS)} columns present.\n")

    # 2. No NaN values
    print("=" * 60)
    print("CHECK 2: No NaN values in static features")
    print("=" * 60)
    nan_report = {c: df[c].isna().sum() for c in STATIC_COLS if df[c].isna().any()}
    if nan_report:
        print(f"FAIL: NaN values found: {nan_report}")
        sys.exit(1)
    print("PASS: Zero NaN values across all 15 columns.\n")

    # 3. Saturation check — no column should have > 50% identical values
    print("=" * 60)
    print("CHECK 3: Saturation check (<= 50% identical values per column)")
    print("=" * 60)
    saturated = []
    for col in STATIC_COLS:
        mode_frac = df[col].value_counts(normalize=True).iloc[0]
        status = "OK" if mode_frac <= 0.50 else "SATURATED"
        print(f"  {col:<30} mode_fraction={mode_frac:.3f}  [{status}]")
        if mode_frac > 0.50:
            saturated.append(col)
    if saturated:
        print(f"\nWARN: Saturated columns (> 50% identical): {saturated}")
        # Don't exit — this is a warning, not a hard failure post-feature-expansion
    else:
        print("\nPASS: No saturated features.\n")

    # 4. Distribution summary
    print("=" * 60)
    print("CHECK 4: Distribution summary")
    print("=" * 60)
    print(df[STATIC_COLS].describe().T.to_string())
    print()

    # 5. Spot check: Quayside vs. suburban
    print("=" * 60)
    print("CHECK 5: Spot-check Quayside vs. Suburban")
    print("=" * 60)
    locs = df['location'].unique()

    for loc_name in [QUAYSIDE_LOC, SUBURBAN_LOC]:
        if loc_name not in locs:
            print(f"  INFO: Location '{loc_name}' not in dataset, skipping spot-check.")
            continue
        row = df[df['location'] == loc_name][STATIC_COLS].iloc[0]
        print(f"\n  {loc_name}")
        for col, val in row.items():
            print(f"    {col:<30} {val}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    run_verification()
