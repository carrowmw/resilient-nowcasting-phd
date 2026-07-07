import pandas as pd
import numpy as np

def check_traffic_gaps(file_path: str):
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows.")

    # Sort by location, category, veh_class, dir, and dt
    df = df.sort_values(['location', 'category', 'veh_class', 'dir', 'dt'])

    # Calculate time difference between consecutive rows within the same groups
    df['dt_diff'] = df.groupby(['location', 'category', 'veh_class', 'dir'])['dt'].diff()

    # Report NaN count (first row of each group) before excluding them from stats
    nan_diff_count = df['dt_diff'].isna().sum()
    print(f"\nRows with no predecessor in their group (expected = number of groups): {nan_diff_count}")
    assert nan_diff_count > 0, "Expected at least one NaN dt_diff (first row per group)"

    gaps = df['dt_diff'].dropna()
    assert len(gaps) > 0, "No valid gaps found after dropping first-row NaNs — data may be empty"

    print("\nGap Statistics (Time Deltas):")
    print(gaps.describe())

    # Count large gaps (e.g., > 15 minutes if data is expected to be more frequent)
    # Let's see what the most common frequency is first.
    mode_diff = gaps.mode()[0]
    print(f"\nMost common gap (Mode): {mode_diff}")

    large_gaps = gaps[gaps > mode_diff]
    print(f"Number of gaps larger than mode: {len(large_gaps)} ({len(large_gaps)/len(gaps)*100:.2f}%)")

    print("\nSample of large gaps:")
    print(large_gaps.value_counts().head(10))

if __name__ == "__main__":
    check_traffic_gaps('data/traffic_20230108_20260608.parquet')
