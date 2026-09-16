"""
Reproducible extraction of calibration statistics from real NYC TLC
trip data.

Run this script to regenerate the constants hardcoded in
data/tlc_calibration.py from the raw source file. It is not imported
by the environment at runtime (env/config.py imports the pre-computed
constants directly, so training and evaluation never depend on this
multi-hundred-MB raw file being present) -- this script exists purely
for transparency and reproducibility of where those numbers came from.

Usage:
    python -m data.tlc_extract_calibration --input data/raw/yellow_tripdata_2019-01.csv.gz
"""
import argparse

import numpy as np
import pandas as pd


def load_and_clean(path: str, sample_rows: int = 2_000_000) -> pd.DataFrame:
    """Load a representative sample of the raw TLC CSV and apply basic sanity filtering."""
    usecols = ["tpep_pickup_datetime", "trip_distance", "PULocationID", "DOLocationID", "total_amount"]
    chunks = []
    rows_loaded = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, parse_dates=["tpep_pickup_datetime"]):
        chunks.append(chunk)
        rows_loaded += len(chunk)
        if rows_loaded >= sample_rows:
            break

    df = pd.concat(chunks, ignore_index=True)

    # Known data-quality issues in this dataset: a small number of rows
    # carry corrupted timestamps outside the nominal month, and a small
    # number have non-positive fares/distances or implausibly long trips.
    df = df[(df["tpep_pickup_datetime"] >= "2019-01-01") & (df["tpep_pickup_datetime"] < "2019-02-01")]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 50)]
    df = df[(df["total_amount"] > 0) & (df["total_amount"] < 300)]
    return df


def compute_hourly_multipliers(df: pd.DataFrame) -> np.ndarray:
    """Normalized (mean=1.0) hourly trip-count multipliers, indexed 0-23."""
    hourly_counts = df["tpep_pickup_datetime"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    return (hourly_counts / hourly_counts.mean()).values


def compute_distance_stats(df: pd.DataFrame) -> dict:
    return {
        "median_miles": float(df["trip_distance"].median()),
        "mean_miles": float(df["trip_distance"].mean()),
        "p75_miles": float(df["trip_distance"].quantile(0.75)),
    }


def compute_fare_per_mile_stats(df: pd.DataFrame) -> dict:
    valid = df[df["trip_distance"] >= 0.5]
    fare_per_mile = valid["total_amount"] / valid["trip_distance"]
    return {
        "median": float(fare_per_mile.median()),
        "mean": float(fare_per_mile.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw yellow_tripdata_*.csv.gz")
    parser.add_argument("--sample-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    print(f"Loading and cleaning {args.sample_rows:,} rows from {args.input}...")
    df = load_and_clean(args.input, args.sample_rows)
    print(f"Retained {len(df):,} rows after sanity filtering.\n")

    multipliers = compute_hourly_multipliers(df)
    print("Hourly multipliers (paste into data/tlc_calibration.py's REAL_TLC_HOURLY_MULTIPLIERS):")
    print(tuple(round(float(m), 3) for m in multipliers))

    print("\nTrip distance stats:")
    for k, v in compute_distance_stats(df).items():
        print(f"  {k}: {v:.3f}")

    print("\nFare-per-mile stats:")
    for k, v in compute_fare_per_mile_stats(df).items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
