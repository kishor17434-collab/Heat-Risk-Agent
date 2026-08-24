"""
pipeline.py
───────────
Aligns temperature and electricity demand data by hourly timestamp and
writes the combined dataset to disk.

Outputs
-------
  data/raw/temp_raw.csv      — raw temperature DataFrame
  data/raw/demand_raw.csv    — raw demand DataFrame
  data/processed/combined.csv — hourly-aligned, NaN-free combined dataset

Combined dataset columns:
    timestamp    datetime64[ns]   UTC, tz-naive
    location     str              Temperature location name
    region       str              ERCOT weather zone
    temp_f       float64          Temperature in °F
    temp_c       float64          Temperature in °C
    demand_mw    float64          Electricity demand in MW
    hour         int              0–23
    day_of_week  int              0=Mon … 6=Sun
    month        int              1–12
    is_weekend   bool

Usage
-----
    from src.data.pipeline import run_pipeline
    df = run_pipeline(start="2024-06-01", end="2024-08-31")

Or via CLI:
    python scripts/run_pipeline.py --start 2024-06-01 --end 2024-08-31
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.ercot_client import ERCOTClient
from src.data.temperature_client import TemperatureClient

load_dotenv()
logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def run_pipeline(
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    temp_mode: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch, align, and optionally save the combined temperature + demand dataset.

    Parameters
    ----------
    start     : First date (inclusive), e.g. "2024-06-01"
    end       : Last date  (inclusive), e.g. "2024-08-31"
    temp_mode : Override TEMP_DATA_MODE env var (simulate | open_meteo | fortyguard)
    save      : Write CSVs to data/raw/ and data/processed/ if True

    Returns
    -------
    pd.DataFrame — combined, cleaned, feature-enriched dataset
    """
    if start is None or end is None:
        if (temp_mode or os.getenv("TEMP_DATA_MODE", "simulate")).lower() == "fortyguard":
            # FortyGuard heatmap requests are asynchronous and hourly. Keep the
            # default live refresh small enough for an interactive run.
            end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            start = (end_dt - timedelta(hours=1)).isoformat()
            end = end_dt.isoformat()
        else:
            end_dt = datetime.utcnow().date()
            start_dt = end_dt - timedelta(days=30)
            start = start_dt.isoformat()
            end = end_dt.isoformat()

    logger.info("Pipeline starting: %s → %s", start, end)

    # ── 1. Fetch temperature ───────────────────────────────────────────────────
    temp_client = TemperatureClient(mode=temp_mode)
    temp_res = temp_client.fetch(start=start, end=end)
    temp_df = temp_res["data"]
    logger.info("Temperature: %d rows", len(temp_df))

    if save:
        _save_csv(temp_df, _RAW_DIR / "temp_raw.csv")

    # ── 2. Fetch demand ────────────────────────────────────────────────────────
    demand_client = ERCOTClient()
    demand_res = demand_client.fetch(start=start, end=end)
    demand_df = demand_res["data"]
    logger.info("Demand: %d rows", len(demand_df))

    if save:
        _save_csv(demand_df, _RAW_DIR / "demand_raw.csv")

    # ── 3. Align on hourly timestamp ───────────────────────────────────────────
    # Round both to the start of each hour before merging
    temp_df["timestamp"] = temp_df["timestamp"].dt.floor("h")
    demand_df["timestamp"] = demand_df["timestamp"].dt.floor("h")

    # Aggregate if there are multiple readings per hour (shouldn't happen, but safe)
    temp_hourly = (
        temp_df.groupby("timestamp", as_index=False)
        .agg(location=("location", "first"), temp_f=("temp_f", "mean"), temp_c=("temp_c", "mean"))
    )
    demand_hourly = (
        demand_df.groupby("timestamp", as_index=False)
        .agg(region=("region", "first"), demand_mw=("demand_mw", "mean"))
    )

    combined = pd.merge(temp_hourly, demand_hourly, on="timestamp", how="inner")
    logger.info("After inner join: %d rows", len(combined))

    if combined.empty:
        raise ValueError(
            "Merge produced 0 rows — timestamp ranges don't overlap. "
            "Check that both clients are pulling the same date range."
        )

    # ── 4. Drop NaNs ──────────────────────────────────────────────────────────
    before = len(combined)
    combined = combined.dropna(subset=["temp_f", "demand_mw"])
    dropped = before - len(combined)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN values", dropped)

    # ── 5. Enrich with time features ───────────────────────────────────────────
    combined = _add_time_features(combined)

    # ── 6. Sort and reset index ────────────────────────────────────────────────
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # ── 7. Save combined dataset ───────────────────────────────────────────────
    if save:
        _save_csv(combined, _PROCESSED_DIR / "combined.csv")

    logger.info(
        "Pipeline complete: %d rows from %s to %s",
        len(combined),
        combined["timestamp"].min(),
        combined["timestamp"].max(),
    )

    # ── 8. Print summary ───────────────────────────────────────────────────────
    _print_summary(combined)

    return combined


def load_combined(path: str | Path | None = None) -> pd.DataFrame:
    """Load the processed combined dataset from disk."""
    if path is None:
        path = _PROCESSED_DIR / "combined.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("Loaded combined dataset: %d rows from %s", len(df), path)
    return df


# ── Private helpers ────────────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features useful for both analysis and modelling."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek          # 0=Mon
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    return df


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved: %s (%d rows)", path, len(df))


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Rows          : {len(df):,}")
    print(f"  Date range    : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  Temp range    : {df['temp_f'].min():.1f}F - {df['temp_f'].max():.1f}F")
    print(f"  Demand range  : {df['demand_mw'].min():.0f} MW - {df['demand_mw'].max():.0f} MW")
    print(f"  NaN rows      : {df.isnull().any(axis=1).sum()}")
    print("=" * 60 + "\n")
