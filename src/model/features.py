"""
features.py
───────────
Feature engineering for the Heat Risk model.

Takes the raw combined DataFrame and returns an augmented version ready
for training or inference.

Feature set:
    hour              int   0–23
    day_of_week       int   0=Mon … 6=Sun
    month             int   1–12
    is_weekend        int   0 or 1 (int for sklearn compatibility)
    temp_f            float Current temperature
    temp_rolling_3h   float Rolling 3h mean temp (smooths noise)
    temp_rolling_24h  float Rolling 24h mean temp (captures heat waves)
    temp_delta_6h     float temp_f − temp 6h ago (heat trend direction)
    temp_delta_24h    float temp_f − temp 24h ago
    season            int   0=Winter 1=Spring 2=Summer 3=Fall

Target:
    demand_mw  — raw MW demand used for regression
"""

from __future__ import annotations

import pandas as pd

# ── Column ordering expected by the trained model ─────────────────────────────
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temp_f",
    "temp_rolling_3h",
    "temp_rolling_24h",
    "temp_delta_6h",
    "temp_delta_24h",
    "season",
]

TARGET_COLUMN = "demand_mw"


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Add all engineered features to the combined DataFrame.

    Parameters
    ----------
    df       : Combined dataset (output of pipeline.run_pipeline)
    drop_na  : Drop rows where rolling features produce NaN (first 24h)

    Returns
    -------
    pd.DataFrame with FEATURE_COLUMNS + TARGET_COLUMN present
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # ── Ensure base time columns exist ─────────────────────────────────────────
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Rolling temperature features ───────────────────────────────────────────
    df["temp_rolling_3h"] = (
        df["temp_f"].rolling(window=3, min_periods=1).mean()
    )
    df["temp_rolling_24h"] = (
        df["temp_f"].rolling(window=24, min_periods=6).mean()
    )

    # ── Temperature delta (trend) ──────────────────────────────────────────────
    df["temp_delta_6h"] = df["temp_f"] - df["temp_f"].shift(6)
    df["temp_delta_24h"] = df["temp_f"] - df["temp_f"].shift(24)

    # ── Season encoding ────────────────────────────────────────────────────────
    df["season"] = df["month"].apply(_month_to_season)

    # ── Cast is_weekend to int ─────────────────────────────────────────────────
    df["is_weekend"] = df["is_weekend"].astype(int)

    if drop_na:
        before = len(df)
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            pass  # Expected — first 24h have no rolling history

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) ready for sklearn fit/predict.

    Parameters
    ----------
    df : DataFrame that has already been passed through build_features()

    Returns
    -------
    X : pd.DataFrame with FEATURE_COLUMNS
    y : pd.Series with TARGET_COLUMN (demand_mw)
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def build_inference_row(
    temp_f: float,
    hour: int,
    day_of_week: int,
    month: int,
    temp_rolling_3h: float | None = None,
    temp_rolling_24h: float | None = None,
    temp_delta_6h: float | None = None,
    temp_delta_24h: float | None = None,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for inference (no history required).
    Missing rolling features default to temp_f (conservative estimate).
    """
    tr3 = temp_rolling_3h if temp_rolling_3h is not None else temp_f
    tr24 = temp_rolling_24h if temp_rolling_24h is not None else temp_f
    d6 = temp_delta_6h if temp_delta_6h is not None else 0.0
    d24 = temp_delta_24h if temp_delta_24h is not None else 0.0

    row = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(day_of_week >= 5),
        "temp_f": temp_f,
        "temp_rolling_3h": tr3,
        "temp_rolling_24h": tr24,
        "temp_delta_6h": d6,
        "temp_delta_24h": d24,
        "season": _month_to_season(month),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _month_to_season(month: int) -> int:
    """0=Winter 1=Spring 2=Summer 3=Fall"""
    if month in (12, 1, 2):
        return 0
    elif month in (3, 4, 5):
        return 1
    elif month in (6, 7, 8):
        return 2
    else:
        return 3
