"""
train.py
────────
Trains two models (Linear Regression + Gradient Boosting) on the combined
temperature + demand dataset, evaluates with a time-series split, and saves
the best model plus a calibrated RiskScorer.

Outputs
-------
  models/best_model.pkl    — best sklearn pipeline (scaler + model)
  models/model_meta.json   — metadata: type, RMSE, MAE, R², feature list,
                              demand percentiles for risk scoring

Usage
-----
    from src.model.train import train_and_save
    meta = train_and_save()

Or via CLI:
    python scripts/run_train.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.pipeline import load_combined
from src.model.features import FEATURE_COLUMNS, build_features, get_feature_matrix

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# Train on first 80%, test on last 20% (time-series safe split)
_TRAIN_SPLIT = 0.80


def train_and_save(
    combined_path: str | Path | None = None,
    save: bool = True,
) -> dict:
    """
    Train LinearRegression and GradientBoosting, pick best by RMSE, save.

    Returns
    -------
    dict — model metadata (type, RMSE, MAE, R², percentile thresholds)
    """
    # ── 1. Load data ───────────────────────────────────────────────────────────
    df = load_combined(combined_path)
    df = build_features(df, drop_na=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Training on %d rows after feature engineering", len(df))

    if len(df) < 100:
        raise ValueError(f"Need at least 100 rows to train; got {len(df)}")

    # ── 2. Time-series split ───────────────────────────────────────────────────
    split_idx = int(len(df) * _TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = get_feature_matrix(train_df)
    X_test, y_test = get_feature_matrix(test_df)
    logger.info("Train: %d rows  |  Test: %d rows", len(train_df), len(test_df))

    # ── 3. Train both models ───────────────────────────────────────────────────
    candidates = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ]),
    }

    results: dict[str, dict] = {}
    for name, pipeline in candidates.items():
        logger.info("Training %s …", name)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        results[name] = {"pipeline": pipeline, "rmse": rmse, "mae": mae, "r2": r2}
        logger.info("%s → RMSE=%.0f MW  MAE=%.0f MW  R²=%.4f", name, rmse, mae, r2)

    # ── 4. Pick best model ─────────────────────────────────────────────────────
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best = results[best_name]
    logger.info("Best model: %s (RMSE=%.0f MW)", best_name, best["rmse"])

    # ── 5. Compute demand percentiles for risk scoring ────────────────────────
    all_demands = df["demand_mw"].values
    percentile_thresholds = {
        f"p{p}": float(np.percentile(all_demands, p))
        for p in [50, 60, 70, 75, 80, 85, 90, 95, 99]
    }

    # ── 6. Build metadata ──────────────────────────────────────────────────────
    meta = {
        "model_type": best_name,
        "features": FEATURE_COLUMNS,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_rmse_mw": round(best["rmse"], 1),
        "test_mae_mw": round(best["mae"], 1),
        "test_r2": round(best["r2"], 4),
        "demand_percentile_thresholds_mw": percentile_thresholds,
        "demand_min": float(all_demands.min()),
        "demand_max": float(all_demands.max()),
        "all_model_results": {
            n: {"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]}
            for n, r in results.items()
        },
    }

    # ── 7. Print summary ───────────────────────────────────────────────────────
    _print_summary(meta)

    # ── 8. Save ────────────────────────────────────────────────────────────────
    if save:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = _MODELS_DIR / "best_model.pkl"
        meta_path = _MODELS_DIR / "model_meta.json"
        joblib.dump(best["pipeline"], model_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved model → %s", model_path)
        logger.info("Saved meta  → %s", meta_path)

    return meta


# ── Private helpers ────────────────────────────────────────────────────────────

def _print_summary(meta: dict) -> None:
    print("\n" + "=" * 60)
    print("  MODEL TRAINING RESULTS")
    print("=" * 60)
    print(f"  Best model    : {meta['model_type']}")
    print(f"  Test RMSE     : {meta['test_rmse_mw']:,.0f} MW")
    print(f"  Test MAE      : {meta['test_mae_mw']:,.0f} MW")
    print(f"  Test R2       : {meta['test_r2']:.4f}")
    print()
    print("  All models:")
    for name, r in meta["all_model_results"].items():
        marker = " << best" if name == meta["model_type"] else ""
        print(f"    {name:25s}  RMSE={r['rmse']:,.0f}  R2={r['r2']:.4f}{marker}")
    print()
    print("  Demand percentiles (risk thresholds):")
    for p, v in meta["demand_percentile_thresholds_mw"].items():
        print(f"    {p}: {v:,.0f} MW")
    print("=" * 60 + "\n")
