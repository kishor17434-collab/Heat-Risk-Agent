"""
predict.py
──────────
Inference layer: loads the saved model and returns risk scores.

Risk score definition
─────────────────────
The model predicts raw demand (MW). We convert this to a 0–100 risk score
using the demand percentile thresholds saved in model_meta.json:

    predicted_demand < p50   →  score  0–30  (normal)
    p50 ≤ demand < p75       →  score 30–60  (elevated)
    p75 ≤ demand < p90       →  score 60–80  (high)
    demand ≥ p90             →  score 80–100 (critical)

This makes the score intuitive: 70 means "demand is in the top 25%".

Usage
-----
    from src.model.predict import load_model, predict_risk, forecast_24h

    model = load_model()
    score = predict_risk(model, temp_f=103.5, hour=17, day_of_week=1, month=7)
    # → e.g. 82.3

    df = forecast_24h(model, current_temp_f=103.5, hour=16, day_of_week=1, month=7)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model.features import build_inference_row
from src.preflight import ProjectPreflightError, validate_required_paths

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_path: str | Path | None = None) -> dict:
    """
    Load the saved sklearn pipeline and metadata.

    Returns
    -------
    dict with keys:
        pipeline  — sklearn Pipeline object
        meta      — dict from model_meta.json
        thresholds — demand percentile thresholds (MW)
    """
    if model_path is None:
        model_path = _MODELS_DIR / "best_model.pkl"
    meta_path = Path(str(model_path).replace(".pkl", "")).parent / "model_meta.json"

    if not Path(model_path).exists():
        raise ProjectPreflightError(
            f"Model artifact missing: {model_path}. "
            "Run 'python scripts/run_train.py' first to train the model."
        )

    validate_required_paths([("model artifact", model_path), ("model metadata", meta_path)])

    pipeline = joblib.load(model_path)
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    thresholds = meta.get("demand_percentile_thresholds_mw", {})
    d_min = meta.get("demand_min", 8000)
    d_max = meta.get("demand_max", 35000)

    logger.info(
        "Loaded model: %s  (RMSE=%.0f MW, R²=%.4f)",
        meta.get("model_type", "unknown"),
        meta.get("test_rmse_mw", 0),
        meta.get("test_r2", 0),
    )

    return {
        "pipeline": pipeline,
        "meta": meta,
        "thresholds": thresholds,
        "demand_min": d_min,
        "demand_max": d_max,
    }


# ── Single-point prediction ───────────────────────────────────────────────────

def predict_risk(
    model_bundle: dict,
    temp_f: float,
    hour: int,
    day_of_week: int,
    month: int,
    temp_rolling_3h: float | None = None,
    temp_rolling_24h: float | None = None,
    temp_delta_6h: float | None = None,
    temp_delta_24h: float | None = None,
) -> dict:
    """
    Predict demand and convert to a 0–100 risk score for a single point in time.

    Returns
    -------
    dict with keys:
        predicted_demand_mw  float
        risk_score           float  0–100
        risk_level           str    "normal" | "elevated" | "high" | "critical"
    """
    X = build_inference_row(
        temp_f=temp_f,
        hour=hour,
        day_of_week=day_of_week,
        month=month,
        temp_rolling_3h=temp_rolling_3h,
        temp_rolling_24h=temp_rolling_24h,
        temp_delta_6h=temp_delta_6h,
        temp_delta_24h=temp_delta_24h,
    )

    pipeline = model_bundle["pipeline"]
    predicted_demand = float(pipeline.predict(X)[0])
    risk_score = _demand_to_risk(predicted_demand, model_bundle["thresholds"])

    return {
        "predicted_demand_mw": round(predicted_demand, 1),
        "risk_score": round(risk_score, 1),
        "risk_level": _risk_level(risk_score),
    }


# ── 24-hour forecast ───────────────────────────────────────────────────────────

def forecast_24h(
    model_bundle: dict,
    current_temp_f: float,
    hour: int,
    day_of_week: int,
    month: int,
    hours_ahead: int = 24,
) -> pd.DataFrame:
    """
    Generate an hourly risk forecast for the next N hours.

    Simulates future temperatures using a simple sinusoidal assumption
    anchored at the current temperature.

    Returns
    -------
    pd.DataFrame with columns:
        forecast_hour, timestamp, temp_f, predicted_demand_mw, risk_score, risk_level
    """
    from datetime import timezone
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    records = []

    for h in range(hours_ahead):
        future_hour = (hour + h) % 24
        future_dow = (day_of_week + (hour + h) // 24) % 7
        future_month = month  # simplified: assume same month for 24h ahead

        # Simple temperature projection: sinusoidal daily cycle around current temp
        # Peak at 15:00, trough at 06:00 — ±8°F amplitude
        temp_proj = current_temp_f + 8.0 * (
            np.sin(np.pi * (future_hour - 6.0) / 12.0)
            - np.sin(np.pi * (hour - 6.0) / 12.0)
        )

        result = predict_risk(
            model_bundle,
            temp_f=temp_proj,
            hour=future_hour,
            day_of_week=future_dow,
            month=future_month,
        )
        records.append({
            "forecast_hour": h,
            "timestamp": now + timedelta(hours=h),
            "temp_f": round(temp_proj, 1),
            "predicted_demand_mw": result["predicted_demand_mw"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
        })

    return pd.DataFrame(records)


# Configurable Risk Bands
_BAND_NORMAL = 30.0
_BAND_ELEVATED = 60.0
_BAND_HIGH = 80.0
_BAND_CRITICAL = 100.0

def _demand_to_risk(demand_mw: float, thresholds: dict) -> float:
    """
    Convert predicted demand (MW) to a 0–100 risk score using percentile thresholds.

    Score bands:
        0–30   : below p50 (normal)
        30–60  : p50–p75  (elevated)
        60–80  : p75–p90  (high)
        80–100 : above p90 (critical)
    """
    p50 = thresholds.get("p50")
    p75 = thresholds.get("p75")
    p90 = thresholds.get("p90")
    p95 = thresholds.get("p95")
    
    # Fallback to sensible defaults if thresholds are missing
    p50 = p50 if p50 else 10000.0
    p75 = p75 if p75 else 15000.0
    p90 = p90 if p90 else 20000.0
    p95 = p95 if p95 else 25000.0

    if demand_mw < p50:
        # Scale 0–30 based on how close to p50
        fraction = demand_mw / p50 if p50 > 0 else 0
        return max(0.0, fraction * _BAND_NORMAL)
    elif demand_mw < p75:
        fraction = (demand_mw - p50) / (p75 - p50) if p75 > p50 else 0
        return _BAND_NORMAL + fraction * (_BAND_ELEVATED - _BAND_NORMAL)
    elif demand_mw < p90:
        fraction = (demand_mw - p75) / (p90 - p75) if p90 > p75 else 0
        return _BAND_ELEVATED + fraction * (_BAND_HIGH - _BAND_ELEVATED)
    elif demand_mw < p95:
        fraction = (demand_mw - p90) / (p95 - p90) if p95 > p90 else 0
        return _BAND_HIGH + fraction * (90.0 - _BAND_HIGH)
    else:
        # Above p95, scale from 90 to 100
        fraction = (demand_mw - p95) / max(p95, 1)
        return min(_BAND_CRITICAL, 90.0 + fraction * 50.0)


def _risk_level(score: float) -> str:
    if score < 50:
        return "normal"
    elif score < 70:
        return "elevated"
    elif score < 85:
        return "high"
    else:
        return "critical"
