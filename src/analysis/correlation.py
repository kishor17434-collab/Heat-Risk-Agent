"""
correlation.py
──────────────
Statistical analysis of the temperature ↔ electricity demand relationship.

Outputs
-------
  • Console: formatted "demo quote" with Pearson r, slope, and lag
  • data/processed/correlation_report.json — machine-readable summary for dashboard
  • data/processed/correlation_plot.png   — scatter plot (optional, if matplotlib present)

Usage
-----
    from src.analysis.correlation import run_analysis
    report = run_analysis()
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_PIPELINE_META = _PROCESSED_DIR / "pipeline_meta.json"


def run_analysis(
    combined_path: str | Path | None = None,
    save: bool = True,
) -> dict:
    """
    Run correlation + lag analysis on the combined temperature + demand dataset.

    Returns
    -------
    dict with keys: pearson_r, pearson_p, slope_pct_per_10f, best_lag_hours,
                    n_samples, demo_quote, lag_correlations
    """
    if combined_path is None:
        combined_path = _PROCESSED_DIR / "combined.csv"

    df = pd.read_csv(combined_path, parse_dates=["timestamp"])
    logger.info("Loaded %d rows from %s", len(df), combined_path)

    if len(df) < 24:
        raise ValueError(f"Need at least 24 rows for meaningful analysis; got {len(df)}")

    # ── 1. Pearson correlation (same-hour, no lag) ─────────────────────────────
    r, p_value = stats.pearsonr(df["temp_f"], df["demand_mw"])
    logger.info("Pearson r=%.4f  p=%.4e  n=%d", r, p_value, len(df))
    sign_warning = bool(r < 0.2)
    sign_warning_message = (
        f"UNEXPECTED CORRELATION SIGN: r = {r:.2f}. Temperature and demand "
        "are expected to be positively correlated in a summer cooling-load grid. "
        "This usually indicates inconsistent data sources; investigate before using the demo finding."
        if sign_warning else ""
    )
    if sign_warning:
        logger.warning("⚠️ %s", sign_warning_message)
        print("\n" + "!" * 70 + "\n  ⚠️  " + sign_warning_message + "\n" + "!" * 70 + "\n")

    # ── 2. OLS slope — how much does demand change per 1°F? ────────────────────
    slope, intercept, *_ = stats.linregress(df["temp_f"], df["demand_mw"])
    mean_demand = df["demand_mw"].mean()
    # Express slope as % demand change per 10°F
    slope_pct_per_10f = (slope * 10.0 / mean_demand) * 100.0
    logger.info(
        "OLS slope: %.1f MW/°F  →  %.1f%% per 10°F (at mean demand %.0f MW)",
        slope, slope_pct_per_10f, mean_demand,
    )

    # ── 3. Lag analysis — does demand respond to temp with a delay? ────────────
    lag_results = _compute_lag_correlations(df, lags=[0, 1, 2, 3, 6, 12, 24])
    best_lag = max(lag_results, key=lambda x: abs(x["pearson_r"]))
    logger.info("Best lag: %d hours (r=%.4f)", best_lag["lag_hours"], best_lag["pearson_r"])

    # ── 4. Percentile thresholds (for risk scoring) ────────────────────────────
    demand_p70 = float(np.percentile(df["demand_mw"], 70))
    demand_p85 = float(np.percentile(df["demand_mw"], 85))
    demand_p95 = float(np.percentile(df["demand_mw"], 95))
    temp_p70 = float(np.percentile(df["temp_f"], 70))
    temp_p85 = float(np.percentile(df["temp_f"], 85))
    temp_p95 = float(np.percentile(df["temp_f"], 95))

    # ── 5. Assemble report ─────────────────────────────────────────────────────
    report = {
        "n_samples": len(df),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "pearson_r": round(float(r), 4),
        "pearson_p": float(p_value),
        "slope_mw_per_f": round(float(slope), 2),
        "slope_pct_per_10f": round(float(slope_pct_per_10f), 2),
        "mean_demand_mw": round(float(mean_demand), 0),
        "best_lag_hours": int(best_lag["lag_hours"]),
        "best_lag_r": round(float(best_lag["pearson_r"]), 4),
        "sign_warning": sign_warning,
        "sign_warning_message": sign_warning_message,
        "lag_correlations": lag_results,
        "demand_percentiles": {
            "p70": round(demand_p70, 0),
            "p85": round(demand_p85, 0),
            "p95": round(demand_p95, 0),
        },
        "temp_percentiles": {
            "p70": round(temp_p70, 1),
            "p85": round(temp_p85, 1),
            "p95": round(temp_p95, 1),
        },
    }

    # ── 6. Generate the demo quote ─────────────────────────────────────────────
    lag_note = (
        "same-hour response" if best_lag["lag_hours"] == 0
        else f"{best_lag['lag_hours']}h lag"
    )
    direction = "increase" if slope_pct_per_10f > 0 else "decrease"
    demo_quote = (
        f"Every 10°F increase in temperature correlates with a "
        f"{abs(slope_pct_per_10f):.1f}% {direction} in electricity demand "
        f"(Pearson r = {r:.2f}, {lag_note}, n = {len(df):,} hours)."
    )
    report["demo_quote"] = demo_quote

    # ── 7. Save ────────────────────────────────────────────────────────────────
    if save:
        out_path = _PROCESSED_DIR / "correlation_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved correlation report → %s", out_path)

    # ── 8. Pretty-print results ────────────────────────────────────────────────
    _print_results(report, demo_quote)

    # ── 9. Try to save a plot ──────────────────────────────────────────────────
    if save:
        _try_save_plot(df, r, slope, intercept)

    return report


# ── Private helpers ────────────────────────────────────────────────────────────

def _compute_lag_correlations(df: pd.DataFrame, lags: list[int]) -> list[dict]:
    """
    Compute Pearson r between temp_f (lagged by N hours) and demand_mw.
    Positive lag = demand responds to temperature from N hours ago.
    """
    results = []
    for lag in lags:
        if lag == 0:
            shifted = df["temp_f"]
        else:
            shifted = df["temp_f"].shift(lag)  # shift temp forward; compare to current demand

        mask = shifted.notna()
        if mask.sum() < 20:
            continue
        r, p = stats.pearsonr(shifted[mask], df["demand_mw"][mask])
        results.append({"lag_hours": lag, "pearson_r": round(float(r), 4), "p_value": float(p)})
        logger.debug("Lag %2dh → r=%.4f  p=%.4e", lag, r, p)
    return results


def _print_results(report: dict, demo_quote: str) -> None:
    print("\n" + "=" * 70)
    print("  CORRELATION ANALYSIS RESULTS")
    print("=" * 70)
    print("\n  [DEMO QUOTE]")
    print(f"  \"{demo_quote}\"")
    print()
    print(f"  Pearson r            : {report['pearson_r']} (p = {report['pearson_p']:.2e})")
    print(f"  Slope                : {report['slope_mw_per_f']} MW per °F")
    print(f"  Slope (normalised)   : {report['slope_pct_per_10f']}% demand change per 10°F")
    print(f"  Best lag             : {report['best_lag_hours']}h (r = {report['best_lag_r']})")
    print(f"  Sample size          : {report['n_samples']:,} hourly observations")
    if report.get("sign_warning"):
        print(f"\n  ⚠️  WARNING: {report['sign_warning_message']}")
    print()
    print("  Lag table:")
    for lag in report["lag_correlations"]:
        bar = "|" * int(abs(lag["pearson_r"]) * 30)
        print(f"    {lag['lag_hours']:3d}h lag  r={lag['pearson_r']:+.4f}  {bar}")
    print()
    print("  Demand percentiles (MW):")
    p = report["demand_percentiles"]
    print(f"    70th: {p['p70']:,.0f}  |  85th: {p['p85']:,.0f}  |  95th: {p['p95']:,.0f}")
    print("=" * 70 + "\n")


def _try_save_plot(
    df: pd.DataFrame,
    r: float,
    slope: float,
    intercept: float,
) -> None:
    """Save a scatter plot if matplotlib is available (non-essential)."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Temperature vs Electricity Demand — ERCOT COAST", fontsize=13)

        # Scatter
        ax = axes[0]
        ax.scatter(df["temp_f"], df["demand_mw"], alpha=0.3, s=8, color="#2196F3")
        x_line = np.linspace(df["temp_f"].min(), df["temp_f"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", lw=2, label=f"r = {r:.2f}")
        ax.set_xlabel("Temperature (°F)")
        ax.set_ylabel("Demand (MW)")
        ax.set_title("Scatter: Temp vs Demand")
        ax.legend()

        # Time series overlay (sample last 30 days)
        ax2 = axes[1]
        recent = df.tail(30 * 24)
        ax2_twin = ax2.twinx()
        ax2.plot(recent["timestamp"], recent["demand_mw"], color="#2196F3", alpha=0.7, label="Demand (MW)")
        ax2_twin.plot(recent["timestamp"], recent["temp_f"], color="#F44336", alpha=0.7, label="Temp (°F)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Demand (MW)", color="#2196F3")
        ax2_twin.set_ylabel("Temp (°F)", color="#F44336")
        ax2.set_title("Last 30 Days: Demand + Temperature")
        fig.autofmt_xdate()

        out = _PROCESSED_DIR / "correlation_plot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        logger.info("Saved plot → %s", out)

    except ImportError:
        logger.info("matplotlib not installed — skipping plot generation")
    except RuntimeError as exc:
        logger.warning("Plot generation failed: %s", exc)
