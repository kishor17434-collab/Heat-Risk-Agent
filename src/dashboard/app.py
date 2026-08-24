"""
src/dashboard/app.py
─────────────────────
Streamlit dashboard for the Heat Risk Agent.

Three panels:
  1. Current Conditions — latest temp + demand, risk gauge
  2. 24h Forecast       — hourly risk score line chart (color-coded zones)
  3. Agent Log          — live-updating decision table + alerts

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _risk_color_hex(score: float) -> str:
    if score >= 85:
        return "#ef4444"
    elif score >= 70:
        return "#f97316"
    elif score >= 50:
        return "#eab308"
    else:
        return "#22c55e"

def _dark_layout() -> dict:
    return {
        "paper_bgcolor": "#0f1117",
        "plot_bgcolor": "#0f1117",
        "font": {"color": "#e0e0e0"},
        "xaxis": {"gridcolor": "#1f2937", "zerolinecolor": "#374151"},
        "yaxis": {"gridcolor": "#1f2937", "zerolinecolor": "#374151"},
        "margin": {"t": 30, "b": 30, "l": 50, "r": 50},
    }
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heat Risk Agent",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_MODELS_DIR = _PROJECT_ROOT / "models"
_LOGS_DIR = _PROJECT_ROOT / "logs"
_COMBINED_CSV = _PROCESSED_DIR / "combined.csv"
_CORRELATION_JSON = _PROCESSED_DIR / "correlation_report.json"
_MODEL_META_JSON = _MODELS_DIR / "model_meta.json"
_DECISION_LOG = _LOGS_DIR / "agent_decisions.log"
_ALERT_LOG = _LOGS_DIR / "agent_alerts.log"

_RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "70"))

_CACHE_TTL = 30


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
        color: #e0e0e0;
    }
    .risk-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
    .alert-row { background: rgba(239,68,68,0.1); border-left: 3px solid #ef4444; padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 4px; font-family: monospace; font-size: 0.82rem; }
    .ok-row    { color: #4b5563; font-family: monospace; font-size: 0.82rem; padding: 0.2rem 1rem; }
    .demo-quote { background: linear-gradient(90deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15)); border-left: 4px solid #3b82f6; padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; font-style: italic; font-size: 1.05rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Heat Risk Agent")
    st.markdown("---")

    threshold = st.slider(
        "Alert Threshold (Risk Score)",
        min_value=0, max_value=100, value=int(_RISK_THRESHOLD), step=5,
    )
    refresh_interval = st.selectbox(
        "Auto-refresh interval",
        options=[10, 30, 60, 120],
        index=1,
        format_func=lambda x: f"Every {x}s",
    )
    st.markdown("---")
    st.markdown("**Data Source**")
    st.caption(f"Mode: `{os.getenv('TEMP_DATA_MODE', 'simulate')}`")
    st.caption(f"Region: ERCOT `{os.getenv('ERCOT_REGION', 'COAST')}`")
    st.caption(f"Location: {os.getenv('LOCATION_NAME', 'Houston_TX')}")
    st.markdown("---")

    if st.button("🔄 Refresh Now"):
        st.rerun()

    if st.button("▶️ Run Agent Simulation (10 steps)"):
        with st.spinner("Running agent simulation..."):
            try:
                from src.agent.loop import AgentLoop
                loop = AgentLoop(threshold=threshold, poll_interval=1)
                loop.run(mode="simulate", max_steps=10)
                st.success("Simulation complete! Logs updated.")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
        import time
        time.sleep(1.5)
        st.rerun()

    st.markdown("---")
    st.caption("Heat Risk Agent • Hackathon Demo")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🌡️ Heat Risk Agent")
st.markdown("*Real-time heat-driven grid strain prediction and autonomous alerting*")

from streamlit_autorefresh import st_autorefresh

# Auto-refresh (smooth background refresh without full page reload)
st_autorefresh(interval=refresh_interval * 1000, key="data_autorefresh")

# ── Load data ──────────────────────────────────────────────────────────────────

def _recent_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if df.empty:
        return df
    # CSV timestamps are stored as UTC without timezone metadata.
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    cut_off = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    recent = df[df["timestamp"] >= cut_off].copy()
    if recent.empty:
        return df.tail(24).copy()
    return recent.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=_CACHE_TTL)
def load_combined_data() -> tuple[pd.DataFrame | None, str | None]:
    if not _COMBINED_CSV.exists():
        return None, "File combined.csv not found"
    try:
        df = pd.read_csv(_COMBINED_CSV, parse_dates=["timestamp"])
        return _recent_window(df), None
    except Exception as e:
        return None, f"Error reading combined.csv: {e}"


@st.cache_data(ttl=_CACHE_TTL)
def load_correlation_report() -> dict | None:
    if not _CORRELATION_JSON.exists():
        return None
    with open(_CORRELATION_JSON) as f:
        return json.load(f)


@st.cache_data(ttl=_CACHE_TTL)
def load_model_meta() -> dict | None:
    if not _MODEL_META_JSON.exists():
        return None
    with open(_MODEL_META_JSON) as f:
        return json.load(f)


def load_forecast() -> pd.DataFrame | None:
    """Generate a live 24h forecast using the trained model."""
    try:
        from src.model.predict import forecast_24h, load_model
        model = load_model()
        now = datetime.now(timezone.utc)
        df, _ = load_combined_data()
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        return forecast_24h(
            model,
            current_temp_f=float(latest["temp_f"]),
            hour=now.hour,
            day_of_week=now.weekday(),
            month=now.month,
        )
    except RuntimeError:
        return None


df, load_err = load_combined_data()
corr = load_correlation_report()
meta = load_model_meta()

# ── Demo quote banner ──────────────────────────────────────────────────────────
if corr and corr.get("demo_quote"):
    st.markdown(
        f'<div class="demo-quote">📊 {corr["demo_quote"]}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — Current Conditions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📡 Current Conditions")

if df is None or df.empty:
    if load_err:
        st.error(load_err)
    else:
        st.warning(
            "No data found. Run the pipeline first:\n"
            "```\npython scripts/run_pipeline.py\n```"
        )
else:
    if os.getenv("TEMP_DATA_MODE", "simulate") == "simulate":
        st.warning("⚠️ Warning: Using simulated temperature data.")

    latest = df.iloc[-1]
    recent_24h = df.tail(24)
    if len(df) > 24:
        recent_24h = df.tail(24)

    # Get current risk score
    current_risk: float | None = None
    current_demand_pred: float | None = None
    try:
        from src.model.predict import load_model, predict_risk
        model_bundle = load_model()
        result = predict_risk(
            model_bundle,
            temp_f=float(latest["temp_f"]),
            hour=int(latest.get("hour", latest["timestamp"].hour)),
            day_of_week=int(latest.get("day_of_week", latest["timestamp"].dayofweek)),
            month=int(latest.get("month", latest["timestamp"].month)),
        )
        current_risk = result["risk_score"]
        current_demand_pred = result["predicted_demand_mw"]
        risk_level = result["risk_level"]
    except RuntimeError:
        risk_level = "unknown"

    # ── Top metrics row ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{latest['temp_f']:.1f}°F",
            delta=f"{latest['temp_f'] - df.iloc[-2]['temp_f']:+.1f}°F vs 1h ago"
                  if len(df) >= 2 else None,
        )
    with col2:
        st.metric(
            label="⚡ Actual Demand",
            value=f"{latest['demand_mw']:,.0f} MW",
            delta=f"{latest['demand_mw'] - df.iloc[-2]['demand_mw']:+,.0f} MW"
                  if len(df) >= 2 else None,
        )
    with col3:
        if current_demand_pred:
            st.metric(label="🔮 Predicted Demand", value=f"{current_demand_pred:,.0f} MW")
        else:
            st.metric(label="🔮 Predicted Demand", value="—")
    with col4:
        if current_risk is not None:
            risk_color = "🔴" if current_risk >= 85 else "🟠" if current_risk >= 70 else "🟡" if current_risk >= 50 else "🟢"
            st.metric(
                label="⚠️ Risk Score",
                value=f"{risk_color} {current_risk:.0f}/100",
                delta=f"{risk_level.upper()}",
            )
            if current_risk >= threshold:
                st.error(f"🚨 Risk {current_risk:.0f} exceeds threshold {threshold} — ALERT TRIGGERED")
        else:
            st.metric(label="⚠️ Risk Score", value="— (run train first)")

    # ── Risk gauge ─────────────────────────────────────────────────────────────
    if current_risk is not None:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_risk,
            title={"text": "Current Risk Score", "font": {"color": "#e0e0e0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892a4"},
                "bar": {"color": _risk_color_hex(current_risk)},
                "bgcolor": "#1a1f2e",
                "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0, 50],  "color": "rgba(34,197,94,0.2)"},
                    {"range": [50, 70], "color": "rgba(234,179,8,0.2)"},
                    {"range": [70, 85], "color": "rgba(249,115,22,0.2)"},
                    {"range": [85, 100],"color": "rgba(239,68,68,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 3},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
            number={"font": {"color": "#e0e0e0"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0f1117",
            font={"color": "#e0e0e0"},
            height=250,
            margin={"t": 40, "b": 10, "l": 20, "r": 20},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── 24h historical chart ───────────────────────────────────────────────────
    st.markdown("**Last 24 hours — Temperature & Demand**")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=recent_24h["timestamp"], y=recent_24h["temp_f"],
        name="Temp (°F)", line={"color": "#ef4444", "width": 2},
        yaxis="y2",
    ))
    fig_hist.add_trace(go.Scatter(
        x=recent_24h["timestamp"], y=recent_24h["demand_mw"],
        name="Demand (MW)", line={"color": "#3b82f6", "width": 2},
        fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
    ))
    layout_hist = _dark_layout()
    layout_hist.update({
        "yaxis": {**layout_hist.get("yaxis", {}), "title": "Demand (MW)", "color": "#3b82f6"},
        "yaxis2": {"title": "Temp (°F)", "color": "#ef4444", "overlaying": "y", "side": "right"},
        "legend": {"orientation": "h", "y": 1.1},
        "height": 300,
    })
    fig_hist.update_layout(**layout_hist)
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — 24h Risk Forecast
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔮 24-Hour Risk Forecast")

if meta is None:
    st.info("No trained model found. Run `python scripts/run_train.py` to enable forecasting.")
else:
    forecast_df = load_forecast()
    if forecast_df is not None and not forecast_df.empty:
        # Color-coded risk bands
        fig_fc = go.Figure()

        # Background bands
        for lo, hi, col in [(0, 50, "rgba(34,197,94,0.08)"), (50, 70, "rgba(234,179,8,0.08)"),
                             (70, 85, "rgba(249,115,22,0.1)"), (85, 100, "rgba(239,68,68,0.12)")]:
            fig_fc.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0)

        # Threshold line
        fig_fc.add_hline(
            y=threshold, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Alert threshold ({threshold})",
            annotation_position="bottom right",
        )

        # Risk line
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["risk_score"],
            mode="lines+markers",
            name="Risk Score",
            line={"color": "#a78bfa", "width": 2.5},
            marker={
                "color": [_risk_color_hex(r) for r in forecast_df["risk_score"]],
                "size": 8,
            },
        ))

        # Temp overlay
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["temp_f"],
            name="Forecast Temp (°F)",
            line={"color": "#f87171", "width": 1.5, "dash": "dot"},
            yaxis="y2",
        ))

        layout_fc = _dark_layout()
        layout_fc.update({
            "yaxis": {**layout_fc.get("yaxis", {}), "title": "Risk Score (0–100)", "range": [0, 105]},
            "yaxis2": {"title": "Temp (°F)", "overlaying": "y", "side": "right", "color": "#f87171"},
            "height": 360,
            "legend": {"orientation": "h", "y": 1.05},
        })
        fig_fc.update_layout(**layout_fc)
        st.plotly_chart(fig_fc, use_container_width=True)

        # Summary table
        st.markdown("**Forecast summary**")
        display = forecast_df[["timestamp", "temp_f", "predicted_demand_mw", "risk_score", "risk_level"]].copy()
        display.columns = ["Time (UTC)", "Temp (°F)", "Demand (MW)", "Risk Score", "Level"]
        display["Time (UTC)"] = display["Time (UTC)"].dt.strftime("%H:%M")
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Model info
        st.caption(
            f"Model: {meta.get('model_type')} | "
            f"R²={meta.get('test_r2', '?'):.4f} | "
            f"RMSE={meta.get('test_rmse_mw', '?'):,.0f} MW"
        )
    else:
        st.info("Could not generate forecast. Make sure the model is trained.")


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — Agent Decision Log
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🤖 Agent Decision Log")

col_dec, col_alert = st.columns([2, 1])

with col_dec:
    st.markdown("**All Decisions**")
    if _DECISION_LOG.exists():
        lines = _DECISION_LOG.read_text(encoding="utf-8").strip().split("\n")
        if df is not None and not df.empty:
            log_start = df["timestamp"].min().strftime("%Y-%m-%d")
            lines = [line for line in lines if line.startswith("[") and line[1:11] >= log_start]
        # Show last 50 decisions, most recent first
        recent_lines = lines[-50:][::-1]
        if recent_lines:
            log_html = ""
            for line in recent_lines:
                is_alert = "ALERT" in line and "ok" not in line[:30]
                css_class = "alert-row" if is_alert else "ok-row"
                log_html += f'<div class="{css_class}">{line}</div>'
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info("No decisions recorded for the current data window.")
    else:
        st.info(
            "No decisions logged yet. Start the agent:\n"
            "```\npython scripts/run_agent.py --mode simulate\n```"
        )

with col_alert:
    st.markdown("**⚠️ Triggered Alerts**")
    if _ALERT_LOG.exists():
        alert_lines = _ALERT_LOG.read_text(encoding="utf-8").strip().split("\n")
        if df is not None and not df.empty:
            log_start = df["timestamp"].min().strftime("%Y-%m-%d")
            alert_lines = [line for line in alert_lines if line.startswith("[") and line[1:11] >= log_start]
        alert_lines = [l for l in alert_lines if l.strip()][-20:][::-1]
        if alert_lines:
            for line in alert_lines:
                st.markdown(
                    f'<div class="alert-row">🚨 {line}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No alerts triggered.")
    else:
        st.success("No alerts triggered yet.")


# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — Correlation Insights (collapsed by default)
# ─────────────────────────────────────────────────────────────────────────────
if corr:
    with st.expander("📈 Correlation Analysis Details", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Statistical Summary**")
            st.markdown(f"- Pearson r: **{corr.get('pearson_r', '?')}**")
            st.markdown(f"- Slope: **{corr.get('slope_mw_per_f', '?')} MW/°F**")
            st.markdown(f"- Best lag: **{corr.get('best_lag_hours', '?')} hours**")
            st.markdown(f"- Sample size: **{corr.get('n_samples', '?'):,} hours**")
        with col_b:
            st.markdown("**Lag Correlation Table**")
            if corr.get("lag_correlations"):
                lag_df = pd.DataFrame(corr["lag_correlations"])
                lag_df.columns = ["Lag (h)", "Pearson r", "p-value"]
                st.dataframe(lag_df, use_container_width=True, hide_index=True)

        if corr.get("correlation_plot"):
            st.image(str(_PROCESSED_DIR / "correlation_plot.png"))


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Heat Risk Agent | Last render: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
    f"Data: ERCOT {os.getenv('ERCOT_REGION', 'COAST')} | "
    f"Threshold: {threshold}/100"
)
