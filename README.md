# Heat Risk Agent 🌡️⚡

An end-to-end system that predicts heat-driven electricity grid strain and autonomously alerts when risk crosses a threshold. Built for a hackathon — prioritises a working pipeline over polish.

---

## Architecture

```
Temperature API  ──┐
                   ├──► Data Pipeline ──► Analysis ──► Model ──► Agent Loop ──► Alerts
ERCOT Demand API ──┘                                               │
                                                                   └──► Streamlit Dashboard
```

### Five layers

| Layer | What it does |
|-------|-------------|
| **Data** | Pulls temperature + ERCOT demand, aligns hourly, saves `combined.csv` |
| **Analysis** | Pearson correlation + lag analysis; prints a quotable finding |
| **Model** | Trains LinearRegression + GradientBoosting; saves best as `models/best_model.pkl` |
| **Agent** | Autonomous loop: polls temp, scores risk, alerts via Slack + log when threshold crossed |
| **Dashboard** | Streamlit app showing current conditions, 24h forecast, and agent decision log |

---

## Quickstart

### 1. Clone and create environment

```bash
git clone <repo>
cd heat-risk-agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Run the test suite:

```bash
pytest
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — the only required change for a zero-key demo is:
#   TEMP_DATA_MODE=simulate   (already the default)
```

### 3. Run each layer in order

```bash
# Layer 1: Pull and align data (default: June–Aug 2024 simulated)
python scripts/run_pipeline.py

# Layer 2: Correlation analysis
python scripts/run_analysis.py

# Layer 3: Train the model
python scripts/run_train.py

# Layer 4: Run the agent (simulate mode replays historical data)
python scripts/run_agent.py --mode simulate

# Layer 5: Launch the dashboard
streamlit run src/dashboard/app.py
```

---

## Environment Variables (`.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TEMP_DATA_MODE` | No | `simulate` | Temperature backend: `simulate` / `open_meteo` / `fortyguard` |
| `DEMAND_DATA_MODE` | No | `auto` | Demand backend mode: `auto` or `simulate` |
| `FORTYGUARD_API_KEY` | Only if `fortyguard` mode | — | FortyGuard API key |
| `EIA_API_KEY` | No | — | EIA Open Data key (enables EIA fallback for demand data) |
| `ERCOT_REGION` | No | `COAST` | ERCOT weather zone for demand data |
| `RISK_THRESHOLD` | No | `70` | Risk score (0–100) above which agent alerts |
| `AGENT_POLL_INTERVAL_SECONDS` | No | `300` | How often the agent polls (seconds) |
| `ALERT_COOLDOWN_MINUTES` | No | `60` | Minimum time between repeated alerts |
| `ALERT_BACKENDS` | No | `log,slack` | Comma-separated: `log`, `slack` |
| `SLACK_WEBHOOK_URL` | Only if `slack` backend | — | Slack incoming webhook URL |
| `LOCATION_LAT` | No | `29.7604` | Latitude (Houston, TX default) |
| `LOCATION_LON` | No | `-95.3698` | Longitude |
| `LOCATION_NAME` | No | `Houston_TX` | Human-readable location name |

---

## Data Sources

| Source | Type | Key needed? |
|--------|------|-------------|
| **Simulate** | Synthetic Texas summer sinusoid + noise | ❌ None |
| **Open-Meteo** | Free historical weather API | ❌ None |
| **FortyGuard** | Commercial temperature API | ✅ `FORTYGUARD_API_KEY` |
| **ERCOT** | Public Texas grid demand data | ❌ None |
| **EIA** | US energy demand API (fallback) | ✅ `EIA_API_KEY` (free) |

---

## Project Structure

```
heat-risk-agent/
├── src/
│   ├── data/
│   │   ├── temperature_client.py   # Temperature backends
│   │   ├── ercot_client.py         # Demand data
│   │   └── pipeline.py             # Align + save combined dataset
│   ├── analysis/
│   │   └── correlation.py          # Correlation + lag analysis
│   ├── model/
│   │   ├── features.py             # Feature engineering
│   │   ├── train.py                # Model training + evaluation
│   │   └── predict.py              # Inference + risk scoring
│   ├── agent/
│   │   ├── alerts.py               # Slack + log alert backends
│   │   └── loop.py                 # Autonomous agent loop
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard
├── scripts/
│   ├── run_pipeline.py             # CLI for Layer 1
│   ├── run_analysis.py             # CLI for Layer 2
│   ├── run_train.py                # CLI for Layer 3
│   └── run_agent.py                # CLI for Layer 4
├── data/
│   ├── raw/                        # Raw API pull CSVs (.gitignored)
│   └── processed/                  # combined.csv (.gitignored)
├── models/                         # Saved model .pkl files (.gitignored)
├── logs/                           # Agent decision + alert logs
├── .env.example                    # Copy to .env and fill in keys
├── requirements.txt
└── README.md
```

---

## Demo Notes

**Quotable finding** (printed by `run_analysis.py` after running):
> *"In ERCOT COAST zone summer data, every 10°F increase in temperature correlates with an X% increase in electricity demand (Pearson r = Y, peak lag = Z hours)."*

**Risk score scale:**
- 0–49: Normal operations (green)
- 50–69: Elevated risk — monitor (yellow)
- 70–84: High risk — agent alerts (orange)
- 85–100: Critical risk — immediate response (red)

**Slack alert format:**
```
🚨 HEAT RISK ALERT — Houston_TX / ERCOT COAST
Time     : 2024-07-15 17:00 UTC
Temp     : 104.3°F
Demand   : 24,150 MW
Risk     : 82/100 (threshold: 70)
Action   : Alert sent — grid operators should prepare demand response.
```

---

## Tech Stack

- **Python 3.11+**
- **pandas + scipy** — data alignment and correlation
- **scikit-learn + joblib** — model training and inference
- **requests + python-dotenv** — API clients
- **Streamlit + Plotly** — dashboard
