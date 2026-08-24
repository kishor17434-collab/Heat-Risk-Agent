"""
loop.py
────────
Autonomous agent loop for the Heat Risk system.

Two modes
─────────
  live      — polls the temperature API every AGENT_POLL_INTERVAL_SECONDS seconds
               (useful when you have a real API key)
  simulate  — replays historical data from combined.csv at accelerated speed
               (great for demos — shows a full summer's worth of events in minutes)

The agent:
  1. Gets current temperature (real or replayed)
  2. Runs the trained model to get a risk score
  3. Logs the decision to logs/agent_decisions.log
  4. If risk ≥ RISK_THRESHOLD → fires AlertManager

Usage
-----
    from src.agent.loop import AgentLoop
    loop = AgentLoop()
    loop.run(mode="simulate")     # blocking

Or via CLI:
    python scripts/run_agent.py --mode simulate
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.agent.alerts import AlertManager
from src.model.predict import load_model, predict_risk
from src.preflight import validate_required_paths

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS_DIR = _PROJECT_ROOT / "logs"
_DECISION_LOG = _LOGS_DIR / "agent_decisions.log"
_COMBINED_CSV = _PROJECT_ROOT / "data" / "processed" / "combined.csv"

_RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "70"))
_POLL_INTERVAL = int(os.getenv("AGENT_POLL_INTERVAL_SECONDS", "300"))

# In simulate mode, each historical hour is replayed with this wall-clock delay
_SIMULATE_STEP_SECONDS = float(os.getenv("SIMULATE_STEP_SECONDS", "0.5"))


class AgentLoop:
    """
    Autonomous heat risk monitoring agent.
    """

    def __init__(
        self,
        threshold: float | None = None,
        poll_interval: int | None = None,
    ) -> None:
        self.threshold = threshold if threshold is not None else _RISK_THRESHOLD
        self.poll_interval = poll_interval or _POLL_INTERVAL
        self.alert_manager = AlertManager()

        logger.info(
            "AgentLoop initialised — threshold=%.0f  poll_interval=%ds",
            self.threshold, self.poll_interval,
        )

        # Load model once at startup
        logger.info("Loading model …")
        self.model = load_model()
        logger.info("Model loaded: %s", self.model["meta"].get("model_type", "unknown"))

    def __enter__(self):
        logger.info("AgentLoop started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("AgentLoop shutting down.")
        # Any cleanup goes here
        if exc_type:
            logger.error("AgentLoop exited with error: %s", exc_val)
        return False

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(self, mode: str = "simulate", max_steps: int | None = None) -> None:
        """
        Start the agent loop.

        Parameters
        ----------
        mode      : "simulate" | "live"
        max_steps : Stop after N steps (useful for testing; None = run forever)
        """
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)

        if mode == "simulate":
            self._run_simulate(max_steps=max_steps)
        elif mode == "live":
            self._run_live(max_steps=max_steps)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use: simulate | live")

    # ── Simulate mode ──────────────────────────────────────────────────────────

    def _run_simulate(self, max_steps: int | None) -> None:
        """Replay historical combined.csv data row by row."""
        validate_required_paths([("combined dataset", _COMBINED_CSV)])

        df = pd.read_csv(_COMBINED_CSV, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "Simulate mode: replaying %d hours from %s to %s",
            len(df), df["timestamp"].min(), df["timestamp"].max(),
        )

        print(f"\n{'='*65}")
        print("  HEAT RISK AGENT — SIMULATE MODE")
        print(f"  Replaying {len(df):,} hours | Threshold: {self.threshold}/100")
        print(f"  Step delay: {_SIMULATE_STEP_SECONDS}s per hour")
        print(f"{'='*65}")

        steps = 0
        for _, row in df.iterrows():
            if max_steps is not None and steps >= max_steps:
                break

            self._process_observation(
                timestamp=row["timestamp"],
                temp_f=float(row["temp_f"]),
                demand_actual_mw=float(row["demand_mw"]),
                hour=int(row.get("hour", row["timestamp"].hour)),
                day_of_week=int(row.get("day_of_week", row["timestamp"].dayofweek)),
                month=int(row.get("month", row["timestamp"].month)),
            )
            steps += 1
            time.sleep(_SIMULATE_STEP_SECONDS)

        print(f"\n[DONE] Simulation complete ({steps} steps processed).")

    # ── Live mode ──────────────────────────────────────────────────────────────

    def _run_live(self, max_steps: int | None) -> None:
        """Poll the temperature API in real time."""
        from datetime import timedelta

        from src.data.temperature_client import TemperatureClient

        client = TemperatureClient()
        steps = 0

        print(f"\n{'='*65}")
        print("  HEAT RISK AGENT — LIVE MODE")
        print(f"  Poll interval: {self.poll_interval}s | Threshold: {self.threshold}/100")
        print("  Press Ctrl+C to stop.")
        print(f"{'='*65}")

        try:
            while True:
                if max_steps is not None and steps >= max_steps:
                    break

                from datetime import timezone
                now = datetime.now(timezone.utc)
                start = now - timedelta(hours=1)

                try:
                    temp_res = client.fetch(start=start, end=now)
                    temp_df = temp_res["data"]
                    if temp_df.empty:
                        logger.warning("Temperature API returned no data — skipping cycle")
                        time.sleep(self.poll_interval)
                        continue

                    latest = temp_df.iloc[-1]
                    self._process_observation(
                        timestamp=now,
                        temp_f=float(latest["temp_f"]),
                        demand_actual_mw=None,
                        hour=now.hour,
                        day_of_week=now.weekday(),
                        month=now.month,
                    )
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    logger.error("Error in live poll cycle: %s", exc)

                steps += 1
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print(f"\n[STOPPED] Agent stopped by user after {steps} cycles.")

    # ── Core decision logic ────────────────────────────────────────────────────

    def _process_observation(
        self,
        timestamp: datetime,
        temp_f: float,
        hour: int,
        day_of_week: int,
        month: int,
        demand_actual_mw: float | None = None,
    ) -> dict:
        """Run the model, log the decision, and fire alerts if needed."""
        decision = self._make_decision(
            timestamp=timestamp,
            temp_f=temp_f,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            demand_actual_mw=demand_actual_mw,
        )
        self._execute_side_effects(decision)
        return decision

    def _make_decision(
        self,
        timestamp: datetime,
        temp_f: float,
        hour: int,
        day_of_week: int,
        month: int,
        demand_actual_mw: float | None = None,
    ) -> dict:
        """Pure function: calculate risk score and build the decision object."""
        result = predict_risk(
            self.model,
            temp_f=temp_f,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
        )

        risk_score = result["risk_score"]
        predicted_demand = result["predicted_demand_mw"]
        risk_level = result["risk_level"]
        triggered = risk_score >= self.threshold

        return {
            "timestamp": str(timestamp)[:16],
            "temp_f": round(temp_f, 1),
            "predicted_demand_mw": predicted_demand,
            "actual_demand_mw": round(demand_actual_mw, 0) if demand_actual_mw else None,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "threshold": self.threshold,
            "alert_triggered": triggered,
        }

    def _execute_side_effects(self, decision: dict) -> None:
        """Execute side effects (logging and alerting) based on a decision."""
        # ── Log decision ───────────────────────────────────────────────────────
        self._write_decision_log(decision)

        # ── Console output ─────────────────────────────────────────────────────
        triggered = decision["alert_triggered"]
        risk_score = decision["risk_score"]
        emoji = "[!!]" if triggered else ("[!] " if risk_score >= 50 else "[ok]")
        print(
            f"  {emoji} [{decision['timestamp']}]  "
            f"Temp={decision['temp_f']:.1f}°F  "
            f"Demand≈{decision['predicted_demand_mw']:,.0f}MW  "
            f"Risk={risk_score:.0f}/100 ({decision['risk_level']})"
            + (" ← ALERT" if triggered else "")
        )

        # ── Fire alert ─────────────────────────────────────────────────────────
        if triggered:
            self.alert_manager.send_alert(
                temp_f=decision['temp_f'],
                demand_mw=decision['predicted_demand_mw'],
                risk_score=risk_score,
                threshold=self.threshold,
            )

    # ── Decision log ───────────────────────────────────────────────────────────

    def _write_decision_log(self, decision: dict) -> None:
        """Append a decision unless the exact entry is already the latest line."""
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        flag = "ALERT" if decision["alert_triggered"] else "ok"
        line = (
            f"[{decision['timestamp']}] {flag} | "
            f"temp={decision['temp_f']}°F | "
            f"demand≈{decision['predicted_demand_mw']:,.0f}MW | "
            f"risk={decision['risk_score']:.1f} | "
            f"level={decision['risk_level']}"
        )
        if _DECISION_LOG.exists():
            with open(_DECISION_LOG, "r", encoding="utf-8") as f:
                if f.read().splitlines()[-1:] == [line]:
                    return
        with open(_DECISION_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
