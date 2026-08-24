"""
alerts.py
──────────
Alert backends for the Heat Risk Agent.

Two backends, both implementing the same interface:
    LogFileAlert   — writes plain-language alerts to logs/agent_alerts.log
    SlackWebhookAlert — posts to a Slack incoming webhook URL

AlertManager coordinates both and handles deduplication (cooldown window).

Usage
-----
    from src.agent.alerts import AlertManager
    manager = AlertManager()
    manager.send_alert(temp_f=104.2, demand_mw=24500, risk_score=82.3, threshold=70)
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS_DIR = _PROJECT_ROOT / "logs"

_SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
_ALERT_BACKENDS = [b.strip() for b in os.getenv("ALERT_BACKENDS", "log").split(",")]
_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))


# ── AlertManager ───────────────────────────────────────────────────────────────

class AlertManager:
    """
    Coordinates alert delivery across configured backends with deduplication.
    """

    def __init__(
        self,
        backends: list[str] | None = None,
        cooldown_minutes: int | None = None,
    ) -> None:
        self._backends = backends or _ALERT_BACKENDS
        self._cooldown = timedelta(minutes=cooldown_minutes or _COOLDOWN_MINUTES)
        self._state_file = _LOGS_DIR / ".last_alert_time"
        self._last_alert_time = self._load_last_alert_time()

        logger.info(
            "AlertManager ready — backends=%s  cooldown=%s",
            self._backends, self._cooldown,
        )

    def _load_last_alert_time(self) -> datetime | None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    timestamp_str = f.read().strip()
                    if timestamp_str:
                        return datetime.fromisoformat(timestamp_str)
            except Exception as e:
                logger.warning("Failed to read last alert time from %s: %s", self._state_file, e)
        return None

    def _save_last_alert_time(self, dt: datetime) -> None:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._state_file, "w") as f:
                f.write(dt.isoformat())
        except Exception as e:
            logger.warning("Failed to save last alert time to %s: %s", self._state_file, e)

    def send_alert(
        self,
        temp_f: float,
        demand_mw: float,
        risk_score: float,
        threshold: float,
        location: str = "",
        region: str = "",
        extra: dict | None = None,
    ) -> bool:
        """
        Send an alert through all configured backends, respecting cooldown.

        Returns True if alert was sent, False if suppressed by cooldown.
        """
        now = datetime.now(timezone.utc)
        if self._last_alert_time and (now - self._last_alert_time) < self._cooldown:
            remaining = self._cooldown - (now - self._last_alert_time)
            logger.info(
                "Alert suppressed by cooldown (%.0f min remaining)",
                remaining.total_seconds() / 60,
            )
            return False

        payload = _build_payload(
            temp_f=temp_f,
            demand_mw=demand_mw,
            risk_score=risk_score,
            threshold=threshold,
            location=location or os.getenv("LOCATION_NAME", "Houston_TX"),
            region=region or os.getenv("ERCOT_REGION", "COAST"),
            extra=extra,
        )

        sent = False
        for backend in self._backends:
            try:
                if backend == "log":
                    _log_alert(payload)
                    sent = True
                elif backend == "slack":
                    _slack_alert(payload)
                    sent = True
                else:
                    logger.warning("Unknown alert backend: %s", backend)
            except requests.RequestException as exc:
                logger.error("Alert backend '%s' failed: %s", backend, exc)

        if sent:
            self._last_alert_time = now
            self._save_last_alert_time(now)
        return sent


# ── Log backend ────────────────────────────────────────────────────────────────

def _log_alert(payload: dict) -> None:
    """Write a plain-language alert to logs/agent_alerts.log."""
    alert_log = _LOGS_DIR / "agent_alerts.log"
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    line = (
        f"[{payload['timestamp']}] [ALERT] | "
        f"{payload['location']} / ERCOT {payload['region']} | "
        f"Temp={payload['temp_f']:.1f}°F | "
        f"Demand={payload['demand_mw']:,.0f} MW | "
        f"Risk={payload['risk_score']:.1f}/100 (threshold={payload['threshold']}) | "
        f"{payload['message']}"
    )

    with open(alert_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    logger.info("Alert written to %s", alert_log)
    print(f"\n{'='*60}")
    print(line)
    print(f"{'='*60}\n")


# ── Slack backend ──────────────────────────────────────────────────────────────

def _slack_alert(payload: dict) -> None:
    """Post a formatted alert to a Slack incoming webhook."""
    if not _SLACK_WEBHOOK_URL or not _SLACK_WEBHOOK_URL.startswith("https://hooks.slack.com/services/T"):
        # Default placeholder — skip silently but warn
        logger.warning(
            "SLACK_WEBHOOK_URL not configured. "
            "Set it in .env to enable Slack alerts."
        )
        return

    risk_emoji = (
        "🔴" if payload["risk_score"] >= 85
        else "🟠" if payload["risk_score"] >= 70
        else "🟡"
    )

    slack_body = {
        "text": f"{risk_emoji} *HEAT RISK ALERT* — {payload['location']} / ERCOT {payload['region']}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{risk_emoji} HEAT RISK ALERT — {payload['location']}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Time (UTC):*\n{payload['timestamp']}"},
                    {"type": "mrkdwn", "text": f"*Region:*\nERCOT {payload['region']}"},
                    {"type": "mrkdwn", "text": f"*Temperature:*\n{payload['temp_f']:.1f}°F"},
                    {"type": "mrkdwn", "text": f"*Demand:*\n{payload['demand_mw']:,.0f} MW"},
                    {"type": "mrkdwn", "text": f"*Risk Score:*\n{payload['risk_score']:.1f} / 100"},
                    {"type": "mrkdwn", "text": f"*Threshold:*\n{payload['threshold']}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Action:* {payload['message']}"},
            },
            {"type": "divider"},
        ],
    }

    resp = requests.post(
        _SLACK_WEBHOOK_URL,
        data=json.dumps(slack_body),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    logger.info("Slack alert sent (status %d)", resp.status_code)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_payload(
    temp_f: float,
    demand_mw: float,
    risk_score: float,
    threshold: float,
    location: str,
    region: str,
    extra: dict | None,
) -> dict:
    level = (
        "CRITICAL" if risk_score >= 85
        else "HIGH" if risk_score >= 70
        else "ELEVATED"
    )
    msg = (
        f"Grid strain risk is {level} at {risk_score:.0f}/100. "
        f"Demand forecast ({demand_mw:,.0f} MW) exceeds threshold. "
        f"Grid operators should prepare demand response measures."
    )
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "location": location,
        "region": region,
        "temp_f": temp_f,
        "demand_mw": demand_mw,
        "risk_score": risk_score,
        "threshold": threshold,
        "risk_level": level,
        "message": msg,
        **(extra or {}),
    }
