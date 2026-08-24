"""
config.py
──────────
Centralized configuration management for the Heat Risk Agent.
Loads and validates environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    location_name: str
    ercot_region: str
    lat: float
    lon: float
    timezone: str
    slack_webhook_url: str
    alert_backends: list[str]
    alert_cooldown_minutes: int
    risk_threshold: float
    agent_poll_interval_seconds: int

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()
        
        # Validations and parsing
        try:
            lat = float(os.getenv("LAT", "29.7604"))
            lon = float(os.getenv("LON", "-95.3698"))
        except ValueError:
            lat, lon = 29.7604, -95.3698
            
        try:
            cooldown = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))
        except ValueError:
            cooldown = 60
            
        try:
            threshold = float(os.getenv("RISK_THRESHOLD", "70"))
        except ValueError:
            threshold = 70.0
            
        try:
            poll_interval = int(os.getenv("AGENT_POLL_INTERVAL_SECONDS", "300"))
        except ValueError:
            poll_interval = 300

        backends_raw = os.getenv("ALERT_BACKENDS", "log")
        backends = [b.strip() for b in backends_raw.split(",") if b.strip()]

        return cls(
            location_name=os.getenv("LOCATION_NAME", "Houston_TX"),
            ercot_region=os.getenv("ERCOT_REGION", "COAST"),
            lat=lat,
            lon=lon,
            timezone=os.getenv("TIMEZONE", "America/Chicago"),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            alert_backends=backends,
            alert_cooldown_minutes=cooldown,
            risk_threshold=threshold,
            agent_poll_interval_seconds=poll_interval,
        )

# Global config instance
config = Config.from_env()
