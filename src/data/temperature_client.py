"""
temperature_client.py
─────────────────────
Unified temperature data client with three interchangeable backends:

  simulate   — Synthetic Texas summer data; works with zero keys.
  open_meteo — Free Open-Meteo historical/forecast API; no key required.
  fortyguard — FortyGuard Temperature API; requires FORTYGUARD_API_KEY env var.

All three return the same DataFrame schema:
    timestamp (UTC, tz-naive)  |  location  |  temp_f  |  temp_c

Usage:
    client = TemperatureClient(mode="simulate")
    df = client.fetch(start="2024-06-01", end="2024-08-31")
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.rate_limit import rate_limit

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_LAT = float(os.getenv("LOCATION_LAT", "29.7604"))   # Houston, TX
_DEFAULT_LON = float(os.getenv("LOCATION_LON", "-95.3698"))
_DEFAULT_LOCATION = os.getenv("LOCATION_NAME", "Houston_TX")

# Typical ERCOT COAST summer temp stats (°F) used in simulation
_SIM_SUMMER_BASE_F = 95.0     # Mean daily high
_SIM_AMPLITUDE_F = 8.0        # ±amplitude of daily cycle
_SIM_NOISE_STD_F = 2.5        # Random noise std dev


# ── Public API ─────────────────────────────────────────────────────────────────

class TemperatureClient:
    """Fetch hourly temperature data from the configured backend."""

    def __init__(
        self,
        mode: str | None = None,
        lat: float = _DEFAULT_LAT,
        lon: float = _DEFAULT_LON,
        location: str = _DEFAULT_LOCATION,
    ) -> None:
        self.mode = (mode or os.getenv("TEMP_DATA_MODE", "simulate")).lower()
        self.lat = lat
        self.lon = lon
        self.location = location
        logger.info("TemperatureClient initialised in mode=%s", self.mode)

    # ── Main entry point ───────────────────────────────────────────────────────

    def fetch(
        self,
        start: str | datetime,
        end: str | datetime,
    ) -> dict:
        """
        Fetch hourly temperature data for [start, end] (inclusive).

        Parameters
        ----------
        start : str | datetime  e.g. "2024-06-01" or datetime object
        end   : str | datetime  e.g. "2024-08-31"

        Returns
        -------
        dict with keys:
            data: pd.DataFrame with columns: timestamp, location, temp_f, temp_c
            source: str
            warning: str
        """
        start_dt = _parse_date(start)
        end_dt = _parse_date(end, end_of_day=True)

        warning = ""
        if self.mode == "simulate":
            df = self._fetch_simulate(start_dt, end_dt)
        elif self.mode == "open_meteo":
            df = self._fetch_open_meteo(start_dt, end_dt)
        elif self.mode == "fortyguard":
            try:
                df = self._fetch_fortyguard(start_dt, end_dt)
            except Exception as e:
                warning = f"FortyGuard API failed ({e}). Falling back to simulate."
                logger.warning(warning)
                self.mode = "simulate"
                df = self._fetch_simulate(start_dt, end_dt)
        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Choose: simulate | open_meteo | fortyguard"
            )

        df = _validate_and_clean(df)
        logger.info(
            "Fetched %d temperature records (%s → %s) via mode=%s",
            len(df), df["timestamp"].min() if not df.empty else "N/A", df["timestamp"].max() if not df.empty else "N/A", self.mode,
        )
        return {"data": df, "source": self.mode, "warning": warning}

    # ── Backend: simulate ──────────────────────────────────────────────────────

    def _fetch_simulate(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Generate synthetic hourly temperature data mimicking Texas summers."""
        rng = np.random.default_rng(seed=42)  # deterministic for reproducibility

        hours = pd.date_range(start=start_dt, end=end_dt, freq="h", tz=None)
        records = []

        for ts in hours:
            # Daily sinusoid: peak at 15:00 local, trough at 06:00
            hour_of_day = ts.hour + ts.minute / 60.0
            daily_cycle = _SIM_AMPLITUDE_F * math.sin(
                math.pi * (hour_of_day - 6.0) / 12.0
            )

            # Seasonal trend: hotter June–August
            day_of_year = ts.timetuple().tm_yday
            seasonal = 5.0 * math.sin(math.pi * (day_of_year - 90) / 180.0)

            # Random noise
            noise = rng.normal(0, _SIM_NOISE_STD_F)

            temp_f = _SIM_SUMMER_BASE_F + daily_cycle + seasonal + noise
            records.append(
                {
                    "timestamp": ts.replace(tzinfo=None),  # tz-naive UTC
                    "location": self.location,
                    "temp_f": round(temp_f, 2),
                    "temp_c": round(_f_to_c(temp_f), 2),
                }
            )

        return pd.DataFrame(records)

    # ── Backend: Open-Meteo (free, no key) ────────────────────────────────────

    @rate_limit(1.0)
    def _fetch_open_meteo(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Pull hourly temperature from the Open-Meteo free historical API.
        Docs: https://open-meteo.com/en/docs/historical-weather-api
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
        }

        logger.info("Requesting Open-Meteo: %s", params)
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        times = data["hourly"]["time"]          # list of "YYYY-MM-DDTHH:MM"
        temps = data["hourly"]["temperature_2m"]

        records = []
        for t, temp_f in zip(times, temps):
            if temp_f is None:
                continue
            records.append(
                {
                    "timestamp": datetime.fromisoformat(t),
                    "location": self.location,
                    "temp_f": round(float(temp_f), 2),
                    "temp_c": round(_f_to_c(float(temp_f)), 2),
                }
            )

        return pd.DataFrame(records)

    # ── Backend: FortyGuard (requires API key) ─────────────────────────────────

    @rate_limit(1.0)
    def _fetch_fortyguard(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Pull hourly temperature snapshots through FortyGuard's documented API.
        The API submits asynchronous heatmap tasks rather than returning a
        temperature series directly, so each requested hour is polled and the
        temperature is extracted from the returned GeoJSON tiles.
        """
        api_key = os.getenv("FORTYGUARD_API_KEY", "")
        if not api_key:
            raise OSError(
                "FORTYGUARD_API_KEY is not set. "
                "Set it in your .env file or switch TEMP_DATA_MODE to 'open_meteo'."
            )

        if end_dt - start_dt > timedelta(hours=24):
            raise ValueError(
                "FortyGuard heatmap tasks are requested one hour at a time. "
                "Use a maximum 24-hour window for fortyguard mode or use open_meteo for history."
            )

        url = "https://api.fortyguard.com/v1/heatmap"
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        all_records: list[dict] = []
        request_time = start_dt.replace(minute=0, second=0, microsecond=0)
        while request_time <= end_dt:
            payload = {
                "polygon_aoi": self._fortyguard_polygon(),
                "date_time": {
                    "start_date": request_time.strftime("%Y-%m-%d"),
                    "start_time": request_time.strftime("%H:%M"),
                    "filter_type": 1,
                },
                "granularity": 100,
            }

            logger.info("Submitting FortyGuard heatmap task for %s", request_time)
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            activity_id = response.json().get("data", {}).get("activity_id")
            if not activity_id:
                raise ValueError(f"FortyGuard response did not include activity_id: {response.text[:500]}")

            result = self._poll_fortyguard(activity_id, api_key)
            temperature_c = self._extract_fortyguard_temperature(result)
            if temperature_c is None:
                raise ValueError("FortyGuard completed task contained no temperature tile")
            temperature_f = temperature_c * 9.0 / 5.0 + 32.0
            all_records.append({
                "timestamp": request_time,
                "location": self.location,
                "temp_f": round(temperature_f, 2),
                "temp_c": round(temperature_c, 2),
            })
            request_time += timedelta(hours=1)

        if not all_records:
            raise ValueError("FortyGuard returned no temperature records")

        return pd.DataFrame(all_records)

    def _fortyguard_polygon(self) -> dict:
        """Return a small valid GeoJSON area centered on the configured point."""
        delta = 0.005
        ring = [
            [self.lon - delta, self.lat - delta],
            [self.lon + delta, self.lat - delta],
            [self.lon + delta, self.lat + delta],
            [self.lon - delta, self.lat + delta],
            [self.lon - delta, self.lat - delta],
        ]
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }],
        }

    @staticmethod
    def _poll_fortyguard(activity_id: str, api_key: str) -> dict:
        """Poll a submitted FortyGuard task with a bounded wait."""
        import time

        url = f"https://api.fortyguard.com/v1/status/{activity_id}"
        for _ in range(60):
            response = requests.get(url, headers={"api-key": api_key}, timeout=60)
            response.raise_for_status()
            body = response.json()
            data = body.get("data", {})
            status = str(data.get("status", "")).lower()
            if status in {"completed", "succeeded"}:
                return data
            if status in {"failed", "error"}:
                raise RuntimeError(f"FortyGuard task {activity_id} failed: {body}")
            time.sleep(5)
        raise TimeoutError(f"FortyGuard task {activity_id} did not complete within 5 minutes")

    @staticmethod
    def _extract_fortyguard_temperature(data: dict) -> float | None:
        """Extract the mean Celsius temperature from completed heatmap tiles."""
        result = data.get("result", {})
        features = result.get("map_data", {}).get("features", [])
        values = []
        for feature in features:
            properties = feature.get("properties", {})
            for key in ("temperature", "temp", "value", "temperature_celsius"):
                if properties.get(key) is not None:
                    values.append(float(properties[key]))
                    break
        return sum(values) / len(values) if values else None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_date(value: str | datetime, end_of_day: bool = False) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    dt = datetime.fromisoformat(str(value))
    if end_of_day and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=None)


def _f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column types, sort, drop dupes/NaNs."""
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "location"])
    df = df.dropna(subset=["temp_f"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
