"""
ercot_client.py
───────────────
Pulls hourly electricity demand data for ERCOT (Texas grid).

Two data sources, tried in order:
  1. ERCOT public data portal — free, no key required.
     Uses the ERCOT Hourly Load (LZASP) or System-Wide Actual Load reports.
  2. EIA Open Data API v2 — free with a key (EIA_API_KEY env var).
     Falls back to this when ERCOT direct pull fails.

Both return the same DataFrame schema:
    timestamp (UTC, tz-naive)  |  region  |  demand_mw

Usage:
    client = ERCOTClient()
    df = client.fetch(start="2024-06-01", end="2024-08-31")
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.rate_limit import rate_limit

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.getenv("ERCOT_REGION", "COAST")
_DEMAND_DATA_MODE = os.getenv("DEMAND_DATA_MODE", "auto").lower()
_EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# ERCOT public reports base — Hourly Load data by weather zone
# ERCOT publishes daily CSV zips at this endpoint pattern.
_ERCOT_REPORTS_BASE = "https://www.ercot.com/api/1/services/read/dashboards/systemWideDemand"

# EIA API v2 endpoint for RTO regional data
_EIA_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# Simulate constants
_SIM_BASE_DEMAND_MW = 18_000
_SIM_PEAK_ADDITION = 7_000
_SIM_WEEKEND_FACTOR = 0.88


class ERCOTClient:
    """Fetch hourly electricity demand data from ERCOT / EIA."""

    def __init__(self, region: str | None = None, mode: str | None = None) -> None:
        self.region = (region or _DEFAULT_REGION).upper()
        self.mode = (mode or os.getenv("DEMAND_DATA_MODE", _DEMAND_DATA_MODE)).lower()
        if self.mode not in {"auto", "simulate"}:
            raise ValueError("DEMAND_DATA_MODE must be 'auto' or 'simulate'")
        logger.info("ERCOTClient initialised for region=%s", self.region)

    # ── Main entry point ───────────────────────────────────────────────────────

    def fetch(
        self,
        start: str | datetime,
        end: str | datetime,
    ) -> dict:
        """
        Fetch hourly demand data for [start, end].

        Returns
        -------
        dict with keys:
            data: pd.DataFrame with columns: timestamp, region, demand_mw
            source: str ("ERCOT", "EIA", or "Simulated")
            warning: str (any warning messages, or empty)
        """
        start_dt = _parse_date(start)
        end_dt = _parse_date(end, end_of_day=True)

        if self.mode == "simulate":
            msg = "Demand data forced to simulation by DEMAND_DATA_MODE=simulate."
            logger.info(msg)
            return {
                "data": _validate_and_clean(self._fetch_simulate(start_dt, end_dt)),
                "source": "Simulated",
                "warning": msg,
            }

        # Try ERCOT dashboard API first
        logger.info("Attempting ERCOT public API …")
        try:
            df = self._fetch_ercot_dashboard(start_dt, end_dt)
            if not df.empty:
                logger.info("ERCOT public API returned %d records", len(df))
                return {"data": _validate_and_clean(df), "source": "ERCOT", "warning": ""}
        except requests.RequestException as exc:
            logger.warning("ERCOT public API failed (%s) — trying EIA fallback", exc)

        # EIA fallback
        if _EIA_API_KEY:
            logger.info("Attempting EIA API fallback …")
            try:
                df = self._fetch_eia(start_dt, end_dt)
                if not df.empty:
                    logger.info("EIA API returned %d records", len(df))
                    return {"data": _validate_and_clean(df), "source": "EIA", "warning": ""}
            except requests.RequestException as exc:
                logger.warning("EIA API also failed (%s) — using simulated demand", exc)
        else:
            logger.info("EIA_API_KEY not set — skipping EIA fallback")

        # Last resort: simulate realistic ERCOT demand
        msg = "All real sources failed. Generating simulated demand data."
        logger.warning(msg)
        df = self._fetch_simulate(start_dt, end_dt)
        return {"data": _validate_and_clean(df), "source": "Simulated", "warning": msg}

    # ── Backend: ERCOT Dashboard API ───────────────────────────────────────────

    @rate_limit(1.0)
    def _fetch_ercot_dashboard(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Pull from the ERCOT system-wide demand dashboard endpoint.
        Returns aggregated hourly load in MW for the COAST zone (or TOTAL).
        """
        # ERCOT's public dashboard provides system-wide & zone data.
        # We request the time range in chunks since ERCOT API can time out on long ranges.
        all_records: list[dict] = []
        total_fallback_count = 0
        chunk_start = start_dt

        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + timedelta(days=7), end_dt)
            params = {
                "startDateTime": chunk_start.strftime("%Y-%m-%dT%H:%M:%S"),
                "endDateTime": chunk_end.strftime("%Y-%m-%dT%H:%M:%S"),
                "fields": "OprDt,HourEnding,COAST,TOTAL",  # request weather zones
            }
            resp = requests.get(_ERCOT_REPORTS_BASE, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            # ERCOT returns {"data": [{"OprDt": "...", "HourEnding": "01:00", "COAST": 12345.0, ...}]}
            for entry in data.get("data", []):
                ts = _parse_ercot_timestamp(
                    entry.get("OprDt", ""), entry.get("HourEnding", "")
                )
                if ts is None:
                    continue
                regional_demand = entry.get(self.region)
                used_total_fallback = regional_demand is None
                demand = regional_demand if regional_demand is not None else entry.get("TOTAL")
                if demand is None:
                    continue
                if used_total_fallback:
                    total_fallback_count += 1
                all_records.append(
                    {
                        "timestamp": ts,
                        "region": self.region,
                        "region_source": "TOTAL_fallback" if used_total_fallback else self.region,
                        "demand_mw": float(demand),
                    }
                )
            chunk_start = chunk_end + timedelta(hours=1)

        if total_fallback_count:
            logger.warning(
                "ERCOT response missing '%s' field for %d records — used "
                "system-wide TOTAL instead. Demand values reflect ERCOT-wide "
                "load, not the %s zone.",
                self.region,
                total_fallback_count,
                self.region,
            )

        return pd.DataFrame(all_records)

    # ── Backend: EIA API v2 ────────────────────────────────────────────────────

    @rate_limit(1.0)
    def _fetch_eia(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Pull hourly demand from EIA Open Data API v2.
        Region code for ERCOT in EIA: 'ERCO'
        Docs: https://www.eia.gov/opendata/
        """
        all_records: list[dict] = []
        offset = 0
        page_size = 5000

        while True:
            params = {
                "api_key": _EIA_API_KEY,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": "ERCO",
                "facets[type][]": "D",       # D = demand
                "start": start_dt.strftime("%Y-%m-%dT%H"),
                "end": end_dt.strftime("%Y-%m-%dT%H"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": page_size,
            }
            resp = requests.get(_EIA_BASE, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()

            rows = body.get("response", {}).get("data", [])
            if not rows:
                break

            for row in rows:
                period = row.get("period", "")   # "2024-06-01T00"
                value = row.get("value")
                if value is None:
                    continue
                try:
                    from datetime import timezone
                    ts = datetime.strptime(period, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                all_records.append(
                    {
                        "timestamp": ts,
                        "region": self.region,
                        "region_source": "EIA",
                        "demand_mw": float(value),
                    }
                )

            if len(rows) < page_size:
                break
            offset += page_size

        return pd.DataFrame(all_records)

    # ── Backend: Simulate (last resort / demo) ─────────────────────────────────

    def _fetch_simulate(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Generate realistic ERCOT COAST demand data.
        Based on actual ERCOT summer 2023 profiles:
          - Weekday summer peak: ~72 GW system-wide; COAST zone ~20-25 GW
          - Demand follows temperature closely with a 1-2h lag
          - Evening peak around 17:00-19:00 local time
        """
        import numpy as np

        rng = np.random.default_rng(seed=123)
        hours = pd.date_range(start=start_dt, end=end_dt, freq="h", tz=None)
        records = []

        for ts in hours:
            hour = ts.hour
            dow = ts.dayofweek  # 0=Mon … 6=Sun
            month = ts.month

            # Hour-of-day shape: trough at 04:00, peak at 17:00
            hour_factor = 0.75 + 0.25 * max(
                0, 1 - abs(hour - 17) / 10.0
            )  # simple triangle peaking at 17:00

            # Seasonal factor: higher June–Aug
            seasonal = 1.0 + 0.2 * max(0, 1 - abs(month - 7) / 3.0)

            # Weekend reduction
            weekend = _SIM_WEEKEND_FACTOR if dow >= 5 else 1.0

            # Shared cooling-load signal aligned with TemperatureClient's
            # synthetic daily and seasonal temperature pattern.
            daily_heat = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))
            seasonal_heat = max(0.0, 1.0 + math.sin(math.pi * (datetime(2024, month, 15).timetuple().tm_yday - 90) / 180.0))
            heat_load = 2_500.0 * daily_heat * seasonal_heat

            # Random noise
            noise = rng.normal(0, 400)

            demand = (
                (_SIM_BASE_DEMAND_MW + _SIM_PEAK_ADDITION * hour_factor) * seasonal * weekend
                + heat_load * weekend
                + noise
            )
            records.append(
                {
                    "timestamp": ts.replace(tzinfo=None),
                    "region": self.region,
                    "region_source": "Simulated",
                    "demand_mw": round(max(demand, 8_000), 1),
                }
            )

        return pd.DataFrame(records)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_date(value: str | datetime, end_of_day: bool = False) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    dt = datetime.fromisoformat(str(value))
    if end_of_day and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=None)


def _parse_ercot_timestamp(opr_dt: str, hour_ending: str) -> datetime | None:
    """
    ERCOT uses OprDt='2024-06-15' and HourEnding='14:00' (1-indexed, so 14:00 = hour 14).
    Converts to a UTC-naive datetime at the start of that hour.
    Strictly validates that HourEnding is between 1 and 24.
    """
    try:
        from datetime import timezone
        date_part = datetime.strptime(opr_dt.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        hour_str = hour_ending.strip().split(":")[0]
        hour = int(hour_str)
        if not 1 <= hour <= 24:
            raise ValueError(f"Invalid HourEnding: {hour}. Must be between 1 and 24.")
        
        # If hour is 24, it means midnight of the NEXT day
        if hour == 24:
            return date_part.replace(hour=0) + timedelta(days=1)
            
        return date_part.replace(hour=hour - 1)
    except (ValueError, IndexError):
        return None


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "region"])
    df = df.dropna(subset=["demand_mw"])
    df = df[df["demand_mw"] > 0]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
