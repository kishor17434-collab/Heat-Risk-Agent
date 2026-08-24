import pytest
from datetime import datetime, timezone
import pandas as pd
import requests
from src.data.ercot_client import ERCOTClient, _parse_ercot_timestamp

def test_ercot_client_simulate(sample_timestamps):
    start, end = sample_timestamps
    client = ERCOTClient()
    
    # We use mode=simulate since it will fallback to _fetch_simulate by default
    # Or in our implementation, ERCOTClient doesn't have mode, it fetches EIA, but we can test if it returns dict
    result = client.fetch(start=start, end=end)
    
    assert isinstance(result, dict)
    assert "data" in result
    assert "source" in result
    assert isinstance(result["data"], pd.DataFrame)
    
    df = result["data"]
    assert not df.empty
    assert "demand_mw" in df.columns
    assert "timestamp" in df.columns

def test_ercot_parse_timestamp():
    # We fixed _parse_ercot_timestamp, let's test it directly
    dt = _parse_ercot_timestamp("2023-08-01", "1") # Hour 1
    assert dt.hour == 0
    assert dt.day == 1
    assert dt.month == 8
    assert dt.year == 2023
    
    dt2 = _parse_ercot_timestamp("2023-08-01", "24") # Hour 24
    assert dt2.hour == 0
    assert dt2.day == 2
    
    assert _parse_ercot_timestamp("2023-08-01", "25") is None
    assert _parse_ercot_timestamp("2023-08-01", "0") is None


def test_ercot_total_fallback_is_labeled_and_warned(monkeypatch, caplog):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"OprDt": "2024-08-01", "HourEnding": "01", "TOTAL": 65000}]}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    with caplog.at_level("WARNING"):
        result = ERCOTClient(region="COAST")._fetch_ercot_dashboard(
            datetime(2024, 8, 1), datetime(2024, 8, 1)
        )

    assert result.iloc[0]["demand_mw"] == 65000
    assert result.iloc[0]["region_source"] == "TOTAL_fallback"
    assert "used system-wide TOTAL instead" in caplog.text


def test_demand_simulation_mode_skips_network(monkeypatch, sample_timestamps):
    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be called in simulate mode")

    monkeypatch.setattr(requests, "get", fail_network)
    monkeypatch.setattr(requests, "post", fail_network)
    result = ERCOTClient(mode="simulate").fetch(*sample_timestamps)

    assert result["source"] == "Simulated"
    assert not result["data"].empty
