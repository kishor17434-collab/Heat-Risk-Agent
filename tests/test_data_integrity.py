from datetime import datetime

import pandas as pd

from src.analysis.correlation import run_analysis
from src.data import pipeline


def _frame(start, values, column):
    timestamps = pd.date_range(start, periods=len(values), freq="h")
    rows = [{"timestamp": ts, column: value} for ts, value in zip(timestamps, values)]
    return pd.DataFrame(rows)


def test_pipeline_warns_on_mismatched_sources(monkeypatch, caplog, capsys):
    temp = _frame("2024-08-01", range(24), "temp_f")
    temp["location"] = "Houston_TX"
    temp["temp_c"] = temp["temp_f"]
    demand = _frame("2024-08-01", range(24000, 24024), "demand_mw")
    demand["region"] = "COAST"

    class TempClient:
        def __init__(self, mode=None):
            pass

        def fetch(self, start, end):
            return {"data": temp.copy(), "source": "simulate", "warning": ""}

    class DemandClient:
        def __init__(self, mode=None):
            pass

        def fetch(self, start, end):
            return {"data": demand.copy(), "source": "EIA", "warning": ""}

    monkeypatch.setattr(pipeline, "TemperatureClient", TempClient)
    monkeypatch.setattr(pipeline, "ERCOTClient", DemandClient)
    with caplog.at_level("WARNING"):
        result = pipeline.run_pipeline("2024-08-01", "2024-08-01", save=False)

    assert result.attrs["pipeline_meta"]["source_mismatch"] is True
    assert "DATA SOURCE MISMATCH" in caplog.text
    assert "DATA SOURCE MISMATCH" in capsys.readouterr().out


def test_pipeline_does_not_warn_for_matched_sources(monkeypatch, caplog):
    temp = _frame("2024-08-01", range(24), "temp_f")
    temp["location"] = "Houston_TX"
    temp["temp_c"] = temp["temp_f"]
    demand = _frame("2024-08-01", range(24000, 24024), "demand_mw")
    demand["region"] = "COAST"

    class TempClient:
        def __init__(self, mode=None): pass
        def fetch(self, start, end): return {"data": temp.copy(), "source": "simulate", "warning": ""}

    class DemandClient:
        def __init__(self, mode=None): pass
        def fetch(self, start, end): return {"data": demand.copy(), "source": "Simulated", "warning": ""}

    monkeypatch.setattr(pipeline, "TemperatureClient", TempClient)
    monkeypatch.setattr(pipeline, "ERCOTClient", DemandClient)
    with caplog.at_level("WARNING"):
        result = pipeline.run_pipeline("2024-08-01", "2024-08-01", save=False)

    assert result.attrs["pipeline_meta"]["source_mismatch"] is False
    assert "DATA SOURCE MISMATCH" not in caplog.text


def test_correlation_sign_warning_for_negative_data(tmp_path, caplog, capsys):
    path = tmp_path / "combined.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-08-01", periods=24, freq="h"),
        "temp_f": range(24),
        "demand_mw": range(24000, 23976, -1),
    }).to_csv(path, index=False)

    with caplog.at_level("WARNING"):
        report = run_analysis(path, save=False)

    assert report["sign_warning"] is True
    assert "UNEXPECTED CORRELATION SIGN" in caplog.text
    assert "UNEXPECTED CORRELATION SIGN" in capsys.readouterr().out


def test_correlation_sign_warning_is_false_for_positive_data(tmp_path, caplog):
    path = tmp_path / "combined.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-08-01", periods=24, freq="h"),
        "temp_f": range(24),
        "demand_mw": range(24000, 24024),
    }).to_csv(path, index=False)

    with caplog.at_level("WARNING"):
        report = run_analysis(path, save=False)

    assert report["sign_warning"] is False