from datetime import datetime, timezone
import pytest
from src.agent.loop import AgentLoop

# Mock load_model to avoid actually loading from disk during tests if needed,
# or if it exists it will just load it. Let's assume it loads successfully or we monkeypatch it.
# Wait, load_model is imported inside loop.py, so we can monkeypatch it.

def test_agent_loop_context_manager(monkeypatch):
    # Mock load_model to return a dummy model dictionary
    monkeypatch.setattr("src.agent.loop.load_model", lambda: {"meta": {"model_type": "dummy"}, "model": None, "thresholds": {}})
    
    with AgentLoop(threshold=75, poll_interval=1) as loop:
        assert loop.threshold == 75
        assert loop.poll_interval == 1
        assert loop.model["meta"]["model_type"] == "dummy"
    # Reaching here means context manager works and exited cleanly

def test_make_decision_pure(monkeypatch):
    # Mock predict_risk to return deterministic values
    def mock_predict_risk(*args, **kwargs):
        return {
            "predicted_demand_mw": 15000.0,
            "risk_score": 80.0,
            "risk_level": "High"
        }
    monkeypatch.setattr("src.agent.loop.predict_risk", mock_predict_risk)
    monkeypatch.setattr("src.agent.loop.load_model", lambda: {"meta": {}, "model": None, "thresholds": {}})
    
    loop = AgentLoop(threshold=75)
    
    ts = datetime(2023, 8, 1, 12, 0, tzinfo=timezone.utc)
    decision = loop._make_decision(
        timestamp=ts,
        temp_f=100.0,
        hour=12,
        day_of_week=1,
        month=8,
        demand_actual_mw=14500.0
    )
    
    assert decision["alert_triggered"] is True
    assert decision["risk_score"] == 80.0
    assert decision["temp_f"] == 100.0
    assert decision["actual_demand_mw"] == 14500.0
    assert decision["predicted_demand_mw"] == 15000.0
