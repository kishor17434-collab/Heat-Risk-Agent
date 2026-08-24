import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.agent.alerts import AlertManager

def test_alert_manager_persistence(tmp_path, monkeypatch):
    # Mock _LOGS_DIR to use temp directory
    monkeypatch.setattr("src.agent.alerts._LOGS_DIR", tmp_path)
    
    # Init first instance
    manager1 = AlertManager(backends=["log"], cooldown_minutes=10)
    assert manager1._last_alert_time is None
    
    # Send alert, which should save state
    dt1 = datetime(2023, 8, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("src.agent.alerts.datetime", type("MockDatetime", (), {"now": lambda tz=None: dt1, "fromisoformat": datetime.fromisoformat}))
    
    sent = manager1.send_alert(100.0, 15000.0, 85.0, 75.0)
    assert sent is True
    assert manager1._last_alert_time == dt1
    
    # Create new instance, it should load the state
    manager2 = AlertManager(backends=["log"], cooldown_minutes=10)
    assert manager2._last_alert_time == dt1
    
    # Try sending another alert before cooldown
    dt2 = dt1 + timedelta(minutes=5)
    monkeypatch.setattr("src.agent.alerts.datetime", type("MockDatetime", (), {"now": lambda tz=None: dt2, "fromisoformat": datetime.fromisoformat}))
    
    sent2 = manager2.send_alert(101.0, 15100.0, 86.0, 75.0)
    assert sent2 is False # In cooldown

    # Try sending after cooldown
    dt3 = dt1 + timedelta(minutes=15)
    monkeypatch.setattr("src.agent.alerts.datetime", type("MockDatetime", (), {"now": lambda tz=None: dt3, "fromisoformat": datetime.fromisoformat}))
    
    sent3 = manager2.send_alert(102.0, 15200.0, 87.0, 75.0)
    assert sent3 is True
    assert manager2._last_alert_time == dt3
