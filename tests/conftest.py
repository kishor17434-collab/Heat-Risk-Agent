import os
import pytest
from datetime import datetime, timezone
import pandas as pd

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Ensure predictable environment variables for tests."""
    monkeypatch.setenv("FORTYGUARD_API_KEY", "test_key")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/fake-slack-webhook")
    monkeypatch.setenv("RISK_THRESHOLD", "75")
    monkeypatch.setenv("TEMP_DATA_MODE", "simulate")
    monkeypatch.setenv("ERCOT_REGION", "COAST")

@pytest.fixture
def sample_timestamps():
    start = datetime(2023, 8, 1, tzinfo=timezone.utc)
    end = datetime(2023, 8, 2, tzinfo=timezone.utc)
    return start, end
