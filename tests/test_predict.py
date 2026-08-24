from src.model.predict import _demand_to_risk

def test_demand_to_risk_with_thresholds():
    thresholds = {
        "p50": 10000.0,
        "p75": 12000.0,
        "p90": 14000.0,
        "p95": 15000.0
    }
    
    # Below p50
    risk = _demand_to_risk(5000.0, thresholds)
    assert 0 <= risk <= 30.0
    
    # Above p95
    risk = _demand_to_risk(16000.0, thresholds)
    assert risk >= 80.0

def test_demand_to_risk_missing_thresholds():
    thresholds = {} # Missing
    
    # Should fallback to defaults
    risk = _demand_to_risk(12000.0, thresholds)
    assert risk > 0
    
def test_demand_to_risk_zero_thresholds():
    thresholds = {
        "p50": 0.0,
        "p75": 0.0,
        "p90": 0.0,
        "p95": 0.0
    }
    # Should not raise ZeroDivisionError and should safely return defaults or scaled max
    risk = _demand_to_risk(100.0, thresholds)
    # With our defaults overriding 0.0 because of `if not p50`, it uses 10000.0. 
    # Or if p50 was 0.0 and we didn't override, it would hit 100.0 or 0.0.
    assert risk >= 0
