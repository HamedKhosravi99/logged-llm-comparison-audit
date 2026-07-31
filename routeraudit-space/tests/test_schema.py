import pandas as pd, pytest
from routeraudit import AuditConfig, SchemaError, analyze_trace

CFG = AuditConfig()

def test_missing_required_column_is_named(log, prices):
    bad = log.drop(columns=["request_cost_usd"])
    with pytest.raises(SchemaError, match="request_cost_usd"):
        analyze_trace(bad, prices, CFG)

def test_negative_cost_rejected(log, prices):
    bad = log.copy(); bad.loc[0, "request_cost_usd"] = -1.0
    with pytest.raises(SchemaError, match="negative"):
        analyze_trace(bad, prices, CFG)

def test_duplicate_request_model_rows_detected(log, prices):
    bad = pd.concat([log, log.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicated"):
        analyze_trace(bad, prices, CFG)

def test_non_numeric_score_rejected(log, prices):
    bad = log.copy(); bad.loc[0, "score"] = "high"
    with pytest.raises(SchemaError, match="not numeric"):
        analyze_trace(bad, prices, CFG)

def test_single_model_rejected(log, prices):
    bad = log[log.model == log.model.iloc[0]]
    with pytest.raises(SchemaError, match="fewer than two"):
        analyze_trace(bad, prices, CFG)

def test_invalid_traffic_shares_rejected(log, prices, traffic):
    bad = traffic.copy(); bad["traffic_share"] = bad["traffic_share"] * 2
    with pytest.raises(SchemaError, match="approximately 1"):
        analyze_trace(log, prices, CFG, traffic=bad)

def test_unknown_traffic_request_type_rejected(log, prices):
    bad = pd.DataFrame(dict(request_type=["nonexistent"], traffic_share=[1.0]))
    with pytest.raises(SchemaError, match="not present"):
        analyze_trace(log, prices, CFG, traffic=bad)

def test_prices_without_request_type_are_expanded(log, prices):
    short = prices.drop(columns=["request_type"]).drop_duplicates("model")
    r = analyze_trace(log, short, CFG)
    assert any("expanded to every request type" in w for w in r.warnings)

def test_empty_table_rejected(prices):
    with pytest.raises(SchemaError):
        analyze_trace(pd.DataFrame(columns=[
            "request_id", "request_type", "model", "score", "request_cost_usd"]), prices, CFG)

@pytest.mark.parametrize("field,value", [
    ("quality_tolerance", -0.1), ("future_requests", 0), ("audit_budget_usd", -1),
    ("verification_cost_usd", -1), ("confidence_level", 1.0), ("minimum_support", 0),
    ("audit_readout_size", 0)])
def test_config_validates(field, value):
    with pytest.raises(ValueError):
        AuditConfig(**{field: value})
