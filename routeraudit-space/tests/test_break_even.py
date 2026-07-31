import numpy as np, pandas as pd
from routeraudit import AuditConfig, analyze_trace
from routeraudit.core.estimate import break_even

def test_no_saving_gives_no_break_even():
    r = break_even(np.array([0.0, -0.01]), np.array([0.5, 0.5]), 10.0, 0.95)
    assert r["point_break_even"] is None
    assert r["lower_bound_break_even"] is None

def test_positive_saving_gives_finite_point_break_even():
    r = break_even(np.array([0.01, 0.01]), np.array([0.5, 0.5]), 10.0, 0.95)
    assert r["point_break_even"] == 1000
    assert r["lower_bound_break_even"] is not None

def test_lower_bound_break_even_is_never_below_the_point():
    r = break_even(np.array([0.02, 0.001]), np.array([0.5, 0.5]), 5.0, 0.95)
    if r["lower_bound_break_even"] is not None:
        assert r["lower_bound_break_even"] >= r["point_break_even"]

def test_break_even_none_is_reported_as_text(log, prices, audit, traffic):
    p = prices.copy()
    p.loc[p.model == "premium-large", "expected_cost_usd"] = 0.0001
    r = analyze_trace(log, p, AuditConfig(), randomized_audit=audit, traffic=traffic)
    vals = dict(zip(r.summary["Item"], r.summary["Value"]))
    assert vals["Point break-even"] == "No finite supported break-even"
    assert any("No finite lower-bound economic break-even" in w for w in r.warnings)

def test_few_units_are_flagged(log, prices, audit, traffic):
    r = analyze_trace(log, prices, AuditConfig(), randomized_audit=audit, traffic=traffic)
    assert any("resamples only" in w for w in r.warnings)
