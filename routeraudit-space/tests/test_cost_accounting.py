import numpy as np, pandas as pd
from routeraudit import AuditConfig, analyze_trace

def test_total_audit_cost_is_calls_plus_verification(log, prices, audit, verified, traffic):
    cfg = AuditConfig(verification_cost_usd=0.25)
    r = analyze_trace(log, prices, cfg, randomized_audit=audit,
                      verified_outcomes=verified, traffic=traffic)
    d = r.diagnostics
    assert np.isclose(d["model_call_audit_cost_usd"], audit["audit_cost_usd"].sum())
    assert np.isclose(d["verification_cost_usd"], verified["verification_cost_usd"].sum())
    assert np.isclose(d["total_audit_cost_usd"],
                      d["model_call_audit_cost_usd"] + d["verification_cost_usd"])

def test_verification_cost_enters_audit_inclusive_cost(log, prices, audit, traffic):
    free = analyze_trace(log, prices, AuditConfig(verification_cost_usd=0.0),
                         randomized_audit=audit, traffic=traffic)
    paid = analyze_trace(log, prices, AuditConfig(verification_cost_usd=1.0),
                         randomized_audit=audit, traffic=traffic)
    assert paid.diagnostics["total_audit_cost_usd"] > free.diagnostics["total_audit_cost_usd"]
    assert paid.diagnostics["audit_inclusive_cost_per_1k"] > \
           free.diagnostics["audit_inclusive_cost_per_1k"]
    # deployment cost itself must not move
    assert np.isclose(paid.diagnostics["recommended_cost_per_1k"],
                      free.diagnostics["recommended_cost_per_1k"])

def test_missing_price_is_never_treated_as_zero(log, prices, traffic):
    p = prices[~((prices.model == "compact-small") & (prices.request_type == "billing"))]
    r = analyze_trace(log, p, AuditConfig(), traffic=traffic)
    assert any("never treated as zero" in w for w in r.warnings)

def test_audit_queue_respects_the_budget(log, prices, traffic):
    r = analyze_trace(log, prices, AuditConfig(audit_budget_usd=0.10), traffic=traffic)
    assert r.audit_queue["audit_price_usd"].sum() <= 0.10 + 1e-9

def test_zero_budget_buys_nothing(log, prices, traffic):
    r = analyze_trace(log, prices, AuditConfig(audit_budget_usd=0.0), traffic=traffic)
    assert len(r.audit_queue) == 0
