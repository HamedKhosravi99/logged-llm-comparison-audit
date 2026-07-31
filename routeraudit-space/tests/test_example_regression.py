"""The bundled example must keep exercising all three switch outcomes.

These are properties of the synthetic example, not results from the paper.
"""
import numpy as np
from routeraudit import AuditConfig, analyze_trace

CFG = AuditConfig(verification_cost_usd=0.25)

def test_example_produces_one_valid_and_one_violating_switch(log, prices, audit, verified, traffic):
    r = analyze_trace(log, prices, CFG, randomized_audit=audit,
                      verified_outcomes=verified, traffic=traffic)
    st = dict(zip(r.candidate_switches["request_type"], r.candidate_switches["status"]))
    assert st.get("billing") == "Valid cost-saving switch"
    assert st.get("account") == "Quality-violating economical switch"
    assert "technical" not in st, "technical has no good-enough cheaper model and must be retained"

def test_example_is_deterministic(log, prices, audit, verified, traffic):
    a = analyze_trace(log, prices, CFG, randomized_audit=audit,
                      verified_outcomes=verified, traffic=traffic)
    b = analyze_trace(log, prices, CFG, randomized_audit=audit,
                      verified_outcomes=verified, traffic=traffic)
    assert a.summary.equals(b.summary)
    assert a.candidate_switches.equals(b.candidate_switches)
    assert np.isclose(a.diagnostics["economics"]["saving_per_request"],
                      b.diagnostics["economics"]["saving_per_request"])

def test_pooling_is_reported_and_used(log, prices, audit, traffic):
    r = analyze_trace(log, prices, CFG, randomized_audit=audit, traffic=traffic)
    assert r.diagnostics["pooling"] == "inverse-variance"

def test_no_paper_number_is_hard_coded():
    """Guard against transcribing paper results into shipped code."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    banned = ["0.0499", "0.0338", "18.3", "6.7%", "0.780", "0.155", "559"]
    for f in list((root / "routeraudit").rglob("*.py")) + [root / "app.py"]:
        src = f.read_text()
        for b in banned:
            assert b not in src, f"{f.name} contains what looks like a paper result: {b}"
