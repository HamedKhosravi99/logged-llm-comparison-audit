import numpy as np, pandas as pd, pytest
from routeraudit import AuditConfig, analyze_trace

CFG = AuditConfig()

def test_plan_mode_makes_no_validity_claim(log, prices, traffic):
    r = analyze_trace(log, prices, CFG, traffic=traffic)
    assert r.mode == "plan"
    assert len(r.candidate_switches) == 0
    assert any("No randomized audit supplied" in w for w in r.warnings)
    assert any("Measurement validity is not established" in w for w in r.warnings)
    assert not r.manifest["measurement_validity_established"]

def test_analyze_mode_classifies_against_verified(log, prices, audit, verified, traffic):
    r = analyze_trace(log, prices, CFG, randomized_audit=audit,
                      verified_outcomes=verified, traffic=traffic)
    assert r.mode == "analyze"
    assert r.manifest["measurement_validity_established"]
    st = set(r.candidate_switches["status"])
    assert "Candidate switch under provided score" not in st

def test_without_verified_no_switch_is_called_valid(log, prices, audit, traffic):
    r = analyze_trace(log, prices, CFG, randomized_audit=audit, traffic=traffic)
    assert set(r.candidate_switches["status"]) <= {"Candidate switch under provided score"}
    assert all(v == "Not established (no verified outcomes)"
               for v in r.candidate_switches["incumbent_relative"])
    assert any("Measurement validity is not established" in w for w in r.warnings)

def test_no_opportunity_retains_incumbent(log, prices, audit, traffic):
    """With the incumbent priced below every alternative there is nothing to switch to."""
    p = prices.copy()
    p.loc[p.model == "premium-large", "expected_cost_usd"] = 0.0001
    r = analyze_trace(log, p, CFG, randomized_audit=audit, traffic=traffic)
    assert len(r.candidate_switches) == 0
    assert any("No cheaper model" in w for w in r.warnings)

def test_traffic_assumption_is_warned_when_derived(log, prices):
    r = analyze_trace(log, prices, CFG)
    assert any("derived from historical routed-log frequencies" in w for w in r.warnings)

def test_thin_support_is_warned_and_excluded(log, prices, traffic):
    r = analyze_trace(log, prices, AuditConfig(minimum_support=10_000), traffic=traffic)
    assert any("fewer than the minimum supported observations" in w for w in r.warnings)

def test_manifest_records_every_setting(log, prices, traffic):
    cfg = AuditConfig(quality_tolerance=0.05, future_requests=250_000, audit_budget_usd=3.5,
                      verification_cost_usd=0.25, confidence_level=0.9,
                      minimum_support=12, audit_readout_size=17)
    m = analyze_trace(log, prices, cfg, traffic=traffic).manifest
    for k, v in vars(cfg).items():
        assert m[k] == v
    assert m["routeraudit_version"] and m["timestamp_utc"]

def test_no_network_imports_in_package():
    """No module in the package may import a networking library (checked on the AST)."""
    import ast
    from pathlib import Path
    banned = {"requests", "urllib", "urllib3", "httpx", "socket", "http", "aiohttp", "ftplib",
              "telnetlib", "smtplib", "xmlrpc"}
    root = Path(__file__).resolve().parents[1] / "routeraudit"
    for f in root.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            hit = banned.intersection(names)
            assert not hit, f"{f.name} imports a networking module: {sorted(hit)}"


def test_report_html_is_self_contained(log, prices, audit, traffic):
    html = analyze_trace(log, prices, CFG, randomized_audit=audit, traffic=traffic).report_html
    assert "<html" in html
    assert "http://" not in html and "https://" not in html
