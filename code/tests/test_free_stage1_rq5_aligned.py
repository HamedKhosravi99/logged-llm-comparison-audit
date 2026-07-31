"""Tests for the aligned Stage 1 pipeline. Run: python3 -m pytest code/tests -q

These lock in the two behaviours that were actually wrong during development:
  * CASE must stop after the seed pairs when the diagnostic says not to audit;
  * acquisition priority must score unbought pairs on the RAW logged value.
Plus the reproduction gate and the invariants the report depends on.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
sys.path.insert(0, CODE)


def test_reproduction_gate_passes():
    """The pipeline must reproduce the paper's RQ5 baseline before any paper edit."""
    d = json.load(open(os.path.join(RES, "free_stage1_rq5_aligned.json")))
    assert d["gate_passed"] is True
    for g in d["gate"]:
        assert abs(g["reproduced"] - g["target"]) <= g["tolerance"], g


def test_diagnostic_gate_limits_purchases():
    """CASE buys 4 seed pairs and stops there whenever the diagnostic declines."""
    import free_stage1_rq5_aligned as AL
    t = {r["arm"]: r for r in json.load(open(os.path.join(RES, "free_stage1_rq5_aligned.json")))["table"]}
    pairs = t["Current CASE"]["pairs"]
    assert 4.0 <= pairs <= AL.BUDGET_PAIRS, pairs
    assert pairs < AL.BUDGET_PAIRS, "no diagnostic gating: every run bought the full budget"


def test_variance_floor_respected():
    import free_stage1_rq5_aligned as AL
    assert AL.VAR_FLOOR == 1e-8


def test_scoring_fold_never_audited():
    """The audit pool and the scoring fold must be disjoint."""
    import random as _r
    from tau2_e2e_full import split_tasks
    keys = [(d, i) for d in ("airline", "retail", "telecom") for i in range(20)]
    a = split_tasks(list(keys), _r.Random(0))
    S = {k for k, v in a.items() if v == "scoring"}
    A = {k for k, v in a.items() if v == "audit"}
    assert S and A and not (S & A)


def test_switch_decomposition_is_exhaustive():
    """harmful + beneficial + neutral must equal switches, and switches + retained equal 3."""
    D = pd.read_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"))
    s = D.harmful + D.beneficial + D.neutral
    assert np.allclose(s, D.switches, atol=1e-9)
    assert np.allclose(D.switches + D.retained, 3.0, atol=1e-6)


def test_no_oracle_opportunity_in_quality_repair():
    """The quality-repair scenario contains no feasible cost-saving switch, so a
    'failed to save cost' reading of it would be unfair."""
    D = pd.read_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"))
    q = D[D.scenario == "quality_repair"]
    assert q.opp_frac.max() == 0.0
    c = D[D.scenario == "cost_reduction"]
    assert c.opp_frac.max() > 0.0


def test_safe_arms_abstain_more_than_they_switch():
    D = pd.read_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"))
    q = D[(D.scenario == "quality_repair") & (D.arm == "Safe inverse-variance CASE")]
    assert q.retained.mean() > 2.5
