"""Joint four-category switch classification, per decision. NO new API calls.

A switch is classified against the TRUE best in its domain (not against the incumbent), which is
what reconciles the categories with the tolerance-satisfaction metric:

  A valid cost-saving          within delta of the true best  AND net saving at H > 0
  B quality-safe, uneconomical within delta of the true best  AND net saving <= 0
  C economical, quality-violating  violates delta             AND net saving > 0
  D invalid on both                violates delta             AND net saving <= 0

Emits one row per (scenario, fold, split seed, acquisition seed, arm, domain).
"""
import collections, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
from tau2_e2e_full import (load, split_tasks, fit_router, cheapest_good_enough,
                           DELTA, OUTER_FOLDS, SPLIT_SEEDS, ACQ_SEEDS, EXPLORE)
from free_stage1_rq5_aligned import (build_log, run_case, greedy_policy, safe_policy,
                                     H_PRIMARY, TAU_PRIMARY)
from free_stage1_diagnostics import incumbent_log

ARM_NAME = {"safe_iv": "Confidence-gated IV CASE", "safe": "Confidence-gated CASE",
            "iv": "Inverse-variance CASE", "cur": "Current CASE"}


def classify_switch(d, base, cand, TRUE, COSTS, doms, audit_usd, H=H_PRIMARY, delta=DELTA):
    """Return category and the two underlying quantities."""
    best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
    q_ok = TRUE[(d, cand)] >= best - delta
    w = 1.0 / len(doms)
    net = H * w * (COSTS[(d, base)] - COSTS[(d, cand)]) - audit_usd * w
    econ = net > 0
    cat = "A" if (q_ok and econ) else "B" if (q_ok and not econ) else "C" if econ else "D"
    return cat, float(net), bool(q_ok), float(TRUE[(d, cand)] - TRUE[(d, base)])


def main():
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                      cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth.itertuples()}
    TRUE = {(r.domain, r.agent_model): r.success for r in truth.itertuples()}
    ideal = {d: cheapest_good_enough(truth[truth.domain == d], DELTA) for d in doms}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})
    import random as _r
    rows = []

    for scenario in ("quality_repair", "cost_reduction"):
        for fold in range(OUTER_FOLDS):
            for sd in range(SPLIT_SEEDS):
                rng = np.random.default_rng(1000 * fold + sd)
                asg = split_tasks(list(keys), _r.Random(1000 * fold + sd))
                S = [k for k, v in asg.items() if v == "scoring"]
                R = set(k for k, v in asg.items() if v == "router")
                L = [k for k, v in asg.items() if v == "log"]
                A = [k for k, v in asg.items() if v == "audit"]
                if not (S and R and L and A): continue

                if scenario == "quality_repair":
                    router = fit_router(df, R, models, doms)
                    M, Mvar = build_log(df, tc, L, router, models, doms, COSTS, rng)
                    base_pol = greedy_policy({(d, m): M.get((d, m), np.nan)
                                              for d in doms for m in models}, models, doms, COSTS)
                    extra = {}
                else:
                    TRUEtr = {}
                    for d in doms:
                        for m in models:
                            v = tc[(tc.domain == d) & (tc.agent_model == m)
                                   & (tc.task_id.isin([t for (dd, t) in R if dd == d]))]
                            TRUEtr[(d, m)] = float(v.success.mean()) if len(v) else 0.0
                    M, Mvar, inc = incumbent_log(df, tc, L, R, models, doms, COSTS, TRUEtr, rng)
                    base_pol = {d: inc for d in doms}
                    extra = {"Raw-log downrouting": greedy_policy(
                        {(d, m): M.get((d, m), np.nan) for d in doms for m in models},
                        models, doms, COSTS)}

                # oracle opportunity per domain, against the deployed baseline
                opp = {}
                for d in doms:
                    b = base_pol[d]
                    best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
                    opp[d] = any(TRUE[(d, x)] >= best - DELTA and COSTS[(d, x)] < COSTS[(d, b)]
                                 for (dd, x) in TRUE if dd == d)

                def emit(arm, pol, acq, audit_usd):
                    for d in doms:
                        b, c = base_pol[d], pol.get(d)
                        if c is None: continue
                        sw = (c != b)
                        cat, net, q_ok, dq = (classify_switch(d, b, c, TRUE, COSTS, doms, audit_usd)
                                              if sw else (None, 0.0, None, 0.0))
                        best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
                        rows.append(dict(scenario=scenario, fold=fold, seed=sd, acq=acq, arm=arm,
                                         domain=d, switched=int(sw), category=cat,
                                         opportunity=int(opp[d]), net_saving=net,
                                         tol_ok=int(TRUE[(d, c)] >= best - DELTA),
                                         dq=dq, chosen=c, base=b,
                                         wrong=int(c != ideal[d]), audit=audit_usd))

                for nm, pol in extra.items():
                    emit(nm, pol, -1, 0.0)
                for acq in range(ACQ_SEEDS):
                    for iv in (False, True):
                        q, qv, spend, _ = run_case(M, Mvar, tc, A, models, doms, COSTS,
                                                   np.random.default_rng(7000 + acq), iv)
                        emit(ARM_NAME["iv" if iv else "cur"], greedy_policy(q, models, doms, COSTS),
                             acq, spend)
                        pol, _acts = safe_policy(q, qv, models, doms, COSTS, base_pol, spend,
                                                 TAU_PRIMARY, H_PRIMARY,
                                                 np.random.default_rng(9000 + acq))
                        emit(ARM_NAME["safe_iv" if iv else "safe"], pol, acq, spend)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(RES, "free_stage1_switch_categories.parquet"), index=False)
    print(f"  wrote free_stage1_switch_categories.parquet  ({len(out):,} decision rows)")
    return out


if __name__ == "__main__":
    main()
