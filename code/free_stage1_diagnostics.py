"""Stage 1 diagnostics (items 1-7). NO new API calls.

Two operational replays from the same 440 tau2 simulations:
  quality-repair scenario : the paper's fitted-router log (RQ5 pipeline, gate-verified)
  cost-reduction scenario : a quality-first incumbent, 0.85 incumbent / 0.15 uniform exploration

For every arm we record DOMAIN-LEVEL decisions so switch quality, quality-constraint validity and
oracle opportunity can be measured, not inferred.
"""
import json, os, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

from tau2_e2e_full import (load, split_tasks, fit_router, cheapest_good_enough, diagnostic,
                           DELTA, N_AUDIT, BUDGET_PAIRS, OUTER_FOLDS, SPLIT_SEEDS, ACQ_SEEDS,
                           ROUTER_TOL, EXPLORE)
from free_stage1_rq5_aligned import (build_log, audit_pair, run_case, greedy_policy, safe_policy,
                                     score, H_PRIMARY, TAU_PRIMARY, VAR_FLOOR)


def incumbent_log(df, tc, L, R, models, doms, COSTS, TRUEtr, rng):
    """Cost-reduction scenario: a quality-first incumbent serves 85% of traffic."""
    inc = max(models, key=lambda m: np.mean([TRUEtr.get((d, m), 0.0) for d in doms]))
    seen = collections.defaultdict(list)
    for d, t in L:
        m = inc if rng.random() >= EXPLORE else models[rng.integers(len(models))]
        r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
        if len(r): seen[(d, m)].append(float(r.success.iloc[0]))
    M = {k: float(np.mean(v)) for k, v in seen.items()}
    Mvar = {k: max(float(np.var(v, ddof=1) / len(v)) if len(v) > 1 else 0.25 / max(len(v), 1), VAR_FLOOR)
            for k, v in seen.items()}
    return M, Mvar, inc


def classify(pol, base_pol, TRUE, COSTS, doms, delta=DELTA):
    """Per-domain switch outcome against the deployed baseline policy."""
    out = []
    for d in doms:
        a, b = base_pol.get(d), pol.get(d)
        if a is None or b is None:
            continue
        if a == b:
            out.append(("retain", d)); continue
        dq = TRUE[(d, b)] - TRUE[(d, a)]
        dc = COSTS[(d, b)] - COSTS[(d, a)]
        if dq < -delta:
            out.append(("harmful", d))
        elif dq >= -delta and dc < 0:
            out.append(("beneficial", d))
        else:
            out.append(("neutral", d))
    return out


def quality_metrics(pol, TRUE, doms, delta=DELTA):
    """Does the deployed choice meet the tolerance against the TRUE best in that domain?"""
    sat, viol = [], []
    for d in doms:
        m = pol.get(d)
        if m is None: continue
        best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
        gap = (best - delta) - TRUE[(d, m)]
        sat.append(gap <= 0); viol.append(max(0.0, gap))
    return (float(np.mean(sat)) if sat else np.nan, bool(all(sat)) if sat else False,
            float(np.mean(viol)) if viol else np.nan, float(np.max(viol)) if viol else np.nan)


def oracle_opportunity(base_pol, TRUE, COSTS, doms, delta=DELTA):
    """Is a genuine quality-preserving cheaper alternative available at all?"""
    have, savings = [], []
    for d in doms:
        inc = base_pol.get(d)
        if inc is None: continue
        best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
        cands = [x for (dd, x) in TRUE if dd == d
                 and TRUE[(d, x)] >= best - delta and COSTS[(d, x)] < COSTS[(d, inc)]]
        have.append(bool(cands))
        savings.append(max([COSTS[(d, inc)] - COSTS[(d, x)] for x in cands], default=0.0))
    return (float(np.mean(have)) if have else np.nan, bool(any(have)),
            float(np.sum(savings)) / max(len(doms), 1))


def run_scenario(scenario):
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
                arms = [("Current CASE", False, False), ("Inverse-variance CASE", True, False),
                        ("Safe CASE", False, True), ("Safe inverse-variance CASE", True, True)]
            else:
                TRUEtr = {}
                for d in doms:
                    for m in models:
                        v = tc[(tc.domain == d) & (tc.agent_model == m)
                               & (tc.task_id.isin([t for (dd, t) in R if dd == d]))]
                        TRUEtr[(d, m)] = float(v.success.mean()) if len(v) else 0.0
                M, Mvar, inc = incumbent_log(df, tc, L, R, models, doms, COSTS, TRUEtr, rng)
                base_pol = {d: inc for d in doms}
                arms = [("Current CASE", False, False), ("Inverse-variance CASE", True, False),
                        ("Safe inverse-variance CASE", True, True)]

            log_pol = greedy_policy({(d, m): M.get((d, m), np.nan) for d in doms for m in models},
                                    models, doms, COSTS)
            opp_frac, opp_any, opp_sav = oracle_opportunity(base_pol, TRUE, COSTS, doms)

            def emit(arm, pol, audit, pairs, retained):
                s, c, _ = score(pol, tc, S)
                cls = collections.Counter(x for x, _ in classify(pol, base_pol, TRUE, COSTS, doms))
                q_sat, q_all, q_mean, q_max = quality_metrics(pol, TRUE, doms)
                rows.append(dict(scenario=scenario, fold=fold, seed=sd, arm=arm,
                                 success=s, cost=c, audit=audit, pairs=pairs,
                                 switches=cls["harmful"] + cls["beneficial"] + cls["neutral"],
                                 harmful=cls["harmful"], beneficial=cls["beneficial"],
                                 neutral=cls["neutral"], retained=retained,
                                 q_sat_frac=q_sat, q_all_ok=int(q_all), q_viol_mean=q_mean,
                                 q_viol_max=q_max, opp_frac=opp_frac, opp_any=int(opp_any),
                                 opp_savings_per_req=opp_sav,
                                 wrong=float(np.mean([pol.get(d) != ideal[d] for d in doms]))))

            base_name = "Raw routed log" if scenario == "quality_repair" else "Raw incumbent"
            emit(base_name, base_pol, 0.0, 0.0, len(doms))
            emit("No audit / retain", base_pol, 0.0, 0.0, len(doms))
            if scenario == "cost_reduction":
                emit("Raw-log downrouting", log_pol, 0.0, 0.0,
                     sum(1 for d in doms if log_pol.get(d) == base_pol.get(d)))
            emit("Ideal complete matrix", ideal, 0.0, 0.0,
                 sum(1 for d in doms if ideal.get(d) == base_pol.get(d)))

            acc = collections.defaultdict(list)
            for a in range(ACQ_SEEDS):
                r2 = np.random.default_rng(7000 + a)
                cache = {}
                for arm, iv, safe in arms:
                    if iv not in cache:
                        cache[iv] = run_case(M, Mvar, tc, A, models, doms, COSTS,
                                             np.random.default_rng(7000 + a), iv)
                    q, qv, spend, npairs = cache[iv]
                    if safe:
                        pol, acts = safe_policy(q, qv, models, doms, COSTS, base_pol, spend,
                                                TAU_PRIMARY, H_PRIMARY, r2)
                        ret = sum(1 for v in acts.values() if v == "RETAIN")
                    else:
                        pol = greedy_policy(q, models, doms, COSTS)
                        ret = sum(1 for d in doms if pol.get(d) == base_pol.get(d))
                    s, c, _ = score(pol, tc, S)
                    cls = collections.Counter(x for x, _ in classify(pol, base_pol, TRUE, COSTS, doms))
                    q_sat, q_all, q_mean, q_max = quality_metrics(pol, TRUE, doms)
                    acc[arm].append(dict(success=s, cost=c, audit=spend, pairs=npairs,
                                         switches=cls["harmful"]+cls["beneficial"]+cls["neutral"],
                                         harmful=cls["harmful"], beneficial=cls["beneficial"],
                                         neutral=cls["neutral"], retained=ret, q_sat_frac=q_sat,
                                         q_all_ok=int(q_all), q_viol_mean=q_mean, q_viol_max=q_max,
                                         wrong=float(np.mean([pol.get(d) != ideal[d] for d in doms]))))
            for arm, rs in acc.items():
                rec = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
                rows.append(dict(scenario=scenario, fold=fold, seed=sd, arm=arm,
                                 opp_frac=opp_frac, opp_any=int(opp_any),
                                 opp_savings_per_req=opp_sav, **rec))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for sc in ("quality_repair", "cost_reduction"):
        print(f"  running {sc} ...", flush=True)
        out.append(run_scenario(sc))
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"), index=False)
    print(f"  wrote free_stage1_router_sensitivity.parquet  ({len(D)} unit-arm rows)")
