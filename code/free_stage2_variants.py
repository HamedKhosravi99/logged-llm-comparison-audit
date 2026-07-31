"""Stage 2, part 2: protocol variants for the cost-reduction decision. NO new API calls.

The stage-2 diagnosis (free_stage2_diagnosis.py) shows the current gate certifies an
infeasible hypothesis: "candidate within delta of the TRUE BEST" has slack <= delta = 0.02 by
construction, needing ~1,700 audit tasks per pair against a pool of 3-5. Testing the candidate
against the DEPLOYED INCUMBENT instead needs ~3 tasks, because genuine opportunities sit far
above the incumbent in their domain.

Variants (identical units, seeds, scoring, costs to stage 1):
  S0   Confidence-gated IV CASE, tau=0.95      -- exact stage-1 arm, reproduction gate
  V1u  Incumbent-anchored non-inferiority gate, incumbent estimated from its own log (free)
  V1p  Same, incumbent re-scored on the audit tasks (paired same-task test, incumbent audit
       cost charged)
  V3   Best-anchored certificate with the same candidate-restricted deep audits
       (shows the hypothesis, not the sample allocation, is what binds)

All variants audit ONLY models cheaper than the incumbent (candidate-restricted acquisition)
on the FULL audit pool of the domain, and switch to the cheapest candidate that passes.

A cost-saving claim requires, pre-stated: paired 95% CI on deployment cost excluding zero,
or a finite 95%-lower-bound break-even including audit spend, reported against BOTH
comparators (Raw incumbent, Raw-log downrouting).
"""
import collections, json, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")

from tau2_e2e_full import (load, split_tasks, cheapest_good_enough,
                           DELTA, OUTER_FOLDS, SPLIT_SEEDS, ACQ_SEEDS)
from free_stage1_rq5_aligned import run_case, greedy_policy, safe_policy, score, H_PRIMARY, TAU_PRIMARY, VAR_FLOOR
from free_stage1_diagnostics import incumbent_log
from free_stage1_classify import classify_switch

TAUS_NI = (0.80, 0.90, 0.95)
TAU_NI_PRIMARY = 0.95

# stage-1 numbers the S0 arm must reproduce (from free_stage1_switch_categories.parquet
# and free_stage1_router_sensitivity.parquet)
S0_TARGETS = dict(A=(0.064, 0.005), C=(0.091, 0.005), total=(0.155, 0.005),
                  audit=(0.78, 0.03))


def audit_full_pool(tc, d, m, A):
    """Deterministic audit of model m on the ENTIRE audit pool of domain d.

    Returns per-task successes (aligned to the sorted pool), their cost, or None."""
    pool = sorted(t for (dd, t) in A if dd == d)
    if not pool:
        return None
    sub = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(pool))]
    sub = sub.set_index("task_id").reindex(pool)
    if sub.success.isna().all():
        return None
    ok = sub.success.notna()
    return dict(tasks=[t for t, o in zip(pool, ok) if o],
                y=sub.success[ok].to_numpy(float), cost=float(sub.cost[ok].sum()))


def ni_gate(M, Mvar, tc, A, models, doms, COSTS, inc, tau, paired):
    """Incumbent-anchored non-inferiority gate.

    Audits every cheaper-than-incumbent candidate on the full domain pool, then switches to
    the cheapest candidate whose posterior P(q_cand >= q_inc - delta) >= tau."""
    pol, spend, audited = {}, 0.0, 0
    for d in doms:
        pol[d] = inc
        cands = [m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]]
        if not cands:
            continue
        if paired:
            ref = audit_full_pool(tc, d, inc, A)
            if ref is None:
                continue
            spend += ref["cost"]; audited += 1
        passing = []
        for m in sorted(cands, key=lambda m: COSTS[(d, m)]):
            r = audit_full_pool(tc, d, m, A)
            if r is None:
                continue
            spend += r["cost"]; audited += 1
            n = len(r["y"])
            if paired:
                common = [i for i, t in enumerate(r["tasks"]) if t in ref["tasks"]]
                ridx = {t: i for i, t in enumerate(ref["tasks"])}
                dvec = np.array([r["y"][i] - ref["y"][ridx[r["tasks"][i]]] for i in common])
                if not len(dvec):
                    continue
                dbar = float(dvec.mean())
                sd = float(dvec.std(ddof=1) / np.sqrt(len(dvec))) if len(dvec) > 1 else 0.5
                # a tiny paired sample that happens to tie must not read as certainty:
                # floor the SE at the binomial worst case for this n
                sd = max(sd, float(np.sqrt(0.25 / len(dvec))))
                p = float(stats.norm.cdf((dbar + DELTA) / sd))
            else:
                qc = float(np.mean(r["y"])); vc = max(np.var(r["y"], ddof=1) / n if n > 1 else 0.25 / n, VAR_FLOOR)
                qi = M.get((d, inc), np.nan)
                vi = Mvar.get((d, inc), 0.25)
                if not np.isfinite(qi):
                    continue
                p = float(stats.norm.cdf((qc - qi + DELTA) / np.sqrt(vc + vi)))
            if p >= tau:
                passing.append(m)
        if passing:
            pol[d] = min(passing, key=lambda m: (COSTS[(d, m)], m))
    return pol, spend, audited


def best_cert_gate(M, Mvar, tc, A, models, doms, COSTS, inc, tau, rng):
    """V3: the ORIGINAL hypothesis (within delta of the estimated best) with the same
    candidate-restricted full-pool audits. Isolates hypothesis choice from sample allocation."""
    pol, spend = {}, 0.0
    for d in doms:
        pol[d] = inc
        cands = [m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]]
        est = {}
        for m in cands:
            r = audit_full_pool(tc, d, m, A)
            if r is None:
                continue
            spend += r["cost"]
            n = len(r["y"])
            est[m] = (float(np.mean(r["y"])),
                      max(np.var(r["y"], ddof=1) / n if n > 1 else 0.25 / n, VAR_FLOOR))
        est[inc] = (M.get((d, inc), np.nan), Mvar.get((d, inc), 0.25))
        ms = [m for m in est if np.isfinite(est[m][0])]
        if len(ms) < 2 or inc not in ms:
            continue
        D = np.vstack([rng.normal(est[m][0], np.sqrt(est[m][1]), 500) for m in ms])
        best = D.max(axis=0)
        passing = []
        for m in ms:
            if m == inc:
                continue
            if float(np.mean(D[ms.index(m)] >= best - DELTA)) >= tau:
                passing.append(m)
        if passing:
            pol[d] = min(passing, key=lambda m: (COSTS[(d, m)], m))
    return pol, spend


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

    unit_rows, dec_rows = [], []
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            rng = np.random.default_rng(1000 * fold + sd)
            asg = split_tasks(list(keys), _r.Random(1000 * fold + sd))
            S = [k for k, v in asg.items() if v == "scoring"]
            R = set(k for k, v in asg.items() if v == "router")
            L = [k for k, v in asg.items() if v == "log"]
            A = [k for k, v in asg.items() if v == "audit"]
            if not (S and R and L and A): continue

            TRUEtr = {}
            for d in doms:
                rt = [t for (dd, t) in R if dd == d]
                for m in models:
                    v = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(rt))]
                    TRUEtr[(d, m)] = float(v.success.mean()) if len(v) else 0.0
            M, Mvar, inc = incumbent_log(df, tc, L, R, models, doms, COSTS, TRUEtr, rng)
            base_pol = {d: inc for d in doms}
            log_pol = greedy_policy({(d, m): M.get((d, m), np.nan) for d in doms for m in models},
                                    models, doms, COSTS)
            opp = {}
            for d in doms:
                best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
                opp[d] = any(TRUE[(d, x)] >= best - DELTA and COSTS[(d, x)] < COSTS[(d, inc)]
                             for (dd, x) in TRUE if dd == d)

            def emit(arm, pol, acq, audit_usd):
                s, c, _ = score(pol, tc, S)
                cats = collections.Counter()
                for d in doms:
                    b, cand = base_pol[d], pol.get(d)
                    if cand is None: continue
                    sw = cand != b
                    cat = classify_switch(d, b, cand, TRUE, COSTS, doms, audit_usd)[0] if sw else None
                    if cat: cats[cat] += 1
                    dec_rows.append(dict(fold=fold, seed=sd, acq=acq, arm=arm, domain=d,
                                         switched=int(sw), category=cat, opportunity=int(opp[d]),
                                         chosen=cand, base=b))
                unit_rows.append(dict(fold=fold, seed=sd, acq=acq, arm=arm, success=s, cost=c,
                                      audit=audit_usd, A=cats["A"], B=cats["B"], C=cats["C"],
                                      D=cats["D"], switches=sum(cats.values())))

            emit("Raw incumbent", base_pol, 0, 0.0)
            emit("Raw-log downrouting", log_pol, 0, 0.0)
            emit("Ideal complete matrix", ideal, 0, 0.0)

            # deterministic variants: computed once, recorded once per unit
            for tau in TAUS_NI:
                p, sp, _ = ni_gate(M, Mvar, tc, A, models, doms, COSTS, inc, tau, paired=False)
                emit(f"NI gate log-anchored tau={tau}", p, 0, sp)
                p, sp, _ = ni_gate(M, Mvar, tc, A, models, doms, COSTS, inc, tau, paired=True)
                emit(f"NI gate paired tau={tau}", p, 0, sp)
            p, sp = best_cert_gate(M, Mvar, tc, A, models, doms, COSTS, inc, 0.95,
                                   np.random.default_rng(500 + 10 * fold + sd))
            emit("Best-anchored cert, deep audits", p, 0, sp)

            # S0 reproduction arm: exact stage-1 code path and seeds
            for a in range(ACQ_SEEDS):
                q, qv, spend, _ = run_case(M, Mvar, tc, A, models, doms, COSTS,
                                           np.random.default_rng(7000 + a), True)
                pol, _acts = safe_policy(q, qv, models, doms, COSTS, base_pol, spend,
                                         TAU_PRIMARY, H_PRIMARY, np.random.default_rng(9000 + a))
                emit("S0 Confidence-gated IV CASE", pol, a, spend)

    U = pd.DataFrame(unit_rows); Dec = pd.DataFrame(dec_rows)

    # ---- collapse: mean over acq within unit, then stats over the 50 units ----
    per_unit = U.groupby(["arm", "fold", "seed"]).mean(numeric_only=True).reset_index()
    n_units = per_unit.groupby("arm").size()

    # ---- reproduction gate ----
    s0 = per_unit[per_unit.arm == "S0 Confidence-gated IV CASE"]
    got = dict(A=s0.A.mean(), C=s0.C.mean(), total=s0.switches.mean(), audit=s0.audit.mean())
    print("\n  REPRODUCTION GATE (stage-1 Confidence-gated IV CASE)")
    ok = True
    for k, (t, tol) in S0_TARGETS.items():
        v = got[k]; passed = abs(v - t) <= tol; ok &= passed
        print(f"    {k:6} target {t:.3f}  got {v:.3f}  {'PASS' if passed else 'FAIL'}")
    print(f"    GATE: {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("reproduction gate failed; conclusions below would not be trustworthy")

    # ---- summary with paired CIs against both comparators ----
    piv_c = per_unit.pivot_table(index=["fold", "seed"], columns="arm", values="cost")
    piv_s = per_unit.pivot_table(index=["fold", "seed"], columns="arm", values="success")

    def ci(v):
        v = np.asarray(v, float); m = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        h = float(stats.t.ppf(0.975, len(v) - 1) * se)
        return m, m - h, m + h

    def be(sav_per_req, audit_usd):
        v = np.asarray(sav_per_req, float)
        rng2 = np.random.default_rng(0)
        boots = [np.mean(rng2.choice(v, len(v), replace=True)) for _ in range(4000)]
        lo = float(np.quantile(boots, 0.05)); m = float(v.mean())
        point = audit_usd / m if m > 1e-12 and audit_usd > 0 else np.nan
        cert = audit_usd / lo if lo > 1e-12 and audit_usd > 0 else np.nan
        return point, cert, lo

    summary = []
    g = per_unit.groupby("arm").mean(numeric_only=True)
    for arm in g.index:
        row = dict(arm=arm, success=g.loc[arm, "success"], cost_1k=g.loc[arm, "cost"] * 1000,
                   audit=g.loc[arm, "audit"], A=g.loc[arm, "A"], B=g.loc[arm, "B"],
                   C=g.loc[arm, "C"], D=g.loc[arm, "D"], switches=g.loc[arm, "switches"],
                   n_units=int(n_units[arm]))
        for comp, tag in (("Raw incumbent", "inc"), ("Raw-log downrouting", "dr")):
            if arm == comp: continue
            dc = (piv_c[comp] - piv_c[arm]).dropna() * 1000     # positive = arm is cheaper
            ds = (piv_s[arm] - piv_s[comp]).dropna()
            m1, l1, h1 = ci(dc); m2, l2, h2 = ci(ds)
            pt, cert, lb = be((piv_c[comp] - piv_c[arm]).dropna().values, g.loc[arm, "audit"])
            row.update({f"save_{tag}": m1, f"save_{tag}_lo": l1, f"save_{tag}_hi": h1,
                        f"dsucc_{tag}": m2, f"dsucc_{tag}_lo": l2, f"dsucc_{tag}_hi": h2,
                        f"be_point_{tag}": pt, f"be_cert_{tag}": cert})
        summary.append(row)
    T = pd.DataFrame(summary).set_index("arm")

    # ---- opportunity recall / precision ----
    dd = Dec.groupby(["arm", "fold", "seed", "acq", "domain"]).first().reset_index()
    per = dd.groupby(["arm", "fold", "seed", "domain"]).agg(
        switched=("switched", "mean"), opportunity=("opportunity", "first"),
        valid=("category", lambda s: np.mean([c == "A" for c in s]))).reset_index()
    rec = per[per.opportunity == 1].groupby("arm").valid.mean().rename("recall_valid_switch")
    fpr = per[per.opportunity == 0].groupby("arm").switched.mean().rename("switch_rate_no_opp")
    T = T.join(rec).join(fpr)

    os.makedirs(RES, exist_ok=True)
    per_unit.to_parquet(os.path.join(RES, "free_stage2_variants_units.parquet"), index=False)
    Dec.to_parquet(os.path.join(RES, "free_stage2_variants_decisions.parquet"), index=False)
    T.reset_index().to_json(os.path.join(RES, "free_stage2_variants.json"), orient="records", indent=1)

    ORDER = ["Raw incumbent", "Raw-log downrouting", "S0 Confidence-gated IV CASE",
             "Best-anchored cert, deep audits",
             "NI gate log-anchored tau=0.8", "NI gate log-anchored tau=0.9",
             "NI gate log-anchored tau=0.95",
             "NI gate paired tau=0.8", "NI gate paired tau=0.9", "NI gate paired tau=0.95",
             "Ideal complete matrix"]
    cols = ["success", "cost_1k", "audit", "A", "B", "C", "switches",
            "save_inc", "save_inc_lo", "save_inc_hi", "be_cert_inc",
            "save_dr", "save_dr_lo", "save_dr_hi",
            "recall_valid_switch", "switch_rate_no_opp"]
    avail = [a for a in ORDER if a in T.index]
    print("\n  SUMMARY (means over 50 fold x seed units; saving in $/1k tasks, + = cheaper)")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(T.loc[avail, [c for c in cols if c in T.columns]].round(4).to_string())
    print("\n  wrote free_stage2_variants_units.parquet, free_stage2_variants_decisions.parquet,"
          " free_stage2_variants.json")


if __name__ == "__main__":
    main()
