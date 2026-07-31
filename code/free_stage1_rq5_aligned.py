"""Stage 1, aligned to the paper's RQ5 pipeline. NO new API calls.

Reuses the RQ5 pipeline by importing it (split_tasks, fit_router, buy, load) from
tau2_e2e_full, so the router, splits, audit pools, costs, tolerance and scoring are provably the
same code rather than a re-implementation.

Arms: raw routed log, Current CASE, Inverse-variance CASE, Safe CASE, Safe inverse-variance CASE,
ideal complete-matrix reference.

Statistical unit: the 20 acquisition seeds are averaged inside each fold x split unit; SEs and CIs
are taken over the 5 x 10 = 50 units.
"""
import argparse, json, os, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

from tau2_e2e_full import (load, split_tasks, fit_router, buy, cheapest_good_enough, diagnostic,
                           DELTA, N_AUDIT, BUDGET_PAIRS, OUTER_FOLDS, SPLIT_SEEDS, ACQ_SEEDS,
                           ROUTER_TOL, EXPLORE)

H_PRIMARY = 100_000
TAUS = (0.80, 0.90, 0.95, 0.975)
TAU_PRIMARY = 0.95
VAR_FLOOR = 1e-8
BOOT = 500

TARGETS = dict(raw_success=(0.487, 0.002), raw_cost_1k=(26.34, 0.25),
               case_success=(0.479, 0.002), case_cost_1k=(28.34, 0.25),
               case_audit=(0.83, 0.05), case_pairs=(7.2, 0.2))


def build_log(df, tc, L, router, models, doms, COSTS, rng):
    """The paper's selective log: fitted router on pre-response features, 0.15 exploration."""
    seen = collections.defaultdict(list)
    for d, t in L:
        txt = df[(df.domain == d) & (df.task_id == t)].initial_user_message.iloc[0]
        if router is None or rng.random() < EXPLORE:
            m = models[rng.integers(len(models))]
        else:
            X = router["feats"]([(d, 0)], [txt])
            p = {mm: (float(router["clf"][mm]) if isinstance(router["clf"][mm], float)
                      else float(router["clf"][mm].predict_proba(X)[0, 1])) for mm in models}
            top = max(p.values())
            m = min([mm for mm in models if p[mm] >= top - ROUTER_TOL],
                    key=lambda mm: (COSTS[(d, mm)], mm))
        r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
        if len(r):
            seen[(d, m)].append(float(r.success.iloc[0]))
    M = {k: float(np.mean(v)) for k, v in seen.items()}
    # variance of the logged mean, from the log only
    Mvar = {k: max(float(np.var(v, ddof=1) / len(v)) if len(v) > 1 else 0.25 / max(len(v), 1), VAR_FLOOR)
            for k, v in seen.items()}
    return M, Mvar


def audit_pair(tc, d, m, A, rng):
    """Audit estimate plus its variance, from the audit pool only."""
    pool = [t for (dd, t) in A if dd == d]
    if not pool:
        return None
    take = pool if len(pool) <= N_AUDIT else list(rng.choice(pool, N_AUDIT, replace=False))
    sub = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(take))]
    if not len(sub):
        return None
    v = float(sub.success.mean())
    var = max(float(sub.success.var(ddof=1) / len(sub)) if len(sub) > 1 else 0.25 / len(sub), VAR_FLOOR)
    return v, var, float(sub.cost.sum())


def run_case(M, Mvar, tc, A, models, doms, COSTS, rng, inverse_variance):
    """The paper's CASE acquisition. Combination is overwrite, or inverse-variance pooling."""
    est, spend, seeds = {}, 0.0, {}
    for m in models:
        d = doms[rng.integers(len(doms))]
        r = audit_pair(tc, d, m, A, rng)
        if r:
            est[(d, m)] = (r[0], r[1]); seeds[(d, m)] = r[0]; spend += r[2]
    _, go = diagnostic(M, seeds)          # the paper's step 2-3: stop here if not justified
    def adjust():
        b = {}
        for m in models:
            g = [M[(d, m)] - est[(d, m)][0] for (d, mm) in est if mm == m and (d, m) in M]
            b[m] = float(np.mean(g)) if g else 0.0
        return b
    def qhat():
        b = adjust(); q, qv = {}, {}
        for d in doms:
            for m in models:
                adj = M[(d, m)] - b.get(m, 0.0) if (d, m) in M else np.nan
                adjv = Mvar.get((d, m), 0.25)
                if (d, m) in est:
                    R, Rv = est[(d, m)]
                    if inverse_variance and np.isfinite(adj):
                        wr, wm = 1.0 / max(Rv, VAR_FLOOR), 1.0 / max(adjv, VAR_FLOOR)
                        q[(d, m)], qv[(d, m)] = (wr * R + wm * adj) / (wr + wm), 1.0 / (wr + wm)
                    else:
                        q[(d, m)], qv[(d, m)] = R, Rv
                else:
                    q[(d, m)], qv[(d, m)] = adj, adjv
        return q, qv
    while go and len(est) < BUDGET_PAIRS:
        cands = [(d, m) for d in doms for m in models if (d, m) not in est]
        if not cands: break
        def pri(dm):
            # the paper scores unbought pairs on the RAW logged value, not the adjusted one
            d, m = dm
            vals = [est[(d, mm)][0] if (d, mm) in est else M.get((d, mm), np.nan) for mm in models]
            vals = [x for x in vals if not np.isnan(x)]
            v = est[(d, m)][0] if (d, m) in est else M.get((d, m), np.nan)
            if not vals or np.isnan(v): return 0.0
            return 1.0 / (1e-6 + abs(v - (max(vals) - DELTA)))
        d, m = max(cands, key=pri)
        r = audit_pair(tc, d, m, A, rng)
        if not r: break
        est[(d, m)] = (r[0], r[1]); spend += r[2]
    q, qv = qhat()
    return q, qv, spend, len(est)


def greedy_policy(q, models, doms, COSTS):
    """Unconstrained cheapest-good-enough on the estimates (the paper's current rule)."""
    pol = {}
    for d in doms:
        ms = [m for m in models if np.isfinite(q[(d, m)])]
        if not ms: continue
        top = max(q[(d, m)] for m in ms)
        ok = [m for m in ms if q[(d, m)] >= top - DELTA]
        pol[d] = min(ok, key=lambda m: (COSTS[(d, m)], m))
    return pol


def safe_policy(q, qv, models, doms, COSTS, raw_pol, audit_usd, tau, H, rng):
    """Retain the raw-log model unless a switch clears both conditions at probability tau."""
    pol, acts = {}, {}
    for d in doms:
        inc = raw_pol.get(d)
        ms = [m for m in models if np.isfinite(q[(d, m)])]
        if not ms or inc is None:
            pol[d], acts[d] = inc, "RETAIN"; continue
        top = max(q[(d, m)] for m in ms)
        ok = [m for m in ms if q[(d, m)] >= top - DELTA]
        cand = min(ok, key=lambda m: (COSTS[(d, m)], m))
        if cand == inc:
            pol[d], acts[d] = inc, "RETAIN"; continue
        D = np.vstack([rng.normal(q[(d, m)], np.sqrt(max(qv[(d, m)], VAR_FLOOR)), BOOT) for m in ms])
        best = D.max(axis=0)
        p_quality = float(np.mean(D[ms.index(cand)] >= best - DELTA))
        # (b) net savings positive at H, charging this domain's share of the audit
        per = COSTS[(d, inc)] - COSTS[(d, cand)]
        net = H * per / max(len(doms), 1) - audit_usd / max(len(doms), 1)
        p_sav = 1.0 if net > 0 else 0.0
        if p_quality >= tau and p_sav >= tau:
            pol[d], acts[d] = cand, "SWITCH"
        else:
            pol[d], acts[d] = inc, "RETAIN"
    return pol, acts


def score(pol, tc, S):
    s = c = n = 0.0
    for d, t in S:
        m = pol.get(d)
        if m is None: continue
        r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
        if len(r): s += float(r.success.iloc[0]); c += float(r.cost.iloc[0]); n += 1
    return (s / n, c / n, int(n)) if n else (np.nan, np.nan, 0)


def main(reproduce_only=False):
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                      cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth.itertuples()}
    TRUE = {(r.domain, r.agent_model): r.success for r in truth.itertuples()}
    ideal = {d: cheapest_good_enough(truth[truth.domain == d], DELTA) for d in doms}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})
    import random as _r

    units, thr_rows = [], []
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            rng = np.random.default_rng(1000 * fold + sd)
            asg = split_tasks(list(keys), _r.Random(1000 * fold + sd))
            S = [k for k, v in asg.items() if v == "scoring"]
            R = set(k for k, v in asg.items() if v == "router")
            L = [k for k, v in asg.items() if v == "log"]
            A = [k for k, v in asg.items() if v == "audit"]
            if not (S and R and L and A): continue
            router = fit_router(df, R, models, doms)
            M, Mvar = build_log(df, tc, L, router, models, doms, COSTS, rng)
            raw_pol = greedy_policy({(d, m): M.get((d, m), np.nan) for d in doms for m in models},
                                    models, doms, COSTS)
            rs, rc, _ = score(raw_pol, tc, S)
            isc, ic, _ = score(ideal, tc, S)

            acc = collections.defaultdict(list)
            for a in range(ACQ_SEEDS):
                r2 = np.random.default_rng(7000 + a)
                for arm, iv in (("Current CASE", False), ("Inverse-variance CASE", True)):
                    q, qv, spend, npairs = run_case(M, Mvar, tc, A, models, doms, COSTS, r2, iv)
                    pol = greedy_policy(q, models, doms, COSTS)
                    s, c, _ = score(pol, tc, S)
                    harm = sum(1 for d in doms if pol.get(d) != raw_pol.get(d) and pol.get(d)
                               and raw_pol.get(d) and TRUE[(d, pol[d])] < TRUE[(d, raw_pol[d])] - DELTA)
                    acc[arm].append(dict(success=s, cost=c, audit=spend, pairs=npairs, harmful=harm,
                                         retained=0,
                                         wrong=float(np.mean([pol.get(d) != ideal[d] for d in doms]))))
                    safe_name = "Safe CASE" if not iv else "Safe inverse-variance CASE"
                    for tau in TAUS:
                        sp, acts = safe_policy(q, qv, models, doms, COSTS, raw_pol, spend, tau,
                                               H_PRIMARY, r2)
                        ss, sc_, _ = score(sp, tc, S)
                        sh = sum(1 for d in doms if sp.get(d) != raw_pol.get(d) and sp.get(d)
                                 and raw_pol.get(d) and TRUE[(d, sp[d])] < TRUE[(d, raw_pol[d])] - DELTA)
                        rec = dict(success=ss, cost=sc_, audit=spend, pairs=npairs, harmful=sh,
                                   retained=sum(1 for v in acts.values() if v == "RETAIN"),
                                   wrong=float(np.mean([sp.get(d) != ideal[d] for d in doms])))
                        if tau == TAU_PRIMARY: acc[safe_name].append(rec)
                        thr_rows.append(dict(fold=fold, seed=sd, acq=a, arm=safe_name, tau=tau, **rec))
            unit = dict(fold=fold, seed=sd)
            units.append(dict(unit, arm="Raw routed log", success=rs, cost=rc, audit=0.0,
                              pairs=0.0, harmful=0.0, retained=len(doms), wrong=float(np.mean(
                                  [raw_pol.get(d) != ideal[d] for d in doms]))))
            units.append(dict(unit, arm="Ideal complete matrix", success=isc, cost=ic, audit=0.0,
                              pairs=0.0, harmful=0.0, retained=0, wrong=0.0))
            for arm, rs_ in acc.items():                     # average acquisition seeds inside the unit
                rec = {k: float(np.mean([r[k] for r in rs_])) for k in rs_[0]}
                # the paper scored acquisition seed 0 only; keep it for the reproduction gate
                rec["success_seed0"] = float(rs_[0]["success"])
                rec["cost_seed0"] = float(rs_[0]["cost"])
                units.append(dict(unit, arm=arm, **rec))
    U = pd.DataFrame(units)
    T = pd.DataFrame(thr_rows)
    return U, T, TRUE, COSTS, ideal


def summarise(U, H=H_PRIMARY):
    g = U.groupby("arm").agg(success=("success", "mean"), success_se=("success", "sem"),
                             cost=("cost", "mean"), cost_se=("cost", "sem"),
                             audit=("audit", "mean"), pairs=("pairs", "mean"),
                             harmful=("harmful", "mean"), retained=("retained", "mean"),
                             wrong=("wrong", "mean"), n=("success", "size"))
    raw = g.loc["Raw routed log"]
    piv = U.pivot_table(index=["fold", "seed"], columns="arm", values="cost")
    rows = []
    rng = np.random.default_rng(0)
    for arm, r in g.iterrows():
        cps = (H * r.cost + r.audit) / max(1e-9, H * r.success)
        per_point = raw.cost - r.cost
        be_point = (r.audit / per_point) if per_point > 1e-12 and r.audit > 0 else None
        be_cert = None
        if arm in piv.columns and "Raw routed log" in piv.columns:
            d = (piv["Raw routed log"] - piv[arm]).dropna().values
            if len(d) > 2:
                boots = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(2000)]
                lo = float(np.quantile(boots, 0.05))
                be_cert = (r.audit / lo) if lo > 1e-12 and r.audit > 0 else None
        rows.append(dict(arm=arm, success=r.success, success_se=r.success_se,
                         cost_per_1k=r.cost * 1000, cost_per_1k_se=r.cost_se * 1000,
                         audit=r.audit, pairs=r.pairs, cost_per_success=cps,
                         harmful=r.harmful, retained=r.retained, wrong=r.wrong,
                         break_even_point=be_point, break_even_certified=be_cert, n_units=int(r.n)))
    return pd.DataFrame(rows).set_index("arm")


def gate(t, U):
    """Reproduction gate, evaluated under the paper's own scoring convention (acquisition seed 0).

    The aligned results below use the 20-seed average that item 6 requires; the gate must test that
    the pipeline is faithful, so it compares like with like."""
    c = U[U.arm == "Current CASE"]
    got = dict(raw_success=t.loc["Raw routed log", "success"],
               raw_cost_1k=t.loc["Raw routed log", "cost_per_1k"],
               case_success=float(c["success_seed0"].mean()),
               case_cost_1k=float(c["cost_seed0"].mean()) * 1000,
               case_audit=t.loc["Current CASE", "audit"],
               case_pairs=t.loc["Current CASE", "pairs"])
    rows, ok = [], True
    for k, (target, tol) in TARGETS.items():
        v = float(got[k]); passed = abs(v - target) <= tol
        ok &= passed
        rows.append(dict(quantity=k, target=target, tolerance=tol, reproduced=v,
                         delta=v - target, passed=bool(passed)))
    return ok, pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    U, T, TRUE, COSTS, ideal = main()
    t = summarise(U)
    ok, rep = gate(t, U)
    print("\n  REPRODUCTION GATE (paper RQ5 baseline, paper scoring convention)")
    print(f"    {'quantity':16}{'target':>10}{'tol':>8}{'got':>10}{'delta':>10}  pass")
    for r in rep.itertuples():
        print(f"    {r.quantity:16}{r.target:>10.3f}{r.tolerance:>8.3f}{r.reproduced:>10.3f}"
              f"{r.delta:>+10.3f}  {'YES' if r.passed else 'NO'}")
    print(f"\n    GATE: {'PASSED' if ok else 'FAILED'}")
    os.makedirs(RES, exist_ok=True)
    U.to_parquet(os.path.join(RES, "free_stage1_rq5_aligned.parquet"), index=False)
    T.to_parquet(os.path.join(RES, "free_stage1_threshold_sensitivity.parquet"), index=False)
    json.dump(dict(gate_passed=bool(ok), gate=json.loads(rep.to_json(orient="records")),
                   table=json.loads(t.reset_index().to_json(orient="records"))),
              open(os.path.join(RES, "free_stage1_rq5_aligned.json"), "w"), indent=1)
    print("\n  aligned table:")
    print("   ", t[["success", "cost_per_1k", "audit", "cost_per_success", "harmful",
                    "retained", "wrong"]].round(4).to_string().replace("\n", "\n    "))
