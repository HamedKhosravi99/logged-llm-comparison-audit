"""Stage 1 of the free plan: fix the method with data already saved. NO new API calls.

Implements and evaluates, on the existing 440 tau2 simulations:
  1. richer routing-bias corrections   - one-adjustment / additive / partially-pooled / low-rank
  2. estimate combination              - audit overwrite / inverse-variance / cross-fitted shrinkage
  3. SWITCH / RETAIN / VALIDATE        - quality and net-savings conditions, both at 95%
  4. expected-value-of-information     - replaces the transferred threshold as the audit trigger
  5. consequence-aware acquisition     - margin / dollar / traffic / full expected value

Results go to results/free_stage1_safe_audit.{json,md}. Nothing here touches the paper.
"""
import json, os, itertools, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "data", "processed", "tau2_complete_matrix.parquet")
RES = os.path.join(HERE, "..", "results")

DELTA = 0.02
H_PRIMARY, H_SENS = 100_000, (10_000, 1_000_000)
CONF = 0.95
N_AUDIT, BUDGET_PAIRS = 10, 8
OUTER_FOLDS, SPLIT_SEEDS = 5, 10
BOOT = 400
LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
RANKS = (1, 2, 3)


def load():
    df = pd.read_parquet(MATRIX)
    tc = df.groupby(["domain", "task_id", "agent_model"]).agg(
        success=("task_success", "mean"), cost=("agent_cost_usd", "mean")).reset_index()
    return df, tc


# ---------------------------------------------------------------- corrections
def fit_correction(dobs, models, doms, kind, lam=1.0, rank=1):
    """dobs: {(dom, model): (d, var)} observed logged-minus-audit gaps on bought pairs.
    Returns b(dom, model)."""
    keys = list(dobs)
    if not keys:
        return lambda d, m: 0.0
    y = np.array([dobs[k][0] for k in keys])
    w = np.array([1.0 / max(dobs[k][1], 1e-6) for k in keys])
    K, J = len(models), len(doms)
    mi = {m: i for i, m in enumerate(models)}; ji = {d: i for i, d in enumerate(doms)}
    if kind == "one_adjustment":
        b = {}
        for m in models:
            v = [dobs[k][0] for k in keys if k[1] == m]
            b[m] = float(np.mean(v)) if v else 0.0
        return lambda d, m: b.get(m, 0.0)
    # design matrix for alpha (K) + beta (J) [+ gamma (K*J)]
    cols = K + J + (K * J if kind == "pooled" else 0)
    X = np.zeros((len(keys), cols))
    for r, (d, m) in enumerate(keys):
        X[r, mi[m]] = 1.0
        X[r, K + ji[d]] = 1.0
        if kind == "pooled":
            X[r, K + J + mi[m] * J + ji[d]] = 1.0
    P = np.diag([1e-3] * K + [1e-3] * J + ([lam] * (K * J) if kind == "pooled" else []))
    W = np.diag(w)
    theta = np.linalg.solve(X.T @ W @ X + P, X.T @ W @ y)
    if kind in ("additive", "pooled"):
        def b(d, m):
            v = theta[mi[m]] + theta[K + ji[d]]
            if kind == "pooled":
                v += theta[K + J + mi[m] * J + ji[d]]
            return float(v)
        return b
    if kind == "lowrank":
        # alpha + beta from the additive fit, then a rank-r fit to the residual
        base = np.linalg.solve(X[:, :K+J].T @ W @ X[:, :K+J] + np.eye(K+J)*1e-3,
                               X[:, :K+J].T @ W @ y)
        resid = np.full((K, J), np.nan)
        for r_, (d, m) in enumerate(keys):
            resid[mi[m], ji[d]] = y[r_] - (base[mi[m]] + base[K + ji[d]])
        Rm = np.nan_to_num(resid, nan=0.0)
        U, S, Vt = np.linalg.svd(Rm, full_matrices=False)
        r = min(rank, len(S))
        low = (U[:, :r] * S[:r]) @ Vt[:r]
        return lambda d, m: float(base[mi[m]] + base[K + ji[d]] + low[mi[m], ji[d]])
    raise ValueError(kind)


# ---------------------------------------------------------------- combination
def combine(kind, M, Mvar, R, Rvar, b):
    """Return the deployed quality estimate and its variance."""
    adj, adjvar = (M - b), Mvar
    if R is None or not np.isfinite(R):
        return adj, adjvar
    if kind == "overwrite":
        return R, Rvar
    if kind == "inv_var":
        wr, wm = 1.0 / max(Rvar, 1e-9), 1.0 / max(adjvar, 1e-9)
        return (wr * R + wm * adj) / (wr + wm), 1.0 / (wr + wm)
    if kind == "shrinkage":
        # James-Stein style pull of the noisy audit toward the adjusted log
        lam = adjvar / max(adjvar + Rvar, 1e-9)
        return lam * R + (1 - lam) * adj, min(Rvar, adjvar)
    raise ValueError(kind)


# ---------------------------------------------------------------- decision rule
def decide(dom, models, q, qv, costs, w_j, H, delta=DELTA, conf=CONF, incumbent=None, rng=None):
    """SWITCH / RETAIN / VALIDATE for one request type, from posterior draws."""
    ms = [m for m in models if np.isfinite(q[m])]
    if len(ms) < 2:
        return "VALIDATE", incumbent, dict(reason="insufficient support")
    draws = {m: rng.normal(q[m], np.sqrt(max(qv[m], 1e-12)), BOOT) for m in ms}
    D = np.vstack([draws[m] for m in ms])                 # models x BOOT
    best = D.max(axis=0)
    # candidate: cheapest model whose point estimate is within delta of the best point estimate
    top = max(q[m] for m in ms)
    ok = [m for m in ms if q[m] >= top - delta]
    cand = min(ok, key=lambda m: (costs[(dom, m)], m))
    inc = incumbent if incumbent in ms else cand
    if cand == inc:
        return "RETAIN", inc, dict(reason="candidate is the incumbent")
    i = ms.index(cand)
    p_quality = float(np.mean(D[i] >= best - delta))
    # net savings over H requests at this type's traffic share, audit cost charged elsewhere
    per_req = costs[(dom, inc)] - costs[(dom, cand)]
    p_savings = 1.0 if per_req > 0 else 0.0
    if p_quality >= conf and p_savings >= conf:
        return "SWITCH", cand, dict(p_quality=p_quality, p_savings=p_savings)
    if p_quality < conf:
        return "RETAIN", inc, dict(p_quality=p_quality, p_savings=p_savings)
    return "VALIDATE", inc, dict(p_quality=p_quality, p_savings=p_savings)


# ---------------------------------------------------------------- acquisition
def priority(kind, dom, m, q, qv, costs, w_j, H, audit_cost, delta=DELTA):
    ms = [x for x in q if np.isfinite(q[x])]
    if not ms or not np.isfinite(q.get(m, np.nan)):
        return 0.0
    top = max(q[x] for x in ms)
    margin = abs(q[m] - (top - delta))
    p_change = float(np.exp(-(margin ** 2) / (2 * max(qv.get(m, 1e-4), 1e-6))))
    if kind == "margin":
        return 1.0 / (1e-6 + margin)
    if kind == "dollar":
        return (1.0 / (1e-6 + margin)) / max(audit_cost, 1e-9)
    if kind == "traffic":
        return (1.0 / (1e-6 + margin)) * w_j
    if kind == "full_ev":
        cheapest_ok = min([x for x in ms if q[x] >= top - delta],
                          key=lambda x: costs[(dom, x)])
        dcost = abs(costs[(dom, m)] - costs[(dom, cheapest_ok)])
        return p_change * H * w_j * dcost / max(audit_cost, 1e-9)
    raise ValueError(kind)


def evi(dom, m, q, qv, costs, w_j, H, audit_cost, delta=DELTA):
    """Expected value of buying this audit, in dollars, against its price."""
    return priority("full_ev", dom, m, q, qv, costs, w_j, H, audit_cost, delta) * audit_cost - audit_cost


def split_tasks(keys, rnd):
    out = {}
    for dom in sorted({d for d, _ in keys}):
        ks = [k for k in keys if k[0] == dom]; rnd.shuffle(ks); n = len(ks)
        ns, nr, na = max(1, round(.20*n)), max(1, round(.25*n)), max(1, round(.15*n))
        out.update({k: "scoring" for k in ks[:ns]}); out.update({k: "router" for k in ks[ns:ns+nr]})
        out.update({k: "audit" for k in ks[ns+nr:ns+nr+na]}); out.update({k: "log" for k in ks[ns+nr+na:]})
    return out


ARMS = [
    ("Raw routed log",              None,             None,        None,      None),
    ("Current CASE",                "one_adjustment", "overwrite", "margin",  False),
    ("Additive correction",         "additive",       "overwrite", "margin",  False),
    ("Partially pooled correction", "pooled",         "overwrite", "margin",  False),
    ("Low-rank correction",         "lowrank",        "overwrite", "margin",  False),
    ("Inverse-variance combination","pooled",         "inv_var",   "margin",  False),
    ("Cross-fitted shrinkage",      "pooled",         "shrinkage", "margin",  False),
    ("Traffic-aware acquisition",   "pooled",         "inv_var",   "traffic", False),
    ("Full expected-value acq.",    "pooled",         "inv_var",   "full_ev", False),
    ("Safe CASE with abstention",   "pooled",         "inv_var",   "full_ev", True),
]


def main():
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                      cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth.itertuples()}
    TRUE = {(r.domain, r.agent_model): r.success for r in truth.itertuples()}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})
    w_j = {d: 1.0 / len(doms) for d in doms}
    ideal = {}
    for d in doms:
        sub = truth[truth.domain == d]; top = sub.success.max()
        ok = sub[sub.success >= top - DELTA].sort_values(["cost", "success"], ascending=[True, False])
        ideal[d] = ok.iloc[0].agent_model

    import random as _r
    rows = []
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            rng = np.random.default_rng(4000*fold + sd)
            asg = split_tasks(list(keys), _r.Random(4000*fold + sd))
            S = [k for k, v in asg.items() if v == "scoring"]
            L = [k for k, v in asg.items() if v == "log"]
            A = [k for k, v in asg.items() if v == "audit"]
            if not (S and L and A): continue

            # selective log: cheapest model per task wins most traffic, with exploration
            seen = collections.defaultdict(list)
            for d, t in L:
                m = (models[rng.integers(len(models))] if rng.random() < 0.15
                     else min(models, key=lambda x: COSTS[(d, x)]))
                r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
                if len(r): seen[(d, m)].append(float(r.success.iloc[0]))
            M = {k: float(np.mean(v)) for k, v in seen.items()}
            Mvar = {k: float(np.var(v, ddof=1) / len(v)) if len(v) > 1 else 0.05 for k, v in seen.items()}
            raw_pol = {}
            for d in doms:
                cand = [(m, M[(d, m)]) for m in models if (d, m) in M]
                if not cand: continue
                top = max(v for _, v in cand)
                ok = [m for m, v in cand if v >= top - DELTA]
                raw_pol[d] = min(ok, key=lambda m: (COSTS[(d, m)], m))

            for name, corr, comb, acq, safe in ARMS:
                if corr is None:
                    pol, spend, absten = dict(raw_pol), 0.0, 0
                else:
                    est, dobs, spend, order = {}, {}, 0.0, []
                    def audit(d, m):
                        nonlocal spend
                        pool = [t for (dd, t) in A if dd == d]
                        if not pool: return None
                        take = pool if len(pool) <= N_AUDIT else list(rng.choice(pool, N_AUDIT, replace=False))
                        sub = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(take))]
                        if not len(sub): return None
                        spend += float(sub.cost.sum())
                        v = float(sub.success.mean())
                        var = float(sub.success.var(ddof=1) / len(sub)) if len(sub) > 1 else 0.05
                        est[(d, m)] = (v, var)
                        if (d, m) in M: dobs[(d, m)] = (M[(d, m)] - v, max(var + Mvar[(d, m)], 1e-6))
                        return v
                    for m in models:                       # seed one pair per model
                        audit(doms[rng.integers(len(doms))], m)
                    lam = 1.0
                    if corr == "pooled":                   # grouped CV over the ridge grid
                        best, lam = np.inf, 1.0
                        for lg in LAMBDA_GRID:
                            errs = []
                            for hold in list(dobs):
                                tr = {k: v for k, v in dobs.items() if k != hold}
                                if len(tr) < 2: continue
                                b = fit_correction(tr, models, doms, "pooled", lam=lg)
                                errs.append((dobs[hold][0] - b(*hold[::-1][::-1]))**2)
                            if errs and np.mean(errs) < best: best, lam = np.mean(errs), lg
                    while len(est) < BUDGET_PAIRS:
                        b = fit_correction(dobs, models, doms, corr, lam=lam)
                        q = {(d, m): (est[(d, m)][0] if (d, m) in est
                                      else (M[(d, m)] - b(d, m) if (d, m) in M else np.nan))
                             for d in doms for m in models}
                        qv = {(d, m): (est[(d, m)][1] if (d, m) in est
                                       else Mvar.get((d, m), 0.05)) for d in doms for m in models}
                        cands = [(d, m) for d in doms for m in models if (d, m) not in est]
                        if not cands: break
                        def sc(dm):
                            d, m = dm
                            ac = COSTS[(d, m)] * N_AUDIT
                            qq = {x: q[(d, x)] for x in models}; vv = {x: qv[(d, x)] for x in models}
                            return priority(acq, d, m, qq, vv, COSTS, w_j[d], H_PRIMARY, ac)
                        d, m = max(cands, key=sc)
                        if safe:                            # EVI gate replaces the threshold
                            ac = COSTS[(d, m)] * N_AUDIT
                            qq = {x: q[(d, x)] for x in models}; vv = {x: qv[(d, x)] for x in models}
                            if evi(d, m, qq, vv, COSTS, w_j[d], H_PRIMARY, ac) <= 0: break
                        if audit(d, m) is None: break
                    b = fit_correction(dobs, models, doms, corr, lam=lam)
                    pol, absten = {}, 0
                    for d in doms:
                        q, qv = {}, {}
                        for m in models:
                            Rv, Rvar = est.get((d, m), (np.nan, np.nan))
                            Mv, Mv2 = M.get((d, m), np.nan), Mvar.get((d, m), 0.05)
                            if not np.isfinite(Mv) and not np.isfinite(Rv): q[m], qv[m] = np.nan, np.nan; continue
                            if not np.isfinite(Mv): q[m], qv[m] = Rv, Rvar
                            else:
                                qq, vv = combine(comb, Mv, Mv2, Rv if np.isfinite(Rv) else None,
                                                 Rvar, b(d, m))
                                q[m], qv[m] = qq, vv
                        if safe:
                            act, choice, _ = decide(d, models, q, qv, COSTS, w_j[d], H_PRIMARY,
                                                    incumbent=raw_pol.get(d), rng=rng)
                            pol[d] = choice if choice else raw_pol.get(d)
                            absten += int(act != "SWITCH")
                        else:
                            ms = [m for m in models if np.isfinite(q[m])]
                            if not ms: pol[d] = raw_pol.get(d); continue
                            top = max(q[m] for m in ms)
                            ok = [m for m in ms if q[m] >= top - DELTA]
                            pol[d] = min(ok, key=lambda m: (COSTS[(d, m)], m))
                # score on held-out tasks
                s = c = n = 0.0
                for d, t in S:
                    m = pol.get(d)
                    r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
                    if len(r): s += float(r.success.iloc[0]); c += float(r.cost.iloc[0]); n += 1
                harmful = sum(1 for d in doms if pol.get(d) != raw_pol.get(d)
                              and pol.get(d) and raw_pol.get(d)
                              and TRUE[(d, pol[d])] < TRUE[(d, raw_pol[d])] - DELTA)
                rows.append(dict(arm=name, fold=fold, seed=sd, success=s/n if n else np.nan,
                                 cost=c/n if n else np.nan, audit=spend, harmful=harmful,
                                 abstentions=absten,
                                 wrong=float(np.mean([pol.get(d) != ideal[d] for d in doms]))))
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(HERE, "..", "data", "processed", "free_stage1_raw.parquet"), index=False)
    return out, TRUE, COSTS, ideal


if __name__ == "__main__":
    main()
