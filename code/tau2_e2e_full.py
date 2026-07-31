"""Full end-to-end replay: everything RQ5 and Figure 3 need, with no new API calls.

Adds to the earlier runner: the audit/no-audit diagnostic decision, the purchase ORDER of each
model-domain pair, the resulting per-domain policies, true tolerance violations, tool-error rates,
and human-fallback sensitivity with break-even under each assumed fallback price.
"""
import json, os, collections
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "data", "processed", "tau2_complete_matrix.parquet")
OUT = os.path.join(HERE, "..", "results", "tau2_e2e_full.json")

DELTA = 0.02
BUDGET_PAIRS = 8
N_AUDIT = 10
OUTER_FOLDS, SPLIT_SEEDS, ACQ_SEEDS = 5, 10, 20
ROUTER_TOL, EXPLORE = 0.05, 0.15
DIAG_THRESHOLD = 1.054           # frozen by leave-one-partition-out CV on LLMRouterBench (P2)
FALLBACK = [0, 2, 5, 10, 25]
HORIZONS = [1_000, 10_000, 100_000, 1_000_000]


def load():
    df = pd.read_parquet(MATRIX)
    tc = df.groupby(["domain", "task_id", "agent_model"]).agg(
        success=("task_success", "mean"), cost=("agent_cost_usd", "mean"),
        tool_err=("tool_error_count", "mean")).reset_index()
    return df, tc


def cheapest_good_enough(sub, delta):
    best = sub["success"].max()
    ok = sub[sub["success"] >= best - delta].sort_values(
        ["cost", "success", "agent_model"], ascending=[True, False, True])
    return ok.iloc[0]["agent_model"]


def split_tasks(keys, rnd):
    out = {}
    for dom in sorted({d for d, _ in keys}):
        ks = [k for k in keys if k[0] == dom]; rnd.shuffle(ks); n = len(ks)
        ns, nr, na = max(1, round(.20*n)), max(1, round(.25*n)), max(1, round(.15*n))
        out.update({k: "scoring" for k in ks[:ns]})
        out.update({k: "router" for k in ks[ns:ns+nr]})
        out.update({k: "audit" for k in ks[ns+nr:ns+nr+na]})
        out.update({k: "log" for k in ks[ns+nr+na:]})
    return out


def fit_router(df, tr_keys, models, doms):
    tr = df[[(d, t) in tr_keys for d, t in zip(df.domain, df.task_id)]]
    txt = tr.groupby(["domain", "task_id"]).initial_user_message.first()
    if len(txt) < 4:
        return None
    vec = TfidfVectorizer(max_features=500, min_df=1, ngram_range=(1, 2))
    Xt = vec.fit_transform(txt.values)
    idx = list(txt.index)
    def feats(pairs, texts):
        T = vec.transform(texts)
        ex = np.array([[len(t), len(t.split()), sum(c.isdigit() for c in t), t.count("?")]
                       + [1.0 if d == dd else 0.0 for dd in doms] for (d, _), t in zip(pairs, texts)])
        return np.hstack([T.toarray(), ex])
    X = feats(idx, txt.values)
    clf = {}
    for m in models:
        y = np.array([tr[(tr.domain == d) & (tr.task_id == t) & (tr.agent_model == m)]
                      .task_success.mean() for d, t in idx])
        y = np.nan_to_num(y, nan=0.0) >= 0.5
        if len(set(y)) < 2:
            clf[m] = float(y.mean()); continue
        bs, bc = -1, 1.0
        for C in (0.01, 0.1, 1, 10):
            sc = LogisticRegression(C=C, max_iter=600).fit(X, y).score(X, y)
            if sc > bs: bs, bc = sc, C
        clf[m] = LogisticRegression(C=bc, max_iter=600).fit(X, y)
    return dict(feats=feats, clf=clf)


def buy(tc, dom, model, audit_keys, rng):
    pool = [t for (d, t) in audit_keys if d == dom]
    if not pool: return None, 0.0
    take = pool if len(pool) <= N_AUDIT else list(rng.choice(pool, N_AUDIT, replace=False))
    sub = tc[(tc.domain == dom) & (tc.agent_model == model) & (tc.task_id.isin(take))]
    if not len(sub): return None, 0.0
    return float(sub.success.mean()), float(sub.cost.sum())


def diagnostic(logged, seeds):
    """Dispersion of logged-minus-audited over the seed pairs, standardised. Audit if above tau."""
    d = [logged[k] - v for k, v in seeds.items() if k in logged and np.isfinite(logged[k])]
    if len(d) < 2: return 0.0, False
    se = np.std(d, ddof=1) / np.sqrt(len(d)) or 1e-9
    stat = float(np.sqrt(np.mean((np.array(d) / max(se, 1e-9)) ** 2)))
    return stat, stat >= DIAG_THRESHOLD


def run_case(logged, tc, A, models, doms, COSTS, rng, dollar=False):
    """Returns policy, audit spend, ordered purchase list, diagnostic stat/decision."""
    est, order, spend = {}, [], 0.0
    seeds = {}
    for m in models:                                    # step 1: one seed pair per model
        d = doms[rng.integers(len(doms))]
        v, c = buy(tc, d, m, A, rng)
        if v is not None:
            est[(d, m)] = v; seeds[(d, m)] = v; spend += c
            order.append(dict(domain=d, model=m, order=len(order) + 1, cost=c, seed=True))
    stat, go = diagnostic(logged, seeds)                # step 2-3: audit or stop
    def adj():
        b = {}
        for m in models:
            g = [logged[(d, m)] - est[(d, m)] for (d, mm) in est if mm == m
                 and (d, m) in logged and np.isfinite(logged[(d, m)])]
            b[m] = float(np.mean(g)) if g else 0.0
        return b
    def pol_from():
        b = adj(); rows = []
        for d in doms:
            for m in models:
                v = est.get((d, m), logged.get((d, m), np.nan) - b.get(m, 0.0))
                rows.append(dict(domain=d, agent_model=m, success=v, cost=COSTS[(d, m)]))
        t = pd.DataFrame(rows).dropna(subset=["success"])
        return {d: cheapest_good_enough(t[t.domain == d], DELTA) for d in doms
                if len(t[t.domain == d])}
    pol = pol_from()
    if go:
        while len(order) < BUDGET_PAIRS:
            cands = [(d, m) for d in doms for m in models if (d, m) not in est]
            if not cands: break
            def pri(dm):
                d, m = dm
                vals = [est.get((d, mm), logged.get((d, mm), np.nan)) for mm in models]
                vals = [x for x in vals if not np.isnan(x)]
                v = est.get((d, m), logged.get((d, m), np.nan))
                if not vals or np.isnan(v): return 0.0
                p = 1.0 / (1e-6 + abs(v - (max(vals) - DELTA)))
                return p / max(1e-9, COSTS[(d, m)] * N_AUDIT) if dollar else p
            d, m = max(cands, key=pri)
            v, c = buy(tc, d, m, A, rng)
            if v is None: break
            est[(d, m)] = v; spend += c
            order.append(dict(domain=d, model=m, order=len(order) + 1, cost=c, seed=False))
            pol = pol_from()
    return pol, spend, order, stat, go


def score(pol, tc, S, truth_cell):
    """Held-out success, cost, tool errors, and true tolerance violations."""
    s = c = te = n = 0.0
    for d, t in S:
        m = pol.get(d)
        r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
        if len(r):
            s += float(r.success.iloc[0]); c += float(r.cost.iloc[0])
            te += float(r.tool_err.iloc[0]); n += 1
    viol = []
    for d in sorted({d for d, _ in S}):
        m = pol.get(d)
        sub = truth_cell[truth_cell.domain == d]
        if m is not None and len(sub):
            viol.append(float(sub[sub.agent_model == m].success.iloc[0] < sub.success.max() - DELTA))
    return (s/n if n else np.nan, c/n if n else np.nan, te/n if n else np.nan,
            float(np.mean(viol)) if viol else np.nan, int(n))


def main():
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth_cell = tc.groupby(["domain", "agent_model"]).agg(
        success=("success", "mean"), cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth_cell.itertuples()}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})

    recs, purchases, triggered, total = [], [], 0, 0
    import random as _r
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            rng = np.random.default_rng(1000*fold + sd)
            asg = split_tasks(list(keys), _r.Random(1000*fold + sd))
            S = [k for k, v in asg.items() if v == "scoring"]
            R = set(k for k, v in asg.items() if v == "router")
            L = [k for k, v in asg.items() if v == "log"]
            A = [k for k, v in asg.items() if v == "audit"]
            if not (S and R and L and A): continue
            router = fit_router(df, R, models, doms)
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
                if len(r): seen[(d, m)].append(float(r.success.iloc[0]))
            logged = {k: float(np.mean(v)) for k, v in seen.items()}
            log_tbl = pd.DataFrame([dict(domain=d, agent_model=m, success=logged.get((d, m), np.nan),
                                         cost=COSTS[(d, m)]) for d in doms for m in models]).dropna(subset=["success"])

            pols = {"Cheapest model": {d: min(models, key=lambda m: COSTS[(d, m)]) for d in doms},
                    "Ideal complete matrix": {d: cheapest_good_enough(truth_cell[truth_cell.domain == d], DELTA) for d in doms},
                    "Raw routed log": {d: cheapest_good_enough(log_tbl[log_tbl.domain == d], DELTA)
                                       for d in doms if len(log_tbl[log_tbl.domain == d])}}
            gm = max(models, key=lambda m: np.nanmean([logged.get((d, m), np.nan) for d in doms]))
            pols["Highest-success model"] = {d: gm for d in doms}
            pols["Best single model"] = {d: gm for d in doms}
            audit_usd = collections.defaultdict(float)
            for name, dollar in (("CASE", False), ("Dollar-aware CASE", True)):
                acc = []
                for a in range(ACQ_SEEDS):
                    p2, sp, order, stat, go = run_case(logged, tc, A, models, doms, COSTS,
                                                       np.random.default_rng(7000+a), dollar)
                    acc.append((p2, sp, order, stat, go))
                    if name == "CASE":
                        total += 1; triggered += int(go)
                        if a == 0 and fold == 0 and sd == 0: purchases = order
                pols[name] = acc[0][0]; audit_usd[name] = float(np.mean([x[1] for x in acc]))
                recs.append(dict(kind="meta", fold=fold, seed=sd, policy=name,
                                 n_pairs=float(np.mean([len(x[2]) for x in acc])),
                                 diag_stat=float(np.mean([x[3] for x in acc]))))
            r3 = np.random.default_rng(99); allp = [(d, m) for d in doms for m in models]; r3.shuffle(allp)
            uest, usp = {}, 0.0
            for (d, m) in allp[:BUDGET_PAIRS]:
                v, c = buy(tc, d, m, A, r3)
                if v is not None: uest[(d, m)] = v; usp += c
            ut = pd.DataFrame([dict(domain=d, agent_model=m, success=uest.get((d, m), logged.get((d, m), np.nan)),
                                    cost=COSTS[(d, m)]) for d in doms for m in models]).dropna(subset=["success"])
            pols["Uniform audit"] = {d: cheapest_good_enough(ut[ut.domain == d], DELTA)
                                     for d in doms if len(ut[ut.domain == d])}
            audit_usd["Uniform audit"] = usp
            ideal = pols["Ideal complete matrix"]
            for name, pol in pols.items():
                s, c, te, viol, n = score(pol, tc, S, truth_cell)
                recs.append(dict(kind="score", fold=fold, seed=sd, policy=name, success=s,
                                 agent_cost=c, tool_err=te, violation=viol, n=n,
                                 audit_usd=audit_usd.get(name, 0.0),
                                 switches=sum(pol.get(d) != pols["Raw routed log"].get(d) for d in doms),
                                 wrong=float(np.mean([pol.get(d) != ideal.get(d) for d in doms])),
                                 policy_map=json.dumps(pol)))
    out = dict(diagnostic_triggered=triggered, diagnostic_total=total,
               diagnostic_threshold=DIAG_THRESHOLD, example_purchase_order=purchases,
               records=recs)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"  {len(recs)} records; diagnostic recommended auditing in {triggered}/{total} runs")
    print(f"  wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
