"""Priority 1 (sections 2.3-2.11): end-to-end CASE replay on the existing tau2 simulations.

Selective routed log -> diagnostic audit -> CASE buys more audits -> routing-adjusted estimates
-> cheapest-good-enough policy -> held-out operational result -> audit-inclusive cost -> break-even.

No new API calls: every outcome and cost is replayed from tau2_complete_matrix.parquet.
"""
import json, os, hashlib, itertools, collections
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "data", "processed", "tau2_complete_matrix.parquet")
RES = os.path.join(HERE, "..", "results")

DELTAS = [0.00, 0.02, 0.05]
PRIMARY_DELTA = 0.02
PAIR_BUDGETS = [4, 6, 8, 10, 12]
N_AUDIT = 10                 # n = min(10, available audit tasks)
ACQ_SEEDS = 20
OUTER_FOLDS, SPLIT_SEEDS = 5, 10
ROUTER_TOL, EXPLORE = 0.05, 0.15
HORIZONS = [1_000, 10_000, 100_000, 1_000_000]
FALLBACK = [0, 2, 5, 10, 25]


def cell_table(df):
    """(domain, model) -> mean success and mean agent cost, averaged over tasks and trials."""
    g = df.groupby(["domain", "agent_model"]).agg(
        success=("task_success", "mean"), cost=("agent_cost_usd", "mean"),
        n=("task_success", "size")).reset_index()
    return g


def task_cells(df):
    """(domain, task, model) -> success prob over trials, mean cost. The complete matrix."""
    return df.groupby(["domain", "task_id", "agent_model"]).agg(
        success=("task_success", "mean"), cost=("agent_cost_usd", "mean")).reset_index()


def cheapest_good_enough(sub, delta):
    """sub: rows for one domain with columns model, success, cost. Ties: cost, then success, then id."""
    best = sub["success"].max()
    ok = sub[sub["success"] >= best - delta]
    ok = ok.sort_values(["cost", "success", "agent_model"], ascending=[True, False, True])
    return ok.iloc[0]["agent_model"]


def split_tasks(keys, rng):
    """20% scoring / 25% router-train / 40% log / 15% audit, grouped by task, stratified by domain."""
    out = {}
    for dom in sorted({d for d, _ in keys}):
        ks = [k for k in keys if k[0] == dom]
        rng.shuffle(ks)
        n = len(ks)
        n_s, n_r, n_a = max(1, round(.20*n)), max(1, round(.25*n)), max(1, round(.15*n))
        out.update({k: "scoring" for k in ks[:n_s]})
        out.update({k: "router" for k in ks[n_s:n_s+n_r]})
        out.update({k: "audit" for k in ks[n_s+n_r:n_s+n_r+n_a]})
        out.update({k: "log" for k in ks[n_s+n_r+n_a:]})
    return out


def fit_router(df, tr_keys, models, rng):
    """Pre-response features only; one L2 logistic per candidate model."""
    tr = df[df.set_index(["domain", "task_id"]).index.isin(tr_keys)]
    txt = tr.groupby(["domain", "task_id"]).initial_user_message.first()
    if txt.nunique() < 3:
        return None
    vec = TfidfVectorizer(max_features=500, min_df=1, ngram_range=(1, 2))
    X_txt = vec.fit_transform(txt.values)
    doms = sorted(df.domain.unique())
    def feats(idx, texts):
        Xt = vec.transform(texts)
        extra = np.array([[len(t), len(t.split()), sum(c.isdigit() for c in t), t.count("?")]
                          + [1.0 if d == dd else 0.0 for dd in doms] for (d, _), t in zip(idx, texts)])
        return np.hstack([Xt.toarray(), extra])
    idx = list(txt.index); X = feats(idx, txt.values)
    clf = {}
    for m in models:
        y = np.array([tr[(tr.domain == d) & (tr.task_id == t) & (tr.agent_model == m)]
                      .task_success.mean() for d, t in idx])
        y = np.nan_to_num(y, nan=0.0) >= 0.5
        if len(set(y)) < 2:
            clf[m] = float(y.mean())
            continue
        best, bestC = None, 1.0
        for C in (0.01, 0.1, 1, 10):
            lr = LogisticRegression(C=C, max_iter=400).fit(X, y)
            s = lr.score(X, y)
            if best is None or s > best: best, bestC = s, C
        clf[m] = LogisticRegression(C=bestC, max_iter=400).fit(X, y)
    return dict(vec=vec, doms=doms, clf=clf, feats=feats)


def route(router, models, dom, text, costs, rng):
    if router is None:
        return rng.choice(models)
    X = router["feats"]([(dom, 0)], [text])
    p = {}
    for m in models:
        c = router["clf"][m]
        p[m] = float(c) if isinstance(c, float) else float(c.predict_proba(X)[0, 1])
    if rng.random() < EXPLORE:
        return rng.choice(models)
    top = max(p.values())
    ok = [m for m in models if p[m] >= top - ROUTER_TOL]
    return min(ok, key=lambda m: (costs.get((dom, m), 9e9), m))


def audit_pair(tc, dom, model, audit_keys, rng):
    """Buy one model x domain pair: n = min(10, available audit tasks), without replacement."""
    pool = [t for (d, t) in audit_keys if d == dom]
    if not pool:
        return None, 0.0
    take = pool if len(pool) <= N_AUDIT else list(rng.choice(pool, N_AUDIT, replace=False))
    sub = tc[(tc.domain == dom) & (tc.agent_model == model) & (tc.task_id.isin(take))]
    if not len(sub):
        return None, 0.0
    return float(sub.success.mean()), float(sub.cost.sum())


def evaluate(policy, tc, score_keys):
    """Held-out realized success and agent cost for a domain->model policy."""
    succ, cost, n = 0.0, 0.0, 0
    for d, t in score_keys:
        m = policy.get(d)
        r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
        if len(r):
            succ += float(r.success.iloc[0]); cost += float(r.cost.iloc[0]); n += 1
    return (succ / n, cost / n, n) if n else (np.nan, np.nan, 0)


def run_case(logged, tc, audit_keys, models, doms, budget, delta, rng, dollar_aware=False):
    """Steps 1-6 of section 2.7. Returns policy, audit cost, pairs bought."""
    bought, est, spend = {}, {}, 0.0
    for m in models:                                   # 1. one seeded pair per model
        d = doms[rng.integers(len(doms))]
        v, c = audit_pair(tc, d, m, audit_keys, rng)
        if v is not None:
            bought[(d, m)] = v; est[(d, m)] = v; spend += c
    def adjust():
        b = {}
        for m in models:
            gaps = [logged.get((d, m), np.nan) - est[(d, m)] for (d, mm) in est if mm == m
                    if not np.isnan(logged.get((d, m), np.nan))]
            b[m] = float(np.mean(gaps)) if gaps else 0.0
        return b
    def policy_from(b):
        rows = []
        for d in doms:
            for m in models:
                v = est.get((d, m), logged.get((d, m), np.nan) - b.get(m, 0.0))
                rows.append(dict(domain=d, agent_model=m, success=v,
                                 cost=COSTS.get((d, m), np.nan)))
        t = pd.DataFrame(rows).dropna(subset=["success"])
        return {d: cheapest_good_enough(t[t.domain == d], delta) for d in doms if len(t[t.domain == d])}
    pol = policy_from(adjust())
    while len(bought) < budget:                        # 5-6. buy where it may change the decision
        cands = [(d, m) for d in doms for m in models if (d, m) not in bought]
        if not cands:
            break
        def priority(dm):
            d, m = dm
            sub = [est.get((d, mm), logged.get((d, mm), np.nan)) for mm in models]
            sub = [x for x in sub if not np.isnan(x)]
            if not sub: return 0.0
            v = est.get((d, m), logged.get((d, m), np.nan))
            if np.isnan(v): return 0.0
            p = 1.0 / (1e-6 + abs(v - (max(sub) - delta)))     # closeness to the threshold
            return p / max(1e-9, COSTS.get((d, m), 1e-9) * N_AUDIT) if dollar_aware else p
        d, m = max(cands, key=priority)
        v, c = audit_pair(tc, d, m, audit_keys, rng)
        if v is None:
            break
        bought[(d, m)] = v; est[(d, m)] = v; spend += c
        newpol = policy_from(adjust())
        if newpol == pol and len(bought) >= budget:
            break
        pol = newpol
    return pol, spend, len(bought)


def main():
    df = pd.read_parquet(MATRIX)
    tc = task_cells(df)
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    global COSTS
    COSTS = {(r.domain, r.agent_model): r.cost for r in cell_table(df).itertuples()}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})
    print(f"  {len(keys)} domain-task cells, {len(models)} models, {len(doms)} domains")

    recs = []
    for fold in range(OUTER_FOLDS):
        for seed in range(SPLIT_SEEDS):
            rng = np.random.default_rng(1000 * fold + seed)
            import random as _r
            assign = split_tasks(list(keys), _r.Random(1000 * fold + seed))
            S = [k for k, v in assign.items() if v == "scoring"]
            R = [k for k, v in assign.items() if v == "router"]
            L = [k for k, v in assign.items() if v == "log"]
            A = [k for k, v in assign.items() if v == "audit"]
            if not (S and R and L and A):
                continue
            router = fit_router(df, set(R), models, rng)

            # 2.5 selective routed log: only the selected model's outcome is revealed
            seen = collections.defaultdict(list)
            for d, t in L:
                txt = df[(df.domain == d) & (df.task_id == t)].initial_user_message.iloc[0]
                m = route(router, models, d, txt, COSTS, rng)
                r = tc[(tc.domain == d) & (tc.task_id == t) & (tc.agent_model == m)]
                if len(r): seen[(d, m)].append(float(r.success.iloc[0]))
            logged = {k: float(np.mean(v)) for k, v in seen.items()}

            truth = pd.DataFrame([dict(domain=d, agent_model=m,
                                       success=float(tc[(tc.domain == d) & (tc.agent_model == m)].success.mean()),
                                       cost=COSTS[(d, m)]) for d in doms for m in models])
            log_tbl = pd.DataFrame([dict(domain=d, agent_model=m, success=logged.get((d, m), np.nan),
                                         cost=COSTS[(d, m)]) for d in doms for m in models]).dropna(subset=["success"])

            pols = {}
            pols["Cheapest model"] = {d: min(models, key=lambda m: COSTS[(d, m)]) for d in doms}
            gm = max(models, key=lambda m: np.nanmean([logged.get((d, m), np.nan) for d in doms]))
            pols["Highest-success model"] = {d: gm for d in doms}
            pols["Best single model"] = {d: gm for d in doms}
            pols["Selective router (free log)"] = {d: (log_tbl[log_tbl.domain == d]
                                                       .sort_values("success", ascending=False).iloc[0].agent_model
                                                       if len(log_tbl[log_tbl.domain == d]) else models[0]) for d in doms}
            pols["Raw routed log"] = {d: cheapest_good_enough(log_tbl[log_tbl.domain == d], PRIMARY_DELTA)
                                      for d in doms if len(log_tbl[log_tbl.domain == d])}
            pols["Ideal reference"] = {d: cheapest_good_enough(truth[truth.domain == d], PRIMARY_DELTA) for d in doms}

            for name, dollar in (("CASE", False), ("Dollar-aware CASE", True)):
                accs = []
                for a in range(ACQ_SEEDS):
                    r2 = np.random.default_rng(7000 + a)
                    p, spend, nb = run_case(logged, tc, A, models, doms, 8, PRIMARY_DELTA, r2, dollar)
                    accs.append((p, spend, nb))
                pols[name] = accs[0][0]
                recs.append(dict(fold=fold, seed=seed, policy=name,
                                 audit_usd=float(np.mean([x[1] for x in accs])),
                                 pairs=float(np.mean([x[2] for x in accs]))))
            # uniform audit: buy pairs at random
            r3 = np.random.default_rng(99)
            allp = [(d, m) for d in doms for m in models]; r3.shuffle(allp)
            uest, uspend = {}, 0.0
            for (d, m) in allp[:8]:
                v, c = audit_pair(tc, d, m, A, r3)
                if v is not None: uest[(d, m)] = v; uspend += c
            ut = pd.DataFrame([dict(domain=d, agent_model=m,
                                    success=uest.get((d, m), logged.get((d, m), np.nan)),
                                    cost=COSTS[(d, m)]) for d in doms for m in models]).dropna(subset=["success"])
            pols["Uniform audit"] = {d: cheapest_good_enough(ut[ut.domain == d], PRIMARY_DELTA)
                                     for d in doms if len(ut[ut.domain == d])}
            recs.append(dict(fold=fold, seed=seed, policy="Uniform audit", audit_usd=uspend, pairs=8))

            ideal = pols["Ideal reference"]
            for name, pol in pols.items():
                s, c, n = evaluate(pol, tc, S)
                wrong = np.mean([pol.get(d) != ideal.get(d) for d in doms])
                recs.append(dict(fold=fold, seed=seed, policy=name, success=s,
                                 agent_cost=c, wrong_choice=wrong, n_scored=n))
    out = pd.DataFrame(recs)
    out.to_parquet(os.path.join(HERE, "..", "data", "processed", "tau2_end_to_end_raw.parquet"), index=False)
    print(f"  {len(out)} records over {OUTER_FOLDS} folds x {SPLIT_SEEDS} seeds")
    return out


if __name__ == "__main__":
    main()
