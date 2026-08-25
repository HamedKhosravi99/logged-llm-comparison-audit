"""Prop. 3.4 certificate on LLMRouterBench at stated serving budgets, over 30 partitions.

Answers option (a) of the reviewer's objection: the budgeted program of Eq. (1) is exercised on a
real corpus at a stated Gamma, and the settledness certificate is reported, rather than being
checked only on sampled synthetic instances.

Two facts make 30 seeds tractable.
  * xhat is independent of eta.  e_kj = eta_j/2 is constant across models within a workload, so
    muhat = mu_obs - eta/2 shifts every allocation's objective by the same constant.  Solve once.
  * settledness is monotone in eta.  The boxes are nested, so binary search finds the margin.

Both are used only to speed the search; every verdict is still the certificate of Prop. 3.4,
computed by solving at the adverse table.

OVERWRITES the git-tracked cert30.json next to this file. Pass a path as the last argument
to write somewhere else, which is what you want if you are only checking the numbers.
"""
import itertools, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = None            # set from argv; None means overwrite the tracked file
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "code"))
from eval_validity.grid import build_base          # noqa: E402
from econ_core import cost_tables                  # noqa: E402

TOL = 1e-12
MULT = [1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]      # Gamma as a multiple of the cheapest feasible spend


def cell_tables(base):
    Q, ctx, coh, K, J = base["Q"], base["ctx"], base["coh"], base["K"], base["J"]
    mu = np.full((K, J), np.nan)
    for j in range(J):
        m = (coh == "score") & (ctx == j)
        if m.sum():
            mu[:, j] = Q[m].mean(0)
    ct = cost_tables(base)
    return mu, ct["c_score"], ct["p"]


def enumerate_allocations(K, J):
    return np.array(list(itertools.product(range(K), repeat=J)), dtype=np.int64)


def solver(mu, p, w, X):
    """Vectorised: returns (value, cost) for every allocation in X."""
    J = X.shape[1]
    cols = np.arange(J)
    cost = (p[X, cols] * w).sum(1)
    def value(m):
        return (m[X, cols] * w).sum(1)
    return value, cost


def best(value_fn, cost, mu, gamma):
    v = value_fn(mu)
    feas = cost <= gamma + TOL
    if not feas.any():
        return None, -np.inf
    vv = np.where(feas, v, -np.inf)
    i = int(np.argmax(vv))
    return i, float(vv[i])


def settled_at(mu_obs, p, w, X, value_fn, cost, gamma, ihat, eta):
    """Prop. 3.4 second form at spread eta (scalar, applied to every workload)."""
    K, J = mu_obs.shape
    e = np.full((K, J), eta / 2.0)
    mu_hat = mu_obs - e
    xhat = X[ihat]
    mu_chk = mu_hat + e                                  # = mu_obs
    for j in range(J):
        mu_chk[xhat[j], j] = mu_hat[xhat[j], j] - e[xhat[j], j]
    iadv, _ = best(value_fn, cost, mu_chk, gamma)
    if iadv is None:
        return None
    return bool(np.array_equal(X[iadv], xhat))


def margin(mu_obs, p, w, X, value_fn, cost, gamma, ihat, hi=0.30, tol=5e-4):
    """Largest eta with a settled allocation, by bisection (settledness is monotone in eta)."""
    if not settled_at(mu_obs, p, w, X, value_fn, cost, gamma, ihat, 0.0):
        return 0.0, False                                 # not settled even with no error
    if settled_at(mu_obs, p, w, X, value_fn, cost, gamma, ihat, hi):
        return hi, True                                   # settled across the whole range probed
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if settled_at(mu_obs, p, w, X, value_fn, cost, gamma, ihat, mid):
            lo = mid
        else:
            hi = mid
    return lo, False


def main(n_seeds=30):
    rows, t0 = [], time.time()
    for seed in range(n_seeds):
        base = build_base(seed=seed, n_clusters=8, corpus="llmrouterbench")
        mu, p, w = cell_tables(base)
        K, J = mu.shape
        if not (np.isfinite(mu).all() and np.isfinite(p).all()):
            print("seed %d: missing cells, skipped" % seed); continue
        X = enumerate_allocations(K, J)
        value_fn, cost = solver(mu, p, w, X)

        cheap = float(sum(w[j] * p[:, j].min() for j in range(J)))
        # unconstrained optimum = per-workload argmax
        xu = np.array([int(np.argmax(mu[:, j])) for j in range(J)])
        qu = float(sum(w[j] * mu[xu[j], j] for j in range(J)))
        # per-workload best-to-runner-up gaps, and exact ties
        gaps = np.array([np.sort(mu[:, j])[-1] - np.sort(mu[:, j])[-2] for j in range(J)])
        n_ties = int((gaps <= 0).sum())

        for m in MULT:
            g = cheap * m
            ihat, vhat = best(value_fn, cost, mu, g)
            if ihat is None:
                continue
            xhat = X[ihat]
            q = float(sum(w[j] * mu[xhat[j], j] for j in range(J)))
            c = float(cost[ihat])
            active = q < qu - 1e-12
            mg, capped = margin(mu, p, w, X, value_fn, cost, g, ihat)
            rows.append(dict(seed=seed, mult=m, gamma=g, spend=c, quality=q,
                             gap_to_unconstrained=qu - q, constraint_active=bool(active),
                             n_feasible=int((cost <= g + TOL).sum()),
                             margin=float(mg), margin_capped=bool(capped),
                             min_gap=float(gaps.min()), n_exact_ties=n_ties))
        print("seed %2d done  (%.0fs elapsed)" % (seed, time.time() - t0), flush=True)

    out = OUT_PATH or os.path.join(HERE, "cert30.json")
    json.dump(rows, open(out, "w"), indent=1)
    print("\nwrote %s" % out)

    print("\n=== settledness margin over %d partitions ===" % n_seeds)
    print("%-7s %-9s %-9s %-9s %-11s %-9s %s" %
          ("Gam/ch", "median", "IQR", "min", "max", "active", "seeds w/ exact tie"))
    for m in MULT:
        r = [x for x in rows if x["mult"] == m]
        if not r: continue
        v = np.array([x["margin"] for x in r])
        act = sum(x["constraint_active"] for x in r)
        tie = sum(x["n_exact_ties"] > 0 for x in r)
        print("%-7.2f %-9.4f %-9s %-9.4f %-11.4f %-9s %d/%d" %
              (m, np.median(v), "%.4f-%.4f" % (np.percentile(v, 25), np.percentile(v, 75)),
               v.min(), v.max(), "%d/%d" % (act, len(r)), tie, len(r)))

    print("\n=== does the slack-budget margin equal the tightest per-workload gap? ===")
    sl = [x for x in rows if not x["constraint_active"]]
    if sl:
        d = np.array([x["margin"] - x["min_gap"] for x in sl])
        print("  %d slack rows; margin - min_gap: median %.5f, max |diff| %.5f (bisection tol 5e-4)"
              % (len(sl), float(np.median(d)), float(np.abs(d).max())))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        OUT_PATH = sys.argv[2]
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
