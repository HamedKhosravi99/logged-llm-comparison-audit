"""Prop. 3.4 certificate on either corpus, using the exact MCKP solver instead of enumeration.

`run30.py` enumerates every allocation, which caps it at LLMRouterBench (K=5, J=8, 390,625 rows).
RouterBench is K=11, J=8, so enumeration would need 214,358,881 rows. This script calls
`code/mckp.py`, a Pareto-frontier dynamic program that is exact in floating point, so both corpora
run through the same path in about a second each.

Conventions follow run30.py exactly. The admissible box is centred at muhat = mu_obs - eta/2 with
half-width e = eta/2, as Sec. 3 states, so the adverse table leaves rivals at mu_obs and pushes each
selected cell down to mu_obs - eta. The incumbent xhat does not depend on eta, because a uniform
eta shifts every allocation's objective by the same constant, so it is solved once per budget.

One deliberate difference. run30.py calls an allocation settled when the adverse argmax is
identical to xhat; this script calls it settled when xhat attains the adverse optimum, which is
Prop. 3.4's own condition, xhat in X*(mu_check), and admits ties. Where the two disagree the
allocation was tied at the adverse table, and this script is the one that matches the proposition.

Usage:
    python source/budgeted_certificate/run_certificate_mckp.py routerbench       [n_seeds]
    python source/budgeted_certificate/run_certificate_mckp.py llmrouterbench    [n_seeds]

OVERWRITES the git-tracked cert30_<corpus>_mckp.json next to this file. Pass a path as the third
argument to write somewhere else, which is what you want if you are only checking the numbers.
"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, HERE)

from eval_validity.grid import build_base            # noqa: E402
from run_budgeted_certificate import cell_tables     # noqa: E402
from mckp import solve_mckp, allocation_value, is_settled   # noqa: E402

MULT = [1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]         # Gamma as a multiple of cheapest feasible spend
HI, TOL_ETA = 0.30, 5e-4


def margin(mu_obs, p, w, gamma, xhat):
    """Largest eta at which xhat is still certified, by bisection (settledness is monotone)."""
    def settled(eta):
        e = np.full(mu_obs.shape, eta / 2.0)
        return is_settled(mu_obs - e, p, w, gamma, e, xhat)

    if not settled(0.0):
        return 0.0, False
    if settled(HI):
        return HI, True
    lo, hi = 0.0, HI
    while hi - lo > TOL_ETA:
        mid = (lo + hi) / 2.0
        if settled(mid):
            lo = mid
        else:
            hi = mid
    return lo, False


def main(corpus="routerbench", n_seeds=30, out_path=None):
    rows, t0 = [], time.time()
    for seed in range(n_seeds):
        base = build_base(seed=seed, n_clusters=8, corpus=corpus)
        mu, p, w = cell_tables(base)
        if not (np.isfinite(mu).all() and np.isfinite(p).all()):
            print("seed %d: missing cells, skipped" % seed, flush=True)
            continue
        J = mu.shape[1]

        cheapest = float(sum(w[j] * p[:, j].min() for j in range(J)))
        xu = np.array([int(np.argmax(mu[:, j])) for j in range(J)])
        qu = allocation_value(mu, w, xu)
        gaps = np.array([np.sort(mu[:, j])[-1] - np.sort(mu[:, j])[-2] for j in range(J)])

        for m in MULT:
            gamma = cheapest * m
            q, xhat, feasible = solve_mckp(mu, p, w, gamma)
            if not feasible:
                continue
            mg, capped = margin(mu, p, w, gamma, xhat)
            rows.append(dict(
                corpus=corpus, seed=seed, mult=m, gamma=gamma,
                spend=float(sum(w[j] * p[xhat[j], j] for j in range(J))),
                quality=q, gap_to_unconstrained=qu - q,
                constraint_active=bool(q < qu - 1e-12),
                margin=float(mg), margin_capped=bool(capped),
                min_gap=float(gaps.min()), n_exact_ties=int((gaps <= 0).sum()),
            ))
        print("seed %2d done  (%.0fs elapsed)" % (seed, time.time() - t0), flush=True)

    out = out_path or os.path.join(HERE, "cert30_%s_mckp.json" % corpus)
    json.dump(rows, open(out, "w"), indent=1)

    print("\n=== settledness margin, %s, %d partitions ===" % (corpus, n_seeds))
    print("%-8s %-9s %-9s %-9s %-9s %s" % ("Gam/ch", "median", "min", "max", "active", "capped"))
    for m in MULT:
        r = [x for x in rows if x["mult"] == m]
        if not r:
            continue
        v = np.array([x["margin"] for x in r])
        print("%-8s %-9.4f %-9.4f %-9.4f %-9d %d"
              % (m, np.median(v), v.min(), v.max(),
                 sum(x["constraint_active"] for x in r),
                 sum(x["margin_capped"] for x in r)))
    print("\nwrote %s  (%.0fs total)" % (out, time.time() - t0))


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "routerbench"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(corpus, seeds, sys.argv[3] if len(sys.argv) > 3 else None)
