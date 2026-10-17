"""The decision margin over the ROW-STRUCTURED admissible set, beside the free box.

Eq. (12) admits C_kj in [0, eta_j] independently for every cell, a K x J box. But Thm. C.4
proves non-identification over P, the class whose error is purely model-level, C_kj = C_k --
a K-dimensional family. The box is therefore a strict relaxation of what the paper's own
theorem licenses, and an unsettled verdict over the box may be an artifact of the relaxation.

This measures both. For row-constant C the objective is linear in C over [0, eta]^K, so the
worst case sits at a vertex -- but at a DIFFERENT vertex for each rival, which is exactly why
Prop. 3.1's single-adverse-table argument does not carry over. So settledness over the row set
is decided by enumerating all 2^K vertices and asking whether xhat attains the optimum at each.

Conventions follow run_certificate_mckp.py exactly, so the box column reproduces Table 3.

    python run_certificate_row.py <corpus> [n_seeds] [out.json]
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "source", "budgeted_certificate"))

from eval_validity.grid import build_base
from run_budgeted_certificate import cell_tables
from mckp import solve_mckp, allocation_value, is_settled

MULT = [1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
HI, TOL_ETA = 0.30, 5e-4


def settled_box(mu, p, w, gamma, xhat, eta):
    """The paper's test: one adverse vertex, worst against every rival at once."""
    e = np.full(mu.shape, eta / 2.0)
    return is_settled(mu - e, p, w, gamma, e, xhat)


def settled_row(mu, p, w, gamma, xhat, eta):
    """xhat optimal at every vertex of {C : C_kj = C_k, C_k in [0, eta]}."""
    K = mu.shape[0]
    for mask in range(1 << K):
        C = np.array([eta if (mask >> k) & 1 else 0.0 for k in range(K)])
        mu_v = mu - C[:, None]
        best, _, feasible = solve_mckp(mu_v, p, w, gamma)
        if not feasible:
            continue
        if allocation_value(mu_v, w, xhat) < best - 1e-9:
            return False
    return True


def margin(test, mu, p, w, gamma, xhat):
    if not test(mu, p, w, gamma, xhat, 0.0):
        return 0.0, False
    if test(mu, p, w, gamma, xhat, HI):
        return HI, True
    lo, hi = 0.0, HI
    while hi - lo > TOL_ETA:
        mid = (lo + hi) / 2.0
        if test(mu, p, w, gamma, xhat, mid):
            lo = mid
        else:
            hi = mid
    return lo, False


def main(corpus, n_seeds, out_path):
    rows, t0 = [], time.time()
    for seed in range(n_seeds):
        base = build_base(seed=seed, n_clusters=8, corpus=corpus)
        mu, p, w = cell_tables(base)
        if not (np.isfinite(mu).all() and np.isfinite(p).all()):
            continue
        K, J = mu.shape
        cheapest = float(sum(w[j] * p[:, j].min() for j in range(J)))
        for m in MULT:
            gamma = cheapest * m
            q, xhat, feasible = solve_mckp(mu, p, w, gamma)
            if not feasible:
                continue
            mb, _ = margin(settled_box, mu, p, w, gamma, xhat)
            mr, _ = margin(settled_row, mu, p, w, gamma, xhat)
            rows.append(dict(corpus=corpus, seed=seed, mult=m, K=int(K), J=int(J),
                             margin_box=float(mb), margin_row=float(mr)))
        print("seed %2d done  (%.0fs)" % (seed, time.time() - t0), flush=True)
    json.dump(rows, open(out_path, "w"), indent=1)
    print("\nwrote %s  (%d rows, %.0fs)" % (out_path, len(rows), time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30, sys.argv[3])
