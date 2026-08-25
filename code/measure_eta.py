"""Measure the evaluator-error spread eta on a corpus where both signals exist.

Everywhere else in this paper eta is assumed. It can be measured wherever the same requests carry
both a proxy score and the designated outcome, which on KuaiRand-Pure they do: `is_click` is the
cheap signal an operator would log, `is_like` is the outcome the operator actually wants. Then

    C_kj  = E[is_click - is_like | arm k, context j]          per-cell evaluator error
    eta_j = max_k C_kj - min_k C_kj                           within-workload spread

which is exactly the quantity (Spread) bounds in Sec. 3. Only within-workload differences enter the
decision, so a constant common to a whole workload is irrelevant and eta_j is the right summary.

C is computed on the randomized-exposure stream, so selection bias does not contaminate it: under
random exposure E[Y_ref | arm] = mu_ref, and the click-minus-like difference is evaluator error
alone. The scoring pool is used by default, matching Sec. 5.1; --pool all uses the whole randomized
stream as a sensitivity check.

Usage:
    python code/measure_eta.py --data code/data/kuairand/KuaiRand-Pure/data
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kuairand_validity import J, K, POOLS, cell_means, load_frames   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--pool", default="score", choices=["score", "all"])
    ap.add_argument("--min-rows", type=int, default=30, help="minimum rows for a cell to count")
    ap.add_argument("--out", default="results/eta_measured.json", type=Path)
    args = ap.parse_args()

    _, rnd, ctx_levels = load_frames(args.data)
    if args.pool == "score":
        lo, hi = POOLS["score"]
        rnd = rnd[(rnd.bucket >= lo) & (rnd.bucket < hi)]

    mu_click, n_c = cell_means(rnd, "is_click")
    mu_like, n_l = cell_means(rnd, "is_like")
    C = mu_click - mu_like
    ok = (n_c >= args.min_rows) & (n_l >= args.min_rows) & np.isfinite(C)

    print("cells with >= %d rows: %d of %d" % (args.min_rows, ok.sum(), K * J))
    print()
    print("%-28s %8s %8s %8s %8s" % ("workload (context)", "cells", "min C", "max C", "eta_j"))
    etas, rows = [], []
    for j in range(J):
        ks = np.flatnonzero(ok[:, j])
        if len(ks) < 2:
            print("%-28s %8d %8s %8s %8s" % (ctx_levels[j][:28], len(ks), "--", "--", "--"))
            continue
        c = C[ks, j]
        eta = float(c.max() - c.min())
        etas.append(eta)
        rows.append(dict(j=j, context=ctx_levels[j], n_cells=int(len(ks)),
                         C_min=float(c.min()), C_max=float(c.max()), eta=eta))
        print("%-28s %8d %8.4f %8.4f %8.4f" % (ctx_levels[j][:28], len(ks), c.min(), c.max(), eta))

    etas = np.array(etas)
    print()
    print("eta over %d workloads: median %.4f, min %.4f, max %.4f, mean %.4f"
          % (len(etas), np.median(etas), etas.min(), etas.max(), etas.mean()))
    print()
    for assumed in (0.005, 0.01, 0.02, 0.05):
        print("  workloads whose measured eta exceeds an assumed %.3f : %d of %d"
              % (assumed, int((etas > assumed).sum()), len(etas)))

    grand = float(np.nanmean(C[ok]))
    print()
    print("mean evaluator error C over supported cells: %.4f" % grand)
    print("(a constant common to a workload does not affect the decision; only eta_j does)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(pool=args.pool, min_rows=args.min_rows, by_workload=rows,
                   eta_median=float(np.median(etas)), eta_min=float(etas.min()),
                   eta_max=float(etas.max()), C_mean=grand),
              open(args.out, "w"), indent=1)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
