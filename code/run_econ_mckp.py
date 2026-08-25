"""Table 4 under Eq. (1): the same three decision surfaces, allocated by the budgeted rule.

The committed table applies the quality-constrained rule of Eq. (2) to the log, to the
routing-adjusted scores and to the target outcome, and reports what each decision costs. Under
Eq. (1) the spend is capped by the serving budget Gamma instead of chosen by the rule, so the
comparison becomes quality and wrong-workload rate at a fixed budget, with the spend reported to
show how much of Gamma each surface actually uses.

The decision uses the estimate; realized cost and quality are always scored on the target outcome,
exactly as before.

Usage:
    python code/run_econ_mckp.py [corpus] [n_partitions] [acq_seeds] [mults]
"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import case_core as cc                                     # noqa: E402
from eval_validity import assignment as A                  # noqa: E402
from eval_validity import metrics as MT                    # noqa: E402
from eval_validity.grid import build_base                  # noqa: E402
from eval_validity.pipeline import surfaces_for            # noqa: E402
from econ_core import cost_tables                          # noqa: E402
from mckp import solve_mckp, allocation_cost, cheapest_spend   # noqa: E402
from run_case_mckp import mckp_margin_cell                 # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
SURFACES = ("log", "routing_adjusted", "target")


def case_surface(surf, supp, p, w, gamma, B, seed, n_read):
    """The routing-adjusted table CASE deploys after spending B readouts under Eq. (1)."""
    M, cellq = surf["M"], surf["cellq"]
    K, J = M.shape
    rng = np.random.default_rng(seed)
    R = np.zeros((K, J), bool)
    reads = {}

    def do_read(k, j):
        reads[(k, j)] = cc.readout(cellq, k, j, rng, n_read)
        R[k, j] = True

    for k in range(K):
        js = np.where(supp[k])[0]
        if len(js) and R.sum() < B:
            do_read(k, int(js[rng.integers(len(js))]))
    while R.sum() < B:
        V = cc.corrected(M, R, reads, cc.bias_rows(M, R, reads))
        cell = mckp_margin_cell(V, p, w, gamma, supp, R)
        if cell is None:
            break
        do_read(*cell)
    return cc.corrected(M, R, reads, cc.bias_rows(M, R, reads)), int(R.sum())


def main(corpus="llmrouterbench", n_partitions=30, acq_seeds=6, mults=(1.25, 2.0, 4.0, 8.0),
         budget_frac=0.35, n_read=30, eps=0.15, conds=("D1", "D0")):
    rows, t0 = [], time.time()
    for part in range(n_partitions):
        base = build_base(part, n_clusters=8, corpus=corpus)
        K, J = base["K"], base["J"]
        cellsA1, n_kj, _ = A.make_selective_log(base, part, eps=eps)
        supp = A.support(n_kj, base, n_read=n_read)
        tab = MT.scoring_tables(base, supp)
        ct = cost_tables(base, supp)
        p, w = ct["c_score"], ct["p"]
        mu_ref = tab["mu"]
        if not (np.isfinite(mu_ref).all() and np.isfinite(p).all()):
            continue
        cheapest = cheapest_spend(p, w, supp)
        B = max(K, int(round(budget_frac * int(supp.sum()))))

        for mult in mults:
            gamma = cheapest * mult
            _, x_ref, feasible = solve_mckp(mu_ref, p, w, gamma, supp)
            if not feasible:
                continue
            for cond in conds:
                surf = surfaces_for(base, cellsA1, cond, 1.0, n_read)
                for s in range(acq_seeds):
                    seed = 10_000 * part + 100 * int(budget_frac * 100) + s
                    V, used = case_surface(surf, supp, p, w, gamma, B, seed, n_read)
                    tables = {"log": surf["M"], "routing_adjusted": V, "target": mu_ref}
                    for name in SURFACES:
                        if name != "routing_adjusted" and s > 0:
                            continue               # no acquisition seed to average over
                        _, x, ok = solve_mckp(tables[name], p, w, gamma, supp)
                        if not ok:
                            continue
                        rows.append(dict(
                            corpus=corpus, partition=int(part), acq_seed=int(s), cond=cond,
                            mult=float(mult), surface=name, reads=int(used) if name ==
                            "routing_adjusted" else 0,
                            quality=float(sum(w[j] * mu_ref[x[j], j] for j in range(J))),
                            usd_per_1m=1e6 * allocation_cost(p, w, x),
                            gamma_per_1m=1e6 * gamma,
                            wrong=float(np.mean([x[j] != x_ref[j] for j in range(J)]))))
        print("partition %2d done (%.0fs, %d rows)" % (part, time.time() - t0, len(rows)),
              flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "econ_mckp_%s.json" % corpus)
    json.dump(rows, open(out, "w"), indent=1)
    report(rows, corpus)
    print("\nwrote %s  (%.0fs)" % (out, time.time() - t0))
    return out


def report(rows, corpus):
    import collections
    print("\n=== %s: Eq. (1) at each Gamma, decision on three surfaces (A1) ===" % corpus)
    print("%-5s %-6s %-18s %10s %14s %12s %8s"
          % ("eval", "Gamma", "decide on", "quality", "$/1M req", "Gamma $/1M", "wrong"))
    ser = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        ser[(r["cond"], r["mult"], r["surface"])][r["partition"]].append(r)
    for key in sorted(ser, key=lambda k: (k[0], k[1], SURFACES.index(k[2]))):
        d = ser[key]
        def col(f):
            v = np.array([np.mean([x[f] for x in d[q]]) for q in sorted(d)])
            return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        q, qse = col("quality")
        c, cse = col("usd_per_1m")
        g, _ = col("gamma_per_1m")
        wr, wse = col("wrong")
        print("%-5s %-6g %-18s %6.4f+-%.4f %8.0f+-%-4.0f %12.0f %6.1f%%"
              % (key[0], key[1], key[2], q, qse, c, cse, g, 100 * wr))


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "llmrouterbench"
    parts = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    mults = tuple(float(m) for m in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1.25, 2.0, 4.0, 8.0)
    main(corpus, parts, seeds, mults)
