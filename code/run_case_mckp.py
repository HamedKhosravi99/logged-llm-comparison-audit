"""CASE under the budgeted allocation of Eq. (1), rather than the per-workload rule.

Section 5's numbers come from an acquisition loop whose decision is an unconstrained per-workload
argmax -- `metrics.deployed_policy` -- which is Eq. (2) at delta = 0, and whose acquisition scores
cells by a per-workload top-two gap (`case_core._smallest_margin_cell`). Neither survives the move
to Eq. (1), because the shared budget couples the workloads: the best model for a workload may be
unaffordable once the other workloads are paid for, and a cell can matter to the allocation without
being close to the top of its own column.

This script runs the same experiment with both replaced.

  decision      X*[V] under the serving budget Gamma, solved exactly by code/mckp.py
  acquisition   the cell involved in the closest competing allocation.  Solve X*[V]; then for each
                (k, j) with k not selected, solve again with model k forced onto workload j.  The
                smallest value gap identifies the allocation that most nearly displaces the
                incumbent, and the unread cell in that near-tie is bought.
  loss          traffic-weighted target-value loss of the deployed allocation against X*[mu_ref],
                both under the same Gamma

Arms: trust-the-log (no evidence spent), uniform (random cells), case-mckp (the rule above).

Section 5 is now produced by `run_section5_mckp.py`, which wraps this decision and this acquisition
rule in section 5's design (partitions crossed with acquisition seeds, the evidence-budget sweep and
the two oracle arms). This script stays as the minimal version of the same experiment.

Note on the committed `results/case_mckp_*.json`: they predate the fix that quotes Gamma against the
cheapest SUPPORTED allocation (`mckp.cheapest_spend`), so re-running this script will not reproduce
them. The paper cites the `section5_mckp_*` files instead.

Usage:
    python code/run_case_mckp.py routerbench [n_seeds] [budget_frac]
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
from mckp import solve_mckp, allocation_value, cheapest_spend   # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")


def _forced(supp, k, j):
    m = supp.copy()
    m[:, j] = False
    m[k, j] = True
    return m


def mckp_margin_cell(V, p, w, gamma, supp, R):
    """Unread cell in the closest competing allocation; None when nothing is left to buy."""
    v_hat, xhat, feasible = solve_mckp(V, p, w, gamma, supp)
    if not feasible:
        return None
    J = V.shape[1]
    best_gap, cell = np.inf, None
    for j in range(J):
        for k in np.where(supp[:, j])[0]:
            if k == xhat[j]:
                continue
            v_alt, _, ok = solve_mckp(V, p, w, gamma, _forced(supp, int(k), j))
            if not ok:
                continue
            gap = v_hat - v_alt
            for cand in ((int(k), j), (int(xhat[j]), j)):
                if not R[cand] and gap < best_gap:
                    best_gap, cell = gap, cand
    if cell is None:                                       # everything in the near-tie already read
        free = np.argwhere(supp & ~R)
        cell = tuple(int(x) for x in free[0]) if len(free) else None
    return cell


def run_arm(arm, surf, supp, p, w, gamma, B, seed, n_read):
    """Deployed surface for one arm, and the number of readouts used."""
    M, cellq = surf["M"], surf["cellq"]
    K, J = M.shape
    if arm == "trust":
        return M.copy(), 0

    rng = np.random.default_rng(seed)
    R = np.zeros((K, J), bool)
    reads = {}

    def do_read(k, j):
        reads[(k, j)] = cc.readout(cellq, k, j, rng, n_read)
        R[k, j] = True

    for k in range(K):                                     # one seed readout per model
        js = np.where(supp[k])[0]
        if len(js) and R.sum() < B:
            do_read(k, int(js[rng.integers(len(js))]))

    while R.sum() < B:
        if arm == "uniform":
            free = np.argwhere(supp & ~R)
            if not len(free):
                break
            k, j = free[rng.integers(len(free))]
            do_read(int(k), int(j))
        else:                                              # case-mckp
            V = cc.corrected(M, R, reads, cc.bias_rows(M, R, reads))
            cell = mckp_margin_cell(V, p, w, gamma, supp, R)
            if cell is None:
                break
            do_read(*cell)

    return cc.corrected(M, R, reads, cc.bias_rows(M, R, reads)), int(R.sum())


def main(corpus="routerbench", n_seeds=30, budget_frac=0.35, n_read=30, mult=2.0, eps=0.15):
    rows, t0 = [], time.time()
    for seed in range(n_seeds):
        base = build_base(seed, n_clusters=8, corpus=corpus)
        K, J = base["K"], base["J"]
        cellsA1, n_kj, _ = A.make_selective_log(base, seed, eps=eps)
        cellsA0 = A.make_matched_common_population_log(base, seed, n_kj)
        supp = A.support(n_kj, base, n_read=n_read)
        tab = MT.scoring_tables(base, supp)
        ct = cost_tables(base)
        p, w = ct["c_score"], ct["p"]
        mu_ref = tab["mu"]

        if not (np.isfinite(mu_ref).all() and np.isfinite(p).all()):
            print("seed %d: missing cells, skipped" % seed, flush=True)
            continue

        cheapest = cheapest_spend(p, w, supp)   # over SUPPORTED cells; see mckp.cheapest_spend
        gamma = cheapest * mult
        v_ref, x_ref, feasible = solve_mckp(mu_ref, p, w, gamma, supp)
        if not feasible:
            print("seed %d: no feasible allocation at Gamma, skipped" % seed, flush=True)
            continue

        B = max(K, int(round(budget_frac * int(supp.sum()))))
        for cond in ("D0", "D1"):
            for aname, cells in (("A0", cellsA0), ("A1", cellsA1)):
                surf = surfaces_for(base, cells, cond, 1.0, n_read)
                for arm in ("trust", "uniform", "case"):
                    V, used = run_arm(arm, surf, supp, p, w, gamma, B, seed, n_read)
                    _, xhat, ok = solve_mckp(V, p, w, gamma, supp)
                    loss = (v_ref - allocation_value(mu_ref, w, xhat)) if ok else float("nan")
                    rows.append(dict(corpus=corpus, seed=seed, cond=cond, assign=aname, arm=arm,
                                     budget_frac=budget_frac, mult=mult, reads=used,
                                     loss=float(loss),
                                     wrong=int(sum(1 for j in range(J) if xhat[j] != x_ref[j]))
                                     if ok else -1))
        print("seed %2d done  (%.0fs elapsed)" % (seed, time.time() - t0), flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "case_mckp_%s_frac%g_mult%g_eps%g_n%d.json"
                       % (corpus, budget_frac, mult, eps, n_read))
    json.dump(rows, open(out, "w"), indent=1)

    print("\n=== target-value loss under Eq. (1), %s, budget %.0f%% of pairs, Gamma = %.2gx cheapest ==="
          % (corpus, 100 * budget_frac, mult))
    print("%-6s %-6s %-10s %-10s %-10s" % ("eval", "assign", "trust", "uniform", "case"))
    for cond in ("D0", "D1"):
        for aname in ("A0", "A1"):
            cell = [r for r in rows if r["cond"] == cond and r["assign"] == aname]
            line = "%-6s %-6s" % (cond, aname)
            for arm in ("trust", "uniform", "case"):
                v = np.array([r["loss"] for r in cell if r["arm"] == arm])
                line += " %-10s" % ("%.4f+-%.4f" % (v.mean(), v.std(ddof=1) / max(1, np.sqrt(len(v))))
                                    if len(v) else "--")
            print(line)
    print("\nwrote %s  (%.0fs total)" % (out, time.time() - t0))


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "routerbench"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    main(corpus, seeds, frac)
