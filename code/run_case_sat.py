"""CASE under the quality-constrained rule of Eq. (2), the relegated marginal-threshold rule.

The companion of `run_case_mckp.py`. Same corpora, same partitions, same assignment and evaluator
conditions, same evidence budget and readout size -- only the decision changes:

  Eq. (1)  X*[mu]        one model per workload maximizing traffic-weighted quality s.t. spend <= Gamma
  Eq. (2)  pi_delta[mu]  per workload, the CHEAPEST model within delta of that workload's best

Section 5's committed numbers were produced by `metrics.deployed_policy`, an unconstrained
per-workload argmax, which is Eq. (2) at delta = 0 without the price tie-break. Nothing in the repo
has ever run the acquisition loop at delta > 0. This script does, so the two rules can be compared
on the same partitions instead of across experiments.

Arms
  trust      deploy the log, no evidence bought
  uniform    B readouts on random supported cells
  case       B readouts by the section 5 rule: smallest corrected TOP-TWO gap (delta-blind)
  case_sat   B readouts by the delta-BOUNDARY margin (below), the acquisition Eq. (2) implies

Under Eq. (2) the decision flips when a model crosses the admissibility line
t_j = max_k V_kj - delta, so the contested cells are the ones near that line, not the ones near the
column top. `sat_margin_cell` scores an unread cell by its distance to t_j when it could move the
decision -- a model cheaper than the incumbent crossing up, the incumbent itself falling off -- and
scores the column leader by how far it must fall to admit a cheaper model, since the leader sets t_j.
Cells more expensive than the incumbent cannot change a cost-minimizing choice and are skipped.

Outcomes, all traffic weighted by the scoring-cohort share w_j, against pi_delta[mu_ref]
  loss        target-value gap; can be negative, since Eq. (2) does not maximize quality
  violation   share of workloads whose deployed model is NOT within delta on mu_ref, the rule's
              own failure mode: the quality standard the operator asked for was not met
  spend_rel   deployed bill / reference bill on the scoring cost surface

Usage:
    python code/run_case_sat.py [corpus] [n_seeds] [budget_frac]
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

RESULTS = os.path.join(os.path.dirname(HERE), "results")
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.05)
TOL = 1e-12


def sat_policy(V, price, supp, delta):
    """pi_delta[V](j): cheapest model within delta of the column best; ties to higher V, then index."""
    K, J = V.shape
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if not len(ks):
            continue
        adm = ks[V[ks, j] >= V[ks, j].max() - delta - TOL]
        pol[j] = int(min(adm, key=lambda k: (price[k, j], -V[k, j], k)))
    return pol


def sat_margin_cell(V, price, supp, R, delta):
    """Unread supported cell closest to changing pi_delta; None when nothing relevant is left."""
    K, J = V.shape
    best, cell = np.inf, None
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if len(ks) < 2:
            continue
        t = V[ks, j].max() - delta
        leader = int(ks[int(np.argmax(V[ks, j]))])
        inc = int(min(ks[V[ks, j] >= t - TOL], key=lambda k: (price[k, j], -V[k, j], k)))
        cheaper = [int(k) for k in ks if price[k, j] < price[inc, j]]
        drop = min([t - V[k, j] for k in cheaper], default=np.inf)   # leader's fall to admit a cheaper model
        for k in ks:
            k = int(k)
            if R[k, j]:
                continue
            if k == leader:
                score = drop
            elif k == inc or price[k, j] < price[inc, j]:
                score = abs(V[k, j] - t)
            else:
                continue                                   # dearer than the incumbent: cannot move the choice
            if score < best:
                best, cell = score, (k, j)
    if cell is None:                                       # every decision-relevant cell already read
        free = np.argwhere(supp & ~R)
        cell = tuple(int(x) for x in free[0]) if len(free) else None
    return cell


def run_arm(arm, surf, supp, price, B, seed, n_read, delta):
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
            continue
        V = cc.corrected(M, R, reads, cc.bias_rows(M, R, reads))
        cell = (cc._smallest_margin_cell(V, supp, R) if arm == "case"
                else sat_margin_cell(V, price, supp, R, delta))
        if cell is None:
            break
        do_read(*cell)

    return cc.corrected(M, R, reads, cc.bias_rows(M, R, reads)), int(R.sum())


def score(pol, pol_ref, mu_ref, price, w, supp, delta):
    """Traffic-weighted target-value loss, tolerance-violation share, relative spend, disagreements."""
    J = len(pol)
    loss = float(sum(w[j] * (mu_ref[pol_ref[j], j] - mu_ref[pol[j], j]) for j in range(J)))
    unw = float(np.mean([mu_ref[pol_ref[j], j] - mu_ref[pol[j], j] for j in range(J)]))
    viol, seen = 0.0, 0
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if not len(ks):
            continue
        seen += 1
        viol += float(mu_ref[pol[j], j] < mu_ref[ks, j].max() - delta - TOL)
    bill = float(sum(w[j] * price[pol[j], j] for j in range(J)))
    bill_ref = float(sum(w[j] * price[pol_ref[j], j] for j in range(J)))
    return dict(loss=loss,
                loss_unw=unw,                              # unweighted, comparable to reference_regret
                violation=viol / max(1, seen),
                spend_rel=bill / bill_ref if bill_ref > 0 else float("nan"),
                wrong=int(sum(1 for j in range(J) if pol[j] != pol_ref[j])))


def main(corpus="llmrouterbench", n_seeds=30, budget_frac=0.35, n_read=30, eps=0.15):
    rows, t0 = [], time.time()
    for seed in range(n_seeds):
        base = build_base(seed, n_clusters=8, corpus=corpus)
        K, J = base["K"], base["J"]
        cellsA1, n_kj, _ = A.make_selective_log(base, seed, eps=eps)
        cellsA0 = A.make_matched_common_population_log(base, seed, n_kj)
        supp = A.support(n_kj, base, n_read=n_read)
        tab = MT.scoring_tables(base, supp)
        ct = cost_tables(base)
        price, w = ct["c_score"], ct["p"]
        mu_ref = tab["mu"]

        if not (np.isfinite(mu_ref).all() and np.isfinite(price).all()):
            print("seed %d: missing cells, skipped" % seed, flush=True)
            continue

        B = max(K, int(round(budget_frac * int(supp.sum()))))
        for cond in ("D0", "D1"):
            for aname, cells in (("A0", cellsA0), ("A1", cellsA1)):
                surf = surfaces_for(base, cells, cond, 1.0, n_read)
                for delta in DELTAS:
                    pol_ref = sat_policy(mu_ref, price, supp, delta)
                    for arm in ("trust", "uniform", "case", "case_sat"):
                        V, used = run_arm(arm, surf, supp, price, B, seed, n_read, delta)
                        pol = sat_policy(V, price, supp, delta)
                        rows.append(dict(corpus=corpus, seed=seed, cond=cond, assign=aname,
                                         delta=delta, arm=arm, budget_frac=budget_frac,
                                         reads=used,
                                         **score(pol, pol_ref, mu_ref, price, w, supp, delta)))
        print("seed %2d done  (%.0fs elapsed)" % (seed, time.time() - t0), flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "case_sat_%s_frac%g_eps%g_n%d.json"
                       % (corpus, budget_frac, eps, n_read))
    json.dump(rows, open(out, "w"), indent=1)
    report(rows, corpus, budget_frac)
    print("\nwrote %s  (%.0fs total)" % (out, time.time() - t0))


def report(rows, corpus, budget_frac):
    arms = ("trust", "uniform", "case", "case_sat")
    for metric in ("loss", "violation"):
        print("\n=== %s under Eq. (2), %s, budget %.0f%% of pairs ==="
              % (metric, corpus, 100 * budget_frac))
        print("%-6s %-4s %-6s %-9s %-9s %-9s %-9s   %s"
              % ("delta", "eval", "assign", *arms, "case-trust paired [95% CI]"))
        for delta in DELTAS:
            for cond in ("D0", "D1"):
                for aname in ("A1", "A0"):
                    sub = [r for r in rows if r["delta"] == delta and r["cond"] == cond
                           and r["assign"] == aname]
                    by = {}
                    for r in sub:
                        by.setdefault(r["seed"], {})[r["arm"]] = r[metric]
                    ks = [s for s in by if len(by[s]) == len(arms)]
                    if not ks:
                        continue
                    line = "%-6.3f %-4s %-6s" % (delta, cond, aname)
                    for arm in arms:
                        line += " %-9.4f" % np.mean([by[s][arm] for s in ks])
                    d = np.array([by[s]["case"] - by[s]["trust"] for s in ks])
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                    print(line + "   %+.4f [%+.4f,%+.4f] n=%d"
                          % (d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se, len(d)))


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "llmrouterbench"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    main(corpus, seeds, frac)
