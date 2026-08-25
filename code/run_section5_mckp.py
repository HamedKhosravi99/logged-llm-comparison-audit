"""Section 5's experiment with the budgeted allocation of Eq. (1) as the decision.

`run_case_mckp.py` established that the Eq. (1) decision and its acquisition rule can be run on
both corpora; this script runs them on section 5's design, so the committed numbers can be
replaced rather than compared against. What changes from `pipeline.run_partition`:

  decision   X*[V] under the serving budget Gamma, solved exactly by code/mckp.py, in place of
             `metrics.deployed_policy`, the unconstrained per-workload argmax that is Eq. (2) at
             delta = 0
  loss       traffic-weighted target-value loss of Eq. (5) against X*[mu_ref] at the same Gamma,
             in place of the unweighted mean over workloads
  acquisition the cell in the closest competing allocation (`run_case_mckp.mckp_margin_cell`), in
             place of the per-workload top-two margin, which does not transfer: under a shared
             budget a cell can decide the allocation without being near the top of its own column

The design is otherwise section 5's, so the arms are paired the same way: one partition fixes the
cohorts, the incumbent router and the evaluator constants; the selective log is drawn once and the
matched log is matched to its per-cell counts; every evaluator condition, budget and acquisition
seed re-uses those objects.

Arms
  trust        deploy the log, no evidence bought
  uniform      B readouts on random supported cells
  case         B readouts by the Eq. (1) acquisition rule above
  oracle_sel   cellwise-perfect assignment repair, mu^op: every cell's selection bias removed and
               the operational evaluator kept. Unachievable, and the floor a randomized audit
               works towards
  oracle_ref   the target table itself; zero loss by construction, and a check on the harness

Usage:
    python code/run_section5_mckp.py [corpus] [n_partitions] [acq_seeds] [mults]
    python code/run_section5_mckp.py llmrouterbench 30 6 1.25,2,4,8
"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import case_core as cc                                     # noqa: E402
from eval_validity import assignment as A                  # noqa: E402
from eval_validity import metrics as MT                    # noqa: E402
from eval_validity import oracles as OR                    # noqa: E402
from eval_validity.grid import build_base                  # noqa: E402
from eval_validity.pipeline import surfaces_for            # noqa: E402
from econ_core import cost_tables                          # noqa: E402
from mckp import solve_mckp, allocation_value, allocation_cost, cheapest_spend   # noqa: E402
from run_case_mckp import mckp_margin_cell                 # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
BUDGET_FRACS = (0.15, 0.25, 0.35)
ARMS = ("trust", "uniform", "case", "oracle_sel", "oracle_ref")


def deployed_surface(arm, surf, supp, p, w, gamma, B, seed, n_read):
    """The table an arm deploys on, and the readouts it spent to get there."""
    if arm == "trust":
        return surf["M"].copy(), 0
    if arm == "oracle_sel":
        return OR.oracle_selection_corrected_surface(surf), 0
    if arm == "oracle_ref":
        return OR.reference_outcome_oracle_surface(surf), 0

    M, cellq = surf["M"], surf["cellq"]
    K, J = M.shape
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
        cell = mckp_margin_cell(V, p, w, gamma, supp, R)
        if cell is None:
            break
        do_read(*cell)

    return cc.corrected(M, R, reads, cc.bias_rows(M, R, reads)), int(R.sum())


def run_partition(part_seed, corpus, mults, acq_seeds, budget_fracs, n_read, eps):
    """Every (Gamma, evaluator, assignment, budget, acquisition seed, arm) cell for one partition."""
    base = build_base(part_seed, n_clusters=8, corpus=corpus)
    K, J = base["K"], base["J"]
    cellsA1, n_kj, _ = A.make_selective_log(base, part_seed, eps=eps)
    cellsA0 = A.make_matched_common_population_log(base, part_seed, n_kj)
    supp = A.support(n_kj, base, n_read=n_read)
    tab = MT.scoring_tables(base, supp)
    ct = cost_tables(base)
    p, w = ct["c_score"], ct["p"]
    mu_ref = tab["mu"]

    if not (np.isfinite(mu_ref).all() and np.isfinite(p).all()):
        print("partition %d: missing cells, skipped" % part_seed, flush=True)
        return []

    cheapest = cheapest_spend(p, w, supp)
    cells = {"A0": cellsA0, "A1": cellsA1}
    rows = []

    for mult in mults:
        gamma = cheapest * mult
        v_ref, x_ref, feasible = solve_mckp(mu_ref, p, w, gamma, supp)
        if not feasible:
            print("partition %d: infeasible at %gx" % (part_seed, mult), flush=True)
            continue
        for cond in ("D0", "D1"):
            for aname in ("A0", "A1"):
                surf = surfaces_for(base, cells[aname], cond, 1.0, n_read)
                for frac in budget_fracs:
                    B = max(K, int(round(frac * int(supp.sum()))))
                    for s in range(acq_seeds):
                        seed = 10_000 * part_seed + 100 * int(frac * 100) + s
                        for arm in ARMS:
                            if arm in ("trust", "oracle_sel", "oracle_ref") and s > 0:
                                continue           # deterministic: no acquisition seed to average
                            V, used = deployed_surface(arm, surf, supp, p, w, gamma, B,
                                                       seed, n_read)
                            _, xhat, ok = solve_mckp(V, p, w, gamma, supp)
                            if not ok:
                                continue
                            rows.append(dict(
                                corpus=corpus, partition=int(part_seed), acq_seed=int(s),
                                mult=float(mult), cond=cond, assign=aname, budget_frac=float(frac),
                                B=int(B), arm=arm, reads=int(used),
                                loss=float(v_ref - allocation_value(mu_ref, w, xhat)),
                                wrong=int(sum(1 for j in range(J) if xhat[j] != x_ref[j])),
                                J=int(J), n_supported=int(supp.sum()),
                                spend=allocation_cost(p, w, xhat), gamma=float(gamma)))
    return rows


def main(corpus="llmrouterbench", n_partitions=30, acq_seeds=6, mults=(1.25, 2.0, 4.0, 8.0),
         n_read=30, eps=0.15, budget_fracs=BUDGET_FRACS):
    """Run every partition not already in the output file, checkpointing after each.

    A partition takes minutes on the eleven-model corpus, so the file is rewritten after every
    one and re-running the same command resumes where the last left off. A partition is complete
    or absent, never half-written, because the rewrite happens between partitions.
    """
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "section5_mckp_%s_eps%g_n%d.json" % (corpus, eps, n_read))
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {r["partition"] for r in rows
            if r["corpus"] == corpus and r["acq_seed"] == acq_seeds - 1}
    rows = [r for r in rows if r["partition"] in done]     # drop any partition run under other settings
    if done:
        print("resuming: %d partitions already in %s" % (len(done), out), flush=True)

    t0 = time.time()
    for part in range(n_partitions):
        if part in done:
            continue
        rows += run_partition(part, corpus, mults, acq_seeds, budget_fracs, n_read, eps)
        json.dump(rows, open(out, "w"), indent=1)
        print("partition %2d done  (%.0fs elapsed, %d rows, checkpointed)"
              % (part, time.time() - t0, len(rows)), flush=True)

    print("\nwrote %s  (%.0fs this run)" % (out, time.time() - t0))
    return out


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "llmrouterbench"
    parts = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    mults = tuple(float(m) for m in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1.25, 2.0, 4.0, 8.0)
    main(corpus, parts, seeds, mults)
