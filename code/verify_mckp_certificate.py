"""Brute-force check of the robust-optimality certificate for budgeted model allocation.

The allocation problem is a multiple-choice knapsack: choose one model k per workload j to
maximise sum_j w_j mu[k][j] subject to sum_j w_j p[k][j] <= B.  The coefficients mu are not
observed; the evidence gives a nominal table muhat and a per-cell bound e, so the admissible
tables are the box {mu : |mu - muhat| <= e}.

CLAIM (the certificate).  Let xhat maximise the nominal objective among budget-feasible
allocations.  Build one modified table:

    incumbent cell  (j, xhat[j]) :  muhat - e     (pushed down)
    every other cell (j, k)      :  muhat + e     (pushed up)

Then xhat is optimal for EVERY admissible table if and only if xhat is still optimal for that
single modified table.  Prices are known, so budget feasibility is unaffected.

Why it should hold: for a rival x, the workloads where x agrees with xhat cancel from the
difference, and the worst case sets each disagreeing coordinate to its own extreme.  The modified
table realises all of those extremes simultaneously, and it does so consistently for every rival
at once, which is what makes one solve enough.

This script does not assume that argument.  It computes robust optimality directly, rival by
rival, by minimising the gap coordinate-by-coordinate, and compares against the certificate.  It
also samples admissible tables at random as an independent check.

    python3 code/verify_mckp_certificate.py
"""

import itertools
import random

TOL = 1e-9


def feasible(J, K, w, p, B):
    """All allocations (one model per workload) that fit the budget."""
    out = []
    for x in itertools.product(range(K), repeat=J):
        if sum(w[j] * p[x[j]][j] for j in range(J)) <= B + TOL:
            out.append(x)
    return out


def value(x, mu, w, J):
    return sum(w[j] * mu[x[j]][j] for j in range(J))


def nominal_best(feas, muhat, w, J):
    return max(feas, key=lambda x: (value(x, muhat, w, J), tuple(-t for t in x)))


def robust_by_rival(xhat, feas, muhat, e, w, J):
    """Ground truth, computed independently of the certificate.

    For each rival x, minimise value(xhat) - value(x) over the box.  Coordinates are independent
    and the objective is linear, so the minimum sets the incumbent's disagreeing cells low and the
    rival's disagreeing cells high.  Agreeing workloads contribute exactly zero.
    """
    for x in feas:
        if x == xhat:
            continue
        gap = 0.0
        for j in range(J):
            if x[j] == xhat[j]:
                continue
            lo = muhat[xhat[j]][j] - e[xhat[j]][j]
            hi = muhat[x[j]][j] + e[x[j]][j]
            gap += w[j] * (lo - hi)
        if gap < -TOL:
            return False
    return True


def certificate(xhat, feas, muhat, e, w, J, K):
    """The one-solve test: re-solve with the incumbent pessimistic and rivals optimistic."""
    mod = [[muhat[k][j] + e[k][j] for j in range(J)] for k in range(K)]
    for j in range(J):
        mod[xhat[j]][j] = muhat[xhat[j]][j] - e[xhat[j]][j]
    best = max(value(x, mod, w, J) for x in feas)
    return value(xhat, mod, w, J) >= best - TOL


def sampled_counterexample(xhat, feas, muhat, e, w, J, K, rng, draws=200):
    """Independent check: draw admissible tables and look for one where xhat is beaten."""
    for _ in range(draws):
        mu = [[muhat[k][j] + rng.uniform(-e[k][j], e[k][j]) for j in range(J)] for k in range(K)]
        if max(value(x, mu, w, J) for x in feas) > value(xhat, mu, w, J) + 1e-7:
            return True
    return False


def regime(name, trials, seed, sizes, continuous, emax, budgets):
    """One sweep. Returns a record; asserts nothing so the caller can report all of them."""
    rng = random.Random(seed)
    disagree = missed = skipped = certified = 0
    for _ in range(trials):
        J, K = rng.choice(sizes), rng.choice(sizes)
        w = [rng.choice([1, 2, 5]) for _ in range(J)]
        p = [[rng.choice([1, 2, 4, 8]) for _ in range(J)] for _ in range(K)]
        if continuous:
            muhat = [[rng.uniform(0, 1) for _ in range(J)] for _ in range(K)]
            e = [[rng.uniform(0, emax) for _ in range(J)] for _ in range(K)]
        else:
            muhat = [[rng.choice([0, .1, .2, .3, .4, .5]) for _ in range(J)] for _ in range(K)]
            e = [[rng.choice([0, .05, .1, emax]) for _ in range(J)] for _ in range(K)]
        feas = feasible(J, K, w, p, rng.choice(budgets))
        if not feas:
            skipped += 1
            continue
        xhat = nominal_best(feas, muhat, w, J)
        truth = robust_by_rival(xhat, feas, muhat, e, w, J)
        cert = certificate(xhat, feas, muhat, e, w, J, K)
        if truth != cert:
            disagree += 1
        if cert:
            certified += 1
            if sampled_counterexample(xhat, feas, muhat, e, w, J, K, rng, draws=300):
                missed += 1
    return {"regime": name, "seed": seed, "instances": trials - skipped,
            "skipped_infeasible": skipped, "certified": certified,
            "certificate_vs_truth_mismatches": disagree, "sampled_violations": missed}


def main(out="results/mckp_certificate.json"):
    import json
    import os

    regimes = [
        regime("lattice coefficients, J,K<=4", 20000, 0, [2, 3, 4], False, 0.1, [8, 16, 32, 64]),
        regime("continuous coefficients, J,K<=4", 8000, 1, [2, 3, 4], True, 0.15, [8, 16, 32, 64]),
        regime("continuous coefficients, J,K<=5", 3000, 2, [3, 4, 5], True, 0.25, [8, 16, 32, 64, 128]),
        regime("wide error bounds", 8000, 3, [2, 3, 4], False, 0.3, [8, 16, 32, 64]),
    ]
    # degenerate check: with exact coefficients every nominal optimum must certify
    rng = random.Random(9)
    n = c = 0
    for _ in range(4000):
        J, K = rng.choice([2, 3, 4]), rng.choice([2, 3, 4])
        w = [rng.choice([1, 2, 5]) for _ in range(J)]
        p = [[rng.choice([1, 2, 4, 8]) for _ in range(J)] for _ in range(K)]
        muhat = [[rng.uniform(0, 1) for _ in range(J)] for _ in range(K)]
        e = [[0.0] * J for _ in range(K)]
        feas = feasible(J, K, w, p, rng.choice([8, 16, 32, 64]))
        if not feas:
            continue
        n += 1
        c += certificate(nominal_best(feas, muhat, w, J), feas, muhat, e, w, J, K)
    regimes.append({"regime": "exact coefficients (e=0), must always certify", "seed": 9,
                    "instances": n, "skipped_infeasible": 0, "certified": c,
                    "certificate_vs_truth_mismatches": 0 if c == n else n - c,
                    "sampled_violations": 0})

    tot = sum(r["instances"] for r in regimes)
    bad = sum(r["certificate_vs_truth_mismatches"] + r["sampled_violations"] for r in regimes)
    for r in regimes:
        print(f"  {r['regime']:46s} n={r['instances']:6d} certified={r['certified']:6d} "
              f"mismatch={r['certificate_vs_truth_mismatches']} violations={r['sampled_violations']}")
    print(f"  {'TOTAL':46s} n={tot:6d} failures={bad}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"claim": "xhat is optimal for every admissible coefficient table iff it is optimal "
                        "for the single table that sets the incumbent cell to muhat-e and every "
                        "other cell to muhat+e",
               "total_instances": tot, "total_failures": bad, "regimes": regimes},
              open(out, "w"), indent=1)
    print(f"wrote {out}")
    assert bad == 0, "the one-solve certificate is not exact"
    print("OK -- the certificate agreed with brute force on every instance")


if __name__ == "__main__":
    main()
