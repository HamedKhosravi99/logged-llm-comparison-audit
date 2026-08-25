"""Exact multiple-choice knapsack for Eq. (1), and the Prop. 3.4 settledness test.

Eq. (1) chooses one model per workload to maximise weighted quality subject to a shared serving
budget:

    maximise    sum_j w_j mu[x(j), j]
    subject to  sum_j w_j p[x(j), j] <= gamma,      one model per workload.

`source/budgeted_certificate` solves this by enumerating every allocation. That is fine at K=5,
J=8 (5^8 = 390,625 rows) and impossible at K=11, J=8 (11^8 = 214,358,881 rows, ~13.7 GB just to
hold the index array). This module solves the same problem exactly by a Pareto-frontier dynamic
program over workloads, which is exact in floating point -- no cost discretisation is involved --
and runs in milliseconds on both corpora.

The recurrence is the standard one for multiple-choice knapsack. Process workloads one at a time,
carrying the set of (cost, value) pairs reachable so far. Extend each pair by every candidate model
for the next workload, drop anything already over budget, then discard dominated pairs: a state is
dominated when some other state costs no more and is worth no less. Costs are non-negative, so
pruning over budget mid-way is safe -- a partial allocation that already exceeds gamma cannot be
rescued by adding workloads.

Self-test:  python code/mckp.py         # checks the DP against brute force on random instances
"""
import itertools

import numpy as np

TOL = 1e-12


def solve_mckp(mu, p, w, gamma, supported=None):
    """Exact optimum of Eq. (1).

    mu, p      : (K, J) quality and per-request price tables.
    w          : (J,) traffic shares.
    gamma      : serving budget, in the same units as sum_j w_j p[x(j), j].
    supported  : optional (K, J) boolean mask; False marks a model unavailable on that workload.
                 Cells that are NaN in mu or p are treated as unsupported.

    Returns (value, x, feasible) with x an int array of length J giving the chosen model per
    workload. If no allocation fits the budget, returns (-inf, None, False).
    """
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    K, J = mu.shape

    ok = np.isfinite(mu) & np.isfinite(p)
    if supported is not None:
        ok &= np.asarray(supported, dtype=bool)

    # Frontier over workloads processed so far: cost, value, and the choices that produced them.
    cost = np.zeros(1)
    value = np.zeros(1)
    choice = np.zeros((1, 0), dtype=np.int16)

    for j in range(J):
        ks = np.flatnonzero(ok[:, j])
        if ks.size == 0:
            return -np.inf, None, False               # workload with no supported model

        add_c = w[j] * p[ks, j]
        add_v = w[j] * mu[ks, j]

        # Cartesian extension of every surviving state by every candidate model.
        cost = (cost[:, None] + add_c[None, :]).ravel()
        value = (value[:, None] + add_v[None, :]).ravel()
        choice = np.concatenate(
            [np.repeat(choice, ks.size, axis=0),
             np.tile(ks, choice.shape[0]).astype(np.int16)[:, None]],
            axis=1,
        )

        keep = cost <= gamma + TOL                    # safe: remaining additions are non-negative
        if not keep.any():
            return -np.inf, None, False
        cost, value, choice = cost[keep], value[keep], choice[keep]

        # Pareto prune: sort by cost ascending, value descending; keep strict value records.
        order = np.lexsort((-value, cost))
        cost, value, choice = cost[order], value[order], choice[order]
        best_so_far = np.maximum.accumulate(
            np.concatenate(([-np.inf], value[:-1]))
        )
        keep = value > best_so_far + TOL
        cost, value, choice = cost[keep], value[keep], choice[keep]

    i = int(np.argmax(value))
    return float(value[i]), choice[i].astype(int), True


def cheapest_spend(p, w, supported=None):
    """Cost of the cheapest allocation, which is the natural unit for quoting Gamma.

    The minimum runs over SUPPORTED cells only. Taking it over all cells can put Gamma below the
    cheapest allocation the instance actually admits, which makes the program infeasible and
    silently tightens every budget quoted as a multiple of it.
    """
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    K, J = p.shape
    ok = np.isfinite(p)
    if supported is not None:
        ok &= np.asarray(supported, dtype=bool)
    total = 0.0
    for j in range(J):
        ks = np.flatnonzero(ok[:, j])
        if ks.size == 0:
            return float("inf")
        total += w[j] * float(p[ks, j].min())
    return float(total)


def allocation_value(mu, w, x):
    """Weighted quality of a given allocation."""
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(sum(w[j] * mu[x[j], j] for j in range(len(x))))


def allocation_cost(p, w, x):
    """Per-request serving cost of a given allocation."""
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(sum(w[j] * p[x[j], j] for j in range(len(x))))


def adverse_table(mu, e, xhat):
    """The least favourable admissible table of Prop. 3.4.

    Every cell the incumbent selects goes down to its lower bound; every rival goes up to its
    upper bound. `e` is the (K, J) array of per-cell half-widths.
    """
    mu = np.asarray(mu, dtype=float)
    e = np.asarray(e, dtype=float)
    check = mu + e
    for j, k in enumerate(xhat):
        check[:, j] = mu[:, j] + e[:, j]
        check[k, j] = mu[k, j] - e[k, j]
    return check


def is_settled(mu, p, w, gamma, e, xhat, supported=None):
    """Prop. 3.4, second form: solve at the adverse table and ask whether xhat is still optimal."""
    check = adverse_table(mu, e, xhat)
    best, _, feasible = solve_mckp(check, p, w, gamma, supported)
    if not feasible:
        return False
    return allocation_value(check, w, xhat) >= best - 1e-9


# ---------------------------------------------------------------------------------------------
# Self-test: the DP must agree with brute force wherever brute force is affordable.
# ---------------------------------------------------------------------------------------------
def _brute(mu, p, w, gamma):
    K, J = mu.shape
    best_v, best_x = -np.inf, None
    for x in itertools.product(range(K), repeat=J):
        c = sum(w[j] * p[x[j], j] for j in range(J))
        if c > gamma + TOL:
            continue
        v = sum(w[j] * mu[x[j], j] for j in range(J))
        if v > best_v:
            best_v, best_x = v, x
    return best_v, best_x


def _self_test(n_instances=200, seed=0):
    rng = np.random.default_rng(seed)
    mismatches, infeasible, checked = 0, 0, 0
    for t in range(n_instances):
        K = int(rng.integers(2, 6))
        J = int(rng.integers(2, 7))
        mu = rng.random((K, J))
        p = rng.random((K, J)) * 10.0
        w = rng.random(J)
        w /= w.sum()
        cheapest = sum(w[j] * p[:, j].min() for j in range(J))
        gamma = cheapest * float(rng.uniform(1.0, 3.0))

        v_dp, x_dp, feas = solve_mckp(mu, p, w, gamma)
        v_bf, x_bf = _brute(mu, p, w, gamma)

        if not feas:
            infeasible += 1
            if x_bf is not None:
                mismatches += 1
                print("instance %d: DP says infeasible, brute force found one" % t)
            continue

        checked += 1
        if abs(v_dp - v_bf) > 1e-9:
            mismatches += 1
            print("instance %d: value %.12f (DP) vs %.12f (brute)" % (t, v_dp, v_bf))
        if allocation_cost(p, w, x_dp) > gamma + 1e-9:
            mismatches += 1
            print("instance %d: DP returned an over-budget allocation" % t)

    print("%d instances, %d solved and compared, %d infeasible, %d mismatches"
          % (n_instances, checked, infeasible, mismatches))
    return mismatches == 0


if __name__ == "__main__":
    import sys

    ok = _self_test()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
