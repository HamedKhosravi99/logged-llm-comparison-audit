"""Brute-force check of Prop. 3.4 (when the routed-log decision is point-identified).

The proposition classifies each model as admissible under every C in the spread class,
under no C, or ambiguous, and states that the decision is point-identified iff some model
is always admissible and no ambiguous model is strictly cheaper than the cheapest such.

Both directions have a proof (App. A). This script is the corroboration the appendix
refers to: it enumerates the achievable admissible sets and winners over a lattice of C
and compares them against the closed form. Instances are drawn on a coarse lattice on
purpose, so exact boundary hits (D = delta - eta) and price ties occur often -- those are
the cases a continuous random search almost surely misses.

    python3 code/verify_prop_idset.py
"""

import itertools
import random

TOL = 1e-12


def admissible(mu, delta):
    top = max(mu)
    return [k for k in range(len(mu)) if top - mu[k] <= delta + TOL]


def winner(mu, price, delta):
    """Cheapest admissible model; index breaks price ties."""
    return min(admissible(mu, delta), key=lambda k: (price[k], k))


def achievable(muop, price, delta, eta, ngrid):
    """Winners and per-model admissibility over a lattice of the spread class.

    The decision depends on C only through within-type differences, so gridding each
    coordinate on [0, eta] and discarding the points of spread > eta covers the class.
    """
    n = len(muop)
    grid = [eta * i / (ngrid - 1) for i in range(ngrid)] if ngrid > 1 else [0.0]
    wins, seen = set(), [set() for _ in range(n)]
    for c in itertools.product(grid, repeat=n):
        if max(c) - min(c) > eta + TOL:
            continue
        mu = [muop[k] - c[k] for k in range(n)]
        adm = set(admissible(mu, delta))
        wins.add(winner(mu, price, delta))
        for k in range(n):
            seen[k].add(1 if k in adm else 0)
    return wins, seen


def closed_form(muop, price, delta, eta):
    """Prop. 3.4 as stated: returns (always, ambiguous, point_identified)."""
    n = len(muop)
    # gap to the best OTHER model; negative when k leads
    gap = [max(muop[:k] + muop[k + 1:]) - muop[k] for k in range(n)]
    always = [k for k in range(n) if gap[k] <= delta - eta + TOL]
    never = [k for k in range(n) if gap[k] > delta + eta + TOL]
    ambiguous = [k for k in range(n) if k not in always and k not in never]
    if not always:
        return always, ambiguous, False
    k0 = min(always, key=lambda k: (price[k], k))
    undercut = any((price[k], k) < (price[k0], k0) for k in ambiguous)
    return always, ambiguous, not undercut


def main(trials=30000, ngrid=9, seed=0):
    rng = random.Random(seed)
    bad_class = bad_id = 0
    for _ in range(trials):
        n = rng.choice([2, 2, 3, 3, 4])
        muop = [rng.choice([0, .1, .2, .3, .4, .5]) for _ in range(n)]
        price = rng.sample(range(1, 20), n)
        delta = rng.choice([0, .1, .2, .3])
        eta = rng.choice([0, .1, .2, .3])

        wins, seen = achievable(muop, price, delta, eta, ngrid)
        always, ambiguous, point_id = closed_form(muop, price, delta, eta)

        for k in range(n):
            pred_always, pred_never = k in always, k not in always and k not in ambiguous
            if pred_always != (seen[k] == {1}) or pred_never != (seen[k] == {0}):
                bad_class += 1
                break
        if point_id != (len(wins) == 1):
            bad_id += 1

    print(f"instances checked:              {trials} (lattice, ngrid={ngrid}, seed={seed})")
    print(f"classification mismatches:      {bad_class}")
    print(f"point-identification mismatches:{bad_id}")
    assert bad_class == 0, "Prop. 3.4 classification disagrees with brute force"
    assert bad_id == 0, "Prop. 3.4 point-identification claim disagrees with brute force"
    print("OK -- Prop. 3.4 agrees with brute force on every instance")


if __name__ == "__main__":
    main()
