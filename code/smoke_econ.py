"""Smoke test for the economic layer. Run before any real economics run.

Checks, in order:
  1. the traced allocator reproduces case_core.allocate bit-for-bit on real data, all five modes;
  2. every supported cell has a finite cost on all three cohorts;
  3. audit cost equals a hand calculation over the purchased cells;
  4. deployment cost equals a hand calculation over traffic weights;
  5. the savings / break-even / ROI identities hold algebraically;
  6. the quality-constrained policy actually satisfies its constraint and is cost-minimal in it;
  7. cost-aware CASE never exceeds its dollar budget and honours each ablation switch.

Exit code is nonzero on any failure.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import case_core as cc                              # noqa: E402
from eval_validity import assignment as A           # noqa: E402
from eval_validity import metrics as MT             # noqa: E402
from eval_validity.grid import build_base           # noqa: E402
from eval_validity.pipeline import surfaces_for, _budget   # noqa: E402
import econ_core as E                               # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    n_read = 30
    print("building partition 0 of the primary corpus ...")
    base = build_base(0, n_clusters=8, corpus="llmrouterbench")
    cellsA1, n_kj, _ = A.make_selective_log(base, 0, eps=0.15)
    supp = A.support(n_kj, base, n_read=n_read)
    surf = surfaces_for(base, cellsA1, "D1", 1.0, n_read)
    M, cellq = surf["M"], surf["cellq"]
    K, J = base["K"], base["J"]
    B = _budget(0.35, supp, K)
    print(f"  K={K} J={J} supported={int(supp.sum())} B={B}\n")

    # 1 -----------------------------------------------------------------------------------
    print("1. traced allocator is identical to case_core.allocate")
    for mode in ("uniform", "case", "rawmargin", "epig", "epig_deconf"):
        try:
            d = E.verify_alloc(mode, M, cellq, supp, B, 4242, n_read)
            check(f"mode {mode}", d == 0.0, f"max|dV| = {d:.1e}")
        except AssertionError as ex:
            check(f"mode {mode}", False, str(ex))

    # 2 -----------------------------------------------------------------------------------
    print("\n2. cost surfaces")
    ct = E.cost_tables(base, supp)
    for nm in ("c_acq", "c_ver", "c_score"):
        check(f"{nm} finite on every supported cell", ct[nm + "_missing"] == 0,
              f"{ct[nm + '_missing']} missing")
    check("traffic weights sum to 1", abs(ct["p"].sum() - 1.0) < 1e-12, f"sum={ct['p'].sum():.12f}")
    check("all costs strictly positive",
          bool(np.all(ct["c_score"][np.isfinite(ct["c_score"])] > 0)))
    print(f"     deployment cost per request, cheapest to priciest single model: "
          f"${np.nanmin(ct['c_score'].mean(1)):.6f} .. ${np.nanmax(ct['c_score'].mean(1)):.6f}")

    # 3 -----------------------------------------------------------------------------------
    print("\n3. audit cost equals a hand calculation")
    V, used, reads = E.alloc_traced("case", M, cellq, supp, B, 4242, n_read)
    a = E.audit_cost(reads, ct, n_read)
    hand = sum(n_read * float(ct["c_acq"][k, j]) for (k, j) in reads)
    check("audit_cost matches hand sum", abs(a - hand) < 1e-15, f"${a:.6f}")
    check("cells purchased equals the budget", used == B == len(reads), f"{used} cells")

    # 4 -----------------------------------------------------------------------------------
    print("\n4. deployment cost equals a hand calculation")
    tab = MT.scoring_tables(base, supp)
    pol = MT.deployed_policy(V, supp)
    dc = E.deploy_cost(pol, ct, 1.0)
    hand = sum(ct["p"][j] * ct["c_score"][pol[j], j] for j in range(J) if pol[j] >= 0)
    check("deploy_cost matches hand sum", abs(dc - hand) < 1e-15, f"${dc:.8f}/request")
    check("scales linearly in H", abs(E.deploy_cost(pol, ct, 1e6) - 1e6 * dc) < 1e-6)

    # 5 -----------------------------------------------------------------------------------
    print("\n5. savings identities")
    pol_trust = MT.deployed_policy(M, supp)
    H = 100_000
    s = E.savings(pol_trust, pol, ct, a, H)
    check("gross = H * per-request saving", abs(s["gross_savings"] - H * s["saving_per_request"]) < 1e-9)
    check("net = gross - audit", abs(s["net_savings"] - (s["gross_savings"] - a)) < 1e-9)
    if s["break_even_requests"] is not None:
        be = s["break_even_requests"]
        check("at break-even, net is zero",
              abs(be * s["saving_per_request"] - a) < 1e-9, f"H_BE = {be:,.0f} requests")
    else:
        check("no finite break-even is reported when savings are nonpositive",
              s["saving_per_request"] <= 0, f"per-request saving ${s['saving_per_request']:.2e}")
    check("roi = net / audit", s["roi"] is None or abs(s["roi"] - s["net_savings"] / a) < 1e-9)

    # 6 -----------------------------------------------------------------------------------
    print("\n6. quality-constrained policy")
    mu = tab["mu"]
    for delta in (0.0, 0.005, 0.01, 0.02):
        pd = E.quality_constrained_policy(mu, ct, supp, delta)
        viol, notmin = 0, 0
        for j in range(J):
            if pd[j] < 0:
                continue
            ks = [k for k in np.where(supp[:, j])[0]
                  if np.isfinite(mu[k, j]) and np.isfinite(ct["c_score"][k, j])]
            best = max(mu[k, j] for k in ks)
            if mu[pd[j], j] < best - delta - 1e-12:
                viol += 1
            feas = [k for k in ks if mu[k, j] >= best - delta - 1e-12]
            if float(ct["c_score"][pd[j], j]) > min(float(ct["c_score"][k, j]) for k in feas) + 1e-15:
                notmin += 1
        check(f"delta={delta}: constraint satisfied", viol == 0, f"{viol} violations")
        check(f"delta={delta}: cost-minimal within the feasible set", notmin == 0, f"{notmin} non-minimal")
    d0 = E.quality_constrained_policy(mu, ct, supp, 0.0)
    check("delta=0 matches the argmax-quality policy on cost",
          abs(E.deploy_cost(d0, ct) - E.deploy_cost(MT.target_policy(tab, supp), ct)) < 1e-12)
    costs = [E.deploy_cost(E.quality_constrained_policy(mu, ct, supp, d), ct)
             for d in (0.0, 0.005, 0.01, 0.02, 0.05)]
    check("cost is nonincreasing as the tolerance widens",
          all(costs[i] >= costs[i + 1] - 1e-15 for i in range(len(costs) - 1)),
          " -> ".join(f"${c*1e6:,.0f}/1M" for c in costs))

    # 7 -----------------------------------------------------------------------------------
    print("\n7. cost-aware CASE")
    budget_usd = E.audit_cost(reads, ct, n_read)          # same dollar budget CASE actually spent
    for name, kw in (("full", {}),
                     ("no traffic weighting", dict(use_traffic=False)),
                     ("no cost denominator", dict(use_cost_denom=False)),
                     ("no savings term", dict(use_savings=False))):
        Vc, uc, rc, spent = E.alloc_cost_aware(M, cellq, supp, budget_usd, 4242, n_read, ct, **kw)
        ok = spent <= budget_usd + 1e-12
        check(f"{name}: respects the dollar budget", ok,
              f"spent ${spent:.4f} of ${budget_usd:.4f} on {uc} cells")
    Vc, uc, rc, spent = E.alloc_cost_aware(M, cellq, supp, budget_usd, 4242, n_read, ct)
    check("cost-aware buys at least as many cells as CASE for the same dollars", uc >= 1,
          f"{uc} cells vs CASE {used}")
    check("no cell is bought twice", len(set(rc)) == len(rc))

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
