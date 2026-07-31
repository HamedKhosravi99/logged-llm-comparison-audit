"""Work Package A. Economic reanalysis of the two routing corpora.

No new inference, no changed experimental result. Everything below is a derived dollar quantity
computed from cost matrices the pipeline already loads and never spent.

Covers, in the plans' numbering:
  A2  eleven baselines
  A3  quality-constrained selection over the pre-specified delta and eta grids
  A4  audit cost in dollars, three cost categories kept separate
  A5  deployment horizons, gross and net savings, break-even, ROI
  A6  quality-loss valuation sensitivity
  A7  cost-aware CASE and its five ablations
  A8  the registered output fields
  A11 uncertainty on every savings figure, paired across partitions

Usage
  python run_econ.py --partitions 30 --corpus llmrouterbench
  python run_econ.py --partitions 20 --corpus routerbench
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from eval_validity import assignment as A            # noqa: E402
from eval_validity import metrics as MT              # noqa: E402
from eval_validity.grid import build_base            # noqa: E402
from eval_validity.pipeline import surfaces_for, _budget   # noqa: E402
import econ_core as E                                # noqa: E402

HORIZONS = (10_000, 100_000, 1_000_000)
DELTAS = (0.0, 0.005, 0.010, 0.020)
ETAS = (0.0, 0.01, 0.02, 0.05)
GAMMAS = (0.0, 0.001, 0.005, 0.010, 0.050)     # dollars per 0.01 quality lost, per request
PRIMARY_H = 100_000
PRIMARY_DELTA = 0.010


def one_partition(part, corpus, eps, n_read, acq_seeds, cond, budget_frac):
    base = build_base(part, n_clusters=8, corpus=corpus)
    K, J = base["K"], base["J"]
    cellsA1, n_kj, _ = A.make_selective_log(base, part, eps=eps)
    cellsA0 = A.make_matched_common_population_log(base, part, n_kj)
    supp = A.support(n_kj, base, n_read=n_read)
    tab = MT.scoring_tables(base, supp)
    mu = tab["mu"]
    pol_ref = MT.target_policy(tab, supp)
    ct = E.cost_tables(base, supp)
    B = _budget(budget_frac, supp, K)
    out = []

    for aname, cells in (("A1", cellsA1), ("A0", cellsA0)):
        surf = surfaces_for(base, cells, cond, 1.0, n_read)
        M, cellq = surf["M"], surf["cellq"]

        # ---- A2 baselines that need no audit --------------------------------------------
        fixed = {
            "target_best":      pol_ref,
            "trust":            MT.deployed_policy(M, supp),
            "cheapest":         E.cheapest_policy(ct, supp),
            "priciest":         E.priciest_policy(ct, supp),
            "single_best":      E.single_best_model_policy(mu, supp, ct),
            "oracle_complete":  pol_ref,
        }
        for d in DELTAS:
            fixed[f"qc_delta_{d}"] = E.quality_constrained_policy(mu, ct, supp, d)
        for e in ETAS:
            fixed[f"qc_eta_{e}"] = E.quality_constrained_policy(mu, ct, supp, e, relative=True)

        for name, pol in fixed.items():
            out.append(_row(part, corpus, aname, cond, name, pol, mu, ct, tab, supp, pol_ref,
                            0.0, 0, B))

        # ---- the decisive economic contrast -------------------------------------------
        # A team applies a quality tolerance to whatever surface it has. The DECISION uses the
        # estimate; the realised cost and quality are always scored on truth. Comparing the same
        # rule applied to the log, to the audited surface, and to the oracle isolates what an
        # accurate comparison is worth in dollars.
        for d in DELTAS:
            pol_log = E.quality_constrained_policy(M, ct, supp, d)
            out.append(_row(part, corpus, aname, cond, f"qcLOG_delta_{d}", pol_log, mu, ct, tab,
                            supp, pol_ref, 0.0, 0, B))

        # ---- A2 audit arms, averaged over acquisition seeds -----------------------------
        for s in range(acq_seeds):
            seed = 10_000 * part + 100 * int(budget_frac * 100) + s
            for name, mode in (("uniform", "uniform"), ("case", "case"),
                               ("rawmargin", "rawmargin"), ("epig_obs", "epig"),
                               ("epig_corr", "epig_deconf")):
                V, used, reads = E.alloc_traced(mode, M, cellq, supp, B, seed, n_read)
                pol = MT.deployed_policy(V, supp)
                usd = E.audit_cost(reads, ct, n_read)
                out.append(_row(part, corpus, aname, cond, name, pol, mu, ct, tab, supp, pol_ref,
                                usd, used, B, acq_seed=s))
                if name == "case":
                    for d in DELTAS:
                        pq = E.quality_constrained_policy(V, ct, supp, d)
                        out.append(_row(part, corpus, aname, cond, f"qcCASE_delta_{d}", pq, mu, ct,
                                        tab, supp, pol_ref, usd, used, B, acq_seed=s))

            # ---- A7 cost-aware CASE and ablations, same dollar budget as CASE -----------
            Vc, _, rc = E.alloc_traced("case", M, cellq, supp, B, seed, n_read)
            budget_usd = E.audit_cost(rc, ct, n_read)
            for name, kw in (("case_cost_aware", {}),
                             ("ca_no_traffic", dict(use_traffic=False)),
                             ("ca_traffic_only", dict(use_savings=False, use_cost_denom=False)),
                             ("ca_cost_denom_only", dict(use_traffic=False, use_savings=False)),
                             ("ca_savings_only", dict(use_traffic=False, use_cost_denom=False))):
                V, used, reads, spent = E.alloc_cost_aware(
                    M, cellq, supp, budget_usd, seed, n_read, ct, **kw)
                pol = MT.deployed_policy(V, supp)
                out.append(_row(part, corpus, aname, cond, name, pol, mu, ct, tab, supp, pol_ref,
                                spent, used, B, acq_seed=s))
    return out


def _row(part, corpus, aname, cond, method, pol, mu, ct, tab, supp, pol_ref, usd, used, B,
         acq_seed=-1):
    q = E.deploy_quality(pol, mu, ct)
    cpr = E.deploy_cost(pol, ct, 1.0)
    r = dict(partition=part, corpus=corpus, assignment=aname, evaluator=cond, method=method,
             acq_seed=acq_seed, B_cells=int(B), cells_read=int(used), audit_usd=float(usd),
             quality=q, target_value_loss=MT.reference_regret(pol, tab, supp, pol_ref),
             wrong_model_rate=MT.top1_error(pol, tab, supp, pol_ref),
             cost_per_request=cpr, deployment_cost_per_1k=1000.0 * cpr)
    for g in GAMMAS:
        v, loss = E.net_value(pol, mu, ct, g / 0.01)     # gamma quoted per 0.01 quality
        r[f"net_value_gamma_{g}"] = v
    r["quality_loss"] = loss
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=30)
    ap.add_argument("--acq-seeds", type=int, default=6)
    ap.add_argument("--corpus", default="llmrouterbench")
    ap.add_argument("--cond", default="D1")
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--n-read", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "econ.json"))
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for p in range(args.partitions):
        rows += one_partition(p, args.corpus, args.eps, args.n_read, args.acq_seeds,
                              args.cond, args.budget)
        print(f"  partition {p:2d}  rows {len(rows):5d}  [{time.time()-t0:.0f}s]", flush=True)

    summary = aggregate(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(spec=dict(corpus=args.corpus, partitions=args.partitions,
                             acq_seeds=args.acq_seeds, evaluator=args.cond,
                             budget_frac=args.budget, eps=args.eps, n_read=args.n_read,
                             horizons=list(HORIZONS), deltas=list(DELTAS), etas=list(ETAS),
                             gammas=list(GAMMAS), primary_horizon=PRIMARY_H,
                             primary_delta=PRIMARY_DELTA,
                             generated_by="code/run_econ.py"),
                   rows=rows, summary=summary),
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    report(summary)


def aggregate(rows):
    """Per-partition means first, then paired contrasts across partitions."""
    import collections
    ser = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        ser[(r["assignment"], r["method"])][r["partition"]].append(r)

    def arr(key, field):
        d = ser[key]
        return np.array([np.mean([x[field] for x in d[p]]) for p in sorted(d)])

    out = {}
    for key in ser:
        a, m = key
        out[f"{a}.{m}"] = {f: _ms(arr(key, f)) for f in
                           ("quality", "cost_per_request", "target_value_loss",
                            "wrong_model_rate", "audit_usd", "cells_read",
                            "deployment_cost_per_1k")}
    # paired savings against trusting the log, at each horizon
    for a in ("A1", "A0"):
        base = arr((a, "trust"), "cost_per_request")
        for m in [k[1] for k in ser if k[0] == a and k[1] != "trust"]:
            cost = arr((a, m), "cost_per_request")
            usd = arr((a, m), "audit_usd")
            per = base - cost
            for H in HORIZONS:
                net = H * per - usd
                out[f"{a}.{m}.net_savings_at_{H}"] = _ms(net)
                out[f"{a}.{m}.gross_savings_at_{H}"] = _ms(H * per)
            out[f"{a}.{m}.saving_per_request"] = _ms(per)
            with np.errstate(divide="ignore", invalid="ignore"):
                be = np.where(per > 0, usd / np.where(per > 0, per, np.nan), np.nan)
                roi = np.where(usd > 0, (PRIMARY_H * per - usd) / np.where(usd > 0, usd, np.nan), np.nan)
            out[f"{a}.{m}.break_even_requests"] = _ms(be, allow_nan=True)
            out[f"{a}.{m}.roi_at_{PRIMARY_H}"] = _ms(roi, allow_nan=True)
    return out


def _ms(x, allow_nan=False):
    x = np.asarray(x, float)
    if allow_nan:
        x = x[np.isfinite(x)]
        if not x.size:
            return dict(mean=None, se=None, n=0)
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
                n=int(len(x)))


def report(s):
    print("\n" + "=" * 92)
    print("A1--D1, deployment economics per 1M requests, paired over partitions")
    print("=" * 92)
    print(f"  {'arm':22}{'quality':>10}{'$/1M req':>12}{'audit $':>10}{'cells':>7}"
          f"{'net @1M req':>14}")
    order = ["trust", "uniform", "case", "case_cost_aware", "epig_corr", "cheapest",
             "qc_delta_0.01", "qc_delta_0.02", "target_best", "priciest"]
    for m in order:
        k = f"A1.{m}"
        if k not in s:
            continue
        v = s[k]
        net = s.get(f"A1.{m}.net_savings_at_1000000", {}).get("mean")
        print(f"  {m:22}{v['quality']['mean']:10.4f}"
              f"{v['cost_per_request']['mean']*1e6:12,.0f}"
              f"{v['audit_usd']['mean']:10.2f}{v['cells_read']['mean']:7.0f}"
              + (f"{net:14,.0f}" if net is not None else f"{'-':>14}"))


if __name__ == "__main__":
    main()
