"""Replicate the whole design on a second released routing matrix.

Every number in the main study comes from one corpus, so its intervals are properties of that
corpus. This runs the identical pipeline -- same cohorts, same context construction, same cost-cascade
incumbent, same pre-specified evaluator distortion, same methods, same budgets -- on RouterBench
(36,497 prompts x 11 models, five benchmark families), which shares no prompts, no models and no
scoring convention with the primary corpus. A conclusion that holds on both is a replicated finding;
one that does not is a property of the first matrix.

    python run_tae_replication.py [--partitions 20] [--acq-seeds 6] [--corpus routerbench]
"""
import argparse
import collections
import json
import os
import time

import numpy as np

from eval_validity import validation
from eval_validity.pipeline import run_partition

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "results", "tae_replication.json")


def _round(o, nd=6):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, np.generic):
        return _round(o.item(), nd)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=20)
    ap.add_argument("--acq-seeds", type=int, default=6)
    ap.add_argument("--corpus", default="routerbench")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--n-read", type=int, default=30)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    rows, gate_rows, metas = [], [], []
    for p in range(args.partitions):
        r, g, m = run_partition(p, acq_seeds=args.acq_seeds, eps=args.eps, n_read=args.n_read,
                                corpus=args.corpus, extra_methods=("rawmargin", "epig_obs"),
                                factorial_conds=("E0", "E1", "E2"))
        rows += r
        gate_rows += g
        metas.append(m)
        d = m["diagnostics"]
        print(f"  partition {p:2d}  K={m['K']} J={m['J']}  supported {m['n_supported']}/{m['K']*m['J']}"
              f"  lambda {m['lam']:.4f}  log-misrank(A1) {d['log_misrank_A1']:.2f}"
              f"  target reversal {d['reference_ranking_reversal_rate']:.2f}  [{time.time()-t0:.0f}s]",
              flush=True)

    out = dict(spec=dict(corpus=args.corpus, partitions=args.partitions, acq_seeds=args.acq_seeds,
                         eps=args.eps, n_read=args.n_read, budgets=[0.15, 0.25, 0.35],
                         generated_by="code/run_tae_replication.py"),
               rows=rows, gate_rows=gate_rows, meta=metas)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(_round(out), open(args.out, "w"), separators=(",", ":"))

    # ---- headline comparison against the primary corpus -------------------------------------
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["budget"] == 0.35:
            agg[(r["assignment"], r["evaluator"], r["method"])][r["partition"]].append(r)
    def st(key, f):
        x = agg[key]
        v = np.array([np.mean([q[f] for q in x[p]]) for p in sorted(x)])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    print(f"\n{args.corpus}: reference regret at a 35% budget\n")
    ms = ["trust", "uniform", "epig_obs", "case", "epig_corr", "oracle_sel", "oracle_ref"]
    print(f"{'cell':<8}" + "".join(f"{m:>13}" for m in ms))
    for A in ("A0", "A1"):
        for E in ("E0", "E1"):
            print(f"{A}-{E:<5}" + "".join(f"{st((A, E, m), 'reference_regret')[0]:13.4f}" for m in ms))
    print(f"\n{'cell':<8}" + "".join(f"{m:>13}" for m in ms) + "   (top-1 identification error)")
    for A in ("A0", "A1"):
        for E in ("E0", "E1"):
            print(f"{A}-{E:<5}" + "".join(f"{st((A, E, m), 'top1_error')[0]:13.3f}" for m in ms))
    for A, E in (("A1", "E0"), ("A1", "E1"), ("A0", "E0")):
        a = np.array([np.mean([q["reference_regret"] for q in agg[(A, E, "case")][p]])
                      for p in sorted(agg[(A, E, "case")])])
        b = np.array([np.mean([q["reference_regret"] for q in agg[(A, E, "trust")][p]])
                      for p in sorted(agg[(A, E, "trust")])])
        c = np.array([np.mean([q["reference_regret"] for q in agg[(A, E, "epig_corr")][p]])
                      for p in sorted(agg[(A, E, "epig_corr")])])
        d = a - b
        e = a - c
        print(f"\n{A}-{E}: CASE - trust {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
              f"(better on {np.mean(d < 0):.0%} of partitions) | "
              f"CASE - R-EPIG_corr {e.mean():+.4f} +- {e.std(ddof=1)/np.sqrt(len(e)):.4f}")
    dg = [m["diagnostics"] for m in metas]
    for k in ("log_misrank_A1", "log_misrank_A0", "reference_ranking_reversal_rate",
              "mean_abs_effective_C", "median_target_margin", "per_action_residual"):
        v = np.array([x[k] for x in dg])
        print(f"  [diag] {k:<34} {v.mean():.4f} +- {v.std(ddof=1)/np.sqrt(len(v)):.4f}")
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
