"""Why do the two deconfounded-surface acquisition rules tie? Because they buy the same cells.

Reference-regret equality between two acquisition rules is weak evidence that the rule does not
matter -- two different allocations can happen to score alike. This measures the thing directly: the
overlap between the SETS of cells each rule purchases, at a matched budget, seed and support.

If the corrected surface (not the rule) is what decides where the budget goes, then two rules run on
that surface should buy nearly the same cells, while the same rule run on the observed surface should
not. That is the prediction the workshop paper's Q2 rests on.

    python run_tae_overlap.py [--partitions 20] [--acq-seeds 6] [--budget 0.35]
"""
import argparse
import json
import os

import numpy as np

import case_core as cc
from eval_validity import assignment as A
from eval_validity import evaluators as EV
from eval_validity import metrics as MT
from eval_validity.grid import build_base

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "results", "tae_acquisition_overlap.json")


def purchased_cells(rule, surface, M, cellq, supp, B, seed, n_read=30):
    """The set of cells one (rule, surface) pair buys. Seeding, budget and support are shared."""
    K, J = M.shape
    rng = np.random.default_rng(seed)
    R = np.zeros((K, J), bool)
    reads = {}

    def do(k, j):
        reads[(k, j)] = cc.readout(cellq, k, j, rng, n_read)
        R[k, j] = True

    for k in range(K):                                   # shared per-action seed
        js = np.where(supp[k])[0]
        if len(js) and R.sum() < B:
            do(k, int(js[rng.integers(len(js))]))
    while R.sum() < B:
        b = cc.bias_rows(M, R, reads) if surface == "corrected" else np.zeros(K)
        V = cc.corrected(M, R, reads, b)
        cell = (cc._smallest_margin_cell(V, supp, R) if rule == "margin"
                else cc._epig_cell(V, supp, R))
        if cell is None:
            break
        do(*cell)
    return set(reads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=20)
    ap.add_argument("--acq-seeds", type=int, default=6)
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--n-read", type=int, default=30)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    arms = {"epig_corrected": ("epig", "corrected"),      # different rule, same surface
            "epig_observed": ("epig", "observed"),        # same rule, different surface
            "margin_observed": ("margin", "observed")}    # CASE's rule off the corrected surface
    rows = []
    for cond in ("E0", "E1"):
        for p in range(args.partitions):
            base = build_base(p)
            cells, n_kj, _ = A.make_selective_log(base, p, eps=args.eps)
            supp = A.support(n_kj, base, n_read=args.n_read)
            surf = EV.build_surfaces(base, cells, EV.distortions(base, cond), args.n_read)
            B = max(base["K"], int(round(args.budget * int(supp.sum()))))
            for s in range(args.acq_seeds):
                seed = 10_000 * p + 100 * int(args.budget * 100) + s
                base_set = purchased_cells("margin", "corrected", surf["M"], surf["cellq"],
                                           supp, B, seed, args.n_read)
                rng = np.random.default_rng(seed)
                free = list(map(tuple, np.argwhere(supp)))
                unif = {free[i] for i in rng.permutation(len(free))[:B]}
                row = dict(partition=p, acq_seed=s, evaluator=cond, B=B,
                           budget=args.budget, n_supported=int(supp.sum()),
                           uniform=len(base_set & unif) / len(base_set))
                for name, (rule, sfc) in arms.items():
                    other = purchased_cells(rule, sfc, surf["M"], surf["cellq"], supp, B, seed,
                                            args.n_read)
                    row[name] = len(base_set & other) / len(base_set)
                rows.append(row)
        print(f"  {cond}: {args.partitions} partitions x {args.acq_seeds} seeds", flush=True)

    out = dict(spec=dict(reference_arm="CASE: smallest corrected margin on the corrected surface",
                         budget=args.budget, partitions=args.partitions, acq_seeds=args.acq_seeds,
                         n_read=args.n_read, eps=args.eps,
                         generated_by="code/run_tae_overlap.py"),
               rows=rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), separators=(",", ":"))

    print("\ncell-set overlap with CASE's purchases (shared budget, seed and support):")
    for cond in ("E0", "E1"):
        sel = [r for r in rows if r["evaluator"] == cond]
        parts = sorted({r["partition"] for r in sel})
        print(f"  A1-{cond}")
        for name in ("epig_corrected", "margin_observed", "epig_observed", "uniform"):
            v = np.array([np.mean([r[name] for r in sel if r["partition"] == p]) for p in parts])
            print(f"    {name:<18} {v.mean():.1%} +- {v.std(ddof=1)/np.sqrt(len(v)):.1%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
