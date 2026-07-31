"""Is the un-repaired third budget/noise, or is it the action-level correction family itself?

Prop. 1 is cellwise: the selection distortion is B^sel_kj. The deployed correction is action-level,
b_k = mean over purchased cells of (M_kj - R_kj), applied to every context of row k. If the router
induces strongly context-dependent selection, that family is misspecified even when the
evaluator-comparability condition of Prop. 1(a) holds -- and the incomplete recovery reported in the
paper would then be structural rather than a budget effect. This separates the two.

Arms, all scored by reference-channel regret on the untouched scoring cohort:

  trust             the logged surface M.
  CASE              fitted action-level correction at CASE's allocation (the deployed method).
  CASE + two-way    same allocation, additive row+column correction fitted on the purchased cells.
  CASE + low-rank   same allocation, rank-r completion of the purchased bias residuals.
  ORACLE row        M minus the BEST POSSIBLE action-level correction, computed from the truth with
                    no budget at all. This is the ceiling of the correction family: whatever regret
                    remains here cannot be bought with more readouts, only with a richer corrector.
  ORACLE cell       the population-operational surface mu^op (cellwise-perfect assignment repair).
  ORACLE ref        the designated target surface mu^ref.

The gap CASE -> ORACLE row is budget, acquisition and readout noise; the gap ORACLE row -> ORACLE
cell is what the action-level family structurally cannot represent.

    python run_tae_correction.py [--partitions 20] [--acq-seeds 6] [--budget 0.35]
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
DEFAULT_OUT = os.path.join(HERE, "..", "results", "tae_correction_family.json")


def case_reads(M, cellq, supp, B, seed, n_read=30):
    """CASE's allocation, returned as (reads, R) so every correction family sees the same cells."""
    K, J = M.shape
    rng = np.random.default_rng(seed)
    R = np.zeros((K, J), bool)
    reads = {}

    def do(k, j):
        reads[(k, j)] = cc.readout(cellq, k, j, rng, n_read)
        R[k, j] = True

    for k in range(K):
        js = np.where(supp[k])[0]
        if len(js) and R.sum() < B:
            do(k, int(js[rng.integers(len(js))]))
    while R.sum() < B:
        b = cc.bias_rows(M, R, reads)
        cell = cc._smallest_margin_cell(cc.corrected(M, R, reads, b), supp, R)
        if cell is None:
            break
        do(*cell)
    return reads, R


def lowrank_bias(M, reads, R, rank=2):
    """Rank-r completion of the purchased bias residuals D_kj = M_kj - R_kj, then subtract."""
    K, J = M.shape
    D = np.zeros((K, J))
    for (k, j), r in reads.items():
        D[k, j] = M[k, j] - r
    Dhat = cc._lr_complete(D, R, rank)
    V = M - Dhat
    return cc._place_reads(V, reads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=20)
    ap.add_argument("--acq-seeds", type=int, default=6)
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--n-read", type=int, default=30)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = []
    for cond in ("D0", "D1"):
        for p in range(args.partitions):
            base = build_base(p)
            cells, n_kj, _ = A.make_selective_log(base, p, eps=args.eps)
            supp = A.support(n_kj, base, n_read=args.n_read)
            tab = MT.scoring_tables(base, supp)
            pol_ref = MT.target_policy(tab, supp)
            surf = EV.build_surfaces(base, cells, EV.distortions(base, cond), args.n_read)
            M, Ef_op, Ef_ref = surf["M"], surf["Ef_op"], surf["Ef_ref"]
            B = max(base["K"], int(round(args.budget * int(supp.sum()))))

            # --- the correction family's own ceiling, computed from the truth, with NO budget ---
            D = M - Ef_op                                   # the full cellwise selection distortion
            b_best = np.array([D[k, supp[k]].mean() if supp[k].any() else 0.0
                               for k in range(base["K"])])
            V_row_oracle = M - b_best[:, None]
            resid = float(((D - b_best[:, None]) ** 2)[supp].sum() / max((D ** 2)[supp].sum(), 1e-12))
            spread = float(np.mean([D[k, supp[k]].std() for k in range(base["K"]) if supp[k].sum() > 1]))

            reg = lambda V: MT.reference_regret(MT.deployed_policy(V, supp), tab, supp, pol_ref)
            fixed = dict(trust=reg(M), oracle_row=reg(V_row_oracle),
                         oracle_cell=reg(Ef_op), oracle_ref=reg(Ef_ref))

            for s in range(args.acq_seeds):
                seed = 10_000 * p + 100 * int(args.budget * 100) + s
                reads, R = case_reads(M, surf["cellq"], supp, B, seed, args.n_read)
                b = cc.bias_rows(M, R, reads)
                row = dict(partition=p, acq_seed=s, evaluator=cond, B=B, budget=args.budget,
                           case=reg(cc.corrected(M, R, reads, b)),
                           case_twoway=reg(cc.vm_twoway(M, reads, R)),
                           case_lowrank=reg(lowrank_bias(M, reads, R, rank=2)),
                           case_athey=reg(cc.vm_athey(M, reads, R)),
                           bias_residual_not_row_constant=resid,
                           within_row_sd_of_selection=spread,
                           row_correction_error=float(np.abs(b - b_best)[
                               [k for k in range(base["K"]) if supp[k].any()]].mean()),
                           **fixed)
                rows.append(row)
        print(f"  {cond} done", flush=True)

    out = dict(spec=dict(budget=args.budget, partitions=args.partitions, acq_seeds=args.acq_seeds,
                         n_read=args.n_read, eps=args.eps,
                         generated_by="code/run_tae_correction.py"), rows=rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), separators=(",", ":"))

    fields = ["trust", "case", "case_twoway", "case_lowrank", "case_athey",
              "oracle_row", "oracle_cell", "oracle_ref"]
    print("\nreference regret at the {:.0%} budget (mean +- paired SE over partitions)".format(args.budget))
    for cond in ("D0", "D1"):
        sel = [r for r in rows if r["evaluator"] == cond]
        parts = sorted({r["partition"] for r in sel})
        print(f"  A1-{cond}")
        for f in fields:
            v = np.array([np.mean([r[f] for r in sel if r["partition"] == p]) for p in parts])
            print(f"    {f:<14} {v.mean():.4f} +- {v.std(ddof=1)/np.sqrt(len(v)):.4f}")
        for f in ("bias_residual_not_row_constant", "within_row_sd_of_selection",
                  "row_correction_error"):
            v = np.array([np.mean([r[f] for r in sel if r["partition"] == p]) for p in parts])
            print(f"    [diag] {f:<34} {v.mean():.4f} +- {v.std(ddof=1)/np.sqrt(len(v)):.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
