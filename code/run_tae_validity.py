"""Mandatory experiment package for the selection-vs-measurement validity study.

Runs, on the LLMRouterBench performance-cost replay:

  * the 2x2 factorial   A0/A1 assignment x D0/D1 evaluator, at 15%/25%/35% cell budgets, for
                        trust / uniform / CASE / corrected-surface R-EPIG plus the two labelled
                        oracle controls;
  * the A1-D2 evaluator-comparability diagnostic;
  * the deployment-decision comparison  ungated / same-evaluator held-out check /
                                        reference-channel held-out check, under A1-D1 and A1-D2;
  * every leakage and algebra test in `eval_validity.validation`.

The replication unit is the prompt-cohort partition; acquisition seeds within a partition are
averaged before any interval is formed.

    python run_tae_validity.py [--partitions 20] [--acq-seeds 5] [--out ../results/tae_validity.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from eval_validity import validation
from eval_validity.pipeline import run_partition

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "results", "tae_validity.json")


def _round(o, nd=6):
    """Compact JSON: round every float so the long-form result files stay small but exact to 1e-6."""
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
    ap.add_argument("--acq-seeds", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--n-read", type=int, default=30)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    print("running leakage / algebra tests ...", flush=True)
    tests = validation.run_all(part_seed=0, eps=args.eps, n_read=args.n_read, verbose=False)
    for k, v in tests.items():
        if isinstance(v, dict):
            print(f"  [{'PASS' if v['pass_'] else 'FAIL'}] {k}")
    if not tests["all_pass"]:
        sys.exit("leakage/algebra tests failed -- refusing to produce a result file")

    rows, gate_rows, metas = [], [], []
    for p in range(args.partitions):
        r, g, m = run_partition(p, acq_seeds=args.acq_seeds, eps=args.eps, n_read=args.n_read,
                                extra_methods=("rawmargin", "epig_obs"),
                                factorial_conds=("D0", "D1", "D2"))
        rows += r
        gate_rows += g
        metas.append(m)
        d = m["diagnostics"]
        print(f"  partition {p:2d}  supported {m['n_supported']}/{m['K']*m['J']}  "
              f"lambda {m['lam']:.4f}  |C_eff| {d['mean_abs_effective_C']:.4f}  "
              f"log-misrank(A1) {d['log_misrank_A1']:.2f}  "
              f"target-ranking reversal {d['reference_ranking_reversal_rate']:.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    out = dict(
        spec=dict(dataset="LLMRouterBench performance-cost track (offline replay)",
                  n_prompts=11289, slices=["simpleqa", "mmlupro", "hle", "livecodebench", "arenahard"],
                  K=metas[0]["K"], J=metas[0]["J"], partitions=args.partitions,
                  acq_seeds=args.acq_seeds, eps=args.eps, n_read=args.n_read,
                  budgets=[0.15, 0.25, 0.35], baseline="log-derived deployment policy (argmax_k M_kj)",
                  bias_formula="C_kj = lambda * z_k * s_j, lambda = 0.5 * median_j Delta^ref_{j,train}",
                  generated_by="code/run_tae_validity.py"),
        tests=tests, rows=rows, gate_rows=gate_rows, meta=metas)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(_round(out), open(args.out, "w"), separators=(",", ":"))
    print(f"\nwrote {args.out}  ({len(rows)} factorial rows, {len(gate_rows)} gate rows, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
