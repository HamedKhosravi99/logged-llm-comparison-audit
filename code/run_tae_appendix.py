"""Pre-specified sensitivity package for the selection-vs-measurement validity study.

Run only after the mandatory package (`run_tae_validity.py`) is complete. Every sweep below was
fixed before the results were inspected:

  A. evaluator-bias strength   lambda in {0.5, 1.0} x median_j Delta^ref_{j,train}  (0x = E0)
  B. stream mismatch           attenuation C^i = 0.5 C^o  vs  sign reversal C^i = -0.5 C^o
  C. readout size              n in {15, 30, 60} prompts per purchased cell
  D. baseline definition       log-derived deployment policy (main) vs the incumbent router policy
  F. evaluator-bias shape      context-invariant model favouritism C_kj = C_k
  G. context granularity       J in {6, 8, 12, 16} embedding clusters (KMeans seed varies by partition)
  H. routing aggressiveness    incumbent exploration rate eps in {0.05, 0.15, 0.30, 0.50}
  E. appendix acquisition rules raw-margin and observed-surface R-EPIG (already in the main file)

    python run_tae_appendix.py [--partitions 20] [--acq-seeds 6]
"""
import argparse
import json
import os
import time

import numpy as np

from eval_validity.pipeline import run_partition

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "results", "tae_validity_appendix.json")


def sweep(name, partitions, **kw):
    rows, grows, metas = [], [], []
    t0 = time.time()
    for p in range(partitions):
        r, g, m = run_partition(p, **kw)
        for x in r:
            x["sweep"] = name
        for x in g:
            x["sweep"] = name
        rows += r
        grows += g
        metas.append(m)
    print(f"  {name:<28} {len(rows):6d} rows  {len(grows):5d} gate rows  [{time.time()-t0:.0f}s]",
          flush=True)
    return rows, grows, metas


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
    ap.add_argument("--acq-seeds", type=int, default=6)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    P, S = args.partitions, args.acq_seeds

    rows, grows, metas = [], [], {}
    specs = [
        # A. evaluator-bias strength (lambda multiplier on the pre-specified 0.5 x median margin)
        ("lambda_x1.0", dict(lam_scale=2.0, factorial_conds=("D1",), gate_conds=("D1",))),
        # B. stream mismatch: sign reversal instead of attenuation
        ("stream_sign_reversal", dict(factorial_conds=("D2rev",), gate_conds=("D2rev",))),
        # C. readout size
        ("readout_15", dict(n_read=15, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        ("readout_60", dict(n_read=60, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        # D. status-quo definition for the deployment gate
        ("baseline_router", dict(baseline="router", factorial_conds=(), gate_conds=("D1", "D2"))),
        # F. context-invariant model favouritism C_kj = C_k -- the cleanest case of Prop. 1(b)
        ("favoritism_Ck", dict(factorial_conds=("D1k",), gate_conds=("D1k",))),
        # H. routing aggressiveness: the assignment factor's whole magnitude is set by how sharply
        #    the incumbent router selects, so the exploration rate is swept end to end. eps -> 1 is
        #    uniform routing, which is A0 by construction; eps = 0.15 is the main setting.
        ("router_eps0.05", dict(eps=0.05, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        ("router_eps0.30", dict(eps=0.30, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        ("router_eps0.50", dict(eps=0.50, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        # G. context granularity: the whole design is defined over J embedding clusters, so J is a
        #    design choice that could drive the margins, the reversals and the acquisition order
        ("contexts_J6", dict(n_clusters=6, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        ("contexts_J12", dict(n_clusters=12, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
        ("contexts_J16", dict(n_clusters=16, factorial_conds=("D0", "D1"), gate_conds=("D1",))),
    ]
    for name, kw in specs:
        kw.setdefault("acq_seeds", S)
        r, g, m = sweep(name, P, **kw)
        rows += r
        grows += g
        metas[name] = m

    out = dict(spec=dict(partitions=P, acq_seeds=S, generated_by="code/run_tae_appendix.py",
                         sweeps=[n for n, _ in specs]),
               rows=rows, gate_rows=grows, meta=metas)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(_round(out), open(args.out, "w"), separators=(",", ":"))
    print(f"\nwrote {args.out}  ({len(rows)} rows, {len(grows)} gate rows)")


if __name__ == "__main__":
    main()
