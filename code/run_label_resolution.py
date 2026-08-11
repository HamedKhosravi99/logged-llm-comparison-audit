#!/usr/bin/env python3
"""P0.5: is the 'true domain best' label resolved at the tolerance it adjudicates?

Per docs/PHASE0_ANALYSIS_PLAN.md. Table 2 classifies every switch against the true domain best,
computed from ~440 tau2-bench simulations. This asks whether that label is determined at delta=0.02
given the per-cell sample sizes, and how often a switch's classification flips under resampling.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
from tau2_e2e_full import load, DELTA          # noqa: E402

B = 1000
SEED = 20260810


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    df, tc = load()
    doms = sorted(tc.domain.unique())
    models = sorted(tc.agent_model.unique())

    # ---- per (domain, model) success with n and Wilson interval -------------
    cells = []
    for d in doms:
        for m in models:
            s = tc[(tc.domain == d) & (tc.agent_model == m)].success
            if len(s) == 0:
                continue
            n = len(s)
            k = float(s.sum())
            lo, hi = wilson(k, n)
            cells.append(dict(domain=d, model=m, n=int(n), success=float(s.mean()),
                              ci_lo=float(lo), ci_hi=float(hi)))
    C = pd.DataFrame(cells)

    # ---- bootstrap: resample simulations within (domain, model) ------------
    rng = np.random.default_rng(SEED)
    by = {(d, m): tc[(tc.domain == d) & (tc.agent_model == m)].success.values
          for d in doms for m in models
          if len(tc[(tc.domain == d) & (tc.agent_model == m)]) > 0}

    # decision = (domain, candidate). in-tolerance iff TRUE[cand] >= best - delta
    decisions = [(d, m) for d in doms for m in models if (d, m) in by]
    flips = {k: 0 for k in decisions}
    intol_point = {}
    for d in doms:
        best = max(by[(d, m)].mean() for m in models if (d, m) in by)
        for m in models:
            if (d, m) in by:
                intol_point[(d, m)] = bool(by[(d, m)].mean() >= best - DELTA)

    for _ in range(B):
        res = {k: v[rng.integers(0, len(v), len(v))].mean() for k, v in by.items()}
        for d in doms:
            best = max(res[(d, m)] for m in models if (d, m) in by)
            for m in models:
                if (d, m) in by:
                    if bool(res[(d, m)] >= best - DELTA) != intol_point[(d, m)]:
                        flips[(d, m)] += 1

    rows = []
    for (d, m) in decisions:
        fr = flips[(d, m)] / B
        rows.append(dict(domain=d, model=m, in_tolerance=intol_point[(d, m)],
                         flip_rate=fr, indeterminate=bool(fr >= 0.05)))
    D = pd.DataFrame(rows)

    n_dec = len(D)
    n_ind = int(D.indeterminate.sum())
    share = n_ind / n_dec if n_dec else 0.0
    trigger = ("INDICATIVE_ONLY" if share > 0.25 else
               "BRACKETED_RANGE" if share >= 0.10 else "PROCEED")

    out = dict(
        spec=dict(delta=DELTA, bootstrap=B, seed=SEED,
                  flip_threshold=0.05,
                  note="a decision is indeterminate if its in-tolerance classification "
                       "flips in >=5% of resamples"),
        cells=C.to_dict("records"),
        decisions=D.to_dict("records"),
        summary=dict(n_domain_decisions=n_dec, n_indeterminate=n_ind,
                     share_indeterminate=share, trigger=trigger),
    )
    p = os.path.join(RES, "label_resolution.json")
    json.dump(out, open(p, "w"), indent=1)

    print(f"cells: {len(C)}  decisions: {n_dec}")
    print(C.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"indeterminate: {n_ind}/{n_dec} = {share:.1%}   TRIGGER: {trigger}")
    print()
    print(D[D.indeterminate].to_string(index=False, float_format=lambda v: f"{v:.3f}")
          if n_ind else "(no indeterminate decisions)")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
