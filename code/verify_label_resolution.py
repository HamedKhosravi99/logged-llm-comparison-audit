#!/usr/bin/env python3
"""Independent recomputation of P0.5 by a different method.

run_label_resolution.py uses a nonparametric bootstrap flip-rate. This script instead computes,
analytically, whether the paired per-task difference (candidate - domain argmax) has a confidence
interval straddling -delta. Different estimator, different resampling logic; the indeterminate set
should agree.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
from tau2_e2e_full import load, DELTA          # noqa: E402


def main():
    _, tc = load()
    doms = sorted(tc.domain.unique())
    piv = tc.pivot_table(index=["domain", "task_id"], columns="agent_model",
                         values="success", aggfunc="mean")
    ind = []
    for d in doms:
        sub = piv.loc[d].dropna(how="all")
        means = sub.mean()
        argmax = means.idxmax()
        for m in sub.columns:
            if m == argmax:          # the argmax is in tolerance by construction
                ind.append(dict(domain=d, model=m, argmax=argmax, n_tasks=int(sub[m].notna().sum()),
                                mean_diff=0.0, ci_lo=0.0, ci_hi=0.0, indeterminate=False))
                continue
            # paired per-task difference against the domain argmax
            pair = sub[[m, argmax]].dropna()
            diff = pair[m] - pair[argmax]
            n = len(diff)
            se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
            lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
            # in-tolerance means diff >= -delta ; indeterminate if the CI straddles -delta
            straddles = bool(lo < -DELTA < hi)
            ind.append(dict(domain=d, model=m, argmax=argmax, n_tasks=int(n),
                            mean_diff=float(diff.mean()), ci_lo=float(lo), ci_hi=float(hi),
                            indeterminate=straddles))
    V = pd.DataFrame(ind)

    ref = json.load(open(os.path.join(RES, "label_resolution.json")))
    R = pd.DataFrame(ref["decisions"])
    a = set(map(tuple, R[R.indeterminate][["domain", "model"]].values))
    b = set(map(tuple, V[V.indeterminate][["domain", "model"]].values))

    print(V.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"bootstrap method  indeterminate: {len(a)}/{len(R)}  {sorted(a)}")
    print(f"analytic  method  indeterminate: {len(b)}/{len(V)}  {sorted(b)}")
    print()
    print("AGREE" if a == b else f"DISAGREE  only-bootstrap={sorted(a-b)}  only-analytic={sorted(b-a)}")


if __name__ == "__main__":
    main()
