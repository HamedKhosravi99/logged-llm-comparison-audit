"""Intervals for the App E break-even figures (1,514 / 1,976 requests). NO new runs.

The published saving is the deployment-cost difference between the complete-matrix strict-best
policy (oracle_complete) and the corrected-comparison tolerance policy (qc_delta_0.02),
identified by exact reconstruction from the stored per-partition rows; the break-even divides
CASE's audit cost by that saving. Partition-level percentile bootstrap (ratio of means,
4000 replicates) supplies the intervals the reviewer asked for.

    python code/make_appE_breakeven_ci.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAIR = ("oracle_complete", "qc_delta_0.02")
AUDIT_ARM = "case"
REPS, SEED = 4000, 0


def one(corpus_file, published_saving_per_M, published_be):
    d = json.load(open(os.path.join(RES, corpus_file)))
    r = pd.DataFrame(d["rows"])
    r = r[r.assignment == "A1"]                               # the published figures are A1-only
    p = r.groupby(["method", "partition"]).agg(cost=("cost_per_request", "mean"),
                                               usd=("audit_usd", "mean")).reset_index()
    w = p.pivot(index="partition", columns="method", values="cost")
    u = p.pivot(index="partition", columns="method", values="usd")
    per = (w[PAIR[0]] - w[PAIR[1]]).to_numpy(float)          # saving per request, per partition
    usd = u[AUDIT_ARM].to_numpy(float)                        # CASE audit spend, per partition
    n = len(per)

    save_M = per.mean() * 1e6
    be = usd.mean() / per.mean()
    assert abs(save_M - published_saving_per_M) < 3, \
        f"{corpus_file}: saving ${save_M:,.0f}/M != published ${published_saving_per_M}"
    assert abs(be - published_be) < 4, f"{corpus_file}: break-even {be:,.0f} != {published_be}"

    rng = np.random.default_rng(SEED)
    bs_save, bs_be = [], []
    for _ in range(REPS):
        i = rng.integers(0, n, n)
        s = per[i].mean()
        bs_save.append(s * 1e6)
        bs_be.append(usd[i].mean() / s if s > 0 else np.inf)
    save_ci = (float(np.quantile(bs_save, .025)), float(np.quantile(bs_save, .975)))
    finite = np.isfinite(bs_be)
    be_ci = (float(np.quantile(np.asarray(bs_be)[finite], .025)),
             float(np.quantile(np.asarray(bs_be)[finite], .975)))
    frac_inf = float(1 - finite.mean())
    return dict(n_partitions=n, saving_per_M=save_M, saving_ci=save_ci,
                break_even=be, break_even_ci=be_ci, frac_nonpositive_saving=frac_inf)


def main():
    out = {}
    for f, corpus, sM, be in [("econ_llmrouterbench.json", "primary", 1579, 1514),
                              ("econ_routerbench.json", "RouterBench", 451, 1976)]:
        out[corpus] = one(f, sM, be)
        v = out[corpus]
        print(f"{corpus:12} saving ${v['saving_per_M']:,.0f}/M "
              f"[{v['saving_ci'][0]:,.0f}, {v['saving_ci'][1]:,.0f}]  "
              f"break-even {v['break_even']:,.0f} "
              f"[{v['break_even_ci'][0]:,.0f}, {v['break_even_ci'][1]:,.0f}] requests  "
              f"(P(saving<=0) in bootstrap: {v['frac_nonpositive_saving']:.3f})")
    json.dump(out, open(os.path.join(RES, "appE_breakeven_ci.json"), "w"), indent=1)
    print("wrote appE_breakeven_ci.json")


if __name__ == "__main__":
    main()
