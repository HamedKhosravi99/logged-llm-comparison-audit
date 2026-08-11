#!/usr/bin/env python3
"""Encouragement head-to-head on Criteo-UPLIFT v2.1, per docs/CRITEO_ANALYSIS_PLAN.md.

Naive (within-encouraged self-selection) vs ITT vs Wald, on the proxy (visit) and the
designated target (conversion), over J=8 covariate cells and 30 fixed partitions.
Everything here follows the frozen plan; deviations must be logged there.
"""
import argparse, gzip, json, os, sys
import numpy as np
import pandas as pd

EXPECTED_ROWS = 13_979_592
EXPECTED_HEADER = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
J_TOP_FEATURES = 3          # top-variance features, median-split -> 8 cells
N_PARTITIONS = 30
RNG_SEED = 777
FS_MIN = 0.005              # per-cell first-stage support floor
ARM_MIN = 200               # per-cell per-arm row floor
A1_TOL = 0.02               # +/- 2pp on relative ITT uplifts
A1_ANCHORS = {"visit": 0.687, "conversion": 0.372}


def tmean(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    from scipy import stats
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se) if se > 0 else (m, m)
    return {"mean": m, "se": se, "ci": [lo, hi], "n": len(x),
            "pos": int((x > 0).sum()), "neg": int((x < 0).sum())}


def cell_stats(g, outcome):
    """Per (partition, cell) estimator set for one outcome column."""
    t1 = g[g.treatment == 1]
    t0 = g[g.treatment == 0]
    if len(t1) < ARM_MIN or len(t0) < ARM_MIN:
        return None
    fs = t1.exposure.mean() - t0.exposure.mean()
    if fs < FS_MIN:
        return None
    exp1 = t1[t1.exposure == 1]
    une1 = t1[t1.exposure == 0]
    if len(exp1) == 0 or len(une1) == 0:
        return None
    naive = exp1[outcome].mean() - une1[outcome].mean()
    itt = t1[outcome].mean() - t0[outcome].mean()
    wald = itt / fs
    return {"naive": naive, "itt": itt, "wald": wald, "fs": fs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # ---------- load with schema verification ----------
    opener = gzip.open if args.data.endswith(".gz") else open
    with opener(args.data, "rt") as f:
        header = f.readline().strip().split(",")
    assert header == EXPECTED_HEADER, f"schema mismatch: {header}"

    dtypes = {f"f{i}": "float32" for i in range(12)}
    dtypes.update({"treatment": "int8", "conversion": "int8", "visit": "int8", "exposure": "int8"})
    df = pd.read_csv(args.data, dtype=dtypes)
    assert len(df) == EXPECTED_ROWS, f"row count {len(df)} != {EXPECTED_ROWS}"

    meta = {"rows": int(len(df)),
            "treatment_share": float(df.treatment.mean()),
            "exposure_given_control": float(df.loc[df.treatment == 0, "exposure"].mean())}
    # A2 one-sided compliance
    assert meta["exposure_given_control"] < 1e-4, meta

    # ---------- cells from covariates only ----------
    variances = {c: float(df[c].var()) for c in [f"f{i}" for i in range(12)]}
    top = sorted(variances, key=variances.get, reverse=True)[:J_TOP_FEATURES]
    medians = {c: float(df[c].median()) for c in top}
    cell = np.zeros(len(df), dtype=np.int8)
    for b, c in enumerate(top):
        cell |= ((df[c].values > medians[c]).astype(np.int8) << b)
    df["cell"] = cell
    weights = df.cell.value_counts(normalize=True).sort_index()

    # ---------- partitions ----------
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(df))
    part = np.empty(len(df), dtype=np.int8)
    part[perm] = np.arange(len(df)) % N_PARTITIONS
    df["part"] = part

    # ---------- A1 anchors on full data ----------
    anchors = {}
    for out, ref in A1_ANCHORS.items():
        c0 = df.loc[df.treatment == 0, out].mean()
        itt = df.loc[df.treatment == 1, out].mean() - c0
        rel = itt / c0
        anchors[out] = {"relative_itt": float(rel), "published": ref,
                        "ok": bool(abs(rel - ref) <= A1_TOL)}

    # ---------- per partition x cell estimators ----------
    rows = []
    for (p, j), g in df.groupby(["part", "cell"], sort=True):
        rec = {"part": int(p), "cell": int(j), "w": float(weights[j])}
        ok = True
        for out in ("visit", "conversion"):
            st = cell_stats(g, out)
            if st is None:
                ok = False
                break
            for k, v in st.items():
                rec[f"{out}_{k}"] = float(v)
        if ok:
            # algebraic identity check (same denominators by construction)
            for out in ("visit", "conversion"):
                assert abs(rec[f"{out}_itt"] - rec[f"{out}_fs"] * rec[f"{out}_wald"]) < 1e-12
            rows.append(rec)
    cells = pd.DataFrame(rows)
    n_expected = N_PARTITIONS * 8
    excluded = n_expected - len(cells)

    # ---------- pooled per partition ----------
    pooled = []
    for p, g in cells.groupby("part"):
        w = g.w / g.w.sum()
        rec = {"part": int(p)}
        for out in ("visit", "conversion"):
            for est in ("naive", "itt", "wald"):
                rec[f"{out}_{est}"] = float((g[f"{out}_{est}"] * w).sum())
        for out in ("visit", "conversion"):
            rec[f"selgap_{out}"] = rec[f"{out}_naive"] - rec[f"{out}_wald"]
        for est in ("naive", "itt", "wald"):
            rec[f"measgap_{est}"] = rec[f"visit_{est}"] - rec[f"conversion_{est}"]
        rec["k1_ratio"] = rec["selgap_conversion"] / rec["conversion_wald"]
        rec["k2_ratio"] = rec["measgap_wald"] / rec["measgap_naive"]
        pooled.append(rec)
    P = pd.DataFrame(pooled)

    out = {"meta": meta, "cells_def": {"features": top, "medians": medians},
           "weights": {int(k): float(v) for k, v in weights.items()},
           "anchors_A1": anchors, "excluded_partition_cells": int(excluded),
           "k1_selection_gap_conversion": tmean(P.selgap_conversion),
           "k1_ratio_to_wald_effect": tmean(P.k1_ratio),
           "k1_selection_gap_visit": tmean(P.selgap_visit),
           "k2_measurement_gap_wald": tmean(P.measgap_wald),
           "k2_measurement_gap_naive": tmean(P.measgap_naive),
           "k2_measurement_gap_itt": tmean(P.measgap_itt),
           "k2_ratio_wald_over_naive": tmean(P.k2_ratio),
           "pooled_estimates": {f"{out_}_{est}": tmean(P[f"{out_}_{est}"])
                                for out_ in ("visit", "conversion")
                                for est in ("naive", "itt", "wald")}}

    # ---------- K'3: cell-constant offset transfer ----------
    full_cells = []
    for j, g in df.groupby("cell"):
        st_c = cell_stats(g, "conversion")
        if st_c is not None:
            full_cells.append({"cell": int(j), "gap": st_c["naive"] - st_c["wald"],
                               "w": float(weights[j])})
    fc = pd.DataFrame(full_cells)
    rng2 = np.random.default_rng(RNG_SEED)
    fit_half = rng2.choice(fc.cell.values, size=len(fc) // 2, replace=False)
    b_hat = float(fc[fc.cell.isin(fit_half)].gap.mean())
    held = fc[~fc.cell.isin(fit_half)]
    var_before = float((held.gap ** 2).mean())
    var_after = float(((held.gap - b_hat) ** 2).mean())
    out["k3"] = {"fit_cells": sorted(int(c) for c in fit_half), "b_hat": b_hat,
                 "heldout_msq_before": var_before, "heldout_msq_after": var_after,
                 "share_explained": 1.0 - var_after / var_before if var_before > 0 else None,
                 "cellwise_gaps": fc.to_dict("records")}

    # per-partition ANOVA-style share: variance of per-cell gap explained by grand mean
    shares = []
    for p, g in cells.groupby("part"):
        gaps = g["conversion_naive"] - g["conversion_wald"]
        gm = gaps.mean()
        tot = float((gaps ** 2).mean())
        res = float(((gaps - gm) ** 2).mean())
        if tot > 0:
            shares.append(1.0 - res / tot)
    out["k3_share_explained_partitions"] = tmean(np.array(shares))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(args.output, "w") as f:
        json.dump(clean(out), f, indent=1)
    print("written", args.output)
    print("A1 anchors:", {k: (round(v['relative_itt'], 4), v['ok']) for k, v in anchors.items()})
    print("K'1 selection gap (conversion):", out["k1_selection_gap_conversion"]["mean"])
    print("K'2 measurement gap wald:", out["k2_measurement_gap_wald"]["mean"],
          " ratio wald/naive:", out["k2_ratio_wald_over_naive"]["mean"])
    print("K'3 share explained (partitions):", out["k3_share_explained_partitions"]["mean"])


if __name__ == "__main__":
    main()
