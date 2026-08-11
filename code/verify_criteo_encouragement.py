#!/usr/bin/env python3
"""Independent recomputation of results/criteo_encouragement.json on a second code path.

Uses numpy bincount aggregation (no pandas groupby) and re-derives cells, partitions,
per-cell estimators, pooled contrasts K'1/K'2, and the A1 anchors from the raw file.
Fails loudly on any mismatch beyond tolerance. Per docs/CRITEO_ANALYSIS_PLAN.md.
"""
import argparse, json
import numpy as np
import pandas as pd

TOL = 1e-9          # deterministic pipeline: mismatches indicate a real bug
RNG_SEED = 777
N_PART = 30
FS_MIN = 0.005
ARM_MIN = 200


def agg(idx, val, n_groups):
    return np.bincount(idx, weights=val, minlength=n_groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    R = json.load(open(args.results))

    df = pd.read_csv(args.data,
                     dtype={**{f"f{i}": "float32" for i in range(12)},
                            **{c: "int8" for c in ("treatment", "conversion", "visit", "exposure")}})
    assert len(df) == R["meta"]["rows"]

    # cells: recompute feature ranking and medians independently
    variances = {c: float(df[c].var()) for c in [f"f{i}" for i in range(12)]}
    top = sorted(variances, key=variances.get, reverse=True)[:3]
    assert top == R["cells_def"]["features"], (top, R["cells_def"]["features"])
    cell = np.zeros(len(df), dtype=np.int64)
    for b, c in enumerate(top):
        med = float(df[c].median())
        assert abs(med - R["cells_def"]["medians"][c]) < 1e-6
        cell |= ((df[c].values > med).astype(np.int64) << b)

    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(df))
    part = np.empty(len(df), dtype=np.int64)
    part[perm] = np.arange(len(df)) % N_PART

    T = df.treatment.values.astype(np.int64)
    D = df.exposure.values.astype(np.int64)
    Y = {"visit": df.visit.values.astype(np.float64),
         "conversion": df.conversion.values.astype(np.float64)}

    # weights
    wfull = np.bincount(cell, minlength=8) / len(df)
    for j in range(8):
        assert abs(wfull[j] - R["weights"][str(j)]) < 1e-12

    # A1 anchors
    for out, ref in (("visit", None), ("conversion", None)):
        c0 = Y[out][T == 0].mean()
        rel = (Y[out][T == 1].mean() - c0) / c0
        assert abs(rel - R["anchors_A1"][out]["relative_itt"]) < 1e-9, out

    # per (partition, cell) estimators via bincount on a combined index
    gidx = part * 8 + cell
    G = N_PART * 8
    ones = np.ones(len(df))
    n_t1 = agg(gidx[T == 1], ones[T == 1], G)
    n_t0 = agg(gidx[T == 0], ones[T == 0], G)
    d_t1 = agg(gidx[T == 1], D[T == 1].astype(float), G)
    d_t0 = agg(gidx[T == 0], D[T == 0].astype(float), G)
    fs = np.where((n_t1 > 0) & (n_t0 > 0), d_t1 / np.maximum(n_t1, 1) - d_t0 / np.maximum(n_t0, 1), np.nan)

    est = {}
    for out in ("visit", "conversion"):
        y = Y[out]
        y_t1 = agg(gidx[T == 1], y[T == 1], G)
        y_t0 = agg(gidx[T == 0], y[T == 0], G)
        m1 = (T == 1) & (D == 1)
        m0 = (T == 1) & (D == 0)
        y_e1 = agg(gidx[m1], y[m1], G)
        n_e1 = agg(gidx[m1], ones[m1], G)
        y_e0 = agg(gidx[m0], y[m0], G)
        n_e0 = agg(gidx[m0], ones[m0], G)
        itt = y_t1 / np.maximum(n_t1, 1) - y_t0 / np.maximum(n_t0, 1)
        naive = y_e1 / np.maximum(n_e1, 1) - y_e0 / np.maximum(n_e0, 1)
        wald = itt / fs
        est[out] = {"itt": itt, "naive": naive, "wald": wald,
                    "n_e1": n_e1, "n_e0": n_e0}

    supported = ((n_t1 >= ARM_MIN) & (n_t0 >= ARM_MIN) & (fs >= FS_MIN)
                 & (est["visit"]["n_e1"] > 0) & (est["visit"]["n_e0"] > 0))
    n_excluded = int(G - supported.sum())
    assert n_excluded == R["excluded_partition_cells"], (n_excluded, R["excluded_partition_cells"])

    # pooled per partition, weighted by full-data cell weights renormalized over supported cells
    selgap_conv, measgap_wald, measgap_naive, k1r, k2r = [], [], [], [], []
    for p in range(N_PART):
        gs = np.array([p * 8 + j for j in range(8)])
        ok = supported[gs]
        w = wfull[np.arange(8)][ok]
        w = w / w.sum()
        pooled = {}
        for out in ("visit", "conversion"):
            for name in ("naive", "itt", "wald"):
                pooled[f"{out}_{name}"] = float((est[out][name][gs][ok] * w).sum())
        selgap_conv.append(pooled["conversion_naive"] - pooled["conversion_wald"])
        mw = pooled["visit_wald"] - pooled["conversion_wald"]
        mn = pooled["visit_naive"] - pooled["conversion_naive"]
        measgap_wald.append(mw)
        measgap_naive.append(mn)
        k1r.append((pooled["conversion_naive"] - pooled["conversion_wald"]) / pooled["conversion_wald"])
        k2r.append(mw / mn)

    def close(name, mine, theirs):
        assert abs(np.mean(mine) - theirs["mean"]) < 1e-9, (name, np.mean(mine), theirs["mean"])
        print(f"  ok {name}: {np.mean(mine):+.6g}")

    close("K'1 selection gap (conversion)", selgap_conv, R["k1_selection_gap_conversion"])
    close("K'1 ratio", k1r, R["k1_ratio_to_wald_effect"])
    close("K'2 measurement gap (wald)", measgap_wald, R["k2_measurement_gap_wald"])
    close("K'2 measurement gap (naive)", measgap_naive, R["k2_measurement_gap_naive"])
    close("K'2 ratio wald/naive", k2r, R["k2_ratio_wald_over_naive"])
    print("ALL INDEPENDENT RECOMPUTATIONS MATCH")


if __name__ == "__main__":
    main()
