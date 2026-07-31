"""Table 1: confidence-gated variance-pooled CASE against free raw-log downrouting.

Regenerates every number printed in the paper's operational table from the stored replay,
including the four-way switch decomposition that replaced the old Figure 2. Hard-coded values
appear only in assertions; nothing is transcribed by hand.

Usage:
    python code/make_table1_operational.py \
      --switches results/free_stage1_switch_categories.parquet \
      --units    results/free_stage1_router_sensitivity.parquet \
      --output   results/table1_operational.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

SCENARIO = "cost_reduction"
N_UNITS = 50
N_SWITCH_ROWS = 24_150
H = 100_000

# Arms are matched by exact name; the two result files use different vintages of the same arm.
ARM_ALIASES = {
    "Safe inverse-variance CASE": "Confidence-gated IV CASE",
    "Confidence-gated IV CASE": "Confidence-gated IV CASE",
    "Raw-log downrouting": "Raw-log downrouting",
    "Raw incumbent": "Incumbent",
}
BASE, GATED, INC = "Raw-log downrouting", "Confidence-gated IV CASE", "Incumbent"
CATS = ["A", "B", "C", "D"]


def alias(df):
    out = df.copy()
    out["arm_std"] = out["arm"].map(ARM_ALIASES)
    return out[out["arm_std"].notna()]


def switch_units(path):
    """Mean switches per run by true category, per fold x seed unit.

    `category` is null for a non-switch, so each arm is reindexed onto its own complete
    fold x seed x acq grid with zeros before averaging; otherwise units with no switch vanish.
    """
    df = pd.read_parquet(path)
    assert len(df) == N_SWITCH_ROWS, f"{path}: {len(df):,} rows, expected {N_SWITCH_ROWS:,}"
    df = alias(df[df.scenario == SCENARIO])
    frames = []
    for a, g in df.groupby("arm_std"):
        grid = pd.MultiIndex.from_product(
            [sorted(g.fold.unique()), sorted(g.seed.unique()), sorted(g.acq.unique())],
            names=["fold", "seed", "acq"])
        ct = (g.pivot_table(index=["fold", "seed", "acq"], columns="category",
                            values="switched", aggfunc="count", fill_value=0)
              .reindex(grid, fill_value=0))
        for k in CATS:
            if k not in ct.columns:
                ct[k] = 0
        u = ct.groupby(level=["fold", "seed"])[CATS].mean()
        assert len(u) == N_UNITS, f"{a}: {len(u)} units, expected {N_UNITS}"
        u["total"] = u[CATS].sum(axis=1)
        u = u.reset_index(); u["arm"] = a
        frames.append(u)
    out = pd.concat(frames, ignore_index=True)
    # the incumbent never switches; its category counts are identically zero on every unit
    zero = out[out.arm == BASE][["fold", "seed"]].drop_duplicates().copy()
    for k in CATS + ["total"]:
        zero[k] = 0.0
    zero["arm"] = INC
    return pd.concat([out, zero], ignore_index=True)


def unit_outcomes(path):
    df = pd.read_parquet(path)
    df = alias(df[df.scenario == SCENARIO])
    dup = df.groupby(["arm_std", "fold", "seed"]).size()
    assert (dup == 1).all(), "unit file is not one row per arm x fold x seed"
    w = df.pivot_table(index=["fold", "seed"], columns="arm_std",
                       values=["success", "cost", "audit"])
    assert len(w) == N_UNITS, f"{len(w)} units, expected {N_UNITS}"
    return w


def paired(v, reps=4000, seed=0):
    """Percentile bootstrap over the paired per-unit differences.

    Matches the convention already used for this table in free_stage1_write_report.py, so the
    two switch-decomposition rows added here are computed the same way as the rows beside them.
    """
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = np.array([np.mean(rng.choice(v, len(v), replace=True)) for _ in range(reps)])
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    S = switch_units(args.switches)
    W = unit_outcomes(args.units)
    sw = S.pivot_table(index=["fold", "seed"], columns="arm", values=CATS + ["total"])

    rows = []

    def add(label, base_v, gate_v, diff, fmt, conclusion, inc_v=np.nan, diff_inc=None):
        m, lo, hi = diff
        mi, li, hi_i = diff_inc if diff_inc else (np.nan, np.nan, np.nan)
        rows.append(dict(outcome=label, incumbent=inc_v, raw_log=base_v, conf_gated=gate_v,
                         diff=m, ci_lo=lo, ci_hi=hi,
                         diff_inc=mi, ci_inc_lo=li, ci_inc_hi=hi_i,
                         fmt=fmt, conclusion=conclusion))

    succ = (W[("success", GATED)] - W[("success", BASE)]).dropna()
    succ_i = (W[("success", GATED)] - W[("success", INC)]).dropna()
    add("Task success", W[("success", BASE)].mean(), W[("success", GATED)].mean(),
        paired(succ), "f4", "No clear change",
        inc_v=W[("success", INC)].mean(), diff_inc=paired(succ_i, seed=1))

    cost = (W[("cost", GATED)] - W[("cost", BASE)]).dropna() * 1000
    cost_i = (W[("cost", GATED)] - W[("cost", INC)]).dropna() * 1000
    add("Deployment cost / 1k tasks", W[("cost", BASE)].mean() * 1000,
        W[("cost", GATED)].mean() * 1000, paired(cost), "usd2", "No clear change",
        inc_v=W[("cost", INC)].mean() * 1000, diff_inc=paired(cost_i, seed=1))

    # level is the ratio of means (total cost over total successes, the paper's convention);
    # the interval is over the paired per-unit differences of per-unit ratios
    cps, lvl = {}, {}
    for a in (BASE, GATED):
        cps[a] = (W[("cost", a)] * H + W[("audit", a)]) / (H * W[("success", a)])
        lvl[a] = float((W[("cost", a)].mean() * H + W[("audit", a)].mean())
                       / (H * W[("success", a)].mean()))
    cps[INC] = (W[("cost", INC)] * H + W[("audit", INC)]) / (H * W[("success", INC)])
    lvl[INC] = float((W[("cost", INC)].mean() * H + W[("audit", INC)].mean())
                     / (H * W[("success", INC)].mean()))
    add("Audit-incl. cost / success", lvl[BASE], lvl[GATED],
        paired((cps[GATED] - cps[BASE]).dropna()), "usd4", "No clear change",
        inc_v=lvl[INC], diff_inc=paired((cps[GATED] - cps[INC]).dropna(), seed=1))

    for key, label, concl in [("A", "Valid cost-saving switches / run", "No clear change"),
                              ("B", "Quality-safe, no saving / run", "Reduced"),
                              ("C", "Quality-violating switches / run", "Reduced"),
                              ("total", "Total switches / run", "Reduced")]:
        d = (sw[(key, GATED)] - sw[(key, BASE)]).dropna()
        d_i = (sw[(key, GATED)] - sw[(key, INC)]).dropna()
        add(label, sw[(key, BASE)].mean(), sw[(key, GATED)].mean(), paired(d), "f3", concl,
            inc_v=sw[(key, INC)].mean(), diff_inc=paired(d_i, seed=1))

    T = pd.DataFrame(rows)

    # assertions only; the table is written from the computed columns above
    exp = {"Task success": (0.4878, 0.4907, +0.0029),
           "Audit-incl. cost / success": (0.0675, 0.0643, -0.0020),
           "Deployment cost / 1k tasks": (32.93, 31.54, -1.39),
           "Valid cost-saving switches / run": (0.040, 0.064, +0.024),
           "Quality-safe, no saving / run": (0.200, 0.000, -0.200),
           "Quality-violating switches / run": (0.540, 0.091, -0.449),
           "Total switches / run": (0.780, 0.155, -0.625)}
    for lab, (b, g, d) in exp.items():
        r = T[T.outcome == lab].iloc[0]
        tol = 5e-3 if abs(b) < 10 else 5e-2
        assert abs(r.raw_log - b) < tol, f"{lab} raw {r.raw_log:.4f} != {b}"
        assert abs(r.conf_gated - g) < tol, f"{lab} gated {r.conf_gated:.4f} != {g}"
        assert abs(r["diff"] - d) < tol, f"{lab} diff {r['diff']:+.4f} != {d}"
    assert (T[T.outcome.str.contains("switches / run|no saving")]
            .iloc[:3]["conf_gated"].sum() - T[T.outcome == "Total switches / run"]
            .iloc[0]["conf_gated"]) < 1e-9, "switch rows do not decompose the total"

    # incumbent-column assertions (all switch counts identically zero; levels as published)
    ti = T.set_index("outcome")
    assert abs(ti.loc["Task success", "incumbent"] - 0.4850) < 5e-3
    assert abs(ti.loc["Deployment cost / 1k tasks", "incumbent"] - 35.14) < 5e-2
    for lab in ["Valid cost-saving switches / run", "Quality-safe, no saving / run",
                "Quality-violating switches / run", "Total switches / run"]:
        assert ti.loc[lab, "incumbent"] == 0.0, f"incumbent switch row {lab} not zero"
    # symmetric precisions (the P5 preemptive guard): 5% raw-log, 41% gated
    prec_raw = ti.loc["Valid cost-saving switches / run", "raw_log"] / \
        ti.loc["Total switches / run", "raw_log"]
    prec_gate = ti.loc["Valid cost-saving switches / run", "conf_gated"] / \
        ti.loc["Total switches / run", "conf_gated"]
    assert abs(prec_raw - 0.051) < 0.01, f"raw-log precision {prec_raw:.3f}"
    assert abs(prec_gate - 0.413) < 0.01, f"gated precision {prec_gate:.3f}"
    # the three deployment-cost legs (gated-raw, gated-inc supported, raw-inc unsupported)
    third = paired((W[("cost", BASE)] - W[("cost", INC)]).dropna() * 1000, seed=2)
    assert third[1] < 0 < third[2], f"raw-inc leg unexpectedly supported: {third}"
    gi = (ti.loc["Deployment cost / 1k tasks", "ci_inc_lo"],
          ti.loc["Deployment cost / 1k tasks", "ci_inc_hi"])
    assert gi[1] < 0, f"gated-inc saving no longer supported: {gi}"

    # break-even against the free comparator, with a bootstrap lower bound
    per_req = (W[("cost", BASE)] - W[("cost", GATED)]).dropna().to_numpy(float)
    audit = float(W[("audit", GATED)].mean())
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(per_req, len(per_req), replace=True)) for _ in range(4000)]
    lb = float(np.quantile(boots, 0.05))
    be_point = audit / per_req.mean() if per_req.mean() > 0 else np.inf
    be_lb = audit / lb if lb > 0 else np.inf
    assert abs(be_point - 559) < 25, f"point break-even {be_point:.0f} != 559"
    assert not np.isfinite(be_lb), "a finite 95%-lower-bound break-even now exists"
    # break-even against the incumbent (post-hoc comparator; table + one RQ5 clause only)
    per_inc = (W[("cost", INC)] - W[("cost", GATED)]).dropna().to_numpy(float)
    boots_i = [np.mean(rng.choice(per_inc, len(per_inc), replace=True)) for _ in range(4000)]
    lb_i = float(np.quantile(boots_i, 0.05))
    be_point_i = audit / per_inc.mean()
    be_lb_i = audit / lb_i if lb_i > 0 else np.inf
    assert abs(be_point_i - 216) < 20, f"incumbent point break-even {be_point_i:.0f} != 216"
    assert np.isfinite(be_lb_i) and abs(be_lb_i - 495) < 60, \
        f"incumbent LB break-even {be_lb_i:.0f} != ~495"
    print(f"\nbreak-even vs incumbent: point {be_point_i:.0f}, 95%-LB {be_lb_i:.0f} tasks")
    third_m = paired((W[("cost", BASE)] - W[("cost", INC)]).dropna() * 1000, seed=2)
    print(f"third leg (raw - incumbent): {third_m[0]:+.2f} [{third_m[1]:+.2f}, {third_m[2]:+.2f}] $/1k")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    T.to_csv(args.output, index=False)

    print(f"{'Outcome':34}{'Raw-log':>10}{'Conf.-gated':>13}{'Paired diff [95% CI]':>32}")
    for r in T.itertuples():
        if r.fmt == "f4":
            b, g = f"{r.raw_log:.4f}", f"{r.conf_gated:.4f}"
            d = f"{r.diff:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        elif r.fmt == "usd2":
            b, g = f"${r.raw_log:.2f}", f"${r.conf_gated:.2f}"
            d = f"${r.diff:+.2f} [${r.ci_lo:+.2f}, ${r.ci_hi:+.2f}]"
        elif r.fmt == "usd4":
            b, g = f"${r.raw_log:.4f}", f"${r.conf_gated:.4f}"
            d = f"${r.diff:+.4f} [${r.ci_lo:+.4f}, ${r.ci_hi:+.4f}]"
        else:
            b, g = f"{r.raw_log:.3f}", f"{r.conf_gated:.3f}"
            d = f"{r.diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
        print(f"{r.outcome:34}{b:>10}{g:>13}{d:>32}")
    print(f"{'Economic break-even':34}{'---':>10}{'':>13}"
          f"{f'point {be_point:.0f} tasks; 95%-LB none':>32}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
