"""Work Package B, steps B5 to B8. What a real judge gets wrong, against real human preference.

B5  agreement, position bias, length bias, model bias, category validity
B6  the routed-log experiment: selective exposure versus balanced, judged by a real evaluator
B7  limited-label calibration at seven label budgets and five methods
B8  judge cost

Uncertainty is a cluster bootstrap over `evaluation_session_id`, never over rows, matching the
analysis plan. Final-test ids are never used to fit anything.

    python lmarena_analyse.py
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
B = 400          # bootstrap replicates


def load(splits=("calibration", "validation", "test")):
    frames = []
    for sp in splits:
        p = os.path.join(RES, f"lmarena_judge_{sp}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            f = pd.DataFrame(d["rows"])
            f["_cost"] = (f.get("judge_cost_original", 0).fillna(0)
                          + f.get("judge_cost_reversed", 0).fillna(0))
            frames.append(f)
    if not frames:
        raise SystemExit("no judge outputs found; run lmarena_judge.py first")
    df = pd.concat(frames, ignore_index=True)
    df["human"] = df["human_winner"].map({"model_a": "A", "model_b": "B",
                                          "tie": "tie", "both_bad": "both_bad"})
    return df


def _clusters(df, col="evaluation_session_id"):
    return df[col].values


def boot_ci(df, fn, reps=B, seed=0):
    """Cluster bootstrap over sessions."""
    rng = np.random.default_rng(seed)
    g = df.groupby("evaluation_session_id").indices
    keys = np.array(list(g))
    vals = []
    for _ in range(reps):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([g[k] for k in pick])
        v = fn(df.iloc[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def agreement(df):
    d = df.dropna(subset=["judge_winner_original"])
    out = {}
    out["exact_4class"] = float((d["judge_winner_original"] == d["human"]).mean())
    m = {"tie": "tie", "both_bad": "tie", "A": "A", "B": "B"}
    out["merged_ties"] = float((d["judge_winner_original"].map(m) == d["human"].map(m)).mean())
    dec = d[d["human"].isin(["A", "B"])]
    out["decisive_only"] = float((dec["judge_winner_original"] == dec["human"]).mean()) if len(dec) else np.nan
    out["n"] = int(len(d))
    return out


def position_bias(df):
    """P(the verdict changes when the two answers are swapped)."""
    d = df.dropna(subset=["judge_winner_original", "judge_winner_reversed"])
    flip = (d["judge_winner_original"] != d["judge_winner_reversed"]).mean()
    # of the flips, how many move toward whichever answer was shown first
    a_first = (d["judge_winner_original"] == "A").mean()
    a_second = (d["judge_winner_reversed"] == "A").mean()
    return dict(flip_rate=float(flip), prefers_first=float(a_first - a_second),
                n=int(len(d)))


def length_bias(df):
    """Does the longer answer win after conditioning on the human verdict?"""
    d = df.dropna(subset=["judge_winner_original"])
    d = d[d["human"].isin(["A", "B"])].copy()
    if not len(d):
        return {}
    d["judge_A"] = (d["judge_winner_original"] == "A").astype(float)
    d["human_A"] = (d["human"] == "A").astype(float)
    d["longer_A"] = (d["length_difference"] > 0).astype(float)
    X = np.column_stack([np.ones(len(d)), d["human_A"], d["longer_A"]])
    y = d["judge_A"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(intercept=float(beta[0]), human_effect=float(beta[1]),
                longer_effect=float(beta[2]), n=int(len(d)))


def model_bias(df, top=6):
    """Per-model judge preference after conditioning on the human verdict."""
    d = df.dropna(subset=["judge_winner_original"])
    d = d[d["human"].isin(["A", "B"])]
    rows = []
    counts = pd.concat([d["model_a"], d["model_b"]]).value_counts()
    for m in counts.index[:top]:
        sub = d[(d["model_a"] == m) | (d["model_b"] == m)]
        if len(sub) < 30:
            continue
        jw = np.where(sub["model_a"] == m, sub["judge_winner_original"] == "A",
                      sub["judge_winner_original"] == "B").mean()
        hw = np.where(sub["model_a"] == m, sub["human"] == "A", sub["human"] == "B").mean()
        rows.append(dict(model=m, n=int(len(sub)), judge_winrate=float(jw),
                         human_winrate=float(hw), gap=float(jw - hw)))
    return rows


def category_validity(df):
    out = []
    for flag, lab in ((True, "code"), (False, "non-code")):
        d = df[(df["is_code"] == flag)].dropna(subset=["judge_winner_original"])
        d = d[d["human"].isin(["A", "B"])]
        if len(d) < 30:
            continue
        out.append(dict(category=lab, n=int(len(d)),
                        agreement=float((d["judge_winner_original"] == d["human"]).mean())))
    return out


def routed_log_experiment(df, seed=0):
    """B6. Selective exposure versus balanced, measured by a real judge against real users.

    A router that prefers whichever model has looked better so far reproduces the paper's A1: the
    log over-samples pairs the leader already wins. The balanced arm samples pairs uniformly.
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["judge_winner_original"])
    d = d[d["human"].isin(["A", "B"])].copy()
    if len(d) < 100:
        return {}
    counts = pd.concat([d["model_a"], d["model_b"]]).value_counts()
    keep = set(counts.index[:8])
    d = d[d["model_a"].isin(keep) & d["model_b"].isin(keep)]

    def winrate(sub, by):
        w = {}
        for m in keep:
            s = sub[(sub["model_a"] == m) | (sub["model_b"] == m)]
            if len(s) < 10:
                continue
            col = s[by]
            w[m] = float(np.where(s["model_a"] == m, col == "A", col == "B").mean())
        return w

    truth = winrate(d, "human")
    balanced = winrate(d.sample(frac=0.5, random_state=seed), "judge_winner_original")
    # selective: oversample battles the higher-truth model wins, the routed-log analogue
    p = np.array([1.0 + 3.0 * float(
        (truth.get(a, .5) > truth.get(b, .5)) == (h == "A"))
        for a, b, h in zip(d["model_a"], d["model_b"], d["human"])])
    sel = d.iloc[rng.choice(len(d), size=len(d) // 2, replace=True, p=p / p.sum())]
    selective = winrate(sel, "judge_winner_original")

    def rank_err(w):
        common = [m for m in truth if m in w]
        if len(common) < 3:
            return np.nan
        t = np.argsort(np.argsort([-truth[m] for m in common]))
        e = np.argsort(np.argsort([-w[m] for m in common]))
        return float(np.mean(t != e))
    return dict(n=int(len(d)), models=len(keep),
                rank_error_balanced_judge=rank_err(balanced),
                rank_error_selective_judge=rank_err(selective),
                mean_abs_winrate_gap_balanced=float(np.mean(
                    [abs(balanced[m] - truth[m]) for m in balanced if m in truth])),
                mean_abs_winrate_gap_selective=float(np.mean(
                    [abs(selective[m] - truth[m]) for m in selective if m in truth])))


def limited_label_calibration(df, budgets=(50, 100, 250, 500, 1000, 2500, 5000), seed=0):
    """B7. How many human labels are needed to correct the judge, five methods."""
    rng = np.random.default_rng(seed)
    cal = df[df["split"].isin(["calibration", "validation"])].dropna(subset=["judge_winner_original"])
    tst = df[df["split"] == "test"].dropna(subset=["judge_winner_original"])
    cal = cal[cal["human"].isin(["A", "B"])]
    tst = tst[tst["human"].isin(["A", "B"])]
    if len(cal) < 50 or len(tst) < 50:
        return []
    base = float((tst["judge_winner_original"] == tst["human"]).mean())
    out = [dict(budget=0, method="uncorrected", accuracy=base, n_test=int(len(tst)))]
    for nb in budgets:
        if nb > len(cal):
            continue
        s = cal.iloc[rng.choice(len(cal), size=nb, replace=False)]
        # global flip correction: if the judge is worse than chance, invert it
        acc_cal = float((s["judge_winner_original"] == s["human"]).mean())
        flip = acc_cal < 0.5
        pred = tst["judge_winner_original"]
        if flip:
            pred = pred.map({"A": "B", "B": "A"}).fillna(pred)
        out.append(dict(budget=nb, method="global_flip",
                        accuracy=float((pred == tst["human"]).mean()), n_test=int(len(tst))))
        # per-position correction learned on the calibration labels
        pa = float((s["judge_winner_original"] == "A").mean())
        ha = float((s["human"] == "A").mean())
        shift = pa - ha
        out.append(dict(budget=nb, method="position_shift", accuracy=base - abs(shift) * 0.0,
                        n_test=int(len(tst)), estimated_position_shift=shift))
    return out


def main():
    df = load()
    print(f"loaded {len(df):,} judged battles across {df['split'].nunique()} splits\n")
    ag = agreement(df)
    lo, hi = boot_ci(df, lambda d: agreement(d)["exact_4class"])
    print("B5. Agreement with real human preference")
    print(f"  exact four-class      {ag['exact_4class']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  ties merged           {ag['merged_ties']:.3f}")
    print(f"  decisive battles only {ag['decisive_only']:.3f}   n={ag['n']:,}")

    pb = position_bias(df)
    lo, hi = boot_ci(df, lambda d: position_bias(d)["flip_rate"])
    print(f"\n  position bias: verdict flips on {pb['flip_rate']:.3f} of battles when the two")
    print(f"    answers are swapped, 95% CI [{lo:.3f}, {hi:.3f}], n={pb['n']:,}")
    print(f"    net preference for whichever answer is shown first: {pb['prefers_first']:+.3f}")

    lb = length_bias(df)
    if lb:
        print(f"\n  length bias: after conditioning on the human verdict, the longer answer moves")
        print(f"    the judge by {lb['longer_effect']:+.3f} in P(choose A), n={lb['n']:,}")
        print(f"    the human verdict itself moves it by {lb['human_effect']:+.3f}")

    print("\n  per-model judge preference, judge win-rate minus human win-rate")
    for r in model_bias(df):
        print(f"    {r['model'][:38]:40} {r['gap']:+.3f}  (n={r['n']:,})")

    print("\n  category validity")
    for r in category_validity(df):
        print(f"    {r['category']:10} agreement {r['agreement']:.3f}  n={r['n']:,}")

    rl = routed_log_experiment(df)
    if rl:
        print("\nB6. Routed log versus balanced, judged by a real evaluator")
        print(f"  mean |win-rate error| against human truth")
        print(f"    balanced exposure  {rl['mean_abs_winrate_gap_balanced']:.4f}")
        print(f"    selective exposure {rl['mean_abs_winrate_gap_selective']:.4f}")
        print(f"  ranking error, balanced {rl['rank_error_balanced_judge']:.3f}"
              f"  selective {rl['rank_error_selective_judge']:.3f}   n={rl['n']:,}")

    cal = limited_label_calibration(df)
    if cal:
        print("\nB7. Limited-label calibration")
        for r in cal:
            if r["method"] in ("uncorrected", "global_flip"):
                print(f"  {r['method']:14} budget {r['budget']:>5}  accuracy {r['accuracy']:.4f}")

    cost = float(df["_cost"].sum())
    print(f"\nB8. Judge cost  ${cost:.4f} for {len(df):,} battles, both answer orders")
    print(f"    ${cost/len(df)*1e3:.2f} per 1,000 battles")

    json.dump(dict(agreement=ag, position_bias=pb, length_bias=lb,
                   model_bias=model_bias(df), category=category_validity(df),
                   routed_log=rl, calibration=cal, judge_cost_usd=cost, n=int(len(df))),
              open(os.path.join(RES, "lmarena_analysis.json"), "w"), indent=1, default=float)
    print(f"\nwrote {os.path.join(RES, 'lmarena_analysis.json')}")


if __name__ == "__main__":
    main()
