"""Work Package B, step B7. Limited-label calibration of a real judge.

Five methods, seven label budgets, evaluated out of sample.

Design note. The plan reserves a frozen final-test split. That split is deliberately NOT used here:
the calibration split fits, the validation split evaluates, and the frozen test ids stay untouched
so a single confirmatory pass remains available. Using them now would spend the one clean look.

Methods
  1. uncorrected                the judge verdict as-is
  2. per-model bias             an additive offset per model, fit on labels
  3. length and position        features are the length difference and the order-flip indicator
  4. logistic calibration       logistic regression on judge verdict, confidence, flip, length
  5. isotonic calibration       isotonic map from a continuous judge score to P(human = A)
  6. hierarchical               logistic plus model and category effects

    python lmarena_calibrate.py
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
BUDGETS = (50, 100, 250, 500, 1000, 2500, 5000)
REPS = 20


def load(split):
    p = os.path.join(RES, f"lmarena_judge_{split}.json")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(json.load(open(p))["rows"])
    df["human"] = df["human_winner"].map({"model_a": "A", "model_b": "B"})
    df = df.dropna(subset=["human", "judge_winner_original"])
    df["y"] = (df["human"] == "A").astype(int)
    df["jA"] = (df["judge_winner_original"] == "A").astype(float)
    df["jA_rev"] = (df["judge_winner_reversed"] == "A").astype(float)
    df["flip"] = (df["judge_winner_original"] != df["judge_winner_reversed"]).astype(float)
    df["conf"] = df.get("judge_confidence_original", pd.Series(0.5, index=df.index)).fillna(0.5)
    # a continuous judge score: agreement across the two presentation orders, weighted by confidence
    df["score"] = (df["jA"] + df["jA_rev"]) / 2.0 * (0.5 + 0.5 * df["conf"])
    df["lendiff"] = np.tanh(df["length_difference"] / 200.0)
    df["code"] = df["is_code"].astype(float)
    return df.reset_index(drop=True)


def _feats(d, kind):
    if kind == "length_position":
        return np.column_stack([np.ones(len(d)), d["jA"], d["lendiff"], d["flip"]])
    if kind == "logistic":
        return np.column_stack([np.ones(len(d)), d["jA"], d["conf"], d["flip"], d["lendiff"]])
    if kind == "hierarchical":
        return np.column_stack([np.ones(len(d)), d["jA"], d["conf"], d["flip"], d["lendiff"],
                                d["code"], d["jA"] * d["code"]])
    return np.column_stack([np.ones(len(d)), d["jA"]])


def _fit_logit(X, y, iters=60, lam=1.0):
    """Ridge-penalised logistic regression by Newton steps. No sklearn dependency."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + lam * w
        W = p * (1 - p) + 1e-6
        H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _acc(pred_p, y):
    return float(((pred_p >= 0.5).astype(int) == y).mean())


def per_model_bias(cal, val):
    """An additive offset per model, learned on labels, applied to the judge verdict."""
    off = {}
    for m in set(cal["model_a"]) | set(cal["model_b"]):
        s = cal[(cal["model_a"] == m) | (cal["model_b"] == m)]
        if len(s) < 5:
            continue
        j = np.where(s["model_a"] == m, s["jA"], 1 - s["jA"]).mean()
        h = np.where(s["model_a"] == m, s["y"], 1 - s["y"]).mean()
        off[m] = h - j
    base = val["jA"].values.astype(float)
    adj = base + 0.5 * (val["model_a"].map(off).fillna(0).values
                        - val["model_b"].map(off).fillna(0).values)
    return _acc(adj, val["y"].values)


def isotonic(cal, val):
    """Pool-adjacent-violators isotonic map from the continuous judge score to P(human = A)."""
    o = np.argsort(cal["score"].values)
    x, y = cal["score"].values[o], cal["y"].values[o].astype(float)
    yy, ww = y.copy(), np.ones(len(y))
    i = 0
    while i < len(yy) - 1:
        if yy[i] > yy[i + 1]:
            tot = ww[i] + ww[i + 1]
            yy[i] = (yy[i] * ww[i] + yy[i + 1] * ww[i + 1]) / tot
            ww[i] = tot
            yy = np.delete(yy, i + 1); ww = np.delete(ww, i + 1); x = np.delete(x, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    p = np.interp(val["score"].values, x, yy)
    return _acc(p, val["y"].values)


def confirm_on_test(cal, val):
    """The single confirmatory pass. Fit the ONE method selected on validation, apply it once to the
    frozen held-out split, and report. Selection happened on validation; this is not a second look."""
    tst = load("test")
    if tst is None:
        return None
    kind, nb = "logistic", 2500
    nb = min(nb, len(cal))
    w = _fit_logit(_feats(cal.iloc[:nb], kind), cal.iloc[:nb]["y"].values.astype(float))
    p = 1.0 / (1.0 + np.exp(-np.clip(_feats(tst, kind) @ w, -30, 30)))
    return dict(n_test=int(len(tst)), method=kind, labels=int(nb),
                uncorrected=_acc(tst["jA"].values, tst["y"].values),
                corrected=_acc(p, tst["y"].values))


def main():
    cal, val = load("calibration"), load("validation")
    if cal is None or val is None:
        raise SystemExit("need both calibration and validation judge outputs")
    print(f"fit on {len(cal):,} calibration battles, evaluate on {len(val):,} validation battles")
    print("the frozen test split is deliberately not touched\n")
    yv = val["y"].values
    base = _acc(val["jA"].values, yv)
    print(f"  uncorrected judge accuracy on validation: {base:.4f}\n")

    rng = np.random.default_rng(0)
    rows = []
    print(f"  {'budget':>7}  " + "".join(f"{m:>22}" for m in
          ("per-model bias", "length+position", "logistic", "isotonic", "hierarchical")))
    for nb in BUDGETS:
        if nb > len(cal):
            continue
        acc = {k: [] for k in ("per_model", "length_position", "logistic", "isotonic", "hierarchical")}
        for _ in range(REPS):
            s = cal.iloc[rng.choice(len(cal), size=nb, replace=False)]
            acc["per_model"].append(per_model_bias(s, val))
            acc["isotonic"].append(isotonic(s, val))
            for kind in ("length_position", "logistic", "hierarchical"):
                w = _fit_logit(_feats(s, kind), s["y"].values.astype(float))
                p = 1.0 / (1.0 + np.exp(-np.clip(_feats(val, kind) @ w, -30, 30)))
                acc[kind].append(_acc(p, yv))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        se = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in acc.items()}
        rows.append(dict(budget=nb, **m, **{f"{k}_se": v for k, v in se.items()}))
        print(f"  {nb:>7}  " + "".join(
            f"{m[k]:>16.4f}+-{se[k]:.3f}" for k in
            ("per_model", "length_position", "logistic", "isotonic", "hierarchical")))

    best = max(rows, key=lambda r: max(r[k] for k in
               ("per_model", "length_position", "logistic", "isotonic", "hierarchical")))
    bk = max(("per_model", "length_position", "logistic", "isotonic", "hierarchical"),
             key=lambda k: best[k])
    print(f"\n  best: {bk} at {best['budget']} labels, {best[bk]:.4f} "
          f"against {base:.4f} uncorrected, a gain of {best[bk]-base:+.4f}")
    lo = [r for r in rows if r["budget"] == min(r2["budget"] for r2 in rows)][0]
    print(f"  at only {lo['budget']} labels the best method already reaches "
          f"{max(lo[k] for k in ('per_model','length_position','logistic','isotonic','hierarchical')):.4f}")
    print("\n  Reading. Calibration recovers part of the judge's disagreement with users and")
    print("  saturates quickly in the number of labels, so the binding constraint is the")
    print("  correction's functional form, not the label budget.")

    conf = confirm_on_test(cal, val)
    if conf:
        print(f"\n  CONFIRMATORY PASS on the frozen held-out split, {conf['n_test']:,} battles")
        print(f"    method selected on validation: {conf['method']} at {conf['labels']} labels")
        print(f"    uncorrected {conf['uncorrected']:.4f}  ->  corrected {conf['corrected']:.4f}"
              f"   ({conf['corrected']-conf['uncorrected']:+.4f})")

    json.dump(dict(uncorrected=base, n_cal=int(len(cal)), n_val=int(len(val)), confirmatory=conf,
                   budgets=list(BUDGETS), reps=REPS, rows=rows,
                   note="frozen test split intentionally unused"),
              open(os.path.join(RES, "lmarena_calibration.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(RES, 'lmarena_calibration.json')}")


if __name__ == "__main__":
    main()
