"""Priority 4 (section 5): joint routing-versus-measurement budget on stored LMArena outputs.

Routing-audit item  = reveal a stored gpt-4o-mini judgment for a battle outside the selective log.
Measurement-label item = reveal the public human vote.
Everything is replayed from saved files; no new API calls.
"""
import json, os, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "p4_joint_budget.json")

BUDGETS = [50, 100, 250, 500, 1000, 2500]
SPLITS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
REPEATS = 100
LABEL_COSTS = [0.25, 1.0, 5.0, 20.0]


def load():
    fr = []
    for sp in ("calibration", "validation", "test"):
        p = os.path.join(RES, f"lmarena_judge_{sp}.json")
        if os.path.exists(p):
            fr.append(pd.DataFrame(json.load(open(p))["rows"]))
    d = pd.concat(fr, ignore_index=True)
    d["human"] = d.human_winner.map({"model_a": "A", "model_b": "B"})
    d = d.dropna(subset=["human", "judge_winner_original"]).copy()
    d["y"] = (d.human == "A").astype(int)
    d["judge_A"] = (d.judge_winner_original == "A").astype(int)
    d["flip"] = (d.judge_winner_original != d.judge_winner_reversed).astype(int)
    d["conf"] = pd.to_numeric(d.get("judge_confidence_original"), errors="coerce").fillna(0.5)
    d["lendiff"] = pd.to_numeric(d.length_difference, errors="coerce").fillna(0.0)
    d["judge_cost"] = (pd.to_numeric(d.get("judge_cost_original"), errors="coerce").fillna(0)
                       + pd.to_numeric(d.get("judge_cost_reversed"), errors="coerce").fillna(0))
    return d


FEATS = ["judge_A", "conf", "flip", "lendiff"]


def fit_predict(method, tr, te):
    """Error of the deployed decision rule against the human vote."""
    if method == "none" or len(tr) < 10 or tr.y.nunique() < 2:
        return float((te.judge_A != te.y).mean())
    if method == "offset":
        # per-model additive offset on the judge verdict
        off = tr.groupby("model_a").apply(lambda g: g.y.mean() - g.judge_A.mean()).to_dict()
        adj = te.judge_A + te.model_a.map(off).fillna(0.0)
        return float(((adj > 0.5).astype(int) != te.y).mean())
    if method == "logistic":
        m = LogisticRegression(max_iter=500).fit(tr[FEATS], tr.y)
        return float(((m.predict_proba(te[FEATS])[:, 1] > 0.5).astype(int) != te.y).mean())
    if method == "isotonic":
        s = tr.judge_A + 0.01 * tr.conf
        iso = IsotonicRegression(out_of_bounds="clip").fit(s, tr.y)
        p = iso.predict(te.judge_A + 0.01 * te.conf)
        return float(((p > 0.5).astype(int) != te.y).mean())
    raise ValueError(method)


def main():
    d = load()
    rng = np.random.default_rng(0)
    judge_item_cost = float(d.judge_cost.mean())
    print(f"  {len(d):,} decisive battles; mean stored judge cost per item ${judge_item_cost:.5f}")

    te = d.sample(n=min(4000, len(d)), random_state=1)
    pool = d.drop(te.index)

    rows = []
    for B, (fr_route, fr_label) in itertools.product(BUDGETS, SPLITS):
        n_label = int(round(B * fr_label))
        errs = {m: [] for m in ("none", "offset", "logistic", "isotonic")}
        for r in range(REPEATS if B <= 500 else 25):
            lab = pool.sample(n=min(n_label, len(pool)), random_state=r) if n_label else pool.iloc[:0]
            for m in errs:
                errs[m].append(fit_predict(m, lab, te))
        for m, v in errs.items():
            rows.append(dict(budget=B, route_frac=fr_route, label_frac=fr_label, method=m,
                             wrong_decision_rate=float(np.mean(v)), se=float(np.std(v, ddof=1) / max(1, np.sqrt(len(v)))),
                             n_label=n_label, n_route=B - n_label))
    df = pd.DataFrame(rows)

    print(f"\n  wrong deployment-decision rate against the human vote (logistic calibration)")
    piv = df[df.method == "logistic"].pivot_table(index="budget", columns="label_frac",
                                                  values="wrong_decision_rate")
    print(piv.round(4).to_string().replace("\n", "\n  ").rjust(0))

    # adaptive allocation, 50-item batches, value per dollar
    adaptive = []
    for lc in LABEL_COSTS:
        spent_r, spent_l, n_l = 0.0, 0.0, 0
        err = fit_predict("none", pool.iloc[:0], te)
        hist = []
        for step in range(20):
            cand = pool.sample(n=min(n_l + 50, len(pool)), random_state=step)
            e_lab = fit_predict("logistic", cand, te)
            gain_label = max(0.0, err - e_lab) / (50 * lc)
            gain_route = max(0.0, err - err) / max(1e-9, 50 * judge_item_cost)   # stored judge adds no new signal on the test split
            if gain_label >= gain_route:
                n_l += 50; spent_l += 50 * lc; err = e_lab
            else:
                spent_r += 50 * judge_item_cost
            hist.append(dict(step=step, err=err, n_label=n_l, spent_label=spent_l, spent_route=spent_r))
        share = spent_l / max(1e-9, spent_l + spent_r)
        adaptive.append(dict(label_cost_usd=lc, final_error=err, n_labels=n_l,
                             spent_label=spent_l, spent_route=spent_r, label_budget_share=share))
        print(f"  adaptive, label ${lc:>5}: final error {err:.4f}, {n_l} labels, "
              f"label share of spend {share:.2f}")

    json.dump(dict(judge_item_cost_usd=judge_item_cost,
                   grid=json.loads(df.to_json(orient="records")),
                   adaptive=adaptive), open(OUT, "w"), indent=1)
    print(f"\n  wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
