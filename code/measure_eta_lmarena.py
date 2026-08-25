"""An LLM-native measurement of the evaluator-error spread eta, from LMArena.

`measure_eta.py` measures eta on KuaiRand, where clicking stands in for liking. Clicking and liking
are further apart than an LLM judge and its target plausibly are, so that figure bounds rather than
transfers. LMArena supplies the judge-native counterpart: the same battles carry a human verdict,
which is the designated outcome, and a `gpt-4o-mini` verdict, which is the proxy an operator would
log.

The paper's mu is an absolute per-cell quality; LMArena is pairwise, so the natural analogue of a
model's quality is its win rate against the field:

    mu_ref_k = P(human picks k  | k appears)          designated outcome
    mu_obs_k = P(judge picks k  | k appears)          proxy
    C_k      = mu_obs_k - mu_ref_k                    evaluator error for model k
    eta_j    = max_k C_kj - min_k C_kj                spread within workload j

Workloads are the only context the corpus carries, `is_code`. Ties and both-bad verdicts are scored
0.5 for both models, so a judge that ties more often than humans do is not counted as favouring
anyone.

Position bias is reported both ways: `original` uses the judge's first-order verdict, which is what
an operator who scores once would log, and `order-averaged` averages the original and reversed
verdicts, which removes position effects and isolates model preference.

Usage:
    python code/measure_eta_lmarena.py
"""
import json, os, sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SPLITS = ("calibration", "validation", "test")
MIN_APPEARANCES = 200


def load_rows():
    rows = []
    for sp in SPLITS:
        p = os.path.join(RES, "lmarena_judge_%s.json" % sp)
        if os.path.exists(p):
            rows += json.load(open(p))["rows"]
    return rows


def _score(winner, side):
    """1.0 if `side` won, 0.0 if it lost, 0.5 for a tie or a double-bad verdict."""
    if winner is None:
        return None
    w = str(winner).lower()
    if w in ("tie", "both_bad", "none"):
        return 0.5
    if w in ("model_a", "a"):
        return 1.0 if side == "a" else 0.0
    if w in ("model_b", "b"):
        return 1.0 if side == "b" else 0.0
    return None


def win_rates(rows, verdict_fields, ctx=None):
    """Per-model win rate under a verdict, averaging over the given fields."""
    num, den = {}, {}
    for r in rows:
        if ctx is not None and bool(r.get("is_code")) != ctx:
            continue
        for side, model in (("a", r["model_a"]), ("b", r["model_b"])):
            vals = [_score(r.get(f), side) for f in verdict_fields]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            num[model] = num.get(model, 0.0) + float(np.mean(vals))
            den[model] = den.get(model, 0) + 1
    return {m: num[m] / den[m] for m in num if den[m] > 0}, den


def main():
    rows = load_rows()
    print("battles: %d across %d splits" % (len(rows), len(SPLITS)))

    schemes = {
        "original (single order, what an operator logs)": ["judge_winner_original"],
        "order-averaged (position bias removed)": ["judge_winner_original", "judge_winner_reversed"],
    }
    out = {}
    for label, fields in schemes.items():
        print("\n=== %s ===" % label)
        print("%-14s %7s %9s %9s %9s" % ("workload", "models", "min C", "max C", "eta"))
        etas = []
        for ctx, name in ((False, "non-code"), (True, "code")):
            ref, den = win_rates(rows, ["human_winner"], ctx)
            obs, _ = win_rates(rows, fields, ctx)
            ks = [m for m in ref if m in obs and den.get(m, 0) >= MIN_APPEARANCES]
            if len(ks) < 2:
                print("%-14s %7d %9s %9s %9s" % (name, len(ks), "--", "--", "--"))
                continue
            C = np.array([obs[m] - ref[m] for m in ks])
            eta = float(C.max() - C.min())
            etas.append(eta)
            print("%-14s %7d %9.4f %9.4f %9.4f" % (name, len(ks), C.min(), C.max(), eta))
        if etas:
            print("eta across workloads: median %.4f" % float(np.median(etas)))
            out[label] = dict(eta_by_workload=etas, eta_median=float(np.median(etas)))

    # pooled, ignoring the workload split, for a single headline figure
    print("\n=== pooled over both workloads ===")
    for label, fields in schemes.items():
        ref, den = win_rates(rows, ["human_winner"])
        obs, _ = win_rates(rows, fields)
        ks = [m for m in ref if m in obs and den.get(m, 0) >= MIN_APPEARANCES]
        C = np.array([obs[m] - ref[m] for m in ks])
        print("%-46s models %2d   C in [%+.4f, %+.4f]   eta %.4f   sd %.4f"
              % (label, len(ks), C.min(), C.max(), C.max() - C.min(), C.std(ddof=1)))
        out.setdefault("pooled", {})[label] = dict(
            n_models=len(ks), C_min=float(C.min()), C_max=float(C.max()),
            eta=float(C.max() - C.min()), C_sd=float(C.std(ddof=1)))

    json.dump(out, open(os.path.join(RES, "eta_lmarena.json"), "w"), indent=1)
    print("\nwrote %s" % os.path.join(RES, "eta_lmarena.json"))


if __name__ == "__main__":
    main()
