#!/usr/bin/env python3
"""Is the LLM judge's error model-level, the way the click signal's error is arm-level?

Referee point: the blindness corollary needs one deployed evaluator's error to be roughly constant
across request types, and the paper establishes that only for clicks on a short-video recommender.
This script runs the same two-way decomposition used on KuaiRand, on the LMArena judge error.

DESIGN NOTE, and the reason this script has two grids. A first pass used the only request-type axis
carried in the scored file, `is_code`, giving a 45 x 2 grid: 44 df on the model axis against 1 on
the category axis. That asymmetry inflates the model share by construction, because a 1-df category
term has almost nowhere to put variance even when request-type dependence is real. The primary
analysis here is therefore a BALANCED 8 x 8 grid with 7 df on each axis, matching KuaiRand's design
exactly, so the two shares are like-for-like. The 45 x 2 grid is retained as a sensitivity and is
reported as such.

Request types come from LMArena's own released prompt annotations (`category_tag`), not from any
clustering of ours: is_code x complexity x instruction-following score >= 3, which is 2x2x2 = 8.

Cell quantity, the analogue of KuaiRand's C_kj = click-mean - like-mean:

    C_{k,j} = judge win-rate of model k in request type j  -  human win-rate of model k in type j

both on the same decisive battles (ties and both_bad excluded from each side).

    python code/run_lmarena_share_test.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# the judge answers "A"/"B"; the released human verdict says "model_a"/"model_b"
SLOT = {"A": "model_a", "B": "model_b", "model_a": "model_a", "model_b": "model_b"}


def request_types(sample_path):
    """J=8 request types from LMArena's released per-prompt annotations."""
    df = pd.read_parquet(sample_path, columns=["id", "category_tag", "is_code", "split"])
    df = df[df.split == "validation"].copy()

    def tag(t):
        return ast.literal_eval(t) if isinstance(t, str) else (t or {})

    d = df.category_tag.map(tag)
    cx = d.map(lambda x: bool(x.get("criteria_v0.1", {}).get("complexity")))
    hi = d.map(lambda x: float(x.get("if_v0.1", {}).get("score") or 0) >= 3)
    code = df.is_code.astype(bool)
    label = ["code" if c else "text" for c in code]
    label = [f"{a}/{'cx' if b else 'simple'}/{'if+' if c else 'if-'}"
             for a, b, c in zip(label, cx, hi)]
    return dict(zip(df.id.astype(str), label))


def winrates(rows, side_key, rtype, models=None):
    """Per (model, request type) win-rate under `side_key`, decisive battles only."""
    won, tot = {}, {}
    for r in rows:
        v = SLOT.get(r.get(side_key))
        if v is None:
            continue
        j = rtype.get(str(r["battle_id"]))
        if j is None:
            continue
        for slot, model in (("model_a", r["model_a"]), ("model_b", r["model_b"])):
            if models is not None and model not in models:
                continue
            key = (model, j)
            tot[key] = tot.get(key, 0) + 1
            if v == slot:
                won[key] = won.get(key, 0) + 1
    return {k: won.get(k, 0) / n for k, n in tot.items()}, tot


def decompose(C, ok):
    """Two-way additive decomposition, identical in form to run_kuairand_validity.c_decomp."""
    K, J = C.shape
    Cc = np.where(ok, C, 0.0)
    grand = Cc[ok].mean()
    a = np.array([Cc[k, ok[k]].mean() - grand if ok[k].any() else 0.0 for k in range(K)])
    b = np.array([Cc[ok[:, j], j].mean() - grand if ok[:, j].any() else 0.0 for j in range(J)])
    fitted = grand + a[:, None] + b[None, :]
    ss = float(((Cc - grand)[ok] ** 2).sum())
    if ss == 0:
        return None
    return dict(share_model=float(((a[:, None] * np.ones((K, J)))[ok] ** 2).sum()) / ss,
                share_type=float(((b[None, :] * np.ones((K, J)))[ok] ** 2).sum()) / ss,
                share_resid=float(((Cc - fitted)[ok] ** 2).sum()) / ss,
                model_fx=a.tolist(), type_fx=b.tolist())


def grid(rows, rtype, models, types, min_cell):
    jr, jn = winrates(rows, "judge_winner_original", rtype, set(models))
    hr, hn = winrates(rows, "human_winner", rtype, set(models))
    C = np.full((len(models), len(types)), np.nan)
    for (m, j) in set(jr) & set(hr):
        if jn[(m, j)] >= min_cell and hn[(m, j)] >= min_cell:
            C[models.index(m), types.index(j)] = jr[(m, j)] - hr[(m, j)]
    return C, ~np.isnan(C)


def run(rows, rtype, models, types, min_cell, n_boot, seed=0, label=""):
    C, ok = grid(rows, rtype, models, types, min_cell)
    dec = decompose(C, ok)
    print(f"\n{label}: {len(models)} models x {len(types)} request types, "
          f"{int(ok.sum())} cells with n >= {min_cell} on both sides")
    if dec is None:
        print("   grid too thin to decompose")
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = [rows[i] for i in rng.choice(len(rows), len(rows), replace=True)]
        Cb, okb = grid(s, rtype, models, types, min_cell)
        if okb.sum() >= 8:
            d = decompose(Cb, okb)
            if d:
                boot.append(d["share_model"])
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan,) * 2

    # permutation null: break the model signal, keep the design
    perm = []
    for _ in range(400):
        Cp = C.copy()
        for j in range(len(types)):
            col = np.where(ok[:, j])[0]
            if len(col) > 1:
                Cp[col, j] = Cp[rng.permutation(col), j]
        d = decompose(Cp, ok)
        if d:
            perm.append(d["share_model"])
    pmed = float(np.median(perm)) if perm else float("nan")

    # can this grid SEE request-type dependence? inject a type-varying term and re-fit.
    inj = C + (rng.standard_normal(len(types)) * float(np.nanstd(C)))[None, :]
    dinj = decompose(np.where(ok, inj, np.nan), ok)

    print(f"   model main effects   {dec['share_model']:.3f}   bootstrap [{lo:.3f}, {hi:.3f}]")
    print(f"   type  main effects   {dec['share_type']:.3f}")
    print(f"   residual             {dec['share_resid']:.3f}")
    print(f"   permutation-null median model share {pmed:.3f}")
    if dinj:
        print(f"   with an injected type-varying term, type share -> {dinj['share_type']:.3f} "
              f"(so the grid can resolve it)")
    return dict(models=len(models), types=len(types), cells=int(ok.sum()),
                share_model=dec["share_model"], share_model_ci=[lo, hi],
                share_type=dec["share_type"], share_resid=dec["share_resid"],
                permutation_null_median=pmed,
                injected_type_share=(dinj or {}).get("share_type"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation",
                    default=os.path.join(ROOT, "results", "lmarena_judge_validation.json"))
    ap.add_argument("--sample", default=os.path.join(ROOT, "data", "lmarena_sample.parquet"))
    ap.add_argument("--n-models", type=int, default=8)
    ap.add_argument("--min-cell", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lmarena_share_test.json"))
    args = ap.parse_args()

    rows = json.load(open(args.validation))["rows"]
    rtype = request_types(args.sample)
    print(f"read {len(rows):,} battles; {len(set(rtype.values()))} request types from "
          f"LMArena's released prompt annotations")

    support = {}
    for r in rows:
        for m in (r["model_a"], r["model_b"]):
            support[m] = support.get(m, 0) + 1
    top = [m for m, _ in sorted(support.items(), key=lambda kv: -kv[1])[:args.n_models]]
    types = sorted(set(rtype.values()))

    primary = run(rows, rtype, top, types, args.min_cell, args.bootstrap,
                  label="PRIMARY, balanced grid")

    # sensitivity: the unbalanced axis the first pass used
    allm = sorted(support)
    flat = {k: ("code" if v.startswith("code") else "text") for k, v in rtype.items()}
    sens = run(rows, flat, allm, ["code", "text"], 30, args.bootstrap,
               label="SENSITIVITY, unbalanced 2-level axis")

    hw = [r["human_winner"] for r in rows]
    base = max(hw.count(v) for v in set(hw)) / len(hw)
    print(f"\n  human-verdict majority-class base rate: {base:.4f}")

    with open(args.out, "w") as fh:
        json.dump(dict(n_battles=len(rows), request_types=types, top_models=top,
                       primary_balanced=primary, sensitivity_unbalanced=sens,
                       human_majority_base_rate=base), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
