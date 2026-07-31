"""Table 2: measurement validity of real evaluators, which no randomized rerun can repair.

Every number is read from the stored evaluator results; hard-coded values appear only in
assertions. Emits a CSV and the LaTeX body rows.

Usage:
    python code/make_table2_measurement.py --results results \
      --output results/table2_measurement.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def load(res, name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    A = load(args.results, "lmarena_analysis.json")
    C = load(args.results, "lmarena_judge_compare.json")
    H = load(args.results, "helpsteer_judge.json")

    rows = []

    # 1. gpt-4o-mini over every human-voted LMArena battle
    rows.append(dict(
        evaluator="\\texttt{gpt-4o-mini}", corpus="LMArena, all human-voted",
        n=int(A["agreement"]["n"]),
        agree=float(A["agreement"]["exact_4class"]),
        flip=float(A["position_bias"]["flip_rate"]),
        longer=float(A["length_bias"]["longer_effect"])))

    # 2-3. two evaluators scored on the SAME battles
    rows.append(dict(
        evaluator="\\texttt{gpt-4o-mini}", corpus="LMArena, shared subset",
        n=int(C["n_shared"]),
        agree=float(C["exact_4class"]["api"]),
        flip=float(C["flip_rate"]["api"]),
        longer=float(C["longer_effect"]["api"])))
    rows.append(dict(
        evaluator="\\texttt{Qwen2.5-7B} (local)", corpus="LMArena, shared subset",
        n=int(C["n_shared"]),
        agree=float(C["exact_4class"]["local"]),
        flip=float(C["flip_rate"]["local"]),
        longer=float(C["longer_effect"]["local"])))

    # 4. gpt-4o-mini against human correctness on HelpSteer2
    hr = pd.DataFrame(H["rows"])
    need = {"correctness", "judge_overall_score", "prompt"}
    if not need <= set(hr.columns):
        raise SystemExit(f"helpsteer rows missing {sorted(need - set(hr.columns))}")
    # Rows are one per response; the responses to a prompt share the prompt text. The
    # decision-focused criterion of helpsteer_run.py: which response would each criterion
    # pick? Ties are broken by row order on both sides, so the two criteria are compared
    # under identical rules.
    dis = tot = 0
    for _, d in hr.groupby("prompt"):
        if len(d) < 2:
            continue
        tot += 1
        dis += int(d["judge_overall_score"].idxmax() != d["correctness"].idxmax())
    agree_hs = 1.0 - dis / tot if tot else np.nan
    rows.append(dict(
        evaluator="\\texttt{gpt-4o-mini}", corpus="HelpSteer2, human correctness",
        n=int(H["meta"]["n"]), agree=agree_hs, flip=np.nan,
        longer=float(H["meta"].get("longer_effect", np.nan))))

    T = pd.DataFrame(rows)

    # assertions only
    r0 = T.iloc[0]
    assert abs(r0.agree - 0.430) < 5e-3, f"LMArena agreement {r0.agree:.4f}"
    assert abs(r0.flip - 0.335) < 5e-3, f"LMArena flip {r0.flip:.4f}"
    assert abs(r0.longer - 0.243) < 5e-3, f"LMArena longer {r0.longer:.4f}"
    assert T.iloc[0].n == 19368 or T.iloc[0].n == 19377, T.iloc[0].n
    assert T.iloc[1].agree > T.iloc[2].agree, "shared-subset ordering changed"
    assert T.iloc[1].longer > 0 > T.iloc[2].longer, \
        "the two judges no longer disagree in the direction of the length effect"
    assert 0.70 < T.iloc[3].agree < 0.73, f"HelpSteer2 agreement {T.iloc[3].agree:.4f} " \
        "(paper states the judge differs from human correctness on 28.3% of prompts)"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    T.to_csv(args.output, index=False)

    def f(x, nd=3):
        return "---" if not np.isfinite(x) else f"${x:.{nd}f}$"

    def s(x, nd=3):
        return "---" if not np.isfinite(x) else f"${x:+.{nd}f}$"

    print(f"{'evaluator':26}{'corpus':32}{'n':>8}{'agree':>8}{'flip':>8}{'longer':>9}")
    for r in T.itertuples():
        print(f"{r.evaluator:26}{r.corpus:32}{r.n:>8,}{r.agree:>8.3f}"
              f"{r.flip if np.isfinite(r.flip) else float('nan'):>8.3f}{r.longer:>9.3f}")
    print("\nLaTeX body rows:\n")
    for r in T.itertuples():
        print(f"{r.evaluator} & {r.corpus} & ${r.n:,}$".replace(",", "{,}")
              + f" & {f(r.agree)} & {f(r.flip)} & {s(r.longer)}\\\\")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
