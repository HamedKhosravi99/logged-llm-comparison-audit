"""Work Package E, step E6. Expected-cost routing on dollar-valued work.

Given a roster of models with different accuracy and price, and a human fallback that costs the
task's released payout when the model gets it wrong, the deployment rule is to send each task to
whichever option minimises expected total cost:

    E[C_ik] = c_ik + v_i + (1 - p_ik) * h_i

with c_ik the model attempt cost, v_i verification, p_ik the estimated success probability and h_i
the released payout. Direct human fallback is allowed when every model's expected cost exceeds it.

The point for this paper: p_ik is an ESTIMATE, and where it comes from decides the bill. Routing on
a per-task-value-band estimate is compared against routing on truth, so the value of an accurate
comparison is priced on work with real dollar stakes.

No new inference. Reads `results/swelancer_manager.json`.

    python swelancer_routing.py
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
BANDS = [0, 500, 1000, 2000, 1e9]
LABELS = ["<=500", "501-1000", "1001-2000", ">2000"]
V_COST = 0.0          # the managerial task is graded by comparison, so verification is free here


def load():
    d = json.load(open(os.path.join(RES, "swelancer_manager.json")))
    df = pd.DataFrame(d["rows"])
    df["correct"] = df["correct_selection"].astype(bool)
    df["band"] = pd.cut(df["task_value_usd"], BANDS, labels=LABELS)
    return df


def expected_cost(p, c, h, v=V_COST):
    return c + v + (1.0 - p) * h


def main():
    df = load()
    models = sorted(df["model_id"].unique())
    tasks = df["question_id"].unique()
    print(f"{len(tasks)} tasks, {len(models)} models, "
          f"released value ${df.groupby('question_id')['task_value_usd'].first().sum():,.0f}\n")

    # ---- success-probability estimates ------------------------------------------------
    # truth:   the model's realised outcome on that task
    # banded:  the model's success rate within the task's value band, the estimate a team could form
    # global:  the model's overall success rate, the crudest estimate
    band_p = df.groupby(["model_id", "band"], observed=True)["correct"].mean()
    glob_p = df.groupby("model_id")["correct"].mean()
    cost_m = df.groupby("model_id")["model_cost_usd"].mean()

    print("  model accuracy and mean attempt cost")
    for m in models:
        print(f"    {m:14} p={glob_p[m]:.3f}  c=${cost_m[m]:.5f}")

    rows = []
    for qid, g in df.groupby("question_id"):
        h = float(g["task_value_usd"].iloc[0])
        rec = dict(question_id=qid, task_value_usd=h)
        for name in ("truth", "banded", "global"):
            best, best_ec = None, float(h)          # direct human fallback is the default option
            for _, r in g.iterrows():
                m = r["model_id"]
                if name == "truth":
                    p = 1.0 if r["correct"] else 0.0
                elif name == "banded":
                    p = float(band_p.get((m, r["band"]), glob_p[m]))
                else:
                    p = float(glob_p[m])
                ec = expected_cost(p, float(r["model_cost_usd"]), h)
                if ec < best_ec:
                    best, best_ec = m, ec
            # realised cost of that choice: attempt cost plus payout if it actually failed
            if best is None:
                realised = h
            else:
                rr = g[g["model_id"] == best].iloc[0]
                realised = float(rr["model_cost_usd"]) + (0.0 if rr["correct"] else h)
            rec[f"{name}_pick"] = best or "human"
            rec[f"{name}_realised"] = realised
        # single-model baselines
        for m in models:
            rr = g[g["model_id"] == m].iloc[0]
            rec[f"single_{m}"] = float(rr["model_cost_usd"]) + (0.0 if rr["correct"] else h)
        rows.append(rec)

    out = pd.DataFrame(rows)
    human_only = out["task_value_usd"].sum()
    print(f"\nE6. Expected-cost routing, total realised cost over {len(out)} tasks")
    print(f"  {'policy':34}{'total $':>12}{'vs human-only':>16}{'human fallbacks':>17}")
    print(f"  {'human only, no model':34}{human_only:12,.0f}{'-':>16}{len(out):>17}")
    res = {}
    for m in models:
        t = out[f"single_{m}"].sum()
        res[f"single_{m}"] = t
        print(f"  {'always ' + m:34}{t:12,.0f}{human_only - t:16,.0f}"
              f"{int((out[f'single_{m}'] > out['task_value_usd'] * 0.9).sum()):>17}")
    for name, lab in (("global", "route on a global accuracy estimate"),
                      ("banded", "route on a per-value-band estimate"),
                      ("truth", "route on truth (oracle)")):
        t = out[f"{name}_realised"].sum()
        res[name] = t
        nh = int((out[f"{name}_pick"] == "human").sum())
        print(f"  {lab:34}{t:12,.0f}{human_only - t:16,.0f}{nh:>17}")

    best_single = min(models, key=lambda m: res[f"single_{m}"])
    print(f"\n  best single model is {best_single} at ${res['single_' + best_single]:,.0f}")
    print(f"  routing on a banded estimate saves a further "
          f"${res['single_' + best_single] - res['banded']:,.0f}")
    print(f"  the oracle would save ${res['single_' + best_single] - res['truth']:,.0f}, so an "
          f"accurate comparison is worth\n  ${res['banded'] - res['truth']:,.0f} more than the "
          f"estimate a team could actually form")

    print("\n  by task-value band, realised cost")
    out["band"] = pd.cut(out["task_value_usd"], BANDS, labels=LABELS)
    for b, g in out.groupby("band", observed=True):
        print(f"    {b:>10}  n={len(g):3d}  human ${g['task_value_usd'].sum():8,.0f}  "
              f"banded ${g['banded_realised'].sum():8,.0f}  truth ${g['truth_realised'].sum():8,.0f}")

    print("\n  Released payouts are a fallback-cost scenario, not realised company savings.")
    json.dump(dict(n_tasks=int(len(out)), human_only=float(human_only),
                   totals={k: float(v) for k, v in res.items()}),
              open(os.path.join(RES, "swelancer_routing.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(RES, 'swelancer_routing.json')}")


if __name__ == "__main__":
    main()
