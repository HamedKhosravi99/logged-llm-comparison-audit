"""Priority 7 (section 8): free stress tests on the tau2 replay. No new API calls."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "..", "data", "processed", "tau2_complete_matrix.parquet")
OUT = os.path.join(HERE, "..", "results", "p7_stress.json")
DELTA = 0.02
AUDIT_BASE = 10.33          # measured audit cost from the practitioner report


def cells(df):
    return df.groupby(["domain", "agent_model"]).agg(
        success=("task_success", "mean"), cost=("agent_cost_usd", "mean")).reset_index()


def decide(cell, weights, delta=DELTA, drop=(), extra=None):
    """Cheapest-good-enough per domain, then traffic-weighted quality and cost."""
    c = cell[~cell.agent_model.isin(drop)].copy()
    notes = []
    if extra is not None:
        for d in sorted(c.domain.unique()):
            c = pd.concat([c, pd.DataFrame([dict(domain=d, agent_model=extra,
                                                 success=np.nan, cost=0.0)])], ignore_index=True)
    pol, q, k = {}, [], []
    for d in sorted(c.domain.unique()):
        sub = c[c.domain == d]
        unsupported = sub[sub.success.isna()].agent_model.tolist()
        if unsupported:
            notes.append(f"{d}: insufficient evidence; exploration required for "
                         f"{', '.join(unsupported)}")
        sub = sub.dropna(subset=["success"])
        if sub.empty:
            continue
        best = sub.success.max()
        ok = sub[sub.success >= best - delta].sort_values(["cost", "success"], ascending=[True, False])
        pol[d] = ok.iloc[0].agent_model
        w = weights.get(d, 1.0)
        q.append(w * ok.iloc[0].success); k.append(w * ok.iloc[0].cost)
    W = sum(weights.get(d, 1.0) for d in pol)
    return pol, (sum(q) / W if W else np.nan), (sum(k) / W * 1000 if W else np.nan), notes


def main():
    df = pd.read_parquet(M)
    cell = cells(df)
    doms = sorted(cell.domain.unique())
    base_w = {d: 1.0 for d in doms}
    res = {}

    print("  traffic scenarios (delta = 0.02):")
    traffic = {"original": base_w, "uniform": base_w}
    for d in doms:
        traffic[f"2x {d}"] = {**base_w, d: 2.0}
    for name, w in traffic.items():
        pol, q, c, _ = decide(cell, w)
        res[f"traffic:{name}"] = dict(policy=pol, quality=q, cost_per_1k=c)
        print(f"    {name:14} quality {q:.3f}  ${c:>7.2f}/1k  {pol}")
    print("    2x state-changing: NOT RUN - Priority 6 found the smallest model x subtype cell is "
          "2 tasks, below the plan's own 15-task floor")

    print("\n  price scenarios:")
    cheapest = cell.groupby("domain").apply(lambda g: g.sort_values("cost").iloc[0].agent_model).to_dict()
    priciest = cell.groupby("domain").apply(lambda g: g.sort_values("cost").iloc[-1].agent_model).to_dict()
    for name in ("cheapest x0.5", "priciest x0.5", "all prices x0.5",
                 "audit x0.5", "audit x2", "audit x5"):
        c2 = cell.copy()
        if name == "cheapest x0.5":
            for d, m in cheapest.items():
                c2.loc[(c2.domain == d) & (c2.agent_model == m), "cost"] *= 0.5
        elif name == "priciest x0.5":
            for d, m in priciest.items():
                c2.loc[(c2.domain == d) & (c2.agent_model == m), "cost"] *= 0.5
        elif name == "all prices x0.5":
            c2["cost"] *= 0.5
        mult = {"audit x0.5": 0.5, "audit x2": 2.0, "audit x5": 5.0}.get(name)
        pol, q, c, _ = decide(c2, base_w)
        aud = AUDIT_BASE * mult if mult else AUDIT_BASE
        res[f"price:{name}"] = dict(policy=pol, quality=q, cost_per_1k=c, audit_usd=aud)
        print(f"    {name:18} quality {q:.3f}  ${c:>7.2f}/1k  audit ${aud:>6.2f}  {pol}")

    print("\n  candidate-set scenarios:")
    models = sorted(cell.agent_model.unique())
    scen = {"remove cheapest": [min(models, key=lambda m: cell[cell.agent_model == m].cost.mean())],
            "remove priciest": [max(models, key=lambda m: cell[cell.agent_model == m].cost.mean())]}
    for m in models:
        scen[f"remove {m}"] = [m]
    for name, drop in scen.items():
        pol, q, c, _ = decide(cell, base_w, drop=drop)
        res[f"candidates:{name}"] = dict(policy=pol, quality=q, cost_per_1k=c)
        print(f"    {name:26} quality {q:.3f}  ${c:>7.2f}/1k  {pol}")
    pol, q, c, notes = decide(cell, base_w, extra="hypothetical-unseen-model")
    res["candidates:add unsupported model"] = dict(policy=pol, quality=q, cost_per_1k=c, notes=notes)
    print(f"    {'add unsupported model':26} -> {notes[0] if notes else 'NO WARNING (bug)'}")
    print(f"       triggered on all {len(notes)} request types: {len(notes) == len(doms)}")

    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"\n  wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
