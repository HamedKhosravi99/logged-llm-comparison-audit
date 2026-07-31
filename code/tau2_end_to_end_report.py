"""Priority 1, sections 2.9-2.11: primary outcome, payback and the main table."""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "processed", "tau2_end_to_end_raw.parquet")
OUT = os.path.join(HERE, "..", "results", "tau2_end_to_end.json")
HORIZONS = [1_000, 10_000, 100_000, 1_000_000]
REPORT_H = 10_000

df = pd.read_parquet(RAW)
perf = df.dropna(subset=["success"]).groupby("policy").agg(
    success=("success", "mean"), success_se=("success", "sem"),
    agent_cost=("agent_cost", "mean"), wrong=("wrong_choice", "mean"), n=("success", "size"))
aud = df.dropna(subset=["audit_usd"]).groupby("policy").agg(
    audit_usd=("audit_usd", "mean"), pairs=("pairs", "mean"))
t = perf.join(aud, how="left").fillna({"audit_usd": 0.0, "pairs": 0.0})

base = t.loc["Raw routed log"]
rows = []
for name, r in t.iterrows():
    cps = (REPORT_H * r.agent_cost + r.audit_usd) / max(1e-9, REPORT_H * r.success)
    sav_per_task = base.agent_cost - r.agent_cost
    be = (r.audit_usd / sav_per_task) if sav_per_task > 0 and r.audit_usd > 0 else None
    rows.append(dict(policy=name, success=r.success, success_se=r.success_se,
                     agent_cost_per_1k=r.agent_cost * 1000, audit_usd=r.audit_usd,
                     cost_per_success=cps, wrong_choice=r.wrong, pairs=r.pairs,
                     break_even=be,
                     net_savings={str(H): H * sav_per_task - r.audit_usd for H in HORIZONS}))
out = pd.DataFrame(rows).set_index("policy")

order = ["Cheapest model", "Best single model", "Raw routed log", "Selective router (free log)",
         "Uniform audit", "CASE", "Dollar-aware CASE", "Ideal reference"]
print(f"\n  Main table (section 2.11). Held-out scoring fold, {int(t.n.iloc[0])} evaluations")
print(f"  Cost per success amortised at H={REPORT_H:,} tasks. delta = 0.02.\n")
hdr = f"  {'Policy':30}{'Success':>9}{'$/1k tasks':>12}{'Audit $':>10}{'$/success':>11}{'Wrong':>8}{'Break-even':>13}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for p in order:
    if p not in out.index: continue
    r = out.loc[p]
    be = "—" if r.break_even is None or not np.isfinite(r.break_even) else f"{r.break_even:,.0f}"
    if p in ("Cheapest model", "Best single model", "Raw routed log", "Selective router (free log)",
             "Ideal reference"): be = "—"
    print(f"  {p:30}{r.success:>9.3f}{r.agent_cost_per_1k:>12.2f}{r.audit_usd:>10.4f}"
          f"{r.cost_per_success:>11.5f}{r.wrong_choice:>8.2f}{be:>13}")

print("\n  Payback (section 2.10), net savings versus the raw routed log:")
for p in ("Uniform audit", "CASE", "Dollar-aware CASE"):
    if p not in out.index: continue
    ns = out.loc[p, "net_savings"]
    s = "  ".join(f"{H//1000}k:${ns[str(H)]:+.2f}" for H in HORIZONS)
    print(f"    {p:22} {s}")
    if all(ns[str(H)] <= 0 for H in HORIZONS):
        print(f"      -> No finite break-even volume under this tolerance.")

json.dump(json.loads(out.reset_index().to_json(orient="records")), open(OUT, "w"), indent=1)
print(f"\n  wrote {os.path.abspath(OUT)}")
