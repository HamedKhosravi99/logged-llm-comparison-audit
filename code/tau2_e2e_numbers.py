"""Every number the revised RQ5, abstract, Figure 3 and conclusion need."""
import json, os, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "results", "tau2_e2e_full.json")))
OUT = os.path.join(HERE, "..", "results", "tau2_e2e_numbers.json")
FALLBACK = [0, 2, 5, 10, 25]
DELTA = 0.02

sc = pd.DataFrame([r for r in D["records"] if r["kind"] == "score"])
meta = pd.DataFrame([r for r in D["records"] if r["kind"] == "meta"])
g = sc.groupby("policy").agg(success=("success", "mean"), success_se=("success", "sem"),
                             agent_cost=("agent_cost", "mean"), tool_err=("tool_err", "mean"),
                             violation=("violation", "mean"), audit_usd=("audit_usd", "mean"),
                             wrong=("wrong", "mean"), switches=("switches", "mean"))
raw = g.loc["Raw routed log"]

def cps(r, fb=0.0):
    """audit-inclusive cost per successful task, with an assumed fallback price on failures."""
    per_task = r.agent_cost + fb * (1 - r.success)
    return per_task / max(1e-9, r.success)

rows = []
for name, r in g.iterrows():
    per_task_saving = raw.agent_cost - r.agent_cost
    be = (r.audit_usd / per_task_saving) if per_task_saving > 1e-12 and r.audit_usd > 0 else None
    fb_be = {}
    for fb in FALLBACK:
        tot_raw = raw.agent_cost + fb * (1 - raw.success)
        tot_r = r.agent_cost + fb * (1 - r.success)
        s = tot_raw - tot_r
        fb_be[str(fb)] = (r.audit_usd / s) if s > 1e-12 and r.audit_usd > 0 else None
    rows.append(dict(policy=name, success=r.success, success_se=r.success_se,
                     agent_cost_per_1k=r.agent_cost*1000, audit_usd=r.audit_usd,
                     cost_per_success=cps(r), tool_err=r.tool_err, violation=r.violation,
                     wrong=r.wrong, switches=r.switches, break_even=be, break_even_by_fallback=fb_be))
t = pd.DataFrame(rows).set_index("policy")

order = ["Cheapest model", "Highest-success model", "Best single model", "Raw routed log",
         "Uniform audit", "CASE", "Dollar-aware CASE", "Ideal complete matrix"]
print(f"  {'Policy':24}{'Succ':>7}{'$/1k':>9}{'Audit$':>9}{'$/succ':>9}{'Viol':>7}{'ToolErr':>9}{'BE':>10}")
for p in order:
    if p not in t.index: continue
    r = t.loc[p]
    be = "—" if r.break_even is None or not np.isfinite(r.break_even) else f"{r.break_even:,.0f}"
    print(f"  {p:24}{r.success:>7.3f}{r.agent_cost_per_1k:>9.2f}{r.audit_usd:>9.4f}"
          f"{r.cost_per_success:>9.4f}{r.violation:>7.2f}{r.tool_err:>9.2f}{be:>10}")

# most common policies
def modal(p):
    c = collections.Counter(sc[sc.policy == p].policy_map)
    return json.loads(c.most_common(1)[0][0]), c.most_common(1)[0][1] / max(1, len(sc[sc.policy == p]))
raw_pol, raw_fr = modal("Raw routed log"); case_pol, case_fr = modal("CASE")
ideal_pol, _ = modal("Ideal complete matrix")
print(f"\n  modal raw-log policy   {raw_pol}  ({raw_fr:.0%} of runs)")
print(f"  modal CASE policy      {case_pol}  ({case_fr:.0%} of runs)")
print(f"  ideal policy           {ideal_pol}")
print(f"  domain decisions CASE changes vs raw log: "
      f"{sum(case_pol.get(k)!=raw_pol.get(k) for k in ideal_pol)} of {len(ideal_pol)}")

npairs = float(meta[meta.policy == "CASE"].n_pairs.mean())
print(f"\n  diagnostic recommended auditing in {D['diagnostic_triggered']}/{D['diagnostic_total']} "
      f"runs ({100*D['diagnostic_triggered']/D['diagnostic_total']:.0f}%), threshold {D['diagnostic_threshold']}")
print(f"  CASE purchased {npairs:.1f} of 12 model-domain pairs on average")
print(f"  fallback sensitivity, CASE break-even by assumed $/failed task:")
for fb, v in t.loc["CASE", "break_even_by_fallback"].items():
    print(f"     ${fb:>2}: {'no finite payback' if v is None else f'{v:,.0f} tasks'}")

json.dump(dict(table=json.loads(t.reset_index().to_json(orient="records")),
               modal_policies=dict(raw=raw_pol, case=case_pol, ideal=ideal_pol),
               n_pairs_mean=npairs, diagnostic_triggered=D["diagnostic_triggered"],
               diagnostic_total=D["diagnostic_total"],
               purchase_order=D["example_purchase_order"]), open(OUT, "w"), indent=1, default=float)
print(f"\n  wrote {os.path.abspath(OUT)}")
