"""Work Package D, steps D3 and D4. Operational routing economics on tau2-bench-verified.

Reads the simulation files tau2 writes and computes, per domain and per agent model, task success,
agent cost, user-simulator cost (benchmark infrastructure, reported separately), cost per attempted
task and cost per successful task. Then the routing question: does picking the best model per
domain beat any single model, and what does a quality-constrained choice cost.

    python tau2_analyse.py
"""
import glob, json, os, re

MIN_N = 20   # a cell below this cannot support a success-rate estimate and is excluded
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "ext", "tau2-bench-verified",
                                    "data", "simulations", "*.json"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
PAT = re.compile(r"_(?P<domain>airline|retail|telecom|mock)_llm_agent_(?P<model>.+?)_user_simulator_")


def load():
    rows = []
    for f in sorted(glob.glob(SIMS)):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            rows.append(dict(domain=m["domain"], agent_model=m["model"],
                             task_id=s.get("task_id"), trial=s.get("trial"),
                             reward=float(ri.get("reward", 0.0)),
                             success=float(ri.get("reward", 0.0)) >= 1.0,
                             agent_cost_usd=float(s.get("agent_cost") or 0.0),
                             user_cost_usd=float(s.get("user_cost") or 0.0),
                             duration_s=float(s.get("duration") or 0.0),
                             termination_reason=s.get("termination_reason")))
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        raise SystemExit("no tau2 simulations found")
    print(f"{len(df):,} simulations, {df['domain'].nunique()} domain(s), "
          f"{df['agent_model'].nunique()} agent models\n")

    print("D4. Per domain and model. Agent cost is the deployable cost; the user simulator is")
    print("    benchmark infrastructure and is reported separately, never merged in.\n")
    print(f"  {'domain':9}{'agent model':15}{'n':>5}{'success':>9}{'$/task':>10}"
          f"{'$/success':>11}{'sim $/task':>12}{'turns s':>9}")
    g = df.groupby(["domain", "agent_model"])
    summ = []
    for (dom, mod), s in g:
        succ = s["success"].mean()
        cpt = s["agent_cost_usd"].mean()
        cps = (s["agent_cost_usd"].sum() / s["success"].sum()) if s["success"].sum() else np.nan
        summ.append(dict(domain=dom, model=mod, n=len(s), success=succ,
                         cost_per_task=cpt, cost_per_success=cps,
                         sim_cost_per_task=s["user_cost_usd"].mean()))
        print(f"  {dom:9}{mod:15}{len(s):5d}{succ:9.3f}{cpt:10.4f}"
              + (f"{cps:11.4f}" if np.isfinite(cps) else f"{'n/a':>11}")
              + f"{s['user_cost_usd'].mean():12.5f}{s['duration_s'].mean():9.1f}")

    sm = pd.DataFrame(summ)
    thin = sm[sm["n"] < MIN_N]
    if len(thin):
        print(f"\n  EXCLUDED from every routing claim, fewer than {MIN_N} simulations:")
        for _, r in thin.iterrows():
            print(f"    {r['domain']}/{r['model']}  n={int(r['n'])}  "
                  f"(success {r['success']:.3f} is not estimable at this n)")
    sm = sm[sm["n"] >= MIN_N]
    print("\nD3. Routing versus any single model, per 1,000 tasks\n")
    for dom, s in sm.groupby("domain"):
        best_q = s.loc[s["success"].idxmax()]
        cheap = s.loc[s["cost_per_task"].idxmin()]
        print(f"  {dom}")
        print(f"    highest success   {best_q['model']:14} {best_q['success']:.3f} "
              f"at ${best_q['cost_per_task']*1000:,.2f} per 1k tasks")
        print(f"    cheapest          {cheap['model']:14} {cheap['success']:.3f} "
              f"at ${cheap['cost_per_task']*1000:,.2f} per 1k tasks")
        for tol in (0.00, 0.02, 0.05):
            ok = s[s["success"] >= best_q["success"] - tol]
            pick = ok.loc[ok["cost_per_task"].idxmin()]
            print(f"    tolerance {tol:.2f}      {pick['model']:14} {pick['success']:.3f} "
                  f"at ${pick['cost_per_task']*1000:,.2f} per 1k tasks")
        dom_rows = s.sort_values("cost_per_task")
        dominated = [r["model"] for _, r in dom_rows.iterrows()
                     if ((dom_rows["cost_per_task"] < r["cost_per_task"])
                         & (dom_rows["success"] > r["success"])).any()]
        if dominated:
            print(f"    strictly dominated on this domain: {', '.join(dominated)}")

    if sm["domain"].nunique() > 1:
        print("\n  per-domain routing against the best single model overall")
        for tol in (0.00, 0.02):
            routed_c = routed_s = 0.0
            for dom, s in sm.groupby("domain"):
                b = s["success"].max()
                ok = s[s["success"] >= b - tol]
                p = ok.loc[ok["cost_per_task"].idxmin()]
                routed_c += p["cost_per_task"]; routed_s += p["success"]
            n = sm["domain"].nunique()
            best_single = sm.groupby("model").agg(s=("success", "mean"),
                                                  c=("cost_per_task", "mean"))
            bs = best_single.loc[best_single["s"].idxmax()]
            print(f"    tolerance {tol:.2f}: routed {routed_s/n:.3f} success at "
                  f"${routed_c/n*1000:,.2f}/1k  vs best single "
                  f"{bs['s']:.3f} at ${bs['c']*1000:,.2f}/1k")

    print("\n  termination reasons:", df["termination_reason"].value_counts().to_dict())
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, "tau2_analysis.json")
    json.dump(dict(n_simulations=int(len(df)), per_cell=summ,
                   termination=df["termination_reason"].value_counts().to_dict()),
              open(out, "w"), indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
