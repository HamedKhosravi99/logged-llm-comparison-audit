"""Assemble the Stage 1 aligned report from the diagnostics. No new API calls."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
D = pd.read_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"))
GATE = json.load(open(os.path.join(RES, "free_stage1_rq5_aligned.json")))
H, rng = 100_000, np.random.default_rng(0)
L = []

def paired(d, a, b, col, reps=4000):
    p = d.pivot_table(index=["fold", "seed"], columns="arm", values=col)
    if a not in p or b not in p: return None
    x = (p[a] - p[b]).dropna().values
    if len(x) < 3: return None
    bs = np.array([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(reps)])
    return x.mean(), np.quantile(bs, .025), np.quantile(bs, .975)

L += ["# Stage 1, aligned to the paper's RQ5 pipeline", "",
      "No new API calls. Two operational replays of the same 440 $\\tau^2$ simulations.",
      "Statistical unit: 20 acquisition seeds averaged inside each fold x split unit, then 50 units.",
      "", "## Reproduction gate", "",
      "| quantity | target | tol | reproduced | pass |", "|---|---:|---:|---:|:--:|"]
for g in GATE["gate"]:
    L.append(f"| {g['quantity']} | {g['target']} | {g['tolerance']} | {g['reproduced']:.3f} | "
             f"{'YES' if g['passed'] else 'NO'} |")
L += ["", f"**Gate: {'PASSED' if GATE['gate_passed'] else 'FAILED'}** (paper scoring convention: "
      "acquisition seed 0). The aligned results below use the 20-seed average that the analysis "
      "plan requires, which moves Current CASE from 0.479/\\$28.34 to 0.482/\\$29.53.", ""]

for sc, base, title in (("quality_repair", "Raw routed log", "Quality-repair scenario (paper's fitted-router log)"),
                        ("cost_reduction", "Raw incumbent", "Cost-reduction scenario (quality-first incumbent)")):
    d = D[D.scenario == sc]
    g = d.groupby("arm").agg(success=("success","mean"), se=("success","sem"), cost=("cost","mean"),
                             audit=("audit","mean"), sw=("switches","mean"), harm=("harmful","mean"),
                             ben=("beneficial","mean"), neu=("neutral","mean"), ret=("retained","mean"),
                             qsat=("q_sat_frac","mean"), qall=("q_all_ok","mean"), wrong=("wrong","mean"))
    nb = g.loc["No audit / retain"]
    piv = d.pivot_table(index=["fold","seed"], columns="arm", values="cost")
    o = d.groupby(["fold","seed"]).agg(f=("opp_frac","first"), any_=("opp_any","first"),
                                       sav=("opp_savings_per_req","first"))
    L += [f"## {title}", "",
          f"**Oracle opportunity.** A quality-preserving cheaper alternative exists for "
          f"**{o.f.mean():.1%}** of domain decisions; **{1-o.any_.mean():.0%}** of runs contain no "
          f"feasible cost-saving switch at all. Maximum oracle saving \\${1000*o.sav.max():.2f} per "
          f"1k tasks.", ""]
    if o.f.mean() < 1e-9:
        L += ["> Cost saving is **impossible** in this scenario. Any apparent saving comes from "
              "quality-reducing downgrades, and no audit method can be faulted for failing to find "
              "savings that do not exist.", ""]
    L += ["| arm | success | \\$/1k | audit \\$ | \\$/success | switches | harmful | benef | harm\\|sw | ben\\|sw | retained | tol-ok | BE point | BE certified |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = [a for a in [base, "No audit / retain", "Raw-log downrouting", "Current CASE",
                         "Inverse-variance CASE", "Safe CASE", "Safe inverse-variance CASE",
                         "Ideal complete matrix"] if a in g.index]
    for a in order:
        r = g.loc[a]
        cps = (H*r.cost + r.audit) / max(1e-9, H*r.success)
        hs = f"{r.harm/r.sw:.2f}" if r.sw > 1e-9 else "n/a"
        bs_ = f"{r.ben/r.sw:.2f}" if r.sw > 1e-9 else "n/a"
        per = nb.cost - r.cost
        bep = f"{r.audit/per:,.0f}" if per > 1e-12 and r.audit > 0 else "none"
        cert = "none"
        if a in piv.columns:
            x = (piv["No audit / retain"] - piv[a]).dropna().values
            lo = float(np.quantile([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(4000)], .05))
            cert = f"{r.audit/lo:,.0f}" if lo > 1e-12 and r.audit > 0 else "none"
        L.append(f"| {a} | {r.success:.4f} | {1000*r.cost:.2f} | {r.audit:.2f} | {cps:.4f} | "
                 f"{r.sw:.3f} | {r.harm:.3f} | {r.ben:.3f} | {hs} | {bs_} | {r.ret:.2f} | "
                 f"{r.qsat:.3f} | {bep} | {cert} |")
    L += ["", "Paired 95% intervals over the 50 units (starred rows exclude zero):", ""]
    cons = [("Inverse-variance CASE","Current CASE"), ("Inverse-variance CASE", base),
            ("Safe inverse-variance CASE", base), ("Safe inverse-variance CASE","No audit / retain")]
    L += ["| contrast | outcome | mean | 95% CI |", "|---|---|---:|---|"]
    for a, b in cons:
        for col, lab in (("success","success"), ("cost","cost/request"),
                         ("wrong","wrong-good-enough"), ("harmful","harmful switches")):
            r = paired(d, a, b, col)
            if r is None: continue
            m, lo, hi = r
            star = " *" if (lo > 0 or hi < 0) else ""
            L.append(f"| {a} − {b} | {lab} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}]{star} |")
    L.append("")
open(os.path.join(RES, "free_stage1_rq5_aligned.md"), "w").write("\n".join(L) + "\n")
print("  wrote results/free_stage1_rq5_aligned.md")
