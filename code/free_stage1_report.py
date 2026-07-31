"""Stage 1 comparison table and the minimum success criterion. Writes a standalone report."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "processed", "free_stage1_raw.parquet")
OUT_MD = os.path.join(HERE, "..", "results", "free_stage1_safe_audit.md")
OUT_JS = os.path.join(HERE, "..", "results", "free_stage1_safe_audit.json")
H = 100_000
DELTA = 0.02

df = pd.read_parquet(RAW)
g = df.groupby("arm").agg(success=("success", "mean"), success_se=("success", "sem"),
                          cost=("cost", "mean"), audit=("audit", "mean"),
                          harmful=("harmful", "mean"), abst=("abstentions", "mean"),
                          wrong=("wrong", "mean"), n=("success", "size"))
raw = g.loc["Raw routed log"]
rows = []
for name, r in g.iterrows():
    cps = (H * r.cost + r.audit) / max(1e-9, H * r.success)
    per = raw.cost - r.cost
    be = (r.audit / per) if per > 1e-12 and r.audit > 0 else None
    rows.append(dict(arm=name, success=r.success, success_se=r.success_se,
                     cost_per_1k=r.cost * 1000, audit=r.audit, cost_per_success=cps,
                     harmful=r.harmful, abstentions=r.abst, wrong=r.wrong, break_even=be))
t = pd.DataFrame(rows).set_index("arm")

order = ["Raw routed log", "Current CASE", "Additive correction", "Partially pooled correction",
         "Low-rank correction", "Inverse-variance combination", "Cross-fitted shrinkage",
         "Traffic-aware acquisition", "Full expected-value acq.", "Safe CASE with abstention"]
L = ["# Stage 1 of the free plan: a safer audit from data already saved", "",
     "No new API calls. Every number is replayed from the existing 440 $\\tau^2$ simulations.",
     f"Tolerance {DELTA}, horizon {H:,}, five outer folds by ten split seeds.", "",
     "| Method | Success | Agent \\$/1k | Audit \\$ | \\$/success | Harmful switches | Abstentions | Break-even |",
     "|---|---:|---:|---:|---:|---:|---:|---:|"]
print(f"  {'Method':30}{'Succ':>7}{'$/1k':>9}{'Audit$':>9}{'$/succ':>10}{'Harm':>7}{'Abst':>7}")
for a in order:
    if a not in t.index: continue
    r = t.loc[a]
    be = "none" if r.break_even is None or not np.isfinite(r.break_even) else f"{r.break_even:,.0f}"
    L.append(f"| {a} | {r.success:.3f} | {r.cost_per_1k:.2f} | {r.audit:.2f} | "
             f"{r.cost_per_success:.4f} | {r.harmful:.2f} | {r.abstentions:.2f} | {be} |")
    print(f"  {a:30}{r.success:>7.3f}{r.cost_per_1k:>9.2f}{r.audit:>9.2f}"
          f"{r.cost_per_success:>10.4f}{r.harmful:>7.2f}{r.abstentions:>7.2f}")

# minimum success criterion
safe = t.loc["Safe CASE with abstention"] if "Safe CASE with abstention" in t.index else None
cur = t.loc["Current CASE"]
L += ["", "## Minimum success criterion", ""]
if safe is not None:
    # the criterion needs a positive LOWER BOUND on net savings, not a point estimate
    pair = df[df.arm.isin(["Safe CASE with abstention", "Raw routed log"])].pivot_table(
        index=["fold", "seed"], columns="arm", values="cost")
    d = (pair["Raw routed log"] - pair["Safe CASE with abstention"]).dropna()
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(d.values, len(d), replace=True)) for _ in range(2000)]
    sav_lo = float(np.quantile(boots, 0.05))          # one-sided 95% lower bound, $/request
    net_lo = H * sav_lo - safe.audit
    switch_ok = (safe.success >= raw.success - DELTA and net_lo > 0
                 and safe.cost_per_success < raw.cost_per_success)
    sd = df[df.arm == "Raw routed log"].groupby(["fold", "seed"]).success.mean().sem()
    abstain_ok = (safe.harmful < cur.harmful and safe.success >= raw.success - 2 * float(sd))
    L.append(f"- lower 95% bound on saving per request: \\${sav_lo:.6f}; "
             f"net over {H:,} requests after the audit: \\${net_lo:,.2f}")
    L.append(f"- **Successful switching**: {'MET' if switch_ok else 'not met'} "
             f"(needs tolerance met, positive net savings, better cost per success)")
    L.append(f"- **Successful abstention**: {'MET' if abstain_ok else 'not met'} "
             f"(needs fewer harmful switches than current CASE, and no worse than the raw log "
             f"within uncertainty)")
    L.append("")
    L.append(f"Current CASE makes {cur.harmful:.2f} harmful switches per run; the safe protocol makes "
             f"{safe.harmful:.2f} and abstains on {safe.abstentions:.2f} of three decisions.")
    verdict = ("The safe audit prevented a model switch that the simpler CASE protocol would have made."
               if abstain_ok and not switch_ok else
               "The safe audit met the switching criterion." if switch_ok else
               "Neither criterion is met on this workload.")
    L += ["", f"**Verdict.** {verdict}"]
L += ["", "## Scope", "",
      "This is a public-benchmark protocol result. It does not establish deployment readiness, which",
      "additionally requires prospective traffic, stable model versions, validated operational",
      "outcomes and a canary or shadow deployment."]
open(OUT_MD, "w").write("\n".join(L) + "\n")
json.dump(json.loads(t.reset_index().to_json(orient="records")), open(OUT_JS, "w"), indent=1)
print(f"\n  wrote {os.path.abspath(OUT_MD)}")
