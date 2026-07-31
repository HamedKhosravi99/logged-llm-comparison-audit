"""Two Stage 1 figures. Standalone; not referenced by the paper unless integrated later."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
BLUE, GREEN, GREY, BLACK, ORANGE = "#2a78d6", "#1baf7a", "#6b6b68", "#0b0b0b", "#eb6834"

# ---------------- risk / coverage
T = pd.read_parquet(os.path.join(RES, "free_stage1_threshold_sensitivity.parquet"))
T["switches"] = 3 - T["retained"]        # three domains per run
D = pd.read_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"))
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.15), gridspec_kw=dict(wspace=0.30))
g = T.groupby(["arm", "tau"]).agg(harm=("harmful", "mean"), sw=("switches", "mean")).reset_index()
for arm, col, mk in (("Safe CASE", BLUE, "o"), ("Safe inverse-variance CASE", GREEN, "D")):
    s = g[g.arm == arm].sort_values("tau")
    cov = s.sw.values
    risk = np.where(cov > 1e-9, s.harm.values / np.maximum(cov, 1e-9), np.nan)
    ax[0].plot(cov, risk, "-", color=col, marker=mk, ms=4, lw=1.0, label=arm)
    for x, y, tau in zip(cov, risk, s.tau):
        ax[0].annotate(f"{tau:g}", (x, y), textcoords="offset points", xytext=(3, 4),
                       fontsize=5.4, color=col)
qr = D[D.scenario == "quality_repair"].groupby("arm").agg(h=("harmful","mean"), s=("switches","mean"))
for arm, col in (("Current CASE", GREY), ("Inverse-variance CASE", ORANGE)):
    if arm in qr.index:
        r = qr.loc[arm]
        ax[0].scatter([r.s], [r.h / max(r.s, 1e-9)], color=col, marker="s", s=18, zorder=4)
        ax[0].annotate(arm.replace(" CASE", ""), (r.s, r.h/max(r.s,1e-9)), textcoords="offset points",
                       xytext=(-6, -10), fontsize=5.8, color=col)
ax[0].axhline(0.5, color="#c9c9c6", lw=0.6, ls=":")
ax[0].set_xlabel("switches attempted per run (coverage)")
ax[0].set_ylabel("harmful share of switches (risk)")
ax[0].set_title("(a) risk vs coverage, quality-repair", loc="left", pad=4)
ax[0].legend(frameon=False, loc="upper right")
ax[0].grid(color="#eeeeec", lw=0.4); ax[0].set_axisbelow(True)

cr = D[D.scenario == "cost_reduction"].groupby("arm").agg(h=("harmful","mean"), b=("beneficial","mean"),
                                                          s=("switches","mean"))
arms = [a for a in ["Raw-log downrouting", "Current CASE", "Inverse-variance CASE",
                    "Safe inverse-variance CASE"] if a in cr.index]
x = np.arange(len(arms))
ax[1].bar(x - 0.18, [cr.loc[a].h/max(cr.loc[a].s,1e-9) for a in arms], 0.34, color=ORANGE, label="harmful")
ax[1].bar(x + 0.18, [cr.loc[a].b/max(cr.loc[a].s,1e-9) for a in arms], 0.34, color=GREEN, label="beneficial")
ax[1].set_xticks(x, [a.replace(" CASE","").replace("Safe inverse-variance","Safe IV")
                     .replace("Inverse-variance","IV").replace("Raw-log downrouting","raw-log")
                     for a in arms], fontsize=6.0)
ax[1].set_ylabel("share of attempted switches")
ax[1].set_title("(b) switch quality, cost-reduction", loc="left", pad=4)
ax[1].legend(frameon=False); ax[1].grid(axis="y", color="#eeeeec", lw=0.4); ax[1].set_axisbelow(True)
fig.savefig(os.path.join(FIG, "safe_case_risk_coverage.pdf"))
fig.savefig(os.path.join(FIG, "safe_case_risk_coverage.png"), dpi=200)

# ---------------- updated RQ5 panel
fig2, ax2 = plt.subplots(1, 2, figsize=(6.2, 2.15), gridspec_kw=dict(wspace=0.32))
for k, (sc, base, ttl) in enumerate((("quality_repair", "Raw routed log", "(a) quality-repair"),
                                     ("cost_reduction", "Raw incumbent", "(b) cost-reduction"))):
    d = D[D.scenario == sc].groupby("arm").agg(s=("success","mean"), se=("success","sem"),
                                               c=("cost","mean"))
    pts = [(base, BLUE, "o"), ("Current CASE", GREY, "s"), ("Inverse-variance CASE", ORANGE, "^"),
           ("Safe inverse-variance CASE", GREEN, "D"), ("Ideal complete matrix", BLACK, "s")]
    for name, col, mk in pts:
        if name not in d.index: continue
        r = d.loc[name]
        ax2[k].errorbar(1000*r.c, r.s, yerr=r.se, color=col, marker=mk, ms=4.4, lw=0,
                        elinewidth=0.8, capsize=1.8, mec="white", mew=0.5, zorder=3,
                        label=name.replace(" complete matrix","").replace("Safe inverse-variance","Safe IV")
                              .replace("Inverse-variance","IV"))
    ax2[k].set_xlabel("agent cost per 1,000 tasks (\\$)")
    if k == 0: ax2[k].set_ylabel("verified task success")
    ax2[k].set_title(ttl, loc="left", pad=4)
    ax2[k].grid(color="#eeeeec", lw=0.4); ax2[k].set_axisbelow(True)
    ax2[k].legend(frameon=False, loc="lower right", fontsize=5.6)
fig2.savefig(os.path.join(FIG, "rq5_safe_audit_updated.pdf"))
fig2.savefig(os.path.join(FIG, "rq5_safe_audit_updated.png"), dpi=200)
print("  wrote figures/safe_case_risk_coverage.pdf and figures/rq5_safe_audit_updated.pdf")
