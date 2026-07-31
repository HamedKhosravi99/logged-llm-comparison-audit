"""Cost-vs-inferior-switch frontier for the confirmatory run. Conventional academic style.

Every plotted value is computed from free_stage2_conf_{llmrb,rb}_units.parquet and the
decisions parquets; nothing is hard-coded. Not part of the paper (NO-GO verdict).

Usage: python make_fig_stage2_frontier.py --outdir ../figures
"""
import argparse, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import seaborn as sns
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
H, V = 100_000, 0.25
GREY, BLUE, ORANGE = "#6f6f6f", "#2a78d6", "#e8703a"

ARMS = [("Incumbent", GREY, "o"), ("Raw-log downrouting", ORANGE, "^"),
        ("Free log NI tau=0.5", GREY, "v"), ("Free log NI tau=0.8", GREY, "s"),
        ("Free log NI tau=0.9", GREY, "D"), ("Free log NI tau=0.95", GREY, "P"),
        ("Locked paired audit", BLUE, "*")]


def style():
    sns.set_context("paper")
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.7, "lines.markersize": 5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=os.path.join(RES, "..", "figures"))
    a = ap.parse_args()
    style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6))
    rec = []
    for ax, (tag, name) in zip(axes, (("llmrb", "LLMRouterBench"), ("rb", "RouterBench"))):
        U = pd.read_parquet(os.path.join(RES, f"free_stage2_conf_{tag}_units.parquet"))
        Dec = pd.read_parquet(os.path.join(RES, f"free_stage2_conf_{tag}_decisions.parquet"))
        inf = Dec.groupby(["arm", "seed", "domain"]).first().reset_index() \
            .groupby("arm").apply(lambda g: (g.cat_inc == "inferior_inc").mean(),
                                  include_groups=False)
        for arm, color, marker in ARMS:
            g = U[U.arm == arm]
            if not len(g):
                continue
            y = (g.cost + (g.audit + V * g.nver) / H) * 1000
            m = float(y.mean())
            h = float(stats.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))) \
                if y.std(ddof=1) > 0 else 0.0
            x = float(inf.get(arm, 0.0))
            ax.errorbar([x], [m], yerr=[h], fmt=marker, color=color, ecolor=color,
                        elinewidth=0.9, capsize=2.5, markeredgecolor=color,
                        label=arm.replace("Free log NI tau=", "log gate $\\tau$=")
                                 .replace("Locked paired audit", "incumbent-anchored audit"))
            rec.append(dict(corpus=name, arm=arm, inferior_rate=x, cost_incl_1k=m, ci_half=h))
        ax.set_xlabel("Incumbent-relative inferior switches\nper domain-decision")
        ax.set_ylabel("Audit-inclusive cost per 1,000 tasks ($)" if tag == "llmrb" else "")
        ax.set_title(name)
        ax.xaxis.grid(True, lw=0.4, color="#e6e6e6"); ax.yaxis.grid(True, lw=0.4, color="#e6e6e6")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right", handletextpad=0.3, borderpad=0.2,
                   labelspacing=0.3)
    sns.despine(fig=fig)
    fig.tight_layout(pad=0.4, w_pad=1.6)
    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.outdir, f"fig_stage2_conf_frontier.{ext}"),
                    dpi=400 if ext == "png" else None)
    pd.DataFrame(rec).to_csv(os.path.join(RES, "free_stage2_conf_frontier.csv"), index=False)
    print("wrote fig_stage2_conf_frontier.{pdf,png} and free_stage2_conf_frontier.csv")
    for r in rec:
        print(f"  {r['corpus']:omit15}" if False else
              f"  {r['corpus']:15} {r['arm']:26} x={r['inferior_rate']:.4f} "
              f"y={r['cost_incl_1k']:.3f} ±{r['ci_half']:.3f}")


if __name__ == "__main__":
    main()
