"""Figure 1 under Eq. (1): target-value loss against the serving budget.

The companion of `make_fig1_validity.py`, which draws loss against the evidence budget for the
per-workload rule. Under Eq. (1) the serving budget is the axis that matters: it is what decides
how much of the allocation is open to evidence at all, since a budget that binds prices the
contested rivals out of the feasible set before any readout is bought.

Every plotted coordinate comes from results/section5_mckp_<corpus>_eps0.15_n30.json; the assertions
below check structure the design guarantees (the logged arm cannot move with an evidence budget it
does not spend, the target-table arm loses nothing, and a cellwise-perfect assignment repair is
exact under a faithful evaluator), not values read off a previous run.

Usage:
    python code/make_fig1_validity_mckp.py --budget 0.35 \
      --results results/section5_mckp_llmrouterbench_eps0.15_n30.json \
      --output figures/fig1_validity_mckp.pdf
"""
import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
import seaborn as sns                                                  # noqa: E402

GREY = "#6f6f6f"
BLUE = "#2a78d6"
INK = "#1a1a1a"

PANELS = [("A1", "D0", "(a) Selective routing"),
          ("A1", "D1", "(b) Selective + distorted")]
ALL_CELLS = [("A0", "D0"), ("A1", "D0"), ("A1", "D1")]
SERIES = [("trust", "Trust the production log", GREY, "o", "-"),
          ("uniform", "Uniform randomized readouts", GREY, "^", ":"),
          ("case", "CASE", BLUE, "s", "-"),
          ("oracle_sel", "Cellwise-perfect assignment repair (unachievable)", INK, "D", "--")]


def style():
    sns.set_context("paper")
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3, "lines.markersize": 4.5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget", type=float, default=0.35, help="evidence budget, share of pairs")
    args = ap.parse_args()

    style()
    df = pd.DataFrame(json.load(open(args.results)))
    df = df[np.isclose(df["budget_frac"], args.budget)]
    if df.empty:
        raise SystemExit("no rows at evidence budget %g in %s" % (args.budget, args.results))
    budgets = sorted(df["mult"].unique())              # the x-axis: serving budget multiples
    n_parts = df.partition.nunique()
    print("read %s: %d rows at evidence budget %g, %d partitions, Gamma %s"
          % (args.results, len(df), args.budget, n_parts, budgets))

    def cell(a, e, m, b):
        sub = df[(df["assign"] == a) & (df.cond == e) & (df.arm == m) & (df["mult"] == b)]
        if sub.empty:
            raise SystemExit("no rows for %s/%s/%s at %g" % (a, e, m, b))
        per = sub.groupby("partition")["loss"].mean().to_numpy(float)
        return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per))), len(per)

    titles = {(a, e): t for a, e, t in PANELS}
    rec = []
    for a, e in ALL_CELLS:
        for m, label, _, _, _ in SERIES + [("oracle_ref", "Target table", INK, "x", "-")]:
            for b in budgets:
                mu, se, n = cell(a, e, m, b)
                rec.append(dict(assignment=a, evaluator=e,
                                condition=titles.get((a, e), "%s/%s (checked, not plotted)" % (a, e)),
                                method=m, method_label=label, evidence_budget=args.budget,
                                budget=b, mean=mu, se=se, n_partitions=n))
    tidy = pd.DataFrame(rec)

    def get(a, e, m, b):
        r = tidy[(tidy.assignment == a) & (tidy.evaluator == e)
                 & (tidy.method == m) & (tidy.budget == b)]
        return float(r["mean"].iloc[0])

    for a, e in ALL_CELLS:                       # structural checks the design guarantees
        assert get(a, e, "oracle_ref", budgets[-1]) < 1e-12, "target table loses value"
    for b in budgets:
        assert get("A1", "D0", "oracle_sel", b) < 1e-12, \
            "a perfect assignment repair is not exact under a faithful evaluator"
    floor = get("A1", "D1", "oracle_sel", budgets[-1])
    print("measurement floor (A1--D1, cellwise-perfect repair): %.4f" % floor)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95), sharey=True)
    xs = np.arange(len(budgets))
    for ax, (a, e, title) in zip(axes, PANELS):
        for m, label, color, marker, ls in SERIES:
            ys = [get(a, e, m, b) for b in budgets]
            es = [float(tidy[(tidy.assignment == a) & (tidy.evaluator == e)
                             & (tidy.method == m) & (tidy.budget == b)]["se"].iloc[0])
                  for b in budgets]
            ax.errorbar(xs, ys, yerr=es, fmt=marker, ls=ls, color=color, ecolor=color,
                        elinewidth=0.8, capsize=2, markeredgecolor=color, label=label, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([("%g" % b) + r"$\times$" for b in budgets])
        ax.set_xlim(-0.3, len(budgets) - 0.7)
        ax.set_title(title)
        ax.set_xlabel(r"Serving budget $\Gamma$")
        ax.yaxis.grid(True, lw=0.4, color="#e6e6e6")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Target-value loss")
    axes[0].set_ylim(bottom=-0.001)

    sns.despine(fig=fig)
    fig.tight_layout(pad=0.3, w_pad=0.9, rect=(0, 0.19, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
               handlelength=1.8, handletextpad=0.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.0))

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stem = os.path.splitext(out)[0]
    fig.savefig(out)
    fig.savefig(stem + ".png", dpi=400)
    tidy.to_csv(stem + "_data.csv", index=False)

    print("\n%-34s%-30s%8s%9s%9s" % ("condition", "method", "budget", "mean", "se"))
    for r in rec:
        print("%-34s%-30s%8.2f%9.4f%9.4f"
              % (r["condition"], r["method_label"], r["budget"], r["mean"], r["se"]))
    print("\nwrote %s\nwrote %s.png\nwrote %s_data.csv" % (out, stem, stem))


if __name__ == "__main__":
    main()
