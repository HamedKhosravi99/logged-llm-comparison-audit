"""Figure 1: target-value loss against the randomized-audit budget.

One message: buying randomized comparisons lowers target-value loss only when the routed log
is selectively assigned, and no budget removes evaluator-induced loss.

Every plotted coordinate is computed from results/tae_validity.json. The hard-coded numbers
in this file appear only inside assertions that verify the computation; none of them is used
to position a marker or an error bar.

Usage:
    python code/make_fig1_validity.py \
      --validity results/tae_validity.json \
      --paper-numbers results/tae_paper_numbers.json \
      --output figures/fig1_validity.pdf
"""
import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

N_PARTITIONS = 30

# three semantic colors: logged-only, audited, oracle
GREY = "#6f6f6f"
BLUE = "#2a78d6"
INK = "#1a1a1a"

# (assignment, evaluator, panel title). Titles are two lines in every panel so the axes align.
# Plotted panels. The A0/D0 cell is no longer drawn -- its content is one sentence of text
# in RQ2 -- but it is still checked in ALL_CELLS below, so dropping it costs no verification.
PANELS = [("A1", "D0", "(a) Selective routing"),
          ("A1", "D1", "(b) Selective + distorted")]
ALL_CELLS = [("A0", "D0"), ("A1", "D0"), ("A1", "D1")]

# (method, label, colour, marker, linestyle)
SERIES = [("trust", "Trust routed log", GREY, "o", "-"),
          ("uniform", "Uniform randomized audit", GREY, "^", ":"),
          ("case", "CASE", BLUE, "s", "-"),
          ("oracle_sel", "Cellwise-perfect assignment repair (unachievable)", INK, "D", "--")]


def style():
    sns.set_context("paper")
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load_rows(path):
    with open(path) as fh:
        blob = json.load(fh)
    if "rows" not in blob:
        raise SystemExit(f"{path}: expected a top-level 'rows' key")
    df = pd.DataFrame(blob["rows"])
    need = {"assignment", "evaluator", "method", "budget", "partition",
            "acq_seed", "reference_regret", "top1_error"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df


def cell(df, assignment, evaluator, method, budget):
    """Average acquisition seeds within each partition, then mean and SE over partitions."""
    sub = df[(df.assignment == assignment) & (df.evaluator == evaluator)
             & (df.method == method) & (df.budget == budget)]
    if sub.empty:
        raise SystemExit(f"no rows for {assignment}/{evaluator}/{method} at budget {budget}")
    per = sub.groupby("partition")["reference_regret"].mean()
    if len(per) != N_PARTITIONS:
        raise SystemExit(f"{assignment}/{evaluator}/{method}@{budget}: "
                         f"{len(per)} partitions, expected {N_PARTITIONS}")
    v = per.to_numpy(float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validity", required=True)
    ap.add_argument("--paper-numbers", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    style()
    df = load_rows(args.validity)
    print(f"read {args.validity}: {len(df):,} rows")

    budgets = sorted(df.budget.unique())
    for b in (0.15, 0.25, 0.35):
        if b not in budgets:
            raise SystemExit(f"budget {b} missing from the source (found {budgets})")
    print(f"budgets on the x-axis: {budgets}")

    # computed over ALL_CELLS, not just the plotted ones, so the assertions below and the
    # exported _data.csv still cover the comparable-request cell that panel (a) used to show.
    titles = {(a, e): t for a, e, t in PANELS}
    rec = []
    for a, e in ALL_CELLS:
        title = titles.get((a, e), f"{a}/{e} (verified, not plotted)")
        for m, label, _, _, _ in SERIES:
            for b in budgets:
                mu, se, n = cell(df, a, e, m, b)
                rec.append(dict(assignment=a, evaluator=e, condition=title.replace("\n", " "),
                                method=m, method_label=label, budget=b,
                                mean=mu, se=se, n_partitions=n))
    tidy = pd.DataFrame(rec)

    def get(a, e, m, b):
        r = tidy[(tidy.assignment == a) & (tidy.evaluator == e)
                 & (tidy.method == m) & (tidy.budget == b)]
        return float(r["mean"].iloc[0])

    # assertions only; nothing below positions a marker from these
    for (a, e, m, b), want in [(("A0", "D0", "trust", 0.35), 0.0214),
                               (("A0", "D0", "case", 0.35), 0.0321),
                               (("A1", "D0", "trust", 0.35), 0.0499),
                               (("A1", "D0", "case", 0.35), 0.0338),
                               (("A1", "D1", "trust", 0.35), 0.0469),
                               (("A1", "D1", "case", 0.35), 0.0338),
                               (("A1", "D1", "oracle_sel", 0.35), 0.0030)]:
        got = get(a, e, m, b)
        assert abs(got - want) < 5e-4, f"{a}/{e}/{m}@{b}: {got:.4f} != {want}"
    # the logged-only arm cannot move with the audit budget
    for a, e in ALL_CELLS:
        vals = {get(a, e, "trust", b) for b in budgets}
        assert max(vals) - min(vals) < 1e-9, f"trust varies with budget in {a}/{e}"
    # the sign flip the figure exists to show
    assert get("A0", "D0", "case", 0.35) > get("A0", "D0", "trust", 0.35)
    assert get("A1", "D0", "case", 0.35) < get("A1", "D0", "trust", 0.35)
    # CASE improves monotonically with budget in every panel
    for a, e in ALL_CELLS:
        seq = [get(a, e, "case", b) for b in budgets]
        assert all(x > y for x, y in zip(seq, seq[1:])), f"CASE not monotone in {a}/{e}"

    top1 = df[(df.assignment == "A1") & (df.evaluator == "D1")
              & (df.budget == 0.35) & (df.method == "oracle_sel")]
    top1_rate = float(top1.groupby("partition")["top1_error"].mean().mean())
    assert abs(top1_rate - 0.183) < 5e-3, f"top-1 error {top1_rate:.4f} != 0.183"
    print(f"caption check: perfect assignment repair still picks the wrong model in "
          f"{top1_rate:.1%} of request types")

    if os.path.exists(args.paper_numbers):
        with open(args.paper_numbers) as fh:
            json.load(fh)
        print(f"read {args.paper_numbers} (decomposition cross-check available)")

    # ---------------------------------------------------------------- draw
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.35), sharey=True)
    xs = np.arange(len(budgets))
    handles = None
    for ax, (a, e, title) in zip(axes, PANELS):
        for m, label, color, marker, ls in SERIES:
            ys = [get(a, e, m, b) for b in budgets]
            es = [float(tidy[(tidy.assignment == a) & (tidy.evaluator == e)
                             & (tidy.method == m) & (tidy.budget == b)]["se"].iloc[0])
                  for b in budgets]
            ax.errorbar(xs, ys, yerr=es, fmt=marker, ls=ls, color=color, ecolor=color,
                        elinewidth=0.8, capsize=2, markeredgecolor=color,
                        label=label, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(round(b * 100))}\\%" if False else f"{int(round(b * 100))}%"
                            for b in budgets])
        ax.set_xlim(-0.3, len(budgets) - 0.7)
        ax.set_title(title)
        ax.set_xlabel("Randomized audit budget")
        ax.yaxis.grid(True, lw=0.4, color="#e6e6e6")
        ax.set_axisbelow(True)
    # one line: a two-line rotated label is clipped at this panel height
    axes[0].set_ylabel("Target-value loss")
    axes[0].set_ylim(bottom=-0.002)
    # the panels are short, so matplotlib drops to two ticks; pin three so the scale stays readable
    axes[0].set_yticks([0.00, 0.02, 0.04])

    sns.despine(fig=fig)
    fig.tight_layout(pad=0.3, w_pad=0.9, rect=(0, 0.20, 1, 1))
    # one shared legend below the panels, so no series is hidden behind a legend box
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
               handlelength=1.8, handletextpad=0.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.03))

    # ---------------------------------------------------------------- write
    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stem = os.path.splitext(out)[0]
    fig.savefig(out)
    fig.savefig(stem + ".png", dpi=400)
    tidy.to_csv(stem + "_data.csv", index=False)

    print(f"\n{'condition':34}{'method':28}{'budget':>8}{'mean':>9}{'se':>9}")
    for r in rec:
        print(f"{r['condition']:34}{r['method_label']:28}{r['budget']:>8.2f}"
              f"{r['mean']:>9.4f}{r['se']:>9.4f}")
    print(f"\nwrote {out}\nwrote {stem}.png\nwrote {stem}_data.csv")


if __name__ == "__main__":
    main()
