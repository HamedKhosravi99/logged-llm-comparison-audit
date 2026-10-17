"""Decision margins as an eta-sensitivity curve.

A partition is settled at evaluator spread eta exactly when its decision margin is at least eta,
since the margin is the largest common spread at which the incumbent allocation stays optimal over
U and the uncertainty sets are nested in eta. The share of the 30 partitions settled at each eta is
therefore a re-count of the stored margins, with no new solves. Margins of zero (exact ties at the
estimated table) never count as settled.

Every plotted coordinate comes from source/budgeted_certificate/cert30_<corpus>_row.json, whose
margin_box is the margin over the box U of the main text and margin_row the margin over the model-level set
of Assumption B.2. The printed table reproduces the median and count columns of Table 2.

Usage:
    python code/make_fig_eta_sensitivity.py --output figures/fig_eta_sensitivity.pdf
"""
import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import seaborn as sns                                                  # noqa: E402

GREY = "#6f6f6f"
BLUE = "#2a78d6"
INK = "#1a1a1a"

SRC = "source/budgeted_certificate/cert30_{}_row.json"
PANELS = [("llmrouterbench", "(a) LLMRouterBench"), ("routerbench", "(b) RouterBench")]
BUDGETS = [(1.25, r"$1.25\times$", BLUE), (2.0, r"$2\times$", GREY), (8.0, r"$8\times$", INK)]
MARKERS = [(0.02, "replays"), (0.128, "KuaiRand"), (0.336, "LMArena judge")]
ETA_LO, ETA_HI = 0.002, 0.4


def style():
    sns.set_context("paper")
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def load(corpus):
    rows = json.load(open(SRC.format(corpus)))
    out = {}
    for r in rows:
        out.setdefault(float(r["mult"]), []).append((r["margin_box"], r["margin_row"]))
    for mult, ms in out.items():
        assert len(ms) == 30, (corpus, mult, len(ms))
    return out


def settled_share(margins, grid):
    m = np.asarray(margins)[:, None]
    return ((m >= grid[None, :]) & (m > 0)).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    style()
    grid = np.geomspace(ETA_LO, ETA_HI, 800)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)

    print(f"{'corpus':16}{'budget':>8}{'median':>9}{'>=0.01':>8}{'>=0.02':>8}"
          f"{'>=0.128 box':>13}{'>=0.128 row':>13}")
    for ax, (corpus, title) in zip(axes, PANELS):
        data = load(corpus)
        for mult, label, colour in BUDGETS:
            box = [b for b, _ in data[mult]]
            row = [r for _, r in data[mult]]
            ax.plot(grid, settled_share(box, grid), color=colour, ls="-", label=label)
            ax.plot(grid, settled_share(row, grid), color=colour, ls="--")
            print(f"{corpus:16}{mult:>8}{np.median(box):>9.4f}"
                  f"{sum(b >= 0.01 for b in box):>8}{sum(b >= 0.02 for b in box):>8}"
                  f"{sum(b >= 0.128 for b in box):>13}{sum(r >= 0.128 for r in row):>13}")
        for eta, name in MARKERS:
            ax.axvline(eta, color=GREY, lw=0.7, ls=":")
            ax.text(eta * 1.06, 1.02, name, fontsize=7, color=GREY, rotation=90,
                    va="top", ha="left")
        ax.set_xscale("log")
        ax.set_xlim(ETA_LO, ETA_HI)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel(r"evaluator spread $\eta$")
        sns.despine(ax=ax)
    axes[0].set_ylabel("share settled")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [mpl.lines.Line2D([], [], color=INK, ls="-"),
                mpl.lines.Line2D([], [], color=INK, ls="--")]
    labels += ["box", "model-level set"]
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=5,
               handlelength=1.8, bbox_to_anchor=(0.5, -0.2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
