"""Decision margins by serving budget, against the evaluator spreads measured on real data.

A partition is settled at evaluator spread eta exactly when its decision margin is at least eta,
since the margin is the largest common spread at which the incumbent allocation stays optimal over
U and the uncertainty sets are nested in eta. Plotting the margins against the measured spreads is
therefore the eta-sensitivity of every verdict, with no new solves.

Every plotted coordinate comes from source/budgeted_certificate/cert30_<corpus>_row.json, whose
margin_box is the margin over the box U of the main text and margin_row the margin over the
model-level set of Assumption B.2. The printed table reproduces the median and count columns of
Table 2. Margins of zero (exact ties at the estimated table) are drawn at the axis floor.

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

SRC = "source/budgeted_certificate/cert30_{}_row.json"
PANELS = [("llmrouterbench", "(a) LLMRouterBench"), ("routerbench", "(b) RouterBench")]
BLUE = "#2a78d6"
ORANGE = "#e07b39"
REF_GREY = "0.35"
REFS = [(0.336, "LMArena judge", (0, (6, 2))), (0.128, "KuaiRand", "-."), (0.02, "replays", ":")]
FLOOR = 1e-4


def style():
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "lines.markersize": 5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def load(corpus):
    rows = json.load(open(SRC.format(corpus)))
    mults = sorted({float(r["mult"]) for r in rows})
    box = [np.array([r["margin_box"] for r in rows if float(r["mult"]) == m]) for m in mults]
    row = [np.array([r["margin_row"] for r in rows if float(r["mult"]) == m]) for m in mults]
    for b in box:
        assert len(b) == 30, (corpus, len(b))
    return mults, box, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    print(f"{'corpus':16}{'budget':>8}{'median':>9}{'>=0.01':>8}{'>=0.02':>8}{'>=0.128':>9}")
    for ax, (corpus, title) in zip(axes, PANELS):
        mults, box, row = load(corpus)
        xs = np.arange(len(mults))
        q25 = np.maximum([np.percentile(b, 25) for b in box], FLOOR)
        q75 = [np.percentile(b, 75) for b in box]
        ax.fill_between(xs, q25, q75, color=BLUE, alpha=0.15, lw=0,
                        label="middle half of partitions")
        ax.plot(xs, np.maximum([np.median(b) for b in box], FLOOR), color=BLUE, ls="-",
                marker="o", label="median, error per cell")
        ax.plot(xs, np.maximum([np.median(r) for r in row], FLOOR), color=ORANGE, ls="--",
                marker="s", mfc="white", mew=1.2, label="median, one error per model")
        for m, b in zip(mults, box):
            print(f"{corpus:16}{m:>8g}{np.median(b):>9.4f}{(b >= 0.01).sum():>8}"
                  f"{(b >= 0.02).sum():>8}{(b >= 0.128).sum():>9}")
        for eta, name, ls in REFS:
            ax.axhline(eta, color=REF_GREY, lw=0.9, ls=ls)
            ax.text(len(mults) - 0.55, eta * 1.12, f"{name} {eta:g}", fontsize=7,
                    color=REF_GREY, ha="right", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m:g}" for m in mults])
        ax.set_xlabel(r"serving budget $\Gamma$ ($\times$ cheapest feasible)")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 0.6)
        ax.set_xlim(-0.4, len(mults) - 0.6)
        ax.set_title(title)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("decision margin")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
