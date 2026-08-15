"""Illustrative instance of the budgeted allocation problem for Sec. 2 (Fig. fig:alloc).

ILLUSTRATIVE, NOT AN EMPIRICAL RESULT. Every number below is invented for exposition; the
caption says so and the paper's empirical instances are in Sec. 5. This script exists so the
figure is reproducible and its numbers auditable, not because it reads any data.

Panel (a) solves the multiple-choice knapsack of Eq. (1) on an estimated quality table. Panel
(b) solves it on an admissible table that differs only by a model-level evaluator offset C on
the "Small" model -- the form Thm. 3.3 shows no audit budget can rule out. Only the Small row
changes numerically, yet the coding and drafting allocations both move, because the shared
budget row couples workloads that the delta-satisficing rule of Eq. (2) would decide separately.
Those two moved decisions carry the heavier double outline; the other two are outlined thinly.

The instance was chosen so that both panels bind the budget and exactly two workloads flip; see
the parameter search recorded in docs/V5_HANDOFF.md.

Usage: python make_fig_allocation.py --outdir ../figures
"""
import argparse, os
from itertools import product
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# --- the invented instance -------------------------------------------------------------------
WORKLOAD = ["coding", "support", "analysis", "drafting"]
VOLUME = np.array([120., 400., 60., 220.])          # thousands of requests per month
MODEL = ["Large", "Mid", "Small"]
PRICE = np.array([26., 12., 5.])                    # dollars per thousand requests
MU = np.array([[0.92, 0.90, 0.88, 0.86],            # rows models, columns workloads
               [0.84, 0.88, 0.81, 0.85],
               [0.71, 0.83, 0.68, 0.82]])
BUDGET = 10_000.                                    # Gamma, dollars per month
OFFSET = np.array([0., 0., 0.06])                   # C_k: the evaluator over-rates "Small"

# Dark navy reads as a single dark tone in greyscale; the pale ramp never approaches it.
ACCENT = "#17365d"
PALE = LinearSegmentedColormap.from_list("pale_grey", ["#fcfcfc", "#bdbdbd"])


def solve(mu):
    """Exact multiple-choice knapsack by enumeration; 4^4 allocations, so enumeration is honest."""
    best_value, best_x = -np.inf, None
    for x in product(range(len(MODEL)), repeat=len(WORKLOAD)):
        cost = sum(VOLUME[j] * PRICE[x[j]] for j in range(len(WORKLOAD)))
        if cost > BUDGET:
            continue
        value = sum(VOLUME[j] * mu[x[j], j] for j in range(len(WORKLOAD)))
        if value > best_value:
            best_value, best_x = value, x
    cost = sum(VOLUME[j] * PRICE[best_x[j]] for j in range(len(WORKLOAD)))
    return best_x, cost, best_value / VOLUME.sum()


def main(outdir):
    mu_b = MU - OFFSET[:, None]
    xa, cost_a, q_a = solve(MU)
    xb, cost_b, q_b = solve(mu_b)
    print(f"(a) {[MODEL[k] for k in xa]}  spend ${cost_a:,.0f}  quality {q_a:.4f}")
    print(f"(b) {[MODEL[k] for k in xb]}  spend ${cost_b:,.0f}  quality {q_b:.4f}")

    changed = [j for j in range(len(WORKLOAD)) if xa[j] != xb[j]]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black"})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 1.80))
    panels = [(axes[0], MU, xa, cost_a, q_a, "(a) Observed quality table"),
              (axes[1], mu_b, xb, cost_b, q_b, "(b) Another admissible target table")]
    for ax, mu, x, cost, quality, title in panels:
        # Pale ramp only: the numbers carry the exact values, the shading carries the ordering,
        # and every cell stays light enough for black text and for greyscale printing.
        ax.imshow(mu, cmap=PALE, vmin=.45, vmax=1.00, aspect="auto")
        for k in range(len(MODEL)):
            for j in range(len(WORKLOAD)):
                ax.text(j, k, f"{mu[k, j]:.2f}", ha="center", va="center",
                        fontsize=9.2, color="black")
        for j in range(len(WORKLOAD)):
            # Every allocated cell gets a thin outline; the decisions that move between panels
            # get a heavier double outline, which survives greyscale as a weight difference.
            # Inset slightly so outlines on horizontally adjacent cells stay separate boxes
            # rather than merging into one wide rectangle.
            if j in changed:
                ax.add_patch(Rectangle((j - .46, x[j] - .46), .92, .92, fill=False,
                                       lw=2.5, ec=ACCENT))
                ax.add_patch(Rectangle((j - .38, x[j] - .38), .76, .76, fill=False,
                                       lw=0.9, ec="white"))
            else:
                ax.add_patch(Rectangle((j - .46, x[j] - .46), .92, .92, fill=False,
                                       lw=0.9, ec=ACCENT))
        ax.set_xticks(range(len(WORKLOAD)))
        ax.set_yticks(range(len(MODEL)))
        ax.set_xticklabels([f"{w}\n{v:.0f}k" for w, v in zip(WORKLOAD, VOLUME)], fontsize=8.4)
        ax.set_yticklabels([f"{m} \\${p:.0f}" for m, p in zip(MODEL, PRICE)], fontsize=8.4)
        ax.set_title(title, fontsize=9.6, pad=6)
        ax.set_xlabel(f"Spend \\${cost:,.0f} of \\${BUDGET:,.0f} · weighted quality "
                      f"{quality:.3f}", fontsize=8.8, labelpad=3)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=.35)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 210})):
        path = os.path.join(outdir, f"fig_allocation_illustrative.{ext}")
        fig.savefig(path, facecolor="white", **kw)
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
    main(ap.parse_args().outdir)
