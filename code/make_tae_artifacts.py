"""Build every figure, table and reported number of the validity paper from the result files.

    python make_tae_artifacts.py

Reads  results/tae_validity.json, results/tae_validity_appendix.json
Writes figs/tae_fig1.pdf, figs/tae_fig2.pdf
       tae_tables.tex               (LaTeX fragments the paper \\input{}s)
       results/tae_paper_numbers.json (claim registry: every number quoted in the text)

Aggregation rule (fixed in the analysis plan): acquisition seeds are averaged WITHIN a prompt-cohort
partition first; the partition is the replication unit, and every interval is a paired standard
error over partitions.
"""
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")

SLICES = ["simpleqa", "mmlupro", "hle", "livecodebench", "arenahard"]
SLICE_LABEL = {"simpleqa": "SimpleQA", "mmlupro": "MMLU-Pro", "hle": "HLE",
               "livecodebench": "LiveCodeBench", "arenahard": "ArenaHard"}
BUDGETS = [0.15, 0.25, 0.35]

# Five distinct validated hues (adjacent-pair CVD dE 9.1, normal-vision 22.9 on the light surface);
# marker and dash carry the secondary encoding, and every figure is backed by a table, which
# discharges the low-contrast relief rule on the aqua and yellow slots. Slots are assigned once, by
# entity, never cycled. Series order below IS the adjacency order the palette was validated in.
STYLE = {
    "trust":      dict(label="Trust operational eval.", color="#2a78d6", marker="o", ls="-"),
    "uniform":    dict(label="Uniform randomized audit", color="#eb6834", marker="s", ls="--"),
    "epig_obs":   dict(label="R-EPIG, observed surface", color="#4a3aa7", marker="^", ls="-."),
    "case":       dict(label="CASE, deconfounded surface", color="#1baf7a", marker="D", ls="-"),
    "epig_corr":  dict(label="R-EPIG, deconfounded surface", color="#eda100", marker="v",
                       ls=(0, (1.6, 1.4))),
    "rawmargin":  dict(label="Raw margin, observed surface", color="#6b6b68", marker="v", ls=":"),
    "oracle_sel": dict(label="Population-operational oracle", color="#6b6b68", marker=None, ls=(0, (4, 2))),
    "oracle_ref": dict(label="Reference-outcome oracle", color="#0b0b0b", marker=None, ls=":"),
}

# the paper labels the evaluator factor D (distortion); the result files key it E. Display only.
CELL_LABEL = {("A0", "E0"): "A0–D0\nneither", ("A1", "E0"): "A1–D0\nassign.",
              ("A0", "E1"): "A0–D1\nmeasure.", ("A1", "E1"): "A1–D1\nboth"}
COND_LABEL = {"E0": "D0", "E1": "D1", "E2": "D2", "E1k": "D1k", "E2rev": "D2rev"}

plt.rcParams.update({
    "font.size": 7.2, "axes.titlesize": 7.8, "axes.labelsize": 7.2,
    "legend.fontsize": 6.6, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.6,
    "grid.color": "#e2e2de", "grid.linewidth": 0.5,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------- aggregation
def load(name):
    return json.load(open(os.path.join(RES, name)))


class Agg:
    """Partition-level aggregation of the long-form result rows."""

    def __init__(self, rows, key_fields, sweep=None):
        self.d = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            if sweep is not None and r.get("sweep") != sweep:
                continue
            self.d[tuple(r[f] for f in key_fields)][r["partition"]].append(r)
        self.key_fields = key_fields

    def series(self, key, field, sub=None):
        """One value per partition (mean over acquisition seeds)."""
        recs = self.d[tuple(key)]
        parts = sorted(recs)
        vals = []
        for p in parts:
            v = [(r[field][sub] if sub is not None else r[field]) for r in recs[p]]
            vals.append(np.nanmean(v))
        return np.asarray(vals, float)

    def stat(self, key, field, sub=None):
        v = self.series(key, field, sub)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / np.sqrt(len(v))), len(v)

    def paired(self, k_a, k_b, field, sub=None):
        d = self.series(k_a, field, sub) - self.series(k_b, field, sub)
        return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                    win=float(np.mean(d < 0)), n=len(d))


def fmt(m, se, dec=4):
    return f"${m:.{dec}f}{{\\scriptstyle\\pm{se:.{dec}f}}}$"


# --------------------------------------------------------------------------- figures
def figure1(F, out):
    """Method comparison only. The oracles are an order of magnitude below every method, so plotting
    them here would squeeze the whole comparison into the top third of the panel; Fig. 2 carries the
    oracle gap instead. The y-axis is therefore fitted to the methods and does not start at zero,
    which the caption states."""
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.88), gridspec_kw=dict(wspace=0.08))
    x = np.arange(len(BUDGETS))
    lo, hi = 0.0305, 0.0555
    for a, (cond, title) in zip(ax, [("E0", "(a) faithful evaluator (A1\u2013D0)"),
                                     ("E1", "(b) distorted evaluator (A1\u2013D1)")]):
        for m in ["trust", "uniform", "epig_obs", "case", "epig_corr"]:
            y = [F.stat(("A1", cond, b, m), "reference_regret") for b in BUDGETS]
            st = STYLE[m]
            a.errorbar(x, [v[0] for v in y], yerr=[v[1] for v in y], color=st["color"],
                       marker=st["marker"], ls=st["ls"], lw=1.6, ms=5.2, capsize=2.6,
                       elinewidth=0.9, label=st["label"], zorder=3)
        v_sel = F.stat(("A1", cond, 0.35, "oracle_sel"), "reference_regret")[0]
        # full-scale inset, so the truncation cannot make the method spread look larger than the
        # gap to the oracle that it actually sits inside
        ins = a.inset_axes([0.105, 0.07, 0.355, 0.375])
        for m in ["trust", "case"]:
            yy = [F.stat(("A1", cond, b_, m), "reference_regret")[0] for b_ in BUDGETS]
            ins.plot(x, yy, color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=1.0,
                     marker=STYLE[m]["marker"], ms=2.2)
        for m in ["oracle_sel", "oracle_ref"]:
            ins.axhline(F.stat(("A1", cond, 0.35, m), "reference_regret")[0],
                        color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=0.8)
        ins.set_ylim(-0.003, 0.058)
        ins.set_xlim(-0.28, 2.28)
        ins.set_xticks(x, ["15", "25", "35"], fontsize=5.2)
        ins.set_yticks([0, 0.05], ["0", "0.05"], fontsize=5.2)
        ins.tick_params(length=1.6, pad=1)
        ins.set_title("full scale, with oracles", fontsize=5.6, pad=1.5, color="#52514e",
                      bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        ins.grid(False)
        for sp in ins.spines.values():
            sp.set_linewidth(0.5)
        a.set_xticks(x, [f"{int(b*100)}%" for b in BUDGETS], fontsize=8.0)
        a.set_xlabel("randomized cell budget", labelpad=2, fontsize=8.0)
        a.set_title(title, pad=5, loc="left", fontsize=8.4)
        a.grid(axis="y", zorder=0)
        a.set_ylim(lo, hi)
        a.set_xlim(-0.28, 2.28)
        a.set_yticks(np.arange(0.035, 0.0551, 0.005))
        a.tick_params(labelsize=8.0)
        a.spines["bottom"].set_visible(False)
        a.plot([0, 1], [0, 0], transform=a.transAxes, color="#8a8a86", lw=0.6,
               clip_on=False, dashes=(3, 3), zorder=1)   # broken baseline: axis is truncated
    ax[0].set_ylabel("reference-channel\ndeployment regret", labelpad=3, fontsize=8.0)
    ax[1].set_yticklabels([])
    h, l = ax[0].get_legend_handles_labels()
    # the two oracle lines are drawn only inside the inset, so they need their own legend entries
    for m in ["oracle_sel", "oracle_ref"]:
        st = STYLE[m]
        h.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=1.0))
        l.append(st["label"] + " (inset)")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.34),
               handlelength=1.9, columnspacing=1.0, handletextpad=0.5, labelspacing=0.35,
               fontsize=7.0)
    fig.savefig(out)
    plt.close(fig)


def figure3(F, out):
    """Per-slice contributions, moved out of figure 1 so the budget panels can be full size."""
    fig, a = plt.subplots(figsize=(5.5, 1.85))
    ms = ["trust", "uniform", "epig_obs", "case", "epig_corr"]
    w, xs = 0.168, np.arange(len(SLICES))
    for i, m in enumerate(ms):
        st = STYLE[m]
        vals = [F.stat(("A1", "E1", 0.35, m), "slice_regret", sub=s) for s in SLICES]
        a.bar(xs + (i - 2) * w, [v[0] for v in vals], w * 0.84, yerr=[v[1] for v in vals],
              color=st["color"], edgecolor="white", linewidth=0.6, capsize=1.4,
              error_kw=dict(elinewidth=0.7, ecolor="#52514e"), label=st["label"], zorder=3)
    a.set_xticks(xs, [SLICE_LABEL[s] for s in SLICES], fontsize=7.6)
    a.tick_params(axis="x", pad=1.5)
    a.set_ylabel("reference regret", labelpad=3, fontsize=7.8)
    a.grid(axis="y", zorder=0)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.30),
               handlelength=2.2, columnspacing=1.3, fontsize=7.2)
    fig.savefig(out)
    plt.close(fig)


def figure2(F, out):
    cells = [("A0", "E0"), ("A1", "E0"), ("A0", "E1"), ("A1", "E1")]
    ms = ["trust", "case", "oracle_sel"]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.50), gridspec_kw=dict(wspace=0.30))
    xs = np.arange(len(cells))
    w = 0.24
    for a, field, ylab, top in [(ax[0], "reference_regret",
                                 "reference-channel\ndeployment regret", 0.062),
                                (ax[1], "top1_error", "top-1\nidentification error", 0.80)]:
        for i, m in enumerate(ms):
            st = STYLE[m]
            vals = [F.stat((A, E, 0.35, m), field) for A, E in cells]
            a.bar(xs + (i - 1) * w, [v[0] for v in vals], w * 0.84, yerr=[v[1] for v in vals],
                  color=st["color"], edgecolor="white", linewidth=0.5, capsize=1.4,
                  error_kw=dict(elinewidth=0.5, ecolor="#52514e"), label=st["label"], zorder=3)
            for xi, v in zip(xs, vals):                       # zero bars must stay legible
                if v[0] < top * 0.012:
                    a.annotate("0", (xi + (i - 1) * w, top * 0.012), fontsize=5.4, ha="center",
                               va="bottom", color=st["color"])
        # shade the two common-population cells: there is no assignment invalidity to repair there,
        # so CASE sitting ABOVE trust is the expected cost of auditing, not a failure of the method
        for xi in (0, 2):
            a.axvspan(xi - 0.48, xi + 0.48, color="#f0efe9", zorder=0)
        for xi in (0, 2):
            a.annotate("nothing\nto repair", (xi, 0.90),
                       xycoords=("data", "axes fraction"), fontsize=5.8, color="#6b6b68",
                       ha="center", va="top", linespacing=1.15)
        a.axhline(0, color="#0b0b0b", ls=":", lw=0.9, zorder=2)
        a.set_xticks(xs, [CELL_LABEL[c] for c in cells], fontsize=5.9, linespacing=1.25)
        a.tick_params(axis="x", pad=1.5)
        a.set_ylabel(ylab, labelpad=2, fontsize=6.8)
        a.set_ylim(0, top)
        a.set_xlim(-0.62, 4.35)
        a.grid(axis="y", zorder=0)

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17),
               handlelength=1.6, columnspacing=1.4)
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- tables
def table_deployment(G, meta):
    arms = [("ungated", "CASE, ungated"),
            ("same_evaluator", "\\;$+$ same-evaluator check"),
            ("reference_channel", "\\;$+$ reference-channel check")]
    lines = []
    for cond, cname in [("E1", "A1--D1 (stable evaluator distortion)"),
                        ("E2", "A1--D2 (evaluator differs between log and audit)")]:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{cname}}}}}\\\\")
        for arm, label in arms:
            k = (cond, 0.35, arm)
            reg = G.stat(k, "reference_regret")
            fs = G.stat(k, "false_switch")
            fr = G.stat(k, "false_retention")
            sw = G.stat(k, "switch_rate")
            inv = G.stat(k, "channel_verdict_inversion")
            acq = G.stat(k, "acq_cells")[0]
            ver = G.stat(k, "ver_cells_used")[0]
            lines.append(
                f"{label} & {fmt(*reg[:2])} & {fmt(*fs[:2], 3)} & {fmt(*fr[:2], 3)} & "
                f"{sw[0]:.2f} & {acq:.0f}\\,/\\,{ver:.0f} & "
                + ("--" if arm == "ungated" else f"${inv[0]*100:.1f}\\%$") + "\\\\")
        lines.append("\\addlinespace[2pt]")
    return "\n".join(lines)


def table_factorial(F):
    ms = ["trust", "uniform", "case", "epig_corr", "oracle_sel", "oracle_ref"]
    out = []
    for A, E in [("A0", "E0"), ("A1", "E0"), ("A0", "E1"), ("A1", "E1")]:
        row = [f"{A}--{COND_LABEL[E]}"]
        for m in ms:
            mu, se, _ = F.stat((A, E, 0.35, m), "reference_regret")
            row.append(fmt(mu, se))
        out.append(" & ".join(row) + "\\\\")
    return "\n".join(out)


SHORT = {"trust": "Trust log", "uniform": "Uniform audit",
         "epig_obs": r"R-EPIG, obs.\ surface", "rawmargin": r"Raw margin, obs.\ surface",
         "case": "CASE, deconf.\ surface", "epig_corr": r"R-EPIG, deconf.\ surface"}


def table_slices(F):
    ms = ["trust", "uniform", "rawmargin", "epig_obs", "case", "epig_corr"]
    out = []
    for cond in ["E0", "E1"]:
        out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{A1--{COND_LABEL[cond]}}}}}\\\\")
        for m in ms:
            row = [SHORT[m]]
            vals = []
            for s in SLICES:
                mu, se, _ = F.stat(("A1", cond, 0.35, m), "slice_regret", sub=s)
                vals.append(mu)
                row.append(f"${mu:.4f}$")
            row.append(f"${np.mean(vals):.4f}$")
            out.append(" & ".join(row) + "\\\\")
        out.append("\\addlinespace[2pt]")
    return "\n".join(out)


# --------------------------------------------------------------------------- numbers
def paper_numbers(D, A, F, G):
    P = {}
    dg = [m["diagnostics"] for m in D["meta"]]
    for k in dg[0]:
        v = np.array([d[k] for d in dg])
        P[f"diag.{k}"] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))))
    lam = np.array([m["lam"] for m in D["meta"]])
    P["diag.lambda"] = dict(mean=float(lam.mean()), lo=float(lam.min()), hi=float(lam.max()))
    P["diag.n_supported"] = float(np.mean([m["n_supported"] for m in D["meta"]]))

    for A_, E_ in [("A0", "E0"), ("A1", "E0"), ("A0", "E1"), ("A1", "E1")]:
        for b in BUDGETS:
            for m in ["trust", "uniform", "case", "epig_corr", "oracle_sel", "oracle_ref",
                      "rawmargin", "epig_obs"]:
                mu, se, n = F.stat((A_, E_, b, m), "reference_regret")
                P[f"reg.{A_}.{E_}.{int(b*100)}.{m}"] = dict(mean=mu, se=se, n=n)
                mu, se, _ = F.stat((A_, E_, b, m), "top1_error")
                P[f"top1.{A_}.{E_}.{int(b*100)}.{m}"] = dict(mean=mu, se=se)
        tr = F.series((A_, E_, 0.35, "trust"), "reference_regret")
        ca = F.series((A_, E_, 0.35, "case"), "reference_regret")
        os_ = F.series((A_, E_, 0.35, "oracle_sel"), "reference_regret")
        orf = F.series((A_, E_, 0.35, "oracle_ref"), "reference_regret")
        P[f"gap.{A_}.{E_}.G_sel"] = dict(mean=float((tr - os_).mean()),
                                         se=float((tr - os_).std(ddof=1) / np.sqrt(len(tr))))
        P[f"gap.{A_}.{E_}.G_meas"] = dict(mean=float((os_ - orf).mean()),
                                          se=float((os_ - orf).std(ddof=1) / np.sqrt(len(tr))))
        ok = (tr - os_) > 1e-4
        P[f"gap.{A_}.{E_}.F_repair"] = dict(
            mean=float(np.mean((tr[ok] - ca[ok]) / (tr[ok] - os_[ok]))) if ok.any() else None,
            n_defined=int(ok.sum()), n=len(tr))
        # the ratio of the aggregate means is a different estimand from the mean of the
        # partition-level ratios; the paper quotes both rather than letting them be confused
        P[f"gap.{A_}.{E_}.F_repair_ratio_of_means"] = dict(
            mean=float((tr.mean() - ca.mean()) / (tr.mean() - os_.mean()))
            if tr.mean() - os_.mean() > 1e-4 else None)

    for A_, E_, b in [("A1", "E0", 0.35), ("A1", "E1", 0.35), ("A0", "E1", 0.35),
                      ("A0", "E0", 0.35), ("A1", "E0", 0.25), ("A1", "E1", 0.25)]:
        P[f"paired.{A_}.{E_}.{int(b*100)}.case_vs_trust"] = F.paired(
            (A_, E_, b, "case"), (A_, E_, b, "trust"), "reference_regret")
        P[f"paired.{A_}.{E_}.{int(b*100)}.case_vs_epigcorr"] = F.paired(
            (A_, E_, b, "case"), (A_, E_, b, "epig_corr"), "reference_regret")
        P[f"paired.{A_}.{E_}.{int(b*100)}.uniform_vs_trust"] = F.paired(
            (A_, E_, b, "uniform"), (A_, E_, b, "trust"), "reference_regret")
    P["paired.A1.construct_gap"] = F.paired(("A1", "E1", 0.35, "case"),
                                            ("A1", "E0", 0.35, "case"), "reference_regret")
    P["paired.A1.stream_mismatch"] = F.paired(("A1", "E2", 0.35, "case"),
                                              ("A1", "E1", 0.35, "case"), "reference_regret")
    for s in SLICES:
        for cond in ["E0", "E1"]:
            P[f"slice.{cond}.{s}.case_vs_trust"] = F.paired(
                ("A1", cond, 0.35, "case"), ("A1", cond, 0.35, "trust"), "slice_regret", sub=s)

    for cond in ["E1", "E2"]:
        for b in BUDGETS:
            for arm in ["ungated", "same_evaluator", "reference_channel"]:
                for f in ["reference_regret", "false_switch", "false_retention", "switch_rate",
                          "top1_error", "channel_verdict_inversion", "ver_cells_used",
                          "n_contested", "n_contested_scored"]:
                    mu, se, _ = G.stat((cond, b, arm), f)
                    P[f"gate.{cond}.{int(b*100)}.{arm}.{f}"] = dict(mean=mu, se=se)
            P[f"gate.{cond}.{int(b*100)}.same_vs_ref.false_switch"] = G.paired(
                (cond, b, "same_evaluator"), (cond, b, "reference_channel"), "false_switch")

    # RouterBench replication and the four reviewer diagnostics
    rp_path = os.path.join(RES, "tae_replication.json")
    if os.path.exists(rp_path):
        RP = json.load(open(rp_path))
        RPa = Agg(RP["rows"], ["assignment", "evaluator", "budget", "method"])
        for key in RPa.d:
            if key[2] != 0.35:
                continue
            for f in ("reference_regret", "top1_error"):
                mu, se, _ = RPa.stat(key, f)
                P[f"rep.{f.split('_')[0]}.{key[0]}.{key[1]}.{key[3]}"] = dict(mean=mu, se=se)
        for A_, E_ in (("A1", "E0"), ("A1", "E1"), ("A0", "E0")):
            P[f"rep.paired.{A_}.{E_}.case_vs_trust"] = RPa.paired(
                (A_, E_, 0.35, "case"), (A_, E_, 0.35, "trust"), "reference_regret")
            P[f"rep.paired.{A_}.{E_}.case_vs_epigcorr"] = RPa.paired(
                (A_, E_, 0.35, "case"), (A_, E_, 0.35, "epig_corr"), "reference_regret")
        dg = [m["diagnostics"] for m in RP["meta"]]
        for k in dg[0]:
            v = np.array([d[k] for d in dg])
            P[f"rep.diag.{k}"] = dict(mean=float(v.mean()),
                                      se=float(v.std(ddof=1) / np.sqrt(len(v))))
    dx_path = os.path.join(RES, "tae_diagnostics.json")
    if os.path.exists(dx_path):
        for block, val in json.load(open(dx_path)).items():
            for k, v in val.items():
                P[f"diagx.{block}.{k}"] = v

    # 95% confidence intervals for the confirmatory contrasts, three resampling schemes
    iv_path = os.path.join(RES, "tae_intervals.json")
    if os.path.exists(iv_path):
        for name, v in json.load(open(iv_path))["contrasts"].items():
            key = ("ci." + name.replace(" ", "_").replace(":", "").replace("(", "")
                   .replace(")", "").replace(",", "").replace("-", "_"))
            P[key] = v

    # correction-family ceiling: is the un-repaired part budget, or misspecification?
    cf_path = os.path.join(RES, "tae_correction_family.json")
    if os.path.exists(cf_path):
        CF = json.load(open(cf_path))["rows"]
        for cond in ("E0", "E1"):
            sel = [r for r in CF if r["evaluator"] == cond]
            parts = sorted({r["partition"] for r in sel})
            ser = {}
            for f in ("trust", "case", "case_twoway", "case_lowrank", "case_athey", "oracle_row",
                      "oracle_cell", "oracle_ref", "bias_residual_not_row_constant",
                      "within_row_sd_of_selection", "row_correction_error"):
                v = np.array([np.mean([r[f] for r in sel if r["partition"] == p]) for p in parts])
                ser[f] = v
                P[f"corr.{cond}.{f}"] = dict(mean=float(v.mean()),
                                             se=float(v.std(ddof=1) / np.sqrt(len(v))))
            d = ser["case"] - ser["oracle_row"]
            P[f"corr.{cond}.case_vs_row_ceiling"] = dict(
                mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
            tot = ser["trust"] - ser["oracle_cell"]
            P[f"corr.{cond}.share_repaired"] = float(np.mean((ser["trust"] - ser["case"]) / tot))
            P[f"corr.{cond}.share_left_in_family"] = float(np.mean(d / tot))
            P[f"corr.{cond}.share_outside_family"] = float(
                np.mean((ser["oracle_row"] - ser["oracle_cell"]) / tot))

    # acquisition-overlap measurement: does the RULE or the SURFACE decide where the budget goes?
    ov_path = os.path.join(RES, "tae_acquisition_overlap.json")
    if os.path.exists(ov_path):
        OV = json.load(open(ov_path))["rows"]
        for cond in ("E0", "E1"):
            sel = [r for r in OV if r["evaluator"] == cond]
            parts = sorted({r["partition"] for r in sel})
            for arm in ("epig_corrected", "margin_observed", "epig_observed", "uniform"):
                v = np.array([np.mean([r[arm] for r in sel if r["partition"] == p]) for p in parts])
                P[f"overlap.{cond}.{arm}"] = dict(mean=float(v.mean()),
                                                  se=float(v.std(ddof=1) / np.sqrt(len(v))))

    # appendix sweeps
    for sw in A["spec"]["sweeps"]:
        FA = Agg(A["rows"], ["assignment", "evaluator", "budget", "method"], sweep=sw)
        GA = Agg(A["gate_rows"], ["evaluator", "budget", "arm"], sweep=sw)
        for key in FA.d:
            if key[2] != 0.35:
                continue
            mu, se, _ = FA.stat(key, "reference_regret")
            P[f"appx.{sw}.reg.{key[0]}.{key[1]}.{key[3]}"] = dict(mean=mu, se=se)
            mu, se, _ = FA.stat(key, "top1_error")
            P[f"appx.{sw}.top1.{key[0]}.{key[1]}.{key[3]}"] = dict(mean=mu, se=se)
        for key in GA.d:
            if key[1] != 0.35:
                continue
            for f in ["reference_regret", "false_switch", "false_retention", "switch_rate",
                      "channel_verdict_inversion"]:
                mu, se, _ = GA.stat(key, f)
                P[f"appx.{sw}.gate.{key[0]}.{key[2]}.{f}"] = dict(mean=mu, se=se)
    return P


def main():
    os.makedirs(FIGS, exist_ok=True)
    D = load("tae_validity.json")
    A = load("tae_validity_appendix.json")
    F = Agg(D["rows"], ["assignment", "evaluator", "budget", "method"])
    G = Agg(D["gate_rows"], ["evaluator", "budget", "arm"])

    figure1(F, os.path.join(FIGS, "tae_fig1.pdf"))
    figure2(F, os.path.join(FIGS, "tae_fig2.pdf"))
    figure3(F, os.path.join(FIGS, "tae_fig3.pdf"))

    tex = ["% auto-generated by code/make_tae_artifacts.py -- do not edit by hand",
           "\\newcommand{\\TAEdeploymentrows}{%", table_deployment(G, D["meta"]), "}",
           "\\newcommand{\\TAEfactorialrows}{%", table_factorial(F), "}",
           "\\newcommand{\\TAEslicerows}{%", table_slices(F), "}"]
    open(os.path.join(ROOT, "tae_tables.tex"), "w").write("\n".join(tex) + "\n")

    P = paper_numbers(D, A, F, G)
    json.dump(P, open(os.path.join(RES, "tae_paper_numbers.json"), "w"), indent=1, default=float)
    print(f"wrote figs/tae_fig1.pdf, figs/tae_fig2.pdf, tae_tables.tex, "
          f"results/tae_paper_numbers.json ({len(P)} registered numbers)")


if __name__ == "__main__":
    main()
