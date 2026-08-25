"""Section 5's quantities, read off the Eq. (1) runs of `run_section5_mckp.py`.

Acquisition seeds are averaged within a partition first, then contrasts are paired across
partitions, which is the protocol the committed Eq. (2) numbers use.

Usage:
    python code/summarize_section5_mckp.py [results/section5_mckp_llmrouterbench_eps0.15_n30.json]
"""
import collections, json, os, sys

import numpy as np

ARMS = ("trust", "uniform", "case", "oracle_sel", "oracle_ref")


def load(path):
    """(mult, cond, assign, frac) -> arm -> partition -> mean over acquisition seeds."""
    rows = json.load(open(path))
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    for r in rows:
        key = (r["mult"], r["cond"], r["assign"], r["budget_frac"])
        acc[key][r["arm"]][r["partition"]].append(r)
    return rows, acc


def per_partition(cell, arm, field):
    d = cell.get(arm, {})
    return {p: float(np.mean([x[field] for x in v])) for p, v in d.items()}


def paired(cell, a, b, field="loss"):
    xa, xb = per_partition(cell, a, field), per_partition(cell, b, field)
    ps = sorted(set(xa) & set(xb))
    if len(ps) < 2:
        return None
    d = np.array([xa[p] - xb[p] for p in ps])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se, len(d)


def fmt(t):
    return "--" if t is None else "%+.4f [%+.4f,%+.4f]" % t[:3]


def main(path):
    rows, acc = load(path)
    corpus = rows[0]["corpus"]
    J = rows[0]["J"]
    fracs = sorted({r["budget_frac"] for r in rows})
    mults = sorted({r["mult"] for r in rows})
    parts = len({r["partition"] for r in rows})
    seeds = len({r["acq_seed"] for r in rows})
    print("%s: %d partitions x %d acquisition seeds, J=%d, budgets %s"
          % (corpus, parts, seeds, J, ", ".join("%g%%" % (100 * f) for f in fracs)))

    for frac in fracs:
        print("\n" + "=" * 108)
        print("target-value loss under Eq. (1), evidence budget %g%% of supported pairs"
              % (100 * frac))
        print("=" * 108)
        print("%-6s %-7s %-9s %-9s %-9s %-9s %-9s  %-26s %s"
              % ("Gamma", "cell", "trust", "uniform", "CASE", "cell-perf", "target",
                 "CASE - trust [95% CI]", "CASE - uniform [95% CI]"))
        for mult in mults:
            for cond in ("D0", "D1"):
                for assign in ("A1", "A0"):
                    cell = acc.get((mult, cond, assign, frac))
                    if not cell:
                        continue
                    means = []
                    for arm in ARMS:
                        v = list(per_partition(cell, arm, "loss").values())
                        means.append("%.4f" % np.mean(v) if v else "--")
                    print("%-6g %-7s %-9s %-9s %-9s %-9s %-9s  %-26s %s"
                          % (mult, cond + "-" + assign, *means,
                             fmt(paired(cell, "case", "trust")),
                             fmt(paired(cell, "case", "uniform"))))

    print("\n" + "=" * 108)
    print("wrong-workload rate (share of J), and readouts bought, at the largest evidence budget")
    print("=" * 108)
    frac = fracs[-1]
    print("%-6s %-7s %-9s %-9s %-9s %-9s  %s"
          % ("Gamma", "cell", "trust", "uniform", "CASE", "cell-perf", "CASE cells read"))
    for mult in mults:
        for cond in ("D0", "D1"):
            for assign in ("A1", "A0"):
                cell = acc.get((mult, cond, assign, frac))
                if not cell:
                    continue
                out = []
                for arm in ("trust", "uniform", "case", "oracle_sel"):
                    v = list(per_partition(cell, arm, "wrong").values())
                    out.append("%.1f%%" % (100 * np.mean(v) / J) if v else "--")
                reads = list(per_partition(cell, "case", "reads").values())
                supp = np.mean([r["n_supported"] for r in rows])
                print("%-6g %-7s %-9s %-9s %-9s %-9s  %.1f of %.0f"
                      % (mult, cond + "-" + assign, *out, np.mean(reads), supp))


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "section5_mckp_llmrouterbench_eps0.15_n30.json")
    main(sys.argv[1] if len(sys.argv) > 1 else default)
