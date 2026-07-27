"""Four analyses a sceptical reader asks for, and the paper should answer rather than assert.

  D1  Is the 18.3% reversal rate evidence, or arithmetic? lambda is fixed at half the median
      training margin and Cor. 1 says a context can flip when the distortion exceeds half its
      target margin, so the reversal rate is largely determined by (lambda, margin distribution).
      This computes the rate Cor. 1 PREDICTS from the margin distribution alone and compares it with
      the observed one. If they agree, the number characterises the stress test, not any judge.

  D2  A usable test for assignment invalidity. Q2b says "buy an audit against evidence of assignment
      invalidity, not by default" without giving a test. After only the K per-action seed readouts,
      the dispersion of the logged-minus-randomized differences, standardised by the sampling noise
      those differences would have under a common-population log, separates A1 from A0. Reported
      with its ROC-style separation and with the regret consequence of acting on it.

  D3  When does gating pay? A held-out check trades false switches for false retentions, so
      "reduces harm" only holds under a loss asymmetry. This reports the break-even ratio
      c_switch / c_retention above which gating lowers expected loss.

  D4  Equivalence, not just failure to reject. Minimum detectable difference at 30 partitions
      (alpha .05, power .80) and a two-one-sided-tests result against a pre-specified margin, for
      the two claims stated as ties.

    python run_tae_diagnostics.py
"""
import argparse
import collections
import json
import os

import numpy as np
from scipy import stats

import case_core as cc
from eval_validity import assignment as A
from eval_validity import evaluators as EV
from eval_validity import metrics as MT
from eval_validity.grid import build_base

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DEFAULT_OUT = os.path.join(RES, "tae_diagnostics.json")
# pre-specified equivalence margin: 10% of the loss of trusting the selective log (0.0499),
# i.e. a difference smaller than this is not decision-relevant at this scale
TOST_MARGIN = 0.005


# ---------------------------------------------------------------- D1: is the reversal arithmetic?
def d1_reversal(n_parts=30, n_read=30, eps=0.15):
    obs, pred, both = [], [], []
    for p in range(n_parts):
        base = build_base(p)
        cells, n_kj, _ = A.make_selective_log(base, p, eps=eps)
        supp = A.support(n_kj, base, n_read=n_read)
        surf = EV.build_surfaces(base, cells, EV.distortions(base, "E1"), n_read)
        ref, op = surf["Ef_ref"], surf["Ef_op"]
        Ceff = op - ref
        o = p_ = 0
        n = 0
        for j in range(base["J"]):
            ks = np.where(supp[:, j])[0]
            if len(ks) < 2:
                continue
            n += 1
            best = ks[int(np.argmax(ref[ks, j]))]
            srt = np.sort(ref[ks, j])[::-1]
            margin = srt[0] - srt[1]
            # Cor. 1: the target-best model can lose its context when some rival's distortion
            # advantage exceeds the target margin -- computable from the margin and C^eff alone
            adv = np.max(Ceff[ks, j] - Ceff[best, j])
            p_ += int(adv > margin)
            o += int(int(ks[int(np.argmax(op[ks, j]))]) != int(best))
        obs.append(o / n)
        pred.append(p_ / n)
        both.append(n)
    obs, pred = np.array(obs), np.array(pred)
    return dict(observed=float(obs.mean()), observed_se=float(obs.std(ddof=1) / np.sqrt(len(obs))),
                predicted_from_margins=float(pred.mean()),
                predicted_se=float(pred.std(ddof=1) / np.sqrt(len(pred))),
                agreement=float(np.mean(np.abs(obs - pred) < 1e-9)),
                mean_abs_gap=float(np.abs(obs - pred).mean()))


# ------------------------------------------------- D2: a test for assignment invalidity, from seeds
def _seed_reads(base, cells, surf, supp, seed, n_read):
    """The K per-action seed readouts every arm buys first, and the log-minus-readout differences."""
    K, J = surf["M"].shape
    rng = np.random.default_rng(seed)
    d, se = [], []
    for k in range(K):
        js = np.where(supp[k])[0]
        if not len(js):
            continue
        j = int(js[rng.integers(len(js))])
        pool = surf["cellq"][(k, j)]
        take = rng.integers(0, len(pool), size=min(n_read, len(pool)))
        r = float(pool[take].mean())
        d.append(surf["M"][k, j] - r)
        # noise this difference would carry under a common-population log: readout + log-cell error
        n_log = max(len(cells[(k, j)]), 1)
        logvar = float(np.var(base["Q"][cells[(k, j)], k])) if n_log > 1 else 0.0
        se.append(np.sqrt(float(np.var(pool)) / len(take) + logvar / n_log))
    d, se = np.array(d), np.array(se)
    return float(np.sqrt(np.mean((d / np.maximum(se, 1e-9)) ** 2)))


def d2_diagnostic(n_parts=30, acq_seeds=6, n_read=30, eps=0.15, budget=0.35):
    stat = {"A0": [], "A1": []}
    reg = collections.defaultdict(list)
    for p in range(n_parts):
        base = build_base(p)
        c1, n_kj, _ = A.make_selective_log(base, p, eps=eps)
        c0 = A.make_matched_common_population_log(base, p, n_kj)
        supp = A.support(n_kj, base, n_read=n_read)
        tab = MT.scoring_tables(base, supp)
        pref = MT.target_policy(tab, supp)
        B = max(base["K"], int(round(budget * int(supp.sum()))))
        for aname, cells in (("A0", c0), ("A1", c1)):
            surf = EV.build_surfaces(base, cells, EV.distortions(base, "E0"), n_read)
            ss, rc, rt = [], [], []
            for s in range(acq_seeds):
                sd = 10_000 * p + 100 * int(budget * 100) + s
                ss.append(_seed_reads(base, cells, surf, supp, sd, n_read))
                V, _ = cc.allocate("case", None, surf["M"], surf["cellq"], supp, B, sd, n_read)
                rc.append(MT.reference_regret(MT.deployed_policy(V, supp), tab, supp, pref))
                rt.append(MT.reference_regret(MT.deployed_policy(surf["M"], supp), tab, supp, pref))
            stat[aname].append(float(np.mean(ss)))
            reg[(aname, "case")].append(float(np.mean(rc)))
            reg[(aname, "trust")].append(float(np.mean(rt)))
    a0, a1 = np.array(stat["A0"]), np.array(stat["A1"])
    # separation and the best single threshold on this data
    grid = np.linspace(min(a0.min(), a1.min()), max(a0.max(), a1.max()), 400)
    acc = [(np.mean(a0 < t) + np.mean(a1 >= t)) / 2 for t in grid]
    tau = float(grid[int(np.argmax(acc))])
    auc = float(np.mean([(x < y) + 0.5 * (x == y) for x in a0 for y in a1]))
    # regret if the rule is followed: audit only when the statistic exceeds tau
    followed = []
    for i in range(len(a0)):
        followed.append(reg[("A0", "case")][i] if a0[i] >= tau else reg[("A0", "trust")][i])
        followed.append(reg[("A1", "case")][i] if a1[i] >= tau else reg[("A1", "trust")][i])
    always = [reg[("A0", "case")][i] for i in range(len(a0))] + \
             [reg[("A1", "case")][i] for i in range(len(a1))]
    never = [reg[("A0", "trust")][i] for i in range(len(a0))] + \
            [reg[("A1", "trust")][i] for i in range(len(a1))]
    return dict(stat_A0=float(a0.mean()), stat_A0_se=float(a0.std(ddof=1) / np.sqrt(len(a0))),
                stat_A1=float(a1.mean()), stat_A1_se=float(a1.std(ddof=1) / np.sqrt(len(a1))),
                threshold=tau, auc=auc, accuracy=float(max(acc)),
                mean_regret_rule=float(np.mean(followed)),
                mean_regret_always_audit=float(np.mean(always)),
                mean_regret_never_audit=float(np.mean(never)),
                n_seed_cells="K per-action seeds only (5 of 13 cells at the 35% budget)")


# ---------------------------------------------------------------- D3: when does gating pay?
def d3_gate_asymmetry():
    D = json.load(open(os.path.join(RES, "tae_validity.json")))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in D["gate_rows"]:
        g[(r["evaluator"], r["budget"], r["arm"])][r["partition"]].append(r)
    out = {}
    for cond in ("E1", "E2"):
        for b in (0.15, 0.25, 0.35):
            def m(arm, f):
                x = g[(cond, b, arm)]
                return float(np.mean([np.mean([q[f] for q in x[p]]) for p in sorted(x)]))
            for arm in ("same_evaluator", "reference_channel"):
                dfs = m(arm, "false_switch") - m("ungated", "false_switch")
                dfr = m(arm, "false_retention") - m("ungated", "false_retention")
                # gating lowers expected loss iff c_switch*|dFS| > c_retention*dFR, so the
                # break-even is on c_switch / c_retention = dFR / |dFS|
                out[f"{cond}.{int(b*100)}.{arm}"] = dict(
                    delta_false_switch=dfs, delta_false_retention=dfr,
                    breakeven_cost_ratio=(float(dfr / -dfs) if dfs < 0 and dfr > 0 else None),
                    delta_reference_regret=m(arm, "reference_regret") - m("ungated", "reference_regret"))
    return out


# ---------------------------------------------------------------- D4: equivalence, not just p > .05
def d4_equivalence(margin=TOST_MARGIN):
    D = json.load(open(os.path.join(RES, "tae_validity.json")))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in D["rows"]:
        if r["budget"] == 0.35:
            g[(r["assignment"], r["evaluator"], r["method"])][r["partition"]].append(r["reference_regret"])
    def series(A_, E, m):
        x = g[(A_, E, m)]
        return np.array([np.mean(x[p]) for p in sorted(x)])
    CF = json.load(open(os.path.join(RES, "tae_correction_family.json")))["rows"]
    cf = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in CF:
        cf[(r["evaluator"], "case")][r["partition"]].append(r["case"])
        cf[(r["evaluator"], "oracle_row")][r["partition"]].append(r["oracle_row"])
    pairs = [
        ("A1-E0: CASE vs R-EPIG_corr", series("A1", "E0", "case") - series("A1", "E0", "epig_corr")),
        ("A1-E1: CASE vs R-EPIG_corr", series("A1", "E1", "case") - series("A1", "E1", "epig_corr")),
        ("A1-E0: CASE vs the action-level ceiling",
         np.array([np.mean(cf[("E0", "case")][p]) for p in sorted(cf[("E0", "case")])]) -
         np.array([np.mean(cf[("E0", "oracle_row")][p]) for p in sorted(cf[("E0", "oracle_row")])])),
    ]
    out = {}
    for name, d in pairs:
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        mdd = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * se
        t_lo = (d.mean() + margin) / se
        t_hi = (d.mean() - margin) / se
        p_tost = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
        out[name] = dict(mean=float(d.mean()), se=float(se), n=n,
                         minimum_detectable_difference=float(mdd),
                         tost_margin=margin, tost_p=float(p_tost),
                         equivalent_at_margin=bool(p_tost < 0.05))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    res = {}
    print("D1  is the reversal rate arithmetic?", flush=True)
    res["D1_reversal_is_predicted_by_margins"] = d1_reversal()
    print("   ", res["D1_reversal_is_predicted_by_margins"])
    print("D2  a test for assignment invalidity from the seed readouts alone", flush=True)
    res["D2_assignment_invalidity_test"] = d2_diagnostic()
    print("   ", res["D2_assignment_invalidity_test"])
    print("D3  break-even loss asymmetry for gating", flush=True)
    res["D3_gate_breakeven"] = d3_gate_asymmetry()
    for k, v in res["D3_gate_breakeven"].items():
        if k.endswith("same_evaluator") or k.endswith("reference_channel"):
            r = v["breakeven_cost_ratio"]
            print(f"    {k:<34} dFS {v['delta_false_switch']:+.3f}  dFR {v['delta_false_retention']:+.3f}"
                  f"  dRegret {v['delta_reference_regret']:+.4f}"
                  f"  gate pays iff c_switch/c_retention > {r:.2f}" if r else f"    {k}: n/a")
    print("D4  equivalence tests", flush=True)
    res["D4_equivalence"] = d4_equivalence()
    for k, v in res["D4_equivalence"].items():
        print(f"    {k:<44} mean {v['mean']:+.4f}  MDD {v['minimum_detectable_difference']:.4f}"
              f"  TOST p={v['tost_p']:.4f} at margin {v['tost_margin']}"
              f"  -> {'equivalent' if v['equivalent_at_margin'] else 'NOT shown equivalent'}")
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
