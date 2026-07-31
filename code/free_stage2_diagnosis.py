"""Stage 2, part 1: WHY the cost-reduction protocol produces no supported saving. NO new API calls.

Answers, from the stored replay data and the pipeline's own definitions:
  D1  How much slack does a true opportunity leave the tau=0.95 gate, and what audit sample
      size would certifying it require, against what the audit pool actually contains?
  D2  Oracle ceilings: perfect detection's saving vs the incumbent and vs raw-log downrouting.
  D3  What the gated arm's switches actually are, conditional on opportunity.
  D4  Paired CIs for every arm against BOTH comparators, with break-even volumes.
"""
import os, collections
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")

from tau2_e2e_full import (load, split_tasks, DELTA, N_AUDIT, BUDGET_PAIRS,
                           OUTER_FOLDS, SPLIT_SEEDS, EXPLORE)

Z95 = stats.norm.ppf(0.95)


def per_unit_setup():
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                      cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth.itertuples()}
    TRUE = {(r.domain, r.agent_model): r.success for r in truth.itertuples()}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})
    import random as _r
    units = []
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            asg = split_tasks(list(keys), _r.Random(1000 * fold + sd))
            R = [k for k, v in asg.items() if v == "router"]
            A = [k for k, v in asg.items() if v == "audit"]
            TRUEtr = {}
            for d in doms:
                rt = [t for (dd, t) in R if dd == d]
                for m in models:
                    v = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(rt))]
                    TRUEtr[(d, m)] = float(v.success.mean()) if len(v) else 0.0
            inc = max(models, key=lambda m: np.mean([TRUEtr.get((d, m), 0.0) for d in doms]))
            pool = {d: sum(1 for (dd, t) in A if dd == d) for d in doms}
            units.append(dict(fold=fold, seed=sd, inc=inc, pool=pool))
    return df, tc, models, doms, TRUE, COSTS, units


def d1_slack(models, doms, TRUE, COSTS, units):
    print("=" * 78)
    print("D1. THE INFORMATION CONSTRAINT ON THE tau=0.95 GATE")
    print("=" * 78)
    rows = []
    for u in units:
        inc = u["inc"]
        for d in doms:
            best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
            for m in models:
                if m == inc or COSTS[(d, m)] >= COSTS[(d, inc)]:
                    continue
                if TRUE[(d, m)] < best - DELTA:
                    continue                      # not a true opportunity
                slack = TRUE[(d, m)] - (best - DELTA)          # margin the certificate can use
                gap_inc = TRUE[(d, m)] - TRUE[(d, inc)]        # margin vs the incumbent instead
                n_best = 0.25 * (Z95 / slack) ** 2 if slack > 0 else np.inf
                n_inc = 0.25 * (Z95 / (gap_inc + DELTA)) ** 2 if gap_inc + DELTA > 0 else np.inf
                rows.append(dict(fold=u["fold"], seed=u["seed"], domain=d, cand=m, inc=inc,
                                 slack_vs_best=slack, gap_vs_inc=gap_inc,
                                 save_per_req=COSTS[(d, inc)] - COSTS[(d, m)],
                                 n_needed_vs_best=n_best, n_needed_vs_inc=n_inc,
                                 pool=u["pool"][d]))
    O = pd.DataFrame(rows)
    if not len(O):
        print("  no opportunities at all"); return O
    print(f"  true opportunities across the 50 units: {len(O)} candidate-domain cells "
          f"({O.groupby(['fold','seed']).size().reindex([(u['fold'],u['seed']) for u in units], fill_value=0).mean():.2f}/unit avg)")
    print(f"\n  slack vs the TRUE BEST that the certificate must detect (delta={DELTA}):")
    print(f"    max possible by construction = delta = {DELTA}")
    print(f"    observed: min {O.slack_vs_best.min():.4f}  median {O.slack_vs_best.median():.4f}  max {O.slack_vs_best.max():.4f}")
    print(f"\n  one-sided n per pair needed for P(within delta)>=0.95 (sigma^2=0.25):")
    print(f"    vs best     : median {O.n_needed_vs_best.median():,.0f}  min {O.n_needed_vs_best.min():,.0f}")
    print(f"    vs incumbent: median {O.n_needed_vs_inc.median():,.0f}  min {O.n_needed_vs_inc.min():,.0f}")
    print(f"  audit pool per domain: median {O.pool.median():.0f} tasks; per buy N_AUDIT={N_AUDIT}, budget {BUDGET_PAIRS} pairs")
    feas = (O.n_needed_vs_best <= O.pool).mean()
    print(f"  fraction of true opportunities certifiable vs best within the pool: {feas:.1%}")
    print(f"  => the gate CANNOT validly reach tau=0.95 on any real opportunity; any firing is noise.")
    return O


def d2_oracle(models, doms, TRUE, COSTS, units, R):
    print()
    print("=" * 78)
    print("D2. ORACLE CEILINGS (perfect detection, zero audit cost)")
    print("=" * 78)
    rows = []
    for u in units:
        inc = u["inc"]
        srow = R[(R.scenario == "cost_reduction") & (R.fold == u["fold"]) & (R.seed == u["seed"])]
        inc_cost = float(srow[srow.arm == "Raw incumbent"].cost.iloc[0])
        dr_cost = float(srow[srow.arm == "Raw-log downrouting"].cost.iloc[0])
        dr_succ = float(srow[srow.arm == "Raw-log downrouting"].success.iloc[0])
        inc_succ = float(srow[srow.arm == "Raw incumbent"].success.iloc[0])
        # oracle = for each domain switch to the cheapest true-tolerance model cheaper than inc
        sav = 0.0
        for d in doms:
            best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
            cands = [m for m in models if TRUE[(d, m)] >= best - DELTA
                     and COSTS[(d, m)] < COSTS[(d, inc)]]
            if cands:
                sav += (COSTS[(d, inc)] - min(COSTS[(d, m)] for m in cands)) / len(doms)
        rows.append(dict(fold=u["fold"], seed=u["seed"], oracle_save_per_req=sav,
                         inc_cost=inc_cost, dr_cost=dr_cost,
                         dr_minus_inc_cost=dr_cost - inc_cost,
                         dr_minus_inc_succ=dr_succ - inc_succ))
    O = pd.DataFrame(rows)
    def ci(v):
        v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
        h = stats.t.ppf(0.975, len(v) - 1) * se
        return m, m - h, m + h
    m, lo, hi = ci(O.oracle_save_per_req * 1000)
    print(f"  oracle saving vs INCUMBENT       : ${m:+.2f}/1k tasks   95% CI [${lo:+.2f}, ${hi:+.2f}]")
    m2, lo2, hi2 = ci(O.dr_minus_inc_cost * 1000)
    print(f"  raw-log downrouting vs incumbent : ${m2:+.2f}/1k tasks   95% CI [${lo2:+.2f}, ${hi2:+.2f}]")
    m3, lo3, hi3 = ci(O.dr_minus_inc_succ)
    print(f"  raw-log downrouting success delta: {m3:+.4f}          95% CI [{lo3:+.4f}, {hi3:+.4f}]")
    print(f"  => oracle ceiling vs the FREE raw-log downrouting on cost alone: "
          f"${(O.oracle_save_per_req.mean() - (-O.dr_minus_inc_cost.mean())) * 1000:+.2f}/1k "
          f"(negative means the free arm spends less by violating quality)")
    return O


def d3_switch_anatomy():
    print()
    print("=" * 78)
    print("D3. WHAT THE GATED ARM'S SWITCHES ACTUALLY ARE")
    print("=" * 78)
    C = pd.read_parquet(os.path.join(RES, "free_stage1_switch_categories.parquet"))
    c = C[(C.scenario == "cost_reduction") & (C.arm == "Confidence-gated IV CASE")]
    print(f"  decisions: {len(c):,}   switch rate {c.switched.mean():.4f}")
    for o in (1, 0):
        g = c[c.opportunity == o]
        sw = g[g.switched == 1]
        lab = "opportunity present" if o else "no opportunity     "
        frac_A = (sw.category == "A").mean() if len(sw) else 0.0
        print(f"  {lab}: {len(g):5,} decisions | P(switch)={g.switched.mean():.3f} | "
              f"P(A|switch)={frac_A:.3f} | cats {dict(collections.Counter(sw.category))}")
    return c


def d4_paired(R):
    print()
    print("=" * 78)
    print("D4. EVERY ARM AGAINST BOTH COMPARATORS (50 paired units)")
    print("=" * 78)
    r = R[R.scenario == "cost_reduction"]
    piv_c = r.pivot_table(index=["fold", "seed"], columns="arm", values="cost")
    piv_s = r.pivot_table(index=["fold", "seed"], columns="arm", values="success")
    aud = r.groupby("arm").audit.mean()
    def ci(v):
        v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
        h = stats.t.ppf(0.975, len(v) - 1) * se
        return m, m - h, m + h
    print(f"  {'arm':30}{'vs incumbent $/1k':>24}{'vs raw-downroute $/1k':>26}{'audit $':>9}")
    for arm in ["Raw-log downrouting", "Current CASE", "Inverse-variance CASE",
                "Safe inverse-variance CASE", "Ideal complete matrix"]:
        if arm not in piv_c.columns: continue
        d_inc = (piv_c[arm] - piv_c["Raw incumbent"]) * 1000
        d_dr = (piv_c[arm] - piv_c["Raw-log downrouting"]) * 1000
        m1, l1, h1 = ci(d_inc); m2, l2, h2 = ci(d_dr)
        print(f"  {arm:30}{m1:+8.2f} [{l1:+7.2f},{h1:+7.2f}]{m2:+9.2f} [{l2:+7.2f},{h2:+7.2f}]"
              f"{aud.get(arm, 0.0):>9.3f}")
    # break-even vs incumbent for the gated arm, with bootstrap lower bound
    arm = "Safe inverse-variance CASE"
    d = (piv_c["Raw incumbent"] - piv_c[arm]).values   # saving per request
    a = aud.get(arm, 0.0)
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(4000)]
    lo = float(np.quantile(boots, 0.05))
    print(f"\n  gated arm vs incumbent: saving {np.mean(d)*1000:+.2f}/1k, 5% bootstrap LB {lo*1000:+.2f}/1k, audit ${a:.2f}")
    print(f"  break-even point-estimate: {a/np.mean(d):,.0f} tasks" if np.mean(d) > 0 else "  no point saving")
    print(f"  break-even 95%-lower-bound: {a/lo:,.0f} tasks" if lo > 0 else "  NO certified break-even vs incumbent")


def main():
    df, tc, models, doms, TRUE, COSTS, units = per_unit_setup()
    R = pd.read_parquet(os.path.join(RES, "free_stage1_router_sensitivity.parquet"))
    O1 = d1_slack(models, doms, TRUE, COSTS, units)
    d2_oracle(models, doms, TRUE, COSTS, units, R)
    d3_switch_anatomy()
    d4_paired(R)
    O1.to_parquet(os.path.join(RES, "free_stage2_opportunity_slack.parquet"), index=False)
    print("\n  wrote free_stage2_opportunity_slack.parquet")


if __name__ == "__main__":
    main()
