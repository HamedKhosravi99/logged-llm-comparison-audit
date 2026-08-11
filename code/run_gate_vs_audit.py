#!/usr/bin/env python3
"""P0.3: is Table 2's headline the audit, or the gate?

R2's largest objection: gated pooled CASE differs from raw-log downrouting in TWO ways - purchased
randomized evidence AND a confidence gate - so the -0.449 reduction in quality-violating switches
cannot be attributed to the audit. This adds the missing arm, raw-log downrouting with the SAME
gate at tau=0.95 and ZERO audit spend, and reports the four-arm ladder so the two effects are
separately identified:

  raw ungated -> raw gated      : the gate at zero audit spend
  raw gated   -> CASE gated     : the audit, given the gate
  raw ungated -> CASE ungated   : the audit at no gate
  CASE ungated-> CASE gated     : the gate, given the audit

Per docs/PHASE0_ANALYSIS_PLAN.md and plan v3 P0.3 (which drops v2's claim of a single
order-independent decomposition - with three arms it is not identified; the fourth arm identifies it).
"""
import collections, json, os, random as _r, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
from tau2_e2e_full import (load, split_tasks, fit_router, cheapest_good_enough,   # noqa: E402
                           DELTA, OUTER_FOLDS, SPLIT_SEEDS)
from free_stage1_rq5_aligned import (build_log, run_case, greedy_policy, safe_policy,  # noqa: E402
                                     H_PRIMARY, TAU_PRIMARY, score)

ACQ = 3


def harmful(pol, base, TRUE, doms):
    return sum(1 for d in doms if pol.get(d) != base.get(d) and pol.get(d) and base.get(d)
               and TRUE[(d, pol[d])] < TRUE[(d, base[d])] - DELTA)


def switches(pol, base, doms):
    return sum(1 for d in doms if pol.get(d) != base.get(d))


def main():
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                      cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth.itertuples()}
    TRUE = {(r.domain, r.agent_model): r.success for r in truth.itertuples()}
    ideal = {d: cheapest_good_enough(truth[truth.domain == d], DELTA) for d in doms}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})

    rows = []
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            rng = np.random.default_rng(1000 * fold + sd)
            asg = split_tasks(list(keys), _r.Random(1000 * fold + sd))
            S = [k for k, v in asg.items() if v == "scoring"]
            R = set(k for k, v in asg.items() if v == "router")
            L = [k for k, v in asg.items() if v == "log"]
            A = [k for k, v in asg.items() if v == "audit"]
            if not (S and R and L and A):
                continue
            router = fit_router(df, R, models, doms)
            M, Mvar = build_log(df, tc, L, router, models, doms, COSTS, rng)
            q_raw = {(d, m): M.get((d, m), np.nan) for d in doms for m in models}
            qv_raw = {(d, m): Mvar.get((d, m), np.nan) for d in doms for m in models}
            inc = greedy_policy(q_raw, models, doms, COSTS)      # log-derived status quo

            # ---- ARM 1: raw ungated (free raw-log downrouting) ---------------------------------
            s1, c1, _ = score(inc, tc, S)

            # ---- ARM 2: raw GATED, zero audit spend (the missing arm) --------------------------
            acc = collections.defaultdict(list)
            for a in range(ACQ):
                r2 = np.random.default_rng(7000 + a)
                gp, _acts = safe_policy(q_raw, qv_raw, models, doms, COSTS, inc,
                                        0.0, TAU_PRIMARY, H_PRIMARY, r2)
                s2, c2, _ = score(gp, tc, S)
                acc["raw_gated"].append(dict(success=s2, cost=c2, audit=0.0,
                                             harmful=harmful(gp, inc, TRUE, doms),
                                             switches=switches(gp, inc, doms)))

                # ---- ARM 3/4: CASE ungated and CASE gated ------------------------------------
                q, qv, spend, _n = run_case(M, Mvar, tc, A, models, doms, COSTS, r2, True)
                cu = greedy_policy(q, models, doms, COSTS)
                s3, c3, _ = score(cu, tc, S)
                acc["case_ungated"].append(dict(success=s3, cost=c3, audit=spend,
                                                harmful=harmful(cu, inc, TRUE, doms),
                                                switches=switches(cu, inc, doms)))
                cg, _ = safe_policy(q, qv, models, doms, COSTS, inc, spend,
                                    TAU_PRIMARY, H_PRIMARY, r2)
                s4, c4, _ = score(cg, tc, S)
                acc["case_gated"].append(dict(success=s4, cost=c4, audit=spend,
                                              harmful=harmful(cg, inc, TRUE, doms),
                                              switches=switches(cg, inc, doms)))
            rec = dict(fold=fold, seed=sd, raw_ungated_success=s1, raw_ungated_cost=c1,
                       raw_ungated_harmful=0.0, raw_ungated_switches=0.0)
            for arm, rs in acc.items():
                for k in ("success", "cost", "audit", "harmful", "switches"):
                    rec[f"{arm}_{k}"] = float(np.mean([r[k] for r in rs]))
            rows.append(rec)

    U = pd.DataFrame(rows)

    def paired(a, b):
        d = U[a] - U[b]
        lo, hi = np.percentile([np.mean(d.sample(len(d), replace=True))
                                for _ in range(2000)], [2.5, 97.5])
        return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=int(len(d)))

    out = dict(
        n_units=int(len(U)),
        levels={c: float(U[c].mean()) for c in U.columns if c not in ("fold", "seed")},
        gate_at_zero_audit=dict(
            harmful=paired("raw_gated_harmful", "raw_ungated_harmful"),
            switches=paired("raw_gated_switches", "raw_ungated_switches")),
        audit_given_gate=dict(
            harmful=paired("case_gated_harmful", "raw_gated_harmful"),
            switches=paired("case_gated_switches", "raw_gated_switches")),
        audit_at_no_gate=dict(
            harmful=paired("case_ungated_harmful", "raw_ungated_harmful"),
            switches=paired("case_ungated_switches", "raw_ungated_switches")),
        gate_given_audit=dict(
            harmful=paired("case_gated_harmful", "case_ungated_harmful"),
            switches=paired("case_gated_switches", "case_ungated_switches")),
        total_case_gated_vs_raw=dict(
            harmful=paired("case_gated_harmful", "raw_ungated_harmful"),
            switches=paired("case_gated_switches", "raw_ungated_switches")),
    )
    p = os.path.join(RES, "gate_vs_audit.json")
    json.dump(out, open(p, "w"), indent=1)

    print(f"units {len(U)}")
    for k in ("gate_at_zero_audit", "audit_given_gate", "audit_at_no_gate",
              "gate_given_audit", "total_case_gated_vs_raw"):
        h = out[k]["harmful"]; s = out[k]["switches"]
        print(f"  {k:24s} harmful {h['mean']:+.3f} [{h['lo']:+.3f},{h['hi']:+.3f}]   "
              f"switches {s['mean']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}]")
    print("wrote", p)


if __name__ == "__main__":
    main()
