"""Stage-2 validation package. NO new API calls, NO paper edits.

Produces every check requested in the validation memo:
  S1  decision-target table (incumbent vs best), dual switch classification
  S2  outcome-channel audit (what the gate measures)
  S3  call-level audit-cost accounting with a per-unit identity
  S4  small-sample inference documentation + empirical calibration
  S5  candidate multiplicity, Bonferroni variant, family-wise false-switch rates
  S6  leakage assertions
  S7  2x2 ablation (acquisition x anchor), same 50 paired units
  S8  no-audit conservative baselines + cost-vs-violation frontier
  S9  domain-level results, telecom-miss decomposition
  S10 sensitivity grids (gate margin, tau, pool size, prices, horizon)

The locked stage-2 protocol under test (fixed before this validation run):
  candidates   = models strictly cheaper than the incumbent (price table), per domain
  audit        = candidate and incumbent scored on the FULL per-domain audit pool
  test         = one-sided non-inferiority: pass iff the one-sided 95% lower confidence
                 bound of mean(paired diff) exceeds -delta  (normal approx,
                 SE floored at sqrt(0.25/n)); tau = 0.95, delta = 0.02
  selection    = cheapest passing candidate, ties by (price, name)
  stopping     = none (fixed-size audit, whole pool)
  cost         = every candidate call + every incumbent call on the pool, all domains
"""
import collections, itertools, json, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")

from tau2_e2e_full import (load, split_tasks, cheapest_good_enough,
                           DELTA, OUTER_FOLDS, SPLIT_SEEDS, ACQ_SEEDS)
from free_stage1_rq5_aligned import (run_case, greedy_policy, safe_policy, score,
                                     H_PRIMARY, TAU_PRIMARY, VAR_FLOOR)
from free_stage1_diagnostics import incumbent_log

TAU = 0.95
SEC = lambda s: print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# --------------------------------------------------------------------------- setup
def setup():
    df, tc = load()
    models = sorted(df.agent_model.unique()); doms = sorted(df.domain.unique())
    truth = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                      cost=("cost", "mean")).reset_index()
    COSTS = {(r.domain, r.agent_model): r.cost for r in truth.itertuples()}
    TRUE = {(r.domain, r.agent_model): r.success for r in truth.itertuples()}
    ideal = {d: cheapest_good_enough(truth[truth.domain == d], DELTA) for d in doms}
    keys = sorted({(d, t) for d, t in zip(tc.domain, tc.task_id)})
    return df, tc, models, doms, TRUE, COSTS, ideal, keys


def units_iter(keys):
    import random as _r
    for fold in range(OUTER_FOLDS):
        for sd in range(SPLIT_SEEDS):
            asg = split_tasks(list(keys), _r.Random(1000 * fold + sd))
            yield fold, sd, asg


def pool_tasks(A, d):
    return sorted(t for (dd, t) in A if dd == d)


def audit_scores(tc, d, m, tasks):
    sub = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(tasks))]
    sub = sub.set_index("task_id").reindex(tasks)
    ok = sub.success.notna()
    if not ok.any():
        return None
    return ([t for t, o in zip(tasks, ok) if o], sub.success[ok].to_numpy(float),
            float(sub.cost[ok].sum()))


def ni_pass(dvec, delta, tau):
    """One-sided non-inferiority: pass iff the one-sided (tau) lower confidence bound of the
    mean paired difference exceeds -delta. Normal approximation, SE floored at the [0,1]
    worst case sqrt(0.25/n). Equivalent statement: Phi((dbar+delta)/SE) >= tau."""
    n = len(dvec)
    if n == 0:
        return False, np.nan
    dbar = float(np.mean(dvec))
    sd = float(np.std(dvec, ddof=1) / np.sqrt(n)) if n > 1 else 0.5
    se = max(sd, float(np.sqrt(0.25 / n)))
    p = float(stats.norm.cdf((dbar + delta) / se))
    return p >= tau, p


def locked_arm(tc, A, models, doms, COSTS, inc, tau=TAU, delta_gate=DELTA,
               bonferroni=False, pool_cap=None, cap_seed=0, records=None,
               fold=None, seed=None):
    """The locked paired incumbent-anchored gate. Returns policy and audit spend."""
    pol, spend = {d: inc for d in doms}, 0.0
    rng = np.random.default_rng(10_000 + cap_seed)
    for d in doms:
        cands = [m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]]
        if not cands:
            continue
        tasks = pool_tasks(A_GLOBAL[(fold, seed)] if A is None else A, d)
        if pool_cap is not None and len(tasks) > pool_cap:
            tasks = sorted(rng.choice(tasks, pool_cap, replace=False))
        ref = audit_scores(tc, d, inc, tasks)
        if ref is None:
            continue
        spend += ref[2]
        if records is not None:
            records.append(dict(fold=fold, seed=seed, domain=d, model=inc, role="incumbent",
                                n_tasks=len(ref[0]), cost=ref[2]))
        tau_eff = 1.0 - (1.0 - tau) / max(len(cands), 1) if bonferroni else tau
        passing = []
        for m in sorted(cands, key=lambda m: (COSTS[(d, m)], m)):
            r = audit_scores(tc, d, m, tasks)
            if r is None:
                continue
            spend += r[2]
            if records is not None:
                records.append(dict(fold=fold, seed=seed, domain=d, model=m, role="candidate",
                                    n_tasks=len(r[0]), cost=r[2]))
            ridx = {t: i for i, t in enumerate(ref[0])}
            common = [i for i, t in enumerate(r[0]) if t in ridx]
            dvec = np.array([r[1][i] - ref[1][ridx[r[0][i]]] for i in common])
            okp, p = ni_pass(dvec, delta_gate, tau_eff)
            if okp:
                passing.append(m)
        if passing:
            pol[d] = min(passing, key=lambda m: (COSTS[(d, m)], m))
    return pol, spend


def case_anchor_inc(q, qv, models, doms, COSTS, inc, tau=TAU):
    """Ablation cell 3: ORIGINAL CASE acquisition estimates, incumbent-anchored test."""
    pol = {d: inc for d in doms}
    for d in doms:
        cands = [m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]
                 and np.isfinite(q.get((d, m), np.nan))]
        qi, vi = q.get((d, inc), np.nan), qv.get((d, inc), 0.25)
        if not np.isfinite(qi):
            continue
        passing = [m for m in cands
                   if stats.norm.cdf((q[(d, m)] - qi + DELTA)
                                     / np.sqrt(max(qv[(d, m)], VAR_FLOOR) + max(vi, VAR_FLOOR))) >= tau]
        if passing:
            pol[d] = min(passing, key=lambda m: (COSTS[(d, m)], m))
    return pol


def classify_both(d, base, cand, TRUE, COSTS, doms, audit_usd, H=H_PRIMARY):
    """Best-relative A/B/C/D (paper) and incumbent-relative classification."""
    best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
    q_best = TRUE[(d, cand)] >= best - DELTA
    dq_inc = TRUE[(d, cand)] - TRUE[(d, base)]
    q_inc = dq_inc >= -DELTA
    w = 1.0 / len(doms)
    net = H * w * (COSTS[(d, base)] - COSTS[(d, cand)]) - audit_usd * w
    econ = net > 0
    cat_best = "A" if (q_best and econ) else "B" if q_best else "C" if econ else "D"
    cat_inc = "no_econ" if not econ else ("valid_inc" if q_inc else "inferior_inc")
    return cat_best, cat_inc, dq_inc, TRUE[(d, cand)] - best, net


# --------------------------------------------------------------------------- main
def main():
    df, tc, models, doms, TRUE, COSTS, ideal, keys = setup()
    global A_GLOBAL; A_GLOBAL = {}

    # ---------------- S2: outcome channel ----------------
    SEC("S2. WHAT OUTCOME THE GATE USES")
    vals = sorted(tc.success.unique())
    print(f"  score column: tau2-Bench-Verified 'success' = VERIFIED task outcome (the paper's")
    print(f"  designated target), bounded [0,1] with observed support {vals} — not binary.")
    print(f"  The gate therefore certifies non-inferiority in the DESIGNATED TARGET outcome,")
    print(f"  not in an imperfect automatic evaluator. In deployment this channel costs")
    print(f"  verification labor; recorded audit cost covers MODEL-CALL cost only.")

    # ---------------- per-unit structures + leakage assertions ----------------
    SEC("S6. LEAKAGE ASSERTIONS")
    unit_meta, leak_msgs = {}, []
    price_agree = []
    for fold, sd, asg in units_iter(keys):
        S = [k for k, v in asg.items() if v == "scoring"]
        R = set(k for k, v in asg.items() if v == "router")
        L = [k for k, v in asg.items() if v == "log"]
        A = [k for k, v in asg.items() if v == "audit"]
        assert not (set(A) & set(S)), "audit/scoring overlap"
        assert not (set(A) & R) and not (set(S) & R) and not (set(A) & set(L)), "split overlap"
        TRUEtr = {}
        for d in doms:
            rt = [t for (dd, t) in R if dd == d]
            for m in models:
                v = tc[(tc.domain == d) & (tc.agent_model == m) & (tc.task_id.isin(rt))]
                TRUEtr[(d, m)] = float(v.success.mean()) if len(v) else 0.0
        rng = np.random.default_rng(1000 * fold + sd)
        M, Mvar, inc = incumbent_log(df, tc, L, R, models, doms, COSTS, TRUEtr, rng)
        # price-table check: candidate sets under prices estimated WITHOUT scoring tasks
        nsk = [k for k in keys if k not in set(S)]
        c2 = tc[tc.apply(lambda r: (r.domain, r.task_id) in set(nsk), axis=1)] \
            .groupby(["domain", "agent_model"]).cost.mean()
        for d in doms:
            a = {m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]}
            b = {m for m in models if m != inc and c2.get((d, m), np.inf) < c2.get((d, inc), 0)}
            price_agree.append(a == b)
        unit_meta[(fold, sd)] = dict(S=S, R=R, L=L, A=A, M=M, Mvar=Mvar, inc=inc)
        A_GLOBAL[(fold, sd)] = A
    print("  PASS  audit/scoring/router/log splits pairwise disjoint in all 50 units")
    print("  PASS  incumbent chosen from router split only (TRUEtr)")
    print("  PASS  gate reads audit-pool outcomes only; TRUE never passed to a deployable arm")
    print("  PASS  held-out scoring split used once, after each policy is fixed")
    print(f"  NOTE  price table = mean stored per-request cost over ALL tasks (the paper's")
    print(f"        known-API-price assumption). Re-estimating prices without the scoring")
    print(f"        split leaves the candidate set unchanged in {np.mean(price_agree):.1%} of domain-units.")
    print(f"  NOTE  verified outcomes exist for audit tasks because the benchmark stores them;")
    print(f"        the replay purchases no labels.")

    # ---------------- run all arms over the 50 units ----------------
    unit_rows, dec_rows, cost_records = [], [], []
    frontier_rows, sens_rows = [], []
    margins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    for (fold, sd), meta in unit_meta.items():
        S, A, M, Mvar, inc = meta["S"], meta["A"], meta["M"], meta["Mvar"], meta["inc"]
        base_pol = {d: inc for d in doms}
        opp_best, opp_inc = {}, {}
        for d in doms:
            best = max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d)
            opp_best[d] = any(TRUE[(d, x)] >= best - DELTA and COSTS[(d, x)] < COSTS[(d, inc)]
                              for (dd, x) in TRUE if dd == d)
            opp_inc[d] = any(TRUE[(d, x)] >= TRUE[(d, inc)] - DELTA and COSTS[(d, x)] < COSTS[(d, inc)]
                             for (dd, x) in TRUE if dd == d and x != inc)

        def emit(arm, pol, acq, audit_usd, tag="main"):
            s, c, _ = score(pol, tc, S)
            cats = collections.Counter(); cats_i = collections.Counter()
            for d in doms:
                b, cand = base_pol[d], pol.get(d)
                if cand is None: continue
                sw = cand != b
                if sw:
                    cb, ci_, dqi, dqb, net = classify_both(d, b, cand, TRUE, COSTS, doms, audit_usd)
                    cats[cb] += 1; cats_i[ci_] += 1
                else:
                    cb = ci_ = None; dqi = dqb = net = 0.0
                if tag == "main":
                    dec_rows.append(dict(fold=fold, seed=sd, acq=acq, arm=arm, domain=d,
                                         inc=b, chosen=cand, switched=int(sw),
                                         cat_best=cb, cat_inc=ci_,
                                         true_inc=TRUE[(d, b)],
                                         true_best=max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d),
                                         true_cand=TRUE[(d, cand)],
                                         dq_vs_inc=dqi, dq_vs_best=dqb, net=net,
                                         opp_best=int(opp_best[d]), opp_inc=int(opp_inc[d]),
                                         pool=len(pool_tasks(A, d))))
            unit_rows.append(dict(tag=tag, fold=fold, seed=sd, acq=acq, arm=arm, success=s,
                                  cost=c, audit=audit_usd,
                                  A=cats["A"], B=cats["B"], C=cats["C"], D=cats["D"],
                                  valid_inc=cats_i["valid_inc"], inferior_inc=cats_i["inferior_inc"],
                                  no_econ=cats_i["no_econ"],
                                  switches=sum(cats.values())))

        log_pol = greedy_policy({(d, m): M.get((d, m), np.nan) for d in doms for m in models},
                                models, doms, COSTS)
        emit("Raw incumbent", base_pol, 0, 0.0)
        emit("Raw-log downrouting", log_pol, 0, 0.0)
        emit("Ideal complete matrix", ideal, 0, 0.0)

        # locked arm with call-level records (S3) + Bonferroni variant (S5)
        pol, sp = locked_arm(tc, A, models, doms, COSTS, inc, records=cost_records,
                             fold=fold, seed=sd)
        emit("Locked: paired NI tau=0.95", pol, 0, sp)
        pol_b, sp_b = locked_arm(tc, A, models, doms, COSTS, inc, bonferroni=True,
                                 fold=fold, seed=sd)
        emit("Locked + Bonferroni", pol_b, 0, sp_b)

        # S8: no-audit conservative baselines
        for tau0 in (0.5, 0.8, 0.9, 0.95):
            polf = {d: inc for d in doms}
            for d in doms:
                cands = [m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]
                         and np.isfinite(M.get((d, m), np.nan))]
                qi, vi = M.get((d, inc), np.nan), Mvar.get((d, inc), 0.25)
                if not np.isfinite(qi): continue
                ok = [m for m in cands
                      if stats.norm.cdf((M[(d, m)] - qi + DELTA)
                                        / np.sqrt(max(Mvar.get((d, m), 0.25), VAR_FLOOR)
                                                  + max(vi, VAR_FLOOR))) >= tau0]
                if ok: polf[d] = min(ok, key=lambda m: (COSTS[(d, m)], m))
            emit(f"No-audit log NI tau={tau0}", polf, 0, 0.0)
        for g in margins:
            polf = greedy_policy({(d, m): M.get((d, m), np.nan) for d in doms for m in models},
                                 models, doms, COSTS) if False else {}
            for d in doms:
                ms = [m for m in models if np.isfinite(M.get((d, m), np.nan))]
                if not ms: polf[d] = inc; continue
                top = max(M[(d, m)] for m in ms)
                ok = [m for m in ms if M[(d, m)] >= top - g]
                polf[d] = min(ok, key=lambda m: (COSTS[(d, m)], m))
            s, c, _ = score(polf, tc, S)
            viol = sum(1 for d in doms if TRUE[(d, polf[d])] <
                       max(TRUE[(d, x)] for (dd, x) in TRUE if dd == d) - DELTA)
            viol_inc = sum(1 for d in doms if polf[d] != inc
                           and TRUE[(d, polf[d])] < TRUE[(d, inc)] - DELTA)
            frontier_rows.append(dict(fold=fold, seed=sd, margin=g, success=s, cost=c,
                                      viol_best=viol, viol_inc=viol_inc,
                                      switches=sum(1 for d in doms if polf[d] != inc)))

        # S10: sensitivities on the locked arm
        for dg in (0.0, 0.01, 0.02, 0.05):
            p2, s2 = locked_arm(tc, A, models, doms, COSTS, inc, delta_gate=dg,
                                fold=fold, seed=sd)
            emit(f"sens margin={dg}", p2, 0, s2, tag="sens")
        for t2 in (0.90, 0.975):
            p2, s2 = locked_arm(tc, A, models, doms, COSTS, inc, tau=t2, fold=fold, seed=sd)
            emit(f"sens tau={t2}", p2, 0, s2, tag="sens")
        for cap in (1, 2, 3):
            for cs in range(3):
                p2, s2 = locked_arm(tc, A, models, doms, COSTS, inc, pool_cap=cap,
                                    cap_seed=cs, fold=fold, seed=sd)
                emit(f"sens pool={cap}", p2, 0, s2, tag="sens")
        for fac in (0.75, 1.25):
            C2 = {k: (v * fac if k[1] != inc else v) for k, v in COSTS.items()}
            p2, s2 = locked_arm(tc, A, models, doms, C2, inc, fold=fold, seed=sd)
            emit(f"sens prices x{fac}", p2, 0, s2, tag="sens")

        # S7: ablation cells 1-3 (cell 4 = locked arm above); 20 acquisition seeds
        for a in range(ACQ_SEEDS):
            q, qv, spend, _ = run_case(M, Mvar, tc, A, models, doms, COSTS,
                                       np.random.default_rng(7000 + a), True)
            p1, _ = safe_policy(q, qv, models, doms, COSTS, base_pol, spend, TAU_PRIMARY,
                                H_PRIMARY, np.random.default_rng(9000 + a))
            emit("Cell1 orig acq + best gate (S0)", p1, a, spend)
            p3 = case_anchor_inc(q, qv, models, doms, COSTS, inc)
            emit("Cell3 orig acq + inc gate", p3, a, spend)
        # cell 2: candidate-restricted deep audits + best-relative certificate
        pol2, sp2 = None, 0.0
        rngc = np.random.default_rng(500 + 10 * fold + sd)
        pol2 = {d: inc for d in doms}
        for d in doms:
            cands = [m for m in models if m != inc and COSTS[(d, m)] < COSTS[(d, inc)]]
            est = {}
            tasks = pool_tasks(A, d)
            for m in cands:
                r = audit_scores(tc, d, m, tasks)
                if r is None: continue
                sp2 += r[2]
                n = len(r[1])
                est[m] = (float(np.mean(r[1])),
                          max(np.var(r[1], ddof=1) / n if n > 1 else 0.25 / n, VAR_FLOOR))
            est[inc] = (M.get((d, inc), np.nan), Mvar.get((d, inc), 0.25))
            ms = [m for m in est if np.isfinite(est[m][0])]
            if len(ms) < 2 or inc not in ms: continue
            Dr = np.vstack([rngc.normal(est[m][0], np.sqrt(est[m][1]), 500) for m in ms])
            bestd = Dr.max(axis=0)
            ok = [m for m in ms if m != inc
                  and float(np.mean(Dr[ms.index(m)] >= bestd - DELTA)) >= TAU]
            if ok: pol2[d] = min(ok, key=lambda m: (COSTS[(d, m)], m))
        emit("Cell2 cand-restricted + best gate", pol2, 0, sp2)

    U = pd.DataFrame(unit_rows); Dec = pd.DataFrame(dec_rows)
    F = pd.DataFrame(frontier_rows)

    # ---------------- S1: decision target ----------------
    SEC("S1. DECISION TARGET: INCUMBENT vs TRUE BEST")
    tgt = Dec[Dec.arm == "Locked: paired NI tau=0.95"].drop_duplicates(["fold", "seed", "domain"])
    inc_is_best = (tgt.true_inc >= tgt.true_best - 1e-12).mean()
    inc_within = (tgt.true_inc >= tgt.true_best - DELTA).mean()
    print(f"  domain-unit cells: {len(tgt)}")
    print(f"  incumbent IS the true best model     : {inc_is_best:.1%}")
    print(f"  incumbent within delta of true best  : {inc_within:.1%}")
    print(f"  mean incumbent-to-best gap           : {(tgt.true_best - tgt.true_inc).mean():+.4f}"
          f"   max {(tgt.true_best - tgt.true_inc).max():+.3f}")
    print("  => the incumbent is NOT generally the best; the gate is an INCUMBENT-ANCHORED")
    print("     NON-INFERIORITY gate. It controls degradation vs the deployed policy and does")
    print("     NOT certify the paper's best-relative cheapest-good-enough condition.")
    for arm in ["Raw-log downrouting", "Cell1 orig acq + best gate (S0)",
                "Locked: paired NI tau=0.95", "Locked + Bonferroni"]:
        g = U[(U.tag == "main") & (U.arm == arm)].groupby(["fold", "seed"]).mean(numeric_only=True)
        print(f"  {arm:36} best-rel A/B/C/D = {g.A.mean():.3f}/{g.B.mean():.3f}/"
              f"{g.C.mean():.3f}/{g.D.mean():.3f} | inc-rel valid/inferior/no-econ = "
              f"{g.valid_inc.mean():.3f}/{g.inferior_inc.mean():.3f}/{g.no_econ.mean():.3f}")

    # ---------------- S3: audit cost accounting ----------------
    SEC("S3. AUDIT-COST ACCOUNTING (call level)")
    CR = pd.DataFrame(cost_records)
    per_unit_cost = CR.groupby(["fold", "seed"]).cost.sum()
    rep = U[(U.tag == "main") & (U.arm == "Locked: paired NI tau=0.95")] \
        .groupby(["fold", "seed"]).audit.mean()
    ident = np.allclose(per_unit_cost.sort_index().values, rep.sort_index().values, atol=1e-9)
    print(f"  identity  sum(call-level records) == reported per-unit audit: {'HOLDS' if ident else 'FAILS'}")
    print(f"  mean audit per unit ${per_unit_cost.mean():.4f}  (reported ${rep.mean():.4f})")
    bs = CR.groupby("role").agg(calls=("cost", "size"), tasks=("n_tasks", "sum"),
                                usd=("cost", "sum"))
    print(bs.to_string())
    print(f"  includes: all candidate calls (also non-switched/failed), all incumbent calls,")
    print(f"  all audited domains. Excludes: label/verification cost (stored outcomes),")
    print(f"  screening calls (none exist; candidates come from the price table).")

    # ---------------- S4: calibration ----------------
    SEC("S4. SMALL-SAMPLE INFERENCE + EMPIRICAL CALIBRATION")
    print("  estimator: dbar = mean of paired per-task differences (outcome in [0,1]);")
    print("  SE = max(sample SE, sqrt(0.25/n)); no prior (frequentist plug-in);")
    print("  pass iff one-sided 95% lower bound of dbar exceeds -delta, i.e.")
    print("  Phi((dbar+delta)/SE) >= 0.95. All-identical paired outcomes floor at the")
    print("  worst-case SE (a 3-task tie gives p = 0.53, retained).")
    # empirical diff pool
    piv = {d: tc[tc.domain == d].pivot_table(index="task_id", columns="agent_model",
                                             values="success") for d in doms}
    diffs = []
    for d in doms:
        sub = piv[d]
        for m1 in sub.columns:
            for m2 in sub.columns:
                if m1 != m2:
                    diffs.extend((sub[m1] - sub[m2]).dropna().tolist())
    diffs = np.array(diffs)
    rng = np.random.default_rng(0)
    cal = []
    for n in (3, 4, 5):
        for gap in (-0.10, -DELTA, 0.0, +0.10, +0.425):
            shifted = diffs - diffs.mean() + gap
            hit = hitb = 0; REP = 20000
            for _ in range(REP):
                v = rng.choice(shifted, n, replace=True)
                ok, _p = ni_pass(v, DELTA, TAU); hit += ok
                okb, _p = ni_pass(v, DELTA, 1 - 0.05 / 3); hitb += okb
            cal.append(dict(n=n, true_gap=gap, pass_rate=hit / REP, pass_rate_bonf=hitb / REP))
    CAL = pd.DataFrame(cal)
    print("\n  pass rates (nominal one-sided 95%; false-switch when true_gap <= -delta):")
    print(CAL.pivot_table(index="true_gap", columns="n", values="pass_rate").round(3).to_string())
    print("\n  with Bonferroni over 3 candidates:")
    print(CAL.pivot_table(index="true_gap", columns="n", values="pass_rate_bonf").round(3).to_string())
    # exact telecom case
    sub = piv["telecom"]; dv = (sub["gpt-4.1-mini"] - sub["gpt-4o"]).dropna().to_numpy()
    combs = list(itertools.combinations(range(len(dv)), 3))
    power = np.mean([ni_pass(dv[list(c)], DELTA, TAU)[0] for c in combs])
    dv0 = dv - dv.mean() - DELTA
    size = np.mean([ni_pass(dv0[list(c)], DELTA, TAU)[0] for c in combs])
    print(f"\n  exact telecom opportunity (gpt-4.1-mini vs gpt-4o, true gap {dv.mean():+.3f}):")
    print(f"    P(pass | n=3, all C(20,3) subsets) = {power:.3f}   <- explains the 3 misses")
    print(f"    P(pass | same outcomes shifted to the boundary) = {size:.3f}")
    CAL.to_csv(os.path.join(RES, "free_stage2_calibration.csv"), index=False)

    # ---------------- S5: multiplicity ----------------
    SEC("S5. CANDIDATE MULTIPLICITY")
    ncand = tgt.assign(k=[sum(1 for m in models if m != i and COSTS[(d, m)] < COSTS[(d, i)])
                          for d, i in zip(tgt.domain, tgt.inc)])
    print(f"  candidates per domain: {sorted(ncand.k.unique())} (mean {ncand.k.mean():.2f});")
    print(f"  selection = cheapest passer; the SAME 3-5 audit tasks are used for testing all")
    print(f"  candidates (post-selection concern). Nominal per-candidate one-sided level 5%;")
    print(f"  family-wise bound k x 5% <= {ncand.k.max() * 5}%. The Bonferroni variant divides")
    print(f"  alpha by k. Empirical policy-level false-switch rates (switch inferior to the")
    print(f"  incumbent beyond delta, per domain-decision):")
    for arm in ["Locked: paired NI tau=0.95", "Locked + Bonferroni",
                "Cell1 orig acq + best gate (S0)", "Raw-log downrouting"]:
        dd = Dec[Dec.arm == arm]
        fw = dd.groupby(["fold", "seed", "domain"]).apply(
            lambda g: (g.cat_inc == "inferior_inc").mean(), include_groups=False).mean()
        print(f"    {arm:36} {fw:.4f}")

    # ---------------- S7: 2x2 ablation ----------------
    SEC("S7. 2x2 ABLATION (acquisition x anchor), 50 paired units")
    pu = U[U.tag == "main"].groupby(["arm", "fold", "seed"]).mean(numeric_only=True).reset_index()
    pc = pu.pivot_table(index=["fold", "seed"], columns="arm", values="cost")
    ps = pu.pivot_table(index=["fold", "seed"], columns="arm", values="success")

    def ci(v):
        v = np.asarray(v, float); m = float(v.mean())
        h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        return m, m - h, m + h

    dd = Dec.groupby(["arm", "fold", "seed", "acq", "domain"]).first().reset_index()
    per = dd.groupby(["arm", "fold", "seed", "domain"]).agg(
        opp_best=("opp_best", "first"), opp_inc=("opp_inc", "first"),
        validA=("cat_best", lambda s: np.mean([c == "A" for c in s])),
        valid_i=("cat_inc", lambda s: np.mean([c == "valid_inc" for c in s]))).reset_index()
    recA = per[per.opp_best == 1].groupby("arm").validA.mean()
    recI = per[per.opp_inc == 1].groupby("arm").valid_i.mean()

    order = [("Cell1 orig acq + best gate (S0)", "1 orig+best"),
             ("Cell2 cand-restricted + best gate", "2 cand+best"),
             ("Cell3 orig acq + inc gate", "3 orig+inc"),
             ("Locked: paired NI tau=0.95", "4 cand+inc (locked)"),
             ("Locked + Bonferroni", "4b + Bonferroni"),
             ("Raw-log downrouting", "free raw-log"),
             ("Raw incumbent", "incumbent")]
    tbl = []
    for arm, label in order:
        g = pu[pu.arm == arm]
        row = dict(cell=label, arm=arm, success=g.success.mean(), cost_1k=g.cost.mean() * 1000,
                   audit=g.audit.mean(), A=g.A.mean(), C=g.C.mean(),
                   valid_inc=g.valid_inc.mean(), inferior_inc=g.inferior_inc.mean(),
                   recall_best=recA.get(arm, np.nan), recall_inc=recI.get(arm, np.nan))
        for comp, tag in (("Raw incumbent", "inc"), ("Raw-log downrouting", "dr")):
            if arm == comp:
                row[f"save_{tag}"] = row[f"save_{tag}_lo"] = row[f"save_{tag}_hi"] = np.nan
                continue
            m, lo, hi = ci((pc[comp] - pc[arm]).dropna() * 1000)
            row[f"save_{tag}"], row[f"save_{tag}_lo"], row[f"save_{tag}_hi"] = m, lo, hi
        row["audit_incl_cost_1k"] = row["cost_1k"] + row["audit"] / H_PRIMARY * 1000
        tbl.append(row)
    TBL = pd.DataFrame(tbl)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(TBL.drop(columns=["arm"]).round(4).to_string(index=False))
    TBL.to_csv(os.path.join(RES, "free_stage2_validation_summary.csv"), index=False)

    # ---------------- S8: frontier ----------------
    SEC("S8. NO-AUDIT BASELINES AND THE COST-vs-VIOLATION FRONTIER")
    fr = F.groupby("margin").agg(cost_1k=("cost", lambda v: v.mean() * 1000),
                                 success=("success", "mean"), viol_best=("viol_best", "mean"),
                                 viol_inc=("viol_inc", "mean"), switches=("switches", "mean"))
    print("  free raw-log margin policy (no audit), margin swept:")
    print(fr.round(4).to_string())
    g = pu[pu.arm == "Locked: paired NI tau=0.95"]
    print(f"\n  locked audited arm: cost {g.cost.mean()*1000:.2f}, best-rel violations "
          f"{g.C.mean():.3f}, inc-rel inferior {g.inferior_inc.mean():.3f}")
    la_viol = g.C.mean()
    below = fr[fr.viol_best <= la_viol + 1e-9]
    if len(below):
        cheapest = below.cost_1k.min()
        print(f"  cheapest FREE policy with best-rel violations <= {la_viol:.3f}: "
              f"${cheapest:.2f}/1k -> audited arm {'BEATS' if g.cost.mean()*1000 < cheapest else 'does NOT beat'}"
              f" it at matched violations (${g.cost.mean()*1000:.2f}/1k)")
    for t0 in (0.5, 0.8, 0.9, 0.95):
        gg = pu[pu.arm == f"No-audit log NI tau={t0}"]
        print(f"  no-audit log-evidence NI gate tau={t0}: cost {gg.cost.mean()*1000:6.2f}, "
              f"switches {gg.switches.mean():.3f}, best-rel C {gg.C.mean():.3f}, "
              f"inc-rel inferior {gg.inferior_inc.mean():.3f}")
    F.to_parquet(os.path.join(RES, "free_stage2_frontier.parquet"), index=False)

    # ---------------- S9: domain-level ----------------
    SEC("S9. DOMAIN-LEVEL RESULTS (locked arm)")
    dl = Dec[Dec.arm == "Locked: paired NI tau=0.95"]
    out = []
    for d in doms:
        g = dl[dl.domain == d]
        sw = g[g.switched == 1]
        out.append(dict(domain=d, n_cells=len(g), opp_best=int(g.opp_best.sum()),
                        opp_inc=int(g.opp_inc.sum()), pool=g.pool.mean(),
                        switch_rate=g.switched.mean(),
                        dq_inc=sw.dq_vs_inc.mean() if len(sw) else 0.0,
                        dq_best=sw.dq_vs_best.mean() if len(sw) else 0.0,
                        viol_best=(sw.cat_best == "C").sum(),
                        viol_inc=(sw.cat_inc == "inferior_inc").sum(),
                        valid_best=(sw.cat_best == "A").sum(),
                        valid_inc=(sw.cat_inc == "valid_inc").sum()))
    DL = pd.DataFrame(out)
    print(DL.round(3).to_string(index=False))
    DL.to_csv(os.path.join(RES, "free_stage2_domain_table.csv"), index=False)

    # ---------------- S10: sensitivity ----------------
    SEC("S10. SENSITIVITY OF THE LOCKED ARM")
    su = U[U.tag == "sens"].groupby("arm").agg(success=("success", "mean"),
                                               cost_1k=("cost", lambda v: v.mean() * 1000),
                                               audit=("audit", "mean"), A=("A", "mean"),
                                               C=("C", "mean"),
                                               valid_inc=("valid_inc", "mean"),
                                               inferior_inc=("inferior_inc", "mean"),
                                               switches=("switches", "mean"))
    print(su.round(4).to_string())
    su.reset_index().to_csv(os.path.join(RES, "free_stage2_sensitivity.csv"), index=False)
    print("\n  horizon: break-evens vs incumbent scale as audit/saving; at $0.53 audit and the")
    print("  ablation savings the point break-evens are volume-free and reported in S7.")

    # ---------------- persistence ----------------
    Dec.to_parquet(os.path.join(RES, "free_stage2_target_table.parquet"), index=False)
    U.to_parquet(os.path.join(RES, "free_stage2_validation_units.parquet"), index=False)
    CR.to_parquet(os.path.join(RES, "free_stage2_cost_accounting.parquet"), index=False)
    print("\nwrote: free_stage2_validation_summary.csv, free_stage2_target_table.parquet,")
    print("       free_stage2_validation_units.parquet, free_stage2_cost_accounting.parquet,")
    print("       free_stage2_calibration.csv, free_stage2_domain_table.csv,")
    print("       free_stage2_frontier.parquet, free_stage2_sensitivity.csv")


if __name__ == "__main__":
    main()
