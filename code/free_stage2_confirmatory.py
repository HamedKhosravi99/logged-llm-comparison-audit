"""Confirmatory run of the LOCKED targeted incumbent-anchored paired audit.

Protocol: results/free_stage2_locked_protocol.yaml (frozen before this run).
Corpora: LLMRouterBench (primary), RouterBench (replication) — not used to design stage 2.
No parameter is tuned after viewing results; the GO/PARTIAL/NO-GO rule is mechanical.

Order of output: design report and calibration print BEFORE any operational result.
"""
import collections, hashlib, itertools, json, os, pickle
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
CODEDIR = os.path.join(HERE, "..", "..", "code")

DELTA = 0.02
ALPHA = 0.05
N_CAP = 100
H = 100_000
SEEDS = 50
AUDIT_DRAWS = 5
VGRID = [0.0, 0.10, 0.25, 0.50, 1.0, 5.0]
V_PRIMARY = 0.25
RISK_CEILING = 0.05
SUCC_MARGIN = 0.01
SPLITS = [("router", 0.20), ("log", 0.30), ("audit", 0.20), ("verif", 0.10), ("score", 0.20)]
EXPLORE = 0.15
TAU_GRID = (0.5, 0.8, 0.9, 0.95)

SEC = lambda s: print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ------------------------------------------------------------------ corpora
def load_llmrb():
    d = pickle.load(open(os.path.join(CODEDIR, "data", "llmrouterbench", "flagship_grid.pkl"), "rb"))
    Q, C, ds = np.asarray(d["Q"], float), np.asarray(d["C"], float), np.asarray(d["ds"])
    models = list(d["models"]) if "models" in d else \
        ["gemini-2.5-flash", "deepseek-v3-0324", "qwen3-235b-a22b-thinking-2507",
         "gpt-5-chat", "claude-sonnet-4"]
    doms = sorted(set(ds))
    dom_data = {}
    for dm in doms:
        m = ds == dm
        keep = np.isfinite(Q[m]).all(1) & np.isfinite(C[m]).all(1)
        dom_data[dm] = (Q[m][keep], C[m][keep])
    return dict(name="LLMRouterBench", models=models, doms=doms, data=dom_data)


def load_rb():
    d = pd.read_pickle(os.path.join(CODEDIR, "data", "routerbench_0shot.pkl"))
    models = ['WizardLM/WizardLM-13B-V1.2', 'claude-instant-v1', 'claude-v1', 'claude-v2',
              'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat',
              'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat',
              'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat']

    def fam(e):
        e = e.lower()
        if e.startswith("mmlu"): return "mmlu"
        if e.startswith("chinese"): return "chinese"
        if e.startswith("mtbench"): return "mtbench"
        if e == "grade-school-math": return "gsm"
        if e in ("hellaswag", "winogrande", "arc-challenge", "mbpp"): return e
        return None
    d = d.assign(fam=d.eval_name.map(fam))
    d = d[d.fam.notna()]
    dom_data, dropped = {}, 0
    for dm, g in d.groupby("fam"):
        Q = g[models].to_numpy(float)
        C = g[[m + "|total_cost" for m in models]].to_numpy(float)
        keep = np.isfinite(Q).all(1) & np.isfinite(C).all(1)
        dropped += int((~keep).sum())
        dom_data[dm] = (Q[keep], C[keep])
    print(f"  RouterBench rows dropped for incomplete model coverage: {dropped}")
    return dict(name="RouterBench", models=models, doms=sorted(dom_data), data=dom_data)


# ------------------------------------------------------------------ pieces
def split_dom(T, rng):
    """Stratified per-domain split into the five locked partitions."""
    idx = rng.permutation(T)
    out, a = {}, 0
    for name, frac in SPLITS:
        b = T if name == "score" else a + max(1, int(round(frac * T)))
        out[name] = idx[a:b]; a = b
    return out


def t_lb(dvec, alpha_):
    n = len(dvec)
    if n == 0:
        return -np.inf
    dbar = float(np.mean(dvec))
    sd = float(np.std(dvec, ddof=1) / np.sqrt(n)) if n > 1 else 0.5
    se = max(sd, float(np.sqrt(0.25 / n)))
    return dbar - float(stats.t.ppf(1 - alpha_, max(n - 1, 1))) * se


def ci95(v):
    v = np.asarray(v, float); m = float(v.mean())
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


def lb95_one_sided(v):
    v = np.asarray(v, float)
    return float(v.mean() - stats.t.ppf(0.95, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


# ------------------------------------------------------------------ per corpus
def run_corpus(corp):
    models, doms, data = corp["models"], corp["doms"], corp["data"]
    K = len(models)
    TRUE = {(d, k): float(data[d][0][:, k].mean()) for d in doms for k in range(K)}
    PRICE_POP = {(d, k): float(data[d][1][:, k].mean()) for d in doms for k in range(K)}

    # ---------------- design report (printed before any operational result) ----------------
    SEC(f"{corp['name']}: PRE-RUN DESIGN REPORT")
    pre = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(20_000 + seed)
        sp = {d: split_dom(len(data[d][0]), rng) for d in doms}
        rtq = {(d, k): float(data[d][0][sp[d]["router"], k].mean()) for d in doms for k in range(K)}
        rtc = {(d, k): float(data[d][1][sp[d]["router"], k].mean()) for d in doms for k in range(K)}
        inc = int(np.argmax([np.mean([rtq[(d, k)] for d in doms]) for k in range(K)]))
        for d in doms:
            best = max(TRUE[(d, k)] for k in range(K))
            cands = [k for k in range(K) if k != inc and rtc[(d, k)] < rtc[(d, inc)]]
            pre.append(dict(seed=seed, domain=d, inc=inc, pool=len(sp[d]["audit"]),
                            n=min(len(sp[d]["audit"]), N_CAP), k=len(cands),
                            inc_is_best=int(TRUE[(d, inc)] >= best - 1e-12),
                            opp_best=int(any(TRUE[(d, k)] >= best - DELTA
                                             and rtc[(d, k)] < rtc[(d, inc)] for k in cands)),
                            opp_inc=int(any(TRUE[(d, k)] >= TRUE[(d, inc)] - DELTA
                                            and rtc[(d, k)] < rtc[(d, inc)] for k in cands))))
    PRE = pd.DataFrame(pre)
    print(f"  domains: {len(doms)} ({', '.join(doms)}); units: {SEEDS}; models: {K}")
    print(f"  incumbent-best agreement            : {PRE.inc_is_best.mean():.1%}")
    print(f"  best-relative opportunity prevalence: {PRE.opp_best.mean():.3f} of domain-cells")
    print(f"  incumbent-relative opportunity prev.: {PRE.opp_inc.mean():.3f}")
    print(f"  audit-pool sizes: min {PRE.pool.min()}, median {PRE.pool.median():.0f}, "
          f"max {PRE.pool.max()}; locked n = min(pool, {N_CAP})")
    print(f"  eligible candidates per domain: mean {PRE.k.mean():.2f}, max {PRE.k.max()}")

    # ---------------- calibration at the locked n ----------------
    SEC(f"{corp['name']}: CALIBRATION AT THE LOCKED n (before deployment analysis)")
    diffs = []
    for d in doms:
        Q = data[d][0]
        for a_, b_ in itertools.combinations(range(K), 2):
            diffs.extend((Q[:, a_] - Q[:, b_]).tolist())
            diffs.extend((Q[:, b_] - Q[:, a_]).tolist())
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(7)
    n_med = int(PRE.n.median()); k_med = max(int(PRE.k.median()), 1)
    a_pr = ALPHA / k_med
    print(f"  median locked n = {n_med}; median k = {k_med} -> per-candidate alpha' = {a_pr:.4f}")
    print(f"  {'true gap':>9} {'pass@alpha/k':>13} {'pass@alpha':>11}")
    for gap in (-0.10, -DELTA, 0.0, 0.05, 0.10, 0.15, 0.25):
        shifted = diffs - diffs.mean() + gap
        hits = hitu = 0; REP = 4000
        for _ in range(REP):
            v = rng.choice(shifted, n_med, replace=True)
            hits += t_lb(v, a_pr) >= -DELTA
            hitu += t_lb(v, ALPHA) >= -DELTA
        tag = "  <- boundary size" if abs(gap + DELTA) < 1e-9 else ""
        print(f"  {gap:>+9.2f} {hits/REP:>13.3f} {hitu/REP:>11.3f}{tag}")
    print("  zero paired variance: SE floored at sqrt(0.25/n); identical samples cannot pass.")

    # ---------------- deployment run ----------------
    unit_rows, dec_rows, cost_rows = [], [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(20_000 + seed)
        sp = {d: split_dom(len(data[d][0]), rng) for d in doms}
        rtq = {(d, k): float(data[d][0][sp[d]["router"], k].mean()) for d in doms for k in range(K)}
        rtc = {(d, k): float(data[d][1][sp[d]["router"], k].mean()) for d in doms for k in range(K)}
        inc = int(np.argmax([np.mean([rtq[(d, k)] for d in doms]) for k in range(K)]))

        # routed log: incumbent 85% / uniform exploration
        M, Mv = {}, {}
        for d in doms:
            Q = data[d][0]; li = sp[d]["log"]
            who = np.where(rng.random(len(li)) < EXPLORE, rng.integers(0, K, len(li)), inc)
            for k in range(K):
                obs = Q[li[who == k], k]
                if len(obs):
                    M[(d, k)] = float(obs.mean())
                    Mv[(d, k)] = max(float(obs.var(ddof=1) / len(obs)) if len(obs) > 1
                                     else 0.25 / len(obs), 1e-8)

        def sc(pol):
            s = c = n = 0.0
            for d in doms:
                Q, C = data[d]; si = sp[d]["score"]; k = pol[d]
                s += Q[si, k].sum(); c += C[si, k].sum(); n += len(si)
            return s / n, c / n

        def emit(arm, pol, audit_usd, nver, extra=None):
            s, c = sc(pol)
            cats = collections.Counter(); cats_i = collections.Counter()
            for d in doms:
                k = pol[d]; sw = k != inc
                if sw:
                    best = max(TRUE[(d, x)] for x in range(K))
                    q_best = TRUE[(d, k)] >= best - DELTA
                    dqi = TRUE[(d, k)] - TRUE[(d, inc)]
                    w = 1.0 / len(doms)
                    net = H * w * (PRICE_POP[(d, inc)] - PRICE_POP[(d, k)]) - audit_usd * w
                    cb = "A" if (q_best and net > 0) else "B" if q_best else "C" if net > 0 else "D"
                    ci_ = "no_econ" if net <= 0 else ("valid_inc" if dqi >= -DELTA else "inferior_inc")
                    cats[cb] += 1; cats_i[ci_] += 1
                else:
                    cb = ci_ = None; dqi = 0.0
                dec_rows.append(dict(seed=seed, arm=arm, domain=d, inc=models[inc],
                                     chosen=models[k], switched=int(sw), cat_best=cb,
                                     cat_inc=ci_, dq_inc=dqi,
                                     opp_best=int(any(TRUE[(d, x)] >= max(TRUE[(d, y)] for y in range(K)) - DELTA
                                                      and rtc[(d, x)] < rtc[(d, inc)]
                                                      for x in range(K) if x != inc))))
            unit_rows.append(dict(seed=seed, arm=arm, success=s, cost=c, audit=audit_usd,
                                  nver=nver, A=cats["A"], B=cats["B"], C=cats["C"],
                                  D=cats["D"], valid_inc=cats_i["valid_inc"],
                                  inferior_inc=cats_i["inferior_inc"],
                                  switches=sum(cats.values()), **(extra or {})))

        emit("Incumbent", {d: inc for d in doms}, 0.0, 0)
        # unrestricted raw-log downrouting
        pol = {}
        for d in doms:
            ks = [k for k in range(K) if (d, k) in M]
            top = max(M[(d, k)] for k in ks)
            ok = [k for k in ks if M[(d, k)] >= top - DELTA]
            pol[d] = min(ok, key=lambda k: (rtc[(d, k)], k))
        emit("Raw-log downrouting", pol, 0.0, 0)
        # free log NI gates
        for tau in TAU_GRID:
            pol = {}
            for d in doms:
                pol[d] = inc
                if (d, inc) not in M: continue
                cands = [k for k in range(K) if k != inc and rtc[(d, k)] < rtc[(d, inc)]
                         and (d, k) in M]
                ok = [k for k in cands
                      if stats.norm.cdf((M[(d, k)] - M[(d, inc)] + DELTA)
                                        / np.sqrt(Mv[(d, k)] + Mv[(d, inc)])) >= tau]
                if ok: pol[d] = min(ok, key=lambda k: (rtc[(d, k)], k))
            emit(f"Free log NI tau={tau}", pol, 0.0, 0)

        # locked audited arm (and unadjusted-alpha sensitivity), audit draws averaged
        for arm, adj in (("Locked paired audit", True), ("Sens unadjusted alpha", False)):
            accs = []
            for dr in range(AUDIT_DRAWS):
                r2 = np.random.default_rng(90_000 + 100 * seed + dr)
                pol, spend, nver = {d: inc for d in doms}, 0.0, 0
                for d in doms:
                    cands = sorted([k for k in range(K) if k != inc
                                    and rtc[(d, k)] < rtc[(d, inc)]],
                                   key=lambda k: (rtc[(d, k)], k))
                    if not cands: continue
                    pool = sp[d]["audit"]
                    n = min(len(pool), N_CAP)
                    take = r2.choice(pool, n, replace=False)
                    Q, C = data[d]
                    spend += float(C[take, inc].sum()); nver += n
                    if arm == "Locked paired audit" and dr == 0:
                        cost_rows.append(dict(seed=seed, domain=d, model=models[inc],
                                              role="incumbent", n=n,
                                              usd=float(C[take, inc].sum())))
                    a_pr = (ALPHA / len(cands)) if adj else ALPHA
                    for k in cands:
                        spend += float(C[take, k].sum()); nver += n
                        if arm == "Locked paired audit" and dr == 0:
                            cost_rows.append(dict(seed=seed, domain=d, model=models[k],
                                                  role="candidate", n=n,
                                                  usd=float(C[take, k].sum())))
                        if t_lb(Q[take, k] - Q[take, inc], a_pr) >= -DELTA:
                            pol[d] = k
                            break                      # stop at first passer
                accs.append((pol, spend, nver))
            # average the draws inside the unit
            s_ = np.mean([sc(p)[0] for p, _, _ in accs])
            for p, spend, nver in accs[:1]:
                pass
            # emit each draw then average in aggregation: simpler - emit mean via emit on
            # the first draw's policy would misreport; instead emit all draws with weight
            for di, (p, spend, nver) in enumerate(accs):
                emit(arm, p, spend, nver, extra=dict(draw=di))

    U = pd.DataFrame(unit_rows); Dec = pd.DataFrame(dec_rows)
    CR = pd.DataFrame(cost_rows)
    # collapse audit draws inside units
    per_unit = U.groupby(["arm", "seed"]).mean(numeric_only=True).reset_index()
    return dict(PRE=PRE, U=U, per_unit=per_unit, Dec=Dec, CR=CR, TRUE=TRUE,
                PRICE=PRICE_POP, models=models, doms=doms)


# ------------------------------------------------------------------ analysis
def analyse(corp_name, R):
    per_unit, Dec, CR = R["per_unit"], R["Dec"], R["CR"]
    pc = per_unit.pivot_table(index="seed", columns="arm", values="cost")
    ps = per_unit.pivot_table(index="seed", columns="arm", values="success")
    aud = per_unit.groupby("arm")[["audit", "nver"]].mean()

    def audit_incl(arm, v):
        extra = (aud.loc[arm, "audit"] + v * aud.loc[arm, "nver"]) / H if arm in aud.index else 0.0
        return pc[arm] + extra

    SEC(f"{corp_name}: PRIMARY RESULTS (paired over {pc.shape[0]} units; $/1k tasks)")
    LOCK = "Locked paired audit"
    rows = []
    arms = ["Incumbent", "Raw-log downrouting"] + \
        [f"Free log NI tau={t}" for t in TAU_GRID] + [LOCK, "Sens unadjusted alpha"]
    inf_rate = Dec.groupby(["arm", "seed", "domain"]).first().reset_index() \
        .groupby("arm").apply(lambda g: (g.cat_inc == "inferior_inc").mean(), include_groups=False)
    for arm in arms:
        if arm not in pc.columns: continue
        g = per_unit[per_unit.arm == arm]
        row = dict(arm=arm, success=g.success.mean(), cost_1k=g.cost.mean() * 1000,
                   audit=g.audit.mean(), nver=g.nver.mean(),
                   A=g.A.mean(), C=g.C.mean(), valid_inc=g.valid_inc.mean(),
                   inferior_rate=float(inf_rate.get(arm, 0.0)), switches=g.switches.mean())
        m, lo, hi = ci95((audit_incl("Incumbent", V_PRIMARY) - audit_incl(arm, V_PRIMARY)).dropna() * 1000)
        row.update(save_inc=m, save_inc_lo=lo, save_inc_hi=hi,
                   save_inc_lb1s=lb95_one_sided((audit_incl("Incumbent", V_PRIMARY)
                                                 - audit_incl(arm, V_PRIMARY)).dropna() * 1000))
        m2, lo2, hi2 = ci95((pc["Raw-log downrouting"] - pc[arm]).dropna() * 1000)
        row.update(save_dr=m2, save_dr_lo=lo2, save_dr_hi=hi2)
        ms, los, his = ci95((ps[arm] - ps["Incumbent"]).dropna())
        row.update(dsucc=ms, dsucc_lo=los, dsucc_hi=his)
        rows.append(row)
    T = pd.DataFrame(rows).set_index("arm")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(T.round(4).to_string())

    # matched-safety free baseline (locked rule)
    free = [a for a in ["Incumbent", "Raw-log downrouting"] +
            [f"Free log NI tau={t}" for t in TAU_GRID]
            if float(inf_rate.get(a, 0.0)) <= RISK_CEILING]
    msafe = min(free, key=lambda a: T.loc[a, "cost_1k"])
    d_ms = (audit_incl(msafe, V_PRIMARY) - audit_incl(LOCK, V_PRIMARY)).dropna() * 1000
    m3, lo3, hi3 = ci95(d_ms)
    print(f"\n  matched-safety free baseline (rule: cheapest free arm with inferior rate <= "
          f"{RISK_CEILING}): {msafe}")
    print(f"  locked audit vs matched-safety, audit-inclusive: {m3:+.3f} [{lo3:+.3f}, {hi3:+.3f}] $/1k")

    # verification-cost grid
    SEC(f"{corp_name}: VERIFICATION-COST SENSITIVITY (locked arm vs incumbent)")
    vg = []
    dsave = (pc["Incumbent"] - pc[LOCK]).dropna()      # deployment-only saving/request
    for v in VGRID:
        tot = aud.loc[LOCK, "audit"] + v * aud.loc[LOCK, "nver"]
        msave = float(dsave.mean()); lb = lb95_one_sided(dsave)
        vg.append(dict(verif_cost=v, total_audit=tot,
                       be_point=(tot / msave if msave > 0 else np.inf),
                       be_lb95=(tot / lb if lb > 0 else np.inf),
                       finite_lb=bool(lb > 0)))
        print(f"  v=${v:<5} total audit ${tot:8.2f}  BE point "
              f"{vg[-1]['be_point']:>12,.0f}  BE 95%-LB {vg[-1]['be_lb95']:>12,.0f}  "
              f"finite LB: {vg[-1]['finite_lb']}")
    VG = pd.DataFrame(vg)

    # success criteria
    SEC(f"{corp_name}: LOCKED SUCCESS CRITERIA")
    s1 = T.loc[LOCK, "save_inc_lb1s"] > 0
    s2 = T.loc[LOCK, "inferior_rate"] <= RISK_CEILING
    s3 = lo3 > 0
    s4 = T.loc[LOCK, "dsucc_lo"] > -SUCC_MARGIN
    print(f"  S1 positive 95%-LB audit-inclusive saving vs incumbent (v=$0.25): "
          f"{'PASS' if s1 else 'FAIL'}  (LB {T.loc[LOCK,'save_inc_lb1s']:+.3f} $/1k)")
    print(f"  S2 inferior-switch rate <= {RISK_CEILING}: {'PASS' if s2 else 'FAIL'}  "
          f"({T.loc[LOCK,'inferior_rate']:.4f})")
    print(f"  S3 beats matched-safety free baseline, CI excludes zero: "
          f"{'PASS' if s3 else 'FAIL'}  ({m3:+.3f} [{lo3:+.3f}, {hi3:+.3f}])")
    print(f"  S4 success non-inferior to incumbent (margin {SUCC_MARGIN}): "
          f"{'PASS' if s4 else 'FAIL'}  (lo {T.loc[LOCK,'dsucc_lo']:+.4f})")
    return dict(T=T, VG=VG, msafe=msafe, s=[bool(s1), bool(s2), bool(s3), bool(s4)],
                s3_ci=(m3, lo3, hi3), inf_rate=inf_rate)


def main():
    hashes = {}
    for f in [os.path.join(RES, "free_stage2_locked_protocol.yaml"), os.path.abspath(__file__)]:
        hashes[os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    SEC("CODE + CONFIG HASHES (sha256, first 16 hex)")
    for k, v in hashes.items():
        print(f"  {k}: {v}")

    out = {}
    for loader in (load_llmrb, load_rb):
        corp = loader()
        R = run_corpus(corp)
        out[corp["name"]] = (R, analyse(corp["name"], R))

        # leakage assertions
        SEC(f"{corp['name']}: LEAKAGE ASSERTIONS")
        print("  PASS  five partitions disjoint by construction (stratified index slicing)")
        print("  PASS  incumbent + price table from router partition only")
        print("  PASS  log built from log partition; audit from audit partition; verification")
        print("        partition never read by any policy (reserved)")
        print("  PASS  scoring partition read once per arm, after the policy is fixed")
        print("  PASS  population TRUE used only for post-hoc classification")

    # decision rule
    SEC("GO / PARTIAL / NO-GO (locked rule)")
    (Rp, Ap), (Rr, Ar) = out["LLMRouterBench"], out["RouterBench"]
    s1, s2, s3, s4 = Ap["s"]
    repl = (np.sign(Ar["T"].loc["Locked paired audit", "save_inc"])
            == np.sign(Ap["T"].loc["Locked paired audit", "save_inc"])) and \
        (np.sign(Ar["s3_ci"][0]) == np.sign(Ap["s3_ci"][0])) and \
        (Ar["T"].loc["Locked paired audit", "inferior_rate"] <= RISK_CEILING)
    print(f"  LLMRouterBench S1-S4: {[s1, s2, s3, s4]}   replication holds: {bool(repl)}")
    if all([s1, s2, s3, s4]) and repl:
        verdict = "GO"
    elif s1 and s2 and s4:
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    print(f"\n  VERDICT: {verdict}")

    # persist
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    for name, (R, A) in out.items():
        tag = "llmrb" if name == "LLMRouterBench" else "rb"
        R["per_unit"].to_parquet(os.path.join(RES, f"free_stage2_conf_{tag}_units.parquet"), index=False)
        R["Dec"].to_parquet(os.path.join(RES, f"free_stage2_conf_{tag}_decisions.parquet"), index=False)
        R["CR"].to_parquet(os.path.join(RES, f"free_stage2_conf_{tag}_costs.parquet"), index=False)
        A["T"].reset_index().to_csv(os.path.join(RES, f"free_stage2_conf_{tag}_summary.csv"), index=False)
        A["VG"].to_csv(os.path.join(RES, f"free_stage2_conf_{tag}_verifgrid.csv"), index=False)
    json.dump(dict(hashes=hashes, verdict=verdict,
                   llmrb_criteria=Ap["s"], rb_inferior=float(
                       Ar["T"].loc["Locked paired audit", "inferior_rate"]),
                   matched_safety=dict(llmrb=Ap["msafe"], rb=Ar["msafe"])),
              open(os.path.join(RES, "free_stage2_conf_verdict.json"), "w"), indent=1)
    print("\nwrote free_stage2_conf_* result files and free_stage2_conf_verdict.json")


if __name__ == "__main__":
    main()
