"""External-validity study on KuaiRand-Pure: the paper's audit protocol on a real production log
with a built-in randomized-exposure stream.

Design is predeclared in docs/KUAIRAND_ANALYSIS_PLAN.md (frozen before any outcome column was
read). Estimator code is reused verbatim from code/case_core.py; decision metrics mirror
code/eval_validity/metrics.py. Vocabulary note: the JSON keys keep the historical name
`reference_regret`; all prose calls it target-value loss.

Usage:
  python code/run_kuairand_validity.py --data <KuaiRand-Pure/data dir> \
      --out results/kuairand_validity.json
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from case_core import allocate                                    # noqa: E402
from eval_validity.metrics import deployed_policy                 # noqa: E402

# ------------------------------- frozen design constants (see the plan) ------------------------
ARMS = ["39", "9", "6", "12", "3", "20", "17", "28"]     # top-8 first-tags, early-window support
K = len(ARMS)
ACT_LEVELS = ["full_active", "high_active", "middle_active"]      # rest -> "other"
J = 8                                                             # 4 activity x 2 follower groups
POOLS = dict(log=(0, 40), audit=(40, 65), score=(65, 100))        # sha1(user) mod 100
N_PARTITIONS = 30
ACQ_SEEDS = 3
BUDGET_FRACS = (0.15, 0.25, 0.35)
MIN_LOG_ROWS = 10                                                 # per-partition cell support
MIN_AUDIT_ROWS = 30                                               # global audit-pool support
NREAD = dict(click=30, like=300)                                  # D0 deviation declared in plan
NREAD_SENS = (30, 100, 300)
EVALUATORS = ("click", "like")                                    # D1, D0
COL = dict(click="is_click", like="is_like")


def _hash_mod(series, salt, mod):
    return series.astype(str).map(
        lambda s: int(hashlib.sha1(f"{salt}:{s}".encode()).hexdigest(), 16) % mod)


def load_frames(data: Path):
    use = ["user_id", "video_id", "is_click", "is_like"]
    std = pd.read_csv(data / "log_standard_4_22_to_5_08_pure.csv", usecols=use)
    rnd = pd.read_csv(data / "log_random_4_22_to_5_08_pure.csv", usecols=use)
    vid = pd.read_csv(data / "video_features_basic_pure.csv", usecols=["video_id", "tag"])
    usr = pd.read_csv(data / "user_features_pure.csv",
                      usecols=["user_id", "user_active_degree", "follow_user_num"])

    vid["arm"] = vid.tag.astype(str).str.split(",").str[0]
    vid = vid[vid.arm.isin(ARMS)]
    vid["k"] = vid.arm.map({a: i for i, a in enumerate(ARMS)})

    usr["act"] = usr.user_active_degree.where(usr.user_active_degree.isin(ACT_LEVELS), "other")
    usr["fol"] = np.where(usr.follow_user_num > 100, "hiF", "loF")
    ctx_levels = [f"{a}_{f}" for a in ACT_LEVELS + ["other"] for f in ("hiF", "loF")]
    usr["j"] = (usr.act + "_" + usr.fol).map({c: i for i, c in enumerate(ctx_levels)})
    usr["bucket"] = _hash_mod(usr.user_id, "pool", 100)
    usr["part"] = _hash_mod(usr.user_id, "partition", N_PARTITIONS)

    def prep(df):
        d = df.merge(vid[["video_id", "k"]], on="video_id")
        d = d.merge(usr[["user_id", "j", "bucket", "part"]], on="user_id")
        return d

    return prep(std), prep(rnd), ctx_levels


def cell_means(df, value_col, K=K, J=J):
    """K x J means and counts (NaN where empty)."""
    g = df.groupby(["k", "j"])[value_col].agg(["mean", "size"])
    M = np.full((K, J), np.nan)
    n = np.zeros((K, J), int)
    for (k, j), row in g.iterrows():
        M[int(k), int(j)] = row["mean"]
        n[int(k), int(j)] = int(row["size"])
    return M, n


def weighted_tvl(pol, pol_ref, mu, w):
    """Traffic-weighted target-value loss; unweighted twin returned alongside."""
    gaps, wts = [], []
    for j in range(len(pol)):
        if pol[j] >= 0 and pol_ref[j] >= 0:
            gaps.append(mu[pol_ref[j], j] - mu[pol[j], j])
            wts.append(w[j])
    if not gaps:
        return float("nan"), float("nan"), float("nan")
    gaps, wts = np.array(gaps), np.array(wts)
    top1 = float(np.mean([pol[j] != pol_ref[j] for j in range(len(pol))
                          if pol[j] >= 0 and pol_ref[j] >= 0]))
    return float((gaps * wts).sum() / wts.sum()), float(gaps.mean()), top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default="results/kuairand_validity.json", type=Path)
    args = ap.parse_args()

    std, rnd, ctx_levels = load_frames(args.data)

    pools = {name: (lambda lo_hi: (lambda b: (b >= lo_hi[0]) & (b < lo_hi[1])))(r)
             for name, r in POOLS.items()}
    std_log = std[pools["log"](std.bucket)]
    rnd_log = rnd[pools["log"](rnd.bucket)]          # same users' randomized rows (for A0)
    rnd_aud = rnd[pools["audit"](rnd.bucket)]
    rnd_sco = rnd[pools["score"](rnd.bucket)]

    # fixed reference tables (scoring pool) and traffic weights (full standard-late stream)
    mu_ref, n_sco = cell_means(rnd_sco, "is_like")
    mu_click_sco, _ = cell_means(rnd_sco, "is_click")
    w = std.groupby("j").size().reindex(range(J), fill_value=0).to_numpy(float)
    w = w / w.sum()

    # global audit pools per evaluator
    cellq = {ev: {} for ev in EVALUATORS}
    aud_n = np.zeros((K, J), int)
    for (k, j), grp in rnd_aud.groupby(["k", "j"]):
        aud_n[int(k), int(j)] = len(grp)
        for ev in EVALUATORS:
            cellq[ev][(int(k), int(j))] = grp[COL[ev]].to_numpy(float)

    rows, sens_rows = [], []
    rng_global = np.random.default_rng(20260805)
    for p in range(N_PARTITIONS):
        sl = std_log[std_log.part == p]
        rl = rnd_log[rnd_log.part == p]
        M_n = {}
        # A1 surfaces from the partition's standard rows
        for ev in EVALUATORS:
            M_n[("A1", ev)] = cell_means(sl, COL[ev])
        n_A1 = M_n[("A1", EVALUATORS[0])][1]
        # A0: same users' randomized rows, subsampled per cell to the A1 cell volume
        sub_idx = []
        for (k, j), grp in rl.groupby(["k", "j"]):
            take = min(len(grp), n_A1[int(k), int(j)])
            if take > 0:
                sub_idx.append(grp.sample(n=take, random_state=int(
                    rng_global.integers(2**31)) ).index)
        rl_sub = rl.loc[np.concatenate(sub_idx)] if sub_idx else rl.iloc[:0]
        for ev in EVALUATORS:
            M_n[("A0", ev)] = cell_means(rl_sub, COL[ev])

        for assign in ("A1", "A0"):
            n_here = M_n[(assign, EVALUATORS[0])][1]
            supp = (n_here >= MIN_LOG_ROWS) & (aud_n >= MIN_AUDIT_ROWS) & (n_sco > 0) \
                & ~np.isnan(mu_ref)                       # guard: never argmax a NaN target cell
            pol_ref = deployed_policy(np.where(supp, mu_ref, -np.inf), supp)
            refs = {"target_ref": pol_ref,
                    "full_click_ref": deployed_policy(
                        np.where(supp, mu_click_sco, -np.inf), supp)}
            if assign == "A1":
                counts = np.where(supp, n_here.astype(float), -np.inf)
                refs["traffic_vote"] = deployed_policy(counts, supp)
            for name, pol in refs.items():
                tw, uw, t1 = weighted_tvl(pol, pol_ref, mu_ref, w)
                rows.append(dict(partition=p, acq_seed=-1, assignment=assign, evaluator="ref",
                                 budget=0.0, B=0, method=name, reference_regret=tw,
                                 reference_regret_unweighted=uw, top1_error=t1))
            for ev in EVALUATORS:
                M = np.nan_to_num(M_n[(assign, ev)][0], nan=0.0)
                n_read = NREAD[ev]
                for frac in BUDGET_FRACS:
                    B = max(K, int(round(frac * supp.sum())))
                    for s in range(ACQ_SEEDS):
                        seed = 10_000 * p + 100 * int(frac * 100) + s
                        for method in ("raw", "uniform", "case"):
                            V, nr = allocate(method, None, M, cellq[ev], supp, B, seed, n_read)
                            pol = deployed_policy(np.where(supp, V, -np.inf), supp)
                            tw, uw, t1 = weighted_tvl(pol, pol_ref, mu_ref, w)
                            rows.append(dict(partition=p, acq_seed=s, assignment=assign,
                                             evaluator=ev, budget=frac, B=B, method=method,
                                             cells_read=nr, reference_regret=tw,
                                             reference_regret_unweighted=uw, top1_error=t1))
                # n_read sensitivity, CASE at the primary budget only
                for nr_s in NREAD_SENS:
                    if nr_s == n_read:
                        continue
                    B = max(K, int(round(0.35 * supp.sum())))
                    for s in range(ACQ_SEEDS):
                        seed = 10_000 * p + 100 * 35 + s
                        V, nr = allocate("case", None, M, cellq[ev], supp, B, seed, nr_s)
                        pol = deployed_policy(np.where(supp, V, -np.inf), supp)
                        tw, uw, t1 = weighted_tvl(pol, pol_ref, mu_ref, w)
                        sens_rows.append(dict(partition=p, acq_seed=s, assignment=assign,
                                              evaluator=ev, n_read=nr_s, budget=0.35,
                                              method="case", reference_regret=tw,
                                              reference_regret_unweighted=uw, top1_error=t1))

    # ---------------- descriptives (no acquisition involved) -----------------------------------
    def c_decomp(frame):
        """Two-way additive decomposition of C_kj = click-mean - like-mean. Descriptive: one
        observation per cell, residual = interaction."""
        mc, _ = cell_means(frame, "is_click")
        ml, _ = cell_means(frame, "is_like")
        C = mc - ml
        ok = ~np.isnan(C)
        Cc = np.where(ok, C, 0.0)
        grand = Cc[ok].mean()
        arm_fx = np.array([Cc[k, ok[k]].mean() - grand if ok[k].any() else 0.0
                           for k in range(K)])
        ctx_fx = np.array([Cc[ok[:, j], j].mean() - grand if ok[:, j].any() else 0.0
                           for j in range(J)])
        fitted = grand + arm_fx[:, None] + ctx_fx[None, :]
        ss_tot = float(((Cc - grand)[ok] ** 2).sum())
        return dict(grand=grand, arm_fx=arm_fx.tolist(), ctx_fx=ctx_fx.tolist(),
                    share_arm=float(((arm_fx[:, None] * np.ones((K, J)))[ok] ** 2).sum())/ss_tot,
                    share_ctx=float(((ctx_fx[None, :] * np.ones((K, J)))[ok] ** 2).sum())/ss_tot,
                    share_resid=float(((Cc - fitted)[ok] ** 2).sum())/ss_tot)

    dec_scoring = c_decomp(rnd_sco)      # predeclared primary (plan: scoring pool)
    dec_full = c_decomp(rnd)             # full randomized stream, sensitivity
    grand, arm_fx = dec_scoring["grand"], np.array(dec_scoring["arm_fx"])
    ctx_fx = np.array(dec_scoring["ctx_fx"])

    from scipy.stats import spearmanr
    exposure = std.groupby(["k", "j"]).size().unstack(fill_value=0).reindex(
        index=range(K), columns=range(J), fill_value=0).to_numpy(float)
    rho = []
    for j in range(J):
        m = ~np.isnan(mu_ref[:, j])
        if m.sum() >= 3:
            rho.append(float(spearmanr(exposure[m, j], mu_ref[m, j]).statistic))
    descriptives = dict(
        C_grand=grand, C_arm_effects=arm_fx.tolist(), C_ctx_effects=ctx_fx.tolist(),
        C_var_share_arm=dec_scoring["share_arm"], C_var_share_ctx=dec_scoring["share_ctx"],
        C_var_share_resid=dec_scoring["share_resid"],
        C_decomp_scoring=dec_scoring, C_decomp_full_stream=dec_full,
        spearman_exposure_vs_target_by_ctx=rho,
        spearman_mean=float(np.mean(rho)),
        mu_ref=np.where(np.isnan(mu_ref), None, mu_ref).tolist(),
        mu_click_scoring=np.where(np.isnan(mu_click_sco), None, mu_click_sco).tolist(),
        scoring_rows_per_cell=n_sco.tolist(), audit_rows_per_cell=aud_n.tolist(),
        traffic_weights=w.tolist(), ctx_levels=ctx_levels, arms=ARMS)

    spec = dict(dataset="KuaiRand-Pure", zenodo="10439422", license="CC BY 4.0",
                arms=ARMS, J=J, ctx_levels=ctx_levels, pools=POOLS,
                n_partitions=N_PARTITIONS, acq_seeds=ACQ_SEEDS, budget_fracs=BUDGET_FRACS,
                min_log_rows=MIN_LOG_ROWS, min_audit_rows=MIN_AUDIT_ROWS, n_read=NREAD,
                n_read_sensitivity=NREAD_SENS,
                windows=dict(M="standard 4/22-5/08", audit="random 4/22-5/08",
                             design="standard 4/08-4/21"),
                plan="docs/KUAIRAND_ANALYSIS_PLAN.md")
    out = dict(spec=spec, rows=rows, sensitivity_rows=sens_rows, descriptives=descriptives)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(rows)} rows, {len(sens_rows)} sensitivity rows")


if __name__ == "__main__":
    main()
