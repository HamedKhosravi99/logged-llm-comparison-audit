"""Observational causal baselines on the SAME replay partitions (plan B §3.1-3.3).

Log-only estimators of the per-cell evaluator surface, deployed through the identical decision
and scored by the identical metrics as the published arms:

  reg        outcome regression only (cross-fitted ridge on prompt embeddings)
  ipw_true   stabilized (Hajek) IPW, exact logging propensities
  ipw_est    stabilized IPW, cross-fitted multinomial-logistic propensities
  dr_true    cross-fitted AIPW/DR, exact propensities
  dr_est     cross-fitted AIPW/DR, estimated propensities

Reproduction gate: the trust arm is recomputed per partition and must match
results/tae_validity.json to its stored rounding. No match, no results.

Primary predeclared contrast: CASE vs dr_true on A1-D0 target-value loss; everything else
exploratory. Diagnostics per §3.3/§3.5 of the plan: overlap, ESS, clipping sensitivity,
propensity log-loss/Brier/calibration, estimated-vs-true comparison.

    python run_causal_baselines.py [--partitions 30]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from eval_validity import assignment as A
from eval_validity import evaluators as EV
from eval_validity import metrics as MT
from eval_validity.grid import build_base, load_raw

RES = os.path.join(HERE, "..", "results")
EPS, N_READ = 0.15, 30
CLIPS = (0.0, 0.01, 0.025, 0.05, 0.10)
PRIMARY_CLIP = 0.01
FOLDS = 5


def observed_scores(base, log_data, Cs):
    """Per logged request: observed evaluator score of the ASSIGNED model (o-stream)."""
    Q, ctx = base["Q"], log_data["ctx"]
    idx, asg = log_data["idx"], log_data["assign"]
    c = Cs["o"][asg, ctx]
    return np.clip(Q[idx, asg] + c, 0.0, 1.0)


def true_propensities(base, log_data):
    K = base["K"]
    pi0 = base["inc_prompt"][log_data["idx"]]
    e = np.full((len(pi0), K), EPS / K)
    e[np.arange(len(pi0)), pi0] += 1.0 - EPS
    return e


def estimated_propensities(EMB, log_data, seed):
    """Cross-fitted multinomial logistic on pre-assignment prompt embeddings."""
    X = EMB[log_data["idx"]]
    y = log_data["assign"]
    K = int(log_data["K"])
    e = np.zeros((len(y), K))
    for tr, te in KFold(FOLDS, shuffle=True, random_state=seed).split(X):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        P = np.zeros((len(te), K))
        P[:, clf.classes_] = clf.predict_proba(X[te])
        e[te] = P
    return e


def outcome_model(EMB, K, log_data, Y, seed):
    """Cross-fitted per-model ridge \\hat m_k(x) on embeddings, predictions for every logged x."""
    X = EMB[log_data["idx"]]
    asg = log_data["assign"]
    M = np.zeros((len(Y), K))
    for tr, te in KFold(FOLDS, shuffle=True, random_state=seed).split(X):
        for k in range(K):
            trk = tr[asg[tr] == k]
            if len(trk) < 8:                     # too thin: fall back to the fold's mean
                M[te, k] = Y[tr].mean()
                continue
            m = Ridge(alpha=1.0).fit(X[trk], Y[trk])
            M[te, k] = m.predict(X[te])
    return np.clip(M, 0.0, 1.0)


def cell_surfaces(base, log_data, Y, e, mhat, clip):
    """IPW (Hajek) and DR surfaces over the log cohort, per (k, j)."""
    K, J = base["K"], base["J"]
    ctx, asg = log_data["ctx"], log_data["assign"]
    ec = np.clip(e, clip, None) if clip > 0 else e
    V_ipw = np.full((K, J), np.nan)
    V_dr = np.full((K, J), np.nan)
    V_reg = np.full((K, J), np.nan)
    ess = np.zeros((K, J))
    for j in range(J):
        mj = ctx == j
        nj = int(mj.sum())
        for k in range(K):
            sel = mj & (asg == k)
            w = 1.0 / ec[sel, k]
            if w.size:
                V_ipw[k, j] = float(np.sum(w * Y[sel]) / np.sum(w))
                ess[k, j] = float(np.sum(w) ** 2 / np.sum(w ** 2))
            if mhat is not None:
                aug = np.zeros(nj)
                aug[asg[mj] == k] = (Y[sel] - mhat[sel, k]) / ec[sel, k]
                V_dr[k, j] = float(np.mean(mhat[mj, k] + aug))
                V_reg[k, j] = float(np.mean(mhat[mj, k]))
    return V_ipw, np.clip(V_dr, 0.0, 1.0), V_reg, ess


def prop_quality(e_est, e_true, asg):
    """Multiclass log-loss, Brier, calibration error, and agreement with the exact truth."""
    p_asg = e_est[np.arange(len(asg)), asg]
    ll = float(-np.mean(np.log(np.clip(p_asg, 1e-12, None))))
    onehot = np.zeros_like(e_est)
    onehot[np.arange(len(asg)), asg] = 1.0
    brier = float(np.mean(np.sum((e_est - onehot) ** 2, axis=1)))
    # reliability in ten bins over the assigned-model probability
    bins = np.clip((p_asg * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.sum():
            ece += m.mean() * abs(p_asg[m].mean() - 1.0)          # placeholder replaced below
    # proper ECE: confidence vs empirical accuracy of the argmax class
    conf = e_est.max(1)
    correct = (e_est.argmax(1) == asg).astype(float)
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = float(sum((bins == b).mean() * abs(conf[bins == b].mean() - correct[bins == b].mean())
                    for b in range(10) if (bins == b).sum()))
    mae_vs_true = float(np.abs(e_est - e_true).mean())
    return dict(log_loss=ll, brier=brier, ece=ece, mae_vs_true=mae_vs_true,
                min_est_propensity=float(e_est.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(RES, "tae_causal_baselines.json"))
    args = ap.parse_args()

    pub = json.load(open(os.path.join(RES, "tae_validity.json")))["rows"]
    EMB = load_raw("llmrouterbench")["E"]

    t0 = time.time()
    rows, diags = [], []
    for p in range(args.partitions):
        base = build_base(p)
        cellsA1, n_kj, log_data = A.make_selective_log(base, p, eps=EPS)
        cellsA0 = A.make_matched_common_population_log(base, p, n_kj)
        supp = A.support(n_kj, base, n_read=N_READ)
        tab = MT.scoring_tables(base, supp)
        pol_ref = MT.target_policy(tab, supp)

        e_true = true_propensities(base, log_data)
        e_est = estimated_propensities(EMB, log_data, seed=1000 + p)
        pq = prop_quality(e_est, e_true, log_data["assign"])
        pq["partition"] = p
        diags.append(pq)

        for cond in ("D0", "D1"):
            Cs = EV.distortions(base, cond)

            # ---------------- reproduction gate: trust must match the published rows
            surfA1 = EV.build_surfaces(base, cellsA1, Cs, N_READ)
            mine = MT.reference_regret(MT.deployed_policy(surfA1["M"], supp), tab, supp, pol_ref)
            ref = [r["reference_regret"] for r in pub
                   if r["partition"] == p and r["assignment"] == "A1"
                   and r["evaluator"] == cond and r["method"] == "trust" and r["budget"] == 0.35]
            if not ref or abs(mine - ref[0]) > 1e-6:
                sys.exit(f"REPRODUCTION GATE FAILED p={p} {cond}: {mine} vs {ref}")

            # ---------------- estimators on the A1 log
            Y = observed_scores(base, log_data, Cs)
            mhat = outcome_model(EMB, base["K"], log_data, Y, seed=2000 + p)
            for prop_name, e in (("true", e_true), ("est", e_est)):
                for clip in (CLIPS if prop_name == "est" else (0.0,)):
                    V_ipw, V_dr, V_reg, ess = cell_surfaces(base, log_data, Y, e, mhat, clip)
                    arms = {f"ipw_{prop_name}": V_ipw, f"dr_{prop_name}": V_dr}
                    if prop_name == "true" and clip == 0.0:
                        arms["reg"] = V_reg
                    for name, V in arms.items():
                        pol = MT.deployed_policy(V, supp)
                        rows.append(dict(partition=p, assignment="A1", evaluator=cond,
                                         method=name, clip=clip,
                                         reference_regret=MT.reference_regret(pol, tab, supp, pol_ref),
                                         top1_error=MT.top1_error(pol, tab, supp, pol_ref),
                                         min_ess=float(np.nanmin(np.where(supp, ess, np.nan))),
                                         median_ess=float(np.nanmedian(np.where(supp, ess, np.nan)))))
        print(f"  partition {p:2d} done  [{time.time()-t0:.0f}s]", flush=True)

    out = dict(spec=dict(eps=EPS, n_read=N_READ, folds=FOLDS, clips=list(CLIPS),
                         primary_clip=PRIMARY_CLIP, partitions=args.partitions,
                         primary_contrast="CASE vs dr_true, A1-D0, reference_regret",
                         generated_by="code/run_causal_baselines.py"),
               rows=rows, propensity_diagnostics=diags)
    json.dump(out, open(args.out, "w"), separators=(",", ":"))
    print(f"wrote {args.out} ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
