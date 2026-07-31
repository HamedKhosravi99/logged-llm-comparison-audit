"""Priority 2 (section 3): out-of-sample audit / no-audit rule.

The paper's threshold 1.17 was chosen in sample. Here the threshold is chosen on LLMRouterBench by
leave-one-partition-out cross-validation, optimising the expected target-value loss of the audit /
no-audit policy rather than classification accuracy, then frozen and applied unchanged to
RouterBench. No new API calls.
"""
import collections, json, os
import numpy as np

import case_core as cc
from eval_validity import assignment as A
from eval_validity import evaluators as EV
from eval_validity import metrics as MT
from eval_validity.grid import build_base
from run_tae_diagnostics import _seed_reads

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "p2_diagnostic_oos.json")


def per_partition(corpus, n_parts, acq_seeds=6, n_read=30, eps=0.15, budget=0.35):
    """For each partition and each of A0/A1: the seed statistic and the loss of auditing vs not."""
    rows = []
    for p in range(n_parts):
        base = build_base(p, corpus=corpus)
        c1, n_kj, _ = A.make_selective_log(base, p, eps=eps)
        c0 = A.make_matched_common_population_log(base, p, n_kj)
        supp = A.support(n_kj, base, n_read=n_read)
        tab = MT.scoring_tables(base, supp)
        pref = MT.target_policy(tab, supp)
        B = max(base["K"], int(round(budget * int(supp.sum()))))
        for aname, cells in (("A0", c0), ("A1", c1)):
            surf = EV.build_surfaces(base, cells, EV.distortions(base, "D0"), n_read)
            ss, rc, rt = [], [], []
            for s in range(acq_seeds):
                sd = 10_000 * p + 100 * int(budget * 100) + s
                ss.append(_seed_reads(base, cells, surf, supp, sd, n_read))
                V, _ = cc.allocate("case", None, surf["M"], surf["cellq"], supp, B, sd, n_read)
                rc.append(MT.reference_regret(MT.deployed_policy(V, supp), tab, supp, pref))
                rt.append(MT.reference_regret(MT.deployed_policy(surf["M"], supp), tab, supp, pref))
            rows.append(dict(corpus=corpus, partition=p, arm=aname,
                             stat=float(np.mean(ss)), loss_audit=float(np.mean(rc)),
                             loss_no_audit=float(np.mean(rt))))
        print(f"    {corpus} partition {p+1}/{n_parts}", flush=True)
    return rows


def pick_threshold(rows):
    """Threshold minimising the expected loss of the resulting audit / no-audit policy."""
    stats = np.array([r["stat"] for r in rows])
    grid = np.linspace(stats.min(), stats.max(), 400)
    best, best_t = np.inf, float(grid[0])
    for t in grid:
        loss = np.mean([r["loss_audit"] if r["stat"] >= t else r["loss_no_audit"] for r in rows])
        if loss < best:
            best, best_t = loss, float(t)
    return best_t


def evaluate(rows, tau):
    followed = np.mean([r["loss_audit"] if r["stat"] >= tau else r["loss_no_audit"] for r in rows])
    always = np.mean([r["loss_audit"] for r in rows])
    never = np.mean([r["loss_no_audit"] for r in rows])
    oracle = np.mean([min(r["loss_audit"], r["loss_no_audit"]) for r in rows])
    a1 = [r for r in rows if r["arm"] == "A1"]; a0 = [r for r in rows if r["arm"] == "A0"]
    sens = float(np.mean([r["stat"] >= tau for r in a1])) if a1 else np.nan
    spec = float(np.mean([r["stat"] < tau for r in a0])) if a0 else np.nan
    auc = float(np.mean([(x["stat"] < y["stat"]) + 0.5 * (x["stat"] == y["stat"])
                         for x in a0 for y in a1])) if a0 and a1 else np.nan
    wrong = float(np.mean([(r["stat"] >= tau) != (r["loss_audit"] < r["loss_no_audit"]) for r in rows]))
    unnecessary = float(np.mean([max(0.0, r["loss_audit"] - r["loss_no_audit"])
                                 for r in rows if r["stat"] >= tau])) if any(r["stat"] >= tau for r in rows) else 0.0
    missed = float(np.mean([max(0.0, r["loss_no_audit"] - r["loss_audit"])
                            for r in rows if r["stat"] < tau])) if any(r["stat"] < tau for r in rows) else 0.0
    return dict(threshold=tau, auc=auc, sensitivity=sens, specificity=spec,
                mean_loss_rule=float(followed), mean_loss_always=float(always),
                mean_loss_never=float(never), mean_loss_oracle=float(oracle),
                loss_vs_oracle=float(followed - oracle), wrong_audit_decision_rate=wrong,
                unnecessary_audit_loss=unnecessary, missed_improvement_loss=missed,
                beats_always=bool(followed < always), beats_never=bool(followed < never))


def main():
    print("  training corpus: LLMRouterBench")
    tr = per_partition("llmrouterbench", 30)
    parts = sorted({r["partition"] for r in tr})
    # leave-one-partition-out CV
    lopo = []
    for p in parts:
        tau = pick_threshold([r for r in tr if r["partition"] != p])
        held = [r for r in tr if r["partition"] == p]
        lopo += [dict(r, tau=tau) for r in held]
    cv_loss = float(np.mean([r["loss_audit"] if r["stat"] >= r["tau"] else r["loss_no_audit"] for r in lopo]))
    tau_frozen = pick_threshold(tr)
    print(f"  LOPO-CV loss {cv_loss:.4f}; frozen threshold {tau_frozen:.3f}")

    print("  external test corpus: RouterBench (threshold applied unchanged, no refit)")
    te = per_partition("routerbench", 20)

    out = dict(
        frozen_threshold=tau_frozen, lopo_cv_mean_loss=cv_loss,
        original_in_sample_threshold=1.17,
        train_llmrouterbench=evaluate(tr, tau_frozen),
        train_in_sample_1p17=evaluate(tr, 1.17),
        test_routerbench_frozen=evaluate(te, tau_frozen),
        test_routerbench_refit_would_be=evaluate(te, pick_threshold(te)),
        rows=tr + te)
    json.dump(out, open(OUT, "w"), indent=1)
    t = out["test_routerbench_frozen"]
    print(f"\n  RouterBench, frozen threshold {tau_frozen:.3f}:")
    print(f"    rule {t['mean_loss_rule']:.4f} | always {t['mean_loss_always']:.4f} | "
          f"never {t['mean_loss_never']:.4f} | oracle {t['mean_loss_oracle']:.4f}")
    print(f"    AUC {t['auc']:.3f}  sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}")
    verdict = ("deployment-ready rule" if t["beats_always"] and t["beats_never"]
               else "a diagnostic signal rather than a deployment-ready rule")
    print(f"    verdict: {verdict}")
    print(f"  wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
