"""One partition's worth of the 2x2 validity experiment, plus the deployment-gate comparison.

The core loop is deliberately paired: a single prompt-cohort partition fixes the contexts, the
incumbent router, the cohorts and the evaluator-design constants; the selective log is drawn once
and the common-population log is matched to its per-cell counts; every evaluator condition, budget,
acquisition seed and method then re-uses exactly those objects, so any difference between arms is
attributable to the factor that was changed.
"""
import numpy as np
import case_core as cc

from . import assignment as A
from . import evaluators as EV
from . import gates as G
from . import metrics as MT
from . import oracles as OR
from . import validation as VAL
from .grid import build_base

BUDGET_FRACS = (0.15, 0.25, 0.35)
METHODS = ("trust", "uniform", "case", "epig_corr", "oracle_sel", "oracle_ref")
ALLOC_MODE = {"uniform": "uniform", "case": "case", "epig_corr": "epig_deconf",
              "rawmargin": "rawmargin", "epig_obs": "epig"}


def _budget(frac, supp, K):
    return max(K, int(round(frac * int(supp.sum()))))


def _verification_cells(B, mv_frac=0.30):
    """Verification cells taken OUT of the same total budget, kept even so checks are paired."""
    return 2 * max(1, int(round(mv_frac * B / 2)))


def surfaces_for(base, cells, cond, lam_scale=1.0, n_read=30):
    Cs = EV.distortions(base, cond, lam_scale)
    surf = EV.build_surfaces(base, cells, Cs, n_read)
    surf["Cs"] = Cs
    return surf


def _method_surface(name, surf, supp, B, seed, n_read):
    """Deployed surface for one method. Only the two labelled oracles read scoring outcomes."""
    if name == "trust":
        return surf["M"].copy(), 0
    if name == "oracle_sel":
        return OR.oracle_selection_corrected_surface(surf), 0
    if name == "oracle_ref":
        return OR.reference_outcome_oracle_surface(surf), 0
    V, used = cc.allocate(ALLOC_MODE[name], None, surf["M"], surf["cellq"], supp, B, seed, n_read)
    return V, used


def run_partition(part_seed, acq_seeds=4, eps=0.15, n_read=30, lam_scale=1.0, corpus="llmrouterbench",
                  budget_fracs=BUDGET_FRACS, methods=METHODS, gate_conds=("E1", "E2"),
                  factorial_conds=("E0", "E1"), extra_methods=(), baseline="logged", n_clusters=8):
    """Run every (assignment, evaluator, budget, acquisition-seed, method) cell for one partition."""
    base = build_base(part_seed, n_clusters=n_clusters, corpus=corpus)
    K, J = base["K"], base["J"]
    cellsA1, n_kj, log_data = A.make_selective_log(base, part_seed, eps=eps)
    cellsA0 = A.make_matched_common_population_log(base, part_seed, n_kj)
    supp = A.support(n_kj, base, n_read=n_read)
    tab = MT.scoring_tables(base, supp)
    pol_ref = MT.target_policy(tab, supp)
    cells = {"A0": cellsA0, "A1": cellsA1}
    use_methods = tuple(methods) + tuple(extra_methods)

    rows, gate_rows = [], []
    for cond in factorial_conds:
        for aname in ("A0", "A1"):
            surf = surfaces_for(base, cells[aname], cond, lam_scale, n_read)
            log_pol = MT.deployed_policy(surf["M"], supp)
            reversal = MT.top1_error(log_pol, tab, supp, pol_ref)
            for frac in budget_fracs:
                B = _budget(frac, supp, K)
                for s in range(acq_seeds):
                    seed = 10_000 * part_seed + 100 * int(frac * 100) + s
                    for m in use_methods:
                        V, used = _method_surface(m, surf, supp, B, seed, n_read)
                        pol = MT.deployed_policy(V, supp)
                        row = dict(partition=int(part_seed), acq_seed=int(s), assignment=aname,
                                   evaluator=cond, budget=float(frac), B=int(B), method=m,
                                   cells_read=int(used),
                                   reference_regret=MT.reference_regret(pol, tab, supp, pol_ref),
                                   operational_regret=MT.operational_regret(pol, surf["Ef_op"], supp),
                                   top1_error=MT.top1_error(pol, tab, supp, pol_ref),
                                   ranking_reversal_rate=reversal,
                                   deployment_value=MT.deployment_value(pol, tab))
                        row["slice_regret"] = MT.reference_regret_by_slice(pol, tab, supp, pol_ref)
                        rows.append(row)

    # ---- deployment decisions: ungated vs same-evaluator vs reference-channel check -----------
    for cond in gate_conds:
        surf = surfaces_for(base, cellsA1, cond, lam_scale, n_read)
        pol_base = G.baseline_policy(base, surf, supp, kind=baseline)
        for frac in budget_fracs:
            B = _budget(frac, supp, K)
            mv = _verification_cells(B)
            for s in range(acq_seeds):
                seed = 10_000 * part_seed + 100 * int(frac * 100) + s
                V_ung, _ = cc.allocate("case", None, surf["M"], surf["cellq"], supp, B, seed, n_read)
                pol_ung = MT.deployed_policy(V_ung, supp)
                V_g, _ = cc.allocate("case", None, surf["M"], surf["cellq"], supp, B - mv, seed, n_read)
                pol_cand = MT.deployed_policy(V_g, supp)
                arms = {
                    "ungated": (pol_ung, G.ungated_candidate(pol_ung)),
                    "same_evaluator": (pol_cand, G.same_evaluator_decision_check(
                        pol_base, pol_cand, surf, supp, mv, seed, n_read)),
                    "reference_channel": (pol_cand, G.reference_channel_decision_check(
                        pol_base, pol_cand, surf, supp, mv, seed, n_read)),
                }
                contested = arms["same_evaluator"][1].get("contested", [])
                vv = G.channel_verdict_validity(pol_base, pol_cand, surf, contested)
                for arm, (pc, dec) in arms.items():
                    deployed = pc if dec["deploy_candidate"] else pol_base
                    out = MT.switch_outcomes(pol_base, pc, dec["deploy_candidate"], tab)
                    gate_rows.append(dict(
                        channel_verdict_inversion=vv.get(arm, 0.0),
                        n_contested_scored=int(vv.get(arm + "_n", 0)),
                        partition=int(part_seed), acq_seed=int(s), evaluator=cond,
                        budget=float(frac), B=int(B), arm=arm,
                        mv=0 if arm == "ungated" else int(mv),
                        acq_cells=int(B if arm == "ungated" else B - mv),
                        ver_cells_used=int(dec["n_ver_cells"]),
                        n_contested=int(dec.get("n_contested", 0)),
                        reference_regret=MT.reference_regret(deployed, tab, supp, pol_ref),
                        top1_error=MT.top1_error(deployed, tab, supp, pol_ref),
                        false_switch=out["false_switch"], false_retention=out["false_retention"],
                        switch_rate=out["switch"], deployment_value=MT.deployment_value(deployed, tab),
                        baseline_value=out["v_base"], candidate_value=out["v_cand"]))

    meta = dict(partition=int(part_seed), corpus=corpus, K=K, J=J, n_supported=int(supp.sum()),
                lam=float(base["lam"]), z=[float(v) for v in base["z"]],
                s=[float(v) for v in base["s"]], eps=float(eps), n_read=int(n_read),
                n_kj=n_kj.tolist(), target_policy=[int(v) for v in pol_ref],
                incumbent_router=[int(v) for v in base["inc"]],
                models=list(base["models"]),
                diagnostics=condition_diagnostics(base, cellsA0, cellsA1, supp, tab, pol_ref,
                                                  lam_scale, n_read))
    return rows, gate_rows, meta


def condition_diagnostics(base, cellsA0, cellsA1, supp, tab, pol_ref, lam_scale=1.0, n_read=30):
    """Design diagnostics for one partition: trap strength and realised evaluator distortion."""
    s0 = surfaces_for(base, cellsA1, "E0", lam_scale, n_read)
    s1 = surfaces_for(base, cellsA1, "E1", lam_scale, n_read)
    a0 = surfaces_for(base, cellsA0, "E0", lam_scale, n_read)
    Bias = s0["M"] - s0["Ef_ref"]
    bk = Bias.mean(1)
    Ceff = s1["Ef_op"] - s1["Ef_ref"]
    nominal = s1["Cs"]["C"]
    return dict(
        log_misrank_A1=MT.top1_error(MT.deployed_policy(s0["M"], supp), tab, supp, pol_ref),
        log_misrank_A0=MT.top1_error(MT.deployed_policy(a0["M"], supp), tab, supp, pol_ref),
        bias_energy=float(np.linalg.norm(Bias[supp]) / np.linalg.norm(s0["Ef_ref"][supp])),
        per_action_residual=float(((Bias - bk[:, None]) ** 2)[supp].sum() / (Bias ** 2)[supp].sum()),
        mean_abs_effective_C=float(np.abs(Ceff[supp]).mean()),
        mean_abs_nominal_C=float(np.abs(nominal[supp]).mean()),
        clipping_attenuation=float(np.abs(Ceff[supp]).mean() / max(np.abs(nominal[supp]).mean(), 1e-12)),
        reference_ranking_reversal_rate=MT.top1_error(
            MT.deployed_policy(s1["Ef_op"], supp), tab, supp, pol_ref),
        **{f"assignment_invariance_{k}": v
           for k, v in VAL.assignment_invariance(base, cellsA1, cellsA0, supp,
                                                 n_read=n_read).items() if k != "pass_"},
        median_target_margin=float(np.median(
            [np.sort(s1["Ef_ref"][np.where(supp[:, j])[0], j])[::-1][:2].ptp()
             for j in range(base["J"]) if supp[:, j].sum() >= 2])))
