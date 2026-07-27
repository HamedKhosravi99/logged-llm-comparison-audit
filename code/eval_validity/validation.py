"""Leakage and validity tests. Every one of these must pass before a result file is trusted.

  1. cohort isolation          the five prompt cohorts are pairwise disjoint and exhaustive;
  2. assignment matching       a paired A0/A1 run has identical per-cell counts, identical total
                               log volume and identical evaluator distortion -- only the assignment
                               distribution differs;
  3. reference-channel leakage replacing the scoring cohort's target outcomes with random values
                               leaves the acquired cells, the fitted correction, the candidate
                               policy and the same-evaluator gate decision bit-identical;
  4. stable-evaluator algebra   with C^o = C^i the evaluator distortion cancels from the estimated
                               logged-minus-randomized difference but survives in the oracle
                               selection-corrected surface;
  5. assignment invariance     Prop. 1 needs the evaluator's mean error on a cell's ROUTED
                               sub-population to match its error on the context population; clipping
                               makes that only approximate, so the residual is measured, not assumed;
  6. stream-mismatch algebra    with C^i = 0.5 C^o the estimated cellwise difference gains the
                               predicted extra term aligned with 0.5 C^o.
"""
import numpy as np
import case_core as cc

from . import assignment as A
from . import evaluators as EV
from . import gates as G
from . import metrics as MT
from .grid import build_base


def cohort_isolation(base):
    coh, N = base["coh"], len(base["coh"])
    sets = {c: set(np.where(coh == c)[0].tolist()) for c in ["train", "log", "acq", "ver", "score"]}
    names = list(sets)
    inter = {f"{a}|{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return dict(pass_=all(v == 0 for v in inter.values()) and sum(map(len, sets.values())) == N,
                intersections=inter, sizes={k: len(v) for k, v in sets.items()}, n_prompts=int(N))


def assignment_matching(base, cellsA1, cellsA0, n_kj):
    K, J = base["K"], base["J"]
    same_counts = all(len(cellsA0[(k, j)]) == n_kj[k, j] for k in range(K) for j in range(J))
    tot1 = sum(len(cellsA1[(k, j)]) for k in range(K) for j in range(J))
    tot0 = sum(len(cellsA0[(k, j)]) for k in range(K) for j in range(J))
    # every A0 cell must draw from the same context's log cohort
    ok_ctx = all(np.all(base["ctx"][cellsA0[(k, j)]] == j) and
                 np.all(base["coh"][cellsA0[(k, j)]] == "log")
                 for k in range(K) for j in range(J) if len(cellsA0[(k, j)]))
    return dict(pass_=bool(same_counts and tot0 == tot1 and ok_ctx),
                total_log_A1=int(tot1), total_log_A0=int(tot0), per_cell_counts_match=bool(same_counts),
                cohort_and_context_respected=bool(ok_ctx))


def reference_channel_leakage(base, cells, supp, cond="E1", B=14, seed=0, n_read=30):
    """Scramble the scoring cohort's target outcomes; nothing an audit method does may change."""
    def artefacts(basex):
        surf = EV.build_surfaces(basex, cells, EV.distortions(basex, cond), n_read)
        V, used = cc.allocate("case", None, surf["M"], surf["cellq"], supp, B, seed, n_read)
        pol_base = G.baseline_policy(basex, surf, supp)
        pol_cand = MT.deployed_policy(V, supp)
        dec = G.same_evaluator_decision_check(pol_base, pol_cand, surf, supp, 4, seed, n_read)
        return dict(V=V, used=used, pol=pol_cand, deploy=dec["deploy_candidate"],
                    vhat=(dec["v_hat_cand"], dec["v_hat_base"]), M=surf["M"], cellq=surf["cellq"])
    a = artefacts(base)
    rng = np.random.default_rng(12345)
    scrambled = dict(base)
    Q2 = base["Q"].copy()
    sc = np.where(base["coh"] == "score")[0]
    Q2[sc] = rng.random((len(sc), base["K"]))            # arbitrary values on the scoring cohort
    scrambled["Q"] = Q2
    b = artefacts(scrambled)
    return dict(pass_=bool(np.allclose(a["V"], b["V"]) and a["used"] == b["used"]
                           and np.array_equal(a["pol"], b["pol"]) and a["deploy"] == b["deploy"]
                           and np.allclose(a["M"], b["M"])),
                max_surface_diff=float(np.max(np.abs(a["V"] - b["V"]))),
                policy_identical=bool(np.array_equal(a["pol"], b["pol"])),
                gate_decision_identical=bool(a["deploy"] == b["deploy"]))


def effective_distortion(base, cells, cond="E1", n_read=30):
    """C^eff_kj = mu^op_kj - mu^ref_kj, the distortion that actually reaches the cell means.

    The nominal distortion lambda*z_k*s_j is attenuated because the target outcomes are (near-)binary
    and the operational score is clipped to [0,1]: adding c to a prompt already scored 1 changes
    nothing. C^eff is the quantity every downstream claim should be scaled against.
    """
    surf = EV.build_surfaces(base, cells, EV.distortions(base, cond), n_read)
    return surf["Ef_op"] - surf["Ef_ref"], surf


def stable_evaluator_algebra(base, cellsA1, cellsA0, supp, n_read=30):
    """C^o = C^i: distortion cancels from logged-minus-randomized, survives in the corrected surface."""
    Cs1 = EV.distortions(base, "E1")
    Cs0 = EV.distortions(base, "E0")
    K, J = base["K"], base["J"]
    out = {}
    for aname, cells in (("A0", cellsA0), ("A1", cellsA1)):
        s1 = EV.build_surfaces(base, cells, Cs1, n_read)
        s0 = EV.build_surfaces(base, cells, Cs0, n_read)
        # cell-mean randomized readout of the acquisition pool (population version, no sampling noise)
        R1 = np.array([[s1["cellq"][(k, j)].mean() for j in range(J)] for k in range(K)])
        R0 = np.array([[s0["cellq"][(k, j)].mean() for j in range(J)] for k in range(K)])
        d1, d0 = s1["M"] - R1, s0["M"] - R0
        Ceff = s1["Ef_op"] - s1["Ef_ref"]
        pol_ref = s1["Ef_ref"].argmax(0)
        pol_op = s1["Ef_op"].argmax(0)
        out[aname] = dict(
            mean_abs_diff_of_differences=float(np.abs((d1 - d0)[supp]).mean()),
            mean_abs_difference_E0=float(np.abs(d0[supp]).mean()),
            mean_abs_effective_C=float(np.abs(Ceff[supp]).mean()),
            mean_abs_nominal_C=float(np.abs(Cs1["C"][supp]).mean()),
            clipping_attenuation=float(np.abs(Ceff[supp]).mean() / np.abs(Cs1["C"][supp]).mean()),
            reference_ranking_reversal_rate=float(np.mean(pol_ref != pol_op)))
    out["pass_"] = bool(
        out["A1"]["mean_abs_diff_of_differences"] < 0.25 * out["A1"]["mean_abs_effective_C"] and
        out["A0"]["mean_abs_diff_of_differences"] < 0.25 * out["A0"]["mean_abs_effective_C"] and
        out["A1"]["mean_abs_effective_C"] > 0.0)
    return out


def assignment_invariance(base, cellsA1, cellsA0, supp, cond="E1", n_read=30):
    """How assignment-invariant is the REALISED evaluator distortion?

    Prop. 1's cancellation needs more than "the same judge scored both streams": it needs the
    judge's mean error to be the same on the routed sub-population of a cell as on the context
    population, i.e. C^{o,sel}_kj = C^{i,pop}_kj where

        C^{o,sel}_kj = E[Ytilde^o(k) - Yref(k) | A=k, Z=j],
        C^{i,pop}_kj = E[Ytilde^i(k) - Yref(k) | Z=j].

    Clipping to [0,1] makes the per-prompt shift depend on the prompt's own target outcome, so a
    router that selects on difficulty breaks exact invariance even under an identical evaluator.
    This measures the residual, and reports it against the selection bias it is meant to identify.
    The A0 column is the same quantity with the routing removed -- it is the cohort-sampling floor.
    """
    Q, ctx, coh, K, J = base["Q"], base["ctx"], base["coh"], base["K"], base["J"]
    C = EV.distortions(base, cond)["C"]
    shift = lambda idx, k, j: float(np.mean(np.clip(Q[idx, k] + C[k, j], 0, 1) - Q[idx, k]))
    o_sel = np.zeros((K, J)); o_sel0 = np.zeros((K, J))
    i_pop = np.zeros((K, J)); b_sel = np.zeros((K, J))
    for j in range(J):
        ac = np.where((coh == "acq") & (ctx == j))[0]
        sc = np.where((coh == "score") & (ctx == j))[0]
        for k in range(K):
            lg, lg0 = cellsA1[(k, j)], cellsA0[(k, j)]
            i_pop[k, j] = shift(ac, k, j)
            if len(lg):
                o_sel[k, j] = shift(lg, k, j)
                b_sel[k, j] = float(Q[lg, k].mean() - Q[sc, k].mean())
            if len(lg0):
                o_sel0[k, j] = shift(lg0, k, j)
    resid = float(np.abs((o_sel - i_pop)[supp]).mean())
    floor = float(np.abs((o_sel0 - i_pop)[supp]).mean())
    bs = float(np.abs(b_sel[supp]).mean())
    ce = float(np.abs(i_pop[supp]).mean())
    return dict(pass_=bool(resid < 0.25 * bs), residual_A1=resid, cohort_floor_A0=floor,
                mean_abs_selection_bias=bs, mean_abs_effective_C=ce,
                residual_over_selection_bias=resid / bs, residual_over_effective_C=resid / ce)


def stream_mismatch_algebra(base, cellsA1, supp, n_read=30):
    """C^i = 0.5 C^o: the estimated cellwise difference must gain an extra term aligned with C^o/2.

    Clipping attenuates the nominal 0.5*C, so the test is on alignment and on a slope inside the
    attenuation range rather than on exact equality with the nominal term.
    """
    K, J = base["K"], base["J"]
    s1 = EV.build_surfaces(base, cellsA1, EV.distortions(base, "E1"), n_read)
    s2 = EV.build_surfaces(base, cellsA1, EV.distortions(base, "E2"), n_read)
    R = lambda s: np.array([[s["cellq"][(k, j)].mean() for j in range(J)] for k in range(K)])
    extra = (s2["M"] - R(s2)) - (s1["M"] - R(s1))
    pred = 0.5 * EV.distortions(base, "E1")["C"]
    x, y = pred[supp].ravel(), extra[supp].ravel()
    slope = float((x @ y) / (x @ x))
    corr = float(np.corrcoef(x, y)[0, 1])
    return dict(pass_=bool(corr > 0.9 and 0.2 < slope <= 1.05),
                corr=corr, slope_vs_nominal=slope,
                mean_abs_extra=float(np.abs(y).mean()),
                mean_abs_nominal=float(np.abs(x).mean()))


def run_all(part_seed=0, eps=0.15, n_read=30, verbose=True):
    base = build_base(part_seed)
    cellsA1, n_kj, _ = A.make_selective_log(base, part_seed, eps=eps)
    cellsA0 = A.make_matched_common_population_log(base, part_seed, n_kj)
    supp = A.support(n_kj, base, n_read=n_read)
    res = dict(
        cohort_isolation=cohort_isolation(base),
        assignment_matching=assignment_matching(base, cellsA1, cellsA0, n_kj),
        reference_channel_leakage=reference_channel_leakage(base, cellsA1, supp, n_read=n_read),
        stable_evaluator_algebra=stable_evaluator_algebra(base, cellsA1, cellsA0, supp, n_read),
        assignment_invariance=assignment_invariance(base, cellsA1, cellsA0, supp, n_read=n_read),
        stream_mismatch_algebra=stream_mismatch_algebra(base, cellsA1, supp, n_read))
    res["all_pass"] = all(v["pass_"] for v in res.values() if isinstance(v, dict))
    if verbose:
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"[{'PASS' if v['pass_'] else 'FAIL'}] {k}")
                for kk, vv in v.items():
                    if kk != "pass_":
                        print(f"        {kk}: {vv}")
        print(f"\nALL PASS: {res['all_pass']}")
    return res


if __name__ == "__main__":
    run_all()
