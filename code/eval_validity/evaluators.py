"""Factor E: what the operational evaluator measures, and the surfaces each condition produces.

Every prompt i carries a designated target outcome Y^ref_i(k) (the released benchmark score). An
operational evaluator on stream s in {o (logging), i (randomized audit), v (verification)} reports

    Ytilde^s_i(k) = clip( Y^ref_i(k) + C^s_kj , 0, 1 ),

with C^s_kj a systematic model-by-context distortion; finite-sample evaluator noise enters through
the finite readout size, exactly as in the audit protocol itself.

    E0  construct-aligned         C^o = C^i = C^v = 0
    E1  stable construct-biased   C^o = C^i = C^v = C,      C_kj = lambda * z_k * s_j
    E2  evaluator-stream mismatch C^o = C, C^i = 0.5 C, same-evaluator verification C^v = C

z_k (standardized model capability rank), s_j (context difficulty sign) and lambda are fixed on the
router-train cohort by `grid.build_base` before any run; nothing here is tuned on results.
"""
import numpy as np

CONDITIONS = ("E0", "E1", "E1k", "E2", "E2rev")


def distortions(base, cond, lam_scale=1.0):
    """Per-stream systematic distortion matrices C^o, C^i, C^v (same-evaluator) for one condition.

    The reference channel is by definition undistorted, so it needs no matrix.
    """
    K, J = base["K"], base["J"]
    Z = np.zeros((K, J))
    C = lam_scale * base["lam"] * base["z"][:, None] * base["s"][None, :]
    if cond == "E0":
        return dict(o=Z, i=Z, v=Z, C=Z)
    if cond == "E1":
        return dict(o=C, i=C, v=C, C=C)
    if cond == "E1k":                                    # context-invariant model favouritism C_kj = C_k
        Ck = np.repeat((lam_scale * base["lam"] * base["z"])[:, None], J, axis=1)
        return dict(o=Ck, i=Ck, v=Ck, C=Ck)
    if cond == "E2":
        return dict(o=C, i=0.5 * C, v=C, C=C)
    if cond == "E2rev":                                  # appendix: sign-reversed audit stream
        return dict(o=C, i=-0.5 * C, v=C, C=C)
    raise ValueError(f"unknown evaluator condition {cond!r}")


def _channel(Q, idx, k, c):
    """Operational scores of model k on prompts `idx` under a scalar cell distortion c."""
    return np.clip(Q[idx, k] + c, 0.0, 1.0)


def build_surfaces(base, cells, Cs, n_read=30):
    """All surfaces a run needs, given an assignment (`cells`) and a distortion set (`Cs`).

    Returns
      M         [K,J] logged operational evaluation, E[Ytilde^o | A=k, Jctx=j] on the log cohort
      cellq     per-cell pools of Ytilde^i on the acquisition cohort (randomized audit readouts)
      verq      per-cell pools of Ytilde^v on the verification cohort (same-evaluator gate)
      verq_ref  per-cell pools of Y^ref on the verification cohort (reference-channel gate ONLY)
      Ef_ref    [K,J] mu^ref, the designated target surface on the untouched scoring cohort
      Ef_op     [K,J] mu^op, the common-population OPERATIONAL surface on the scoring cohort
                      (= what a perfect assignment-validity repair can recover; it retains C^o)
    """
    Q, ctx, coh, K, J = base["Q"], base["ctx"], base["coh"], base["K"], base["J"]
    M = np.zeros((K, J))
    Ef_ref = np.zeros((K, J))
    Ef_op = np.zeros((K, J))
    cellq, verq, verq_ref = {}, {}, {}
    for j in range(J):
        ac = np.where((coh == "acq") & (ctx == j))[0]
        vr = np.where((coh == "ver") & (ctx == j))[0]
        sc = np.where((coh == "score") & (ctx == j))[0]
        for k in range(K):
            lg = cells[(k, j)]
            M[k, j] = float(_channel(Q, lg, k, Cs["o"][k, j]).mean()) if len(lg) else np.nan
            cellq[(k, j)] = _channel(Q, ac, k, Cs["i"][k, j])
            verq[(k, j)] = _channel(Q, vr, k, Cs["v"][k, j])
            verq_ref[(k, j)] = Q[vr, k].astype(float)
            Ef_ref[k, j] = float(Q[sc, k].mean())
            Ef_op[k, j] = float(_channel(Q, sc, k, Cs["o"][k, j]).mean())
    # unsupported cells never enter a decision; fill so arithmetic stays finite
    M = np.where(np.isnan(M), Ef_ref, M)
    return dict(M=M, cellq=cellq, verq=verq, verq_ref=verq_ref, Ef_ref=Ef_ref, Ef_op=Ef_op)
