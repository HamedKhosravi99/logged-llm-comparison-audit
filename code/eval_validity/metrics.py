"""Outcomes. Two confirmatory primaries (reference-channel deployment regret, false-switch rate)
plus a small set of descriptive secondaries.

Reference-channel deployment regret of a deployed policy pihat, relative to the target policy
pi^ref(j) = argmax_k mu^ref_kj:

    R^ref(pihat) = (1/J) sum_j [ mu^ref_{pi^ref(j), j} - mu^ref_{pihat(j), j} ].

The same formula restricted to one benchmark slice replaces mu^ref_kj by the slice's own scoring
prompts in context j, keeping the pooled target policy pi^ref fixed -- so the reference-outcome
oracle scores exactly zero on the macro average and on every slice.
"""
import numpy as np

MIN_SLICE_CELL = 10          # scoring prompts of a slice needed in a context to score that context


def deployed_policy(V, supp):
    """pihat(j) = argmax over supported models of the deployed surface V (-1 if no support)."""
    K, J = V.shape
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if len(ks):
            pol[j] = int(ks[int(np.argmax(V[ks, j]))])
    return pol


def scoring_tables(base, supp):
    """Per-context cell means of the designated target outcome, pooled and per benchmark slice."""
    Q, ctx, coh, ds, K, J = base["Q"], base["ctx"], base["coh"], base["ds"], base["K"], base["J"]
    sc = coh == "score"
    mu = np.zeros((K, J))
    n = np.zeros(J, int)
    for j in range(J):
        m = np.where(sc & (ctx == j))[0]
        n[j] = len(m)
        mu[:, j] = Q[m].mean(0) if len(m) else np.nan
    slices = {}
    for d in np.unique(ds):
        mud = np.full((K, J), np.nan)
        ok = np.zeros(J, bool)
        for j in range(J):
            m = np.where(sc & (ctx == j) & (ds == d))[0]
            if len(m) >= MIN_SLICE_CELL:
                mud[:, j] = Q[m].mean(0)
                ok[j] = True
        slices[str(d)] = dict(mu=mud, ok=ok)
    # per-prompt scoring records, for traffic-weighted deployment value
    idx = np.where(sc)[0]
    return dict(mu=mu, n=n, slices=slices, sc_idx=idx, sc_ctx=ctx[idx], sc_ds=ds[idx], sc_Q=Q[idx])


def target_policy(tab, supp):
    """pi^ref(j) = argmax over supported models of mu^ref."""
    return deployed_policy(tab["mu"], supp)


def reference_regret(pol, tab, supp, pol_ref=None):
    """Pooled R^ref of a deployed policy (mean over contexts of the target-value gap)."""
    pol_ref = target_policy(tab, supp) if pol_ref is None else pol_ref
    mu, J = tab["mu"], len(pol)
    g = [mu[pol_ref[j], j] - mu[pol[j], j] for j in range(J) if pol[j] >= 0 and pol_ref[j] >= 0]
    return float(np.mean(g)) if g else float("nan")


def reference_regret_by_slice(pol, tab, supp, pol_ref=None):
    """R^ref restricted to each benchmark slice's scoring prompts, pooled target policy fixed."""
    pol_ref = target_policy(tab, supp) if pol_ref is None else pol_ref
    out = {}
    for d, S in tab["slices"].items():
        mud, ok = S["mu"], S["ok"]
        g = [mud[pol_ref[j], j] - mud[pol[j], j]
             for j in range(len(pol)) if ok[j] and pol[j] >= 0 and pol_ref[j] >= 0]
        out[d] = float(np.mean(g)) if g else float("nan")
    return out


def operational_regret(pol, Ef_op, supp):
    """Regret against the common-population OPERATIONAL surface (what the evaluator would rank)."""
    pol_op = deployed_policy(Ef_op, supp)
    g = [Ef_op[pol_op[j], j] - Ef_op[pol[j], j]
         for j in range(len(pol)) if pol[j] >= 0 and pol_op[j] >= 0]
    return float(np.mean(g)) if g else float("nan")


def top1_error(pol, tab, supp, pol_ref=None):
    """Fraction of contexts whose deployed model is not the target-best model."""
    pol_ref = target_policy(tab, supp) if pol_ref is None else pol_ref
    e = [float(pol[j] != pol_ref[j]) for j in range(len(pol)) if pol[j] >= 0 and pol_ref[j] >= 0]
    return float(np.mean(e)) if e else float("nan")


def deployment_value(pol, tab):
    """Traffic-weighted reference-channel value of a policy on the untouched scoring cohort."""
    ctxs, Q = tab["sc_ctx"], tab["sc_Q"]
    v = [Q[t, pol[int(ctxs[t])]] for t in range(len(ctxs)) if pol[int(ctxs[t])] >= 0]
    return float(np.mean(v)) if v else float("nan")


def switch_outcomes(pol_base, pol_cand, deployed_candidate, tab):
    """False switch / false retention / switch indicator for one gate decision.

    A false switch: the gate replaces the log-derived deployment policy with the audited candidate
    and the candidate has strictly lower reference-channel value on the untouched scoring cohort.
    A false retention: the gate keeps the log-derived policy although the candidate is strictly
    better.
    """
    v_base = deployment_value(pol_base, tab)
    v_cand = deployment_value(pol_cand, tab)
    harmful = v_cand < v_base
    beneficial = v_cand > v_base
    return dict(false_switch=float(bool(deployed_candidate) and harmful),
                false_retention=float((not deployed_candidate) and beneficial),
                switch=float(bool(deployed_candidate) and (pol_cand != pol_base).any()),
                v_base=v_base, v_cand=v_cand)


def repair_fraction(r_trust, r_case, r_oracle_sel, tol=1e-4):
    """F_repair = (R_trust - R_CASE) / (R_trust - R_oracle-selection). None when undefined."""
    den = r_trust - r_oracle_sel
    if not np.isfinite(den) or den <= tol:
        return None
    return float((r_trust - r_case) / den)
