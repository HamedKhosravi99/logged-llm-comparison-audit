"""Economic layer for the routed-evaluation study.

Everything here is derived from quantities the existing pipeline already computes. No experimental
result is recomputed and no reported number changes. The one thing this module adds is the dollar
axis that both corpora carry and the paper never spent: `load_llmrouterbench.py` returns a per-prompt
cost matrix `C [N,K]` and RouterBench ships `|total_cost` columns, and `grid.py` uses them only to
rank models cheapest-first for the incumbent cascade.

Three cost surfaces, not one, because the pipeline reads three disjoint cohorts.

    c_acq[k,j]    mean dollar cost of model k over the ACQUISITION-cohort prompts of context j.
                  An audit readout of cell (k,j) draws n_read prompts from exactly this pool
                  (`evaluators.build_surfaces`, `coh == "acq"`), so this is what an audit costs.
    c_ver[k,j]    same over the VERIFICATION cohort. The deployment gate reads this pool.
    c_score[k,j]  same over the SCORING cohort. Deployment value is measured there, so this is
                  what running a deployed policy costs.

Keeping them apart is required by the cost-accounting rules: an audit dollar and a deployment
dollar are not the same dollar.
"""
import numpy as np

import case_core as cc


# --------------------------------------------------------------------------- cost surfaces
def cost_tables(base, supp=None):
    """Per-cell mean dollar cost on each cohort, plus scoring-traffic weights.

    Traffic weight p_j is the scoring-cohort share of context j, matching the plan's
    p_j = (# scoring requests in type j) / (# all scoring requests).
    """
    C, ctx, coh, K, J = base["C"], base["ctx"], base["coh"], base["K"], base["J"]
    out = {}
    for name, tag in (("c_acq", "acq"), ("c_ver", "ver"), ("c_score", "score")):
        M = np.full((K, J), np.nan)
        for j in range(J):
            m = np.where((coh == tag) & (ctx == j))[0]
            if len(m):
                M[:, j] = C[m].mean(0)
        out[name] = M
    n_sc = np.array([int(((coh == "score") & (ctx == j)).sum()) for j in range(J)], float)
    out["n_score"] = n_sc
    out["p"] = n_sc / n_sc.sum() if n_sc.sum() else np.zeros(J)
    # a missing cost must never be silently zero
    for name in ("c_acq", "c_ver", "c_score"):
        bad = ~np.isfinite(out[name])
        if supp is not None:
            bad &= supp
        out[name + "_missing"] = int(bad.sum())
    return out


# --------------------------------------------------------------------------- traced allocation
def alloc_traced(mode, M, cellq, supp, B, seed, n_read):
    """`case_core.allocate` with the purchased cells recorded.

    Byte-identical to the original by construction: the control flow below is copied from
    `case_core.allocate` and the same rng stream is consumed in the same order. `verify_alloc`
    asserts the equality on real data rather than trusting the copy.
    """
    K, J = M.shape
    rng = np.random.default_rng(seed)
    if mode == "raw":
        return M.copy(), 0, {}
    R = np.zeros((K, J), bool)
    reads = {}

    def do_read(k, j):
        reads[(k, j)] = cc.readout(cellq, k, j, rng, n_read)
        R[k, j] = True

    for k in range(K):
        js = np.where(supp[k])[0]
        if len(js) and R.sum() < B:
            do_read(k, int(js[rng.integers(len(js))]))
    while R.sum() < B:
        if mode == "uniform":
            free = np.argwhere(supp & ~R)
            if not len(free):
                break
            k, j = free[rng.integers(len(free))]
            do_read(int(k), int(j))
        elif mode == "rawmargin":
            Vraw = cc._place_reads(M.copy(), reads)
            cell = cc._smallest_margin_cell(Vraw, supp, R)
            if cell is None:
                break
            do_read(*cell)
        elif mode == "epig":
            Vobs = cc._place_reads(M.copy(), reads)
            cell = cc._epig_cell(Vobs, supp, R)
            if cell is None:
                break
            do_read(*cell)
        elif mode == "epig_deconf":
            b = cc.bias_rows(M, R, reads)
            cell = cc._epig_cell(cc.corrected(M, R, reads, b), supp, R)
            if cell is None:
                break
            do_read(*cell)
        else:
            b = cc.bias_rows(M, R, reads)
            cell = cc._smallest_margin_cell(cc.corrected(M, R, reads, b), supp, R)
            if cell is None:
                break
            do_read(*cell)
    b = cc.bias_rows(M, R, reads)
    return cc.corrected(M, R, reads, b), int(R.sum()), dict(reads)


def verify_alloc(mode, M, cellq, supp, B, seed, n_read):
    """Assert the traced allocator reproduces the original exactly. Returns the max abs deviation."""
    V0, u0 = cc.allocate(mode, None, M, cellq, supp, B, seed, n_read)
    V1, u1 = alloc_traced(mode, M, cellq, supp, B, seed, n_read)[:2]
    assert u0 == u1, f"{mode}: cells read {u0} != {u1}"
    d = float(np.max(np.abs(np.nan_to_num(V0) - np.nan_to_num(V1))))
    assert d == 0.0, f"{mode}: surfaces differ by {d}"
    return d


# --------------------------------------------------------------------------- cost accounting
def audit_cost(reads, ct, n_read):
    """Dollar cost of the purchased readouts.

    A readout of (k,j) runs model k on n_read acquisition-cohort prompts of context j, so its price
    is n_read * c_acq[k,j]. Sampling is with replacement, so the count is n_read even when the pool
    is smaller, except that `case_core.readout` caps the draw at the pool size; the cap is applied
    here too via `n_eff`.
    """
    tot = 0.0
    for (k, j) in reads:
        c = ct["c_acq"][k, j]
        if not np.isfinite(c):
            raise ValueError(f"missing acquisition cost for cell ({k},{j})")
        tot += n_read * float(c)
    return tot


def verification_cost(n_cells, ct, n_read, cells=None):
    """Dollar cost of gate verification readouts, priced on the verification cohort."""
    if cells:
        return sum(n_read * float(ct["c_ver"][k, j]) for (k, j) in cells)
    finite = ct["c_ver"][np.isfinite(ct["c_ver"])]
    return float(n_cells * n_read * finite.mean()) if finite.size else float("nan")


def deploy_cost(pol, ct, H=1):
    """C_deploy(pi, H) = H * sum_j p_j * c_score[pi(j), j]. Per-request when H = 1."""
    p, c = ct["p"], ct["c_score"]
    tot = 0.0
    for j in range(len(pol)):
        if pol[j] >= 0 and np.isfinite(c[pol[j], j]):
            tot += p[j] * float(c[pol[j], j])
    return H * tot


def deploy_quality(pol, mu, ct):
    """Traffic-weighted target quality of a policy, on the same weights as deploy_cost."""
    p = ct["p"]
    tot, w = 0.0, 0.0
    for j in range(len(pol)):
        if pol[j] >= 0 and np.isfinite(mu[pol[j], j]):
            tot += p[j] * float(mu[pol[j], j])
            w += p[j]
    return tot / w if w else float("nan")


# --------------------------------------------------------------------------- policies
def quality_constrained_policy(mu, ct, supp, delta=0.0, relative=False):
    """pi_delta(j) = argmin_k c_score[k,j] subject to mu[k,j] >= max_k mu[k,j] - delta.

    With `relative`, the tolerance is a fraction of the context's best quality rather than absolute.
    Ties in cost break toward the higher-quality model, then the lower index, so the rule is
    deterministic.
    """
    K, J = mu.shape
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        ks = np.array([k for k in ks if np.isfinite(mu[k, j]) and np.isfinite(ct["c_score"][k, j])])
        if not len(ks):
            continue
        best = mu[ks, j].max()
        tol = best * delta if relative else delta
        ok = ks[mu[ks, j] >= best - tol]
        pol[j] = int(min(ok, key=lambda k: (float(ct["c_score"][k, j]), -float(mu[k, j]), int(k))))
    return pol


def cheapest_policy(ct, supp):
    K, J = ct["c_score"].shape
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = [k for k in np.where(supp[:, j])[0] if np.isfinite(ct["c_score"][k, j])]
        if ks:
            pol[j] = int(min(ks, key=lambda k: float(ct["c_score"][k, j])))
    return pol


def priciest_policy(ct, supp):
    K, J = ct["c_score"].shape
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = [k for k in np.where(supp[:, j])[0] if np.isfinite(ct["c_score"][k, j])]
        if ks:
            pol[j] = int(max(ks, key=lambda k: float(ct["c_score"][k, j])))
    return pol


def single_best_model_policy(mu, supp, ct=None, budget=None):
    """One model for every context. With `budget`, the cheapest single model meeting it."""
    K, J = mu.shape
    cands = []
    for k in range(K):
        js = [j for j in range(J) if supp[k, j] and np.isfinite(mu[k, j])]
        if len(js) < J:
            continue
        q = float(np.mean([mu[k, j] for j in js]))
        c = float(np.mean([ct["c_score"][k, j] for j in js])) if ct is not None else 0.0
        cands.append((k, q, c))
    if not cands:
        return np.full(J, -1, int)
    if budget is not None:
        ok = [t for t in cands if t[2] <= budget]
        cands = ok or cands
    k = max(cands, key=lambda t: t[1])[0]
    return np.array([k if supp[k, j] else -1 for j in range(J)], int)


# --------------------------------------------------------------------------- economics
def savings(pol_base, pol_new, ct, audit_usd, H):
    """Gross and net savings, break-even horizon and ROI at horizon H."""
    c0 = deploy_cost(pol_base, ct, 1.0)
    c1 = deploy_cost(pol_new, ct, 1.0)
    per_req = c0 - c1
    gross = H * per_req
    net = gross - audit_usd
    be = (audit_usd / per_req) if per_req > 0 else None      # no finite break-even otherwise
    roi = (net / audit_usd) if audit_usd > 0 else None
    return dict(cost_per_request_base=c0, cost_per_request_new=c1, saving_per_request=per_req,
                gross_savings=gross, audit_usd=audit_usd, net_savings=net,
                break_even_requests=be, roi=roi, horizon=H)


def net_value(pol, mu, ct, gamma, mu_best=None):
    """V(pi) = -C_deploy(pi) - gamma * L_quality(pi), per request.

    L_quality is the traffic-weighted target-quality shortfall against the best feasible policy,
    so gamma is dollars per unit of quality lost per request.
    """
    if mu_best is None:
        mu_best = np.nanmax(np.where(np.isfinite(mu), mu, -np.inf), axis=0)
    p = ct["p"]
    loss = 0.0
    for j in range(len(pol)):
        if pol[j] >= 0 and np.isfinite(mu[pol[j], j]):
            loss += p[j] * (float(mu_best[j]) - float(mu[pol[j], j]))
    return -deploy_cost(pol, ct, 1.0) - gamma * loss, loss


# --------------------------------------------------------------------------- cost-aware CASE
def _decision_uncertainty(V, supp, j, sigma=0.2):
    """u_j, the probability the current top-two ordering in context j is wrong under a Gaussian
    working posterior of width sigma. Same posterior convention as `case_core._epig_cell`."""
    from math import erf, sqrt
    ks = np.where(supp[:, j])[0]
    if len(ks) < 2:
        return 0.0
    o = ks[np.argsort(-V[ks, j])]
    gap = float(V[o[0], j] - V[o[1], j])
    return 0.5 * (1.0 + erf((-gap / (sigma * sqrt(2.0))) / sqrt(2.0)))


def alloc_cost_aware(M, cellq, supp, B_usd, seed, n_read, ct, lam=1.0,
                     use_traffic=True, use_cost_denom=True, use_savings=True, sigma=0.2,
                     B_cells=None):
    """Cost-aware CASE. Spends a DOLLAR budget, not a cell count.

        Priority(k,j) = w_j * u_j * (|c_cur,j - c_kj| + lam * |q_cur,j - q_kj|) / C_audit(k,j)

    w_j is the traffic weight, u_j the decision uncertainty, c_cur/q_cur the currently deployed
    model's cost and estimated quality in context j. The ablation switches drop one term each,
    exactly as the plan's five-way ablation requires.

    Seeding matches CASE: one random supported cell per model first, so the arms are comparable.
    Purchases stop when no remaining cell fits the dollar budget.
    """
    K, J = M.shape
    rng = np.random.default_rng(seed)
    R = np.zeros((K, J), bool)
    reads = {}
    spent = 0.0

    def price(k, j):
        return n_read * float(ct["c_acq"][k, j])

    def do_read(k, j):
        nonlocal spent
        reads[(k, j)] = cc.readout(cellq, k, j, rng, n_read)
        R[k, j] = True
        spent += price(k, j)

    for k in range(K):
        js = np.where(supp[k])[0]
        if len(js):
            j = int(js[rng.integers(len(js))])
            if spent + price(k, j) <= B_usd:
                do_read(k, j)
    while True:
        if B_cells is not None and R.sum() >= B_cells:
            break
        b = cc.bias_rows(M, R, reads)
        V = cc.corrected(M, R, reads, b)
        pol = np.array([int(np.where(supp[:, j])[0][np.argmax(V[np.where(supp[:, j])[0], j])])
                        if supp[:, j].any() else -1 for j in range(J)])
        best, cell = -np.inf, None
        for j in range(J):
            u = _decision_uncertainty(V, supp, j, sigma) if use_savings else 1.0
            w = float(ct["p"][j]) if use_traffic else 1.0
            cur = pol[j]
            for k in np.where(supp[:, j] & ~R[:, j])[0]:
                pr = price(int(k), j)
                if spent + pr > B_usd:
                    continue
                if use_savings and cur >= 0:
                    dc = abs(float(ct["c_score"][cur, j]) - float(ct["c_score"][k, j]))
                    dq = abs(float(V[cur, j]) - float(V[k, j]))
                    gain = dc + lam * dq
                else:
                    gain = 1.0
                s = w * u * gain / (pr if use_cost_denom else 1.0)
                if s > best:
                    best, cell = s, (int(k), j)
        if cell is None:
            break
        do_read(*cell)
    b = cc.bias_rows(M, R, reads)
    return cc.corrected(M, R, reads, b), int(R.sum()), dict(reads), spent


def net_value_policy(mu, ct, supp, V_per_quality):
    """Decide on net value directly. pi(j) = argmax_k [ V * mu[k,j] - c_score[k,j] ].

    This is decision 0.3(iii) in the plans. The paper's setup already licenses any target fixed in
    advance, so with this target deployment regret is literally dollars per request. V is the value
    of one unit of target quality on one request, swept rather than asserted.
    """
    K, J = mu.shape
    pol = np.full(J, -1, int)
    for j in range(J):
        ks = [k for k in np.where(supp[:, j])[0]
              if np.isfinite(mu[k, j]) and np.isfinite(ct["c_score"][k, j])]
        if ks:
            pol[j] = int(max(ks, key=lambda k: V_per_quality * float(mu[k, j])
                             - float(ct["c_score"][k, j])))
    return pol
