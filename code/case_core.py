"""Self-contained CASE core for the routing study (same method as the paper, re-implemented
here so this folder stands alone).

A routed cell (model k, category j) can be *randomized*: run model k on a fresh random
sample of category-j prompts and average -> an unbiased, finite-sample estimate of Ef[k,j].
CASE spends a budget of such readouts, deconfounds the log with a per-action (per-model)
working correction b_k, allocates the next readout to the smallest corrected top-two margin,
and re-estimates b_k after every readout (the closed loop).
"""
import numpy as np


def readout(cellq, k, j, rng, n_read):
    """Unbiased noisy estimate of Ef[k,j]: mean quality of model k over n_read random category-j prompts."""
    pool = cellq[(k, j)]
    n = min(n_read, len(pool))
    return float(pool[rng.integers(0, len(pool), size=n)].mean())


def bias_rows(M, R, reads):
    """Per-model working correction b_k = mean over revealed cells in row k of (log - readout).
    Rows with no readout get b_k = 0 (no evidence yet)."""
    K, J = M.shape
    b = np.zeros(K)
    for k in range(K):
        js = np.where(R[k])[0]
        if len(js):
            b[k] = np.mean([M[k, j] - reads[(k, j)] for j in js])
    return b


def corrected(M, R, reads, b):
    """Deployed surface: readout on revealed cells, log minus per-model correction elsewhere."""
    V = M - b[:, None]
    for (k, j), r in reads.items():
        V[k, j] = r
    return V


def _smallest_margin_cell(V, supp, R):
    """The unrevealed supported cell whose column has the smallest top-two gap (the contested context)."""
    K, J = V.shape
    best, cell = np.inf, None
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if len(ks) < 2:
            continue
        o = ks[np.argsort(-V[ks, j])]
        gap = V[o[0], j] - V[o[1], j]
        for rank, k in enumerate(o):
            if not R[k, j]:
                score = gap + 0.01 * rank            # prefer the contested column, top contenders first
                if score < best:
                    best, cell = score, (k, j)
    return cell


def _epig_cell(V, supp, R, sigma=0.2):
    """R-EPIG / decision-information acquisition (gao2026observationally): reveal the top-two
    contender, in the most decision-uncertain column, that most reduces the expected wrong-argmax
    probability -- computed on the OBSERVED (confounded) surface, i.e. it never deconfounds the
    boundary. Posterior std is sigma on unread cells, ~0.3*sigma on read (resolved) cells."""
    from math import erf, sqrt
    ncdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    K, J = V.shape
    best, cell = -1.0, None
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if len(ks) < 2:
            continue
        o = ks[np.argsort(-V[ks, j])]
        top, second = int(o[0]), int(o[1])
        gap = V[top, j] - V[second, j]
        s_t = sigma * (1.0 if not R[top, j] else 0.3)
        s_s = sigma * (1.0 if not R[second, j] else 0.3)
        p_wrong = ncdf(-gap / (np.sqrt(s_t ** 2 + s_s ** 2) + 1e-9))     # P(second beats top)
        for k in (top, second):
            if not R[k, j] and p_wrong > best:
                best, cell = p_wrong, (k, j)
    return cell


def allocate(mode, Ef, M, cellq, supp, B, seed, n_read, quota=1):
    """Return (V deployed surface, n_reads_used). mode in {raw, uniform, case}.
      raw     : no readouts; deploy the confounded log directly.
      uniform : spend B readouts on random cells; deploy per-model-corrected + readouts.
      case    : spend B readouts by corrected-margin allocation, re-deconfounding each step.
    A per-model seed (one random cell per model) is included in the budget for uniform/case."""
    K, J = M.shape
    rng = np.random.default_rng(seed)
    if mode == "raw":
        return M.copy(), 0
    R = np.zeros((K, J), bool); reads = {}
    def do_read(k, j):
        reads[(k, j)] = readout(cellq, k, j, rng, n_read); R[k, j] = True
    for k in range(K):                                   # seed: one random cell per model
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
        elif mode == "rawmargin":                        # allocate by the RAW log margin (never deconfound the boundary)
            Vraw = M.copy()
            for (k, j), r in reads.items():
                Vraw[k, j] = r
            cell = _smallest_margin_cell(Vraw, supp, R)
            if cell is None:
                break
            do_read(*cell)
        elif mode == "epig":                             # R-EPIG (gao2026observationally): decision-uncertainty
            Vobs = M.copy()                              # acquisition on the OBSERVED (confounded) surface -- no deconfounding
            for (k, j), r in reads.items():
                Vobs[k, j] = r
            cell = _epig_cell(Vobs, supp, R)
            if cell is None:
                break
            do_read(*cell)
        elif mode == "epig_deconf":                      # R-EPIG's rule, but acquiring on the DECONFOUNDED surface
            b = bias_rows(M, R, reads)                   # (isolates "deconfound-before-acquire" from the acquisition rule)
            cell = _epig_cell(corrected(M, R, reads, b), supp, R)
            if cell is None:
                break
            do_read(*cell)
        else:                                            # case: re-deconfound, then smallest corrected margin
            b = bias_rows(M, R, reads)
            V = corrected(M, R, reads, b)
            cell = _smallest_margin_cell(V, supp, R)
            if cell is None:
                break
            do_read(*cell)
    b = bias_rows(M, R, reads)
    return corrected(M, R, reads, b), int(R.sum())


def policy_regret(V, Ef, supp=None):
    """Mean value gap vs the oracle: pick argmax_k V per context, score by true Ef.
    If supp is given, choose only among supported models in each context."""
    K, J = Ef.shape
    if supp is None:
        chosen = V.argmax(0)
        return float(np.mean(Ef.max(0) - Ef[chosen, np.arange(J)]))
    reg = []
    for j in range(J):
        ks = np.where(supp[:, j])[0]
        if len(ks) < 1:
            continue
        chosen = ks[np.argmax(V[ks, j])]
        reg.append(Ef[ks, j].max() - Ef[chosen, j])
    return float(np.mean(reg)) if reg else 0.0


# ----------------------------------------------------------------------------
# Value-model roster: the paper's published *correction* baselines (Table tab:roster).
# Each takes the confounded log M, the revealed readout cells (reads/R from a uniform
# budget), and returns a deployed surface V. They differ in HOW they turn the log + a
# small unbiased sample into an estimate of Ef -- the paper's R1 (value-model) axis --
# and are scored at their native *uniform* allocation, exactly as in tab:roster. CASE
# (per-action correction + active corrected-margin allocation) is scored separately.
# ----------------------------------------------------------------------------
def _place_reads(V, reads):
    for (k, j), r in reads.items():
        V[k, j] = r
    return V


def vm_log(M, reads, R):
    """log-greedy / behavior cloning: trust the confounded log (no correction)."""
    return _place_reads(M.copy(), reads)


def vm_peraction(M, reads, R):
    """CASE's value model in isolation: per-action (row-constant) working correction.
    Rank-1 bias b_k 1^T; the paper's per-action candidate (Prop. peraction)."""
    b = bias_rows(M, R, reads)
    return corrected(M, R, reads, b)


def vm_twoway(M, reads, R, iters=50):
    """Two-way / additive fixed-effects estimate-and-extrapolate (Kallus, two-way FE
    family): fit an additive row+column bias b_k + c_j on the revealed residuals by
    alternating means, extrapolate, subtract. Rank<=2 bias."""
    K, J = M.shape
    bk, cj = np.zeros(K), np.zeros(J)
    rows = {k: np.where(R[k])[0] for k in range(K)}
    cols = {j: np.where(R[:, j])[0] for j in range(J)}
    for _ in range(iters):
        for k in range(K):
            js = rows[k]
            if len(js):
                bk[k] = np.mean([(M[k, j] - reads[(k, j)]) - cj[j] for j in js])
        for j in range(J):
            ks = cols[j]
            if len(ks):
                cj[j] = np.mean([(M[k, j] - reads[(k, j)]) - bk[k] for k in ks])
    V = M - (bk[:, None] + cj[None, :])
    return _place_reads(V, reads)


def vm_athey(M, reads, R, tau=3.0):
    """Athey-combine / experiment-grounded shrinkage (athey2020combining): per-action
    correction shrunk toward 0 by how much unbiased evidence supports it (weight
    n_k/(n_k+tau)) -- trust the log where the experimental sample is thin. (With a dense
    uniform budget the shrinkage is small, so it tracks the per-action corrector.)"""
    K, J = M.shape
    b = np.zeros(K)
    for k in range(K):
        js = np.where(R[k])[0]
        if len(js):
            raw_b = np.mean([M[k, j] - reads[(k, j)] for j in js])
            b[k] = (len(js) / (len(js) + tau)) * raw_b
    return corrected(M, R, reads, b)


def _lr_complete(RD, R, rank, iters=80):
    """Rank-r hard-impute completion of a matrix whose OBSERVED entries are RD on mask R
    (column-mean warm start on the unobserved). Anchors observed cells each iteration."""
    K, J = RD.shape
    E = RD.copy()
    for j in range(J):
        ks = np.where(R[:, j])[0]
        E[~R[:, j], j] = E[ks, j].mean() if len(ks) else (RD[R].mean() if R.any() else 0.0)
    r = int(min(rank, K, J))
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(E, full_matrices=False)
        s[r:] = 0.0
        E = np.where(R, RD, (U * s) @ Vt)
    return E


def vm_causalmc(M, reads, R, ranks=(1, 2, 3, 4)):
    """Faithful causal matrix completion (Causal-MC, SpectralRep, G-ESTT family): treat the
    unbiased readouts as the observed entries of the OUTCOME matrix E and low-rank-complete
    it -- it has no bias model and does NOT use the confounded log's level. Rank is chosen
    by held-out error on a 25% split of the revealed cells (data-driven, no oracle peeking)."""
    K, J = M.shape
    RD = np.zeros((K, J))
    for (k, j), v in reads.items():
        RD[k, j] = v
    cells = list(reads.keys())
    if len(cells) < 6:
        return _place_reads(_lr_complete(RD, R, ranks[0]), reads)
    order = np.argsort([k * J + j for (k, j) in cells])   # deterministic split (seed-free)
    val = {cells[int(i)] for i in order[::4]}             # ~25% held out for rank selection
    Rf = np.zeros((K, J), bool); RDf = np.zeros((K, J))
    for (k, j) in cells:
        if (k, j) not in val:
            Rf[k, j] = True; RDf[k, j] = reads[(k, j)]
    best_rank, best_err = ranks[0], np.inf
    for rk in ranks:
        E = _lr_complete(RDf, Rf, rk)
        err = np.mean([(E[k, j] - reads[(k, j)]) ** 2 for (k, j) in val])
        if err < best_err:
            best_err, best_rank = err, rk
    return _place_reads(_lr_complete(RD, R, best_rank), reads)


VALUE_MODELS = {
    "log": vm_log,                 # log-greedy / behavior cloning
    "peraction": vm_peraction,     # CASE value model (per-action), uniform alloc
    "twoway": vm_twoway,           # Kallus / two-way fixed-effects additive correction
    "athey": vm_athey,             # Athey-combine / experiment-grounded shrinkage
    "causalmc": vm_causalmc,       # Causal-MC / SpectralRep / G-ESTT (OUTCOME low-rank completion)
}


def uniform_reveal(Ef, M, cellq, supp, B, seed, n_read):
    """The native uniform allocation shared by every roster baseline: per-action seed
    (one random cell each) then random fill to budget B. Returns (reads, R)."""
    K, J = M.shape
    rng = np.random.default_rng(seed)
    R = np.zeros((K, J), bool)
    reads = {}
    def do_read(k, j):
        reads[(k, j)] = readout(cellq, k, j, rng, n_read); R[k, j] = True
    for k in range(K):                                   # seed: one random cell per model
        js = np.where(supp[k])[0]
        if len(js) and R.sum() < B:
            do_read(k, int(js[rng.integers(len(js))]))
    while R.sum() < B:
        free = np.argwhere(supp & ~R)
        if not len(free):
            break
        k, j = free[rng.integers(len(free))]
        do_read(int(k), int(j))
    return reads, R


def allocate_roster(vm, Ef, M, cellq, supp, B, seed, n_read):
    """Native uniform allocation (as tab:roster gives the published baselines) + a value
    model. Returns the deployed surface V. Same seeding/budget as CASE's uniform arm."""
    reads, R = uniform_reveal(Ef, M, cellq, supp, B, seed, n_read)
    return VALUE_MODELS[vm](M, reads, R)
