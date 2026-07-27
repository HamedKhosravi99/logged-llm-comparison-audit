"""Factor A: how the observational log assigns models to prompts.

A1 (selective router)     the incumbent cost cascade with exploration rate eps picks the model for
                          each logged prompt, so a model is evaluated on the prompts the router
                          sends it -- the population a model is measured on depends on the model.
A0 (common population)    the SAME logged prompts, the SAME per-cell counts n_kj, but the
                          assignment is a uniform random partition of each context's log prompts.
                          Router-induced selection bias B^sel is then zero up to sampling noise.

Matching the per-cell counts exactly is what isolates assignment validity from log sample size:
between a paired A1/A0 run, only the assignment *distribution* differs.
"""
import numpy as np


def make_selective_log(base, seed, eps=0.15):
    """A1. Returns (cells, n_kj, log_data) with cells[(k,j)] = prompt indices logged for model k."""
    ctx, coh, K, J = base["ctx"], base["coh"], base["K"], base["J"]
    inc_prompt = base["inc_prompt"]
    rng = np.random.default_rng(seed)
    cells = {(k, j): [] for k in range(K) for j in range(J)}
    asg_all, ctx_all, idx_all = [], [], []
    for j in range(J):
        lg = np.where((coh == "log") & (ctx == j))[0]
        assign = np.empty(len(lg), int)
        for t, x in enumerate(lg):
            p = np.full(K, eps / K)
            p[inc_prompt[x]] += (1.0 - eps)             # known logging propensity
            assign[t] = rng.choice(K, p=p)
        for k in range(K):
            cells[(k, j)] = lg[assign == k]
        asg_all.append(assign); ctx_all.append(np.full(len(lg), j)); idx_all.append(lg)
    n_kj = np.array([[len(cells[(k, j)]) for j in range(J)] for k in range(K)], int)
    log_data = dict(idx=np.concatenate(idx_all), ctx=np.concatenate(ctx_all),
                    assign=np.concatenate(asg_all), eps=eps, K=K, J=J)
    return cells, n_kj, log_data


def make_matched_common_population_log(base, seed, n_kj):
    """A0. Same log prompts and same per-cell counts n_kj, assigned uniformly at random.

    Each context's log prompts are randomly permuted and split into consecutive blocks of sizes
    n_kj[:, j], so every prompt keeps exactly one role and the per-cell counts match A1 exactly.
    """
    ctx, coh, K, J = base["ctx"], base["coh"], base["K"], base["J"]
    rng = np.random.default_rng(seed + 10_000)
    cells = {}
    for j in range(J):
        lg = np.where((coh == "log") & (ctx == j))[0]
        perm = rng.permutation(lg)
        cut = np.cumsum(n_kj[:, j])[:-1]
        for k, block in enumerate(np.split(perm, cut)):
            cells[(k, j)] = block
    return cells


def support(n_kj, base, min_assign=8, n_read=30):
    """Supported cells: enough logged prompts AND a randomizable acquisition pool of >= n_read."""
    ctx, coh, K, J = base["ctx"], base["coh"], base["K"], base["J"]
    n_acq = np.array([int(((coh == "acq") & (ctx == j)).sum()) for j in range(J)])
    return (n_kj >= min_assign) & (n_acq[None, :] >= n_read)
