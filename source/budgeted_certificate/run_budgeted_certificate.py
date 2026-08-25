"""Prop. 3.4 (robust-optimality certificate) on a real corpus at a stated serving budget.

The paper formulates budgeted allocation as a multiple-choice knapsack, Eq. (1), and states the
settledness certificate as Prop. 3.4. Until now that proposition was exercised only by brute force
over sampled synthetic instances; every empirical section runs the quality-constrained rule of
Eq. (2) instead. App. A shows the two rules do not reduce to each other, so those empirics do not
transfer. This script closes that gap: it runs Eq. (1) on LLMRouterBench at stated budgets Gamma
and reports the certificate verdict.

Construction, all from the paper's own definitions:
  mu_obs   per-cell mean target score on the SCORING cohort  (K x J)
  p_kj     per-cell mean dollar cost on the scoring cohort   (K x J)
  w_j      scoring-cohort traffic share of workload j
  Gamma    serving budget, dollars per request served (so the constraint is sum_j w_j p_x(j),j <= Gamma)
  eta_j    assumed evaluator-error spread; the admissible box is centred at mu_obs - eta/2
           with half-width e = eta/2, exactly as Sec. 3 states.

Verdict, Prop. 3.4 second form: solve at the nominal table, then at the adverse table (incumbent
cells pushed down, all rivals pushed up). The allocation is settled over the box iff the nominal
optimum is still optimal adversely.

OVERWRITES the git-tracked budgeted_certificate.json next to this file.
"""
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "code"))

from eval_validity.grid import build_base            # noqa: E402
from econ_core import cost_tables                    # noqa: E402

TOL = 1e-12


def cell_tables(base):
    """mu_obs on the scoring cohort, plus prices and traffic shares."""
    Q, ctx, coh, K, J = base["Q"], base["ctx"], base["coh"], base["K"], base["J"]
    mu = np.full((K, J), np.nan)
    for j in range(J):
        m = (coh == "score") & (ctx == j)
        if m.sum():
            mu[:, j] = Q[m].mean(0)
    ct = cost_tables(base)
    return mu, ct["c_score"], ct["p"]


def solve(mu, p, w, gamma):
    """Exact MCKP by enumeration: one model per workload, maximise sum_j w_j mu, s.t. cost <= gamma."""
    K, J = mu.shape
    best_v, best_x, n_feas = -np.inf, None, 0
    for x in itertools.product(range(K), repeat=J):
        cost = sum(w[j] * p[x[j], j] for j in range(J))
        if cost > gamma + TOL:
            continue
        n_feas += 1
        v = sum(w[j] * mu[x[j], j] for j in range(J))
        if v > best_v + TOL:
            best_v, best_x = v, x
    return best_x, best_v, n_feas


def certificate(mu_obs, p, w, gamma, eta):
    """Prop. 3.4. Returns the verdict and the quantities behind it."""
    K, J = mu_obs.shape
    e = np.tile(eta / 2.0, (K, 1))                 # e_kj = eta_j / 2
    mu_hat = mu_obs - e                            # centred, as Sec. 3 states
    xhat, vhat, n_feas = solve(mu_hat, p, w, gamma)
    if xhat is None:
        return dict(feasible=False)
    # adverse table: incumbent down, every rival up
    mu_chk = mu_hat + e
    for j in range(J):
        mu_chk[xhat[j], j] = mu_hat[xhat[j], j] - e[xhat[j], j]
    xadv, vadv, _ = solve(mu_chk, p, w, gamma)
    settled = (xadv == xhat)
    cost_hat = sum(w[j] * p[xhat[j], j] for j in range(J))
    return dict(feasible=True, settled=bool(settled), xhat=xhat, xadv=xadv,
                n_feasible=n_feas, cost=float(cost_hat),
                binds=bool(cost_hat > gamma - 1e-9),
                quality=float(sum(w[j] * mu_obs[xhat[j], j] for j in range(J))),
                n_moved=int(sum(1 for j in range(J) if xhat[j] != xadv[j])),
                moved=[j for j in range(J) if xhat[j] != xadv[j]])


def main():
    base = build_base(seed=0, n_clusters=8, corpus="llmrouterbench")
    mu, p, w = cell_tables(base)
    K, J = mu.shape
    assert np.isfinite(mu).all() and np.isfinite(p).all(), "missing cells"
    models = list(base["models"])
    print("corpus=llmrouterbench  K=%d  J=%d  models=%s" % (K, J, models))
    print("traffic shares w_j = %s  (sum %.3f)" % (np.round(w, 3).tolist(), w.sum()))
    print("price range per request: %.5f .. %.5f dollars" % (p.min(), p.max()))
    cheap = sum(w[j] * p[:, j].min() for j in range(J))
    rich = sum(w[j] * p[:, j].max() for j in range(J))
    print("cheapest feasible spend %.5f   richest %.5f  (dollars per request)\n" % (cheap, rich))

    gammas = [cheap * 1.0, cheap * 1.25, cheap * 1.5, cheap * 2.0, cheap * 3.0,
              cheap * 5.0, rich * 0.5, rich * 0.8, rich]
    etas = [0.0, 0.005, 0.01, 0.02, 0.05]
    out = []
    print("%-12s %-8s %-9s %-7s %-6s %-8s %s" %
          ("Gamma", "spend", "binds?", "quality", "eta", "settled", "workloads that move"))
    for g in gammas:
        for eta_s in etas:
            eta = np.full(J, eta_s)
            r = certificate(mu, p, w, g, eta)
            if not r["feasible"]:
                print("%-12.5f  infeasible" % g); continue
            print("%-12.5f %-8.5f %-9s %-7.4f %-6.3f %-8s %s" %
                  (g, r["cost"], "yes" if r["binds"] else "no", r["quality"], eta_s,
                   "YES" if r["settled"] else "no", r["moved"] if r["moved"] else "-"))
            out.append(dict(gamma=float(g), eta=float(eta_s), **{k: v for k, v in r.items()
                                                                 if k not in ("xhat", "xadv")}))
        print()

    # the useful summary: largest eta at which each budget is still settled
    print("=== settledness margin: largest eta with a settled allocation ===")
    fine = np.round(np.arange(0.0, 0.101, 0.001), 3)
    for g in gammas:
        last = None
        for eta_s in fine:
            r = certificate(mu, p, w, g, np.full(J, eta_s))
            if not r["feasible"]:
                break
            if r["settled"]:
                last = eta_s
            else:
                break
        print("  Gamma=%.5f  settled up to eta = %s" %
              (g, ("%.3f" % last) if last is not None else "not settled even at eta=0"))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "budgeted_certificate.json"), "w"), indent=1)
    print("\nwrote budgeted_certificate.json (a tracked file in the repository)")


if __name__ == "__main__":
    main()
