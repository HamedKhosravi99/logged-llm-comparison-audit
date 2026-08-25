"""Cross-validate the certificate against DIRECT robust optimality on the real instance.

The certificate says: one adverse solve suffices. This checks that claim the hard way on this
corpus -- rival by rival, minimising the gap coordinate-by-coordinate over the box -- exactly as
code/verify_mckp_certificate.py does for synthetic instances.
"""
import itertools, sys, os
import numpy as np
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"code")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_budgeted_certificate import cell_tables, solve, certificate
from eval_validity.grid import build_base
TOL=1e-12

def robust_direct(mu_hat, e, p, w, gamma, xhat):
    """True robust optimality: for every feasible rival, worst-case gap >= 0?"""
    K,J = mu_hat.shape
    for x in itertools.product(range(K), repeat=J):
        if sum(w[j]*p[x[j],j] for j in range(J)) > gamma+1e-12: continue
        if x == xhat: continue
        # worst case for xhat: its cells at lower bound, rival cells at upper bound
        gap = 0.0
        for j in range(J):
            if x[j]==xhat[j]: continue
            gap += w[j]*((mu_hat[xhat[j],j]-e[xhat[j],j]) - (mu_hat[x[j],j]+e[x[j],j]))
        if gap < -TOL: return False, x, gap
    return True, None, None

base=build_base(seed=0,n_clusters=8,corpus="llmrouterbench")
mu,p,w=cell_tables(base); K,J=mu.shape
print("%-10s %-7s %-11s %-11s %s"%("Gamma","eta","certificate","direct","agree?"))
for g in [0.00067, 0.00080, 0.00107, 0.00267]:
    for eta_s in [0.001, 0.005, 0.01, 0.045, 0.05]:
        eta=np.full(J,eta_s); e=np.tile(eta/2.0,(K,1)); mu_hat=mu-e
        xhat,_,_=solve(mu_hat,p,w,g)
        cert=certificate(mu,p,w,g,eta)["settled"]
        direct,witness,gap=robust_direct(mu_hat,e,p,w,g,xhat)
        print("%-10.5f %-7.3f %-11s %-11s %s"%(g,eta_s,"settled" if cert else "not",
              "settled" if direct else "not", "OK" if cert==direct else "** MISMATCH **"))
# why is 0.00067 fragile? show the margin in the workload that moves
print("\n=== the near-tie at Gamma=0.00067 ===")
g=0.00067; eta=np.full(J,0.001); e=np.tile(eta/2.0,(K,1)); mu_hat=mu-e
xhat,_,_=solve(mu_hat,p,w,g)
r=certificate(mu,p,w,g,eta)
for j in r["moved"]:
    inc,riv = xhat[j], r["xadv"][j]
    print("  workload %d: incumbent model %d mu=%.4f p=%.5f | rival model %d mu=%.4f p=%.5f | w_j=%.3f | quality margin %.5f"
          %(j,inc,mu[inc,j],p[inc,j],riv,mu[riv,j],p[riv,j],w[j],mu[inc,j]-mu[riv,j]))
