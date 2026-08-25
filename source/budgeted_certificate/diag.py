import itertools, sys, os
import numpy as np
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_budgeted_certificate import cell_tables, solve, certificate
from eval_validity.grid import build_base

base=build_base(seed=0,n_clusters=8,corpus="llmrouterbench")
mu,p,w=cell_tables(base); K,J=mu.shape
# unconstrained optimum = per-workload argmax, ignoring the budget
xu=tuple(int(np.argmax(mu[:,j])) for j in range(J))
qu=sum(w[j]*mu[xu[j],j] for j in range(J)); cu=sum(w[j]*p[xu[j],j] for j in range(J))
print("UNCONSTRAINED optimum: quality %.4f at spend %.5f\n"%(qu,cu))
cheap=sum(w[j]*p[:,j].min() for j in range(J)); rich=sum(w[j]*p[:,j].max() for j in range(J))
gammas=[cheap,cheap*1.25,cheap*1.5,cheap*2,cheap*3,cheap*5,rich*0.5,rich*0.8,rich]
print("%-10s %-9s %-8s %-8s %-9s %-11s %s"%("Gamma","n_feas","spend","quality","gap to unc","constraint","settled up to eta"))
for g in gammas:
    x,v,nf=solve(mu,p,w,g)
    c=sum(w[j]*p[x[j],j] for j in range(J)); q=sum(w[j]*mu[x[j],j] for j in range(J))
    active = q < qu-1e-12
    last=None
    for e in np.round(np.arange(0,0.2005,0.001),3):
        r=certificate(mu,p,w,g,np.full(J,e))
        if not r["feasible"]: break
        if r["settled"]: last=e
        else: break
    print("%-10.5f %-9d %-8.5f %-8.4f %-9.4f %-11s %s"%(g,nf,c,q,qu-q,
          "ACTIVE" if active else "slack", ("%.3f"%last) if last is not None else "0.000 (fails at 0.001)"))
