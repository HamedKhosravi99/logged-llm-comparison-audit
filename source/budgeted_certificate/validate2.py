import itertools, sys, os
import numpy as np
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"code")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_budgeted_certificate import cell_tables, solve, certificate
from eval_validity.grid import build_base
from validate import robust_direct
base=build_base(seed=0,n_clusters=8,corpus="llmrouterbench")
mu,p,w=cell_tables(base); K,J=mu.shape
cheap=sum(w[j]*p[:,j].min() for j in range(J))
GAM={"cheap*1.25":cheap*1.25,"cheap*1.5":cheap*1.5,"cheap*2":cheap*2,"cheap*5":cheap*5}
print("%-12s %-12s %-7s %-11s %-11s %s"%("label","Gamma","eta","certificate","direct","agree?"))
bad=0
for lab,g in GAM.items():
    for eta_s in [0.001,0.005,0.017,0.018,0.045,0.05]:
        eta=np.full(J,eta_s); e=np.tile(eta/2.0,(K,1)); mu_hat=mu-e
        xhat,_,_=solve(mu_hat,p,w,g)
        cert=certificate(mu,p,w,g,eta)["settled"]
        direct,_,_=robust_direct(mu_hat,e,p,w,g,xhat)
        ok = cert==direct; bad += (not ok)
        print("%-12s %-12.8f %-7.3f %-11s %-11s %s"%(lab,g,eta_s,"settled" if cert else "not",
              "settled" if direct else "not","OK" if ok else "** MISMATCH **"))
print("\nmismatches: %d / %d"%(bad,len(GAM)*6))
print("\n=== is the workload-1 tie exact? (Gamma=cheap*1.25) ===")
g=cheap*1.25; eta=np.full(J,0.001); e=np.tile(eta/2.0,(K,1))
xhat,_,_=solve(mu-e,p,w,g); r=certificate(mu,p,w,g,eta)
for j in r["moved"]:
    inc,riv=xhat[j],r["xadv"][j]
    print("  workload %d  incumbent m%d mu=%.10f  rival m%d mu=%.10f  |diff|=%.3e  w_j=%.4f"
          %(j,inc,mu[inc,j],riv,mu[riv,j],abs(mu[inc,j]-mu[riv,j]),w[j]))
    print("     n scoring requests in this workload: %d"%int((( base["coh"]=="score")&(base["ctx"]==j)).sum()))
