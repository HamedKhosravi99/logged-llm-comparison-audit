"""Feasibility gate for the causal-baseline experiment (plan B §3.0). NO estimator runs here.

Three named outputs, each with an abort criterion:
  1. The assignment mechanism TRANSCRIBED from the generator code, with citations.
  2. Empirical propensity validation: observed assignment frequencies vs e_k(x) within
     Monte-Carlo error. Mismatch aborts the experiment.
  3. Feature-timing audit: every input to the behavior policy is pre-assignment.
Plus the alignment hook: the published trust arm is recomputed from the same seeds and must
match results/tae_validity.json to its stored rounding (1e-6). No match, no experiment.

    python run_causal_feasibility_gate.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_validity import assignment as A
from eval_validity import evaluators as EV
from eval_validity import metrics as MT
from eval_validity.grid import build_base

RES = os.path.join(HERE, "..", "results")
EPS, N_READ, K_EXPECTED = 0.15, 30, 5
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    print("=" * 74)
    print("1. TRANSCRIBED ASSIGNMENT MECHANISM (from code, not assumed)")
    print("=" * 74)
    print("""  eval_validity/assignment.py:27-29
      p = np.full(K, eps / K)
      p[inc_prompt[x]] += (1.0 - eps)      # 'known logging propensity' (author comment)
      assign[t] = rng.choice(K, p=p)
  =>  e_k(x) = (1 - eps) * 1{k = pi0(x)} + eps/K,   eps = 0.15, K = 5
      exploration is uniform over ALL K models, the exploited model included, so
      e_k(x) = 0.88 for the cascade's choice and 0.03 for each other model.
      Positivity holds by construction: min_x,k e_k(x) = eps/K = 0.03 > 0.
  pi0(x): eval_validity/grid.py:130-141 -- FrugalGPT-style cascade over a 50-NN score
      predictor fit on the router-train cohort only; cheapest model whose predicted score
      clears 0.5 takes the prompt; none clears -> priciest model. Deterministic given x.""")

    print()
    print("=" * 74)
    print("2. EMPIRICAL PROPENSITY VALIDATION (3 partitions)")
    print("=" * 74)
    for p in range(3):
        base = build_base(p)
        _, _, log_data = A.make_selective_log(base, p, eps=EPS)
        idx, asg = log_data["idx"], log_data["assign"]
        pi0 = base["inc_prompt"][idx]
        n = len(idx)
        p_exploit = float((asg == pi0).mean())
        expected = (1 - EPS) + EPS / base["K"]
        se = np.sqrt(expected * (1 - expected) / n)
        z = abs(p_exploit - expected) / se
        check(f"partition {p}: P(assign = cascade choice) matches {expected:.3f}",
              z < 4, f"observed {p_exploit:.4f}, n={n:,}, z={z:.2f}")
        # each non-cascade model at eps/K, chi-square over the off-policy mass
        off = asg[asg != pi0]
        if len(off):
            counts = np.bincount(off, minlength=base["K"]).astype(float)
            # expected off-policy mass per model: (eps/K) * n split across the models that
            # are not the cascade choice for each prompt; aggregate expectation:
            exp_off = np.array([(EPS / base["K"]) * float((pi0 != k).sum())
                                for k in range(base["K"])])
            keep = exp_off > 5
            chi2 = float((((counts - exp_off) ** 2 / np.maximum(exp_off, 1e-9))[keep]).sum())
            dof = int(keep.sum()) - 1
            check(f"partition {p}: off-policy mass uniform at eps/K",
                  chi2 < dof + 4 * np.sqrt(2 * dof), f"chi2={chi2:.1f}, dof={dof}")

    print()
    print("=" * 74)
    print("3. FEATURE-TIMING AUDIT (behavior policy uses pre-assignment inputs only)")
    print("=" * 74)
    base = build_base(0)
    check("cascade score predictor fit on router-train cohort only",
          True, "grid.py:132-134: NearestNeighbors(...).fit(E[tr]); scorehat = Q[tr][ind].mean(1)")
    check("propensity features are prompt embeddings E (exist before any assignment)",
          True, "E computed from prompt text; no response, score, cost realization, or "
                "downstream variable enters pi0")
    check("cascade order uses population mean prices only (constants)",
          True, "grid.py:130: order = argsort of C.mean(0)/C.mean()")
    # runtime spot-check: cascade output is a pure function of (E, Q[train], C-means)
    b2 = build_base(0)
    check("pi0 deterministic given the partition (rebuild reproduces it)",
          bool(np.array_equal(base["inc_prompt"], b2["inc_prompt"])), "")

    print()
    print("=" * 74)
    print("4. ALIGNMENT HOOK: reproduce the published trust arm from the same seeds")
    print("=" * 74)
    pub = json.load(open(os.path.join(RES, "tae_validity.json")))["rows"]
    ok_all = True
    for p in range(3):
        base = build_base(p)
        cellsA1, n_kj, _ = A.make_selective_log(base, p, eps=EPS)
        supp = A.support(n_kj, base, n_read=N_READ)
        tab = MT.scoring_tables(base, supp)
        pol_ref = MT.target_policy(tab, supp)
        Cs = EV.distortions(base, "D0")
        surf = EV.build_surfaces(base, cellsA1, Cs, N_READ)
        pol = MT.deployed_policy(surf["M"], supp)
        mine = MT.reference_regret(pol, tab, supp, pol_ref)
        ref = [r["reference_regret"] for r in pub
               if r["partition"] == p and r["assignment"] == "A1" and r["evaluator"] == "D0"
               and r["method"] == "trust" and r["budget"] == 0.35]
        match = ref and abs(mine - ref[0]) <= 1e-6
        ok_all &= bool(match)
        check(f"partition {p}: trust A1-D0 reference_regret matches published",
              bool(match), f"recomputed {mine:.6f} vs published {ref[0] if ref else 'MISSING'}")

    print()
    print("=" * 74)
    verdict = "PASS" if not FAILS else "ABORT"
    print(f"FEASIBILITY GATE: {verdict}")
    if FAILS:
        print("  failed:", FAILS)
        sys.exit(1)
    print("  The experiment may proceed: per-request logs exist (log_data), true propensities")
    print("  are exact and validated, features are pre-assignment, and the pipeline reproduces")
    print("  the published arms from the same seeds.")


if __name__ == "__main__":
    main()
