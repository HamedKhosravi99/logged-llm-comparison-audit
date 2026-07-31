"""Priority 3 (section 4): full quality-tolerance and payback map.

delta in {0.000,0.005,0.010,0.015,0.020,0.030,0.050}; horizons 1k..1M; on LLMRouterBench,
RouterBench and the tau2 replay. Overrides run_econ's frozen grids at import time rather than
editing it, so the committed results stay reproducible. No new API calls.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "p3_tolerance_map.json")

DELTAS = (0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050)
HORIZONS = (1_000, 2_000, 5_000, 10_000, 100_000, 1_000_000)
PRIMARY_DELTA = 0.010


def tau2_map():
    """The tau2 replay: sweep delta over the complete matrix, decide, and price it."""
    m = pd.read_parquet(os.path.join(HERE, "..", "data", "processed", "tau2_complete_matrix.parquet"))
    tc = m.groupby(["domain", "task_id", "agent_model"]).agg(
        success=("task_success", "mean"), cost=("agent_cost_usd", "mean")).reset_index()
    cell = tc.groupby(["domain", "agent_model"]).agg(success=("success", "mean"),
                                                     cost=("cost", "mean")).reset_index()
    doms = sorted(cell.domain.unique())
    out = []
    for d in DELTAS:
        pol, q, c, viol = {}, [], [], []
        for dom in doms:
            sub = cell[cell.domain == dom]
            best = sub.success.max()
            ok = sub[sub.success >= best - d].sort_values(["cost", "success"], ascending=[True, False])
            pick = ok.iloc[0]
            pol[dom] = pick.agent_model
            q.append(pick.success); c.append(pick.cost)
            viol.append(float(pick.success < best - d))     # true tolerance violation on truth
        out.append(dict(corpus="tau2", delta=d, quality=float(np.mean(q)),
                        cost_per_1k=float(np.mean(c) * 1000),
                        violation_rate=float(np.mean(viol)), policy=pol))
    return out


def corpus_map(corpus, partitions):
    """Wider delta and horizon grids on a routing corpus, by overriding run_econ's frozen
    constants at import time. run_econ.py itself is untouched, so the committed results stay
    reproducible."""
    sys.path.insert(0, HERE)
    import run_econ
    run_econ.DELTAS, run_econ.HORIZONS = DELTAS, HORIZONS
    rows = []
    for p in range(partitions):
        rows += run_econ.one_partition(p, corpus, 0.15, 30, 6, "D1", 0.35)
        print(f"    {corpus} partition {p+1}/{partitions}", flush=True)
    out = []
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        for d in DELTAS:
            k = f"qc_delta_{d}"       # decide on truth under tolerance d
            kl = f"qcLOG_delta_{d}"   # decide on the free log
            kc = f"qcCASE_delta_{d}"  # decide on routing-adjusted scores
            for tag, key in (("truth", k), ("log", kl), ("case", kc)):
                if r.get("method") == key and r.get("assignment") == "A1":
                    by[(d, tag)].append(r)
    for (d, tag), rs in sorted(by.items()):
        if not rs: continue
        out.append(dict(corpus=corpus, delta=float(d), basis=tag,
                        quality=float(np.mean([x["quality"] for x in rs])),
                        cost_per_1k=float(np.mean([x["deployment_cost_per_1k"] for x in rs])),
                        wrong_model_rate=float(np.mean([x["wrong_model_rate"] for x in rs])),
                        audit_usd=float(np.mean([x.get("audit_usd") or 0 for x in rs]))))
    return out


def tau2_only():
    res = dict(deltas=list(DELTAS), horizons=list(HORIZONS), primary_delta=PRIMARY_DELTA)
    res["tau2"] = tau2_map()
    print("  tau2 tolerance map:")
    print(f"    {'delta':>7}{'quality':>10}{'$/1k':>10}{'violation':>11}  policy")
    for r in res["tau2"]:
        star = " <- primary" if abs(r["delta"] - 0.02) < 1e-9 else ""
        print(f"    {r['delta']:>7.3f}{r['quality']:>10.3f}{r['cost_per_1k']:>10.2f}"
              f"{r['violation_rate']:>11.2f}  {r['policy']}{star}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  wrote {os.path.abspath(OUT)}")

def run_corpora():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for corpus, parts in (("llmrouterbench", 30), ("routerbench", 20)):
        res[corpus] = corpus_map(corpus, parts)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"  {corpus}: {len(res[corpus])} tolerance rows")
    print(f"  wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    import sys
    if "--corpora" in sys.argv:
        run_corpora()
    else:
        tau2_only()
