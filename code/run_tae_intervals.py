"""95% confidence intervals for the confirmatory comparisons, under three resampling schemes.

The paper's default `+-` is a paired standard error over prompt-cohort partitions. A partition
resamples cohort membership, the context clustering and the router inside ONE fixed corpus, so that
interval covers sensitivity to the split and the acquisition process -- not uncertainty over which
prompts are in the corpus, and not uncertainty over which benchmarks are in the suite. Those are
different questions and they get different intervals here:

  partition     paired t interval over the 30 partitions (and its percentile-bootstrap twin).
                Question: would a different split and acquisition draw give the same answer?
  prompt        nonparametric bootstrap of the SCORING COHORT prompts, with each run's deployed
                policy held fixed. The target surface mu^ref and the target policy pi^ref are
                re-estimated on every replicate, so this propagates the uncertainty in the thing the
                paper scores against. Question: is the answer an artifact of which prompts happened
                to land in the scoring cohort?
  slice         cluster bootstrap over the FIVE benchmark slices, resampled with replacement, using
                each run's slice-restricted regret. Question: would the answer survive a different
                benchmark suite? With only five clusters this interval is coarse and wide by
                construction -- that is the point, and it should not be reported as if it were
                sharp.

    python run_tae_intervals.py [--boot 10000] [--prompt-boot 2000]
"""
import argparse
import collections
import json
import os

import numpy as np
from scipy import stats

import case_core as cc
from eval_validity import assignment as A
from eval_validity import evaluators as EV
from eval_validity import metrics as MT
from eval_validity.grid import build_base

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DEFAULT_OUT = os.path.join(RES, "tae_intervals.json")
SLICES = ["simpleqa", "mmlupro", "hle", "livecodebench", "arenahard"]

# the four pre-specified confirmatory contrasts, plus the two headline levels they refer to
CONTRASTS = [
    ("A1-E0: CASE - trust", "A1", "E0", "case", "trust"),
    ("A1-E0: CASE - R-EPIG_corr", "A1", "E0", "case", "epig_corr"),
    ("A1-E1: CASE - trust", "A1", "E1", "case", "trust"),
    ("A1-E1: CASE - population-operational oracle", "A1", "E1", "case", "oracle_sel"),
    ("A1-E1: measurement floor (oracle_sel - oracle_ref)", "A1", "E1", "oracle_sel", "oracle_ref"),
    ("A0-E0: CASE - trust (cost of auditing)", "A0", "E0", "case", "trust"),
]


def paired_t(d, alpha=0.05):
    n = len(d)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return float(d.mean() - h), float(d.mean() + h)


def boot_ci(d, rng, reps, alpha=0.05):
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def per_partition(rows, budget=0.35):
    """series[(A, E, method)][field] -> one value per partition (seeds averaged first)."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["budget"] == budget:
            g[(r["assignment"], r["evaluator"], r["method"])][r["partition"]].append(r)
    out = {}
    for k, byp in g.items():
        parts = sorted(byp)
        out[k] = dict(
            partitions=parts,
            reference_regret=np.array([np.mean([q["reference_regret"] for q in byp[p]]) for p in parts]),
            slice_regret=np.array([[np.mean([q["slice_regret"][s] for q in byp[p]]) for s in SLICES]
                                   for p in parts]))
    return out


def prompt_bootstrap(contrasts, reps, seed=0, budget=0.35, n_read=30, eps=0.15, n_parts=30):
    """Re-score every run's deployed policy on bootstrap resamples of the scoring cohort.

    The policies are held fixed (they never saw the scoring cohort); mu^ref, pi^ref and therefore the
    regret are recomputed on each resample, so the interval covers uncertainty in the designated
    target itself rather than in the audit.
    """
    rng = np.random.default_rng(seed)
    need = sorted({m for _, _, _, a, b in contrasts for m in (a, b)})
    # per partition: the scoring-cohort outcomes, and each method's deployed policy per seed
    store = []
    for p in range(n_parts):
        base = build_base(p)
        cellsA1, n_kj, _ = A.make_selective_log(base, p, eps=eps)
        cellsA0 = A.make_matched_common_population_log(base, p, n_kj)
        supp = A.support(n_kj, base, n_read=n_read)
        sc = np.where(base["coh"] == "score")[0]
        rec = dict(Q=base["Q"][sc], ctx=base["ctx"][sc], supp=supp, pols={})
        for aname, cells in (("A0", cellsA0), ("A1", cellsA1)):
            for cond in ("E0", "E1"):
                surf = EV.build_surfaces(base, cells, EV.distortions(base, cond), n_read)
                B = max(base["K"], int(round(budget * int(supp.sum()))))
                for m in need:
                    pols = []
                    for s in range(6):
                        sd = 10_000 * p + 100 * int(budget * 100) + s
                        if m == "trust":
                            V = surf["M"]
                        elif m == "oracle_sel":
                            V = surf["Ef_op"]
                        elif m == "oracle_ref":
                            V = surf["Ef_ref"]
                        else:
                            mode = {"case": "case", "epig_corr": "epig_deconf"}[m]
                            V, _ = cc.allocate(mode, None, surf["M"], surf["cellq"], supp, B, sd, n_read)
                        pols.append(MT.deployed_policy(V, supp))
                    rec["pols"][(aname, cond, m)] = np.array(pols)
        store.append(rec)
        if (p + 1) % 10 == 0:
            print(f"    prompt-bootstrap setup {p+1}/{n_parts}", flush=True)

    K = store[0]["Q"].shape[1]
    J = store[0]["supp"].shape[1]
    keys = sorted({(a, c, m) for _, a, c, ma, mb in contrasts for m in (ma, mb)
                   for a, c in [(a, c)]})
    # one set of replicates shared by every contrast (common random numbers), so the intervals are
    # directly comparable and each replicate does the resampling work once rather than once per pair
    draws = {k: np.empty((reps, len(store))) for k in keys}
    for b in range(reps):
        for pi, rec in enumerate(store):
            n = len(rec["Q"])
            take = rng.integers(0, n, size=n)
            Qb, ctxb, supp = rec["Q"][take], rec["ctx"][take], rec["supp"]
            mu = np.full((K, J), np.nan)
            for j in range(J):
                sel = ctxb == j
                if sel.any():
                    mu[:, j] = Qb[sel].mean(0)
            pol_ref = np.full(J, -1, int)
            for j in range(J):
                ks = np.where(supp[:, j] & np.isfinite(mu[:, j]))[0]
                if len(ks):
                    pol_ref[j] = int(ks[int(np.argmax(mu[ks, j]))])
            best = np.array([mu[pol_ref[j], j] if pol_ref[j] >= 0 else np.nan for j in range(J)])
            for k in keys:
                pols = rec["pols"][k]
                v = []
                for pol in pols:
                    g = [best[j] - mu[pol[j], j] for j in range(J)
                         if pol[j] >= 0 and pol_ref[j] >= 0 and np.isfinite(mu[pol[j], j])]
                    v.append(np.mean(g) if g else np.nan)
                draws[k][b, pi] = np.nanmean(v)
        if (b + 1) % 200 == 0:
            print(f"    replicate {b+1}/{reps}", flush=True)
    out = {}
    for name, aname, cond, ma, mb in contrasts:
        d = draws[(aname, cond, ma)].mean(1) - draws[(aname, cond, mb)].mean(1)
        out[name] = (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--prompt-boot", type=int, default=2000)   # matches the shipped results
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    D = json.load(open(os.path.join(RES, "tae_validity.json")))
    ser = per_partition(D["rows"])

    res = {}
    for name, aname, cond, ma, mb in CONTRASTS:
        d = ser[(aname, cond, ma)]["reference_regret"] - ser[(aname, cond, mb)]["reference_regret"]
        sl = ser[(aname, cond, ma)]["slice_regret"] - ser[(aname, cond, mb)]["slice_regret"]
        # cluster bootstrap over the five benchmark slices (coarse by construction: 5 clusters)
        pick = rng.integers(0, len(SLICES), size=(args.boot, len(SLICES)))
        slice_draws = sl[:, pick].mean(0).mean(1)
        res[name] = dict(
            mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n_partitions=len(d),
            ci_partition_t=paired_t(d), ci_partition_boot=boot_ci(d, rng, args.boot),
            ci_slice_cluster_boot=(float(np.quantile(slice_draws, 0.025)),
                                   float(np.quantile(slice_draws, 0.975))),
            slice_mean=float(sl.mean()))

    # gate contrast lives in the gate rows
    gg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in D["gate_rows"]:
        if r["budget"] == 0.35:
            gg[(r["evaluator"], r["arm"])][r["partition"]].append(r["false_switch"])
    for cond in ("E1", "E2"):
        parts = sorted(gg[(cond, "same_evaluator")])
        d = np.array([np.mean(gg[(cond, "same_evaluator")][p]) for p in parts]) - \
            np.array([np.mean(gg[(cond, "reference_channel")][p]) for p in parts])
        res[f"A1-{cond}: false switch, same-evaluator - reference-channel"] = dict(
            mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n_partitions=len(d),
            ci_partition_t=paired_t(d), ci_partition_boot=boot_ci(d, rng, args.boot),
            ci_slice_cluster_boot=None, slice_mean=None)

    print("running the prompt-level bootstrap (policies fixed, scoring cohort resampled) ...",
          flush=True)
    pb = prompt_bootstrap(CONTRASTS, args.prompt_boot)
    for k, v in pb.items():
        res[k]["ci_prompt_boot"] = v

    out = dict(spec=dict(budget=0.35, boot=args.boot, prompt_boot=args.prompt_boot,
                         generated_by="code/run_tae_intervals.py",
                         note="'+-' elsewhere in the paper is a paired SE over partitions"),
               contrasts=res)
    json.dump(out, open(args.out, "w"), indent=1)

    w = max(len(k) for k in res)
    print(f"\n{'contrast':<{w}}  {'mean':>9}  {'95% CI (partition, t)':>26}  "
          f"{'95% CI (prompt boot)':>24}  {'95% CI (slice clusters)':>26}")
    for k, v in res.items():
        pt = f"[{v['ci_partition_t'][0]:+.4f}, {v['ci_partition_t'][1]:+.4f}]"
        pb_ = (f"[{v['ci_prompt_boot'][0]:+.4f}, {v['ci_prompt_boot'][1]:+.4f}]"
               if v.get("ci_prompt_boot") else "--")
        sb = (f"[{v['ci_slice_cluster_boot'][0]:+.4f}, {v['ci_slice_cluster_boot'][1]:+.4f}]"
              if v.get("ci_slice_cluster_boot") else "--")
        print(f"{k:<{w}}  {v['mean']:+9.4f}  {pt:>26}  {pb_:>24}  {sb:>26}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
