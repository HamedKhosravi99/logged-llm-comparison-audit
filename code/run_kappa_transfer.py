#!/usr/bin/env python3
"""P0.6: does the audit-worth diagnostic's threshold transfer to RouterBench?

Three reviewers flagged an internal contradiction: Alg. 1 line 5 calls kappa "externally
validated"; S5 says its threshold was "applied unchanged to RouterBench"; S6.2 says it "does not
transfer well enough to deploy"; and App. C.10 reports only in-sample primary-corpus numbers
(AUC 0.788, threshold 1.171, accuracy 0.75). No cross-corpus number exists anywhere.

This computes the missing number: apply the LLMRouterBench-fitted threshold, unchanged, to
RouterBench, and report AUC and accuracy there.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random A1 statistic exceeds a random A0 statistic)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    prim = json.load(open(os.path.join(RES, "tae_diagnostics.json")))
    d2 = prim["D2_assignment_invalidity_test"]
    thr = d2["threshold"]

    rep_path = os.path.join(RES, "tae_replication.json")
    rep = json.load(open(rep_path))

    # locate per-partition seed statistics for RouterBench, if the replication stored them
    def find(o, want, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from find(v, want, f"{p}.{k}" if p else k)
        elif isinstance(o, list) and want in p.lower():
            yield p, o

    cands = dict(find(rep, "stat"))
    out = dict(primary=dict(threshold=thr, auc=d2["auc"], accuracy=d2["accuracy"],
                            stat_A0=d2["stat_A0"], stat_A1=d2["stat_A1"],
                            n_seed_cells=d2["n_seed_cells"]),
                routerbench_keys_found=sorted(cands.keys()))

    if not cands:
        out["result"] = "NOT_COMPUTABLE_FROM_STORED_ARTIFACTS"
        out["reading"] = (
            "results/tae_replication.json does not store the per-partition kappa statistics for "
            "RouterBench, so the transferred AUC/accuracy cannot be recovered without re-running "
            "the replication. The paper therefore reports no cross-corpus performance for kappa "
            "anywhere, while Alg. 1 calls it 'externally validated' and S5 says the threshold was "
            "'applied unchanged to RouterBench'. Under the plan's decision rule this is the "
            "'does not transfer' branch by default: the claim is unsupported by any stored "
            "artifact and must be removed from Alg. 1 unless the replication is re-run.")
    p = os.path.join(RES, "kappa_transfer.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1)[:1200])
    print("wrote", p)


if __name__ == "__main__":
    main()
