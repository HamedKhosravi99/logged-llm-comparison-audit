"""Second-evaluator comparison. Are the judge's failures a property of gpt-4o-mini, or of
automatic judges generally?

Both judges are scored with the SAME functions (`agreement`, `position_bias`, `length_bias`
imported from `lmarena_analyse`) on the SAME battles: the API arm is restricted to exactly the
subsample the local judge covered, so nothing here is a comparison across different item sets.

Reads  results/lmarena_local_judge.json
Writes results/lmarena_judge_compare.json
"""
import os, json
import numpy as np
import pandas as pd

from lmarena_analyse import RES, load, agreement, position_bias, length_bias, boot_ci

import sys, glob

def _pick(arg=None):
    """Which local-judge file to compare. Defaults to the newest one."""
    if arg:
        return arg if os.path.exists(arg) else os.path.join(RES, arg)
    cands = sorted(glob.glob(os.path.join(RES, "lmarena_local_judge_*.json")), key=os.path.getmtime)
    if not cands:
        raise SystemExit("no local-judge results found; run lmarena_local_judge.py first")
    return cands[-1]

LOCAL = _pick(sys.argv[1] if len(sys.argv) > 1 else None)
OUT = os.path.join(RES, "lmarena_judge_compare_" +
                   os.path.basename(LOCAL).replace("lmarena_local_judge_", ""))


def _local_frame():
    d = json.load(open(LOCAL))
    f = pd.DataFrame(d["rows"])
    f["human"] = f["human_winner"].map({"model_a": "A", "model_b": "B",
                                        "tie": "tie", "both_bad": "both_bad"})
    return d["meta"], f


def main():
    meta, loc = _local_frame()
    api = load()
    shared = set(loc["battle_id"]) & set(api["battle_id"])
    loc = loc[loc["battle_id"].isin(shared)].copy()
    api = api[api["battle_id"].isin(shared)].copy()
    print(f"comparing two judges on the same {len(shared):,} battles")
    print(f"  local judge: {meta['judge']}  ({meta.get('wall_hours',0):.2f}h, "
          f"${meta.get('api_cost_usd',0):.2f})")
    print(f"  parse failures: {meta.get('parse_failures')}\n")

    res = {"n_shared": len(shared), "local_judge": meta["judge"], "api_judge": "gpt-4o-mini"}
    print(f"{'metric':34}{'gpt-4o-mini':>14}{'local':>14}   replicates?")
    for label, fn, key in [
        ("agreement with human",        agreement,     "exact_4class"),
        ("agreement, decisive only",    agreement,     "decisive_only"),
        ("position-flip rate",          position_bias, "flip_rate"),
        ("prefers-first",               position_bias, "prefers_first"),
        ("length effect on the judge",  length_bias,   "longer_effect"),
        ("human effect on the judge",   length_bias,   "human_effect"),
    ]:
        a, b = fn(api).get(key, np.nan), fn(loc).get(key, np.nan)
        alo, ahi = boot_ci(api, lambda d, f=fn, k=key: f(d).get(k, np.nan))
        blo, bhi = boot_ci(loc, lambda d, f=fn, k=key: f(d).get(k, np.nan))
        # "replicates" = same sign, and the local point estimate is inside the API interval
        # or the two intervals overlap
        overlap = not (bhi < alo or blo > ahi)
        same_sign = np.sign(a) == np.sign(b)
        verdict = "yes" if (overlap and same_sign) else ("same direction" if same_sign else "NO")
        res[key] = dict(api=float(a), api_ci=[alo, ahi], local=float(b), local_ci=[blo, bhi],
                        replicates=verdict)
        print(f"{label:34}{a:>14.3f}{b:>14.3f}   {verdict}")

    # judge-vs-judge agreement on the shared items
    m = api.set_index("battle_id")["judge_winner_original"]
    l = loc.set_index("battle_id")["judge_winner_original"]
    both = pd.concat([m.rename("api"), l.rename("loc")], axis=1).dropna()
    jj = float((both["api"] == both["loc"]).mean())
    res["judge_vs_judge_agreement"] = dict(value=jj, n=int(len(both)))
    print(f"\n  the two judges agree with each other on {jj:.3f} of {len(both):,} battles")

    # do they fail on the same items?
    both["api_right"] = (both["api"] == api.set_index("battle_id")["human"].reindex(both.index))
    both["loc_right"] = (both["loc"] == loc.set_index("battle_id")["human"].reindex(both.index))
    tbl = pd.crosstab(both["api_right"], both["loc_right"])
    res["error_overlap"] = {str(k): {str(k2): int(v2) for k2, v2 in v.items()}
                            for k, v in tbl.to_dict().items()}
    print("\n  correctness overlap (rows: gpt-4o-mini right, cols: local right)")
    print(tbl.to_string().replace("\n", "\n    "))

    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
