#!/usr/bin/env python3
"""P0.2 + P0.1: is the 79.6% arm-main-effect share real, and what is its interval?

Per docs/PHASE0_ANALYSIS_PLAN.md.

P0.2  permutation null  - permute ARM LABELS across items within a context, preserving each
      context's composition and both signals' base rates, breaking only the arm<->content
      association. (The v2 plan's within-cell pairing permutation was degenerate: it leaves both
      cell means, and hence C_kj, exactly unchanged. A non-degeneracy check below asserts this
      null actually moves the statistic.)
P0.2b positive control - inject a known CONTEXT-VARYING C_kj and confirm the fit attributes it to
      interaction (a power check for the estimator at the realized cell sizes).
P0.2c diagnostic       - arm share of the click table alone and the like table alone.
P0.1  bootstrap        - cluster bootstrap over USERS, refitting the decomposition every replicate.
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
B_NULL, B_BOOT, SEED = 1000, 1000, 20260810
ARMS = ['39', '9', '6', '12', '3', '20', '17', '28']


def build(data_dir):
    """Reproduce the App. I scoring-pool frame: randomized stream, in-scope arms, 8 contexts."""
    d = os.path.join(data_dir, "data")
    rnd = pd.read_csv(os.path.join(d, "log_random_4_22_to_5_08_pure.csv"),
                      usecols=["user_id", "video_id", "is_click", "is_like"])
    v = pd.read_csv(os.path.join(d, "video_features_basic_pure.csv"), usecols=["video_id", "tag"])
    u = pd.read_csv(os.path.join(d, "user_features_pure.csv"),
                    usecols=["user_id", "user_active_degree", "follow_user_num"])
    v["arm"] = v.tag.astype(str).str.split(",").str[0]
    v = v[v.arm.isin(ARMS)]
    lvl = {"full_active": 0, "high_active": 1, "middle_active": 2}
    u["a"] = u.user_active_degree.map(lambda x: lvl.get(x, 3))
    u["f"] = (u.follow_user_num > 100).astype(int)
    u["ctx"] = u.a * 2 + u.f
    # same sha1 bucket split as run_kuairand_validity.py: score pool = bucket in [65,100)
    u["bucket"] = u.user_id.astype(str).map(
        lambda s: int(hashlib.sha1(f"pool:{s}".encode()).hexdigest(), 16) % 100)
    df = rnd.merge(v[["video_id", "arm"]], on="video_id").merge(
        u[["user_id", "ctx", "bucket"]], on="user_id")
    df["k"] = df.arm.map({a: i for i, a in enumerate(ARMS)})
    return df


def decomp(df, kcol="k", value=None):
    """Two-way additive decomposition, matching run_kuairand_validity.c_decomp."""
    K, J = 8, 8
    if value is None:
        g = df.groupby([kcol, "ctx"]).agg(c=("is_click", "mean"), l=("is_like", "mean"))
        C = np.full((K, J), np.nan)
        for (k, j), r in g.iterrows():
            C[int(k), int(j)] = r.c - r.l
    else:
        g = df.groupby([kcol, "ctx"])[value].mean()
        C = np.full((K, J), np.nan)
        for (k, j), val in g.items():
            C[int(k), int(j)] = val
    ok = ~np.isnan(C)
    Cc = np.where(ok, C, 0.0)
    grand = Cc[ok].mean()
    arm = np.array([Cc[k, ok[k]].mean() - grand if ok[k].any() else 0.0 for k in range(K)])
    ctx = np.array([Cc[ok[:, j], j].mean() - grand if ok[:, j].any() else 0.0 for j in range(J)])
    fit = grand + arm[:, None] + ctx[None, :]
    sst = float(((Cc - grand)[ok] ** 2).sum())
    if sst == 0:
        return dict(share_arm=np.nan, share_ctx=np.nan, share_resid=np.nan)
    return dict(share_arm=float(((arm[:, None] * np.ones((K, J)))[ok] ** 2).sum()) / sst,
                share_ctx=float(((ctx[None, :] * np.ones((K, J)))[ok] ** 2).sum()) / sst,
                share_resid=float(((Cc - fit)[ok] ** 2).sum()) / sst)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    ap.add_argument("--pool", default="scoring", choices=["scoring", "full"])
    a = ap.parse_args()
    full = build(a.data)
    df = full[full.bucket >= 65] if a.pool == "scoring" else full
    print(f"pool={a.pool}")
    rng = np.random.default_rng(SEED)
    obs = decomp(df)
    out = {"observed": obs, "n_rows": int(len(df)), "n_users": int(df.user_id.nunique())}
    print(f"rows {len(df):,}  users {df.user_id.nunique():,}   observed arm share {obs['share_arm']:.4f}")

    # ---- P0.2 non-degeneracy check (standing rule) --------------------------------------------
    d2 = df.copy()
    d2["k"] = d2.groupby("ctx")["k"].transform(lambda s: rng.permutation(s.values))
    moved = abs(decomp(d2)["share_arm"] - obs["share_arm"]) > 1e-12
    print(f"non-degeneracy check: statistic moves under the null = {moved}")
    assert moved, "null leaves the statistic invariant - degenerate, do not use"

    # ---- P0.2 null ----------------------------------------------------------------------------
    null = []
    for _ in range(B_NULL):
        t = df.copy()
        t["k"] = t.groupby("ctx")["k"].transform(lambda s: rng.permutation(s.values))
        null.append(decomp(t)["share_arm"])
    null = np.array([x for x in null if np.isfinite(x)])
    pct = float((null < obs["share_arm"]).mean())
    out["null"] = dict(n=len(null), mean=float(null.mean()), p05=float(np.percentile(null, 5)),
                       median=float(np.median(null)), p95=float(np.percentile(null, 95)),
                       observed_percentile=pct)
    print(f"null: mean {null.mean():.4f}  median {np.median(null):.4f}  p95 {np.percentile(null,95):.4f}"
          f"   observed at percentile {pct:.3f}")

    # ---- P0.2b positive control: inject context-varying C, check it lands in interaction -------
    ctl = df.copy()
    bump = rng.normal(0, 0.05, (8, 8))                  # pure interaction, no arm structure
    bump -= bump.mean(1, keepdims=True); bump -= bump.mean(0, keepdims=True)
    ctl["is_click"] = np.clip(ctl.is_click + bump[ctl.k.values, ctl.ctx.values], 0, 1)
    c = decomp(ctl)
    out["positive_control"] = dict(injected="context-varying (pure interaction)", **c)
    print(f"positive control: arm {c['share_arm']:.4f}  ctx {c['share_ctx']:.4f}  resid {c['share_resid']:.4f}"
          f"   (resid should RISE vs observed {obs['share_resid']:.4f})")

    # ---- P0.2c diagnostic: click table and like table separately ------------------------------
    dc = decomp(df, value="is_click"); dl = decomp(df, value="is_like")
    out["click_only"] = dc; out["like_only"] = dl
    print(f"click-only arm share {dc['share_arm']:.4f}   like-only arm share {dl['share_arm']:.4f}")

    # ---- P0.1 cluster bootstrap over users, refitting each replicate --------------------------
    users = df.user_id.unique()
    idx = {u: g for u, g in df.groupby("user_id").indices.items()}
    boot = []
    for _ in range(B_BOOT):
        pick = rng.choice(users, len(users), replace=True)
        rows = np.concatenate([idx[u] for u in pick])
        boot.append(decomp(df.iloc[rows])["share_arm"])
    boot = np.array([x for x in boot if np.isfinite(x)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["bootstrap"] = dict(n=len(boot), mean=float(boot.mean()), lo=float(lo), hi=float(hi))
    print(f"bootstrap: {obs['share_arm']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    trig = ("KEEP_IN_ABSTRACT" if lo >= 0.60 else
            "DELETE_FROM_ABSTRACT" if lo < 0.50 else "APPENDIX_ONLY")
    out["trigger_P01"] = trig
    out["trigger_P02"] = ("WITHDRAW" if obs["share_arm"] <= np.median(null) else
                          "STANDS" if obs["share_arm"] > np.percentile(null, 95) else "WEAK")
    print(f"TRIGGER P0.1 {trig}   TRIGGER P0.2 {out['trigger_P02']}")

    out["pool"] = a.pool
    p = os.path.join(RES, f"kuairand_share_tests_{a.pool}.json")
    json.dump(out, open(p, "w"), indent=1); print("wrote", p)


if __name__ == "__main__":
    main()
