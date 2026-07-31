"""Work Package B, steps B1 and B2. Build the LMArena evaluation sample.

Splitting is by `evaluation_session_id`, never by row, so the same user session cannot appear on
both sides of a split. Final-test ids are written once and never rewritten, so no prompt or
threshold can be tuned against them.

    python lmarena_sample.py --n 20000
"""
import argparse, hashlib, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
SPLIT = dict(calibration=5_000, validation=5_000, test=10_000)


def _tokens(conv):
    """Crude token proxy for the length-bias stratum. Whitespace count over the assistant turns."""
    try:
        turns = conv if isinstance(conv, (list, np.ndarray)) else json.loads(conv)
        return sum(len(str(t.get("content", "")).split())
                   for t in turns if str(t.get("role", "")).lower() == "assistant")
    except Exception:
        return len(str(conv).split())


def build(n_total, seed=0, shard="lmarena_shard0.parquet"):
    df = pd.read_parquet(os.path.join(DATA, shard))
    df = df[df["winner"].isin(["model_a", "model_b", "tie", "both_bad"])].copy()

    df["tok_a"] = df["conversation_a"].map(_tokens)
    df["tok_b"] = df["conversation_b"].map(_tokens)
    df["len_diff"] = df["tok_a"] - df["tok_b"]
    df["len_q"] = pd.qcut(df["len_diff"].rank(method="first"), 5, labels=False)
    df["pair"] = [tuple(sorted((a, b))) for a, b in zip(df["model_a"], df["model_b"])]
    df["conv_turns"] = df["conversation_a"].map(
        lambda c: len(c) if isinstance(c, (list, np.ndarray)) else 1)

    # stratify on the plan's variables, then take a proportional sample of whole sessions
    df["stratum"] = (df["is_code"].astype(str) + "|" + df["winner"] + "|"
                     + df["len_q"].astype(str) + "|" + (df["conv_turns"] > 1).astype(str))
    rng = np.random.default_rng(seed)
    sess = df.groupby("evaluation_session_id")["stratum"].first().reset_index()
    take = min(n_total, len(df))
    frac = take / len(df)
    keep = []
    for st, g in sess.groupby("stratum"):
        k = int(round(len(g) * frac))
        if k:
            keep += list(rng.choice(g["evaluation_session_id"].values,
                                    size=min(k, len(g)), replace=False))
    sub = df[df["evaluation_session_id"].isin(set(keep))].copy()

    # session-level split, deterministic by hash so it is reproducible without storing a seed
    def bucket(sid):
        h = int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) % 100
        return "calibration" if h < 25 else ("validation" if h < 50 else "test")
    sub["split"] = sub["evaluation_session_id"].map(bucket)

    out = os.path.join(DATA, "lmarena_sample.parquet")
    sub.to_parquet(out)
    frozen = os.path.join(DATA, "lmarena_test_ids.json")
    if not os.path.exists(frozen):
        json.dump(sorted(sub.loc[sub["split"] == "test", "id"].astype(str).tolist()),
                  open(frozen, "w"))
        note = "written now and never rewritten"
    else:
        prev = set(json.load(open(frozen)))
        cur = set(sub.loc[sub["split"] == "test", "id"].astype(str))
        note = "unchanged" if prev == cur else f"WARNING drifted, {len(prev ^ cur)} ids differ"

    print(f"sampled {len(sub):,} battles from {len(df):,} in the shard")
    print("  split by session:", sub["split"].value_counts().to_dict())
    print("  sessions:", sub["evaluation_session_id"].nunique())
    print("  human winner:", sub["winner"].value_counts().to_dict())
    print("  is_code:", sub["is_code"].value_counts().to_dict())
    print("  distinct model pairs:", sub["pair"].nunique())
    print(f"  no session spans two splits:",
          bool((sub.groupby('evaluation_session_id')['split'].nunique() == 1).all()))
    print(f"  frozen test ids -> {frozen} ({note})")
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args().n, ap.parse_args().seed)
