"""Generate the bundled synthetic example. Run from the repository root:

    python examples/make_examples.py

The scenario is shaped like the paper's cost-reduction replay -- a quality-first incumbent, a
selectively routed log, and one request type where a cheaper model is genuinely good enough --
but every value here is synthetic. No company data and no paper result is embedded.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = ["premium-large", "balanced-mid", "compact-small"]
TYPES = ["billing", "technical", "account"]

# True per (type, model) quality and price of the synthetic world. The scenario is built so
# that each request type exercises a different outcome:
#   billing   - a cheaper model is genuinely good enough      -> valid cost-saving switch
#   account   - the judge overrates a cheaper model            -> quality-violating switch
#   technical - no cheaper model is close                      -> retain the incumbent
TRUE_Q = {
    ("billing", "premium-large"): 0.82, ("billing", "balanced-mid"): 0.815,
    ("billing", "compact-small"): 0.62,
    ("technical", "premium-large"): 0.75, ("technical", "balanced-mid"): 0.58,
    ("technical", "compact-small"): 0.41,
    ("account", "premium-large"): 0.88, ("account", "balanced-mid"): 0.80,
    ("account", "compact-small"): 0.85,
}
PRICE = {"premium-large": 0.0180, "balanced-mid": 0.0052, "compact-small": 0.0011}
# the automatic evaluator rewards the terse compact model beyond its true quality
JUDGE_BONUS = {"premium-large": 0.0, "balanced-mid": 0.0, "compact-small": 0.05}
INCUMBENT = "premium-large"
EXPLORE = 0.15
N_LOG = 3000
N_AUDIT_PER_CELL = 80


def main() -> None:
    rng = np.random.default_rng(20260730)

    # ---- routed log: the incumbent serves most traffic, exploration covers the rest
    rows = []
    for i in range(N_LOG):
        t = TYPES[rng.integers(len(TYPES))]
        m = INCUMBENT if rng.random() >= EXPLORE else MODELS[rng.integers(len(MODELS))]
        judged = np.clip(TRUE_Q[(t, m)] + JUDGE_BONUS[m] + rng.normal(0, 0.06), 0, 1)
        rows.append(dict(request_id=f"req-{i:05d}", request_type=t, model=m,
                         score=round(float(judged), 4),
                         request_cost_usd=round(float(PRICE[m] * rng.uniform(0.9, 1.1)), 6),
                         incumbent=INCUMBENT))
    log = pd.DataFrame(rows)
    log.to_csv(os.path.join(HERE, "routed_log.csv"), index=False)

    # ---- prices, per request type
    pd.DataFrame([dict(request_type=t, model=m, expected_cost_usd=PRICE[m])
                  for t in TYPES for m in MODELS]
                 ).to_csv(os.path.join(HERE, "model_prices.csv"), index=False)

    # ---- randomized audit: every cheaper-than-incumbent cell, sampled independently
    arows = []
    n = 0
    for t in TYPES:
        for m in MODELS:
            if PRICE[m] >= PRICE[INCUMBENT] and m != INCUMBENT:
                continue
            for _ in range(N_AUDIT_PER_CELL):
                judged = np.clip(TRUE_Q[(t, m)] + JUDGE_BONUS[m] + rng.normal(0, 0.06), 0, 1)
                arows.append(dict(request_id=f"aud-{n:05d}", request_type=t, model=m,
                                  audit_score=round(float(judged), 4),
                                  audit_cost_usd=round(float(PRICE[m]), 6)))
                n += 1
    pd.DataFrame(arows).to_csv(os.path.join(HERE, "randomized_audit.csv"), index=False)

    # ---- verified outcomes on a subset: the designated target, free of the judge's bonus
    vrows = []
    for i, r in enumerate(arows):
        if i % 3:
            continue
        v = np.clip(TRUE_Q[(r["request_type"], r["model"])] + rng.normal(0, 0.07), 0, 1)
        vrows.append(dict(request_id=r["request_id"], request_type=r["request_type"],
                          model=r["model"], verified_outcome=round(float(v), 4),
                          verification_cost_usd=0.25))
    pd.DataFrame(vrows).to_csv(os.path.join(HERE, "verified_outcomes.csv"), index=False)

    # ---- traffic weights
    share = log["request_type"].value_counts(normalize=True).sort_index()
    pd.DataFrame(dict(request_type=share.index,
                      traffic_share=share.to_numpy().round(6))
                 ).to_csv(os.path.join(HERE, "traffic.csv"), index=False)

    print(f"routed_log.csv        {len(log):,} rows")
    print(f"model_prices.csv      {len(TYPES) * len(MODELS)} rows")
    print(f"randomized_audit.csv  {len(arows):,} rows")
    print(f"verified_outcomes.csv {len(vrows):,} rows")
    print("traffic.csv           %d rows" % len(share))


if __name__ == "__main__":
    main()
