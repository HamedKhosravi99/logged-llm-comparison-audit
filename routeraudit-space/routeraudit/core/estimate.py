"""The statistical core: surfaces, diagnostic, acquisition priority, pooling, economics.

No I/O, no global state, no network. Every function takes explicit inputs and returns plain
Python or numpy values. The interface layer is not allowed to compute anything itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

VAR_FLOOR = 1e-8


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF, Acklam's rational approximation plus one Halley step.

    Implemented here rather than imported from SciPy so the package has no compiled dependency
    and can run in a browser (Pyodide) build. Agreement with the SciPy implementation is better
    than 1e-12 over the range this package uses; a test enforces that where SciPy is installed.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf requires 0 < p < 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return float(x - u / (1 + x * u / 2))




@dataclass
class Surfaces:
    """Per (model, request_type) cell summaries built from one table."""
    models: list[str]
    types: list[str]
    n: np.ndarray            # logged row count
    mean: np.ndarray         # logged mean score
    var: np.ndarray          # variance of the logged mean
    cost: np.ndarray         # expected serving cost per request
    traffic: np.ndarray      # traffic share per request type, sums to 1
    supported: np.ndarray    # boolean mask, n >= minimum_support

    @property
    def K(self) -> int:
        return len(self.models)

    @property
    def J(self) -> int:
        return len(self.types)


def build_surfaces(log: pd.DataFrame, prices: pd.DataFrame, minimum_support: int,
                   traffic: pd.DataFrame | None = None) -> Surfaces:
    models = sorted(log["model"].unique())
    types = sorted(log["request_type"].unique())
    mi = {m: i for i, m in enumerate(models)}
    ji = {t: j for j, t in enumerate(types)}
    K, J = len(models), len(types)

    n = np.zeros((K, J), int)
    mean = np.full((K, J), np.nan)
    var = np.full((K, J), np.nan)
    for (m, t), g in log.groupby(["model", "request_type"], sort=False):
        k, j = mi[m], ji[t]
        v = g["score"].to_numpy(float)
        n[k, j] = len(v)
        mean[k, j] = float(v.mean())
        var[k, j] = max(float(v.var(ddof=1) / len(v)) if len(v) > 1 else 0.25 / len(v), VAR_FLOOR)

    cost = np.full((K, J), np.nan)
    for r in prices.itertuples():
        if r.model in mi and r.request_type in ji:
            cost[mi[r.model], ji[r.request_type]] = float(r.expected_cost_usd)

    if traffic is not None:
        w = np.zeros(J)
        for r in traffic.itertuples():
            if r.request_type in ji:
                w[ji[r.request_type]] = float(r.traffic_share)
    else:
        w = np.array([float((log["request_type"] == t).sum()) for t in types])
    w = w / w.sum() if w.sum() > 0 else np.full(J, 1.0 / J)

    return Surfaces(models, types, n, mean, var, cost, w, n >= minimum_support)


def assignment_diagnostic(s: Surfaces) -> dict:
    """Is the routed log imbalanced enough that a randomized audit is worth considering?

    Reports the coefficient of variation of logged counts across supported cells, and the mean
    standardized dispersion of logged scores within a request type. Neither is a test; both are
    descriptive signals, which is how the paper reports the corresponding statistic.
    """
    z = []
    for j in range(s.J):
        ks = np.where(s.supported[:, j])[0]
        if len(ks) < 2:
            continue
        v = s.mean[ks, j]
        sd = v.std(ddof=1)
        if sd > 0:
            z.append(float(np.sqrt(np.mean(((v - v.mean()) / sd) ** 2))))
    disp = float(np.mean(z)) if z else float("nan")
    if s.supported.any():
        cnt = s.n[s.supported].astype(float)
        imbalance = float(cnt.std() / max(cnt.mean(), 1e-9))
    else:
        imbalance = float("nan")
    if not np.isfinite(imbalance):
        verdict = "Insufficient evidence"
    elif imbalance > 0.5:
        verdict = "Selective assignment appears important"
    else:
        verdict = "Selective assignment does not appear important"
    return dict(count_imbalance=imbalance, score_dispersion=disp, verdict=verdict,
                supported_pairs=int(s.supported.sum()), total_pairs=int(s.K * s.J),
                unobserved_pairs=int((s.n == 0).sum()))


def incumbent_policy(s: Surfaces, declared: dict[str, str] | None = None) -> np.ndarray:
    """Which model currently serves each request type: the declared incumbent if the log
    carries one, otherwise the model that served the most logged requests."""
    pol = np.full(s.J, -1, int)
    for j in range(s.J):
        if declared and s.types[j] in declared:
            m = declared[s.types[j]]
            if m in s.models:
                pol[j] = s.models.index(m)
                continue
        col = s.n[:, j]
        if col.max() > 0:
            pol[j] = int(np.argmax(col))
    return pol


def cheaper_candidates(s: Surfaces, inc: np.ndarray) -> list[dict]:
    """Models priced below the incumbent in a request type. Eligibility uses price only, so it
    never depends on any outcome."""
    out = []
    for j in range(s.J):
        i = inc[j]
        if i < 0 or not np.isfinite(s.cost[i, j]):
            continue
        for k in range(s.K):
            if k == i or not np.isfinite(s.cost[k, j]):
                continue
            if s.cost[k, j] < s.cost[i, j]:
                out.append(dict(request_type=s.types[j], candidate=s.models[k],
                                incumbent=s.models[i],
                                candidate_cost_usd=float(s.cost[k, j]),
                                incumbent_cost_usd=float(s.cost[i, j]),
                                saving_per_request_usd=float(s.cost[i, j] - s.cost[k, j]),
                                logged_rows=int(s.n[k, j]),
                                supported=bool(s.supported[k, j])))
    return out


def audit_queue(s: Surfaces, inc: np.ndarray, tolerance: float, budget_usd: float,
                readout_size: int) -> tuple[pd.DataFrame, float]:
    """Which (model, request_type) comparisons to buy, cheapest-informative first.

    Priority rises with traffic share and with how close the pair sits to the decision boundary,
    and falls with what the readout costs. Pairs are added until the budget is exhausted.
    """
    rows = []
    for j in range(s.J):
        i = inc[j]
        if i < 0:
            continue
        ks = [k for k in range(s.K) if np.isfinite(s.cost[k, j])]
        if len(ks) < 2:
            continue
        obs = [k for k in ks if np.isfinite(s.mean[k, j])]
        best = max((s.mean[k, j] for k in obs), default=np.nan)
        for k in ks:
            if k == i:
                continue
            if s.cost[k, j] >= s.cost[i, j]:
                continue                       # only cheaper models can produce a saving
            price = readout_size * float(s.cost[k, j])
            m = s.mean[k, j]
            # distance from the decision boundary; unobserved pairs are maximally uncertain
            dist = abs(float(m) - (best - tolerance)) if np.isfinite(m) and np.isfinite(best) else 0.0
            rows.append(dict(request_type=s.types[j], model=s.models[k],
                             incumbent=s.models[i],
                             traffic_share=float(s.traffic[j]),
                             logged_rows=int(s.n[k, j]),
                             distance_to_boundary=dist,
                             readouts=int(readout_size),
                             audit_price_usd=price,
                             priority=float(s.traffic[j] / (dist + 1e-3) / max(price, 1e-9))))
    if not rows:
        return pd.DataFrame(columns=["request_type", "model", "incumbent", "traffic_share",
                                     "logged_rows", "distance_to_boundary", "readouts",
                                     "audit_price_usd", "priority"]), 0.0
    q = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    spent, keep = 0.0, []
    for r in q.itertuples():
        if spent + r.audit_price_usd > budget_usd:
            continue
        spent += r.audit_price_usd
        keep.append(r.Index)
    return q.loc[keep].reset_index(drop=True), float(spent)


def routing_adjusted(s: Surfaces, audit_mean: np.ndarray, audit_var: np.ndarray,
                     pool: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Correct the logged surface with the audit, then optionally pool by inverse variance.

    One additive adjustment per model, fitted on the audited cells only, matching the paper's
    deployed correction. Where a cell was audited, the randomized estimate and the adjusted
    logged estimate are combined by inverse-variance weights instead of one replacing the other.
    """
    bought = np.isfinite(audit_mean)
    b = np.zeros(s.K)
    for k in range(s.K):
        d = [s.mean[k, j] - audit_mean[k, j] for j in range(s.J)
             if bought[k, j] and np.isfinite(s.mean[k, j])]
        b[k] = float(np.mean(d)) if d else 0.0

    q = np.full((s.K, s.J), np.nan)
    qv = np.full((s.K, s.J), np.nan)
    for k in range(s.K):
        for j in range(s.J):
            adj = s.mean[k, j] - b[k] if np.isfinite(s.mean[k, j]) else np.nan
            adjv = s.var[k, j] if np.isfinite(s.var[k, j]) else 0.25
            if bought[k, j]:
                R, Rv = audit_mean[k, j], max(audit_var[k, j], VAR_FLOOR)
                if pool and np.isfinite(adj):
                    wr, wm = 1.0 / Rv, 1.0 / max(adjv, VAR_FLOOR)
                    q[k, j] = (wr * R + wm * adj) / (wr + wm)
                    qv[k, j] = 1.0 / (wr + wm)
                else:
                    q[k, j], qv[k, j] = R, Rv
            else:
                q[k, j], qv[k, j] = adj, adjv
    return q, qv


def cheapest_good_enough(s: Surfaces, q: np.ndarray, tolerance: float) -> np.ndarray:
    """The paper's decision rule applied to whichever score surface is supplied."""
    pol = np.full(s.J, -1, int)
    for j in range(s.J):
        ks = [k for k in range(s.K) if s.supported[k, j]
              and np.isfinite(q[k, j]) and np.isfinite(s.cost[k, j])]
        if not ks:
            continue
        best = max(q[k, j] for k in ks)
        ok = [k for k in ks if q[k, j] >= best - tolerance]
        pol[j] = int(min(ok, key=lambda k: (float(s.cost[k, j]), -float(q[k, j]), k)))
    return pol


def policy_cost(s: Surfaces, pol: np.ndarray) -> float:
    return float(sum(s.traffic[j] * s.cost[pol[j], j] for j in range(s.J)
                     if pol[j] >= 0 and np.isfinite(s.cost[pol[j], j])))


def break_even(saving_per_request_by_type: np.ndarray, traffic: np.ndarray,
               audit_total_usd: float, confidence: float,
               n_boot: int = 4000, seed: int = 0) -> dict:
    """Point and lower-bound break-even volume.

    The lower bound resamples request types with replacement, weighted by traffic, which is the
    only resampling unit available from aggregated inputs. With few request types the interval is
    wide, and that is reported rather than hidden.
    """
    w = np.asarray(traffic, float)
    v = np.asarray(saving_per_request_by_type, float)
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if len(v) == 0 or w.sum() <= 0:
        return dict(saving_per_request=float("nan"), lower_bound=float("nan"),
                    point_break_even=None, lower_bound_break_even=None, n_units=0)
    w = w / w.sum()
    point = float((w * v).sum())

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=(n_boot, len(v)), replace=True)
    draws = np.array([float((w[i] / w[i].sum() * v[i]).sum()) if w[i].sum() > 0 else np.nan
                      for i in idx])
    lo = float(np.nanquantile(draws, 1.0 - confidence))

    return dict(saving_per_request=point, lower_bound=lo, n_units=int(len(v)),
                point_break_even=(audit_total_usd / point) if point > 1e-12 and audit_total_usd > 0 else None,
                lower_bound_break_even=(audit_total_usd / lo) if lo > 1e-12 and audit_total_usd > 0 else None)


def classify_switches(s: Surfaces, inc: np.ndarray, pol: np.ndarray, q: np.ndarray,
                      qv: np.ndarray, tolerance: float, confidence: float,
                      verified: np.ndarray | None) -> pd.DataFrame:
    """One row per request type where the recommended policy differs from the incumbent.

    Status vocabulary is deliberately conservative. Without verified target outcomes a switch is
    only ever `Candidate switch under provided score`, never `valid`.
    """
    zc = norm_ppf(confidence)
    rows = []
    for j in range(s.J):
        i, k = inc[j], pol[j]
        if i < 0 or k < 0 or i == k:
            continue
        dq = float(q[k, j] - q[i, j]) if np.isfinite(q[k, j]) and np.isfinite(q[i, j]) else np.nan
        se = float(np.sqrt(max(qv[k, j], VAR_FLOOR) + max(qv[i, j], VAR_FLOOR)))
        lo = dq - zc * se
        dcost = float(s.cost[i, j] - s.cost[k, j])

        if verified is not None and np.isfinite(verified[k, j]) and np.isfinite(verified[i, j]):
            vbest = np.nanmax(verified[:, j])
            within_best = bool(verified[k, j] >= vbest - tolerance)
            not_worse = bool(verified[k, j] >= verified[i, j] - tolerance)
            if within_best and dcost > 0:
                status = "Valid cost-saving switch"
            elif within_best:
                status = "Quality-safe, no saving"
            elif dcost > 0:
                status = "Quality-violating economical switch"
            else:
                status = "Invalid on both"
            inc_rel = "Non-inferior to incumbent" if not_worse else "Inferior to incumbent"
        else:
            status = "Candidate switch under provided score"
            inc_rel = "Not established (no verified outcomes)"

        rows.append(dict(request_type=s.types[j], incumbent=s.models[i], candidate=s.models[k],
                         traffic_share=float(s.traffic[j]),
                         score_difference=dq, score_diff_lower_bound=lo,
                         non_inferior_at_confidence=bool(np.isfinite(lo) and lo >= -tolerance),
                         saving_per_request_usd=dcost,
                         status=status, incumbent_relative=inc_rel))
    cols = ["request_type", "incumbent", "candidate", "traffic_share", "score_difference",
            "score_diff_lower_bound", "non_inferior_at_confidence", "saving_per_request_usd",
            "status", "incumbent_relative"]
    return pd.DataFrame(rows, columns=cols)
