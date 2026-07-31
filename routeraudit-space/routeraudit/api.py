"""The one stable entry point. The interface layer calls `analyze_trace` and nothing else.

Guarantees: no network calls, no global mutable state, no held-out outcome is read while a
candidate policy is being selected, and every unsupported assumption becomes a warning rather
than a silent default.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .core import estimate as est
from .validation import (SchemaError, validate_audit, validate_prices, validate_routed_log,
                         validate_traffic, validate_verified)

__all__ = ["AuditConfig", "AuditResult", "analyze_trace", "SchemaError"]


@dataclass(frozen=True)
class AuditConfig:
    quality_tolerance: float = 0.02
    future_requests: int = 100_000
    audit_budget_usd: float = 10.0
    verification_cost_usd: float = 0.0
    confidence_level: float = 0.95
    minimum_support: int = 8
    audit_readout_size: int = 30

    def __post_init__(self) -> None:
        if self.quality_tolerance < 0:
            raise ValueError("quality_tolerance must be non-negative")
        if self.future_requests < 1:
            raise ValueError("future_requests must be at least 1")
        if self.audit_budget_usd < 0:
            raise ValueError("audit_budget_usd must be non-negative")
        if self.verification_cost_usd < 0:
            raise ValueError("verification_cost_usd must be non-negative")
        if not 0.5 <= self.confidence_level < 1:
            raise ValueError("confidence_level must lie in [0.5, 1)")
        if self.minimum_support < 1:
            raise ValueError("minimum_support must be at least 1")
        if self.audit_readout_size < 1:
            raise ValueError("audit_readout_size must be at least 1")


@dataclass
class AuditResult:
    mode: str
    summary: pd.DataFrame
    diagnostics: dict[str, Any]
    audit_queue: pd.DataFrame
    candidate_switches: pd.DataFrame
    warnings: list[str]
    manifest: dict[str, Any]
    report_html: str


def _cell_matrix(df: pd.DataFrame, models: list[str], types: list[str],
                 value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell mean, variance of the mean, and row count from a long table."""
    mi = {m: i for i, m in enumerate(models)}
    ji = {t: j for j, t in enumerate(types)}
    mean = np.full((len(models), len(types)), np.nan)
    var = np.full((len(models), len(types)), np.nan)
    cnt = np.zeros((len(models), len(types)), int)
    for (m, t), g in df.groupby(["model", "request_type"], sort=False):
        if m not in mi or t not in ji:
            continue
        k, j = mi[m], ji[t]
        v = g[value_col].to_numpy(float)
        cnt[k, j] = len(v)
        mean[k, j] = float(v.mean())
        var[k, j] = max(float(v.var(ddof=1) / len(v)) if len(v) > 1 else 0.25 / len(v),
                        est.VAR_FLOOR)
    return mean, var, cnt


def analyze_trace(
    routed_log: pd.DataFrame,
    model_prices: pd.DataFrame,
    config: AuditConfig,
    randomized_audit: pd.DataFrame | None = None,
    verified_outcomes: pd.DataFrame | None = None,
    traffic: pd.DataFrame | None = None,
) -> AuditResult:
    """Run RouterAudit without reading or writing global state.

    Mode A (plan) when `randomized_audit` is None: reports support, the assignment diagnostic,
    cheaper candidates and a proposed audit queue. It never asserts that a switch is valid.

    Mode B (analyze) otherwise: adds routing-adjusted and inverse-variance-pooled estimates, a
    candidate policy with intervals, switch classification and audit-inclusive economics.
    """
    warnings: list[str] = []

    log, w = validate_routed_log(routed_log)
    warnings += w
    types = sorted(log["request_type"].unique())
    prices, w = validate_prices(model_prices, types)
    warnings += w

    traffic_v = None
    if traffic is not None:
        traffic_v, w = validate_traffic(traffic, types)
        warnings += w
    else:
        warnings.append("Traffic weights were derived from historical routed-log frequencies. "
                        "Future traffic may differ.")

    audit_v = None
    if randomized_audit is not None:
        audit_v, w = validate_audit(randomized_audit)
        warnings += w
    else:
        warnings.append("No randomized audit supplied. Assignment correction has not been "
                        "performed.")

    verified_v = None
    if verified_outcomes is not None:
        verified_v, w = validate_verified(verified_outcomes)
        warnings += w
    else:
        warnings.append("No verified target outcomes supplied. Measurement validity is not "
                        "established: the randomized audit can address selective assignment, "
                        "but this run does not establish that the automatic score represents "
                        "the outcome the organization values.")

    s = est.build_surfaces(log, prices, config.minimum_support, traffic_v)

    thin = int(((~s.supported) & (s.n > 0)).sum())
    if thin:
        warnings.append(f"{thin} model/request-type pair(s) have fewer than "
                        f"{config.minimum_support} logged rows and are excluded from every "
                        f"decision. The result depends on fewer than the minimum supported "
                        f"observations in those cells.")
    missing_price = int((~np.isfinite(s.cost) & s.supported).sum())
    if missing_price:
        warnings.append(f"{missing_price} supported pair(s) have no price. A missing price is "
                        f"never treated as zero, so those pairs are excluded from cost output.")

    declared = None
    if "incumbent" in log.columns:
        d = log.dropna(subset=["incumbent"]).groupby("request_type")["incumbent"].agg(
            lambda x: x.astype(str).mode().iat[0])
        declared = d.to_dict()
    inc = est.incumbent_policy(s, declared)

    diag = est.assignment_diagnostic(s)
    if diag["verdict"] == "Selective assignment does not appear important":
        warnings.append("The production log already contains sufficient exploratory support for "
                        "a free comparison; a randomized audit may not repay its cost here.")

    cands = est.cheaper_candidates(s, inc)
    if not cands:
        warnings.append("No cheaper model lies within the estimated quality tolerance: no "
                        "candidate is priced below the incumbent in any request type.")

    queue, queue_cost = est.audit_queue(s, inc, config.quality_tolerance,
                                        config.audit_budget_usd, config.audit_readout_size)
    n_verified_outputs = int(queue["readouts"].sum()) if len(queue) else 0
    planned_total = queue_cost + config.verification_cost_usd * n_verified_outputs

    max_saving = float(sum(s.traffic[j] * max(
        (s.cost[i, j] - s.cost[k, j] for k in range(s.K)
         if np.isfinite(s.cost[k, j]) and np.isfinite(s.cost[i, j])), default=0.0)
        for j, i in enumerate(inc) if i >= 0)) * config.future_requests
    if planned_total > max_saving > 0:
        warnings.append(f"The requested audit costs ${planned_total:,.2f}, which exceeds its "
                        f"maximum estimated deployment saving of ${max_saving:,.2f} over "
                        f"{config.future_requests:,} requests.")

    manifest = dict(
        routeraudit_version=__version__,
        timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        mode="plan" if audit_v is None else "analyze",
        measurement_validity_established=bool(verified_v is not None),
        randomized_audit_supplied=bool(audit_v is not None),
        traffic_supplied=bool(traffic_v is not None),
        models=s.models, request_types=s.types,
        logged_rows=int(len(log)),
        **{k: v for k, v in asdict(config).items()},
    )

    # ---------------------------------------------------------------- Mode A
    if audit_v is None:
        summary = pd.DataFrame([
            ("Logged rows", f"{len(log):,}"),
            ("Models", f"{s.K}"),
            ("Request types", f"{s.J}"),
            ("Supported pairs", f"{diag['supported_pairs']} of {diag['total_pairs']}"),
            ("Never-observed pairs", f"{diag['unobserved_pairs']}"),
            ("Data check", "Sufficient evidence" if diag["supported_pairs"] >= 2
             else "Insufficient evidence"),
            ("Assignment diagnostic", diag["verdict"]),
            ("Cheaper candidates found", f"{len(cands)}"),
            ("Proposed audit purchases", f"{len(queue)}"),
            ("Estimated model-call audit cost", f"${queue_cost:,.2f}"),
            ("Estimated verification cost", f"${config.verification_cost_usd * n_verified_outputs:,.2f}"),
            ("Estimated total audit cost", f"${planned_total:,.2f}"),
            ("Measurement validity", "Measurement validity not established"),
            ("Recommendation", "Collect the proposed randomized comparisons, then re-run in "
                               "analyze mode"),
        ], columns=["Item", "Value"])
        switches = pd.DataFrame(columns=[
            "request_type", "incumbent", "candidate", "traffic_share", "score_difference",
            "score_diff_lower_bound", "non_inferior_at_confidence", "saving_per_request_usd",
            "status", "incumbent_relative"])
        diagnostics = dict(assignment=diag, cheaper_candidates=cands,
                           planned_audit_cost_usd=queue_cost,
                           planned_verification_cost_usd=config.verification_cost_usd * n_verified_outputs,
                           planned_total_audit_cost_usd=planned_total,
                           verified_outputs_required=n_verified_outputs)
        from .reporting import render_report
        html = render_report("plan", summary, queue, switches, diagnostics, warnings, manifest)
        return AuditResult("plan", summary, diagnostics, queue, switches, warnings, manifest, html)

    # ---------------------------------------------------------------- Mode B
    a_mean, a_var, a_cnt = _cell_matrix(audit_v, s.models, s.types, "audit_score")
    audited_cells = int(np.isfinite(a_mean).sum())
    if audited_cells == 0:
        raise SchemaError("Randomized audit contains no rows matching a model and request type "
                          "present in the routed log.")
    spent_audit = float(audit_v["audit_cost_usd"].sum())

    ver_mat = None
    ver_cost = 0.0
    if verified_v is not None:
        ver_mat, _, _ = _cell_matrix(verified_v, s.models, s.types, "verified_outcome")
        if "verification_cost_usd" in verified_v.columns:
            ver_cost = float(verified_v["verification_cost_usd"].sum())
        else:
            ver_cost = config.verification_cost_usd * len(verified_v)
    else:
        ver_cost = config.verification_cost_usd * int(len(audit_v))
    audit_total = spent_audit + ver_cost

    q_pool, qv_pool = est.routing_adjusted(s, a_mean, a_var, pool=True)
    q_over, _ = est.routing_adjusted(s, a_mean, a_var, pool=False)

    # candidate selection reads only the pooled estimate; verified outcomes are used afterwards
    pol = est.cheapest_good_enough(s, q_pool, config.quality_tolerance)
    for j in range(s.J):
        if pol[j] < 0 and inc[j] >= 0:
            pol[j] = inc[j]

    switches = est.classify_switches(s, inc, pol, q_pool, qv_pool,
                                     config.quality_tolerance, config.confidence_level, ver_mat)

    inc_cost = est.policy_cost(s, inc)
    pol_cost = est.policy_cost(s, pol)
    per_type = np.array([
        (s.cost[inc[j], j] - s.cost[pol[j], j])
        if inc[j] >= 0 and pol[j] >= 0 and np.isfinite(s.cost[inc[j], j])
        and np.isfinite(s.cost[pol[j], j]) else np.nan
        for j in range(s.J)])
    be = est.break_even(per_type, s.traffic, audit_total, config.confidence_level)

    if be["point_break_even"] is None:
        warnings.append("The recommended policy does not reduce deployment cost against the "
                        "incumbent, so no break-even volume exists.")
    if be["lower_bound_break_even"] is None:
        warnings.append("No finite lower-bound economic break-even exists: the lower bound on "
                        "the per-request saving is not positive.")
    if np.isfinite(be["lower_bound"]) and be["lower_bound"] <= 0 < be["saving_per_request"]:
        warnings.append("The cost difference is not statistically distinguishable from zero at "
                        f"the {config.confidence_level:.0%} level.")
    if be["n_units"] < 5:
        warnings.append(f"The economic interval resamples only {be['n_units']} request type(s), "
                        f"so it is wide and should be read as indicative.")

    deploy_1k = 1000 * pol_cost
    inc_1k = 1000 * inc_cost
    incl_1k = 1000 * pol_cost + 1000 * audit_total / config.future_requests

    def _fmt_be(x):
        return "No finite supported break-even" if x is None else f"{x:,.0f} requests"

    summary = pd.DataFrame([
        ("Logged rows", f"{len(log):,}"),
        ("Audited cells", f"{audited_cells}"),
        ("Supported pairs", f"{diag['supported_pairs']} of {diag['total_pairs']}"),
        ("Data check", "Sufficient evidence" if diag["supported_pairs"] >= 2
         else "Insufficient evidence"),
        ("Assignment diagnostic", diag["verdict"]),
        ("Request types retaining the incumbent", f"{int((pol == inc).sum())} of {s.J}"),
        ("Request types with a candidate switch", f"{len(switches)}"),
        ("Incumbent deployment cost / 1k requests", f"${inc_1k:,.4f}"),
        ("Recommended deployment cost / 1k requests", f"${deploy_1k:,.4f}"),
        ("Audit-inclusive cost / 1k requests", f"${incl_1k:,.4f}"),
        ("Model-call audit cost", f"${spent_audit:,.2f}"),
        ("Verification cost", f"${ver_cost:,.2f}"),
        ("Total audit cost", f"${audit_total:,.2f}"),
        ("Estimated saving / request", f"${be['saving_per_request']:,.6f}"),
        (f"Lower bound at {config.confidence_level:.0%}", f"${be['lower_bound']:,.6f}"),
        ("Point break-even", _fmt_be(be["point_break_even"])),
        (f"{config.confidence_level:.0%} lower-bound break-even", _fmt_be(be["lower_bound_break_even"])),
        ("Measurement validity", "Established from verified outcomes" if ver_mat is not None
         else "Measurement validity not established"),
    ], columns=["Item", "Value"])

    diagnostics = dict(
        assignment=diag, cheaper_candidates=cands, economics=be,
        model_call_audit_cost_usd=spent_audit, verification_cost_usd=ver_cost,
        total_audit_cost_usd=audit_total,
        incumbent_cost_per_1k=inc_1k, recommended_cost_per_1k=deploy_1k,
        audit_inclusive_cost_per_1k=incl_1k,
        incumbent_policy={s.types[j]: s.models[inc[j]] for j in range(s.J) if inc[j] >= 0},
        recommended_policy={s.types[j]: s.models[pol[j]] for j in range(s.J) if pol[j] >= 0},
        pooling="inverse-variance",
        overwrite_policy_differs=bool(
            not np.array_equal(est.cheapest_good_enough(s, q_over, config.quality_tolerance), pol)),
    )
    if diagnostics["overwrite_policy_differs"]:
        warnings.append("Replacing the logged estimate with the audit instead of pooling would "
                        "give a different policy; the pooled estimate is reported.")

    from .reporting import render_report
    html = render_report("analyze", summary, queue, switches, diagnostics, warnings, manifest)
    return AuditResult("analyze", summary, diagnostics, queue, switches, warnings, manifest, html)
