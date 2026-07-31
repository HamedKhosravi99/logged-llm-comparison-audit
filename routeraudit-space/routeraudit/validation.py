"""Schema and sanity validation for every table RouterAudit accepts.

Validation never repairs data silently. It either raises SchemaError, which the interface shows
verbatim, or appends a warning that travels into the report.
"""
from __future__ import annotations

import pandas as pd

MIN_SUPPORT_FLOOR = 1

ROUTED_LOG_REQUIRED = ["request_id", "request_type", "model", "score", "request_cost_usd"]
ROUTED_LOG_OPTIONAL = ["timestamp", "traffic_weight", "incumbent"]

PRICES_REQUIRED_LONG = ["request_type", "model", "expected_cost_usd"]
PRICES_REQUIRED_SHORT = ["model", "expected_cost_usd"]

AUDIT_REQUIRED = ["request_id", "request_type", "model", "audit_score", "audit_cost_usd"]
VERIFIED_REQUIRED = ["request_id", "request_type", "model", "verified_outcome"]
VERIFIED_OPTIONAL = ["verification_cost_usd"]
TRAFFIC_REQUIRED = ["request_type", "traffic_share"]


class SchemaError(ValueError):
    """Raised when an input table cannot be used without guessing."""


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    if df is None:
        raise SchemaError(f"{name}: no table supplied")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{name} is missing required column(s): {', '.join(missing)}. "
            f"Expected at least: {', '.join(cols)}")
    if len(df) == 0:
        raise SchemaError(f"{name} contains no rows")


def _numeric(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            bad = int(v.isna().sum())
            raise SchemaError(f"{name}: column '{c}' has {bad} value(s) that are not numeric "
                              f"or are missing")
        df[c] = v


def _nonnegative(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        if c in df.columns and (df[c] < 0).any():
            n = int((df[c] < 0).sum())
            raise SchemaError(f"{name}: column '{c}' has {n} negative value(s); costs and "
                              f"weights must be non-negative")


def validate_routed_log(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    _require(df, ROUTED_LOG_REQUIRED, "Routed log")
    df = df.copy()
    _numeric(df, ["score", "request_cost_usd", "traffic_weight"], "Routed log")
    _nonnegative(df, ["request_cost_usd", "traffic_weight"], "Routed log")
    warnings: list[str] = []

    dup = df.duplicated(subset=["request_id", "model"]).sum()
    if dup:
        raise SchemaError(f"Routed log: {int(dup)} duplicated (request_id, model) row(s). "
                          f"Each request must appear once per model that served it.")

    for c in ("request_type", "model", "request_id"):
        if df[c].isna().any():
            raise SchemaError(f"Routed log: column '{c}' contains missing values")
        df[c] = df[c].astype(str)

    if df["model"].nunique() < 2:
        raise SchemaError("Routed log: fewer than two distinct models, so there is no "
                          "comparison to audit")

    lo, hi = float(df["score"].min()), float(df["score"].max())
    if lo < 0 or hi > 1:
        warnings.append(
            f"Automatic scores fall outside [0, 1] (observed {lo:.4g} to {hi:.4g}). The quality "
            f"tolerance is interpreted on the same scale as the supplied score.")
    return df, warnings


def validate_prices(df: pd.DataFrame, request_types: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if df is None:
        raise SchemaError("Model prices: no table supplied")
    df = df.copy()
    warnings: list[str] = []
    if "request_type" in df.columns:
        _require(df, PRICES_REQUIRED_LONG, "Model prices")
        _numeric(df, ["expected_cost_usd"], "Model prices")
        _nonnegative(df, ["expected_cost_usd"], "Model prices")
        df["request_type"] = df["request_type"].astype(str)
    else:
        _require(df, PRICES_REQUIRED_SHORT, "Model prices")
        _numeric(df, ["expected_cost_usd"], "Model prices")
        _nonnegative(df, ["expected_cost_usd"], "Model prices")
        rows = [dict(request_type=t, model=m, expected_cost_usd=c)
                for t in request_types
                for m, c in zip(df["model"].astype(str), df["expected_cost_usd"])]
        df = pd.DataFrame(rows)
        warnings.append("Model prices were supplied without a request_type column and were "
                        "expanded to every request type at the same price.")
    df["model"] = df["model"].astype(str)
    if df.duplicated(subset=["request_type", "model"]).any():
        raise SchemaError("Model prices: duplicated (request_type, model) rows")
    return df, warnings


def validate_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    _require(df, AUDIT_REQUIRED, "Randomized audit")
    df = df.copy()
    _numeric(df, ["audit_score", "audit_cost_usd"], "Randomized audit")
    _nonnegative(df, ["audit_cost_usd"], "Randomized audit")
    for c in ("request_type", "model", "request_id"):
        df[c] = df[c].astype(str)
    if df.duplicated(subset=["request_id", "model"]).any():
        raise SchemaError("Randomized audit: duplicated (request_id, model) rows")
    return df, []


def validate_verified(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    _require(df, VERIFIED_REQUIRED, "Verified outcomes")
    df = df.copy()
    _numeric(df, ["verified_outcome", "verification_cost_usd"], "Verified outcomes")
    _nonnegative(df, ["verification_cost_usd"], "Verified outcomes")
    for c in ("request_type", "model", "request_id"):
        df[c] = df[c].astype(str)
    return df, []


def validate_traffic(df: pd.DataFrame, request_types: list[str]) -> tuple[pd.DataFrame, list[str]]:
    _require(df, TRAFFIC_REQUIRED, "Traffic weights")
    df = df.copy()
    _numeric(df, ["traffic_share"], "Traffic weights")
    _nonnegative(df, ["traffic_share"], "Traffic weights")
    df["request_type"] = df["request_type"].astype(str)
    total = float(df["traffic_share"].sum())
    if abs(total - 1.0) > 1e-3:
        raise SchemaError(f"Traffic weights: shares sum to {total:.6f}, which is not "
                          f"approximately 1. Supply normalised shares.")
    unknown = sorted(set(df["request_type"]) - set(request_types))
    if unknown:
        raise SchemaError(f"Traffic weights: request type(s) not present in the routed log: "
                          f"{', '.join(unknown[:5])}")
    return df, []
