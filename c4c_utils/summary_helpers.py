"""
Summary Helpers for OrgGraph

Provides filter-consistent grant summary computation.
When a region filter is active, every metric shown together must be computed 
on the same filtered dataset. No mixing scopes.

Usage:
    all_summary = build_grants_summary(grants_df, scope_label="All grants (unfiltered)")
    filtered_summary = build_grants_summary(filtered_df, scope_label="Great Lakes grants")
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd


DEFAULT_BUCKET_ORDER = ["3a", "3b", "schedule_i"]


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce series to numeric safely (handles strings, NaNs)."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def build_grants_summary(
    df: pd.DataFrame,
    *,
    scope_label: str,
    bucket_col: str = "grant_bucket",
    amount_col: str = "grant_amount",
    bucket_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a single, filter-consistent summary object from the given dataframe.

    Returns:
      {
        "scope_label": "...",
        "total": {"count": int, "amount": float},
        "by_bucket": {
            "3a": {"count": int, "amount": float},
            "3b": {"count": int, "amount": float},
            "schedule_i": {"count": int, "amount": float},
            "__other__": {"count": int, "amount": float},   # optional if present
        },
        "bucket_order": [...]
      }
    """
    bucket_order = bucket_order or DEFAULT_BUCKET_ORDER

    if df is None or df.empty:
        return {
            "scope_label": scope_label,
            "total": {"count": 0, "amount": 0.0},
            "by_bucket": {b: {"count": 0, "amount": 0.0} for b in bucket_order},
            "bucket_order": bucket_order,
        }

    # Ensure required cols exist
    if bucket_col not in df.columns:
        # If no bucket column, treat all as "__other__"
        working = df.copy()
        working[bucket_col] = "__other__"
    else:
        working = df.copy()
        # Normalize bucket to string
        working[bucket_col] = working[bucket_col].astype(str).str.strip()

    if amount_col not in df.columns:
        raise KeyError(f"Missing column '{amount_col}'")

    # Normalize amount to numeric
    working[amount_col] = _safe_numeric(working[amount_col])

    # Compute totals
    total_count = int(len(working))
    total_amount = float(working[amount_col].sum())

    # Group by bucket
    grp = (
        working.groupby(bucket_col, dropna=False)[amount_col]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={bucket_col: "bucket", "sum": "amount"})
    )

    # Initialize with bucket_order
    by_bucket: Dict[str, Dict[str, Any]] = {
        b: {"count": 0, "amount": 0.0} for b in bucket_order
    }

    other_count = 0
    other_amount = 0.0

    for _, row in grp.iterrows():
        b = str(row["bucket"]).strip()
        c = int(row["count"])
        a = float(row["amount"])

        if b in by_bucket:
            by_bucket[b]["count"] = c
            by_bucket[b]["amount"] = a
        else:
            other_count += c
            other_amount += a

    if other_count > 0:
        by_bucket["__other__"] = {"count": other_count, "amount": other_amount}

    return {
        "scope_label": scope_label,
        "total": {"count": total_count, "amount": total_amount},
        "by_bucket": by_bucket,
        "bucket_order": bucket_order + (["__other__"] if "__other__" in by_bucket else []),
    }


def get_filtered_grants(
    grants_df: pd.DataFrame,
    region_def: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply region filter to grants_df if region_def is active.
    
    Returns filtered dataframe (or original if no filter active).
    """
    if grants_df is None or grants_df.empty:
        return grants_df
    
    if region_def and region_def.get("id") != "none" and "region_relevant" in grants_df.columns:
        return grants_df[grants_df["region_relevant"] == True].copy()
    
    return grants_df
