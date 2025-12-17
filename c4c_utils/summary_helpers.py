# summary_helpers.py
"""
Summary Helpers for OrgGraph

Single source of truth for grant summary metrics in the UI.
Provides filter-consistent summaries with proper bucket normalization.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


BUCKET_LABELS = {
    "3a_paid": "3a (Paid)",
    "3b_future": "3b (Future)",
    "schedule_i": "Schedule I",
}

BUCKET_ORDER = ["3a_paid", "3b_future", "schedule_i"]


def _coerce_amount(series: pd.Series) -> pd.Series:
    """Safely coerce a series to numeric, filling NaN with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _normalize_bucket(v) -> str:
    """
    Normalize bucket strings into canonical buckets.
    Accepts: '3a', '3b', '3a_paid', '3b_future', 'schedule_i', etc.
    """
    if not isinstance(v, str):
        return ""
    v2 = v.strip().lower()
    if v2 in ("3a", "3a_paid"):
        return "3a_paid"
    if v2 in ("3b", "3b_future"):
        return "3b_future"
    if v2 in ("schedulei", "schedule_i", "sched_i"):
        return "schedule_i"
    return v.strip()


def summarize_grants(grants_df: pd.DataFrame) -> Dict:
    """
    Returns a stable summary dict used everywhere in UI:
      - per-bucket counts + amounts
      - totals
      
    Returns:
        {
            "buckets": {
                "3a_paid": {"count": int, "amount": float, "label": str},
                "3b_future": {"count": int, "amount": float, "label": str},
                "schedule_i": {"count": int, "amount": float, "label": str},
            },
            "total_count": int,
            "total_amount": float,
            "other_count": int,  # grants not in known buckets
            "other_amount": float,
        }
    """
    if grants_df is None or grants_df.empty:
        buckets = {
            b: {"count": 0, "amount": 0.0, "label": BUCKET_LABELS.get(b, b)}
            for b in BUCKET_ORDER
        }
        return {
            "buckets": buckets, 
            "total_count": 0, 
            "total_amount": 0.0,
            "other_count": 0,
            "other_amount": 0.0,
        }

    df = grants_df.copy()

    # Required columns (best-effort)
    if "grant_bucket" not in df.columns:
        df["grant_bucket"] = ""
    if "grant_amount" not in df.columns:
        df["grant_amount"] = 0.0

    df["_normalized_bucket"] = df["grant_bucket"].apply(_normalize_bucket)
    df["grant_amount"] = _coerce_amount(df["grant_amount"])

    buckets: Dict[str, Dict] = {}
    total_count = 0
    total_amount = 0.0

    for b in BUCKET_ORDER:
        sub = df[df["_normalized_bucket"] == b]
        c = int(len(sub))
        a = float(sub["grant_amount"].sum())
        buckets[b] = {"count": c, "amount": a, "label": BUCKET_LABELS.get(b, b)}
        total_count += c
        total_amount += a

    # Track unknown/other buckets separately
    unknown = df[~df["_normalized_bucket"].isin(BUCKET_ORDER)]
    other_count = int(len(unknown))
    other_amount = float(unknown["grant_amount"].sum())
    
    # Include other in totals
    total_count += other_count
    total_amount += other_amount

    return {
        "buckets": buckets, 
        "total_count": total_count, 
        "total_amount": total_amount,
        "other_count": other_count,
        "other_amount": other_amount,
    }


def split_region_filtered(
    grants_df: pd.DataFrame,
    region_flag_col: str = "region_relevant",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (all_df, region_df). Works with either region_relevant or gl_relevant.
    """
    if grants_df is None or grants_df.empty:
        return grants_df, grants_df

    df = grants_df.copy()

    # Support older column name
    if region_flag_col not in df.columns and "gl_relevant" in df.columns:
        region_flag_col = "gl_relevant"

    if region_flag_col not in df.columns:
        # No region flag available → treat as "no filtering"
        return df, df

    region_df = df[df[region_flag_col].astype(bool)].copy()
    return df, region_df


def build_grant_network_summary(
    grants_df: pd.DataFrame,
    region_flag_col: str = "region_relevant",
) -> Dict:
    """
    Single canonical object the UI should use.
    
    Returns:
        {
            "all": {"df": DataFrame, "summary": dict},
            "region": {"df": DataFrame, "summary": dict},
            "has_region_filter": bool,
            "region_flag_col_used": str,
        }
    """
    all_df, region_df = split_region_filtered(grants_df, region_flag_col=region_flag_col)

    all_summary = summarize_grants(all_df)
    region_summary = summarize_grants(region_df)

    # Determine which column was actually used
    actual_col = ""
    if grants_df is not None and not grants_df.empty:
        if region_flag_col in grants_df.columns:
            actual_col = region_flag_col
        elif "gl_relevant" in grants_df.columns:
            actual_col = "gl_relevant"

    return {
        "all": {"df": all_df, "summary": all_summary},
        "region": {"df": region_df, "summary": region_summary},
        "has_region_filter": (len(region_df) != len(all_df)) and (len(region_df) > 0),
        "region_flag_col_used": actual_col,
    }


# =============================================================================
# Backward compatibility aliases
# =============================================================================

def get_filtered_grants(grants_df: pd.DataFrame, region_def: dict = None) -> pd.DataFrame:
    """
    Legacy helper - use build_grant_network_summary instead.
    """
    if grants_df is None or grants_df.empty:
        return grants_df
    
    if region_def and region_def.get("id") != "none" and "region_relevant" in grants_df.columns:
        return grants_df[grants_df["region_relevant"] == True].copy()
    
    return grants_df


def build_grants_summary(df: pd.DataFrame, *, scope_label: str, **kwargs) -> Dict:
    """
    Legacy helper - wraps summarize_grants with scope_label for compatibility.
    """
    summary = summarize_grants(df)
    
    # Convert to old format for backward compatibility
    return {
        "scope_label": scope_label,
        "total": {"count": summary["total_count"], "amount": summary["total_amount"]},
        "by_bucket": {
            "3a": {"count": summary["buckets"]["3a_paid"]["count"], "amount": summary["buckets"]["3a_paid"]["amount"]},
            "3b": {"count": summary["buckets"]["3b_future"]["count"], "amount": summary["buckets"]["3b_future"]["amount"]},
            "schedule_i": {"count": summary["buckets"]["schedule_i"]["count"], "amount": summary["buckets"]["schedule_i"]["amount"]},
            "__other__": {"count": summary["other_count"], "amount": summary["other_amount"]},
        },
        "bucket_order": ["3a", "3b", "schedule_i"],
    }
