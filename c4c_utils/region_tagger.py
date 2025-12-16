# region_tagger.py
"""
Region Tagger for OrgGraph
==========================
Pure functions to apply region tagging to grants dataframe.
Region tagging is optional post-processing after 990-PF parsing.

A grant is region_relevant=True if grantee_state matches any state/province
in the region definition.
"""

import re
import pandas as pd
from typing import Dict, Any, Optional

# Pattern to extract state code from address: ", ST 12345" or ", ST 12345-6789"
STATE_ZIP_RE = re.compile(r",\s*([A-Z]{2})\s+\d{5}(?:-\d{4})?\b")

# Pattern for Canadian postal codes (to detect CA addresses)
CA_POSTAL_RE = re.compile(r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b")


def normalize_admin1(df: pd.DataFrame, state_col: str = "grantee_state") -> pd.Series:
    """
    Normalize the admin1 (state/province) column to uppercase.
    
    Args:
        df: Grants dataframe
        state_col: Name of state column
        
    Returns:
        Series of normalized state codes
    """
    if state_col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    
    return (
        df[state_col]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )


def fallback_state_from_address(address: str) -> str:
    """
    Try to extract state code from raw address string.
    Looks for pattern: ", ST 12345"
    
    Args:
        address: Raw address string
        
    Returns:
        Two-letter state code or empty string
    """
    if not address:
        return ""
    
    # Try US pattern first: ", ST 12345"
    match = STATE_ZIP_RE.search(address.upper())
    if match:
        return match.group(1)
    
    return ""


def apply_region_tagging(
    grants_df: pd.DataFrame,
    region_def: Dict[str, Any],
    *,
    enable_address_fallback: bool = True,
    state_col: str = "grantee_state",
    address_col: str = "grantee_address_raw",
) -> pd.DataFrame:
    """
    Apply region tagging to a grants dataframe.
    
    Adds columns:
    - region_id: ID of the region definition
    - region_name: Human-readable region name
    - region_rule_source: "preset", "project", or "custom"
    - region_relevant: Boolean indicating if grant is in region
    
    Args:
        grants_df: Dataframe with grant data (must have grantee_state column)
        region_def: Region definition dict with include_us_states, include_ca_provinces
        enable_address_fallback: If True, try to parse state from grantee_address_raw
                                  when grantee_state is missing
        state_col: Name of state column (default: grantee_state)
        address_col: Name of raw address column for fallback (default: grantee_address_raw)
        
    Returns:
        Copy of dataframe with region columns added
    """
    # Handle empty/None inputs
    if grants_df is None or grants_df.empty:
        return grants_df
    
    # No region mode - return unchanged
    if not region_def or region_def.get("id") in (None, "", "none"):
        return grants_df
    
    # Work on a copy
    df = grants_df.copy()
    
    # Add region metadata columns
    df["region_id"] = region_def.get("id", "")
    df["region_name"] = region_def.get("name", "")
    df["region_rule_source"] = region_def.get("source", "")
    df["region_relevant"] = False
    
    # Build sets of included states/provinces (uppercase)
    include_states = set(
        s.upper() for s in region_def.get("include_us_states", [])
    )
    include_provinces = set(
        p.upper() for p in region_def.get("include_ca_provinces", [])
    )
    all_included = include_states.union(include_provinces)
    
    # Get normalized state column
    admin1 = normalize_admin1(df, state_col)
    
    # Apply fallback parsing if enabled
    if enable_address_fallback and address_col in df.columns:
        # Find rows with missing state
        missing = admin1.eq("")
        if missing.any():
            # Try to parse state from raw address
            fallback_states = (
                df.loc[missing, address_col]
                .fillna("")
                .apply(fallback_state_from_address)
            )
            # Update admin1 where we found a fallback
            admin1 = admin1.mask(missing, fallback_states)
    
    # Mark as region_relevant if state/province is in the included set
    df["region_relevant"] = admin1.isin(all_included)
    
    return df


def get_region_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for region-tagged grants.
    
    Args:
        df: Dataframe with region columns
        
    Returns:
        Dict with summary stats
    """
    if df is None or df.empty:
        return {
            "total_grants": 0,
            "region_relevant_count": 0,
            "region_relevant_pct": 0,
            "region_relevant_amount": 0,
            "total_amount": 0,
            "region_name": "",
            "region_id": "",
        }
    
    # Check if region tagging was applied
    has_region = "region_relevant" in df.columns
    
    total = len(df)
    relevant = df["region_relevant"].sum() if has_region else 0
    
    amount_col = "grant_amount" if "grant_amount" in df.columns else "amount"
    total_amount = df[amount_col].sum() if amount_col in df.columns else 0
    relevant_amount = df.loc[df["region_relevant"], amount_col].sum() if has_region and amount_col in df.columns else 0
    
    # Get region name/id from first row (all rows should have same region)
    region_name = df["region_name"].iloc[0] if "region_name" in df.columns and not df.empty else ""
    region_id = df["region_id"].iloc[0] if "region_id" in df.columns and not df.empty else ""
    
    return {
        "total_grants": total,
        "region_relevant_count": int(relevant),
        "region_relevant_pct": round(relevant / total * 100, 1) if total > 0 else 0,
        "region_relevant_amount": int(relevant_amount),
        "total_amount": int(total_amount),
        "region_name": region_name,
        "region_id": region_id,
    }


def filter_region_relevant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only region-relevant grants.
    
    Args:
        df: Dataframe with region_relevant column
        
    Returns:
        Filtered dataframe
    """
    if df is None or df.empty or "region_relevant" not in df.columns:
        return df
    
    return df[df["region_relevant"]].copy()
