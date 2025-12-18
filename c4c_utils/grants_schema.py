"""
Canonical Grants Detail Schema — Shared by OrgGraph US, OrgGraph CA, and Insight Engine

This defines the single source of truth for grants_detail.csv schema.
All grant analytics across apps must use this schema.

Usage:
    from c4c_utils.grants_schema import (
        GRANTS_DETAIL_COLUMNS,
        GRANT_BUCKETS,
        create_grants_detail_row,
        validate_grants_detail_df
    )
"""

import pandas as pd
from typing import Optional, Any

# =============================================================================
# Canonical Schema Columns
# =============================================================================

# Required columns (must exist in all outputs)
GRANTS_DETAIL_REQUIRED = [
    "foundation_name",      # Name of filing foundation
    "foundation_ein",       # EIN (US) or BN/registration ID (CA)
    "tax_year",             # Filing year (string)
    "grantee_name",         # Recipient organization
    "grantee_city",         # City
    "grantee_state",        # US state or CA province code
    "grant_amount",         # Numeric amount
    "grant_purpose_raw",    # Raw text purpose (may be empty)
    "grant_bucket",         # Source bucket (3a, 3b, schedule_i, ca_t3010)
    "region_relevant",      # Boolean computed from region mode
    "source_file",          # Original uploaded file name
]

# Recommended columns (non-breaking, safe to add)
GRANTS_DETAIL_RECOMMENDED = [
    "grantee_country",      # "US" or "CA"
    "foundation_country",   # "US" or "CA"
    "source_system",        # e.g. irs_990pf_pdf, irs_990_xml, cra_t3010_csv
    "grant_amount_cash",    # Cash portion (CA data has this split)
    "grant_amount_in_kind", # In-kind portion
    "currency",             # USD or CAD
    "fiscal_year",          # Numeric year (parsed from tax_year/reporting_period)
    "reporting_period",     # Original reporting period string
]

# All columns
GRANTS_DETAIL_COLUMNS = GRANTS_DETAIL_REQUIRED + GRANTS_DETAIL_RECOMMENDED


# =============================================================================
# Grant Bucket Constants
# =============================================================================

class GrantBucket:
    """
    Grant bucket constants.
    
    These describe WHERE the data came from, not its meaning.
    Never sum buckets blindly - UI decides what to include.
    """
    # US (IRS 990-PF)
    US_3A = "3a"                    # Part XIV line 3a (paid this year)
    US_3B = "3b"                    # Part XIV line 3b (future commitments)
    US_SCHEDULE_I = "schedule_i"   # Schedule I grants
    
    # Canada (CRA T3010)
    CA_T3010 = "ca_t3010"          # CRA T3010 grant disclosures
    
    # Unknown/unspecified
    UNKNOWN = "unknown"


# List of valid buckets
GRANT_BUCKETS = [
    GrantBucket.US_3A,
    GrantBucket.US_3B,
    GrantBucket.US_SCHEDULE_I,
    GrantBucket.CA_T3010,
    GrantBucket.UNKNOWN,
]


# =============================================================================
# Row Creation Helper
# =============================================================================

def create_grants_detail_row(
    foundation_name: str,
    foundation_ein: str,
    tax_year: str,
    grantee_name: str,
    grantee_city: str,
    grantee_state: str,
    grant_amount: float,
    grant_purpose_raw: str,
    grant_bucket: str,
    region_relevant: bool,
    source_file: str,
    # Optional fields
    grantee_country: str = "",
    foundation_country: str = "",
    source_system: str = "",
    grant_amount_cash: float = None,
    grant_amount_in_kind: float = None,
    currency: str = "",
    fiscal_year: int = None,
    reporting_period: str = "",
) -> dict:
    """
    Create a canonical grants_detail row.
    
    Returns a dict with all columns (required + recommended).
    Empty/None values are converted to empty strings or appropriate defaults.
    """
    def safe_str(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val).strip()
    
    def safe_float(val) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0
        try:
            return float(val)
        except:
            return 0.0
    
    def safe_int(val) -> Optional[int]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return int(val)
        except:
            return None
    
    return {
        # Required
        "foundation_name": safe_str(foundation_name),
        "foundation_ein": safe_str(foundation_ein),
        "tax_year": safe_str(tax_year),
        "grantee_name": safe_str(grantee_name),
        "grantee_city": safe_str(grantee_city),
        "grantee_state": safe_str(grantee_state),
        "grant_amount": safe_float(grant_amount),
        "grant_purpose_raw": safe_str(grant_purpose_raw),
        "grant_bucket": safe_str(grant_bucket) if grant_bucket in GRANT_BUCKETS else GrantBucket.UNKNOWN,
        "region_relevant": bool(region_relevant),
        "source_file": safe_str(source_file),
        # Recommended
        "grantee_country": safe_str(grantee_country),
        "foundation_country": safe_str(foundation_country),
        "source_system": safe_str(source_system),
        "grant_amount_cash": safe_float(grant_amount_cash) if grant_amount_cash is not None else "",
        "grant_amount_in_kind": safe_float(grant_amount_in_kind) if grant_amount_in_kind is not None else "",
        "currency": safe_str(currency),
        "fiscal_year": safe_int(fiscal_year) if fiscal_year else "",
        "reporting_period": safe_str(reporting_period),
    }


# =============================================================================
# DataFrame Creation Helper
# =============================================================================

def create_empty_grants_detail_df() -> pd.DataFrame:
    """Create an empty DataFrame with canonical schema."""
    return pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS)


def ensure_grants_detail_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has all canonical columns.
    Adds missing columns with empty values.
    """
    df = df.copy()
    for col in GRANTS_DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    # Reorder to canonical order
    return df[GRANTS_DETAIL_COLUMNS + [c for c in df.columns if c not in GRANTS_DETAIL_COLUMNS]]


# =============================================================================
# Validation
# =============================================================================

def validate_grants_detail_df(df: pd.DataFrame) -> tuple:
    """
    Validate a grants_detail DataFrame.
    
    Returns:
        (is_valid: bool, errors: list[str])
    """
    errors = []
    
    # Check required columns exist
    for col in GRANTS_DETAIL_REQUIRED:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if errors:
        return False, errors
    
    # Check grant_bucket values
    if "grant_bucket" in df.columns:
        invalid_buckets = df[~df["grant_bucket"].isin(GRANT_BUCKETS + [""])]["grant_bucket"].unique()
        if len(invalid_buckets) > 0:
            errors.append(f"Invalid grant_bucket values: {list(invalid_buckets)}")
    
    # Check grant_amount is numeric
    if "grant_amount" in df.columns:
        try:
            pd.to_numeric(df["grant_amount"], errors="raise")
        except:
            errors.append("grant_amount column contains non-numeric values")
    
    return len(errors) == 0, errors


# =============================================================================
# Merge Helper (append without overwrite)
# =============================================================================

def merge_grants_detail(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> tuple:
    """
    Merge new grants into existing grants_detail, avoiding duplicates.
    
    Uses a composite key of: foundation_ein + grantee_name + grant_amount + fiscal_year
    
    Returns:
        (merged_df, stats_dict)
    """
    if existing_df.empty:
        return ensure_grants_detail_columns(new_df), {
            "existing": 0,
            "new": len(new_df),
            "added": len(new_df),
            "skipped": 0
        }
    
    if new_df.empty:
        return ensure_grants_detail_columns(existing_df), {
            "existing": len(existing_df),
            "new": 0,
            "added": 0,
            "skipped": 0
        }
    
    # Create composite key for deduplication
    def make_key(row):
        return f"{row.get('foundation_ein', '')}|{row.get('grantee_name', '')}|{row.get('grant_amount', '')}|{row.get('fiscal_year', '')}"
    
    existing_df = ensure_grants_detail_columns(existing_df)
    new_df = ensure_grants_detail_columns(new_df)
    
    existing_keys = set(existing_df.apply(make_key, axis=1))
    new_keys = new_df.apply(make_key, axis=1)
    
    mask = ~new_keys.isin(existing_keys)
    to_add = new_df[mask]
    
    merged = pd.concat([existing_df, to_add], ignore_index=True)
    
    return merged, {
        "existing": len(existing_df),
        "new": len(new_df),
        "added": len(to_add),
        "skipped": len(new_df) - len(to_add)
    }
