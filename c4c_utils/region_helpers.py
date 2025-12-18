"""
Region Mode Helper — Shared by OrgGraph US and OrgGraph CA

Provides consistent region filtering logic across all C4C apps.

Region Modes:
- OFF: No regional filtering (region_relevant = True for all)
- PRESET: Use predefined region sets (e.g., Great Lakes)
- CUSTOM: User-specified admin1 codes + country codes

Usage:
    from c4c_utils.region_helpers import RegionMode, REGION_PRESETS, compute_region_relevant

    # Get preset
    preset = REGION_PRESETS["great_lakes"]
    
    # Compute relevance for a grant
    is_relevant = compute_region_relevant(
        grantee_state="MI",
        grantee_country="US",
        region_mode=RegionMode.PRESET,
        admin1_codes=preset["admin1_codes"],
        country_codes=preset["country_codes"]
    )
"""

from enum import Enum
from typing import Optional

# =============================================================================
# Region Mode Enum
# =============================================================================

class RegionMode(Enum):
    OFF = "off"
    PRESET = "preset"
    CUSTOM = "custom"


# =============================================================================
# Region Presets
# =============================================================================

REGION_PRESETS = {
    "great_lakes": {
        "label": "Great Lakes Region",
        "description": "US Great Lakes states + Ontario & Quebec",
        "admin1_codes": ["MI", "OH", "MN", "WI", "IN", "IL", "NY", "PA", "ON", "QC"],
        "country_codes": ["US", "CA"],
    },
    "midwest_us": {
        "label": "US Midwest",
        "description": "US Midwest states only",
        "admin1_codes": ["MI", "OH", "MN", "WI", "IN", "IL", "IA", "MO", "ND", "SD", "NE", "KS"],
        "country_codes": ["US"],
    },
    "northeast_us": {
        "label": "US Northeast",
        "description": "US Northeast states",
        "admin1_codes": ["NY", "PA", "NJ", "CT", "MA", "RI", "VT", "NH", "ME"],
        "country_codes": ["US"],
    },
    "california": {
        "label": "California",
        "description": "California only",
        "admin1_codes": ["CA"],
        "country_codes": ["US"],
    },
    "ontario": {
        "label": "Ontario",
        "description": "Ontario only",
        "admin1_codes": ["ON"],
        "country_codes": ["CA"],
    },
    "british_columbia": {
        "label": "British Columbia",
        "description": "British Columbia only",
        "admin1_codes": ["BC"],
        "country_codes": ["CA"],
    },
    "canada_all": {
        "label": "All Canada",
        "description": "All Canadian provinces and territories",
        "admin1_codes": ["AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"],
        "country_codes": ["CA"],
    },
    "us_all": {
        "label": "All US",
        "description": "All US states",
        "admin1_codes": [],  # Empty means all US states
        "country_codes": ["US"],
    },
}

# Default preset key
DEFAULT_PRESET = "great_lakes"


# =============================================================================
# US State and Canadian Province Codes (for validation/UI)
# =============================================================================

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP"
]

CA_PROVINCES = [
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"
]

ALL_ADMIN1_CODES = US_STATES + CA_PROVINCES


# =============================================================================
# Region Relevance Computation
# =============================================================================

def normalize_code(code: Optional[str]) -> str:
    """Normalize a state/province/country code to uppercase."""
    if not code or str(code).strip().lower() in ("", "nan", "none"):
        return ""
    return str(code).strip().upper()


def compute_region_relevant(
    grantee_state: Optional[str],
    grantee_country: Optional[str],
    region_mode: RegionMode,
    admin1_codes: list = None,
    country_codes: list = None
) -> bool:
    """
    Compute whether a grant is region-relevant.
    
    Args:
        grantee_state: State/province code (e.g., "MI", "ON")
        grantee_country: Country code (e.g., "US", "CA")
        region_mode: OFF, PRESET, or CUSTOM
        admin1_codes: List of state/province codes to match
        country_codes: List of country codes to match
    
    Returns:
        True if the grant is region-relevant, False otherwise.
    
    Rules:
        - OFF mode: Always returns True
        - PRESET/CUSTOM mode:
            - grantee_state must be in admin1_codes (if admin1_codes is non-empty)
            - grantee_country must be in country_codes (if country_codes is non-empty)
            - If admin1_codes is empty, only country_codes is checked
    """
    if region_mode == RegionMode.OFF:
        return True
    
    # Normalize inputs
    state = normalize_code(grantee_state)
    country = normalize_code(grantee_country)
    
    admin1_codes = [normalize_code(c) for c in (admin1_codes or [])]
    admin1_codes = [c for c in admin1_codes if c]  # Remove empties
    
    country_codes = [normalize_code(c) for c in (country_codes or [])]
    country_codes = [c for c in country_codes if c]
    
    # Check country first
    if country_codes:
        if not country or country not in country_codes:
            return False
    
    # Check admin1 (state/province)
    if admin1_codes:
        if not state or state not in admin1_codes:
            return False
    
    # If we get here, either:
    # - Both checks passed
    # - admin1_codes was empty (only country checked)
    # - Both were empty (should use OFF mode instead, but return True)
    return True


def get_preset(preset_key: str) -> dict:
    """Get a region preset by key. Returns None if not found."""
    return REGION_PRESETS.get(preset_key)


def get_preset_options() -> list:
    """Get list of (key, label) tuples for UI dropdown."""
    return [(k, v["label"]) for k, v in REGION_PRESETS.items()]


# =============================================================================
# Streamlit UI Helpers
# =============================================================================

def render_region_mode_selector(st_module, default_mode: RegionMode = RegionMode.OFF):
    """
    Render region mode selector in Streamlit.
    
    Args:
        st_module: The streamlit module (pass `st`)
        default_mode: Default region mode
    
    Returns:
        dict with keys: mode, preset_key, admin1_codes, country_codes
    """
    st = st_module
    
    # Mode selector
    mode_options = {
        "Off (include all grants)": RegionMode.OFF,
        "Preset region": RegionMode.PRESET,
        "Custom selection": RegionMode.CUSTOM,
    }
    
    mode_labels = list(mode_options.keys())
    default_idx = 0
    for i, (_, mode) in enumerate(mode_options.items()):
        if mode == default_mode:
            default_idx = i
            break
    
    selected_label = st.radio(
        "Region Mode",
        mode_labels,
        index=default_idx,
        horizontal=True,
        help="Filter grants by recipient location"
    )
    
    selected_mode = mode_options[selected_label]
    
    result = {
        "mode": selected_mode,
        "preset_key": None,
        "admin1_codes": [],
        "country_codes": [],
    }
    
    if selected_mode == RegionMode.OFF:
        return result
    
    if selected_mode == RegionMode.PRESET:
        # Preset dropdown
        preset_options = get_preset_options()
        preset_labels = [label for _, label in preset_options]
        preset_keys = [key for key, _ in preset_options]
        
        default_preset_idx = preset_keys.index(DEFAULT_PRESET) if DEFAULT_PRESET in preset_keys else 0
        
        selected_preset_label = st.selectbox(
            "Select region preset",
            preset_labels,
            index=default_preset_idx
        )
        
        selected_preset_key = preset_keys[preset_labels.index(selected_preset_label)]
        preset = get_preset(selected_preset_key)
        
        result["preset_key"] = selected_preset_key
        result["admin1_codes"] = preset["admin1_codes"]
        result["country_codes"] = preset["country_codes"]
        
        # Show what's included
        st.caption(f"**Includes:** {', '.join(preset['admin1_codes'])} ({', '.join(preset['country_codes'])})")
    
    elif selected_mode == RegionMode.CUSTOM:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_countries = st.multiselect(
                "Countries",
                ["US", "CA"],
                default=["US", "CA"]
            )
        
        with col2:
            # Show relevant admin1 codes based on selected countries
            available_admin1 = []
            if "US" in selected_countries:
                available_admin1.extend(US_STATES)
            if "CA" in selected_countries:
                available_admin1.extend(CA_PROVINCES)
            
            selected_admin1 = st.multiselect(
                "States/Provinces",
                available_admin1,
                default=[]
            )
        
        result["admin1_codes"] = selected_admin1
        result["country_codes"] = selected_countries
    
    return result


# =============================================================================
# Batch Processing Helper
# =============================================================================

def add_region_relevant_column(
    df,
    state_col: str,
    country_col: str,
    region_mode: RegionMode,
    admin1_codes: list = None,
    country_codes: list = None
):
    """
    Add 'region_relevant' column to a DataFrame.
    
    Args:
        df: pandas DataFrame with grant data
        state_col: Name of state/province column
        country_col: Name of country column
        region_mode: Region mode to apply
        admin1_codes: List of admin1 codes (for PRESET/CUSTOM)
        country_codes: List of country codes (for PRESET/CUSTOM)
    
    Returns:
        DataFrame with 'region_relevant' boolean column added
    """
    import pandas as pd
    
    df = df.copy()
    
    if region_mode == RegionMode.OFF:
        df["region_relevant"] = True
        return df
    
    df["region_relevant"] = df.apply(
        lambda row: compute_region_relevant(
            grantee_state=row.get(state_col),
            grantee_country=row.get(country_col),
            region_mode=region_mode,
            admin1_codes=admin1_codes,
            country_codes=country_codes
        ),
        axis=1
    )
    
    return df
