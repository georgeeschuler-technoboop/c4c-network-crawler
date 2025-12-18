# Canonical grants_detail.csv Schema

**Version:** 1.0 (December 2024)  
**Used by:** OrgGraph US v0.15.0+, OrgGraph CA, Insight Engine v0.6.0+

## Overview

This document defines the canonical schema for `grants_detail.csv`, the shared grant-level data file produced by OrgGraph US and OrgGraph CA and consumed by Insight Engine.

## Column Definitions

### Core Columns (Required)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `foundation_name` | string | Name of the granting organization | "Joyce Foundation" |
| `foundation_ein` | string | Tax ID (EIN for US, BN for CA) | "366079185" |
| `tax_year` | string | Tax year of the filing | "2023" |
| `grantee_name` | string | Name of the recipient organization | "Alliance for the Great Lakes" |
| `grantee_city` | string | City of the grantee | "Chicago" |
| `grantee_state` | string | State/province code | "IL", "ON" |
| `grant_amount` | float | Grant amount in local currency | 150000.00 |
| `grant_purpose_raw` | string | Raw purpose text from filing | "For water quality monitoring" |
| `grant_bucket` | string | Source bucket within the filing | "3a", "3b", "schedule_i", "ca_t3010" |
| `region_relevant` | boolean | True if grantee is in the selected region | true |
| `source_file` | string | Original filename processed | "joyce_2023.xml" |

### Extended Columns (Optional but Recommended)

| Column | Type | Description | Default |
|--------|------|-------------|---------|
| `grantee_country` | string | Country code | "US", "CA" |
| `foundation_country` | string | Funder country code | "US", "CA" |
| `source_system` | string | Which OrgGraph produced this | "IRS_990", "CHARITYDATA_CA" |
| `grant_amount_cash` | float | Cash portion (if split) | "" |
| `grant_amount_in_kind` | float | In-kind portion (if split) | "" |
| `currency` | string | Currency code | "USD", "CAD" |
| `fiscal_year` | string | Fiscal year (may differ from tax_year) | "" |
| `reporting_period` | string | Filing period | "" |

## Grant Bucket Values

| Value | Description | Source |
|-------|-------------|--------|
| `3a` | Part XIV Line 3a - Grants paid this year | US 990-PF |
| `3b` | Part XIV Line 3b - Future commitments | US 990-PF |
| `schedule_i` | Schedule I grants | US 990 |
| `ca_t3010` | T3010 Schedule 2 | Canada T3010 |
| `unknown` | Bucket not determined | Fallback |

## Column Aliases (for Backward Compatibility)

Insight Engine maintains aliases for legacy naming:

| Canonical (OrgGraph) | Legacy (Insight Engine) |
|----------------------|-------------------------|
| `foundation_name` | `funder_name` |
| `foundation_ein` | `funder_id` |
| `foundation_country` | `funder_country` |
| `grantee_state` | `grantee_admin1` |

## Append Behavior

When adding new filings to an existing project:

1. Load existing `grants_detail.csv` if present
2. Deduplicate using composite key: `foundation_ein + grantee_name + grant_amount + fiscal_year`
3. Append only new rows
4. Save merged result

This ensures:
- No duplicate grants when re-processing filings
- Cumulative build-up across multiple uploads
- Safe re-runs without data loss

## File Location

```
demo_data/
└── {project_name}/
    ├── nodes.csv           # Required
    ├── edges.csv           # Required
    ├── grants_detail.csv   # Optional (enables analytics)
    ├── config.json         # Project settings
    └── ...
```

## Usage Examples

### OrgGraph US (Producer)

```python
from app import ensure_grants_detail_columns, merge_grants_detail

# After parsing files
grants_df = ensure_grants_detail_columns(parsed_grants, source_file="joyce_2023.xml")

# Merge with existing
existing = pd.read_csv(project_path / "grants_detail.csv")
merged, stats = merge_grants_detail(existing, grants_df)
merged.to_csv(project_path / "grants_detail.csv", index=False)
```

### Insight Engine (Consumer)

```python
from app import normalize_grants_detail_columns, summarize_grants_by_bucket

# Load and normalize
grants_df = pd.read_csv(project_path / "grants_detail.csv")
grants_df = normalize_grants_detail_columns(grants_df)

# Summarize
all_summary = summarize_grants_by_bucket(grants_df, region_only=False)
region_summary = summarize_grants_by_bucket(grants_df, region_only=True)
```

## Migration Notes

### From OrgGraph US v0.14.x to v0.15.0

- `grants_detail.csv` now saved to project folder (not just ZIP)
- All 19 canonical columns ensured on export
- Append behavior implemented

### From Insight Engine v0.5.x to v0.6.0

- Column aliasing handles both naming conventions
- Grant Network Results section shows All vs Region Relevant
- Bucket breakdown includes CA buckets
