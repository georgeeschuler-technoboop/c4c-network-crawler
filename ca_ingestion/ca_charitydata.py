"""
Canadian Charity Ingestion (charitydata.ca)

Ingests charitydata.ca exports for Canadian foundations/charities.

Expected input structure:
    /data/ca_charities/<org_slug>/
        assets.csv
        directors-trustees.csv
        grants_recent.csv

Output structure:
    /outputs/ca_charities/<org_slug>/
        org_attributes.json
        nodes_people.csv
        nodes_donees.csv
        edges_board_membership.csv
        edges_grants.csv

Usage:
    python ca_charitydata.py <org_slug>
    python ca_charitydata.py --all
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Header Parsing
# =============================================================================

HEADER_RE = re.compile(r"^(.*)\s+\((\d{9}RR\d{4})\)\s*$")


def read_charitydata_header(csv_path: Path) -> Dict[str, Optional[str]]:
    """
    charitydata.ca exports begin with a non-CSV header line like:
      TORONTO FOUNDATION (136491875RR0001)

    Returns:
      {
        "source_header": "...",
        "legal_name": "...",
        "cra_bn": "#########RR####" or None
      }
    """
    if not csv_path.exists():
        return {"source_header": "", "legal_name": "", "cra_bn": None}
    
    with csv_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().lstrip("\ufeff").strip().strip('"').rstrip(",")

    m = HEADER_RE.match(first)
    if m:
        return {"source_header": first, "legal_name": m.group(1).strip(), "cra_bn": m.group(2)}
    return {"source_header": first, "legal_name": first, "cra_bn": None}


# =============================================================================
# CSV Loaders
# =============================================================================

def load_assets_table(path: Path) -> pd.DataFrame:
    """Load assets.csv, skipping the charitydata header line."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, skiprows=1)


def load_directors_table(path: Path) -> pd.DataFrame:
    """Load directors-trustees.csv, skipping the charitydata header line."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, skiprows=1)


def load_grants_table(path: Path) -> pd.DataFrame:
    """Load grants_recent.csv, skipping the charitydata header line."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, skiprows=1)


# =============================================================================
# Asset Extraction
# =============================================================================

def latest_year_column(df: pd.DataFrame) -> Optional[str]:
    """
    Finds the latest (max) year column among columns that look like YYYY.
    Returns the column name as a string, or None.
    """
    years = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit() and len(s) == 4:
            years.append(int(s))
    if not years:
        return None
    return str(max(years))


def extract_total_assets_latest(assets_df: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    """
    Extracts Total assets ($) for the latest year available.
    
    Assets.csv structure:
        Assets,2020,2021,2022,2023
        Total assets ($),1000000,1100000,...
    """
    if assets_df.empty:
        return None, None
    
    year_col = latest_year_column(assets_df)
    if not year_col:
        return None, None

    # Find the "Total assets ($)" row (exact match preferred)
    if "Assets" not in assets_df.columns:
        return int(year_col), None
    
    row = assets_df.loc[assets_df["Assets"].astype(str).str.strip() == "Total assets ($)"]
    if row.empty:
        # fallback: contains 'Total assets'
        row = assets_df.loc[assets_df["Assets"].astype(str).str.contains("Total assets", case=False, na=False)]

    if row.empty:
        return int(year_col), None

    val = row.iloc[0].get(year_col)
    try:
        return int(year_col), float(val) if pd.notna(val) else None
    except Exception:
        return int(year_col), None


# =============================================================================
# Slug Utilities
# =============================================================================

def slugify_loose(text: str) -> str:
    """
    Lightweight slug for donee names (MVP only).
    """
    text = (text or "").strip().lower()
    text = re.sub(r"&|\+", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


# =============================================================================
# Main Ingestion Function
# =============================================================================

def ingest_ca_charity(org_slug: str, org_dir: Path, out_dir: Path) -> Dict:
    """
    Ingests charitydata.ca exports for a single Canadian org.

    Inputs (expected):
      - assets.csv
      - directors-trustees.csv
      - grants_recent.csv

    Writes:
      - org_attributes.json
      - nodes_people.csv
      - nodes_donees.csv
      - edges_board_membership.csv
      - edges_grants.csv
    
    Returns:
      dict with diagnostics (files_found, files_missing, counts, errors)
    """
    org_dir = Path(org_dir)
    out_org_dir = Path(out_dir) / org_slug
    out_org_dir.mkdir(parents=True, exist_ok=True)

    assets_path = org_dir / "assets.csv"
    directors_path = org_dir / "directors-trustees.csv"
    grants_path = org_dir / "grants_recent.csv"

    # Track diagnostics
    diagnostics = {
        "org_slug": org_slug,
        "files_found": [],
        "files_missing": [],
        "directors_count": 0,
        "grants_count": 0,
        "donees_count": 0,
        "errors": [],
    }
    
    for path, name in [(assets_path, "assets.csv"), (directors_path, "directors-trustees.csv"), (grants_path, "grants_recent.csv")]:
        if path.exists():
            diagnostics["files_found"].append(name)
        else:
            diagnostics["files_missing"].append(name)

    # --- Org header (try assets.csv first; fall back to grants if needed)
    if assets_path.exists():
        header = read_charitydata_header(assets_path)
    elif grants_path.exists():
        header = read_charitydata_header(grants_path)
    elif directors_path.exists():
        header = read_charitydata_header(directors_path)
    else:
        header = {"source_header": "", "legal_name": org_slug, "cra_bn": None}
        diagnostics["errors"].append("No input files found")

    # --- Assets
    assets_df = load_assets_table(assets_path)
    latest_year, total_assets = extract_total_assets_latest(assets_df)

    org_attributes = {
        "org_slug": org_slug,
        "jurisdiction": "CA",
        "data_source": "charitydata.ca",
        "legal_name": header.get("legal_name"),
        "cra_bn": header.get("cra_bn"),
        "total_assets_latest_year": latest_year,
        "total_assets_latest_value": total_assets,
    }
    (out_org_dir / "org_attributes.json").write_text(json.dumps(org_attributes, indent=2), encoding="utf-8")

    # --- Directors / trustees -> people nodes + membership edges
    directors_df = load_directors_table(directors_path)

    people_rows = []
    membership_rows = []

    if not directors_df.empty:
        for _, r in directors_df.iterrows():
            last = str(r.get("Last Name", "")).strip()
            first = str(r.get("First Name", "")).strip()
            position = str(r.get("Position", "")).strip()
            appointed = str(r.get("Appointed", "")).strip()
            ceased = str(r.get("Ceased", "")).strip()
            arms = str(r.get("At Arm's Length", "")).strip()

            # Skip rows with no name
            if not last and not first:
                continue
            
            # Clean up 'nan' strings
            if last.lower() == "nan":
                last = ""
            if first.lower() == "nan":
                first = ""
            if position.lower() == "nan":
                position = ""
            if appointed.lower() == "nan":
                appointed = ""
            if ceased.lower() == "nan":
                ceased = ""
            if arms.lower() == "nan":
                arms = ""

            person_id = f"person:{org_slug}:{last}|{first}|{appointed}"

            people_rows.append({
                "person_id": person_id,
                "org_slug_context": org_slug,
                "first_name": first,
                "last_name": last,
            })

            membership_rows.append({
                "edge_type": "BOARD_MEMBERSHIP",
                "source_id": person_id,
                "target_id": f"org:{org_slug}",
                "position": position,
                "appointed": appointed,
                "ceased": ceased,
                "at_arms_length": arms,
            })

    diagnostics["directors_count"] = len(people_rows)
    
    pd.DataFrame(people_rows).drop_duplicates().to_csv(out_org_dir / "nodes_people.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(out_org_dir / "edges_board_membership.csv", index=False)

    # --- Grants -> org -> donee edges (+ minimal donee node ids)
    grants_df = load_grants_table(grants_path)

    grant_edges = []
    donee_nodes = []

    if not grants_df.empty:
        for _, r in grants_df.iterrows():
            period = r.get("Reporting Period", "")
            donee = str(r.get("Donee Name", "")).strip()
            city = str(r.get("City", "")).strip()
            prov = str(r.get("Prov", "")).strip()
            amt = r.get("Reported Amount ($)")
            gik = r.get("Gifts In Kind ($)")

            # Skip rows with no donee name
            if not donee or donee.lower() == "nan":
                continue
            
            # Clean up 'nan' strings
            if city.lower() == "nan":
                city = ""
            if prov.lower() == "nan":
                prov = ""

            donee_slug = slugify_loose(donee)
            donee_id = f"donee:{donee_slug}:{prov}" if prov else f"donee:{donee_slug}"

            donee_nodes.append({
                "donee_id": donee_id,
                "donee_name": donee,
                "city": city,
                "prov": prov,
            })

            grant_edges.append({
                "edge_type": "GRANT",
                "source_id": f"org:{org_slug}",
                "target_id": donee_id,
                "reporting_period": period,
                "reported_amount": amt,
                "gifts_in_kind": gik,
                "city": city,
                "prov": prov,
            })

    diagnostics["grants_count"] = len(grant_edges)
    diagnostics["donees_count"] = len(pd.DataFrame(donee_nodes).drop_duplicates()) if donee_nodes else 0

    pd.DataFrame(donee_nodes).drop_duplicates().to_csv(out_org_dir / "nodes_donees.csv", index=False)
    pd.DataFrame(grant_edges).to_csv(out_org_dir / "edges_grants.csv", index=False)
    
    diagnostics["legal_name"] = header.get("legal_name", "")
    diagnostics["output_dir"] = str(out_org_dir)
    
    return diagnostics


# =============================================================================
# Batch Processing
# =============================================================================

def ingest_all_ca_charities(data_dir: Path, out_dir: Path) -> List[Dict]:
    """
    Ingest all Canadian charities in a directory.
    
    Args:
        data_dir: Path to ca_charities directory containing org_slug folders
        out_dir: Output directory base
    
    Returns:
        List of diagnostics dicts, one per org
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return []
    
    results = []
    
    org_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    for org_dir in org_dirs:
        org_slug = org_dir.name
        result = ingest_ca_charity(org_slug, org_dir, out_dir)
        results.append(result)
    
    return results


# =============================================================================
# CLI
# =============================================================================

def print_diagnostics(diag: Dict) -> None:
    """Pretty-print diagnostics for one org."""
    status = "✓" if not diag.get("errors") else "⚠"
    print(f"  {status} {diag['org_slug']}")
    if diag.get("legal_name"):
        print(f"      {diag['legal_name']}")
    print(f"      Directors: {diag['directors_count']}, Grants: {diag['grants_count']}, Donees: {diag['donees_count']}")
    if diag.get("files_missing"):
        print(f"      ⚠ Missing: {', '.join(diag['files_missing'])}")
    if diag.get("errors"):
        for err in diag["errors"]:
            print(f"      ❌ {err}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ca_charitydata.py <org_slug> [data_dir] [output_dir]")
        print("       python ca_charitydata.py --all [data_dir] [output_dir]")
        print("")
        print("Defaults:")
        print("  data_dir   = ./data/ca_charities")
        print("  output_dir = ./outputs/ca_charities")
        sys.exit(1)
    
    org_slug = sys.argv[1]
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./data/ca_charities")
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./outputs/ca_charities")
    
    if org_slug == "--all":
        print(f"\n=== Ingesting all CA charities from {data_dir} ===\n")
        results = ingest_all_ca_charities(data_dir, output_dir)
        
        print(f"Processed {len(results)} organizations:\n")
        for diag in results:
            print_diagnostics(diag)
            print()
        
        # Summary
        total_directors = sum(r["directors_count"] for r in results)
        total_grants = sum(r["grants_count"] for r in results)
        total_donees = sum(r["donees_count"] for r in results)
        print(f"=== Totals: {total_directors} directors, {total_grants} grants, {total_donees} donees ===")
    else:
        org_dir = data_dir / org_slug
        
        if not org_dir.exists():
            print(f"❌ Directory not found: {org_dir}")
            sys.exit(1)
        
        print(f"\n=== Ingesting {org_slug} ===\n")
        diag = ingest_ca_charity(org_slug, org_dir, output_dir)
        print_diagnostics(diag)
        print(f"\n  Output written to: {diag['output_dir']}")
