"""
C4C Canadian Charity Ingestion — Streamlit App

Upload charitydata.ca exports and generate network-ready CSVs.

Outputs conform to C4C Network Schema v1 (MVP):
- nodes.csv: ORG and PERSON nodes
- edges.csv: GRANT and BOARD_MEMBERSHIP edges

Supports two modes:
- Existing Project: Load pre-curated data from repo (e.g., GLFN demo)
- New Project: Upload files for ad-hoc analysis
"""

import streamlit as st
import pandas as pd
import json
import re
import hashlib
import zipfile
import os
from pathlib import Path
from io import BytesIO, StringIO

# =============================================================================
# Config
# =============================================================================

C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_9e232fe9e6914305a7ea8746e2e77125~mv2.png"
SOURCE_SYSTEM = "CHARITYDATA_CA"
JURISDICTION = "CA"
CURRENCY = "CAD"

# Demo data location (relative to app.py)
DEMO_DATA_ROOT = Path(__file__).parent.parent / "demo_data" / "ca"

st.set_page_config(
    page_title="C4C CA Charity Ingestion",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)

# =============================================================================
# Header
# =============================================================================

col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.image(C4C_LOGO_URL, width=60)
with col_title:
    st.title("C4C Canadian Charity Ingestion")

st.markdown("""
Parse **charitydata.ca** exports for Canadian foundations and generate:
- **Unified nodes.csv** (organizations + people)
- **Unified edges.csv** (grants + board memberships)

*Outputs conform to C4C Network Schema v1.*
""")

st.divider()

# =============================================================================
# Project Mode Selector
# =============================================================================

st.subheader("📂 Select Project Mode")

project_mode = st.radio(
    "How would you like to load data?",
    ["🗂️ Existing Project (GLFN Demo)", "📤 New Project (Upload Files)"],
    horizontal=True,
    help="Existing Project loads pre-curated data. New Project lets you upload your own files."
)

is_existing_project = "Existing" in project_mode

st.divider()

# =============================================================================
# Parsing Functions
# =============================================================================

HEADER_RE = re.compile(r"^(.*)\s+\((\d{9}RR\d{4})\)\s*$")


def read_charitydata_csv_from_upload(uploaded_file) -> tuple:
    """
    Read a charitydata.ca CSV file from Streamlit upload.
    
    Returns: (DataFrame, org_name, cra_bn)
    """
    content = uploaded_file.getvalue().decode("utf-8")
    return _parse_charitydata_content(content)


def read_charitydata_csv_from_path(file_path: Path) -> tuple:
    """
    Read a charitydata.ca CSV file from disk.
    
    Returns: (DataFrame, org_name, cra_bn)
    """
    if not file_path.exists():
        return pd.DataFrame(), "", ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return _parse_charitydata_content(content)


def _parse_charitydata_content(content: str) -> tuple:
    """
    Parse charitydata.ca CSV content.
    
    Returns: (DataFrame, org_name, cra_bn)
    """
    lines = content.split("\n")
    
    if not lines:
        return pd.DataFrame(), "", ""
    
    # Parse header line
    header_line = lines[0].strip().lstrip("\ufeff").strip().strip('"').rstrip(",")
    m = HEADER_RE.match(header_line)
    
    if m:
        org_name = m.group(1).strip()
        cra_bn = m.group(2)
    else:
        org_name = header_line
        cra_bn = ""
    
    # Parse CSV (skip header line)
    remaining = "\n".join(lines[1:])
    if remaining.strip():
        df = pd.read_csv(StringIO(remaining))
    else:
        df = pd.DataFrame()
    
    return df, org_name, cra_bn


def scan_demo_orgs(demo_root: Path) -> list:
    """
    Scan demo data folder for available org_slug directories.
    
    Returns: list of (org_slug, display_name) tuples
    """
    if not demo_root.exists():
        return []
    
    orgs = []
    for org_dir in sorted(demo_root.iterdir()):
        if org_dir.is_dir() and not org_dir.name.startswith('.'):
            # Try to get display name from assets.csv header
            display_name = org_dir.name.replace("-", " ").title()
            assets_path = org_dir / "assets.csv"
            if assets_path.exists():
                _, org_name, _ = read_charitydata_csv_from_path(assets_path)
                if org_name:
                    display_name = org_name
            orgs.append((org_dir.name, display_name))
    
    return orgs


def latest_year_column(df: pd.DataFrame) -> str:
    """Find the latest year column (YYYY format)."""
    years = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit() and len(s) == 4:
            years.append(int(s))
    return str(max(years)) if years else ""


def extract_total_assets(assets_df: pd.DataFrame) -> tuple:
    """Extract total assets for latest year."""
    if assets_df.empty or "Assets" not in assets_df.columns:
        return None, None
    
    year_col = latest_year_column(assets_df)
    if not year_col:
        return None, None
    
    row = assets_df.loc[assets_df["Assets"].astype(str).str.strip() == "Total assets ($)"]
    if row.empty:
        row = assets_df.loc[assets_df["Assets"].astype(str).str.contains("Total assets", case=False, na=False)]
    
    if row.empty:
        return int(year_col), None
    
    val = row.iloc[0].get(year_col)
    try:
        return int(year_col), float(val) if pd.notna(val) else None
    except:
        return int(year_col), None


def slugify_loose(text: str) -> str:
    """Lightweight slug for org names."""
    text = (text or "").strip().lower()
    text = re.sub(r"&|\+", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def clean_nan(val) -> str:
    """Clean up 'nan' strings."""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def extract_fiscal_year(reporting_period: str) -> int:
    """Extract year from reporting period (e.g., '2023-12-31' -> 2023)."""
    if not reporting_period:
        return None
    rp = str(reporting_period).strip()
    # Try to extract 4-digit year
    match = re.search(r'(\d{4})', rp)
    if match:
        return int(match.group(1))
    return None


def generate_edge_hash(s: str) -> str:
    """Generate short hash for edge ID."""
    return hashlib.md5(s.encode()).hexdigest()[:8]


# =============================================================================
# Data Input (Mode-Dependent)
# =============================================================================

# Initialize file/data holders
assets_df = pd.DataFrame()
directors_df = pd.DataFrame()
grants_df = pd.DataFrame()
org_name = ""
cra_bn = ""
data_loaded = False

if is_existing_project:
    # -------------------------------------------------------------------------
    # Existing Project Mode: Load from demo_data folder
    # -------------------------------------------------------------------------
    st.subheader("🗂️ Select Organization")
    
    available_orgs = scan_demo_orgs(DEMO_DATA_ROOT)
    
    if not available_orgs:
        st.warning(f"""
        **No demo data found.**
        
        To use Existing Project mode, add organization folders to:
        ```
        demo_data/ca/<org_slug>/
          ├── assets.csv
          ├── directors-trustees.csv
          └── grants.csv
        ```
        
        Or switch to **New Project** mode to upload files.
        """)
    else:
        # Build selection options
        org_options = {f"{display} ({slug})": slug for slug, display in available_orgs}
        
        selected_display = st.selectbox(
            "Choose an organization to analyze:",
            options=list(org_options.keys()),
            help=f"Found {len(available_orgs)} organizations in demo data"
        )
        
        if selected_display:
            selected_slug = org_options[selected_display]
            org_dir = DEMO_DATA_ROOT / selected_slug
            
            # Show what files are available
            assets_path = org_dir / "assets.csv"
            directors_path = org_dir / "directors-trustees.csv"
            grants_path = org_dir / "grants.csv"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if assets_path.exists():
                    st.success("✅ assets.csv")
                else:
                    st.warning("⚠️ assets.csv not found")
            with col2:
                if directors_path.exists():
                    st.success("✅ directors-trustees.csv")
                else:
                    st.warning("⚠️ directors-trustees.csv not found")
            with col3:
                if grants_path.exists():
                    st.success("✅ grants.csv")
                else:
                    st.warning("⚠️ grants.csv not found")
            
            # Load the data
            if assets_path.exists():
                assets_df, org_name, cra_bn = read_charitydata_csv_from_path(assets_path)
            if directors_path.exists():
                directors_df, dir_org, dir_bn = read_charitydata_csv_from_path(directors_path)
                if not org_name:
                    org_name = dir_org
                if not cra_bn:
                    cra_bn = dir_bn
            if grants_path.exists():
                grants_df, gr_org, gr_bn = read_charitydata_csv_from_path(grants_path)
                if not org_name:
                    org_name = gr_org
                if not cra_bn:
                    cra_bn = gr_bn
            
            data_loaded = not assets_df.empty or not directors_df.empty or not grants_df.empty

else:
    # -------------------------------------------------------------------------
    # New Project Mode: Upload files
    # -------------------------------------------------------------------------
    st.subheader("📤 Upload Files")
    
    st.markdown("""
    Upload the CSV files exported from **charitydata.ca** for one organization:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        assets_file = st.file_uploader(
            "assets.csv",
            type=["csv"],
            help="Financial data with total assets by year"
        )
    
    with col2:
        directors_file = st.file_uploader(
            "directors-trustees.csv",
            type=["csv"],
            help="Board members and trustees"
        )
    
    with col3:
        grants_file = st.file_uploader(
            "grants.csv",
            type=["csv"],
            help="Grants to qualified donees (auto-filters to most recent year)"
        )
    
    # Load uploaded files
    if assets_file:
        assets_df, org_name, cra_bn = read_charitydata_csv_from_upload(assets_file)
    if directors_file:
        directors_df, dir_org, dir_bn = read_charitydata_csv_from_upload(directors_file)
        if not org_name:
            org_name = dir_org
        if not cra_bn:
            cra_bn = dir_bn
    if grants_file:
        grants_df, gr_org, gr_bn = read_charitydata_csv_from_upload(grants_file)
        if not org_name:
            org_name = gr_org
        if not cra_bn:
            cra_bn = gr_bn
    
    data_loaded = assets_file or directors_file or grants_file

st.divider()

# =============================================================================
# Process Data
# =============================================================================

if data_loaded:
    
    # Initialize processing variables
    org_slug = slugify_loose(org_name) if org_name else ""
    latest_year = None
    total_assets = None
    fiscal_year = None
    foundation_node_id = ""
    
    # Canonical schema collections
    nodes = []  # All nodes (ORG + PERSON)
    edges = []  # All edges (GRANT + BOARD_MEMBERSHIP)
    
    seen_node_ids = set()  # Deduplication
    
    # -------------------------------------------------------------------------
    # Extract assets info
    # -------------------------------------------------------------------------
    if not assets_df.empty:
        latest_year, total_assets = extract_total_assets(assets_df)
        st.success(f"✅ **assets.csv** — Loaded")
    
    # -------------------------------------------------------------------------
    # Create foundation ORG node
    # -------------------------------------------------------------------------
    if org_slug:
        foundation_node_id = f"org:{org_slug}"
        if foundation_node_id not in seen_node_ids:
            nodes.append({
                "node_id": foundation_node_id,
                "node_type": "ORG",
                "label": org_name,
                "org_slug": org_slug,
                "jurisdiction": JURISDICTION,
                "tax_id": cra_bn,
                "city": "",
                "region": "",
                "source_system": SOURCE_SYSTEM,
                "source_ref": cra_bn,
                "assets_latest": total_assets,
                "assets_year": latest_year,
                "first_name": "",
                "last_name": "",
            })
            seen_node_ids.add(foundation_node_id)
    
    # -------------------------------------------------------------------------
    # Process directors-trustees
    # -------------------------------------------------------------------------
    board_count = 0
    if not directors_df.empty and foundation_node_id:
        for _, r in directors_df.iterrows():
            last = clean_nan(r.get("Last Name", ""))
            first = clean_nan(r.get("First Name", ""))
            position = clean_nan(r.get("Position", ""))
            appointed = clean_nan(r.get("Appointed", ""))
            ceased = clean_nan(r.get("Ceased", ""))
            arms = clean_nan(r.get("At Arm's Length", ""))
            
            if not last and not first:
                continue
            
            # Person node ID (contextual identity)
            person_key = f"{org_slug}:{last}|{first}|{appointed}"
            person_node_id = f"person:{person_key}"
            
            # Add PERSON node
            if person_node_id not in seen_node_ids:
                nodes.append({
                    "node_id": person_node_id,
                    "node_type": "PERSON",
                    "label": f"{first} {last}".strip(),
                    "org_slug": "",
                    "jurisdiction": "",
                    "tax_id": "",
                    "city": "",
                    "region": "",
                    "source_system": SOURCE_SYSTEM,
                    "source_ref": f"{org_slug}/directors-trustees.csv",
                    "assets_latest": None,
                    "assets_year": None,
                    "first_name": first,
                    "last_name": last,
                })
                seen_node_ids.add(person_node_id)
            
            # Add BOARD_MEMBERSHIP edge
            edge_id = f"bm:{person_node_id}->{foundation_node_id}:{appointed or 'unknown'}"
            
            edges.append({
                "edge_id": edge_id,
                "from_id": person_node_id,
                "to_id": foundation_node_id,
                "edge_type": "BOARD_MEMBERSHIP",
                "amount": None,
                "amount_cash": None,
                "amount_in_kind": None,
                "currency": "",
                "fiscal_year": None,
                "reporting_period": "",
                "purpose": "",
                "role": position,
                "start_date": appointed,
                "end_date": ceased,
                "at_arms_length": arms,
                "city": "",
                "region": "",
                "source_system": SOURCE_SYSTEM,
                "source_ref": f"{org_slug}/directors-trustees.csv",
            })
            
            board_count += 1
        
        st.success(f"✅ **directors-trustees.csv** — {board_count} board members")
    
    # -------------------------------------------------------------------------
    # Process grants
    # -------------------------------------------------------------------------
    grant_count = 0
    if not grants_df.empty and foundation_node_id:
        # MVP DESIGN CHOICE (Canada parity):
        # charitydata.ca exports grants across many years. For sprint parity with US IRS 990 analysis
        # (which is typically based on the most recent filing), we filter grant rows to the most
        # recent fiscal year present in the dataset. We preserve the ability to analyze multi-year
        # trends later, but for now we standardize on "latest-year grants" for comparability.
        
        # Filter to most recent reporting period only
        total_rows = len(grants_df)
        grants_filtered = grants_df.copy()
        
        if "Reporting Period" in grants_filtered.columns:
            # Get unique periods and find the most recent one
            periods = grants_filtered["Reporting Period"].dropna().unique()
            if len(periods) > 0:
                # Handle edge case: some periods might be malformed
                try:
                    latest_period = sorted(periods, reverse=True)[0]
                except Exception:
                    latest_period = periods[0]  # Fallback to first available
                
                grants_filtered = grants_filtered[grants_filtered["Reporting Period"] == latest_period]
                filtered_rows = len(grants_filtered)
                fiscal_year = extract_fiscal_year(latest_period)
                st.info(f"📅 Filtered to most recent period: **{latest_period}** ({filtered_rows} of {total_rows} grants)")
        
        for _, r in grants_filtered.iterrows():
            donee = clean_nan(r.get("Donee Name", ""))
            if not donee:
                continue
            
            city = clean_nan(r.get("City", ""))
            prov = clean_nan(r.get("Prov", ""))
            period = r.get("Reporting Period", "")
            amt = r.get("Reported Amount ($)", 0)
            gik = r.get("Gifts In Kind ($)", 0)
            
            # Parse amounts
            try:
                amt_cash = float(amt) if pd.notna(amt) else 0
            except:
                amt_cash = 0
            try:
                amt_in_kind = float(gik) if pd.notna(gik) else 0
            except:
                amt_in_kind = 0
            amt_total = amt_cash + amt_in_kind
            
            # Donee org_slug
            donee_slug = f"donee-{slugify_loose(donee)}"
            if prov:
                donee_slug = f"{donee_slug}-{prov.lower()}"
            
            donee_node_id = f"org:{donee_slug}"
            
            # Add donee ORG node
            if donee_node_id not in seen_node_ids:
                nodes.append({
                    "node_id": donee_node_id,
                    "node_type": "ORG",
                    "label": donee,
                    "org_slug": donee_slug,
                    "jurisdiction": JURISDICTION,
                    "tax_id": "",
                    "city": city,
                    "region": prov,
                    "source_system": SOURCE_SYSTEM,
                    "source_ref": f"{org_slug}/grants.csv",
                    "assets_latest": None,
                    "assets_year": None,
                    "first_name": "",
                    "last_name": "",
                })
                seen_node_ids.add(donee_node_id)
            
            # Add GRANT edge
            fy = extract_fiscal_year(period) or fiscal_year
            if amt_total > 0:
                edge_id = f"gr:{foundation_node_id}->{donee_node_id}:{fy}:{int(amt_total)}"
            else:
                hash_input = f"{foundation_node_id}{donee_node_id}{fy}{period}"
                edge_id = f"gr:{foundation_node_id}->{donee_node_id}:{fy}:h{generate_edge_hash(hash_input)}"
            
            edges.append({
                "edge_id": edge_id,
                "from_id": foundation_node_id,
                "to_id": donee_node_id,
                "edge_type": "GRANT",
                "amount": amt_total,
                "amount_cash": amt_cash if amt_cash > 0 else None,
                "amount_in_kind": amt_in_kind if amt_in_kind > 0 else None,
                "currency": CURRENCY,
                "fiscal_year": fy,
                "reporting_period": str(period) if period else "",
                "purpose": "",
                "role": "",
                "start_date": "",
                "end_date": "",
                "at_arms_length": "",
                "city": city,
                "region": prov,
                "source_system": SOURCE_SYSTEM,
                "source_ref": f"{org_slug}/grants.csv",
            })
            
            grant_count += 1
        
        st.success(f"✅ **grants** — {grant_count} grants loaded")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    st.subheader("📊 Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Organization", org_name or "Unknown")
    with col2:
        st.metric("CRA BN", cra_bn or "—")
    with col3:
        if total_assets:
            st.metric(f"Total Assets ({latest_year})", f"${total_assets:,.0f}")
        else:
            st.metric("Total Assets", "—")
    
    # Count by type
    nodes_df = pd.DataFrame(nodes) if nodes else pd.DataFrame()
    edges_df = pd.DataFrame(edges) if edges else pd.DataFrame()
    
    org_nodes = len(nodes_df[nodes_df["node_type"] == "ORG"]) if not nodes_df.empty else 0
    person_nodes = len(nodes_df[nodes_df["node_type"] == "PERSON"]) if not nodes_df.empty else 0
    grant_edges_count = len(edges_df[edges_df["edge_type"] == "GRANT"]) if not edges_df.empty else 0
    board_edges = len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]) if not edges_df.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ORG Nodes", org_nodes)
    with col2:
        st.metric("PERSON Nodes", person_nodes)
    with col3:
        st.metric("GRANT Edges", grant_edges_count)
    with col4:
        st.metric("BOARD Edges", board_edges)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Data Previews
    # -------------------------------------------------------------------------
    st.subheader("📋 Data Preview")
    st.caption("Preview of extracted data. Downloads include full schema with all IDs.")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        if not nodes_df.empty:
            # Show user-friendly columns
            display_cols = ["node_type", "label", "jurisdiction", "city", "region", "tax_id"]
            display_cols = [c for c in display_cols if c in nodes_df.columns]
            st.dataframe(nodes_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"{len(nodes_df)} total nodes")
        else:
            st.info("No nodes found.")
    
    with tab2:
        if not edges_df.empty:
            # Show user-friendly columns
            display_cols = ["edge_type", "from_id", "to_id", "amount", "role", "fiscal_year"]
            display_cols = [c for c in display_cols if c in edges_df.columns]
            st.dataframe(edges_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"{len(edges_df)} total edges")
        else:
            st.info("No edges found.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Downloads
    # -------------------------------------------------------------------------
    st.subheader("📥 Download Outputs")
    
    # Define canonical column order
    NODE_COLUMNS = [
        "node_id", "node_type", "label", "org_slug", "jurisdiction", "tax_id",
        "city", "region", "source_system", "source_ref", "assets_latest", "assets_year",
        "first_name", "last_name"
    ]
    
    EDGE_COLUMNS = [
        "edge_id", "from_id", "to_id", "edge_type",
        "amount", "amount_cash", "amount_in_kind", "currency",
        "fiscal_year", "reporting_period", "purpose",
        "role", "start_date", "end_date", "at_arms_length",
        "city", "region",
        "source_system", "source_ref"
    ]
    
    # Reorder columns to canonical order
    if not nodes_df.empty:
        nodes_df = nodes_df.reindex(columns=NODE_COLUMNS)
    if not edges_df.empty:
        edges_df = edges_df.reindex(columns=EDGE_COLUMNS)
    
    # Build org_attributes.json
    org_attributes = {
        "schema_version": "1.0-mvp",
        "org_slug": org_slug,
        "jurisdiction": JURISDICTION,
        "source_system": SOURCE_SYSTEM,
        "org_legal_name": org_name,
        "org_display_name": org_name.title() if org_name else "",
        "tax_id": cra_bn,
        "source_url": "",
        "assets_latest": total_assets if total_assets is not None else "",
        "assets_year": latest_year if latest_year is not None else "",
        "notes": ""
    }
    
    def create_zip_download():
        """Create a zip file with canonical outputs."""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Canonical outputs
            if not nodes_df.empty:
                zf.writestr("nodes.csv", nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr("edges.csv", edges_df.to_csv(index=False))
            # Org metadata
            zf.writestr("org_attributes.json", json.dumps(org_attributes, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # Download All button
    st.download_button(
        label="📦 Download All (ZIP)",
        data=create_zip_download(),
        file_name=f"c4c_ca_{org_slug or 'export'}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
    # Individual downloads
    st.markdown("**Or download individually:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not nodes_df.empty:
            st.download_button(
                label="📄 nodes.csv",
                data=nodes_df.to_csv(index=False),
                file_name="nodes.csv",
                mime="text/csv"
            )
    
    with col2:
        if not edges_df.empty:
            st.download_button(
                label="📄 edges.csv",
                data=edges_df.to_csv(index=False),
                file_name="edges.csv",
                mime="text/csv"
            )
    
    with col3:
        st.download_button(
            label="📄 org_attributes.json",
            data=json.dumps(org_attributes, indent=2),
            file_name="org_attributes.json",
            mime="application/json"
        )

else:
    if is_existing_project:
        st.info("👆 Select an organization from the demo data to analyze.")
    else:
        st.info("👆 Upload at least one CSV file to get started.")
