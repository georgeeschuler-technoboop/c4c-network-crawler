"""
C4C Network Intelligence — CA Charity Ingestion

Dual-mode Streamlit app:
- GLFN Demo: Load pre-built canonical graph from repo
- New Project: Upload charitydata.ca exports and generate canonical outputs

Outputs conform to C4C Network Schema v1 (MVP):
- nodes.csv: ORG and PERSON nodes
- edges.csv: GRANT and BOARD_MEMBERSHIP edges
"""

import streamlit as st
import pandas as pd
import json
import re
import hashlib
import zipfile
from pathlib import Path
from io import BytesIO, StringIO

# =============================================================================
# Config
# =============================================================================

C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_9e232fe9e6914305a7ea8746e2e77125~mv2.png"
SOURCE_SYSTEM = "CHARITYDATA_CA"
JURISDICTION = "CA"
CURRENCY = "CAD"

# Demo data paths (relative to app location)
REPO_ROOT = Path(__file__).resolve().parent.parent
GLFN_DEMO_DIR = REPO_ROOT / "demo_data" / "glfn"

st.set_page_config(
    page_title="C4C CA Charity Ingestion",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)

# =============================================================================
# Canonical Schema Columns
# =============================================================================

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

# =============================================================================
# Parsing Functions
# =============================================================================

HEADER_RE = re.compile(r"^(.*)\s+\((\d{9}RR\d{4})\)\s*$")


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


def read_charitydata_csv_from_upload(uploaded_file) -> tuple:
    """Read a charitydata.ca CSV from Streamlit upload."""
    content = uploaded_file.getvalue().decode("utf-8")
    return _parse_charitydata_content(content)


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
    match = re.search(r'(\d{4})', rp)
    if match:
        return int(match.group(1))
    return None


def generate_edge_hash(s: str) -> str:
    """Generate short hash for edge ID."""
    return hashlib.md5(s.encode()).hexdigest()[:8]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_glfn_demo() -> tuple:
    """
    Load pre-built canonical graph from demo_data/glfn/.
    Returns: (nodes_df, edges_df, org_attributes)
    """
    nodes_path = GLFN_DEMO_DIR / "nodes.csv"
    edges_path = GLFN_DEMO_DIR / "edges.csv"
    org_attr_path = GLFN_DEMO_DIR / "org_attributes.json"
    
    if not nodes_path.exists() or not edges_path.exists():
        return None, None, None
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    org_attributes = {}
    if org_attr_path.exists():
        with open(org_attr_path, "r") as f:
            org_attributes = json.load(f)
    
    return nodes_df, edges_df, org_attributes


def process_uploaded_files(assets_file, directors_file, grants_file) -> tuple:
    """
    Process uploaded charitydata.ca files and return canonical outputs.
    Returns: (nodes_df, edges_df, org_attributes)
    """
    # Initialize
    assets_df = pd.DataFrame()
    directors_df = pd.DataFrame()
    grants_df = pd.DataFrame()
    org_name = ""
    cra_bn = ""
    
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
    
    if not org_name:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Processing variables
    org_slug = slugify_loose(org_name)
    latest_year, total_assets = extract_total_assets(assets_df) if not assets_df.empty else (None, None)
    fiscal_year = None
    foundation_node_id = f"org:{org_slug}"
    
    # Collections
    nodes = []
    edges = []
    seen_node_ids = set()
    
    # Create foundation ORG node
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
    
    # Process directors
    if not directors_df.empty:
        for _, r in directors_df.iterrows():
            last = clean_nan(r.get("Last Name", ""))
            first = clean_nan(r.get("First Name", ""))
            position = clean_nan(r.get("Position", ""))
            appointed = clean_nan(r.get("Appointed", ""))
            ceased = clean_nan(r.get("Ceased", ""))
            arms = clean_nan(r.get("At Arm's Length", ""))
            
            if not last and not first:
                continue
            
            person_key = f"{org_slug}:{last}|{first}|{appointed}"
            person_node_id = f"person:{person_key}"
            
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
    
    # Process grants
    if not grants_df.empty:
        # MVP DESIGN CHOICE (Canada parity):
        # charitydata.ca exports grants across many years. For sprint parity with US IRS 990 analysis
        # (which is typically based on the most recent filing), we filter grant rows to the most
        # recent fiscal year present in the dataset. We preserve the ability to analyze multi-year
        # trends later, but for now we standardize on "latest-year grants" for comparability.
        
        grants_filtered = grants_df.copy()
        
        if "Reporting Period" in grants_filtered.columns:
            periods = grants_filtered["Reporting Period"].dropna().unique()
            if len(periods) > 0:
                try:
                    latest_period = sorted(periods, reverse=True)[0]
                except Exception:
                    latest_period = periods[0]
                
                grants_filtered = grants_filtered[grants_filtered["Reporting Period"] == latest_period]
                fiscal_year = extract_fiscal_year(latest_period)
        
        for _, r in grants_filtered.iterrows():
            donee = clean_nan(r.get("Donee Name", ""))
            if not donee:
                continue
            
            city = clean_nan(r.get("City", ""))
            prov = clean_nan(r.get("Prov", ""))
            period = r.get("Reporting Period", "")
            amt = r.get("Reported Amount ($)", 0)
            gik = r.get("Gifts In Kind ($)", 0)
            
            try:
                amt_cash = float(amt) if pd.notna(amt) else 0
            except:
                amt_cash = 0
            try:
                amt_in_kind = float(gik) if pd.notna(gik) else 0
            except:
                amt_in_kind = 0
            amt_total = amt_cash + amt_in_kind
            
            donee_slug = f"donee-{slugify_loose(donee)}"
            if prov:
                donee_slug = f"{donee_slug}-{prov.lower()}"
            
            donee_node_id = f"org:{donee_slug}"
            
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
    
    # Build DataFrames
    nodes_df = pd.DataFrame(nodes).reindex(columns=NODE_COLUMNS) if nodes else pd.DataFrame()
    edges_df = pd.DataFrame(edges).reindex(columns=EDGE_COLUMNS) if edges else pd.DataFrame()
    
    # Build org_attributes
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
    
    return nodes_df, edges_df, org_attributes


# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, org_attributes: dict) -> None:
    """Render summary metrics and data preview for the loaded graph."""
    
    if nodes_df is None or nodes_df.empty:
        st.warning("No graph data loaded.")
        return
    
    # Org info header
    if org_attributes:
        st.subheader(f"📊 {org_attributes.get('org_display_name', 'Graph Summary')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Organization", org_attributes.get("org_legal_name", "—"))
        with col2:
            st.metric("Tax ID", org_attributes.get("tax_id", "—"))
        with col3:
            assets = org_attributes.get("assets_latest")
            year = org_attributes.get("assets_year")
            if assets and assets != "":
                st.metric(f"Total Assets ({year})", f"${float(assets):,.0f}")
            else:
                st.metric("Total Assets", "—")
    else:
        st.subheader("📊 Graph Summary")
    
    st.divider()
    
    # Node/Edge counts
    org_nodes = len(nodes_df[nodes_df["node_type"] == "ORG"]) if "node_type" in nodes_df.columns else 0
    person_nodes = len(nodes_df[nodes_df["node_type"] == "PERSON"]) if "node_type" in nodes_df.columns else 0
    grant_edges = len(edges_df[edges_df["edge_type"] == "GRANT"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    board_edges = len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ORG Nodes", org_nodes)
    with col2:
        st.metric("PERSON Nodes", person_nodes)
    with col3:
        st.metric("GRANT Edges", grant_edges)
    with col4:
        st.metric("BOARD Edges", board_edges)
    
    st.divider()
    
    # Data preview
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        display_cols = ["node_type", "label", "jurisdiction", "city", "region", "tax_id"]
        display_cols = [c for c in display_cols if c in nodes_df.columns]
        st.dataframe(nodes_df[display_cols], use_container_width=True, hide_index=True)
        st.caption(f"{len(nodes_df)} total nodes")
    
    with tab2:
        if not edges_df.empty:
            display_cols = ["edge_type", "from_id", "to_id", "amount", "role", "fiscal_year"]
            display_cols = [c for c in display_cols if c in edges_df.columns]
            st.dataframe(edges_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"{len(edges_df)} total edges")
        else:
            st.info("No edges found.")


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, org_attributes: dict) -> None:
    """Render download buttons for canonical outputs."""
    
    if nodes_df is None or nodes_df.empty:
        return
    
    st.divider()
    st.subheader("📥 Download Outputs")
    
    org_slug = org_attributes.get("org_slug", "export") if org_attributes else "export"
    
    def create_zip_download():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if not nodes_df.empty:
                zf.writestr("nodes.csv", nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr("edges.csv", edges_df.to_csv(index=False))
            if org_attributes:
                zf.writestr("org_attributes.json", json.dumps(org_attributes, indent=2))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    st.download_button(
        label="📦 Download All (ZIP)",
        data=create_zip_download(),
        file_name=f"c4c_{org_slug}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
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
        if org_attributes:
            st.download_button(
                label="📄 org_attributes.json",
                data=json.dumps(org_attributes, indent=2),
                file_name="org_attributes.json",
                mime="application/json"
            )


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("C4C Network Intelligence")
    
    st.markdown("""
    Parse nonprofit data and generate canonical network graphs for analysis.
    
    *Outputs conform to C4C Network Schema v1.*
    """)
    
    st.divider()
    
    # Project Mode Selector
    project_mode = st.selectbox(
        "Project",
        ["GLFN Demo (pre-loaded)", "New Project (upload)"],
        index=0,
        help="GLFN Demo loads pre-built data. New Project lets you upload files."
    )
    
    st.divider()
    
    # Initialize outputs
    nodes_df = pd.DataFrame()
    edges_df = pd.DataFrame()
    org_attributes = {}
    
    if project_mode == "GLFN Demo (pre-loaded)":
        # ---------------------------------------------------------------------
        # GLFN Demo Mode
        # ---------------------------------------------------------------------
        st.caption("📂 Loading canonical graph from `demo_data/glfn/`...")
        
        nodes_df, edges_df, org_attributes = load_glfn_demo()
        
        if nodes_df is None:
            st.error(f"""
            **Demo data not found.**
            
            Expected files at:
            ```
            {GLFN_DEMO_DIR}/nodes.csv
            {GLFN_DEMO_DIR}/edges.csv
            ```
            
            Please ensure demo data is committed to the repo, or switch to **New Project** mode.
            """)
            st.stop()
        
        st.success(f"✅ Loaded GLFN demo data: {len(nodes_df)} nodes, {len(edges_df)} edges")
        
    else:
        # ---------------------------------------------------------------------
        # New Project (Upload) Mode
        # ---------------------------------------------------------------------
        st.subheader("📤 Upload charitydata.ca Files")
        
        st.markdown("""
        Upload CSV files exported from **charitydata.ca** for one organization.
        Grants are automatically filtered to the most recent reporting period.
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
                help="Grants to qualified donees"
            )
        
        if not (assets_file or directors_file or grants_file):
            st.info("👆 Upload at least one CSV file to generate the graph.")
            st.stop()
        
        # Process uploads
        with st.spinner("Processing uploads..."):
            nodes_df, edges_df, org_attributes = process_uploaded_files(
                assets_file, directors_file, grants_file
            )
        
        if nodes_df.empty:
            st.warning("Could not extract any data from uploaded files.")
            st.stop()
        
        st.success(f"✅ Processed: {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    # ---------------------------------------------------------------------
    # Common Output (both modes)
    # ---------------------------------------------------------------------
    render_graph_summary(nodes_df, edges_df, org_attributes)
    render_downloads(nodes_df, edges_df, org_attributes)


if __name__ == "__main__":
    main()
