"""
OrgGraph (CA) — Canadian Nonprofit Registry Ingestion

Dual-mode Streamlit app:
- GLFN Demo: Load pre-built canonical graph from repo
- Add to GLFN: Upload charitydata.ca exports, auto-merge with existing GLFN data

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

APP_VERSION = "0.3.0"  # Track app version
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_71fc58eb3cdf401cb972889063c2c132~mv2.png"
SOURCE_SYSTEM = "CHARITYDATA_CA"
JURISDICTION = "CA"
CURRENCY = "CAD"

# Demo data paths (relative to app location)
REPO_ROOT = Path(__file__).resolve().parent.parent
GLFN_DEMO_DIR = REPO_ROOT / "demo_data" / "glfn"

st.set_page_config(
    page_title="OrgGraph (CA)",
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
    """Parse charitydata.ca CSV content."""
    lines = content.split("\n")
    
    if not lines:
        return pd.DataFrame(), "", ""
    
    header_line = lines[0].strip().lstrip("\ufeff").strip().strip('"').rstrip(",")
    m = HEADER_RE.match(header_line)
    
    if m:
        org_name = m.group(1).strip()
        cra_bn = m.group(2)
    else:
        org_name = header_line
        cra_bn = ""
    
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
    """Extract year from reporting period."""
    if not reporting_period:
        return None
    match = re.search(r'(\d{4})', str(reporting_period).strip())
    if match:
        return int(match.group(1))
    return None


def generate_edge_hash(s: str) -> str:
    """Generate short hash for edge ID."""
    return hashlib.md5(s.encode()).hexdigest()[:8]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_glfn_data() -> tuple:
    """
    Load existing GLFN data from demo_data/glfn/.
    Returns: (nodes_df, edges_df) - empty DataFrames if files don't exist
    """
    nodes_path = GLFN_DEMO_DIR / "nodes.csv"
    edges_path = GLFN_DEMO_DIR / "edges.csv"
    
    nodes_df = pd.DataFrame(columns=NODE_COLUMNS)
    edges_df = pd.DataFrame(columns=EDGE_COLUMNS)
    
    if nodes_path.exists():
        try:
            df = pd.read_csv(nodes_path)
            if not df.empty and len(df) > 0:
                nodes_df = df
        except:
            pass
    
    if edges_path.exists():
        try:
            df = pd.read_csv(edges_path)
            if not df.empty and len(df) > 0:
                edges_df = df
        except:
            pass
    
    return nodes_df, edges_df


def get_existing_foundations(nodes_df: pd.DataFrame) -> list:
    """
    Extract list of foundation names already in GLFN data.
    Foundations are ORG nodes that have a tax_id (CRA BN).
    Returns: list of (label, source_system) tuples
    """
    if nodes_df.empty or "node_type" not in nodes_df.columns:
        return []
    
    # Filter to ORG nodes with tax_id (these are foundations, not donees)
    orgs = nodes_df[nodes_df["node_type"] == "ORG"].copy()
    
    if "tax_id" not in orgs.columns:
        return []
    
    # Foundations have tax_ids, donees don't
    foundations = orgs[orgs["tax_id"].notna() & (orgs["tax_id"] != "")]
    
    if foundations.empty:
        return []
    
    result = []
    for _, row in foundations.iterrows():
        label = row.get("label", "Unknown")
        source = row.get("source_system", "")
        result.append((label, source))
    
    return result


def merge_graph_data(existing_nodes: pd.DataFrame, existing_edges: pd.DataFrame,
                     new_nodes: pd.DataFrame, new_edges: pd.DataFrame) -> tuple:
    """
    Merge new graph data with existing, deduplicating by ID.
    Returns: (merged_nodes, merged_edges, stats)
    """
    stats = {
        "existing_nodes": len(existing_nodes),
        "existing_edges": len(existing_edges),
        "new_nodes_total": len(new_nodes),
        "new_edges_total": len(new_edges),
        "nodes_added": 0,
        "edges_added": 0,
        "nodes_skipped": 0,
        "edges_skipped": 0,
    }
    
    # Merge nodes
    if existing_nodes.empty or "node_id" not in existing_nodes.columns:
        merged_nodes = new_nodes.copy()
        stats["nodes_added"] = len(new_nodes)
    elif new_nodes.empty:
        merged_nodes = existing_nodes.copy()
    else:
        existing_ids = set(existing_nodes["node_id"].dropna().astype(str))
        new_mask = ~new_nodes["node_id"].astype(str).isin(existing_ids)
        nodes_to_add = new_nodes[new_mask]
        
        stats["nodes_added"] = len(nodes_to_add)
        stats["nodes_skipped"] = len(new_nodes) - len(nodes_to_add)
        
        merged_nodes = pd.concat([existing_nodes, nodes_to_add], ignore_index=True)
    
    # Merge edges
    if existing_edges.empty or "edge_id" not in existing_edges.columns:
        merged_edges = new_edges.copy()
        stats["edges_added"] = len(new_edges)
    elif new_edges.empty:
        merged_edges = existing_edges.copy()
    else:
        existing_ids = set(existing_edges["edge_id"].dropna().astype(str))
        new_mask = ~new_edges["edge_id"].astype(str).isin(existing_ids)
        edges_to_add = new_edges[new_mask]
        
        stats["edges_added"] = len(edges_to_add)
        stats["edges_skipped"] = len(new_edges) - len(edges_to_add)
        
        merged_edges = pd.concat([existing_edges, edges_to_add], ignore_index=True)
    
    return merged_nodes, merged_edges, stats


def process_uploaded_files(assets_file, directors_file, grants_file) -> tuple:
    """
    Process uploaded charitydata.ca files and return canonical outputs.
    Returns: (nodes_df, edges_df, org_attributes)
    """
    assets_df = pd.DataFrame()
    directors_df = pd.DataFrame()
    grants_df = pd.DataFrame()
    org_name = ""
    cra_bn = ""
    
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
    
    org_slug = slugify_loose(org_name)
    latest_year, total_assets = extract_total_assets(assets_df) if not assets_df.empty else (None, None)
    fiscal_year = None
    foundation_node_id = f"org:{org_slug}"
    
    nodes = []
    edges = []
    seen_node_ids = set()
    
    # Foundation node
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
    
    # Directors
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
    
    # Grants
    if not grants_df.empty:
        grants_filtered = grants_df.copy()
        
        if "Reporting Period" in grants_filtered.columns:
            periods = grants_filtered["Reporting Period"].dropna().unique()
            if len(periods) > 0:
                try:
                    latest_period = sorted(periods, reverse=True)[0]
                except:
                    latest_period = periods[0]
                
                grants_filtered = grants_filtered[grants_filtered["Reporting Period"] == latest_period]
                fiscal_year = extract_fiscal_year(latest_period)
        
        # Track edge base IDs to add sequence numbers for duplicates
        edge_base_counts = {}
        
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
            
            # Build edge ID with sequence number to preserve multiple identical grants
            fy = extract_fiscal_year(period) or fiscal_year
            if amt_total > 0:
                edge_base = f"gr:{foundation_node_id}->{donee_node_id}:{fy}:{int(amt_total)}"
            else:
                hash_input = f"{foundation_node_id}{donee_node_id}{fy}{period}"
                edge_base = f"gr:{foundation_node_id}->{donee_node_id}:{fy}:h{generate_edge_hash(hash_input)}"
            
            # Add sequence number
            if edge_base not in edge_base_counts:
                edge_base_counts[edge_base] = 0
            edge_base_counts[edge_base] += 1
            seq = edge_base_counts[edge_base]
            edge_id = f"{edge_base}:{seq}"
            
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
    
    nodes_df = pd.DataFrame(nodes).reindex(columns=NODE_COLUMNS) if nodes else pd.DataFrame()
    edges_df = pd.DataFrame(edges).reindex(columns=EDGE_COLUMNS) if edges else pd.DataFrame()
    
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

def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Render summary metrics and data preview."""
    
    if nodes_df is None or nodes_df.empty:
        st.warning("No graph data loaded.")
        return
    
    st.subheader("📊 Graph Summary")
    
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
    
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        display_cols = ["node_type", "label", "jurisdiction", "city", "region", "tax_id", "source_system"]
        display_cols = [c for c in display_cols if c in nodes_df.columns]
        st.dataframe(nodes_df[display_cols], use_container_width=True, hide_index=True)
        st.caption(f"{len(nodes_df)} total nodes")
    
    with tab2:
        if not edges_df.empty:
            display_cols = ["edge_type", "from_id", "to_id", "amount", "role", "fiscal_year", "source_system"]
            display_cols = [c for c in display_cols if c in edges_df.columns]
            st.dataframe(edges_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"{len(edges_df)} total edges")
        else:
            st.info("No edges found.")


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Render download buttons."""
    
    if nodes_df is None or nodes_df.empty:
        return
    
    st.divider()
    st.subheader("📥 Download & Upload to GitHub")
    
    st.info("⬇️ **Download these files and upload to `demo_data/glfn/` on GitHub** (replace existing files)")
    
    def create_zip_download():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if not nodes_df.empty:
                zf.writestr("nodes.csv", nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr("edges.csv", edges_df.to_csv(index=False))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    st.download_button(
        label="📦 Download All (ZIP)",
        data=create_zip_download(),
        file_name="c4c_glfn_update.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("**Or download individually:**")
    
    col1, col2 = st.columns(2)
    
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


# =============================================================================
# Main App
# =============================================================================

def main():
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("OrgGraph (CA)")
    
    st.markdown("""
    OrgGraph currently supports US and Canadian nonprofit registries; additional sources will be added in the future.
    """)
    st.caption(f"v{APP_VERSION}")
    
    st.divider()
    
    project_mode = st.selectbox(
        "Mode",
        ["GLFN Demo (view existing)", "Add to GLFN (upload + merge)"],
        index=0,
        help="View existing GLFN data or add a new organization"
    )
    
    st.divider()
    
    nodes_df = pd.DataFrame()
    edges_df = pd.DataFrame()
    
    if project_mode == "GLFN Demo (view existing)":
        # ---------------------------------------------------------------------
        # View Mode
        # ---------------------------------------------------------------------
        st.caption("📂 Loading from `demo_data/glfn/`...")
        
        nodes_df, edges_df = load_glfn_data()
        
        if nodes_df.empty and edges_df.empty:
            st.warning("""
            **No GLFN data found yet.**
            
            Switch to **"Add to GLFN"** mode to start building the dataset.
            """)
            st.stop()
        
        st.success(f"✅ GLFN data: {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        # Show existing foundations
        existing_foundations = get_existing_foundations(nodes_df)
        if existing_foundations:
            with st.expander(f"📋 Foundations in GLFN ({len(existing_foundations)})", expanded=True):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
    else:
        # ---------------------------------------------------------------------
        # Add to GLFN Mode
        # ---------------------------------------------------------------------
        existing_nodes, existing_edges = load_glfn_data()
        
        if not existing_nodes.empty or not existing_edges.empty:
            st.success(f"📂 **Existing GLFN data:** {len(existing_nodes)} nodes, {len(existing_edges)} edges")
            
            # Show existing foundations
            existing_foundations = get_existing_foundations(existing_nodes)
            if existing_foundations:
                with st.expander(f"📋 Foundations already in GLFN ({len(existing_foundations)})", expanded=False):
                    for label, source in existing_foundations:
                        flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                        st.write(f"{flag} {label}")
            
            st.caption("New data will be merged. Duplicates automatically skipped.")
        else:
            st.info("📂 **No existing GLFN data.** This will be the first organization.")
        
        st.divider()
        st.subheader("📤 Upload charitydata.ca Files")
        
        st.markdown("Upload CSV files for **one organization**:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            assets_file = st.file_uploader("assets.csv", type=["csv"])
        with col2:
            directors_file = st.file_uploader("directors-trustees.csv", type=["csv"])
        with col3:
            grants_file = st.file_uploader("grants.csv", type=["csv"])
        
        if not (assets_file or directors_file or grants_file):
            st.info("👆 Upload at least one CSV file")
            st.stop()
        
        with st.spinner("Processing..."):
            new_nodes, new_edges, org_attributes = process_uploaded_files(
                assets_file, directors_file, grants_file
            )
        
        if new_nodes.empty:
            st.warning("Could not extract data from uploaded files.")
            st.stop()
        
        org_name = org_attributes.get("org_legal_name", "Unknown")
        st.success(f"✅ Processed **{org_name}**: {len(new_nodes)} nodes, {len(new_edges)} edges")
        
        st.divider()
        
        # Merge
        nodes_df, edges_df, stats = merge_graph_data(
            existing_nodes, existing_edges, new_nodes, new_edges
        )
        
        st.subheader(f"🔀 Merge Results — {org_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Nodes:**")
            st.write(f"- Existing: {stats['existing_nodes']}")
            st.write(f"- From this org: {stats['new_nodes_total']}")
            st.write(f"- ✅ **Added: {stats['nodes_added']}**")
            if stats['nodes_skipped'] > 0:
                st.write(f"- ⏭️ Skipped (duplicates): {stats['nodes_skipped']}")
        
        with col2:
            st.markdown("**Edges:**")
            st.write(f"- Existing: {stats['existing_edges']}")
            st.write(f"- From this org: {stats['new_edges_total']}")
            st.write(f"- ✅ **Added: {stats['edges_added']}**")
            if stats['edges_skipped'] > 0:
                st.write(f"- ⏭️ Skipped (duplicates): {stats['edges_skipped']}")
        
        st.divider()
        st.success(f"📊 **Combined GLFN dataset:** {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    # ---------------------------------------------------------------------
    # Common Output
    # ---------------------------------------------------------------------
    render_graph_summary(nodes_df, edges_df)
    render_downloads(nodes_df, edges_df)


if __name__ == "__main__":
    main()
