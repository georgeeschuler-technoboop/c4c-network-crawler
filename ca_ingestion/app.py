"""
OrgGraph (CA) — Canadian Nonprofit Registry Ingestion

Multi-project Streamlit app:
- New Project: Create a new project and upload initial data
- Add to Existing: Select existing project and merge new data
- View Demo: Read-only view of sample demo data

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

APP_VERSION = "0.4.0"
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_71fc58eb3cdf401cb972889063c2c132~mv2.png"
SOURCE_SYSTEM = "CHARITYDATA_CA"
JURISDICTION = "CA"
CURRENCY = "CAD"

# Demo data paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
DEMO_PROJECT_NAME = "_demo"  # Reserved name for demo dataset

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
# Project Management Functions
# =============================================================================

def get_projects() -> list:
    """Get list of existing projects from demo_data folder."""
    if not DEMO_DATA_DIR.exists():
        return []
    
    projects = []
    for item in DEMO_DATA_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            has_data = (item / "nodes.csv").exists() or (item / "edges.csv").exists()
            projects.append({
                "name": item.name,
                "path": item,
                "has_data": has_data,
                "is_demo": item.name == DEMO_PROJECT_NAME
            })
    
    projects.sort(key=lambda x: (not x["is_demo"], x["name"].lower()))
    return projects


def get_project_display_name(project_name: str) -> str:
    """Convert folder name to display name."""
    if project_name == DEMO_PROJECT_NAME:
        return "Demo Dataset"
    return project_name.replace("_", " ").replace("-", " ").title()


def get_folder_name(display_name: str) -> str:
    """Convert display name to folder name."""
    folder = display_name.lower().strip()
    folder = re.sub(r'[^a-z0-9\s]', '', folder)
    folder = re.sub(r'\s+', '_', folder)
    return folder


def create_project(project_name: str) -> tuple:
    """Create a new project folder. Returns (success, message)."""
    folder_name = get_folder_name(project_name)
    
    if not folder_name:
        return False, "Invalid project name"
    
    if folder_name == DEMO_PROJECT_NAME:
        return False, f"'{DEMO_PROJECT_NAME}' is reserved for demo data"
    
    project_path = DEMO_DATA_DIR / folder_name
    
    if project_path.exists():
        return False, f"Project '{project_name}' already exists"
    
    try:
        project_path.mkdir(parents=True, exist_ok=True)
        return True, f"Created project: {project_name}"
    except Exception as e:
        return False, f"Failed to create project: {str(e)}"


def load_project_data(project_name: str) -> tuple:
    """Load existing data from a project folder."""
    project_path = DEMO_DATA_DIR / project_name
    nodes_path = project_path / "nodes.csv"
    edges_path = project_path / "edges.csv"
    
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
    """Get list of existing foundations from nodes."""
    if nodes_df.empty or "node_type" not in nodes_df.columns:
        return []
    
    orgs = nodes_df[nodes_df["node_type"] == "ORG"]
    if orgs.empty:
        return []
    
    foundations = []
    for _, row in orgs.iterrows():
        label = row.get("label", "Unknown")
        source = row.get("source_system", "")
        foundations.append((label, source))
    
    return foundations


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
    match = re.search(r"(\d{4})", str(reporting_period))
    if match:
        return int(match.group(1))
    return None


def deterministic_person_id(first: str, last: str, org_slug: str) -> str:
    """Generate deterministic person node ID."""
    raw = f"{first}|{last}|{org_slug}".lower()
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"person-{h}"


def deterministic_grant_edge_id(from_id: str, to_id: str, amount: float, fiscal_year: int) -> str:
    """Generate deterministic grant edge ID."""
    raw = f"{from_id}|{to_id}|{amount}|{fiscal_year}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"grant-{h}"


def deterministic_board_edge_id(from_id: str, to_id: str, role: str) -> str:
    """Generate deterministic board edge ID."""
    raw = f"{from_id}|{to_id}|{role}".lower()
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"board-{h}"


# =============================================================================
# Data Processing Functions
# =============================================================================

def process_directors_file(directors_df: pd.DataFrame, org_slug: str, cra_bn: str) -> tuple:
    """Process directors/trustees CSV into canonical format."""
    nodes = []
    edges = []
    
    year_col = latest_year_column(directors_df)
    
    for _, row in directors_df.iterrows():
        name_field = row.get("Director / Trustee") or row.get("Name") or ""
        if not name_field or pd.isna(name_field):
            continue
        
        if year_col and year_col in row.index:
            active_val = row.get(year_col)
            if pd.isna(active_val) or str(active_val).strip() == "":
                continue
        
        # Parse name
        name_str = str(name_field).strip()
        parts = name_str.split(",", 1)
        if len(parts) == 2:
            last_name = parts[0].strip()
            first_name = parts[1].strip()
        else:
            name_parts = name_str.split()
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        
        # Generate IDs
        person_id = deterministic_person_id(first_name, last_name, org_slug)
        org_id = f"org-{cra_bn}" if cra_bn else f"org-{org_slug}"
        
        # Create person node
        person_node = {col: "" for col in NODE_COLUMNS}
        person_node.update({
            "node_id": person_id,
            "node_type": "PERSON",
            "label": f"{first_name} {last_name}".strip(),
            "first_name": first_name,
            "last_name": last_name,
            "jurisdiction": JURISDICTION,
            "source_system": SOURCE_SYSTEM,
        })
        nodes.append(person_node)
        
        # Create board edge
        role = "Director/Trustee"
        edge_id = deterministic_board_edge_id(person_id, org_id, role)
        
        board_edge = {col: "" for col in EDGE_COLUMNS}
        board_edge.update({
            "edge_id": edge_id,
            "from_id": person_id,
            "to_id": org_id,
            "edge_type": "BOARD_MEMBERSHIP",
            "role": role,
            "source_system": SOURCE_SYSTEM,
        })
        edges.append(board_edge)
    
    return nodes, edges


def process_grants_file(grants_df: pd.DataFrame, org_slug: str, cra_bn: str) -> tuple:
    """Process grants CSV into canonical format."""
    nodes = []
    edges = []
    
    org_id = f"org-{cra_bn}" if cra_bn else f"org-{org_slug}"
    
    for _, row in grants_df.iterrows():
        grantee_name = row.get("Qualified donee") or row.get("Donee") or ""
        if not grantee_name or pd.isna(grantee_name):
            continue
        
        # Extract values
        cash = row.get("Cash ($)", 0)
        in_kind = row.get("In-kind ($)", 0)
        
        try:
            cash = float(cash) if pd.notna(cash) else 0
        except:
            cash = 0
        
        try:
            in_kind = float(in_kind) if pd.notna(in_kind) else 0
        except:
            in_kind = 0
        
        total_amount = cash + in_kind
        
        # Get reporting period
        reporting_period = clean_nan(row.get("Reporting period", ""))
        fiscal_year = extract_fiscal_year(reporting_period)
        
        # Get location
        city = clean_nan(row.get("City", ""))
        province = clean_nan(row.get("Prov", ""))
        
        # Create grantee org node
        grantee_slug = slugify_loose(str(grantee_name))
        grantee_id = f"org-{grantee_slug}"
        
        grantee_node = {col: "" for col in NODE_COLUMNS}
        grantee_node.update({
            "node_id": grantee_id,
            "node_type": "ORG",
            "label": str(grantee_name).strip(),
            "org_slug": grantee_slug,
            "city": city,
            "region": province,
            "jurisdiction": JURISDICTION,
            "source_system": SOURCE_SYSTEM,
        })
        nodes.append(grantee_node)
        
        # Create grant edge
        edge_id = deterministic_grant_edge_id(org_id, grantee_id, total_amount, fiscal_year or 0)
        
        grant_edge = {col: "" for col in EDGE_COLUMNS}
        grant_edge.update({
            "edge_id": edge_id,
            "from_id": org_id,
            "to_id": grantee_id,
            "edge_type": "GRANT",
            "amount": total_amount,
            "amount_cash": cash,
            "amount_in_kind": in_kind,
            "currency": CURRENCY,
            "fiscal_year": fiscal_year or "",
            "reporting_period": reporting_period,
            "city": city,
            "region": province,
            "source_system": SOURCE_SYSTEM,
        })
        edges.append(grant_edge)
    
    return nodes, edges


def process_uploaded_files(assets_file, directors_file, grants_file) -> tuple:
    """Process uploaded charitydata.ca files."""
    
    all_nodes = []
    all_edges = []
    org_name = ""
    cra_bn = ""
    total_assets = None
    latest_year = None
    
    # Parse assets file first (contains org metadata)
    if assets_file:
        assets_df, org_name, cra_bn = read_charitydata_csv_from_upload(assets_file)
        latest_year, total_assets = extract_total_assets(assets_df)
    
    # Try to get org info from other files if not in assets
    if not org_name and directors_file:
        _, org_name, cra_bn = read_charitydata_csv_from_upload(directors_file)
        directors_file.seek(0)  # Reset for later use
    
    if not org_name and grants_file:
        _, org_name, cra_bn = read_charitydata_csv_from_upload(grants_file)
        grants_file.seek(0)  # Reset for later use
    
    if not org_name:
        org_name = "Unknown Organization"
    
    org_slug = slugify_loose(org_name)
    org_id = f"org-{cra_bn}" if cra_bn else f"org-{org_slug}"
    
    # Create org node
    org_node = {col: "" for col in NODE_COLUMNS}
    org_node.update({
        "node_id": org_id,
        "node_type": "ORG",
        "label": org_name,
        "org_slug": org_slug,
        "jurisdiction": JURISDICTION,
        "tax_id": cra_bn,
        "source_system": SOURCE_SYSTEM,
        "assets_latest": total_assets if total_assets is not None else "",
        "assets_year": latest_year if latest_year is not None else "",
    })
    all_nodes.append(org_node)
    
    # Process directors
    if directors_file:
        directors_df, _, _ = read_charitydata_csv_from_upload(directors_file)
        if not directors_df.empty:
            person_nodes, board_edges = process_directors_file(directors_df, org_slug, cra_bn)
            all_nodes.extend(person_nodes)
            all_edges.extend(board_edges)
    
    # Process grants
    if grants_file:
        grants_df, _, _ = read_charitydata_csv_from_upload(grants_file)
        if not grants_df.empty:
            grantee_nodes, grant_edges = process_grants_file(grants_df, org_slug, cra_bn)
            all_nodes.extend(grantee_nodes)
            all_edges.extend(grant_edges)
    
    # Create DataFrames
    nodes_df = pd.DataFrame(all_nodes, columns=NODE_COLUMNS)
    edges_df = pd.DataFrame(all_edges, columns=EDGE_COLUMNS)
    
    # Deduplicate nodes by node_id
    if not nodes_df.empty:
        nodes_df = nodes_df.drop_duplicates(subset=["node_id"], keep="first")
    
    org_attributes = {
        "org_id": org_id,
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


def merge_graph_data(existing_nodes: pd.DataFrame, existing_edges: pd.DataFrame,
                     new_nodes: pd.DataFrame, new_edges: pd.DataFrame) -> tuple:
    """Merge new graph data with existing, deduplicating by ID."""
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
        st.metric("🏛️ Organizations", org_nodes)
    with col2:
        st.metric("👤 People", person_nodes)
    with col3:
        st.metric("💰 Grant Edges", grant_edges)
    with col4:
        st.metric("🪪 Board Edges", board_edges)
    
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


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, project_name: str = None) -> None:
    """Render download buttons."""
    
    if nodes_df is None or nodes_df.empty:
        return
    
    st.divider()
    st.subheader("💾 Download")
    
    if project_name and project_name != DEMO_PROJECT_NAME:
        st.info(f"⬇️ **Download these files and upload to `demo_data/{project_name}/` on GitHub** (replace existing files)")
    
    def create_zip_download():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if not nodes_df.empty:
                zf.writestr("nodes.csv", nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr("edges.csv", edges_df.to_csv(index=False))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    file_prefix = project_name if project_name else "orggraph_ca"
    
    st.download_button(
        label="📦 Download All (ZIP)",
        data=create_zip_download(),
        file_name=f"{file_prefix}_export.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("**Or download individually:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not nodes_df.empty:
            st.download_button(
                label="📥 nodes.csv",
                data=nodes_df.to_csv(index=False),
                file_name="nodes.csv",
                mime="text/csv"
            )
    
    with col2:
        if not edges_df.empty:
            st.download_button(
                label="📥 edges.csv",
                data=edges_df.to_csv(index=False),
                file_name="edges.csv",
                mime="text/csv"
            )


# =============================================================================
# Upload Interface
# =============================================================================

def render_upload_interface(project_name: str):
    """Render the upload and processing interface for a project."""
    display_name = get_project_display_name(project_name)
    
    # Load existing data
    existing_nodes, existing_edges = load_project_data(project_name)
    
    # Show existing data status
    if not existing_nodes.empty or not existing_edges.empty:
        st.success(f"📂 **Existing {display_name} data:** {len(existing_nodes)} nodes, {len(existing_edges)} edges")
        
        existing_foundations = get_existing_foundations(existing_nodes)
        if existing_foundations:
            with st.expander(f"📋 Organizations already in {display_name} ({len(existing_foundations)})", expanded=False):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        st.caption("New data will be merged. Duplicates automatically skipped.")
    else:
        st.info(f"📂 **No existing {display_name} data.** This will be the first organization.")
    
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
    st.success(f"📊 **Combined {display_name} dataset:** {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    render_graph_summary(nodes_df, edges_df)
    render_downloads(nodes_df, edges_df, project_name)


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
    
    # ==========================================================================
    # Project Mode Selection
    # ==========================================================================
    
    st.subheader("📁 Project")
    
    projects = get_projects()
    existing_project_names = [p["name"] for p in projects if not p["is_demo"]]
    has_demo = any(p["is_demo"] for p in projects)
    
    # Mode selection
    mode_options = ["➕ New Project"]
    if existing_project_names:
        mode_options.append("📂 Add to Existing Project")
    if has_demo:
        mode_options.append("👁️ View Demo")
    
    project_mode = st.radio(
        "What would you like to do?",
        mode_options,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # ==========================================================================
    # NEW PROJECT MODE
    # ==========================================================================
    if project_mode == "➕ New Project":
        st.markdown("### Create New Project")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_project_name = st.text_input(
                "Project Name",
                placeholder="e.g., BC Foundations Network",
                help="Choose a descriptive name for your project"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            create_btn = st.button("Create Project", type="primary", disabled=not new_project_name)
        
        if new_project_name:
            folder_name = get_folder_name(new_project_name)
            st.caption(f"📁 Will create folder: `demo_data/{folder_name}/`")
        
        if create_btn and new_project_name:
            success, message = create_project(new_project_name)
            if success:
                st.success(f"✅ {message}")
                st.session_state.current_project = get_folder_name(new_project_name)
                st.rerun()
            else:
                st.error(f"❌ {message}")
        
        # If project was just created, show upload interface
        if "current_project" in st.session_state and st.session_state.current_project:
            project_name = st.session_state.current_project
            st.divider()
            render_upload_interface(project_name)
    
    # ==========================================================================
    # ADD TO EXISTING PROJECT MODE
    # ==========================================================================
    elif project_mode == "📂 Add to Existing Project":
        st.markdown("### Select Project")
        
        # Build dropdown options with node/edge counts
        project_options = []
        for p in projects:
            if not p["is_demo"]:
                display_name = get_project_display_name(p["name"])
                if p["has_data"]:
                    nodes_df, edges_df = load_project_data(p["name"])
                    display_name += f" ({len(nodes_df)} nodes, {len(edges_df)} edges)"
                else:
                    display_name += " (empty)"
                project_options.append((p["name"], display_name))
        
        if not project_options:
            st.info("No existing projects found. Create a new project first.")
            st.stop()
        
        selected_display = st.selectbox(
            "Select project to add data to:",
            [display for _, display in project_options],
            label_visibility="collapsed"
        )
        
        # Find selected project name
        selected_project = None
        for name, display in project_options:
            if display == selected_display:
                selected_project = name
                break
        
        if selected_project:
            st.divider()
            render_upload_interface(selected_project)
    
    # ==========================================================================
    # VIEW DEMO MODE
    # ==========================================================================
    elif project_mode == "👁️ View Demo":
        st.markdown("### Demo Dataset")
        st.caption(f"📂 Loading from `demo_data/{DEMO_PROJECT_NAME}/`...")
        
        nodes_df, edges_df = load_project_data(DEMO_PROJECT_NAME)
        
        if nodes_df.empty and edges_df.empty:
            st.warning("""
            **No demo data found.**
            
            The demo dataset hasn't been set up yet. Create a new project to get started.
            """)
            st.stop()
        
        st.success(f"✅ Demo data: {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        # Show existing foundations
        existing_foundations = get_existing_foundations(nodes_df)
        if existing_foundations:
            with st.expander(f"📋 Organizations in Demo ({len(existing_foundations)})", expanded=True):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        render_graph_summary(nodes_df, edges_df)
        render_downloads(nodes_df, edges_df, DEMO_PROJECT_NAME)


if __name__ == "__main__":
    main()
