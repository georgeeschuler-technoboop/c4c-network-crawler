"""
OrgGraph (US) — US Nonprofit Registry Ingestion

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
from io import BytesIO
import zipfile
import sys
import os
import re
from pathlib import Path
from datetime import datetime

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4c_utils.irs990_parser import parse_990_pdf
from c4c_utils.network_export import build_nodes_df, build_edges_df, NODE_COLUMNS, EDGE_COLUMNS, get_existing_foundations


# =============================================================================
# Constants
# =============================================================================

APP_VERSION = "0.4.0"
MAX_FILES = 50
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_25063966d6cd496eb2fe3f6ee5cde0fa~mv2.png"
SOURCE_SYSTEM = "IRS_990"
JURISDICTION = "US"

# Demo data paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
DEMO_PROJECT_NAME = "_demo"  # Reserved name for demo dataset


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="OrgGraph (US)",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)


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
            # Check if it has nodes.csv or edges.csv (a real project)
            has_data = (item / "nodes.csv").exists() or (item / "edges.csv").exists()
            projects.append({
                "name": item.name,
                "path": item,
                "has_data": has_data,
                "is_demo": item.name == DEMO_PROJECT_NAME
            })
    
    # Sort: demo first (if exists), then alphabetically
    projects.sort(key=lambda x: (not x["is_demo"], x["name"].lower()))
    return projects


def get_project_display_name(project_name: str) -> str:
    """Convert folder name to display name."""
    if project_name == DEMO_PROJECT_NAME:
        return "Demo Dataset"
    # Convert snake_case or kebab-case to Title Case
    return project_name.replace("_", " ").replace("-", " ").title()


def get_folder_name(display_name: str) -> str:
    """Convert display name to folder name."""
    # Convert to lowercase, replace spaces with underscores
    folder = display_name.lower().strip()
    folder = re.sub(r'[^a-z0-9\s]', '', folder)  # Remove special chars
    folder = re.sub(r'\s+', '_', folder)  # Spaces to underscores
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


def get_project_path(project_name: str) -> Path:
    """Get the path for a project folder."""
    return DEMO_DATA_DIR / project_name


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables for persistent results."""
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "nodes_df" not in st.session_state:
        st.session_state.nodes_df = pd.DataFrame()
    if "edges_df" not in st.session_state:
        st.session_state.edges_df = pd.DataFrame()
    if "grants_df" not in st.session_state:
        st.session_state.grants_df = None
    if "parse_results" not in st.session_state:
        st.session_state.parse_results = []
    if "merge_stats" not in st.session_state:
        st.session_state.merge_stats = {}
    if "processed_orgs" not in st.session_state:
        st.session_state.processed_orgs = []
    if "current_project" not in st.session_state:
        st.session_state.current_project = None


def clear_session_state():
    """Clear all processing results from session state."""
    st.session_state.processed = False
    st.session_state.nodes_df = pd.DataFrame()
    st.session_state.edges_df = pd.DataFrame()
    st.session_state.grants_df = None
    st.session_state.parse_results = []
    st.session_state.merge_stats = {}
    st.session_state.processed_orgs = []


def store_results(nodes_df, edges_df, grants_df, parse_results, merge_stats, processed_orgs):
    """Store processing results in session state."""
    st.session_state.processed = True
    st.session_state.nodes_df = nodes_df
    st.session_state.edges_df = edges_df
    st.session_state.grants_df = grants_df
    st.session_state.parse_results = parse_results
    st.session_state.merge_stats = merge_stats
    st.session_state.processed_orgs = processed_orgs


# =============================================================================
# Data Merging Functions
# =============================================================================

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


def process_uploaded_files(uploaded_files, tax_year_override: str = "") -> tuple:
    """Process uploaded 990-PF files and return canonical outputs."""
    all_grants = []
    all_people = []
    foundations_meta = []
    parse_results = []
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            result = parse_990_pdf(file_bytes, uploaded_file.name, tax_year_override)
            
            diagnostics = result.get('diagnostics', {})
            org_name = diagnostics.get('org_name', '') or result['foundation_meta'].get('foundation_name', 'Unknown')
            grants_count = len(result['grants_df'])
            people_count = len(result['people_df'])
            total_amount = result['grants_df']['grant_amount'].sum() if grants_count > 0 else 0
            
            foundations_meta.append(result['foundation_meta'])
            
            if not result['grants_df'].empty:
                all_grants.append(result['grants_df'])
            
            if not result['people_df'].empty:
                all_people.append(result['people_df'])
            
            # Determine status
            if grants_count > 0:
                status = "success"
                message = f"{grants_count} grants (${total_amount:,.0f}), {people_count} board members"
            elif not diagnostics.get('is_990pf', True):
                status = "wrong_form"
                message = f"Form {diagnostics.get('form_type', 'unknown')} (public charity) - no itemized grants"
            else:
                status = "no_grants"
                message = f"No grants found. {people_count} board members extracted."
            
            parse_results.append({
                "file": uploaded_file.name,
                "status": status,
                "org_name": org_name,
                "message": message
            })
            
        except Exception as e:
            parse_results.append({
                "file": uploaded_file.name,
                "status": "error",
                "org_name": "",
                "message": str(e)
            })
    
    # Combine all results
    if all_grants:
        combined_grants = pd.concat(all_grants, ignore_index=True)
    else:
        combined_grants = pd.DataFrame()
    
    if all_people:
        combined_people = pd.concat(all_people, ignore_index=True)
    else:
        combined_people = pd.DataFrame()
    
    # Build canonical format
    nodes_df = build_nodes_df(combined_grants, combined_people, foundations_meta)
    edges_df = build_edges_df(combined_grants, combined_people, foundations_meta)
    
    return nodes_df, edges_df, combined_grants, foundations_meta, parse_results


# =============================================================================
# Rendering Functions
# =============================================================================

def render_parse_status(parse_results: list):
    """Render the parsing status for each file."""
    if not parse_results:
        return
    
    success_count = sum(1 for r in parse_results if r["status"] == "success")
    error_count = sum(1 for r in parse_results if r["status"] == "error")
    no_grants_count = sum(1 for r in parse_results if r["status"] == "no_grants")
    wrong_form_count = sum(1 for r in parse_results if r["status"] == "wrong_form")
    
    st.subheader("📋 Parsing Status")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Success", success_count)
    col2.metric("⚠️ No Grants", no_grants_count)
    col3.metric("📄 Wrong Form", wrong_form_count)
    col4.metric("❌ Errors", error_count)
    
    with st.expander("View Details", expanded=False):
        for result in parse_results:
            status = result["status"]
            file_name = result["file"]
            org_name = result.get("org_name", "")
            message = result.get("message", "")
            
            if status == "success":
                st.success(f"✅ **{org_name or file_name}**: {message}")
            elif status == "no_grants":
                st.warning(f"⚠️ **{org_name or file_name}**: {message}")
            elif status == "wrong_form":
                st.info(f"📄 **{file_name}**: {message}")
            else:
                st.error(f"❌ **{file_name}**: {message}")


def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, grants_df: pd.DataFrame = None):
    """Render summary metrics for the graph."""
    st.subheader("📊 Graph Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Count node types
    org_count = len(nodes_df[nodes_df["node_type"] == "ORG"]) if not nodes_df.empty and "node_type" in nodes_df.columns else 0
    person_count = len(nodes_df[nodes_df["node_type"] == "PERSON"]) if not nodes_df.empty and "node_type" in nodes_df.columns else 0
    
    # Count edge types
    grant_count = len(edges_df[edges_df["edge_type"] == "GRANT"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    board_count = len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    
    col1.metric("🏛️ Organizations", org_count)
    col2.metric("👤 People", person_count)
    col3.metric("💰 Grant Edges", grant_count)
    col4.metric("🪪 Board Edges", board_count)
    
    # Show total funding if available
    if grants_df is not None and not grants_df.empty and "grant_amount" in grants_df.columns:
        total_funding = grants_df["grant_amount"].sum()
        st.metric("💵 Total Grant Funding", f"${total_funding:,.0f}")


def render_analytics(grants_df: pd.DataFrame):
    """Render grant analytics."""
    if grants_df is None or grants_df.empty:
        st.info("No grant data available for analytics")
        return
    
    st.subheader("📈 Grant Analytics")
    
    # Top grantees by amount
    if "grantee_name" in grants_df.columns and "grant_amount" in grants_df.columns:
        grantee_totals = grants_df.groupby("grantee_name")["grant_amount"].sum().sort_values(ascending=False).head(10)
        
        if not grantee_totals.empty:
            st.markdown("**Top 10 Grantees by Total Funding:**")
            for i, (grantee, amount) in enumerate(grantee_totals.items(), 1):
                st.write(f"{i}. **{grantee}**: ${amount:,.0f}")
    
    # Multi-funder grantees
    if "grantee_name" in grants_df.columns and "foundation_name" in grants_df.columns:
        funder_counts = grants_df.groupby("grantee_name")["foundation_name"].nunique()
        multi_funded = funder_counts[funder_counts > 1].sort_values(ascending=False)
        
        if not multi_funded.empty:
            st.markdown("**Multi-Funder Grantees:**")
            for grantee, count in multi_funded.head(10).items():
                st.write(f"- **{grantee}**: {count} funders")


def render_data_preview(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Render data preview expanders."""
    with st.expander("👀 Preview Nodes", expanded=False):
        if not nodes_df.empty:
            st.dataframe(nodes_df, use_container_width=True)
        else:
            st.info("No nodes to display")
    
    with st.expander("👀 Preview Edges", expanded=False):
        if not edges_df.empty:
            st.dataframe(edges_df, use_container_width=True)
        else:
            st.info("No edges to display")


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                    grants_df: pd.DataFrame = None, parse_results: list = None,
                    project_name: str = None):
    """Render download buttons."""
    st.subheader("💾 Download")
    
    display_name = get_project_display_name(project_name) if project_name else "project"
    
    # Instruction for saving to project
    if project_name and project_name != DEMO_PROJECT_NAME:
        st.info(f"⬇️ **Download these files and upload to `demo_data/{project_name}/` on GitHub** (replace existing files)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not nodes_df.empty:
            nodes_csv = nodes_df.to_csv(index=False)
            st.download_button(
                "📥 Download nodes.csv",
                data=nodes_csv,
                file_name="nodes.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if not edges_df.empty:
            edges_csv = edges_df.to_csv(index=False)
            st.download_button(
                "📥 Download edges.csv",
                data=edges_csv,
                file_name="edges.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # ZIP download
    if not nodes_df.empty or not edges_df.empty:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if not nodes_df.empty:
                zip_file.writestr('nodes.csv', nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zip_file.writestr('edges.csv', edges_df.to_csv(index=False))
            if grants_df is not None and not grants_df.empty:
                zip_file.writestr('grants_detail.csv', grants_df.to_csv(index=False))
            if parse_results:
                zip_file.writestr('parse_log.json', json.dumps(parse_results, indent=2))
        
        zip_buffer.seek(0)
        
        file_prefix = project_name if project_name else "orggraph"
        st.download_button(
            "📦 Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{file_prefix}_export.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
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
            with st.expander(f"📋 Foundations already in {display_name} ({len(existing_foundations)})", expanded=False):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        st.caption("New data will be merged. Duplicates automatically skipped.")
    else:
        st.info(f"📂 **No existing {display_name} data.** This will be the first upload.")
    
    st.divider()
    
    # Check if we have results in session state
    if st.session_state.processed:
        # Show Clear Results button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Results", type="secondary"):
                clear_session_state()
                st.rerun()
        
        # Show stored results
        st.subheader("📤 Last Processing Results")
        
        if st.session_state.processed_orgs:
            orgs_label = ", ".join(st.session_state.processed_orgs[:3])
            if len(st.session_state.processed_orgs) > 3:
                orgs_label += f" + {len(st.session_state.processed_orgs) - 3} more"
            st.info(f"**Processed:** {orgs_label}")
        
        # Render processing log
        if st.session_state.parse_results:
            render_parse_status(st.session_state.parse_results)
        
        st.divider()
        
        # Merge stats
        if st.session_state.merge_stats:
            stats = st.session_state.merge_stats
            st.subheader("🔀 Merge Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Nodes:**")
                st.write(f"- Existing: {stats['existing_nodes']}")
                st.write(f"- From this upload: {stats['new_nodes_total']}")
                st.write(f"- ✅ **Added: {stats['nodes_added']}**")
                if stats['nodes_skipped'] > 0:
                    st.write(f"- ⏭️ Skipped (duplicates): {stats['nodes_skipped']}")
            
            with col2:
                st.markdown("**Edges:**")
                st.write(f"- Existing: {stats['existing_edges']}")
                st.write(f"- From this upload: {stats['new_edges_total']}")
                st.write(f"- ✅ **Added: {stats['edges_added']}**")
                if stats['edges_skipped'] > 0:
                    st.write(f"- ⏭️ Skipped (duplicates): {stats['edges_skipped']}")
        
        st.divider()
        st.success(f"📊 **Combined {display_name} dataset:** {len(st.session_state.nodes_df)} nodes, {len(st.session_state.edges_df)} edges")
        
        # Render outputs from session state
        render_graph_summary(st.session_state.nodes_df, st.session_state.edges_df, st.session_state.grants_df)
        
        show_analytics = st.checkbox("📊 Show Network Analytics", value=False)
        if show_analytics:
            render_analytics(st.session_state.grants_df)
        
        render_data_preview(st.session_state.nodes_df, st.session_state.edges_df)
        render_downloads(st.session_state.nodes_df, st.session_state.edges_df, 
                       st.session_state.grants_df, st.session_state.parse_results,
                       project_name)
        
    else:
        # Show upload interface
        st.subheader("📤 Upload 990-PF PDFs")
        
        st.markdown(f"""
        Upload up to **{MAX_FILES} private foundation 990-PF filings**.
        
        *Note: Works with **990-PF** (private foundations). Standard **990** filings 
        from public charities don't include itemized grants.*
        """)
        
        uploaded_files = st.file_uploader(
            f"Upload 990-PF PDF(s) (max {MAX_FILES})",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) > MAX_FILES:
            st.warning(f"⚠️ Max {MAX_FILES} files. Processing first {MAX_FILES}.")
            uploaded_files = uploaded_files[:MAX_FILES]
        
        tax_year_override = st.text_input(
            "Tax Year (optional)",
            help="Override auto-detection if needed"
        )
        
        parse_button = st.button("🔍 Parse 990 Filings", type="primary", disabled=not uploaded_files)
        
        if not uploaded_files:
            st.info("👆 Upload 990-PF PDF files")
            st.stop()
        
        if parse_button:
            # Process files
            with st.spinner("Parsing filings..."):
                new_nodes, new_edges, grants_df, foundations_meta, parse_results = process_uploaded_files(
                    uploaded_files, tax_year_override
                )
            
            if new_nodes.empty:
                st.warning("No data extracted from uploaded files.")
                # Still store results to show errors
                store_results(
                    pd.DataFrame(), pd.DataFrame(), None,
                    parse_results, {}, []
                )
                st.rerun()
            
            # Merge with existing
            nodes_df, edges_df, merge_stats = merge_graph_data(
                existing_nodes, existing_edges, new_nodes, new_edges
            )
            
            # Get processed org names
            processed_orgs = [r["org_name"] for r in parse_results if r.get("status") == "success" and r.get("org_name")]
            
            # Store in session state
            store_results(nodes_df, edges_df, grants_df, parse_results, merge_stats, processed_orgs)
            
            # Rerun to show results
            st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    init_session_state()
    
    # Header
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image(C4C_LOGO_URL, width=80)
    with col2:
        st.title("OrgGraph (US)")
    
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
                placeholder="e.g., Water Funders Network",
                help="Choose a descriptive name for your project"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            create_btn = st.button("Create Project", type="primary", disabled=not new_project_name)
        
        if new_project_name:
            folder_name = get_folder_name(new_project_name)
            st.caption(f"📁 Will create folder: `demo_data/{folder_name}/`")
        
        if create_btn and new_project_name:
            success, message = create_project(new_project_name)
            if success:
                st.success(f"✅ {message}")
                st.session_state.current_project = get_folder_name(new_project_name)
                clear_session_state()
                st.rerun()
            else:
                st.error(f"❌ {message}")
        
        # If project was just created, show upload interface
        if st.session_state.current_project:
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
            st.session_state.current_project = selected_project
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
            with st.expander(f"📋 Foundations in Demo ({len(existing_foundations)})", expanded=True):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        # Reconstruct grants_df for analytics
        grants_df = None
        if not edges_df.empty and "edge_type" in edges_df.columns:
            grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy()
            if not grant_edges.empty:
                grants_df = pd.DataFrame({
                    'foundation_name': grant_edges['from_id'],
                    'grantee_name': grant_edges['to_id'],
                    'grant_amount': grant_edges['amount'],
                    'grantee_state': grant_edges.get('region', ''),
                })
        
        # Render outputs (read-only)
        render_graph_summary(nodes_df, edges_df, grants_df)
        
        show_analytics = st.checkbox("📊 Show Network Analytics", value=False)
        if show_analytics:
            render_analytics(grants_df)
        
        render_data_preview(nodes_df, edges_df)
        render_downloads(nodes_df, edges_df, grants_df, None, DEMO_PROJECT_NAME)


if __name__ == "__main__":
    main()
