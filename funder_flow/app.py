"""
C4C 990 Funder Flow — Network Intelligence Engine

Dual-mode Streamlit app:
- GLFN Demo: Load pre-built canonical graph from repo
- New Project: Upload IRS 990-PF filings and generate canonical outputs

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
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4c_utils.irs990_parser import parse_990_pdf
from c4c_utils.network_export import build_nodes_df, build_edges_df


# =============================================================================
# Constants
# =============================================================================

MAX_FILES = 50
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_9e232fe9e6914305a7ea8746e2e77125~mv2.png"
SOURCE_SYSTEM = "IRS_990"
JURISDICTION = "US"

# Demo data paths
REPO_ROOT = Path(__file__).resolve().parent.parent
GLFN_DEMO_DIR = REPO_ROOT / "demo_data" / "glfn"


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="C4C 990 Funder Flow",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)


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


def process_uploaded_files(uploaded_files, tax_year_override: str = "") -> tuple:
    """
    Process uploaded 990-PF files and return canonical outputs.
    Returns: (nodes_df, edges_df, grants_df, foundations_meta, parse_results)
    """
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
            
            foundations_meta.append(result['foundation_meta'])
            
            if not result['grants_df'].empty:
                all_grants.append(result['grants_df'])
            
            if not result['people_df'].empty:
                all_people.append(result['people_df'])
            
            parse_results.append({
                "file": uploaded_file.name,
                "success": True,
                "org_name": org_name,
                "grants": grants_count,
                "people": people_count,
                "total_amount": result['grants_df']['grant_amount'].sum() if grants_count > 0 else 0,
                "is_990pf": diagnostics.get('is_990pf', False),
                "form_type": diagnostics.get('form_type', 'unknown'),
            })
            
        except Exception as e:
            parse_results.append({
                "file": uploaded_file.name,
                "success": False,
                "error": str(e),
            })
    
    # Combine DataFrames
    grants_df = pd.concat(all_grants, ignore_index=True) if all_grants else pd.DataFrame()
    people_df = pd.concat(all_people, ignore_index=True) if all_people else pd.DataFrame()
    
    # Build canonical nodes and edges
    nodes_df = build_nodes_df(grants_df, people_df, foundations_meta)
    edges_df = build_edges_df(grants_df, people_df, foundations_meta)
    
    return nodes_df, edges_df, grants_df, foundations_meta, parse_results


# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_parse_status(parse_results: list) -> None:
    """Render parsing status for each file."""
    st.markdown("### Parsing Status")
    
    for result in parse_results:
        if result.get("success"):
            if result["grants"] > 0:
                st.success(
                    f"✅ **{result['file']}** – "
                    f"{result['grants']} grants (${result['total_amount']:,.0f}), "
                    f"{result['people']} board members ({result['org_name']})"
                )
            elif not result.get("is_990pf"):
                st.warning(
                    f"📋 **{result['file']}** is a **Form {result['form_type']}** (public charity) – "
                    f"no itemized grants. Found {result['people']} board members."
                )
            else:
                st.warning(
                    f"⚠️ **{result['file']}** – No grants found. "
                    f"Found {result['people']} board members."
                )
        else:
            st.error(f"❌ **{result['file']}**: {result.get('error', 'Unknown error')}")


def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, grants_df: pd.DataFrame = None) -> None:
    """Render summary metrics for the graph."""
    
    if nodes_df is None or nodes_df.empty:
        st.warning("No graph data loaded.")
        return
    
    st.subheader("📊 Summary")
    
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
    
    # Grant totals if available
    if grants_df is not None and not grants_df.empty:
        total_amount = grants_df['grant_amount'].sum()
        unique_grantees = grants_df['grantee_name'].nunique()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Grant Amount", f"${total_amount:,.0f}")
        with col2:
            st.metric("Unique Grantees", unique_grantees)


def render_analytics(grants_df: pd.DataFrame, people_df: pd.DataFrame = None) -> None:
    """Render network analytics section."""
    
    if grants_df is None or grants_df.empty:
        st.info("📊 Analytics require grant data.")
        return
    
    st.subheader("📊 Network Analytics")
    
    # --- Funder Overlap Analysis ---
    st.markdown("#### 🔗 Funder Overlap: Grantees Receiving from Multiple Foundations")
    
    grantee_funder_counts = grants_df.groupby('grantee_name')['foundation_name'].nunique().reset_index()
    grantee_funder_counts.columns = ['Grantee', 'Number of Funders']
    
    grantee_totals = grants_df.groupby('grantee_name')['grant_amount'].sum().reset_index()
    grantee_totals.columns = ['Grantee', 'Total Funding']
    
    overlap_df = grantee_funder_counts.merge(grantee_totals, on='Grantee')
    overlap_df = overlap_df.sort_values('Number of Funders', ascending=False)
    
    multi_funder = overlap_df[overlap_df['Number of Funders'] > 1].copy()
    
    if not multi_funder.empty:
        st.success(f"**{len(multi_funder)}** grantees receive funding from multiple foundations")
        multi_funder['Total Funding'] = multi_funder['Total Funding'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(multi_funder.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("No grantees receive funding from multiple foundations in this dataset.")
    
    st.divider()
    
    # --- Geographic Distribution ---
    st.markdown("#### 🌍 Geographic Distribution of Grantees")
    
    if 'grantee_state' in grants_df.columns:
        state_funding = grants_df.groupby('grantee_state').agg({
            'grantee_name': 'nunique',
            'grant_amount': 'sum'
        }).reset_index()
        state_funding.columns = ['State', 'Unique Grantees', 'Total Funding']
        state_funding = state_funding.sort_values('Total Funding', ascending=False)
        
        state_funding['Total Funding'] = state_funding['Total Funding'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(state_funding.head(15), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # --- Grant Size Distribution ---
    st.markdown("#### 💰 Grant Size Distribution")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Grant", f"${grants_df['grant_amount'].min():,.0f}")
    with col2:
        st.metric("Median Grant", f"${grants_df['grant_amount'].median():,.0f}")
    with col3:
        st.metric("Mean Grant", f"${grants_df['grant_amount'].mean():,.0f}")
    with col4:
        st.metric("Max Grant", f"${grants_df['grant_amount'].max():,.0f}")


def render_data_preview(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Render data preview tables."""
    
    st.subheader("📋 Data Preview")
    st.caption("Preview of canonical graph data. Downloads include full schema.")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        if not nodes_df.empty:
            display_cols = ["node_type", "label", "jurisdiction", "city", "region", "tax_id"]
            display_cols = [c for c in display_cols if c in nodes_df.columns]
            st.dataframe(nodes_df[display_cols].head(50), use_container_width=True, hide_index=True)
            if len(nodes_df) > 50:
                st.caption(f"Showing 50 of {len(nodes_df)} nodes")
        else:
            st.info("No nodes found.")
    
    with tab2:
        if not edges_df.empty:
            display_cols = ["edge_type", "from_id", "to_id", "amount", "role", "fiscal_year"]
            display_cols = [c for c in display_cols if c in edges_df.columns]
            st.dataframe(edges_df[display_cols].head(50), use_container_width=True, hide_index=True)
            if len(edges_df) > 50:
                st.caption(f"Showing 50 of {len(edges_df)} edges")
        else:
            st.info("No edges found.")


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, grants_df: pd.DataFrame = None) -> None:
    """Render download buttons."""
    
    if nodes_df is None or nodes_df.empty:
        return
    
    st.divider()
    st.subheader("📥 Download Outputs")
    
    def create_zip_download():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if not nodes_df.empty:
                zf.writestr("nodes.csv", nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr("edges.csv", edges_df.to_csv(index=False))
            if grants_df is not None and not grants_df.empty:
                zf.writestr("grants_raw.csv", grants_df.to_csv(index=False))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    st.download_button(
        label="📦 Download All (ZIP)",
        data=create_zip_download(),
        file_name="c4c_990_funder_flow.zip",
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
        if grants_df is not None and not grants_df.empty:
            st.download_button(
                label="📄 grants_raw.csv",
                data=grants_df.to_csv(index=False),
                file_name="grants_raw.csv",
                mime="text/csv"
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
        st.title("C4C 990 Funder Flow")
    
    st.markdown("""
    Parse IRS **990-PF** filings and generate canonical network graphs.
    
    *Outputs conform to C4C Network Schema v1.*
    """)
    
    st.divider()
    
    # Project Mode Selector
    project_mode = st.selectbox(
        "Project",
        ["GLFN Demo (pre-loaded)", "New Project (upload 990-PFs)"],
        index=0,
        help="GLFN Demo loads pre-built data. New Project lets you upload files."
    )
    
    st.divider()
    
    # Initialize outputs
    nodes_df = pd.DataFrame()
    edges_df = pd.DataFrame()
    grants_df = None
    
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
        
        # For demo mode, try to reconstruct grants_df from edges for analytics
        if not edges_df.empty and "edge_type" in edges_df.columns:
            grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy()
            if not grant_edges.empty:
                # Create minimal grants_df for analytics
                grants_df = pd.DataFrame({
                    'foundation_name': grant_edges['from_id'],
                    'grantee_name': grant_edges['to_id'],
                    'grant_amount': grant_edges['amount'],
                    'grantee_state': grant_edges.get('region', ''),
                })
        
    else:
        # ---------------------------------------------------------------------
        # New Project (Upload) Mode
        # ---------------------------------------------------------------------
        st.subheader("📤 Upload 990-PF PDFs")
        
        st.markdown(f"""
        Upload up to **{MAX_FILES} private foundation 990-PF filings** to extract grant data.
        
        *Note: This works with **990-PF** (private foundations). Standard **990** filings 
        from public charities typically don't include itemized grants.*
        """)
        
        uploaded_files = st.file_uploader(
            f"Upload 990-PF PDF(s) (max {MAX_FILES})",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Upload 1–{MAX_FILES} IRS 990-PF PDF files"
        )
        
        if uploaded_files and len(uploaded_files) > MAX_FILES:
            st.warning(f"⚠️ Please upload at most {MAX_FILES} files.")
            uploaded_files = uploaded_files[:MAX_FILES]
        
        tax_year_override = st.text_input(
            "Tax Year (optional)",
            help="Override auto-detection of tax year if needed."
        )
        
        parse_button = st.button("🔍 Parse 990 Filings", type="primary", disabled=not uploaded_files)
        
        if not uploaded_files:
            st.info("👆 Upload 990-PF PDF files to generate the graph.")
            st.stop()
        
        if not parse_button:
            st.stop()
        
        # Process uploads
        with st.spinner("Parsing filings..."):
            nodes_df, edges_df, grants_df, foundations_meta, parse_results = process_uploaded_files(
                uploaded_files, tax_year_override
            )
        
        render_parse_status(parse_results)
        
        if nodes_df.empty:
            st.warning("No data extracted from uploaded files.")
            st.stop()
        
        st.divider()
    
    # ---------------------------------------------------------------------
    # Common Output (both modes)
    # ---------------------------------------------------------------------
    render_graph_summary(nodes_df, edges_df, grants_df)
    
    st.divider()
    
    # Analytics toggle
    show_analytics = st.checkbox("📊 Show Network Analytics", value=False)
    
    if show_analytics:
        render_analytics(grants_df)
    
    st.divider()
    
    render_data_preview(nodes_df, edges_df)
    render_downloads(nodes_df, edges_df, grants_df)


if __name__ == "__main__":
    main()
