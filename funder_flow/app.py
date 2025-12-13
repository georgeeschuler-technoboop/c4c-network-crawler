"""
C4C 990 Funder Flow — Network Intelligence Engine

Dual-mode Streamlit app:
- GLFN Demo: Load pre-built canonical graph from repo
- Add to GLFN: Upload IRS 990-PF filings, auto-merge with existing GLFN data

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
from datetime import datetime

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4c_utils.irs990_parser import parse_990_pdf
from c4c_utils.network_export import build_nodes_df, build_edges_df, NODE_COLUMNS, EDGE_COLUMNS, get_existing_foundations


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
# Data Loading Functions
# =============================================================================

def load_glfn_data() -> tuple:
    """Load existing GLFN data from demo_data/glfn/."""
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
                "grants": grants_count,
                "people": people_count,
                "total_amount": total_amount,
                "is_990pf": diagnostics.get('is_990pf', False),
                "form_type": diagnostics.get('form_type', 'unknown'),
                "message": message,
                "error": None,
            })
            
        except Exception as e:
            parse_results.append({
                "file": uploaded_file.name,
                "status": "error",
                "org_name": "",
                "grants": 0,
                "people": 0,
                "total_amount": 0,
                "is_990pf": False,
                "form_type": "unknown",
                "message": "",
                "error": str(e),
            })
    
    grants_df = pd.concat(all_grants, ignore_index=True) if all_grants else pd.DataFrame()
    people_df = pd.concat(all_people, ignore_index=True) if all_people else pd.DataFrame()
    
    nodes_df = build_nodes_df(grants_df, people_df, foundations_meta)
    edges_df = build_edges_df(grants_df, people_df, foundations_meta)
    
    return nodes_df, edges_df, grants_df, foundations_meta, parse_results


def create_processing_log(parse_results: list) -> pd.DataFrame:
    """Create a DataFrame of processing results for download."""
    rows = []
    for r in parse_results:
        rows.append({
            "File": r.get("file", ""),
            "Status": r.get("status", ""),
            "Organization": r.get("org_name", ""),
            "Form Type": r.get("form_type", ""),
            "Is 990-PF": r.get("is_990pf", ""),
            "Grants Found": r.get("grants", 0),
            "Total Amount": r.get("total_amount", 0),
            "Board Members": r.get("people", 0),
            "Message": r.get("message", ""),
            "Error": r.get("error", ""),
        })
    return pd.DataFrame(rows)


# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_parse_status(parse_results: list) -> None:
    """Render parsing status for each file."""
    st.markdown("### 📋 Processing Log")
    
    # Summary counts
    success_count = sum(1 for r in parse_results if r["status"] == "success")
    warning_count = sum(1 for r in parse_results if r["status"] in ["wrong_form", "no_grants"])
    error_count = sum(1 for r in parse_results if r["status"] == "error")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Success", success_count)
    with col2:
        st.metric("⚠️ Warnings", warning_count)
    with col3:
        st.metric("❌ Errors", error_count)
    
    # Detailed log in expander
    with st.expander("View detailed log", expanded=True):
        for result in parse_results:
            if result["status"] == "success":
                st.success(f"✅ **{result['file']}** — {result['message']} ({result['org_name']})")
            elif result["status"] == "wrong_form":
                st.warning(f"📋 **{result['file']}** — {result['message']} ({result['org_name']})")
            elif result["status"] == "no_grants":
                st.warning(f"⚠️ **{result['file']}** — {result['message']} ({result['org_name']})")
            elif result["status"] == "error":
                st.error(f"❌ **{result['file']}** — Error: {result['error']}")


def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, grants_df: pd.DataFrame = None) -> None:
    """Render summary metrics for the graph."""
    
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
    
    if grants_df is not None and not grants_df.empty:
        total_amount = grants_df['grant_amount'].sum()
        unique_grantees = grants_df['grantee_name'].nunique()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Grant Amount", f"${total_amount:,.0f}")
        with col2:
            st.metric("Unique Grantees", unique_grantees)


def render_analytics(grants_df: pd.DataFrame) -> None:
    """Render network analytics section."""
    
    if grants_df is None or grants_df.empty:
        st.info("📊 Analytics require grant data.")
        return
    
    st.subheader("📊 Network Analytics")
    
    # Funder Overlap
    st.markdown("#### 🔗 Funder Overlap: Grantees with Multiple Funders")
    
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
        st.info("No grantees receive funding from multiple foundations yet.")
    
    st.divider()
    
    # Geographic Distribution
    if 'grantee_state' in grants_df.columns:
        st.markdown("#### 🌍 Geographic Distribution")
        
        state_funding = grants_df.groupby('grantee_state').agg({
            'grantee_name': 'nunique',
            'grant_amount': 'sum'
        }).reset_index()
        state_funding.columns = ['State', 'Grantees', 'Total Funding']
        state_funding = state_funding.sort_values('Total Funding', ascending=False)
        state_funding['Total Funding'] = state_funding['Total Funding'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(state_funding.head(15), use_container_width=True, hide_index=True)


def render_data_preview(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Render data preview tables."""
    
    st.divider()
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Nodes", "Edges"])
    
    with tab1:
        if not nodes_df.empty:
            display_cols = ["node_type", "label", "jurisdiction", "city", "region", "tax_id", "source_system"]
            display_cols = [c for c in display_cols if c in nodes_df.columns]
            st.dataframe(nodes_df[display_cols].head(50), use_container_width=True, hide_index=True)
            st.caption(f"{len(nodes_df)} total nodes")
        else:
            st.info("No nodes found.")
    
    with tab2:
        if not edges_df.empty:
            display_cols = ["edge_type", "from_id", "to_id", "amount", "role", "fiscal_year", "source_system"]
            display_cols = [c for c in display_cols if c in edges_df.columns]
            st.dataframe(edges_df[display_cols].head(50), use_container_width=True, hide_index=True)
            st.caption(f"{len(edges_df)} total edges")
        else:
            st.info("No edges found.")


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                     grants_df: pd.DataFrame = None, parse_results: list = None) -> None:
    """Render download buttons."""
    
    if nodes_df is None or nodes_df.empty:
        return
    
    st.divider()
    st.subheader("📥 Download & Upload to GitHub")
    
    st.info("⬇️ **Download these files and upload to `demo_data/glfn/` on GitHub** (replace existing files)")
    
    # Create processing log if available
    processing_log = None
    if parse_results:
        processing_log = create_processing_log(parse_results)
    
    def create_zip_download():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if not nodes_df.empty:
                zf.writestr("nodes.csv", nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr("edges.csv", edges_df.to_csv(index=False))
            if grants_df is not None and not grants_df.empty:
                zf.writestr("grants_raw.csv", grants_df.to_csv(index=False))
            if processing_log is not None:
                zf.writestr("processing_log.csv", processing_log.to_csv(index=False))
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
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        if processing_log is not None:
            st.download_button(
                label="📄 processing_log.csv",
                data=processing_log.to_csv(index=False),
                file_name="processing_log.csv",
                mime="text/csv"
            )


# =============================================================================
# Main App
# =============================================================================

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("C4C 990 Funder Flow")
    
    st.markdown("""
    Parse IRS **990-PF** filings and build the GLFN network graph.
    
    *Outputs conform to C4C Network Schema v1.*
    """)
    
    st.divider()
    
    project_mode = st.selectbox(
        "Mode",
        ["GLFN Demo (view existing)", "Add to GLFN (upload + merge)"],
        index=0,
        help="View existing GLFN data or add new 990-PF filings"
    )
    
    st.divider()
    
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
        
        # Render outputs
        render_graph_summary(nodes_df, edges_df, grants_df)
        
        show_analytics = st.checkbox("📊 Show Network Analytics", value=False)
        if show_analytics:
            render_analytics(grants_df)
        
        render_data_preview(nodes_df, edges_df)
        render_downloads(nodes_df, edges_df, grants_df)
        
    else:
        # ---------------------------------------------------------------------
        # Add to GLFN Mode
        # ---------------------------------------------------------------------
        existing_nodes, existing_edges = load_glfn_data()
        
        # Show existing data status
        if not existing_nodes.empty or not existing_edges.empty:
            st.success(f"📂 **Existing GLFN data:** {len(existing_nodes)} nodes, {len(existing_edges)} edges")
            
            existing_foundations = get_existing_foundations(existing_nodes)
            if existing_foundations:
                with st.expander(f"📋 Foundations already in GLFN ({len(existing_foundations)})", expanded=False):
                    for label, source in existing_foundations:
                        flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                        st.write(f"{flag} {label}")
            
            st.caption("New data will be merged. Duplicates automatically skipped.")
        else:
            st.info("📂 **No existing GLFN data.** This will be the first upload.")
        
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
            st.success(f"📊 **Combined GLFN dataset:** {len(st.session_state.nodes_df)} nodes, {len(st.session_state.edges_df)} edges")
            
            # Render outputs from session state
            render_graph_summary(st.session_state.nodes_df, st.session_state.edges_df, st.session_state.grants_df)
            
            show_analytics = st.checkbox("📊 Show Network Analytics", value=False)
            if show_analytics:
                render_analytics(st.session_state.grants_df)
            
            render_data_preview(st.session_state.nodes_df, st.session_state.edges_df)
            render_downloads(st.session_state.nodes_df, st.session_state.edges_df, 
                           st.session_state.grants_df, st.session_state.parse_results)
            
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


if __name__ == "__main__":
    main()
