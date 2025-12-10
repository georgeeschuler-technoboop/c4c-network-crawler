"""
C4C 990 Funder Flow Prototype

Upload IRS 990/990-PF filings and generate Polinode-ready network data:
- Grants table (funder → grantee)
- Unified nodes list (foundations, grantees, people)
- Unified edges list (grant + board membership edges)

Part of the C4C Network Intelligence Engine.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from io import BytesIO
import sys
import os

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from c4c_utils.irs990_parser import parse_990_pdf
from c4c_utils.network_export import build_nodes_df, build_edges_df


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="C4C 990 Funder Flow",
    page_icon="🔍",
    layout="wide"
)


# =============================================================================
# App Header
# =============================================================================
st.title("🔍 C4C 990 Funder Flow Prototype")

st.markdown("""
Upload IRS 990/990-PF filings (e.g., Porter Family Foundation) and generate:
- A **grants table** (funder → grantee)
- A **unified nodes list** (foundations, grantees, people)
- A **unified edges list** (grant + board membership edges)

These CSVs are **Polinode-ready** and will eventually plug into the broader 
C4C Network Intelligence Engine.
""")

st.divider()


# =============================================================================
# Step 1: File Upload
# =============================================================================
st.subheader("Step 1 – Upload 990/990-PF PDFs")

st.markdown("""
*Start with one filing (like the Porter Family Foundation 990-PF) to validate 
the parser. You can upload more than one later to see overlap across multiple funders.*
""")

uploaded_files = st.file_uploader(
    "Upload 990/990-PF PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more IRS 990-PF PDF files"
)

tax_year_override = st.text_input(
    "Tax Year (optional)",
    help="If left blank, the app will try to detect the tax year from each filing. "
         "You can override this if needed (e.g., 2022)."
)

st.divider()


# =============================================================================
# Step 2: Parse Button
# =============================================================================
st.subheader("Step 2 – Parse Filings")

st.markdown("""
*The app will extract grant schedules and board/officer tables from each filing 
and build three CSVs: `grants.csv`, `nodes.csv`, `edges.csv`*
""")

parse_button = st.button("🔍 Parse 990 Filings", type="primary", disabled=not uploaded_files)


# =============================================================================
# Processing & Results
# =============================================================================
if parse_button and uploaded_files:
    
    # Initialize storage
    all_grants = []
    all_people = []
    foundations_meta = []
    
    # Status container
    status_container = st.container()
    
    with status_container:
        st.markdown("### Parsing Status")
        
        for uploaded_file in uploaded_files:
            with st.spinner(f"Parsing: {uploaded_file.name}..."):
                try:
                    # Read file bytes
                    file_bytes = uploaded_file.read()
                    
                    # Parse the PDF
                    result = parse_990_pdf(
                        file_bytes, 
                        uploaded_file.name, 
                        tax_year_override
                    )
                    
                    # Collect results
                    foundations_meta.append(result['foundation_meta'])
                    
                    if not result['grants_df'].empty:
                        all_grants.append(result['grants_df'])
                    
                    if not result['people_df'].empty:
                        all_people.append(result['people_df'])
                    
                    # Report success
                    grants_count = len(result['grants_df'])
                    people_count = len(result['people_df'])
                    foundation_name = result['foundation_meta'].get('foundation_name', 'Unknown')
                    
                    if grants_count > 0:
                        st.success(
                            f"✅ Parsed **{uploaded_file.name}** – "
                            f"{grants_count} grants, {people_count} board members "
                            f"({foundation_name})"
                        )
                    else:
                        st.warning(
                            f"⚠️ Could not find a grants schedule in **{uploaded_file.name}**. "
                            f"Check that this is a 990-PF with a grants schedule."
                        )
                
                except Exception as e:
                    st.error(f"❌ Error parsing **{uploaded_file.name}**: {str(e)}")
    
    st.divider()
    
    # =============================================================================
    # Build Combined DataFrames
    # =============================================================================
    
    # Combine grants
    if all_grants:
        grants_df = pd.concat(all_grants, ignore_index=True)
    else:
        grants_df = pd.DataFrame()
    
    # Combine people
    if all_people:
        people_df = pd.concat(all_people, ignore_index=True)
    else:
        people_df = pd.DataFrame()
    
    # Build nodes and edges
    nodes_df = build_nodes_df(grants_df, people_df, foundations_meta)
    edges_df = build_edges_df(grants_df, people_df, foundations_meta)
    
    # =============================================================================
    # Summary
    # =============================================================================
    st.subheader("📊 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Foundations Parsed", len(foundations_meta))
    
    with col2:
        st.metric("Total Grants", len(grants_df))
    
    with col3:
        unique_grantees = grants_df['grantee_name'].nunique() if not grants_df.empty else 0
        st.metric("Unique Grantees", unique_grantees)
    
    with col4:
        st.metric("Board Members", len(people_df))
    
    if not grants_df.empty:
        total_amount = grants_df['grant_amount'].sum()
        st.metric("Total Grant Amount", f"${total_amount:,.0f}")
    
    st.divider()
    
    # =============================================================================
    # Data Previews
    # =============================================================================
    
    # Grants Preview
    st.subheader("📋 Grants: Funder → Grantee")
    st.markdown("""
    *This table shows each grant line item extracted from the filings. 
    Download as `grants.csv` for deeper analysis if needed.*
    """)
    
    if not grants_df.empty:
        st.dataframe(grants_df.head(50), use_container_width=True)
        if len(grants_df) > 50:
            st.caption(f"Showing 50 of {len(grants_df)} grants")
    else:
        st.info("No grants found in uploaded files.")
    
    st.divider()
    
    # Nodes Preview
    st.subheader("🔵 Nodes: Foundations, Grantees, People")
    st.markdown("""
    *This is the unified node list, combining:*
    - **Foundations** (type = foundation)
    - **Grantees** (type = grantee)  
    - **People** (type = person)
    
    *Each row has a `node_id` that matches the `from_id` / `to_id` columns in `edges.csv`.*
    """)
    
    if not nodes_df.empty:
        st.dataframe(nodes_df.head(50), use_container_width=True)
        if len(nodes_df) > 50:
            st.caption(f"Showing 50 of {len(nodes_df)} nodes")
    else:
        st.info("No nodes generated.")
    
    st.divider()
    
    # Edges Preview
    st.subheader("➡️ Edges: Grants + Board Memberships")
    st.markdown("""
    *This file combines:*
    - **Grant edges** (edge_type = "grant") from foundations to grantees
    - **Board membership edges** (edge_type = "board_membership") from people to foundations
    
    *You can filter by `edge_type` in Polinode or your analysis tools.*
    """)
    
    if not edges_df.empty:
        st.dataframe(edges_df.head(50), use_container_width=True)
        if len(edges_df) > 50:
            st.caption(f"Showing 50 of {len(edges_df)} edges")
    else:
        st.info("No edges generated.")
    
    st.divider()
    
    # =============================================================================
    # Download Buttons
    # =============================================================================
    st.subheader("📥 Download CSVs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grants_csv = grants_df.to_csv(index=False) if not grants_df.empty else ""
        st.download_button(
            label="⬇️ Download grants.csv",
            data=grants_csv,
            file_name="grants.csv",
            mime="text/csv",
            help="Raw grant table (one row per grant line item)"
        )
    
    with col2:
        nodes_csv = nodes_df.to_csv(index=False) if not nodes_df.empty else ""
        st.download_button(
            label="⬇️ Download nodes.csv",
            data=nodes_csv,
            file_name="nodes.csv",
            mime="text/csv",
            help="Unified list of foundations, grantees, and people (Polinode-ready)"
        )
    
    with col3:
        edges_csv = edges_df.to_csv(index=False) if not edges_df.empty else ""
        st.download_button(
            label="⬇️ Download edges.csv",
            data=edges_csv,
            file_name="edges.csv",
            mime="text/csv",
            help="Unified list of grant and board membership edges (Polinode-ready)"
        )


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption(
    "C4C 990 Funder Flow Prototype v0.1 | "
    "Part of the [Connecting for Change](https://connectingforchange.io) Network Intelligence Engine"
)
