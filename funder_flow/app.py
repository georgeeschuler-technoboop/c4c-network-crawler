"""
C4C 990 Funder Flow Prototype

Upload IRS 990-PF filings and generate Polinode-ready network data:
- Grants table (funder → grantee)
- Unified nodes list (foundations, grantees, people)
- Unified edges list (grant + board membership edges)

Part of the C4C Network Intelligence Engine.
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import zipfile
import sys
import os

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4c_utils.irs990_parser import parse_990_pdf
from c4c_utils.network_export import build_nodes_df, build_edges_df


# =============================================================================
# Constants
# =============================================================================
MAX_FILES = 5
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_9e232fe9e6914305a7ea8746e2e77125~mv2.png"


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="C4C 990 Funder Flow",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)


# =============================================================================
# App Header
# =============================================================================
# Logo + Title row
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.image(C4C_LOGO_URL, width=60)
with col_title:
    st.title("C4C 990 Funder Flow Prototype")

st.markdown("""
Upload IRS **990-PF** filings (private foundations) and generate:
- A **grants table** (funder → grantee)
- A **unified nodes list** (foundations, grantees, people)
- A **unified edges list** (grant + board membership edges)

These CSVs are **Polinode-ready** and will eventually plug into the broader 
C4C Network Intelligence Engine.
""")

# Mode toggle
st.divider()
mode = st.radio(
    "Mode",
    ["📋 Basic (Parse & Export)", "📊 Advanced (Analytics)"],
    horizontal=True,
    help="Basic mode: Parse 990-PFs and download CSVs. Advanced mode: Additional network analytics and insights."
)
is_analytics_mode = "Advanced" in mode

st.divider()


# =============================================================================
# Step 1: File Upload
# =============================================================================
st.subheader("Step 1 – Upload 990-PF PDFs")

st.markdown(f"""
Upload up to **{MAX_FILES} private foundation 990-PF filings** to extract grant data.

*Note: This tool works with **990-PF** (private foundation) filings, which include 
grants schedules. Standard **990** filings from public charities typically don't 
include itemized grants.*
""")

uploaded_files = st.file_uploader(
    f"Upload 990-PF PDF(s) (max {MAX_FILES})",
    type=["pdf"],
    accept_multiple_files=True,
    help=f"Upload 1–{MAX_FILES} IRS 990-PF PDF files from private foundations"
)

# Enforce file limit
if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.warning(f"⚠️ Please upload at most {MAX_FILES} files. You uploaded {len(uploaded_files)}.")
    uploaded_files = uploaded_files[:MAX_FILES]
    st.info(f"Processing first {MAX_FILES} files only.")

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
The app will extract grant schedules and board/officer tables from each filing 
and build three CSVs: `grants.csv`, `nodes.csv`, `edges.csv`
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
                    
                    # Get diagnostics if available
                    diagnostics = result.get('diagnostics', {})
                    form_type = diagnostics.get('form_type', 'unknown')
                    is_990pf = diagnostics.get('is_990pf', False)
                    grants_reported = diagnostics.get('grants_reported_amount')
                    org_name = diagnostics.get('org_name', '') or result['foundation_meta'].get('foundation_name', 'Unknown')
                    
                    # Collect results
                    foundations_meta.append(result['foundation_meta'])
                    
                    if not result['grants_df'].empty:
                        all_grants.append(result['grants_df'])
                    
                    if not result['people_df'].empty:
                        all_people.append(result['people_df'])
                    
                    # Report results with appropriate messaging
                    grants_count = len(result['grants_df'])
                    people_count = len(result['people_df'])
                    
                    if grants_count > 0:
                        # Success - found grants
                        total_amount = result['grants_df']['grant_amount'].sum()
                        st.success(
                            f"✅ **{uploaded_file.name}** – "
                            f"{grants_count} grants (${total_amount:,.0f}), {people_count} board members "
                            f"({org_name})"
                        )
                    
                    elif not is_990pf and form_type != 'unknown':
                        # Wrong form type - this is a 990, not 990-PF
                        msg = f"📋 **{uploaded_file.name}** is a **Form {form_type}** (public charity)"
                        
                        msg += f"\n\n• Organization: **{org_name}**"
                        
                        if grants_reported is not None:
                            if grants_reported == 0:
                                msg += "\n• This organization reported **$0 in grants** paid during the year."
                            else:
                                msg += f"\n• Grants reported: **${grants_reported:,}**"
                                if grants_reported < 5000:
                                    msg += " *(below $5,000 itemization threshold)*"
                        
                        msg += "\n\n*Form 990 is for public charities. Grants are only itemized on Schedule I if over $5,000. "
                        msg += "This tool is designed for **990-PF** (private foundation) filings, which include detailed grant schedules.*"
                        
                        st.info(msg)
                        
                        # Still report board members if found
                        if people_count > 0:
                            st.caption(f"   ↳ Found {people_count} board members (added to network)")
                    
                    elif is_990pf and grants_count == 0:
                        # Is a 990-PF but no grants found
                        msg = f"⚠️ **{uploaded_file.name}** – No grants extracted"
                        msg += f"\n\n• Organization: **{org_name}**"
                        msg += "\n• This appears to be a 990-PF filing."
                        msg += "\n\n**Possible reasons:**"
                        msg += "\n• The foundation may not have made any grants this year"
                        msg += "\n• Grants may be listed in a different format than expected"
                        msg += "\n• The PDF structure may differ from typical ProPublica exports"
                        msg += "\n\n*Tip: Check Part XV of the original filing to verify grant activity.*"
                        
                        st.warning(msg)
                        
                        if people_count > 0:
                            st.caption(f"   ↳ Found {people_count} board members (added to network)")
                    
                    else:
                        # Unknown form type, no grants
                        msg = f"⚠️ **{uploaded_file.name}** – No grants found"
                        if org_name:
                            msg += f"\n\n• Organization: **{org_name}**"
                        
                        msg += "\n\n**Possible reasons:**"
                        msg += "\n• This may be a **Form 990** (public charity) rather than a **990-PF** (private foundation)"
                        msg += "\n• The organization may have reported **less than $5,000** in grants (not itemized on Form 990)"
                        msg += "\n• The organization may not have made any grants this year"
                        msg += "\n• The PDF format may differ from expected structure"
                        msg += "\n\n*Tip: Check the original filing to verify grant activity.*"
                        
                        st.warning(msg)
                        
                        if people_count > 0:
                            st.caption(f"   ↳ Found {people_count} board members (added to network)")
                
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
    # Analytics Mode Section
    # =============================================================================
    if is_analytics_mode and not grants_df.empty:
        st.subheader("📊 Network Analytics")
        
        # --- Funder Overlap Analysis ---
        st.markdown("#### 🔗 Funder Overlap: Grantees Receiving from Multiple Foundations")
        st.markdown("*Which organizations are funded by multiple foundations in this network?*")
        
        # Count how many foundations fund each grantee
        grantee_funder_counts = grants_df.groupby('grantee_name')['foundation_name'].nunique().reset_index()
        grantee_funder_counts.columns = ['Grantee', 'Number of Funders']
        
        # Get total funding per grantee
        grantee_totals = grants_df.groupby('grantee_name')['grant_amount'].sum().reset_index()
        grantee_totals.columns = ['Grantee', 'Total Funding']
        
        # Merge
        overlap_df = grantee_funder_counts.merge(grantee_totals, on='Grantee')
        overlap_df = overlap_df.sort_values('Number of Funders', ascending=False)
        
        # Filter to multi-funder grantees
        multi_funder = overlap_df[overlap_df['Number of Funders'] > 1].copy()
        
        if not multi_funder.empty:
            st.success(f"**{len(multi_funder)}** grantees receive funding from multiple foundations")
            
            # Format for display
            multi_funder['Total Funding'] = multi_funder['Total Funding'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(multi_funder.head(20), use_container_width=True, hide_index=True)
            
            if len(multi_funder) > 20:
                st.caption(f"Showing top 20 of {len(multi_funder)} multi-funder grantees")
            
            # Show which funders for top overlaps
            st.markdown("**Funding details for top overlapping grantees:**")
            for _, row in multi_funder.head(5).iterrows():
                grantee = row['Grantee']
                funders = grants_df[grants_df['grantee_name'] == grantee].groupby('foundation_name')['grant_amount'].sum()
                funder_list = ", ".join([f"{f} (${amt:,.0f})" for f, amt in funders.items()])
                st.caption(f"• **{grantee}**: {funder_list}")
        else:
            st.info("No grantees receive funding from multiple foundations in this dataset.")
        
        st.divider()
        
        # --- Top Grantees by Total Funding ---
        st.markdown("#### 💰 Top Grantees by Total Funding")
        st.markdown("*Which organizations receive the most funding across all foundations?*")
        
        top_grantees = overlap_df.sort_values('Total Funding' if 'Total Funding' in overlap_df.columns else overlap_df.columns[2], ascending=False).head(15).copy()
        
        # Re-merge to get numeric values for sorting
        top_grantees = grants_df.groupby('grantee_name').agg({
            'grant_amount': 'sum',
            'foundation_name': 'nunique'
        }).reset_index()
        top_grantees.columns = ['Grantee', 'Total Funding', 'Number of Funders']
        top_grantees = top_grantees.sort_values('Total Funding', ascending=False).head(15)
        
        # Display as bar chart
        chart_data = top_grantees.set_index('Grantee')['Total Funding']
        st.bar_chart(chart_data)
        
        # Also show table
        top_grantees_display = top_grantees.copy()
        top_grantees_display['Total Funding'] = top_grantees_display['Total Funding'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_grantees_display, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # --- Geographic Distribution ---
        st.markdown("#### 🗺️ Geographic Distribution")
        st.markdown("*Where are grants going? (by state)*")
        
        if 'grantee_state' in grants_df.columns:
            # Group by state
            state_funding = grants_df.groupby('grantee_state').agg({
                'grant_amount': 'sum',
                'grantee_name': 'nunique'
            }).reset_index()
            state_funding.columns = ['State', 'Total Funding', 'Unique Grantees']
            state_funding = state_funding.sort_values('Total Funding', ascending=False)
            
            # Filter out empty states
            state_funding = state_funding[state_funding['State'].str.strip() != '']
            
            if not state_funding.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**By Total Funding:**")
                    top_states = state_funding.head(10).copy()
                    top_states['Total Funding'] = top_states['Total Funding'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(top_states, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Funding by State (Top 10):**")
                    chart_states = state_funding.head(10).set_index('State')['Total Funding']
                    st.bar_chart(chart_states)
            else:
                st.info("No state data available in grants.")
        else:
            st.info("State data not available in parsed grants.")
        
        st.divider()
        
        # --- Grant Size Analysis ---
        st.markdown("#### 📏 Grant Size Distribution")
        st.markdown("*What's the typical grant size?*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        amounts = grants_df['grant_amount']
        
        with col1:
            st.metric("Median Grant", f"${amounts.median():,.0f}")
        with col2:
            st.metric("Average Grant", f"${amounts.mean():,.0f}")
        with col3:
            st.metric("Smallest Grant", f"${amounts.min():,.0f}")
        with col4:
            st.metric("Largest Grant", f"${amounts.max():,.0f}")
        
        # Quartile breakdown
        st.markdown("**Grant Size Quartiles:**")
        q1, q2, q3 = amounts.quantile([0.25, 0.5, 0.75])
        st.caption(f"• 25th percentile: ${q1:,.0f}")
        st.caption(f"• 50th percentile (median): ${q2:,.0f}")
        st.caption(f"• 75th percentile: ${q3:,.0f}")
        
        st.divider()
        
        # --- Foundation Profiles ---
        st.markdown("#### 🏛️ Foundation Profiles")
        st.markdown("*Summary statistics for each foundation*")
        
        foundation_stats = grants_df.groupby('foundation_name').agg({
            'grant_amount': ['sum', 'mean', 'median', 'count'],
            'grantee_name': 'nunique',
            'grantee_state': 'nunique'
        }).reset_index()
        
        # Flatten column names
        foundation_stats.columns = [
            'Foundation', 'Total Grants ($)', 'Avg Grant ($)', 'Median Grant ($)', 
            'Grant Count', 'Unique Grantees', 'States Reached'
        ]
        
        # Add board member count
        if not people_df.empty:
            board_counts = people_df.groupby('foundation_name').size().reset_index()
            board_counts.columns = ['Foundation', 'Board Members']
            foundation_stats = foundation_stats.merge(board_counts, on='Foundation', how='left')
            foundation_stats['Board Members'] = foundation_stats['Board Members'].fillna(0).astype(int)
        
        # Format currency columns
        for col in ['Total Grants ($)', 'Avg Grant ($)', 'Median Grant ($)']:
            foundation_stats[col] = foundation_stats[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(foundation_stats, use_container_width=True, hide_index=True)
        
        st.divider()
    
    elif is_analytics_mode and grants_df.empty:
        st.info("📊 Analytics require grant data. Upload 990-PF files to see network insights.")
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
        # Show type breakdown
        type_counts = nodes_df['type'].value_counts()
        cols = st.columns(len(type_counts))
        for i, (node_type, count) in enumerate(type_counts.items()):
            cols[i].metric(f"{node_type.title()}s", count)
        
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
        # Show edge type breakdown
        edge_type_counts = edges_df['edge_type'].value_counts()
        cols = st.columns(len(edge_type_counts))
        for i, (edge_type, count) in enumerate(edge_type_counts.items()):
            label = edge_type.replace('_', ' ').title()
            cols[i].metric(f"{label} Edges", count)
        
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
    
    # Create zip file with all CSVs
    def create_zip_download():
        """Create a zip file containing all CSVs."""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if not grants_df.empty:
                zf.writestr('grants.csv', grants_df.to_csv(index=False))
            if not nodes_df.empty:
                zf.writestr('nodes.csv', nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zf.writestr('edges.csv', edges_df.to_csv(index=False))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # Download All button (prominent)
    has_data = not grants_df.empty or not nodes_df.empty or not edges_df.empty
    
    if has_data:
        st.download_button(
            label="📦 Download All (ZIP)",
            data=create_zip_download(),
            file_name="c4c_funder_flow_export.zip",
            mime="application/zip",
            help="Download all CSVs in a single ZIP file",
            type="primary",
            use_container_width=True
        )
        
        st.caption("Or download individual files:")
    
    # Individual file downloads
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grants_csv = grants_df.to_csv(index=False) if not grants_df.empty else ""
        st.download_button(
            label="⬇️ Download grants.csv",
            data=grants_csv,
            file_name="grants.csv",
            mime="text/csv",
            help="Raw grant table (one row per grant line item)",
            disabled=grants_df.empty
        )
    
    with col2:
        nodes_csv = nodes_df.to_csv(index=False) if not nodes_df.empty else ""
        st.download_button(
            label="⬇️ Download nodes.csv",
            data=nodes_csv,
            file_name="nodes.csv",
            mime="text/csv",
            help="Unified list of foundations, grantees, and people (Polinode-ready)",
            disabled=nodes_df.empty
        )
    
    with col3:
        edges_csv = edges_df.to_csv(index=False) if not edges_df.empty else ""
        st.download_button(
            label="⬇️ Download edges.csv",
            data=edges_csv,
            file_name="edges.csv",
            mime="text/csv",
            help="Unified list of grant and board membership edges (Polinode-ready)",
            disabled=edges_df.empty
        )


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption(
    "C4C 990 Funder Flow Prototype v0.2 | "
    "Part of the [Connecting for Change](https://connectingforchange.io) Network Intelligence Engine"
)
