"""
C4C Canadian Charity Ingestion — Streamlit App

Upload charitydata.ca exports and generate network-ready CSVs.
"""

import streamlit as st
import pandas as pd
import json
import re
import zipfile
from io import BytesIO, StringIO

# =============================================================================
# Config
# =============================================================================

C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_9e232fe9e6914305a7ea8746e2e77125~mv2.png"

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
Upload **charitydata.ca** exports for a Canadian foundation and generate:
- **Organization attributes** (name, CRA BN, total assets)
- **Board member nodes** and **membership edges**
- **Grant edges** to donee organizations

Grants are **automatically filtered** to the most recent reporting period for parity with US 990 data.
""")

st.divider()

# =============================================================================
# Parsing Functions
# =============================================================================

HEADER_RE = re.compile(r"^(.*)\s+\((\d{9}RR\d{4})\)\s*$")


def read_charitydata_csv(uploaded_file) -> tuple:
    """
    Read a charitydata.ca CSV file.
    
    Returns: (DataFrame, org_name, cra_bn)
    """
    content = uploaded_file.getvalue().decode("utf-8")
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
    """Lightweight slug for donee names."""
    text = (text or "").strip().lower()
    text = re.sub(r"&|\+", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def clean_nan(val) -> str:
    """Clean up 'nan' strings."""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


# =============================================================================
# File Upload
# =============================================================================

st.subheader("📁 Upload Files")

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

# Org slug input
st.text_input(
    "Organization slug (optional)",
    key="org_slug_input",
    placeholder="e.g., toronto-foundation",
    help="Used for node IDs. If blank, will be derived from org name."
)

st.divider()

# =============================================================================
# Process Files
# =============================================================================

if assets_file or directors_file or grants_file:
    
    # Initialize results
    org_name = ""
    cra_bn = ""
    org_slug = st.session_state.get("org_slug_input", "")
    latest_year = None
    total_assets = None
    
    people_rows = []
    membership_rows = []
    grant_edges = []
    donee_nodes = []
    
    # -------------------------------------------------------------------------
    # Parse assets.csv
    # -------------------------------------------------------------------------
    if assets_file:
        assets_df, org_name, cra_bn = read_charitydata_csv(assets_file)
        latest_year, total_assets = extract_total_assets(assets_df)
        st.success(f"✅ **assets.csv** — Loaded")
    
    # -------------------------------------------------------------------------
    # Parse directors-trustees.csv
    # -------------------------------------------------------------------------
    if directors_file:
        directors_df, dir_org_name, dir_cra_bn = read_charitydata_csv(directors_file)
        
        # Use org info from directors if not from assets
        if not org_name and dir_org_name:
            org_name = dir_org_name
        if not cra_bn and dir_cra_bn:
            cra_bn = dir_cra_bn
        
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
                
                # Derive org_slug if not provided
                if not org_slug and org_name:
                    org_slug = slugify_loose(org_name)
                
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
        
        st.success(f"✅ **directors-trustees.csv** — {len(people_rows)} board members")
    
    # -------------------------------------------------------------------------
    # Parse grants_recent.csv
    # -------------------------------------------------------------------------
    if grants_file:
        grants_df, grants_org_name, grants_cra_bn = read_charitydata_csv(grants_file)
        
        # Use org info from grants if not set
        if not org_name and grants_org_name:
            org_name = grants_org_name
        if not cra_bn and grants_cra_bn:
            cra_bn = grants_cra_bn
        
        if not grants_df.empty:
            # Filter to most recent reporting period only
            total_rows = len(grants_df)
            
            if "Reporting Period" in grants_df.columns:
                # Get unique periods and find the most recent one
                periods = grants_df["Reporting Period"].dropna().unique()
                if len(periods) > 0:
                    # Sort periods (format is typically YYYY-MM-DD or similar)
                    latest_period = sorted(periods, reverse=True)[0]
                    grants_df = grants_df[grants_df["Reporting Period"] == latest_period]
                    filtered_rows = len(grants_df)
                    st.info(f"📅 Filtered to most recent period: **{latest_period}** ({filtered_rows} of {total_rows} grants)")
            
            for _, r in grants_df.iterrows():
                donee = clean_nan(r.get("Donee Name", ""))
                if not donee:
                    continue
                
                city = clean_nan(r.get("City", ""))
                prov = clean_nan(r.get("Prov", ""))
                period = r.get("Reporting Period", "")
                amt = r.get("Reported Amount ($)", 0)
                gik = r.get("Gifts In Kind ($)", 0)
                
                # Derive org_slug if not provided
                if not org_slug and org_name:
                    org_slug = slugify_loose(org_name)
                
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
        
        st.success(f"✅ **grants** — {len(grant_edges)} grants loaded")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    st.subheader("📊 Summary")
    
    # Derive org_slug if still empty
    if not org_slug and org_name:
        org_slug = slugify_loose(org_name)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Organization", org_name or "Unknown")
    with col2:
        st.metric("CRA BN", cra_bn or "—")
    with col3:
        if total_assets:
            st.metric(f"Total Assets ({latest_year})", f"${total_assets:,.0f}")
        else:
            st.metric("Total Assets", "—")
    with col4:
        st.metric("Org Slug", org_slug or "—")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Board Members", len(people_rows))
    with col2:
        st.metric("Grants", len(grant_edges))
    with col3:
        unique_donees = len(pd.DataFrame(donee_nodes).drop_duplicates()) if donee_nodes else 0
        st.metric("Unique Donees", unique_donees)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Build Output DataFrames
    # -------------------------------------------------------------------------
    
    # Org attributes
    org_attributes = {
        "org_slug": org_slug,
        "jurisdiction": "CA",
        "data_source": "charitydata.ca",
        "legal_name": org_name,
        "cra_bn": cra_bn,
        "total_assets_latest_year": latest_year,
        "total_assets_latest_value": total_assets,
    }
    
    # DataFrames
    nodes_people_df = pd.DataFrame(people_rows).drop_duplicates() if people_rows else pd.DataFrame()
    nodes_donees_df = pd.DataFrame(donee_nodes).drop_duplicates() if donee_nodes else pd.DataFrame()
    edges_board_df = pd.DataFrame(membership_rows) if membership_rows else pd.DataFrame()
    edges_grants_df = pd.DataFrame(grant_edges) if grant_edges else pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # Data Previews
    # -------------------------------------------------------------------------
    st.subheader("📋 Data Preview")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Board Members", "Donees", "Board Edges", "Grant Edges"])
    
    with tab1:
        if not nodes_people_df.empty:
            st.dataframe(nodes_people_df, use_container_width=True, hide_index=True)
        else:
            st.info("No board members found.")
    
    with tab2:
        if not nodes_donees_df.empty:
            st.dataframe(nodes_donees_df, use_container_width=True, hide_index=True)
        else:
            st.info("No donees found.")
    
    with tab3:
        if not edges_board_df.empty:
            st.dataframe(edges_board_df, use_container_width=True, hide_index=True)
        else:
            st.info("No board membership edges.")
    
    with tab4:
        if not edges_grants_df.empty:
            st.dataframe(edges_grants_df, use_container_width=True, hide_index=True)
        else:
            st.info("No grant edges.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Downloads
    # -------------------------------------------------------------------------
    st.subheader("📥 Download Outputs")
    
    def create_zip_download():
        """Create a zip file with all outputs."""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Org attributes JSON
            zf.writestr("org_attributes.json", json.dumps(org_attributes, indent=2, default=str))
            
            # CSVs
            if not nodes_people_df.empty:
                zf.writestr("nodes_people.csv", nodes_people_df.to_csv(index=False))
            if not nodes_donees_df.empty:
                zf.writestr("nodes_donees.csv", nodes_donees_df.to_csv(index=False))
            if not edges_board_df.empty:
                zf.writestr("edges_board_membership.csv", edges_board_df.to_csv(index=False))
            if not edges_grants_df.empty:
                zf.writestr("edges_grants.csv", edges_grants_df.to_csv(index=False))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # Download All button
    st.download_button(
        label="📦 Download All (ZIP)",
        data=create_zip_download(),
        file_name=f"ca_charity_{org_slug or 'export'}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
    # Individual downloads
    st.markdown("**Or download individually:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.download_button(
            label="org_attributes.json",
            data=json.dumps(org_attributes, indent=2, default=str),
            file_name="org_attributes.json",
            mime="application/json"
        )
    
    with col2:
        if not nodes_people_df.empty:
            st.download_button(
                label="nodes_people.csv",
                data=nodes_people_df.to_csv(index=False),
                file_name="nodes_people.csv",
                mime="text/csv"
            )
    
    with col3:
        if not nodes_donees_df.empty:
            st.download_button(
                label="nodes_donees.csv",
                data=nodes_donees_df.to_csv(index=False),
                file_name="nodes_donees.csv",
                mime="text/csv"
            )
    
    with col4:
        if not edges_board_df.empty:
            st.download_button(
                label="edges_board.csv",
                data=edges_board_df.to_csv(index=False),
                file_name="edges_board_membership.csv",
                mime="text/csv"
            )
    
    with col5:
        if not edges_grants_df.empty:
            st.download_button(
                label="edges_grants.csv",
                data=edges_grants_df.to_csv(index=False),
                file_name="edges_grants.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Upload at least one CSV file to get started.")
