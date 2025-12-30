"""
C4C Project Browser — Cloud Project Management

Standalone Streamlit app for browsing and managing cloud projects
saved from OrgGraph, ActorGraph, and InsightGraph.

Features:
- Browse all cloud projects with metadata
- Filter by source app, date range, search
- Preview nodes/edges without downloading
- Download bundles
- Manage: rename, delete projects

VERSION HISTORY:
----------------
v0.1.0: Initial release
- Project listing with metadata (app, date, node/edge counts)
- Filter by source app
- Search by project name
- Preview data tables
- Download bundle
- Delete projects

"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from io import BytesIO
import zipfile
import json

# =============================================================================
# Constants
# =============================================================================
APP_VERSION = "0.1.0"
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_25063966d6cd496eb2fe3f6ee5cde0fa~mv2.png"
APP_ICON_URL = "https://static.wixstatic.com/media/275a3f_25063966d6cd496eb2fe3f6ee5cde0fa~mv2.png"

# Source app display names
SOURCE_APPS = {
    "all": "All Apps",
    "orggraph_us": "OrgGraph US",
    "orggraph_ca": "OrgGraph CA",
    "actorgraph": "ActorGraph",
    "insightgraph": "InsightGraph",
}

# App icons for display
APP_ICONS = {
    "orggraph_us": "🏛️",
    "orggraph_ca": "🍁",
    "actorgraph": "🔗",
    "insightgraph": "📊",
}

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="C4C Project Browser",
    page_icon=APP_ICON_URL,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Session State
# =============================================================================
def init_session_state():
    """Initialize session state."""
    if "project_store" not in st.session_state:
        st.session_state.project_store = None
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = None


def init_project_store():
    """Initialize Project Store client."""
    if st.session_state.project_store is not None:
        return st.session_state.project_store
    
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        
        from c4c_utils.c4c_project_store import ProjectStoreClient
        
        client = ProjectStoreClient(url, key)
        st.session_state.project_store = client
        return client
    except Exception as e:
        st.session_state.project_store = None
        return None


def get_authenticated_client():
    """Get authenticated client or None."""
    client = init_project_store()
    if client and client.is_authenticated():
        return client
    return None


# =============================================================================
# Authentication UI
# =============================================================================
def render_auth_sidebar():
    """Render authentication UI in sidebar."""
    client = init_project_store()
    
    if not client:
        st.sidebar.error("☁️ Cloud unavailable")
        st.sidebar.caption("Check Supabase configuration")
        return None
    
    if client.is_authenticated():
        user = client.get_current_user()
        st.sidebar.success(f"☁️ {user['email']}")
        
        if st.sidebar.button("Logout", use_container_width=True):
            client.logout()
            st.rerun()
        
        return client
    else:
        st.sidebar.warning("☁️ Not logged in")
        
        with st.sidebar.expander("🔐 Login / Sign Up", expanded=True):
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_pass")
                
                if st.button("Login", key="login_btn", use_container_width=True):
                    if email and password:
                        success, error = client.login(email, password)
                        if success:
                            st.success("✅ Logged in!")
                            st.rerun()
                        else:
                            st.error(f"Login failed: {error}")
                    else:
                        st.warning("Enter email and password")
            
            with tab2:
                st.caption("First time? Create an account.")
                signup_email = st.text_input("Email", key="signup_email")
                signup_pass = st.text_input("Password", type="password", key="signup_pass")
                
                if st.button("Sign Up", key="signup_btn", use_container_width=True):
                    if signup_email and signup_pass:
                        success, error = client.signup(signup_email, signup_pass)
                        if success:
                            st.success("✅ Check email to confirm")
                        else:
                            st.error(f"Signup failed: {error}")
                    else:
                        st.warning("Enter email and password")
        
        return None


# =============================================================================
# Project Listing
# =============================================================================
def load_projects(client, source_app_filter: str = None):
    """Load projects from cloud."""
    try:
        if source_app_filter and source_app_filter != "all":
            projects, error = client.list_projects(source_app=source_app_filter, include_public=True)
        else:
            projects, error = client.list_projects(include_public=True)
        
        if error:
            st.error(f"Failed to load projects: {error}")
            return []
        
        return projects or []
    except Exception as e:
        st.error(f"Error loading projects: {e}")
        return []


def format_date(date_str: str) -> str:
    """Format ISO date string for display."""
    if not date_str:
        return "—"
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y %H:%M")
    except:
        return date_str[:16] if len(date_str) > 16 else date_str


def render_project_table(projects: list, search_query: str = ""):
    """Render project table with selection."""
    if not projects:
        st.info("No projects found. Save a project from OrgGraph, ActorGraph, or InsightGraph to see it here.")
        return None
    
    # Filter by search
    if search_query:
        search_lower = search_query.lower()
        projects = [p for p in projects if search_lower in (getattr(p, 'name', '') or '').lower()]
    
    if not projects:
        st.warning(f"No projects matching '{search_query}'")
        return None
    
    # Build table data
    table_data = []
    for p in projects:
        source_app = getattr(p, 'source_app', '') or 'unknown'
        icon = APP_ICONS.get(source_app, "📁")
        app_label = SOURCE_APPS.get(source_app, source_app)
        
        table_data.append({
            "": icon,
            "Project": getattr(p, 'name', getattr(p, 'slug', '—')),
            "App": app_label,
            "Nodes": getattr(p, 'node_count', 0) or 0,
            "Edges": getattr(p, 'edge_count', 0) or 0,
            "Updated": format_date(getattr(p, 'updated_at', None)),
            "_project": p,  # Hidden: full project object
        })
    
    # Sort by updated date (newest first)
    table_data.sort(key=lambda x: getattr(x['_project'], 'updated_at', '') or '', reverse=True)
    
    # Display as interactive table
    st.markdown(f"**{len(table_data)} project(s)**")
    
    # Create selection
    selected_idx = None
    
    for i, row in enumerate(table_data):
        col1, col2, col3, col4, col5, col6, col7 = st.columns([0.5, 3, 1.5, 1, 1, 2, 1])
        
        with col1:
            st.write(row[""])
        with col2:
            st.write(row["Project"])
        with col3:
            st.caption(row["App"])
        with col4:
            st.caption(f"{row['Nodes']:,}")
        with col5:
            st.caption(f"{row['Edges']:,}")
        with col6:
            st.caption(row["Updated"])
        with col7:
            if st.button("View", key=f"view_{i}", use_container_width=True):
                st.session_state.selected_project = row["_project"]
                st.rerun()
    
    return None


# =============================================================================
# Project Detail View
# =============================================================================
def render_project_detail(client, project):
    """Render detailed view of a single project."""
    
    # Header with back button
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back"):
            st.session_state.selected_project = None
            st.rerun()
    with col2:
        source_app = getattr(project, 'source_app', '') or 'unknown'
        icon = APP_ICONS.get(source_app, "📁")
        st.subheader(f"{icon} {getattr(project, 'name', 'Unnamed Project')}")
    
    # Metadata
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", f"{getattr(project, 'node_count', 0) or 0:,}")
    with col2:
        st.metric("Edges", f"{getattr(project, 'edge_count', 0) or 0:,}")
    with col3:
        app_label = SOURCE_APPS.get(source_app, source_app)
        st.metric("Source App", app_label)
    with col4:
        jurisdiction = getattr(project, 'jurisdiction', '—') or '—'
        st.metric("Jurisdiction", jurisdiction)
    
    # More metadata
    with st.expander("📋 Project Details", expanded=False):
        details = {
            "Slug": getattr(project, 'slug', '—'),
            "Created": format_date(getattr(project, 'created_at', None)),
            "Updated": format_date(getattr(project, 'updated_at', None)),
            "App Version": getattr(project, 'app_version', '—') or '—',
            "Schema Version": getattr(project, 'schema_version', '—') or '—',
            "Region Preset": getattr(project, 'region_preset', '—') or '—',
        }
        
        for key, value in details.items():
            st.caption(f"**{key}:** {value}")
    
    st.divider()
    
    # Actions
    st.markdown("### 📥 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    # Download bundle
    with col1:
        if st.button("📦 Download Bundle", use_container_width=True, type="primary"):
            with st.spinner("Downloading..."):
                try:
                    slug = getattr(project, 'slug', '')
                    project_id = getattr(project, 'id', None)
                    
                    bundle_data, error = client.load_project(project_id=project_id, slug=slug)
                    if error:
                        st.error(f"Download failed: {error}")
                    elif bundle_data:
                        st.download_button(
                            "💾 Save ZIP",
                            data=bundle_data,
                            file_name=f"{slug or 'project'}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    else:
                        st.warning("No bundle data found")
                except Exception as e:
                    st.error(f"Download error: {e}")
    
    # Preview data
    with col2:
        preview_clicked = st.button("👁️ Preview Data", use_container_width=True)
    
    # Delete
    with col3:
        delete_clicked = st.button("🗑️ Delete", use_container_width=True)
    
    # Preview section
    if preview_clicked:
        st.divider()
        render_data_preview(client, project)
    
    # Delete confirmation
    if delete_clicked:
        st.divider()
        st.warning("⚠️ **Delete this project?** This cannot be undone.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❌ Yes, Delete", use_container_width=True, type="primary"):
                try:
                    project_id = getattr(project, 'id', None)
                    success, error = client.delete_project(project_id=project_id)
                    if success:
                        st.success("✅ Project deleted")
                        st.session_state.selected_project = None
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {error}")
                except Exception as e:
                    st.error(f"Delete error: {e}")
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


def render_data_preview(client, project):
    """Preview nodes/edges data from a project."""
    st.markdown("### 👁️ Data Preview")
    
    try:
        # Download and extract bundle
        slug = getattr(project, 'slug', '')
        project_id = getattr(project, 'id', None)
        
        bundle_data, error = client.load_project(project_id=project_id, slug=slug)
        
        if error:
            st.error(f"Failed to load data: {error}")
            return
        
        if not bundle_data:
            st.warning("No bundle data found")
            return
        
        # Extract CSVs from ZIP
        with zipfile.ZipFile(BytesIO(bundle_data), 'r') as zip_file:
            file_list = zip_file.namelist()
            
            # Find nodes and edges files
            nodes_file = None
            edges_file = None
            grants_file = None
            board_file = None
            
            for f in file_list:
                f_lower = f.lower()
                if 'nodes' in f_lower and f_lower.endswith('.csv') and 'polinode' not in f_lower:
                    nodes_file = f
                elif 'edges' in f_lower and f_lower.endswith('.csv') and 'polinode' not in f_lower:
                    edges_file = f
                elif 'grants_detail' in f_lower and f_lower.endswith('.csv'):
                    grants_file = f
                elif 'board_detail' in f_lower and f_lower.endswith('.csv'):
                    board_file = f
            
            # Display tabs for each file
            tabs = ["Nodes", "Edges"]
            if grants_file:
                tabs.append("Grants")
            if board_file:
                tabs.append("Board")
            tabs.append("Files")
            
            tab_objects = st.tabs(tabs)
            
            # Nodes tab
            with tab_objects[0]:
                if nodes_file:
                    with zip_file.open(nodes_file) as f:
                        nodes_df = pd.read_csv(f)
                    st.caption(f"**{len(nodes_df):,} nodes** from `{nodes_file}`")
                    st.dataframe(nodes_df.head(100), use_container_width=True)
                    if len(nodes_df) > 100:
                        st.caption(f"Showing first 100 of {len(nodes_df):,} rows")
                else:
                    st.info("No nodes.csv found in bundle")
            
            # Edges tab
            with tab_objects[1]:
                if edges_file:
                    with zip_file.open(edges_file) as f:
                        edges_df = pd.read_csv(f)
                    st.caption(f"**{len(edges_df):,} edges** from `{edges_file}`")
                    st.dataframe(edges_df.head(100), use_container_width=True)
                    if len(edges_df) > 100:
                        st.caption(f"Showing first 100 of {len(edges_df):,} rows")
                else:
                    st.info("No edges.csv found in bundle")
            
            # Grants tab (if exists)
            tab_idx = 2
            if grants_file:
                with tab_objects[tab_idx]:
                    with zip_file.open(grants_file) as f:
                        grants_df = pd.read_csv(f)
                    st.caption(f"**{len(grants_df):,} grants** from `{grants_file}`")
                    st.dataframe(grants_df.head(100), use_container_width=True)
                    if len(grants_df) > 100:
                        st.caption(f"Showing first 100 of {len(grants_df):,} rows")
                tab_idx += 1
            
            # Board tab (if exists)
            if board_file:
                with tab_objects[tab_idx]:
                    with zip_file.open(board_file) as f:
                        board_df = pd.read_csv(f)
                    st.caption(f"**{len(board_df):,} board members** from `{board_file}`")
                    
                    # Show interlock summary if available
                    if 'interlock_count' in board_df.columns:
                        interlocked = board_df[board_df['interlock_count'] > 1]
                        if not interlocked.empty:
                            unique_people = interlocked['person_name_normalized'].nunique() if 'person_name_normalized' in interlocked.columns else len(interlocked)
                            st.info(f"🔗 **{unique_people} people** serve on multiple boards")
                    
                    st.dataframe(board_df.head(100), use_container_width=True)
                    if len(board_df) > 100:
                        st.caption(f"Showing first 100 of {len(board_df):,} rows")
                tab_idx += 1
            
            # Files tab
            with tab_objects[tab_idx]:
                st.caption(f"**{len(file_list)} files in bundle:**")
                for f in sorted(file_list):
                    info = zip_file.getinfo(f)
                    size_kb = info.file_size / 1024
                    st.caption(f"  `{f}` ({size_kb:.1f} KB)")
    
    except Exception as e:
        st.error(f"Preview error: {e}")


# =============================================================================
# Main App
# =============================================================================
def main():
    init_session_state()
    
    # Header
    st.title("📂 C4C Project Browser")
    st.caption(f"Manage cloud projects from OrgGraph, ActorGraph, and InsightGraph • v{APP_VERSION}")
    
    # Sidebar: Auth
    st.sidebar.title("☁️ Cloud")
    client = render_auth_sidebar()
    
    if not client:
        # Not logged in - show landing
        st.divider()
        st.markdown("""
        ### Welcome to the Project Browser
        
        This tool lets you browse and manage all your cloud-saved projects from:
        
        - 🏛️ **OrgGraph US** — US nonprofit network data
        - 🍁 **OrgGraph CA** — Canadian nonprofit network data
        - 🔗 **ActorGraph** — LinkedIn network data
        - 📊 **InsightGraph** — Network analysis reports
        
        **Login** in the sidebar to view your projects.
        """)
        return
    
    # Sidebar: Filters
    st.sidebar.divider()
    st.sidebar.markdown("### 🔍 Filters")
    
    source_app_filter = st.sidebar.selectbox(
        "Source App",
        options=list(SOURCE_APPS.keys()),
        format_func=lambda x: SOURCE_APPS[x],
        index=0
    )
    
    search_query = st.sidebar.text_input("🔎 Search projects", placeholder="Project name...")
    
    # Main content
    st.divider()
    
    if st.session_state.selected_project:
        # Detail view
        render_project_detail(client, st.session_state.selected_project)
    else:
        # List view
        st.markdown("### 📋 Your Projects")
        
        # Load projects
        projects = load_projects(client, source_app_filter)
        
        # Render table
        render_project_table(projects, search_query)
        
        # Stats footer
        if projects:
            st.divider()
            total_nodes = sum(getattr(p, 'node_count', 0) or 0 for p in projects)
            total_edges = sum(getattr(p, 'edge_count', 0) or 0 for p in projects)
            st.caption(f"**Total:** {len(projects)} projects • {total_nodes:,} nodes • {total_edges:,} edges")


if __name__ == "__main__":
    main()
