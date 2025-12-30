# =============================================================================
# CloudProjects — Cloud Project Management
#
# Standalone Streamlit app for browsing and managing cloud projects
# saved from OrgGraph, ActorGraph, and InsightGraph.
#
# VERSION HISTORY:
# v0.3.5: Force favicon via base64 injection (bypasses Streamlit/CDN/favicon caching)
# v0.3.4: Attempted local icon path resolution
# v0.3.2: Use PIL Image object for icon (handles format conversion)
# v0.3.1: Icon at very top per technical advisory
# v0.3.0: Fixed icon using PIL Image
# v0.2.9: Fixed icon file path resolution
# v0.2.4: Fixed storage bucket name
# v0.2.1: Renamed to CloudProjects
# v0.2.0: Self-contained version
# v0.1.0: Initial release
# =============================================================================

# =============================================================================
# PAGE CONFIG — MUST BE FIRST (before any other st.* calls)
# =============================================================================
from pathlib import Path
import base64

import streamlit as st
import streamlit.components.v1 as components

SCRIPT_DIR = Path(__file__).parent
ICON_PATH = SCRIPT_DIR / "cloudprojects_icon.png"

st.set_page_config(
    page_title="CloudProjects",
    page_icon=str(ICON_PATH),  # keep this for Streamlit-native path if it works
    layout="wide",
    initial_sidebar_state="expanded",
)

def _force_favicon(png_path: Path) -> None:
    """
    Bypass Streamlit/favicon caching by injecting a data-URL favicon.
    If this runs, the browser should show the icon even if Streamlit/CDN caches
    the default favicon.
    """
    try:
        b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
        components.html(
            f"""
            <script>
              const link = document.querySelector("link[rel~='icon']") || document.createElement('link');
              link.rel = 'icon';
              link.href = "data:image/png;base64,{b64}";
              document.head.appendChild(link);
            </script>
            """,
            height=0,
            width=0,
        )
    except Exception:
        # Fail silently; app should still run even if favicon injection fails.
        pass

_force_favicon(ICON_PATH)

# =============================================================================
# REST OF IMPORTS
# =============================================================================
import pandas as pd
from datetime import datetime, timezone
from io import BytesIO
import zipfile
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

# =============================================================================
# Constants
# =============================================================================
APP_VERSION = "0.3.5"

# Logo/icon files should be in same directory as this script
C4C_LOGO_FILE = SCRIPT_DIR / "c4c_logo.png"

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
# Embedded Project Store Client
# =============================================================================
@dataclass
class Project:
    """Project metadata from Supabase."""
    id: str
    slug: str
    name: str
    source_app: str
    node_count: int
    edge_count: int
    jurisdiction: Optional[str]
    region_preset: Optional[str]
    app_version: Optional[str]
    schema_version: Optional[str]
    bundle_version: Optional[str]
    is_public: bool
    user_id: str
    created_at: str
    updated_at: str
    bundle_path: Optional[str] = None


class EmbeddedProjectStoreClient:
    """
    Lightweight Supabase client for project management.
    Embedded version - no external dependencies beyond supabase-py.
    """

    def __init__(self, url: str, key: str):
        """Initialize with Supabase credentials."""
        from supabase import create_client

        self.url = url
        self.key = key
        self.client = create_client(url, key)
        self._user = None

    def is_authenticated(self) -> bool:
        """Check if user is logged in."""
        try:
            session = self.client.auth.get_session()
            if session and session.user:
                self._user = session.user
                return True
            return False
        except Exception:
            return False

    def get_current_user(self) -> Optional[dict]:
        """Get current user info."""
        if self._user:
            return {"email": self._user.email, "id": self._user.id}
        try:
            session = self.client.auth.get_session()
            if session and session.user:
                return {"email": session.user.email, "id": session.user.id}
        except Exception:
            pass
        return None

    def login(self, email: str, password: str) -> tuple:
        """Login with email/password."""
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if response.user:
                self._user = response.user
                return True, None
            return False, "Login failed"
        except Exception as e:
            return False, str(e)

    def logout(self):
        """Logout current user."""
        try:
            self.client.auth.sign_out()
            self._user = None
        except Exception:
            pass

    def signup(self, email: str, password: str) -> tuple:
        """Create new account."""
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            if response.user:
                return True, None
            return False, "Signup failed"
        except Exception as e:
            return False, str(e)

    def list_projects(self, source_app: str = None, include_public: bool = True) -> tuple:
        """List projects accessible to current user."""
        try:
            user = self.get_current_user()
            if not user:
                return [], "Not authenticated"

            # Build query
            query = self.client.table('projects').select('*')

            # Filter by source app if specified
            if source_app:
                query = query.eq('source_app', source_app)

            # Get user's own projects + public projects
            if include_public:
                query = query.or_(f"user_id.eq.{user['id']},is_public.eq.true")
            else:
                query = query.eq('user_id', user['id'])

            # Order by updated_at desc
            query = query.order('updated_at', desc=True)

            response = query.execute()

            projects = []
            for row in response.data:
                projects.append(Project(
                    id=row.get('id'),
                    slug=row.get('slug'),
                    name=row.get('name', row.get('slug', 'Unnamed')),
                    source_app=row.get('source_app'),
                    node_count=row.get('node_count', 0),
                    edge_count=row.get('edge_count', 0),
                    jurisdiction=row.get('jurisdiction'),
                    region_preset=row.get('region_preset'),
                    app_version=row.get('app_version'),
                    schema_version=row.get('schema_version'),
                    bundle_version=row.get('bundle_version'),
                    is_public=row.get('is_public', False),
                    user_id=row.get('user_id'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at'),
                    bundle_path=row.get('bundle_path'),
                ))

            return projects, None

        except Exception as e:
            return [], str(e)

    def load_project(self, project_id: str = None, slug: str = None) -> tuple:
        """Download project bundle from storage."""
        try:
            # First get project metadata to find bundle path
            if project_id:
                response = self.client.table('projects').select('*').eq('id', project_id).single().execute()
            elif slug:
                response = self.client.table('projects').select('*').eq('slug', slug).single().execute()
            else:
                return None, "Must provide project_id or slug"

            if not response.data:
                return None, "Project not found"

            bundle_path = response.data.get('bundle_path')
            if not bundle_path:
                return None, "No bundle path found"

            # Try bucket name from secrets, then the correct name
            bucket_names = []
            try:
                # Check if bucket name is in secrets
                bucket_names.append(st.secrets["supabase"].get("bucket", None))
            except Exception:
                pass

            # The actual bucket name (underscore, not hyphen)
            bucket_names.append("project_bundles")
            bucket_names = [b for b in bucket_names if b]  # Remove None values

            # Try each bucket name
            last_error = None
            for bucket_name in bucket_names:
                try:
                    data = self.client.storage.from_(bucket_name).download(bundle_path)
                    return data, None
                except Exception as e:
                    last_error = str(e)
                    continue

            return None, f"Could not download from any bucket. Last error: {last_error}. Tried: {bucket_names}"

        except Exception as e:
            return None, str(e)

    def delete_project(self, project_id: str = None, slug: str = None) -> tuple:
        """Delete a project and its bundle."""
        try:
            user = self.get_current_user()
            if not user:
                return False, "Not authenticated"

            # Get project first
            if project_id:
                response = self.client.table('projects').select('*').eq('id', project_id).single().execute()
            elif slug:
                response = self.client.table('projects').select('*').eq('slug', slug).single().execute()
            else:
                return False, "Must provide project_id or slug"

            if not response.data:
                return False, "Project not found"

            project = response.data

            # Check ownership
            if project.get('user_id') != user['id']:
                return False, "Permission denied"

            # Delete bundle from storage if exists
            bundle_path = project.get('bundle_path')
            if bundle_path:
                # Use correct bucket name (underscore)
                bucket_names = ["project_bundles"]
                try:
                    bucket_names.insert(0, st.secrets["supabase"].get("bucket"))
                except Exception:
                    pass
                bucket_names = [b for b in bucket_names if b]

                for bucket_name in bucket_names:
                    try:
                        self.client.storage.from_(bucket_name).remove([bundle_path])
                        break
                    except Exception:
                        continue

            # Delete from database
            self.client.table('projects').delete().eq('id', project['id']).execute()

            return True, None

        except Exception as e:
            return False, str(e)

    def get_project(self, project_id: str = None, slug: str = None) -> tuple:
        """Get project metadata."""
        try:
            if project_id:
                response = self.client.table('projects').select('*').eq('id', project_id).single().execute()
            elif slug:
                response = self.client.table('projects').select('*').eq('slug', slug).single().execute()
            else:
                return None, "Must provide project_id or slug"

            if not response.data:
                return None, "Project not found"

            row = response.data
            project = Project(
                id=row.get('id'),
                slug=row.get('slug'),
                name=row.get('name', row.get('slug', 'Unnamed')),
                source_app=row.get('source_app'),
                node_count=row.get('node_count', 0),
                edge_count=row.get('edge_count', 0),
                jurisdiction=row.get('jurisdiction'),
                region_preset=row.get('region_preset'),
                app_version=row.get('app_version'),
                schema_version=row.get('schema_version'),
                bundle_version=row.get('bundle_version'),
                is_public=row.get('is_public', False),
                user_id=row.get('user_id'),
                created_at=row.get('created_at'),
                updated_at=row.get('updated_at'),
                bundle_path=row.get('bundle_path'),
            )
            return project, None

        except Exception as e:
            return None, str(e)


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

    # Check if secrets exist
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except KeyError as e:
        st.sidebar.error(f"❌ Missing secret: {e}")
        st.sidebar.caption("Add [supabase] section to secrets.toml")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Secrets error: {e}")
        return None

    # Try to initialize embedded client
    try:
        client = EmbeddedProjectStoreClient(url, key)
        st.session_state.project_store = client
        return client
    except ImportError as e:
        st.sidebar.error(f"❌ Import error: {e}")
        st.sidebar.caption("pip install supabase")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Client init error: {e}")
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
        # Error messages already shown by init_project_store
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
    except Exception:
        return date_str[:16] if len(date_str) > 16 else date_str


def render_project_table(projects: list, search_query: str = ""):
    """Render project table with selection."""
    if not projects:
        st.info("No projects found. Save a project from OrgGraph, ActorGraph, or InsightGraph to see it here.")
        return None

    # Filter by search
    if search_query:
        search_lower = search_query.lower()
        projects = [p for p in projects if search_lower in (p.name or '').lower()]

    if not projects:
        st.warning(f"No projects matching '{search_query}'")
        return None

    # Build table data
    table_data = []
    for p in projects:
        source_app = p.source_app or 'unknown'
        icon = APP_ICONS.get(source_app, "📁")
        app_label = SOURCE_APPS.get(source_app, source_app)

        table_data.append({
            "icon": icon,
            "name": p.name or p.slug or '—',
            "app": app_label,
            "nodes": p.node_count or 0,
            "edges": p.edge_count or 0,
            "updated": format_date(p.updated_at),
            "project": p,
        })

    # Sort by updated date (newest first)
    table_data.sort(key=lambda x: x['project'].updated_at or '', reverse=True)

    # Display as interactive table
    st.markdown(f"**{len(table_data)} project(s)**")

    # Header row
    col1, col2, col3, col4, col5, col6, col7 = st.columns([0.5, 3, 1.5, 1, 1, 2, 1])
    with col2:
        st.caption("**Project**")
    with col3:
        st.caption("**App**")
    with col4:
        st.caption("**Nodes**")
    with col5:
        st.caption("**Edges**")
    with col6:
        st.caption("**Updated**")

    st.divider()

    for i, row in enumerate(table_data):
        col1, col2, col3, col4, col5, col6, col7 = st.columns([0.5, 3, 1.5, 1, 1, 2, 1])

        with col1:
            st.write(row["icon"])
        with col2:
            st.write(row["name"])
        with col3:
            st.caption(row["app"])
        with col4:
            st.caption(f"{row['nodes']:,}")
        with col5:
            st.caption(f"{row['edges']:,}")
        with col6:
            st.caption(row["updated"])
        with col7:
            if st.button("View", key=f"view_{i}", use_container_width=True):
                st.session_state.selected_project = row["project"]
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
        source_app = project.source_app or 'unknown'
        icon = APP_ICONS.get(source_app, "📁")
        st.subheader(f"{icon} {project.name or 'Unnamed Project'}")

    # Metadata
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nodes", f"{project.node_count or 0:,}")
    with col2:
        st.metric("Edges", f"{project.edge_count or 0:,}")
    with col3:
        app_label = SOURCE_APPS.get(source_app, source_app)
        st.metric("Source App", app_label)
    with col4:
        jurisdiction = project.jurisdiction or '—'
        st.metric("Jurisdiction", jurisdiction)

    # More metadata
    with st.expander("📋 Project Details", expanded=False):
        details = {
            "Slug": project.slug or '—',
            "Created": format_date(project.created_at),
            "Updated": format_date(project.updated_at),
            "App Version": project.app_version or '—',
            "Schema Version": project.schema_version or '—',
            "Region Preset": project.region_preset or '—',
            "Public": "Yes" if project.is_public else "No",
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
                    bundle_data, error = client.load_project(project_id=project.id, slug=project.slug)
                    if error:
                        st.error(f"Download failed: {error}")
                    elif bundle_data:
                        st.download_button(
                            "💾 Save ZIP",
                            data=bundle_data,
                            file_name=f"{project.slug or 'project'}.zip",
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
                    success, error = client.delete_project(project_id=project.id)
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
        bundle_data, error = client.load_project(project_id=project.id, slug=project.slug)

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
    st.title("☁️ CloudProjects")
    st.caption(f"Manage cloud projects from OrgGraph, ActorGraph, and InsightGraph • v{APP_VERSION}")

    # Sidebar: Auth
    st.sidebar.title("☁️ Cloud")
    client = render_auth_sidebar()

    if not client:
        # Not logged in - show landing
        st.divider()
        st.markdown("""
        ### Welcome to CloudProjects

        Browse and manage all your cloud-saved projects from:

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
            total_nodes = sum(p.node_count or 0 for p in projects)
            total_edges = sum(p.edge_count or 0 for p in projects)
            st.caption(f"**Total:** {len(projects)} projects • {total_nodes:,} nodes • {total_edges:,} edges")


if __name__ == "__main__":
    main()
