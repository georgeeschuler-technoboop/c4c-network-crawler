# =============================================================================
# CloudProjects — Cloud Project Management
#
# Standalone Streamlit app for browsing and managing cloud projects
# saved from OrgGraph, ActorGraph, and InsightGraph.
#
# VERSION HISTORY:
# v0.5.0: Add Docs & Artifacts tab (manifest-driven)
# v0.4.2: Add explicit favicon debug panel + show injection errors when ?debug_favicon=1
# v0.4.1: Favicon debug + fallback logic (local file + data-URL injection with optional diagnostics)
# v0.4.0: Stable release with data URL favicon injection
# v0.3.6: Inject favicon via st.markdown (<link rel="icon"> data URL) to bypass Streamlit/CDN caching
# v0.3.5: Attempted favicon via components.html() (iframe-limited; ineffective)
# v0.3.4: Attempted local icon path resolution
# v0.3.x: Various favicon fix attempts (PIL, path resolution, etc.)
# v0.2.4: Fixed storage bucket name (project_bundles)
# v0.2.1: Renamed to CloudProjects
# v0.2.0: Self-contained version with embedded Supabase client
# v0.1.0: Initial release
# =============================================================================

# =============================================================================
# PAGE CONFIG — MUST BE FIRST (before any other st.* calls)
# =============================================================================
import base64
from pathlib import Path
import streamlit as st

SCRIPT_DIR = Path(__file__).parent
ICON_PATH = SCRIPT_DIR / "cloudprojects_icon.png"

# Optional diagnostics: add ?debug_favicon=1 to the URL to show favicon debug in sidebar.
try:
    DEBUG_FAVICON = str(st.query_params.get("debug_favicon", "0")).strip().lower() in ("1", "true", "yes", "on")
except Exception:
    DEBUG_FAVICON = False

st.set_page_config(
    page_title="CloudProjects",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _inject_data_url_favicon(png_path: Path):
    if not png_path.exists():
        raise FileNotFoundError(f"Icon file not found: {png_path}")
    b64 = base64.b64encode(png_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,{b64}">
        <link rel="icon" type="image/png" sizes="16x16" href="data:image/png;base64,{b64}">
        <link rel="shortcut icon" type="image/png" href="data:image/png;base64,{b64}">
        """,
        unsafe_allow_html=True,
    )

inject_ok = True
inject_msg = "Not attempted"
try:
    _inject_data_url_favicon(ICON_PATH)
    inject_ok = True
    inject_msg = "Injected data-URL favicon tags"
except Exception as e:
    inject_ok = False
    inject_msg = f"{type(e).__name__}: {e}"
    if DEBUG_FAVICON:
        st.sidebar.error(f"Favicon injection failed: {inject_msg}")

if DEBUG_FAVICON:
    st.sidebar.markdown("### 🧪 Favicon debug")
    st.sidebar.write("ICON_PATH:", str(ICON_PATH))
    st.sidebar.write("Exists:", ICON_PATH.exists())
    if ICON_PATH.exists():
        st.sidebar.write("Bytes:", ICON_PATH.stat().st_size)
    st.sidebar.write("Streamlit page_icon used:", str(ICON_PATH) if ICON_PATH.exists() else "☁️")
    st.sidebar.write("Injection status:", inject_ok)
    st.sidebar.write("Injection message:", inject_msg)

# =============================================================================
# REST OF IMPORTS
# =============================================================================
import pandas as pd
from datetime import datetime, timezone
from io import BytesIO
import zipfile
import json
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# =============================================================================
# Constants
# =============================================================================
APP_VERSION = "0.5.0"

# Manifest URL for docs and artifacts
MANIFEST_URL = "https://igbzclkhwnxnypjssdwz.supabase.co/storage/v1/object/public/c4c-artifacts/demo/_manifest.json"

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
    "seed_resolver": "🧹",
    "cloudprojects": "☁️",
}

# =============================================================================
# Manifest Fetching (for Docs & Artifacts)
# =============================================================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_manifest() -> Optional[Dict[str, Any]]:
    """Fetch the manifest from Supabase. Returns None on error."""
    try:
        response = requests.get(MANIFEST_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load manifest: {e}")
        return None


def get_file_icon(file_type: str) -> str:
    """Return an emoji icon based on file type."""
    icons = {
        "md": "📝",
        "pdf": "📄",
        "zip": "📦",
        "xlsx": "📊",
        "html": "🌐",
        "csv": "📋",
    }
    return icons.get(file_type.lower(), "📁")


def get_file_type_from_url(url: str) -> str:
    """Extract file type from URL."""
    if not url:
        return ""
    url_lower = url.lower()
    for ext in ["pdf", "zip", "xlsx", "html", "md", "csv"]:
        if url_lower.endswith(f".{ext}"):
            return ext
    return ""


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
        from supabase import create_client
        self.url = url
        self.key = key
        self.client = create_client(url, key)
        self._user = None

    def is_authenticated(self) -> bool:
        try:
            session = self.client.auth.get_session()
            if session and session.user:
                self._user = session.user
                return True
            return False
        except Exception:
            return False

    def get_current_user(self) -> Optional[dict]:
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
        try:
            response = self.client.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                self._user = response.user
                return True, None
            return False, "Login failed"
        except Exception as e:
            return False, str(e)

    def logout(self):
        try:
            self.client.auth.sign_out()
            self._user = None
        except Exception:
            pass

    def signup(self, email: str, password: str) -> tuple:
        try:
            response = self.client.auth.sign_up({"email": email, "password": password})
            if response.user:
                return True, None
            return False, "Signup failed"
        except Exception as e:
            return False, str(e)

    def list_projects(self, source_app: str = None, include_public: bool = True) -> tuple:
        try:
            user = self.get_current_user()
            if not user:
                return [], "Not authenticated"

            query = self.client.table("projects").select("*")

            if source_app:
                query = query.eq("source_app", source_app)

            if include_public:
                query = query.or_(f"user_id.eq.{user['id']},is_public.eq.true")
            else:
                query = query.eq("user_id", user["id"])

            query = query.order("updated_at", desc=True)
            response = query.execute()

            projects = []
            for row in response.data:
                projects.append(Project(
                    id=row.get("id"),
                    slug=row.get("slug"),
                    name=row.get("name", row.get("slug", "Unnamed")),
                    source_app=row.get("source_app"),
                    node_count=row.get("node_count", 0),
                    edge_count=row.get("edge_count", 0),
                    jurisdiction=row.get("jurisdiction"),
                    region_preset=row.get("region_preset"),
                    app_version=row.get("app_version"),
                    schema_version=row.get("schema_version"),
                    bundle_version=row.get("bundle_version"),
                    is_public=row.get("is_public", False),
                    user_id=row.get("user_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at"),
                    bundle_path=row.get("bundle_path"),
                ))

            return projects, None
        except Exception as e:
            return [], str(e)

    def load_project(self, project_id: str = None, slug: str = None) -> tuple:
        try:
            if project_id:
                response = self.client.table("projects").select("*").eq("id", project_id).single().execute()
            elif slug:
                response = self.client.table("projects").select("*").eq("slug", slug).single().execute()
            else:
                return None, "Must provide project_id or slug"

            if not response.data:
                return None, "Project not found"

            bundle_path = response.data.get("bundle_path")
            if not bundle_path:
                return None, "No bundle path found"

            bucket_names = []
            try:
                bucket_names.append(st.secrets["supabase"].get("bucket", None))
            except Exception:
                pass

            bucket_names.append("project_bundles")
            bucket_names = [b for b in bucket_names if b]

            last_error = None
            for bucket_name in bucket_names:
                try:
                    data = self.client.storage.from_(bucket_name).download(bundle_path)
                    return data, None
                except Exception as e:
                    last_error = str(e)

            return None, f"Could not download from any bucket. Last error: {last_error}. Tried: {bucket_names}"
        except Exception as e:
            return None, str(e)

    def delete_project(self, project_id: str = None, slug: str = None) -> tuple:
        try:
            user = self.get_current_user()
            if not user:
                return False, "Not authenticated"

            if project_id:
                response = self.client.table("projects").select("*").eq("id", project_id).single().execute()
            elif slug:
                response = self.client.table("projects").select("*").eq("slug", slug).single().execute()
            else:
                return False, "Must provide project_id or slug"

            if not response.data:
                return False, "Project not found"

            project = response.data
            if project.get("user_id") != user["id"]:
                return False, "Permission denied"

            bundle_path = project.get("bundle_path")
            if bundle_path:
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

            self.client.table("projects").delete().eq("id", project["id"]).execute()
            return True, None
        except Exception as e:
            return False, str(e)

    def get_project(self, project_id: str = None, slug: str = None) -> tuple:
        try:
            if project_id:
                response = self.client.table("projects").select("*").eq("id", project_id).single().execute()
            elif slug:
                response = self.client.table("projects").select("*").eq("slug", slug).single().execute()
            else:
                return None, "Must provide project_id or slug"

            if not response.data:
                return None, "Project not found"

            row = response.data
            project = Project(
                id=row.get("id"),
                slug=row.get("slug"),
                name=row.get("name", row.get("slug", "Unnamed")),
                source_app=row.get("source_app"),
                node_count=row.get("node_count", 0),
                edge_count=row.get("edge_count", 0),
                jurisdiction=row.get("jurisdiction"),
                region_preset=row.get("region_preset"),
                app_version=row.get("app_version"),
                schema_version=row.get("schema_version"),
                bundle_version=row.get("bundle_version"),
                is_public=row.get("is_public", False),
                user_id=row.get("user_id"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
                bundle_path=row.get("bundle_path"),
            )
            return project, None
        except Exception as e:
            return None, str(e)


# =============================================================================
# Session State
# =============================================================================
def init_session_state():
    if "project_store" not in st.session_state:
        st.session_state.project_store = None
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "projects"


def init_project_store():
    if st.session_state.project_store is not None:
        return st.session_state.project_store

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


def render_auth_sidebar():
    client = init_project_store()

    if not client:
        return None

    if client.is_authenticated():
        user = client.get_current_user()
        st.sidebar.success(f"☁️ {user['email']}")

        if st.sidebar.button("Logout", use_container_width=True):
            client.logout()
            st.rerun()

        return client

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


def load_projects(client, source_app_filter: str = None):
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
    if not date_str:
        return "—"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return date_str[:16] if len(date_str) > 16 else date_str


def render_project_table(projects: list, search_query: str = ""):
    if not projects:
        st.info("No projects found. Save a project from OrgGraph, ActorGraph, or InsightGraph to see it here.")
        return None

    if search_query:
        q = search_query.lower()
        projects = [p for p in projects if q in (p.name or "").lower()]

    if not projects:
        st.warning(f"No projects matching '{search_query}'")
        return None

    table_data = []
    for p in projects:
        source_app = p.source_app or "unknown"
        icon = APP_ICONS.get(source_app, "📁")
        app_label = SOURCE_APPS.get(source_app, source_app)

        table_data.append({
            "icon": icon,
            "name": p.name or p.slug or "—",
            "app": app_label,
            "nodes": p.node_count or 0,
            "edges": p.edge_count or 0,
            "updated": format_date(p.updated_at),
            "project": p,
        })

    table_data.sort(key=lambda x: x["project"].updated_at or "", reverse=True)

    st.markdown(f"**{len(table_data)} project(s)**")

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


def render_project_detail(client, project):
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back"):
            st.session_state.selected_project = None
            st.rerun()
    with col2:
        source_app = project.source_app or "unknown"
        icon = APP_ICONS.get(source_app, "📁")
        st.subheader(f"{icon} {project.name or 'Unnamed Project'}")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", f"{project.node_count or 0:,}")
    with col2:
        st.metric("Edges", f"{project.edge_count or 0:,}")
    with col3:
        st.metric("Source App", SOURCE_APPS.get(source_app, source_app))
    with col4:
        st.metric("Jurisdiction", project.jurisdiction or "—")

    with st.expander("📋 Project Details", expanded=False):
        details = {
            "Slug": project.slug or "—",
            "Created": format_date(project.created_at),
            "Updated": format_date(project.updated_at),
            "App Version": project.app_version or "—",
            "Schema Version": project.schema_version or "—",
            "Region Preset": project.region_preset or "—",
            "Public": "Yes" if project.is_public else "No",
        }
        for k, v in details.items():
            st.caption(f"**{k}:** {v}")

    st.divider()
    st.markdown("### 📥 Actions")

    col1, col2, col3 = st.columns(3)

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
                            use_container_width=True,
                        )
                    else:
                        st.warning("No bundle data found")
                except Exception as e:
                    st.error(f"Download error: {e}")

    with col2:
        preview_clicked = st.button("👁️ Preview Data", use_container_width=True)

    with col3:
        delete_clicked = st.button("🗑️ Delete", use_container_width=True)

    if preview_clicked:
        st.divider()
        render_data_preview(client, project)

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
    st.markdown("### 👁️ Data Preview")

    try:
        bundle_data, error = client.load_project(project_id=project.id, slug=project.slug)
        if error:
            st.error(f"Failed to load data: {error}")
            return
        if not bundle_data:
            st.warning("No bundle data found")
            return

        with zipfile.ZipFile(BytesIO(bundle_data), "r") as z:
            file_list = z.namelist()

            nodes_file = None
            edges_file = None
            grants_file = None
            board_file = None

            for f in file_list:
                fl = f.lower()
                if "nodes" in fl and fl.endswith(".csv") and "polinode" not in fl:
                    nodes_file = f
                elif "edges" in fl and fl.endswith(".csv") and "polinode" not in fl:
                    edges_file = f
                elif "grants_detail" in fl and fl.endswith(".csv"):
                    grants_file = f
                elif "board_detail" in fl and fl.endswith(".csv"):
                    board_file = f

            tabs = ["Nodes", "Edges"]
            if grants_file:
                tabs.append("Grants")
            if board_file:
                tabs.append("Board")
            tabs.append("Files")

            tab_objs = st.tabs(tabs)

            with tab_objs[0]:
                if nodes_file:
                    with z.open(nodes_file) as f:
                        df = pd.read_csv(f)
                    st.caption(f"**{len(df):,} nodes** from `{nodes_file}`")
                    st.dataframe(df.head(100), use_container_width=True)
                    if len(df) > 100:
                        st.caption(f"Showing first 100 of {len(df):,} rows")
                else:
                    st.info("No nodes.csv found in bundle")

            with tab_objs[1]:
                if edges_file:
                    with z.open(edges_file) as f:
                        df = pd.read_csv(f)
                    st.caption(f"**{len(df):,} edges** from `{edges_file}`")
                    st.dataframe(df.head(100), use_container_width=True)
                    if len(df) > 100:
                        st.caption(f"Showing first 100 of {len(df):,} rows")
                else:
                    st.info("No edges.csv found in bundle")

            tab_idx = 2
            if grants_file:
                with tab_objs[tab_idx]:
                    with z.open(grants_file) as f:
                        df = pd.read_csv(f)
                    st.caption(f"**{len(df):,} grants** from `{grants_file}`")
                    st.dataframe(df.head(100), use_container_width=True)
                    if len(df) > 100:
                        st.caption(f"Showing first 100 of {len(df):,} rows")
                tab_idx += 1

            if board_file:
                with tab_objs[tab_idx]:
                    with z.open(board_file) as f:
                        df = pd.read_csv(f)
                    st.caption(f"**{len(df):,} board members** from `{board_file}`")

                    if "interlock_count" in df.columns:
                        interlocked = df[df["interlock_count"] > 1]
                        if not interlocked.empty:
                            unique_people = (
                                interlocked["person_name_normalized"].nunique()
                                if "person_name_normalized" in interlocked.columns
                                else len(interlocked)
                            )
                            st.info(f"🔗 **{unique_people} people** serve on multiple boards")

                    st.dataframe(df.head(100), use_container_width=True)
                    if len(df) > 100:
                        st.caption(f"Showing first 100 of {len(df):,} rows")
                tab_idx += 1

            with tab_objs[tab_idx]:
                st.caption(f"**{len(file_list)} files in bundle:**")
                for f in sorted(file_list):
                    info = z.getinfo(f)
                    st.caption(f"  `{f}` ({info.file_size/1024:.1f} KB)")

    except Exception as e:
        st.error(f"Preview error: {e}")


# =============================================================================
# Docs & Artifacts Tab
# =============================================================================
def render_docs_artifacts_tab():
    """Render the Docs & Artifacts tab with manifest-driven content."""
    
    manifest = fetch_manifest()
    
    if not manifest:
        st.warning("Could not load docs and artifacts manifest.")
        st.caption("Check network connection or try refreshing.")
        return
    
    # Quick Start Guides
    st.markdown("### 📘 Quick Start Guides")
    st.caption("Markdown versions — download or view in any text editor.")
    
    docs = manifest.get("docs", [])
    quickstarts = [d for d in docs if d.get("type") == "quickstart"]
    
    if quickstarts:
        for doc in quickstarts:
            title = doc.get("title", "Untitled")
            notes = doc.get("notes", "")
            md_url = doc.get("md_url", "")
            
            if md_url:
                col1, col2, col3 = st.columns([0.5, 4, 1.5])
                with col1:
                    # Get app icon from quickstart_id
                    qs_id = doc.get("id", "")
                    app_key = qs_id.replace("qs_", "") if qs_id.startswith("qs_") else ""
                    icon = APP_ICONS.get(app_key, "📝")
                    st.write(icon)
                with col2:
                    st.markdown(f"**{title}**")
                    st.caption(notes)
                with col3:
                    st.link_button("⬇ Download MD", md_url, use_container_width=True)
    else:
        st.info("No Quick Start guides found.")
    
    st.divider()
    
    # Technical References
    st.markdown("### 📊 Technical References")
    
    other_docs = [d for d in docs if d.get("type") != "quickstart"]
    schema = manifest.get("schema", {})
    
    if other_docs or schema:
        for doc in other_docs:
            title = doc.get("title", "Untitled")
            notes = doc.get("notes", "")
            url = doc.get("url", "")
            file_type = get_file_type_from_url(url)
            icon = get_file_icon(file_type)
            
            if url:
                col1, col2, col3 = st.columns([0.5, 4, 1.5])
                with col1:
                    st.write(icon)
                with col2:
                    st.markdown(f"**{title}**")
                    st.caption(notes)
                with col3:
                    btn_label = "📄 View PDF" if file_type == "pdf" else "⬇ Download"
                    st.link_button(btn_label, url, use_container_width=True)
        
        # Schema
        if schema and schema.get("url"):
            col1, col2, col3 = st.columns([0.5, 4, 1.5])
            with col1:
                st.write("📋")
            with col2:
                st.markdown(f"**{schema.get('title', 'Schema & Conventions')}**")
                status = schema.get("status", "draft")
                st.caption(f"{schema.get('description', '')} ({status})")
            with col3:
                st.link_button("📄 Open", schema.get("url"), use_container_width=True)
    else:
        st.info("No technical references found.")
    
    st.divider()
    
    # Demo Projects (Artifacts)
    st.markdown("### 📁 Demo Project Artifacts")
    st.caption("Inputs and outputs for demo workflows.")
    
    projects = manifest.get("projects", [])
    
    if projects:
        for project in projects:
            project_title = project.get("title", project.get("id", "Project"))
            project_desc = project.get("description", "")
            readme_url = project.get("readme", "")
            
            with st.expander(f"**{project_title}**", expanded=True):
                st.caption(project_desc)
                
                if readme_url:
                    st.link_button("📄 README", readme_url)
                
                # Inputs
                inputs = project.get("inputs", [])
                if inputs:
                    st.markdown("**Inputs**")
                    for item in inputs:
                        name = item.get("name", "Untitled")
                        notes = item.get("notes", "")
                        url = item.get("url", "")
                        file_type = item.get("type", get_file_type_from_url(url))
                        icon = get_file_icon(file_type)
                        primary_app = item.get("primary_app", "")
                        app_icon = APP_ICONS.get(primary_app, "")
                        
                        if url:
                            col1, col2, col3 = st.columns([0.5, 4, 1.5])
                            with col1:
                                st.write(icon)
                            with col2:
                                app_badge = f" {app_icon}" if app_icon else ""
                                st.markdown(f"{name}{app_badge}")
                                st.caption(notes)
                            with col3:
                                st.link_button("⬇ Download", url, use_container_width=True)
                
                # Outputs
                outputs = project.get("outputs", [])
                if outputs:
                    st.markdown("**Outputs**")
                    for item in outputs:
                        name = item.get("name", "Untitled")
                        notes = item.get("notes", "")
                        url = item.get("url", "")
                        file_type = item.get("type", get_file_type_from_url(url))
                        icon = get_file_icon(file_type)
                        primary_app = item.get("primary_app", "")
                        app_icon = APP_ICONS.get(primary_app, "")
                        
                        if url:
                            col1, col2, col3 = st.columns([0.5, 4, 1.5])
                            with col1:
                                st.write(icon)
                            with col2:
                                app_badge = f" {app_icon}" if app_icon else ""
                                st.markdown(f"{name}{app_badge}")
                                st.caption(notes)
                            with col3:
                                btn_label = "👁 View" if file_type == "html" else "⬇ Download"
                                st.link_button(btn_label, url, use_container_width=True)
    else:
        st.info("No demo projects found.")
    
    # Manifest info
    st.divider()
    manifest_version = manifest.get("manifest_version", "—")
    last_updated = manifest.get("last_updated", "—")
    st.caption(f"📋 Manifest v{manifest_version} • Last updated: {last_updated}")


# =============================================================================
# Main
# =============================================================================
def main():
    init_session_state()

    st.title("☁️ CloudProjects")
    st.caption(f"Manage cloud projects and access docs & artifacts • v{APP_VERSION}")

    st.sidebar.title("☁️ Cloud")
    client = render_auth_sidebar()

    # Main tabs
    tab_projects, tab_docs = st.tabs(["📋 Projects", "📚 Docs & Artifacts"])
    
    with tab_projects:
        if not client:
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
        else:
            st.sidebar.divider()
            st.sidebar.markdown("### 🔍 Filters")

            source_app_filter = st.sidebar.selectbox(
                "Source App",
                options=list(SOURCE_APPS.keys()),
                format_func=lambda x: SOURCE_APPS[x],
                index=0,
            )

            search_query = st.sidebar.text_input("🔎 Search projects", placeholder="Project name...")

            st.divider()

            if st.session_state.selected_project:
                render_project_detail(client, st.session_state.selected_project)
            else:
                st.markdown("### 📋 Your Projects")
                projects = load_projects(client, source_app_filter)
                render_project_table(projects, search_query)

                if projects:
                    st.divider()
                    total_nodes = sum(p.node_count or 0 for p in projects)
                    total_edges = sum(p.edge_count or 0 for p in projects)
                    st.caption(f"**Total:** {len(projects)} projects • {total_nodes:,} nodes • {total_edges:,} edges")
    
    with tab_docs:
        render_docs_artifacts_tab()


if __name__ == "__main__":
    main()
