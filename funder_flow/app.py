"""
OrgGraph (US) — US Nonprofit Registry Ingestion

Multi-project Streamlit app:
- New Project: Create a new project and upload initial data
- Add to Existing: Select existing project and merge new data
- View Demo: Read-only view of sample demo data

Outputs conform to C4C Network Schema v1 (MVP):
- nodes.csv: ORG and PERSON nodes
- edges.csv: GRANT and BOARD_MEMBERSHIP edges
- grants_detail.csv: Canonical grant detail format (shared with CA)

VERSION HISTORY:
----------------
UPDATED v0.25.0: Board Detail Export with Interlock Detection
- NEW: board_detail.csv in ZIP bundle with full board member data
- Extracts all board member info from IRS 990/990-PF XML files
- Includes: name, title, hours/week, address (when available), org info
- Automatic interlock detection: finds people serving on multiple boards
- Name normalization for matching (handles middle initials, suffixes)
- Confidence scoring: exact matches, fuzzy matches, unique entries
- Address type indicator: individual (990-PF) vs organization (990)
- README updated with board_detail.csv column definitions

UPDATED v0.24.0: Download simplification + BOM fix
- Collapsed 8 download buttons to 3 (Save to Project, Save to Cloud, Download ZIP)
- Added README.md to ZIP bundle with column definitions and usage guide
- Project-prefixed filenames in ZIP (e.g., great_lakes_nodes.csv)
- Cleaner export UI with single primary action
- Fixed: UTF-8 BOM handling in XML validation (was rejecting valid XML files)

UPDATED v0.23.1: iPad Safari XML download guardrail + clearer UX
- Detect and block "fake XML" files saved from iPad Safari (rendered text view instead of raw XML)
- Show actionable error message: long-press XML link → "Download Linked File"
- Add UI caption reminding iPad users about the correct download method

UPDATED v0.23.0: Phase 2 - Project Store cloud integration
- Save bundles to cloud (Supabase Storage + projects table)
- ZIP bundles uploaded with full metadata
- Project Store client replaces old row-based storage
- Compatible with InsightGraph cloud loading

UPDATED v0.22.0: Phase 1c - ZIP bundle format with manifest.json
- Download All ZIP now includes manifest.json with bundle metadata
- Files at root level (removed c4c_schema/ subfolder)
- Polinode files in polinode/ folder (renamed from polinode_schema/)
- Manifest includes: schema version, source app, timestamps, row counts

UPDATED v0.21.0: Phase 1b - Unified CSV column ordering
- Exports now use standardized column order from coregraph_schema
- All nodes.csv and edges.csv exports have consistent structure
- Compatible with OrgGraph CA exports for easy merging

UPDATED v0.20.0: CoreGraph v1 schema normalization (Phase 1a)
- node_type normalized to lowercase: ORG→organization, PERSON→person
- edge_type normalized to lowercase: GRANT→grant, BOARD_MEMBERSHIP→board
- Added org_type field for funder/grantee roles
- Added source_app field for provenance tracking
- Node IDs namespaced: org-123→orggraph_us:org-123
- Added directed and weight fields to edges

UPDATED v0.19.0: Supabase cloud storage integration
- Added cloud save/load functionality
- User authentication via Supabase
- Projects persist in cloud database
- Save to Local and Save to Cloud options

UPDATED v0.18.0: Canonical role vocabulary
- Role columns now include: network_role_code, network_role_label, network_role_order
- Labels aligned with spec: "Funder + Grantee" (not "/"), "Individual" (not "Person")
- Codes: FUNDER, GRANTEE, FUNDER_GRANTEE, BOARD_MEMBER, ORGANIZATION, INDIVIDUAL
- Legend order (1-6) for consistent Polinode sorting
- Derivation order documented (BOARD_MEMBER overrides first)

UPDATED v0.17.0: Network Role in Polinode export
- Added `Network Role` column: Funder, Grantee, Funder / Grantee, Board Member
- Derived from edge relationships (who gives grants, receives grants, serves on boards)
- Enables filtering/coloring by role in Polinode visualization

UPDATED v0.16.0: Polinode export improvements
- Name canonicalization (Unicode NFKC, whitespace, quotes, hyphens)
- Duplicate node names deduplicated (prefers ORG over PERSON)
- Post-export validation (warns if duplicate names or missing edge references)
- Excel export with Nodes + Edges tabs for direct Polinode import
- ZIP structure now uses c4c_schema/ and polinode_schema/ folders

UPDATED v0.14.0: Multi-project support
- Project selection UI (New / Add to Existing / View Demo)
- Merge behavior for adding foundations to existing projects
- Region mode with Great Lakes preset

UPDATED v0.14.1: Region tagging improvements
- apply_region_tagging() for grantee state/country analysis
- region_relevant column in grants_detail.csv
- Region summary stats in UI

UPDATED v0.15.0: grants_detail.csv saved to project folder
- grants_detail.csv now saved directly to demo_data/{project}/
- Append-only merge behavior (existing rows preserved)
- Deduplication by composite key (foundation_ein + grantee_name + amount + year)
- All 19 canonical columns ensured on every export
- Schema aligned with OrgGraph CA for cross-border analysis
"""
import streamlit as st
from console_ui import inject_c4c_console_theme, c4c_header, c4c_badge
from console_state import get_cloud_status
import pandas as pd
import json
from io import BytesIO
import zipfile
import sys
import os
import re
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to path for imports (MUST be before c4c_utils imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4c_utils.c4c_supabase import C4CSupabase
from c4c_utils.irs990_parser import PARSER_VERSION
from c4c_utils.irs990pf_xml_parser import parse_990pf_xml
from c4c_utils.irs990_xml_parser import parse_990_xml
from c4c_utils.network_export import build_nodes_df, build_edges_df, NODE_COLUMNS, EDGE_COLUMNS, get_existing_foundations
from c4c_utils.regions_presets import REGION_PRESETS, US_STATES, CA_PROVINCES
from c4c_utils.project_store import list_projects, load_project_config, save_project_config, get_region_from_config, update_region_in_config
from c4c_utils.region_tagger import apply_region_tagging, get_region_summary
from c4c_utils.board_extractor import BoardExtractor
from c4c_utils.irs_return_qa import compute_confidence, render_return_qa_panel
from c4c_utils.irs_return_dispatcher import parse_irs_return
from c4c_utils.summary_helpers import build_grant_network_summary, summarize_grants
from c4c_utils.coregraph_schema import prepare_unified_nodes_csv, prepare_unified_edges_csv, namespace_id, COREGRAPH_VERSION

# =============================================================================
# Constants
# =============================================================================
APP_VERSION = "0.25.0"  # Board detail export with interlock detection
MAX_FILES = 50
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_25063966d6cd496eb2fe3f6ee5cde0fa~mv2.png"
APP_ICON_URL = "https://static.wixstatic.com/media/275a3f_f8ac661446cc49788526af77befbf7b3~mv2.png"
SOURCE_SYSTEM = "IRS_990"
JURISDICTION = "US"
SOURCE_APP = "orggraph_us"  # CoreGraph v1 source app identifier
# Demo data paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
DEMO_PROJECT_NAME = "_demo"  # Reserved name for demo dataset

# =============================================================================
# Canonical grants_detail.csv Schema (Shared with OrgGraph CA)
# =============================================================================
GRANTS_DETAIL_COLUMNS = [
    "foundation_name", "foundation_ein", "tax_year", "grantee_name",
    "grantee_city", "grantee_state", "grant_amount", "grant_purpose_raw",
    "grant_bucket", "region_relevant", "source_file",
    "grantee_country", "foundation_country", "source_system",
    "grant_amount_cash", "grant_amount_in_kind", "currency",
    "fiscal_year", "reporting_period"
]

# US Grant bucket constants
GRANT_BUCKET_3A = "3a"           # Part XIV line 3a (paid this year)
GRANT_BUCKET_3B = "3b"           # Part XIV line 3b (future commitments)
GRANT_BUCKET_SCHEDULE_I = "schedule_i"  # Schedule I grants

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="OrgGraph (US)",
    page_icon=APP_ICON_URL,
    layout="wide"
)
inject_c4c_console_theme()
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
    grants_detail_path = project_path / "grants_detail.csv"
    
    nodes_df = pd.DataFrame(columns=NODE_COLUMNS)
    edges_df = pd.DataFrame(columns=EDGE_COLUMNS)
    grants_detail_df = pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS)
    
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
    
    if grants_detail_path.exists():
        try:
            df = pd.read_csv(grants_detail_path)
            if not df.empty and len(df) > 0:
                grants_detail_df = df
        except:
            pass
    
    return nodes_df, edges_df, grants_detail_df
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
    if "board_detail_df" not in st.session_state:
        st.session_state.board_detail_df = pd.DataFrame()
    if "parse_results" not in st.session_state:
        st.session_state.parse_results = []
    if "merge_stats" not in st.session_state:
        st.session_state.merge_stats = {}
    if "processed_orgs" not in st.session_state:
        st.session_state.processed_orgs = []
    if "current_project" not in st.session_state:
        st.session_state.current_project = None
    if "region_def" not in st.session_state:
        st.session_state.region_def = None
    
    # Initialize Supabase connection (legacy)
    init_supabase()
    
    # Initialize Project Store (Phase 2)
    init_project_store()


# =============================================================================
# Supabase Cloud Integration
# =============================================================================

def init_supabase():
    """Initialize Supabase connection."""
    if "supabase_db" not in st.session_state:
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            st.session_state.supabase_db = C4CSupabase(url=url, key=key)
        except Exception as e:
            st.session_state.supabase_db = None
    return st.session_state.get("supabase_db")


def init_project_store():
    """Initialize Project Store client for bundle storage (Phase 2)."""
    if "project_store" not in st.session_state:
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            
            # Import Project Store client
            from c4c_utils.c4c_project_store import ProjectStoreClient
            
            client = ProjectStoreClient(url, key)
            st.session_state.project_store = client
        except ImportError:
            st.session_state.project_store = None
            # Silently fail if module not available
        except Exception as e:
            st.session_state.project_store = None
    
    return st.session_state.get("project_store")


def get_project_store_authenticated():
    """Get authenticated Project Store client, or None."""
    client = st.session_state.get("project_store")
    if client and client.is_authenticated():
        return client
    return None


def render_cloud_status():
    """Render cloud connection status and login UI in sidebar."""
    # Initialize Project Store
    init_project_store()
    
    client = st.session_state.get("project_store")
    
    if not client:
        st.sidebar.caption("☁️ Cloud unavailable")
        return None
    
    if client.is_authenticated():
        user = client.get_current_user()
        
        # Get project count
        projects, _ = client.list_projects(source_app=SOURCE_APP)
        project_count = len(projects) if projects else 0
        
        # Logged in: show email + project count + logout button
        st.sidebar.caption(f"☁️ {user['email']}")
        st.sidebar.caption(f"📦 {project_count} cloud project(s)")
        
        if st.sidebar.button("Logout", key="cloud_logout", use_container_width=True):
            client.logout()
            st.rerun()
        return client
    else:
        # Not logged in: show status, then collapsible login form
        st.sidebar.caption("☁️ Not connected")
        with st.sidebar.expander("Login / Sign Up", expanded=False):
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                email = st.text_input("Email", key="cloud_login_email")
                password = st.text_input("Password", type="password", key="cloud_login_pass")
                if st.button("Login", key="cloud_login_btn"):
                    success, error = client.login(email, password)
                    if success:
                        st.success("✅ Logged in!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {error}")
            
            with tab2:
                st.caption("First time? Create an account.")
                signup_email = st.text_input("Email", key="cloud_signup_email")
                signup_pass = st.text_input("Password", type="password", key="cloud_signup_pass")
                if st.button("Sign Up", key="cloud_signup_btn"):
                    success, error = client.signup(signup_email, signup_pass)
                    if success:
                        st.success("✅ Check email to confirm")
                    else:
                        st.error(f"Signup failed: {error}")
        
        return client


def save_to_cloud(project_name: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                  grants_df: pd.DataFrame = None, region_def: dict = None):
    """Save project data to Supabase cloud."""
    db = st.session_state.get("supabase_db")
    
    if not db or not db.is_authenticated:
        st.error("❌ Login required to save to cloud")
        return False
    
    # Debug: check what we're receiving
    print(f"DEBUG save_to_cloud: nodes={len(nodes_df) if nodes_df is not None else 'None'}, edges={len(edges_df) if edges_df is not None else 'None'}, grants={len(grants_df) if grants_df is not None else 'None'}")
    if grants_df is not None and not grants_df.empty:
        print(f"DEBUG save_to_cloud: grants_df columns = {list(grants_df.columns)}")
    
    # Create slug from project name
    slug = project_name.lower().replace(" ", "-").replace("_", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    slug = slug[:50]  # Limit length
    
    if not slug:
        slug = "untitled-project"
    
    with st.spinner("☁️ Saving to cloud..."):
        try:
            # Check if project exists
            existing = db.get_project_by_slug(slug)
            
            if existing:
                project = existing
            else:
                # Create new project
                config = {}
                if region_def:
                    config["region_lens"] = region_def
                
                project = db.create_project(
                    name=project_name,
                    slug=slug,
                    source_app="orggraph_us",
                    config=config
                )
                
                if not project:
                    error_msg = db.last_error if hasattr(db, 'last_error') else "Unknown error"
                    st.error(f"❌ Failed to create cloud project: {error_msg}")
                    return False
            
            # Save data
            st.info(f"🔍 Attempting save: nodes={len(nodes_df) if nodes_df is not None else 0}, edges={len(edges_df) if edges_df is not None else 0}, grants={len(grants_df) if grants_df is not None and not grants_df.empty else 0}")
            
            results = db.save_project_data(
                project_id=project["id"],
                nodes_df=nodes_df,
                edges_df=edges_df,
                grants_df=grants_df if grants_df is not None else None
            )
            
            st.info(f"🔍 Results: {results}")
            
            grants_msg = f", {results.get('grants', 0)} grants" if results.get('grants', 0) > 0 else ""
            st.success(f"☁️ Saved to cloud: {results.get('nodes', 0)} nodes, {results.get('edges', 0)} edges{grants_msg}")
            return True
            
        except Exception as e:
            st.error(f"❌ Cloud save failed: {e}")
            return False


def save_bundle_to_cloud(
    project_name: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    grants_df: pd.DataFrame = None,
    region_def: dict = None,
    poli_nodes: pd.DataFrame = None,
    poli_edges: pd.DataFrame = None,
    polinode_excel: bytes = None,
    parse_results: list = None
) -> tuple:
    """
    Save project as ZIP bundle to Project Store (Supabase Storage).
    
    This is the Phase 2 cloud storage approach - stores complete bundles
    rather than individual rows.
    
    Returns:
        Tuple of (success: bool, message: str, project_slug: str or None)
    """
    client = get_project_store_authenticated()
    
    if not client:
        return False, "Login required to save to cloud", None
    
    # Create ZIP bundle in memory
    zip_buffer = BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Prepare grants_detail
            grants_detail = None
            if grants_df is not None and not grants_df.empty:
                grants_detail = ensure_grants_detail_columns(grants_df)
            
            # manifest.json
            manifest = generate_bundle_manifest(
                nodes_df=nodes_df,
                edges_df=edges_df,
                grants_detail_df=grants_detail,
                project_name=project_name
            )
            zip_file.writestr('manifest.json', json.dumps(manifest, indent=2))
            
            # Core data files
            if nodes_df is not None and not nodes_df.empty:
                zip_file.writestr('nodes.csv', nodes_df.to_csv(index=False))
            if edges_df is not None and not edges_df.empty:
                zip_file.writestr('edges.csv', edges_df.to_csv(index=False))
            if grants_detail is not None and not grants_detail.empty:
                zip_file.writestr('grants_detail.csv', grants_detail.to_csv(index=False))
            
            # Polinode files
            if poli_nodes is not None and not poli_nodes.empty:
                zip_file.writestr('polinode/nodes_polinode.csv', poli_nodes.to_csv(index=False))
            if poli_edges is not None and not poli_edges.empty:
                zip_file.writestr('polinode/edges_polinode.csv', poli_edges.to_csv(index=False))
            if polinode_excel:
                zip_file.writestr('polinode/polinode_import.xlsx', polinode_excel)
            
            # Parse log
            if parse_results:
                zip_file.writestr('parse_log.json', json.dumps(parse_results, indent=2, default=str))
        
        zip_buffer.seek(0)
        bundle_data = zip_buffer.getvalue()
        
        # Upload to Project Store
        node_count = len(nodes_df) if nodes_df is not None else 0
        edge_count = len(edges_df) if edges_df is not None else 0
        region_preset = region_def.get('id') if region_def else None
        
        project, error = client.save_project(
            name=project_name,
            bundle_data=bundle_data,
            source_app=SOURCE_APP,
            node_count=node_count,
            edge_count=edge_count,
            jurisdiction=JURISDICTION,
            region_preset=region_preset,
            app_version=APP_VERSION,
            schema_version=COREGRAPH_VERSION,
            bundle_version="1.0"
        )
        
        if error:
            return False, f"Upload failed: {error}", None
        
        return True, f"Saved to cloud: {project.slug}", project.slug
        
    except Exception as e:
        return False, f"Bundle creation failed: {str(e)}", None


def list_cloud_projects():
    """List projects from Supabase cloud."""
    db = st.session_state.get("supabase_db")
    
    if not db or not db.is_authenticated:
        return []
    
    return db.list_projects(source_app="orggraph_us")


def load_from_cloud(project_id: str):
    """Load project data from Supabase cloud."""
    db = st.session_state.get("supabase_db")
    
    if not db:
        return None, None, None
    
    nodes_df = db.get_nodes(project_id)
    edges_df = db.get_edges(project_id)
    grants_df = db.get_grants_detail(project_id)
    
    return nodes_df, edges_df, grants_df


def clear_session_state():
    """Clear all processing results from session state."""
    st.session_state.processed = False
    st.session_state.nodes_df = pd.DataFrame()
    st.session_state.edges_df = pd.DataFrame()
    st.session_state.grants_df = None
    st.session_state.parse_results = []
    st.session_state.merge_stats = {}
    st.session_state.processed_orgs = []
    st.session_state.region_def = None
    st.session_state.board_detail_df = pd.DataFrame()


def store_results(nodes_df, edges_df, grants_df, parse_results, merge_stats, processed_orgs, region_def=None, board_detail_df=None):
    """Store processing results in session state."""
    st.session_state.processed = True
    st.session_state.nodes_df = nodes_df
    st.session_state.edges_df = edges_df
    st.session_state.grants_df = grants_df
    st.session_state.parse_results = parse_results
    st.session_state.merge_stats = merge_stats
    st.session_state.processed_orgs = processed_orgs
    st.session_state.region_def = region_def
    st.session_state.board_detail_df = board_detail_df if board_detail_df is not None else pd.DataFrame()
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


def merge_grants_detail(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> tuple:
    """
    Merge new grants_detail rows with existing, avoiding duplicates.
    
    Uses composite key: foundation_ein + grantee_name + grant_amount + fiscal_year
    
    Returns: (merged_df, stats_dict)
    """
    stats = {
        "existing": len(existing_df) if not existing_df.empty else 0,
        "new": len(new_df) if not new_df.empty else 0,
        "added": 0,
        "skipped": 0,
    }
    
    if existing_df.empty or len(existing_df) == 0:
        stats["added"] = len(new_df) if not new_df.empty else 0
        return new_df.copy() if not new_df.empty else pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS), stats
    
    if new_df.empty or len(new_df) == 0:
        return existing_df.copy(), stats
    
    # Create composite key for deduplication
    def make_key(row):
        return f"{row.get('foundation_ein', '')}|{row.get('grantee_name', '')}|{row.get('grant_amount', '')}|{row.get('fiscal_year', '')}"
    
    existing_keys = set(existing_df.apply(make_key, axis=1))
    new_keys = new_df.apply(make_key, axis=1)
    
    mask = ~new_keys.isin(existing_keys)
    to_add = new_df[mask]
    
    stats["added"] = len(to_add)
    stats["skipped"] = len(new_df) - len(to_add)
    
    merged = pd.concat([existing_df, to_add], ignore_index=True)
    
    return merged, stats


def ensure_grants_detail_columns(grants_df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    """
    Ensure grants_df has all canonical grants_detail columns.
    Maps existing columns to canonical names and fills missing with defaults.
    """
    if grants_df is None or grants_df.empty:
        return pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS)
    
    df = grants_df.copy()
    
    # Map existing column names to canonical names
    column_mappings = {
        "grantee": "grantee_name",
        "org_name": "grantee_name",
        "amount": "grant_amount",
        "state": "grantee_state",
        "city": "grantee_city",
        "purpose": "grant_purpose_raw",
        "grant_purpose": "grant_purpose_raw",
    }
    
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Add missing columns with defaults
    defaults = {
        "foundation_name": "",
        "foundation_ein": "",
        "tax_year": "",
        "grantee_name": "",
        "grantee_city": "",
        "grantee_state": "",
        "grant_amount": 0,
        "grant_purpose_raw": "",
        "grant_bucket": "unknown",
        "region_relevant": True,
        "source_file": source_file,
        "grantee_country": "US",
        "foundation_country": "US",
        "source_system": SOURCE_SYSTEM,
        "grant_amount_cash": "",
        "grant_amount_in_kind": "",
        "currency": "USD",
        "fiscal_year": "",
        "reporting_period": "",
    }
    
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    
    # Ensure all canonical columns exist
    for col in GRANTS_DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    return df


def extract_board_with_fallback(file_bytes: bytes, filename: str, meta: dict) -> pd.DataFrame:
    """
    Extract board members using BoardExtractor.
    
    Writes bytes to temp file since BoardExtractor needs a file path.
    Returns DataFrame with columns matching network_export.py expected format:
    - person_name, org_ein, org_name, role, tax_year
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        extractor = BoardExtractor(tmp_path)
        members = extractor.extract()
        
        os.unlink(tmp_path)  # Clean up temp file
        
        if members:
            # Get foundation info from meta
            org_name = meta.get('foundation_name', '')
            org_ein = meta.get('foundation_ein', '').replace('-', '')
            tax_year = meta.get('tax_year', '')
            
            # Convert to DataFrame with columns matching network_export.py
            people_data = []
            for m in members:
                people_data.append({
                    'person_name': m.name,           # network_export expects person_name
                    'role': m.title,                 # network_export expects role
                    'org_name': org_name,            # network_export expects org_name
                    'org_ein': org_ein,              # network_export expects org_ein
                    'tax_year': tax_year,            # needed for node/edge IDs
                    'compensation': m.compensation or 0,
                    'hours_per_week': m.hours_per_week,
                    'benefits': m.benefits or 0,
                    'expense_allowance': m.expense_allowance or 0,
                })
            return pd.DataFrame(people_data)
        
        return pd.DataFrame()
    
    except Exception as e:
        # Log but don't fail - return empty DataFrame
        return pd.DataFrame()


# =============================================================================
# Board Detail Extraction (Phase 1: Board Detail CSV)
# =============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize a person's name for matching.
    
    Transformations:
    - Lowercase
    - Remove middle initials (single letters followed by period or space)
    - Remove suffixes (Jr, Sr, II, III, IV, MD, PhD, etc.)
    - Remove extra whitespace
    - Remove punctuation
    
    Returns normalized name for comparison.
    """
    import re
    import unicodedata
    
    if not name:
        return ""
    
    # Unicode normalize
    name = unicodedata.normalize('NFKC', name)
    
    # Lowercase
    name = name.lower()
    
    # Remove common suffixes
    suffixes = r'\b(jr\.?|sr\.?|ii|iii|iv|v|md|phd|esq\.?|cpa|jd|mba|dds|do|rn|pe)\b'
    name = re.sub(suffixes, '', name, flags=re.IGNORECASE)
    
    # Remove middle initials (single letter followed by period or space, or between spaces)
    # "John Q. Smith" -> "John Smith"
    # "John Q Smith" -> "John Smith"
    name = re.sub(r'\b[a-z]\.\s*', ' ', name)  # "Q. " -> " "
    name = re.sub(r'\s+[a-z]\s+', ' ', name)   # " Q " -> " "
    
    # Remove punctuation except spaces
    name = re.sub(r'[^\w\s]', '', name)
    
    # Collapse whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def extract_board_details_from_xml(file_bytes: bytes, meta: dict) -> list:
    """
    Extract detailed board member information from IRS 990 XML.
    
    Handles:
    - Form 990 (Part VII Section A)
    - Form 990-PF (Part VIII - Officers, Directors, Trustees)
    - Form 990-EZ
    
    Returns list of dicts with all available fields.
    """
    import xml.etree.ElementTree as ET
    
    # Handle BOM
    if file_bytes.startswith(b'\xef\xbb\xbf'):
        file_bytes = file_bytes[3:]
    
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError:
        return []
    
    ns = {'irs': 'http://www.irs.gov/efile'}
    
    # Get organization info from meta
    org_name = meta.get('foundation_name', '')
    org_ein = meta.get('foundation_ein', '').replace('-', '')
    tax_year = meta.get('tax_year', '')
    form_type = meta.get('form_type', '')
    
    # Get organization address (fallback when individual address not available)
    org_address = {
        'address_line1': '',
        'city': '',
        'state': '',
        'zip': '',
    }
    
    # Try to find filer address
    filer_addr = root.find('.//irs:Filer/irs:USAddress', ns)
    if filer_addr is None:
        filer_addr = root.find('.//irs:USAddress', ns)
    
    if filer_addr is not None:
        addr_line = filer_addr.find('irs:AddressLine1Txt', ns)
        if addr_line is None:
            addr_line = filer_addr.find('irs:AddressLine1', ns)
        city = filer_addr.find('irs:CityNm', ns)
        if city is None:
            city = filer_addr.find('irs:City', ns)
        state = filer_addr.find('irs:StateAbbreviationCd', ns)
        if state is None:
            state = filer_addr.find('irs:State', ns)
        zip_code = filer_addr.find('irs:ZIPCd', ns)
        if zip_code is None:
            zip_code = filer_addr.find('irs:ZIPCode', ns)
        
        org_address['address_line1'] = addr_line.text if addr_line is not None else ''
        org_address['city'] = city.text if city is not None else ''
        org_address['state'] = state.text if state is not None else ''
        org_address['zip'] = zip_code.text if zip_code is not None else ''
    
    board_members = []
    
    # =========================================================================
    # Form 990 - Part VII Section A
    # =========================================================================
    members_990 = root.findall('.//irs:Form990PartVIISectionAGrp', ns)
    
    for m in members_990:
        person_name = m.find('irs:PersonNm', ns)
        title = m.find('irs:TitleTxt', ns)
        hours = m.find('irs:AverageHoursPerWeekRt', ns)
        is_trustee = m.find('irs:IndividualTrusteeOrDirectorInd', ns)
        is_officer = m.find('irs:OfficerInd', ns)
        
        # 990 doesn't have individual addresses, use org address
        member = {
            'person_name': person_name.text if person_name is not None else '',
            'foundation_name': org_name,
            'foundation_ein': org_ein,
            'tax_year': tax_year,
            'title': title.text if title is not None else '',
            'hours_per_week': hours.text if hours is not None else '',
            'is_officer': 'Y' if (is_officer is not None and is_officer.text == 'X') else 'N',
            'is_trustee_director': 'Y' if (is_trustee is not None and is_trustee.text == 'X') else 'N',
            'address_line1': org_address['address_line1'],
            'city': org_address['city'],
            'state': org_address['state'],
            'zip': org_address['zip'],
            'country': 'US',
            'address_type': 'organization',
            'form_type': '990',
        }
        
        if member['person_name']:  # Only add if we have a name
            board_members.append(member)
    
    # =========================================================================
    # Form 990-PF - Part VIII (Officers, Directors, Trustees)
    # =========================================================================
    # Try multiple possible paths for 990-PF
    pf_paths = [
        './/irs:OfficerDirTrstKeyEmplInfoGrp',
        './/irs:OfcrDirTrusteesKeyEmplGrp', 
        './/irs:OfficerDirTrstKeyEmplGrp',
    ]
    
    for pf_path in pf_paths:
        members_pf = root.findall(pf_path, ns)
        
        for m in members_pf:
            # Name can be PersonNm or inside a nested group
            person_name = m.find('.//irs:PersonNm', ns)
            if person_name is None:
                person_name = m.find('.//irs:BusinessNameLine1Txt', ns)
            
            title = m.find('.//irs:TitleTxt', ns)
            hours = m.find('.//irs:AverageHrsPerWkDevotedToPosRt', ns)
            if hours is None:
                hours = m.find('.//irs:AvgHoursPerWkDevotedToPosRt', ns)
            
            # 990-PF often has individual addresses
            ind_addr = m.find('.//irs:USAddress', ns)
            
            if ind_addr is not None:
                addr_line = ind_addr.find('irs:AddressLine1Txt', ns)
                if addr_line is None:
                    addr_line = ind_addr.find('irs:AddressLine1', ns)
                city = ind_addr.find('irs:CityNm', ns)
                if city is None:
                    city = ind_addr.find('irs:City', ns)
                state = ind_addr.find('irs:StateAbbreviationCd', ns)
                if state is None:
                    state = ind_addr.find('irs:State', ns)
                zip_code = ind_addr.find('irs:ZIPCd', ns)
                if zip_code is None:
                    zip_code = ind_addr.find('irs:ZIPCode', ns)
                
                address_type = 'individual'
                address_line1 = addr_line.text if addr_line is not None else ''
                city_val = city.text if city is not None else ''
                state_val = state.text if state is not None else ''
                zip_val = zip_code.text if zip_code is not None else ''
            else:
                # Fall back to org address
                address_type = 'organization'
                address_line1 = org_address['address_line1']
                city_val = org_address['city']
                state_val = org_address['state']
                zip_val = org_address['zip']
            
            member = {
                'person_name': person_name.text if person_name is not None else '',
                'foundation_name': org_name,
                'foundation_ein': org_ein,
                'tax_year': tax_year,
                'title': title.text if title is not None else '',
                'hours_per_week': hours.text if hours is not None else '',
                'is_officer': 'N',  # 990-PF doesn't have this flag separately
                'is_trustee_director': 'Y',  # All in this section are trustees/directors
                'address_line1': address_line1,
                'city': city_val,
                'state': state_val,
                'zip': zip_val,
                'country': 'US',
                'address_type': address_type,
                'form_type': '990-PF',
            }
            
            if member['person_name']:
                board_members.append(member)
    
    return board_members


def detect_board_interlocks(board_members: list) -> pd.DataFrame:
    """
    Detect board interlocks (same person serving on multiple boards).
    
    Uses normalized names to find matches:
    - exact: Normalized names match perfectly
    - fuzzy: Names are similar (Levenshtein distance <= 2)
    - unique: Only appears once
    
    Returns DataFrame with interlock_id, interlock_count, match_confidence added.
    """
    from difflib import SequenceMatcher
    
    if not board_members:
        return pd.DataFrame()
    
    df = pd.DataFrame(board_members)
    
    # Add normalized names
    df['person_name_normalized'] = df['person_name'].apply(normalize_name)
    
    # Group by normalized name to find exact matches
    name_groups = df.groupby('person_name_normalized')
    
    # Track interlock assignments
    interlock_id = 0
    interlock_map = {}  # normalized_name -> interlock_id
    fuzzy_pairs = []  # Track fuzzy matches for later
    
    # First pass: exact matches
    for norm_name, group in name_groups:
        if len(group) > 1:
            # Multiple boards with same normalized name = exact match
            interlock_id += 1
            interlock_map[norm_name] = {
                'id': interlock_id,
                'count': len(group['foundation_ein'].unique()),
                'confidence': 'exact'
            }
        else:
            # Single occurrence so far
            interlock_map[norm_name] = {
                'id': None,  # Will assign later
                'count': 1,
                'confidence': 'unique'
            }
    
    # Second pass: fuzzy matches for names that appear only once
    unique_names = [n for n, v in interlock_map.items() if v['count'] == 1]
    
    def similar(a: str, b: str) -> bool:
        """Check if two names are similar enough to be the same person."""
        if not a or not b:
            return False
        # Use SequenceMatcher ratio - threshold of 0.85 works well for names
        ratio = SequenceMatcher(None, a, b).ratio()
        return ratio >= 0.85
    
    # Find fuzzy matches among unique names
    fuzzy_groups = []
    used = set()
    
    for i, name1 in enumerate(unique_names):
        if name1 in used:
            continue
        
        group = [name1]
        for name2 in unique_names[i+1:]:
            if name2 in used:
                continue
            if similar(name1, name2):
                group.append(name2)
                used.add(name2)
        
        if len(group) > 1:
            used.add(name1)
            fuzzy_groups.append(group)
    
    # Assign interlock IDs to fuzzy matches
    for group in fuzzy_groups:
        interlock_id += 1
        # Count unique foundations across the fuzzy group
        foundations = set()
        for norm_name in group:
            mask = df['person_name_normalized'] == norm_name
            foundations.update(df[mask]['foundation_ein'].unique())
        
        for norm_name in group:
            interlock_map[norm_name] = {
                'id': interlock_id,
                'count': len(foundations),
                'confidence': 'fuzzy'
            }
    
    # Assign remaining unique entries their own IDs
    for norm_name, data in interlock_map.items():
        if data['id'] is None:
            interlock_id += 1
            interlock_map[norm_name]['id'] = interlock_id
    
    # Apply to dataframe
    df['interlock_id'] = df['person_name_normalized'].apply(lambda x: interlock_map.get(x, {}).get('id', 0))
    df['interlock_count'] = df['person_name_normalized'].apply(lambda x: interlock_map.get(x, {}).get('count', 1))
    df['match_confidence'] = df['person_name_normalized'].apply(lambda x: interlock_map.get(x, {}).get('confidence', 'unique'))
    
    # Reorder columns for output
    output_columns = [
        'person_name',
        'person_name_normalized',
        'foundation_name',
        'foundation_ein',
        'title',
        'hours_per_week',
        'is_officer',
        'is_trustee_director',
        'address_line1',
        'city',
        'state',
        'zip',
        'country',
        'address_type',
        'tax_year',
        'interlock_id',
        'interlock_count',
        'match_confidence',
    ]
    
    # Only include columns that exist
    output_columns = [c for c in output_columns if c in df.columns]
    
    return df[output_columns].sort_values(['interlock_count', 'person_name_normalized'], ascending=[False, True])


def build_board_detail_from_results(parse_results: list, raw_file_bytes: dict) -> pd.DataFrame:
    """
    Build board_detail.csv from parse results.
    
    Args:
        parse_results: List of parse result dicts from processing
        raw_file_bytes: Dict mapping filename -> bytes for XML files
    
    Returns:
        DataFrame ready for export as board_detail.csv
    """
    all_board_members = []
    
    for result in parse_results:
        if result.get('status') == 'error':
            continue
        
        filename = result.get('file', '')
        meta = {
            'foundation_name': result.get('org_name', ''),
            'foundation_ein': result.get('foundation_ein', ''),
            'tax_year': result.get('tax_year', ''),
            'form_type': result.get('form_type', ''),
        }
        
        # Get raw bytes for this file
        file_bytes = raw_file_bytes.get(filename)
        
        if file_bytes and filename.lower().endswith('.xml'):
            # Extract detailed board info from XML
            members = extract_board_details_from_xml(file_bytes, meta)
            all_board_members.extend(members)
    
    if not all_board_members:
        return pd.DataFrame()
    
    # Detect interlocks
    return detect_board_interlocks(all_board_members)




# -----------------------------------------------------------------------------
# Upload validation helpers (pre-parse guardrails)
# -----------------------------------------------------------------------------
def _is_real_xml(data: bytes) -> bool:
    """Return True if bytes look like actual XML markup."""
    # Strip UTF-8 BOM if present, then whitespace
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    head = data.lstrip()[:200]
    return head.startswith(b"<") or head.startswith(b"<?xml")

def _looks_like_mobile_saved_text(data: bytes) -> bool:
    """
    Heuristic for the iPad Safari failure mode where the user saves the rendered
    text view instead of the raw XML bytes from ProPublica.
    """
    if _is_real_xml(data):
        return False

    snippet = data[:6000].decode("utf-8", errors="ignore")
    has_any_angle = ("<" in snippet) and (">" in snippet)
    has_ts = bool(re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", snippet))
    has_einish = bool(re.search(r"\b\d{8,9}\b", snippet))
    return (not has_any_angle) and (has_ts or has_einish)

def validate_uploaded_xml_or_raise(file_bytes: bytes, filename: str) -> None:
    """Raise a ValueError with a friendly message if file isn't valid XML."""
    if _is_real_xml(file_bytes):
        return

    if _looks_like_mobile_saved_text(file_bytes):
        raise ValueError(
            "This file is named .xml but it isn't XML markup (it looks like a rendered text view). "
            "On iPad Safari, tapping the ProPublica 'XML' link and saving often produces this format. "
            "Fix: long-press the XML link and choose 'Download Linked File', then upload the downloaded file."
        )

    start = file_bytes[:160].decode("utf-8", errors="replace")
    raise ValueError(
        "This file is named .xml but it doesn't appear to be valid XML (it doesn't start with '<'). "
        f"File starts with: {start!r}"
    )

def process_uploaded_files(uploaded_files, tax_year_override: str = "", region_spec: dict = None) -> tuple:
    """
    Process uploaded 990-PF/990 files and return canonical outputs.
    
    Uses unified dispatcher (parse_irs_return) for PDF/XML routing.
    Region tagging is applied per-file inside the dispatcher if region_spec is provided.
    """
    all_grants = []
    all_people = []
    foundations_meta = []
    parse_results = []
    raw_xml_bytes = {}  # Track XML file bytes for board detail extraction
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name.lower()
            
            # Track XML file bytes for board detail extraction
            if filename.endswith('.xml'):
                raw_xml_bytes[uploaded_file.name] = file_bytes

            # Guardrail: iPad Safari sometimes saves a rendered text view as ".xml" (not real XML)
            if filename.endswith('.xml'):
                validate_uploaded_xml_or_raise(file_bytes, uploaded_file.name)

            # Route via unified dispatcher (PDF/XML) with region tagging
            result = parse_irs_return(file_bytes, uploaded_file.name, tax_year_override, region_spec=region_spec)
            
            # Extract diagnostics
            diagnostics = result.get('diagnostics', {})
            meta = result.get('foundation_meta', {})
            grants_df = result.get('grants_df', pd.DataFrame())
            people_df = result.get('people_df', pd.DataFrame())
            
            org_name = meta.get('foundation_name', 'Unknown')
            grants_count = len(grants_df)
            
            # Use BoardExtractor if parser didn't find people or found few (PDF only)
            # XML parsers already extract board members reliably
            if not filename.endswith('.xml'):
                if people_df.empty or len(people_df) < 3:
                    board_df = extract_board_with_fallback(file_bytes, uploaded_file.name, meta)
                    if not board_df.empty and len(board_df) > len(people_df):
                        people_df = board_df
                        diagnostics['board_extraction'] = 'BoardExtractor'
                    else:
                        diagnostics['board_extraction'] = 'parser'
                else:
                    diagnostics['board_extraction'] = 'parser'
            else:
                diagnostics['board_extraction'] = 'xml_parser'
            
            people_count = len(people_df)
            
            # Totals: prefer parser diagnostics, else compute from grants_df
            grants_3a_total = (
                diagnostics.get('grants_3a_total')
                or diagnostics.get('computed_total_3a')
                or 0
            )
            grants_3b_total = (
                diagnostics.get('grants_3b_total')
                or diagnostics.get('computed_total_3b')
                or 0
            )

            if (grants_3a_total == 0 and grants_3b_total == 0) and not grants_df.empty and 'grant_amount' in grants_df.columns:
                # Non-PF 990 parsers may not populate 3a/3b; treat as a single pool
                total_amount = float(grants_df['grant_amount'].fillna(0).sum())
            else:
                total_amount = grants_3a_total + grants_3b_total
            
            foundations_meta.append(meta)
            
            if not grants_df.empty:
                # Add source columns to grants
                grants_df = grants_df.copy()
                grants_df['foundation_name'] = org_name
                grants_df['foundation_ein'] = meta.get('foundation_ein', '')
                grants_df['tax_year'] = meta.get('tax_year', '')
                # Rename columns to match expected schema
                if 'amount' in grants_df.columns and 'grant_amount' not in grants_df.columns:
                    grants_df['grant_amount'] = grants_df['amount']
                if 'org_name' in grants_df.columns and 'grantee_name' not in grants_df.columns:
                    grants_df['grantee_name'] = grants_df['org_name']
                all_grants.append(grants_df)
            
            if not people_df.empty:
                people_df = people_df.copy()
                
                # Rename columns from parser format to network_export format
                # Parser outputs: name, title, hours, compensation, benefits
                # network_export expects: person_name, role, org_name, org_ein, tax_year
                if 'name' in people_df.columns and 'person_name' not in people_df.columns:
                    people_df['person_name'] = people_df['name']
                if 'title' in people_df.columns and 'role' not in people_df.columns:
                    people_df['role'] = people_df['title']
                
                people_df['org_name'] = org_name              # network_export expects org_name
                people_df['org_ein'] = meta.get('foundation_ein', '').replace('-', '')  # network_export expects org_ein
                people_df['tax_year'] = meta.get('tax_year', '')
                all_people.append(people_df)
            
            # Determine status
            if grants_count > 0:
                status = "success"
            else:
                status = "no_grants"
            
            # Build parse result with full diagnostics
            parse_results.append({
                "file": uploaded_file.name,
                "status": status,
                "org_name": org_name,
                "ein": meta.get('foundation_ein', ''),
                "tax_year": meta.get('tax_year', ''),
                "grants_count": grants_count,
                "grants_total": total_amount,
                "board_count": people_count,
                "foundation_meta": meta,
                "diagnostics": diagnostics
            })
            
        except Exception as e:
            parse_results.append({
                "file": uploaded_file.name,
                "status": "error",
                "org_name": "",
                "message": str(e),
                "diagnostics": {"errors": [str(e)], "parser_version": "unknown"}
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
    
    # Build board detail with interlock detection
    board_detail_df = build_board_detail_from_results(parse_results, raw_xml_bytes)
    
    return nodes_df, edges_df, combined_grants, foundations_meta, parse_results, board_detail_df
# =============================================================================
# v2.5 Diagnostic Display Functions
# =============================================================================
def get_confidence_color(confidence: str) -> str:
    """Return emoji color code for confidence level."""
    colors = {
        "high": "🟢",
        "medium-high": "🟡",
        "medium": "🟠", 
        "low": "🔴",
        "very_low": "⛔"
    }
    return colors.get(confidence, "⚪")
def get_confidence_badge(conf_dict: dict) -> str:
    """Create a formatted confidence badge."""
    if not conf_dict:
        return "—"
    
    match_pct = conf_dict.get("match_pct", 0)
    status = conf_dict.get("status", "unknown")
    confidence = conf_dict.get("confidence", "unknown")
    color = get_confidence_color(confidence)
    
    return f"{color} {match_pct}% ({status})"

def _safe_int(x, default=0):
    """Safely convert to int."""
    try:
        return int(x)
    except Exception:
        return default

def render_single_file_diagnostics(result: dict, expanded: bool = False):
    """Display diagnostics for a single parsed file with QA confidence scoring."""
    diag = result.get("diagnostics", {})
    meta = result.get("foundation_meta", {})
    grants_df = result.get("grants_df", pd.DataFrame())
    
    # Handle error case first
    if result.get("status") == "error":
        error_msg = result.get("message", "Unknown error")
        filename = result.get("file", "Unknown file")
        # Truncate long error messages for title
        short_error = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
        with st.expander(f"❌ {filename}: {short_error}", expanded=True):
            st.error(f"**Error:** {error_msg}")
            if diag.get("errors"):
                for err in diag["errors"]:
                    st.write(f"• {err}")
        return
    
    # Add form_type for confidence scoring (990-PF is our current parser)
    if "form_type_detected" not in diag:
        diag["form_type_detected"] = "990-PF"
    
    # Compute unified confidence score
    conf = compute_confidence(diag)
    
    # Determine overall status icon based on confidence grade
    grade_icons = {
        "high": "✅",
        "medium": "🟡",
        "low": "🟠",
        "failed": "❌"
    }
    status_icon = grade_icons.get(conf.grade, "❓")
    
    # Override with error icon if actual errors present
    if diag.get("errors"):
        status_icon = "❌"
    
    # Foundation name for display
    foundation_name = result.get("org_name") or "Unknown"
    if len(foundation_name) > 45:
        foundation_name = foundation_name[:42] + "..."
    
    with st.expander(f"{status_icon} {foundation_name}", expanded=expanded):
        # Confidence score header
        st.markdown(f"### 📊 Parser Confidence: `{conf.grade.upper()}` ({conf.score}/100)")
        
        # Show confidence reasons
        if conf.reasons:
            for reason in conf.reasons[:3]:
                st.markdown(f"- {reason}")
        
        # Show penalties if any (collapsed by default)
        if conf.penalties:
            with st.expander("⚠️ Penalties", expanded=False):
                for reason, points in conf.penalties:
                    st.markdown(f"- {reason} ({points})")
        
        st.divider()
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Grants (3a)", f"{diag.get('grants_3a_count', 0):,}")
            st.caption(f"${diag.get('grants_3a_total', 0):,}")
        with col2:
            st.metric("Grants (3b)", f"{diag.get('grants_3b_count', 0):,}")
            st.caption(f"${diag.get('grants_3b_total', 0):,}")
        with col3:
            board_method = diag.get('board_extraction', 'parser')
            board_label = "Board Members"
            if board_method == 'BoardExtractor':
                board_label = "Board Members ✨"  # Indicate enhanced extraction
            st.metric(board_label, diag.get("board_count", 0))

    # Coverage & attachment diagnostics (helps users understand "missing detail" cases)
    sched_i = _safe_int(diag.get("schedule_i_pages_detected", 0), 0)
    if sched_i > 0:
        st.caption(f"Schedule coverage: **Schedule I pages detected:** {sched_i}")

    # Use dispatcher's attachment detection from diagnostics
    if diag.get("attachment_reference_detected"):
        st.warning(
            "⚠️ This return references **attached grant detail** (e.g., 'See attached detail') "
            "that may not be included in the current PDF/XML source. "
            "For complete grant detail, download the full filing (with attachments) from **IRS TEOS**: "
            "https://apps.irs.gov/app/eos/"
        )
        
        # Show detected phrases if available
        phrases = diag.get("attachment_reference_phrases", [])
        if phrases:
            st.caption(f"Detected: {', '.join(phrases[:3])}")
        
        # Totals reconciliation (QA check)
        rep_3a = diag.get('reported_total_3a')
        comp_3a = diag.get('grants_3a_total', 0)
        if rep_3a is not None:
            diff = abs(int(rep_3a) - int(comp_3a))
            pct = (diff / float(rep_3a) * 100) if rep_3a else 0
            match_icon = "✅" if pct <= 1.0 else "⚠️"
            st.markdown(f"**3a Totals Check:** Reported ${rep_3a:,} vs Computed ${comp_3a:,} → {match_icon} {100-pct:.1f}% match")
        
        rep_3b = diag.get('reported_total_3b')
        comp_3b = diag.get('grants_3b_total', 0)
        if rep_3b is not None:
            diff_b = abs(int(rep_3b) - int(comp_3b))
            pct_b = (diff_b / float(rep_3b) * 100) if rep_3b else 0
            match_icon_b = "✅" if pct_b <= 1.0 else "⚠️"
            st.markdown(f"**3b Totals Check:** Reported ${rep_3b:,} vs Computed ${comp_3b:,} → {match_icon_b} {100-pct_b:.1f}% match")
        
        # Format detection
        fmt = diag.get("extraction_format", {})
        if fmt:
            dominant = fmt.get("dominant_format", "unknown")
            fmt_conf = fmt.get("format_confidence", 0)
            
            # Human readable format names
            if "erb" in dominant.lower():
                fmt_display = "Erb-style (status→org)"
            elif "joyce" in dominant.lower():
                fmt_display = "Joyce-style (org→status)"
            else:
                fmt_display = dominant
            
            st.markdown(f"**PDF Format:** {fmt_display} ({fmt_conf*100:.0f}% confidence)")
        
        # Sample grants for verification
        samples = diag.get("sample_grants", [])
        if samples:
            st.markdown("**Sample Grants** (verify against source)")
            sample_data = []
            for s in samples[:3]:
                sample_data.append({
                    "Organization": s.get("org", "")[:35],
                    "Amount": f"${s.get('amount', 0):,}",
                    "Format": "A" if "erb" in s.get("format", "").lower() else "B"
                })
            st.dataframe(pd.DataFrame(sample_data), hide_index=True, use_container_width=True)
        
        # Warnings
        warnings = diag.get("warnings", [])
        if warnings:
            st.warning("**Warnings:**\n" + "\n".join(f"• {w}" for w in warnings))
        
        # Errors
        errors = diag.get("errors", [])
        if errors:
            st.error("**Errors:**\n" + "\n".join(f"• {e}" for e in errors))
        
        # Source type and parser info
        source_type = diag.get("source_type", "pdf")
        form_type = diag.get("form_type_detected", "990-PF")
        pages = diag.get("pages_processed", 0)
        
        # Source badge
        if source_type == "xml":
            source_badge = "📄 XML (high accuracy)"
        else:
            source_badge = f"📑 PDF ({pages} pages) — beta"
        
        st.caption(f"{source_badge} • {form_type} • Parser v{diag.get('parser_version', 'unknown')} • {result.get('file', 'unknown')}")
def render_return_diagnostics(parse_results: list):
    """
    Section 1: Return Diagnostics (unfiltered, per-return health)
    
    Shows parsing success/failure stats and per-file details.
    This is about "did parsing work?" - always unfiltered.
    """
    if not parse_results:
        return
    
    # Get parser version from first result
    parser_version = "unknown"
    if parse_results:
        parser_version = parse_results[0].get("diagnostics", {}).get("parser_version", "unknown")
    
    # Calculate file-level stats
    success_count = sum(1 for r in parse_results if r["status"] == "success")
    error_count = sum(1 for r in parse_results if r["status"] == "error")
    no_grants_count = sum(1 for r in parse_results if r["status"] == "no_grants")
    total_board = sum(r.get("board_count", 0) for r in parse_results)
    
    st.subheader("🔎 Return Diagnostics")
    st.caption(f"Return-level parsing stats (unfiltered). Parser v{parser_version}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Returns Processed", success_count)
    col2.metric("⚠️ Zero-Grant Returns", no_grants_count)
    col3.metric("❌ Errors", error_count)
    col4.metric("👤 Board Members", f"{total_board:,}")
    
    # Clarifying note
    if no_grants_count > 0:
        st.caption(f"*Note: {no_grants_count} filing(s) contained no extractable grant rows — this does not affect the combined dataset totals below.*")
    
    # Individual file results in expander
    with st.expander("📁 Individual file results", expanded=False):
        # Sort: errors first, then warnings, then success
        def sort_key(r):
            if r.get("diagnostics", {}).get("errors"):
                return 0
            if r.get("diagnostics", {}).get("warnings"):
                return 1
            if r.get("status") == "no_grants":
                return 2
            return 3
        
        sorted_results = sorted(parse_results, key=sort_key)
        
        for result in sorted_results:
            render_single_file_diagnostics(result, expanded=False)


def _fmt_money(x: float) -> str:
    """Format money value."""
    try:
        return f"${x:,.0f}"
    except Exception:
        return "$0"


def render_grant_metrics_row(summary: dict, show_other: bool = True):
    """Render a row of grant metrics from a summary dict."""
    buckets = summary["buckets"]
    
    cols = st.columns(4)
    
    # 3a (Paid)
    b = buckets.get("3a_paid", {"count": 0, "amount": 0.0, "label": "3a (Paid)"})
    cols[0].metric(b["label"], f'{b["count"]:,}')
    cols[0].caption(_fmt_money(b["amount"]))
    
    # 3b (Future)
    b = buckets.get("3b_future", {"count": 0, "amount": 0.0, "label": "3b (Future)"})
    cols[1].metric(b["label"], f'{b["count"]:,}')
    cols[1].caption(_fmt_money(b["amount"]))
    
    # Schedule I
    b = buckets.get("schedule_i", {"count": 0, "amount": 0.0, "label": "Schedule I"})
    cols[2].metric(b["label"], f'{b["count"]:,}')
    cols[2].caption(_fmt_money(b["amount"]))
    
    # Total
    cols[3].metric("**Total**", f'{summary["total_count"]:,}')
    cols[3].caption(_fmt_money(summary["total_amount"]))
    
    # Show other if present
    if show_other and summary.get("other_count", 0) > 0:
        st.caption(f"*+ {summary['other_count']:,} other grants ({_fmt_money(summary['other_amount'])})*")


def render_grant_network_results(grants_df: pd.DataFrame, region_def: dict = None,
                                  nodes_df: pd.DataFrame = None, edges_df: pd.DataFrame = None):
    """
    Section 3: Grant Network Results
    
    Shows grant totals that power the network:
    - All Grants (Unfiltered)
    - Region-Filtered Grants (only if region mode active)
    """
    st.subheader("📊 Grant Network Results")
    st.caption("Grant totals that power the network. We show totals for all extracted grants, and (if enabled) the subset matching your region filter.")
    
    if grants_df is None or grants_df.empty:
        st.info("No grant data available.")
        return
    
    # Build canonical summary object (single source of truth)
    net = build_grant_network_summary(grants_df, region_flag_col="region_relevant")
    
    # --- All Grants (Unfiltered) ---
    st.markdown("**All Grants (Unfiltered)**")
    st.caption("Includes every grant extracted from all returns in this run.")
    render_grant_metrics_row(net["all"]["summary"])
    
    # Network objects subline
    # CoreGraph v1: Accept both old (ORG/PERSON) and new (organization/person) formats
    if nodes_df is not None and edges_df is not None and not nodes_df.empty:
        org_count = len(nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]) if "node_type" in nodes_df.columns else 0
        person_count = len(nodes_df[nodes_df["node_type"].str.lower().isin(["person"])]) if "node_type" in nodes_df.columns else 0
        grant_edges = len(edges_df[edges_df["edge_type"].str.lower().isin(["grant"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
        board_edges = len(edges_df[edges_df["edge_type"].str.lower().isin(["board", "board_membership"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
        st.caption(f"*Network objects: {org_count:,} organizations · {person_count:,} people · {grant_edges:,} grant edges · {board_edges:,} board edges*")
    
    # --- Region-Filtered Grants (only if region mode active) ---
    if net["has_region_filter"]:
        st.divider()
        region_name = region_def.get("name", "Region") if region_def else "Region"
        st.markdown(f"**{region_name} Grants (Region-Filtered)**")
        st.caption("Only grants matching the current region filter (used for region-relevant analytics/exports).")
        render_grant_metrics_row(net["region"]["summary"])
    elif region_def and region_def.get("id") != "none":
        st.divider()
        st.info("Region filter is enabled but no region-relevant column found in data.")
    
    return net  # Return for use by analytics


# Keep old function name for backward compatibility
def render_parse_status(parse_results: list, grants_df: pd.DataFrame = None, region_def: dict = None,
                        nodes_df: pd.DataFrame = None, edges_df: pd.DataFrame = None):
    """
    DEPRECATED: Split into render_return_diagnostics() and render_grant_network_results()
    Kept for backward compatibility - calls both in sequence.
    """
    render_return_diagnostics(parse_results)
    st.divider()
    return render_grant_network_results(grants_df, region_def, nodes_df, edges_df)


# =============================================================================
# Help System
# =============================================================================

QUICK_START_GUIDE = """
## Quick Start Guide

### 1. Create a Project
- Click **"➕ New Project"** and give it a descriptive name
- Example: "Great Lakes Funders 2024" or "Water Stewardship Network"

### 2. Upload 990 Filings
- **Best option:** Download XML files from [ProPublica Nonprofit Explorer](https://projects.propublica.org/nonprofits/)
- Upload multiple files at once (up to 50)
- Supported: **990-PF** (private foundations) and **990 Schedule I** (public charities)

### 3. Configure Region (Optional)
- Apply regional tagging to identify grants in specific geographic areas
- Choose a preset (Great Lakes, New England, etc.) or build a custom region

### 4. Download Results
- **ZIP** — Complete export with all files, README, and Polinode-ready data

### Data Source Tips

| Source | Accuracy | Notes |
|--------|----------|-------|
| ProPublica XML | ⭐⭐⭐ Excellent | Best choice - 100% accurate |
| ProPublica PDF | ⭐⭐ Good | Beta - may have minor variance |

### Need More Help?
Click **"Request Support"** below to send us a message.
"""


def log_support_request(email: str, message: str, context: dict = None) -> bool:
    """
    Log a support request to a JSON file.
    
    Creates/appends to demo_data/_support_requests.json
    """
    from datetime import datetime
    import json
    
    log_file = DEMO_DATA_DIR / "_support_requests.json"
    
    try:
        # Load existing requests
        if log_file.exists():
            requests = json.loads(log_file.read_text(encoding="utf-8"))
        else:
            requests = []
        
        # Add new request
        requests.append({
            "timestamp": datetime.now().isoformat(),
            "email": email,
            "message": message,
            "app_version": APP_VERSION,
            "context": context or {},
        })
        
        # Save
        DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_file.write_text(json.dumps(requests, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        return False


def render_help_button():
    """Render help button with popover menu."""
    
    # Use popover if available (Streamlit 1.33+), otherwise dialog
    try:
        with st.popover("❓", help="Help & Support"):
            render_help_content()
    except AttributeError:
        # Fallback for older Streamlit versions
        if st.button("❓ Help", key="help_btn"):
            st.session_state.show_help = True
        
        if st.session_state.get("show_help", False):
            render_help_dialog()


def render_help_content():
    """Render help menu content (used inside popover or dialog)."""
    
    tab1, tab2 = st.tabs(["📖 Quick Start", "💬 Request Support"])
    
    with tab1:
        st.markdown(QUICK_START_GUIDE)
    
    with tab2:
        render_support_form()


def render_support_form():
    """Render the support request form."""
    
    st.markdown("### Request Support")
    st.markdown("Have a question, found a bug, or need help? Let us know!")
    
    email = st.text_input(
        "Your email",
        placeholder="you@example.com",
        key="support_email"
    )
    
    message = st.text_area(
        "How can we help?",
        placeholder="Describe your question, issue, or feedback...",
        height=150,
        key="support_message"
    )
    
    # Optional: include context
    include_context = st.checkbox(
        "Include app state (helps with debugging)",
        value=True,
        key="support_include_context"
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Send", type="primary", key="support_send"):
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
            elif not message.strip():
                st.error("Please describe your question or issue.")
            else:
                # Build context
                context = {}
                if include_context:
                    context = {
                        "current_project": st.session_state.get("current_project", ""),
                        "processed": st.session_state.get("processed", False),
                    }
                
                # Log the request
                success = log_support_request(email, message.strip(), context)
                
                if success:
                    st.success("✅ Support request submitted! We'll get back to you soon.")
                    # Also show mailto link as backup
                    st.caption(f"You can also email us directly at info@connectingforchangellc.com")
                else:
                    # Fallback to mailto
                    st.warning("Could not save request. Please email us directly:")
                    mailto = f"mailto:info@connectingforchangellc.com?subject=OrgGraph Support&body={message[:500]}"
                    st.markdown(f"[📧 Email info@connectingforchangellc.com]({mailto})")
    
    with col2:
        st.caption("Or email us directly at info@connectingforchangellc.com")


def render_help_dialog():
    """Render help as a dialog (fallback for older Streamlit)."""
    
    with st.container():
        st.markdown("---")
        st.markdown("## ❓ Help & Support")
        
        render_help_content()
        
        if st.button("Close Help", key="close_help"):
            st.session_state.show_help = False
            st.rerun()


# =============================================================================
# Region Selector UI
# =============================================================================
def region_selector_ui(project_id: str = None) -> dict:
    """
    Render region selector UI and return selected region definition.
    
    Settings are saved to the project's config.json when changed.
    
    Args:
        project_id: Current project ID. If None, returns no region (off mode).
    
    Returns:
        Region definition dict (from presets or custom), or REGION_PRESETS["none"]
    """
    st.subheader("🗺️ Regional Perspective (optional)")
    st.caption("Apply regional tagging to identify grants in specific geographic areas")
    
    # If no project context, just return none
    if not project_id:
        st.info("Select a project to configure region settings.")
        return REGION_PRESETS["none"]
    
    # Load current project config
    cfg = load_project_config(project_id)
    rf = cfg.get("region_filter", {})
    current_mode = rf.get("mode", "off")
    
    # Mode selector
    mode_options = ["off", "preset", "custom"]
    mode_labels = {
        "off": "Off (show all grants)",
        "preset": "Use preset region",
        "custom": "Custom region",
    }
    
    current_index = mode_options.index(current_mode) if current_mode in mode_options else 0
    
    mode = st.radio(
        "Region mode",
        options=mode_options,
        format_func=lambda x: mode_labels.get(x, x),
        index=current_index,
        horizontal=True,
        help="Region tagging is saved per-project",
        key=f"region_mode_{project_id}"
    )
    
    # Track if config changed
    config_changed = (mode != current_mode)
    
    # Off mode
    if mode == "off":
        if config_changed:
            cfg = update_region_in_config(cfg, mode="off")
            save_project_config(project_id, cfg)
            st.success("✅ Region tagging disabled for this project")
        return REGION_PRESETS["none"]
    
    # Preset mode
    if mode == "preset":
        preset_ids = [k for k in REGION_PRESETS.keys() if k != "none"]
        preset_labels = [REGION_PRESETS[k]["name"] for k in preset_ids]
        
        current_preset = rf.get("preset_key", "")
        if current_preset in preset_ids:
            default_index = preset_ids.index(current_preset)
        else:
            # Default to Great Lakes if available
            default_index = preset_ids.index("great_lakes") if "great_lakes" in preset_ids else 0
        
        choice = st.selectbox(
            "Choose preset region",
            preset_labels,
            index=default_index,
            key=f"preset_choice_{project_id}"
        )
        chosen_id = preset_ids[preset_labels.index(choice)]
        
        selected = REGION_PRESETS[chosen_id]
        
        # Show what's included
        with st.expander("Region details", expanded=False):
            if selected.get("include_us_states"):
                st.write(f"**US States:** {', '.join(selected['include_us_states'])}")
            if selected.get("include_ca_provinces"):
                st.write(f"**Canadian Provinces:** {', '.join(selected['include_ca_provinces'])}")
            if selected.get("notes"):
                st.caption(selected["notes"])
        
        # Save if changed
        if config_changed or chosen_id != current_preset:
            cfg = update_region_in_config(cfg, mode="preset", preset_key=chosen_id)
            save_project_config(project_id, cfg)
            st.success(f"✅ Region set to: {selected['name']}")
        
        return selected
    
    # Custom mode
    st.markdown("**Build custom region**")
    
    # Load current custom settings
    current_us = [s for s in rf.get("custom_admin1_codes", []) if s in US_STATES]
    current_ca = [s for s in rf.get("custom_admin1_codes", []) if s in CA_PROVINCES]
    
    col1, col2 = st.columns(2)
    with col1:
        us_states = st.multiselect(
            "US states to include",
            sorted(US_STATES),
            default=current_us,
            help="Select US states for this region",
            key=f"custom_us_{project_id}"
        )
    with col2:
        ca_provinces = st.multiselect(
            "Canadian provinces/territories",
            sorted(CA_PROVINCES),
            default=current_ca,
            help="Select Canadian provinces for this region",
            key=f"custom_ca_{project_id}"
        )
    
    # Build region definition
    all_codes = list(us_states) + list(ca_provinces)
    country_codes = []
    if us_states:
        country_codes.append("US")
    if ca_provinces:
        country_codes.append("CA")
    
    region_def = {
        "id": "custom",
        "name": f"{cfg.get('project_name', project_id)} Custom Region",
        "source": "project_config",
        "include_us_states": us_states,
        "include_ca_provinces": ca_provinces,
        "include_countries": country_codes,
        "notes": "Custom region from project config",
    }
    
    # Show summary
    if all_codes:
        st.caption(f"Selected: {', '.join(sorted(all_codes))}")
    else:
        st.warning("No states/provinces selected. Region tagging will not be applied.")
    
    # Save button
    if st.button("💾 Save region settings", key=f"save_region_{project_id}"):
        cfg = update_region_in_config(
            cfg,
            mode="custom",
            custom_admin1_codes=all_codes,
            custom_country_codes=country_codes
        )
        save_project_config(project_id, cfg)
        st.success("✅ Custom region saved to project config")
    
    # Return none if nothing selected, otherwise return the region
    if not all_codes:
        return REGION_PRESETS["none"]
    
    return region_def


def render_region_summary(grants_df: pd.DataFrame):
    """Render region tagging summary if region columns exist."""
    if grants_df is None or grants_df.empty:
        return
    
    if "region_relevant" not in grants_df.columns:
        return
    
    summary = get_region_summary(grants_df)
    
    region_name = summary.get("region_name", "")
    if region_name and region_name != "(no region tagging)":
        st.subheader(f"🗺️ {region_name} Grants")
        
        col1, col2 = st.columns(2)
        
        region_count = summary.get('region_relevant_count', 0)
        region_amount = summary.get('region_relevant_amount', 0)
        
        with col1:
            st.metric(f"{region_name} Grants", f"{region_count:,}")
        with col2:
            st.metric(f"{region_name} Amount", f"${region_amount:,.0f}")
# =============================================================================
# Other Rendering Functions
# =============================================================================
def render_network_stats(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Render minimal network object counts as a caption line."""
    if nodes_df.empty and edges_df.empty:
        return
    
    # Count node types
    # CoreGraph v1: Accept both old (ORG/PERSON) and new (organization/person) formats
    org_count = len(nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]) if not nodes_df.empty and "node_type" in nodes_df.columns else 0
    person_count = len(nodes_df[nodes_df["node_type"].str.lower().isin(["person"])]) if not nodes_df.empty and "node_type" in nodes_df.columns else 0
    
    # Count edge types
    grant_count = len(edges_df[edges_df["edge_type"].str.lower().isin(["grant"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    board_count = len(edges_df[edges_df["edge_type"].str.lower().isin(["board", "board_membership"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    
    st.caption(f"Network objects: {org_count:,} organizations · {person_count:,} people · {grant_count:,} grant edges · {board_count:,} board edges")


# Keep for backward compatibility but mark as deprecated
def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, grants_df: pd.DataFrame = None):
    """DEPRECATED: Use render_network_stats instead. Kept for backward compatibility."""
    render_network_stats(nodes_df, edges_df)
def render_analytics(grants_df: pd.DataFrame, region_def: dict = None):
    """Render grant analytics, filtered by region if selected."""
    if grants_df is None or grants_df.empty:
        st.info("No grant data available for analytics")
        return
    
    # Apply region filter if active
    if region_def and region_def.get("id") != "none" and "region_relevant" in grants_df.columns:
        analysis_df = grants_df[grants_df["region_relevant"] == True].copy()
        region_label = f" ({region_def.get('name', 'Region')}-Relevant)"
        filter_note = "Analytics run on region-filtered grants."
    else:
        analysis_df = grants_df
        region_label = ""
        filter_note = "Analytics run on all grants (no region filter applied)."
    
    if analysis_df.empty:
        st.info("No grants in selected region")
        return
    
    st.subheader(f"📈 Grant Analytics{region_label}")
    st.caption(filter_note)
    
    # Use columns to create visual separation
    col1, col2 = st.columns(2)
    
    # --- Top 10 Grantees ---
    with col1:
        st.markdown("#### 🏆 Top 10 Grantees")
        st.caption("By total funding received")
        
        if "grantee_name" in analysis_df.columns and "grant_amount" in analysis_df.columns:
            grantee_totals = analysis_df.groupby("grantee_name")["grant_amount"].sum().sort_values(ascending=False).head(10)
            
            if not grantee_totals.empty:
                for i, (grantee, amount) in enumerate(grantee_totals.items(), 1):
                    st.write(f"**{i}.** {grantee}")
                    st.caption(f"${amount:,.0f}")
            else:
                st.info("No grantee data available")
        else:
            st.info("Missing grantee data columns")
    
    # --- Multi-Funder Grantees ---
    with col2:
        st.markdown("#### 🤝 Multi-Funder Grantees")
        st.caption("Organizations funded by multiple foundations")
        
        if "grantee_name" in analysis_df.columns and "foundation_name" in analysis_df.columns:
            funder_counts = analysis_df.groupby("grantee_name")["foundation_name"].nunique()
            multi_funded = funder_counts[funder_counts > 1].sort_values(ascending=False)
            
            if not multi_funded.empty:
                for grantee, count in multi_funded.head(10).items():
                    st.write(f"**{grantee}**")
                    st.caption(f"{count} funders")
            else:
                st.info("No multi-funder grantees found")
        else:
            st.info("Missing foundation data columns")
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


# =============================================================================
# Polinode Export Helpers
# =============================================================================

def canonicalize_name(name: str) -> str:
    """Canonicalize a display name for Polinode consistency.
    
    - Unicode normalize (NFKC)
    - Strip whitespace
    - Collapse multiple spaces
    - Normalize quotes and hyphens
    """
    import unicodedata
    if not name or pd.isna(name):
        return str(name) if name else ""
    name = str(name)
    # Unicode normalize
    name = unicodedata.normalize('NFKC', name)
    # Normalize curly quotes to straight
    name = name.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    # Normalize various hyphens/dashes to standard hyphen
    name = name.replace('–', '-').replace('—', '-').replace('−', '-')
    # Strip and collapse whitespace
    name = ' '.join(name.split())
    return name


def validate_polinode_export(poli_nodes: pd.DataFrame, poli_edges: pd.DataFrame) -> tuple:
    """Validate Polinode export for consistency before saving.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if poli_nodes.empty:
        return True, errors
    
    # Check 1: Node names must be unique
    if 'Name' in poli_nodes.columns:
        duplicates = poli_nodes[poli_nodes.duplicated(subset='Name', keep=False)]['Name'].unique()
        if len(duplicates) > 0:
            errors.append(f"Duplicate node names ({len(duplicates)}): {', '.join(str(d) for d in duplicates[:5])}")
    
    # Check 2: All edge sources/targets must exist in nodes
    if not poli_edges.empty and 'Name' in poli_nodes.columns and 'Source' in poli_edges.columns:
        node_names = set(poli_nodes['Name'].dropna())
        edge_sources = set(poli_edges['Source'].dropna())
        edge_targets = set(poli_edges['Target'].dropna())
        
        missing_sources = edge_sources - node_names
        missing_targets = edge_targets - node_names
        
        if missing_sources:
            errors.append(f"Edge sources not in nodes ({len(missing_sources)}): {', '.join(str(s) for s in list(missing_sources)[:5])}")
        if missing_targets:
            errors.append(f"Edge targets not in nodes ({len(missing_targets)}): {', '.join(str(t) for t in list(missing_targets)[:5])}")
    
    return len(errors) == 0, errors


def generate_bundle_manifest(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    grants_detail_df: pd.DataFrame = None,
    project_name: str = None
) -> dict:
    """
    Generate manifest.json for Phase 1c ZIP bundle.
    
    Returns:
        Dictionary to be serialized as manifest.json
    """
    # Count node types
    node_counts = {}
    if nodes_df is not None and not nodes_df.empty and 'node_type' in nodes_df.columns:
        node_counts = nodes_df['node_type'].value_counts().to_dict()
    
    # Count edge types
    edge_counts = {}
    if edges_df is not None and not edges_df.empty and 'edge_type' in edges_df.columns:
        edge_counts = edges_df['edge_type'].value_counts().to_dict()
    
    # Build file list
    files = []
    if nodes_df is not None and not nodes_df.empty:
        files.append({"path": "nodes.csv", "rows": len(nodes_df)})
    if edges_df is not None and not edges_df.empty:
        files.append({"path": "edges.csv", "rows": len(edges_df)})
    if grants_detail_df is not None and not grants_detail_df.empty:
        files.append({"path": "grants_detail.csv", "rows": len(grants_detail_df)})
    
    manifest = {
        "schema_version": COREGRAPH_VERSION,
        "bundle_version": "1.0",
        "source_app": SOURCE_APP,
        "app_version": APP_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name or "unnamed",
        "jurisdiction": JURISDICTION,
        "contents": {
            "files": files,
            "node_count": len(nodes_df) if nodes_df is not None else 0,
            "edge_count": len(edges_df) if edges_df is not None else 0,
            "node_types": node_counts,
            "edge_types": edge_counts,
        },
        "polinode": {
            "included": True,
            "files": [
                "polinode/nodes_polinode.csv",
                "polinode/edges_polinode.csv",
                "polinode/polinode_import.xlsx"
            ]
        }
    }
    
    return manifest


def generate_polinode_excel(poli_nodes: pd.DataFrame, poli_edges: pd.DataFrame) -> bytes:
    """Generate a single Excel file with Nodes and Edges sheets for Polinode import.
    
    Returns:
        Excel file as bytes
    """
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        poli_nodes.to_excel(writer, sheet_name='Nodes', index=False)
        poli_edges.to_excel(writer, sheet_name='Edges', index=False)
    
    excel_buffer.seek(0)
    return excel_buffer.getvalue()


def derive_network_roles(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    """
    Derive network role for each node based on edge relationships.
    
    Returns canonical role vocabulary per spec:
    - FUNDER: Organization that gives grants
    - GRANTEE: Organization that receives grants  
    - FUNDER_GRANTEE: Organization that both gives and receives grants
    - BOARD_MEMBER: Individual serving on organization's board
    - ORGANIZATION: Organization with no grant edges (fallback)
    - INDIVIDUAL: Person node not acting as board member
    
    Derivation order (per spec):
    1. BOARD_MEMBER (overrides other roles for people)
    2. FUNDER_GRANTEE (appears in both from_id and to_id)
    3. FUNDER (only in from_id)
    4. GRANTEE (only in to_id)
    5. ORGANIZATION (org with no grant edges)
    6. INDIVIDUAL (person with no board role)
    
    Returns:
        dict mapping node_id -> {code, label, order}
    """
    # Role definitions (canonical vocabulary)
    ROLES = {
        'FUNDER':         {'label': 'Funder',            'order': 1},
        'FUNDER_GRANTEE': {'label': 'Funder + Grantee',  'order': 2},
        'GRANTEE':        {'label': 'Grantee',           'order': 3},
        'ORGANIZATION':   {'label': 'Organization',      'order': 4},
        'BOARD_MEMBER':   {'label': 'Board Member',      'order': 5},
        'INDIVIDUAL':     {'label': 'Individual',        'order': 6},
    }
    
    if edges_df.empty:
        # No edges - all orgs are ORGANIZATION, all people are INDIVIDUAL
        roles = {}
        for _, row in nodes_df.iterrows():
            node_id = row['node_id']
            node_type = str(row.get('node_type', '')).lower()
            # CoreGraph v1: Accept both old and new formats
            if node_type in ['person']:
                code = 'INDIVIDUAL'
            else:
                code = 'ORGANIZATION'
            roles[node_id] = {
                'code': code,
                'label': ROLES[code]['label'],
                'order': ROLES[code]['order']
            }
        return roles
    
    # CoreGraph v1: Accept both old (GRANT/BOARD_MEMBERSHIP) and new (grant/board) formats
    grant_edges = edges_df[edges_df['edge_type'].str.lower().isin(['grant'])]
    board_edges = edges_df[edges_df['edge_type'].str.lower().isin(['board', 'board_membership'])]
    
    funder_ids = set(grant_edges['from_id']) if not grant_edges.empty else set()
    grantee_ids = set(grant_edges['to_id']) if not grant_edges.empty else set()
    board_member_ids = set(board_edges['from_id']) if not board_edges.empty else set()
    
    roles = {}
    for _, row in nodes_df.iterrows():
        node_id = row['node_id']
        node_type = str(row.get('node_type', '')).lower()
        
        # Apply derivation rules in order
        # CoreGraph v1: Accept both old and new formats
        if node_type in ['person']:
            # Rule 1: BOARD_MEMBER (overrides for people)
            # Rule 6: INDIVIDUAL (fallback for people)
            code = 'BOARD_MEMBER' if node_id in board_member_ids else 'INDIVIDUAL'
        else:
            # Rules 2-5 for organizations
            is_funder = node_id in funder_ids
            is_grantee = node_id in grantee_ids
            
            if is_funder and is_grantee:
                code = 'FUNDER_GRANTEE'  # Rule 2
            elif is_funder:
                code = 'FUNDER'          # Rule 3
            elif is_grantee:
                code = 'GRANTEE'         # Rule 4
            else:
                code = 'ORGANIZATION'    # Rule 5
        
        roles[node_id] = {
            'code': code,
            'label': ROLES[code]['label'],
            'order': ROLES[code]['order']
        }
    
    return roles


def convert_to_polinode_format(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple:
    """
    Convert internal node/edge format to Polinode-compatible format.
    
    Polinode requires:
    - Nodes: 'Name' column (unique identifier)
    - Edges: 'Source' and 'Target' columns matching Name values exactly
    
    Features:
    - Name canonicalization (Unicode, whitespace, quotes, hyphens)
    - Deduplication by name (prefers ORG over PERSON for same name)
    - Edges reference canonicalized display names
    
    Returns:
        (polinode_nodes_df, polinode_edges_df)
    """
    if nodes_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Compute network roles for all nodes
    network_roles = derive_network_roles(nodes_df, edges_df)
    
    # Build node_id → canonicalized label mapping, handling duplicates
    id_to_label = {}
    label_to_row = {}  # Track best row per label (prefer organization over person for duplicates)
    label_to_id = {}   # Track node_id for each label (for role lookup)
    
    for idx, row in nodes_df.iterrows():
        node_id = row['node_id']
        raw_label = row.get('label', node_id)
        canon_label = canonicalize_name(raw_label)
        node_type = str(row.get('node_type', '')).lower()
        
        # Handle duplicate labels: prefer organization over person
        # CoreGraph v1: Accept both old (ORG/PERSON) and new (organization/person) formats
        if canon_label in label_to_row:
            existing_type = str(label_to_row[canon_label].get('node_type', '')).lower()
            # If existing is person and this one is org, replace
            if existing_type in ['person'] and node_type in ['org', 'organization']:
                label_to_row[canon_label] = row
                label_to_id[canon_label] = node_id
        else:
            label_to_row[canon_label] = row
            label_to_id[canon_label] = node_id
        
        # Always map this ID to the canonical label
        id_to_label[node_id] = canon_label
    
    # --- Nodes (deduplicated) ---
    poli_nodes_data = []
    for canon_label, row in label_to_row.items():
        node_id = label_to_id.get(canon_label)
        role_info = network_roles.get(node_id, {'code': '', 'label': '', 'order': 99})
        node_dict = {
            'Name': canon_label,
            'Type': row.get('node_type', ''),
            'network_role_code': role_info.get('code', ''),
            'network_role_label': role_info.get('label', ''),
            'network_role_order': role_info.get('order', 99),
        }
        # Add optional attributes
        if 'city' in row.index and pd.notna(row.get('city')):
            node_dict['City'] = row['city']
        if 'region' in row.index and pd.notna(row.get('region')):
            node_dict['Region'] = row['region']
        if 'jurisdiction' in row.index and pd.notna(row.get('jurisdiction')):
            node_dict['Jurisdiction'] = row['jurisdiction']
        if 'tax_id' in row.index and pd.notna(row.get('tax_id')):
            node_dict['Tax ID'] = row['tax_id']
        if 'assets_latest' in row.index and pd.notna(row.get('assets_latest')):
            node_dict['Assets'] = row['assets_latest']
        
        poli_nodes_data.append(node_dict)
    
    poli_nodes = pd.DataFrame(poli_nodes_data)
    
    # --- Edges ---
    if edges_df.empty:
        return poli_nodes, pd.DataFrame()
    
    poli_edges_data = []
    for idx, row in edges_df.iterrows():
        source_label = id_to_label.get(row['from_id'])
        target_label = id_to_label.get(row['to_id'])
        
        # Skip edges where we can't resolve the label
        if not source_label or not target_label:
            continue
        
        edge_dict = {
            'Source': source_label,
            'Target': target_label,
        }
        # Add edge attributes
        if 'edge_type' in row.index and pd.notna(row.get('edge_type')):
            edge_dict['Type'] = row['edge_type']
        if 'amount' in row.index and pd.notna(row.get('amount')):
            edge_dict['Amount'] = row['amount']
        if 'fiscal_year' in row.index and pd.notna(row.get('fiscal_year')):
            edge_dict['Fiscal Year'] = row['fiscal_year']
        if 'purpose' in row.index and pd.notna(row.get('purpose')):
            edge_dict['Purpose'] = row['purpose']
        if 'role' in row.index and pd.notna(row.get('role')):
            edge_dict['Role'] = row['role']
        if 'city' in row.index and pd.notna(row.get('city')):
            edge_dict['City'] = row['city']
        if 'region' in row.index and pd.notna(row.get('region')):
            edge_dict['Region'] = row['region']
        
        poli_edges_data.append(edge_dict)
    
    poli_edges = pd.DataFrame(poli_edges_data)
    
    # Remove duplicate edges
    if not poli_edges.empty:
        poli_edges = poli_edges.drop_duplicates()
    
    return poli_nodes, poli_edges


def filter_data_to_region(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                          grants_df: pd.DataFrame, region_def: dict) -> tuple:
    """
    Filter nodes and edges to only include region-relevant grants.
    
    This ensures exports only contain:
    - Grant edges where grantee is in the selected region
    - Board membership edges (always included - people at funders)
    - Nodes referenced by the filtered edges
    
    Returns:
        (filtered_nodes_df, filtered_edges_df, filtered_grants_df)
    """
    if not region_def or region_def.get("id") == "none":
        return nodes_df, edges_df, grants_df
    
    if grants_df is None or grants_df.empty:
        return nodes_df, edges_df, grants_df
    
    if "region_relevant" not in grants_df.columns:
        return nodes_df, edges_df, grants_df
    
    # Filter grants to region-relevant only
    filtered_grants = grants_df[grants_df["region_relevant"] == True].copy()
    
    if filtered_grants.empty:
        # No region-relevant grants - return empty datasets
        return pd.DataFrame(columns=nodes_df.columns), pd.DataFrame(columns=edges_df.columns), filtered_grants
    
    # Build set of grantee names that are region-relevant
    region_grantees = set(filtered_grants["grantee_name"].dropna().unique())
    
    # Filter edges
    # - Keep all BOARD_MEMBERSHIP edges
    # - Keep only GRANT edges where target (grantee) is in region
    
    if edges_df.empty:
        return nodes_df, edges_df, filtered_grants
    
    # Get node_id → label mapping
    if not nodes_df.empty and "node_id" in nodes_df.columns and "label" in nodes_df.columns:
        id_to_label = dict(zip(nodes_df["node_id"], nodes_df["label"]))
    else:
        id_to_label = {}
    
    def is_region_relevant_edge(row):
        # CoreGraph v1: Accept both old and new edge_type formats
        edge_type = str(row.get("edge_type", "")).lower()
        if edge_type in ["board", "board_membership"]:
            return True
        if edge_type in ["grant"]:
            # Check if target (grantee) is in region
            target_id = row.get("to_id", "")
            target_label = id_to_label.get(target_id, target_id)
            return target_label in region_grantees
        return True  # Include other edge types
    
    filtered_edges = edges_df[edges_df.apply(is_region_relevant_edge, axis=1)].copy()
    
    # Filter nodes to only those referenced by filtered edges
    if not filtered_edges.empty:
        referenced_ids = set(filtered_edges["from_id"].dropna()) | set(filtered_edges["to_id"].dropna())
        filtered_nodes = nodes_df[nodes_df["node_id"].isin(referenced_ids)].copy()
    else:
        filtered_nodes = pd.DataFrame(columns=nodes_df.columns)
    
    return filtered_nodes, filtered_edges, filtered_grants


# =============================================================================
# README Generator for ZIP Bundle
# =============================================================================

def generate_readme(project_name: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                   grants_df: pd.DataFrame = None, region_def: dict = None,
                   board_detail_df: pd.DataFrame = None) -> str:
    """
    Generate README.md content for the ZIP bundle.
    
    Includes:
    - Project metadata
    - File descriptions
    - Column definitions
    - Data provenance
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    # Sanitize project name for filenames
    safe_project_name = project_name.replace(" ", "_").lower() if project_name else "orggraph_export"
    
    # Count stats
    node_count = len(nodes_df) if nodes_df is not None and not nodes_df.empty else 0
    edge_count = len(edges_df) if edges_df is not None and not edges_df.empty else 0
    grant_count = len(grants_df) if grants_df is not None and not grants_df.empty else 0
    board_count = len(board_detail_df) if board_detail_df is not None and not board_detail_df.empty else 0
    
    # Interlock stats
    interlock_info = ""
    if board_detail_df is not None and not board_detail_df.empty and 'interlock_count' in board_detail_df.columns:
        people_on_multiple = len(board_detail_df[board_detail_df['interlock_count'] > 1]['person_name_normalized'].unique())
        if people_on_multiple > 0:
            interlock_info = f"- **Board Interlocks:** {people_on_multiple} people on multiple boards\n"
    
    # Region info
    region_info = ""
    if region_def and region_def.get("id") != "none":
        region_name = region_def.get("name", "Custom Region")
        region_info = f"- **Region Filter:** {region_name}\n"
    
    readme = f"""# {project_name} — OrgGraph Export

**Generated:** {timestamp}  
**Source App:** OrgGraph US v{APP_VERSION}  
**Schema Version:** CoreGraph v{COREGRAPH_VERSION}

## Summary

- **Nodes:** {node_count:,}
- **Edges:** {edge_count:,}
- **Grant Details:** {grant_count:,}
- **Board Members:** {board_count:,}
{interlock_info}{region_info}
---

## Files in This Bundle

### Core Data Files (C4C Schema)

| File | Description |
|------|-------------|
| `{safe_project_name}_nodes.csv` | All organizations and people in the network |
| `{safe_project_name}_edges.csv` | Grant and board membership relationships |
| `{safe_project_name}_grants_detail.csv` | Detailed grant records with amounts, purposes, locations |
| `{safe_project_name}_board_detail.csv` | Board member details with interlock detection |
| `manifest.json` | Bundle metadata (schema version, timestamps, counts) |

### Polinode Import Files

| File | Description |
|------|-------------|
| `polinode/{safe_project_name}_polinode.xlsx` | Excel workbook with Nodes + Edges tabs for direct Polinode import |
| `polinode/{safe_project_name}_polinode_nodes.csv` | Polinode-compatible nodes (Name as primary key) |
| `polinode/{safe_project_name}_polinode_edges.csv` | Polinode-compatible edges (Source/Target reference Name) |

---

## Column Definitions

### nodes.csv

| Column | Type | Description |
|--------|------|-------------|
| `node_id` | string | Unique identifier (namespaced: `orggraph_us:org-123456789`) |
| `node_type` | string | `organization` or `person` |
| `label` | string | Display name |
| `org_type` | string | `funder`, `grantee`, or `funder_grantee` (orgs only) |
| `tax_id` | string | EIN for US organizations |
| `city` | string | City location |
| `region` | string | State/province code |
| `jurisdiction` | string | Country code (US, CA) |
| `source_system` | string | Data source (e.g., `IRS_990`) |
| `source_app` | string | App that created this record (`orggraph_us`) |

### edges.csv

| Column | Type | Description |
|--------|------|-------------|
| `edge_id` | string | Unique identifier |
| `edge_type` | string | `grant` or `board` |
| `from_id` | string | Source node_id (funder or board member) |
| `to_id` | string | Target node_id (grantee or organization) |
| `amount` | number | Grant amount in USD (grant edges only) |
| `fiscal_year` | string | Tax year of the grant |
| `purpose` | string | Grant purpose description |
| `directed` | boolean | Always `true` for grants |
| `weight` | number | Edge weight (defaults to 1) |
| `source_app` | string | App that created this record |

### grants_detail.csv

| Column | Type | Description |
|--------|------|-------------|
| `foundation_name` | string | Funder organization name |
| `foundation_ein` | string | Funder EIN |
| `tax_year` | string | Tax year of filing |
| `grantee_name` | string | Recipient organization name |
| `grantee_city` | string | Recipient city |
| `grantee_state` | string | Recipient state code |
| `grantee_country` | string | Recipient country code |
| `grant_amount` | number | Grant amount in USD |
| `grant_purpose_raw` | string | Purpose as stated in filing |
| `grant_bucket` | string | `3a` (paid), `3b` (future), or `schedule_i` |
| `region_relevant` | boolean | Whether grant matches region filter |
| `source_file` | string | Original filing filename |

### board_detail.csv

| Column | Type | Description |
|--------|------|-------------|
| `person_name` | string | Name as filed in IRS form |
| `person_name_normalized` | string | Cleaned name for matching |
| `foundation_name` | string | Organization they serve |
| `foundation_ein` | string | Organization EIN |
| `title` | string | Position/role on board |
| `hours_per_week` | string | Time commitment |
| `is_officer` | string | Y/N - Is this person an officer |
| `is_trustee_director` | string | Y/N - Is this person a trustee/director |
| `address_line1` | string | Street address (individual or org) |
| `city` | string | City |
| `state` | string | State code |
| `zip` | string | ZIP code |
| `country` | string | Country (US) |
| `address_type` | string | `individual` or `organization` |
| `interlock_id` | number | Group ID for same person across boards |
| `interlock_count` | number | Number of boards (1 = no interlock) |
| `match_confidence` | string | `exact`, `fuzzy`, or `unique` |

**Using board_detail.csv for LinkedIn resolution:**
- Filter by `interlock_count > 1` to find people serving on multiple boards
- Use `city`, `state`, and `foundation_name` to help disambiguate common names
- `address_type = individual` indicates the address is for the person (990-PF only)
- `match_confidence = fuzzy` means names were similar but not exact — verify manually

---

## Using with Polinode

1. Open Polinode (https://app.polinode.com)
2. Create new network → Import from Excel
3. Upload `polinode/{safe_project_name}_polinode.xlsx`
4. Map columns:
   - Nodes tab: `Name` as identifier
   - Edges tab: `Source` and `Target` as endpoints
5. Click Import

The `network_role_label` column can be used for node coloring:
- **Funder** — Organizations that give grants
- **Grantee** — Organizations that receive grants  
- **Funder + Grantee** — Organizations that do both
- **Board Member** — Individuals serving on boards

---

## Data Provenance

- **Source:** IRS Form 990-PF and 990 Schedule I filings
- **Parser:** OrgGraph US v{APP_VERSION}
- **Processing:** Grants extracted, board members identified, interlocks detected
- **Schema:** CoreGraph v{COREGRAPH_VERSION} (normalized node/edge types, namespaced IDs)

For questions about this data, contact: info@connectingforchangellc.com
"""
    return readme


# =============================================================================
# Simplified Download UI (v0.24.0)
# =============================================================================

def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                    grants_df: pd.DataFrame = None, parse_results: list = None,
                    project_name: str = None, region_def: dict = None,
                    board_detail_df: pd.DataFrame = None):
    """
    Simplified download UI with 3 actions:
    - 💾 Save to Project (local folder)
    - ☁️ Save to Cloud (Supabase)
    - 📦 Download ZIP (single bundle with everything)
    
    v0.24.0: Collapsed from 8 buttons to 3
    v0.25.0: Added board_detail.csv with interlock detection
    """
    st.subheader("📥 Export Data")
    
    # Sanitize project name for filenames
    safe_project_name = project_name.replace(" ", "_").lower() if project_name else "orggraph_export"
    
    # Region filtering option (keep this - it's useful)
    if region_def and region_def.get("id") != "none" and grants_df is not None and "region_relevant" in grants_df.columns:
        filter_to_region = st.checkbox(
            f"🗺️ Export only {region_def.get('name', 'region')}-relevant grants",
            value=True,
            help="When checked, exports will only include grants to organizations in the selected region"
        )
        
        if filter_to_region:
            export_nodes, export_edges, export_grants = filter_data_to_region(
                nodes_df, edges_df, grants_df, region_def
            )
            
            # Show filtering stats
            original_grants = len(edges_df[edges_df["edge_type"].str.lower().isin(["grant"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
            filtered_grants = len(export_edges[export_edges["edge_type"].str.lower().isin(["grant"])]) if not export_edges.empty and "edge_type" in export_edges.columns else 0
            
            st.caption(f"Exporting {filtered_grants:,} of {original_grants:,} grants ({region_def.get('name', 'region')}-relevant only)")
        else:
            export_nodes, export_edges, export_grants = nodes_df, edges_df, grants_df
    else:
        export_nodes, export_edges, export_grants = nodes_df, edges_df, grants_df
    
    # Apply unified schema
    export_nodes = prepare_unified_nodes_csv(export_nodes, SOURCE_APP)
    export_edges = prepare_unified_edges_csv(export_edges, SOURCE_APP)
    
    # Prepare Polinode files
    poli_nodes, poli_edges = convert_to_polinode_format(export_nodes, export_edges)
    polinode_excel = None
    if not poli_nodes.empty:
        polinode_excel = generate_polinode_excel(poli_nodes, poli_edges if not poli_edges.empty else pd.DataFrame())
    
    # Prepare grants detail
    grants_detail = None
    if export_grants is not None and not export_grants.empty:
        grants_detail = ensure_grants_detail_columns(export_grants)
    
    st.divider()
    
    # ==========================================================================
    # Three-button layout
    # ==========================================================================
    
    col1, col2, col3 = st.columns(3)
    
    # --- Save to Project (Local) ---
    with col1:
        if project_name and project_name != DEMO_PROJECT_NAME:
            if st.button("💾 Save to Project", use_container_width=True, 
                        help="Save here so you can add more foundations to this project later"):
                project_path = get_project_path(project_name)
                try:
                    export_nodes.to_csv(project_path / "nodes.csv", index=False)
                    export_edges.to_csv(project_path / "edges.csv", index=False)
                    if grants_detail is not None and not grants_detail.empty:
                        grants_detail.to_csv(project_path / "grants_detail.csv", index=False)
                    st.success(f"✅ Saved to project folder")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.button("💾 Save to Project", use_container_width=True, disabled=True,
                     help="Create a project first to enable local saves")
    
    # --- Save to Cloud ---
    with col2:
        client = get_project_store_authenticated()
        cloud_enabled = client is not None
        
        if st.button("☁️ Save to Cloud", 
                    disabled=not cloud_enabled,
                    use_container_width=True,
                    help="Login to enable cloud saves" if not cloud_enabled else "Save to cloud so you can analyze in InsightGraph or share with teammates"):
            
            with st.spinner("☁️ Uploading..."):
                success, message, slug = save_bundle_to_cloud(
                    project_name=project_name,
                    nodes_df=export_nodes,
                    edges_df=export_edges,
                    grants_df=export_grants,
                    region_def=region_def,
                    poli_nodes=poli_nodes,
                    poli_edges=poli_edges,
                    polinode_excel=polinode_excel,
                    parse_results=parse_results
                )
                
                if success:
                    st.success(f"☁️ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # --- Download ZIP (Primary Action) ---
    with col3:
        # Build ZIP bundle
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # README.md
            readme_content = generate_readme(
                project_name=project_name or "OrgGraph Export",
                nodes_df=export_nodes,
                edges_df=export_edges,
                grants_df=grants_detail,
                region_def=region_def,
                board_detail_df=board_detail_df
            )
            zip_file.writestr('README.md', readme_content)
            
            # manifest.json
            manifest = generate_bundle_manifest(
                nodes_df=export_nodes,
                edges_df=export_edges,
                grants_detail_df=grants_detail,
                project_name=project_name
            )
            zip_file.writestr('manifest.json', json.dumps(manifest, indent=2))
            
            # Core data files (with project prefix)
            if not export_nodes.empty:
                zip_file.writestr(f'{safe_project_name}_nodes.csv', export_nodes.to_csv(index=False))
            if not export_edges.empty:
                zip_file.writestr(f'{safe_project_name}_edges.csv', export_edges.to_csv(index=False))
            if grants_detail is not None and not grants_detail.empty:
                zip_file.writestr(f'{safe_project_name}_grants_detail.csv', grants_detail.to_csv(index=False))
            
            # Polinode files
            if not poli_nodes.empty:
                zip_file.writestr(f'polinode/{safe_project_name}_polinode_nodes.csv', poli_nodes.to_csv(index=False))
            if not poli_edges.empty:
                zip_file.writestr(f'polinode/{safe_project_name}_polinode_edges.csv', poli_edges.to_csv(index=False))
            if polinode_excel:
                zip_file.writestr(f'polinode/{safe_project_name}_polinode.xlsx', polinode_excel)
            
            # Board detail with interlock detection
            if board_detail_df is not None and not board_detail_df.empty:
                zip_file.writestr(f'{safe_project_name}_board_detail.csv', board_detail_df.to_csv(index=False))
            
            # Parse log (diagnostics)
            if parse_results:
                zip_file.writestr('parse_log.json', json.dumps(parse_results, indent=2, default=str))
        
        zip_buffer.seek(0)
        
        st.download_button(
            "📦 Download ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{safe_project_name}_export.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
            help="Download everything in one bundle — ready for Polinode, sharing, or offline use"
        )
    
    # Help text
    st.caption("""
    **ZIP contains:** README.md · nodes.csv · edges.csv · grants_detail.csv · board_detail.csv · 
    manifest.json · polinode/ folder with Excel + CSVs
    """)


# =============================================================================
# Upload Interface
# =============================================================================
def render_upload_interface(project_name: str):
    """Render the upload and processing interface for a project."""
    display_name = get_project_display_name(project_name)
    
    # Load existing data (now includes grants_detail)
    existing_nodes, existing_edges, existing_grants_detail = load_project_data(project_name)
    
    # Show existing data status
    if not existing_nodes.empty or not existing_edges.empty:
        grants_count = len(existing_grants_detail) if not existing_grants_detail.empty else 0
        st.success(f"📂 **Existing {display_name} data:** {len(existing_nodes)} nodes, {len(existing_edges)} edges, {grants_count} grant details")
        
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
        
        # =================================================================
        # SECTION 1: Return Diagnostics (unfiltered)
        # =================================================================
        if st.session_state.parse_results:
            render_return_diagnostics(st.session_state.parse_results)
        
        st.divider()
        
        # =================================================================
        # SECTION 2: Merge Results
        # =================================================================
        if st.session_state.merge_stats:
            stats = st.session_state.merge_stats
            st.subheader("🔁 Merge Results")
            st.caption("Dataset merge outcome. Counts below reflect what was added to the combined nodes/edges outputs from this upload.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Nodes:**")
                st.write(f"- Existing: {stats['existing_nodes']}")
                st.write(f"- From this upload: {stats['new_nodes_total']}")
                st.write(f"- ✅ **Added: {stats['nodes_added']}**")
                if stats['nodes_skipped'] > 0:
                    st.write(f"- ⏭️ Skipped: {stats['nodes_skipped']}")
            
            with col2:
                st.markdown("**Edges:**")
                st.write(f"- Existing: {stats['existing_edges']}")
                st.write(f"- From this upload: {stats['new_edges_total']}")
                st.write(f"- ✅ **Added: {stats['edges_added']}**")
                if stats['edges_skipped'] > 0:
                    st.write(f"- ⏭️ Skipped: {stats['edges_skipped']}")
            
            with col3:
                st.markdown("**Grant Details:**")
                st.write(f"- Existing: {stats.get('existing_grants', 0)}")
                st.write(f"- From this upload: {stats.get('new_grants_total', 0)}")
                st.write(f"- ✅ **Added: {stats.get('grants_added', 0)}**")
                if stats.get('grants_skipped', 0) > 0:
                    st.write(f"- ⏭️ Skipped: {stats.get('grants_skipped', 0)}")
        
        st.divider()
        
        # =================================================================
        # SECTION 3: Grant Network Results (All + Region-Filtered)
        # =================================================================
        render_grant_network_results(
            st.session_state.grants_df, 
            st.session_state.region_def,
            st.session_state.nodes_df,
            st.session_state.edges_df
        )
        
        st.divider()
        
        # Combined dataset summary
        grants_count = len(st.session_state.grants_df) if st.session_state.grants_df is not None else 0
        st.success(f"📊 **Combined {display_name} dataset:** {len(st.session_state.nodes_df)} nodes, {len(st.session_state.edges_df)} edges, {grants_count} grant details")
        
        # =================================================================
        # SECTION 4: Analytics (toggle, uses region-filtered if available)
        # =================================================================
        show_analytics = st.checkbox("📈 Show Network Analytics", value=False, 
                                     help="Analytics run on region-filtered grants when available")
        if show_analytics:
            render_analytics(st.session_state.grants_df, st.session_state.region_def)
        
        # =================================================================
        # SECTION 5: Data Preview & Downloads
        # =================================================================
        render_data_preview(st.session_state.nodes_df, st.session_state.edges_df)
        render_downloads(st.session_state.nodes_df, st.session_state.edges_df, 
                       st.session_state.grants_df, st.session_state.parse_results,
                       project_name, st.session_state.region_def,
                       st.session_state.board_detail_df)
        
        # =================================================================
        # SECTION 6: Advanced - Graph Structure (optional expander)
        # =================================================================
        with st.expander("🧩 Advanced: Graph Structure", expanded=False):
            st.caption("Structural counts reflect the full merged dataset (unfiltered).")
            render_network_stats(st.session_state.nodes_df, st.session_state.edges_df)
        
    else:
        # Show upload interface
        st.subheader("📤 Upload IRS 990 Filings")
        
        st.markdown(f"""
        Upload up to **{MAX_FILES} IRS 990 filings** (PDF or XML).
        """)
        
        # Data source guidance
        with st.expander("📚 Data source guide (recommended reading)", expanded=False):
            st.markdown("""
            **Which file format should I use?**
            
            | Source | Format | Accuracy | Recommendation |
            |--------|--------|----------|----------------|
            | **ProPublica XML** | `.xml` | ⭐⭐⭐ Excellent | **Best choice** - 100% accurate, includes grantee EINs |
            | **ProPublica PDF** | `.pdf` | ⭐⭐ Good | Beta - works well but may have minor parsing variance |
            | **IRS direct PDF** | `.pdf` | ⭐⭐ Good | Coming soon - not yet optimized |
            
            **How to get XML files (recommended):**
            1. Go to [ProPublica Nonprofit Explorer](https://projects.propublica.org/nonprofits/)
            2. Search for the foundation by name or EIN
            3. Click on a tax filing year
            4. Look for **"XML"** download link (not PDF)
            
            **Supported form types:**
            - **990-PF** (private foundations) — PDF or XML
            - **990 with Schedule I** (public charities making grants) — XML only
            
            *PDF parsing is in beta. XML is preferred for production use.*
            """)
        
        st.caption("📱 iPad tip: long-press the ProPublica 'XML' link → 'Download Linked File'. Tapping and saving may produce a non-XML text file.")

        uploaded_files = st.file_uploader(
            f"Upload 990 files (max {MAX_FILES})",
            type=["pdf", "xml"],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) > MAX_FILES:
            st.warning(f"⚠️ Max {MAX_FILES} files. Processing first {MAX_FILES}.")
            uploaded_files = uploaded_files[:MAX_FILES]
        
        tax_year_override = st.text_input(
            "Tax Year (optional)",
            help="Override auto-detection if needed"
        )
        
        st.divider()
        
        # Region selector (after upload, before processing)
        region_def = region_selector_ui(project_id=project_name)
        
        # Build region_spec for dispatcher (convert from UI format if needed)
        region_spec = None
        if region_def and region_def.get("id") != "none":
            region_spec = region_def
        
        st.divider()
        
        parse_button = st.button("🔍 Parse 990 Filings", type="primary", disabled=not uploaded_files)
        
        if not uploaded_files:
            st.info("👆 Upload 990 PDF or XML files")
            st.stop()
        
        if parse_button:
            # Process files with region_spec passed to dispatcher
            with st.spinner("Parsing filings..."):
                new_nodes, new_edges, grants_df, foundations_meta, parse_results, board_detail_df = process_uploaded_files(
                    uploaded_files, tax_year_override, region_spec=region_spec
                )
            
            # Note: Region tagging is now done inside the dispatcher per-file
            # No need for post-processing region tagging here
            
            if new_nodes.empty:
                st.warning("No data extracted from uploaded files.")
                # Still store results to show errors
                store_results(
                    pd.DataFrame(), pd.DataFrame(), None,
                    parse_results, {}, [], region_def, pd.DataFrame()
                )
                st.rerun()
            
            # Merge nodes and edges with existing
            nodes_df, edges_df, merge_stats = merge_graph_data(
                existing_nodes, existing_edges, new_nodes, new_edges
            )
            
            # Merge grants_detail with existing (append behavior)
            new_grants_detail = ensure_grants_detail_columns(grants_df)
            merged_grants_df, grants_stats = merge_grants_detail(
                existing_grants_detail, new_grants_detail
            )
            
            # Add grants_detail stats to merge_stats
            merge_stats['existing_grants'] = grants_stats['existing']
            merge_stats['new_grants_total'] = grants_stats['new']
            merge_stats['grants_added'] = grants_stats['added']
            merge_stats['grants_skipped'] = grants_stats['skipped']
            
            # Merge board_detail with existing (if any)
            existing_board_detail = st.session_state.get("board_detail_df", pd.DataFrame())
            if not board_detail_df.empty:
                if not existing_board_detail.empty:
                    # Concatenate and re-run interlock detection
                    combined_members = pd.concat([existing_board_detail, board_detail_df], ignore_index=True)
                    # Re-detect interlocks across all data
                    board_detail_df = detect_board_interlocks(combined_members.to_dict('records'))
                # else: just use the new board_detail_df
            else:
                board_detail_df = existing_board_detail
            
            # Get processed org names
            processed_orgs = [r["org_name"] for r in parse_results if r.get("status") == "success" and r.get("org_name")]
            
            # Store in session state (use merged grants)
            store_results(nodes_df, edges_df, merged_grants_df, parse_results, merge_stats, processed_orgs, region_def, board_detail_df)
            
            # Rerun to show results
            st.rerun()
# =============================================================================
# Main Application
# =============================================================================

# =============================================================================
# C4C CONSOLE UI (Streamlit theming helpers)
# =============================================================================
def inject_c4c_console_ui():
    """Inject C4C console-style CSS. Safe to call multiple times."""
    
# (UI helpers moved to console_ui.py)

def main():
    init_session_state()
    
    # Sidebar: Cloud login/status
    render_cloud_status()
    
    # Get cloud status for header display
    cloud_logged_in, cloud_text = get_cloud_status()
    
    c4c_header(
        title="OrgGraph (US)",
        subtitle="Organization-centered network mapping for grants, board interlocks, and ecosystems.",
        icon_url=APP_ICON_URL,
        right_html=f"<span class='c4c-pill' style='margin-right:8px;'>{cloud_text}</span>{c4c_badge(f'v{APP_VERSION}', 'indigo')}"
    )

    # Quick Start - collapsible, links to guide
    with st.expander("📘 Quick Start", expanded=False):
        st.markdown("""
1. **Create or select a project** below
2. **Upload one or more 990-PF files** (or use demo data)
3. **Run ingestion → export a bundle** (ZIP) for Polinode + InsightGraph

[View full guide](https://www.connectingforchangellc.com/orggraph-us)
        """)

    # ==========================================================================
    # STAGE 1: Project Selection
    # ==========================================================================
    current_project = st.session_state.get('current_project')
    has_data = st.session_state.get('processed', False)
    
    # Stage 1 is complete if we have a project selected
    stage1_complete = current_project is not None
    
    # Determine expander label
    if stage1_complete:
        display_name = current_project.replace('_', ' ').title() if current_project != '_demo' else 'Demo'
        stage1_label = f"✓ Project: {display_name}"
    else:
        stage1_label = "📁 Project"
    
    with st.expander(stage1_label, expanded=not stage1_complete):
        if stage1_complete:
            # Collapsed summary view - show change option
            col1, col2 = st.columns([4, 1])
            with col1:
                display_name = current_project.replace('_', ' ').title() if current_project != '_demo' else 'Demo'
                st.markdown(f"**Active project:** `{current_project}`")
            with col2:
                if st.button("Change", key="stage1_change_btn", type="secondary"):
                    st.session_state.current_project = None
                    clear_session_state()
                    st.rerun()
        else:
            # Full project selection UI
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
                st.markdown("**Create New Project**")
                
                st.caption("""
                **Naming tips:** Use a descriptive name like "Great Lakes Funders 2024" or "Water Stewardship Network". 
                Avoid special characters. The name becomes a folder, so "Great Lakes Funders" → `great_lakes_funders/`
                """)
                
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
            
            # ==========================================================================
            # ADD TO EXISTING PROJECT MODE
            # ==========================================================================
            elif project_mode == "📂 Add to Existing Project":
                st.markdown("**Select Project**")
                
                # Build dropdown options with node/edge counts
                project_options = []
                for p in projects:
                    if not p["is_demo"]:
                        display_name = get_project_display_name(p["name"])
                        if p["has_data"]:
                            nodes_df, edges_df, grants_detail_df = load_project_data(p["name"])
                            display_name += f" ({len(nodes_df)} nodes, {len(edges_df)} edges)"
                        else:
                            display_name += " (empty)"
                        project_options.append((p["name"], display_name))
                
                if not project_options:
                    st.info("No existing projects found. Create a new project first.")
                else:
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
                    
                    if selected_project and st.button("Open Project", type="primary"):
                        st.session_state.current_project = selected_project
                        st.rerun()
            
            # ==========================================================================
            # VIEW DEMO MODE
            # ==========================================================================
            elif project_mode == "👁️ View Demo":
                st.markdown("**Demo Dataset**")
                st.caption(f"📂 Explore sample data from `demo_data/{DEMO_PROJECT_NAME}/`")
                
                if st.button("Load Demo", type="secondary"):
                    st.session_state.current_project = DEMO_PROJECT_NAME
                    # Load demo data into state
                    demo_path = DEMO_DATA_DIR / DEMO_PROJECT_NAME
                    nodes_path = demo_path / "nodes.csv"
                    edges_path = demo_path / "edges.csv"
                    grants_path = demo_path / "grants_detail.csv"
                    
                    if nodes_path.exists() and edges_path.exists():
                        nodes_df = pd.read_csv(nodes_path)
                        edges_df = pd.read_csv(edges_path)
                        grants_df = pd.read_csv(grants_path) if grants_path.exists() else None
                        set_processed_state(nodes_df, edges_df, grants_df, [], {}, [], None)
                    st.rerun()

    # ==========================================================================
    # After Stage 1: Show upload or results based on project state
    # ==========================================================================
    if not current_project:
        # No project selected - stop here
        st.info("👆 Select or create a project above to continue.")
        st.stop()
    
    # Project is selected - show appropriate next stage
    if current_project == DEMO_PROJECT_NAME:
        # Demo mode: show demo data directly
        nodes_df, edges_df, grants_detail_df = load_project_data(DEMO_PROJECT_NAME)
        
        if nodes_df.empty and edges_df.empty:
            st.warning("**No demo data found.** The demo dataset hasn't been set up yet.")
            st.stop()
        
        grants_count = len(grants_detail_df) if not grants_detail_df.empty else 0
        st.success(f"✅ Demo data: {len(nodes_df)} nodes, {len(edges_df)} edges, {grants_count} grant details")
        
        # Show existing foundations
        existing_foundations = get_existing_foundations(nodes_df)
        if existing_foundations:
            with st.expander(f"📋 Foundations in Demo ({len(existing_foundations)})", expanded=False):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        # Use grants_detail.csv if available
        grants_df = grants_detail_df if not grants_detail_df.empty else None
        if grants_df is None and not edges_df.empty and "edge_type" in edges_df.columns:
            grant_edges = edges_df[edges_df["edge_type"].str.lower().isin(["grant"])].copy()
            if not grant_edges.empty:
                grants_df = pd.DataFrame({
                    'foundation_name': grant_edges['from_id'],
                    'grantee_name': grant_edges['to_id'],
                    'grant_amount': grant_edges['amount'],
                    'grantee_state': grant_edges.get('region', ''),
                })
        
        # Render outputs (read-only demo)
        render_graph_summary(nodes_df, edges_df, grants_df)
        
        show_analytics = st.checkbox("📊 Show Network Analytics", value=False)
        if show_analytics:
            render_analytics(grants_df, None)
        
        render_data_preview(nodes_df, edges_df)
        render_downloads(nodes_df, edges_df, grants_df, None, DEMO_PROJECT_NAME, None, None)
    else:
        # Regular project: show upload interface
        render_upload_interface(current_project)


if __name__ == "__main__":
    main()
