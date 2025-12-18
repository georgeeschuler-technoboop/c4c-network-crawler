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
import tempfile
from pathlib import Path
from datetime import datetime
# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# =============================================================================
# Constants
# =============================================================================
APP_VERSION = "0.15.0"  # Added grants_detail.csv project save + append behavior
MAX_FILES = 50
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_25063966d6cd496eb2fe3f6ee5cde0fa~mv2.png"
SOURCE_SYSTEM = "IRS_990"
JURISDICTION = "US"
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
def store_results(nodes_df, edges_df, grants_df, parse_results, merge_stats, processed_orgs, region_def=None):
    """Store processing results in session state."""
    st.session_state.processed = True
    st.session_state.nodes_df = nodes_df
    st.session_state.edges_df = edges_df
    st.session_state.grants_df = grants_df
    st.session_state.parse_results = parse_results
    st.session_state.merge_stats = merge_stats
    st.session_state.processed_orgs = processed_orgs
    st.session_state.region_def = region_def
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
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name.lower()
            
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
    
    return nodes_df, edges_df, combined_grants, foundations_meta, parse_results
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
    foundation_name = result.get("org_name", "Unknown")
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
    if nodes_df is not None and edges_df is not None and not nodes_df.empty:
        org_count = len(nodes_df[nodes_df["node_type"] == "ORG"]) if "node_type" in nodes_df.columns else 0
        person_count = len(nodes_df[nodes_df["node_type"] == "PERSON"]) if "node_type" in nodes_df.columns else 0
        grant_edges = len(edges_df[edges_df["edge_type"] == "GRANT"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
        board_edges = len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
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
- **nodes.csv** — Organizations and people
- **edges.csv** — Grant and board relationships
- **ZIP** — Complete export with grant details and diagnostics

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
    org_count = len(nodes_df[nodes_df["node_type"] == "ORG"]) if not nodes_df.empty and "node_type" in nodes_df.columns else 0
    person_count = len(nodes_df[nodes_df["node_type"] == "PERSON"]) if not nodes_df.empty and "node_type" in nodes_df.columns else 0
    
    # Count edge types
    grant_count = len(edges_df[edges_df["edge_type"] == "GRANT"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    board_count = len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    
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
def convert_to_polinode_format(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple:
    """
    Convert internal node/edge format to Polinode-compatible format.
    
    Polinode requires:
    - Nodes: 'Name' column (unique identifier)
    - Edges: 'Source' and 'Target' columns matching Name values
    
    Returns:
        (polinode_nodes_df, polinode_edges_df)
    """
    if nodes_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Build node_id → label mapping
    id_to_label = dict(zip(nodes_df['node_id'], nodes_df['label']))
    
    # --- Nodes ---
    # Create Name column from label, keep useful attributes
    poli_nodes = pd.DataFrame()
    poli_nodes['Name'] = nodes_df['label']
    poli_nodes['Type'] = nodes_df['node_type']  # ORG or PERSON
    
    # Add optional attributes that Polinode can use
    if 'city' in nodes_df.columns:
        poli_nodes['City'] = nodes_df['city']
    if 'region' in nodes_df.columns:
        poli_nodes['Region'] = nodes_df['region']
    if 'jurisdiction' in nodes_df.columns:
        poli_nodes['Jurisdiction'] = nodes_df['jurisdiction']
    if 'tax_id' in nodes_df.columns:
        poli_nodes['Tax ID'] = nodes_df['tax_id']
    if 'assets_latest' in nodes_df.columns:
        poli_nodes['Assets'] = nodes_df['assets_latest']
    
    # Keep internal ID for reference (prefixed with ! so Polinode doesn't parse it)
    poli_nodes['!Internal ID'] = nodes_df['node_id']
    
    # --- Edges ---
    if edges_df.empty:
        return poli_nodes, pd.DataFrame()
    
    poli_edges = pd.DataFrame()
    
    # Convert from_id/to_id to Source/Target using labels
    poli_edges['Source'] = edges_df['from_id'].map(id_to_label)
    poli_edges['Target'] = edges_df['to_id'].map(id_to_label)
    
    # Add edge attributes
    if 'edge_type' in edges_df.columns:
        poli_edges['Type'] = edges_df['edge_type']  # GRANT or BOARD_MEMBERSHIP
    if 'amount' in edges_df.columns:
        poli_edges['Amount'] = edges_df['amount']
    if 'fiscal_year' in edges_df.columns:
        poli_edges['Fiscal Year'] = edges_df['fiscal_year']
    if 'purpose' in edges_df.columns:
        poli_edges['Purpose'] = edges_df['purpose']
    if 'role' in edges_df.columns:
        poli_edges['Role'] = edges_df['role']
    if 'city' in edges_df.columns:
        poli_edges['City'] = edges_df['city']
    if 'region' in edges_df.columns:
        poli_edges['Region'] = edges_df['region']
    
    # Drop rows where Source or Target couldn't be mapped (shouldn't happen, but safety)
    poli_edges = poli_edges.dropna(subset=['Source', 'Target'])
    
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
        if row.get("edge_type") == "BOARD_MEMBERSHIP":
            return True
        if row.get("edge_type") == "GRANT":
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


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                    grants_df: pd.DataFrame = None, parse_results: list = None,
                    project_name: str = None, region_def: dict = None):
    """Render download buttons with region filtering option."""
    st.subheader("📥 Download Data")
    
    # Region filtering option
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
            original_grants = len(edges_df[edges_df["edge_type"] == "GRANT"]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
            filtered_grants = len(export_edges[export_edges["edge_type"] == "GRANT"]) if not export_edges.empty and "edge_type" in export_edges.columns else 0
            
            st.caption(f"Exporting {filtered_grants:,} of {original_grants:,} grants ({region_def.get('name', 'region')}-relevant only)")
        else:
            export_nodes, export_edges, export_grants = nodes_df, edges_df, grants_df
    else:
        export_nodes, export_edges, export_grants = nodes_df, edges_df, grants_df
    
    # Save to project folder
    if project_name and project_name != DEMO_PROJECT_NAME:
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            if st.button("💾 Save to Project", type="primary"):
                project_path = get_project_path(project_name)
                try:
                    export_nodes.to_csv(project_path / "nodes.csv", index=False)
                    export_edges.to_csv(project_path / "edges.csv", index=False)
                    # Save grants_detail.csv (append behavior handled by merge earlier)
                    if export_grants is not None and not export_grants.empty:
                        grants_detail = ensure_grants_detail_columns(export_grants)
                        grants_detail.to_csv(project_path / "grants_detail.csv", index=False)
                    st.success(f"✅ Saved to `{project_path}/`")
                except Exception as e:
                    st.error(f"Error saving: {e}")
        with save_col2:
            st.caption("Saves nodes.csv, edges.csv, and grants_detail.csv to the project folder")
    
    st.divider()
    
    # Convert to Polinode format
    poli_nodes, poli_edges = convert_to_polinode_format(export_nodes, export_edges)
    
    # Download buttons in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not export_nodes.empty:
            st.download_button(
                "📥 nodes.csv",
                data=export_nodes.to_csv(index=False),
                file_name="nodes.csv",
                mime="text/csv",
                use_container_width=True,
                help="C4C schema format"
            )
    
    with col2:
        if not export_edges.empty:
            st.download_button(
                "📥 edges.csv",
                data=export_edges.to_csv(index=False),
                file_name="edges.csv",
                mime="text/csv",
                use_container_width=True,
                help="C4C schema format"
            )
    
    with col3:
        if not poli_nodes.empty:
            st.download_button(
                "📥 nodes_polinode.csv",
                data=poli_nodes.to_csv(index=False),
                file_name="nodes_polinode.csv",
                mime="text/csv",
                use_container_width=True,
                help="Polinode-compatible format"
            )
    
    with col4:
        if not poli_edges.empty:
            st.download_button(
                "📥 edges_polinode.csv",
                data=poli_edges.to_csv(index=False),
                file_name="edges_polinode.csv",
                mime="text/csv",
                use_container_width=True,
                help="Polinode-compatible format"
            )
    
    # ZIP download with everything
    if not export_nodes.empty or not export_edges.empty:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # C4C schema files
            if not export_nodes.empty:
                zip_file.writestr('nodes.csv', export_nodes.to_csv(index=False))
            if not export_edges.empty:
                zip_file.writestr('edges.csv', export_edges.to_csv(index=False))
            
            # Polinode-compatible files
            if not poli_nodes.empty:
                zip_file.writestr('nodes_polinode.csv', poli_nodes.to_csv(index=False))
            if not poli_edges.empty:
                zip_file.writestr('edges_polinode.csv', poli_edges.to_csv(index=False))
            
            # Grants detail (canonical schema)
            if export_grants is not None and not export_grants.empty:
                grants_detail = ensure_grants_detail_columns(export_grants)
                zip_file.writestr('grants_detail.csv', grants_detail.to_csv(index=False))
            if parse_results:
                zip_file.writestr('parse_log.json', json.dumps(parse_results, indent=2, default=str))
        
        zip_buffer.seek(0)
        
        file_prefix = project_name if project_name else "orggraph"
        
        # Explain what's in the ZIP
        st.caption("""
        **Complete export includes:**
        `nodes.csv` + `edges.csv` + `grants_detail.csv` (C4C schema) •
        `nodes_polinode.csv` + `edges_polinode.csv` (Polinode-ready) •
        `parse_log.json` (diagnostics)
        """)
        
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
                       project_name, st.session_state.region_def)
        
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
                new_nodes, new_edges, grants_df, foundations_meta, parse_results = process_uploaded_files(
                    uploaded_files, tax_year_override, region_spec=region_spec
                )
            
            # Note: Region tagging is now done inside the dispatcher per-file
            # No need for post-processing region tagging here
            
            if new_nodes.empty:
                st.warning("No data extracted from uploaded files.")
                # Still store results to show errors
                store_results(
                    pd.DataFrame(), pd.DataFrame(), None,
                    parse_results, {}, [], region_def
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
            
            # Get processed org names
            processed_orgs = [r["org_name"] for r in parse_results if r.get("status") == "success" and r.get("org_name")]
            
            # Store in session state (use merged grants)
            store_results(nodes_df, edges_df, merged_grants_df, parse_results, merge_stats, processed_orgs, region_def)
            
            # Rerun to show results
            st.rerun()
# =============================================================================
# Main Application
# =============================================================================
def main():
    init_session_state()
    
    # Header
    # Header with logo, title, and help button
    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        st.image(C4C_LOGO_URL, width=80)
    with col2:
        st.title("OrgGraph (US)")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        render_help_button()
    
    st.markdown("""
    OrgGraph currently supports US and Canadian nonprofit registries; additional sources will be added in the future.
    """)
    st.caption(f"App v{APP_VERSION} • Parser v{PARSER_VERSION}")
    
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
                    nodes_df, edges_df, grants_detail_df = load_project_data(p["name"])
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
        
        nodes_df, edges_df, grants_detail_df = load_project_data(DEMO_PROJECT_NAME)
        
        if nodes_df.empty and edges_df.empty:
            st.warning("""
            **No demo data found.**
            
            The demo dataset hasn't been set up yet. Create a new project to get started.
            """)
            st.stop()
        
        grants_count = len(grants_detail_df) if not grants_detail_df.empty else 0
        st.success(f"✅ Demo data: {len(nodes_df)} nodes, {len(edges_df)} edges, {grants_count} grant details")
        
        # Show existing foundations
        existing_foundations = get_existing_foundations(nodes_df)
        if existing_foundations:
            with st.expander(f"📋 Foundations in Demo ({len(existing_foundations)})", expanded=True):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        # Use grants_detail.csv if available, otherwise reconstruct from edges
        grants_df = grants_detail_df if not grants_detail_df.empty else None
        if grants_df is None and not edges_df.empty and "edge_type" in edges_df.columns:
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
            render_analytics(grants_df, None)  # Demo mode has no region_def
        
        render_data_preview(nodes_df, edges_df)
        render_downloads(nodes_df, edges_df, grants_df, None, DEMO_PROJECT_NAME)
if __name__ == "__main__":
    main()
