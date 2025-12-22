"""
OrgGraph (CA) — Canadian Nonprofit Registry Ingestion

Multi-project Streamlit app:
- New Project: Create a new project and upload initial data
- Add to Existing: Select existing project and merge new data
- View Demo: Read-only view of sample demo data

Outputs conform to C4C Network Schema v1 (MVP):
- nodes.csv: ORG and PERSON nodes
- edges.csv: GRANT and BOARD_MEMBERSHIP edges
- grants_detail.csv: Canonical grant detail format (shared with US)

VERSION HISTORY:
----------------
UPDATED v0.12.0: CoreGraph v1 schema normalization (Phase 1a)
- node_type normalized to lowercase: ORG→organization, PERSON→person
- edge_type normalized to lowercase: GRANT→grant, BOARD_MEMBERSHIP→board
- Added org_type field for funder/grantee roles
- Added source_app field for provenance tracking
- Node IDs namespaced: org-123→orggraph_ca:org-123
- Added directed and weight fields to edges

UPDATED v0.11.0: Supabase cloud storage integration
- Added cloud save/load functionality
- User authentication via Supabase
- Projects persist in cloud database

UPDATED v0.10.1: Regional Perspective UI improvements
- Renamed "Region Mode" to "Regional Perspective" for consistency with US
- Added preset regions: Quebec, Prairie Provinces, Atlantic Canada
- Improved UI layout with dropdown selector and expandable details
- Great Lakes now labeled "Great Lakes (Binational)"

UPDATED v0.10.0: Canonical role vocabulary
- Role columns now include: network_role_code, network_role_label, network_role_order
- Labels aligned with spec: "Funder + Grantee" (not "/"), "Individual" (not "Person")
- Codes: FUNDER, GRANTEE, FUNDER_GRANTEE, BOARD_MEMBER, ORGANIZATION, INDIVIDUAL
- Legend order (1-6) for consistent Polinode sorting
- Derivation order documented (BOARD_MEMBER overrides first)

UPDATED v0.9.0: Polinode export parity with OrgGraph US v0.17.0
- Added `Network Role` column: Funder, Grantee, Funder / Grantee, Board Member
- Name canonicalization (Unicode NFKC, whitespace, quotes, hyphens)
- Duplicate node handling (prefers ORG over PERSON)
- Pre-export validation (warns on duplicate names, missing edge refs)
- Excel export with Nodes + Edges sheets for direct Polinode import
- ZIP now uses c4c_schema/ and polinode_schema/ folders

UPDATED v0.8.1: Fixed CA grants column mapping
- Canonical grants_detail.csv schema (same as OrgGraph US)
- Region mode with Great Lakes preset
- grant_bucket = "ca_t3010" for all CA grants

UPDATED v0.8.0: Added Save to Project + batch upload
- Save to Project button (writes directly to demo_data/{project}/)
- Robust currency amount parsing (handles $, CAD, commas, spaces)
- Schema fully aligned with OrgGraph US v0.15.0+
- BATCH UPLOAD: Process 5-10 organizations at once!
  - Upload multiple files per category (assets, directors, grants)
  - Files auto-grouped by CRA Business Number in header
  - Per-org results shown with combined merge stats

UPDATED v0.8.1: Fixed CA grants column mapping
- standardize_ca_grants_columns() normalizes column names before processing
- Handles: Donee Name → Qualified donee, Reported Amount → Cash ($), 
  Gifts In Kind → In-kind ($), Province → Prov, Fiscal Year → Reporting period
- Fixes $0 grant amounts from CRA-style CSV exports (e.g., Consecon)
"""

import streamlit as st
import pandas as pd
import json
import re
import hashlib
import zipfile
import sys
import os
from pathlib import Path
from io import BytesIO, StringIO
from enum import Enum

# =============================================================================
# Config
# =============================================================================

APP_VERSION = "0.12.0"  # CoreGraph v1 schema normalization (Phase 1a)
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_bcf888c01ebe499ca978b82f5291947b~mv2.png"
SOURCE_SYSTEM = "CHARITYDATA_CA"
JURISDICTION = "CA"
CURRENCY = "CAD"
SUPPORT_EMAIL = "info@connectingforchangellc.com"
SOURCE_APP = "orggraph_ca"  # CoreGraph v1 source app identifier

# Demo data paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
DEMO_PROJECT_NAME = "_demo"  # Reserved name for demo dataset

# Add the project root to path for imports (MUST be before c4c_utils imports)
sys.path.insert(0, str(REPO_ROOT))

from c4c_utils.c4c_supabase import C4CSupabase

st.set_page_config(
    page_title="OrgGraph (CA)",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)

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


def render_cloud_status():
    """Render cloud connection status and login UI in sidebar."""
    db = st.session_state.get("supabase_db")
    
    if not db:
        st.sidebar.caption("☁️ Cloud unavailable")
        return None
    
    if db.is_authenticated:
        user = db.get_user()
        # Logged in: show email + logout button
        st.sidebar.caption(f"☁️ {user['email']}")
        if st.sidebar.button("Logout", key="cloud_logout", use_container_width=True):
            db.logout()
            st.rerun()
        return db
    else:
        # Not logged in: show status, then collapsible login form
        st.sidebar.caption("☁️ Not connected")
        with st.sidebar.expander("Login / Sign Up", expanded=False):
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                email = st.text_input("Email", key="cloud_login_email")
                password = st.text_input("Password", type="password", key="cloud_login_pass")
                if st.button("Login", key="cloud_login_btn"):
                    if db.login(email, password):
                        st.success("✅ Logged in!")
                        st.rerun()
                    else:
                        st.error("Login failed")
            
            with tab2:
                st.caption("First time? Create an account.")
                signup_email = st.text_input("Email", key="cloud_signup_email")
                signup_pass = st.text_input("Password", type="password", key="cloud_signup_pass")
                if st.button("Sign Up", key="cloud_signup_btn"):
                    if db.signup(signup_email, signup_pass):
                        st.success("✅ Check email to confirm")
                    else:
                        st.error("Signup failed")
        
        return db


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
                    source_app="orggraph_ca",
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


# =============================================================================
# Regional Perspective (Shared Logic with OrgGraph US)
# =============================================================================

class RegionMode(Enum):
    OFF = "off"
    PRESET = "preset"
    CUSTOM = "custom"

REGION_PRESETS = {
    "great_lakes": {
        "label": "Great Lakes Region",
        "description": "US Great Lakes states + Ontario & Quebec",
        "admin1_codes": ["MI", "OH", "MN", "WI", "IN", "IL", "NY", "PA", "ON", "QC"],
        "country_codes": ["US", "CA"],
    },
    "ontario": {
        "label": "Ontario",
        "description": "Ontario only",
        "admin1_codes": ["ON"],
        "country_codes": ["CA"],
    },
    "quebec": {
        "label": "Quebec",
        "description": "Quebec only",
        "admin1_codes": ["QC"],
        "country_codes": ["CA"],
    },
    "british_columbia": {
        "label": "British Columbia", 
        "description": "British Columbia only",
        "admin1_codes": ["BC"],
        "country_codes": ["CA"],
    },
    "prairies": {
        "label": "Prairie Provinces",
        "description": "Alberta, Saskatchewan, Manitoba",
        "admin1_codes": ["AB", "SK", "MB"],
        "country_codes": ["CA"],
    },
    "atlantic": {
        "label": "Atlantic Canada",
        "description": "NB, NS, PE, NL",
        "admin1_codes": ["NB", "NS", "PE", "NL"],
        "country_codes": ["CA"],
    },
    "canada_all": {
        "label": "All Canada",
        "description": "All Canadian provinces and territories",
        "admin1_codes": ["AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"],
        "country_codes": ["CA"],
    },
}

CA_PROVINCES = ["AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"]

def compute_region_relevant(grantee_state: str, grantee_country: str, 
                           region_mode: RegionMode, admin1_codes: list, country_codes: list) -> bool:
    """Compute whether a grant is region-relevant."""
    if region_mode == RegionMode.OFF:
        return True
    
    state = str(grantee_state).strip().upper() if grantee_state else ""
    country = str(grantee_country).strip().upper() if grantee_country else ""
    
    # Check country
    if country_codes and country and country not in country_codes:
        return False
    
    # Check admin1 (state/province)
    if admin1_codes and state and state not in admin1_codes:
        return False
    
    return True

# =============================================================================
# Canonical grants_detail.csv Schema
# =============================================================================

GRANTS_DETAIL_COLUMNS = [
    "foundation_name", "foundation_ein", "tax_year", "grantee_name",
    "grantee_city", "grantee_state", "grant_amount", "grant_purpose_raw",
    "grant_bucket", "region_relevant", "source_file",
    "grantee_country", "foundation_country", "source_system",
    "grant_amount_cash", "grant_amount_in_kind", "currency",
    "fiscal_year", "reporting_period"
]

GRANT_BUCKET_CA = "ca_t3010"  # CRA T3010 grant disclosures

# =============================================================================
# Canonical Schema Columns
# =============================================================================

NODE_COLUMNS = [
    "node_id", "node_type", "label", "org_slug", "jurisdiction", "tax_id",
    "city", "region", "source_system", "source_ref", "assets_latest", "assets_year",
    "first_name", "last_name",
    "org_type", "source_app"  # CoreGraph v1: org_type for funder/grantee, source_app for provenance
]

EDGE_COLUMNS = [
    "edge_id", "from_id", "to_id", "edge_type",
    "amount", "amount_cash", "amount_in_kind", "currency",
    "fiscal_year", "reporting_period", "purpose",
    "role", "start_date", "end_date", "at_arms_length",
    "city", "region",
    "source_system", "source_ref",
    "directed", "weight", "source_app"  # CoreGraph v1: directed defaults, weight, source_app
]


# =============================================================================
# CoreGraph v1 Helper Functions
# =============================================================================

def namespace_id(node_id: str, source_app: str = SOURCE_APP) -> str:
    """
    Namespace a node ID to prevent collisions across apps.
    
    If the ID already contains ':', it's assumed to be namespaced
    and is returned unchanged (prevents double-prefixing).
    
    Args:
        node_id: Original node ID (e.g., 'org-123')
        source_app: Source app identifier (default: SOURCE_APP)
    
    Returns:
        Namespaced ID (e.g., 'orggraph_ca:org-123')
    """
    if not node_id:
        return node_id
    
    node_id_str = str(node_id)
    
    # Already namespaced? Return as-is
    if ':' in node_id_str:
        return node_id_str
    
    return f"{source_app}:{node_id_str}"


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
            has_data = (item / "nodes.csv").exists() or (item / "edges.csv").exists()
            projects.append({
                "name": item.name,
                "path": item,
                "has_data": has_data,
                "is_demo": item.name == DEMO_PROJECT_NAME
            })
    
    projects.sort(key=lambda x: (not x["is_demo"], x["name"].lower()))
    return projects


def get_project_display_name(project_name: str) -> str:
    """Convert folder name to display name."""
    if project_name == DEMO_PROJECT_NAME:
        return "Demo Dataset"
    return project_name.replace("_", " ").replace("-", " ").title()


def get_folder_name(display_name: str) -> str:
    """Convert display name to folder name."""
    folder = display_name.lower().strip()
    folder = re.sub(r'[^a-z0-9\s]', '', folder)
    folder = re.sub(r'\s+', '_', folder)
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


def get_project_path(project_name: str) -> Path:
    """Get the path for a project folder."""
    return DEMO_DATA_DIR / project_name


def load_project_data(project_name: str) -> tuple:
    """Load existing data from a project folder."""
    project_path = get_project_path(project_name)
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


def get_existing_foundations(nodes_df: pd.DataFrame) -> list:
    """Get list of existing foundations from nodes."""
    if nodes_df.empty or "node_type" not in nodes_df.columns:
        return []
    
    # CoreGraph v1: Accept both old (ORG) and new (organization) formats
    orgs = nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]
    if orgs.empty:
        return []
    
    foundations = []
    for _, row in orgs.iterrows():
        label = row.get("label", "Unknown")
        source = row.get("source_system", "")
        foundations.append((label, source))
    
    return foundations


# =============================================================================
# Parsing Functions
# =============================================================================

HEADER_RE = re.compile(r"^(.*)\s+\((\d{9}RR\d{4})\)\s*$")


def parse_currency_amount(value) -> float:
    """
    Parse a currency amount robustly.
    Handles: numbers, strings with $, commas, spaces, CAD prefix, etc.
    Returns 0.0 if parsing fails.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string and clean
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return 0.0
    
    # Remove currency symbols and common formatting
    s = s.replace("$", "").replace("CAD", "").replace(",", "").replace(" ", "").strip()
    
    # Handle parentheses for negative numbers: (123) -> -123
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _norm_col(c: str) -> str:
    """Normalize column name for matching."""
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")


def standardize_ca_grants_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CRA-style grants CSV columns into the canonical names expected by process_grants_file().
    
    Expected output columns:
      - Qualified donee
      - Cash ($)
      - In-kind ($)
      - Reporting period
      - City
      - Prov
    
    Handles input variations:
      - Donee Name / Qualified donee / Donee
      - Reported Amount ($) / Cash ($)
      - Gifts In Kind / In-kind ($)
      - Fiscal Year / Reporting period
      - Province / Prov
    """
    if df is None or df.empty:
        return df

    # Build lookup of normalized -> original column names
    col_map = {_norm_col(c): c for c in df.columns}

    def pick(*candidates):
        """Find first matching column from candidates."""
        for cand in candidates:
            if cand in col_map:
                return col_map[cand]
        return None

    # Identify source columns (order matters - more specific first)
    c_donee = pick("qualified_donee", "donee_name", "donee_name_organization", "donee")
    c_reported = pick("reported_amount", "reported_amount_", "total_amount", "amount")
    c_cash = pick("cash", "cash_")
    c_inkind = pick("gifts_in_kind", "gifts_in_kind_", "in_kind", "in_kind_")
    c_city = pick("city")
    c_prov = pick("prov", "province", "state_province", "state")
    c_fy = pick("fiscal_year", "fiscalyear", "year", "tax_year", "reporting_period")

    # Create output with canonical columns (preserve originals)
    out = df.copy()

    # Qualified donee
    if c_donee:
        out["Qualified donee"] = out[c_donee].astype(str).fillna("").str.strip()
    elif "Qualified donee" not in out.columns:
        out["Qualified donee"] = ""

    # In-kind amount
    if c_inkind:
        out["In-kind ($)"] = (
            out[c_inkind].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
    elif "In-kind ($)" not in out.columns:
        out["In-kind ($)"] = "0"

    # Cash / Reported amount handling:
    # - If explicit Cash column exists, use it
    # - Otherwise use Reported Amount
    # - If both Reported and In-kind exist, Cash = Reported - In-kind
    if c_cash and "Cash ($)" not in out.columns:
        out["Cash ($)"] = (
            out[c_cash].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
    elif c_reported:
        rep = (
            out[c_reported].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        # If we have in-kind, compute cash = reported - in-kind
        if c_inkind:
            rep_num = pd.to_numeric(rep, errors="coerce").fillna(0)
            ink_num = pd.to_numeric(out["In-kind ($)"], errors="coerce").fillna(0)
            cash_num = (rep_num - ink_num).clip(lower=0)
            out["Cash ($)"] = cash_num.astype(str)
        else:
            out["Cash ($)"] = rep
    elif "Cash ($)" not in out.columns:
        out["Cash ($)"] = "0"

    # Location
    if c_city and "City" not in out.columns:
        out["City"] = out[c_city].astype(str).fillna("").str.strip()
    elif "City" not in out.columns:
        out["City"] = ""
    
    if c_prov and "Prov" not in out.columns:
        out["Prov"] = out[c_prov].astype(str).fillna("").str.strip()
    elif "Prov" not in out.columns:
        out["Prov"] = ""

    # Reporting period (derive from fiscal year if needed)
    if c_fy and "Reporting period" not in out.columns:
        fy = out[c_fy].astype(str).fillna("").str.extract(r"(\d{4})", expand=False).fillna("")
        out["Reporting period"] = fy.apply(lambda y: f"FY {y}" if y else "")
    elif "Reporting period" not in out.columns:
        out["Reporting period"] = ""

    return out


def _parse_charitydata_content(content: str) -> tuple:
    """Parse charitydata.ca CSV content."""
    lines = content.split("\n")
    
    if not lines:
        return pd.DataFrame(), "", ""
    
    header_line = lines[0].strip().lstrip("\ufeff").strip().strip('"').rstrip(",")
    m = HEADER_RE.match(header_line)
    
    if m:
        org_name = m.group(1).strip()
        cra_bn = m.group(2)
    else:
        org_name = header_line
        cra_bn = ""
    
    remaining = "\n".join(lines[1:])
    if remaining.strip():
        df = pd.read_csv(StringIO(remaining))
    else:
        df = pd.DataFrame()
    
    return df, org_name, cra_bn


def read_charitydata_csv_from_upload(uploaded_file) -> tuple:
    """Read a charitydata.ca CSV from Streamlit upload."""
    content = uploaded_file.getvalue().decode("utf-8")
    return _parse_charitydata_content(content)


def extract_org_identity_from_file(uploaded_file) -> tuple:
    """
    Extract org name and CRA BN from a charitydata.ca file header without fully parsing.
    Returns: (org_name, cra_bn, file_type)
    """
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.split("\n")
    
    if not lines:
        return "", "", "unknown"
    
    # Parse header line
    header_line = lines[0].strip().lstrip("\ufeff").strip().strip('"').rstrip(",")
    m = HEADER_RE.match(header_line)
    
    if m:
        org_name = m.group(1).strip()
        cra_bn = m.group(2)
    else:
        org_name = header_line
        cra_bn = ""
    
    # Detect file type from content
    file_type = "unknown"
    if len(lines) > 1:
        second_line = lines[1].lower() if len(lines) > 1 else ""
        if "assets" in second_line or "total revenue" in second_line:
            file_type = "assets"
        elif "director" in second_line or "trustee" in second_line or "position" in second_line or "first name" in second_line:
            file_type = "directors"
        elif "donee" in second_line or "qualified donee" in second_line or "cash" in second_line:
            file_type = "grants"
    
    # Reset file pointer for later use
    uploaded_file.seek(0)
    
    return org_name, cra_bn, file_type


def group_files_by_organization(all_files: list) -> dict:
    """
    Group uploaded files by organization (CRA BN).
    
    Returns: {
        cra_bn: {
            "org_name": str,
            "assets": file or None,
            "directors": file or None,
            "grants": file or None,
        }
    }
    """
    orgs = {}
    
    for f in all_files:
        if f is None:
            continue
        
        org_name, cra_bn, file_type = extract_org_identity_from_file(f)
        
        # Use CRA BN as key, or org_name if no BN
        key = cra_bn if cra_bn else org_name
        if not key:
            continue
        
        if key not in orgs:
            orgs[key] = {
                "org_name": org_name,
                "cra_bn": cra_bn,
                "assets": None,
                "directors": None,
                "grants": None,
            }
        
        # Assign file to appropriate slot (allow override if same type uploaded twice)
        if file_type == "assets":
            orgs[key]["assets"] = f
        elif file_type == "directors":
            orgs[key]["directors"] = f
        elif file_type == "grants":
            orgs[key]["grants"] = f
        else:
            # Unknown type - try to infer from filename
            fname = f.name.lower() if hasattr(f, 'name') else ""
            if "asset" in fname:
                orgs[key]["assets"] = f
            elif "director" in fname or "trustee" in fname:
                orgs[key]["directors"] = f
            elif "grant" in fname:
                orgs[key]["grants"] = f
    
    return orgs


def process_batch_organizations(
    org_files: dict,
    region_mode: RegionMode,
    admin1_codes: list,
    country_codes: list
) -> tuple:
    """
    Process multiple organizations in batch.
    
    Args:
        org_files: dict from group_files_by_organization()
        region_mode: RegionMode enum
        admin1_codes: list of state/province codes
        country_codes: list of country codes
    
    Returns: (all_nodes_df, all_edges_df, all_grants_detail_df, batch_stats)
    """
    all_nodes = []
    all_edges = []
    all_grants_detail = []
    
    batch_stats = {
        "orgs_processed": 0,
        "orgs_failed": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "total_grants": 0,
        "org_results": []  # List of per-org stats
    }
    
    for key, files in org_files.items():
        org_name = files["org_name"]
        cra_bn = files["cra_bn"]
        
        try:
            # Process this organization's files
            nodes_df, edges_df, org_attributes = process_uploaded_files(
                files["assets"],
                files["directors"],
                files["grants"]
            )
            
            if nodes_df.empty:
                batch_stats["orgs_failed"] += 1
                batch_stats["org_results"].append({
                    "org_name": org_name,
                    "cra_bn": cra_bn,
                    "status": "failed",
                    "error": "No data extracted"
                })
                continue
            
            # Build grants_detail if grants file present
            grants_detail_df = pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS)
            if files["grants"]:
                files["grants"].seek(0)
                grants_df, _, _ = read_charitydata_csv_from_upload(files["grants"])
                if not grants_df.empty:
                    source_filename = files["grants"].name if hasattr(files["grants"], 'name') else "grants.csv"
                    grants_detail_df = build_grants_detail_from_grants_csv(
                        grants_df=grants_df,
                        foundation_name=org_attributes.get("org_legal_name", org_name),
                        foundation_ein=org_attributes.get("tax_id", cra_bn),
                        source_file=source_filename,
                        region_mode=region_mode,
                        admin1_codes=admin1_codes,
                        country_codes=country_codes
                    )
            
            # Accumulate results
            all_nodes.append(nodes_df)
            all_edges.append(edges_df)
            if not grants_detail_df.empty:
                all_grants_detail.append(grants_detail_df)
            
            # Track stats
            batch_stats["orgs_processed"] += 1
            batch_stats["total_nodes"] += len(nodes_df)
            batch_stats["total_edges"] += len(edges_df)
            batch_stats["total_grants"] += len(grants_detail_df)
            batch_stats["org_results"].append({
                "org_name": org_attributes.get("org_legal_name", org_name),
                "cra_bn": org_attributes.get("tax_id", cra_bn),
                "status": "success",
                "nodes": len(nodes_df),
                "edges": len(edges_df),
                "grants": len(grants_detail_df)
            })
            
        except Exception as e:
            batch_stats["orgs_failed"] += 1
            batch_stats["org_results"].append({
                "org_name": org_name,
                "cra_bn": cra_bn,
                "status": "failed",
                "error": str(e)
            })
    
    # Combine all results
    combined_nodes = pd.concat(all_nodes, ignore_index=True) if all_nodes else pd.DataFrame(columns=NODE_COLUMNS)
    combined_edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame(columns=EDGE_COLUMNS)
    combined_grants = pd.concat(all_grants_detail, ignore_index=True) if all_grants_detail else pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS)
    
    # Deduplicate nodes by node_id
    if not combined_nodes.empty:
        combined_nodes = combined_nodes.drop_duplicates(subset=["node_id"], keep="first")
    
    # Deduplicate edges by edge_id
    if not combined_edges.empty:
        combined_edges = combined_edges.drop_duplicates(subset=["edge_id"], keep="first")
    
    return combined_nodes, combined_edges, combined_grants, batch_stats


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
    """Lightweight slug for org names."""
    text = (text or "").strip().lower()
    text = re.sub(r"&|\+", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def clean_nan(val) -> str:
    """Clean up 'nan' strings."""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def extract_fiscal_year(reporting_period: str) -> int:
    """Extract year from reporting period."""
    if not reporting_period:
        return None
    match = re.search(r"(\d{4})", str(reporting_period))
    if match:
        return int(match.group(1))
    return None


def deterministic_person_id(first: str, last: str, org_slug: str) -> str:
    """Generate deterministic person node ID."""
    raw = f"{first}|{last}|{org_slug}".lower()
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"person-{h}"


def deterministic_grant_edge_id(from_id: str, to_id: str, amount: float, fiscal_year: int) -> str:
    """Generate deterministic grant edge ID."""
    raw = f"{from_id}|{to_id}|{amount}|{fiscal_year}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"grant-{h}"


def deterministic_board_edge_id(from_id: str, to_id: str, role: str) -> str:
    """Generate deterministic board edge ID."""
    raw = f"{from_id}|{to_id}|{role}".lower()
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"board-{h}"


# =============================================================================
# Data Processing Functions
# =============================================================================

def process_directors_file(directors_df: pd.DataFrame, org_slug: str, cra_bn: str) -> tuple:
    """
    Process directors/trustees CSV into canonical format.
    
    Handles two charitydata.ca export formats:
    1. Combined name: "Director / Trustee" or "Name" column
    2. Separate columns: "Last Name", "First Name", "Position", etc.
    """
    nodes = []
    edges = []
    
    # Detect format: separate columns vs combined name
    has_separate_names = "Last Name" in directors_df.columns and "First Name" in directors_df.columns
    has_combined_name = "Director / Trustee" in directors_df.columns or "Name" in directors_df.columns
    
    year_col = latest_year_column(directors_df)
    
    for _, row in directors_df.iterrows():
        # Extract name based on format
        if has_separate_names:
            # Format 2: Separate columns (newer charitydata.ca export)
            last_name = str(row.get("Last Name", "")).strip()
            first_name = str(row.get("First Name", "")).strip()
            
            # Skip if no name
            if not last_name and not first_name:
                continue
            if pd.isna(last_name) and pd.isna(first_name):
                continue
            
            # Clean up NaN strings
            if last_name.lower() == "nan":
                last_name = ""
            if first_name.lower() == "nan":
                first_name = ""
            
            # Get role from Position column
            position = str(row.get("Position", "")).strip()
            if position.lower() == "nan" or not position:
                role = "Director/Trustee"
            else:
                role = position.title()
            
            # Get at arms length
            at_arms_length = str(row.get("At Arm's Length", "")).strip().upper() == "Y"
            
            # Get dates
            start_date = clean_nan(str(row.get("Appointed", "")))
            end_date = clean_nan(str(row.get("Ceased", "")))
            
        elif has_combined_name:
            # Format 1: Combined name column (older format)
            name_field = row.get("Director / Trustee") or row.get("Name") or ""
            if not name_field or pd.isna(name_field):
                continue
            
            # Check year column for active status
            if year_col and year_col in row.index:
                active_val = row.get(year_col)
                if pd.isna(active_val) or str(active_val).strip() == "":
                    continue
            
            # Parse name (LASTNAME, FIRSTNAME format)
            name_str = str(name_field).strip()
            parts = name_str.split(",", 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_name = parts[1].strip()
            else:
                name_parts = name_str.split()
                first_name = name_parts[0] if name_parts else ""
                last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            
            role = "Director/Trustee"
            at_arms_length = True
            start_date = ""
            end_date = ""
        else:
            # Unknown format - skip
            continue
        
        # Skip if we still don't have a valid name
        if not first_name and not last_name:
            continue
        
        # Generate IDs
        person_id = deterministic_person_id(first_name, last_name, org_slug)
        org_id = f"org-{cra_bn}" if cra_bn else f"org-{org_slug}"
        
        # Create person node (CoreGraph v1: lowercase node_type, namespaced ID)
        person_node = {col: "" for col in NODE_COLUMNS}
        person_node.update({
            "node_id": namespace_id(person_id),
            "node_type": "person",  # CoreGraph v1: lowercase
            "label": f"{first_name} {last_name}".strip(),
            "first_name": first_name,
            "last_name": last_name,
            "jurisdiction": JURISDICTION,
            "source_system": SOURCE_SYSTEM,
            "source_app": SOURCE_APP,  # CoreGraph v1: provenance
        })
        nodes.append(person_node)
        
        # Create board edge (CoreGraph v1: lowercase edge_type, namespaced IDs)
        edge_id = deterministic_board_edge_id(person_id, org_id, role)
        
        board_edge = {col: "" for col in EDGE_COLUMNS}
        board_edge.update({
            "edge_id": edge_id,
            "from_id": namespace_id(person_id),
            "to_id": namespace_id(org_id),
            "edge_type": "board",  # CoreGraph v1: lowercase (was BOARD_MEMBERSHIP)
            "role": role,
            "start_date": start_date,
            "end_date": end_date,
            "at_arms_length": "Y" if at_arms_length else "N",
            "source_system": SOURCE_SYSTEM,
            "source_app": SOURCE_APP,  # CoreGraph v1: provenance
            "directed": True,  # CoreGraph v1: explicit default
            "weight": 1,  # CoreGraph v1: explicit default
        })
        edges.append(board_edge)
    
    return nodes, edges


def process_grants_file(grants_df: pd.DataFrame, org_slug: str, cra_bn: str) -> tuple:
    """Process grants CSV into canonical format."""
    nodes = []
    edges = []
    
    # Standardize column names (handles Donee Name → Qualified donee, etc.)
    grants_df = standardize_ca_grants_columns(grants_df)
    
    if grants_df.empty:
        return nodes, edges
    
    org_id = f"org-{cra_bn}" if cra_bn else f"org-{org_slug}"
    
    for _, row in grants_df.iterrows():
        grantee_name = row.get("Qualified donee") or row.get("Donee") or ""
        if not grantee_name or pd.isna(grantee_name):
            continue
        
        # Extract values with robust parsing
        cash = parse_currency_amount(row.get("Cash ($)", 0))
        in_kind = parse_currency_amount(row.get("In-kind ($)", 0))
        total_amount = cash + in_kind
        
        # Get reporting period
        reporting_period = clean_nan(row.get("Reporting period", ""))
        fiscal_year = extract_fiscal_year(reporting_period)
        
        # Get location
        city = clean_nan(row.get("City", ""))
        province = clean_nan(row.get("Prov", ""))
        
        # Create grantee org node (CoreGraph v1: lowercase node_type, namespaced ID)
        grantee_slug = slugify_loose(str(grantee_name))
        grantee_id = f"org-{grantee_slug}"
        
        grantee_node = {col: "" for col in NODE_COLUMNS}
        grantee_node.update({
            "node_id": namespace_id(grantee_id),
            "node_type": "organization",  # CoreGraph v1: lowercase (was ORG)
            "org_type": "grantee",  # CoreGraph v1: functional role
            "label": str(grantee_name).strip(),
            "org_slug": grantee_slug,
            "city": city,
            "region": province,
            "jurisdiction": JURISDICTION,
            "source_system": SOURCE_SYSTEM,
            "source_app": SOURCE_APP,  # CoreGraph v1: provenance
        })
        nodes.append(grantee_node)
        
        # Create grant edge (CoreGraph v1: lowercase edge_type, namespaced IDs)
        edge_id = deterministic_grant_edge_id(org_id, grantee_id, total_amount, fiscal_year or 0)
        
        grant_edge = {col: "" for col in EDGE_COLUMNS}
        grant_edge.update({
            "edge_id": edge_id,
            "from_id": namespace_id(org_id),
            "to_id": namespace_id(grantee_id),
            "edge_type": "grant",  # CoreGraph v1: lowercase (was GRANT)
            "amount": total_amount,
            "amount_cash": cash,
            "amount_in_kind": in_kind,
            "currency": CURRENCY,
            "fiscal_year": fiscal_year or "",
            "reporting_period": reporting_period,
            "city": city,
            "region": province,
            "source_system": SOURCE_SYSTEM,
            "source_app": SOURCE_APP,  # CoreGraph v1: provenance
            "directed": True,  # CoreGraph v1: explicit default
            "weight": 1,  # CoreGraph v1: explicit default
        })
        edges.append(grant_edge)
    
    return nodes, edges


def build_grants_detail_from_grants_csv(
    grants_df: pd.DataFrame,
    foundation_name: str,
    foundation_ein: str,
    source_file: str,
    region_mode: RegionMode = RegionMode.OFF,
    admin1_codes: list = None,
    country_codes: list = None
) -> pd.DataFrame:
    """
    Build grants_detail.csv rows from CA grants CSV.
    
    Maps charitydata.ca columns to canonical grants_detail schema:
    - Donee Name / Qualified donee → grantee_name
    - Prov → grantee_state
    - City → grantee_city
    - Cash ($) + In-kind ($) → grant_amount
    - grant_purpose_raw = "" (CA data doesn't have purpose text)
    - grant_bucket = "ca_t3010"
    """
    if grants_df.empty:
        return pd.DataFrame(columns=GRANTS_DETAIL_COLUMNS)
    
    # Standardize column names (handles Donee Name → Qualified donee, etc.)
    grants_df = standardize_ca_grants_columns(grants_df)
    
    rows = []
    
    for _, row in grants_df.iterrows():
        # Get grantee name (handle different column names)
        grantee_name = row.get("Qualified donee") or row.get("Donee") or row.get("Donee Name") or ""
        if not grantee_name or pd.isna(grantee_name):
            continue
        
        # Get amounts with robust parsing
        cash = parse_currency_amount(row.get("Cash ($)", 0))
        in_kind = parse_currency_amount(row.get("In-kind ($)", 0))
        total_amount = cash + in_kind
        
        # Get location
        city = str(row.get("City", "")).strip()
        if city.lower() == "nan":
            city = ""
        province = str(row.get("Prov", "")).strip()
        if province.lower() == "nan":
            province = ""
        
        # Get reporting period / fiscal year
        reporting_period = str(row.get("Reporting period", "")).strip()
        if reporting_period.lower() == "nan":
            reporting_period = ""
        fiscal_year = None
        if reporting_period:
            match = re.search(r"(\d{4})", reporting_period)
            if match:
                fiscal_year = int(match.group(1))
        
        # Compute region relevance
        is_relevant = compute_region_relevant(
            grantee_state=province,
            grantee_country="CA",
            region_mode=region_mode,
            admin1_codes=admin1_codes or [],
            country_codes=country_codes or []
        )
        
        # Build canonical row
        detail_row = {
            "foundation_name": foundation_name,
            "foundation_ein": foundation_ein,
            "tax_year": str(fiscal_year) if fiscal_year else "",
            "grantee_name": str(grantee_name).strip(),
            "grantee_city": city,
            "grantee_state": province,
            "grant_amount": total_amount,
            "grant_purpose_raw": "",  # CA data doesn't have purpose text
            "grant_bucket": GRANT_BUCKET_CA,
            "region_relevant": is_relevant,
            "source_file": source_file,
            "grantee_country": "CA",
            "foundation_country": "CA",
            "source_system": SOURCE_SYSTEM,
            "grant_amount_cash": cash,
            "grant_amount_in_kind": in_kind,
            "currency": CURRENCY,
            "fiscal_year": fiscal_year if fiscal_year else "",
            "reporting_period": reporting_period,
        }
        rows.append(detail_row)
    
    return pd.DataFrame(rows, columns=GRANTS_DETAIL_COLUMNS)


def process_uploaded_files(assets_file, directors_file, grants_file) -> tuple:
    """Process uploaded charitydata.ca files."""
    
    all_nodes = []
    all_edges = []
    org_name = ""
    cra_bn = ""
    total_assets = None
    latest_year = None
    
    # Parse assets file first (contains org metadata)
    if assets_file:
        assets_df, org_name, cra_bn = read_charitydata_csv_from_upload(assets_file)
        latest_year, total_assets = extract_total_assets(assets_df)
    
    # Try to get org info from other files if not in assets
    if not org_name and directors_file:
        _, org_name, cra_bn = read_charitydata_csv_from_upload(directors_file)
        directors_file.seek(0)  # Reset for later use
    
    if not org_name and grants_file:
        _, org_name, cra_bn = read_charitydata_csv_from_upload(grants_file)
        grants_file.seek(0)  # Reset for later use
    
    if not org_name:
        org_name = "Unknown Organization"
    
    org_slug = slugify_loose(org_name)
    org_id = f"org-{cra_bn}" if cra_bn else f"org-{org_slug}"
    
    # Create org node (CoreGraph v1: lowercase node_type, namespaced ID)
    org_node = {col: "" for col in NODE_COLUMNS}
    org_node.update({
        "node_id": namespace_id(org_id),
        "node_type": "organization",  # CoreGraph v1: lowercase (was ORG)
        "org_type": "funder",  # CoreGraph v1: functional role
        "label": org_name,
        "org_slug": org_slug,
        "jurisdiction": JURISDICTION,
        "tax_id": cra_bn,
        "source_system": SOURCE_SYSTEM,
        "source_app": SOURCE_APP,  # CoreGraph v1: provenance
        "assets_latest": total_assets if total_assets is not None else "",
        "assets_year": latest_year if latest_year is not None else "",
    })
    all_nodes.append(org_node)
    
    # Process directors
    if directors_file:
        directors_df, _, _ = read_charitydata_csv_from_upload(directors_file)
        if not directors_df.empty:
            person_nodes, board_edges = process_directors_file(directors_df, org_slug, cra_bn)
            all_nodes.extend(person_nodes)
            all_edges.extend(board_edges)
    
    # Process grants
    if grants_file:
        grants_df, _, _ = read_charitydata_csv_from_upload(grants_file)
        if not grants_df.empty:
            grantee_nodes, grant_edges = process_grants_file(grants_df, org_slug, cra_bn)
            all_nodes.extend(grantee_nodes)
            all_edges.extend(grant_edges)
    
    # Create DataFrames
    nodes_df = pd.DataFrame(all_nodes, columns=NODE_COLUMNS)
    edges_df = pd.DataFrame(all_edges, columns=EDGE_COLUMNS)
    
    # Deduplicate nodes by node_id
    if not nodes_df.empty:
        nodes_df = nodes_df.drop_duplicates(subset=["node_id"], keep="first")
    
    org_attributes = {
        "org_id": org_id,
        "org_slug": org_slug,
        "jurisdiction": JURISDICTION,
        "source_system": SOURCE_SYSTEM,
        "org_legal_name": org_name,
        "org_display_name": org_name.title() if org_name else "",
        "tax_id": cra_bn,
        "source_url": "",
        "assets_latest": total_assets if total_assets is not None else "",
        "assets_year": latest_year if latest_year is not None else "",
        "notes": ""
    }
    
    return nodes_df, edges_df, org_attributes


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


# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_network_results(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """
    Render network results summary.
    Aligned with OrgGraph US patterns for UI consistency.
    """
    if nodes_df is None or nodes_df.empty:
        st.warning("No graph data loaded.")
        return
    
    st.subheader("📊 Network Results")
    st.caption("Network constructed from uploaded charity data")
    
    # CoreGraph v1: Accept both old (ORG/PERSON) and new (organization/person) formats
    org_nodes = len(nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]) if "node_type" in nodes_df.columns else 0
    person_nodes = len(nodes_df[nodes_df["node_type"].str.lower().isin(["person"])]) if "node_type" in nodes_df.columns else 0
    grant_edges = len(edges_df[edges_df["edge_type"].str.lower().isin(["grant"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    board_edges = len(edges_df[edges_df["edge_type"].str.lower().isin(["board", "board_membership"])]) if not edges_df.empty and "edge_type" in edges_df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏛️ Organizations", org_nodes)
    col2.metric("👤 People", person_nodes)
    col3.metric("💰 Grant Edges", grant_edges)
    col4.metric("🪪 Board Edges", board_edges)
    
    # Calculate total grant funding if available
    if not edges_df.empty and "edge_type" in edges_df.columns and "amount" in edges_df.columns:
        grant_df = edges_df[edges_df["edge_type"].str.lower().isin(["grant"])].copy()
        if not grant_df.empty:
            grant_df["amount"] = pd.to_numeric(grant_df["amount"], errors="coerce").fillna(0)
            total_funding = grant_df["amount"].sum()
            if total_funding > 0:
                st.caption(f"*Total grant funding: ${total_funding:,.0f} CAD*")


def render_grant_analytics(edges_df: pd.DataFrame) -> None:
    """
    Render grant analytics section.
    Aligned with OrgGraph US patterns for UI consistency.
    """
    if edges_df is None or edges_df.empty:
        return
    
    if "edge_type" not in edges_df.columns:
        return
    
    # CoreGraph v1: Accept both old (GRANT) and new (grant) formats
    grant_df = edges_df[edges_df["edge_type"].str.lower().isin(["grant"])].copy()
    if grant_df.empty:
        return
    
    st.subheader("📈 Grant Analytics")
    st.caption("Analysis of grant relationships in the network")
    
    # Ensure amount is numeric
    if "amount" in grant_df.columns:
        grant_df["amount"] = pd.to_numeric(grant_df["amount"], errors="coerce").fillna(0)
    
    # Use columns for visual separation (aligned with US app)
    col1, col2 = st.columns(2)
    
    # --- Top Grantees ---
    with col1:
        st.markdown("#### 🏆 Top 10 Grantees")
        st.caption("By total funding received")
        
        if "to_id" in grant_df.columns and "amount" in grant_df.columns:
            grantee_totals = grant_df.groupby("to_id")["amount"].sum().sort_values(ascending=False).head(10)
            
            if not grantee_totals.empty:
                for i, (grantee, amount) in enumerate(grantee_totals.items(), 1):
                    # Clean up the grantee ID for display
                    display_name = str(grantee).replace("ORG:", "").replace("_", " ").title()
                    st.write(f"**{i}.** {display_name}")
                    st.caption(f"${amount:,.0f} CAD")
            else:
                st.info("No grantee data available")
        else:
            st.info("Missing grantee data columns")
    
    # --- Top Grantors ---
    with col2:
        st.markdown("#### 🎁 Top 10 Grantors")
        st.caption("By total funding given")
        
        if "from_id" in grant_df.columns and "amount" in grant_df.columns:
            grantor_totals = grant_df.groupby("from_id")["amount"].sum().sort_values(ascending=False).head(10)
            
            if not grantor_totals.empty:
                for i, (grantor, amount) in enumerate(grantor_totals.items(), 1):
                    # Clean up the grantor ID for display
                    display_name = str(grantor).replace("ORG:", "").replace("_", " ").title()
                    st.write(f"**{i}.** {display_name}")
                    st.caption(f"${amount:,.0f} CAD")
            else:
                st.info("No grantor data available")
        else:
            st.info("Missing grantor data columns")


def render_data_preview(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """
    Render data preview in expanders.
    Aligned with OrgGraph US patterns for UI consistency.
    """
    with st.expander("👀 Preview Nodes", expanded=False):
        if not nodes_df.empty:
            display_cols = ["node_type", "label", "jurisdiction", "city", "region", "tax_id", "source_system"]
            display_cols = [c for c in display_cols if c in nodes_df.columns]
            st.dataframe(nodes_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"{len(nodes_df)} total nodes")
        else:
            st.info("No nodes to display")
    
    with st.expander("👀 Preview Edges", expanded=False):
        if not edges_df.empty:
            display_cols = ["edge_type", "from_id", "to_id", "amount", "role", "fiscal_year", "source_system"]
            display_cols = [c for c in display_cols if c in edges_df.columns]
            st.dataframe(edges_df[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"{len(edges_df)} total edges")
        else:
            st.info("No edges to display")


def render_graph_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """
    DEPRECATED: Split into render_network_results, render_grant_analytics, render_data_preview.
    Kept for backward compatibility.
    """
    render_network_results(nodes_df, edges_df)
    
    # Show analytics toggle (aligned with US app)
    show_analytics = st.checkbox("📈 Show Grant Analytics", value=False)
    if show_analytics:
        render_grant_analytics(edges_df)
    
    render_data_preview(nodes_df, edges_df)


# =============================================================================
# Help System
# =============================================================================

QUICK_START_GUIDE = """
## Quick Start Guide

### 1. Create a Project
- Click **"➕ New Project"** and give it a descriptive name
- Example: "BC Foundations Network" or "Ontario Grantmakers 2024"

### 2. Get Data from charitydata.ca
1. Go to [charitydata.ca](https://www.charitydata.ca/)
2. Search for a Canadian charity by name or BN
3. Download the CSV files:
   - **assets.csv** — Financial information
   - **directors-trustees.csv** — Board members
   - **grants.csv** — Grants made (for foundations)

### 3. Upload Files
- Upload one or more CSV files for an organization
- The app will extract and merge the data automatically

### 4. Download Results
- **nodes.csv** — Organizations and people
- **edges.csv** — Grant and board relationships
- **Polinode format** — Ready to import into Polinode for visualization

### Data Quality
charitydata.ca provides clean, structured data from CRA filings. No parsing variance — what you see is what's reported.

### Need More Help?
Click **"Request Support"** below to send us a message.
"""


def log_support_request(email: str, message: str, context: dict = None) -> bool:
    """Log a support request to a JSON file."""
    from datetime import datetime
    
    log_file = DEMO_DATA_DIR / "_support_requests.json"
    
    try:
        if log_file.exists():
            requests = json.loads(log_file.read_text(encoding="utf-8"))
        else:
            requests = []
        
        requests.append({
            "timestamp": datetime.now().isoformat(),
            "email": email,
            "message": message,
            "app_version": APP_VERSION,
            "app": "CA",
            "context": context or {},
        })
        
        DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_file.write_text(json.dumps(requests, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def render_help_button():
    """Render help button with popover menu."""
    try:
        with st.popover("❓", help="Help & Support"):
            render_help_content()
    except AttributeError:
        if st.button("❓ Help", key="help_btn"):
            st.session_state.show_help = True
        
        if st.session_state.get("show_help", False):
            render_help_dialog()


def render_help_content():
    """Render help menu content."""
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
                context = {}
                if include_context:
                    context = {
                        "current_project": st.session_state.get("current_project", ""),
                    }
                
                success = log_support_request(email, message.strip(), context)
                
                if success:
                    st.success("✅ Support request submitted! We'll get back to you soon.")
                    st.caption(f"You can also email us directly at {SUPPORT_EMAIL}")
                else:
                    st.warning("Could not save request. Please email us directly:")
                    st.markdown(f"[📧 Email {SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL}?subject=OrgGraph%20CA%20Support)")
    
    with col2:
        st.caption(f"Or email us directly at {SUPPORT_EMAIL}")


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
# Export Formats (Polinode) — Aligned with OrgGraph US v0.17.0
# =============================================================================

def canonicalize_name(name: str) -> str:
    """
    Canonicalize a name for Polinode export.
    
    Handles:
    - Unicode normalization (NFKC)
    - Leading/trailing whitespace
    - Multiple internal spaces
    - Smart quotes → straight quotes
    - En/em dashes → hyphens
    """
    import unicodedata
    if not name or pd.isna(name):
        return ""
    
    # Unicode normalize
    s = unicodedata.normalize('NFKC', str(name))
    
    # Smart quotes to straight
    s = s.replace('"', '"').replace('"', '"')
    s = s.replace(''', "'").replace(''', "'")
    
    # Dashes to hyphen
    s = s.replace('–', '-').replace('—', '-')
    
    # Whitespace cleanup
    s = ' '.join(s.split())
    
    return s.strip()


def validate_polinode_export(poli_nodes: pd.DataFrame, poli_edges: pd.DataFrame) -> tuple:
    """
    Validate Polinode export for common issues.
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    # Check for duplicate names
    if not poli_nodes.empty and 'Name' in poli_nodes.columns:
        dupes = poli_nodes[poli_nodes['Name'].duplicated(keep=False)]
        if not dupes.empty:
            dupe_names = dupes['Name'].unique()[:5]
            errors.append(f"Duplicate node names ({len(dupes)}): {', '.join(str(n) for n in dupe_names)}")
    
    # Check for edges referencing missing nodes
    if not poli_edges.empty and not poli_nodes.empty:
        node_names = set(poli_nodes['Name'])
        missing_sources = set(poli_edges['Source']) - node_names
        missing_targets = set(poli_edges['Target']) - node_names
        
        if missing_sources:
            errors.append(f"Edge sources not in nodes ({len(missing_sources)}): {', '.join(str(s) for s in list(missing_sources)[:5])}")
        if missing_targets:
            errors.append(f"Edge targets not in nodes ({len(missing_targets)}): {', '.join(str(t) for t in list(missing_targets)[:5])}")
    
    return len(errors) == 0, errors


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
    - Network Role derived from edge relationships
    
    Returns:
        (polinode_nodes_df, polinode_edges_df)
    """
    if nodes_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Compute network roles for all nodes
    network_roles = derive_network_roles(nodes_df, edges_df)
    
    # Build node_id → canonicalized label mapping, handling duplicates
    id_to_label = {}
    label_to_row = {}  # Track best row per label (prefer ORG over PERSON for duplicates)
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


def render_downloads(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                     grants_detail_df: pd.DataFrame = None, project_name: str = None) -> None:
    """Render download buttons with C4C, Polinode, and grants_detail formats."""
    
    if nodes_df is None or nodes_df.empty:
        return
    
    st.divider()
    st.subheader("📥 Download Data")
    
    # Check if we have grants_detail
    has_grants_detail = grants_detail_df is not None and not grants_detail_df.empty
    
    # Save options (Local + Cloud)
    if project_name and project_name != DEMO_PROJECT_NAME:
        # DEBUG: Show what data we have
        st.caption(f"🔍 Debug: nodes={len(nodes_df) if nodes_df is not None else 'None'}, edges={len(edges_df) if edges_df is not None else 'None'}, grants={len(grants_detail_df) if grants_detail_df is not None and not grants_detail_df.empty else 'None/Empty'}")
        
        st.markdown("**💾 Save Project**")
        
        save_col1, save_col2, save_col3 = st.columns([2, 2, 1])
        
        # Local save
        with save_col1:
            if st.button("💾 Save to Local", type="primary", use_container_width=True):
                project_path = get_project_path(project_name)
                try:
                    nodes_df.to_csv(project_path / "nodes.csv", index=False)
                    edges_df.to_csv(project_path / "edges.csv", index=False)
                    if has_grants_detail:
                        grants_detail_df.to_csv(project_path / "grants_detail.csv", index=False)
                    st.success(f"✅ Saved to `{project_path}/`")
                    st.caption("Files saved: nodes.csv, edges.csv" + (", grants_detail.csv" if has_grants_detail else ""))
                except Exception as e:
                    st.error(f"Error saving: {e}")
        
        # Cloud save
        with save_col2:
            db = st.session_state.get("supabase_db")
            cloud_enabled = db and db.is_authenticated
            
            if st.button("☁️ Save to Cloud", 
                        disabled=not cloud_enabled,
                        use_container_width=True,
                        help="Login to enable cloud save" if not cloud_enabled else "Save to Supabase"):
                save_to_cloud(
                    project_name=project_name,
                    nodes_df=nodes_df,
                    edges_df=edges_df,
                    grants_df=grants_detail_df
                )
        
        with save_col3:
            if not cloud_enabled:
                st.caption("☁️ Login to enable")
            else:
                st.caption("☁️ Ready")
        
        st.divider()
    
    # Generate Polinode format
    poli_nodes, poli_edges = convert_to_polinode_format(nodes_df, edges_df)
    
    # Validate Polinode export
    is_valid, validation_errors = validate_polinode_export(poli_nodes, poli_edges)
    if not is_valid:
        st.warning("⚠️ **Polinode Export Warnings:**")
        for err in validation_errors:
            st.caption(f"  • {err}")
    
    # --- C4C Schema Section ---
    st.markdown("**📁 C4C Schema** (for Insight Engine)")
    c4c_col1, c4c_col2, c4c_col3 = st.columns(3)
    
    with c4c_col1:
        if not nodes_df.empty:
            st.download_button(
                "📥 nodes.csv",
                data=nodes_df.to_csv(index=False),
                file_name="nodes.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with c4c_col2:
        if not edges_df.empty:
            st.download_button(
                "📥 edges.csv",
                data=edges_df.to_csv(index=False),
                file_name="edges.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with c4c_col3:
        if has_grants_detail:
            st.download_button(
                "📥 grants_detail.csv",
                data=grants_detail_df.to_csv(index=False),
                file_name="grants_detail.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # --- Polinode Import Section ---
    st.markdown("**🔗 Polinode Import** (for visualization)")
    poli_col1, poli_col2, poli_col3 = st.columns(3)
    
    with poli_col1:
        if not poli_nodes.empty and not poli_edges.empty:
            try:
                excel_bytes = generate_polinode_excel(poli_nodes, poli_edges)
                st.download_button(
                    "📥 Polinode.xlsx (recommended)",
                    data=excel_bytes,
                    file_name="polinode_import.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Single Excel file with Nodes + Edges sheets"
                )
            except Exception as e:
                st.caption(f"Excel export unavailable: {e}")
    
    with poli_col2:
        if not poli_nodes.empty:
            st.download_button(
                "📥 nodes_polinode.csv",
                data=poli_nodes.to_csv(index=False),
                file_name="nodes_polinode.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with poli_col3:
        if not poli_edges.empty:
            st.download_button(
                "📥 edges_polinode.csv",
                data=poli_edges.to_csv(index=False),
                file_name="edges_polinode.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # ZIP download with everything (organized folders)
    if not nodes_df.empty or not edges_df.empty:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # C4C schema files (in c4c_schema/ folder)
            if not nodes_df.empty:
                zip_file.writestr('c4c_schema/nodes.csv', nodes_df.to_csv(index=False))
            if not edges_df.empty:
                zip_file.writestr('c4c_schema/edges.csv', edges_df.to_csv(index=False))
            if has_grants_detail:
                zip_file.writestr('c4c_schema/grants_detail.csv', grants_detail_df.to_csv(index=False))
            
            # Polinode-compatible files (in polinode_schema/ folder)
            if not poli_nodes.empty:
                zip_file.writestr('polinode_schema/nodes_polinode.csv', poli_nodes.to_csv(index=False))
            if not poli_edges.empty:
                zip_file.writestr('polinode_schema/edges_polinode.csv', poli_edges.to_csv(index=False))
            # Add Excel file to polinode folder
            if not poli_nodes.empty and not poli_edges.empty:
                try:
                    excel_bytes = generate_polinode_excel(poli_nodes, poli_edges)
                    zip_file.writestr('polinode_schema/polinode_import.xlsx', excel_bytes)
                except:
                    pass
        
        zip_buffer.seek(0)
        
        file_prefix = project_name if project_name else "orggraph_ca"
        
        st.caption("""
        **Complete export includes:**
        `c4c_schema/` — nodes.csv, edges.csv, grants_detail.csv •
        `polinode_schema/` — nodes_polinode.csv, edges_polinode.csv, polinode_import.xlsx
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
            with st.expander(f"📋 Organizations already in {display_name} ({len(existing_foundations)})", expanded=False):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        st.caption("New data will be merged. Duplicates automatically skipped.")
    else:
        st.info(f"📂 **No existing {display_name} data.** This will be the first organization.")
    
    st.divider()
    st.subheader("📤 Upload charitydata.ca Files")
    
    # Data source guidance
    with st.expander("📚 Data source guide", expanded=False):
        st.markdown("""
        **How to get data from charitydata.ca:**
        
        1. Go to [charitydata.ca](https://www.charitydata.ca/)
        2. Search for a Canadian charity by name or Business Number (BN)
        3. Click on the charity to view its profile
        4. Download the CSV files you need:
        
        | File | Contains | Required? |
        |------|----------|-----------|
        | **assets.csv** | Financial information, total assets | Recommended |
        | **directors-trustees.csv** | Board members | Optional |
        | **grants.csv** | Grants made (for foundations) | Optional |
        
        **Batch Upload:** You can upload files for multiple organizations at once!
        Files are automatically grouped by the organization name in the header.
        
        **Data Quality:** charitydata.ca provides clean, structured data directly from CRA T3010 filings. 
        Unlike PDF parsing, there's no extraction variance — the data is exactly as reported.
        """)
    
    st.markdown("**Batch Upload:** Upload CSV files for **one or more organizations**")
    st.caption("Files are automatically grouped by the organization header (CRA Business Number)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        assets_files = st.file_uploader(
            "assets.csv files", 
            type=["csv"], 
            accept_multiple_files=True,
            help="Upload one or more assets.csv files"
        )
    with col2:
        directors_files = st.file_uploader(
            "directors-trustees.csv files", 
            type=["csv"], 
            accept_multiple_files=True,
            help="Upload one or more directors-trustees.csv files"
        )
    with col3:
        grants_files = st.file_uploader(
            "grants.csv files", 
            type=["csv"], 
            accept_multiple_files=True,
            help="Upload one or more grants.csv files"
        )
    
    # Combine all uploaded files
    all_files = (assets_files or []) + (directors_files or []) + (grants_files or [])
    
    if not all_files:
        st.info("👆 Upload CSV files for one or more organizations")
        st.stop()
    
    # Group files by organization
    org_files = group_files_by_organization(all_files)
    
    if not org_files:
        st.warning("Could not identify any organizations from uploaded files.")
        st.stop()
    
    # Show detected organizations
    st.divider()
    st.subheader(f"🏢 Detected Organizations ({len(org_files)})")
    
    for key, files in org_files.items():
        file_types = []
        if files["assets"]:
            file_types.append("📊 assets")
        if files["directors"]:
            file_types.append("👥 directors")
        if files["grants"]:
            file_types.append("💰 grants")
        
        st.write(f"🇨🇦 **{files['org_name']}** ({files['cra_bn'] or 'no BN'}) — {', '.join(file_types)}")
    
    # Regional Perspective Selection (for grants_detail)
    st.divider()
    st.subheader("🗺️ Regional Perspective (optional)")
    st.caption("Apply regional tagging to identify grants in specific geographic areas")
    
    region_mode_options = {
        "Off (show all grants)": ("off", None),
        "Use preset region": ("preset", None),
    }
    
    col1, col2 = st.columns([1, 2])
    with col1:
        region_choice = st.radio(
            "Region mode",
            ["Off (show all grants)", "Use preset region"],
            horizontal=False,
            label_visibility="collapsed"
        )
    
    if region_choice == "Use preset region":
        with col2:
            preset_options = {
                "Great Lakes (Binational)": "great_lakes",
                "Ontario": "ontario",
                "Quebec": "quebec",
                "British Columbia": "british_columbia",
                "Prairie Provinces": "prairies",
                "Atlantic Canada": "atlantic",
                "All Canada": "canada_all",
            }
            selected_preset = st.selectbox(
                "Choose preset region",
                list(preset_options.keys()),
                label_visibility="visible"
            )
            preset_key = preset_options[selected_preset]
            preset = REGION_PRESETS[preset_key]
            admin1_codes = preset["admin1_codes"]
            country_codes = preset["country_codes"]
            
            with st.expander("Region details"):
                st.caption(f"**Includes:** {', '.join(admin1_codes)}")
                st.caption(f"**Countries:** {', '.join(country_codes)}")
        
        region_mode = RegionMode.PRESET
    else:
        admin1_codes = []
        country_codes = []
        region_mode = RegionMode.OFF
    
    st.divider()
    
    # Process all organizations in batch
    with st.spinner(f"Processing {len(org_files)} organization(s)..."):
        new_nodes, new_edges, new_grants_detail, batch_stats = process_batch_organizations(
            org_files=org_files,
            region_mode=region_mode,
            admin1_codes=admin1_codes,
            country_codes=country_codes
        )
    
    if new_nodes.empty:
        st.warning("Could not extract data from uploaded files.")
        st.stop()
    
    # Show batch processing results
    st.subheader("✅ Batch Processing Complete")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Organizations", batch_stats["orgs_processed"])
    col2.metric("Total Nodes", batch_stats["total_nodes"])
    col3.metric("Total Edges", batch_stats["total_edges"])
    col4.metric("Total Grants", batch_stats["total_grants"])
    
    if batch_stats["orgs_failed"] > 0:
        st.warning(f"⚠️ {batch_stats['orgs_failed']} organization(s) failed to process")
    
    # Show per-org breakdown
    with st.expander(f"📋 Per-Organization Results ({len(batch_stats['org_results'])})", expanded=True):
        for result in batch_stats["org_results"]:
            if result["status"] == "success":
                st.write(f"✅ **{result['org_name']}** — {result['nodes']} nodes, {result['edges']} edges, {result['grants']} grants")
            else:
                st.write(f"❌ **{result['org_name']}** — {result.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Merge with existing data
    nodes_df, edges_df, stats = merge_graph_data(
        existing_nodes, existing_edges, new_nodes, new_edges
    )
    
    # Merge grants_detail
    grants_detail_df, grants_stats = merge_grants_detail(
        existing_grants_detail, new_grants_detail
    )
    
    st.subheader(f"🔁 Merge Results")
    st.caption(f"Dataset merge outcome for {batch_stats['orgs_processed']} organizations. Counts reflect what was added to the combined outputs.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Nodes:**")
        st.write(f"- Existing: {stats['existing_nodes']}")
        st.write(f"- From batch: {stats['new_nodes_total']}")
        st.write(f"- ✅ **Added: {stats['nodes_added']}**")
        if stats['nodes_skipped'] > 0:
            st.write(f"- ⏭️ Skipped: {stats['nodes_skipped']}")
    
    with col2:
        st.markdown("**Edges:**")
        st.write(f"- Existing: {stats['existing_edges']}")
        st.write(f"- From batch: {stats['new_edges_total']}")
        st.write(f"- ✅ **Added: {stats['edges_added']}**")
        if stats['edges_skipped'] > 0:
            st.write(f"- ⏭️ Skipped: {stats['edges_skipped']}")
    
    with col3:
        st.markdown("**Grant Details:**")
        st.write(f"- Existing: {grants_stats['existing']}")
        st.write(f"- From batch: {grants_stats['new']}")
        st.write(f"- ✅ **Added: {grants_stats['added']}**")
        if grants_stats['skipped'] > 0:
            st.write(f"- ⏭️ Skipped: {grants_stats['skipped']}")
    
    # Show region-relevant count
    if not grants_detail_df.empty and "region_relevant" in grants_detail_df.columns:
        relevant_count = grants_detail_df["region_relevant"].sum()
        total_count = len(grants_detail_df)
        st.caption(f"*Region-relevant grants: {relevant_count} of {total_count} ({100*relevant_count/total_count:.0f}%)*")
    
    st.divider()
    st.success(f"📊 **Combined {display_name} dataset:** {len(nodes_df)} nodes, {len(edges_df)} edges, {len(grants_detail_df)} grant details")
    
    render_graph_summary(nodes_df, edges_df)
    render_downloads(nodes_df, edges_df, grants_detail_df, project_name)


# =============================================================================
# Main App
# =============================================================================

def main():
    # Initialize Supabase
    init_supabase()
    
    # Header with logo, title, and help button
    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        st.image(C4C_LOGO_URL, width=60)
    with col2:
        st.title("OrgGraph (CA)")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        render_help_button()
    
    st.markdown("""
    OrgGraph currently supports US and Canadian nonprofit registries; additional sources will be added in the future.
    """)
    st.caption(f"App v{APP_VERSION}")
    
    # Cloud status in sidebar
    render_cloud_status()
    
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
        **Naming tips:** Use a descriptive name like "BC Foundations 2024" or "Ontario Grantmakers Network". 
        Avoid special characters. The name becomes a folder, so "BC Foundations" → `bc_foundations/`
        """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_project_name = st.text_input(
                "Project Name",
                placeholder="e.g., BC Foundations Network",
                help="Choose a descriptive name for your project"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            create_btn = st.button("Create Project", type="primary", disabled=not new_project_name)
        
        if new_project_name:
            folder_name = get_folder_name(new_project_name)
            st.caption(f"📁 Will create folder: `demo_data/{folder_name}/`")
        
        if create_btn and new_project_name:
            success, message = create_project(new_project_name)
            if success:
                st.success(f"✅ {message}")
                st.session_state.current_project = get_folder_name(new_project_name)
                st.rerun()
            else:
                st.error(f"❌ {message}")
        
        # If project was just created, show upload interface
        if "current_project" in st.session_state and st.session_state.current_project:
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
            with st.expander(f"📋 Organizations in Demo ({len(existing_foundations)})", expanded=True):
                for label, source in existing_foundations:
                    flag = "🇨🇦" if source == "CHARITYDATA_CA" else "🇺🇸" if source == "IRS_990" else "📄"
                    st.write(f"{flag} {label}")
        
        render_graph_summary(nodes_df, edges_df)
        render_downloads(nodes_df, edges_df, grants_detail_df, DEMO_PROJECT_NAME)


if __name__ == "__main__":
    main()
