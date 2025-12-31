"""
InsightGraph — Streamlit App

Structured insight from complex networks.
Reads exported data from OrgGraph US/CA projects.

VERSION HISTORY:
----------------
UPDATED v0.15.10: Network-Type-Aware Analysis
- NEW: Automatic detection of network type (funder vs social)
- NEW: SocialAnalyzer for ActorGraph LinkedIn data
- NEW: FunderAnalyzer for OrgGraph grant/board data
- Uses analyzers/ module with standardized output schema
- Brokerage ecosystem analysis for both network types
- Backward compatible with existing projects

UPDATED v0.15.9: Download simplification
- Collapsed individual file downloads into ZIP bundle
- Added benefit-oriented tooltips to all buttons
- Cleaner export UI with clear primary/secondary actions
- README.md added to bundle with column definitions

UPDATED v0.15.8: Immediate Save After Linking
- NEW: "Save Linked Project to Cloud" dialog appears immediately after creating linked network
- NEW: No need to run insights first before saving
- Shows overlap % in the save dialog
- Improved UX flow: Link → Save → Run Insights

UPDATED v0.15.7: Overlap Analysis Diagnostics
- NEW: Shows manifest status after loading linked project from cloud
- NEW: Warning if linked project was saved without overlap_analysis
- NEW: Debug logging for manifest in compute_insights_from_dataframes

UPDATED v0.15.6: Save Linked Projects to Cloud
- NEW: save_linked_to_cloud() function
- NEW: "Save Linked Project to Cloud" button in Downloads section
- Linked projects now persist with full manifest (match_stats, overlap_analysis)
- Description includes source projects and overlap percentage

UPDATED v0.15.5: Overlap Analysis in Reports
- NEW: Overlap analysis now included in markdown report for linked projects
- NEW: Overlap analysis stored in manifest for persistence
- Report includes: overlap summary table, match statistics, interpretation, strategic implications

UPDATED v0.15.4: Strategic Network Overlap Analysis
- UPGRADED: Interpretation tiers now signal action, not just status
  - High (≥70%): "Structural Alignment" — risk of groupthink/incumbency
  - Moderate (40-69%): "Partial Alignment" — hybrid ecosystem
  - Low (<40%): "Structural Disconnect" — movement before money
- NEW: Strategic Implications tabs for Funders / Coalitions / Intermediaries
- NEW: Downloadable CSV of unmatched organizations
- NEW: Scrollable dataframe view for all unmatched orgs
- REFINED: Thresholds as configurable constants
- REFINED: "Surfaces, not judges" — clear that analysis is diagnostic, not normative

UPDATED v0.15.3: Network Overlap Insights
- NEW: Shows all suggested matches (not limited to 15)
- NEW: Bulk actions: "Confirm All" / "Reject All" buttons
- NEW: Undo button for individual confirmations
- NEW: Network Overlap Analysis section with strategic interpretation
- Calculates overlap percentage between networks
- Provides context: High/Moderate/Low alignment meanings
- Strategic implications for unmatched organizations

UPDATED v0.15.2: Entity Linking Column Fix
- FIXED: Now detects 'label' column from CoreGraph schema (not just 'name')
- Checks multiple possible column names for flexibility

UPDATED v0.15.1: Entity Linking Stats Display
- NEW: Shows link stats in Downloads section after linking
- Displays: auto-matched, user confirmed, rejected, total linked
- Shows coverage percentage (linked orgs / total orgs)
- Clears link state when loading different projects

UPDATED v0.15.0: Phase 4 - Entity Linking
- NEW: "🔗 Link Entities" mode in Cloud Projects tab
- Select ActorGraph (LinkedIn) + OrgGraph projects to link
- Auto-match by normalized organization name
- Fuzzy matching (>70% similarity) for suggested matches
- Review UI: confirm/reject suggested matches
- Creates linked network with LinkedIn metadata on OrgGraph orgs
- Adds linkedin_url, linkedin_industry, linkedin_website columns

UPDATED v0.14.0: Phase 3B - Project Management UI
- NEW: "⚙️ Manage" tab for cloud project management
- View all projects with source icons and metadata
- 🗑️ Delete projects (with confirmation)
- ✏️ Rename projects (name only, slug preserved)
- 🔒/🌐 Toggle public/private visibility

UPDATED v0.13.1: Save merged projects to cloud
- NEW: "Save Merged to Cloud" button in Downloads section
- Enter custom name for merged project
- Bundle includes nodes, edges, grants, and analysis artifacts
- source_app="insightgraph" for merged projects
- Manifest tracks source_projects list

UPDATED v0.13.0: Phase 3 - Multi-project merge
- NEW: "Merge Multiple" mode in Cloud Projects tab
- Select 2+ projects with checkboxes to merge
- Combines nodes, edges, grants with deduplication
- Dedup by node_id, source+target+edge_type, grant keys
- Adds merge_source column for provenance tracking
- Preview shows combined counts before merge

UPDATED v0.12.1: Disabled legacy Save to Cloud button
- FIX: Removed broken save_artifacts_to_cloud call (incompatible with Phase 2 schema)
- InsightGraph is a consumer - loads bundles from OrgGraph/ActorGraph
- Saving artifacts to cloud may be added in future release

UPDATED v0.12.0: Phase 2 - Load from Cloud
- NEW: "Load from Cloud" tab in Project section
- Lists all cloud projects from OrgGraph US, CA, and ActorGraph
- Downloads and extracts ZIP bundles from Supabase Storage
- Project Store client integration

UPDATED v0.11.2: Fixed HTML download timing
- FIX: Generate HTML once and reuse for both ZIP and download button
- Prevents race condition where HTML button fails but ZIP works

UPDATED v0.11.1: Fixed HTML download
- FIX: Encode HTML as UTF-8 bytes for Safari compatibility
- NEW: Show error message if HTML generation fails

UPDATED v0.11.0: HTML report rendering
- NEW: HTML download button as primary (styled, print-ready)
- NEW: index.html included in bundle ZIP
- Three-column download layout: HTML | Bundle | Markdown
- Updated app icon to InsightGraph icon

UPDATED v0.10.0: Bundle export with manifest.json
- NEW: Structured bundle ZIP (data/, analysis/ folders)
- NEW: manifest.json with metadata, inputs, outputs, stats
- Traceability: SHA256 hashes, row counts, config snapshot
- Backward-compatible flat ZIP still available

UPDATED v0.9.0: Supabase cloud storage integration
- Added cloud save/load functionality
- User authentication via Supabase
- Artifacts persist in cloud database

UPDATED v0.8.0: Renamed to InsightGraph + Roles × Region Lens
- RENAMED: "Insight Engine" → "InsightGraph" for product consistency
- NEW: Roles × Region Lens section in reports (requires run.py v3.0.5)
- Matches naming pattern: ActorGraph, OrgGraph, InsightGraph

UPDATED v0.7.0: Critical metrics fixes (run.py v3.0.4)
- FIX: Betweenness now computed on undirected graph (was always 0)
- FIX: Hidden broker thresholds computed among connectors only
- FIX: Bridge detection focuses on largest component, ranked by impact
- FIX: Health score includes governance factor (board interlocks)
- Aligned V1 output with V2 metrics

UPDATED v0.6.0: Canonical schema alignment with OrgGraph US/CA
- Added Grant Network Results section (All vs Region Relevant)
- Support for grant_bucket column (3a, 3b, schedule_i, ca_t3010)
- Column aliasing: foundation_name ↔ funder_name for compatibility
- Region filtering support via region_relevant column
- Source breakdown (US vs CA) when multiple source_systems present

UPDATED v0.5.4: Use grant_purpose_raw column
- Purpose column is now explicitly 'grant_purpose_raw' (OrgGraph US standard)

UPDATED v0.5.3: Fixed input/output flow
- Inputs: nodes.csv + edges.csv (required), grants_detail.csv (optional)
- Outputs: project_summary.json, insight_cards.json, insight_report.md, node_metrics.csv
- Clear "Run" vs "Load Previous" workflow
- No longer requires pre-computed artifacts to run

UPDATED v0.5.2: Data-calibrated purpose keywords
- Keywords calibrated from GLFN grants_detail.csv (5,160 grants)
- Added Arts & Culture category
- Improved coverage: 80% of grants now categorized

UPDATED v0.5.1: Added Grant Purpose Explorer
- Loads grants_detail.csv when available
- Keyword-based purpose classification (no AI)
- Filter grants by purpose tags

UPDATED v0.5.0: Aligned with OrgGraph US/CA visual patterns
- Multi-project support (scans demo_data/)
- Loads pre-exported artifacts when available
- Consistent section headers and metric labels
- Clean separation: Network Results → Insight Cards → Downloads
"""

import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
import zipfile
import sys
import os
import importlib.util

# Get paths first (needed for sys.path)
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Add the project root to path for imports (MUST be before c4c_utils imports)
sys.path.insert(0, str(REPO_ROOT))

from c4c_utils.c4c_supabase import C4CSupabase

# =============================================================================
# Config
# =============================================================================

APP_VERSION = "0.15.10"  # Network-type-aware analysis
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_9c48d5079fcf4b688606c81d8f34d5a5~mv2.jpg"
INSIGHTGRAPH_ICON_URL = "https://static.wixstatic.com/media/275a3f_7736e28c9f5e40c1b2407e09dc5cb6e7~mv2.png"

DEMO_DATA_DIR = REPO_ROOT / "demo_data"

# =============================================================================
# Canonical grants_detail.csv Schema (Shared with OrgGraph US/CA)
# =============================================================================
# OrgGraph US/CA output these columns. InsightGraph should handle both
# naming conventions for backward compatibility.

GRANTS_DETAIL_COLUMN_ALIASES = {
    # OrgGraph US uses foundation_*, InsightGraph historically used funder_*
    "foundation_name": "funder_name",
    "foundation_ein": "funder_id",
    "foundation_country": "funder_country",
    # Ensure these exist
    "grantee_state": "grantee_admin1",
}

# Grant bucket constants (US + CA)
GRANT_BUCKETS = {
    "3a": "Part XIV 3a (US)",
    "3b": "Part XIV 3b (US)", 
    "schedule_i": "Schedule I (US)",
    "ca_t3010": "T3010 (Canada)",
    "unknown": "Unknown",
}

st.set_page_config(
    page_title="InsightGraph",
    page_icon=INSIGHTGRAPH_ICON_URL,
    layout="wide"
)

# =============================================================================
# Grant Purpose Classification (keyword-based, no AI)
# =============================================================================

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^a-z0-9\s]")

def normalize_purpose_text(s: str) -> str:
    """Normalize purpose text for keyword matching."""
    if not s or pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace("w/", " with ")
    s = _NONWORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def contains_word(text: str, word: str) -> bool:
    """Check for whole-word match."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def contains_phrase(text: str, phrase: str) -> bool:
    """Check for phrase match."""
    return phrase in text


# Purpose categories with phrases and words
# CALIBRATED from GLFN grants_detail.csv (5,160 grants)
CORE_PURPOSE_CATEGORIES = {
    "water": {
        "label": "💧 Water",
        "phrases": ["drinking water", "stormwater", "water quality", "watershed restoration", 
                   "water stewardship", "clean water", "great lakes", "wetland restoration",
                   "lakes region", "flint water", "river restoration", "creek restoration"],
        "words": ["water", "watershed", "river", "rivers", "lake", "lakes", "wetland", "wetlands", 
                 "aquifer", "freshwater", "groundwater", "stream", "coastal", "flint", "creek"]
    },
    "environment_nature": {
        "label": "🌲 Environment & Nature",
        "phrases": ["habitat restoration", "land conservation", "natural resources", 
                   "salmon habitat", "land mgmt", "habitat for", "forest restoration"],
        "words": ["environment", "environmental", "nature", "conservation", "biodiversity", 
                 "ecosystem", "habitat", "wildlife", "forest", "forests", "forestry", 
                 "land", "parks", "grassland", "island", "restore", "restoring", 
                 "restoration", "protect", "monitoring"]
    },
    "climate_energy": {
        "label": "🌡️ Climate & Energy",
        "phrases": ["climate change", "renewable energy", "clean energy", "climate action",
                   "climate resilience"],
        "words": ["climate", "carbon", "emissions", "decarbonization", "resilience", 
                 "solar", "wind", "energy", "sustainability", "sustainable", "renewable"]
    },
    "education_research": {
        "label": "📚 Education & Research",
        "phrases": ["environmental education", "stem education", "afterschool network",
                   "statewide afterschool", "quality afterschool", "summer learning",
                   "learning opportunities", "school linked"],
        "words": ["education", "educational", "research", "scholarship", "scholarships", 
                 "university", "college", "school", "learning", "training", "fellows", 
                 "fellowship", "afterschool"]
    },
    "community_social": {
        "label": "👥 Community & Social",
        "phrases": ["community development", "civic engagement", "public health",
                   "gun violence", "violence prevention", "justice reform", 
                   "economic mobility", "underserved youth"],
        "words": ["community", "communities", "civic", "social", "equity", "justice", 
                 "health", "neighborhood", "urban", "rural", "youth", "children",
                 "violence", "prevention", "mobility", "underserved"]
    },
    "agriculture_food": {
        "label": "🌾 Agriculture & Food",
        "phrases": ["food security", "sustainable agriculture"],
        "words": ["agriculture", "agricultural", "farm", "farming", "food", "nutrition"]
    },
    "policy_advocacy": {
        "label": "📢 Policy & Advocacy",
        "phrases": ["policy research", "public policy", "reform efforts"],
        "words": ["policy", "advocacy", "legislation", "regulatory", "government", "reform",
                 "democracy", "democratic", "voting", "civic"]
    },
    "arts_culture": {
        "label": "🎭 Arts & Culture",
        "phrases": ["arts and culture", "cultural programs"],
        "words": ["arts", "culture", "cultural", "museum", "theater", "theatre", 
                 "music", "dance", "heritage", "journalism", "media"]
    },
    "general_support": {
        "label": "🎯 General Support",
        "phrases": ["general support", "operating support", "core support", "general operating",
                   "general purposes", "charitable purposes", "charitable support", 
                   "mission fund", "discretionary fund", "membership grants"],
        "words": ["operations", "operating", "unrestricted", "capacity", "mission",
                 "discretionary", "membership"]
    },
}


def classify_grant_purpose(raw_purpose: str) -> dict:
    """
    Classify grant purpose using keyword matching.
    Returns: {"primary": str, "tags": list, "confidence": str}
    """
    norm = normalize_purpose_text(raw_purpose)
    if not norm:
        return {"primary": "uncategorized", "primary_label": "❓ Uncategorized", "tags": [], "confidence": "low"}

    matched = []
    matched_labels = []
    
    for cat, kws in CORE_PURPOSE_CATEGORIES.items():
        hit = False

        for ph in kws.get("phrases", []):
            ph_norm = normalize_purpose_text(ph)
            if contains_phrase(norm, ph_norm):
                hit = True
                break

        if not hit:
            for w in kws.get("words", []):
                w_norm = normalize_purpose_text(w)
                if contains_word(norm, w_norm):
                    hit = True
                    break

        if hit:
            matched.append(cat)
            matched_labels.append(kws["label"])

    matched = list(dict.fromkeys(matched))
    matched_labels = list(dict.fromkeys(matched_labels))

    if not matched:
        return {"primary": "uncategorized", "primary_label": "❓ Uncategorized", "tags": [], "confidence": "low"}

    confidence = "high" if len(matched) >= 2 else "medium"
    return {
        "primary": matched[0], 
        "primary_label": matched_labels[0],
        "tags": matched, 
        "tag_labels": matched_labels,
        "confidence": confidence
    }


def add_purpose_classifications(grants_df: pd.DataFrame) -> pd.DataFrame:
    """Add purpose classification columns to grants dataframe."""
    if grants_df.empty:
        return grants_df
    
    # Use grant_purpose_raw column (standard field from OrgGraph US export)
    purpose_col = "grant_purpose_raw"
    
    if purpose_col not in grants_df.columns:
        st.warning(f"Purpose column '{purpose_col}' not found in grants_detail.csv. Available columns: {list(grants_df.columns)}")
        return grants_df
    
    # Classify each grant
    classifications = grants_df[purpose_col].apply(classify_grant_purpose)
    
    grants_df = grants_df.copy()
    grants_df["purpose_primary"] = classifications.apply(lambda x: x["primary"])
    grants_df["purpose_primary_label"] = classifications.apply(lambda x: x["primary_label"])
    grants_df["purpose_tags"] = classifications.apply(lambda x: "|".join(x["tags"]))
    grants_df["purpose_confidence"] = classifications.apply(lambda x: x["confidence"])
    
    return grants_df


def normalize_grants_detail_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize grants_detail.csv columns to canonical names.
    Handles both OrgGraph US (foundation_*) and legacy (funder_*) naming.
    Adds missing columns with sensible defaults.
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Apply aliases: create funder_name from foundation_name if missing, etc.
    for canonical, alias in GRANTS_DETAIL_COLUMN_ALIASES.items():
        if canonical in df.columns and alias not in df.columns:
            df[alias] = df[canonical]
        elif alias in df.columns and canonical not in df.columns:
            df[canonical] = df[alias]
    
    # Ensure critical columns exist with defaults
    defaults = {
        "grant_bucket": "unknown",
        "region_relevant": True,  # Default to True (show all)
        "source_system": "unknown",
        "grant_amount": 0.0,
        "grantee_country": "",
        "fiscal_year": "",
    }
    
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    
    # Normalize grant_amount to numeric
    if "grant_amount" in df.columns:
        df["grant_amount"] = pd.to_numeric(df["grant_amount"], errors="coerce").fillna(0.0)
    
    # Normalize region_relevant to boolean
    if "region_relevant" in df.columns:
        df["region_relevant"] = df["region_relevant"].fillna(True).astype(bool)
    
    return df


def summarize_grants_by_bucket(df: pd.DataFrame, region_only: bool = False) -> dict:
    """
    Summarize grants by bucket (3a, 3b, schedule_i, ca_t3010, etc).
    
    Returns: {
        "row_count": int,
        "total_amount": float,
        "by_bucket": pd.DataFrame with [grant_bucket, count, amount]
    }
    """
    if df is None or df.empty:
        return {
            "row_count": 0,
            "total_amount": 0.0,
            "by_bucket": pd.DataFrame(columns=["grant_bucket", "count", "amount"]),
        }
    
    work = df.copy()
    
    # Ensure required columns
    if "grant_bucket" not in work.columns:
        work["grant_bucket"] = "unknown"
    if "grant_amount" not in work.columns:
        work["grant_amount"] = 0.0
    if "region_relevant" not in work.columns:
        work["region_relevant"] = True
    
    work["grant_amount"] = pd.to_numeric(work["grant_amount"], errors="coerce").fillna(0.0)
    
    # Filter by region if requested
    if region_only:
        work = work[work["region_relevant"] == True]
    
    # Aggregate by bucket
    by_bucket = (
        work.groupby("grant_bucket", dropna=False)
        .agg(count=("grant_bucket", "size"), amount=("grant_amount", "sum"))
        .reset_index()
        .sort_values("amount", ascending=False)
    )
    
    # Add bucket labels
    by_bucket["bucket_label"] = by_bucket["grant_bucket"].map(GRANT_BUCKETS).fillna(by_bucket["grant_bucket"])
    
    return {
        "row_count": int(len(work)),
        "total_amount": float(work["grant_amount"].sum()),
        "by_bucket": by_bucket,
    }


def summarize_grants_by_source(df: pd.DataFrame) -> dict:
    """Summarize grants by source_system (US vs CA)."""
    if df is None or df.empty:
        return {"by_source": pd.DataFrame()}
    
    work = df.copy()
    
    if "source_system" not in work.columns:
        work["source_system"] = "unknown"
    if "grant_amount" not in work.columns:
        work["grant_amount"] = 0.0
    
    work["grant_amount"] = pd.to_numeric(work["grant_amount"], errors="coerce").fillna(0.0)
    
    by_source = (
        work.groupby("source_system", dropna=False)
        .agg(count=("source_system", "size"), amount=("grant_amount", "sum"))
        .reset_index()
        .sort_values("amount", ascending=False)
    )
    
    return {"by_source": by_source}

# =============================================================================
# Dynamic Import of run.py (for compute fallback)
# =============================================================================

def load_run_module():
    """Load run.py module for computing insights when pre-exports don't exist."""
    run_path = APP_DIR / "run.py"
    if not run_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("run", run_path)
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)
        return run_module
    except Exception as e:
        st.warning(f"Could not load run.py: {e}")
        return None


# =============================================================================
# Project Discovery
# =============================================================================

def get_projects() -> list:
    """Discover available projects in demo_data/."""
    if not DEMO_DATA_DIR.exists():
        return []
    
    projects = []
    for item in DEMO_DATA_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check what files exist
            has_nodes = (item / "nodes.csv").exists()
            has_edges = (item / "edges.csv").exists()
            has_grants_detail = (item / "grants_detail.csv").exists()
            
            # Pre-computed artifacts (outputs from previous runs)
            has_summary = (item / "project_summary.json").exists()
            has_cards = (item / "insight_cards.json").exists()
            has_report = (item / "insight_report.md").exists()
            has_metrics = (item / "node_metrics.csv").exists()
            
            # Only show projects that have the required inputs
            if has_nodes and has_edges:
                projects.append({
                    "id": item.name,
                    "path": item,
                    # Required inputs
                    "has_nodes": has_nodes,
                    "has_edges": has_edges,
                    # Optional input
                    "has_grants_detail": has_grants_detail,
                    # Pre-computed outputs (from previous runs)
                    "has_summary": has_summary,
                    "has_cards": has_cards,
                    "has_report": has_report,
                    "has_metrics": has_metrics,
                    "has_precomputed": has_summary and has_cards,
                })
    
    # Sort alphabetically
    projects.sort(key=lambda x: x["id"].lower())
    return projects


def load_project_inputs(project: dict) -> dict:
    """Load input files for a project (nodes, edges, grants_detail)."""
    path = project["path"]
    data = {
        "project_id": project["id"],
        "nodes_df": None,
        "edges_df": None,
        "grants_df": None,
        "has_grants_detail": project.get("has_grants_detail", False),
        "has_region_data": False,  # Set to True if region_relevant column exists
    }
    
    # Load required inputs
    if project["has_nodes"]:
        data["nodes_df"] = pd.read_csv(path / "nodes.csv")
    if project["has_edges"]:
        data["edges_df"] = pd.read_csv(path / "edges.csv")
    
    # Load optional grants_detail.csv, normalize columns, add purpose classifications
    if project.get("has_grants_detail"):
        try:
            grants_df = pd.read_csv(path / "grants_detail.csv")
            # Normalize to canonical schema (handles foundation_name → funder_name, etc.)
            grants_df = normalize_grants_detail_columns(grants_df)
            # Check if region data exists (not all True/False)
            if "region_relevant" in grants_df.columns:
                unique_vals = grants_df["region_relevant"].dropna().unique()
                data["has_region_data"] = len(unique_vals) > 1 or (len(unique_vals) == 1 and not unique_vals[0])
            # Add purpose classifications
            grants_df = add_purpose_classifications(grants_df)
            data["grants_df"] = grants_df
        except Exception as e:
            st.warning(f"Could not load grants_detail.csv: {e}")
            data["grants_df"] = None
            data["has_grants_detail"] = False
    
    return data


def load_precomputed_artifacts(project: dict) -> dict:
    """Load pre-computed artifacts from a previous run (if they exist)."""
    path = project["path"]
    artifacts = {
        "project_summary": None,
        "insight_cards": None,
        "markdown_report": None,
        "metrics_df": None,
    }
    
    if project.get("has_summary"):
        with open(path / "project_summary.json", "r") as f:
            artifacts["project_summary"] = json.load(f)
    
    if project.get("has_cards"):
        with open(path / "insight_cards.json", "r") as f:
            artifacts["insight_cards"] = json.load(f)
    
    if project.get("has_report"):
        with open(path / "insight_report.md", "r") as f:
            artifacts["markdown_report"] = f.read()
    
    if project.get("has_metrics"):
        artifacts["metrics_df"] = pd.read_csv(path / "node_metrics.csv")
    
    return artifacts


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    if "current_project_id" not in st.session_state:
        st.session_state.current_project_id = None
    if "project_data" not in st.session_state:
        st.session_state.project_data = None
    if "cloud_project_data" not in st.session_state:
        st.session_state.cloud_project_data = None
    if "merged_projects" not in st.session_state:
        st.session_state.merged_projects = None
    
    # Entity linking state (Phase 4)
    if "entity_matches" not in st.session_state:
        st.session_state.entity_matches = None
    if "match_confirmations" not in st.session_state:
        st.session_state.match_confirmations = {}
    if "link_actor_data" not in st.session_state:
        st.session_state.link_actor_data = None
    if "link_org_data" not in st.session_state:
        st.session_state.link_org_data = None
    if "linked_project_info" not in st.session_state:
        st.session_state.linked_project_info = None
    if "show_linked_save_dialog" not in st.session_state:
        st.session_state.show_linked_save_dialog = False
    
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
        except Exception as e:
            st.session_state.project_store = None
    
    return st.session_state.get("project_store")


def get_project_store_authenticated():
    """Get authenticated Project Store client, or None."""
    client = st.session_state.get("project_store")
    if client and client.is_authenticated():
        return client
    return None


def list_cloud_projects():
    """List all available cloud projects from Project Store."""
    client = get_project_store_authenticated()
    if not client:
        return []
    
    # Get projects from all source apps
    all_projects = []
    for source_app in ['orggraph_us', 'orggraph_ca', 'actorgraph', 'insightgraph']:
        projects, error = client.list_projects(source_app=source_app, include_public=True)
        if projects:
            all_projects.extend(projects)
    
    # Sort by updated_at (most recent first)
    all_projects.sort(key=lambda p: p.updated_at, reverse=True)
    return all_projects


def load_cloud_project(project_id: str = None, slug: str = None) -> dict:
    """
    Load a project bundle from cloud storage.
    
    Downloads ZIP, extracts nodes.csv, edges.csv, grants_detail.csv.
    
    Returns:
        dict with nodes_df, edges_df, grants_df, project metadata
    """
    client = get_project_store_authenticated()
    if not client:
        return {"error": "Not authenticated"}
    
    # Download bundle
    bundle_data, error = client.load_project(project_id=project_id, slug=slug)
    if error:
        return {"error": error}
    
    # Get project metadata
    project, _ = client.get_project(project_id=project_id, slug=slug)
    
    # Extract files from ZIP
    try:
        with zipfile.ZipFile(BytesIO(bundle_data), 'r') as zf:
            file_list = zf.namelist()
            
            result = {
                "project_id": project.slug if project else "cloud_project",
                "source_app": project.source_app if project else "unknown",
                "name": project.name if project else "Cloud Project",
                "node_count": project.node_count if project else 0,
                "edge_count": project.edge_count if project else 0,
                "nodes_df": None,
                "edges_df": None,
                "grants_df": None,
                "has_grants_detail": False,
                "has_region_data": False,
                "manifest": None,
            }
            
            # Load manifest if present
            if 'manifest.json' in file_list:
                with zf.open('manifest.json') as f:
                    result["manifest"] = json.load(f)
            
            # Load nodes.csv
            if 'nodes.csv' in file_list:
                with zf.open('nodes.csv') as f:
                    result["nodes_df"] = pd.read_csv(f)
            
            # Load edges.csv
            if 'edges.csv' in file_list:
                with zf.open('edges.csv') as f:
                    result["edges_df"] = pd.read_csv(f)
            
            # Load grants_detail.csv if present
            if 'grants_detail.csv' in file_list:
                with zf.open('grants_detail.csv') as f:
                    grants_df = pd.read_csv(f)
                    # Normalize columns
                    grants_df = normalize_grants_detail_columns(grants_df)
                    # Check region data
                    if "region_relevant" in grants_df.columns:
                        unique_vals = grants_df["region_relevant"].dropna().unique()
                        result["has_region_data"] = len(unique_vals) > 1 or (len(unique_vals) == 1 and not unique_vals[0])
                    # Add purpose classifications
                    grants_df = add_purpose_classifications(grants_df)
                    result["grants_df"] = grants_df
                    result["has_grants_detail"] = True
            
            return result
            
    except Exception as e:
        return {"error": f"Failed to extract bundle: {str(e)}"}


def merge_cloud_projects(projects: list) -> dict:
    """
    Download and merge multiple cloud projects into a single dataset.
    
    Args:
        projects: List of ProjectSummary objects to merge
    
    Returns:
        dict with merged nodes_df, edges_df, grants_df
    """
    if not projects:
        return {"error": "No projects to merge"}
    
    all_nodes = []
    all_edges = []
    all_grants = []
    source_apps = set()
    project_names = []
    
    # Download and collect data from each project
    for p in projects:
        data = load_cloud_project(project_id=p.id)
        
        if data.get("error"):
            return {"error": f"Failed to load {p.name}: {data['error']}"}
        
        if data.get("nodes_df") is not None:
            nodes_df = data["nodes_df"].copy()
            # Add source tracking if not present
            if "merge_source" not in nodes_df.columns:
                nodes_df["merge_source"] = p.slug
            all_nodes.append(nodes_df)
        
        if data.get("edges_df") is not None:
            edges_df = data["edges_df"].copy()
            if "merge_source" not in edges_df.columns:
                edges_df["merge_source"] = p.slug
            all_edges.append(edges_df)
        
        if data.get("grants_df") is not None:
            grants_df = data["grants_df"].copy()
            if "merge_source" not in grants_df.columns:
                grants_df["merge_source"] = p.slug
            all_grants.append(grants_df)
        
        source_apps.add(p.source_app)
        project_names.append(p.name)
    
    # Merge nodes with deduplication
    if all_nodes:
        merged_nodes = pd.concat(all_nodes, ignore_index=True)
        # Deduplicate by node_id (keep first occurrence)
        before_dedup = len(merged_nodes)
        if "node_id" in merged_nodes.columns:
            merged_nodes = merged_nodes.drop_duplicates(subset=["node_id"], keep="first")
        elif "id" in merged_nodes.columns:
            merged_nodes = merged_nodes.drop_duplicates(subset=["id"], keep="first")
        after_dedup = len(merged_nodes)
        nodes_deduped = before_dedup - after_dedup
    else:
        merged_nodes = pd.DataFrame()
        nodes_deduped = 0
    
    # Merge edges with deduplication
    if all_edges:
        merged_edges = pd.concat(all_edges, ignore_index=True)
        # Deduplicate by source + target + edge_type (keep first)
        before_dedup = len(merged_edges)
        
        # Find which columns exist for deduplication
        dedup_cols = []
        # Try common column names for source/target
        if "source" in merged_edges.columns:
            dedup_cols.append("source")
        elif "source_id" in merged_edges.columns:
            dedup_cols.append("source_id")
        
        if "target" in merged_edges.columns:
            dedup_cols.append("target")
        elif "target_id" in merged_edges.columns:
            dedup_cols.append("target_id")
        
        if "edge_type" in merged_edges.columns:
            dedup_cols.append("edge_type")
        
        # Only deduplicate if we have columns to dedupe on
        if len(dedup_cols) >= 2:
            merged_edges = merged_edges.drop_duplicates(subset=dedup_cols, keep="first")
        
        after_dedup = len(merged_edges)
        edges_deduped = before_dedup - after_dedup
    else:
        merged_edges = pd.DataFrame()
        edges_deduped = 0
    
    # Merge grants with deduplication
    merged_grants = None
    has_grants = False
    has_region_data = False
    
    if all_grants:
        merged_grants = pd.concat(all_grants, ignore_index=True)
        # Deduplicate by key fields that exist
        before_dedup = len(merged_grants)
        dedup_cols = []
        for col in ["funder_id", "funder_ein", "grantee_name", "grant_amount", "grant_year"]:
            if col in merged_grants.columns:
                dedup_cols.append(col)
        
        if len(dedup_cols) >= 2:
            merged_grants = merged_grants.drop_duplicates(subset=dedup_cols, keep="first")
        
        # Check region data
        if "region_relevant" in merged_grants.columns:
            unique_vals = merged_grants["region_relevant"].dropna().unique()
            has_region_data = len(unique_vals) > 1 or (len(unique_vals) == 1 and not unique_vals[0])
        
        has_grants = True
    
    # Build merged name
    merged_name = " + ".join(project_names[:3])
    if len(project_names) > 3:
        merged_name += f" (+{len(project_names) - 3} more)"
    
    return {
        "project_id": f"merged-{len(projects)}-projects",
        "source_app": "merged",
        "name": merged_name,
        "node_count": len(merged_nodes),
        "edge_count": len(merged_edges),
        "nodes_df": merged_nodes if not merged_nodes.empty else None,
        "edges_df": merged_edges if not merged_edges.empty else None,
        "grants_df": merged_grants,
        "has_grants_detail": has_grants,
        "has_region_data": has_region_data,
        "manifest": {
            "merged": True,
            "source_projects": [p.slug for p in projects],
            "source_apps": list(source_apps),
            "dedup_stats": {
                "nodes_removed": nodes_deduped,
                "edges_removed": edges_deduped,
            }
        },
    }


# =============================================================================
# Entity Linking (Phase 4)
# =============================================================================

def normalize_org_name(name: str) -> str:
    """Normalize organization name for matching."""
    if not name or pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common suffixes
    suffixes = [
        ' incorporated', ' inc', ' llc', ' nfp', ' corp', ' corporation',
        ' foundation', ' fund', ' trust', ' society', ' association',
        ' organization', ' org', ' company', ' co', ' ltd', ' limited'
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove "the " prefix
    if name.startswith('the '):
        name = name[4:]
    
    # Remove punctuation except spaces
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name


def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two normalized names (0-100)."""
    if not name1 or not name2:
        return 0.0
    
    # Exact match
    if name1 == name2:
        return 100.0
    
    # Token-based Jaccard similarity
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    jaccard = (intersection / union) * 100 if union > 0 else 0.0
    
    # Boost if one is substring of other
    if name1 in name2 or name2 in name1:
        jaccard = max(jaccard, 80.0)
    
    return jaccard


def find_entity_matches(actor_df: pd.DataFrame, org_df: pd.DataFrame) -> dict:
    """
    Find matches between ActorGraph organizations and OrgGraph nodes.
    
    Args:
        actor_df: ActorGraph nodes dataframe (companies from LinkedIn)
        org_df: OrgGraph nodes dataframe (funders + grantees)
    
    Returns:
        dict with auto_matches, suggested_matches, unmatched_actor, unmatched_org
    """
    if actor_df is None or org_df is None:
        return {"error": "Missing data"}
    
    # Get organization nodes from OrgGraph (filter to orgs only)
    org_nodes = org_df[org_df['node_type'] == 'organization'].copy() if 'node_type' in org_df.columns else org_df.copy()
    
    # Determine name columns - check multiple possible column names
    actor_name_col = None
    for col in ['label', 'name', 'Name', 'organization', 'company']:
        if col in actor_df.columns:
            actor_name_col = col
            break
    
    org_name_col = None
    for col in ['label', 'name', 'Name', 'organization']:
        if col in org_nodes.columns:
            org_name_col = col
            break
    
    if actor_name_col is None:
        return {"error": f"ActorGraph missing name column. Found: {list(actor_df.columns)}"}
    if org_name_col is None:
        return {"error": f"OrgGraph missing label column. Found: {list(org_nodes.columns)}"}
    
    # Normalize names
    actor_df = actor_df.copy()
    org_nodes = org_nodes.copy()
    
    actor_df['_normalized'] = actor_df[actor_name_col].apply(normalize_org_name)
    org_nodes['_normalized'] = org_nodes[org_name_col].apply(normalize_org_name)
    
    # Build lookup dict for OrgGraph
    org_lookup = {}
    for idx, row in org_nodes.iterrows():
        norm_name = row['_normalized']
        if norm_name:
            if norm_name not in org_lookup:
                org_lookup[norm_name] = []
            org_lookup[norm_name].append({
                'idx': idx,
                'node_id': row.get('node_id', ''),
                'label': row.get(org_name_col, ''),
                'node_type': row.get('node_type', 'organization'),
            })
    
    auto_matches = []      # Exact normalized matches
    suggested_matches = [] # Fuzzy matches (>70% similarity)
    unmatched_actor = []   # No match found
    matched_org_ids = set()
    
    for idx, actor_row in actor_df.iterrows():
        actor_name = actor_row.get(actor_name_col, '')
        actor_norm = actor_row.get('_normalized', '')
        
        if not actor_norm:
            continue
        
        actor_info = {
            'idx': idx,
            'name': actor_name,
            'normalized': actor_norm,
            'linkedin_url': actor_row.get('profile_url', actor_row.get('LinkedIn URL', '')),
            'website': actor_row.get('website', actor_row.get('Website', '')),
            'industry': actor_row.get('industry', actor_row.get('Industry', '')),
        }
        
        # Check for exact normalized match
        if actor_norm in org_lookup:
            org_match = org_lookup[actor_norm][0]  # Take first match
            auto_matches.append({
                'actor': actor_info,
                'org': org_match,
                'similarity': 100.0,
                'match_type': 'exact',
            })
            matched_org_ids.add(org_match['node_id'])
            continue
        
        # Fuzzy matching - find best match
        best_match = None
        best_similarity = 0.0
        
        for org_norm, org_list in org_lookup.items():
            similarity = calculate_similarity(actor_norm, org_norm)
            if similarity > best_similarity and similarity >= 70.0:
                best_similarity = similarity
                best_match = org_list[0]
        
        if best_match and best_similarity >= 70.0:
            suggested_matches.append({
                'actor': actor_info,
                'org': best_match,
                'similarity': round(best_similarity, 1),
                'match_type': 'fuzzy',
                'confirmed': None,  # User needs to confirm
            })
        else:
            unmatched_actor.append(actor_info)
    
    # Find unmatched OrgGraph orgs
    unmatched_org = []
    for idx, row in org_nodes.iterrows():
        node_id = row.get('node_id', '')
        if node_id not in matched_org_ids:
            unmatched_org.append({
                'node_id': node_id,
                'label': row.get(org_name_col, ''),
            })
    
    return {
        'auto_matches': auto_matches,
        'suggested_matches': suggested_matches,
        'unmatched_actor': unmatched_actor,
        'unmatched_org': unmatched_org,
        'stats': {
            'total_actor': len(actor_df),
            'total_org': len(org_nodes),
            'auto_matched': len(auto_matches),
            'suggested': len(suggested_matches),
            'unmatched_actor': len(unmatched_actor),
            'unmatched_org': len(unmatched_org),
        }
    }


def render_entity_match_review():
    """Render the entity match review UI."""
    matches = st.session_state.get("entity_matches", {})
    
    if not matches or matches.get("error"):
        st.error(f"❌ {matches.get('error', 'No matches found')}")
        return
    
    stats = matches.get('stats', {})
    auto_matches = matches.get('auto_matches', [])
    suggested_matches = matches.get('suggested_matches', [])
    unmatched_actor = matches.get('unmatched_actor', [])
    
    st.divider()
    st.markdown("### Match Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Auto-matched", stats.get('auto_matched', 0))
    col2.metric("🔶 Review needed", stats.get('suggested', 0))
    col3.metric("❌ Unmatched (Actor)", stats.get('unmatched_actor', 0))
    col4.metric("📊 Total OrgGraph", stats.get('total_org', 0))
    
    # Calculate overlap metrics
    total_actor = stats.get('total_actor', 0)
    total_org = stats.get('total_org', 0)
    auto_matched = stats.get('auto_matched', 0)
    suggested = stats.get('suggested', 0)
    potential_matches = auto_matched + suggested
    unmatched_actor_count = stats.get('unmatched_actor', 0)
    
    # Configurable thresholds
    OVERLAP_HIGH_THRESHOLD = 70
    OVERLAP_MODERATE_THRESHOLD = 40
    
    # Network Overlap Analysis
    if total_actor > 0:
        overlap_pct = (potential_matches / total_actor) * 100
        
        st.markdown("---")
        st.markdown("### 🔍 Network Overlap Analysis")
        st.caption("*Are the organizations shaping this ecosystem also connected to formal funding flows?*")
        
        # Determine overlap level and interpretation
        if overlap_pct >= OVERLAP_HIGH_THRESHOLD:
            overlap_level = "Structural Alignment"
            overlap_color = "🟢"
            interpretation = f"""
There is **strong alignment** between the influence/coalition network and the foundation funding ecosystem.

Coalition activity is well-resourced and institutionally supported. Funding reinforces existing influence pathways.

**What this means:**
- Most organizations shaping the field are also resourced to sustain it
- Funders and network actors share priorities and relationships
- *Risk:* Potential for groupthink or incumbency bias — emergent voices may be crowded out
            """
        elif overlap_pct >= OVERLAP_MODERATE_THRESHOLD:
            overlap_level = "Partial Alignment"
            overlap_color = "🟡"
            interpretation = f"""
There is **partial alignment** between the influence/coalition network and the foundation funding ecosystem.

A core set of organizations are both influential and funded, while a significant share of network participants operate outside formal funding streams. This suggests a hybrid ecosystem where momentum is not fully matched by capital.

**What this means:**
- Some influential organizations may be under-resourced relative to their network role
- Funding may be reinforcing established actors rather than emergent ones
- The network likely relies on informal, corporate, or volunteer support to function
            """
        else:
            overlap_level = "Structural Disconnect"
            overlap_color = "🔴"
            interpretation = f"""
There is **limited alignment** between the influence/coalition network and the foundation funding ecosystem.

Influence and funding are decoupled. The coalition relies on under-resourced actors operating outside formal funding streams.

**What this means:**
- High fragility — key actors may be at risk of burnout or departure
- High leverage for first-mover funders willing to support this space
- The field may be "movement before money" — grassroots energy awaiting institutional support
            """
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(f"{overlap_color} Overlap", f"{overlap_pct:.1f}%", overlap_level)
        with col2:
            st.markdown(interpretation)
        
        # Strategic implications by audience
        if unmatched_actor_count > 0:
            with st.expander("📋 Strategic Implications", expanded=True):
                tab1, tab2, tab3 = st.tabs(["For Funders", "For Coalitions", "For Intermediaries"])
                
                with tab1:
                    st.markdown(f"""
**Blind spot detection:** {unmatched_actor_count} organizations are shaping this field but not in foundation portfolios.

**Early-signal investing:** Grassroots or emergent actors gaining traction before institutional funding arrives.

**Portfolio balance:** Check for over-concentration on "usual suspects."

*Key question: Are we funding the organizations that actually move the system — or just the ones easiest to diligence?*
                    """)
                
                with tab2:
                    st.markdown(f"""
**Sustainability risk:** High-centrality, low-funding organizations are at burnout risk.

**Power asymmetry:** Who coordinates the work vs who controls the resources?

**Growth constraints:** Why does scale stall despite network momentum?

*Key question: Which parts of our network are carrying disproportionate load without capital?*
                    """)
                
                with tab3:
                    st.markdown(f"""
**Broker opportunity mapping:** Organizations sitting between funded and unfunded clusters.

**Translation gaps:** Where do language, governance, or form (501c3 vs coalition) block funding access?

**Intervention design:** Fiscal sponsorship, pooled funds, regranting vehicles.

*Key question: Where would a small structural intervention unlock outsized impact?*
                    """)
            
            st.caption("*This analysis surfaces structural patterns, not recommendations. Interpretation requires context.*")
    
    # Auto-matches (collapsed by default)
    if auto_matches:
        with st.expander(f"✅ Auto-matched ({len(auto_matches)})", expanded=False):
            for match in auto_matches[:20]:  # Show first 20
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{match['actor']['name']}**")
                with col2:
                    st.markdown(f"↔ {match['org']['label']}")
            if len(auto_matches) > 20:
                st.caption(f"...and {len(auto_matches) - 20} more")
    
    # Suggested matches (need review)
    if suggested_matches:
        with st.expander(f"🔶 Review Suggested Matches ({len(suggested_matches)})", expanded=True):
            # Initialize confirmation state
            if "match_confirmations" not in st.session_state:
                st.session_state.match_confirmations = {}
            
            # Bulk actions
            col_all, col_none, col_status = st.columns([1, 1, 2])
            with col_all:
                if st.button("✓ Confirm All", key="confirm_all_suggested"):
                    for i in range(len(suggested_matches)):
                        st.session_state.match_confirmations[i] = True
                    st.rerun()
            with col_none:
                if st.button("✗ Reject All", key="reject_all_suggested"):
                    for i in range(len(suggested_matches)):
                        st.session_state.match_confirmations[i] = False
                    st.rerun()
            with col_status:
                confirmed = sum(1 for i in range(len(suggested_matches)) if st.session_state.match_confirmations.get(i) == True)
                rejected = sum(1 for i in range(len(suggested_matches)) if st.session_state.match_confirmations.get(i) == False)
                pending = len(suggested_matches) - confirmed - rejected
                st.caption(f"✅ {confirmed} confirmed · ❌ {rejected} rejected · ⏳ {pending} pending")
            
            st.divider()
            
            # Show ALL matches
            for i, match in enumerate(suggested_matches):
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
                
                with col1:
                    st.markdown(f"**{match['actor']['name']}**")
                    if match['actor'].get('industry'):
                        st.caption(match['actor']['industry'])
                
                with col2:
                    st.markdown(f"↔ {match['org']['label']}")
                    st.caption(f"{match['similarity']}% similar")
                
                # Show status or buttons based on current state
                if i in st.session_state.match_confirmations:
                    status = "✅" if st.session_state.match_confirmations[i] else "❌"
                    with col3:
                        st.markdown(f"**{status}**")
                    with col4:
                        if st.button("↩️", key=f"undo_{i}", help="Undo"):
                            del st.session_state.match_confirmations[i]
                            st.rerun()
                else:
                    with col3:
                        if st.button("✓", key=f"confirm_{i}", help="Confirm match"):
                            st.session_state.match_confirmations[i] = True
                            st.rerun()
                    
                    with col4:
                        if st.button("✗", key=f"reject_{i}", help="Reject match"):
                            st.session_state.match_confirmations[i] = False
                            st.rerun()
    
    # Unmatched ActorGraph orgs (with CSV download)
    if unmatched_actor:
        with st.expander(f"❌ Unmatched Organizations ({len(unmatched_actor)})", expanded=False):
            st.caption("Organizations in the influence network but not found in the funding network.")
            
            # Create DataFrame for display and download
            unmatched_df = pd.DataFrame(unmatched_actor)
            display_cols = ['name']
            if 'industry' in unmatched_df.columns:
                display_cols.append('industry')
            if 'linkedin_url' in unmatched_df.columns:
                display_cols.append('linkedin_url')
            if 'website' in unmatched_df.columns:
                display_cols.append('website')
            
            # Download button
            csv_data = unmatched_df[display_cols].to_csv(index=False)
            st.download_button(
                "📥 Download Unmatched List (CSV)",
                data=csv_data,
                file_name="unmatched_organizations.csv",
                mime="text/csv",
                use_container_width=True,
                help="Use this list for outreach, research, or funding prospecting"
            )
            
            st.divider()
            
            # Show all orgs in scrollable dataframe
            st.dataframe(
                unmatched_df[display_cols],
                use_container_width=True,
                height=300
            )
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔗 Create Linked Network", type="primary", use_container_width=True):
            # Build linked network from confirmed matches
            linked_data = create_linked_network()
            if linked_data and not linked_data.get("error"):
                st.session_state.cloud_project_data = linked_data
                st.session_state.current_project_id = f"cloud:linked-network"
                st.session_state.project_data = None
                st.session_state.merged_projects = None
                # Track linked project info for display
                st.session_state.linked_project_info = linked_data.get("manifest", {}).get("match_stats", {})
                # Set flag to show save dialog
                st.session_state.show_linked_save_dialog = True
                st.success(f"✅ Created linked network: {linked_data['node_count']} nodes, {linked_data['edge_count']} edges")
                st.rerun()
            else:
                st.error(f"❌ {linked_data.get('error', 'Failed to create linked network')}")
    
    with col2:
        if st.button("🗑️ Clear Matches", use_container_width=True):
            st.session_state.entity_matches = None
            st.session_state.match_confirmations = {}
            st.rerun()


def create_linked_network() -> dict:
    """
    Create a linked network from confirmed matches.
    
    Combines OrgGraph data with ActorGraph metadata for matched orgs.
    """
    matches = st.session_state.get("entity_matches", {})
    confirmations = st.session_state.get("match_confirmations", {})
    org_data = st.session_state.get("link_org_data", {})
    actor_data = st.session_state.get("link_actor_data", {})
    
    if not org_data or not actor_data:
        return {"error": "Missing project data"}
    
    # Start with OrgGraph data as base
    nodes_df = org_data.get("nodes_df")
    edges_df = org_data.get("edges_df")
    grants_df = org_data.get("grants_df")
    
    if nodes_df is None:
        return {"error": "No nodes data"}
    
    nodes_df = nodes_df.copy()
    
    # Add LinkedIn columns if not present
    if 'linkedin_url' not in nodes_df.columns:
        nodes_df['linkedin_url'] = None
    if 'linkedin_industry' not in nodes_df.columns:
        nodes_df['linkedin_industry'] = None
    if 'linkedin_website' not in nodes_df.columns:
        nodes_df['linkedin_website'] = None
    if 'linkedin_matched' not in nodes_df.columns:
        nodes_df['linkedin_matched'] = False
    
    # Apply auto-matches
    auto_matches = matches.get('auto_matches', [])
    for match in auto_matches:
        org_node_id = match['org']['node_id']
        actor_info = match['actor']
        
        mask = nodes_df['node_id'] == org_node_id
        if mask.any():
            nodes_df.loc[mask, 'linkedin_url'] = actor_info.get('linkedin_url')
            nodes_df.loc[mask, 'linkedin_industry'] = actor_info.get('industry')
            nodes_df.loc[mask, 'linkedin_website'] = actor_info.get('website')
            nodes_df.loc[mask, 'linkedin_matched'] = True
    
    # Apply confirmed suggested matches
    suggested_matches = matches.get('suggested_matches', [])
    for i, match in enumerate(suggested_matches):
        if confirmations.get(i) == True:  # Explicitly confirmed
            org_node_id = match['org']['node_id']
            actor_info = match['actor']
            
            mask = nodes_df['node_id'] == org_node_id
            if mask.any():
                nodes_df.loc[mask, 'linkedin_url'] = actor_info.get('linkedin_url')
                nodes_df.loc[mask, 'linkedin_industry'] = actor_info.get('industry')
                nodes_df.loc[mask, 'linkedin_website'] = actor_info.get('website')
                nodes_df.loc[mask, 'linkedin_matched'] = True
    
    # Count matches
    match_count = nodes_df['linkedin_matched'].sum()
    
    # Calculate overlap analysis
    stats = matches.get('stats', {})
    total_actor = stats.get('total_actor', 0)
    auto_matched_count = len(auto_matches)
    suggested_count = len(suggested_matches)
    potential_matches = auto_matched_count + suggested_count
    
    # Configurable thresholds (same as UI)
    OVERLAP_HIGH_THRESHOLD = 70
    OVERLAP_MODERATE_THRESHOLD = 40
    
    overlap_pct = (potential_matches / total_actor * 100) if total_actor > 0 else 0
    
    if overlap_pct >= OVERLAP_HIGH_THRESHOLD:
        overlap_level = "High"
        overlap_signal = "Structural Alignment"
    elif overlap_pct >= OVERLAP_MODERATE_THRESHOLD:
        overlap_level = "Moderate"
        overlap_signal = "Partial Alignment"
    else:
        overlap_level = "Low"
        overlap_signal = "Structural Disconnect"
    
    # Build result
    actor_project = st.session_state.get("link_actor_project")
    org_project = st.session_state.get("link_org_project")
    
    return {
        "project_id": f"linked-{org_project.slug if org_project else 'network'}",
        "source_app": "linked",
        "name": f"Linked: {org_project.name if org_project else 'Network'} + {actor_project.name if actor_project else 'LinkedIn'}",
        "node_count": len(nodes_df),
        "edge_count": len(edges_df) if edges_df is not None else 0,
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "grants_df": grants_df,
        "has_grants_detail": grants_df is not None and not grants_df.empty,
        "has_region_data": org_data.get("has_region_data", False),
        "manifest": {
            "linked": True,
            "source_projects": [
                actor_project.slug if actor_project else "actorgraph",
                org_project.slug if org_project else "orggraph",
            ],
            "match_stats": {
                "auto_matched": auto_matched_count,
                "confirmed": sum(1 for v in confirmations.values() if v == True),
                "rejected": sum(1 for v in confirmations.values() if v == False),
                "total_linked": int(match_count),
            },
            "overlap_analysis": {
                "overlap_pct": round(overlap_pct, 1),
                "overlap_level": overlap_level,
                "overlap_signal": overlap_signal,
                "total_actor_orgs": total_actor,
                "total_org_orgs": stats.get('total_org', 0),
                "potential_matches": potential_matches,
                "unmatched_actor": stats.get('unmatched_actor', 0),
            }
        },
    }


def save_merged_to_cloud(name: str, data: dict, source_projects: list) -> tuple:
    """
    Save a merged project bundle to cloud storage.
    
    Args:
        name: Project name
        data: Dict with nodes_df, edges_df, grants_df, etc.
        source_projects: List of source project slugs
    
    Returns:
        Tuple of (success: bool, message: str, slug: str or None)
    """
    client = get_project_store_authenticated()
    if not client:
        return False, "Not authenticated", None
    
    # Create ZIP bundle
    zip_buffer = BytesIO()
    
    try:
        from datetime import datetime, timezone
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create manifest
            manifest = {
                "schema_version": "c4c_coregraph_v1",
                "bundle_version": "1.0",
                "source_app": "insightgraph_merged",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "merged": True,
                "source_projects": source_projects,
                "node_count": len(data.get("nodes_df", [])) if data.get("nodes_df") is not None else 0,
                "edge_count": len(data.get("edges_df", [])) if data.get("edges_df") is not None else 0,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            
            # Write nodes.csv
            if data.get("nodes_df") is not None and not data["nodes_df"].empty:
                zf.writestr("nodes.csv", data["nodes_df"].to_csv(index=False))
            
            # Write edges.csv
            if data.get("edges_df") is not None and not data["edges_df"].empty:
                zf.writestr("edges.csv", data["edges_df"].to_csv(index=False))
            
            # Write grants_detail.csv
            if data.get("grants_df") is not None and not data["grants_df"].empty:
                zf.writestr("grants_detail.csv", data["grants_df"].to_csv(index=False))
            
            # Write analysis artifacts if present
            if data.get("insight_cards"):
                zf.writestr("analysis/insight_cards.json", json.dumps(data["insight_cards"], indent=2))
            if data.get("project_summary"):
                zf.writestr("analysis/project_summary.json", json.dumps(data["project_summary"], indent=2))
            if data.get("markdown_report"):
                zf.writestr("analysis/insight_report.md", data["markdown_report"])
            if data.get("metrics_df") is not None and not data["metrics_df"].empty:
                zf.writestr("analysis/node_metrics.csv", data["metrics_df"].to_csv(index=False))
        
        zip_buffer.seek(0)
        bundle_data = zip_buffer.getvalue()
        
        # Calculate counts
        node_count = len(data.get("nodes_df", [])) if data.get("nodes_df") is not None else 0
        edge_count = len(data.get("edges_df", [])) if data.get("edges_df") is not None else 0
        
        # Save to Project Store
        project, error = client.save_project(
            name=name,
            bundle_data=bundle_data,
            source_app="insightgraph",  # Use insightgraph as source for merged
            node_count=node_count,
            edge_count=edge_count,
            jurisdiction=None,  # Merged projects may span jurisdictions
            region_preset=None,
            app_version=APP_VERSION,
            schema_version="c4c_coregraph_v1",
            bundle_version="1.0",
            description=f"Merged from: {', '.join(source_projects)}"
        )
        
        if error:
            return False, f"Upload failed: {error}", None
        
        return True, f"Saved as '{project.slug}'", project.slug
        
    except Exception as e:
        return False, f"Failed to create bundle: {str(e)}", None


def save_linked_to_cloud(name: str, data: dict) -> tuple:
    """
    Save a linked project bundle to cloud storage.
    
    Args:
        name: Project name
        data: Dict with nodes_df, edges_df, grants_df, manifest, etc.
    
    Returns:
        Tuple of (success: bool, message: str, slug: str or None)
    """
    client = get_project_store_authenticated()
    if not client:
        return False, "Not authenticated", None
    
    # Create ZIP bundle
    zip_buffer = BytesIO()
    
    try:
        from datetime import datetime, timezone
        
        # Get manifest from linked data
        source_manifest = data.get("manifest", {})
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Create manifest including overlap analysis
            manifest = {
                "schema_version": "c4c_coregraph_v1",
                "bundle_version": "1.0",
                "source_app": "insightgraph_linked",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "linked": True,
                "source_projects": source_manifest.get("source_projects", []),
                "node_count": len(data.get("nodes_df", [])) if data.get("nodes_df") is not None else 0,
                "edge_count": len(data.get("edges_df", [])) if data.get("edges_df") is not None else 0,
                "match_stats": source_manifest.get("match_stats", {}),
                "overlap_analysis": source_manifest.get("overlap_analysis", {}),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            
            # Write nodes.csv (with LinkedIn columns)
            if data.get("nodes_df") is not None and not data["nodes_df"].empty:
                zf.writestr("nodes.csv", data["nodes_df"].to_csv(index=False))
            
            # Write edges.csv
            if data.get("edges_df") is not None and not data["edges_df"].empty:
                zf.writestr("edges.csv", data["edges_df"].to_csv(index=False))
            
            # Write grants_detail.csv
            if data.get("grants_df") is not None and not data["grants_df"].empty:
                zf.writestr("grants_detail.csv", data["grants_df"].to_csv(index=False))
            
            # Write analysis artifacts if present
            if data.get("insight_cards"):
                zf.writestr("analysis/insight_cards.json", json.dumps(data["insight_cards"], indent=2))
            if data.get("project_summary"):
                zf.writestr("analysis/project_summary.json", json.dumps(data["project_summary"], indent=2))
            if data.get("markdown_report"):
                zf.writestr("analysis/insight_report.md", data["markdown_report"])
            if data.get("metrics_df") is not None and not data["metrics_df"].empty:
                zf.writestr("analysis/node_metrics.csv", data["metrics_df"].to_csv(index=False))
        
        zip_buffer.seek(0)
        bundle_data = zip_buffer.getvalue()
        
        # Calculate counts
        node_count = len(data.get("nodes_df", [])) if data.get("nodes_df") is not None else 0
        edge_count = len(data.get("edges_df", [])) if data.get("edges_df") is not None else 0
        
        # Build description from source projects and overlap
        source_projects = source_manifest.get("source_projects", [])
        overlap = source_manifest.get("overlap_analysis", {})
        overlap_pct = overlap.get("overlap_pct", 0)
        description = f"Linked from: {', '.join(source_projects)} | Overlap: {overlap_pct}%"
        
        # Save to Project Store
        project, error = client.save_project(
            name=name,
            bundle_data=bundle_data,
            source_app="insightgraph",  # Use insightgraph as source for linked
            node_count=node_count,
            edge_count=edge_count,
            jurisdiction=None,  # Linked projects may span jurisdictions
            region_preset=None,
            app_version=APP_VERSION,
            schema_version="c4c_coregraph_v1",
            bundle_version="1.0",
            description=description
        )
        
        if error:
            return False, f"Upload failed: {error}", None
        
        return True, f"Saved as '{project.slug}'", project.slug
        
    except Exception as e:
        return False, f"Failed to create bundle: {str(e)}", None


def delete_cloud_project(project_id: str) -> tuple:
    """
    Delete a project from cloud storage.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    client = get_project_store_authenticated()
    if not client:
        return False, "Not authenticated"
    
    try:
        success, error = client.delete_project(project_id=project_id)
        if success:
            return True, "Project deleted"
        else:
            return False, f"Delete failed: {error}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def rename_project(project_id: str, new_name: str) -> tuple:
    """
    Rename a project (update name only, slug stays the same).
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    client = get_project_store_authenticated()
    if not client:
        return False, "Not authenticated"
    
    try:
        # Update project name in database
        response = client.client.table('projects').update({
            'name': new_name
        }).eq('id', project_id).execute()
        
        if response.data:
            return True, f"Renamed to '{new_name}'"
        else:
            return False, "Rename failed"
    except Exception as e:
        return False, f"Error: {str(e)}"


def toggle_project_public(project_id: str, is_public: bool) -> tuple:
    """
    Toggle project public/private visibility.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    client = get_project_store_authenticated()
    if not client:
        return False, "Not authenticated"
    
    try:
        # Update is_public in database
        response = client.client.table('projects').update({
            'is_public': is_public
        }).eq('id', project_id).execute()
        
        if response.data:
            status = "public" if is_public else "private"
            return True, f"Project is now {status}"
        else:
            return False, "Update failed"
    except Exception as e:
        return False, f"Error: {str(e)}"


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
        
        # Get total cloud project count
        cloud_projects = list_cloud_projects()
        project_count = len(cloud_projects) if cloud_projects else 0
        
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


def save_artifacts_to_cloud(project_id: str, data: dict):
    """Save InsightGraph artifacts to Supabase cloud."""
    db = st.session_state.get("supabase_db")
    
    if not db or not db.is_authenticated:
        st.error("❌ Login required to save to cloud")
        return False
    
    # Debug: check what we're receiving
    print(f"DEBUG save_artifacts_to_cloud: project_id={project_id}")
    print(f"DEBUG save_artifacts_to_cloud: data keys={list(data.keys())}")
    
    # Create slug from project name
    slug = project_id.lower().replace(" ", "-").replace("_", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    slug = slug[:50]
    
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
                project = db.create_project(
                    name=project_id,
                    slug=slug,
                    source_app="insightgraph",
                    config={}
                )
                
                if not project:
                    error_msg = db.last_error if hasattr(db, 'last_error') else "Unknown error"
                    st.error(f"❌ Failed to create cloud project: {error_msg}")
                    return False
            
            # Save artifacts
            saved = []
            
            if data.get("project_summary"):
                db.save_artifact(project["id"], "project_summary", data["project_summary"], f"InsightGraph v{APP_VERSION}")
                saved.append("project_summary")
            
            if data.get("insight_cards"):
                db.save_artifact(project["id"], "insight_cards", data["insight_cards"], f"InsightGraph v{APP_VERSION}")
                saved.append("insight_cards")
            
            if data.get("markdown_report"):
                db.save_artifact(project["id"], "insight_report", data["markdown_report"], f"InsightGraph v{APP_VERSION}")
                saved.append("insight_report")
            
            if data.get("metrics_df") is not None and not data["metrics_df"].empty:
                metrics_json = data["metrics_df"].to_dict(orient="records")
                db.save_artifact(project["id"], "node_metrics", metrics_json, f"InsightGraph v{APP_VERSION}")
                saved.append("node_metrics")
            
            st.info(f"🔍 Saved artifacts: {saved}")
            st.success(f"☁️ Saved to cloud: {', '.join(saved)}")
            return True
            
        except Exception as e:
            st.error(f"❌ Cloud save failed: {e}")
            return False


# =============================================================================
# Rendering Functions (Aligned with OrgGraph US/CA)
# =============================================================================

def render_network_results(summary: dict):
    """
    Render Network Results from project_summary.json.
    Uses same icons/labels as OrgGraph US/CA.
    """
    st.subheader("📊 Network Results")
    st.caption("These metrics describe the merged exported network for this project.")
    
    # Row 1: Node and Edge counts
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏛️ Organizations", f"{summary['node_counts']['organizations']:,}")
    col2.metric("👤 People", f"{summary['node_counts']['people']:,}")
    col3.metric("💰 Grant Edges", f"{summary['edge_counts']['grants']:,}")
    col4.metric("🪪 Board Edges", f"{summary['edge_counts']['board_memberships']:,}")
    
    # Row 2: Funding metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Total Funding", f"${summary['funding']['total_amount']:,.0f}")
    col2.metric("🎁 Funders", summary['funding']['funder_count'])
    col3.metric("🎯 Grantees", summary['funding']['grantee_count'])
    
    # Governance metric if available
    if "governance" in summary and "multi_board_people" in summary["governance"]:
        col4.metric("🔗 Multi-Board People", summary['governance']['multi_board_people'])
    
    # Top funders share if available
    if "funding" in summary and "top5_share" in summary["funding"]:
        top5 = summary["funding"]["top5_share"]
        st.caption(f"*Top 5 funders account for {top5:.1%} of total funding*")


def render_grant_network_results(grants_df: pd.DataFrame, has_region_data: bool = False):
    """
    Render Grant Network Results section (aligned with OrgGraph US/CA).
    Shows All Grants (unfiltered) vs Region Relevant (filtered) summaries.
    """
    st.subheader("📌 Grant Network Results")
    st.caption(
        "Grant totals from grants_detail.csv. 'All Grants' is unfiltered. "
        "'Region Relevant' uses region_relevant == True when available."
    )
    
    if grants_df is None or grants_df.empty:
        st.info("No grants_detail.csv found for this project.")
        return
    
    # Compute summaries
    all_summary = summarize_grants_by_bucket(grants_df, region_only=False)
    region_summary = summarize_grants_by_bucket(grants_df, region_only=True) if has_region_data else None
    
    # Display All vs Region side by side
    if has_region_data and region_summary:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**All Grants (unfiltered)**")
            st.metric("Grant rows", f"{all_summary['row_count']:,}")
            st.metric("Total amount", f"${all_summary['total_amount']:,.0f}")
        with col2:
            st.markdown("**Region Relevant**")
            st.metric("Grant rows", f"{region_summary['row_count']:,}")
            st.metric("Total amount", f"${region_summary['total_amount']:,.0f}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**All Grants**")
            st.metric("Grant rows", f"{all_summary['row_count']:,}")
            st.metric("Total amount", f"${all_summary['total_amount']:,.0f}")
        with col2:
            st.caption("*Region filtering not available for this project*")
    
    # Bucket breakdown in expander
    with st.expander("📊 Show bucket breakdown", expanded=False):
        st.markdown("**All Grants by bucket**")
        if not all_summary["by_bucket"].empty:
            display_df = all_summary["by_bucket"][["bucket_label", "count", "amount"]].copy()
            display_df.columns = ["Bucket", "Count", "Amount"]
            display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No bucket data available")
        
        if has_region_data and region_summary and not region_summary["by_bucket"].empty:
            st.markdown("**Region Relevant by bucket**")
            display_df = region_summary["by_bucket"][["bucket_label", "count", "amount"]].copy()
            display_df.columns = ["Bucket", "Count", "Amount"]
            display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Source breakdown (US vs CA) if multiple sources
    source_summary = summarize_grants_by_source(grants_df)
    if not source_summary["by_source"].empty and len(source_summary["by_source"]) > 1:
        with st.expander("🌐 Show source breakdown (US vs CA)", expanded=False):
            display_df = source_summary["by_source"].copy()
            display_df.columns = ["Source", "Count", "Amount"]
            display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_health_banner(insight_cards: dict):
    """Render the Network Health banner."""
    health = insight_cards.get("health", {})
    health_score = health.get("score", 0)
    health_label = health.get("label", "Unknown")
    
    # Color based on score
    if health_score >= 70:
        health_color = "🟢"
    elif health_score >= 40:
        health_color = "🟡"
    else:
        health_color = "🔴"
    
    st.markdown(f"### {health_color} Network Health: **{health_score}/100** — *{health_label}*")
    st.caption("This score reflects funder coordination, governance ties, and funding concentration.")
    
    # Health factors in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Positive Factors**")
        for f in health.get("positive", []):
            st.markdown(f"- {f}")
        if not health.get("positive"):
            st.caption("*No strong positive factors*")
    with col2:
        st.markdown("**⚠️ Risk Factors**")
        for f in health.get("risk", []):
            st.markdown(f"- {f}")
        if not health.get("risk"):
            st.caption("*No significant risks*")


def render_insight_card(card: dict):
    """Render a single insight card with narratives."""
    with st.container():
        st.markdown(f"#### {card['title']}")
        st.caption(f"Use Case: {card['use_case']}")
        
        # Render summary
        st.markdown(card['summary'])
        
        # Health factors (special handling for network health card)
        if "health_factors" in card and card["health_factors"]:
            factors = card["health_factors"]
            if factors.get("positive") or factors.get("risk"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Positive Factors**")
                    for f in factors.get("positive", []):
                        st.markdown(f"- {f}")
                with col2:
                    st.markdown("**⚠️ Risk Factors**")
                    for f in factors.get("risk", []):
                        st.markdown(f"- {f}")
        
        # Ranked rows with narratives
        ranked_rows = card.get("ranked_rows", [])
        if ranked_rows:
            has_narratives = any(r.get("narrative") for r in ranked_rows)
            has_interpretations = any(r.get("interpretation") for r in ranked_rows)
            
            if has_interpretations and not has_narratives:
                # Health-style indicators
                st.markdown("---")
                for row in ranked_rows:
                    indicator = row.get("indicator", "")
                    value = row.get("value", "")
                    interpretation = row.get("interpretation", "")
                    st.markdown(f"**{indicator}:** {value}")
                    if interpretation:
                        st.caption(f"↳ {interpretation}")
                    
            elif has_narratives:
                # Render as expandable narrative sections
                for row in ranked_rows:
                    entity_name = (
                        row.get("grantee") or 
                        row.get("person") or 
                        row.get("org") or 
                        row.get("funder") or 
                        row.get("node") or 
                        row.get("pair") or
                        f"#{row.get('rank', '')}"
                    )
                    
                    with st.expander(f"**{row.get('rank', '')}. {entity_name}**"):
                        if row.get("narrative"):
                            st.markdown(row["narrative"])
                        if row.get("recommendation"):
                            st.markdown(row["recommendation"])
                        
                        # Show key metrics
                        metrics_to_show = ["funders", "boards", "amount", "outflow", "shared", "jaccard", "betweenness"]
                        metric_parts = []
                        for m in metrics_to_show:
                            if m in row:
                                val = row[m]
                                if isinstance(val, float):
                                    val = f"{val:.3f}" if val < 1 else f"{val:,.0f}"
                                metric_parts.append(f"**{m.title()}:** {val}")
                        if metric_parts:
                            st.caption(" • ".join(metric_parts))
            else:
                # Render as simple table
                df = pd.DataFrame(ranked_rows)
                for col in df.columns:
                    if "amount" in col.lower() or "received" in col.lower() or "outflow" in col.lower():
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                
                display_cols = [c for c in df.columns if not c.endswith("_id") and not c.endswith("_ids") and c != "rank"]
                if display_cols:
                    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        
        # Evidence summary
        evidence = card.get("evidence", {})
        node_count = len(evidence.get("node_ids", []))
        edge_count = len(evidence.get("edge_ids", []))
        if node_count > 0 or edge_count > 0:
            st.caption(f"📎 Evidence: {node_count} nodes, {edge_count} edges")
        
        st.divider()


def render_insight_cards_section(insight_cards: dict):
    """Render the Insight Cards section."""
    st.subheader("🧠 Insight Cards")
    st.caption("Cards are generated from the exported network metrics. Each card is a narrative interpretation with ranked evidence.")
    
    cards = insight_cards.get("cards", [])
    
    if not cards:
        st.info("No insight cards available for this project.")
        return
    
    # Card filter
    use_cases = list(set(c["use_case"] for c in cards))
    selected_use_case = st.selectbox(
        "Filter by Use Case",
        ["All"] + sorted(use_cases),
        index=0
    )
    
    filtered_cards = cards if selected_use_case == "All" else [c for c in cards if c["use_case"] == selected_use_case]
    
    for card in filtered_cards:
        render_insight_card(card)


def render_node_metrics(metrics_df: pd.DataFrame):
    """Render node metrics in expanders (aligned with OrgGraph pattern)."""
    with st.expander("👀 Preview Node Metrics", expanded=False):
        tab1, tab2 = st.tabs(["Organizations", "People"])
        
        with tab1:
            org_metrics = metrics_df[metrics_df["node_type"] == "ORG"]
            display_cols = ["label", "degree", "grant_in_degree", "grant_out_degree", 
                          "grant_outflow_total", "betweenness", "shared_board_count",
                          "is_connector", "is_broker", "is_hidden_broker", "is_capital_hub"]
            display_cols = [c for c in display_cols if c in org_metrics.columns]
            
            if not org_metrics.empty and display_cols:
                sort_col = "grant_outflow_total" if "grant_outflow_total" in display_cols else display_cols[0]
                st.dataframe(
                    org_metrics[display_cols].sort_values(sort_col, ascending=False), 
                    use_container_width=True, 
                    hide_index=True
                )
                st.caption(f"{len(org_metrics)} organizations")
            else:
                st.info("No organization metrics available")
        
        with tab2:
            person_metrics = metrics_df[metrics_df["node_type"] == "PERSON"]
            display_cols = ["label", "boards_served", "betweenness", "is_connector"]
            display_cols = [c for c in display_cols if c in person_metrics.columns]
            
            if not person_metrics.empty and display_cols:
                sort_col = "boards_served" if "boards_served" in display_cols else display_cols[0]
                st.dataframe(
                    person_metrics[display_cols].sort_values(sort_col, ascending=False), 
                    use_container_width=True, 
                    hide_index=True
                )
                st.caption(f"{len(person_metrics)} people")
            else:
                st.info("No people metrics available")


def render_grant_purpose_explorer(grants_df: pd.DataFrame, project_id: str):
    """
    Render Grant Purpose Explorer section.
    Uses keyword-based classification (no AI) to categorize grants by purpose.
    """
    st.subheader("🎯 Grant Purpose Explorer")
    st.caption("Keyword-based purpose classification. Filter grants by thematic area to explore funding patterns.")
    
    if grants_df is None or grants_df.empty:
        st.info(f"Grant purpose analysis unavailable for this project (missing `grants_detail.csv` in `demo_data/{project_id}/`).")
        return
    
    # Check if we have purpose classifications
    if "purpose_primary" not in grants_df.columns:
        st.warning("Purpose classifications not available. Ensure grants_detail.csv has a 'grant_purpose_raw' column.")
        return
    
    # Get all unique tags
    all_tags = set()
    for tags_str in grants_df["purpose_tags"].dropna():
        for t in str(tags_str).split("|"):
            if t:
                all_tags.add(t)
    all_tags = sorted(all_tags)
    
    # Get tag labels for display
    tag_label_map = {}
    for cat, info in CORE_PURPOSE_CATEGORIES.items():
        tag_label_map[cat] = info["label"]
    tag_label_map["uncategorized"] = "❓ Uncategorized"
    
    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_grants = len(grants_df)
    total_amount = grants_df["grant_amount"].sum() if "grant_amount" in grants_df.columns else 0
    categorized = len(grants_df[grants_df["purpose_primary"] != "uncategorized"])
    uncategorized = total_grants - categorized
    
    col1.metric("📋 Total Grants", f"{total_grants:,}")
    col2.metric("💵 Total Funding", f"${total_amount:,.0f}")
    col3.metric("✅ Categorized", f"{categorized:,} ({100*categorized/total_grants:.0f}%)" if total_grants > 0 else "0")
    col4.metric("❓ Uncategorized", f"{uncategorized:,}")
    
    st.markdown("---")
    
    # Purpose breakdown by category
    st.markdown("**Funding by Purpose Category**")
    
    # Build summary by primary category
    if "grant_amount" in grants_df.columns:
        purpose_summary = grants_df.groupby("purpose_primary").agg(
            count=("purpose_primary", "count"),
            amount=("grant_amount", "sum")
        ).reset_index()
        purpose_summary["label"] = purpose_summary["purpose_primary"].map(tag_label_map).fillna(purpose_summary["purpose_primary"])
        purpose_summary = purpose_summary.sort_values("amount", ascending=False)
        
        # Display as horizontal metrics
        cols = st.columns(min(len(purpose_summary), 4))
        for i, row in enumerate(purpose_summary.head(8).itertuples()):
            col_idx = i % 4
            with cols[col_idx]:
                st.metric(
                    row.label,
                    f"{row.count:,} grants",
                    f"${row.amount:,.0f}"
                )
    
    st.markdown("---")
    
    # Filter section
    st.markdown("**Filter Grants by Purpose**")
    
    # Tag filter (multi-select)
    filter_options = [tag_label_map.get(t, t) for t in all_tags]
    tag_to_key = {tag_label_map.get(t, t): t for t in all_tags}
    
    selected_labels = st.multiselect(
        "Select purpose categories",
        options=filter_options,
        default=[],
        help="Select one or more purpose categories to filter grants. Leave empty to show all."
    )
    
    # Convert selected labels back to keys
    selected_tags = [tag_to_key.get(lbl, lbl) for lbl in selected_labels]
    
    # Apply filter
    if selected_tags:
        mask = grants_df["purpose_tags"].fillna("").apply(
            lambda s: any(t in str(s).split("|") for t in selected_tags)
        )
        filtered_df = grants_df[mask]
    else:
        filtered_df = grants_df
    
    # Show filtered results
    st.markdown(f"**Showing {len(filtered_df):,} grants** ({100*len(filtered_df)/total_grants:.0f}% of total)")
    
    if "grant_amount" in filtered_df.columns:
        filtered_amount = filtered_df["grant_amount"].sum()
        st.caption(f"Total filtered funding: ${filtered_amount:,.0f}")
    
    # Display grants table
    # Try foundation_name first (OrgGraph US), then funder_name (legacy)
    display_cols = ["foundation_name", "funder_name", "grantee_name", "grant_amount", "grant_purpose_raw", "purpose_primary_label", "grant_bucket", "fiscal_year"]
    # Filter to columns that exist, but avoid duplicates (foundation_name and funder_name are aliases)
    available_cols = []
    seen_funder_col = False
    for c in display_cols:
        if c in filtered_df.columns:
            if c in ("foundation_name", "funder_name"):
                if not seen_funder_col:
                    available_cols.append(c)
                    seen_funder_col = True
            else:
                available_cols.append(c)
    display_cols = available_cols
    
    if display_cols:
        # Format amount column
        show_df = filtered_df[display_cols].copy()
        for col in show_df.columns:
            if "amount" in col.lower():
                show_df[col] = show_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
        
        st.dataframe(show_df.head(100), use_container_width=True, hide_index=True)
        
        if len(filtered_df) > 100:
            st.caption(f"Showing first 100 of {len(filtered_df):,} grants. Download full data below.")
    
    # Download filtered data
    if not filtered_df.empty:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Grants (CSV)",
            data=csv_data,
            file_name=f"{project_id}_grants_filtered.csv",
            mime="text/csv"
        )


def render_downloads(data: dict):
    """Render download section with simplified UI."""
    st.subheader("📥 Export Data")
    
    # Check what's available
    has_report = data.get("markdown_report") is not None
    has_cards = data.get("insight_cards") is not None
    has_summary = data.get("project_summary") is not None
    has_metrics = data.get("metrics_df") is not None
    has_nodes = data.get("nodes_df") is not None
    has_edges = data.get("edges_df") is not None
    
    # Generate HTML report ONCE (used for both ZIP and standalone download)
    html_report = None
    html_error = None
    run_module = load_run_module()
    
    if has_report and run_module and hasattr(run_module, 'render_html_report'):
        try:
            html_report = run_module.render_html_report(
                markdown_content=data["markdown_report"],
                project_summary=data.get("project_summary", {}),
                insight_cards=data.get("insight_cards", {}),
                project_id=data.get("project_id", "report")
            )
        except Exception as e:
            html_error = str(e)
            print(f"Warning: Could not generate HTML report: {e}")
    elif has_report and run_module:
        html_error = "render_html_report function not found in run.py"
    elif has_report:
        html_error = "Could not load run.py module"
    
    def generate_bundle_readme():
        """Generate README.md for the bundle."""
        project_id = data.get("project_id", "unknown")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        
        node_count = len(data.get("nodes_df", [])) if data.get("nodes_df") is not None else 0
        edge_count = len(data.get("edges_df", [])) if data.get("edges_df") is not None else 0
        
        return f"""# {project_id} — InsightGraph Export

**Generated:** {timestamp}  
**Source App:** InsightGraph v{APP_VERSION}

## Summary

- **Nodes:** {node_count:,}
- **Edges:** {edge_count:,}

---

## Files in This Bundle

| File | Description |
|------|-------------|
| `index.html` | Interactive HTML report — open in any browser |
| `report.md` | Markdown source for the report |
| `manifest.json` | Bundle metadata and analysis parameters |
| `analysis/node_metrics.csv` | Centrality scores and network metrics for each node |
| `analysis/insight_cards.json` | Structured insight data (health scores, key findings) |
| `analysis/project_summary.json` | Project-level summary statistics |
| `data/nodes.csv` | Source node data |
| `data/edges.csv` | Source edge data |
| `data/grants_detail.csv` | Grant details (if available) |

---

## Using the HTML Report

1. Open `index.html` in any web browser
2. Use Ctrl/Cmd+P to print to PDF if needed
3. Share the HTML file directly — it's self-contained

---

## Node Metrics Columns

| Column | Description |
|--------|-------------|
| `node_id` | Unique identifier |
| `label` | Display name |
| `degree_centrality` | How connected (0-1) |
| `betweenness_centrality` | How often on shortest paths (0-1) |
| `eigenvector_centrality` | Connected to well-connected nodes (0-1) |
| `community` | Community/cluster assignment |

---

For questions, contact: info@connectingforchangellc.com
"""
    
    def create_bundle_zip():
        """Create structured bundle ZIP with manifest, README, and HTML report."""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # README.md
            zf.writestr("README.md", generate_bundle_readme())
            
            # Analysis outputs
            if has_metrics:
                zf.writestr("analysis/node_metrics.csv", data["metrics_df"].to_csv(index=False))
            if has_cards:
                zf.writestr("analysis/insight_cards.json", json.dumps(data["insight_cards"], indent=2))
            if has_summary:
                zf.writestr("analysis/project_summary.json", json.dumps(data["project_summary"], indent=2))
            
            # Markdown report
            if has_report:
                zf.writestr("report.md", data["markdown_report"])
            
            # HTML report (use pre-generated from outer scope)
            if html_report:
                zf.writestr("index.html", html_report)
            
            # Input data (if available)
            if has_nodes:
                zf.writestr("data/nodes.csv", data["nodes_df"].to_csv(index=False))
            if has_edges:
                zf.writestr("data/edges.csv", data["edges_df"].to_csv(index=False))
            if data.get("grants_df") is not None and not data["grants_df"].empty:
                zf.writestr("data/grants_detail.csv", data["grants_df"].to_csv(index=False))
            
            # Generate manifest
            try:
                if run_module and hasattr(run_module, 'generate_manifest'):
                    manifest = run_module.generate_manifest(
                        project_id=data.get("project_id", "unknown"),
                        project_summary=data.get("project_summary", {}),
                        insight_cards=data.get("insight_cards", {}),
                        nodes_df=data.get("nodes_df"),
                        edges_df=data.get("edges_df"),
                        grants_df=data.get("grants_df"),
                        region_lens_config=data.get("region_lens_config"),
                        cloud_project_id=data.get("cloud_project_id")
                    )
                    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            except Exception as e:
                print(f"Warning: Could not generate manifest: {e}")
                # Continue without manifest
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # ==========================================================================
    # Two-button primary layout: HTML Report + ZIP Bundle
    # ==========================================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        if html_report:
            # Ensure HTML is encoded as UTF-8 bytes
            html_bytes = html_report.encode('utf-8') if isinstance(html_report, str) else html_report
            st.download_button(
                "📄 Download Report (HTML)",
                data=html_bytes,
                file_name=f"{data['project_id']}_report.html",
                mime="text/html",
                type="primary",
                use_container_width=True,
                help="Open in any browser, print to PDF, or share directly"
            )
        elif has_report:
            if html_error:
                st.warning(f"HTML generation failed: {html_error}")
            st.download_button(
                "📝 Download Report (Markdown)",
                data=data["markdown_report"],
                file_name="insight_report.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True,
                help="Source markdown — convert to PDF with any markdown editor"
            )
        else:
            st.info("Run insights first to generate a report")
    
    with col2:
        if has_report or has_cards or has_summary or has_metrics:
            st.download_button(
                "📦 Download ZIP",
                data=create_bundle_zip(),
                file_name=f"{data['project_id']}_insights.zip",
                mime="application/zip",
                use_container_width=True,
                help="Everything in one bundle — report, data, metrics, ready for sharing or archiving"
            )
        else:
            st.button("📦 Download ZIP", disabled=True, use_container_width=True,
                     help="Run insights first to generate exportable data")
    
    # Help text
    st.caption("""
    **ZIP contains:** README.md · index.html · report.md · manifest.json · 
    analysis/ folder with metrics · data/ folder with source files
    """)
    
    # ==========================================================================
    # Cloud save options (for merged/linked projects)
    # ==========================================================================
    
    is_merged = st.session_state.get("merged_projects") is not None
    is_linked = st.session_state.get("linked_project_info") is not None
    client = get_project_store_authenticated()
    cloud_enabled = client is not None
    
    # Display link stats if this is a linked project
    if is_linked:
        st.divider()
        link_stats = st.session_state.get("linked_project_info", {})
        
        st.markdown("**🔗 Entity Linking Results**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Auto-matched", link_stats.get("auto_matched", 0))
        col2.metric("✓ User Confirmed", link_stats.get("confirmed", 0))
        col3.metric("✗ Rejected", link_stats.get("rejected", 0))
        col4.metric("🔗 Total Linked", link_stats.get("total_linked", 0))
        
        # Calculate and show coverage
        total_nodes = data.get("node_count", 0)
        total_linked = link_stats.get("total_linked", 0)
        if total_nodes > 0:
            coverage_pct = (total_linked / total_nodes) * 100
            st.caption(f"📊 {total_linked} of {total_nodes} organizations linked with LinkedIn data ({coverage_pct:.1f}% coverage)")
        
        st.caption("Linked organizations now have: `linkedin_url`, `linkedin_industry`, `linkedin_website` columns")
        
        # Save linked project to cloud
        if cloud_enabled:
            st.markdown("**☁️ Save Linked Project to Cloud**")
            st.caption("Save so you can reload this linked network later without re-linking.")
            
            # Get default name from source projects
            cloud_data = st.session_state.get("cloud_project_data", {})
            source_manifest = cloud_data.get("manifest", {})
            source_projects = source_manifest.get("source_projects", ["actorgraph", "orggraph"])
            default_name = f"Linked: {' + '.join(source_projects[:2])}"
            
            col_name, col_btn = st.columns([3, 1])
            
            with col_name:
                linked_name = st.text_input(
                    "Project name",
                    value=default_name,
                    key="linked_project_name",
                    label_visibility="collapsed",
                    placeholder="Enter name for linked project"
                )
            
            with col_btn:
                if st.button("☁️ Save", type="primary", use_container_width=True, key="save_linked_btn"):
                    if linked_name:
                        with st.spinner("☁️ Saving linked project..."):
                            success, message, slug = save_linked_to_cloud(
                                name=linked_name,
                                data=cloud_data
                            )
                            if success:
                                st.success(f"✅ {message}")
                                # Clear the linked state since it's now saved
                                st.session_state.linked_project_info = None
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("Enter a name for the linked project")
    
    if is_merged and cloud_enabled:
        st.divider()
        st.markdown("**☁️ Save Merged Project to Cloud**")
        st.caption("Save so you can reload this merged dataset later without re-merging.")
        
        # Get default name from merged sources
        merged_slugs = st.session_state.get("merged_projects", [])
        default_name = f"Merged: {' + '.join(merged_slugs[:2])}"
        if len(merged_slugs) > 2:
            default_name += f" (+{len(merged_slugs) - 2})"
        
        col_name, col_btn = st.columns([3, 1])
        
        with col_name:
            merged_name = st.text_input(
                "Project name",
                value=default_name,
                key="merged_project_name",
                label_visibility="collapsed",
                placeholder="Enter name for merged project"
            )
        
        with col_btn:
            if st.button("☁️ Save", type="primary", use_container_width=True):
                if merged_name:
                    with st.spinner("☁️ Saving merged project..."):
                        success, message, slug = save_merged_to_cloud(
                            name=merged_name,
                            data=data,
                            source_projects=merged_slugs
                        )
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("Enter a name for the merged project")


def render_technical_details(data: dict):
    """Render technical details in expander."""
    with st.expander("🛠️ Technical Details", expanded=False):
        st.caption("Developer-facing details for troubleshooting.")
        
        if data.get("project_summary"):
            st.markdown("**project_summary.json:**")
            st.json(data["project_summary"])
        
        if data.get("insight_cards"):
            st.markdown("**insight_cards.json (health + metadata):**")
            cards_preview = {
                "health": data["insight_cards"].get("health", {}),
                "card_count": len(data["insight_cards"].get("cards", [])),
                "generated_at": data["insight_cards"].get("generated_at", "unknown"),
            }
            st.json(cards_preview)


# =============================================================================
# Compute Fallback (when pre-exports don't exist)
# =============================================================================

def compute_insights(project: dict, project_id: str) -> dict:
    """
    Compute insights from nodes/edges when pre-exports don't exist.
    
    Uses network-type-aware analysis (v0.15.10+):
    - Automatically detects funder vs social network
    - Routes to appropriate analyzer
    - Returns standardized result format
    """
    run_module = load_run_module()
    if not run_module:
        st.error("Cannot compute insights: run.py not found")
        return None
    
    path = project["path"]
    nodes_path = path / "nodes.csv"
    edges_path = path / "edges.csv"
    
    with st.spinner("Computing insights from network data..."):
        try:
            # Load data
            nodes_df = pd.read_csv(nodes_path)
            edges_df = pd.read_csv(edges_path)
            
            # Check if new analyzer system is available
            if hasattr(run_module, 'analyze_network'):
                # NEW: Use network-type-aware analyzer
                result = run_module.analyze_network(nodes_df, edges_df, project_id)
                
                # Convert AnalysisResult to dict format expected by rest of app
                return {
                    "project_id": project_id,
                    "network_type": result.network_type,
                    "nodes_df": result.nodes_df if result.nodes_df is not None else nodes_df,
                    "edges_df": result.edges_df if result.edges_df is not None else edges_df,
                    "metrics_df": result.metrics_df,
                    "insight_cards": result.to_insight_cards_dict(),
                    "project_summary": result.to_project_summary_dict(),
                    "markdown_report": result.markdown_report,
                    "roles_region_summary": result.project_summary.roles_region if hasattr(result.project_summary, 'roles_region') else None,
                    "brokerage": result.brokerage.to_dict() if result.brokerage else None,
                }
            else:
                # LEGACY: Fall back to old method for backward compatibility
                nodes_df, edges_df = run_module.load_and_validate(nodes_path, edges_path)
                
                # Build graphs
                grant_graph = run_module.build_grant_graph(nodes_df, edges_df)
                board_graph = run_module.build_board_graph(nodes_df, edges_df)
                interlock_graph = run_module.build_interlock_graph(nodes_df, edges_df)
                
                # Compute metrics
                metrics_df = run_module.compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
                metrics_df = run_module.compute_derived_signals(metrics_df)
                
                # Flow stats
                flow_stats = run_module.compute_flow_stats(edges_df, metrics_df)
                overlap_df = run_module.compute_portfolio_overlap(edges_df)
                
                # Roles × Region Lens (v3.0.5+)
                lens_config = run_module.load_region_lens_config(path)
                nodes_with_roles = run_module.derive_network_roles(nodes_df.copy(), edges_df)
                nodes_with_lens = run_module.compute_region_lens_membership(nodes_with_roles, lens_config)
                roles_region_summary = run_module.generate_roles_region_summary(nodes_with_lens, edges_df, lens_config)
                
                # Generate insights
                insight_cards = run_module.generate_insight_cards(
                    nodes_df, edges_df, metrics_df,
                    interlock_graph, flow_stats, overlap_df,
                    project_id=project_id
                )
                
                # Project summary
                project_summary = run_module.generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
                project_summary['roles_region'] = roles_region_summary
                
                # Generate markdown report (with roles/region section)
                markdown_report = run_module.generate_markdown_report(
                    insight_cards, project_summary, project_id, roles_region_summary
                )
                
                return {
                    "project_id": project_id,
                    "nodes_df": nodes_df,
                    "edges_df": edges_df,
                    "metrics_df": metrics_df,
                    "insight_cards": insight_cards,
                    "project_summary": project_summary,
                    "markdown_report": markdown_report,
                    "roles_region_summary": roles_region_summary,
                }
            
        except Exception as e:
            st.error(f"Error computing insights: {e}")
            import traceback
            traceback.print_exc()
            return None


def compute_insights_from_dataframes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                                      grants_df: pd.DataFrame, project_id: str,
                                      manifest: dict = None) -> dict:
    """
    Compute insights from dataframes directly (for cloud projects).
    
    This is used when loading from cloud storage where we have dataframes
    but no file paths.
    
    Uses network-type-aware analysis (v0.15.10+):
    - Automatically detects funder vs social network
    - Routes to appropriate analyzer
    
    Args:
        manifest: Optional manifest dict that may contain overlap_analysis for linked projects
    """
    run_module = load_run_module()
    if not run_module:
        st.error("Cannot compute insights: run.py not found")
        return None
    
    if nodes_df is None or edges_df is None:
        st.error("Missing nodes or edges data")
        return None
    
    with st.spinner("Computing insights from cloud data..."):
        try:
            # Validate dataframes
            if nodes_df.empty:
                st.error("Nodes dataframe is empty")
                return None
            if edges_df.empty:
                st.warning("Edges dataframe is empty - some metrics may be limited")
            
            # Check if new analyzer system is available
            if hasattr(run_module, 'analyze_network'):
                # NEW: Use network-type-aware analyzer
                result = run_module.analyze_network(nodes_df, edges_df, project_id)
                
                # Get markdown report
                markdown_report = result.markdown_report
                
                # Convert AnalysisResult to dict format
                output = {
                    "project_id": project_id,
                    "network_type": result.network_type,
                    "nodes_df": result.nodes_df if result.nodes_df is not None else nodes_df,
                    "edges_df": result.edges_df if result.edges_df is not None else edges_df,
                    "metrics_df": result.metrics_df,
                    "insight_cards": result.to_insight_cards_dict(),
                    "project_summary": result.to_project_summary_dict(),
                    "markdown_report": markdown_report,
                    "roles_region_summary": result.project_summary.roles_region if hasattr(result.project_summary, 'roles_region') else None,
                    "brokerage": result.brokerage.to_dict() if result.brokerage else None,
                }
            else:
                # LEGACY: Fall back to old method for backward compatibility
                # Build graphs
                grant_graph = run_module.build_grant_graph(nodes_df, edges_df)
                board_graph = run_module.build_board_graph(nodes_df, edges_df)
                interlock_graph = run_module.build_interlock_graph(nodes_df, edges_df)
                
                # Compute metrics
                metrics_df = run_module.compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
                metrics_df = run_module.compute_derived_signals(metrics_df)
                
                # Flow stats
                flow_stats = run_module.compute_flow_stats(edges_df, metrics_df)
                overlap_df = run_module.compute_portfolio_overlap(edges_df)
                
                # Roles × Region Lens - use disabled config for cloud projects
                lens_config = {'enabled': False}
                nodes_with_roles = run_module.derive_network_roles(nodes_df.copy(), edges_df)
                nodes_with_lens = run_module.compute_region_lens_membership(nodes_with_roles, lens_config)
                roles_region_summary = run_module.generate_roles_region_summary(nodes_with_lens, edges_df, lens_config)
                
                # Generate insights
                insight_cards = run_module.generate_insight_cards(
                    nodes_df, edges_df, metrics_df,
                    interlock_graph, flow_stats, overlap_df,
                    project_id=project_id
                )
                
                # Project summary
                project_summary = run_module.generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
                project_summary['roles_region'] = roles_region_summary
                
                # Generate markdown report
                markdown_report = run_module.generate_markdown_report(
                    insight_cards, project_summary, project_id, roles_region_summary
                )
                
                output = {
                    "project_id": project_id,
                    "nodes_df": nodes_df,
                    "edges_df": edges_df,
                    "metrics_df": metrics_df,
                    "insight_cards": insight_cards,
                    "project_summary": project_summary,
                    "markdown_report": markdown_report,
                    "roles_region_summary": roles_region_summary,
                }
            
            # Add overlap analysis section if this is a linked project
            # (This applies to both new and legacy paths)
            # Debug: Log manifest status
            print(f"DEBUG: manifest present: {manifest is not None}")
            if manifest:
                print(f"DEBUG: manifest keys: {list(manifest.keys())}")
                print(f"DEBUG: overlap_analysis present: {'overlap_analysis' in manifest}")
                if 'overlap_analysis' in manifest:
                    print(f"DEBUG: overlap_analysis: {manifest['overlap_analysis']}")
            
            if manifest and manifest.get("overlap_analysis"):
                overlap = manifest["overlap_analysis"]
                match_stats = manifest.get("match_stats", {})
                
                overlap_section = f"""

---

## 🔗 Network Overlap Analysis

*Are the organizations shaping this ecosystem also connected to formal funding flows?*

### Overlap Summary

| Metric | Value |
|--------|-------|
| **Overlap** | {overlap.get('overlap_pct', 0):.1f}% |
| **Signal** | {overlap.get('overlap_signal', 'Unknown')} |
| **Influence Network Orgs** | {overlap.get('total_actor_orgs', 0)} |
| **Funding Network Orgs** | {overlap.get('total_org_orgs', 0)} |
| **Potential Matches** | {overlap.get('potential_matches', 0)} |
| **Unmatched (Influence)** | {overlap.get('unmatched_actor', 0)} |

### Match Statistics

| Metric | Count |
|--------|-------|
| Auto-matched | {match_stats.get('auto_matched', 0)} |
| User Confirmed | {match_stats.get('confirmed', 0)} |
| User Rejected | {match_stats.get('rejected', 0)} |
| **Total Linked** | {match_stats.get('total_linked', 0)} |

### Interpretation

"""
                # Add interpretation based on level
                level = overlap.get('overlap_level', 'Unknown')
                if level == "High":
                    overlap_section += """**Structural Alignment** — Coalition activity is well-resourced and institutionally supported. 
Funding reinforces existing influence pathways. 

*Risk:* Potential for groupthink or incumbency bias — emergent voices may be crowded out.
"""
                elif level == "Moderate":
                    overlap_section += """**Partial Alignment** — A hybrid ecosystem where momentum is not fully matched by capital.

Some influential organizations may be under-resourced relative to their network role. 
Funding may be reinforcing established actors rather than emergent ones.
"""
                else:
                    overlap_section += """**Structural Disconnect** — Influence and funding are decoupled. 
The coalition relies on under-resourced actors operating outside formal funding streams.

*Opportunity:* High leverage for first-mover funders willing to support this space.
"""
                
                # Add strategic implications
                overlap_section += """
### Strategic Implications

**For Funders:**
- Unmatched organizations represent potential early-signal funding opportunities
- Check for over-concentration on "usual suspects"
- Key question: *Are we funding the organizations that actually move the system?*

**For Coalitions:**
- Identify high-centrality, low-funding organizations at risk of burnout
- Assess power asymmetry: who coordinates vs who controls resources
- Key question: *Which parts of our network are carrying disproportionate load without capital?*

**For Intermediaries:**
- Target brokerage or fiscal mechanisms to bridge funding gaps
- Identify translation gaps where structure/form blocks funding access
- Key question: *Where would a small structural intervention unlock outsized impact?*

---

*This analysis surfaces structural patterns, not recommendations. Interpretation requires context.*
"""
                output["markdown_report"] += overlap_section
            
            return output
            
        except Exception as e:
            st.error(f"Error computing insights: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None

def main():
    init_session_state()
    
    # ==========================================================================
    # Header
    # ==========================================================================
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(INSIGHTGRAPH_ICON_URL, width=60)
    with col_title:
        st.title("InsightGraph")
    
    st.markdown("Structured insight from complex networks.")
    st.caption(f"v{APP_VERSION}")
    
    # Cloud status in sidebar
    render_cloud_status()
    
    st.divider()
    
    # ==========================================================================
    # Section 1: Project Selection
    # ==========================================================================
    st.subheader("🗂️ Project")
    
    # Check if user is authenticated for cloud access
    client = get_project_store_authenticated()
    cloud_available = client is not None
    
    # Tabs for Local vs Cloud vs Manage
    if cloud_available:
        tab_local, tab_cloud, tab_manage = st.tabs(["📂 Local Projects", "☁️ Cloud Projects", "⚙️ Manage"])
    else:
        tab_local = st.container()
        tab_cloud = None
        tab_manage = None
    
    selected_project = None
    inputs = None
    is_cloud_project = False
    
    # -------------------------------------------------------------------------
    # LOCAL PROJECTS TAB
    # -------------------------------------------------------------------------
    with tab_local:
        st.caption("Select a project folder containing nodes.csv and edges.csv (exported from OrgGraph).")
        
        projects = get_projects()
        
        if not projects:
            st.warning(f"""
            **No local projects found.**
            
            Expected location: `{DEMO_DATA_DIR}`
            
            Each project folder must contain:
            - `nodes.csv` (required)
            - `edges.csv` (required)
            - `grants_detail.csv` (optional, enables Purpose Explorer)
            """)
            if not cloud_available:
                st.info("💡 Login to access cloud projects from OrgGraph or ActorGraph.")
        else:
            # Build project selector
            project_options = [p["id"] for p in projects]
            
            selected_id = st.selectbox(
                "Select project",
                project_options,
                label_visibility="collapsed",
                key="local_project_select"
            )
            
            selected_project = next((p for p in projects if p["id"] == selected_id), None)
            
            if selected_project:
                # Show project input status
                st.markdown(f"**📂 {selected_id}**")
                
                col1, col2, col3 = st.columns(3)
                col1.markdown("✅ nodes.csv" if selected_project["has_nodes"] else "❌ nodes.csv")
                col2.markdown("✅ edges.csv" if selected_project["has_edges"] else "❌ edges.csv")
                col3.markdown("✅ grants_detail.csv" if selected_project["has_grants_detail"] else "⚪ grants_detail.csv (optional)")
                
                if selected_project["has_precomputed"]:
                    st.caption("💾 *Pre-computed results available from previous run*")
                
                # Load input data
                inputs = load_project_inputs(selected_project)
    
    # -------------------------------------------------------------------------
    # CLOUD PROJECTS TAB
    # -------------------------------------------------------------------------
    if tab_cloud is not None:
        with tab_cloud:
            st.caption("Load network bundles saved from OrgGraph US, OrgGraph CA, or ActorGraph.")
            
            cloud_projects = list_cloud_projects()
            
            if not cloud_projects:
                st.info("No cloud projects found. Save a project from OrgGraph or ActorGraph first.")
            else:
                # Build cloud project options with source app icons
                source_icons = {
                    'orggraph_us': '🇺🇸',
                    'orggraph_ca': '🇨🇦',
                    'actorgraph': '🕸️',
                    'insightgraph': '🔀',  # Merged projects
                }
                
                # Initialize selection state
                if "cloud_project_selections" not in st.session_state:
                    st.session_state.cloud_project_selections = {}
                
                # Mode toggle: Single vs Merge vs Link
                load_mode = st.radio(
                    "Load mode",
                    ["📄 Single Project", "🔀 Merge Multiple", "🔗 Link Entities"],
                    horizontal=True,
                    key="cloud_load_mode"
                )
                
                st.divider()
                
                if load_mode == "📄 Single Project":
                    # Original single-select dropdown
                    cloud_options = []
                    for p in cloud_projects:
                        icon = source_icons.get(p.source_app, '📦')
                        label = f"{icon} {p.name} ({p.node_count} nodes, {p.edge_count} edges)"
                        cloud_options.append((p.id, p.slug, label, p))
                    
                    selected_cloud_label = st.selectbox(
                        "Select cloud project",
                        [opt[2] for opt in cloud_options],
                        label_visibility="collapsed",
                        key="cloud_project_select"
                    )
                    
                    # Find selected cloud project
                    selected_cloud = None
                    for opt in cloud_options:
                        if opt[2] == selected_cloud_label:
                            selected_cloud = opt[3]
                            break
                    
                    if selected_cloud:
                        st.markdown(f"**☁️ {selected_cloud.name}**")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Nodes", selected_cloud.node_count)
                        col2.metric("Edges", selected_cloud.edge_count)
                        col3.markdown(f"**Source:** {selected_cloud.source_app}")
                        
                        if selected_cloud.jurisdiction:
                            st.caption(f"Jurisdiction: {selected_cloud.jurisdiction}")
                        
                        # Load from cloud button
                        if st.button("☁️ Load from Cloud", type="primary", use_container_width=True):
                            with st.spinner("☁️ Downloading bundle..."):
                                cloud_data = load_cloud_project(project_id=selected_cloud.id)
                                
                                if cloud_data.get("error"):
                                    st.error(f"❌ {cloud_data['error']}")
                                else:
                                    # Store in session state
                                    st.session_state.cloud_project_data = cloud_data
                                    st.session_state.current_project_id = f"cloud:{selected_cloud.slug}"
                                    st.session_state.project_data = None  # Clear local project data
                                    st.session_state.merged_projects = None  # Clear merge state
                                    st.session_state.linked_project_info = None  # Clear link state
                                    st.success(f"✅ Loaded {selected_cloud.name}")
                                    st.rerun()
                
                elif load_mode == "🔀 Merge Multiple":
                    # Multi-select with checkboxes
                    st.markdown("**Select projects to merge:**")
                    
                    selected_projects = []
                    total_nodes = 0
                    total_edges = 0
                    
                    for p in cloud_projects:
                        icon = source_icons.get(p.source_app, '📦')
                        col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1.5])
                        
                        with col1:
                            is_selected = st.checkbox(
                                "select",
                                key=f"merge_select_{p.id}",
                                label_visibility="collapsed"
                            )
                        
                        with col2:
                            st.markdown(f"{icon} **{p.name}**")
                        
                        with col3:
                            st.caption(f"{p.node_count} nodes")
                        
                        with col4:
                            st.caption(f"{p.edge_count} edges")
                        
                        if is_selected:
                            selected_projects.append(p)
                            total_nodes += p.node_count
                            total_edges += p.edge_count
                    
                    st.divider()
                    
                    # Show merge preview
                    if len(selected_projects) == 0:
                        st.info("Select 2 or more projects to merge.")
                    elif len(selected_projects) == 1:
                        st.warning("Select at least 2 projects to merge, or use Single Project mode.")
                    else:
                        st.markdown(f"**🔀 Merge Preview:** {len(selected_projects)} projects")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Nodes (before dedup)", total_nodes)
                        col2.metric("Total Edges (before dedup)", total_edges)
                        
                        # Show source breakdown
                        sources = {}
                        for p in selected_projects:
                            src = p.source_app
                            sources[src] = sources.get(src, 0) + 1
                        source_summary = ", ".join([f"{source_icons.get(k, '📦')} {v}" for k, v in sources.items()])
                        col3.markdown(f"**Sources:** {source_summary}")
                        
                        st.caption("⚠️ Node/edge counts may decrease after deduplication.")
                        
                        # Merge button
                        if st.button("🔀 Merge & Load", type="primary", use_container_width=True):
                            with st.spinner(f"☁️ Downloading and merging {len(selected_projects)} projects..."):
                                merged_data = merge_cloud_projects(selected_projects)
                                
                                if merged_data.get("error"):
                                    st.error(f"❌ {merged_data['error']}")
                                else:
                                    # Store in session state
                                    st.session_state.cloud_project_data = merged_data
                                    st.session_state.current_project_id = f"cloud:merged-{len(selected_projects)}"
                                    st.session_state.project_data = None
                                    st.session_state.merged_projects = [p.slug for p in selected_projects]
                                    st.session_state.linked_project_info = None  # Clear link state
                                    st.success(f"✅ Merged {len(selected_projects)} projects: {merged_data['node_count']} nodes, {merged_data['edge_count']} edges")
                                    st.rerun()
                
                elif load_mode == "🔗 Link Entities":
                    # Link ActorGraph orgs to OrgGraph funders/grantees
                    st.markdown("**Link ActorGraph organizations to OrgGraph funders/grantees**")
                    st.caption("Match LinkedIn company data to foundation network organizations.")
                    
                    # Separate projects by source type
                    actorgraph_projects = [p for p in cloud_projects if p.source_app == 'actorgraph']
                    orggraph_projects = [p for p in cloud_projects if p.source_app in ('orggraph_us', 'orggraph_ca', 'insightgraph')]
                    
                    if not actorgraph_projects:
                        st.warning("No ActorGraph projects found. Save a company crawl from ActorGraph first.")
                    elif not orggraph_projects:
                        st.warning("No OrgGraph projects found. Save a project from OrgGraph first.")
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🕸️ ActorGraph (LinkedIn)**")
                            actor_options = []
                            for p in actorgraph_projects:
                                label = f"{p.name} ({p.node_count} orgs)"
                                actor_options.append((p.id, label, p))
                            
                            selected_actor_label = st.selectbox(
                                "Select ActorGraph project",
                                [opt[1] for opt in actor_options],
                                key="link_actor_select",
                                label_visibility="collapsed"
                            )
                            
                            selected_actor = None
                            for opt in actor_options:
                                if opt[1] == selected_actor_label:
                                    selected_actor = opt[2]
                                    break
                        
                        with col2:
                            st.markdown("**🏛️ OrgGraph (Funders/Grantees)**")
                            org_options = []
                            for p in orggraph_projects:
                                icon = source_icons.get(p.source_app, '📦')
                                label = f"{icon} {p.name} ({p.node_count} nodes)"
                                org_options.append((p.id, label, p))
                            
                            selected_org_label = st.selectbox(
                                "Select OrgGraph project",
                                [opt[1] for opt in org_options],
                                key="link_org_select",
                                label_visibility="collapsed"
                            )
                            
                            selected_org = None
                            for opt in org_options:
                                if opt[1] == selected_org_label:
                                    selected_org = opt[2]
                                    break
                        
                        st.divider()
                        
                        if selected_actor and selected_org:
                            # Show selection summary
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("ActorGraph Orgs", selected_actor.node_count)
                            with col2:
                                st.metric("OrgGraph Nodes", selected_org.node_count)
                            
                            # Find Matches button
                            if st.button("🔍 Find Matches", type="primary", use_container_width=True):
                                with st.spinner("🔍 Analyzing organizations for matches..."):
                                    # Load both projects
                                    actor_data = load_cloud_project(project_id=selected_actor.id)
                                    org_data = load_cloud_project(project_id=selected_org.id)
                                    
                                    if actor_data.get("error"):
                                        st.error(f"❌ Failed to load ActorGraph: {actor_data['error']}")
                                    elif org_data.get("error"):
                                        st.error(f"❌ Failed to load OrgGraph: {org_data['error']}")
                                    else:
                                        # Find matches
                                        matches = find_entity_matches(
                                            actor_data.get("nodes_df"),
                                            org_data.get("nodes_df")
                                        )
                                        
                                        # Store in session state for review
                                        st.session_state.entity_matches = matches
                                        st.session_state.link_actor_data = actor_data
                                        st.session_state.link_org_data = org_data
                                        st.session_state.link_actor_project = selected_actor
                                        st.session_state.link_org_project = selected_org
                                        st.rerun()
                            
                            # Show match results if available
                            if st.session_state.get("entity_matches"):
                                render_entity_match_review()
    
    # -------------------------------------------------------------------------
    # MANAGE PROJECTS TAB
    # -------------------------------------------------------------------------
    if tab_manage is not None:
        with tab_manage:
            st.caption("View, rename, delete, or share your cloud projects.")
            
            cloud_projects = list_cloud_projects()
            
            if not cloud_projects:
                st.info("No cloud projects to manage. Save a project first.")
            else:
                # Source icons
                source_icons = {
                    'orggraph_us': '🇺🇸',
                    'orggraph_ca': '🇨🇦',
                    'actorgraph': '🕸️',
                    'insightgraph': '🔀',
                }
                
                # Show project count
                st.markdown(f"**{len(cloud_projects)} project(s)**")
                
                # Project list with actions
                for i, p in enumerate(cloud_projects):
                    icon = source_icons.get(p.source_app, '📦')
                    
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            # Project name and details
                            st.markdown(f"{icon} **{p.name}**")
                            st.caption(f"{p.node_count} nodes · {p.edge_count} edges · {p.source_app}")
                        
                        with col2:
                            # Public/Private toggle
                            is_public = p.is_public
                            public_label = "🌐 Public" if is_public else "🔒 Private"
                            if st.button(public_label, key=f"toggle_public_{p.id}", use_container_width=True):
                                success, msg = toggle_project_public(p.id, not is_public)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        with col3:
                            # Rename button
                            if st.button("✏️ Rename", key=f"rename_btn_{p.id}", use_container_width=True):
                                st.session_state[f"renaming_{p.id}"] = True
                                st.rerun()
                        
                        with col4:
                            # Delete button
                            if st.button("🗑️ Delete", key=f"delete_btn_{p.id}", use_container_width=True):
                                st.session_state[f"confirm_delete_{p.id}"] = True
                                st.rerun()
                        
                        # Rename dialog
                        if st.session_state.get(f"renaming_{p.id}"):
                            new_name = st.text_input(
                                "New name",
                                value=p.name,
                                key=f"rename_input_{p.id}"
                            )
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.button("💾 Save", key=f"rename_save_{p.id}", use_container_width=True):
                                    if new_name and new_name != p.name:
                                        success, msg = rename_project(p.id, new_name)
                                        if success:
                                            st.success(msg)
                                            st.session_state[f"renaming_{p.id}"] = False
                                            st.rerun()
                                        else:
                                            st.error(msg)
                                    else:
                                        st.session_state[f"renaming_{p.id}"] = False
                                        st.rerun()
                            with col_cancel:
                                if st.button("Cancel", key=f"rename_cancel_{p.id}", use_container_width=True):
                                    st.session_state[f"renaming_{p.id}"] = False
                                    st.rerun()
                        
                        # Delete confirmation
                        if st.session_state.get(f"confirm_delete_{p.id}"):
                            st.warning(f"⚠️ Delete **{p.name}**? This cannot be undone.")
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("🗑️ Yes, Delete", key=f"delete_confirm_{p.id}", type="primary", use_container_width=True):
                                    success, msg = delete_cloud_project(p.id)
                                    if success:
                                        st.success(msg)
                                        st.session_state[f"confirm_delete_{p.id}"] = False
                                        st.rerun()
                                    else:
                                        st.error(msg)
                            with col_no:
                                if st.button("Cancel", key=f"delete_cancel_{p.id}", use_container_width=True):
                                    st.session_state[f"confirm_delete_{p.id}"] = False
                                    st.rerun()
                        
                        st.divider()
    
    # -------------------------------------------------------------------------
    # Check for loaded cloud project in session state
    # -------------------------------------------------------------------------
    current_proj_id = st.session_state.get("current_project_id") or ""
    if (current_proj_id.startswith("cloud:") and 
        st.session_state.get("cloud_project_data")):
        
        cloud_data = st.session_state.cloud_project_data
        selected_project = {
            "id": cloud_data["project_id"],
            "path": None,
            "has_nodes": cloud_data.get("nodes_df") is not None,
            "has_edges": cloud_data.get("edges_df") is not None,
            "has_grants_detail": cloud_data.get("has_grants_detail", False),
            "has_precomputed": False,
            "is_cloud": True,
        }
        inputs = cloud_data
        is_cloud_project = True
        
        st.success(f"☁️ **Loaded from cloud:** {cloud_data.get('name', cloud_data['project_id'])}")
        
        # Diagnostic: Show manifest status for linked projects
        manifest = cloud_data.get("manifest", {})
        if manifest.get("linked"):
            has_overlap = "overlap_analysis" in manifest and bool(manifest.get("overlap_analysis"))
            if has_overlap:
                overlap_pct = manifest["overlap_analysis"].get("overlap_pct", "?")
                st.caption(f"🔗 Linked project with overlap analysis ({overlap_pct}% overlap)")
            else:
                st.warning("⚠️ This linked project was saved without overlap analysis. Re-create the link to include it in reports.")
    
    # -------------------------------------------------------------------------
    # Stop if no project selected
    # -------------------------------------------------------------------------
    if not selected_project or not inputs:
        st.info("Select a project above to continue.")
        st.stop()
    
    st.divider()
    
    # ==========================================================================
    # Section 2: Run or Load
    # ==========================================================================
    
    selected_id = selected_project["id"]
    
    # Check session state for computed results
    data = None
    
    if not is_cloud_project and selected_project.get("has_precomputed"):
        # Offer choice: load previous or recompute
        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button("📂 Load Previous Results", type="secondary", use_container_width=True)
        with col2:
            compute_btn = st.button("🚀 Recompute Insights", type="primary", use_container_width=True)
        
        if load_btn:
            artifacts = load_precomputed_artifacts(selected_project)
            data = {**inputs, **artifacts}
            st.session_state.project_data = data
            st.session_state.current_project_id = selected_id
            st.rerun()
        
        if compute_btn:
            computed = compute_insights(selected_project, selected_id)
            if computed:
                data = {**inputs, **computed}
                st.session_state.project_data = data
                st.session_state.current_project_id = selected_id
                st.rerun()
    else:
        # No precomputed - must run (or cloud project)
        
        # Show save option for newly created linked projects
        if is_cloud_project and st.session_state.get("show_linked_save_dialog"):
            client = get_project_store_authenticated()
            if client:
                st.markdown("### ☁️ Save Linked Project to Cloud")
                st.caption("Save this linked network before running insights, so you can reload it later.")
                
                cloud_data = st.session_state.get("cloud_project_data", {})
                source_manifest = cloud_data.get("manifest", {})
                source_projects = source_manifest.get("source_projects", ["actorgraph", "orggraph"])
                overlap = source_manifest.get("overlap_analysis", {})
                overlap_pct = overlap.get("overlap_pct", 0)
                
                default_name = f"Linked: {' + '.join(source_projects[:2])}"
                
                col_name, col_btn = st.columns([3, 1])
                
                with col_name:
                    linked_name = st.text_input(
                        "Project name",
                        value=default_name,
                        key="linked_project_name_immediate",
                        label_visibility="collapsed",
                        placeholder="Enter name for linked project"
                    )
                
                with col_btn:
                    if st.button("☁️ Save", type="primary", use_container_width=True, key="save_linked_immediate"):
                        if linked_name:
                            with st.spinner("☁️ Saving linked project..."):
                                success, message, slug = save_linked_to_cloud(
                                    name=linked_name,
                                    data=cloud_data
                                )
                                if success:
                                    st.success(f"✅ {message}")
                                    # Clear the dialog flag but keep linked_project_info
                                    st.session_state.show_linked_save_dialog = False
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.warning("Enter a name for the linked project")
                
                # Show overlap info
                if overlap_pct > 0:
                    st.caption(f"🔗 Overlap: {overlap_pct:.1f}% | This will be included in reports.")
                
                st.divider()
        
        compute_btn = st.button("🚀 Run Insights Engine", type="primary")
        
        if compute_btn:
            # For cloud projects, create a pseudo-project dict
            if is_cloud_project:
                pseudo_project = {
                    "id": selected_id,
                    "path": None,
                    "has_nodes": True,
                    "has_edges": True,
                }
                computed = compute_insights_from_dataframes(
                    inputs.get("nodes_df"),
                    inputs.get("edges_df"),
                    inputs.get("grants_df"),
                    selected_id,
                    manifest=inputs.get("manifest")
                )
                # Preserve cloud: prefix for session state
                session_project_id = f"cloud:{selected_id}"
            else:
                computed = compute_insights(selected_project, selected_id)
                session_project_id = selected_id
            
            if computed:
                data = {**inputs, **computed}
                st.session_state.project_data = data
                st.session_state.current_project_id = session_project_id
                st.rerun()
    
    # Check if we have data from session state
    current_id = st.session_state.get("current_project_id") or ""
    # Strip cloud: prefix for comparison
    current_id_clean = current_id.replace("cloud:", "") if current_id.startswith("cloud:") else current_id
    if st.session_state.get("project_data") and current_id_clean == selected_id:
        data = st.session_state.project_data
        # Make sure grants_df is included from inputs
        if data.get("grants_df") is None and inputs.get("grants_df") is not None:
            data["grants_df"] = inputs["grants_df"]
            data["has_grants_detail"] = inputs.get("has_grants_detail", False)
            data["has_region_data"] = inputs.get("has_region_data", False)
    
    if not data or not data.get("project_summary"):
        st.info("Click a button above to load or compute insights.")
        st.stop()
    
    st.divider()
    
    # ==========================================================================
    # Section 3: Network Results (from project_summary.json)
    # ==========================================================================
    if data.get("project_summary"):
        render_network_results(data["project_summary"])
        st.divider()
    
    # ==========================================================================
    # Section 3.5: Grant Network Results (from grants_detail.csv)
    # ==========================================================================
    if data.get("has_grants_detail") and data.get("grants_df") is not None:
        render_grant_network_results(data["grants_df"], data.get("has_region_data", False))
        st.divider()
    
    # ==========================================================================
    # Section 4: Health Banner + Insight Cards
    # ==========================================================================
    if data.get("insight_cards"):
        render_health_banner(data["insight_cards"])
        st.divider()
        render_insight_cards_section(data["insight_cards"])
        st.divider()
    
    # ==========================================================================
    # Section 5: Grant Purpose Explorer (if grants_detail.csv available)
    # ==========================================================================
    show_purpose_explorer = st.checkbox("🎯 Show Grant Purpose Explorer", value=False)
    if show_purpose_explorer:
        render_grant_purpose_explorer(data.get("grants_df"), data["project_id"])
        st.divider()
    elif not data.get("has_grants_detail"):
        st.caption(f"*Grant Purpose Explorer unavailable (missing `grants_detail.csv` in `demo_data/{data['project_id']}/`)*")
    
    # ==========================================================================
    # Section 6: Node Metrics (optional)
    # ==========================================================================
    if data.get("metrics_df") is not None and not data["metrics_df"].empty:
        render_node_metrics(data["metrics_df"])
        st.divider()
    
    # ==========================================================================
    # Section 7: Downloads
    # ==========================================================================
    render_downloads(data)
    
    st.divider()
    
    # ==========================================================================
    # Section 8: Technical Details
    # ==========================================================================
    render_technical_details(data)


if __name__ == "__main__":
    main()
