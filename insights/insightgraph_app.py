"""
InsightGraph — Streamlit App

Structured insight from complex networks.
Reads exported data from OrgGraph US/CA projects.

VERSION HISTORY:
----------------
UPDATED v0.8.0: Renamed to InsightGraph
- RENAMED: "Insight Engine" → "InsightGraph" for product consistency
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
import zipfile
import sys
import importlib.util

# =============================================================================
# Config
# =============================================================================

APP_VERSION = "0.8.0"  # Renamed to InsightGraph
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_ed8e76c8495d4799a5d7575822009e93~mv2.png"

# Get paths
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
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
    page_icon=C4C_LOGO_URL,
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
    """Render download section (aligned with OrgGraph pattern)."""
    st.subheader("📥 Downloads")
    st.caption("These are the exact artifacts produced for this project export.")
    
    # Check what's available
    has_report = data.get("markdown_report") is not None
    has_cards = data.get("insight_cards") is not None
    has_summary = data.get("project_summary") is not None
    has_metrics = data.get("metrics_df") is not None
    
    def create_zip():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if has_metrics:
                zf.writestr("node_metrics.csv", data["metrics_df"].to_csv(index=False))
            if has_cards:
                zf.writestr("insight_cards.json", json.dumps(data["insight_cards"], indent=2))
            if has_summary:
                zf.writestr("project_summary.json", json.dumps(data["project_summary"], indent=2))
            if has_report:
                zf.writestr("insight_report.md", data["markdown_report"])
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # Primary downloads
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if has_report:
            st.download_button(
                "📝 Download Insight Report (Markdown)",
                data=data["markdown_report"],
                file_name="insight_report.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("No insight report available")
    
    with col2:
        if has_report or has_cards or has_summary or has_metrics:
            st.download_button(
                "📦 Download All (ZIP)",
                data=create_zip(),
                file_name=f"{data['project_id']}_insights.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    # Individual files
    st.caption("Individual data files:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if has_metrics:
            st.download_button(
                "📄 node_metrics.csv",
                data=data["metrics_df"].to_csv(index=False),
                file_name="node_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if has_cards:
            st.download_button(
                "📄 insight_cards.json",
                data=json.dumps(data["insight_cards"], indent=2),
                file_name="insight_cards.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        if has_summary:
            st.download_button(
                "📄 project_summary.json",
                data=json.dumps(data["project_summary"], indent=2),
                file_name="project_summary.json",
                mime="application/json",
                use_container_width=True
            )


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
    """Compute insights from nodes/edges when pre-exports don't exist."""
    run_module = load_run_module()
    if not run_module:
        st.error("Cannot compute insights: run.py not found")
        return None
    
    path = project["path"]
    nodes_path = path / "nodes.csv"
    edges_path = path / "edges.csv"
    
    with st.spinner("Computing insights from network data..."):
        try:
            # Load and validate
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
            
            # Generate insights
            insight_cards = run_module.generate_insight_cards(
                nodes_df, edges_df, metrics_df,
                interlock_graph, flow_stats, overlap_df,
                project_id=project_id
            )
            
            # Project summary
            project_summary = run_module.generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
            
            # Generate markdown report
            markdown_report = run_module.generate_markdown_report(insight_cards, project_summary, project_id)
            
            return {
                "project_id": project_id,
                "nodes_df": nodes_df,
                "edges_df": edges_df,
                "metrics_df": metrics_df,
                "insight_cards": insight_cards,
                "project_summary": project_summary,
                "markdown_report": markdown_report,
            }
            
        except Exception as e:
            st.error(f"Error computing insights: {e}")
            return None


# =============================================================================
# Main App
# =============================================================================

def main():
    init_session_state()
    
    # ==========================================================================
    # Header
    # ==========================================================================
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("InsightGraph")
    
    st.markdown("Structured insight from complex networks.")
    st.caption(f"v{APP_VERSION}")
    
    st.divider()
    
    # ==========================================================================
    # Section 1: Project Selection
    # ==========================================================================
    st.subheader("🗂️ Project")
    st.caption("Select a project folder containing nodes.csv and edges.csv (exported from OrgGraph).")
    
    projects = get_projects()
    
    if not projects:
        st.warning(f"""
        **No projects found.**
        
        Expected location: `{DEMO_DATA_DIR}`
        
        Each project folder must contain:
        - `nodes.csv` (required)
        - `edges.csv` (required)
        - `grants_detail.csv` (optional, enables Purpose Explorer)
        """)
        st.stop()
    
    # Build project selector
    project_options = [p["id"] for p in projects]
    
    selected_id = st.selectbox(
        "Select project",
        project_options,
        label_visibility="collapsed"
    )
    
    selected_project = next((p for p in projects if p["id"] == selected_id), None)
    
    if not selected_project:
        st.error("Project not found")
        st.stop()
    
    # Show project input status
    st.markdown(f"**📂 {selected_id}**")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown("✅ nodes.csv" if selected_project["has_nodes"] else "❌ nodes.csv")
    col2.markdown("✅ edges.csv" if selected_project["has_edges"] else "❌ edges.csv")
    col3.markdown("✅ grants_detail.csv" if selected_project["has_grants_detail"] else "⚪ grants_detail.csv (optional)")
    
    if selected_project["has_precomputed"]:
        st.caption("💾 *Pre-computed results available from previous run*")
    
    st.divider()
    
    # ==========================================================================
    # Section 2: Run or Load
    # ==========================================================================
    
    # Load input data (nodes, edges, grants_detail)
    inputs = load_project_inputs(selected_project)
    
    # Check session state for computed results
    data = None
    
    if selected_project["has_precomputed"]:
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
        # No precomputed - must run
        compute_btn = st.button("🚀 Run Insights Engine", type="primary")
        
        if compute_btn:
            computed = compute_insights(selected_project, selected_id)
            if computed:
                data = {**inputs, **computed}
                st.session_state.project_data = data
                st.session_state.current_project_id = selected_id
                st.rerun()
    
    # Check if we have data from session state
    if st.session_state.get("project_data") and st.session_state.get("current_project_id") == selected_id:
        data = st.session_state.project_data
        # Make sure grants_df is included from inputs
        if data.get("grants_df") is None and inputs.get("grants_df") is not None:
            data["grants_df"] = inputs["grants_df"]
            data["has_grants_detail"] = inputs["has_grants_detail"]
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
