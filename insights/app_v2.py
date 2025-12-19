"""
Insight Engine v2 — Streamlit App

A systems briefing generator, not a dashboard.
Generates narrative reports with expandable evidence.

The UI is an interface to the briefing, not the product itself.

VERSION HISTORY:
----------------
v2.0.0 (2025-12-19): Initial v2 release
- New report structure based on insight-engine-spec.md
- Signal/Interpretation/Why it matters pattern for all sections
- Reading contract banner (5-min / 20-min / explore)
- System Summary with headline, positives, gaps
- Opportunity/Risk lens labels for Brokers/Bridges
- Strategic Recommendations (always exactly 4)
- Project Outputs organized by audience (Executive/Analyst/Developer)

v2.0.1 (2025-12-19): Fixed recommendations bug
- Each default recommendation now has unique trigger
- Prevents IndexError when fewer than 4 signal-triggered recs

v2.0.2 (2025-12-19): Fixed CSV schema handling
- compute_basic_metrics() now handles actual OrgGraph schema
- Supports node_type = ORG/PERSON (not funder/grantee)
- Supports edge_type = GRANT/BOARD_MEMBERSHIP
- Supports from_id/to_id column names
- Uses grants_detail.csv as primary funding source when available
- Computes funder pairs, overlap percentages, governance metrics
- Proper health score calculation

v2.0.3 (2025-12-19): Updated logo
- Changed C4C_LOGO_URL to new icon

Based on: insight-engine-spec.md (December 2025)

INTEGRATION NOTE:
This v2 app can load data from:
1. v1 artifacts (project_summary.json, insight_cards.json) - adapts to v2 format
2. v2 native format (network_metrics.json) - if available
3. Raw CSVs (nodes.csv, edges.csv, grants_detail.csv) - computes metrics directly
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
import zipfile

# =============================================================================
# Import Insight Engine modules
# =============================================================================
import config as cfg
from narratives import (
    SignalInterpretation,
    interpret_funding_concentration,
    interpret_funder_overlap,
    interpret_portfolio_twins,
    interpret_governance,
    interpret_hidden_brokers,
    interpret_single_point_bridges,
    interpret_network_health,
    generate_system_summary,
    generate_recommendations,
)
from report_generator import ReportData, generate_report

# =============================================================================
# App Configuration
# =============================================================================

APP_VERSION = "2.0.3"
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_ddef70debd0b46c799a9d3d8c73a42da~mv2.png"

# Paths
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"

st.set_page_config(
    page_title="Insight Engine v2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Session State Initialization
# =============================================================================

if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "generated_report" not in st.session_state:
    st.session_state.generated_report = None
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None


# =============================================================================
# Project Discovery (matches v1)
# =============================================================================

def get_projects() -> list[dict]:
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
            
            # v1 pre-computed artifacts
            has_summary = (item / "project_summary.json").exists()
            has_cards = (item / "insight_cards.json").exists()
            has_report = (item / "insight_report.md").exists()
            has_metrics_csv = (item / "node_metrics.csv").exists()
            
            # v2 native format
            has_network_metrics = (item / "network_metrics.json").exists()
            
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
                    # v1 pre-computed outputs
                    "has_summary": has_summary,
                    "has_cards": has_cards,
                    "has_report": has_report,
                    "has_metrics_csv": has_metrics_csv,
                    "has_v1_artifacts": has_summary and has_cards,
                    # v2 native format
                    "has_network_metrics": has_network_metrics,
                    # Overall status
                    "is_complete": has_summary and has_cards,
                })
    
    # Sort alphabetically
    projects.sort(key=lambda x: x["id"].lower())
    return projects


# =============================================================================
# Data Loading & Adaptation
# =============================================================================

def load_v1_artifacts(project_path: Path) -> dict:
    """Load v1 artifacts (project_summary.json, insight_cards.json)."""
    artifacts = {
        "project_summary": None,
        "insight_cards": None,
        "markdown_report": None,
    }
    
    summary_path = project_path / "project_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            artifacts["project_summary"] = json.load(f)
    
    cards_path = project_path / "insight_cards.json"
    if cards_path.exists():
        with open(cards_path, 'r') as f:
            artifacts["insight_cards"] = json.load(f)
    
    report_path = project_path / "insight_report.md"
    if report_path.exists():
        with open(report_path, 'r') as f:
            artifacts["markdown_report"] = f.read()
    
    return artifacts


def adapt_v1_to_v2_metrics(project_summary: dict, insight_cards: dict) -> dict:
    """
    Adapt v1 data structures to v2 metrics format.
    
    v1 project_summary structure:
    {
        "node_counts": {"organizations": N, "people": N},
        "edge_counts": {"grants": N, "board_memberships": N},
        "funding": {"total_amount": N, "funder_count": N, "grantee_count": N, "top5_share": 0.xx},
        "governance": {"multi_board_people": N}
    }
    
    v1 insight_cards structure:
    {
        "health": {"score": N, "label": "...", "positive": [...], "risk": [...]},
        "cards": [...]
    }
    """
    metrics = {
        # Default values
        "node_count": 0,
        "edge_count": 0,
        "total_funding": 0,
        "funder_count": 0,
        "grantee_count": 0,
        "multi_funder_pct": 0,
        "multi_funder_count": 0,
        "connectivity_pct": 50,  # Not in v1, use default
        "top5_share_pct": 0,
        "network_health_score": 50,
        "governance_data_available": False,
        "governance_coverage_pct": 0,
        "shared_board_count": 0,
        "pct_with_interlocks": 0,
        "broker_count": 0,
        "bridge_count": 0,
        "top_shared_grantees": [],
        "top_funder_pairs": [],
        "top_brokers": [],
        "top_bridges": [],
        "max_funder_overlap_pct": 0,
    }
    
    # Extract from project_summary
    if project_summary:
        node_counts = project_summary.get("node_counts", {})
        edge_counts = project_summary.get("edge_counts", {})
        funding = project_summary.get("funding", {})
        governance = project_summary.get("governance", {})
        
        metrics["node_count"] = node_counts.get("organizations", 0) + node_counts.get("people", 0)
        metrics["edge_count"] = edge_counts.get("grants", 0) + edge_counts.get("board_memberships", 0)
        metrics["total_funding"] = funding.get("total_amount", 0)
        metrics["funder_count"] = funding.get("funder_count", 0)
        metrics["grantee_count"] = funding.get("grantee_count", 0)
        
        # top5_share is stored as decimal in v1 (e.g., 0.62), convert to percentage
        top5_share = funding.get("top5_share", 0)
        metrics["top5_share_pct"] = top5_share * 100 if top5_share < 1 else top5_share
        
        # Governance
        multi_board = governance.get("multi_board_people", 0)
        metrics["shared_board_count"] = multi_board
        metrics["governance_data_available"] = multi_board > 0
        if multi_board > 0:
            metrics["governance_coverage_pct"] = 0.5  # Assume moderate coverage if we have data
            metrics["pct_with_interlocks"] = min(multi_board * 5, 50)  # Rough estimate
    
    # Extract from insight_cards
    if insight_cards:
        health = insight_cards.get("health", {})
        metrics["network_health_score"] = health.get("score", 50)
        
        # Try to extract metrics from cards
        cards = insight_cards.get("cards", [])
        for card in cards:
            title = card.get("title", "").lower()
            ranked_rows = card.get("ranked_rows", [])
            
            # Multi-funder grantees card
            if "anchor" in title or "multi-funder" in title or "shared" in title:
                metrics["multi_funder_count"] = len(ranked_rows)
                if metrics["grantee_count"] > 0:
                    metrics["multi_funder_pct"] = (len(ranked_rows) / metrics["grantee_count"]) * 100
                # Extract top shared grantees
                for row in ranked_rows[:cfg.TOP_N_SHARED_GRANTEES]:
                    grantee = {
                        "grantee_name": row.get("grantee") or row.get("org") or row.get("node", "Unknown"),
                        "funder_count": row.get("funders", 0),
                        "total_received": row.get("amount", 0),
                        "top_funders": [],
                    }
                    metrics["top_shared_grantees"].append(grantee)
            
            # Portfolio overlap / twins card
            if "portfolio" in title or "twin" in title or "overlap" in title:
                for row in ranked_rows[:cfg.TOP_N_FUNDER_PAIRS]:
                    pair = {
                        "funder_a": row.get("pair", "").split(" & ")[0] if " & " in row.get("pair", "") else row.get("funder_a", "Unknown"),
                        "funder_b": row.get("pair", "").split(" & ")[1] if " & " in row.get("pair", "") else row.get("funder_b", "Unknown"),
                        "shared_grantee_count": row.get("shared", 0),
                        "overlap_pct": row.get("jaccard", 0) * 100 if row.get("jaccard", 0) < 1 else row.get("jaccard", 0),
                    }
                    metrics["top_funder_pairs"].append(pair)
                    if pair["overlap_pct"] > metrics["max_funder_overlap_pct"]:
                        metrics["max_funder_overlap_pct"] = pair["overlap_pct"]
            
            # Broker card
            if "broker" in title:
                metrics["broker_count"] = len(ranked_rows)
                for row in ranked_rows[:cfg.TOP_N_BROKERS]:
                    broker = {
                        "org_name": row.get("org") or row.get("grantee") or row.get("node", "Unknown"),
                        "betweenness": row.get("betweenness", 0),
                        "visibility_percentile": 30,  # Not in v1, use default
                        "broker_reason": row.get("narrative", "High structural importance"),
                    }
                    metrics["top_brokers"].append(broker)
            
            # Bridge / connector card
            if "bridge" in title or "connector" in title or "critical" in title:
                metrics["bridge_count"] = len(ranked_rows)
                for row in ranked_rows[:cfg.TOP_N_BRIDGES]:
                    bridge = {
                        "node_name": row.get("org") or row.get("person") or row.get("node", "Unknown"),
                        "node_type": "organization" if row.get("org") else "person",
                        "impact_if_removed": row.get("narrative", "Network fragmentation"),
                    }
                    metrics["top_bridges"].append(bridge)
    
    return metrics


def load_metrics(project: dict) -> dict:
    """
    Load metrics from available sources:
    1. v2 native format (network_metrics.json)
    2. v1 artifacts (project_summary.json + insight_cards.json) - adapted
    3. Raw CSVs (nodes.csv, edges.csv) - basic computation only
    """
    project_path = project["path"]
    
    # Try v2 native format first
    if project.get("has_network_metrics"):
        with open(project_path / "network_metrics.json", 'r') as f:
            return json.load(f)
    
    # Try v1 artifacts
    if project.get("has_v1_artifacts"):
        v1_data = load_v1_artifacts(project_path)
        return adapt_v1_to_v2_metrics(
            v1_data.get("project_summary"),
            v1_data.get("insight_cards")
        )
    
    # Fallback: compute basic metrics from CSVs
    return compute_basic_metrics(project_path)


def compute_basic_metrics(project_path: Path) -> dict:
    """
    Compute basic metrics from nodes.csv, edges.csv, and optionally grants_detail.csv.
    
    Expected schemas:
    - nodes.csv: node_id, node_type (ORG/PERSON), label, ...
    - edges.csv: edge_id, from_id, to_id, edge_type (GRANT/BOARD_MEMBERSHIP), amount, ...
    - grants_detail.csv: foundation_name, grantee_name, grant_amount, ...
    """
    nodes_df = pd.read_csv(project_path / "nodes.csv")
    edges_df = pd.read_csv(project_path / "edges.csv")
    
    # Check for grants_detail.csv (better source for funding data)
    grants_path = project_path / "grants_detail.csv"
    grants_df = None
    if grants_path.exists():
        grants_df = pd.read_csv(grants_path)
    
    # Basic counts
    node_count = len(nodes_df)
    edge_count = len(edges_df)
    
    # Node type counts
    # Schema uses: node_type = "ORG" or "PERSON"
    org_count = 0
    person_count = 0
    
    if "node_type" in nodes_df.columns:
        type_values = nodes_df["node_type"].str.upper()
        org_count = (type_values == "ORG").sum()
        person_count = (type_values == "PERSON").sum()
    
    # Edge type counts
    # Schema uses: edge_type = "GRANT" or "BOARD_MEMBERSHIP"
    grant_edge_count = 0
    board_edge_count = 0
    
    if "edge_type" in edges_df.columns:
        edge_types = edges_df["edge_type"].str.upper()
        grant_edge_count = (edge_types == "GRANT").sum()
        board_edge_count = (edge_types == "BOARD_MEMBERSHIP").sum()
    
    # Source/target columns - handle various naming conventions
    source_col = None
    target_col = None
    for s, t in [("from_id", "to_id"), ("source_id", "target_id"), ("source", "target")]:
        if s in edges_df.columns and t in edges_df.columns:
            source_col = s
            target_col = t
            break
    
    # Amount column
    amount_col = None
    for col in ["amount", "grant_amount", "weight"]:
        if col in edges_df.columns:
            amount_col = col
            break
    
    # ==========================================================================
    # If grants_detail.csv exists, use it for funding metrics (more reliable)
    # ==========================================================================
    if grants_df is not None and not grants_df.empty:
        # Funder and grantee counts from grants
        funder_col = "foundation_name" if "foundation_name" in grants_df.columns else "funder_name"
        grantee_col = "grantee_name"
        amount_col_grants = "grant_amount"
        
        if funder_col in grants_df.columns:
            funder_count = grants_df[funder_col].nunique()
        else:
            funder_count = 0
        
        if grantee_col in grants_df.columns:
            grantee_count = grants_df[grantee_col].nunique()
        else:
            grantee_count = 0
        
        # Total funding
        if amount_col_grants in grants_df.columns:
            total_funding = pd.to_numeric(grants_df[amount_col_grants], errors="coerce").fillna(0).sum()
        else:
            total_funding = 0
        
        # Multi-funder grantees
        if funder_col in grants_df.columns and grantee_col in grants_df.columns:
            grantee_funder_counts = grants_df.groupby(grantee_col)[funder_col].nunique()
            multi_funder_grantees = grantee_funder_counts[grantee_funder_counts > 1]
            multi_funder_count = len(multi_funder_grantees)
            multi_funder_pct = (multi_funder_count / grantee_count * 100) if grantee_count > 0 else 0
            
            # Top shared grantees
            top_shared = multi_funder_grantees.sort_values(ascending=False).head(cfg.TOP_N_SHARED_GRANTEES)
            top_shared_grantees = []
            for grantee_name, funder_ct in top_shared.items():
                grantee_grants = grants_df[grants_df[grantee_col] == grantee_name]
                total_received = grantee_grants[amount_col_grants].sum() if amount_col_grants in grantee_grants.columns else 0
                top_funders = grantee_grants[funder_col].unique().tolist()[:3]
                top_shared_grantees.append({
                    "grantee_name": grantee_name,
                    "funder_count": int(funder_ct),
                    "total_received": float(total_received),
                    "top_funders": top_funders,
                })
        else:
            multi_funder_count = 0
            multi_funder_pct = 0
            top_shared_grantees = []
        
        # Top 5 funder concentration
        if funder_col in grants_df.columns and amount_col_grants in grants_df.columns:
            funder_totals = grants_df.groupby(funder_col)[amount_col_grants].sum().sort_values(ascending=False)
            top5_funding = funder_totals.head(5).sum()
            top5_share_pct = (top5_funding / total_funding * 100) if total_funding > 0 else 0
        else:
            top5_share_pct = 0
        
        # Funder pair overlap (Portfolio Twins)
        top_funder_pairs = []
        max_funder_overlap_pct = 0
        if funder_col in grants_df.columns and grantee_col in grants_df.columns:
            # Build funder -> set of grantees mapping
            funder_grantees = grants_df.groupby(funder_col)[grantee_col].apply(set).to_dict()
            funders = list(funder_grantees.keys())
            
            pairs = []
            for i, f1 in enumerate(funders):
                for f2 in funders[i+1:]:
                    shared = funder_grantees[f1] & funder_grantees[f2]
                    if len(shared) > 0:
                        union = funder_grantees[f1] | funder_grantees[f2]
                        jaccard = len(shared) / len(union) * 100 if len(union) > 0 else 0
                        pairs.append({
                            "funder_a": f1,
                            "funder_b": f2,
                            "shared_grantee_count": len(shared),
                            "overlap_pct": round(jaccard, 1),
                        })
            
            # Sort by shared count and take top N
            pairs.sort(key=lambda x: x["shared_grantee_count"], reverse=True)
            top_funder_pairs = pairs[:cfg.TOP_N_FUNDER_PAIRS]
            if top_funder_pairs:
                max_funder_overlap_pct = max(p["overlap_pct"] for p in top_funder_pairs)
    
    else:
        # ==========================================================================
        # Fallback: compute from edges.csv only (less reliable)
        # ==========================================================================
        # Filter to grant edges only for funding metrics
        if "edge_type" in edges_df.columns:
            grant_edges = edges_df[edges_df["edge_type"].str.upper() == "GRANT"]
        else:
            grant_edges = edges_df
        
        # Estimate funder/grantee counts from edges
        if source_col and target_col:
            funder_count = grant_edges[source_col].nunique() if not grant_edges.empty else 0
            grantee_count = grant_edges[target_col].nunique() if not grant_edges.empty else 0
        else:
            funder_count = 0
            grantee_count = 0
        
        # Total funding from edge amounts
        if amount_col and not grant_edges.empty:
            total_funding = pd.to_numeric(grant_edges[amount_col], errors="coerce").fillna(0).sum()
        else:
            total_funding = 0
        
        # Multi-funder calculation
        multi_funder_count = 0
        multi_funder_pct = 0
        top_shared_grantees = []
        
        if source_col and target_col and not grant_edges.empty:
            grantee_funder_counts = grant_edges.groupby(target_col)[source_col].nunique()
            multi_funder_count = (grantee_funder_counts > 1).sum()
            multi_funder_pct = (multi_funder_count / len(grantee_funder_counts) * 100) if len(grantee_funder_counts) > 0 else 0
        
        # Top 5 concentration
        top5_share_pct = 0
        if source_col and amount_col and not grant_edges.empty:
            funder_totals = grant_edges.groupby(source_col)[amount_col].sum().sort_values(ascending=False)
            top5_funding = pd.to_numeric(funder_totals.head(5), errors="coerce").sum()
            top5_share_pct = (top5_funding / total_funding * 100) if total_funding > 0 else 0
        
        top_funder_pairs = []
        max_funder_overlap_pct = 0
    
    # ==========================================================================
    # Governance metrics (from board edges)
    # ==========================================================================
    governance_data_available = board_edge_count > 0
    shared_board_count = 0
    pct_with_interlocks = 0
    
    if governance_data_available and "edge_type" in edges_df.columns:
        board_edges = edges_df[edges_df["edge_type"].str.upper() == "BOARD_MEMBERSHIP"]
        if not board_edges.empty and source_col and target_col:
            # Count people serving on multiple boards
            person_board_counts = board_edges.groupby(source_col)[target_col].nunique()
            shared_board_count = (person_board_counts > 1).sum()
            
            # Estimate interlock percentage
            if org_count > 0:
                orgs_with_shared_people = board_edges[board_edges[source_col].isin(
                    person_board_counts[person_board_counts > 1].index
                )][target_col].nunique()
                pct_with_interlocks = (orgs_with_shared_people / org_count * 100) if org_count > 0 else 0
    
    # ==========================================================================
    # Compute network health score
    # ==========================================================================
    # Simple formula: weighted average of positive indicators
    health_components = []
    
    # Multi-funder presence (max 30 points)
    health_components.append(min(multi_funder_pct, 30))
    
    # Funding diversification (max 25 points) - inverse of concentration
    concentration_penalty = max(0, top5_share_pct - 50) / 2  # Penalty if > 50%
    health_components.append(max(0, 25 - concentration_penalty))
    
    # Governance connectivity (max 20 points)
    if governance_data_available and shared_board_count > 0:
        health_components.append(min(shared_board_count * 2, 20))
    else:
        health_components.append(0)
    
    # Base connectivity (25 points if we have edges)
    if edge_count > 0:
        health_components.append(25)
    else:
        health_components.append(0)
    
    network_health_score = sum(health_components)
    
    # Estimate connectivity percentage
    connectivity_pct = min(100, (edge_count / max(node_count, 1)) * 10) if node_count > 0 else 0
    
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "org_count": org_count,
        "person_count": person_count,
        "grant_edge_count": grant_edge_count,
        "board_edge_count": board_edge_count,
        "funder_count": funder_count,
        "grantee_count": grantee_count,
        "total_funding": total_funding,
        "multi_funder_count": multi_funder_count,
        "multi_funder_pct": multi_funder_pct,
        "top5_share_pct": top5_share_pct,
        "connectivity_pct": connectivity_pct,
        "network_health_score": network_health_score,
        "governance_data_available": governance_data_available,
        "shared_board_count": shared_board_count,
        "pct_with_interlocks": pct_with_interlocks,
        "governance_coverage_pct": 0.5 if governance_data_available else 0,
        "broker_count": 0,  # Requires graph analysis
        "bridge_count": 0,  # Requires graph analysis
        "top_shared_grantees": top_shared_grantees,
        "top_funder_pairs": top_funder_pairs,
        "top_brokers": [],  # Requires graph analysis
        "top_bridges": [],  # Requires graph analysis
        "max_funder_overlap_pct": max_funder_overlap_pct,
    }


# =============================================================================
# UI Helper Functions
# =============================================================================

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.0f}K"
    else:
        return f"${amount:,.0f}"


def confidence_badge(confidence: str) -> str:
    """Return emoji badge for confidence level."""
    badges = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🟠",
        "unavailable": "⚪"
    }
    return badges.get(confidence, "⚪")


def render_signal_section(
    title: str,
    signal: SignalInterpretation,
    lens: str = None,
    show_evidence: bool = True
):
    """Render a signal section with consistent structure."""
    
    # Section header with optional lens
    if lens:
        st.markdown(f"### {title}")
        if lens == "opportunity":
            st.info("**Lens: Opportunity** — This signal suggests untapped potential.")
        elif lens == "risk":
            st.warning("**Lens: Risk** — This signal identifies structural vulnerability.")
    else:
        st.markdown(f"### {title}")
    
    # Confidence indicator
    st.caption(f"{confidence_badge(signal.confidence)} Confidence: {signal.confidence}")
    
    # Signal → Interpretation → Why it matters
    st.markdown(f"**Signal**  \n{signal.signal}")
    st.markdown(f"**Interpretation**  \n{signal.interpretation}")
    st.markdown(f"**Why it matters**  \n{signal.why_it_matters}")
    
    # Evidence (collapsible if there's data)
    if show_evidence and signal.evidence:
        with st.expander("📊 View Evidence", expanded=False):
            render_evidence(signal.evidence)


def render_evidence(evidence: dict):
    """Render evidence data as appropriate UI elements."""
    
    # Handle different evidence types
    if "top_shared_grantees" in evidence and evidence["top_shared_grantees"]:
        st.markdown("**Top Shared Grantees**")
        grantees = evidence["top_shared_grantees"]
        df = pd.DataFrame(grantees)
        if not df.empty:
            display_cols = ["grantee_name", "funder_count", "total_received"]
            display_cols = [c for c in display_cols if c in df.columns]
            if display_cols:
                show_df = df[display_cols].copy()
                if "total_received" in show_df.columns:
                    show_df["total_received"] = show_df["total_received"].apply(
                        lambda x: format_currency(x) if pd.notna(x) and x > 0 else "—"
                    )
                st.dataframe(show_df, use_container_width=True, hide_index=True)
    
    if "top_funder_pairs" in evidence and evidence["top_funder_pairs"]:
        st.markdown("**Top Funder Pairs**")
        pairs = evidence["top_funder_pairs"]
        df = pd.DataFrame(pairs)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    if "top_brokers" in evidence and evidence["top_brokers"]:
        st.markdown("**Hidden Brokers**")
        brokers = evidence["top_brokers"]
        df = pd.DataFrame(brokers)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    if "top_bridges" in evidence and evidence["top_bridges"]:
        st.markdown("**Single-Point Bridges**")
        bridges = evidence["top_bridges"]
        df = pd.DataFrame(bridges)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Render simple key-value metrics
    simple_keys = ["total_funding", "funder_count", "grantee_count", "top5_share_pct", 
                   "multi_funder_count", "multi_funder_pct", "broker_count", "bridge_count"]
    
    metrics_to_show = {k: v for k, v in evidence.items() 
                       if k in simple_keys and v is not None and not (isinstance(v, (int, float)) and v == 0)}
    
    if metrics_to_show:
        cols = st.columns(min(4, len(metrics_to_show)))
        for i, (key, value) in enumerate(metrics_to_show.items()):
            col = cols[i % len(cols)]
            label = key.replace("_", " ").title()
            if "funding" in key.lower() or "received" in key.lower():
                col.metric(label, format_currency(value))
            elif "pct" in key.lower():
                col.metric(label, f"{value:.1f}%")
            else:
                col.metric(label, f"{value:,}" if isinstance(value, (int, float)) else value)


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("Insight Engine")
    
    st.markdown("Systems Briefing Generator")
    st.caption(f"v{APP_VERSION}")
    
    # =========================================================================
    # Reading Contract Banner
    # =========================================================================
    with st.expander("📖 How to Read This Report", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⏱️ 5 minutes (Skim)**")
            st.markdown("""
            - System Summary
            - Network Health
            - Strategic Recommendations
            """)
        
        with col2:
            st.markdown("**📚 20 minutes (Read)**")
            st.markdown("""
            - All signal sections
            - Selected evidence
            """)
        
        with col3:
            st.markdown("**🔍 Optional (Explore)**")
            st.markdown("""
            - Grant themes
            - Data tables
            - Downloads
            """)
        
        st.caption("*If Governance Connectivity shows 'data unavailable,' you may skip that section.*")
    
    st.divider()
    
    # =========================================================================
    # Project Selection
    # =========================================================================
    st.markdown("### 🗂️ Project")
    st.caption("Select a project folder containing nodes.csv and edges.csv (exported from OrgGraph).")
    
    projects = get_projects()
    
    if not projects:
        st.warning(f"""
        **No projects found.**
        
        Expected location: `{DEMO_DATA_DIR}`
        
        Each project folder should contain:
        - `nodes.csv` (required)
        - `edges.csv` (required)
        - `project_summary.json` + `insight_cards.json` (recommended, from v1)
        """)
        st.stop()
    
    # Project selector
    project_options = {p["id"]: p for p in projects}
    selected_id = st.selectbox(
        "Select project",
        options=list(project_options.keys()),
        format_func=lambda x: f"{'✅' if project_options[x]['is_complete'] else '⚠️'} {x}",
        label_visibility="collapsed"
    )
    
    project = project_options[selected_id]
    
    # Status indicators
    st.markdown(f"**📂 {selected_id}**")
    col1, col2, col3 = st.columns(3)
    col1.markdown("✅ nodes.csv" if project["has_nodes"] else "❌ nodes.csv")
    col2.markdown("✅ edges.csv" if project["has_edges"] else "❌ edges.csv")
    
    if project["has_v1_artifacts"]:
        col3.markdown("✅ v1 artifacts (will adapt)")
    elif project["has_network_metrics"]:
        col3.markdown("✅ v2 metrics")
    else:
        col3.markdown("⚠️ Basic mode (limited data)")
    
    # Generate button
    if st.button("🚀 Generate Briefing", type="primary", use_container_width=True):
        with st.spinner("Generating systems briefing..."):
            # Load metrics (adapts from v1 if needed)
            metrics = load_metrics(project)
            
            # Add metadata
            metrics["data_sources"] = "OrgGraph export"
            metrics["date_range"] = "See project metadata"
            
            # Generate report data
            report_data = ReportData(metrics, project_name=selected_id)
            report_markdown = generate_report(report_data)
            
            # Store in session state
            st.session_state.report_data = report_data
            st.session_state.generated_report = report_markdown
            st.session_state.current_project_id = selected_id
            st.session_state.metrics = metrics
        
        st.rerun()
    
    # =========================================================================
    # Report Display
    # =========================================================================
    
    if st.session_state.report_data is None or st.session_state.current_project_id != selected_id:
        st.info("👆 Select a project and click **Generate Briefing** to create your systems briefing.")
        st.stop()
    
    report_data = st.session_state.report_data
    signals = report_data.signals
    summary = report_data.system_summary
    recommendations = report_data.recommendations
    metrics = st.session_state.metrics
    
    st.divider()
    
    # =========================================================================
    # 1. System Summary
    # =========================================================================
    st.markdown("## System Summary")
    
    # Headline
    st.markdown(f"### {summary.headline}")
    
    # Summary paragraph
    st.markdown(summary.summary_paragraph)
    
    # What's working / What's missing
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ What's working**")
        for positive in summary.positives:
            st.markdown(f"- {positive}")
    
    with col2:
        st.markdown("**⚠️ What's missing**")
        for gap in summary.gaps:
            st.markdown(f"- {gap}")
    
    # Why it matters
    st.markdown(f"**Why this matters**  \n{summary.why_it_matters}")
    
    st.divider()
    
    # =========================================================================
    # 2. Network Health
    # =========================================================================
    st.markdown("## Network Health")
    
    health = signals["network_health"]
    health_score = health.evidence.get("health_score", 50)
    health_label = health.evidence.get("health_label", "Moderate")
    
    # Color based on score
    if health_score >= 70:
        health_color = "🟢"
    elif health_score >= 40:
        health_color = "🟡"
    else:
        health_color = "🔴"
    
    st.markdown(f"### {health_color} Overall Score: **{health_score:.0f}/100** — *{health_label}*")
    
    # Key signals as metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Multi-Funder %", f"{metrics.get('multi_funder_pct', 0):.0f}%")
    col2.metric("Connectivity", f"{metrics.get('connectivity_pct', 0):.0f}%")
    col3.metric("Top 5 Share", f"{metrics.get('top5_share_pct', 0):.0f}%")
    col4.metric("Governance", "Available" if metrics.get("governance_data_available") else "Unavailable")
    
    st.markdown(f"**Why this matters**  \n{health.why_it_matters}")
    
    st.divider()
    
    # =========================================================================
    # 3. Funding Concentration
    # =========================================================================
    render_signal_section(
        "Funding Concentration",
        signals["funding_concentration"]
    )
    
    st.divider()
    
    # =========================================================================
    # 4. Funder Overlap Clusters
    # =========================================================================
    render_signal_section(
        "Funder Overlap Clusters",
        signals["funder_overlap"]
    )
    
    st.divider()
    
    # =========================================================================
    # 5. Portfolio Twins
    # =========================================================================
    render_signal_section(
        "Portfolio Twins",
        signals["portfolio_twins"]
    )
    
    st.divider()
    
    # =========================================================================
    # 6. Governance Connectivity
    # =========================================================================
    render_signal_section(
        "Governance Connectivity",
        signals["governance"]
    )
    
    st.divider()
    
    # =========================================================================
    # 7. Hidden Brokers (Opportunity)
    # =========================================================================
    render_signal_section(
        "Hidden Brokers",
        signals["hidden_brokers"],
        lens="opportunity"
    )
    
    st.divider()
    
    # =========================================================================
    # 8. Single-Point Bridges (Risk)
    # =========================================================================
    render_signal_section(
        "Single-Point Bridges",
        signals["single_point_bridges"],
        lens="risk"
    )
    
    st.divider()
    
    # =========================================================================
    # 9. Strategic Recommendations
    # =========================================================================
    st.markdown("## Strategic Recommendations")
    
    st.markdown(cfg.STRATEGIC_INTRO)
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f"**{i}. {rec.title}**")
            st.markdown(rec.text)
    
    st.divider()
    
    # =========================================================================
    # 10. Appendix / Optional Deep Dive
    # =========================================================================
    st.markdown("## Appendix")
    
    # Method Notes
    with st.expander("📋 Method Notes", expanded=False):
        st.markdown(f"""
**Network Health Score (v1):** Weighted composite of coordination (multi-funder %), 
connectivity, concentration (inverse), and governance connectivity proxies.

**Hidden Broker Definition:** Nodes in top {cfg.BROKER_BETWEENNESS_PERCENTILE}th percentile 
of betweenness centrality AND bottom {cfg.BROKER_VISIBILITY_PERCENTILE}th percentile of degree, 
computed within each node type (funder, grantee, person) to avoid cross-type artifacts.

**Single-Point Bridge Definition:** Articulation points — nodes whose removal disconnects 
the network graph.
        """)
    
    # Data Notes
    with st.expander("📊 Data Notes", expanded=False):
        st.markdown(f"""
- **Data Sources:** {metrics.get('data_sources', 'OrgGraph export')}
- **Date Range:** {metrics.get('date_range', 'Not specified')}
- **Processing Timestamp:** {report_data.generated_timestamp}
- **Nodes:** {metrics.get('node_count', 0):,}
- **Edges:** {metrics.get('edge_count', 0):,}
- **Total Funding:** {format_currency(metrics.get('total_funding', 0))}
        """)
    
    # Technical Details (for debugging)
    with st.expander("🛠️ Technical Details", expanded=False):
        st.caption("Raw metrics used for report generation:")
        st.json(metrics)
    
    st.divider()
    
    # =========================================================================
    # Project Outputs (Downloads)
    # =========================================================================
    st.markdown("## 📥 Project Outputs")
    
    st.markdown("Download your briefing and supporting data.")
    
    # Organize by audience
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👔 Executive**")
        st.download_button(
            label="📄 Download Executive Briefing",
            data=st.session_state.generated_report,
            file_name=f"{selected_id}_insight_report_v2.md",
            mime="text/markdown",
            help="Markdown format, convertible to PDF",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**📊 Analyst**")
        
        # Create metrics JSON
        metrics_json = json.dumps(metrics, indent=2, default=str)
        st.download_button(
            label="📊 Download Metrics (JSON)",
            data=metrics_json,
            file_name=f"{selected_id}_metrics_v2.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        st.markdown("**🛠️ Developer**")
        
        # Create full package ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{selected_id}_insight_report_v2.md", st.session_state.generated_report)
            zf.writestr(f"{selected_id}_metrics_v2.json", metrics_json)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 Download Full Package (ZIP)",
            data=zip_buffer,
            file_name=f"{selected_id}_v2_package.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # =========================================================================
    # Footer
    # =========================================================================
    st.divider()
    st.caption(f"*Report generated by {cfg.GENERATOR_CREDIT} v{cfg.APP_VERSION}*")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
