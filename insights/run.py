"""
C4C InsightGraph — Network Analysis & Briefing Generator

Computes network metrics, brokerage roles, and generates insight cards
from canonical nodes.csv and edges.csv.

Usage:
    python -m insights.run --nodes <path> --edges <path> --out <dir>
    
    Or with defaults (GLFN demo):
    python -m insights.run

VERSION HISTORY:
----------------
v3.0.22 (2025-12-22): Synthesis metadata in manifest
- NEW: manifest["synthesis"] block with downstream tool guidance
- NEW: synthesis_guides output paths in manifest
- NEW: VISUAL_SYNTHESIS_GUIDE.md, SYNTHESIS_MODE_PROMPT.md, SYNTHESIS_CHECKLIST.md
- PURPOSE: Self-describing exports for NotebookLM, slide tools, infographics
- CONSTRAINT: Embedded epistemic discipline for AI-generated summaries

v3.0.21 (2025-12-22): Soft recommendation language fix + Decision Options rename
- FIX: Replaced "this suggests/indicates/reveals" with descriptive usage framing
- FIX: Replaced "Opportunity:" labels with "Decision Option:"
- RENAME: "Strategic Recommendations" → "Decision Options"
- NEW: Section note clarifying these are not recommendations
- Language pattern: "Teams often use this signal to decide whether…"

v3.0.20 (2025-12-22): Language audit + markdown rendering fix
- FIX: Section subtitle now renders as HTML (not raw _underscores_)
- FIX: Removed "natural partners/hub/alignment" over-claiming language
- FIX: Replaced "should/must" with conditional phrasing
- FIX: "strong consensus" → "shared investment priorities"
- CSS: Added .section-subtitle class
- Authoring contract enforced (no prescriptive language)

v3.0.19 (2025-12-22): Signal Intensity Framework (interpretive governance)
- NEW: signal_intensity field (low/medium/high) governs reader attention
- NEW: Global "no action is valid" sentence in every Decision Lens footer
- NEW: Intensity-specific badge colors (gray/indigo/orange)
- REMOVED: badge and practical_use fields (replaced by signal_intensity)
- GOVERNANCE: This is the interpretive architecture lock - Portfolio Twins frozen
- CSS: .decision-lens__badge--low/medium/high, .decision-lens__footer

v3.0.18 (2025-12-22): Decision Lens grid layout + badges
- NEW: Decision Lens rendered as 3-column grid (responsive)
- NEW: Optional badge support (Directional signal, High leverage, etc.)
- NEW: Practical use sentence for action-shaped "so what"
- NEW: Speed bump subtitle for Portfolio Twins (prevents skimmer misread)
- CSS: .decision-lens component with header, grid, guardrail, badge styling
- Follows C4C Report Authoring Guide v1.0 schema

v3.0.17 (2025-12-22): Decision Lens guardrails + improved framing
- NEW: "What not to over-interpret" guardrail in every Decision Lens
- IMPROVED: All section intros now clarify what the analysis does NOT show
- IMPROVED: DECISION_LENS content rewritten for clarity and action-orientation
- IMPROVED: Portfolio Twins intro emphasizes "touchpoints, not aligned strategies"
- Follows feedback: prevents client over-reaction, protects analytical credibility

v3.0.16 (2025-12-22): Portfolio Twins tier-consistent language
- FIX: Per-pair narratives now match overall tier (weak/moderate/strong)
- FIX: "Natural partners" only used for high Jaccard (>= 0.15)
- NEW: Portfolio size context (e.g., "20 of 620 vs 540")
- NEW: Adjusted thresholds: Strong >= 0.15, Moderate >= 0.05, Weak < 0.05
- FIX: Opportunity text now tier-appropriate
- FIX: Decision Lens text now tier-neutral

v3.0.15 (2025-12-21): Decision Lens + Executive Summary (C4C Authoring Guide v1.0)
- NEW: Executive Summary section after header with key signals
- NEW: Decision Lens blocks for each section (What/Why/Next)
- NEW: Human-readable section intros
- NEW: .callout-decision CSS class with purple accent
- NEW: :::decision-lens markdown syntax support
- NEW: DECISION_LENS and SECTION_INTROS content constants
- Follows C4C Report Authoring Guide v1.0 structure

v3.0.14 (2025-12-21): Collapsible sections + polish
- NEW: Collapsible sections with Expand/Collapse buttons
- NEW: Skip-link now off-screen until focused (better UX)
- NEW: TOC links auto-expand collapsed sections
- Hide toggle buttons in print
- Section h2 now flexbox layout for button alignment

v3.0.13 (2025-12-21): Accessibility + print polish
- NEW: Skip-to-content link for accessibility
- NEW: html { color-scheme: light; } for consistent form controls
- NEW: print-color-adjust on all callouts (not just banner)
- Hide skip-link in print

v3.0.12 (2025-12-21): Funders always in-lens
- FIX: Funders are ALWAYS in-lens (they define the network)
- The geographic lens is about grantees, not funders
- Removes conditional fallback that wasn't working

v3.0.11 (2025-12-21): Portfolio Twins fix
- FIX: Portfolio Twins now shows pair names (e.g., "Funder A ↔ Funder B")
- FIX: Added rank to each pair for proper rendering
- FIX: Markdown rendering shows entities even without explicit rank

v3.0.10 (2025-12-21): Fixed region lens funder fallback
- FIX: Column name was 'network_role' but should be 'network_role_code'
- Funders without state data now correctly default to in-lens

v3.0.9 (2025-12-21): HTML rendering improvements
- FIX: Blockquotes (> lines) now render as styled callouts
- FIX: Duplicate H1 removed (skip first H1 in markdown)
- FIX: Health banner now appears (fixed insight_cards["health"] key)
- NEW: Use Case labels styled with capsule
- NEW: Signal indicators (🟢🟡🔴) styled as pills
- NEW: CSS for .use-case, .signal-green/yellow/red classes

v3.0.8 (2025-12-21): Fixed region lens membership
- FIX: Check 'state' column first, then 'region' as fallback
- FIX: Funders without state data default to in-lens (they define the network)
- Long-term: state data should be populated for all orgs

v3.0.7 (2025-12-21): Added HTML report rendering
- NEW: render_html_report() function for styled HTML output
- NEW: Embedded CSS template with C4C branding
- NEW: Table of contents generation
- NEW: Section wrapping and callout styling
- Print-friendly and mobile-responsive design

v3.0.6 (2025-12-21): Added manifest.json for bundle traceability
- NEW: generate_manifest() function for structured bundle metadata
- NEW: Bundle format version 1.0
- Prep for HTML report rendering

v3.0.5 (2025-12-21): Added Roles × Region Lens analysis
- NEW: Canonical role vocabulary (FUNDER, GRANTEE, FUNDER_GRANTEE, BOARD_MEMBER, ORGANIZATION, INDIVIDUAL)
- NEW: Region lens configuration (project_config.json or defaults)
- NEW: in_region_lens membership computed per node
- NEW: Roles × Lens cross-tabulation (counts by role, in/out of lens)
- NEW: Edge flow categories (IN→IN, IN→OUT, OUT→IN, OUT→OUT)
- NEW: Roles × Region Lens section in insight_report.md
- Supports Great Lakes (Binational) lens out of the box
- RENAMED: "Insight Engine" → "InsightGraph"

v3.0.4 (2025-12-20): Fixed bridge detection to focus on largest component
- FIX: Articulation points now computed only within largest connected component
- FIX: Impact counts nodes disconnected from main cluster, not peripheral clusters
- Previously small Canadian foundations appeared as top bridges incorrectly

v3.0.3 (2025-12-20): Fixed hidden broker threshold calculation
- FIX: Betweenness percentile now computed only among nodes with btw > 0
- FIX: Hidden broker uses 85th percentile betweenness + below-40th percentile degree
- Previously 75th percentile of all nodes was 0.0, catching no one

v3.0.2 (2025-12-20): Critical fixes for betweenness and bridge detection
- FIX: Betweenness now computed on undirected graph (was always 0 on DiGraph)
- FIX: Bridge ranking now sorted by impact (nodes isolated if removed)
- FIX: Health score now includes governance factor (board interlocks)
- Aligned with V2 metrics output

v3.0.1 (2025-12-19): Fixed hidden broker detection bug
- Added betweenness > 0 check to is_broker and is_hidden_broker
- Previously flagged 2,810 nodes with betweenness=0 as hidden brokers
- Now correctly requires actual brokerage (betweenness > 0)
"""

import argparse
import json
import hashlib
import re
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# =============================================================================
# Version
# =============================================================================

ENGINE_VERSION = "3.0.22"
BUNDLE_FORMAT_VERSION = "1.0"

# C4C logo as base64 (80px, ~4KB) for self-contained HTML reports
C4C_LOGO_BASE64 = "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"

# =============================================================================
# Constants
# =============================================================================

DEFAULT_NODES = Path(__file__).parent.parent / "demo_data" / "glfn" / "nodes.csv"
DEFAULT_EDGES = Path(__file__).parent.parent / "demo_data" / "glfn" / "edges.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "output"

CONNECTOR_THRESHOLD = 75
BROKER_THRESHOLD = 75
HIDDEN_BROKER_DEGREE_CAP = 40
CAPITAL_HUB_THRESHOLD = 75


# =============================================================================
# Data Loading & Validation
# =============================================================================

def load_and_validate(nodes_path: Path, edges_path: Path) -> tuple:
    """Load canonical CSVs and validate required columns."""
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    required_node_cols = {"node_id", "node_type", "label"}
    missing_node_cols = required_node_cols - set(nodes_df.columns)
    if missing_node_cols:
        raise ValueError(f"nodes.csv missing required columns: {missing_node_cols}")
    
    required_edge_cols = {"edge_id", "edge_type", "from_id", "to_id"}
    missing_edge_cols = required_edge_cols - set(edges_df.columns)
    if missing_edge_cols:
        raise ValueError(f"edges.csv missing required columns: {missing_edge_cols}")
    
    print(f"✓ Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")
    return nodes_df, edges_df


# =============================================================================
# Graph Construction
# =============================================================================

def build_grant_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build directed grant graph: ORG → ORG, weighted by amount."""
    G = nx.DiGraph()
    org_nodes = nodes_df[nodes_df["node_type"] == "ORG"]
    for _, row in org_nodes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"]
    for _, row in grant_edges.iterrows():
        amount = row.get("amount", 0) or 0
        if G.has_edge(row["from_id"], row["to_id"]):
            G[row["from_id"]][row["to_id"]]["weight"] += float(amount)
            G[row["from_id"]][row["to_id"]]["grant_count"] += 1
        else:
            G.add_edge(row["from_id"], row["to_id"], weight=float(amount), grant_count=1)
    
    print(f"✓ Grant graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_board_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build bipartite board graph: PERSON — ORG."""
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    board_edges = edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]
    for _, row in board_edges.iterrows():
        G.add_edge(row["from_id"], row["to_id"], edge_type="BOARD_MEMBERSHIP")
    
    print(f"✓ Board graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_interlock_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build ORG—ORG interlock graph weighted by shared board members."""
    G = nx.Graph()
    org_nodes = nodes_df[nodes_df["node_type"] == "ORG"]
    for _, row in org_nodes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    board_edges = edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]
    person_to_orgs = defaultdict(set)
    for _, row in board_edges.iterrows():
        person_to_orgs[row["from_id"]].add(row["to_id"])
    
    interlock_weights = defaultdict(lambda: {"weight": 0, "shared_people": []})
    for person_id, orgs in person_to_orgs.items():
        orgs_list = list(orgs)
        for i, org1 in enumerate(orgs_list):
            for org2 in orgs_list[i+1:]:
                key = tuple(sorted([org1, org2]))
                interlock_weights[key]["weight"] += 1
                interlock_weights[key]["shared_people"].append(person_id)
    
    for (org1, org2), data in interlock_weights.items():
        G.add_edge(org1, org2, weight=data["weight"], shared_people=data["shared_people"])
    
    print(f"✓ Interlock graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# =============================================================================
# Layer 1: Base Metrics
# =============================================================================

def compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph) -> pd.DataFrame:
    """Compute base metrics for all nodes."""
    metrics = []
    
    # FIX: Convert to undirected for betweenness calculation
    # Directed grant graph has no paths through nodes (funder→grantee only)
    grant_undirected = grant_graph.to_undirected() if grant_graph.number_of_edges() > 0 else nx.Graph()
    grant_betweenness = nx.betweenness_centrality(grant_undirected) if grant_undirected.number_of_edges() > 0 else {}
    grant_pagerank = nx.pagerank(grant_graph, weight="weight") if grant_graph.number_of_edges() > 0 else {}
    board_betweenness = nx.betweenness_centrality(board_graph) if board_graph.number_of_edges() > 0 else {}
    
    node_to_component = {}
    if grant_graph.number_of_nodes() > 0:
        grant_undirected = grant_graph.to_undirected()
        for i, comp in enumerate(nx.connected_components(grant_undirected)):
            for node in comp:
                node_to_component[node] = i
    
    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        node_type = row["node_type"]
        
        m = {
            "node_id": node_id,
            "node_type": node_type,
            "label": row.get("label", ""),
            "jurisdiction": row.get("jurisdiction", ""),
            "org_slug": row.get("org_slug", ""),
            "region": row.get("region", ""),
        }
        
        if node_type == "ORG":
            m["degree"] = grant_graph.degree(node_id) if node_id in grant_graph else 0
            m["grant_in_degree"] = grant_graph.in_degree(node_id) if node_id in grant_graph else 0
            m["grant_out_degree"] = grant_graph.out_degree(node_id) if node_id in grant_graph else 0
            
            outflow = sum(d.get("weight", 0) for _, _, d in grant_graph.out_edges(node_id, data=True)) if node_id in grant_graph else 0
            m["grant_outflow_total"] = outflow
            m["betweenness"] = grant_betweenness.get(node_id, 0)
            m["pagerank"] = grant_pagerank.get(node_id, 0)
            m["component_id"] = node_to_component.get(node_id, -1)
            m["shared_board_count"] = interlock_graph.degree(node_id) if node_id in interlock_graph else 0
            m["boards_served"] = None
        else:
            m["boards_served"] = board_graph.degree(node_id) if node_id in board_graph else 0
            m["degree"] = m["boards_served"]
            m["betweenness"] = board_betweenness.get(node_id, 0)
            m["grant_in_degree"] = None
            m["grant_out_degree"] = None
            m["grant_outflow_total"] = None
            m["pagerank"] = None
            m["component_id"] = None
            m["shared_board_count"] = None
        
        metrics.append(m)
    
    return pd.DataFrame(metrics)


# =============================================================================
# Layer 2: Derived Signals
# =============================================================================

def compute_derived_signals(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived boolean flags based on percentile thresholds."""
    df = metrics_df.copy()
    df["is_connector"] = 0
    df["is_broker"] = 0
    df["is_hidden_broker"] = 0
    df["is_capital_hub"] = 0
    df["is_isolated"] = 0
    
    org_mask = df["node_type"] == "ORG"
    org_df = df[org_mask]
    
    if len(org_df) > 0:
        degree_75 = np.percentile(org_df["degree"].dropna(), CONNECTOR_THRESHOLD)
        outflow_vals = org_df["grant_outflow_total"].dropna()
        outflow_75 = np.percentile(outflow_vals, CAPITAL_HUB_THRESHOLD) if len(outflow_vals) > 0 else 0
        
        df.loc[org_mask & (df["degree"] >= degree_75), "is_connector"] = 1
        df.loc[org_mask & (df["grant_outflow_total"] >= outflow_75) & (df["grant_outflow_total"] > 0), "is_capital_hub"] = 1
        df.loc[org_mask & (df["degree"] == 1), "is_isolated"] = 1
        
        # FIX: Compute broker thresholds only among nodes with non-zero betweenness
        # This prevents the 75th percentile from being 0 when most nodes have no betweenness
        connectors = org_df[org_df["betweenness"] > 0]
        if len(connectors) > 0:
            # 85th percentile among actual connectors (matches V2 hidden broker count)
            betweenness_85 = np.percentile(connectors["betweenness"], 85)
            # 40th percentile degree among connectors (hidden = low visibility)
            degree_40 = np.percentile(connectors["degree"], 40)
            
            # is_broker: high betweenness among connectors
            df.loc[org_mask & (df["betweenness"] >= betweenness_85), "is_broker"] = 1
            
            # is_hidden_broker: high betweenness BUT low degree (bottom 40% among connectors)
            df.loc[org_mask & (df["betweenness"] >= betweenness_85) & (df["degree"] <= degree_40), "is_hidden_broker"] = 1
    
    person_mask = df["node_type"] == "PERSON"
    df.loc[person_mask & (df["boards_served"] >= 2), "is_connector"] = 1
    
    return df


# =============================================================================
# Flow Statistics
# =============================================================================

def compute_flow_stats(edges_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
    """Compute system-level funding flow statistics."""
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy()
    
    if grant_edges.empty:
        return {"total_grant_amount": 0, "grant_count": 0, "funder_count": 0, 
                "grantee_count": 0, "top_5_funders_share": 0, "top_10_grantees_share": 0, "multi_funder_grantees": 0}
    
    grant_edges["amount"] = pd.to_numeric(grant_edges["amount"], errors="coerce").fillna(0)
    total_amount = grant_edges["amount"].sum()
    
    funder_totals = grant_edges.groupby("from_id")["amount"].sum().sort_values(ascending=False)
    top_5_share = (funder_totals.head(5).sum() / total_amount * 100) if total_amount > 0 else 0
    
    grantee_totals = grant_edges.groupby("to_id")["amount"].sum().sort_values(ascending=False)
    top_10_share = (grantee_totals.head(10).sum() / total_amount * 100) if total_amount > 0 else 0
    
    grantee_funder_counts = grant_edges.groupby("to_id")["from_id"].nunique()
    multi_funder = (grantee_funder_counts >= 2).sum()
    
    return {
        "total_grant_amount": float(total_amount),
        "grant_count": len(grant_edges),
        "funder_count": len(funder_totals),
        "grantee_count": len(grantee_totals),
        "top_5_funders_share": round(top_5_share, 1),
        "top_10_grantees_share": round(top_10_share, 1),
        "multi_funder_grantees": int(multi_funder),
    }


def compute_portfolio_overlap(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compute funder × funder portfolio overlap matrix."""
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy()
    if grant_edges.empty:
        return pd.DataFrame()
    
    funder_grantees = grant_edges.groupby("from_id")["to_id"].apply(set).to_dict()
    funders = list(funder_grantees.keys())
    overlaps = []
    
    for i, f1 in enumerate(funders):
        for f2 in funders[i+1:]:
            shared = funder_grantees[f1] & funder_grantees[f2]
            if shared:
                jaccard = len(shared) / len(funder_grantees[f1] | funder_grantees[f2])
                overlaps.append({
                    "funder_1": f1, "funder_2": f2,
                    "shared_grantees": len(shared),
                    "funder_1_portfolio": len(funder_grantees[f1]),
                    "funder_2_portfolio": len(funder_grantees[f2]),
                    "jaccard_similarity": round(jaccard, 3),
                    "shared_grantee_ids": list(shared),
                })
    
    return pd.DataFrame(overlaps).sort_values("shared_grantees", ascending=False) if overlaps else pd.DataFrame()


# =============================================================================
# Network Health Score
# =============================================================================

def compute_network_health(flow_stats, metrics_df, n_components, largest_component_pct, multi_funder_pct):
    """Compute 0-100 health score for funder network.
    
    Factors:
    - Coordination (multi-funder grantees): 0-25 points
    - Connectivity (largest component): 0-20 points
    - Concentration (top 5 share): -15 to +10 points
    - Governance (board interlocks): 0-15 points (NEW)
    """
    positive_factors, risk_factors = [], []
    score = 20.0  # Lower base to account for governance factor
    
    # Coordination signal
    if multi_funder_pct >= 10:
        score += 25
        positive_factors.append(f"🟢 **Strong coordination** — {multi_funder_pct:.1f}% of grantees have multiple funders")
    elif multi_funder_pct >= 5:
        score += 15
        positive_factors.append(f"🟡 **Moderate coordination** — {multi_funder_pct:.1f}% have multiple funders")
    elif multi_funder_pct >= 1:
        score += 5
        risk_factors.append(f"🔴 **Low coordination** — only {multi_funder_pct:.1f}% have multiple funders")
    else:
        risk_factors.append("🔴 **No portfolio overlap** — funders operate in silos")
    
    # Connectivity
    if largest_component_pct >= 80:
        score += 20
        positive_factors.append(f"🟢 **Highly connected** — {largest_component_pct:.0f}% of organizations linked through shared funding")
    elif largest_component_pct >= 50:
        score += 10
    else:
        risk_factors.append(f"🔴 **Fragmented** — only {largest_component_pct:.0f}% connected through shared funding, most operate in isolated clusters")
    
    # Concentration
    top5_share = flow_stats.get("top_5_funders_share", 100)
    if top5_share >= 95:
        score -= 15
        risk_factors.append(f"🔴 **Extreme concentration** — top 5 control {top5_share:.0f}%")
    elif top5_share < 80:
        score += 10
        positive_factors.append(f"🟢 **Distributed funding** — top 5 control {top5_share:.0f}%")
    
    # Governance connectivity (NEW)
    org_metrics = metrics_df[metrics_df["node_type"] == "ORG"]
    foundations = org_metrics[org_metrics["grant_outflow_total"] > 0] if "grant_outflow_total" in org_metrics.columns else pd.DataFrame()
    if len(foundations) > 0:
        pct_with_interlocks = (foundations["shared_board_count"] > 0).mean() * 100
        if pct_with_interlocks >= 20:
            score += 15
            positive_factors.append(f"🟢 **Governance ties** — {pct_with_interlocks:.0f}% of funders share board members")
        elif pct_with_interlocks >= 5:
            score += 8
        elif pct_with_interlocks == 0:
            risk_factors.append("🔴 **No governance ties** — funders have no shared board members")
    
    score = max(0, min(100, int(score)))
    label = "Healthy coordination" if score >= 70 else "Mixed signals" if score >= 40 else "Fragmented / siloed"
    
    return score, label, positive_factors, risk_factors


# =============================================================================
# Badge System
# =============================================================================

FUNDER_BADGES = {
    "capital_hub": {"emoji": "💰", "label": "Capital Hub", "color": "#10B981"},
    "hidden_broker": {"emoji": "🔍", "label": "Hidden Broker", "color": "#6366F1"},
    "connector": {"emoji": "🔗", "label": "Connector", "color": "#3B82F6"},
    "isolated": {"emoji": "⚪", "label": "Isolated", "color": "#9CA3AF"},
    "bridge": {"emoji": "🌉", "label": "Critical Bridge", "color": "#F97316"},
}

HEALTH_BADGES = {
    "healthy": {"emoji": "🟢", "label": "Healthy", "color": "#10B981"},
    "mixed": {"emoji": "🟡", "label": "Mixed", "color": "#FBBF24"},
    "fragile": {"emoji": "🔴", "label": "Fragile", "color": "#EF4444"},
}

CONCENTRATION_BADGES = {
    "distributed": {"emoji": "🟢", "label": "Distributed", "color": "#10B981"},
    "moderate": {"emoji": "🟡", "label": "Moderate", "color": "#FBBF24"},
    "concentrated": {"emoji": "🟠", "label": "High", "color": "#F97316"},
    "extreme": {"emoji": "🔴", "label": "Extreme", "color": "#EF4444"},
}


def _pct_bucket(pct: float) -> str:
    """Convert 0-1 percentile to human label."""
    if pct is None:
        return "unknown"
    if pct >= 0.95:
        return "very high"
    if pct >= 0.75:
        return "high"
    if pct >= 0.40:
        return "moderate"
    return "low"


# =============================================================================
# Narrative Helpers
# =============================================================================

def describe_funder_with_recommendation(
    label: str,
    grant_outflow: float,
    grantee_count: int,
    shared_board_count: int,
    betweenness_pct: float,
    is_hidden_broker: bool = False,
    is_capital_hub: bool = False,
    is_isolated: bool = False,
) -> tuple:
    """
    Generate narrative description and recommendation for a funder.
    Returns (blurb, recommendation)
    """
    btw_bucket = _pct_bucket(betweenness_pct)
    
    # Build role-specific framing
    if is_capital_hub and is_hidden_broker:
        role_sentence = (
            f"**{label}** is both a capital hub (${grant_outflow:,.0f} distributed) and a hidden broker — "
            f"they quietly shape funding flows while bridging otherwise disconnected parts of the network."
        )
    elif is_hidden_broker:
        role_sentence = (
            f"**{label}** operates as a hidden broker — despite moderate visibility, they occupy a critical "
            f"bridging position connecting groups that don't otherwise interact."
        )
    elif is_capital_hub:
        role_sentence = (
            f"**{label}** is a capital hub, distributing ${grant_outflow:,.0f} to {grantee_count} grantees. "
            f"Their funding decisions significantly shape the field."
        )
    elif is_isolated:
        role_sentence = (
            f"**{label}** operates independently with minimal network ties. "
            f"They fund ${grant_outflow:,.0f} to {grantee_count} grantees but share few grantees or board members with peers."
        )
    else:
        role_sentence = (
            f"**{label}** distributes ${grant_outflow:,.0f} across {grantee_count} grantees, "
            f"holding a meaningful position in the funding landscape."
        )
    
    # Add governance context
    if shared_board_count >= 3:
        gov_context = f"They share board members with {shared_board_count} other organizations, enabling informal coordination across multiple foundations."
    elif shared_board_count > 0:
        gov_context = f"They share board ties with {shared_board_count} other organization(s), creating potential coordination channels."
    else:
        gov_context = "They have no board interlocks with other network members — operating in governance isolation."
    
    # Add betweenness context if relevant
    if btw_bucket in ("very high", "high") and not is_hidden_broker:
        btw_context = "They often connect funders or grantees who would not otherwise interact."
    else:
        btw_context = ""
    
    blurb = f"{role_sentence} {gov_context}"
    if btw_context:
        blurb += f" {btw_context}"
    
    # Generate recommendation
    recs = []
    
    if is_hidden_broker:
        recs.append(
            "Engage them in cross-funder coordination — they bridge groups that don't otherwise connect. "
            "Their structural position makes them valuable for pilot initiatives or coalition-building."
        )
    
    if is_capital_hub and shared_board_count == 0:
        recs.append(
            "As a major funder with no governance ties, consider facilitating introductions to peer funders "
            "for alignment conversations or joint learning."
        )
    
    if shared_board_count >= 3:
        recs.append(
            "Leverage their board relationships for coalition-building or joint initiatives — "
            "they can convene multiple foundations through existing relationships."
        )
    
    if is_isolated and grantee_count > 20:
        recs.append(
            "Despite isolation, their broad portfolio suggests shared interests with other funders. "
            "Map potential overlap to identify coordination touchpoints worth exploring."
        )
    
    if not recs:
        if grantee_count > 50:
            recs.append(
                "Their broad portfolio makes them a good candidate for field-wide learning, "
                "impact measurement partnerships, or knowledge-sharing initiatives."
            )
        else:
            recs.append(
                "Consider inviting them to funder coordination conversations in their focus areas."
            )
    
    recommendation = " ".join(recs)
    return blurb, f"💡 **Suggested Focus:** {recommendation}"


def describe_grantee(label, funder_count, total_received, funder_labels):
    """Generate narrative for a multi-funder grantee."""
    funders_str = ", ".join(funder_labels[:3])
    if len(funder_labels) > 3:
        funders_str += f" + {len(funder_labels) - 3} more"
    
    if funder_count >= 4:
        blurb = (
            f"**{label}** receives from {funder_count} network funders (${total_received:,.0f} from {funders_str}). "
            f"Teams often use shared investment patterns like this to identify potential coordination touchpoints."
        )
        rec = "Could be a candidate for joint impact measurement, aligned reporting, or a brief funder check-in."
    elif funder_count >= 3:
        blurb = (
            f"**{label}** receives from {funder_count} funders (${total_received:,.0f} from {funders_str}). "
            f"Overlap at this level is commonly used to assess whether coordination conversations would add value."
        )
        rec = "Some teams explore joint site visits, shared evaluation, or coordinated grant timing."
    else:
        blurb = (
            f"**{label}** receives from {funder_count} funders (${total_received:,.0f} from {funders_str}). "
            f"Shared funding at this level is often used to decide whether a brief funder check-in is warranted."
        )
        rec = "Some teams use these touchpoints to explore whether investments could be more intentionally aligned."
    
    return blurb, f"💡 **Decision Option:** {rec}"


def describe_board_connector(label, board_count, org_labels):
    """Generate narrative for a person on multiple boards."""
    orgs_str = ", ".join(org_labels[:3])
    if len(org_labels) > 3:
        orgs_str += f" + {len(org_labels) - 3} more"
    
    if board_count >= 4:
        blurb = (
            f"**{label}** serves on {board_count} boards ({orgs_str}), creating dense governance links. "
            f"They can facilitate informal information flow, relationship-building, and strategic alignment across multiple foundations."
        )
        rec = (
            "High-leverage connector — engage them for strategic introductions or coalition navigation. "
            "Monitor for potential overload; consider whether responsibilities should be shared."
        )
    elif board_count >= 2:
        blurb = (
            f"**{label}** serves on {board_count} boards ({orgs_str}), creating governance bridges between these organizations. "
            f"They enable informal coordination that formal structures often miss."
        )
        rec = "Include them in cross-organization strategy conversations where their multi-board perspective adds value."
    else:
        blurb = f"**{label}** serves on {board_count} board(s), with focused governance involvement."
        rec = "Keep them informed of cross-organizational initiatives relevant to their board."
    
    return blurb, f"💡 **Suggested Focus:** {rec}"


# =============================================================================
# Decision Options Engine
# =============================================================================

def generate_strategic_recommendations(
    health_score: int,
    health_label: str,
    flow_stats: dict,
    multi_funder_pct: float,
    largest_component_pct: float,
    n_isolated_funders: int,
    total_funders: int,
    n_hidden_brokers: int,
    n_board_conduits: int,
) -> str:
    """
    Generate rule-based decision options based on network signals.
    Returns markdown string.
    """
    sections = []
    
    # Section note (authoring contract requirement)
    section_note = (
        "_The options below describe common ways teams apply these signals in practice; "
        "they are not recommendations._"
    )
    sections.append(section_note + "\n")
    
    # Framing based on health
    if health_score >= 70:
        intro = (
            "The funding network shows **healthy coordination signals**. Teams with similar patterns "
            "often focus on **deepening strategic relationships** and **protecting what works**."
        )
    elif health_score >= 40:
        intro = (
            "The network shows **mixed signals**. Some coordination exists, but structural gaps limit "
            "how effectively funders can align. Teams often assess whether targeted interventions would add value."
        )
    else:
        intro = (
            "The network appears **fragmented**. Funders operate largely in silos with minimal coordination. "
            "Teams often assess whether building basic connective tissue would be valuable."
        )
    
    sections.append(f"### 🧭 How to Read This\n\n{intro}\n")
    
    # Coordination recommendations
    coord_recs = []
    
    if multi_funder_pct < 1:
        coord_recs.append(
            "**Build initial overlap:** Almost no grantees receive from multiple funders. Start by mapping "
            "where portfolios *could* overlap based on thematic focus, then facilitate introductions."
        )
    elif multi_funder_pct < 5:
        coord_recs.append(
            "**Explore existing touchpoints:** A small number of shared grantees exist. These could serve as anchors — "
            "consider convening funders around specific grantees to build relationships and explore joint action."
        )
    
    if coord_recs:
        sections.append("### 🔗 Strengthen Funder Coordination\n\n" + 
                       "\n\n".join([f"- {r}" for r in coord_recs]) + "\n")
    
    # Governance recommendations
    gov_recs = []
    
    isolated_pct = n_isolated_funders / max(total_funders, 1) * 100
    if isolated_pct > 70:
        gov_recs.append(
            "**Address governance silos:** Most funders have no shared board members. Consider facilitating "
            "cross-foundation board dialogues or joint trustee convenings to build informal relationships."
        )
    
    if n_board_conduits == 0:
        gov_recs.append(
            "**Identify potential bridge-builders:** No one currently serves on multiple boards. Look for "
            "respected individuals who could be nominated to additional boards to create connective tissue."
        )
    elif n_board_conduits >= 5:
        gov_recs.append(
            "**Leverage existing connectors:** Multiple people serve on 2+ boards. Engage them intentionally "
            "in coordination efforts — they have built-in legitimacy across organizations."
        )
    
    if gov_recs:
        sections.append("### 🏛️ Strengthen Governance Ties\n\n" + 
                       "\n\n".join([f"- {r}" for r in gov_recs]) + "\n")
    
    # Concentration recommendations
    conc_recs = []
    
    top5_share = flow_stats.get("top_5_funders_share", 0)
    if top5_share >= 95:
        conc_recs.append(
            "**Monitor concentration risk:** A handful of funders control nearly all capital. Consider whether "
            "this field-shaping power could benefit from broader funder input."
        )
        conc_recs.append(
            "**Engage smaller funders strategically:** Though they control less capital, smaller funders may "
            "have flexibility, relationships, or risk tolerance that larger funders lack."
        )
    
    if conc_recs:
        sections.append("### 💰 Address Funding Concentration\n\n" + 
                       "\n\n".join([f"- {r}" for r in conc_recs]) + "\n")
    
    # Broker recommendations
    broker_recs = []
    
    if n_hidden_brokers > 0:
        broker_recs.append(
            f"**Engage hidden brokers:** {n_hidden_brokers} organization(s) quietly bridge disconnected parts "
            "of the network. Involve them in coordination design — they see patterns others miss."
        )
    
    if broker_recs:
        sections.append("### 🌉 Work with Network Brokers\n\n" + 
                       "\n\n".join([f"- {r}" for r in broker_recs]) + "\n")
    
    # Fallback
    if len(sections) == 1:
        sections.append(
            "### ✨ No Major Structural Gaps\n\n"
            "The network appears structurally sound. Focus on **clarifying shared purpose** and "
            "**deepening existing relationships** rather than changing the structure itself."
        )
    
    return "\n".join(sections)


# =============================================================================
# Insight Cards Generation
# =============================================================================

def generate_insight_cards(nodes_df, edges_df, metrics_df, interlock_graph, flow_stats, overlap_df, project_id="glfn"):
    """Generate insight cards with narrative descriptions."""
    cards = []
    node_labels = dict(zip(nodes_df["node_id"], nodes_df["label"]))
    
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy() if not edges_df.empty else pd.DataFrame()
    board_edges = edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"].copy() if not edges_df.empty else pd.DataFrame()
    
    if not grant_edges.empty and "amount" in grant_edges.columns:
        grant_edges["amount"] = pd.to_numeric(grant_edges["amount"], errors="coerce").fillna(0)
    
    grantee_funders = grant_edges.groupby("to_id")["from_id"].nunique() if not grant_edges.empty else pd.Series(dtype=int)
    
    # Health metrics
    total_grantees = flow_stats.get("grantee_count", 0)
    multi_funder_count = flow_stats.get("multi_funder_grantees", 0)
    multi_funder_pct = (multi_funder_count / total_grantees * 100) if total_grantees > 0 else 0
    
    grant_graph = nx.Graph()
    if not grant_edges.empty:
        for _, row in grant_edges.iterrows():
            grant_graph.add_edge(row["from_id"], row["to_id"])
    
    n_components = nx.number_connected_components(grant_graph) if grant_graph.number_of_nodes() > 0 else 0
    largest_cc_pct = len(max(nx.connected_components(grant_graph), key=len)) / grant_graph.number_of_nodes() * 100 if n_components > 0 else 0
    
    health_score, health_label, positive_factors, risk_factors = compute_network_health(
        flow_stats, metrics_df, n_components, largest_cc_pct, multi_funder_pct
    )
    
    # Pre-compute counts for recommendations
    org_metrics = metrics_df[metrics_df["node_type"] == "ORG"]
    foundations = org_metrics[org_metrics["grant_outflow_total"] > 0]
    n_isolated_funders = len(foundations[foundations["shared_board_count"] == 0])
    total_funders = len(foundations)
    n_hidden_brokers = len(metrics_df[(metrics_df["is_hidden_broker"] == 1) & (metrics_df["node_type"] == "ORG")])
    person_metrics = metrics_df[metrics_df["node_type"] == "PERSON"]
    n_board_conduits = len(person_metrics[person_metrics["boards_served"] >= 2])
    
    # =========================================================================
    # Card 1: Network Health Overview
    # =========================================================================
    health_emoji = "🟢" if health_score >= 70 else "🟡" if health_score >= 40 else "🔴"
    
    if health_score >= 70:
        health_narrative = (
            "The funding network shows **healthy coordination signals**. Funders share grantees and "
            "governance ties, suggesting organic alignment. Focus on deepening strategic relationships."
        )
    elif health_score >= 40:
        health_narrative = (
            "The network shows **mixed signals**. Some coordination exists, but structural gaps limit "
            "how effectively funders can align. Targeted bridge-building could unlock value."
        )
    else:
        health_narrative = (
            "The network appears **fragmented**. Funders operate largely in silos with minimal portfolio "
            "overlap or governance ties. Building basic coordination infrastructure is the priority."
        )
    
    # Build rich interpretation statements
    if multi_funder_pct >= 10:
        mf_interpretation = f"Strong signal — {multi_funder_pct:.1f}% of grantees have multiple funders, indicating active co-investment"
    elif multi_funder_pct >= 5:
        mf_interpretation = f"Moderate — {multi_funder_pct:.1f}% have multiple funders, suggesting coordination touchpoints"
    elif multi_funder_pct >= 1:
        mf_interpretation = f"Weak signal — only {multi_funder_pct:.1f}% have multiple funders, funders rarely co-invest"
    else:
        mf_interpretation = "No overlap — funders operate in complete silos with no shared grantees"
    
    if largest_cc_pct >= 90:
        cc_interpretation = f"Nearly all organizations ({largest_cc_pct:.0f}%) can reach each other through funding chains — a highly unified network"
    elif largest_cc_pct >= 70:
        cc_interpretation = f"Most organizations ({largest_cc_pct:.0f}%) are linked through overlapping grants, though some isolated clusters exist"
    elif largest_cc_pct >= 50:
        cc_interpretation = f"About half the organizations can reach each other through funding chains. The other {100-largest_cc_pct:.0f}% are in isolated pockets — funders with distinct portfolios that share nothing"
    else:
        cc_interpretation = f"Only {largest_cc_pct:.0f}% of organizations are connected through shared funding. Most funders operate in isolated clusters with completely distinct portfolios"
    
    top5 = flow_stats['top_5_funders_share']
    if top5 >= 95:
        conc_interpretation = f"Extreme — top 5 funders control {top5}%, a few actors dominate the landscape"
    elif top5 >= 80:
        conc_interpretation = f"High — top 5 control {top5}%, limited funder diversity"
    elif top5 >= 60:
        conc_interpretation = f"Moderate — top 5 control {top5}%, reasonable distribution"
    else:
        conc_interpretation = f"Distributed — top 5 control only {top5}%, healthy funder diversity"
    
    cards.append({
        "card_id": "network_health",
        "use_case": "System Framing",
        "title": "Network Health Overview",
        "summary": f"{health_emoji} **Network Health: {health_score}/100** — *{health_label}*\n\n{health_narrative}",
        "ranked_rows": [
            {"indicator": "Health Score", "value": f"{health_score}/100", "interpretation": health_label},
            {"indicator": "Multi-Funder Grantees", "value": f"{multi_funder_pct:.1f}%", "interpretation": mf_interpretation},
            {"indicator": "Connected through Shared Funding", "value": f"{largest_cc_pct:.0f}%", "interpretation": cc_interpretation},
            {"indicator": "Top 5 Funder Share", "value": f"{top5}%", "interpretation": conc_interpretation},
        ],
        "health_factors": {"positive": positive_factors, "risk": risk_factors},
        "evidence": {"node_ids": [], "edge_ids": []},
    })
    
    # =========================================================================
    # Card 2: Funding Concentration
    # =========================================================================
    top5_share = flow_stats['top_5_funders_share']
    total_amount = flow_stats['total_grant_amount']
    
    if top5_share >= 95:
        conc_emoji, conc_label = "🔴", "Extreme concentration"
        conc_narrative = (
            f"The top 5 funders control **{top5_share}%** of all funding (${total_amount:,.0f}). "
            f"This near-total concentration means a handful of actors shape the entire funding landscape. "
            f"If priorities shift at any of these foundations, large parts of the ecosystem could be affected.\n\n"
            f"💡 **Implication:** Track concentration trends. Identify emerging funders who could diversify the base."
        )
    elif top5_share >= 80:
        conc_emoji, conc_label = "🟠", "High concentration"
        conc_narrative = (
            f"The top 5 funders account for **{top5_share}%** of total funding. While some concentration is normal, "
            f"this level suggests limited funder diversity. Smaller funders may struggle to influence field direction.\n\n"
            f"💡 **Implication:** Encourage mid-tier funders to coordinate for collective impact."
        )
    else:
        conc_emoji, conc_label = "🟢", "Healthy distribution"
        conc_narrative = (
            f"Funding is relatively distributed — top 5 funders control {top5_share}% of ${total_amount:,.0f}. "
            f"This diversity provides resilience and multiple pathways for grantees.\n\n"
            f"💡 **Implication:** Maintain diversity. Look for coordination opportunities among mid-tier funders."
        )
    
    cards.append({
        "card_id": "concentration_snapshot",
        "use_case": "Funding Concentration",
        "title": "Funding Concentration",
        "summary": f"{conc_emoji} **{conc_label}**\n\n{conc_narrative}",
        "ranked_rows": [
            {"metric": "Total Funding", "value": f"${total_amount:,.0f}"},
            {"metric": "Funders", "value": str(flow_stats['funder_count'])},
            {"metric": "Grantees", "value": str(flow_stats['grantee_count'])},
            {"metric": "Top 5 Share", "value": f"{top5_share}%"},
            {"metric": "Multi-Funder Grantees", "value": str(multi_funder_count)},
        ],
        "evidence": {"node_ids": [], "edge_ids": []},
    })
    
    # =========================================================================
    # Card 3: Funder Overlap Clusters
    # =========================================================================
    if not grantee_funders.empty:
        multi_funder = grantee_funders[grantee_funders >= 2].sort_values(ascending=False)
        overlap_pct = len(multi_funder) / len(grantee_funders) * 100 if len(grantee_funders) > 0 else 0
        
        if overlap_pct < 1:
            overlap_emoji, overlap_label = "🔴", "Minimal overlap"
            overlap_narrative = (
                f"Only **{len(multi_funder)} grantees** ({overlap_pct:.1f}%) receive funding from multiple network members. "
                f"Funders appear to be operating in near-complete silos with almost no shared investments."
            )
        elif overlap_pct < 5:
            overlap_emoji, overlap_label = "🟡", "Limited overlap"
            overlap_narrative = (
                f"**{len(multi_funder)} grantees** ({overlap_pct:.1f}%) receive from 2+ funders. These shared investments "
                f"represent potential coordination touchpoints, though most grantees depend on a single funder."
            )
        else:
            overlap_emoji, overlap_label = "🟢", "Meaningful overlap"
            overlap_narrative = (
                f"**{len(multi_funder)} grantees** ({overlap_pct:.1f}%) receive from multiple funders. "
                f"Teams often use overlap at this level to assess whether coordination would add value."
            )
        
        ranked_rows = []
        for rank, grantee_id in enumerate(multi_funder.head(5).index, 1):
            funder_ids = grant_edges[grant_edges["to_id"] == grantee_id]["from_id"].unique().tolist()
            funder_labels = [node_labels.get(f, f) for f in funder_ids]
            total_received = grant_edges[grant_edges["to_id"] == grantee_id]["amount"].sum()
            blurb, rec = describe_grantee(node_labels.get(grantee_id, grantee_id), len(funder_ids), total_received, funder_labels)
            ranked_rows.append({
                "rank": rank, 
                "grantee": node_labels.get(grantee_id, grantee_id),
                "funders": len(funder_ids), 
                "amount": f"${total_received:,.0f}",
                "narrative": blurb, 
                "recommendation": rec
            })
        
        cards.append({
            "card_id": "funder_overlap_clusters",
            "use_case": "Funder Flow",
            "title": "Funder Overlap Clusters",
            "summary": f"{overlap_emoji} **{overlap_label}**\n\n{overlap_narrative}",
            "ranked_rows": ranked_rows,
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # =========================================================================
    # Card 4: Portfolio Twins
    # =========================================================================
    # Thresholds: Strong >= 0.15, Moderate >= 0.05, Weak < 0.05
    # (These are calibrated for philanthropic networks where 0.10+ is meaningful)
    if not overlap_df.empty:
        top = overlap_df.iloc[0]
        top_jaccard = top['jaccard_similarity']
        
        # Tier-consistent framing
        if top_jaccard >= 0.15:
            twin_emoji, twin_label = "🟢", "Strong alignment found"
            twin_narrative = (
                f"**{len(overlap_df)} funder pairs** share at least one grantee. The most aligned — "
                f"{node_labels.get(top['funder_1'], '')} & {node_labels.get(top['funder_2'], '')} — "
                f"share {int(top['shared_grantees'])} grantees (Jaccard: {top_jaccard:.2f}). "
                f"Teams often use high overlap like this to assess whether coordination conversations would add value."
            )
            opportunity_text = "💡 **Decision Option:** High-overlap pairs are commonly used to decide whether joint learning, aligned timing, or shared reporting pilots merit exploration."
        elif top_jaccard >= 0.05:
            twin_emoji, twin_label = "🟡", "Moderate alignment"
            twin_narrative = (
                f"**{len(overlap_df)} funder pairs** share grantees. Top pair: "
                f"{node_labels.get(top['funder_1'], '')} & {node_labels.get(top['funder_2'], '')} "
                f"({int(top['shared_grantees'])} shared, Jaccard: {top_jaccard:.2f}). "
                f"Teams use moderate overlap to decide whether checking for duplication or alignment is warranted."
            )
            opportunity_text = "💡 **Decision Option:** Moderate overlap is often used to assess whether lightweight coordination (shared learning, grant timing) would add value."
        else:
            twin_emoji, twin_label = "⚪", "Limited overlap"
            twin_narrative = (
                f"**{len(overlap_df)} funder pairs** share at least one grantee, but similarity is low. "
                f"Even the closest pair ({node_labels.get(top['funder_1'], '')} & {node_labels.get(top['funder_2'], '')}) "
                f"has Jaccard of {top_jaccard:.2f}. Portfolios are largely distinct."
            )
            opportunity_text = "💡 **Context:** Low overlap doesn't mean coordination isn't valuable — it means portfolios are largely distinct. Shared touchpoints may still anchor lightweight coordination."
        
        ranked_rows = []
        for rank, (_, r) in enumerate(overlap_df.head(5).iterrows(), 1):
            f1, f2 = node_labels.get(r['funder_1'], r['funder_1']), node_labels.get(r['funder_2'], r['funder_2'])
            shared = int(r['shared_grantees'])
            jaccard = r['jaccard_similarity']
            p1 = r.get('funder_1_portfolio', 0)
            p2 = r.get('funder_2_portfolio', 0)
            
            # Tier-consistent per-pair narrative
            if jaccard >= 0.15:
                pair_desc = f"High overlap — worth a coordination conversation."
            elif jaccard >= 0.05:
                pair_desc = f"Moderate overlap — a potential touchpoint."
            else:
                pair_desc = f"Shared touchpoints, but portfolios largely distinct."
            
            # Include portfolio context if available
            if p1 and p2:
                narrative = f"Share **{shared} grantees** ({shared} of {p1} vs {p2}, Jaccard: {jaccard:.2f}) — {pair_desc}"
            else:
                narrative = f"Share **{shared} grantees** (Jaccard: {jaccard:.2f}) — {pair_desc}"
            
            ranked_rows.append({
                "rank": rank,
                "pair": f"{f1} ↔ {f2}",
                "shared": shared,
                "jaccard": round(jaccard, 2),
                "narrative": narrative
            })
        
        cards.append({
            "card_id": "portfolio_twins",
            "use_case": "Funding Concentration",
            "title": "Portfolio Twins",
            "summary": f"{twin_emoji} **{twin_label}**\n\n{twin_narrative}\n\n{opportunity_text}",
            "ranked_rows": ranked_rows,
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # =========================================================================
    # Card 5: Board Conduits
    # =========================================================================
    multi_board = person_metrics[person_metrics["boards_served"] >= 2].sort_values("boards_served", ascending=False)
    
    if not multi_board.empty and not board_edges.empty:
        person_orgs = board_edges.groupby("from_id")["to_id"].apply(list).to_dict()
        
        board_narrative = (
            f"**{len(multi_board)} individuals** serve on 2+ boards, creating direct governance links between organizations. "
            f"These 'board conduits' enable informal coordination, information sharing, and relationship-building "
            f"that formal structures often miss."
        )
        
        ranked_rows = []
        for rank, (_, row) in enumerate(multi_board.head(5).iterrows(), 1):
            org_ids = person_orgs.get(row["node_id"], [])
            org_lbls = [node_labels.get(o, o) for o in org_ids]
            blurb, rec = describe_board_connector(row["label"], int(row["boards_served"]), org_lbls)
            ranked_rows.append({
                "rank": rank, 
                "person": row["label"], 
                "boards": int(row["boards_served"]),
                "organizations": org_lbls,
                "narrative": blurb, 
                "recommendation": rec
            })
        
        cards.append({
            "card_id": "shared_board_conduits",
            "use_case": "Board Network & Conduits",
            "title": "Shared Board Conduits",
            "summary": f"🔗 **{len(multi_board)} governance connectors identified**\n\n{board_narrative}",
            "ranked_rows": ranked_rows,
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    else:
        cards.append({
            "card_id": "shared_board_conduits",
            "use_case": "Board Network & Conduits",
            "title": "Shared Board Conduits",
            "summary": "⚪ **No multi-board individuals detected**\n\nNo one serves on multiple boards in this network. Governance structures are fully separate — a potential gap for coordination.",
            "ranked_rows": [],
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # =========================================================================
    # Card 6: Isolated Foundations
    # =========================================================================
    disconnected = foundations[foundations["shared_board_count"] == 0]
    
    if total_funders > 0:
        disc_pct = n_isolated_funders / total_funders * 100
        
        if disc_pct >= 80:
            disc_emoji, disc_label = "🔴", "Governance silos"
            disc_narrative = (
                f"**{n_isolated_funders} of {total_funders} funders** ({disc_pct:.0f}%) have no shared board members "
                f"with other network foundations. This limits informal coordination channels and peer learning."
            )
        elif disc_pct >= 50:
            disc_emoji, disc_label = "🟡", "Mixed governance ties"
            disc_narrative = (
                f"**{n_isolated_funders} funders** ({disc_pct:.0f}%) operate without board interlocks. "
                f"Some governance bridges exist, but many funders remain structurally isolated."
            )
        else:
            disc_emoji, disc_label = "🟢", "Connected governance"
            disc_narrative = f"Most funders share board ties. Only {n_isolated_funders} ({disc_pct:.0f}%) are isolated."
        
        ranked_rows = []
        for i, (_, r) in enumerate(disconnected.sort_values("grant_outflow_total", ascending=False).head(5).iterrows()):
            ranked_rows.append({
                "rank": i+1, 
                "funder": r["label"], 
                "outflow": f"${r['grant_outflow_total']:,.0f}",
                "narrative": f"Distributes ${r['grant_outflow_total']:,.0f} with no governance ties to other network funders."
            })
        
        cards.append({
            "card_id": "no_board_interlocks",
            "use_case": "Board Network & Conduits",
            "title": "Foundations with No Board Interlocks",
            "summary": f"{disc_emoji} **{disc_label}**\n\n{disc_narrative}\n\n💡 **Decision Option:** Teams sometimes use governance isolation data to decide whether introductions between funders with aligned portfolios would add value.",
            "ranked_rows": ranked_rows,
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # =========================================================================
    # Card 7: Hidden Brokers
    # =========================================================================
    hidden = metrics_df[(metrics_df["is_hidden_broker"] == 1) & (metrics_df["node_type"] == "ORG")]
    
    if not hidden.empty:
        broker_narrative = (
            f"**{len(hidden)} organizations** have high betweenness centrality but low visibility — they quietly "
            f"bridge otherwise disconnected parts of the network. These 'hidden brokers' often go unrecognized "
            f"but play critical structural roles in enabling coordination."
        )
        
        ranked_rows = []
        for i, (_, r) in enumerate(hidden.sort_values("betweenness", ascending=False).head(5).iterrows()):
            grantee_count = len(grant_edges[grant_edges["from_id"] == r["node_id"]]) if not grant_edges.empty else 0
            blurb, rec = describe_funder_with_recommendation(
                r["label"],
                r["grant_outflow_total"] or 0,
                grantee_count,
                int(r["shared_board_count"] or 0),
                r["betweenness"],
                is_hidden_broker=True,
                is_capital_hub=bool(r.get("is_capital_hub", 0)),
            )
            ranked_rows.append({
                "rank": i+1, 
                "org": r["label"], 
                "betweenness": round(r["betweenness"], 4),
                "narrative": blurb,
                "recommendation": rec
            })
        
        cards.append({
            "card_id": "hidden_brokers",
            "use_case": "Brokerage Roles",
            "title": "Hidden Brokers",
            "summary": f"🔍 **{len(hidden)} hidden brokers identified**\n\n{broker_narrative}",
            "ranked_rows": ranked_rows,
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    else:
        cards.append({
            "card_id": "hidden_brokers",
            "use_case": "Brokerage Roles",
            "title": "Hidden Brokers",
            "summary": "⚪ **No hidden brokers detected**\n\nAll high-betweenness nodes are also highly visible. No quiet bridges exist in this network.",
            "ranked_rows": [],
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # =========================================================================
    # Card 8: Single-Point Bridges
    # =========================================================================
    if grant_graph.number_of_edges() > 0:
        # FIX: Only compute articulation points within the largest connected component
        # Otherwise small peripheral components can appear as "critical bridges"
        largest_cc = max(nx.connected_components(grant_graph), key=len)
        largest_subgraph = grant_graph.subgraph(largest_cc).copy()
        
        ap = list(nx.articulation_points(largest_subgraph))
        ap_in_network = [a for a in ap if a in metrics_df["node_id"].values]
        
        if ap_in_network:
            # Compute impact of removing each articulation point
            bridge_impacts = []
            for a in ap_in_network:
                G_temp = largest_subgraph.copy()
                G_temp.remove_node(a)
                components = list(nx.connected_components(G_temp))
                # Count nodes that would be disconnected from the main component
                if len(components) > 1:
                    largest_remaining = max(len(c) for c in components)
                    isolated_count = len(largest_cc) - 1 - largest_remaining  # -1 for removed node
                else:
                    isolated_count = 0
                neighbor_count = largest_subgraph.degree(a)
                bridge_impacts.append({
                    "node_id": a,
                    "isolated_nodes": isolated_count,
                    "component_count": len(components),
                    "neighbor_count": neighbor_count
                })
            
            # Sort by impact (isolated nodes descending)
            bridge_impacts.sort(key=lambda x: -x["isolated_nodes"])
            
            bridge_narrative = (
                f"**{len(ap_in_network)} nodes** are critical bridges — removing any one would fragment the network "
                f"into disconnected pieces. These are structural vulnerabilities but also high-leverage positions."
            )
            
            ranked_rows = []
            for i, impact in enumerate(bridge_impacts[:5]):
                a = impact["node_id"]
                row = metrics_df[metrics_df["node_id"] == a].iloc[0]
                node_type = row["node_type"]
                if node_type == "ORG" and (row.get("grant_outflow_total") or 0) > 0:
                    role_desc = f"Funder (${row['grant_outflow_total']:,.0f})"
                elif node_type == "ORG":
                    role_desc = "Grantee connecting funders"
                else:
                    role_desc = f"Person on {int(row.get('boards_served', 0))} boards"
                
                impact_desc = f"Would isolate {impact['isolated_nodes']} nodes across {impact['component_count']} component(s)"
                
                ranked_rows.append({
                    "rank": i+1, 
                    "node": node_labels.get(a, a),
                    "type": node_type,
                    "role": role_desc,
                    "impact": impact_desc,
                    "neighbor_count": impact["neighbor_count"],
                    "narrative": f"Removing {node_labels.get(a, a)} would split the network. {impact_desc}."
                })
            
            cards.append({
                "card_id": "single_point_bridges",
                "use_case": "Brokerage Roles",
                "title": "Single-Point Bridges",
                "summary": f"⚠️ **{len(ap_in_network)} critical bridges**\n\n{bridge_narrative}\n\n💡 **Risk Mitigation:** Build redundant pathways around critical bridges to improve resilience.",
                "ranked_rows": ranked_rows,
                "evidence": {"node_ids": ap_in_network[:10], "edge_ids": []},
            })
        else:
            cards.append({
                "card_id": "single_point_bridges",
                "use_case": "Brokerage Roles",
                "title": "Single-Point Bridges",
                "summary": "🟢 **No single points of failure**\n\nThe network has redundant pathways — no single node's removal would fragment it.",
                "ranked_rows": [],
                "evidence": {"node_ids": [], "edge_ids": []},
            })
    
    # =========================================================================
    # Card 9: Strategic Recommendations
    # =========================================================================
    recommendations_md = generate_strategic_recommendations(
        health_score=health_score,
        health_label=health_label,
        flow_stats=flow_stats,
        multi_funder_pct=multi_funder_pct,
        largest_component_pct=largest_cc_pct,
        n_isolated_funders=n_isolated_funders,
        total_funders=total_funders,
        n_hidden_brokers=n_hidden_brokers,
        n_board_conduits=n_board_conduits,
    )
    
    cards.append({
        "card_id": "decision_options",
        "use_case": "Decision Options",
        "title": "Decision Options",
        "summary": recommendations_md,
        "ranked_rows": [],
        "evidence": {"node_ids": [], "edge_ids": []},
    })
    
    return {
        "schema_version": "1.0-mvp",
        "project_id": project_id,
        "generated_at": datetime.now().isoformat() + "Z",
        "health": {"score": health_score, "label": health_label, "positive": positive_factors, "risk": risk_factors},
        "cards": cards,
    }


# =============================================================================
# Roles × Region Lens
# =============================================================================

# Canonical role vocabulary (must match OrgGraph exports)
ROLE_VOCABULARY = {
    'FUNDER':         {'label': 'Funder',            'order': 1},
    'FUNDER_GRANTEE': {'label': 'Funder + Grantee',  'order': 2},
    'GRANTEE':        {'label': 'Grantee',           'order': 3},
    'ORGANIZATION':   {'label': 'Organization',      'order': 4},
    'BOARD_MEMBER':   {'label': 'Board Member',      'order': 5},
    'INDIVIDUAL':     {'label': 'Individual',        'order': 6},
}

# =============================================================================
# Decision Lens Content (per C4C Report Authoring Guide v1.0)
# =============================================================================
# Each section includes:
#   - what_tells_you: interpretive frame
#   - why_matters: decision context  
#   - teams_do_next: action guidance (descriptive, not prescriptive)
#   - not_over_interpret: misinterpretation guardrail (prevents client over-reaction)
#   - signal_intensity: low / medium / high (governs reader attention)
#     - low: primarily confirmatory / contextual
#     - medium: worth discussion or light exploration  
#     - high: merits active follow-up or strategy review

DECISION_LENS = {
    "network_health": {
        "what_tells_you": "This score reflects structural connectivity and coordination capacity — not effectiveness or impact.",
        "why_matters": "When coordination infrastructure is weak, new funding or initiatives often underperform. This signal helps prioritize where to invest in connective tissue before changing grant strategy.",
        "teams_do_next": "High scores suggest existing infrastructure to leverage. Low scores suggest investing in convening, shared learning, or backbone capacity before expecting coordination to emerge.",
        "not_over_interpret": "Health scores reflect structure, not performance. A low score doesn't mean the network is failing — it means coordination requires intentional effort.",
        "signal_intensity": "medium",
    },
    "roles_region_lens": {
        "what_tells_you": "This shows how organizations are distributed relative to the defined regional lens, revealing alignment between funding sources and place-based impact.",
        "why_matters": "Regional strategies depend on understanding which actors operate within vs. outside the target geography. Misalignment between funder location and grantee location can affect accountability and coordination.",
        "teams_do_next": "Use this to clarify which actors are 'in scope' for regional coordination and which require different engagement strategies.",
        "not_over_interpret": "Out-of-lens funders are not problems — many effective funders operate nationally. This lens shows geographic distribution, not quality.",
        "signal_intensity": "low",
    },
    "funding_concentration": {
        "what_tells_you": "Grant concentration shows how evenly or unevenly funding is distributed across organizations. It helps assess system resilience and exposure to single-funder risk.",
        "why_matters": "High concentration increases fragility — if key funders shift priorities, dependent organizations are exposed. Moderate concentration may reflect intentional focus.",
        "teams_do_next": "High concentration → assess dependency risk and succession planning. Moderate → check if specialization is intentional. Low → portfolios are distributed; resilience may be higher.",
        "not_over_interpret": "Concentration does not imply inefficiency or favoritism. Some issue areas require focused funding by design.",
        "signal_intensity": "medium",
    },
    "multi_funder_grantees": {
        "what_tells_you": "This identifies where multiple funders already support the same organizations — revealing latent alignment even without formal coordination.",
        "why_matters": "Shared grantees represent the lowest-friction entry points for funder coordination. These are places where alignment already exists organically.",
        "teams_do_next": "Dense clusters → opportunities for shared learning or coordination. Sparse overlap → funders operate independently (which may be intentional).",
        "not_over_interpret": "Low overlap doesn't indicate misalignment. Many funders intentionally differentiate portfolios to maximize collective coverage.",
        "signal_intensity": "medium",
    },
    "portfolio_twins": {
        "what_tells_you": "Portfolio overlap signals identify shared grantee touchpoints, not necessarily aligned strategies. Most funder pairs show limited overall similarity even when they fund some of the same organizations.",
        "why_matters": "This helps answer a practical question: Where might coordination be worth exploring — and where is it unlikely to add value?",
        "teams_do_next": "High similarity → review duplication, co-funding, or shared learning. Moderate overlap → consider light coordination (timing, convenings). Low overlap → no action required; portfolios are complementary.",
        "not_over_interpret": "Low similarity does not imply misalignment or inefficiency. Shared grantees do not imply redundant strategies. In most regions, distinct portfolios reflect healthy diversity.",
        "signal_intensity": "low",
    },
    "hidden_brokers": {
        "what_tells_you": "Brokers are organizations that connect otherwise separate parts of the network. They often enable coordination, information flow, and alignment across domains.",
        "why_matters": "High-brokerage actors are critical for coordination and knowledge transfer. Few brokers overall creates fragmentation risk if those actors disengage.",
        "teams_do_next": "Identify whether key brokers are aware of their structural role. Consider engagement, support, or risk mitigation for these connective organizations.",
        "not_over_interpret": "Brokerage is a structural role, not a value judgment. Brokers are not inherently leaders or decision-makers. Peripheral organizations may be highly impactful in niche roles.",
        "signal_intensity": "medium",
    },
    "single_point_bridges": {
        "what_tells_you": "Some connections between network components rely on only one organization or relationship. These create structural fragility.",
        "why_matters": "Single-point bridges are not necessarily problems, but they represent risk. If the bridging actor disengages, entire parts of the network may disconnect.",
        "teams_do_next": "Assess whether bridge actors are stable and well-supported. Consider whether redundancy, diversification, or intentional cross-connection is needed.",
        "not_over_interpret": "Bridges are not failures — they often reflect natural network structure. The question is whether the risk is understood and managed.",
        "signal_intensity": "medium",
    },
    "shared_board_conduits": {
        "what_tells_you": "Shared board memberships create informal pathways for coordination and influence across the network.",
        "why_matters": "In mature networks, informal governance ties are often how alignment happens without formal coordination structures. These are relationship-based coordination channels.",
        "teams_do_next": "Consider whether these connectors are aware of their bridging role and could be engaged more intentionally for network-wide coordination.",
        "not_over_interpret": "Board overlaps indicate potential for coordination, not actual coordination. Shared governance doesn't guarantee aligned strategies.",
        "signal_intensity": "low",
    },
    "shared_board_conduits_empty": {
        "what_tells_you": "There are few or no informal governance ties connecting organizations across the network.",
        "why_matters": "In the absence of organic governance ties, coordination will not emerge naturally. Alignment efforts will need to be intentional rather than emergent.",
        "teams_do_next": "Consider formal convenings, intermediaries, or governance experiments to create the connective tissue that doesn't currently exist organically.",
        "not_over_interpret": "Lack of board overlaps is common and not inherently problematic. Many effective networks coordinate through other mechanisms.",
        "signal_intensity": "low",
    },
    "isolated_funders": {
        "what_tells_you": "These funders have no shared board members with other network foundations, limiting informal coordination pathways.",
        "why_matters": "Without informal governance connections, peer learning and organic coordination are structurally unlikely. This doesn't mean coordination is impossible — just that it requires more intentional effort.",
        "teams_do_next": "Consider whether introductions, joint convenings, or shared initiatives could create connective tissue. Prioritize funders with aligned portfolios.",
        "not_over_interpret": "Governance isolation is common, especially for national funders or those with different geographic focus. It indicates structural distance, not misalignment.",
        "signal_intensity": "low",
    },
}

# Signal Intensity labels (for rendering)
SIGNAL_INTENSITY_LABELS = {
    "low": "Low-intensity signal",
    "medium": "Moderate-intensity signal",
    "high": "High-intensity signal",
}

# Section intro text (human-readable, accessible)
# These answer "what does this show AND what doesn't it show"
SECTION_INTROS = {
    "network_health": "This section provides an overall assessment of network connectivity and coordination capacity — revealing whether the structural foundations exist for effective collaboration, not whether the network is succeeding.",
    "roles_region_lens": "This section examines how organizations are distributed relative to the defined regional focus, showing alignment between where funding originates and where impact occurs.",
    "funding_concentration": "This section examines how concentrated or diversified funding relationships are across the network, revealing whether influence and risk are broadly shared or narrowly held. It assesses resilience, not quality.",
    "multi_funder_grantees": "This section identifies grantees supported by multiple funders, revealing where informal alignment already exists. Shared grantees indicate organic overlap, not necessarily intentional coordination.",
    "portfolio_twins": "This analysis highlights shared grantee touchpoints between funders — not necessarily aligned strategies. In most regions, funder portfolios are intentionally distinct, and even pairs that fund some of the same organizations often differ substantially in scale, focus, or approach.",
    "hidden_brokers": "This section identifies organizations that connect otherwise disconnected parts of the network. These 'brokers' play structural roles in enabling coordination — but brokerage is a position, not a performance measure.",
    "single_point_bridges": "This section identifies structural vulnerabilities where removing a single node would fragment the network. Bridges are not failures — they're features that may require monitoring.",
    "shared_board_conduits": "This section examines whether shared board memberships create informal pathways for coordination. Board overlaps indicate potential channels, not guaranteed alignment.",
    "isolated_funders": "This section identifies funders with no governance connections to other network foundations. Isolation is structural, not a judgment — many effective funders operate independently.",
}

# Section subtitles (speed bumps for skimmers)
# These are short qualifiers rendered in lighter text directly under the section header
SECTION_SUBTITLES = {
    "portfolio_twins": "Most funder pairs show limited overlap; this section highlights where shared touchpoints exist.",
}

# Default region lens for GLFN (can be overridden by project_config.json)
DEFAULT_REGION_LENS = {
    "enabled": True,
    "label": "Great Lakes (Binational)",
    "mode": "preset",
    "boundaries": {
        "us_states": ["MI", "OH", "MN", "WI", "IN", "IL", "NY", "PA"],
        "ca_provinces": ["ON", "QC"]
    }
}


def load_region_lens_config(project_dir: Path) -> dict:
    """
    Load region lens configuration from project_config.json.
    Falls back to DEFAULT_REGION_LENS if not found.
    """
    config_path = project_dir / "project_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("region_lens", DEFAULT_REGION_LENS)
        except Exception:
            pass
    return DEFAULT_REGION_LENS


def derive_network_roles(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive network role for each node if not already present.
    
    If nodes_df already has 'network_role_code', uses that.
    Otherwise derives from edge relationships.
    
    Returns nodes_df with network_role_code, network_role_label, network_role_order columns.
    """
    # Check if already present
    if 'network_role_code' in nodes_df.columns:
        # Ensure label and order columns exist
        if 'network_role_label' not in nodes_df.columns:
            nodes_df['network_role_label'] = nodes_df['network_role_code'].map(
                lambda c: ROLE_VOCABULARY.get(c, {}).get('label', c)
            )
        if 'network_role_order' not in nodes_df.columns:
            nodes_df['network_role_order'] = nodes_df['network_role_code'].map(
                lambda c: ROLE_VOCABULARY.get(c, {}).get('order', 99)
            )
        return nodes_df
    
    # Derive from edges
    grant_edges = edges_df[edges_df['edge_type'] == 'GRANT'] if not edges_df.empty else pd.DataFrame()
    board_edges = edges_df[edges_df['edge_type'] == 'BOARD_MEMBERSHIP'] if not edges_df.empty else pd.DataFrame()
    
    funder_ids = set(grant_edges['from_id']) if not grant_edges.empty else set()
    grantee_ids = set(grant_edges['to_id']) if not grant_edges.empty else set()
    board_member_ids = set(board_edges['from_id']) if not board_edges.empty else set()
    
    def get_role(row):
        node_id = row['node_id']
        node_type = row.get('node_type', '')
        
        if node_type == 'PERSON':
            code = 'BOARD_MEMBER' if node_id in board_member_ids else 'INDIVIDUAL'
        else:
            is_funder = node_id in funder_ids
            is_grantee = node_id in grantee_ids
            
            if is_funder and is_grantee:
                code = 'FUNDER_GRANTEE'
            elif is_funder:
                code = 'FUNDER'
            elif is_grantee:
                code = 'GRANTEE'
            else:
                code = 'ORGANIZATION'
        
        return pd.Series({
            'network_role_code': code,
            'network_role_label': ROLE_VOCABULARY[code]['label'],
            'network_role_order': ROLE_VOCABULARY[code]['order']
        })
    
    role_cols = nodes_df.apply(get_role, axis=1)
    nodes_df = pd.concat([nodes_df, role_cols], axis=1)
    
    return nodes_df


def compute_region_lens_membership(nodes_df: pd.DataFrame, lens_config: dict) -> pd.DataFrame:
    """
    Compute in_region_lens for each node based on lens boundaries.
    
    Returns nodes_df with:
    - in_region_lens: bool
    - region_lens_label: str (same for all nodes)
    """
    if not lens_config.get('enabled', False):
        nodes_df['in_region_lens'] = True  # All nodes in-lens if disabled
        nodes_df['region_lens_label'] = 'All Regions'
        return nodes_df
    
    boundaries = lens_config.get('boundaries', {})
    us_states = set(boundaries.get('us_states', []))
    ca_provinces = set(boundaries.get('ca_provinces', []))
    all_regions = us_states | ca_provinces
    
    lens_label = lens_config.get('label', 'Custom Region')
    
    def is_in_lens(row):
        # Funders define the network - they are ALWAYS "in lens" by definition
        # The geographic lens is about where grantees are located, not funders
        role = str(row.get('network_role_code', '')).upper()
        if role in ('FUNDER', 'FUNDER_GRANTEE'):
            return True
        
        # For other roles, check geography
        state_val = str(row.get('state', '')).strip().upper()
        region_val = str(row.get('region', '')).strip().upper()
        
        # Use state if available, otherwise region
        location = state_val if state_val else region_val
        
        # Direct match (2-letter codes)
        if location and location in all_regions:
            return True
        
        # Check full names (Ontario -> ON, etc.)
        if location:
            region_map = {
                'ONTARIO': 'ON', 'QUEBEC': 'QC', 'MICHIGAN': 'MI', 'OHIO': 'OH',
                'MINNESOTA': 'MN', 'WISCONSIN': 'WI', 'INDIANA': 'IN', 'ILLINOIS': 'IL',
                'NEW YORK': 'NY', 'PENNSYLVANIA': 'PA'
            }
            mapped = region_map.get(location, location)
            if mapped in all_regions:
                return True
        
        return False
    
    nodes_df['in_region_lens'] = nodes_df.apply(is_in_lens, axis=1)
    nodes_df['region_lens_label'] = lens_label
    
    return nodes_df


def compute_roles_by_lens(nodes_df: pd.DataFrame) -> dict:
    """
    Compute role counts by lens membership.
    
    Returns:
        {
            'FUNDER': {'in': 15, 'out': 5, 'pct_in': 75.0},
            'GRANTEE': {'in': 2000, 'out': 899, 'pct_in': 69.0},
            ...
        }
    """
    if 'network_role_code' not in nodes_df.columns or 'in_region_lens' not in nodes_df.columns:
        return {}
    
    result = {}
    for code in ROLE_VOCABULARY.keys():
        role_nodes = nodes_df[nodes_df['network_role_code'] == code]
        in_count = len(role_nodes[role_nodes['in_region_lens'] == True])
        out_count = len(role_nodes[role_nodes['in_region_lens'] == False])
        total = in_count + out_count
        
        result[code] = {
            'in': in_count,
            'out': out_count,
            'total': total,
            'pct_in': (in_count / total * 100) if total > 0 else 0,
            'label': ROLE_VOCABULARY[code]['label'],
            'order': ROLE_VOCABULARY[code]['order']
        }
    
    return result


def compute_edge_flows_by_lens(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> dict:
    """
    Compute grant edge flows by lens category.
    
    Categories:
    - IN_IN: Both funder and grantee in-lens
    - IN_OUT: Funder in-lens, grantee out-of-lens
    - OUT_IN: Funder out-of-lens, grantee in-lens
    - OUT_OUT: Both out-of-lens
    
    Returns:
        {
            'IN_IN': {'count': 3000, 'amount': 400000000},
            'IN_OUT': {'count': 500, 'amount': 50000000},
            ...
        }
    """
    if 'in_region_lens' not in nodes_df.columns:
        return {}
    
    grant_edges = edges_df[edges_df['edge_type'] == 'GRANT'].copy() if not edges_df.empty else pd.DataFrame()
    
    if grant_edges.empty:
        return {}
    
    # Build node_id -> in_lens lookup
    node_lens = dict(zip(nodes_df['node_id'], nodes_df['in_region_lens']))
    
    # Classify each edge
    def classify_edge(row):
        from_in = node_lens.get(row['from_id'], False)
        to_in = node_lens.get(row['to_id'], False)
        
        if from_in and to_in:
            return 'IN_IN'
        elif from_in and not to_in:
            return 'IN_OUT'
        elif not from_in and to_in:
            return 'OUT_IN'
        else:
            return 'OUT_OUT'
    
    grant_edges['flow_category'] = grant_edges.apply(classify_edge, axis=1)
    
    # Parse amounts
    if 'amount' in grant_edges.columns:
        grant_edges['amount_num'] = pd.to_numeric(grant_edges['amount'], errors='coerce').fillna(0)
    else:
        grant_edges['amount_num'] = 0
    
    # Aggregate
    result = {}
    for cat in ['IN_IN', 'IN_OUT', 'OUT_IN', 'OUT_OUT']:
        cat_edges = grant_edges[grant_edges['flow_category'] == cat]
        result[cat] = {
            'count': len(cat_edges),
            'amount': cat_edges['amount_num'].sum(),
            'label': cat.replace('_', '→')
        }
    
    return result


def generate_roles_region_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, lens_config: dict) -> dict:
    """
    Generate complete Roles × Region Lens summary.
    
    Returns a dict suitable for adding to insight_cards.
    """
    if not lens_config.get('enabled', False):
        return {'enabled': False}
    
    # Ensure roles and lens membership are computed
    nodes_df = derive_network_roles(nodes_df, edges_df)
    nodes_df = compute_region_lens_membership(nodes_df, lens_config)
    
    # Compute metrics
    roles_by_lens = compute_roles_by_lens(nodes_df)
    edge_flows = compute_edge_flows_by_lens(edges_df, nodes_df)
    
    # Overall stats
    total_nodes = len(nodes_df)
    in_lens_count = len(nodes_df[nodes_df['in_region_lens'] == True])
    out_lens_count = total_nodes - in_lens_count
    pct_in = (in_lens_count / total_nodes * 100) if total_nodes > 0 else 0
    
    return {
        'enabled': True,
        'lens_label': lens_config.get('label', 'Custom Region'),
        'totals': {
            'in_lens': in_lens_count,
            'out_lens': out_lens_count,
            'total': total_nodes,
            'pct_in': pct_in
        },
        'roles_by_lens': roles_by_lens,
        'edge_flows': edge_flows
    }


def format_roles_region_section(summary: dict, skip_header: bool = False) -> list:
    """
    Format Roles × Region Lens summary as markdown lines.
    
    Args:
        summary: Region lens summary dict
        skip_header: If True, skip the section header (used when header is already rendered)
    """
    if not summary.get('enabled', False):
        return []
    
    lines = []
    
    if not skip_header:
        lines.append("## 🗺️ Roles × Region Lens")
        lines.append("")
    
    lines.append(f"**Lens:** {summary.get('lens_label', 'Unknown')}")
    lines.append("")
    
    # Overall totals
    totals = summary.get('totals', {})
    in_count = totals.get('in_lens', 0)
    out_count = totals.get('out_lens', 0)
    pct_in = totals.get('pct_in', 0)
    
    lines.append(f"- **In-lens nodes:** {in_count:,} ({pct_in:.1f}%)")
    lines.append(f"- **Out-of-lens nodes:** {out_count:,} ({100 - pct_in:.1f}%)")
    lines.append("")
    
    # Roles breakdown (sorted by order, only show non-empty)
    roles = summary.get('roles_by_lens', {})
    sorted_roles = sorted(roles.items(), key=lambda x: x[1].get('order', 99))
    
    lines.append("### By Network Role")
    lines.append("")
    lines.append("| Role | In-Lens | Out-of-Lens | % In-Lens |")
    lines.append("|------|---------|-------------|-----------|")
    
    for code, data in sorted_roles:
        if data.get('total', 0) > 0:
            label = data.get('label', code)
            in_n = data.get('in', 0)
            out_n = data.get('out', 0)
            pct = data.get('pct_in', 0)
            lines.append(f"| {label} | {in_n:,} | {out_n:,} | {pct:.1f}% |")
    
    lines.append("")
    
    # Edge flows
    flows = summary.get('edge_flows', {})
    if flows:
        lines.append("### Grant Flows by Lens Category")
        lines.append("")
        lines.append("| Flow | Grants | Amount |")
        lines.append("|------|--------|--------|")
        
        for cat in ['IN_IN', 'IN_OUT', 'OUT_IN', 'OUT_OUT']:
            if cat in flows:
                data = flows[cat]
                label = data.get('label', cat)
                count = data.get('count', 0)
                amount = data.get('amount', 0)
                if count > 0:
                    lines.append(f"| {label} | {count:,} | ${amount:,.0f} |")
        
        lines.append("")
        
        # Interpretation
        in_in = flows.get('IN_IN', {}).get('amount', 0)
        in_out = flows.get('IN_OUT', {}).get('amount', 0)
        out_in = flows.get('OUT_IN', {}).get('amount', 0)
        out_out = flows.get('OUT_OUT', {}).get('amount', 0)
        total_flow = in_in + in_out + out_in + out_out
        
        if total_flow > 0:
            in_in_pct = in_in / total_flow * 100
            in_out_pct = in_out / total_flow * 100
            out_in_pct = out_in / total_flow * 100
            
            if in_in_pct >= 80:
                lines.append(f"> **Interpretation:** The network is highly regional — {in_in_pct:.0f}% of funding stays within the lens boundaries.")
            elif out_in_pct >= 80:
                lines.append(f"> **Interpretation:** External funding into the region — {out_in_pct:.0f}% of funding comes from out-of-lens funders to in-lens grantees.")
            elif in_out_pct >= 20:
                lines.append(f"> **Interpretation:** Significant outflow — {in_out_pct:.0f}% of funding from in-lens funders goes to out-of-lens grantees.")
            else:
                lines.append(f"> **Interpretation:** Mixed flows — funding crosses lens boundaries in multiple directions.")
            lines.append("")
    
    # Disclaimer
    lines.append("---")
    lines.append("")
    lines.append("*Region lens is defined at project setup (client-defined scope). It is not automatic geocoding.*")
    lines.append("")
    
    return lines


# =============================================================================
# Markdown Report Generator
# =============================================================================

def generate_markdown_report(insight_cards: dict, project_summary: dict, project_id: str = "glfn", roles_region_summary: dict = None) -> str:
    """
    Generate a complete markdown report from insight cards.
    
    Structure follows C4C Report Authoring Guide v1.0:
    - Executive Summary after header
    - Human-readable section intros
    - Decision Lens blocks for each section
    
    Returns formatted markdown string.
    """
    lines = []
    
    # Header
    lines.append(f"# Network Insight Report")
    lines.append(f"**Project:** {project_id.upper()}")
    lines.append(f"**Generated:** {insight_cards.get('generated_at', 'Unknown')[:10]}")
    lines.append("")
    
    # Summary stats
    summary = project_summary
    nodes = summary.get("node_counts", {})
    edges = summary.get("edge_counts", {})
    funding = summary.get("funding", {})
    
    lines.append(f"**Nodes:** {nodes.get('total', 0)} ({nodes.get('organizations', 0)} organizations, {nodes.get('people', 0)} people)")
    lines.append(f"**Edges:** {edges.get('total', 0)} ({edges.get('grants', 0)} grants, {edges.get('board_memberships', 0)} board memberships)")
    lines.append(f"**Total Funding:** ${funding.get('total_amount', 0):,.0f}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # ==========================================================================
    # EXECUTIVE SUMMARY (new)
    # ==========================================================================
    health = insight_cards.get("health", {})
    health_score = health.get("score", 0)
    health_label = health.get("label", "Unknown")
    positive_factors = health.get("positive", [])
    risk_factors = health.get("risk", [])
    
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report provides a structural view of the funding network, focusing on how resources, relationships, and influence are distributed across funders and grantees.")
    lines.append("")
    lines.append("Rather than prescribing actions, the analysis surfaces decision-relevant signals that help teams decide where coordination, investment, or deeper inquiry may be most valuable.")
    lines.append("")
    lines.append("**Key signals:**")
    lines.append("")
    
    # Dynamic key signals based on actual data
    lines.append(f"- Overall network health is **{health_label.lower()}** ({health_score}/100), indicating {'strong' if health_score >= 70 else 'moderate' if health_score >= 40 else 'limited'} coordination capacity.")
    
    # Add signals from positive/risk factors
    if positive_factors:
        # Extract a key positive signal
        lines.append(f"- {positive_factors[0].replace('🟢 ', '').replace('**', '')}")
    if risk_factors:
        # Extract a key risk signal
        lines.append(f"- {risk_factors[0].replace('🔴 ', '').replace('**', '')}")
    
    # Add funding concentration signal
    top_5_share = funding.get("top_5_share", 0)
    if top_5_share:
        lines.append(f"- Funding flows show {'significant' if top_5_share > 80 else 'moderate' if top_5_share > 60 else 'distributed'} concentration (top 5 funders control {top_5_share:.0f}%).")
    
    # Add governance signal
    governance = summary.get("governance", {})
    multi_board = governance.get("multi_board_people", 0)
    if multi_board == 0:
        lines.append("- Informal governance pathways (shared board memberships) are minimal, suggesting coordination will need to be intentional.")
    else:
        lines.append(f"- {multi_board} individuals serve on multiple boards, creating informal coordination pathways.")
    
    lines.append("")
    lines.append("Many organizations use this report as a first-pass diagnostic, followed by facilitated interpretation, peer conversations, or deeper scenario design.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # ==========================================================================
    # NETWORK HEALTH SECTION
    # ==========================================================================
    if health_score >= 70:
        health_emoji = "🟢"
    elif health_score >= 40:
        health_emoji = "🟡"
    else:
        health_emoji = "🔴"
    
    lines.append(f"## {health_emoji} Network Health: {health_score}/100 ({health_label})")
    lines.append("")
    
    # Section intro
    lines.append(SECTION_INTROS.get("network_health", ""))
    lines.append("")
    
    # Decision Lens block
    lens = DECISION_LENS.get("network_health", {})
    if lens:
        intensity = lens.get("signal_intensity", "medium")
        intensity_label = SIGNAL_INTENSITY_LABELS.get(intensity, "Signal")
        lines.append(f':::decision-lens intensity="{intensity}"')
        lines.append(f"**What this tells you**")
        lines.append(lens.get("what_tells_you", ""))
        lines.append("")
        lines.append(f"**Why it matters for decisions**")
        lines.append(lens.get("why_matters", ""))
        lines.append("")
        lines.append(f"**What teams often do next**")
        lines.append(lens.get("teams_do_next", ""))
        if lens.get("not_over_interpret"):
            lines.append("")
            lines.append(f"**What not to over-interpret**")
            lines.append(lens.get("not_over_interpret", ""))
        lines.append(":::")
        lines.append("")
    
    # Health indicators from the network_health card
    cards = insight_cards.get("cards", [])
    health_card = next((c for c in cards if c.get("card_id") == "network_health"), None)
    if health_card:
        indicators = health_card.get("ranked_rows", [])
        for row in indicators:
            indicator = row.get("indicator", "")
            value = row.get("value", "")
            interpretation = row.get("interpretation", "")
            
            if indicator != "Health Score":  # Skip health score since it's in the header
                lines.append(f"**{indicator}:** {value}")
                if interpretation:
                    lines.append(f"> {interpretation}")
                lines.append("")
    
    # Health factors
    if positive_factors:
        lines.append("### ✅ Positive Factors")
        lines.append("")
        for f in positive_factors:
            lines.append(f"- {f}")
        lines.append("")
    
    if risk_factors:
        lines.append("### ⚠️ Risk Factors")
        lines.append("")
        for f in risk_factors:
            lines.append(f"- {f}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # ==========================================================================
    # ROLES × REGION LENS SECTION
    # ==========================================================================
    if roles_region_summary and roles_region_summary.get('enabled', False):
        lines.append("## 🗺️ Roles × Region Lens")
        lines.append("")
        
        # Section intro
        lines.append(SECTION_INTROS.get("roles_region_lens", ""))
        lines.append("")
        
        # Decision Lens block
        lens = DECISION_LENS.get("roles_region_lens", {})
        if lens:
            intensity = lens.get("signal_intensity", "medium")
            lines.append(f':::decision-lens intensity="{intensity}"')
            lines.append(f"**What this tells you**")
            lines.append(lens.get("what_tells_you", ""))
            lines.append("")
            lines.append(f"**Why it matters for decisions**")
            lines.append(lens.get("why_matters", ""))
            lines.append("")
            lines.append(f"**What teams often do next**")
            lines.append(lens.get("teams_do_next", ""))
            if lens.get("not_over_interpret"):
                lines.append("")
                lines.append(f"**What not to over-interpret**")
                lines.append(lens.get("not_over_interpret", ""))
            lines.append(":::")
            lines.append("")
        
        # Add the actual region data (from existing format_roles_region_section logic)
        region_lines = format_roles_region_section(roles_region_summary, skip_header=True)
        lines.extend(region_lines)
        lines.append("---")
        lines.append("")
    
    # ==========================================================================
    # REMAINING CARDS (with Decision Lens blocks)
    # ==========================================================================
    for card in cards:
        card_id = card.get("card_id", "")
        
        # Skip network_health card (already rendered above)
        if card_id == "network_health":
            continue
        
        title = card.get("title", "Untitled")
        use_case = card.get("use_case", "")
        summary_text = card.get("summary", "")
        
        lines.append(f"## {title}")
        
        # Add speed bump subtitle if available (for skimmers)
        subtitle = SECTION_SUBTITLES.get(card_id, "")
        if subtitle:
            # Use HTML directly - markdown underscores don't reliably convert
            lines.append(f'<p class="section-subtitle">{subtitle}</p>')
            lines.append("")
        
        lines.append(f"*Use Case: {use_case}*")
        lines.append("")
        
        # Section intro (if available)
        intro = SECTION_INTROS.get(card_id, "")
        if intro:
            lines.append(intro)
            lines.append("")
        
        # Decision Lens block (if available)
        # Handle special case for empty shared_board_conduits
        lens_key = card_id
        if card_id == "shared_board_conduits" and "No multi-board" in summary_text:
            lens_key = "shared_board_conduits_empty"
        
        lens = DECISION_LENS.get(lens_key, {})
        if lens:
            intensity = lens.get("signal_intensity", "medium")
            lines.append(f':::decision-lens intensity="{intensity}"')
            lines.append(f"**What this tells you**")
            lines.append(lens.get("what_tells_you", ""))
            lines.append("")
            lines.append(f"**Why it matters for decisions**")
            lines.append(lens.get("why_matters", ""))
            lines.append("")
            lines.append(f"**What teams often do next**")
            lines.append(lens.get("teams_do_next", ""))
            if lens.get("not_over_interpret"):
                lines.append("")
                lines.append(f"**What not to over-interpret**")
                lines.append(lens.get("not_over_interpret", ""))
            lines.append(":::")
            lines.append("")
        
        # Original summary/analysis content
        lines.append(summary_text)
        lines.append("")
        
        # Ranked rows
        ranked_rows = card.get("ranked_rows", [])
        if ranked_rows:
            # Check if rows have narratives
            has_narratives = any(r.get("narrative") for r in ranked_rows)
            
            if has_narratives:
                for idx, row in enumerate(ranked_rows, 1):
                    rank = row.get("rank", idx)  # Default to index if no rank
                    entity = (
                        row.get("grantee") or 
                        row.get("person") or 
                        row.get("org") or 
                        row.get("funder") or 
                        row.get("node") or 
                        row.get("pair") or
                        ""
                    )
                    
                    if entity:
                        lines.append(f"### {rank}. {entity}")
                        lines.append("")
                    
                    if row.get("narrative"):
                        lines.append(row["narrative"])
                        lines.append("")
                    
                    if row.get("recommendation"):
                        lines.append(row["recommendation"])
                        lines.append("")
            elif any(r.get("interpretation") for r in ranked_rows):
                # Health-style indicators: render as vertical list
                for row in ranked_rows:
                    indicator = row.get("indicator", "")
                    value = row.get("value", "")
                    interpretation = row.get("interpretation", "")
                    
                    lines.append(f"**{indicator}:** {value}")
                    if interpretation:
                        lines.append(f"> {interpretation}")
                    lines.append("")
            else:
                # Simple table
                if ranked_rows:
                    # Get column headers from first row
                    cols = [k for k in ranked_rows[0].keys() if k not in ["rank", "node_ids", "edge_ids"]]
                    
                    # Header row
                    lines.append("| " + " | ".join(cols) + " |")
                    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
                    
                    # Data rows
                    for row in ranked_rows:
                        vals = [str(row.get(c, "")) for c in cols]
                        lines.append("| " + " | ".join(vals) + " |")
                    
                    lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("*Report generated by C4C InsightGraph*")
    
    return "\n".join(lines)


# =============================================================================
# Project Summary
# =============================================================================

def generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats):
    """Generate top-level project summary."""
    return {
        "generated_at": datetime.now().isoformat(),
        "node_counts": {
            "total": len(nodes_df),
            "organizations": len(nodes_df[nodes_df["node_type"] == "ORG"]),
            "people": len(nodes_df[nodes_df["node_type"] == "PERSON"]),
        },
        "edge_counts": {
            "total": len(edges_df),
            "grants": len(edges_df[edges_df["edge_type"] == "GRANT"]),
            "board_memberships": len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]),
        },
        "funding": {
            "total_amount": flow_stats["total_grant_amount"],
            "funder_count": flow_stats["funder_count"],
            "grantee_count": flow_stats["grantee_count"],
            "top_5_share": flow_stats["top_5_funders_share"],
        },
        "governance": {
            "multi_board_people": len(metrics_df[(metrics_df["node_type"] == "PERSON") & (metrics_df["boards_served"] >= 2)]),
        },
    }


def generate_manifest(
    project_id: str,
    project_summary: dict,
    insight_cards: dict,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    grants_df: pd.DataFrame = None,
    region_lens_config: dict = None,
    cloud_project_id: str = None,
    input_paths: dict = None
) -> dict:
    """
    Generate bundle manifest for traceability and reproducibility.
    
    Args:
        project_id: Project identifier (e.g., "glfn-2025")
        project_summary: Output from generate_project_summary()
        insight_cards: Output from generate_insight_cards()
        nodes_df: Input nodes DataFrame
        edges_df: Input edges DataFrame
        grants_df: Optional grants_detail DataFrame
        region_lens_config: Region lens configuration used
        cloud_project_id: Supabase project UUID (if saved to cloud)
        input_paths: Dict with original file paths {"nodes": ..., "edges": ..., "grants": ...}
    
    Returns:
        Manifest dict ready for JSON serialization.
    """
    now = datetime.now(timezone.utc)
    
    # Helper to compute file hash
    def df_hash(df):
        if df is None or df.empty:
            return None
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
    
    # Build manifest
    manifest = {
        "bundle": {
            "format_version": BUNDLE_FORMAT_VERSION,
            "generated_at": now.isoformat(),
            "generated_by": {
                "app": "InsightGraph",
                "engine": f"run.py v{ENGINE_VERSION}"
            }
        },
        
        "project": {
            "id": project_id,
            "name": project_id.replace("-", " ").replace("_", " ").title(),
        },
        
        "config": {
            "thresholds": {
                "connector_percentile": CONNECTOR_THRESHOLD,
                "broker_percentile": BROKER_THRESHOLD,
                "hidden_broker_degree_cap": HIDDEN_BROKER_DEGREE_CAP,
                "capital_hub_percentile": CAPITAL_HUB_THRESHOLD
            }
        },
        
        "inputs": {
            "nodes": {
                "path": "data/nodes.csv",
                "rows": len(nodes_df) if nodes_df is not None else 0,
                "hash": df_hash(nodes_df)
            },
            "edges": {
                "path": "data/edges.csv",
                "rows": len(edges_df) if edges_df is not None else 0,
                "hash": df_hash(edges_df)
            }
        },
        
        "outputs": {
            "report_html": {
                "path": "index.html"
            },
            "report_md": {
                "path": "report.md",
                "sections": [
                    "Network Health",
                    "Funding Concentration",
                    "Funder Overlap",
                    "Portfolio Twins",
                    "Shared Board Conduits",
                    "Hidden Brokers",
                    "Single-Point Bridges",
                    "Roles × Region Lens",
                    "Decision Options"
                ]
            },
            "project_summary": {
                "path": "analysis/project_summary.json"
            },
            "insight_cards": {
                "path": "analysis/insight_cards.json",
                "cards": len(insight_cards) if insight_cards else 0
            },
            "node_metrics": {
                "path": "analysis/node_metrics.csv"
            },
            "synthesis_guides": {
                "visual_synthesis_guide": "guides/VISUAL_SYNTHESIS_GUIDE.md",
                "synthesis_mode_prompt": "guides/SYNTHESIS_MODE_PROMPT.md",
                "synthesis_checklist": "guides/SYNTHESIS_CHECKLIST.md"
            }
        },
        
        "stats": {
            "network": {
                "nodes": project_summary.get("node_counts", {}).get("total", 0),
                "edges": project_summary.get("edge_counts", {}).get("total", 0),
                "organizations": project_summary.get("node_counts", {}).get("organizations", 0),
                "people": project_summary.get("node_counts", {}).get("people", 0)
            },
            "funding": {
                "total": project_summary.get("funding", {}).get("total_amount", 0),
                "funders": project_summary.get("funding", {}).get("funder_count", 0),
                "grantees": project_summary.get("funding", {}).get("grantee_count", 0),
                "top_5_share": project_summary.get("funding", {}).get("top_5_share", 0)
            }
        }
    }
    
    # Add cloud project ID if available
    if cloud_project_id:
        manifest["project"]["cloud_id"] = cloud_project_id
    
    # Add region lens config if available
    if region_lens_config:
        manifest["config"]["region_lens"] = {
            "id": region_lens_config.get("id", "custom"),
            "name": region_lens_config.get("name", "Custom Region")
        }
        # Include geographic scope if present
        if "include_us_states" in region_lens_config:
            manifest["config"]["region_lens"]["include_us_states"] = region_lens_config["include_us_states"]
        if "include_ca_provinces" in region_lens_config:
            manifest["config"]["region_lens"]["include_ca_provinces"] = region_lens_config["include_ca_provinces"]
    
    # Add grants_detail if available
    if grants_df is not None and not grants_df.empty:
        manifest["inputs"]["grants_detail"] = {
            "path": "data/grants_detail.csv",
            "rows": len(grants_df),
            "hash": df_hash(grants_df)
        }
    
    # Add health score if available in insight_cards
    health = None
    if insight_cards:
        health = insight_cards.get("health") or insight_cards.get("network_health")
    
    if health:
        manifest["stats"]["health"] = {
            "score": health.get("score", 0),
            "label": health.get("label", "Unknown")
        }
    
    # Add original input paths for traceability
    if input_paths:
        for key, path in input_paths.items():
            if key in manifest["inputs"] and path:
                manifest["inputs"][key]["source_path"] = str(path)
    
    # Add synthesis metadata (guidance for downstream tools)
    manifest["synthesis"] = {
        "purpose": (
            "Guidance for generating non-prescriptive visual or narrative "
            "summaries of this report."
        ),
        "visual_synthesis_guide": "guides/VISUAL_SYNTHESIS_GUIDE.md",
        "synthesis_mode_prompt": "guides/SYNTHESIS_MODE_PROMPT.md",
        "synthesis_checklist": "guides/SYNTHESIS_CHECKLIST.md",
        "intended_use": [
            "NotebookLM",
            "slide generation tools",
            "infographic drafting",
            "facilitated discussion prep"
        ],
        "constraints": [
            "Do not imply recommendations from structural signals",
            "Preserve signal intensity labels (Low/Moderate/High)",
            "Include 'no action may be appropriate' framing",
            "Use 'teams often use this to decide' phrasing"
        ]
    }
    
    return manifest


def preprocess_decision_lens(md_content: str) -> str:
    """
    Preprocess decision-lens blocks to HTML before markdown library processing.
    
    Converts :::decision-lens intensity="low" ... ::: to HTML divs.
    This is needed because the markdown library doesn't handle custom fence syntax.
    """
    lines = md_content.split('\n')
    result_lines = []
    in_decision_lens = False
    decision_lens_intensity = "medium"
    decision_lens_content = {}
    current_field = None
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith(':::decision-lens'):
            in_decision_lens = True
            decision_lens_content = {}
            decision_lens_intensity = "medium"
            current_field = None
            
            # Parse intensity
            intensity_match = re.search(r'intensity="([^"]+)"', stripped)
            if intensity_match:
                decision_lens_intensity = intensity_match.group(1)
            continue
        
        if stripped == ':::' and in_decision_lens:
            # Render the decision lens HTML
            intensity_labels = {
                "low": "Low-intensity signal",
                "medium": "Moderate-intensity signal",
                "high": "High-intensity signal",
            }
            intensity_label = intensity_labels.get(decision_lens_intensity, "Signal")
            
            result_lines.append(f'<div class="decision-lens decision-lens--{decision_lens_intensity}" role="note" aria-label="Decision Lens">')
            result_lines.append('  <div class="decision-lens__header">')
            result_lines.append('    <p class="decision-lens__title">Decision Lens</p>')
            result_lines.append(f'    <span class="decision-lens__badge decision-lens__badge--{decision_lens_intensity}">{intensity_label}</span>')
            result_lines.append('  </div>')
            result_lines.append('  <div class="decision-lens__grid">')
            
            # What this tells you
            result_lines.append('    <div class="decision-lens__item">')
            result_lines.append('      <p class="decision-lens__label">What this tells you</p>')
            result_lines.append(f'      <p class="decision-lens__text">{decision_lens_content.get("tells_you", "")}</p>')
            result_lines.append('    </div>')
            
            # Why it matters
            result_lines.append('    <div class="decision-lens__item">')
            result_lines.append('      <p class="decision-lens__label">Why it matters</p>')
            result_lines.append(f'      <p class="decision-lens__text">{decision_lens_content.get("why_matters", "")}</p>')
            result_lines.append('    </div>')
            
            # What teams often do next
            result_lines.append('    <div class="decision-lens__item">')
            result_lines.append('      <p class="decision-lens__label">What teams often do next</p>')
            result_lines.append(f'      <p class="decision-lens__text">{decision_lens_content.get("next_steps", "")}</p>')
            result_lines.append('    </div>')
            
            result_lines.append('  </div>')
            
            # Guardrail (if present)
            if decision_lens_content.get("guardrail"):
                result_lines.append('  <div class="decision-lens__guardrail">')
                result_lines.append('    <p class="decision-lens__guardrail-title">What not to over-interpret</p>')
                result_lines.append(f'    <p class="decision-lens__guardrail-text">{decision_lens_content.get("guardrail", "")}</p>')
                result_lines.append('  </div>')
            
            # Global no-action normalization (always present)
            result_lines.append('  <div class="decision-lens__footer">')
            result_lines.append('    <p class="decision-lens__footer-text">In many cases, the appropriate outcome of this analysis is to confirm that no coordination or intervention is needed.</p>')
            result_lines.append('  </div>')
            
            result_lines.append('</div>')
            result_lines.append('')
            
            in_decision_lens = False
            decision_lens_content = {}
            decision_lens_intensity = "medium"
            current_field = None
            continue
        
        if in_decision_lens:
            if stripped.startswith('**') and stripped.endswith('**'):
                field_name = stripped[2:-2].lower()
                if 'tells you' in field_name:
                    current_field = 'tells_you'
                elif 'why it matters' in field_name or 'matters for decisions' in field_name:
                    current_field = 'why_matters'
                elif 'teams often do' in field_name or 'next' in field_name:
                    current_field = 'next_steps'
                elif 'not to over-interpret' in field_name or 'over-interpret' in field_name:
                    current_field = 'guardrail'
                else:
                    current_field = None
            elif stripped and current_field:
                # Apply basic inline formatting
                text = stripped
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
                decision_lens_content[current_field] = text
            continue
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def render_html_report(
    markdown_content: str,
    project_summary: dict,
    insight_cards: dict = None,
    project_id: str = "report",
    template_path: Path = None
) -> str:
    """
    Render markdown report to styled HTML.
    
    Args:
        markdown_content: The markdown report content
        project_summary: Project summary dict with stats
        insight_cards: Optional insight cards for health score
        project_id: Project identifier
        template_path: Path to HTML template (optional, uses embedded default)
    
    Returns:
        Rendered HTML string
    """
    # Try to import markdown library
    try:
        import markdown
        from markdown.extensions.toc import TocExtension
        HAS_MARKDOWN = True
    except ImportError:
        HAS_MARKDOWN = False
        print("Warning: markdown library not installed. Using basic HTML conversion.")
    
    # Extract health score if available
    health_score = None
    health_label = "Unknown"
    health_summary = ""
    
    # Check for health in insight_cards (can be under "health" or "network_health")
    health = None
    if insight_cards:
        health = insight_cards.get("health") or insight_cards.get("network_health")
    
    if health:
        health_score = health.get("score")
        health_label = health.get("label", "Unknown")
        # Create a brief summary
        if health_score:
            if health_score >= 80:
                health_summary = "Strong network with good connectivity and governance structures."
            elif health_score >= 60:
                health_summary = "Moderate network health with some areas for improvement."
            elif health_score >= 40:
                health_summary = "Network shows vulnerabilities that may warrant attention."
            else:
                health_summary = "Network requires significant strengthening."
    
    # Project name from project_id
    project_name = project_id.replace("-", " ").replace("_", " ").title()
    
    # Get date
    generated_at = project_summary.get("generated_at", datetime.now().isoformat())
    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = dt.strftime("%B %d, %Y")
    except:
        date_str = generated_at[:10] if len(generated_at) >= 10 else generated_at
    
    # Convert markdown to HTML
    if HAS_MARKDOWN:
        # Use markdown library with TOC extension
        md = markdown.Markdown(extensions=[
            'tables',
            'fenced_code',
            TocExtension(permalink=False, toc_depth=3)
        ])
        # Preprocess decision-lens blocks (custom syntax not handled by markdown library)
        preprocessed_md = preprocess_decision_lens(markdown_content)
        content_html = md.convert(preprocessed_md)
        toc_html = md.toc
    else:
        # Basic conversion without library
        content_html = basic_markdown_to_html(markdown_content)
        toc_html = generate_basic_toc(markdown_content)
    
    # Wrap sections in <section> tags for styling
    content_html = wrap_sections(content_html)
    
    # Add callout styling
    content_html = style_callouts(content_html)
    
    # Build HTML from embedded template
    html = build_html_from_template(
        project_name=project_name,
        project_id=project_id,
        date=date_str,
        version=ENGINE_VERSION,
        health_score=health_score,
        health_label=health_label,
        health_summary=health_summary,
        toc=toc_html,
        content=content_html
    )
    
    return html


def basic_markdown_to_html(md_content: str) -> str:
    """Basic markdown to HTML conversion without external library."""
    lines = md_content.split('\n')
    html_lines = []
    in_list = False
    in_table = False
    in_blockquote = False
    in_decision_lens = False
    table_header_done = False
    skip_first_h1 = True  # Skip first H1 (duplicate of header)
    
    # Track decision lens content for grid rendering
    decision_lens_content = {}
    decision_lens_intensity = "medium"
    current_dl_field = None
    
    for line in lines:
        stripped = line.strip()
        
        # Handle decision-lens blocks (with intensity)
        if stripped.startswith(':::decision-lens'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            in_decision_lens = True
            decision_lens_content = {}
            decision_lens_intensity = "medium"
            current_dl_field = None
            
            # Parse intensity: :::decision-lens intensity="low"
            intensity_match = re.search(r'intensity="([^"]+)"', stripped)
            if intensity_match:
                decision_lens_intensity = intensity_match.group(1)
            continue
        
        if stripped == ':::' and in_decision_lens:
            # Get intensity label
            intensity_labels = {
                "low": "Low-intensity signal",
                "medium": "Moderate-intensity signal",
                "high": "High-intensity signal",
            }
            intensity_label = intensity_labels.get(decision_lens_intensity, "Signal")
            intensity_class = f"decision-lens--{decision_lens_intensity}"
            
            # Render the decision lens grid
            html_lines.append(f'<div class="decision-lens {intensity_class}" role="note" aria-label="Decision Lens">')
            html_lines.append('  <div class="decision-lens__header">')
            html_lines.append('    <p class="decision-lens__title">Decision Lens</p>')
            html_lines.append(f'    <span class="decision-lens__badge decision-lens__badge--{decision_lens_intensity}">{intensity_label}</span>')
            html_lines.append('  </div>')
            html_lines.append('  <div class="decision-lens__grid">')
            
            # What this tells you
            html_lines.append('    <div class="decision-lens__item">')
            html_lines.append('      <p class="decision-lens__label">What this tells you</p>')
            html_lines.append(f'      <p class="decision-lens__text">{decision_lens_content.get("tells_you", "")}</p>')
            html_lines.append('    </div>')
            
            # Why it matters
            html_lines.append('    <div class="decision-lens__item">')
            html_lines.append('      <p class="decision-lens__label">Why it matters</p>')
            html_lines.append(f'      <p class="decision-lens__text">{decision_lens_content.get("why_matters", "")}</p>')
            html_lines.append('    </div>')
            
            # What teams often do next
            html_lines.append('    <div class="decision-lens__item">')
            html_lines.append('      <p class="decision-lens__label">What teams often do next</p>')
            html_lines.append(f'      <p class="decision-lens__text">{decision_lens_content.get("next_steps", "")}</p>')
            html_lines.append('    </div>')
            
            html_lines.append('  </div>')
            
            # Guardrail (if present)
            if decision_lens_content.get("guardrail"):
                html_lines.append('  <div class="decision-lens__guardrail">')
                html_lines.append('    <p class="decision-lens__guardrail-title">What not to over-interpret</p>')
                html_lines.append(f'    <p class="decision-lens__guardrail-text">{decision_lens_content.get("guardrail", "")}</p>')
                html_lines.append('  </div>')
            
            # Global no-action normalization (always present)
            html_lines.append('  <div class="decision-lens__footer">')
            html_lines.append('    <p class="decision-lens__footer-text">In many cases, the appropriate outcome of this analysis is to confirm that no coordination or intervention is needed.</p>')
            html_lines.append('  </div>')
            
            html_lines.append('</div>')
            
            in_decision_lens = False
            decision_lens_content = {}
            decision_lens_intensity = "medium"
            current_dl_field = None
            continue
        
        # Inside decision-lens, collect content by field
        if in_decision_lens:
            if stripped.startswith('**') and stripped.endswith('**'):
                # Field header like "**What this tells you**"
                field_name = stripped[2:-2].lower()
                if 'tells you' in field_name:
                    current_dl_field = 'tells_you'
                elif 'why it matters' in field_name or 'matters for decisions' in field_name:
                    current_dl_field = 'why_matters'
                elif 'teams often do' in field_name or 'next' in field_name:
                    current_dl_field = 'next_steps'
                elif 'not to over-interpret' in field_name or 'over-interpret' in field_name:
                    current_dl_field = 'guardrail'
                else:
                    current_dl_field = None
            elif stripped and current_dl_field:
                # Content for current field
                decision_lens_content[current_dl_field] = inline_format(stripped)
            continue
        
        # Close blockquote if we're leaving it
        if in_blockquote and not stripped.startswith('>'):
            html_lines.append('</div>')
            in_blockquote = False
        
        # Blockquotes → Callouts
        if stripped.startswith('> '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            
            content = stripped[2:].strip()
            
            # Detect callout type from content
            callout_class = 'callout'
            if any(word in content.lower() for word in ['warning', 'risk', 'weak', 'vulnerability', 'concern']):
                callout_class = 'callout callout-warning'
            elif any(word in content.lower() for word in ['strong', 'positive', 'opportunity', 'strength']):
                callout_class = 'callout callout-success'
            elif any(word in content.lower() for word in ['note', 'info', 'signal']):
                callout_class = 'callout callout-info'
            
            if not in_blockquote:
                html_lines.append(f'<div class="{callout_class}">')
                in_blockquote = True
            
            html_lines.append(f'<p>{inline_format(content)}</p>')
            continue
        
        # Headers
        if stripped.startswith('# ') and not stripped.startswith('## '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            # Skip first H1 (already in page header)
            if skip_first_h1:
                skip_first_h1 = False
                continue
            html_lines.append(f'<h1 id="{slugify(stripped[2:])}">{inline_format(stripped[2:])}</h1>')
        
        elif stripped.startswith('### '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3 id="{slugify(stripped[4:])}">{inline_format(stripped[4:])}</h3>')
        
        elif stripped.startswith('## '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h2 id="{slugify(stripped[3:])}">{inline_format(stripped[3:])}</h2>')
        
        # Horizontal rule
        elif stripped == '---':
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<hr>')
        
        # Table
        elif '|' in stripped and stripped.startswith('|'):
            if not in_table:
                html_lines.append('<table>')
                in_table = True
                table_header_done = False
            
            # Skip separator row
            if stripped.replace('|', '').replace('-', '').replace(':', '').strip() == '':
                table_header_done = True
                continue
            
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if not table_header_done:
                html_lines.append('<thead><tr>')
                for cell in cells:
                    html_lines.append(f'<th>{cell}</th>')
                html_lines.append('</tr></thead><tbody>')
            else:
                html_lines.append('<tr>')
                for cell in cells:
                    html_lines.append(f'<td>{inline_format(cell)}</td>')
                html_lines.append('</tr>')
        
        # List items
        elif stripped.startswith('- ') or stripped.startswith('* '):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{inline_format(stripped[2:])}</li>')
        
        # Numbered list
        elif re.match(r'^\d+\. ', stripped):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            content = re.sub(r'^\d+\. ', '', stripped)
            if not in_list:
                html_lines.append('<ol>')
                in_list = True
            html_lines.append(f'<li>{inline_format(content)}</li>')
        
        # Empty line
        elif stripped == '':
            if in_list:
                html_lines.append('</ul>' if '</li>' in html_lines[-1] else '</ol>')
                in_list = False
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
        
        # Paragraph (with signal pill detection)
        else:
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            if stripped:
                # Detect "Use Case:" labels
                if stripped.startswith('*Use Case:'):
                    content = stripped.replace('*Use Case:', '').replace('*', '').strip()
                    html_lines.append(f'<p class="use-case"><span class="label">Use Case:</span> {content}</p>')
                # Detect signal indicators
                elif stripped.startswith('🟢') or stripped.startswith('🟡') or stripped.startswith('🔴'):
                    signal_class = 'signal-green' if '🟢' in stripped else ('signal-yellow' if '🟡' in stripped else 'signal-red')
                    html_lines.append(f'<p class="signal {signal_class}">{inline_format(stripped)}</p>')
                else:
                    html_lines.append(f'<p>{inline_format(stripped)}</p>')
    
    # Close any open elements
    if in_blockquote:
        html_lines.append('</div>')
    if in_list:
        html_lines.append('</ul>')
    if in_table:
        html_lines.append('</tbody></table>')
    
    return '\n'.join(html_lines)


def inline_format(text: str) -> str:
    """Apply inline markdown formatting (bold, italic, code)."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text


def slugify(text: str) -> str:
    """Create URL-friendly slug from text."""
    # Remove emoji and special chars
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_]+', '-', slug)
    return slug.strip('-')


def generate_basic_toc(md_content: str) -> str:
    """Generate table of contents HTML from markdown headers."""
    toc_items = []
    for line in md_content.split('\n'):
        if line.startswith('## '):
            title = line[3:].strip()
            slug = slugify(title)
            toc_items.append(f'<li><a href="#{slug}">{title}</a></li>')
    
    if toc_items:
        return f'<ul>{"".join(toc_items)}</ul>'
    return '<ul><li>No sections found</li></ul>'


def wrap_sections(html_content: str) -> str:
    """Wrap H2 sections in <section> tags for styling."""
    # Split on h2 tags
    parts = re.split(r'(<h2[^>]*>)', html_content)
    
    result = []
    in_section = False
    
    for part in parts:
        if part.startswith('<h2'):
            if in_section:
                result.append('</section>')
            result.append('<section>')
            in_section = True
        result.append(part)
    
    if in_section:
        result.append('</section>')
    
    return ''.join(result)


def style_callouts(html_content: str) -> str:
    """Add callout styling for key patterns."""
    # Style "Key Takeaway" or similar patterns
    html_content = re.sub(
        r'<p><strong>(Key Takeaway|Recommendation|Note|Warning|Tip):</strong>',
        r'<div class="callout"><p><strong>\1:</strong>',
        html_content
    )
    # Close divs (simple heuristic - next paragraph)
    html_content = re.sub(
        r'(class="callout"><p>.*?</p>)\s*<p>',
        r'\1</div>\n<p>',
        html_content,
        flags=re.DOTALL
    )
    return html_content


def build_html_from_template(
    project_name: str,
    project_id: str,
    date: str,
    version: str,
    health_score: int,
    health_label: str,
    health_summary: str,
    toc: str,
    content: str
) -> str:
    """Build complete HTML document from embedded template."""
    
    # Health banner HTML (only if score exists)
    health_banner = ""
    if health_score is not None:
        health_banner = f'''
  <div class="health-banner">
    <div class="health-score">{health_score}<span>/100</span></div>
    <div class="health-details">
      <h2>Network Health: {health_label}</h2>
      <p>{health_summary}</p>
    </div>
  </div>
'''
    
    # Complete HTML template (embedded for portability)
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{project_name} — Network Insight Report</title>
  <style>
    /*
     * C4C Design Contract
     * -------------------
     * teal    = structure (headers, section borders)
     * indigo  = interactive (links, focus states)
     * orange  = emphasis (health banner, accent)
     * green   = positive valence (success, factors-positive)
     * red     = negative valence (warning, factors-risk)
     */
    html {{ color-scheme: light; }}
    /* Skip link for accessibility */
    .skip-link {{
      position: absolute;
      left: -999px;
      top: 0;
      background: #fff;
      color: var(--c4c-indigo);
      border: 2px solid var(--c4c-indigo);
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
      z-index: 9999;
    }}
    .skip-link:focus-visible {{
      left: 1rem;
      top: 1rem;
      outline: 3px solid var(--c4c-indigo);
      outline-offset: 2px;
    }}
    /* Section toggle button styles */
    .section-toggle {{
      flex: 0 0 auto;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.35rem 0.6rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--c4c-indigo);
      cursor: pointer;
    }}
    .section-toggle:hover {{ background: rgba(40, 37, 190, 0.06); }}
    .section-toggle:focus-visible {{
      outline: 3px solid var(--c4c-indigo);
      outline-offset: 2px;
    }}
    :root {{
      /* C4C Brand Palette */
      --c4c-teal: #0C7A7A;
      --c4c-teal-light: #0e8f8f;
      --c4c-orange: #EB9001;
      --c4c-indigo: #2825BE;
      --c4c-red: #CF4C38;
      --c4c-green: #2d6a4f;
      --c4c-purple: #6b2e77;
      --c4c-rose: #E8A7A5;
      
      /* Semantic colors */
      --primary: var(--c4c-teal);
      --primary-light: var(--c4c-teal-light);
      --accent: var(--c4c-orange);
      --warning: var(--c4c-red);
      --success: var(--c4c-green);
      
      /* Neutrals */
      --text: #1a1a1a;
      --text-light: #444;
      --muted: #666;
      --bg: #fafafa;
      --card-bg: #fff;
      --border: #e0e0e0;
      --border-light: #f0f0f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      line-height: 1.65;
      color: var(--text);
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem;
      background: var(--bg);
    }}
    /* Links - indigo for interactive elements */
    a {{
      color: var(--c4c-indigo);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    a:focus-visible {{
      outline: 3px solid var(--c4c-indigo);
      outline-offset: 2px;
      border-radius: 4px;
    }}
    header {{
      border-bottom: 3px solid var(--primary);
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }}
    .logo {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }}
    .logo img {{
      height: 32px;
      width: auto;
    }}
    .logo span {{
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--primary);
      letter-spacing: 0.5px;
    }}
    header h1 {{
      font-size: 2rem;
      font-weight: 700;
      margin: 0.25rem 0;
      color: var(--primary);
    }}
    .subtitle {{
      font-size: 1.15rem;
      color: var(--muted);
      margin: 0;
    }}
    .meta {{
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 0.75rem;
    }}
    .health-banner {{
      background: var(--accent);
      color: white;
      padding: 1.5rem 2rem;
      border-radius: 10px;
      margin-bottom: 2rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;
    }}
    .health-score {{
      font-size: 3rem;
      font-weight: 700;
      line-height: 1;
    }}
    .health-score span {{
      font-size: 1.5rem;
      opacity: 0.8;
    }}
    .health-details h2 {{
      margin: 0;
      font-size: 1.25rem;
      font-weight: 600;
    }}
    .health-details p {{
      margin: 0.25rem 0 0;
      opacity: 0.9;
    }}
    .toc {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.5rem 2rem;
      margin-bottom: 2rem;
    }}
    .toc h2 {{
      font-size: 0.9rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin: 0 0 1rem;
      color: var(--muted);
    }}
    .toc ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      columns: 2;
      column-gap: 2rem;
    }}
    .toc li {{
      margin-bottom: 0.5rem;
    }}
    .toc a {{
      color: var(--c4c-indigo);
      text-decoration: none;
    }}
    .toc a:hover {{
      text-decoration: underline;
    }}
    section {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 2rem;
      margin-bottom: 1.5rem;
    }}
    section h2 {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      margin: 0 0 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
      color: var(--primary);
      font-size: 1.35rem;
    }}
    h3 {{
      color: var(--text);
      font-size: 1.1rem;
      margin: 1.5rem 0 0.75rem;
    }}
    h4 {{
      color: var(--text-light);
      font-size: 1rem;
      margin: 1.25rem 0 0.5rem;
    }}
    p {{ margin: 0 0 1rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.9rem;
      display: block;
      overflow-x: auto;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border-light);
      vertical-align: top;
    }}
    th {{
      background: var(--bg);
      font-weight: 600;
      color: var(--text-light);
      font-size: 0.8rem;
      text-transform: uppercase;
    }}
    tr:nth-child(even) {{ background: rgba(0,0,0,0.02); }}
    tr:hover {{ background: rgba(0,0,0,0.04); }}
    ul, ol {{
      padding-left: 1.5rem;
      margin: 0.75rem 0;
    }}
    li {{ margin-bottom: 0.4rem; }}
    /* Callouts - semantic rgba backgrounds */
    .callout {{
      background: rgba(12, 122, 122, 0.08);
      border-left: 4px solid var(--primary);
      padding: 1rem 1.25rem;
      margin: 1rem 0;
      border-radius: 0 8px 8px 0;
    }}
    .callout p {{ margin: 0; }}
    .callout strong {{ color: inherit; }}
    .callout-warning {{
      background: rgba(207, 76, 56, 0.10);
      border-left-color: var(--warning);
    }}
    .callout-success {{
      background: rgba(45, 106, 79, 0.10);
      border-left-color: var(--success);
    }}
    .callout-info {{
      background: rgba(40, 37, 190, 0.08);
      border-left-color: var(--c4c-indigo);
    }}
    /* Decision Lens Component (C4C) */
    .decision-lens {{
      margin: 1rem 0 1.25rem;
      padding: 1rem 1.25rem;
      border-radius: 12px;
      border: 1px solid rgba(40, 37, 190, 0.18);
      background: rgba(40, 37, 190, 0.06);
    }}
    .decision-lens__header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}
    .decision-lens__title {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 800;
      letter-spacing: 0.2px;
      color: var(--c4c-indigo);
      margin: 0;
      font-size: 0.95rem;
      text-transform: uppercase;
    }}
    .decision-lens__badge {{
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      border: 1px solid rgba(40, 37, 190, 0.25);
      background: rgba(255, 255, 255, 0.65);
      color: var(--c4c-indigo);
      white-space: nowrap;
    }}
    .decision-lens__grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
    }}
    @media (min-width: 720px) {{
      .decision-lens__grid {{
        grid-template-columns: 1fr 1fr 1fr;
      }}
    }}
    .decision-lens__item {{
      background: rgba(255, 255, 255, 0.75);
      border: 1px solid rgba(0, 0, 0, 0.06);
      border-radius: 10px;
      padding: 0.75rem 0.85rem;
    }}
    .decision-lens__label {{
      font-size: 0.75rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted);
      margin: 0 0 0.35rem;
    }}
    .decision-lens__text {{
      margin: 0;
      font-size: 0.92rem;
      color: var(--text);
    }}
    .decision-lens__guardrail {{
      margin-top: 0.9rem;
      padding-top: 0.75rem;
      border-top: 1px dashed rgba(40, 37, 190, 0.22);
    }}
    .decision-lens__guardrail-title {{
      margin: 0 0 0.35rem;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted);
    }}
    .decision-lens__guardrail-text {{
      margin: 0;
      font-size: 0.92rem;
      color: var(--text-light);
      font-style: italic;
    }}
    /* Signal Intensity Badge Colors */
    .decision-lens__badge--low {{
      background: rgba(102, 102, 102, 0.12);
      border-color: rgba(102, 102, 102, 0.3);
      color: #555;
    }}
    .decision-lens__badge--medium {{
      background: rgba(40, 37, 190, 0.12);
      border-color: rgba(40, 37, 190, 0.3);
      color: var(--c4c-indigo);
    }}
    .decision-lens__badge--high {{
      background: rgba(235, 144, 1, 0.15);
      border-color: rgba(235, 144, 1, 0.4);
      color: #b87000;
    }}
    /* Decision Lens Footer (global no-action normalization) */
    .decision-lens__footer {{
      margin-top: 0.9rem;
      padding-top: 0.65rem;
      border-top: 1px solid rgba(40, 37, 190, 0.12);
    }}
    .decision-lens__footer-text {{
      margin: 0;
      font-size: 0.82rem;
      color: var(--muted);
      font-style: italic;
      text-align: center;
    }}
    /* Legacy callout-decision (for backward compat) */
    .callout-decision {{
      background: rgba(40, 37, 190, 0.06);
      border-left-color: var(--c4c-indigo);
      margin: 1.25rem 0 1.5rem;
    }}
    .callout-decision .decision-label {{
      font-weight: 700;
      color: var(--c4c-indigo);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 0.75rem;
      display: block;
    }}
    .callout-decision p {{
      margin: 0.5rem 0;
      font-size: 0.95rem;
    }}
    .callout-decision p:last-child {{
      margin-bottom: 0;
    }}
    /* Use Case labels */
    .use-case {{
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 0.5rem;
    }}
    .use-case .label {{
      font-weight: 600;
      color: var(--primary);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.5px;
    }}
    /* Section subtitle (speed bump for skimmers) */
    .section-subtitle {{
      font-size: 0.92rem;
      color: var(--muted);
      font-style: italic;
      margin: -0.25rem 0 0.5rem;
    }}
    /* Signal indicators */
    .signal {{
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-size: 0.85rem;
      font-weight: 500;
      margin: 0.5rem 0;
    }}
    .signal-green {{
      background: rgba(45, 106, 79, 0.15);
      color: var(--c4c-green);
    }}
    .signal-yellow {{
      background: rgba(235, 144, 1, 0.15);
      color: #b36d00;
    }}
    .signal-red {{
      background: rgba(207, 76, 56, 0.15);
      color: var(--c4c-red);
    }}
    code {{
      background: var(--bg);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.85em;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 2rem 0;
    }}
    footer {{
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
      text-align: center;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    footer a {{
      color: var(--c4c-indigo);
      text-decoration: none;
    }}
    footer a:hover {{ text-decoration: underline; }}
    strong {{ color: var(--text); }}
    @media print {{
      body {{ background: #fff; padding: 0; max-width: none; }}
      section {{ break-inside: avoid; border: none; box-shadow: none; }}
      .toc {{ break-after: page; }}
      .skip-link, .section-toggle {{ display: none; }}
      .health-banner, .callout, .callout-warning, .callout-success, .callout-info, .callout-decision, .decision-lens, .factors-positive, .factors-risk {{ 
        -webkit-print-color-adjust: exact; 
        print-color-adjust: exact; 
      }}
      .health-banner {{ background: var(--accent); }}
      a {{ text-decoration: none; color: inherit; }}
    }}
    @media screen {{
      .health-banner {{
        background: linear-gradient(135deg, var(--c4c-orange), #f5a623);
      }}
    }}
    @media (max-width: 600px) {{
      body {{ padding: 1rem; }}
      .toc ul {{ columns: 1; }}
      .health-banner {{ flex-direction: column; text-align: center; }}
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to report content</a>
  <header>
    <div class="logo">
      <img src="data:image/png;base64,{C4C_LOGO_BASE64}" alt="C4C">
      <span>NETWORK INSIGHT</span>
    </div>
    <h1>{project_name}</h1>
    <p class="subtitle">Network Insight Report</p>
    <p class="meta">Generated {date} • InsightGraph v{version}</p>
  </header>

{health_banner}

  <nav class="toc">
    <h2>Contents</h2>
    {toc}
  </nav>

  <main id="main">
    {content}
  </main>

  <footer>
    <p class="data-outputs">
      <strong>Data:</strong>
      <a href="data/nodes.csv">nodes.csv</a> •
      <a href="data/edges.csv">edges.csv</a> •
      <a href="data/grants_detail.csv">grants_detail.csv</a>
    </p>
    <p class="traceability">
      Project: {project_id} •
      <a href="manifest.json">View Manifest</a> •
      <a href="report.md">Source Markdown</a>
    </p>
    <p class="brand">Generated by <strong>C4C InsightGraph</strong> — Network Insight Platform</p>
  </footer>

  <script>
  (function () {{
    function setCollapsed(sectionEl, collapsed) {{
      sectionEl.dataset.collapsed = collapsed ? "true" : "false";
      var h2 = sectionEl.querySelector("h2");
      var btn = sectionEl.querySelector(".section-toggle");
      if (!h2 || !btn) return;

      // Hide all direct children except the H2
      Array.from(sectionEl.children).forEach(function(child) {{
        if (child === h2) return;
        child.style.display = collapsed ? "none" : "";
      }});

      btn.textContent = collapsed ? "Expand" : "Collapse";
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }}

    function ensureToggle(sectionEl) {{
      var h2 = sectionEl.querySelector("h2");
      if (!h2) return;

      // Prevent duplicate buttons if rerun
      if (h2.querySelector(".section-toggle")) return;

      var btn = document.createElement("button");
      btn.className = "section-toggle";
      btn.type = "button";
      btn.textContent = "Collapse";
      btn.setAttribute("aria-expanded", "true");

      btn.addEventListener("click", function(e) {{
        e.preventDefault();
        e.stopPropagation();
        var isCollapsed = sectionEl.dataset.collapsed === "true";
        setCollapsed(sectionEl, !isCollapsed);
      }});

      h2.appendChild(btn);
      sectionEl.dataset.collapsed = "false";
    }}

    // Initialize toggles
    document.querySelectorAll("section").forEach(function(sec) {{
      ensureToggle(sec);
    }});

    // If user clicks a TOC link to a collapsed section, expand it automatically
    function expandToHash() {{
      if (!location.hash) return;
      var target = document.querySelector(location.hash);
      if (!target) return;
      var sec = target.closest("section");
      if (!sec) return;
      setCollapsed(sec, false);
    }}

    window.addEventListener("hashchange", expandToHash);
    expandToHash();
  }})();
  </script>
</body>
</html>'''
    
    return html


# =============================================================================
# Main
# =============================================================================

def run(nodes_path, edges_path, output_dir, project_id="glfn"):
    """Main pipeline."""
    print("\n" + "="*60)
    print("C4C InsightGraph — Network Analysis")
    print("="*60 + "\n")
    
    nodes_df, edges_df = load_and_validate(nodes_path, edges_path)
    
    # Determine project directory for config loading
    project_dir = Path(nodes_path).parent
    
    print("\nBuilding graphs...")
    grant_graph = build_grant_graph(nodes_df, edges_df)
    board_graph = build_board_graph(nodes_df, edges_df)
    interlock_graph = build_interlock_graph(nodes_df, edges_df)
    
    print("\nComputing metrics...")
    metrics_df = compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
    metrics_df = compute_derived_signals(metrics_df)
    flow_stats = compute_flow_stats(edges_df, metrics_df)
    overlap_df = compute_portfolio_overlap(edges_df)
    
    # Compute Roles × Region Lens summary
    print("\nComputing Roles × Region Lens...")
    lens_config = load_region_lens_config(project_dir)
    
    # Derive network roles and compute lens membership
    nodes_with_roles = derive_network_roles(nodes_df.copy(), edges_df)
    nodes_with_lens = compute_region_lens_membership(nodes_with_roles, lens_config)
    roles_region_summary = generate_roles_region_summary(nodes_with_lens, edges_df, lens_config)
    
    print("\nGenerating insights...")
    insight_cards = generate_insight_cards(nodes_df, edges_df, metrics_df, interlock_graph, flow_stats, overlap_df, project_id)
    project_summary = generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
    
    # Add roles/region to project summary
    project_summary['roles_region'] = roles_region_summary
    
    # Generate markdown report
    markdown_report = generate_markdown_report(insight_cards, project_summary, project_id, roles_region_summary)
    
    print("\nWriting outputs...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_dir / "node_metrics.csv", index=False)
    with open(output_dir / "insight_cards.json", "w") as f:
        json.dump(insight_cards, f, indent=2)
    with open(output_dir / "project_summary.json", "w") as f:
        json.dump(project_summary, f, indent=2)
    with open(output_dir / "insight_report.md", "w") as f:
        f.write(markdown_report)
    
    print(f"\n✅ Done! Outputs in {output_dir}")
    return project_summary, markdown_report


def main():
    parser = argparse.ArgumentParser(description="C4C InsightGraph — Network Analysis & Briefing Generator")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project", type=str, default="glfn")
    args = parser.parse_args()
    
    summary, markdown_report = run(args.nodes, args.edges, args.out, args.project)
    print(f"\nNodes: {summary['node_counts']['total']}, Edges: {summary['edge_counts']['total']}")
    print(f"Funding: ${summary['funding']['total_amount']:,.0f}")
    print(f"Report: insight_report.md ({len(markdown_report)} chars)")


if __name__ == "__main__":
    main()
