"""
C4C Network Intelligence Engine — Phase 3

Computes network metrics, brokerage roles, and generates insight cards
from canonical nodes.csv and edges.csv.

Usage:
    python -m insights.run --nodes <path> --edges <path> --out <dir>
    
    Or with defaults (GLFN demo):
    python -m insights.run

VERSION HISTORY:
----------------
v3.0.5 (2025-12-21): Added Roles × Region Lens analysis
- NEW: Canonical role vocabulary (FUNDER, GRANTEE, FUNDER_GRANTEE, BOARD_MEMBER, ORGANIZATION, INDIVIDUAL)
- NEW: Region lens configuration (project_config.json or defaults)
- NEW: in_region_lens membership computed per node
- NEW: Roles × Lens cross-tabulation (counts by role, in/out of lens)
- NEW: Edge flow categories (IN→IN, IN→OUT, OUT→IN, OUT→OUT)
- NEW: Roles × Region Lens section in insight_report.md
- Supports Great Lakes (Binational) lens out of the box

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
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
            "Map potential overlap to identify natural coordination partners."
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
            f"This exceptional overlap signals strong funder consensus around their work — a natural coordination hub."
        )
        rec = "Convene their funders for joint impact measurement, aligned reporting, or a co-funding conversation. This grantee could anchor a funder learning community."
    elif funder_count >= 3:
        blurb = (
            f"**{label}** receives from {funder_count} funders (${total_received:,.0f} from {funders_str}). "
            f"This overlap suggests shared priorities and potential for deeper alignment."
        )
        rec = "Explore joint site visits, shared evaluation, or coordinated grant timing among these funders."
    else:
        blurb = (
            f"**{label}** receives from {funder_count} funders (${total_received:,.0f} from {funders_str}). "
            f"This natural alignment could seed deeper coordination."
        )
        rec = "Introduce these funders to explore whether their investments could be more intentionally aligned."
    
    return blurb, f"💡 **Opportunity:** {rec}"


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
# Strategic Recommendations Engine
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
    Generate rule-based strategic recommendations.
    Returns markdown string.
    """
    sections = []
    
    # Framing based on health
    if health_score >= 70:
        intro = (
            "The funding network shows **healthy coordination signals**. Focus on **deepening strategic "
            "relationships** and **protecting what works** rather than building basic connectivity."
        )
    elif health_score >= 40:
        intro = (
            "The network shows **mixed signals**. Some coordination exists, but structural gaps limit "
            "how effectively funders can align. Targeted interventions could unlock significant value."
        )
    else:
        intro = (
            "The network appears **fragmented**. Funders operate largely in silos with minimal coordination. "
            "Building basic connective tissue should be the priority."
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
            "**Nurture natural alignment:** A small number of shared grantees exist. Use these as anchors — "
            "convene funders around specific grantees to build relationships and explore joint action."
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
            "**Monitor concentration risk:** A handful of funders control nearly all capital. Track whether "
            "this creates field-shaping power that should be balanced with broader funder voice."
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
        mf_interpretation = f"Moderate — {multi_funder_pct:.1f}% have multiple funders, some natural coordination exists"
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
                f"This suggests funders are operating in near-complete silos with almost no shared investments."
            )
        elif overlap_pct < 5:
            overlap_emoji, overlap_label = "🟡", "Limited overlap"
            overlap_narrative = (
                f"**{len(multi_funder)} grantees** ({overlap_pct:.1f}%) receive from 2+ funders. Some natural alignment exists, "
                f"but most grantees depend on a single funder. These shared grantees represent organic coordination points."
            )
        else:
            overlap_emoji, overlap_label = "🟢", "Meaningful overlap"
            overlap_narrative = (
                f"**{len(multi_funder)} grantees** ({overlap_pct:.1f}%) receive from multiple funders. "
                f"This overlap suggests shared priorities and real potential for deeper coordination."
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
    if not overlap_df.empty:
        top = overlap_df.iloc[0]
        top_jaccard = top['jaccard_similarity']
        
        if top_jaccard >= 0.3:
            twin_emoji, twin_label = "🟢", "Strong alignment found"
            twin_narrative = (
                f"**{len(overlap_df)} funder pairs** share at least one grantee. The most aligned — "
                f"{node_labels.get(top['funder_1'], '')} & {node_labels.get(top['funder_2'], '')} — "
                f"share {int(top['shared_grantees'])} grantees (Jaccard: {top_jaccard:.2f}). "
                f"This level of overlap suggests natural partnership potential."
            )
        elif top_jaccard >= 0.1:
            twin_emoji, twin_label = "🟡", "Moderate alignment"
            twin_narrative = (
                f"**{len(overlap_df)} funder pairs** share grantees. Top pair: "
                f"{node_labels.get(top['funder_1'], '')} & {node_labels.get(top['funder_2'], '')} "
                f"({int(top['shared_grantees'])} shared, Jaccard: {top_jaccard:.2f})."
            )
        else:
            twin_emoji, twin_label = "⚪", "Weak alignment"
            twin_narrative = f"Some portfolio overlap exists, but even the closest funder pairs share few grantees."
        
        ranked_rows = []
        for _, r in overlap_df.head(5).iterrows():
            f1, f2 = node_labels.get(r['funder_1'], ''), node_labels.get(r['funder_2'], '')
            ranked_rows.append({
                "pair": f"{f1} & {f2}",
                "shared": int(r['shared_grantees']),
                "jaccard": r['jaccard_similarity'],
                "narrative": f"These funders share {int(r['shared_grantees'])} grantees — natural partners for coordination."
            })
        
        cards.append({
            "card_id": "portfolio_twins",
            "use_case": "Funding Concentration",
            "title": "Portfolio Twins",
            "summary": f"{twin_emoji} **{twin_label}**\n\n{twin_narrative}\n\n💡 **Opportunity:** Funder pairs with high overlap could pilot joint reporting, co-investment, or aligned grantmaking.",
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
            "summary": f"{disc_emoji} **{disc_label}**\n\n{disc_narrative}\n\n💡 **Opportunity:** Consider introductions between isolated funders with aligned portfolios.",
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
        "card_id": "strategic_recommendations",
        "use_case": "Strategic Recommendations",
        "title": "Strategic Recommendations",
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
        # Check region/state/province column
        region = str(row.get('region', '')).strip().upper()
        
        # Handle common variations
        if region in all_regions:
            return True
        
        # Check 2-letter codes
        if len(region) == 2:
            return region in all_regions
        
        # Check full names (Ontario -> ON, etc.)
        region_map = {
            'ONTARIO': 'ON', 'QUEBEC': 'QC', 'MICHIGAN': 'MI', 'OHIO': 'OH',
            'MINNESOTA': 'MN', 'WISCONSIN': 'WI', 'INDIANA': 'IN', 'ILLINOIS': 'IL',
            'NEW YORK': 'NY', 'PENNSYLVANIA': 'PA'
        }
        mapped = region_map.get(region, region)
        return mapped in all_regions
    
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


def format_roles_region_section(summary: dict) -> list:
    """
    Format Roles × Region Lens summary as markdown lines.
    """
    if not summary.get('enabled', False):
        return []
    
    lines = []
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
    
    # Health overview
    health = insight_cards.get("health", {})
    health_score = health.get("score", 0)
    health_label = health.get("label", "Unknown")
    
    if health_score >= 70:
        health_emoji = "🟢"
    elif health_score >= 40:
        health_emoji = "🟡"
    else:
        health_emoji = "🔴"
    
    lines.append(f"## {health_emoji} Network Health: {health_score}/100 ({health_label})")
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
    positive = health.get("positive", [])
    risk = health.get("risk", [])
    
    if positive:
        lines.append("### ✅ Positive Factors")
        lines.append("")
        for f in positive:
            lines.append(f"- {f}")
        lines.append("")
    
    if risk:
        lines.append("### ⚠️ Risk Factors")
        lines.append("")
        for f in risk:
            lines.append(f"- {f}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Roles × Region Lens section (if available)
    if roles_region_summary and roles_region_summary.get('enabled', False):
        region_lines = format_roles_region_section(roles_region_summary)
        lines.extend(region_lines)
        lines.append("---")
        lines.append("")
    
    # Each card
    cards = insight_cards.get("cards", [])
    
    for card in cards:
        card_id = card.get("card_id", "")
        
        # Skip network_health card (already rendered above)
        if card_id == "network_health":
            continue
        
        title = card.get("title", "Untitled")
        use_case = card.get("use_case", "")
        summary_text = card.get("summary", "")
        
        lines.append(f"## {title}")
        lines.append(f"*Use Case: {use_case}*")
        lines.append("")
        lines.append(summary_text)
        lines.append("")
        
        # Ranked rows
        ranked_rows = card.get("ranked_rows", [])
        if ranked_rows:
            # Check if rows have narratives
            has_narratives = any(r.get("narrative") for r in ranked_rows)
            
            if has_narratives:
                for row in ranked_rows:
                    rank = row.get("rank", "")
                    entity = (
                        row.get("grantee") or 
                        row.get("person") or 
                        row.get("org") or 
                        row.get("funder") or 
                        row.get("node") or 
                        row.get("pair") or
                        ""
                    )
                    
                    if rank and entity:
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
    lines.append("*Report generated by C4C Network Insight Engine*")
    
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


# =============================================================================
# Main
# =============================================================================

def run(nodes_path, edges_path, output_dir, project_id="glfn"):
    """Main pipeline."""
    print("\n" + "="*60)
    print("C4C Network Insight Engine — Phase 3")
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
    parser = argparse.ArgumentParser(description="C4C Network Insight Engine")
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
