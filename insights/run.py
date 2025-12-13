"""
C4C Network Intelligence Engine — Phase 3

Computes network metrics, brokerage roles, and generates insight cards
from canonical nodes.csv and edges.csv.

Usage:
    python -m insights.run --nodes <path> --edges <path> --out <dir>
    
    Or with defaults (GLFN demo):
    python -m insights.run
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

# Percentile thresholds for role detection
CONNECTOR_THRESHOLD = 75
BROKER_THRESHOLD = 75
HIDDEN_BROKER_DEGREE_CAP = 40
CAPITAL_HUB_THRESHOLD = 75


# =============================================================================
# Data Loading & Validation
# =============================================================================

def load_and_validate(nodes_path: Path, edges_path: Path) -> tuple:
    """
    Load canonical CSVs and validate required columns.
    Fails fast with readable error messages.
    """
    # Check files exist
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")
    
    # Load CSVs
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    # Validate required columns
    required_node_cols = {"node_id", "node_type", "label"}
    missing_node_cols = required_node_cols - set(nodes_df.columns)
    if missing_node_cols:
        raise ValueError(f"nodes.csv missing required columns: {missing_node_cols}")
    
    required_edge_cols = {"edge_id", "edge_type", "from_id", "to_id"}
    missing_edge_cols = required_edge_cols - set(edges_df.columns)
    if missing_edge_cols:
        raise ValueError(f"edges.csv missing required columns: {missing_edge_cols}")
    
    # Validate node_type values
    valid_types = {"ORG", "PERSON"}
    actual_types = set(nodes_df["node_type"].dropna().unique())
    invalid_types = actual_types - valid_types
    if invalid_types:
        print(f"Warning: Unexpected node_type values: {invalid_types}")
    
    # Validate edge_type values
    valid_edge_types = {"GRANT", "BOARD_MEMBERSHIP"}
    actual_edge_types = set(edges_df["edge_type"].dropna().unique())
    invalid_edge_types = actual_edge_types - valid_edge_types
    if invalid_edge_types:
        print(f"Warning: Unexpected edge_type values: {invalid_edge_types}")
    
    print(f"✓ Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    return nodes_df, edges_df


# =============================================================================
# Graph Construction
# =============================================================================

def build_grant_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed grant graph: ORG → ORG, weighted by amount.
    """
    G = nx.DiGraph()
    
    # Add all ORG nodes
    org_nodes = nodes_df[nodes_df["node_type"] == "ORG"]
    for _, row in org_nodes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    # Add grant edges
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"]
    for _, row in grant_edges.iterrows():
        from_id = row["from_id"]
        to_id = row["to_id"]
        amount = row.get("amount", 0) or 0
        
        if G.has_edge(from_id, to_id):
            # Aggregate multiple grants
            G[from_id][to_id]["weight"] += float(amount)
            G[from_id][to_id]["grant_count"] += 1
        else:
            G.add_edge(from_id, to_id, weight=float(amount), grant_count=1)
    
    print(f"✓ Grant graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_board_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build bipartite board graph: PERSON — ORG (undirected).
    """
    G = nx.Graph()
    
    # Add all nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    # Add board membership edges
    board_edges = edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]
    for _, row in board_edges.iterrows():
        from_id = row["from_id"]  # PERSON
        to_id = row["to_id"]      # ORG
        G.add_edge(from_id, to_id, edge_type="BOARD_MEMBERSHIP")
    
    print(f"✓ Board graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_interlock_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build ORG—ORG interlock graph, weighted by number of shared board members.
    """
    G = nx.Graph()
    
    # Add ORG nodes
    org_nodes = nodes_df[nodes_df["node_type"] == "ORG"]
    for _, row in org_nodes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    # Build person → orgs mapping
    board_edges = edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"]
    person_to_orgs = defaultdict(set)
    for _, row in board_edges.iterrows():
        person_id = row["from_id"]
        org_id = row["to_id"]
        person_to_orgs[person_id].add(org_id)
    
    # Create edges between orgs that share board members
    interlock_weights = defaultdict(lambda: {"weight": 0, "shared_people": []})
    for person_id, orgs in person_to_orgs.items():
        orgs_list = list(orgs)
        for i, org1 in enumerate(orgs_list):
            for org2 in orgs_list[i+1:]:
                key = tuple(sorted([org1, org2]))
                interlock_weights[key]["weight"] += 1
                interlock_weights[key]["shared_people"].append(person_id)
    
    # Add edges to graph
    for (org1, org2), data in interlock_weights.items():
        G.add_edge(org1, org2, weight=data["weight"], shared_people=data["shared_people"])
    
    print(f"✓ Interlock graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (org pairs with shared board members)")
    return G


# =============================================================================
# Layer 1: Base Metrics
# =============================================================================

def compute_base_metrics(
    nodes_df: pd.DataFrame,
    grant_graph: nx.DiGraph,
    board_graph: nx.Graph,
    interlock_graph: nx.Graph
) -> pd.DataFrame:
    """
    Compute base metrics for all nodes.
    Returns DataFrame with one row per node.
    """
    metrics = []
    
    # Compute centrality metrics on graphs
    # Grant graph metrics (for ORGs)
    grant_betweenness = nx.betweenness_centrality(grant_graph, weight=None) if grant_graph.number_of_edges() > 0 else {}
    grant_pagerank = nx.pagerank(grant_graph, weight="weight") if grant_graph.number_of_edges() > 0 else {}
    
    # Board graph metrics (for both)
    board_betweenness = nx.betweenness_centrality(board_graph) if board_graph.number_of_edges() > 0 else {}
    
    # Interlock graph metrics (for ORGs)
    interlock_betweenness = nx.betweenness_centrality(interlock_graph, weight=None) if interlock_graph.number_of_edges() > 0 else {}
    
    # Connected components (grant graph)
    if grant_graph.number_of_nodes() > 0:
        grant_undirected = grant_graph.to_undirected()
        components = list(nx.connected_components(grant_undirected))
        node_to_component = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_component[node] = i
    else:
        node_to_component = {}
    
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
            # Grant graph metrics
            m["degree"] = grant_graph.degree(node_id) if node_id in grant_graph else 0
            m["grant_in_degree"] = grant_graph.in_degree(node_id) if node_id in grant_graph else 0
            m["grant_out_degree"] = grant_graph.out_degree(node_id) if node_id in grant_graph else 0
            
            # Weighted outflow
            outflow = 0
            if node_id in grant_graph:
                for _, _, data in grant_graph.out_edges(node_id, data=True):
                    outflow += data.get("weight", 0)
            m["grant_outflow_total"] = outflow
            
            # Centrality
            m["betweenness"] = grant_betweenness.get(node_id, 0)
            m["pagerank"] = grant_pagerank.get(node_id, 0)
            m["component_id"] = node_to_component.get(node_id, -1)
            
            # Interlock metrics
            m["shared_board_count"] = interlock_graph.degree(node_id) if node_id in interlock_graph else 0
            shared_people_total = 0
            if node_id in interlock_graph:
                for _, _, data in interlock_graph.edges(node_id, data=True):
                    shared_people_total += data.get("weight", 0)
            m["shared_board_people_total"] = shared_people_total
            
            # Board metrics (N/A for ORG)
            m["boards_served"] = None
            
        elif node_type == "PERSON":
            # Board count
            boards_served = board_graph.degree(node_id) if node_id in board_graph else 0
            m["boards_served"] = boards_served
            m["degree"] = boards_served
            
            # Betweenness in board graph
            m["betweenness"] = board_betweenness.get(node_id, 0)
            
            # N/A for PERSON
            m["grant_in_degree"] = None
            m["grant_out_degree"] = None
            m["grant_outflow_total"] = None
            m["pagerank"] = None
            m["component_id"] = None
            m["shared_board_count"] = None
            m["shared_board_people_total"] = None
        
        metrics.append(m)
    
    return pd.DataFrame(metrics)


# =============================================================================
# Layer 2: Derived Signals (Brokerage Roles)
# =============================================================================

def compute_derived_signals(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived boolean flags based on percentile thresholds.
    """
    df = metrics_df.copy()
    
    # Initialize flags
    df["is_connector"] = 0
    df["is_broker"] = 0
    df["is_hidden_broker"] = 0
    df["is_capital_hub"] = 0
    df["is_isolated"] = 0
    
    # Compute thresholds for ORG nodes only
    org_mask = df["node_type"] == "ORG"
    org_df = df[org_mask]
    
    if len(org_df) > 0:
        # Degree percentiles
        degree_75 = np.percentile(org_df["degree"].dropna(), CONNECTOR_THRESHOLD)
        degree_40 = np.percentile(org_df["degree"].dropna(), HIDDEN_BROKER_DEGREE_CAP)
        
        # Betweenness percentiles
        betweenness_75 = np.percentile(org_df["betweenness"].dropna(), BROKER_THRESHOLD)
        
        # Outflow percentiles
        outflow_vals = org_df["grant_outflow_total"].dropna()
        if len(outflow_vals) > 0:
            outflow_75 = np.percentile(outflow_vals, CAPITAL_HUB_THRESHOLD)
        else:
            outflow_75 = 0
        
        # Apply flags (ORG only)
        df.loc[org_mask & (df["degree"] >= degree_75), "is_connector"] = 1
        df.loc[org_mask & (df["betweenness"] >= betweenness_75), "is_broker"] = 1
        df.loc[org_mask & (df["betweenness"] >= betweenness_75) & (df["degree"] <= degree_40), "is_hidden_broker"] = 1
        df.loc[org_mask & (df["grant_outflow_total"] >= outflow_75) & (df["grant_outflow_total"] > 0), "is_capital_hub"] = 1
        df.loc[org_mask & (df["degree"] == 1), "is_isolated"] = 1
    
    # PERSON role: multi-board connector
    person_mask = df["node_type"] == "PERSON"
    df.loc[person_mask & (df["boards_served"] >= 2), "is_connector"] = 1
    
    return df


# =============================================================================
# Layer 3: System-Level Flow Stats
# =============================================================================

def compute_flow_stats(edges_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
    """
    Compute system-level funding flow statistics.
    """
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy()
    
    if grant_edges.empty:
        return {
            "total_grant_amount": 0,
            "grant_count": 0,
            "funder_count": 0,
            "grantee_count": 0,
            "top_5_funders_share": 0,
            "top_10_grantees_share": 0,
        }
    
    # Ensure amount is numeric
    grant_edges["amount"] = pd.to_numeric(grant_edges["amount"], errors="coerce").fillna(0)
    
    total_amount = grant_edges["amount"].sum()
    
    # Funder stats
    funder_totals = grant_edges.groupby("from_id")["amount"].sum().sort_values(ascending=False)
    funder_count = len(funder_totals)
    top_5_funders_amount = funder_totals.head(5).sum()
    top_5_funders_share = (top_5_funders_amount / total_amount * 100) if total_amount > 0 else 0
    
    # Grantee stats
    grantee_totals = grant_edges.groupby("to_id")["amount"].sum().sort_values(ascending=False)
    grantee_count = len(grantee_totals)
    top_10_grantees_amount = grantee_totals.head(10).sum()
    top_10_grantees_share = (top_10_grantees_amount / total_amount * 100) if total_amount > 0 else 0
    
    # Grantees by funder count
    grantee_funder_counts = grant_edges.groupby("to_id")["from_id"].nunique()
    multi_funder_grantees = (grantee_funder_counts >= 2).sum()
    
    return {
        "total_grant_amount": float(total_amount),
        "grant_count": len(grant_edges),
        "funder_count": funder_count,
        "grantee_count": grantee_count,
        "top_5_funders_share": round(top_5_funders_share, 1),
        "top_10_grantees_share": round(top_10_grantees_share, 1),
        "multi_funder_grantees": int(multi_funder_grantees),
    }


def compute_portfolio_overlap(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute funder × funder portfolio overlap matrix.
    Returns DataFrame with overlap scores.
    """
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy()
    
    if grant_edges.empty:
        return pd.DataFrame()
    
    # Build funder → grantees mapping
    funder_grantees = grant_edges.groupby("from_id")["to_id"].apply(set).to_dict()
    
    funders = list(funder_grantees.keys())
    overlaps = []
    
    for i, f1 in enumerate(funders):
        for f2 in funders[i+1:]:
            g1 = funder_grantees[f1]
            g2 = funder_grantees[f2]
            shared = g1 & g2
            if shared:
                # Jaccard similarity
                jaccard = len(shared) / len(g1 | g2)
                overlaps.append({
                    "funder_1": f1,
                    "funder_2": f2,
                    "shared_grantees": len(shared),
                    "jaccard_similarity": round(jaccard, 3),
                    "shared_grantee_ids": list(shared),
                })
    
    return pd.DataFrame(overlaps).sort_values("shared_grantees", ascending=False)


# =============================================================================
# Layer 4: Insight Cards
# =============================================================================

def generate_insight_cards(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    interlock_graph: nx.Graph,
    flow_stats: dict,
    overlap_df: pd.DataFrame,
    project_id: str = "glfn"
) -> dict:
    """
    Generate pre-interpreted insight cards.
    Returns full insight cards document with schema metadata.
    """
    cards = []
    
    # Helper to get label from node_id
    node_labels = dict(zip(nodes_df["node_id"], nodes_df["label"]))
    
    # Get grant and board edges
    grant_edges = edges_df[edges_df["edge_type"] == "GRANT"].copy() if not edges_df.empty else pd.DataFrame()
    board_edges = edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"].copy() if not edges_df.empty else pd.DataFrame()
    
    # Ensure amount is numeric
    if not grant_edges.empty and "amount" in grant_edges.columns:
        grant_edges["amount"] = pd.to_numeric(grant_edges["amount"], errors="coerce").fillna(0)
    
    # Pre-compute grantee funder counts
    grantee_funders = pd.Series(dtype=int)
    if not grant_edges.empty:
        grantee_funders = grant_edges.groupby("to_id")["from_id"].nunique()
    
    # -------------------------------------------------------------------------
    # Card 1: Concentration Snapshot (Must-Have #1)
    # -------------------------------------------------------------------------
    cards.append({
        "card_id": "concentration_snapshot",
        "use_case": "Funding Concentration",
        "title": "Concentration Snapshot",
        "summary": f"Top 5 funders account for {flow_stats['top_5_funders_share']}% of total funding (${flow_stats['total_grant_amount']:,.0f}). {flow_stats['multi_funder_grantees']} grantees receive from multiple funders.",
        "ranked_rows": [
            {"rank": 1, "metric": "Total Grant Volume", "value": f"${flow_stats['total_grant_amount']:,.0f}"},
            {"rank": 2, "metric": "Unique Funders", "value": str(flow_stats['funder_count'])},
            {"rank": 3, "metric": "Unique Grantees", "value": str(flow_stats['grantee_count'])},
            {"rank": 4, "metric": "Top 5 Funders Share", "value": f"{flow_stats['top_5_funders_share']}%"},
            {"rank": 5, "metric": "Multi-Funder Grantees", "value": str(flow_stats['multi_funder_grantees'])},
        ],
        "evidence": {
            "node_ids": [],
            "edge_ids": [],
        },
    })
    
    # -------------------------------------------------------------------------
    # Card 2: Funder Overlap Clusters (Must-Have #2)
    # -------------------------------------------------------------------------
    if not grantee_funders.empty:
        multi_funder = grantee_funders[grantee_funders >= 2].sort_values(ascending=False)
        
        ranked_rows = []
        evidence_node_ids = []
        evidence_edge_ids = []
        
        for rank, grantee_id in enumerate(multi_funder.head(10).index, 1):
            funder_count = int(multi_funder[grantee_id])
            funder_ids = grant_edges[grant_edges["to_id"] == grantee_id]["from_id"].unique().tolist()
            funder_labels = [node_labels.get(f, f) for f in funder_ids]
            total_received = grant_edges[grant_edges["to_id"] == grantee_id]["amount"].sum()
            
            # Get edge IDs for evidence
            edge_ids = grant_edges[grant_edges["to_id"] == grantee_id]["edge_id"].tolist()
            
            ranked_rows.append({
                "rank": rank,
                "grantee_id": grantee_id,
                "grantee_label": node_labels.get(grantee_id, grantee_id),
                "funder_ids": funder_ids,
                "funder_labels": funder_labels,
                "funder_count": funder_count,
                "total_received": float(total_received),
            })
            evidence_node_ids.append(grantee_id)
            evidence_node_ids.extend(funder_ids)
            evidence_edge_ids.extend(edge_ids)
        
        top_label = ranked_rows[0]["grantee_label"] if ranked_rows else "N/A"
        top_count = ranked_rows[0]["funder_count"] if ranked_rows else 0
        
        cards.append({
            "card_id": "funder_overlap_clusters",
            "use_case": "Funder Flow",
            "title": "Funder Overlap Clusters",
            "summary": f"{len(multi_funder)} grantees receive funding from 2+ network members. Top: {top_label} ({top_count} funders).",
            "ranked_rows": ranked_rows,
            "evidence": {
                "node_ids": list(set(evidence_node_ids)),
                "edge_ids": list(set(evidence_edge_ids)),
            },
        })
    else:
        cards.append({
            "card_id": "funder_overlap_clusters",
            "use_case": "Funder Flow",
            "title": "Funder Overlap Clusters",
            "summary": "No grantees receive funding from multiple network members.",
            "ranked_rows": [],
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # -------------------------------------------------------------------------
    # Card 3: Portfolio Twins (Must-Have #3)
    # -------------------------------------------------------------------------
    if not overlap_df.empty:
        ranked_rows = []
        evidence_node_ids = []
        
        for rank, (_, row) in enumerate(overlap_df.head(10).iterrows(), 1):
            ranked_rows.append({
                "rank": rank,
                "funder_1_id": row["funder_1"],
                "funder_1_label": node_labels.get(row["funder_1"], row["funder_1"]),
                "funder_2_id": row["funder_2"],
                "funder_2_label": node_labels.get(row["funder_2"], row["funder_2"]),
                "shared_grantees": int(row["shared_grantees"]),
                "jaccard_similarity": float(row["jaccard_similarity"]),
                "shared_grantee_ids": row["shared_grantee_ids"],
            })
            evidence_node_ids.extend([row["funder_1"], row["funder_2"]])
            evidence_node_ids.extend(row["shared_grantee_ids"])
        
        top_pair = f"{ranked_rows[0]['funder_1_label']} & {ranked_rows[0]['funder_2_label']}" if ranked_rows else "N/A"
        top_shared = ranked_rows[0]["shared_grantees"] if ranked_rows else 0
        
        cards.append({
            "card_id": "portfolio_twins",
            "use_case": "Funding Concentration",
            "title": "Portfolio Twins",
            "summary": f"{len(overlap_df)} funder pairs share grantees. Most aligned: {top_pair} ({top_shared} shared).",
            "ranked_rows": ranked_rows,
            "evidence": {
                "node_ids": list(set(evidence_node_ids)),
                "edge_ids": [],
            },
        })
    else:
        cards.append({
            "card_id": "portfolio_twins",
            "use_case": "Funding Concentration",
            "title": "Portfolio Twins",
            "summary": "No funder pairs share grantees in this network.",
            "ranked_rows": [],
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # -------------------------------------------------------------------------
    # Card 4: Orphaned Grantees (Must-Have #4)
    # -------------------------------------------------------------------------
    if not grantee_funders.empty:
        single_funder = grantee_funders[grantee_funders == 1]
        
        ranked_rows = []
        evidence_node_ids = []
        evidence_edge_ids = []
        
        if not single_funder.empty:
            # Sort by amount received
            grantee_amounts = grant_edges.groupby("to_id")["amount"].sum()
            orphaned_sorted = single_funder.to_frame("count").join(grantee_amounts.rename("amount"))
            orphaned_sorted = orphaned_sorted.sort_values("amount", ascending=False)
            
            for rank, grantee_id in enumerate(orphaned_sorted.head(10).index, 1):
                funder_id = grant_edges[grant_edges["to_id"] == grantee_id]["from_id"].iloc[0]
                edge_ids = grant_edges[grant_edges["to_id"] == grantee_id]["edge_id"].tolist()
                amount = orphaned_sorted.loc[grantee_id, "amount"]
                
                ranked_rows.append({
                    "rank": rank,
                    "grantee_id": grantee_id,
                    "grantee_label": node_labels.get(grantee_id, grantee_id),
                    "funder_id": funder_id,
                    "funder_label": node_labels.get(funder_id, funder_id),
                    "amount_received": float(amount),
                })
                evidence_node_ids.extend([grantee_id, funder_id])
                evidence_edge_ids.extend(edge_ids)
        
        cards.append({
            "card_id": "orphaned_grantees",
            "use_case": "Funder Flow",
            "title": "Orphaned Grantees",
            "summary": f"{len(single_funder)} grantees have only one funder in the network — potential coordination opportunities.",
            "ranked_rows": ranked_rows,
            "evidence": {
                "node_ids": list(set(evidence_node_ids)),
                "edge_ids": list(set(evidence_edge_ids)),
            },
        })
    else:
        cards.append({
            "card_id": "orphaned_grantees",
            "use_case": "Funder Flow",
            "title": "Orphaned Grantees",
            "summary": "No grant data available to identify orphaned grantees.",
            "ranked_rows": [],
            "evidence": {"node_ids": [], "edge_ids": []},
        })
    
    # -------------------------------------------------------------------------
    # Card 5: Shared Board Conduits (Must-Have #5)
    # -------------------------------------------------------------------------
    person_metrics = metrics_df[metrics_df["node_type"] == "PERSON"]
    multi_board = person_metrics[person_metrics["boards_served"] >= 2].sort_values("boards_served", ascending=False)
    
    ranked_rows = []
    evidence_node_ids = []
    evidence_edge_ids = []
    
    if not multi_board.empty and not board_edges.empty:
        person_orgs = board_edges.groupby("from_id")["to_id"].apply(list).to_dict()
        person_edge_ids = board_edges.groupby("from_id")["edge_id"].apply(list).to_dict()
        
        for rank, (_, row) in enumerate(multi_board.head(10).iterrows(), 1):
            person_id = row["node_id"]
            org_ids = person_orgs.get(person_id, [])
            org_labels = [node_labels.get(o, o) for o in org_ids]
            edge_ids = person_edge_ids.get(person_id, [])
            
            ranked_rows.append({
                "rank": rank,
                "person_id": person_id,
                "person_label": row["label"],
                "org_ids": org_ids,
                "org_labels": org_labels,
                "board_count": int(row["boards_served"]),
            })
            evidence_node_ids.append(person_id)
            evidence_node_ids.extend(org_ids)
            evidence_edge_ids.extend(edge_ids)
    
    top_person = ranked_rows[0]["person_label"] if ranked_rows else "N/A"
    top_orgs = ", ".join(ranked_rows[0]["org_labels"][:3]) if ranked_rows else "N/A"
    
    cards.append({
        "card_id": "shared_board_conduits",
        "use_case": "Board Network & Conduits",
        "title": "Shared Board Conduits",
        "summary": f"{len(multi_board)} individuals sit on 2+ boards. Top conduit: {top_person} links {top_orgs}.",
        "ranked_rows": ranked_rows,
        "evidence": {
            "node_ids": list(set(evidence_node_ids)),
            "edge_ids": list(set(evidence_edge_ids)),
        },
    })
    
    # -------------------------------------------------------------------------
    # Card 6: Foundations with No Board Interlocks (Must-Have #6)
    # -------------------------------------------------------------------------
    org_metrics = metrics_df[metrics_df["node_type"] == "ORG"]
    foundations = org_metrics[org_metrics["grant_outflow_total"] > 0]  # Funders
    disconnected = foundations[foundations["shared_board_count"] == 0]
    
    ranked_rows = []
    evidence_node_ids = []
    
    for rank, (_, row) in enumerate(disconnected.sort_values("grant_outflow_total", ascending=False).head(10).iterrows(), 1):
        ranked_rows.append({
            "rank": rank,
            "org_id": row["node_id"],
            "org_label": row["label"],
            "grant_outflow": float(row["grant_outflow_total"]),
        })
        evidence_node_ids.append(row["node_id"])
    
    cards.append({
        "card_id": "no_board_interlocks",
        "use_case": "Board Network & Conduits",
        "title": "Foundations with No Board Interlocks",
        "summary": f"{len(disconnected)} funders have no shared board members with other network foundations.",
        "ranked_rows": ranked_rows,
        "evidence": {
            "node_ids": evidence_node_ids,
            "edge_ids": [],
        },
    })
    
    # -------------------------------------------------------------------------
    # Card 7: Hidden Brokers (Must-Have #7)
    # -------------------------------------------------------------------------
    hidden_brokers = metrics_df[(metrics_df["is_hidden_broker"] == 1) & (metrics_df["node_type"] == "ORG")]
    
    ranked_rows = []
    evidence_node_ids = []
    
    for rank, (_, row) in enumerate(hidden_brokers.sort_values("betweenness", ascending=False).head(10).iterrows(), 1):
        ranked_rows.append({
            "rank": rank,
            "org_id": row["node_id"],
            "org_label": row["label"],
            "betweenness": round(float(row["betweenness"]), 4),
            "degree": int(row["degree"]),
        })
        evidence_node_ids.append(row["node_id"])
    
    top_broker = ranked_rows[0]["org_label"] if ranked_rows else "N/A"
    
    cards.append({
        "card_id": "hidden_brokers",
        "use_case": "Brokerage Roles",
        "title": "Hidden Brokers",
        "summary": f"{len(hidden_brokers)} organizations have high betweenness but low visibility. Top: {top_broker}.",
        "ranked_rows": ranked_rows,
        "evidence": {
            "node_ids": evidence_node_ids,
            "edge_ids": [],
        },
    })
    
    # -------------------------------------------------------------------------
    # Card 8: Single-Point Bridges (Must-Have #8)
    # -------------------------------------------------------------------------
    grant_graph_undirected = nx.Graph()
    if not grant_edges.empty:
        for _, row in grant_edges.iterrows():
            grant_graph_undirected.add_edge(row["from_id"], row["to_id"])
    
    articulation_points = []
    if grant_graph_undirected.number_of_edges() > 0:
        articulation_points = list(nx.articulation_points(grant_graph_undirected))
    
    # Filter to nodes in our metrics
    ap_in_network = [ap for ap in articulation_points if ap in metrics_df["node_id"].values]
    
    ranked_rows = []
    evidence_node_ids = []
    
    for rank, ap in enumerate(ap_in_network[:10], 1):
        row = metrics_df[metrics_df["node_id"] == ap].iloc[0]
        ranked_rows.append({
            "rank": rank,
            "node_id": ap,
            "node_label": node_labels.get(ap, ap),
            "node_type": row["node_type"],
            "degree": int(row["degree"]) if pd.notna(row["degree"]) else 0,
        })
        evidence_node_ids.append(ap)
    
    top_bridge = ranked_rows[0]["node_label"] if ranked_rows else "N/A"
    
    cards.append({
        "card_id": "single_point_bridges",
        "use_case": "Brokerage Roles",
        "title": "Single-Point Bridges",
        "summary": f"{len(ap_in_network)} nodes are critical bridges — removal would fragment the network. Top: {top_bridge}.",
        "ranked_rows": ranked_rows,
        "evidence": {
            "node_ids": evidence_node_ids,
            "edge_ids": [],
        },
    })
    
    # -------------------------------------------------------------------------
    # Return full document
    # -------------------------------------------------------------------------
    return {
        "schema_version": "1.0-mvp",
        "project_id": project_id,
        "generated_at": datetime.now().isoformat() + "Z",
        "cards": cards,
    }


# =============================================================================
# Project Summary
# =============================================================================

def generate_project_summary(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    flow_stats: dict
) -> dict:
    """
    Generate top-level project summary for demo framing.
    """
    org_count = len(nodes_df[nodes_df["node_type"] == "ORG"])
    person_count = len(nodes_df[nodes_df["node_type"] == "PERSON"])
    grant_count = len(edges_df[edges_df["edge_type"] == "GRANT"])
    board_count = len(edges_df[edges_df["edge_type"] == "BOARD_MEMBERSHIP"])
    
    # Foundations (orgs that give grants)
    funders = metrics_df[(metrics_df["node_type"] == "ORG") & (metrics_df["grant_outflow_total"] > 0)]
    
    # Multi-board people
    multi_board = metrics_df[(metrics_df["node_type"] == "PERSON") & (metrics_df["boards_served"] >= 2)]
    
    return {
        "generated_at": datetime.now().isoformat(),
        "node_counts": {
            "total": len(nodes_df),
            "organizations": org_count,
            "people": person_count,
        },
        "edge_counts": {
            "total": len(edges_df),
            "grants": grant_count,
            "board_memberships": board_count,
        },
        "funding": {
            "total_amount": flow_stats["total_grant_amount"],
            "funder_count": flow_stats["funder_count"],
            "grantee_count": flow_stats["grantee_count"],
            "top_5_share": flow_stats["top_5_funders_share"],
        },
        "governance": {
            "multi_board_people": len(multi_board),
            "foundations_with_interlocks": len(funders[funders["shared_board_count"] > 0]),
            "isolated_foundations": len(funders[funders["shared_board_count"] == 0]),
        },
    }


# =============================================================================
# Output Writers
# =============================================================================

def write_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    insight_cards_doc: dict,
    project_summary: dict
) -> None:
    """
    Write all outputs to the specified directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # node_metrics.csv
    metrics_path = output_dir / "node_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Wrote {metrics_path}")
    
    # insight_cards.json (full document with schema metadata)
    cards_path = output_dir / "insight_cards.json"
    with open(cards_path, "w") as f:
        json.dump(insight_cards_doc, f, indent=2)
    print(f"✓ Wrote {cards_path}")
    
    # project_summary.json
    summary_path = output_dir / "project_summary.json"
    with open(summary_path, "w") as f:
        json.dump(project_summary, f, indent=2)
    print(f"✓ Wrote {summary_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run(nodes_path: Path, edges_path: Path, output_dir: Path, project_id: str = "glfn") -> dict:
    """
    Main pipeline: load → compute → generate → write.
    Returns project summary for programmatic use.
    """
    print("\n" + "="*60)
    print("C4C Network Intelligence Engine — Phase 3")
    print("="*60 + "\n")
    
    # Load and validate
    print("Loading data...")
    nodes_df, edges_df = load_and_validate(nodes_path, edges_path)
    
    # Build graphs
    print("\nBuilding graphs...")
    grant_graph = build_grant_graph(nodes_df, edges_df)
    board_graph = build_board_graph(nodes_df, edges_df)
    interlock_graph = build_interlock_graph(nodes_df, edges_df)
    
    # Compute metrics
    print("\nComputing Layer 1 base metrics...")
    metrics_df = compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
    
    print("Computing Layer 2 derived signals...")
    metrics_df = compute_derived_signals(metrics_df)
    
    print("Computing flow statistics...")
    flow_stats = compute_flow_stats(edges_df, metrics_df)
    overlap_df = compute_portfolio_overlap(edges_df)
    
    # Generate insights
    print("\nGenerating insight cards...")
    insight_cards_doc = generate_insight_cards(
        nodes_df, edges_df, metrics_df,
        interlock_graph, flow_stats, overlap_df,
        project_id=project_id
    )
    print(f"✓ Generated {len(insight_cards_doc['cards'])} insight cards")
    
    # Project summary
    project_summary = generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
    
    # Write outputs
    print("\nWriting outputs...")
    write_outputs(output_dir, metrics_df, insight_cards_doc, project_summary)
    
    print("\n" + "="*60)
    print("✅ Phase 3 complete!")
    print("="*60 + "\n")
    
    return project_summary


def main():
    parser = argparse.ArgumentParser(
        description="C4C Network Intelligence Engine — Phase 3"
    )
    parser.add_argument(
        "--nodes", type=Path, default=DEFAULT_NODES,
        help="Path to canonical nodes.csv"
    )
    parser.add_argument(
        "--edges", type=Path, default=DEFAULT_EDGES,
        help="Path to canonical edges.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for results"
    )
    parser.add_argument(
        "--project", type=str, default="glfn",
        help="Project ID for insight cards metadata"
    )
    
    args = parser.parse_args()
    
    summary = run(args.nodes, args.edges, args.out, args.project)
    
    # Print summary
    print("Project Summary:")
    print(f"  Nodes: {summary['node_counts']['total']} ({summary['node_counts']['organizations']} orgs, {summary['node_counts']['people']} people)")
    print(f"  Edges: {summary['edge_counts']['total']} ({summary['edge_counts']['grants']} grants, {summary['edge_counts']['board_memberships']} board)")
    print(f"  Funding: ${summary['funding']['total_amount']:,.0f} from {summary['funding']['funder_count']} funders to {summary['funding']['grantee_count']} grantees")
    print(f"  Governance: {summary['governance']['multi_board_people']} multi-board people, {summary['governance']['isolated_foundations']} isolated foundations")


if __name__ == "__main__":
    main()
