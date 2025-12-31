"""
InsightGraph — Funder Network Analyzer

Analyzes OrgGraph funder networks (foundations, grantees, board members).
Wraps existing run.py logic with the NetworkAnalyzer interface.

VERSION HISTORY:
----------------
v1.0.0 (2025-12-31): Initial release
- FunderAnalyzer class implementing NetworkAnalyzer ABC
- Wraps existing run.py funder analysis functions
- Returns standardized AnalysisResult

NETWORK CHARACTERISTICS:
- node_type: org/organization, person
- edge_type: grant, board_membership
- Metrics: funding flows, portfolio overlap, governance ties
"""

import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

from .base import (
    NetworkAnalyzer,
    AnalysisResult,
    HealthScore,
    InsightCard,
    BrokerageData,
    ProjectSummary,
    compute_brokerage_roles,
    get_top_brokers,
    BROKERAGE_ROLE_CONFIG,
)

# =============================================================================
# Version
# =============================================================================

FUNDER_ANALYZER_VERSION = "1.0.0"

# =============================================================================
# Thresholds (from original run.py)
# =============================================================================

CONNECTOR_THRESHOLD = 75  # Percentile for "connector" designation
CAPITAL_HUB_THRESHOLD = 75  # Percentile for "capital hub" designation


# =============================================================================
# Funder-Specific Graph Builders
# =============================================================================

def build_grant_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build directed ORG—ORG grant graph weighted by funding amount."""
    G = nx.DiGraph()
    
    # Add organization nodes
    org_nodes = nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]
    for _, row in org_nodes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    # Add grant edges
    grant_edges = edges_df[edges_df["edge_type"].str.lower().isin(["grant"])]
    for _, row in grant_edges.iterrows():
        weight = row.get("weight", row.get("amount", 1))
        G.add_edge(row["from_id"], row["to_id"], weight=weight, edge_type="grant")
    
    return G


def build_board_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build bipartite PERSON—ORG board membership graph."""
    G = nx.Graph()
    
    # Add all nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    # Add board edges
    board_edges = edges_df[edges_df["edge_type"].str.lower().isin(["board", "board_membership"])]
    for _, row in board_edges.iterrows():
        G.add_edge(row["from_id"], row["to_id"], edge_type="board")
    
    return G


def build_interlock_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build ORG—ORG interlock graph weighted by shared board members."""
    G = nx.Graph()
    
    # Add organization nodes
    org_nodes = nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]
    for _, row in org_nodes.iterrows():
        G.add_node(row["node_id"], **row.to_dict())
    
    # Find shared board members
    board_edges = edges_df[edges_df["edge_type"].str.lower().isin(["board", "board_membership"])]
    person_to_orgs = defaultdict(set)
    for _, row in board_edges.iterrows():
        person_to_orgs[row["from_id"]].add(row["to_id"])
    
    # Create interlock edges
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
    
    return G


def build_combined_org_graph(grant_graph: nx.DiGraph, interlock_graph: nx.Graph) -> nx.Graph:
    """
    Build combined undirected org-to-org graph for brokerage analysis.
    Combines grant relationships and interlock relationships.
    """
    combined = nx.Graph()
    
    # Add grant edges (as undirected)
    for u, v, data in grant_graph.edges(data=True):
        if combined.has_edge(u, v):
            combined[u][v]['weight'] = combined[u][v].get('weight', 1) + data.get('weight', 1)
            combined[u][v]['sources'].add('grant')
        else:
            combined.add_edge(u, v, weight=data.get('weight', 1), sources={'grant'})
    
    # Add interlock edges
    for u, v, data in interlock_graph.edges(data=True):
        if combined.has_edge(u, v):
            combined[u][v]['weight'] = combined[u][v].get('weight', 1) + data.get('weight', 1)
            combined[u][v]['sources'].add('interlock')
        else:
            combined.add_edge(u, v, weight=data.get('weight', 1), sources={'interlock'})
    
    return combined


# =============================================================================
# Funder-Specific Metrics
# =============================================================================

def compute_funder_metrics(nodes_df: pd.DataFrame, grant_graph: nx.DiGraph, 
                           board_graph: nx.Graph, interlock_graph: nx.Graph) -> pd.DataFrame:
    """Compute funder-specific metrics for all nodes."""
    metrics = []
    
    # Convert to undirected for betweenness calculation
    grant_undirected = grant_graph.to_undirected() if grant_graph.number_of_edges() > 0 else nx.Graph()
    grant_betweenness = nx.betweenness_centrality(grant_undirected) if grant_undirected.number_of_edges() > 0 else {}
    grant_pagerank = nx.pagerank(grant_graph, weight="weight") if grant_graph.number_of_edges() > 0 else {}
    board_betweenness = nx.betweenness_centrality(board_graph) if board_graph.number_of_edges() > 0 else {}
    
    # Component mapping
    node_to_component = {}
    if grant_graph.number_of_nodes() > 0:
        grant_undirected = grant_graph.to_undirected()
        for i, comp in enumerate(nx.connected_components(grant_undirected)):
            for node in comp:
                node_to_component[node] = i
    
    for _, row in nodes_df.iterrows():
        node_id = row["node_id"]
        node_type = str(row.get("node_type", "")).lower()
        
        m = {
            "node_id": node_id,
            "node_type": node_type,
            "label": row.get("label", ""),
            "jurisdiction": row.get("jurisdiction", ""),
            "org_slug": row.get("org_slug", ""),
            "region": row.get("region", ""),
        }
        
        if node_type in ["org", "organization"]:
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
            # Person node
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


def compute_derived_signals(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived boolean flags based on percentile thresholds."""
    df = metrics_df.copy()
    df["is_connector"] = 0
    df["is_broker"] = 0
    df["is_hidden_broker"] = 0
    df["is_capital_hub"] = 0
    df["is_isolated"] = 0
    
    org_mask = df["node_type"].str.lower().isin(["org", "organization"])
    org_df = df[org_mask]
    
    if len(org_df) > 0:
        degree_75 = np.percentile(org_df["degree"].dropna(), CONNECTOR_THRESHOLD)
        outflow_vals = org_df["grant_outflow_total"].dropna()
        outflow_75 = np.percentile(outflow_vals, CAPITAL_HUB_THRESHOLD) if len(outflow_vals) > 0 else 0
        
        df.loc[org_mask & (df["degree"] >= degree_75), "is_connector"] = 1
        df.loc[org_mask & (df["grant_outflow_total"] >= outflow_75) & (df["grant_outflow_total"] > 0), "is_capital_hub"] = 1
        df.loc[org_mask & (df["degree"] == 1), "is_isolated"] = 1
        
        # Broker thresholds among nodes with non-zero betweenness
        connectors = org_df[org_df["betweenness"] > 0]
        if len(connectors) > 0:
            betweenness_85 = np.percentile(connectors["betweenness"], 85)
            degree_40 = np.percentile(connectors["degree"], 40)
            
            df.loc[org_mask & (df["betweenness"] >= betweenness_85), "is_broker"] = 1
            df.loc[org_mask & (df["betweenness"] >= betweenness_85) & (df["degree"] <= degree_40), "is_hidden_broker"] = 1
    
    # Person connectors
    person_mask = df["node_type"].str.lower().isin(["person"])
    df.loc[person_mask & (df["boards_served"] >= 2), "is_connector"] = 1
    
    return df


def compute_flow_stats(edges_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
    """Compute system-level funding flow statistics."""
    grant_edges = edges_df[edges_df["edge_type"].str.lower().isin(["grant"])].copy()
    
    if grant_edges.empty:
        return {
            "total_funding": 0,
            "funder_count": 0,
            "grantee_count": 0,
            "top_5_funders_share": 0,
            "multi_funder_grantees": 0,
            "multi_funder_pct": 0
        }
    
    # Get weight/amount
    if 'weight' in grant_edges.columns:
        grant_edges['amount'] = grant_edges['weight']
    elif 'amount' not in grant_edges.columns:
        grant_edges['amount'] = 1
    
    total_funding = float(grant_edges['amount'].sum())  # Convert numpy to Python float
    funders = int(grant_edges['from_id'].nunique())  # Convert numpy to Python int
    grantees = int(grant_edges['to_id'].nunique())
    
    # Top 5 share
    funder_totals = grant_edges.groupby('from_id')['amount'].sum().sort_values(ascending=False)
    top_5_share = float(funder_totals.head(5).sum() / total_funding * 100) if total_funding > 0 else 0.0
    
    # Multi-funder grantees
    grantee_funder_counts = grant_edges.groupby('to_id')['from_id'].nunique()
    multi_funder = int((grantee_funder_counts > 1).sum())
    multi_funder_pct = float(multi_funder / grantees * 100) if grantees > 0 else 0.0
    
    return {
        "total_funding": total_funding,
        "funder_count": funders,
        "grantee_count": grantees,
        "top_5_funders_share": top_5_share,
        "multi_funder_grantees": multi_funder,
        "multi_funder_pct": multi_funder_pct
    }


def compute_portfolio_overlap(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Jaccard similarity between funder portfolios."""
    grant_edges = edges_df[edges_df["edge_type"].str.lower().isin(["grant"])]
    
    if grant_edges.empty:
        return pd.DataFrame(columns=['funder_a', 'funder_b', 'jaccard', 'shared_count', 'a_count', 'b_count'])
    
    # Build funder -> grantee sets
    funder_portfolios = grant_edges.groupby('from_id')['to_id'].apply(set).to_dict()
    
    # Compute pairwise Jaccard
    results = []
    funders = list(funder_portfolios.keys())
    
    for i, f1 in enumerate(funders):
        for f2 in funders[i+1:]:
            set_a = funder_portfolios[f1]
            set_b = funder_portfolios[f2]
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union > 0 else 0
            
            if intersection > 0:  # Only include pairs with overlap
                results.append({
                    'funder_a': f1,
                    'funder_b': f2,
                    'jaccard': jaccard,
                    'shared_count': intersection,
                    'a_count': len(set_a),
                    'b_count': len(set_b)
                })
    
    return pd.DataFrame(results)


# =============================================================================
# Funder Health Scoring
# =============================================================================

def compute_funder_health(flow_stats: dict, metrics_df: pd.DataFrame, 
                          n_components: int, largest_component_pct: float) -> HealthScore:
    """
    Compute 0-100 health score for funder network.
    
    Factors:
    - Coordination (multi-funder grantees): 0-25 points
    - Connectivity (largest component): 0-20 points
    - Concentration (top 5 share): -15 to +10 points
    - Governance (board interlocks): 0-15 points
    """
    positive_factors = []
    risk_factors = []
    score = 20.0  # Base score
    
    multi_funder_pct = flow_stats.get("multi_funder_pct", 0)
    
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
    
    # Governance connectivity
    org_metrics = metrics_df[metrics_df["node_type"].str.lower().isin(["org", "organization"])]
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
    
    if score >= 70:
        label = "Healthy coordination"
    elif score >= 40:
        label = "Mixed signals"
    else:
        label = "Fragmented / siloed"
    
    return HealthScore(
        score=score,
        label=label,
        positive=positive_factors,
        risk=risk_factors
    )


# =============================================================================
# Funder Analyzer Class
# =============================================================================

class FunderAnalyzer(NetworkAnalyzer):
    """
    Analyzer for funder networks (OrgGraph US/CA data).
    
    Analyzes:
    - Funding flows and concentration
    - Portfolio overlap between funders
    - Governance ties (board interlocks)
    - Hidden brokers and connectors
    - Brokerage ecosystem (shared with social analyzer)
    """
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, project_id: str = "project"):
        super().__init__(nodes_df, edges_df, project_id)
        
        # Funder-specific graphs
        self.grant_graph = None
        self.board_graph = None
        self.interlock_graph = None
        self.combined_org_graph = None
        
        # Computed data
        self.metrics_df = None
        self.flow_stats = None
        self.overlap_df = None
        self.brokerage_data = None
    
    def analyze(self) -> AnalysisResult:
        """Run funder network analysis."""
        
        # Build graphs
        self.grant_graph = build_grant_graph(self.nodes_df, self.edges_df)
        self.board_graph = build_board_graph(self.nodes_df, self.edges_df)
        self.interlock_graph = build_interlock_graph(self.nodes_df, self.edges_df)
        self.combined_org_graph = build_combined_org_graph(self.grant_graph, self.interlock_graph)
        
        # Compute metrics
        self.metrics_df = compute_funder_metrics(
            self.nodes_df, self.grant_graph, self.board_graph, self.interlock_graph
        )
        self.metrics_df = compute_derived_signals(self.metrics_df)
        self.flow_stats = compute_flow_stats(self.edges_df, self.metrics_df)
        self.overlap_df = compute_portfolio_overlap(self.edges_df)
        
        # Compute brokerage (shared logic from base.py)
        betweenness_map = dict(zip(self.metrics_df['node_id'], self.metrics_df['betweenness']))
        self.brokerage_data = compute_brokerage_roles(self.combined_org_graph, betweenness_map)
        self.brokerage_data.top_brokers = get_top_brokers(self.brokerage_data, self.nodes_df)
        
        # Add brokerage to metrics
        if self.brokerage_data.enabled:
            self.metrics_df['brokerage_role'] = self.metrics_df['node_id'].map(self.brokerage_data.roles)
            self.metrics_df['community_id'] = self.metrics_df['node_id'].map(self.brokerage_data.communities)
        
        # Compute component stats
        component_stats = self.compute_component_stats(self.combined_org_graph)
        
        # Compute health
        health = compute_funder_health(
            self.flow_stats,
            self.metrics_df,
            component_stats['n_components'],
            component_stats['largest_component_pct']
        )
        
        # Generate insight cards
        cards = self._generate_insight_cards(health, component_stats)
        
        # Generate project summary
        project_summary = self._generate_project_summary()
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(health, cards)
        
        return AnalysisResult(
            network_type='funder',
            source_app=self.source_app,
            project_id=self.project_id,
            generated_at=self.get_timestamp(),
            health=health,
            cards=cards,
            metrics_df=self.metrics_df,
            project_summary=project_summary,
            brokerage=self.brokerage_data,
            markdown_report=markdown_report,
            nodes_df=self.nodes_df,
            edges_df=self.edges_df
        )
    
    def _generate_insight_cards(self, health: HealthScore, component_stats: dict) -> list[InsightCard]:
        """Generate funder-specific insight cards."""
        cards = []
        
        # Network Health Overview
        cards.append(InsightCard(
            card_id="network_health",
            use_case="System Framing",
            title="Network Health Overview",
            summary=self._format_health_summary(health),
            ranked_rows=[
                {"indicator": "Health Score", "value": f"{health.score}/100", "interpretation": health.label},
                {"indicator": "Multi-Funder Grantees", "value": f"{self.flow_stats['multi_funder_pct']:.1f}%", 
                 "interpretation": self._interpret_multi_funder()},
                {"indicator": "Connected through Shared Funding", "value": f"{component_stats['largest_component_pct']:.0f}%",
                 "interpretation": self._interpret_connectivity(component_stats)},
                {"indicator": "Top 5 Funder Share", "value": f"{self.flow_stats['top_5_funders_share']:.0f}%",
                 "interpretation": self._interpret_concentration()}
            ],
            health_factors={"positive": health.positive, "risk": health.risk}
        ))
        
        # Funding Concentration
        cards.append(InsightCard(
            card_id="concentration_snapshot",
            use_case="Funding Concentration",
            title="Funding Concentration",
            summary=self._format_concentration_summary(),
            ranked_rows=[
                {"metric": "Total Funding", "value": f"${self.flow_stats['total_funding']:,.0f}"},
                {"metric": "Funders", "value": str(self.flow_stats['funder_count'])},
                {"metric": "Grantees", "value": str(self.flow_stats['grantee_count'])},
                {"metric": "Top 5 Share", "value": f"{self.flow_stats['top_5_funders_share']:.0f}%"},
                {"metric": "Multi-Funder Grantees", "value": str(self.flow_stats['multi_funder_grantees'])}
            ]
        ))
        
        # Board Conduits
        multi_board = self.metrics_df[
            (self.metrics_df['node_type'].str.lower() == 'person') & 
            (self.metrics_df['boards_served'] >= 2)
        ] if 'boards_served' in self.metrics_df.columns else pd.DataFrame()
        
        cards.append(InsightCard(
            card_id="shared_board_conduits",
            use_case="Board Network & Conduits",
            title="Shared Board Conduits",
            summary=self._format_board_summary(multi_board)
        ))
        
        # Hidden Brokers
        hidden_brokers = self.metrics_df[self.metrics_df['is_hidden_broker'] == 1] if 'is_hidden_broker' in self.metrics_df.columns else pd.DataFrame()
        
        cards.append(InsightCard(
            card_id="hidden_brokers",
            use_case="Brokerage Roles",
            title="Hidden Brokers",
            summary=self._format_hidden_broker_summary(hidden_brokers)
        ))
        
        # Decision Options
        cards.append(InsightCard(
            card_id="decision_options",
            use_case="Decision Options",
            title="Decision Options",
            summary=self._generate_decision_options(health)
        ))
        
        return cards
    
    def _generate_project_summary(self) -> ProjectSummary:
        """Generate project summary for funder network."""
        org_count = int(len(self.nodes_df[self.nodes_df['node_type'].str.lower().isin(['org', 'organization'])]))
        person_count = int(len(self.nodes_df[self.nodes_df['node_type'].str.lower() == 'person']))
        
        grant_count = int(len(self.edges_df[self.edges_df['edge_type'].str.lower() == 'grant']))
        board_count = int(len(self.edges_df[self.edges_df['edge_type'].str.lower().isin(['board', 'board_membership'])]))
        
        multi_board = self.metrics_df[
            (self.metrics_df['node_type'].str.lower() == 'person') & 
            (self.metrics_df['boards_served'] >= 2)
        ] if 'boards_served' in self.metrics_df.columns else pd.DataFrame()
        
        return ProjectSummary(
            generated_at=self.get_timestamp(),
            network_type='funder',
            source_app=self.source_app,
            node_counts={
                "total": int(len(self.nodes_df)),
                "organizations": org_count,
                "people": person_count
            },
            edge_counts={
                "total": int(len(self.edges_df)),
                "grants": grant_count,
                "board_memberships": board_count
            },
            funding={
                "total_amount": float(self.flow_stats['total_funding']),
                "funder_count": int(self.flow_stats['funder_count']),
                "grantee_count": int(self.flow_stats['grantee_count']),
                "top_5_share": float(self.flow_stats['top_5_funders_share'])
            },
            governance={
                "multi_board_people": int(len(multi_board))
            },
            brokerage=self.brokerage_data.to_dict() if self.brokerage_data else None
        )
    
    def _generate_markdown_report(self, health: HealthScore, cards: list[InsightCard]) -> str:
        """Generate markdown report for funder network."""
        lines = []
        
        # Header
        lines.append(f"# Funder Network Analysis Report")
        lines.append(f"**Project:** {self.project_id.upper()}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"**Network Type:** Funder Network")
        lines.append("")
        
        # Summary stats
        summary = self._generate_project_summary()
        lines.append(f"**Nodes:** {summary.node_counts['total']} ({summary.node_counts['organizations']} organizations, {summary.node_counts['people']} people)")
        lines.append(f"**Edges:** {summary.edge_counts['total']} ({summary.edge_counts['grants']} grants, {summary.edge_counts['board_memberships']} board memberships)")
        lines.append(f"**Total Funding:** ${summary.funding['total_amount']:,.0f}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Health section
        health_emoji = "🟢" if health.score >= 70 else "🟡" if health.score >= 40 else "🔴"
        lines.append(f"## {health_emoji} Network Health: {health.score}/100 ({health.label})")
        lines.append("")
        
        if health.positive:
            lines.append("### ✅ Positive Factors")
            lines.append("")
            for factor in health.positive:
                lines.append(f"- {factor}")
            lines.append("")
        
        if health.risk:
            lines.append("### ⚠️ Risk Factors")
            lines.append("")
            for factor in health.risk:
                lines.append(f"- {factor}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Brokerage section
        if self.brokerage_data and self.brokerage_data.enabled:
            lines.extend(self._format_brokerage_section())
            lines.append("---")
            lines.append("")
        
        # Cards
        for card in cards:
            if card.card_id not in ['network_health']:  # Already rendered
                lines.append(f"## {card.title}")
                lines.append(f"*Use Case: {card.use_case}*")
                lines.append("")
                lines.append(card.summary)
                lines.append("")
                lines.append("---")
                lines.append("")
        
        lines.append("*Report generated by C4C InsightGraph — Funder Network Analyzer*")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Helper Formatters
    # =========================================================================
    
    def _format_health_summary(self, health: HealthScore) -> str:
        emoji = "🔴" if health.score < 40 else "🟡" if health.score < 70 else "🟢"
        return f"{emoji} **Network Health: {health.score}/100** — *{health.label}*"
    
    def _interpret_multi_funder(self) -> str:
        pct = self.flow_stats['multi_funder_pct']
        if pct >= 10:
            return "Strong coordination — multiple funders supporting same grantees"
        elif pct >= 5:
            return "Moderate coordination potential"
        elif pct >= 1:
            return "Limited overlap — funders operate mostly independently"
        else:
            return "No overlap — funders operate in complete silos with no shared grantees"
    
    def _interpret_connectivity(self, component_stats: dict) -> str:
        pct = component_stats['largest_component_pct']
        if pct >= 80:
            return "Highly connected through shared funding relationships"
        elif pct >= 50:
            return "Moderately connected"
        else:
            return f"Only {pct:.0f}% of organizations are connected through shared funding. Most funders operate in isolated clusters with completely distinct portfolios"
    
    def _interpret_concentration(self) -> str:
        share = self.flow_stats['top_5_funders_share']
        if share >= 95:
            return "Extreme concentration — top 5 control nearly all funding"
        elif share >= 80:
            return "High concentration"
        elif share >= 60:
            return "Moderate concentration"
        else:
            return f"Distributed — top 5 control only {share:.0f}%, healthy funder diversity"
    
    def _format_concentration_summary(self) -> str:
        share = self.flow_stats['top_5_funders_share']
        total = self.flow_stats['total_funding']
        
        if share >= 95:
            return f"🔴 **Extreme concentration**\n\nTop 5 funders control {share:.0f}% of ${total:,.0f}. This creates significant dependency risk."
        elif share >= 80:
            return f"🟠 **High concentration**\n\nTop 5 funders control {share:.0f}% of ${total:,.0f}."
        else:
            return f"🟢 **Healthy distribution**\n\nFunding is relatively distributed — top 5 funders control {share:.0f}% of ${total:,.0f}. This diversity provides resilience and multiple pathways for grantees."
    
    def _format_board_summary(self, multi_board: pd.DataFrame) -> str:
        if len(multi_board) == 0:
            return "⚪ **No multi-board individuals detected**\n\nNo one serves on multiple boards in this network. Governance structures are fully separate — a potential gap for coordination."
        else:
            return f"🔗 **{len(multi_board)} multi-board individuals**\n\nThese individuals serve on 2+ boards, creating informal coordination pathways."
    
    def _format_hidden_broker_summary(self, hidden_brokers: pd.DataFrame) -> str:
        if len(hidden_brokers) == 0:
            return "⚪ **No hidden brokers detected**\n\nAll high-betweenness nodes are also highly visible. No quiet bridges exist in this network."
        else:
            return f"🔍 **{len(hidden_brokers)} hidden brokers detected**\n\nThese organizations have high betweenness but low visibility — they quietly bridge different parts of the network."
    
    def _generate_decision_options(self, health: HealthScore) -> str:
        lines = ["_The options below describe common ways teams apply these signals in practice; they are not recommendations._\n"]
        
        if health.score < 40:
            lines.append("### 🧭 How to Read This\n")
            lines.append("The network appears **fragmented**. Funders operate largely in silos with minimal coordination. Teams often assess whether building basic connective tissue would be valuable.\n")
        elif health.score < 70:
            lines.append("### 🧭 How to Read This\n")
            lines.append("The network shows **mixed signals**. Some coordination exists, but structural gaps limit effectiveness.\n")
        else:
            lines.append("### 🧭 How to Read This\n")
            lines.append("The network shows **healthy coordination signals**. Focus on deepening strategic relationships.\n")
        
        if self.flow_stats['multi_funder_pct'] < 5:
            lines.append("### 🔗 Strengthen Funder Coordination\n")
            lines.append("- **Build initial overlap:** Almost no grantees receive from multiple funders. Start by mapping where portfolios *could* overlap based on thematic focus, then facilitate introductions.\n")
        
        multi_board = self.metrics_df[
            (self.metrics_df['node_type'].str.lower() == 'person') & 
            (self.metrics_df['boards_served'] >= 2)
        ] if 'boards_served' in self.metrics_df.columns else pd.DataFrame()
        
        if len(multi_board) == 0:
            lines.append("### 🏛️ Strengthen Governance Ties\n")
            lines.append("- **Identify potential bridge-builders:** No one currently serves on multiple boards. Look for respected individuals who could be nominated to additional boards to create connective tissue.\n")
        
        return "\n".join(lines)
    
    def _format_brokerage_section(self) -> list[str]:
        """Format brokerage ecosystem section for markdown report."""
        lines = []
        lines.append("## 🎭 Brokerage Ecosystem")
        lines.append("")
        lines.append("_How information and influence flow through the network_")
        lines.append("")
        
        if not self.brokerage_data or not self.brokerage_data.enabled:
            lines.append("> Brokerage analysis requires at least 10 organizations.")
            return lines
        
        # Pattern and interpretation
        pattern_display = self.brokerage_data.pattern.replace('-', ' ').title()
        lines.append(f"**Pattern:** {pattern_display}")
        lines.append("")
        lines.append(self.brokerage_data.interpretation)
        lines.append("")
        lines.append(f"The network contains **{self.brokerage_data.community_count} distinct communities** detected via Louvain algorithm.")
        lines.append("")
        
        # Role distribution table
        lines.append("### Role Distribution")
        lines.append("")
        lines.append("| Role | Count | Description |")
        lines.append("|------|-------|-------------|")
        
        role_order = ['liaison', 'gatekeeper', 'representative', 'coordinator', 'consultant', 'peripheral']
        for role in role_order:
            count = self.brokerage_data.role_counts.get(role, 0)
            if count > 0:
                config = BROKERAGE_ROLE_CONFIG[role]
                lines.append(f"| {config['emoji']} {config['label']} | {count} | {config['description']} |")
        
        lines.append("")
        
        # Top brokers
        if self.brokerage_data.top_brokers:
            lines.append("### Key Brokers")
            lines.append("")
            for node_id, label, role in self.brokerage_data.top_brokers[:8]:
                config = BROKERAGE_ROLE_CONFIG.get(role, BROKERAGE_ROLE_CONFIG['peripheral'])
                lines.append(f"- **{label}** — {config['emoji']} {config['label']}")
            lines.append("")
        
        # Strategic implications
        if self.brokerage_data.strategic_implications:
            lines.append("### Strategic Implications")
            lines.append("")
            for impl in self.brokerage_data.strategic_implications:
                lines.append(f"- {impl}")
            lines.append("")
        
        return lines
