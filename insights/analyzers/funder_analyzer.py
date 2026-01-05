"""
InsightGraph — Funder Network Analyzer

Analyzes OrgGraph funder networks (foundations, grantees, board members).
Wraps existing run.py logic with the NetworkAnalyzer interface.

VERSION HISTORY:
----------------
v1.1.1 (2026-01-06): Bug fix for graph building
- FIX: Exclude 'weight' from row dict spread to avoid duplicate keyword argument
- Affects build_grant_graph and build_board_graph

v1.1.0 (2026-01-06): YAML Copy Map Integration
- Health labels now sourced from INSIGHTGRAPH_COPY_MAP_v1.yaml
- Interpretive guardrail added to markdown reports
- Health descriptions from YAML
- Single source of truth for narrative copy
- Graceful fallback if copy_manager unavailable

v1.0.1 (2025-12-31): Fixed funding amount calculation
- build_grant_graph: Prefer 'amount' column over 'weight' for edge weights
- compute_flow_stats: Use 'amount' column for total funding calculation
- Now correctly shows dollar amounts instead of grant counts

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
# YAML Copy Manager Integration
# =============================================================================

# Try to import copy_manager for YAML-driven labels
try:
    from copy_manager import get_copy_manager
    COPY_MANAGER_AVAILABLE = True
except ImportError:
    COPY_MANAGER_AVAILABLE = False


def _get_health_label(score: int) -> str:
    """
    Get health label using current 70/40 thresholds with YAML vocabulary.
    
    Mapping:
    - ≥70 → "Strong" (top tier)
    - ≥40 → "Moderate" (middle tier)  
    - <40 → "Fragile" (bottom tier)
    """
    if COPY_MANAGER_AVAILABLE:
        try:
            copy = get_copy_manager()
            # Map current thresholds to YAML bands
            if score >= 70:
                return copy.get_health_band(85).label  # "Strong"
            elif score >= 40:
                return copy.get_health_band(65).label  # "Moderate"
            else:
                return copy.get_health_band(30).label  # "Fragile"
        except Exception:
            pass
    
    # Fallback to original labels
    if score >= 70:
        return "Healthy coordination"
    elif score >= 40:
        return "Mixed signals"
    else:
        return "Fragmented / siloed"


def _get_health_description(score: int) -> str:
    """Get health description from YAML based on score."""
    if COPY_MANAGER_AVAILABLE:
        try:
            copy = get_copy_manager()
            if score >= 70:
                return copy.get_health_band(85).description
            elif score >= 40:
                return copy.get_health_band(65).description
            else:
                return copy.get_health_band(30).description
        except Exception:
            pass
    
    # Fallback descriptions
    if score >= 70:
        return "Structurally aligned; coordination pathways are robust."
    elif score >= 40:
        return "Mixed signals; coordination is possible but not automatic."
    else:
        return "High structural risk; the network depends on a small number of critical bridges."


def _get_health_guardrail() -> str:
    """Get the interpretive guardrail from YAML."""
    if COPY_MANAGER_AVAILABLE:
        try:
            copy = get_copy_manager()
            return copy.health_score_helper
        except Exception:
            pass
    
    return "Network Health reflects coordination capacity — not impact, effectiveness, or intent."


# =============================================================================
# Version
# =============================================================================

FUNDER_ANALYZER_VERSION = "1.1.1"

# =============================================================================
# Thresholds (from original run.py)
# =============================================================================

CONNECTOR_THRESHOLD = 75  # Percentile for "connector" designation
CAPITAL_HUB_THRESHOLD = 75  # Percentile for "capital hub" designation


# =============================================================================
# Funder-Specific Graph Builders
# =============================================================================

def build_grant_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build directed graph from grant edges only."""
    G = nx.DiGraph()
    
    # Add organization nodes
    org_nodes = nodes_df[nodes_df['node_type'].isin(['org', 'organization'])]
    for _, row in org_nodes.iterrows():
        G.add_node(row['node_id'], **row.to_dict())
    
    # Add grant edges
    grant_edges = edges_df[edges_df['edge_type'].isin(['grant', 'funding'])]
    for _, row in grant_edges.iterrows():
        # Prefer 'amount' column for weight, fall back to 'weight'
        weight = row.get('amount', row.get('weight', 1))
        if pd.isna(weight):
            weight = 1
        # Exclude 'weight' from row dict to avoid duplicate keyword argument
        row_dict = {k: v for k, v in row.to_dict().items() if k != 'weight'}
        G.add_edge(row['from_id'], row['to_id'], weight=weight, **row_dict)
    
    return G


def build_board_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build undirected graph from board membership edges."""
    G = nx.Graph()
    
    # Add all nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row['node_id'], **row.to_dict())
    
    # Add board membership edges
    board_edges = edges_df[edges_df['edge_type'].isin(['board_membership', 'board'])]
    for _, row in board_edges.iterrows():
        # Exclude 'weight' from row dict to avoid potential duplicate
        row_dict = {k: v for k, v in row.to_dict().items() if k != 'weight'}
        G.add_edge(row['from_id'], row['to_id'], **row_dict)
    
    return G


def build_interlock_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build organization-to-organization graph via shared board members.
    
    Two organizations are connected if they share at least one board member.
    """
    G = nx.Graph()
    
    # Get board edges
    board_edges = edges_df[edges_df['edge_type'].isin(['board_membership', 'board'])]
    
    # Map people to their organizations
    person_to_orgs = defaultdict(set)
    for _, row in board_edges.iterrows():
        # Determine which is person, which is org
        from_type = nodes_df[nodes_df['node_id'] == row['from_id']]['node_type'].iloc[0] if len(nodes_df[nodes_df['node_id'] == row['from_id']]) > 0 else 'unknown'
        to_type = nodes_df[nodes_df['node_id'] == row['to_id']]['node_type'].iloc[0] if len(nodes_df[nodes_df['node_id'] == row['to_id']]) > 0 else 'unknown'
        
        if from_type == 'person' and to_type in ['org', 'organization']:
            person_to_orgs[row['from_id']].add(row['to_id'])
        elif to_type == 'person' and from_type in ['org', 'organization']:
            person_to_orgs[row['to_id']].add(row['from_id'])
    
    # Create edges between organizations that share board members
    for person, orgs in person_to_orgs.items():
        org_list = list(orgs)
        for i in range(len(org_list)):
            for j in range(i + 1, len(org_list)):
                if G.has_edge(org_list[i], org_list[j]):
                    G[org_list[i]][org_list[j]]['weight'] += 1
                    G[org_list[i]][org_list[j]]['shared_members'].append(person)
                else:
                    G.add_edge(org_list[i], org_list[j], weight=1, shared_members=[person])
    
    return G


def build_combined_org_graph(grant_graph: nx.DiGraph, board_graph: nx.Graph) -> nx.Graph:
    """
    Build undirected org-only graph combining grant and board relationships.
    
    Used for community detection and centrality metrics.
    """
    G = nx.Graph()
    
    # Add edges from grant graph (as undirected)
    for u, v, data in grant_graph.edges(data=True):
        if G.has_edge(u, v):
            G[u][v]['weight'] += data.get('weight', 1)
        else:
            G.add_edge(u, v, weight=data.get('weight', 1), edge_type='grant')
    
    # Add edges from board graph (org-to-person connections)
    # We want org-to-org through shared people
    org_nodes = set(n for n in grant_graph.nodes())
    for u, v, data in board_graph.edges(data=True):
        if u in org_nodes and v in org_nodes:
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1, edge_type='board')
    
    return G


# =============================================================================
# Funder-Specific Metrics
# =============================================================================

def compute_flow_stats(grant_graph: nx.DiGraph, nodes_df: pd.DataFrame) -> dict:
    """Compute funding flow statistics."""
    stats = {
        'total_funding': 0,
        'funder_count': 0,
        'grantee_count': 0,
        'multi_funder_count': 0,
        'multi_funder_pct': 0,
        'top5_share_pct': 0,
    }
    
    # Identify funders (nodes with outgoing grant edges)
    funders = set(u for u, v in grant_graph.edges())
    # Identify grantees (nodes with incoming grant edges)
    grantees = set(v for u, v in grant_graph.edges())
    
    stats['funder_count'] = len(funders)
    stats['grantee_count'] = len(grantees)
    
    # Total funding - use 'amount' attribute from edges
    for u, v, data in grant_graph.edges(data=True):
        amount = data.get('amount', data.get('weight', 0))
        if pd.notna(amount):
            stats['total_funding'] += float(amount)
    
    # Count grantees with multiple funders
    grantee_funder_counts = defaultdict(int)
    for u, v in grant_graph.edges():
        grantee_funder_counts[v] += 1
    
    multi_funder = sum(1 for g, count in grantee_funder_counts.items() if count > 1)
    stats['multi_funder_count'] = multi_funder
    stats['multi_funder_pct'] = (multi_funder / len(grantees) * 100) if grantees else 0
    
    # Top 5 funder share
    funder_totals = defaultdict(float)
    for u, v, data in grant_graph.edges(data=True):
        amount = data.get('amount', data.get('weight', 0))
        if pd.notna(amount):
            funder_totals[u] += float(amount)
    
    if funder_totals and stats['total_funding'] > 0:
        top5 = sorted(funder_totals.values(), reverse=True)[:5]
        stats['top5_share_pct'] = sum(top5) / stats['total_funding'] * 100
    
    return stats


def compute_portfolio_overlap(grant_graph: nx.DiGraph) -> dict:
    """Compute portfolio overlap between funders."""
    # Map each funder to their grantees
    funder_portfolios = defaultdict(set)
    for u, v in grant_graph.edges():
        funder_portfolios[u].add(v)
    
    funders = list(funder_portfolios.keys())
    overlaps = []
    
    for i in range(len(funders)):
        for j in range(i + 1, len(funders)):
            f1, f2 = funders[i], funders[j]
            shared = funder_portfolios[f1] & funder_portfolios[f2]
            if shared:
                # Jaccard similarity
                union = funder_portfolios[f1] | funder_portfolios[f2]
                overlap_pct = len(shared) / len(union) * 100
                overlaps.append({
                    'funder_a': f1,
                    'funder_b': f2,
                    'shared_count': len(shared),
                    'overlap_pct': overlap_pct
                })
    
    # Sort by shared count
    overlaps.sort(key=lambda x: x['shared_count'], reverse=True)
    
    return {
        'top_pairs': overlaps[:10],
        'max_overlap_pct': max(o['overlap_pct'] for o in overlaps) if overlaps else 0
    }


def compute_governance_stats(interlock_graph: nx.Graph, board_graph: nx.Graph, nodes_df: pd.DataFrame) -> dict:
    """Compute governance/board interlock statistics."""
    stats = {
        'has_governance_data': len(board_graph.edges()) > 0,
        'shared_board_count': 0,
        'pct_with_interlocks': 0,
        'top_connectors': []
    }
    
    if not stats['has_governance_data']:
        return stats
    
    # Count shared board members (people on multiple org boards)
    person_nodes = nodes_df[nodes_df['node_type'] == 'person']['node_id'].tolist()
    multi_board_people = []
    
    for person in person_nodes:
        if person in board_graph:
            orgs = [n for n in board_graph.neighbors(person) 
                   if nodes_df[nodes_df['node_id'] == n]['node_type'].iloc[0] in ['org', 'organization']
                   if len(nodes_df[nodes_df['node_id'] == n]) > 0]
            if len(orgs) > 1:
                multi_board_people.append({'person': person, 'org_count': len(orgs)})
    
    stats['shared_board_count'] = len(multi_board_people)
    
    # Percentage of orgs with interlocks
    orgs_with_interlocks = set()
    for person_data in multi_board_people:
        person = person_data['person']
        if person in board_graph:
            for neighbor in board_graph.neighbors(person):
                orgs_with_interlocks.add(neighbor)
    
    org_nodes = nodes_df[nodes_df['node_type'].isin(['org', 'organization'])]
    if len(org_nodes) > 0:
        stats['pct_with_interlocks'] = len(orgs_with_interlocks) / len(org_nodes) * 100
    
    # Top connectors
    multi_board_people.sort(key=lambda x: x['org_count'], reverse=True)
    stats['top_connectors'] = multi_board_people[:5]
    
    return stats


# =============================================================================
# Health Score (Funder-Specific)
# =============================================================================

def compute_health_score(
    flow_stats: dict,
    portfolio_overlap: dict,
    governance_stats: dict,
    combined_graph: nx.Graph,
    nodes_df: pd.DataFrame
) -> HealthScore:
    """
    Compute network health score for funder networks.
    
    Factors:
    - Multi-funder grantee percentage (coordination)
    - Funding concentration (resilience)
    - Governance connectivity (embedded ties)
    - Network connectivity (largest component)
    """
    score = 0
    positive_factors = []
    risk_factors = []
    
    # 1. Multi-funder percentage (0-30 points)
    multi_funder_pct = flow_stats.get('multi_funder_pct', 0)
    if multi_funder_pct >= 30:
        score += 30
        positive_factors.append(f"🟢 **Strong coordination** — {multi_funder_pct:.1f}% of grantees have multiple funders")
    elif multi_funder_pct >= 10:
        score += 15
        positive_factors.append(f"🟡 **Moderate coordination** — {multi_funder_pct:.1f}% have multiple funders")
    elif multi_funder_pct >= 5:
        score += 8
    else:
        risk_factors.append(f"🔴 **Low coordination** — only {multi_funder_pct:.1f}% have multiple funders")
    
    # 2. Network connectivity (0-25 points)
    if len(combined_graph) > 0:
        largest_cc = max(nx.connected_components(combined_graph), key=len) if combined_graph.number_of_nodes() > 0 else set()
        largest_component_pct = len(largest_cc) / combined_graph.number_of_nodes() * 100 if combined_graph.number_of_nodes() > 0 else 0
        
        if largest_component_pct >= 80:
            score += 25
            positive_factors.append(f"🟢 **Highly connected** — {largest_component_pct:.0f}% of organizations linked through shared funding")
        elif largest_component_pct >= 50:
            score += 15
        else:
            risk_factors.append(f"🔴 **Fragmented** — only {largest_component_pct:.0f}% connected through shared funding, most operate in isolated clusters")
    
    # 3. Funding concentration (0-20 points) - inverse relationship
    top5_share = flow_stats.get('top5_share_pct', 100)
    if top5_share <= 50:
        score += 20
        positive_factors.append(f"🟢 **Distributed funding** — top 5 control {top5_share:.0f}%")
    elif top5_share <= 70:
        score += 12
    elif top5_share <= 85:
        score += 5
    else:
        risk_factors.append(f"🔴 **Extreme concentration** — top 5 control {top5_share:.0f}%")
    
    # 4. Governance connectivity (0-15 points)
    if governance_stats.get('has_governance_data'):
        pct_with_interlocks = governance_stats.get('pct_with_interlocks', 0)
        if pct_with_interlocks >= 20:
            score += 15
            positive_factors.append(f"🟢 **Governance ties** — {pct_with_interlocks:.0f}% of funders share board members")
        elif pct_with_interlocks >= 5:
            score += 8
        elif pct_with_interlocks == 0:
            risk_factors.append("🔴 **No governance ties** — funders have no shared board members")
    
    score = max(0, min(100, int(score)))
    
    # Get label from YAML (with current 70/40 thresholds)
    label = _get_health_label(score)
    
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
        
        # Funder-specific stats
        self.flow_stats = None
        self.portfolio_overlap = None
        self.governance_stats = None
        self.brokerage_data = None
    
    def analyze(self) -> AnalysisResult:
        """Run funder network analysis."""
        
        # Build graphs
        self.grant_graph = build_grant_graph(self.nodes_df, self.edges_df)
        self.board_graph = build_board_graph(self.nodes_df, self.edges_df)
        self.interlock_graph = build_interlock_graph(self.nodes_df, self.edges_df)
        self.combined_org_graph = build_combined_org_graph(self.grant_graph, self.board_graph)
        
        # Compute funder-specific metrics
        self.flow_stats = compute_flow_stats(self.grant_graph, self.nodes_df)
        self.portfolio_overlap = compute_portfolio_overlap(self.grant_graph)
        self.governance_stats = compute_governance_stats(
            self.interlock_graph, self.board_graph, self.nodes_df
        )
        
        # Compute brokerage roles
        self.brokerage_data = self._compute_brokerage()
        
        # Compute health score
        health = compute_health_score(
            self.flow_stats,
            self.portfolio_overlap,
            self.governance_stats,
            self.combined_org_graph,
            self.nodes_df
        )
        
        # Generate insight cards
        cards = self._generate_insight_cards(health)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(health, cards)
        
        # Compute node metrics
        metrics_df = self._compute_node_metrics()
        
        return AnalysisResult(
            health=health,
            cards=cards,
            metrics_df=metrics_df,
            project_summary=self._generate_project_summary(),
            markdown_report=markdown_report,
            network_type='funder',
            brokerage_data=self.brokerage_data
        )
    
    def _compute_brokerage(self) -> BrokerageData:
        """Compute brokerage ecosystem data."""
        # Use combined org graph for community detection and brokerage
        if self.combined_org_graph is None or self.combined_org_graph.number_of_nodes() == 0:
            return BrokerageData(enabled=False, roles={}, communities=[], summary="")
        
        role_assignments, communities = compute_brokerage_roles(
            self.combined_org_graph, 
            self.nodes_df
        )
        
        # Get top brokers
        top_brokers = get_top_brokers(role_assignments, self.nodes_df, limit=8)
        
        # Count roles
        role_counts = defaultdict(int)
        for node_id, role in role_assignments.items():
            role_counts[role] += 1
        
        # Generate summary
        total_nodes = len(role_assignments)
        strategic_roles = ['gatekeeper', 'representative', 'consultant', 'liaison']
        strategic_count = sum(role_counts.get(r, 0) for r in strategic_roles)
        strategic_pct = (strategic_count / total_nodes * 100) if total_nodes > 0 else 0
        
        if strategic_pct >= 10:
            pattern = "Broker-rich"
            summary = f"High brokerage capacity ({strategic_pct:.0f}% in strategic roles). Multiple actors can facilitate cross-community coordination."
        elif strategic_pct >= 3:
            pattern = "Balanced"
            summary = "Brokerage roles are distributed without strong concentration. The network has moderate coordination capacity."
        else:
            pattern = "Coordinator-dominated"
            summary = "Most actors strengthen within-community ties. Cross-community coordination depends on a small number of brokers."
        
        return BrokerageData(
            enabled=True,
            roles=role_assignments,
            communities=communities,
            role_counts=dict(role_counts),
            top_brokers=top_brokers,
            pattern=pattern,
            summary=summary
        )
    
    def _generate_insight_cards(self, health: HealthScore) -> list[InsightCard]:
        """Generate insight cards for funder network."""
        cards = []
        
        # Network Health Overview
        cards.append(InsightCard(
            card_id="network_health",
            title="Network Health Overview",
            use_case="Overall Assessment",
            summary=self._format_health_summary(health),
            details={"score": health.score, "label": health.label},
            signal_strength="high"
        ))
        
        # Funding Concentration
        cards.append(InsightCard(
            card_id="funding_concentration",
            title="Funding Concentration",
            use_case="Funding Concentration",
            summary=self._format_concentration_summary(),
            details=self.flow_stats,
            signal_strength="high"
        ))
        
        # Governance / Board Interlocks
        cards.append(InsightCard(
            card_id="governance",
            title="Shared Board Conduits",
            use_case="Board Network & Conduits",
            summary=self._format_governance_summary(),
            details=self.governance_stats,
            signal_strength="high" if self.governance_stats.get('has_governance_data') else "unavailable"
        ))
        
        # Hidden Brokers
        cards.append(InsightCard(
            card_id="hidden_brokers",
            title="Hidden Brokers",
            use_case="Brokerage Roles",
            summary=self._format_broker_summary(),
            details={},
            signal_strength="medium"
        ))
        
        # Strategic Recommendations
        cards.append(InsightCard(
            card_id="recommendations",
            title="Strategic Considerations",
            use_case="Strategic Considerations",
            summary=self._format_recommendations(health),
            details={},
            signal_strength="high"
        ))
        
        return cards
    
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
        
        # Health section with interpretive guardrail
        health_emoji = "🟢" if health.score >= 70 else "🟡" if health.score >= 40 else "🔴"
        lines.append(f"## {health_emoji} Network Health: {health.score}/100 ({health.label})")
        lines.append("")
        
        # Add interpretive guardrail from YAML
        guardrail = _get_health_guardrail()
        lines.append(f"*{guardrail}*")
        lines.append("")
        
        # Add health description from YAML
        description = _get_health_description(health.score)
        lines.append(description)
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
        if pct >= 30:
            return "Strong coordination — multiple funders supporting same grantees"
        elif pct >= 10:
            return "Moderate coordination potential"
        else:
            return "Low coordination — mostly siloed funding"
    
    def _interpret_connectivity(self) -> str:
        if self.combined_org_graph and self.combined_org_graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(self.combined_org_graph), key=len)
            pct = len(largest_cc) / self.combined_org_graph.number_of_nodes() * 100
            if pct >= 80:
                return "Highly connected"
            elif pct >= 50:
                return "Moderately connected"
            else:
                return f"Fragmented — only {pct:.0f}% of nodes connected"
        return "Unknown"
    
    def _interpret_concentration(self) -> str:
        top5 = self.flow_stats['top5_share_pct']
        if top5 >= 85:
            return "Extreme concentration risk"
        elif top5 >= 70:
            return "Moderate concentration"
        else:
            return "Healthy distribution"
    
    def _format_concentration_summary(self) -> str:
        top5 = self.flow_stats['top5_share_pct']
        total = self.flow_stats['total_funding']
        
        if top5 >= 85:
            emoji = "🔴"
            label = "Extreme concentration"
            detail = f"Top 5 funders control {top5:.0f}% of ${total:,.0f}. This creates significant dependency risk."
        elif top5 >= 70:
            emoji = "🟡"
            label = "Moderate concentration"
            detail = f"Top 5 funders control {top5:.0f}% of ${total:,.0f}. Some diversification exists but top funders dominate."
        else:
            emoji = "🟢"
            label = "Healthy distribution"
            detail = f"Funding is relatively distributed — top 5 funders control {top5:.0f}% of ${total:,.0f}. This diversity provides resilience and multiple pathways for grantees."
        
        return f"{emoji} **{label}**\n\n{detail}"
    
    def _format_governance_summary(self) -> str:
        if not self.governance_stats.get('has_governance_data'):
            return "⚪ **No board membership data available**\n\nGovernance analysis requires board membership edges in the network data."
        
        shared_count = self.governance_stats.get('shared_board_count', 0)
        
        if shared_count == 0:
            return "⚪ **No multi-board individuals detected**\n\nNo one serves on multiple boards in this network. Governance structures are fully separate — a potential gap for coordination."
        else:
            top_connectors = self.governance_stats.get('top_connectors', [])
            connector_list = ", ".join([f"{c['person']} ({c['org_count']} boards)" for c in top_connectors[:3]])
            return f"🟢 **{shared_count} multi-board individuals detected**\n\nThese individuals serve as governance bridges: {connector_list}."
    
    def _format_broker_summary(self) -> str:
        if not self.brokerage_data or not self.brokerage_data.enabled:
            return "⚪ **Brokerage analysis unavailable**"
        
        # Count hidden brokers (high betweenness, low visibility)
        role_counts = self.brokerage_data.role_counts
        hidden_broker_count = role_counts.get('consultant', 0) + role_counts.get('liaison', 0)
        
        if hidden_broker_count == 0:
            return "⚪ **No hidden brokers detected**\n\nAll high-betweenness nodes are also highly visible. No quiet bridges exist in this network."
        else:
            return f"🔍 **{hidden_broker_count} hidden brokers detected**\n\nThese organizations have high betweenness but low visibility — they quietly bridge different parts of the network."
    
    def _format_recommendations(self, health: HealthScore) -> str:
        lines = []
        lines.append("*The options below describe common ways teams apply these signals in practice; they are not recommendations.*")
        lines.append("")
        
        # Opening context based on health
        if health.score >= 70:
            lines.append("### 🧭 How to Read This\n")
            lines.append("The network shows **healthy coordination signals**. Strategic options focus on deepening existing connections.\n")
        elif health.score >= 40:
            lines.append("### 🧭 How to Read This\n")
            lines.append("The network shows **mixed signals**. Some coordination exists, but structural gaps limit effectiveness.\n")
        else:
            lines.append("### 🧭 How to Read This\n")
            lines.append("The network appears **fragmented**. Funders operate largely in silos with minimal coordination. Teams often assess whether building basic connective tissue would be valuable.\n")
        
        # Funder coordination
        if self.flow_stats['multi_funder_pct'] < 10:
            lines.append("### 🔗 Strengthen Funder Coordination\n")
            lines.append("- **Build initial overlap:** Almost no grantees receive from multiple funders. Start by mapping where portfolios *could* overlap based on thematic focus, then facilitate introductions.\n")
        elif self.flow_stats['multi_funder_pct'] < 30:
            lines.append("### 🔗 Strengthen Funder Coordination\n")
            lines.append("- **Expand existing bridges:** Some overlap exists. Identify the grantees already receiving from multiple funders and explore whether they can serve as coordination anchors.\n")
        
        # Governance
        if self.governance_stats.get('shared_board_count', 0) == 0:
            lines.append("### 🏛️ Strengthen Governance Ties\n")
            lines.append("- **Identify potential bridge-builders:** No one currently serves on multiple boards. Look for respected individuals who could be nominated to additional boards to create connective tissue.\n")
        
        return "\n".join(lines)
    
    def _format_brokerage_section(self) -> list[str]:
        """Format brokerage ecosystem section for markdown report."""
        lines = []
        lines.append("## 🎭 Brokerage Ecosystem")
        lines.append("*How information and influence flow through the network*")
        lines.append("")
        
        lines.append(f"**Pattern:** {self.brokerage_data.pattern}")
        lines.append("")
        lines.append(self.brokerage_data.summary)
        lines.append("")
        
        # Community count
        lines.append(f"The network contains **{len(self.brokerage_data.communities)} distinct communities** detected via Louvain algorithm.")
        lines.append("")
        
        # Role distribution table
        lines.append("### Role Distribution")
        lines.append("")
        lines.append("| Role | Count | Description |")
        lines.append("|------|-------|-------------|")
        
        for role_key, config in BROKERAGE_ROLE_CONFIG.items():
            count = self.brokerage_data.role_counts.get(role_key, 0)
            if count > 0:
                lines.append(f"| {config['emoji']} {config['label']} | {count} | {config['description']} |")
        lines.append("")
        
        # Top brokers
        if self.brokerage_data.top_brokers:
            lines.append("### Key Brokers")
            lines.append("")
            for broker in self.brokerage_data.top_brokers:
                role_config = BROKERAGE_ROLE_CONFIG.get(broker['role'], {})
                emoji = role_config.get('emoji', '⚪')
                label = role_config.get('label', broker['role'])
                lines.append(f"- **{broker['label']}** — {emoji} {label}")
            lines.append("")
        
        # Strategic implications
        lines.append("### Strategic Implications")
        lines.append("")
        
        strategic_roles = ['gatekeeper', 'representative', 'consultant', 'liaison']
        strategic_count = sum(self.brokerage_data.role_counts.get(r, 0) for r in strategic_roles)
        
        if strategic_count >= 5:
            lines.append("- Multiple strategic brokers exist — consider engaging them as coordination facilitators")
            lines.append("- Gatekeepers and representatives can be entry points for cross-community initiatives")
        elif strategic_count >= 1:
            lines.append("- Few strategic brokers exist — they carry disproportionate coordination load")
            lines.append("- Consider building redundancy by developing additional cross-community connectors")
        else:
            lines.append("- No urgent structural interventions needed")
            lines.append("- Consider targeted investment in liaison development for strategic priorities")
        lines.append("")
        
        return lines
    
    def _compute_node_metrics(self) -> pd.DataFrame:
        """Compute per-node metrics for funder network."""
        if self.combined_org_graph is None or self.combined_org_graph.number_of_nodes() == 0:
            return pd.DataFrame()
        
        # Get org nodes only
        org_nodes = self.nodes_df[self.nodes_df['node_type'].isin(['org', 'organization'])].copy()
        
        # Compute centrality metrics
        degree = dict(self.combined_org_graph.degree())
        betweenness = nx.betweenness_centrality(self.combined_org_graph)
        
        try:
            eigenvector = nx.eigenvector_centrality(self.combined_org_graph, max_iter=1000)
        except:
            eigenvector = {n: 0 for n in self.combined_org_graph.nodes()}
        
        # Build metrics dataframe
        metrics = []
        for _, row in org_nodes.iterrows():
            node_id = row['node_id']
            if node_id in self.combined_org_graph:
                metrics.append({
                    'node_id': node_id,
                    'label': row.get('label', row.get('name', node_id)),
                    'node_type': row['node_type'],
                    'degree': degree.get(node_id, 0),
                    'betweenness': betweenness.get(node_id, 0),
                    'eigenvector': eigenvector.get(node_id, 0),
                    'brokerage_role': self.brokerage_data.roles.get(node_id, 'unknown') if self.brokerage_data else 'unknown'
                })
        
        return pd.DataFrame(metrics)
    
    def _generate_project_summary(self) -> ProjectSummary:
        """Generate project summary for funder network."""
        org_count = len(self.nodes_df[self.nodes_df['node_type'].isin(['org', 'organization'])])
        person_count = len(self.nodes_df[self.nodes_df['node_type'] == 'person'])
        
        grant_count = len(self.edges_df[self.edges_df['edge_type'].isin(['grant', 'funding'])])
        board_count = len(self.edges_df[self.edges_df['edge_type'].isin(['board_membership', 'board'])])
        
        return ProjectSummary(
            project_id=self.project_id,
            network_type='funder',
            node_counts={
                'total': len(self.nodes_df),
                'organizations': org_count,
                'people': person_count
            },
            edge_counts={
                'total': len(self.edges_df),
                'grants': grant_count,
                'board_memberships': board_count
            },
            funding={
                'total_amount': self.flow_stats.get('total_funding', 0),
                'funder_count': self.flow_stats.get('funder_count', 0),
                'grantee_count': self.flow_stats.get('grantee_count', 0)
            }
        )
