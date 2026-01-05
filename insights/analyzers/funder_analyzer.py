"""
InsightGraph — Funder Network Analyzer

Analyzes OrgGraph funder networks (foundations, grantees, board members).
Wraps existing run.py logic with the NetworkAnalyzer interface.

VERSION HISTORY:
----------------
v1.2.0 (2026-01-06): Proper package structure fix
- FIX: Use simple absolute import 'from insights.copy_manager import ...'
- REQUIRES: insights/__init__.py must exist
- REQUIRES: run.py v4.1.0+ (no sys.path manipulation)
- REMOVED: Multi-strategy import fallbacks (they masked the real bug)

v1.1.9 (2026-01-06): Robust multi-strategy import for copy_manager
- ADD: Three import strategies (absolute, path-based, relative)
- ADD: Detailed error reporting for all strategies
- FIX: Path-based import adds parent dir to sys.path

v1.1.8 (2026-01-06): Fix import path for Streamlit Cloud
- FIX: Use absolute import 'from insights.copy_manager' (Streamlit doesn't treat dirs as packages)
- KEEP: Fallback to relative import for package-style usage

v1.1.7 (2026-01-06): Add YAML loading diagnostics
- ADD: Capture and display specific error when copy_manager fails to load
- ADD: Report footer shows exact error message for debugging

v1.1.6 (2026-01-06): Fix copy_manager import path
- FIX: Import from ..copy_manager (parent directory) not copy_manager
- ADD: Report footer shows (YAML) or (fallback) to confirm copy source

v1.1.5 (2026-01-06): Quality fixes + deployment verification
- FIX: Remove invalid 'summary' kwarg from BrokerageData (field is 'interpretation')
- FIX: generated_at now uses UTC (datetime.now(timezone.utc))
- FIX: source_app now detected via base class (supports orggraph_us/orggraph_ca/actorgraph)
- ADD: ANALYZER_BUILD marker in report header for deployment verification

v1.1.4 (2026-01-06): Fix AnalysisResult fields
- FIX: Use 'brokerage' not 'brokerage_data'
- FIX: Add all required fields: source_app, project_id, generated_at

v1.1.3 (2026-01-06): Fix ProjectSummary fields
- FIX: ProjectSummary uses generated_at, network_type, source_app (not project_id)
- FIX: Matches actual base.py ProjectSummary dataclass signature

v1.1.2 (2026-01-06): Fixed brokerage integration + InsightCard fields
- FIX: _compute_brokerage now matches base.py signature (uses betweenness_map)
- FIX: _format_brokerage_section uses BrokerageData attributes correctly
- FIX: InsightCard uses 'health_factors'/'evidence' instead of non-existent 'details'
- FIX: Removed non-existent 'signal_strength' parameter from InsightCard

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

# Use absolute package import (requires insights/__init__.py)
COPY_MANAGER_AVAILABLE = False
COPY_MANAGER_ERROR = None

try:
    from insights.copy_manager import get_copy_manager
    # Test that it can actually load the YAML
    _test_cm = get_copy_manager()
    COPY_MANAGER_AVAILABLE = True
except ImportError as e:
    COPY_MANAGER_ERROR = f"ImportError: {e}"
except FileNotFoundError as e:
    COPY_MANAGER_ERROR = f"FileNotFoundError: {e}"
except Exception as e:
    COPY_MANAGER_ERROR = f"{type(e).__name__}: {e}"


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

FUNDER_ANALYZER_VERSION = "1.2.0"
ANALYZER_BUILD = "v1.2.0-2026-01-06-package-fix"

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
            network_type='funder',
            source_app=self.source_app,  # Detected in base class __init__
            project_id=self.project_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            health=health,
            cards=cards,
            metrics_df=metrics_df,
            project_summary=self._generate_project_summary(),
            brokerage=self.brokerage_data,
            markdown_report=markdown_report
        )
    
    def _compute_brokerage(self) -> BrokerageData:
        """Compute brokerage ecosystem data."""
        # Use combined org graph for community detection and brokerage
        if self.combined_org_graph is None or self.combined_org_graph.number_of_nodes() == 0:
            return BrokerageData(enabled=False, roles={}, communities={})
        
        # Compute betweenness for the graph
        betweenness_map = nx.betweenness_centrality(self.combined_org_graph)
        
        # Call shared brokerage logic from base.py
        brokerage_data = compute_brokerage_roles(self.combined_org_graph, betweenness_map)
        
        # Add top brokers
        brokerage_data.top_brokers = get_top_brokers(brokerage_data, self.nodes_df)
        
        return brokerage_data
    
    def _generate_insight_cards(self, health: HealthScore) -> list[InsightCard]:
        """Generate insight cards for funder network."""
        cards = []
        
        # Network Health Overview
        cards.append(InsightCard(
            card_id="network_health",
            title="Network Health Overview",
            use_case="Overall Assessment",
            summary=self._format_health_summary(health),
            health_factors={"score": health.score, "label": health.label}
        ))
        
        # Funding Concentration
        cards.append(InsightCard(
            card_id="funding_concentration",
            title="Funding Concentration",
            use_case="Funding Concentration",
            summary=self._format_concentration_summary(),
            evidence={"metrics": self.flow_stats}
        ))
        
        # Governance / Board Interlocks
        cards.append(InsightCard(
            card_id="governance",
            title="Shared Board Conduits",
            use_case="Board Network & Conduits",
            summary=self._format_governance_summary(),
            evidence={"metrics": self.governance_stats}
        ))
        
        # Hidden Brokers
        cards.append(InsightCard(
            card_id="hidden_brokers",
            title="Hidden Brokers",
            use_case="Brokerage Roles",
            summary=self._format_broker_summary()
        ))
        
        # Strategic Recommendations
        cards.append(InsightCard(
            card_id="recommendations",
            title="Strategic Considerations",
            use_case="Strategic Considerations",
            summary=self._format_recommendations(health)
        ))
        
        return cards
    
    def _generate_markdown_report(self, health: HealthScore, cards: list[InsightCard]) -> str:
        """Generate markdown report for funder network."""
        lines = []
        
        # Header
        lines.append(f"# Funder Network Analysis Report")
        lines.append(f"**Project:** {self.project_id.upper()}")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"**Network Type:** Funder Network ({self.source_app})")
        lines.append(f"**Analyzer:** {ANALYZER_BUILD}")
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
        
        if COPY_MANAGER_AVAILABLE:
            copy_status = "YAML"
        else:
            copy_status = f"fallback: {COPY_MANAGER_ERROR}"
        lines.append(f"*Report generated by C4C InsightGraph — Funder Network Analyzer ({copy_status})*")
        
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
            generated_at=datetime.now(timezone.utc).isoformat(),
            network_type='funder',
            source_app=self.source_app,  # Detected in base class __init__
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
