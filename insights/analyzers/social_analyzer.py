"""
InsightGraph — Social Network Analyzer

Analyzes ActorGraph social networks (LinkedIn companies/people, connections).
Provides centrality metrics, brokerage roles, sector analysis, and connector identification.

VERSION HISTORY:
----------------
v1.1.2 (2026-01-06): Fixed brokerage integration
- FIX: _compute_brokerage now matches base.py signature (uses betweenness_map)
- FIX: _format_brokerage_section uses BrokerageData attributes correctly

v1.1.1 (2026-01-06): Bug fix for graph building
- FIX: Exclude 'weight' from row dict spread to avoid duplicate keyword argument
- Affects build_connection_graph

v1.1.0 (2026-01-06): YAML Copy Map Integration
- Health labels now sourced from INSIGHTGRAPH_COPY_MAP_v1.yaml
- Interpretive guardrail added to markdown reports
- Health descriptions from YAML
- Single source of truth for narrative copy
- Graceful fallback if copy_manager unavailable

v1.0.0 (2025-12-31): Initial release
- SocialAnalyzer class implementing NetworkAnalyzer ABC
- 4 centrality metrics: degree, betweenness, eigenvector, closeness
- Sector/industry analysis from LinkedIn data
- Top connector identification
- Social health score formula
- Brokerage ecosystem (shared with funder analyzer)

NETWORK CHARACTERISTICS:
- node_type: company, person
- edge_type: connection (similar_companies, etc.)
- Metrics: centralities, sector distribution, connector rankings
"""

import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timezone
from collections import Counter

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
        return "Well-connected network"
    elif score >= 40:
        return "Moderately connected"
    else:
        return "Fragmented network"


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

SOCIAL_ANALYZER_VERSION = "1.1.2"

# =============================================================================
# Thresholds
# =============================================================================

TOP_CONNECTOR_COUNT = 10  # Number of top connectors to show
MIN_SECTOR_COUNT = 2  # Minimum sectors for diversity analysis


# =============================================================================
# Social Network Graph Builder
# =============================================================================

def build_connection_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build undirected graph from connection edges."""
    G = nx.Graph()
    
    # Add all nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row['node_id'], **row.to_dict())
    
    # Add connection edges
    for _, row in edges_df.iterrows():
        weight = row.get('weight', 1)
        if pd.isna(weight):
            weight = 1
        # Exclude 'weight' from row dict to avoid duplicate keyword argument
        row_dict = {k: v for k, v in row.to_dict().items() if k != 'weight'}
        G.add_edge(row['from_id'], row['to_id'], weight=weight, **row_dict)
    
    return G


# =============================================================================
# Social Network Metrics
# =============================================================================

def compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute centrality metrics for all nodes."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    
    # Degree centrality
    degree = dict(G.degree())
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    
    # Eigenvector centrality (with fallback)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector = {n: 0 for n in G.nodes()}
    
    # Closeness centrality
    closeness = nx.closeness_centrality(G)
    
    # Build dataframe
    metrics = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        metrics.append({
            'node_id': node_id,
            'label': node_data.get('label', node_data.get('name', node_id)),
            'node_type': node_data.get('node_type', 'unknown'),
            'industry': node_data.get('industry', ''),
            'degree': degree.get(node_id, 0),
            'betweenness': betweenness.get(node_id, 0),
            'eigenvector': eigenvector.get(node_id, 0),
            'closeness': closeness.get(node_id, 0),
        })
    
    return pd.DataFrame(metrics)


def analyze_sectors(nodes_df: pd.DataFrame) -> dict:
    """Analyze sector/industry distribution."""
    # Look for industry column
    industry_col = None
    for col in ['industry', 'sector', 'linkedin_industry']:
        if col in nodes_df.columns:
            industry_col = col
            break
    
    if industry_col is None:
        return {'available': False}
    
    # Count industries
    industries = nodes_df[industry_col].dropna()
    if len(industries) == 0:
        return {'available': False}
    
    industry_counts = Counter(industries)
    total = len(industries)
    
    # Top sectors
    top_sectors = industry_counts.most_common(5)
    
    return {
        'available': True,
        'sector_count': len(industry_counts),
        'top_sector': top_sectors[0][0] if top_sectors else 'Unknown',
        'top_sector_pct': (top_sectors[0][1] / total * 100) if top_sectors else 0,
        'top_5': [{'sector': s, 'count': c, 'pct': c/total*100} for s, c in top_sectors],
        'total_with_sector': total
    }


def compute_component_stats(G: nx.Graph) -> dict:
    """Compute connected component statistics."""
    if G.number_of_nodes() == 0:
        return {'largest_pct': 0, 'component_count': 0}
    
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    
    return {
        'largest_pct': len(largest) / G.number_of_nodes() * 100,
        'component_count': len(components),
        'largest_size': len(largest),
        'isolated_count': sum(1 for c in components if len(c) == 1)
    }


# =============================================================================
# Social Health Score
# =============================================================================

def compute_social_health_score(
    G: nx.Graph,
    metrics_df: pd.DataFrame,
    sector_analysis: dict,
    brokerage_data: BrokerageData
) -> HealthScore:
    """
    Compute network health score for social networks.
    
    Factors:
    - Connectivity (largest component %)
    - Brokerage capacity (strategic roles %)
    - Sector diversity
    - Degree distribution (Gini coefficient)
    """
    score = 0
    positive_factors = []
    risk_factors = []
    
    # 1. Connectivity (0-35 points)
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        largest = max(components, key=len) if components else set()
        largest_pct = len(largest) / G.number_of_nodes() * 100
        
        if largest_pct >= 80:
            score += 35
            positive_factors.append(f"🟢 **Highly connected** — {largest_pct:.0f}% in largest component")
        elif largest_pct >= 50:
            score += 20
            positive_factors.append(f"🟡 **Moderately connected** — {largest_pct:.0f}% in largest component")
        elif largest_pct >= 20:
            score += 10
            risk_factors.append(f"🔴 **Fragmented** — only {largest_pct:.0f}% in largest component")
        else:
            risk_factors.append(f"🔴 **Highly fragmented** — only {largest_pct:.0f}% connected")
    
    # 2. Brokerage capacity (0-25 points)
    if brokerage_data and brokerage_data.enabled:
        role_counts = brokerage_data.role_counts
        total_nodes = sum(role_counts.values())
        strategic_roles = ['gatekeeper', 'representative', 'consultant', 'liaison']
        strategic_count = sum(role_counts.get(r, 0) for r in strategic_roles)
        strategic_pct = (strategic_count / total_nodes * 100) if total_nodes > 0 else 0
        
        if strategic_pct >= 10:
            score += 25
            positive_factors.append(f"🟢 **Strong brokerage capacity** — {strategic_pct:.0f}% in strategic roles")
        elif strategic_pct >= 3:
            score += 15
            positive_factors.append(f"🟡 **Moderate brokerage** — {strategic_pct:.0f}% in strategic roles")
        elif strategic_pct >= 1:
            score += 8
        else:
            risk_factors.append("🔴 **Limited brokerage** — few nodes bridge communities")
    else:
        score += 12  # Neutral if no brokerage data
    
    # 3. Sector diversity (0-20 points)
    if sector_analysis.get('available'):
        sector_count = sector_analysis.get('sector_count', 1)
        top_sector_pct = sector_analysis.get('top_sector_pct', 100)
        
        if sector_count >= 10 and top_sector_pct < 40:
            score += 20
            positive_factors.append(f"🟢 **Diverse sectors** — {sector_count} industries, well distributed")
        elif sector_count >= 5:
            score += 12
        elif sector_count >= 2:
            score += 6
        else:
            risk_factors.append("🔴 **Limited sector diversity** — concentrated in one industry")
    else:
        score += 10  # Neutral if no sector data
    
    # 4. Degree distribution / hub concentration (0-20 points)
    # Moderate hubs = healthy, extreme concentration = fragility risk
    if len(metrics_df) > 0 and 'degree' in metrics_df.columns:
        degrees = metrics_df['degree'].values
        if len(degrees) > 1:
            # Compute Gini coefficient for degree distribution
            sorted_degrees = np.sort(degrees)
            n = len(sorted_degrees)
            cumsum = np.cumsum(sorted_degrees)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_degrees))) / (n * np.sum(sorted_degrees)) - (n + 1) / n
            gini = max(0, min(1, gini))  # Clamp to [0, 1]
            
            if gini < 0.3:
                score += 20
                positive_factors.append("🟢 **Balanced influence distribution** — connections spread across many nodes")
            elif gini < 0.5:
                score += 12
            elif gini < 0.7:
                score += 6
            else:
                risk_factors.append("🔴 **Over-centralization risk** — few nodes dominate connections, creating fragility")
        else:
            score += 12
    else:
        score += 12
    
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
# Social Analyzer Class
# =============================================================================

class SocialAnalyzer(NetworkAnalyzer):
    """
    Analyzer for social networks (ActorGraph LinkedIn data).
    
    Analyzes:
    - Centrality metrics (degree, betweenness, eigenvector, closeness)
    - Top connectors and influencers
    - Sector/industry distribution
    - Brokerage ecosystem (shared with funder analyzer)
    - Community structure
    """
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, project_id: str = "project"):
        super().__init__(nodes_df, edges_df, project_id)
        
        # Social-specific data
        self.connection_graph = None
        self.metrics_df = None
        self.sector_analysis = None
        self.brokerage_data = None
    
    def analyze(self) -> AnalysisResult:
        """Run social network analysis."""
        
        # Build connection graph
        self.connection_graph = build_connection_graph(self.nodes_df, self.edges_df)
        
        # Compute metrics
        self.metrics_df = compute_centrality_metrics(self.connection_graph)
        
        # Analyze sectors
        self.sector_analysis = analyze_sectors(self.nodes_df)
        
        # Compute brokerage roles
        self.brokerage_data = self._compute_brokerage()
        
        # Add brokerage roles to metrics
        if self.brokerage_data and self.brokerage_data.enabled:
            self.metrics_df['brokerage_role'] = self.metrics_df['node_id'].map(
                lambda x: self.brokerage_data.roles.get(x, 'unknown')
            )
        
        # Compute component stats
        component_stats = compute_component_stats(self.connection_graph)
        
        # Compute health score
        health = compute_social_health_score(
            self.connection_graph,
            self.metrics_df,
            self.sector_analysis,
            self.brokerage_data
        )
        
        # Generate insight cards
        cards = self._generate_insight_cards(health, component_stats)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(health, cards, component_stats)
        
        return AnalysisResult(
            health=health,
            cards=cards,
            metrics_df=self.metrics_df,
            project_summary=self._generate_project_summary(),
            markdown_report=markdown_report,
            network_type='social',
            brokerage_data=self.brokerage_data
        )
    
    def _compute_brokerage(self) -> BrokerageData:
        """Compute brokerage ecosystem data."""
        if self.connection_graph is None or self.connection_graph.number_of_nodes() == 0:
            return BrokerageData(enabled=False, roles={}, communities={}, summary="")
        
        # Compute betweenness for the graph
        betweenness_map = nx.betweenness_centrality(self.connection_graph)
        
        # Call shared brokerage logic from base.py
        brokerage_data = compute_brokerage_roles(self.connection_graph, betweenness_map)
        
        # Add top brokers
        brokerage_data.top_brokers = get_top_brokers(brokerage_data, self.nodes_df)
        
        return brokerage_data
    
    def _generate_insight_cards(self, health: HealthScore, component_stats: dict) -> list[InsightCard]:
        """Generate insight cards for social network."""
        cards = []
        
        # Network Health
        cards.append(InsightCard(
            card_id="network_health",
            title="Network Health Overview",
            use_case="Overall Assessment",
            summary=self._format_health_summary(health),
            details={"score": health.score, "label": health.label},
            signal_strength="high"
        ))
        
        # Connectivity
        cards.append(InsightCard(
            card_id="connectivity",
            title="Network Connectivity",
            use_case="Structural Analysis",
            summary=self._format_connectivity_summary(component_stats),
            details=component_stats,
            signal_strength="high"
        ))
        
        # Top Connectors
        cards.append(InsightCard(
            card_id="top_connectors",
            title="Top Connectors",
            use_case="Key Players",
            summary=self._format_connector_summary(),
            details={},
            signal_strength="high"
        ))
        
        # Sector Distribution
        if self.sector_analysis.get('available'):
            cards.append(InsightCard(
                card_id="sectors",
                title="Sector Distribution",
                use_case="Industry Analysis",
                summary=self._format_sector_summary(),
                details=self.sector_analysis,
                signal_strength="medium"
            ))
        
        return cards
    
    def _generate_markdown_report(self, health: HealthScore, cards: list[InsightCard], 
                                   component_stats: dict) -> str:
        """Generate markdown report for social network."""
        lines = []
        
        # Header
        lines.append(f"# Social Network Analysis Report")
        lines.append(f"**Project:** {self.project_id.upper()}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"**Network Type:** Social Network (LinkedIn)")
        lines.append("")
        
        # Summary stats
        summary = self._generate_project_summary()
        lines.append(f"**Nodes:** {summary.node_counts['total']} ({summary.node_counts['companies']} companies, {summary.node_counts['people']} people)")
        lines.append(f"**Connections:** {summary.edge_counts['total']}")
        if summary.node_counts.get('seeds', 0) > 0:
            lines.append(f"**Seed Organizations:** {summary.node_counts['seeds']} → **Discovered:** {summary.node_counts['discovered']}")
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
        
        # Top Connectors section
        lines.append("## 🔗 Top Connectors")
        lines.append("")
        lines.append("Organizations with the most connections in the network.")
        lines.append("")
        
        top_connectors = self.metrics_df.nlargest(10, 'degree')
        lines.append("| Rank | Organization | Connections | Betweenness | Sector |")
        lines.append("|------|--------------|-------------|-------------|--------|")
        for i, (_, row) in enumerate(top_connectors.iterrows(), 1):
            sector = row.get('industry', '') or ''
            lines.append(f"| {i} | {row['label']} | {int(row['degree'])} | {row['betweenness']:.3f} | {sector} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Brokerage section
        if self.brokerage_data and self.brokerage_data.enabled:
            lines.extend(self._format_brokerage_section())
            lines.append("---")
            lines.append("")
        
        # Sector analysis
        if self.sector_analysis.get('available'):
            lines.append("## 🏢 Sector Distribution")
            lines.append("")
            lines.append(self._format_sector_summary())
            lines.append("")
            lines.append("---")
            lines.append("")
        
        lines.append("*Report generated by C4C InsightGraph — Social Network Analyzer*")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Helper Formatters
    # =========================================================================
    
    def _format_health_summary(self, health: HealthScore) -> str:
        emoji = "🔴" if health.score < 40 else "🟡" if health.score < 70 else "🟢"
        return f"{emoji} **Network Health: {health.score}/100** — *{health.label}*"
    
    def _format_connectivity_summary(self, stats: dict) -> str:
        pct = stats.get('largest_pct', 0)
        if pct >= 80:
            return "🟢 **Highly connected**\n\nMost nodes are reachable from each other, enabling information flow across the network."
        elif pct >= 50:
            return "🟡 **Moderately connected**\n\nThe network has a core connected component but some isolated clusters exist."
        else:
            return f"🔴 **Fragmented** — only {pct:.0f}% of nodes connected\n\nMultiple disconnected clusters limit cross-network coordination."
    
    def _format_connector_summary(self) -> str:
        if len(self.metrics_df) == 0:
            return "No connector data available."
        
        top = self.metrics_df.nlargest(3, 'degree')
        names = ", ".join(top['label'].tolist())
        return f"🔗 **Top connectors:** {names}\n\nThese organizations have the most connections and can facilitate introductions across the network."
    
    def _format_sector_summary(self) -> str:
        if not self.sector_analysis.get('available'):
            return "⚪ **No sector data available**"
        
        sector_count = self.sector_analysis['sector_count']
        top_sector = self.sector_analysis['top_sector']
        top_pct = self.sector_analysis['top_sector_pct']
        
        if sector_count >= 10:
            return f"🟢 **Diverse sector representation**\n\n{sector_count} industries represented, with {top_sector} being most common ({top_pct:.0f}%)."
        elif sector_count >= 5:
            return f"🟡 **Moderate sector distribution**\n\n{sector_count} industries represented, with {top_sector} being most common ({top_pct:.0f}%)."
        else:
            return f"🔴 **Limited sector diversity**\n\nOnly {sector_count} industries, dominated by {top_sector} ({top_pct:.0f}%)."
    
    def _format_brokerage_section(self) -> list[str]:
        """Format brokerage ecosystem section for markdown report."""
        lines = []
        lines.append("## 🎭 Brokerage Ecosystem")
        lines.append("")
        lines.append("_How information and influence flow through the network_")
        lines.append("")
        
        if not self.brokerage_data or not self.brokerage_data.enabled:
            lines.append("> Brokerage analysis requires at least 10 nodes.")
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
        
        return lines
    
    def _generate_project_summary(self) -> ProjectSummary:
        """Generate project summary for social network."""
        company_count = len(self.nodes_df[self.nodes_df['node_type'].isin(['company', 'organization', 'org'])])
        person_count = len(self.nodes_df[self.nodes_df['node_type'] == 'person'])
        
        # Check for seed/discovered distinction
        seed_count = 0
        discovered_count = 0
        if 'is_seed' in self.nodes_df.columns:
            seed_count = self.nodes_df['is_seed'].sum()
            discovered_count = len(self.nodes_df) - seed_count
        
        return ProjectSummary(
            project_id=self.project_id,
            network_type='social',
            node_counts={
                'total': len(self.nodes_df),
                'companies': company_count,
                'people': person_count,
                'seeds': seed_count,
                'discovered': discovered_count
            },
            edge_counts={
                'total': len(self.edges_df),
                'connections': len(self.edges_df)
            },
            funding={}  # No funding in social networks
        )
