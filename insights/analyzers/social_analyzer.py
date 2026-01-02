"""
InsightGraph — Social Network Analyzer

Analyzes ActorGraph social networks (LinkedIn companies/people, connections).
Provides centrality metrics, brokerage roles, sector analysis, and connector identification.

VERSION HISTORY:
----------------
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
# Version
# =============================================================================

SOCIAL_ANALYZER_VERSION = "1.0.0"

# =============================================================================
# Thresholds
# =============================================================================

TOP_CONNECTOR_PERCENTILE = 90  # Top 10% by degree are "connectors"
SECTOR_CONCENTRATION_THRESHOLD = 50  # >50% in one sector = concentrated
SECTOR_DIVERSITY_MIN = 5  # Minimum sectors for "diverse" label


# =============================================================================
# Social Network Graph Building
# =============================================================================

def build_connection_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build undirected connection graph from ActorGraph data.
    
    Handles edge types: connection, similar_companies, etc.
    """
    G = nx.Graph()
    
    # Add all nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(row['node_id'], **row.to_dict())
    
    # Add connection edges
    connection_types = ['connection', 'similar_companies', 'similar', 'colleague', 'follows']
    connection_mask = edges_df['edge_type'].str.lower().isin(connection_types)
    
    # If no recognized types, add all edges
    if not connection_mask.any():
        connection_mask = pd.Series([True] * len(edges_df))
    
    for _, row in edges_df[connection_mask].iterrows():
        weight = row.get('weight', 1)
        G.add_edge(row['from_id'], row['to_id'], weight=weight)
    
    return G


# =============================================================================
# Social Network Metrics
# =============================================================================

def compute_social_metrics(nodes_df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """
    Compute social network metrics for all nodes.
    
    Metrics:
    - degree: Raw connection count
    - betweenness: Bridge position score
    - eigenvector: Influence score (connected to well-connected nodes)
    - closeness: Reachability score
    """
    metrics = []
    
    # Compute centralities
    degree_dict = dict(G.degree())
    
    betweenness_dict = {}
    eigenvector_dict = {}
    closeness_dict = {}
    
    if G.number_of_edges() > 0:
        betweenness_dict = nx.betweenness_centrality(G)
        closeness_dict = nx.closeness_centrality(G)
        
        try:
            eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500)
        except nx.NetworkXError:
            # Fallback for disconnected graphs - compute per component
            eigenvector_dict = {}
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                if len(component) > 1:
                    try:
                        sub_eigen = nx.eigenvector_centrality(subgraph, max_iter=500)
                        eigenvector_dict.update(sub_eigen)
                    except:
                        for node in component:
                            eigenvector_dict[node] = 0.0
                else:
                    for node in component:
                        eigenvector_dict[node] = 0.0
    
    # Component mapping
    node_to_component = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            node_to_component[node] = i
    
    # Build metrics DataFrame
    for _, row in nodes_df.iterrows():
        node_id = row['node_id']
        
        m = {
            'node_id': node_id,
            'node_type': row.get('node_type', 'unknown'),
            'label': row.get('label', ''),
            'industry': row.get('industry', ''),
            'region': row.get('region', ''),
            'city': row.get('city', ''),
            'source_type': row.get('source_type', ''),  # seed vs discovered
            
            # Centrality metrics
            'degree': degree_dict.get(node_id, 0),
            'betweenness': betweenness_dict.get(node_id, 0),
            'eigenvector': eigenvector_dict.get(node_id, 0),
            'closeness': closeness_dict.get(node_id, 0),
            
            # Component
            'component_id': node_to_component.get(node_id, -1),
        }
        
        metrics.append(m)
    
    df = pd.DataFrame(metrics)
    
    # Add percentile ranks
    if len(df) > 0:
        for metric in ['degree', 'betweenness', 'eigenvector', 'closeness']:
            df[f'{metric}_pct'] = df[metric].rank(pct=True)
    
    # Flag top connectors
    if len(df) > 0:
        threshold = np.percentile(df['degree'], TOP_CONNECTOR_PERCENTILE)
        df['is_top_connector'] = (df['degree'] >= threshold).astype(int)
    else:
        df['is_top_connector'] = 0
    
    return df


def compute_sector_analysis(nodes_df: pd.DataFrame) -> dict:
    """
    Analyze sector/industry distribution from LinkedIn data.
    
    Returns:
        dict with sector_counts, top_sector, concentration metrics
    """
    # Get industry column (may be 'industry' or 'sector')
    industry_col = None
    for col in ['industry', 'sector', 'category']:
        if col in nodes_df.columns:
            industry_col = col
            break
    
    if industry_col is None or nodes_df[industry_col].isna().all():
        return {
            'enabled': False,
            'sector_counts': {},
            'top_sector': None,
            'top_sector_count': 0,
            'top_sector_pct': 0,
            'sector_count': 0,
            'is_concentrated': False,
            'is_diverse': False
        }
    
    # Count sectors
    sector_counts = nodes_df[industry_col].dropna().value_counts().to_dict()
    total = sum(sector_counts.values())
    
    if total == 0:
        return {
            'enabled': False,
            'sector_counts': {},
            'top_sector': None,
            'top_sector_count': 0,
            'top_sector_pct': 0,
            'sector_count': 0,
            'is_concentrated': False,
            'is_diverse': False
        }
    
    top_sector = max(sector_counts, key=sector_counts.get)
    top_count = int(sector_counts[top_sector])
    top_pct = float(top_count / total * 100)
    
    # Convert sector_counts values to Python int for JSON serialization
    sector_counts_clean = {k: int(v) for k, v in sector_counts.items()}
    
    return {
        'enabled': True,
        'sector_counts': sector_counts_clean,
        'top_sector': str(top_sector),
        'top_sector_count': top_count,
        'top_sector_pct': top_pct,
        'sector_count': int(len(sector_counts)),
        'is_concentrated': bool(top_pct > SECTOR_CONCENTRATION_THRESHOLD),
        'is_diverse': bool(len(sector_counts) >= SECTOR_DIVERSITY_MIN and top_pct < 30)
    }


# =============================================================================
# Social Health Scoring
# =============================================================================

def compute_social_health(metrics_df: pd.DataFrame, component_stats: dict, 
                          brokerage_data: BrokerageData, sector_analysis: dict) -> HealthScore:
    """
    Compute 0-100 health score for social network.
    
    Formula (different from funder health):
    - Connectivity (25%): Largest component %
    - Brokerage capacity (25%): % in strategic roles
    - Sector diversity (25%): Number of sectors, concentration
    - Hub distribution (25%): 1 - Gini of degree distribution
    """
    positive_factors = []
    risk_factors = []
    score = 0.0
    
    # 1. Connectivity (0-25 points)
    largest_pct = component_stats.get('largest_component_pct', 0)
    if largest_pct >= 80:
        score += 25
        positive_factors.append(f"🟢 **Highly connected** — {largest_pct:.0f}% of nodes in largest component")
    elif largest_pct >= 50:
        score += 15
        positive_factors.append(f"🟡 **Moderately connected** — {largest_pct:.0f}% in largest component")
    elif largest_pct >= 20:
        score += 8
        risk_factors.append(f"🔴 **Fragmented** — only {largest_pct:.0f}% in largest component")
    else:
        risk_factors.append(f"🔴 **Highly fragmented** — only {largest_pct:.0f}% connected")
    
    # 2. Brokerage capacity (0-25 points)
    if brokerage_data.enabled:
        total_nodes = sum(brokerage_data.role_counts.values())
        strategic = (brokerage_data.role_counts.get('liaison', 0) + 
                    brokerage_data.role_counts.get('gatekeeper', 0) +
                    brokerage_data.role_counts.get('representative', 0))
        strategic_pct = strategic / total_nodes * 100 if total_nodes > 0 else 0
        
        if strategic_pct >= 30:
            score += 25
            positive_factors.append(f"🟢 **Strong brokerage capacity** — {strategic_pct:.0f}% in strategic roles")
        elif strategic_pct >= 15:
            score += 15
            positive_factors.append(f"🟡 **Moderate brokerage** — {strategic_pct:.0f}% in strategic roles")
        elif strategic_pct >= 5:
            score += 8
        else:
            risk_factors.append("🔴 **Limited brokerage** — few nodes bridge communities")
    else:
        score += 12  # Default if brokerage couldn't be computed
    
    # 3. Sector diversity (0-25 points)
    if sector_analysis.get('enabled'):
        if sector_analysis['is_diverse']:
            score += 25
            positive_factors.append(f"🟢 **Sector diversity** — {sector_analysis['sector_count']} industries represented")
        elif not sector_analysis['is_concentrated']:
            score += 15
        else:
            score += 5
            risk_factors.append(f"🔴 **Sector concentration** — {sector_analysis['top_sector_pct']:.0f}% in {sector_analysis['top_sector']}")
    else:
        score += 12  # Default if no sector data
    
    # 4. Hub distribution (0-25 points) - Risk-sensitive framing
    # Moderate hubs = healthy, extreme concentration = fragility risk
    if len(metrics_df) > 0 and 'degree' in metrics_df.columns:
        degrees = metrics_df['degree'].values
        degrees_sorted = np.sort(degrees)
        n = len(degrees_sorted)
        if n > 0 and degrees_sorted.sum() > 0:
            cumulative = np.cumsum(degrees_sorted)
            gini = (2 * np.sum((np.arange(1, n+1) * degrees_sorted))) / (n * np.sum(degrees_sorted)) - (n + 1) / n
            gini = max(0, min(1, gini))  # Clamp to [0,1]
            distribution_score = (1 - gini) * 25
            score += distribution_score
            
            if gini < 0.3:
                positive_factors.append("🟢 **Balanced influence distribution** — connections spread across many nodes")
            elif gini < 0.5:
                pass  # Neutral - no flag needed
            else:
                risk_factors.append("🔴 **Over-centralization risk** — few nodes dominate connections, creating fragility")
        else:
            score += 12
    else:
        score += 12
    
    score = max(0, min(100, int(score)))
    
    if score >= 70:
        label = "Well-connected network"
    elif score >= 40:
        label = "Moderately connected"
    else:
        label = "Fragmented network"
    
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
        self.metrics_df = compute_social_metrics(self.nodes_df, self.connection_graph)
        
        # Compute sector analysis
        self.sector_analysis = compute_sector_analysis(self.nodes_df)
        
        # Compute brokerage (shared logic from base.py)
        betweenness_map = dict(zip(self.metrics_df['node_id'], self.metrics_df['betweenness']))
        self.brokerage_data = compute_brokerage_roles(self.connection_graph, betweenness_map)
        self.brokerage_data.top_brokers = get_top_brokers(self.brokerage_data, self.nodes_df)
        
        # Add brokerage to metrics
        if self.brokerage_data.enabled:
            self.metrics_df['brokerage_role'] = self.metrics_df['node_id'].map(self.brokerage_data.roles)
            self.metrics_df['community_id'] = self.metrics_df['node_id'].map(self.brokerage_data.communities)
        
        # Compute component stats
        component_stats = self.compute_component_stats(self.connection_graph)
        
        # Compute health
        health = compute_social_health(
            self.metrics_df,
            component_stats,
            self.brokerage_data,
            self.sector_analysis
        )
        
        # Generate insight cards
        cards = self._generate_insight_cards(health, component_stats)
        
        # Generate project summary
        project_summary = self._generate_project_summary()
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(health, cards, component_stats)
        
        return AnalysisResult(
            network_type='social',
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
        """Generate social-network-specific insight cards."""
        cards = []
        
        # Network Health Overview
        cards.append(InsightCard(
            card_id="network_health",
            use_case="System Framing",
            title="Network Health Overview",
            summary=self._format_health_summary(health),
            ranked_rows=[
                {"indicator": "Health Score", "value": f"{health.score}/100", "interpretation": health.label},
                {"indicator": "Nodes", "value": str(len(self.nodes_df)), "interpretation": ""},
                {"indicator": "Connections", "value": str(len(self.edges_df)), "interpretation": ""},
                {"indicator": "Largest Component", "value": f"{component_stats['largest_component_pct']:.0f}%",
                 "interpretation": self._interpret_connectivity(component_stats)},
                {"indicator": "Communities", "value": str(self.brokerage_data.community_count) if self.brokerage_data.enabled else "N/A",
                 "interpretation": ""}
            ],
            health_factors={"positive": health.positive, "risk": health.risk}
        ))
        
        # Top Connectors
        top_connectors = self.metrics_df.nlargest(10, 'degree')[['label', 'degree', 'betweenness', 'industry']].copy()
        connector_rows = []
        for i, (_, row) in enumerate(top_connectors.iterrows(), 1):
            connector_rows.append({
                "rank": i,
                "name": row['label'],
                "connections": int(row['degree']),
                "betweenness": f"{row['betweenness']:.3f}",
                "sector": row.get('industry', '') or ''
            })
        
        cards.append(InsightCard(
            card_id="top_connectors",
            use_case="Connector Analysis",
            title="Top Connectors",
            summary=self._format_connector_summary(),
            ranked_rows=connector_rows
        ))
        
        # Sector Distribution
        if self.sector_analysis.get('enabled'):
            sector_rows = []
            for sector, count in sorted(self.sector_analysis['sector_counts'].items(), key=lambda x: -x[1])[:10]:
                pct = count / sum(self.sector_analysis['sector_counts'].values()) * 100
                sector_rows.append({
                    "sector": sector,
                    "count": count,
                    "percentage": f"{pct:.1f}%"
                })
            
            cards.append(InsightCard(
                card_id="sector_distribution",
                use_case="Sector Analysis",
                title="Sector Distribution",
                summary=self._format_sector_summary(),
                ranked_rows=sector_rows
            ))
        
        # Brokerage Ecosystem
        if self.brokerage_data.enabled:
            role_rows = []
            for role in ['liaison', 'gatekeeper', 'representative', 'coordinator', 'consultant', 'peripheral']:
                count = self.brokerage_data.role_counts.get(role, 0)
                if count > 0:
                    config = BROKERAGE_ROLE_CONFIG[role]
                    role_rows.append({
                        "role": f"{config['emoji']} {config['label']}",
                        "count": count,
                        "description": config['description']
                    })
            
            cards.append(InsightCard(
                card_id="brokerage_ecosystem",
                use_case="Brokerage Analysis",
                title="Brokerage Ecosystem",
                summary=self._format_brokerage_summary(),
                ranked_rows=role_rows
            ))
        
        # Decision Options
        cards.append(InsightCard(
            card_id="decision_options",
            use_case="Strategic Considerations",
            title="Strategic Considerations",
            summary=self._generate_decision_options(health)
        ))
        
        return cards
    
    def _generate_project_summary(self) -> ProjectSummary:
        """Generate project summary for social network."""
        # Handle missing node_type column gracefully
        company_count = 0
        person_count = 0
        if 'node_type' in self.nodes_df.columns:
            company_count = int(len(self.nodes_df[self.nodes_df['node_type'].str.lower() == 'company']))
            person_count = int(len(self.nodes_df[self.nodes_df['node_type'].str.lower() == 'person']))
        
        # Handle missing source_type column gracefully
        seed_count = 0
        if 'source_type' in self.nodes_df.columns:
            seed_count = int(len(self.nodes_df[self.nodes_df['source_type'].str.lower() == 'seed']))
        discovered_count = int(len(self.nodes_df)) - seed_count
        
        return ProjectSummary(
            generated_at=self.get_timestamp(),
            network_type='social',
            source_app=self.source_app,
            node_counts={
                "total": int(len(self.nodes_df)),
                "companies": company_count,
                "people": person_count,
                "seeds": seed_count,
                "discovered": discovered_count
            },
            edge_counts={
                "total": int(len(self.edges_df)),
                "connections": int(len(self.edges_df))  # All edges are connections in social networks
            },
            sectors=self.sector_analysis if self.sector_analysis.get('enabled') else None,
            brokerage=self.brokerage_data.to_dict() if self.brokerage_data else None
        )
    
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
        
        # Health section
        health_emoji = "🟢" if health.score >= 70 else "🟡" if health.score >= 40 else "🔴"
        lines.append(f"## {health_emoji} Network Health: {health.score}/100 ({health.label})")
        lines.append("")
        lines.append("This score reflects structural connectivity, brokerage capacity, and sector diversity.")
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
        
        # Sector section
        if self.sector_analysis and self.sector_analysis.get('enabled'):
            lines.extend(self._format_sector_section())
            lines.append("---")
            lines.append("")
        
        # Community structure
        lines.append("## 🧩 Community Structure")
        lines.append("")
        lines.append(f"The network contains **{component_stats['n_components']} connected components**.")
        lines.append(f"The largest component includes **{component_stats['largest_component_pct']:.0f}%** of all nodes.")
        lines.append("")
        
        if self.brokerage_data.enabled:
            lines.append(f"Louvain community detection identified **{self.brokerage_data.community_count} communities** within the network.")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Decision Options
        lines.append("## 🧭 Strategic Considerations")
        lines.append("")
        lines.append(self._generate_decision_options(health))
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
    
    def _interpret_connectivity(self, component_stats: dict) -> str:
        pct = component_stats['largest_component_pct']
        if pct >= 80:
            return "Highly connected — most nodes reachable from each other"
        elif pct >= 50:
            return "Moderately connected"
        else:
            return f"Fragmented — only {pct:.0f}% of nodes connected"
    
    def _format_connector_summary(self) -> str:
        top_count = (self.metrics_df['is_top_connector'] == 1).sum()
        total = len(self.metrics_df)
        return f"🔗 **{top_count} top connectors** (top 10% by connection count)\n\nThese organizations are the most connected in the network and may serve as hubs for information flow and coordination."
    
    def _format_sector_summary(self) -> str:
        if not self.sector_analysis.get('enabled'):
            return "⚪ No sector data available"
        
        if self.sector_analysis['is_diverse']:
            return f"🟢 **Diverse sector representation**\n\n{self.sector_analysis['sector_count']} industries represented. The network spans multiple sectors, providing broad reach."
        elif self.sector_analysis['is_concentrated']:
            return f"🔴 **Sector concentration**\n\n{self.sector_analysis['top_sector_pct']:.0f}% of organizations are in {self.sector_analysis['top_sector']}. Consider expanding to other sectors for broader impact."
        else:
            return f"🟡 **Moderate sector distribution**\n\n{self.sector_analysis['sector_count']} industries represented, with {self.sector_analysis['top_sector']} being most common ({self.sector_analysis['top_sector_pct']:.0f}%)."
    
    def _format_brokerage_summary(self) -> str:
        if not self.brokerage_data or not self.brokerage_data.enabled:
            return "⚪ Brokerage analysis requires at least 10 nodes."
        
        pattern_display = self.brokerage_data.pattern.replace('-', ' ').title()
        return f"**Pattern:** {pattern_display}\n\n{self.brokerage_data.interpretation}"
    
    def _generate_decision_options(self, health: HealthScore) -> str:
        lines = ["_These observations highlight structural patterns; they are not recommendations._\n"]
        
        # Based on health
        if health.score < 40:
            lines.append("### Network Structure")
            lines.append("The network appears **fragmented**. Many nodes are isolated or in small clusters. Consider whether bridging disconnected groups would add value.\n")
        elif health.score < 70:
            lines.append("### Network Structure")
            lines.append("The network shows **moderate connectivity**. Some clusters exist with varying levels of connection between them.\n")
        else:
            lines.append("### Network Structure")
            lines.append("The network is **well-connected**. Most nodes can reach each other through relatively short paths.\n")
        
        # Top connectors
        top_connector = self.metrics_df.nlargest(1, 'degree').iloc[0] if len(self.metrics_df) > 0 else None
        if top_connector is not None:
            lines.append("### Key Connectors")
            lines.append(f"**{top_connector['label']}** has the most connections ({int(top_connector['degree'])}). Key connectors like this often serve as information hubs.\n")
        
        # Brokerage
        if self.brokerage_data.enabled and self.brokerage_data.role_counts.get('liaison', 0) > 0:
            lines.append("### Brokerage Opportunity")
            lines.append(f"{self.brokerage_data.role_counts['liaison']} liaisons bridge different communities. Engaging them may accelerate cross-group coordination.\n")
        
        # Sector
        if self.sector_analysis.get('enabled') and self.sector_analysis['is_concentrated']:
            lines.append("### Sector Consideration")
            lines.append(f"The network is concentrated in {self.sector_analysis['top_sector']}. Intentional outreach to other sectors could broaden reach and resilience.\n")
        
        return "\n".join(lines)
    
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
        
        # Strategic implications
        if self.brokerage_data.strategic_implications:
            lines.append("### Strategic Implications")
            lines.append("")
            for impl in self.brokerage_data.strategic_implications:
                lines.append(f"- {impl}")
            lines.append("")
        
        return lines
    
    def _format_sector_section(self) -> list[str]:
        """Format sector distribution section for markdown report."""
        lines = []
        lines.append("## 📊 Sector Distribution")
        lines.append("")
        
        if not self.sector_analysis.get('enabled'):
            lines.append("> No sector/industry data available.")
            return lines
        
        lines.append(self._format_sector_summary())
        lines.append("")
        
        # Top sectors table
        lines.append("### Top Sectors")
        lines.append("")
        lines.append("| Sector | Count | Percentage |")
        lines.append("|--------|-------|------------|")
        
        total = sum(self.sector_analysis['sector_counts'].values())
        for sector, count in sorted(self.sector_analysis['sector_counts'].items(), key=lambda x: -x[1])[:10]:
            pct = count / total * 100
            lines.append(f"| {sector} | {count} | {pct:.1f}% |")
        
        lines.append("")
        
        return lines
