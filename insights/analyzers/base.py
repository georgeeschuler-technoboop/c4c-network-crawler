"""
InsightGraph — Base Analyzer Module

Defines the abstract base class for network analyzers and standardized output schema.
All network-type-specific analyzers inherit from NetworkAnalyzer.

VERSION HISTORY:
----------------
v1.0.1 (2025-12-31): Added INSIGHT_RESULT_SCHEMA_VERSION per team review
v1.0.0 (2025-12-31): Initial release
- NetworkAnalyzer ABC with analyze() method
- AnalysisResult dataclass for standardized output
- HealthScore, InsightCard, BrokerageData dataclasses
- Shared utility functions for graph building and metrics
- Network type detection logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime, timezone

# =============================================================================
# Version
# =============================================================================

BASE_VERSION = "1.0.1"

# Versioned schema contract - increment when output structure changes
INSIGHT_RESULT_SCHEMA_VERSION = "1.0"

# =============================================================================
# Type Definitions
# =============================================================================

NetworkType = Literal["funder", "social", "hybrid", "unknown"]
SourceApp = Literal["orggraph_us", "orggraph_ca", "actorgraph", "linked", "unknown"]


# =============================================================================
# Output Schema — Dataclasses
# =============================================================================

@dataclass
class HealthScore:
    """Network health assessment."""
    score: int  # 0-100
    label: str  # Human-readable label
    positive: list[str] = field(default_factory=list)  # Green flags
    risk: list[str] = field(default_factory=list)  # Red flags
    
    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "positive": self.positive,
            "risk": self.risk
        }


@dataclass
class InsightCard:
    """A single insight card for the report."""
    card_id: str
    use_case: str
    title: str
    summary: str
    ranked_rows: list[dict] = field(default_factory=list)
    evidence: dict = field(default_factory=lambda: {"node_ids": [], "edge_ids": []})
    health_factors: Optional[dict] = None
    
    def to_dict(self) -> dict:
        result = {
            "card_id": self.card_id,
            "use_case": self.use_case,
            "title": self.title,
            "summary": self.summary,
            "ranked_rows": self.ranked_rows,
            "evidence": self.evidence
        }
        if self.health_factors:
            result["health_factors"] = self.health_factors
        return result


@dataclass
class BrokerageData:
    """Louvain community-based brokerage analysis results."""
    enabled: bool = False
    roles: dict[str, str] = field(default_factory=dict)  # node_id -> role
    communities: dict[str, int] = field(default_factory=dict)  # node_id -> community_id
    role_counts: dict[str, int] = field(default_factory=dict)
    community_count: int = 0
    pattern: str = "unavailable"
    interpretation: str = ""
    strategic_implications: list[str] = field(default_factory=list)
    top_brokers: list[tuple] = field(default_factory=list)  # (node_id, label, role)
    
    def to_dict(self) -> dict:
        # Convert any numpy types to Python native for JSON serialization
        role_counts_clean = {str(k): int(v) for k, v in self.role_counts.items()}
        top_brokers_clean = [
            (str(node_id), str(label), str(role)) 
            for node_id, label, role in self.top_brokers
        ]
        return {
            "enabled": bool(self.enabled),
            "role_counts": role_counts_clean,
            "community_count": int(self.community_count),
            "pattern": str(self.pattern),
            "interpretation": str(self.interpretation),
            "strategic_implications": [str(s) for s in self.strategic_implications],
            "top_brokers": top_brokers_clean
        }


@dataclass
class ProjectSummary:
    """High-level project statistics."""
    generated_at: str
    network_type: NetworkType
    source_app: SourceApp
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    # Funder-specific (optional)
    funding: Optional[dict] = None
    governance: Optional[dict] = None
    # Social-specific (optional)
    sectors: Optional[dict] = None
    # Shared
    brokerage: Optional[dict] = None
    roles_region: Optional[dict] = None
    
    def to_dict(self) -> dict:
        result = {
            "generated_at": self.generated_at,
            "network_type": self.network_type,
            "source_app": self.source_app,
            "node_counts": self.node_counts,
            "edge_counts": self.edge_counts
        }
        if self.funding:
            result["funding"] = self.funding
        if self.governance:
            result["governance"] = self.governance
        if self.sectors:
            result["sectors"] = self.sectors
        if self.brokerage:
            result["brokerage"] = self.brokerage
        if self.roles_region:
            result["roles_region"] = self.roles_region
        return result


@dataclass
class AnalysisResult:
    """
    Standardized output from any network analyzer.
    
    Both FunderAnalyzer and SocialAnalyzer return this structure,
    though the contents of cards and metrics differ by network type.
    """
    # Metadata
    network_type: NetworkType
    source_app: SourceApp
    project_id: str
    generated_at: str
    
    # Core outputs
    health: HealthScore
    cards: list[InsightCard]
    metrics_df: pd.DataFrame
    project_summary: ProjectSummary
    brokerage: BrokerageData
    
    # Report
    markdown_report: str
    
    # Optional: raw data for downstream use
    nodes_df: Optional[pd.DataFrame] = None
    edges_df: Optional[pd.DataFrame] = None
    
    def to_insight_cards_dict(self) -> dict:
        """Convert to insight_cards.json format."""
        return {
            "schema_version": INSIGHT_RESULT_SCHEMA_VERSION,
            "project_id": self.project_id,
            "generated_at": self.generated_at,
            "network_type": self.network_type,
            "health": self.health.to_dict(),
            "brokerage": self.brokerage.to_dict(),
            "cards": [c.to_dict() for c in self.cards]
        }
    
    def to_project_summary_dict(self) -> dict:
        """Convert to project_summary.json format."""
        return self.project_summary.to_dict()


# =============================================================================
# Network Type Detection
# =============================================================================

def detect_network_type(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> NetworkType:
    """
    Detect network type from data characteristics.
    
    Returns:
        'funder' - OrgGraph data with grants/board memberships
        'social' - ActorGraph data with connections
        'hybrid' - Mixed data (linked network)
        'unknown' - Cannot determine
    """
    edge_types = set()
    if 'edge_type' in edges_df.columns:
        edge_types = set(edges_df['edge_type'].str.lower().unique())
    
    source_apps = set()
    if 'source_app' in nodes_df.columns:
        source_apps = set(nodes_df['source_app'].str.lower().unique())
    
    # Use set operations for robust detection
    funder_edge_types = {'grant', 'board', 'board_membership'}
    social_edge_types = {'connection', 'similar_companies', 'similar', 'colleague', 'follows'}
    
    has_funder_edges = bool(funder_edge_types & edge_types)
    has_social_edges = bool(social_edge_types & edge_types)
    has_actorgraph = bool({'actorgraph'} & source_apps)
    has_orggraph = any('orggraph' in app for app in source_apps)
    
    # Hybrid: has both funder and social edge types
    if has_funder_edges and has_social_edges:
        return 'hybrid'
    
    # Funder: has grants or board memberships
    if has_funder_edges:
        return 'funder'
    
    # Social: has connections or is from ActorGraph
    if has_social_edges or has_actorgraph:
        return 'social'
    
    # Fallback based on source app
    if has_orggraph:
        return 'funder'
    
    return 'unknown'


def detect_source_app(nodes_df: pd.DataFrame) -> SourceApp:
    """Detect primary source application from node data."""
    if 'source_app' not in nodes_df.columns:
        return 'unknown'
    
    source_apps = nodes_df['source_app'].str.lower().value_counts()
    if source_apps.empty:
        return 'unknown'
    
    primary = source_apps.index[0]
    
    if 'orggraph_us' in primary or 'orggraph-us' in primary:
        return 'orggraph_us'
    elif 'orggraph_ca' in primary or 'orggraph-ca' in primary:
        return 'orggraph_ca'
    elif 'actorgraph' in primary:
        return 'actorgraph'
    elif len(source_apps) > 1:
        return 'linked'
    
    return 'unknown'


# =============================================================================
# Abstract Base Class
# =============================================================================

class NetworkAnalyzer(ABC):
    """
    Abstract base class for network analyzers.
    
    Subclasses must implement:
    - analyze() -> AnalysisResult
    
    Shared utilities are provided as class methods.
    """
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, project_id: str = "project"):
        """
        Initialize analyzer with network data.
        
        Args:
            nodes_df: DataFrame with node_id, node_type, label, and type-specific columns
            edges_df: DataFrame with edge_id, from_id, to_id, edge_type, and type-specific columns
            project_id: Project identifier for outputs
        """
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        self.project_id = project_id
        self.network_type = detect_network_type(nodes_df, edges_df)
        self.source_app = detect_source_app(nodes_df)
    
    @abstractmethod
    def analyze(self) -> AnalysisResult:
        """
        Run analysis and return standardized result.
        
        Subclasses implement network-type-specific analysis logic.
        """
        pass
    
    # =========================================================================
    # Shared Utilities
    # =========================================================================
    
    @staticmethod
    def build_undirected_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
        """Build basic undirected graph from nodes and edges."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for _, row in nodes_df.iterrows():
            G.add_node(row['node_id'], **row.to_dict())
        
        # Add edges
        for _, row in edges_df.iterrows():
            weight = row.get('weight', 1)
            G.add_edge(row['from_id'], row['to_id'], weight=weight)
        
        return G
    
    @staticmethod
    def compute_basic_centralities(G: nx.Graph) -> dict[str, dict]:
        """
        Compute basic centrality metrics for all nodes.
        
        Returns dict with keys: degree, betweenness, eigenvector, closeness
        Each value is a dict of {node_id: score}
        """
        result = {
            'degree': {},
            'betweenness': {},
            'eigenvector': {},
            'closeness': {}
        }
        
        if G.number_of_nodes() == 0:
            return result
        
        # Degree centrality
        result['degree'] = dict(G.degree())
        
        # Betweenness
        if G.number_of_edges() > 0:
            result['betweenness'] = nx.betweenness_centrality(G)
        
        # Eigenvector (may fail on disconnected graphs)
        try:
            result['eigenvector'] = nx.eigenvector_centrality(G, max_iter=500)
        except nx.NetworkXError:
            # Fallback for disconnected graphs
            result['eigenvector'] = {n: 0.0 for n in G.nodes()}
        
        # Closeness
        result['closeness'] = nx.closeness_centrality(G)
        
        return result
    
    @staticmethod
    def compute_component_stats(G: nx.Graph) -> dict:
        """Compute connected component statistics."""
        if G.number_of_nodes() == 0:
            return {
                'n_components': 0,
                'largest_component_size': 0,
                'largest_component_pct': 0.0
            }
        
        components = list(nx.connected_components(G))
        largest = max(components, key=len) if components else set()
        
        return {
            'n_components': len(components),
            'largest_component_size': len(largest),
            'largest_component_pct': len(largest) / G.number_of_nodes() * 100
        }
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# =============================================================================
# Shared Brokerage Ecosystem Logic
# =============================================================================

# Brokerage Role Configuration (Louvain community-based)
BROKERAGE_ROLE_CONFIG = {
    "liaison": {
        "emoji": "🌉",
        "label": "Liaison",
        "color": "#D97706",
        "description": "Bridges different communities",
        "strategic_use": "Key for cross-sector coordination and information flow between clusters"
    },
    "gatekeeper": {
        "emoji": "🚪",
        "label": "Gatekeeper",
        "color": "#F97316",
        "description": "Controls access between groups",
        "strategic_use": "Engage early in stakeholder processes; they filter what reaches their community"
    },
    "representative": {
        "emoji": "📢",
        "label": "Representative",
        "color": "#10B981",
        "description": "Speaks for their community externally",
        "strategic_use": "Amplifies messages; good partners for communicating to their cluster"
    },
    "coordinator": {
        "emoji": "🧩",
        "label": "Coordinator",
        "color": "#3B82F6",
        "description": "Strengthens within-group connections",
        "strategic_use": "Helps build cohesion; engage for internal community organizing"
    },
    "consultant": {
        "emoji": "🧠",
        "label": "Consultant",
        "color": "#6366F1",
        "description": "External advisor to multiple groups",
        "strategic_use": "Brings outside perspective; useful for introducing new ideas"
    },
    "peripheral": {
        "emoji": "⚪",
        "label": "Peripheral",
        "color": "#9CA3AF",
        "description": "Edge of network",
        "strategic_use": "Potential for outreach and network expansion"
    },
}

MIN_NODES_FOR_BROKERAGE = 10


def compute_brokerage_roles(G: nx.Graph, betweenness_map: dict = None) -> BrokerageData:
    """
    Classify nodes into brokerage roles based on Louvain community structure.
    
    This is shared between funder and social analyzers.
    
    Args:
        G: Undirected graph for community detection
        betweenness_map: Optional pre-computed betweenness scores
        
    Returns:
        BrokerageData with roles, communities, counts, and interpretation
    """
    result = BrokerageData(
        role_counts={role: 0 for role in BROKERAGE_ROLE_CONFIG.keys()}
    )
    
    if G.number_of_nodes() < MIN_NODES_FOR_BROKERAGE:
        return result
    
    # Try to import community detection
    try:
        import community as community_louvain
    except ImportError:
        print("Warning: python-louvain not installed. Brokerage roles unavailable.")
        return result
    
    # Run Louvain community detection
    try:
        partition = community_louvain.best_partition(G)
    except Exception as e:
        print(f"Warning: Community detection failed: {e}")
        return result
    
    result.communities = partition
    result.community_count = len(set(partition.values()))
    result.enabled = True
    
    # Get betweenness if not provided
    if betweenness_map is None:
        betweenness_map = nx.betweenness_centrality(G) if G.number_of_edges() > 0 else {}
    
    # Classify each node
    for node in G.nodes():
        node_community = partition.get(node)
        if node_community is None:
            result.roles[node] = 'peripheral'
            result.role_counts['peripheral'] += 1
            continue
        
        neighbors = list(G.neighbors(node))
        if not neighbors:
            result.roles[node] = 'peripheral'
            result.role_counts['peripheral'] += 1
            continue
        
        # Count neighbors in same vs different communities
        neighbor_communities = [partition.get(n) for n in neighbors if partition.get(n) is not None]
        if not neighbor_communities:
            result.roles[node] = 'peripheral'
            result.role_counts['peripheral'] += 1
            continue
        
        same_community = sum(1 for c in neighbor_communities if c == node_community)
        diff_community = len(neighbor_communities) - same_community
        total = len(neighbor_communities)
        
        same_ratio = same_community / total if total > 0 else 0
        diff_ratio = diff_community / total if total > 0 else 0
        
        betweenness = betweenness_map.get(node, 0)
        
        # Role classification logic
        if betweenness > 0.1 and diff_ratio > 0.5:
            role = 'liaison'
        elif betweenness > 0.05 and diff_ratio > 0.3:
            role = 'gatekeeper'
        elif same_ratio > 0.7 and diff_ratio > 0.1:
            role = 'representative'
        elif same_ratio > 0.8:
            role = 'coordinator'
        elif diff_ratio > 0.5:
            role = 'consultant'
        else:
            role = 'peripheral'
        
        result.roles[node] = role
        result.role_counts[role] += 1
    
    # Generate interpretation
    result = _interpret_brokerage(result)
    
    return result


def _interpret_brokerage(data: BrokerageData) -> BrokerageData:
    """Add interpretation to brokerage data."""
    if not data.enabled:
        data.pattern = 'unavailable'
        data.interpretation = 'Brokerage analysis requires at least 10 nodes.'
        return data
    
    counts = data.role_counts
    total = sum(counts.values())
    if total == 0:
        data.pattern = 'unavailable'
        data.interpretation = 'No nodes could be classified.'
        return data
    
    # Calculate percentages
    liaison_pct = counts['liaison'] / total * 100
    gatekeeper_pct = counts['gatekeeper'] / total * 100
    peripheral_pct = counts['peripheral'] / total * 100
    strategic_roles = counts['liaison'] + counts['gatekeeper'] + counts['representative']
    strategic_pct = strategic_roles / total * 100
    
    if liaison_pct > 15:
        data.pattern = "liaison-rich"
        data.interpretation = (
            f"This network has strong cross-community connectivity ({counts['liaison']} liaisons). "
            "Information flows relatively freely between clusters, creating opportunities for "
            "coordination but also risk of message dilution."
        )
        data.strategic_implications = [
            "Liaisons are natural conveners for cross-sector initiatives",
            "Consider whether key messages are reaching all communities consistently"
        ]
    elif gatekeeper_pct > 15:
        data.pattern = "gatekeeper-concentrated"
        data.interpretation = (
            f"A small number of gatekeepers ({counts['gatekeeper']}) control information flow between communities. "
            "This creates efficiency but also bottleneck risk."
        )
        data.strategic_implications = [
            "Engage gatekeepers early — they determine what reaches their communities",
            "Monitor for single points of failure if key gatekeepers disengage"
        ]
    elif peripheral_pct > 60:
        data.pattern = "periphery-heavy"
        data.interpretation = (
            f"Most nodes ({counts['peripheral']}) sit at the network edge with limited brokerage capacity. "
            "The network may lack connective tissue for coordination."
        )
        data.strategic_implications = [
            "Consider capacity building to develop more connectors",
            "Identify peripheral nodes with potential to bridge communities"
        ]
    elif strategic_pct > 30:
        data.pattern = "well-brokered"
        data.interpretation = (
            f"This network has healthy brokerage capacity ({strategic_roles} nodes in strategic roles). "
            "Multiple pathways exist for information and coordination across communities."
        )
        data.strategic_implications = [
            "Leverage existing brokers for coordination initiatives",
            "The network has built-in resilience if individual brokers disengage"
        ]
    else:
        data.pattern = "balanced"
        data.interpretation = (
            "Brokerage roles are distributed without strong concentration. "
            "The network has moderate coordination capacity."
        )
        data.strategic_implications = [
            "No urgent structural interventions needed",
            "Consider targeted investment in liaison development for strategic priorities"
        ]
    
    return data


def get_top_brokers(brokerage_data: BrokerageData, nodes_df: pd.DataFrame, n: int = 8) -> list[tuple]:
    """
    Get top brokers (liaisons and gatekeepers prioritized) with their labels.
    
    Returns list of (node_id, label, role) tuples.
    """
    if not brokerage_data.enabled:
        return []
    
    roles = brokerage_data.roles
    node_labels = dict(zip(nodes_df['node_id'], nodes_df['label']))
    
    # Prioritize strategic roles
    priority_order = ['liaison', 'gatekeeper', 'representative', 'coordinator', 'consultant']
    
    brokers = []
    for role in priority_order:
        for node_id, node_role in roles.items():
            if node_role == role:
                label = node_labels.get(node_id, node_id)
                brokers.append((node_id, label, role))
        if len(brokers) >= n:
            break
    
    return brokers[:n]


def get_brokerage_badge(role: str) -> str:
    """Get emoji + label for a brokerage role."""
    config = BROKERAGE_ROLE_CONFIG.get(role, BROKERAGE_ROLE_CONFIG['peripheral'])
    return f"{config['emoji']} {config['label']}"
