import streamlit as st
import pandas as pd
import requests
import json
import time
from io import StringIO, BytesIO
from collections import deque
from typing import Dict, List, Tuple, Optional, Sequence
import re
import os
import pathlib
from datetime import datetime
import socket
import zipfile
import networkx as nx
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go


# ============================================================================
# NETWORK INSIGHTS (Badges, Health Score, Breakpoints)
# ============================================================================

CENTRALITY_LEVELS = ["low", "medium", "high", "extreme"]


@dataclass
class MetricBreakpoints:
    """Quantile-based thresholds for a single centrality metric."""
    low: float
    medium: float
    high: float


@dataclass
class NetworkStats:
    """Container for network-level statistics."""
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    avg_clustering: float
    n_components: int
    largest_component_size: int


def compute_breakpoints(values: Sequence[float]) -> MetricBreakpoints:
    """Compute adaptive thresholds based on quantiles (40%/80%/95%)."""
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return MetricBreakpoints(low=0.0, medium=0.0, high=0.0)
    q40, q80, q95 = np.quantile(arr, [0.40, 0.80, 0.95])
    return MetricBreakpoints(low=q40, medium=q80, high=q95)


def classify_value(value: float, bp: MetricBreakpoints) -> str:
    """Map a raw centrality value to low/medium/high/extreme."""
    if value is None:
        return "low"
    if value <= bp.low:
        return "low"
    elif value <= bp.medium:
        return "medium"
    elif value <= bp.high:
        return "high"
    else:
        return "extreme"


# Badge configurations per metric per level
BADGE_CONFIG: Dict[str, Dict[str, Dict[str, str]]] = {
    "degree": {
        "low":     {"emoji": "⚪", "label": "Low connectivity",     "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Moderate connectivity","color": "#3B82F6"},
        "high":    {"emoji": "🟢", "label": "Highly connected",     "color": "#10B981"},
        "extreme": {"emoji": "🔥", "label": "Super hub",            "color": "#F97316"},
    },
    "betweenness": {
        "low":     {"emoji": "⚪", "label": "Within cluster",       "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Occasional bridge",   "color": "#3B82F6"},
        "high":    {"emoji": "🟠", "label": "Key broker",          "color": "#F97316"},
        "extreme": {"emoji": "🚨", "label": "Critical bottleneck", "color": "#DC2626"},
    },
    "closeness": {
        "low":     {"emoji": "⚪", "label": "Hard to reach",        "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Moderate reach",      "color": "#3B82F6"},
        "high":    {"emoji": "💫", "label": "Well positioned",     "color": "#10B981"},
        "extreme": {"emoji": "🚀", "label": "System-wide access",  "color": "#0EA5E9"},
    },
    "eigenvector": {
        "low":     {"emoji": "⚪", "label": "Peripheral influence",   "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Connected to influence","color": "#3B82F6"},
        "high":    {"emoji": "⭐", "label": "Influence hub",         "color": "#FACC15"},
        "extreme": {"emoji": "👑", "label": "Power center",          "color": "#D97706"},
    },
}

METRIC_TOOLTIPS: Dict[str, str] = {
    "degree": "Degree centrality counts how many direct connections a node has. Higher values indicate hubs.",
    "betweenness": "Betweenness centrality measures how often a node sits on the shortest path between others. High values indicate brokers.",
    "closeness": "Closeness centrality captures how easily a node can reach everyone else. Higher values mean shorter average paths.",
    "eigenvector": "Eigenvector centrality reflects influence: being connected to other well-connected nodes.",
}


def render_badge(metric: str, level: str, small: bool = False) -> str:
    """Return HTML snippet for a colored badge."""
    cfg = BADGE_CONFIG[metric][level]
    pad = "2px 6px" if small else "4px 8px"
    font_size = "11px" if small else "13px"
    return (
        f"<span style='"
        f"background-color:{cfg['color']}20;"
        f"border:1px solid {cfg['color']};"
        f"border-radius:999px;"
        f"padding:{pad};"
        f"font-size:{font_size};"
        f"color:#111827;"
        f"margin-right:4px;"
        f"white-space:nowrap;"
        f"'>"
        f"{cfg['emoji']} {cfg['label']}"
        f"</span>"
    )


def get_badge_text(metric: str, level: str) -> str:
    """Return just the emoji + label (no HTML)."""
    cfg = BADGE_CONFIG[metric][level]
    return f"{cfg['emoji']} {cfg['label']}"


def describe_node_role(name: str, organization: str, levels: Dict[str, str]) -> str:
    """Generate human-friendly description of a node's structural role."""
    org_part = f" at {organization}" if organization else ""
    fragments: List[str] = []

    if levels.get("degree") in ("high", "extreme"):
        fragments.append("acts as a hub for many direct relationships")
    elif levels.get("degree") == "low":
        fragments.append("sits on the edge of the network")

    if levels.get("betweenness") == "extreme":
        fragments.append("is a critical bridge between otherwise disconnected groups")
    elif levels.get("betweenness") == "high":
        fragments.append("often brokers interactions between different clusters")

    if levels.get("closeness") in ("high", "extreme"):
        fragments.append("can quickly reach most other actors in the system")

    if levels.get("eigenvector") in ("high", "extreme"):
        fragments.append("is connected to other highly influential actors")

    if not fragments:
        return f"{name}{org_part} plays a more peripheral structural role in this network."

    joined = "; ".join(fragments)
    return f"{name}{org_part} {joined}."


def centralization_index(values: Sequence[float]) -> float:
    """Measure how dominant the top node is (0-1, 1=fully centralized)."""
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size <= 1 or arr.sum() == 0:
        return 0.0
    top_share = arr.max() / arr.sum()
    n = arr.size
    min_share = 1.0 / n
    denom = 1.0 - min_share
    if denom <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (top_share - min_share) / denom)))


def compute_network_health(
    stats: NetworkStats,
    degree_values: Sequence[float],
    betweenness_values: Sequence[float],
) -> Tuple[int, str]:
    """Compute 0-100 health score and label."""
    if stats.n_nodes == 0:
        return 0, "No data"

    # Connectivity score (0-25)
    target_min, target_max = 2.0, 8.0
    deg = stats.avg_degree
    if deg <= target_min:
        connectivity = 0.0
    elif deg >= target_max:
        connectivity = 1.0
    else:
        connectivity = (deg - target_min) / (target_max - target_min)
    connectivity_score = connectivity * 25.0

    # Cohesion score (0-25)
    largest_share = stats.largest_component_size / stats.n_nodes
    cohesion_score = largest_share * 25.0

    # Fragmentation penalty (0-15)
    if stats.n_components <= 1:
        fragmentation_penalty = 0.0
    else:
        frag = min(stats.n_components - 1, 9) / 9.0
        fragmentation_penalty = frag * 15.0

    # Centralization penalty (0-25)
    deg_cent = centralization_index(degree_values)
    btw_cent = centralization_index(betweenness_values)
    centralization = 0.5 * (deg_cent + btw_cent)
    centralization_penalty = centralization * 25.0

    base = 30.0
    raw_score = base + connectivity_score + cohesion_score - fragmentation_penalty - centralization_penalty
    score = int(max(0, min(100, round(raw_score))))

    if score >= 70:
        label = "Healthy cohesion"
    elif score >= 40:
        label = "Mixed signals"
    else:
        label = "Fragile / at risk"

    return score, label


def render_health_summary(score: int, label: str):
    """Render the network health score in Streamlit."""
    if label == "Healthy cohesion":
        color = "🟢"
    elif label == "Mixed signals":
        color = "🟡"
    else:
        color = "🔴"
    st.markdown(f"### {color} Network Health: **{score} / 100** — *{label}*")
    st.caption("This score reflects how well people are connected, how unified the system is, and how influence is distributed.")


def render_health_details(
    stats: NetworkStats,
    degree_values: Sequence[float],
    betweenness_values: Sequence[float],
):
    """Render detailed breakdown of health score in plain language."""
    with st.expander("🔍 Health Score Breakdown"):
        # Calculate values
        largest_share = stats.largest_component_size / max(stats.n_nodes, 1) * 100
        deg_cent = centralization_index(degree_values)
        btw_cent = centralization_index(betweenness_values)
        power_concentrated = deg_cent > 0.5 or btw_cent > 0.5
        
        # Collect positive factors
        positives = []
        if largest_share >= 80:
            positives.append(f"🟢 **The network is highly unified** — {largest_share:.0f}% of people can reach each other through the network, which means ideas and information can spread broadly.")
        if not power_concentrated:
            positives.append("🟢 **Influence is distributed** — no single actor dominates the network, reducing gatekeeping and single points of failure.")
        if stats.avg_degree >= 4:
            positives.append(f"🟢 **Good direct connectivity** — people have an average of {stats.avg_degree:.1f} direct connections, enabling organic collaboration.")
        
        # Collect risk factors
        risks = []
        if stats.avg_degree < 4:
            risks.append(f"🔴 **Low direct connectivity** (avg {stats.avg_degree:.1f} links per person) — people have few direct ties, which means collaboration requires active coordination.")
        if stats.n_components > 1:
            risks.append(f"🟠 **{stats.n_components} isolated groups exist** — these clusters cannot reach each other, even indirectly. This may indicate blind spots, missing sectors, or disconnected communities.")
        if largest_share < 80:
            risks.append(f"🟠 **Fragmented network** — only {largest_share:.0f}% of people are part of the main connected group, limiting how far information can travel.")
        if power_concentrated:
            risks.append("🔴 **Power is concentrated** — influence sits with a few key actors, creating potential bottlenecks and vulnerabilities.")
        
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Positive Factors**")
            if positives:
                for p in positives:
                    st.markdown(p)
            else:
                st.markdown("*No strong positive factors identified*")
        
        with col2:
            st.markdown("**🔴 Risk Factors**")
            if risks:
                for r in risks:
                    st.markdown(r)
            else:
                st.markdown("*No significant risks identified*")
        
        # Glossary
        with st.expander("📖 What do these terms mean?"):
            st.markdown("""
**Direct connections (degree):**  
How many people someone is directly connected to.

**Connected group (component):**  
A group of people who can reach each other through the network, even if indirectly.  
If the network has multiple components, those groups never interact.

**Network unity (cohesion):**  
The share of all actors who are part of the same overall connected system.

**Power concentration:**  
Whether influence (measured through network metrics) sits with a few actors or is spread out.
            """)


# ============================================================================
# SECTOR ANALYSIS (Diversity, Imbalance, Narrative)
# ============================================================================

@dataclass
class SectorAnalysis:
    """Container for sector distribution analysis results."""
    df: pd.DataFrame
    dominant_sectors: List[str]
    underrepresented_sectors: List[str]
    diversity_score: float
    diversity_label: str
    summary_text: str


def analyze_sectors(sector_counts: Dict[str, int], total_nodes: int) -> SectorAnalysis:
    """
    Analyze sector distribution for imbalance and diversity.
    
    Args:
        sector_counts: {"Philanthropy": 12, "Academia": 9, ...}
        total_nodes: total # of nodes in the network
    
    Returns:
        SectorAnalysis with dataframe, dominant/underrepresented lists,
        diversity score, and narrative summary.
    """
    # Build dataframe of counts
    data = []
    total_classified = sum(sector_counts.values()) or 1  # avoid div by zero

    for sector, count in sector_counts.items():
        pct_classified = (count / total_classified) * 100
        pct_network = (count / max(total_nodes, 1)) * 100
        data.append({
            "sector": sector,
            "count": count,
            "pct_classified": pct_classified,
            "pct_network": pct_network,
        })

    df = pd.DataFrame(data).sort_values("pct_classified", ascending=False)

    # Dominant = any sector with >= 40% of all classified actors
    dominant_sectors = df[df["pct_classified"] >= 40.0]["sector"].tolist()

    # Underrepresented = sectors with <= 10% of classified actors
    underrepresented_sectors = df[df["pct_classified"] <= 10.0]["sector"].tolist()

    # Diversity: simple Shannon index normalized 0–1
    p = df["pct_classified"].values / 100.0
    p = p[p > 0]
    if len(p) > 0:
        H = -np.sum(p * np.log(p))
        H_max = np.log(len(p))
        diversity = float(H / H_max) if H_max > 0 else 0.0
    else:
        diversity = 0.0

    if diversity >= 0.75:
        diversity_label = "Broad cross-sector mix"
    elif diversity >= 0.45:
        diversity_label = "Moderate mix with some skew"
    else:
        diversity_label = "Highly concentrated"

    # Narrative summary
    if len(df) > 0:
        top_row = df.iloc[0]
        top_sector = top_row["sector"]
        top_pct = top_row["pct_classified"]

        summary_parts = [
            f"{len(df)} sectors identified among classified actors.",
            f"The largest share comes from **{top_sector}** ({top_pct:.1f}% of classified actors).",
        ]

        if dominant_sectors:
            ds = ", ".join(dominant_sectors)
            summary_parts.append(f"These sector(s) dominate the network: **{ds}**.")

        if underrepresented_sectors and len(underrepresented_sectors) != len(df):
            us = ", ".join(underrepresented_sectors)
            summary_parts.append(f"These sectors are underrepresented: **{us}** (each ≤10% of classified actors).")

        summary_parts.append(f"Overall sector diversity: **{diversity_label}**.")
        summary_text = " ".join(summary_parts)
    else:
        summary_text = "No sector data available."

    return SectorAnalysis(
        df=df,
        dominant_sectors=dominant_sectors,
        underrepresented_sectors=underrepresented_sectors,
        diversity_score=diversity,
        diversity_label=diversity_label,
        summary_text=summary_text,
    )


def render_sector_analysis(sectors: Dict[str, int], total_nodes: int, seen_profiles: Dict = None):
    """Render the enhanced sector distribution with Plotly bar chart and narrative."""
    
    if not sectors:
        st.info("No sector classification available")
        return
    
    # Analyze sectors
    analysis = analyze_sectors(sectors, total_nodes)
    df = analysis.df
    
    # Header metric
    st.metric("Sectors Identified", len(sectors))
    
    # Determine bar colors based on dominance
    def get_bar_color(pct):
        if pct >= 40:
            return "#F97316"  # Orange - dominant
        elif pct <= 10:
            return "#9CA3AF"  # Gray - underrepresented
        else:
            return "#10B981"  # Green - healthy
    
    df["color"] = df["pct_classified"].apply(get_bar_color)
    
    # Sort for display (largest at top for horizontal bars)
    df_sorted = df.sort_values("pct_classified", ascending=True)
    
    # Create Plotly horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted["sector"],
        x=df_sorted["pct_classified"],
        orientation='h',
        marker_color=df_sorted["color"],
        text=[f"{p:.1f}%" for p in df_sorted["pct_classified"]],
        textposition='auto',
        textfont=dict(color='white', size=12),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of network<br>%{customdata} people<extra></extra>",
        customdata=df_sorted["count"]
    ))
    
    fig.update_layout(
        height=max(200, len(df) * 40),  # Dynamic height based on sector count
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="% of Classified Actors",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, max(df["pct_classified"]) * 1.15]  # Add padding for labels
        ),
        yaxis=dict(
            showgrid=False,
        )
    )
    
    st.markdown("**Share of classified actors by sector:**")
    st.plotly_chart(fig, use_container_width=True)
    
    # Color legend
    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 12px; margin-bottom: 10px;">
        <span><span style="color: #F97316;">●</span> Dominant (≥40%)</span>
        <span><span style="color: #10B981;">●</span> Healthy (10-40%)</span>
        <span><span style="color: #9CA3AF;">●</span> Underrepresented (≤10%)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary narrative
    st.markdown("---")
    st.markdown(analysis.summary_text)
    
    # Diversity indicator
    if analysis.diversity_score >= 0.75:
        st.success(f"🟢 **Sector Diversity: {analysis.diversity_label}** — The network includes a healthy mix of perspectives from different sectors.")
    elif analysis.diversity_score >= 0.45:
        st.warning(f"🟡 **Sector Diversity: {analysis.diversity_label}** — Some sectors are more represented than others, but there's reasonable variety.")
    else:
        st.error(f"🔴 **Sector Diversity: {analysis.diversity_label}** — The network is dominated by one or two sectors, which may limit perspective diversity.")
    
    # Callout boxes for imbalance
    if analysis.dominant_sectors:
        st.warning(
            "⚠️ **Sector Dominance**\n\n"
            f"The network is heavily shaped by **{', '.join(analysis.dominant_sectors)}**. "
            "Voices from other sectors may have less visibility or influence."
        )
    
    if analysis.underrepresented_sectors and len(analysis.underrepresented_sectors) != len(df):
        underrep_list = "\n".join([f"- {s}" for s in analysis.underrepresented_sectors])
        st.info(
            "💡 **Underrepresented Sectors**\n\n"
            "These sectors appear in small numbers and may represent "
            "critical but less-heard perspectives:\n\n"
            f"{underrep_list}"
        )
    
    # Sector detail expanders - show who's in each sector
    if seen_profiles:
        st.markdown("---")
        st.markdown("**👥 Explore People by Sector:**")
        
        # Group profiles by sector
        profiles_by_sector = {}
        for node_id, profile in seen_profiles.items():
            sector = profile.get('sector', 'Unknown')
            if sector not in profiles_by_sector:
                profiles_by_sector[sector] = []
            profiles_by_sector[sector].append(profile)
        
        # Create expanders for each sector (sorted by count)
        for _, row in df.iterrows():
            sector = row["sector"]
            count = int(row["count"])
            pct = row["pct_classified"]
            
            # Get color indicator
            if pct >= 40:
                indicator = "🟠"
            elif pct <= 10:
                indicator = "⚪"
            else:
                indicator = "🟢"
            
            profiles = profiles_by_sector.get(sector, [])
            
            with st.expander(f"{indicator} **{sector}** — {count} people ({pct:.1f}%)"):
                if profiles:
                    # Show up to 10 sample profiles
                    sample_profiles = profiles[:10]
                    
                    for p in sample_profiles:
                        name = p.get('name', 'Unknown')
                        org = p.get('organization', '')
                        headline = p.get('headline', '')
                        
                        org_text = f" • {org}" if org else ""
                        headline_text = f"\n  _{headline}_" if headline and len(headline) < 100 else ""
                        
                        st.markdown(f"**{name}**{org_text}{headline_text}")
                    
                    if len(profiles) > 10:
                        st.caption(f"...and {len(profiles) - 10} more people in this sector")
                else:
                    st.caption("No profile details available")
        
        # Special handling for "Unknown" or "Other" category
        unknown_profiles = profiles_by_sector.get('Unknown', []) + profiles_by_sector.get('Other', [])
        if unknown_profiles and 'Unknown' not in [row["sector"] for _, row in df.iterrows()]:
            with st.expander(f"❓ **Unclassified** — {len(unknown_profiles)} people"):
                st.caption("These profiles couldn't be automatically classified into a sector:")
                for p in unknown_profiles[:10]:
                    name = p.get('name', 'Unknown')
                    org = p.get('organization', '')
                    headline = p.get('headline', '')
                    
                    org_text = f" • {org}" if org else ""
                    st.markdown(f"**{name}**{org_text}")
                    if headline:
                        st.caption(f"  {headline[:80]}...")
                
                if len(unknown_profiles) > 10:
                    st.caption(f"...and {len(unknown_profiles) - 10} more unclassified profiles")


# ============================================================================
# BROKERAGE ROLES (Gould & Fernandez Classification)
# ============================================================================

def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """
    Run Louvain community detection.
    Returns {node_id: community_id}
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        return partition
    except ImportError:
        # Fallback: use NetworkX's greedy modularity if python-louvain not available
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition


def compute_brokerage_roles(G: nx.Graph, communities: Dict[str, int]) -> Dict[str, str]:
    """
    Implements simplified Gould & Fernandez brokerage classification.
    Returns {node_id: brokerage_role}
    
    Roles:
    - coordinator: connects people within their own group
    - gatekeeper: controls access into their group  
    - representative: connects their group to others
    - liaison: bridges two groups neither of which is their own
    - consultant: connects to many others across multiple sectors
    - peripheral: low brokerage, purely internal ties
    """
    roles = {}

    for node in G.nodes():
        if node not in communities:
            roles[node] = "peripheral"
            continue
            
        group = communities[node]
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            roles[node] = "peripheral"
            continue

        neighbor_groups = {communities.get(n, -1) for n in neighbors}
        neighbor_groups.discard(-1)  # Remove unknown groups

        # Internal coordinator: neighbors mostly in same group
        same_group_ties = [n for n in neighbors if communities.get(n) == group]
        other_group_ties = [n for n in neighbors if communities.get(n) != group and n in communities]

        if len(neighbor_groups) <= 1 and group in neighbor_groups:
            roles[node] = "coordinator"
            continue

        # Liaison: connects two groups neither of which is their own
        if len(neighbor_groups) >= 2 and group not in neighbor_groups:
            roles[node] = "liaison"
            continue

        # Representative: many outgoing ties to other groups
        if len(other_group_ties) >= len(neighbors) * 0.5:
            roles[node] = "representative"
            continue

        # Gatekeeper: some ties to other groups but primarily internal
        if len(other_group_ties) > 0 and len(other_group_ties) <= len(neighbors) * 0.35:
            roles[node] = "gatekeeper"
            continue

        # Consultant (multi-group advisor)
        if len(neighbor_groups) > 2:
            roles[node] = "consultant"
            continue

        # Default
        roles[node] = "peripheral"

    return roles


BROKER_BADGES: Dict[str, Dict[str, str]] = {
    "coordinator": {"emoji": "🧩", "label": "Internal coordinator", "color": "#3B82F6"},
    "gatekeeper": {"emoji": "🚪", "label": "Gatekeeper", "color": "#F97316"},
    "representative": {"emoji": "🔗", "label": "Representative", "color": "#10B981"},
    "liaison": {"emoji": "🌉", "label": "Cross-group liaison", "color": "#D97706"},
    "consultant": {"emoji": "🧠", "label": "Multi-group advisor", "color": "#6366F1"},
    "peripheral": {"emoji": "⚪", "label": "Peripheral", "color": "#9CA3AF"},
}

BROKER_TOOLTIPS: Dict[str, str] = {
    "coordinator": "Connects people within their own group, keeping it internally cohesive.",
    "gatekeeper": "Controls access into their group — a key node for sector handoffs.",
    "representative": "Brings their group's voice into other groups — outbound connector.",
    "liaison": "The rare 'true bridge' — connects groups they don't belong to. Extremely valuable.",
    "consultant": "Sits across multiple groups with non-hierarchical ties — multi-group advisor.",
    "peripheral": "Low influence, low connectivity — sits at the edge of the system.",
}


def render_broker_badge(role: str, small: bool = True) -> str:
    """Return HTML snippet for a brokerage role badge."""
    cfg = BROKER_BADGES.get(role, BROKER_BADGES["peripheral"])
    pad = "2px 6px" if small else "4px 8px"
    font_size = "11px" if small else "13px"
    return (
        f"<span style='"
        f"background-color:{cfg['color']}20;"
        f"border:1px solid {cfg['color']};"
        f"border-radius:999px;"
        f"padding:{pad};"
        f"font-size:{font_size};"
        f"color:#111827;"
        f"margin-left:4px;"
        f"white-space:nowrap;"
        f"'>"
        f"{cfg['emoji']} {cfg['label']}"
        f"</span>"
    )


def _bucket_from_percentile(p: float) -> str:
    """
    Simple human label for a 0–1 percentile value.
    Expects p in [0, 1] (e.g., 0.87 = 87th percentile).
    """
    if p is None:
        return "unknown"
    if p >= 0.95:
        return "very high"
    if p >= 0.75:
        return "high"
    if p >= 0.40:
        return "moderate"
    return "low"


def describe_node_with_recommendation(
    name: str,
    organization: Optional[str],
    role: Optional[str],
    degree_pct: Optional[float] = None,
    betweenness_pct: Optional[float] = None,
    eigenvector_pct: Optional[float] = None,
    closeness_pct: Optional[float] = None,
    sector: Optional[str] = None,
    is_dominant_sector: bool = False,
    is_underrepresented_sector: bool = False,
    include_recommendation: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Returns (blurb, recommendation) for a single node.
    
    All *_pct arguments are percentiles in [0, 1] for that metric
    (e.g., 0.90 = top 10% of the network).
    is_dominant_sector / is_underrepresented_sector derived from SectorAnalysis.
    """
    org_str = f" at {organization}" if organization else ""
    sector_str = f" in the {sector} space" if sector else ""

    deg_bucket = _bucket_from_percentile(degree_pct)
    btw_bucket = _bucket_from_percentile(betweenness_pct)
    eig_bucket = _bucket_from_percentile(eigenvector_pct)
    clo_bucket = _bucket_from_percentile(closeness_pct)

    # --- Role-specific framing ---
    role_key = (role or "").lower()

    if role_key == "liaison":
        role_sentence = (
            f"{name}{org_str}{sector_str} acts as a cross-group liaison, "
            f"helping information and relationships move between otherwise "
            f"separate parts of the network."
        )
    elif role_key == "gatekeeper":
        role_sentence = (
            f"{name}{org_str}{sector_str} functions as a gatekeeper, controlling "
            f"key entry points into their part of the network."
        )
    elif role_key == "representative":
        role_sentence = (
            f"{name}{org_str}{sector_str} often represents their group to others, "
            f"serving as a visible point of contact between communities."
        )
    elif role_key == "coordinator":
        role_sentence = (
            f"{name}{org_str}{sector_str} primarily coordinates relationships "
            f"within their own group, helping keep that cluster connected."
        )
    elif role_key == "consultant":
        role_sentence = (
            f"{name}{org_str}{sector_str} spans several groups without belonging "
            f"strongly to any single one, often acting as an informal advisor or "
            f"connector across contexts."
        )
    elif role_key == "peripheral":
        role_sentence = (
            f"{name}{org_str}{sector_str} sits closer to the edge of the network, "
            f"connecting in through a smaller number of ties."
        )
    else:
        role_sentence = (
            f"{name}{org_str}{sector_str} holds a meaningful position in the "
            f"network based on their pattern of connections."
        )

    # --- Centrality context phrases ---
    context_bits = []

    if deg_bucket in ("very high", "high"):
        context_bits.append("is connected to many people in the network")
    elif deg_bucket == "moderate":
        context_bits.append("has a solid set of direct relationships")
    elif deg_bucket == "low":
        context_bits.append("has a focused set of direct relationships")

    if btw_bucket in ("very high", "high"):
        context_bits.append("often links people who would not otherwise connect")

    if eig_bucket in ("very high", "high"):
        context_bits.append("is closely tied to other visible or influential actors")

    if clo_bucket in ("very high", "high") and btw_bucket not in ("very high", "high"):
        context_bits.append("can reach much of the network in only a few steps")

    # --- Assemble the blurb ---
    blurb = role_sentence
    if context_bits:
        blurb += " " + f"{name} " + "; ".join(context_bits) + "."

    # --- Generate recommendation ---
    if not include_recommendation:
        return blurb, None

    rec_bits = []

    # High centrality + liaison → protect and support
    if role_key == "liaison" and btw_bucket in ("very high", "high"):
        rec_bits.append(
            "Protect this person from overload and involve them early when you "
            "need to connect groups or sectors that rarely work together."
        )

    # Gatekeeper with high betweenness → backup & transparency
    if role_key == "gatekeeper" and btw_bucket in ("very high", "high"):
        rec_bits.append(
            "Make sure there are backup pathways into their group and that others "
            "understand how to engage without everything running through one person."
        )

    # Coordinator with very high degree → internal organizer
    if role_key == "coordinator" and deg_bucket in ("very high", "high"):
        rec_bits.append(
            "Leverage them as an internal organizer for their group and include "
            "them when you need to align priorities or share updates within that cluster."
        )

    # Representative → outward voice
    if role_key == "representative" and btw_bucket in ("very high", "high", "moderate"):
        rec_bits.append(
            "Include them when presenting to external stakeholders or forming "
            "cross-sector partnerships."
        )

    # Consultant → multi-group advisor
    if role_key == "consultant" and eig_bucket in ("very high", "high"):
        rec_bits.append(
            "Tap them for advice on cross-cutting issues; they see patterns across groups."
        )

    # Representative or liaison from underrepresented sector
    if role_key in ("representative", "liaison") and is_underrepresented_sector:
        rec_bits.append(
            "Treat them as a key voice for an underrepresented sector and invite "
            "them into design, strategy, or sensemaking spaces."
        )

    # Very high degree or betweenness in any role → risk of overload
    if (deg_bucket == "very high" or btw_bucket == "very high") and role_key not in ("peripheral",):
        rec_bits.append(
            "Check in on their workload and relational burden; consider sharing "
            "responsibilities with 1–2 additional connectors."
        )

    # Peripheral in dominant sector → quiet entry point
    if role_key == "peripheral" and is_dominant_sector:
        rec_bits.append(
            "If you need fresh perspectives from the dominant sector, this person "
            "may be a quieter entry point without going through the usual hubs."
        )

    # Peripheral with moderate/high closeness → emerging connector
    if role_key == "peripheral" and clo_bucket in ("very high", "high"):
        rec_bits.append(
            "Despite being peripheral, they can reach others quickly — consider "
            "developing them as an emerging connector."
        )

    # If nothing matched, offer a generic but useful prompt
    if not rec_bits:
        rec_bits.append(
            "Keep them informed and connected to relevant conversations; their "
            "position may become more important as initiatives evolve."
        )

    recommendation = " ".join(rec_bits)

    return blurb, recommendation


# Backward-compatible wrapper
def describe_node_narrative(
    name: str,
    organization: Optional[str],
    role: Optional[str],
    degree_pct: Optional[float] = None,
    betweenness_pct: Optional[float] = None,
    eigenvector_pct: Optional[float] = None,
    closeness_pct: Optional[float] = None,
    sector: Optional[str] = None,
) -> str:
    """Return just the blurb (for backward compatibility)."""
    blurb, _ = describe_node_with_recommendation(
        name=name,
        organization=organization,
        role=role,
        degree_pct=degree_pct,
        betweenness_pct=betweenness_pct,
        eigenvector_pct=eigenvector_pct,
        closeness_pct=closeness_pct,
        sector=sector,
        include_recommendation=False,
    )
    return blurb


def compute_percentiles(values: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw metric values to percentiles (0-1).
    Returns {node_id: percentile}
    """
    if not values:
        return {}
    
    sorted_values = sorted(values.values())
    n = len(sorted_values)
    
    percentiles = {}
    for node_id, value in values.items():
        # Find position in sorted list
        rank = sorted_values.index(value)
        percentiles[node_id] = rank / max(n - 1, 1)
    
    return percentiles


# Legacy function for backward compatibility
def describe_broker_role(name: str, org: str, role: str, betweenness_level: str) -> str:
    """Generate a narrative description of a broker's role (simplified version)."""
    # Map level to approximate percentile
    level_to_pct = {"low": 0.2, "medium": 0.5, "high": 0.85, "extreme": 0.97}
    btw_pct = level_to_pct.get(betweenness_level, 0.5)
    
    return describe_node_narrative(
        name=name,
        organization=org,
        role=role,
        betweenness_pct=btw_pct
    )


# ============================================================================
# RECOMMENDATIONS ENGINE (Rule-based Next Steps)
# ============================================================================

def generate_recommendations(
    stats: NetworkStats,
    sector_analysis: SectorAnalysis,
    degree_values: Sequence[float],
    betweenness_values: Sequence[float],
    health_score: int,
    health_label: str,
    brokerage_roles: Dict[str, str] = None,
    critical_brokers: List[str] = None,
) -> str:
    """
    Generate rule-based recommendations based on network structure.
    Returns markdown string with actionable next steps.
    """
    rec_sections: List[str] = []
    
    # Initialize if None
    if brokerage_roles is None:
        brokerage_roles = {}
    if critical_brokers is None:
        critical_brokers = []

    # ----------------------------------------------------------------------
    # 1. Overall framing based on health score
    # ----------------------------------------------------------------------
    if health_score >= 70:
        intro = (
            "The network is structurally healthy. The goal now is to **protect "
            "what works** and **deepen strategic relationships** rather than fix "
            "basic connectivity."
        )
    elif health_score >= 40:
        intro = (
            "The network shows **mixed signals**. There is a reasonably solid "
            "core, but several structural risks limit how quickly ideas and "
            "collaboration can spread."
        )
    else:
        intro = (
            "The network appears **fragile**. Basic connectivity and inclusion "
            "need attention before deeper collaboration will be reliable."
        )

    rec_sections.append(f"### 🧭 How to Read This\n\n{intro}\n")

    # ----------------------------------------------------------------------
    # 2. Connectivity & fragmentation
    # ----------------------------------------------------------------------
    connectivity_recs: List[str] = []

    if stats.avg_degree < 3:
        connectivity_recs.append(
            "**Increase direct connections:** Most people have few direct ties. "
            "Host small, mixed-group sessions or introductions so key actors meet "
            "each other directly instead of relying on intermediaries."
        )
    elif stats.avg_degree > 8:
        connectivity_recs.append(
            "**Simplify crowded ties:** Many actors have lots of direct links. "
            "Consider clarifying who is responsible for what to avoid overload."
        )

    if stats.n_components > 1:
        connectivity_recs.append(
            f"**Bridge isolated groups:** The network splits into **{stats.n_components} "
            "disconnected clusters**. Identify 1–2 people in each cluster who can "
            "join cross-group conversations or shared projects."
        )

    if connectivity_recs:
        rec_sections.append("### 🔗 Strengthen Basic Connections\n\n" +
                            "\n\n".join([f"- {r}" for r in connectivity_recs]) + "\n")

    # ----------------------------------------------------------------------
    # 3. Sector balance & inclusion
    # ----------------------------------------------------------------------
    sector_recs: List[str] = []

    if sector_analysis and sector_analysis.dominant_sectors:
        dom = ", ".join(sector_analysis.dominant_sectors)
        sector_recs.append(
            f"**Rebalance who is at the table:** The network is strongly shaped "
            f"by **{dom}**. Make sure decisions and narratives do not rely only "
            "on these perspectives."
        )

    if sector_analysis and sector_analysis.underrepresented_sectors and \
       len(sector_analysis.underrepresented_sectors) != len(sector_analysis.df):
        under = ", ".join(sector_analysis.underrepresented_sectors)
        sector_recs.append(
            f"**Invite missing voices:** These sectors are underrepresented "
            f"(**{under}**). Intentionally invite them into sensemaking workshops, "
            "advisory groups, or early-stage design conversations."
        )

    if sector_analysis and sector_analysis.diversity_label == "Highly concentrated":
        sector_recs.append(
            "**Monitor concentration of influence:** Because most actors come "
            "from a narrow set of sectors, keep an eye on how that shapes which "
            "problems are seen as important."
        )

    if sector_recs:
        rec_sections.append("### 🎯 Balance Sectors and Perspectives\n\n" +
                            "\n\n".join([f"- {r}" for r in sector_recs]) + "\n")

    # ----------------------------------------------------------------------
    # 4. Power concentration & brokers
    # ----------------------------------------------------------------------
    power_recs: List[str] = []

    deg_cent = centralization_index(degree_values)
    btw_cent = centralization_index(betweenness_values)
    avg_cent = 0.5 * (deg_cent + btw_cent)

    if critical_brokers:
        power_recs.append(
            "**Reduce dependence on critical brokers:** A small number of "
            "people sit on many key paths. If they disengage or change roles, "
            "collaboration could stall. Identify 2–3 additional connectors who "
            "can share this load."
        )

    if avg_cent > 0.5:
        power_recs.append(
            "**Distribute influence more widely:** Consider rotating facilitation "
            "roles, broadening who convenes meetings, and creating multiple entry "
            "points into the network."
        )
    elif avg_cent < 0.2 and stats.n_nodes > 20:
        power_recs.append(
            "**Clarify leadership signals:** Power is widely spread. This is "
            "healthy for resilience, but can make decision-making fuzzy. Make "
            "sure there are clear ways to align on priorities."
        )

    # Role-specific checks
    if brokerage_roles:
        roles_list = list(brokerage_roles.values())
        n_liaison = roles_list.count("liaison")
        n_gatekeeper = roles_list.count("gatekeeper")
        n_rep = roles_list.count("representative")

        if n_liaison == 0 and stats.n_components > 1:
            power_recs.append(
                "**Cultivate cross-group liaisons:** No clear liaisons were "
                "detected. Support boundary-spanning people who can translate "
                "between groups and sectors."
            )

        if n_gatekeeper > 3:
            power_recs.append(
                "**Support gatekeepers, but avoid bottlenecks:** People who "
                "control access into key groups need backup and transparency so "
                "others are not blocked when they are unavailable."
            )

        if n_rep == 0 and sector_analysis and sector_analysis.dominant_sectors:
            power_recs.append(
                "**Develop representatives for underrepresented sectors:** "
                "Encourage individuals who can speak for missing sectors in "
                "multi-stakeholder forums."
            )

    if power_recs:
        rec_sections.append("### 🧩 Work with Brokers Thoughtfully\n\n" +
                            "\n\n".join([f"- {r}" for r in power_recs]) + "\n")

    # ----------------------------------------------------------------------
    # 5. Periphery & follow-through
    # ----------------------------------------------------------------------
    periphery_recs: List[str] = []

    if stats.largest_component_size / max(stats.n_nodes, 1) > 0.9 and stats.avg_degree < 3:
        periphery_recs.append(
            "**Stabilize the periphery:** Many actors are technically connected "
            "but only via a small number of ties. Provide easy ways for them to "
            "participate (light-touch updates, open calls, or periodic check-ins)."
        )

    if periphery_recs:
        rec_sections.append("### 🌱 Strengthen Participation at the Edges\n\n" +
                            "\n\n".join([f"- {r}" for r in periphery_recs]) + "\n")

    # ----------------------------------------------------------------------
    # 6. Fallback if nothing was triggered
    # ----------------------------------------------------------------------
    if len(rec_sections) == 1:  # Only the intro
        rec_sections.append(
            "### ✨ No Structural Red Flags\n\n"
            "The network appears structurally sound based on the available data. "
            "Focus on **clarifying shared purpose, stories, and norms** rather than "
            "changing the structure itself."
        )

    return "\n".join(rec_sections)


def render_recommendations(
    stats: NetworkStats,
    sector_analysis: SectorAnalysis,
    degree_values: Sequence[float],
    betweenness_values: Sequence[float],
    health_score: int,
    health_label: str,
    brokerage_roles: Dict[str, str] = None,
    critical_brokers: List[str] = None,
):
    """Render recommendations section in Streamlit."""
    st.subheader("🚀 Next-Step Recommendations")
    
    recommendations_md = generate_recommendations(
        stats=stats,
        sector_analysis=sector_analysis,
        degree_values=degree_values,
        betweenness_values=betweenness_values,
        health_score=health_score,
        health_label=health_label,
        brokerage_roles=brokerage_roles,
        critical_brokers=critical_brokers,
    )
    
    st.markdown(recommendations_md)


# ============================================================================
# CONFIGURATION
# ============================================================================

API_DELAY = 3.0  # Seconds between API calls (20 requests/min for $49/mo EnrichLayer plan)
PER_MIN_LIMIT = 20  # Tuned for $49/mo plan (can be changed for other tiers)


# ============================================================================
# RATE LIMITER CLASS
# ============================================================================

class RateLimiter:
    """
    Rate limiter that enforces a per-minute request limit.
    Uses a sliding window approach with a safety buffer.
    """
    def __init__(self, per_min_limit: int, buffer: float = 0.8):
        """
        Args:
            per_min_limit: documented limit (e.g., 20 requests/min)
            buffer: safety factor (0.8 → aim for 16/min so we never hit the hard cap)
        """
        self.per_min_limit = per_min_limit
        self.allowed_per_min = max(1, int(per_min_limit * buffer))
        self.window_start = time.time()
        self.calls_in_window = 0

    def wait_for_slot(self):
        """Wait until we have a safe slot to make a request."""
        now = time.time()
        elapsed = now - self.window_start

        # New minute → reset window
        if elapsed >= 60:
            self.window_start = now
            self.calls_in_window = 0
            return

        # If we've hit our safe quota, sleep until the minute resets
        if self.calls_in_window >= self.allowed_per_min:
            sleep_for = 60 - elapsed
            time.sleep(sleep_for)
            self.window_start = time.time()
            self.calls_in_window = 0

    def record_call(self):
        """Record that we made a call."""
        self.calls_in_window += 1
    
    def get_status(self) -> str:
        """Get current rate limiter status for display."""
        return f"{self.calls_in_window}/{self.allowed_per_min} calls this minute"
DEFAULT_MOCK_MODE = True  # Default to mock mode for safe testing

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_url_stub(profile_url: str) -> str:
    """
    Extract a temporary ID from LinkedIn URL.
    Example: https://www.linkedin.com/in/john-doe → john-doe
    """
    # Remove trailing slashes and query parameters
    clean_url = profile_url.rstrip('/').split('?')[0]
    
    # Extract the username part
    match = re.search(r'/in/([^/]+)', clean_url)
    if match:
        return match.group(1)
    
    # Fallback: use last part of URL
    return clean_url.split('/')[-1]


def extract_organization(occupation: str = '', experiences: List = None) -> str:
    """
    Extract organization name from occupation string or experiences.
    
    Args:
        occupation: String like "Chief Executive Officer at Toniic"
        experiences: List of experience dicts with 'company' field
    
    Returns:
        Organization name or empty string
    """
    # Try occupation field first (most reliable for current role)
    if occupation and ' at ' in occupation:
        # "Chief Executive Officer at Toniic" → "Toniic"
        org = occupation.split(' at ', 1)[1].strip()
        # Clean up common suffixes
        org = org.replace('|', '').strip()
        return org
    
    # Fallback to most recent experience
    if experiences and len(experiences) > 0:
        # Get most recent (first in list, usually current)
        recent = experiences[0]
        if 'company' in recent and recent['company']:
            return recent['company'].strip()
    
    return ''


def infer_sector(organization: str, headline: str = '') -> str:
    """
    Infer sector/industry from organization name and headline.
    Simple keyword-based classification.
    """
    combined = f"{organization} {headline}".lower()
    
    # Keyword mappings
    if any(word in combined for word in ['foundation', 'philanthropy', 'donor', 'giving']):
        return 'Philanthropy'
    elif any(word in combined for word in ['ngo', 'nonprofit', 'charity', 'humanitarian']):
        return 'Nonprofit'
    elif any(word in combined for word in ['government', 'ministry', 'federal', 'state', 'public sector']):
        return 'Government'
    elif any(word in combined for word in ['university', 'college', 'academic', 'research', 'professor']):
        return 'Academia'
    elif any(word in combined for word in ['peace', 'conflict', 'democracy', 'justice', 'rights']):
        return 'Peacebuilding/Democracy'
    elif any(word in combined for word in ['social impact', 'social change', 'community development']):
        return 'Social Impact'
    elif any(word in combined for word in ['consulting', 'consultant', 'advisory']):
        return 'Consulting'
    elif any(word in combined for word in ['finance', 'investment', 'capital', 'fund', 'bank']):
        return 'Finance'
    elif any(word in combined for word in ['tech', 'software', 'digital', 'platform']):
        return 'Technology'
    elif any(word in combined for word in ['corp', 'inc', 'llc', 'ltd', 'company']):
        return 'Corporate'
    else:
        return 'Other'


def canonical_id_from_url(profile_url: str) -> str:
    """Generate temporary canonical ID from URL before API enrichment."""
    return extract_url_stub(profile_url)


def update_canonical_ids(seen_profiles: Dict, edges: List, old_id: str, new_id: str) -> None:
    """
    Update all references to old_id with new_id after API enrichment.
    """
    # Update the node entry
    if old_id in seen_profiles:
        node = seen_profiles[old_id]
        node['id'] = new_id
        seen_profiles[new_id] = node
        if old_id != new_id:
            del seen_profiles[old_id]
    
    # Update all edge references
    for edge in edges:
        if edge['source_id'] == old_id:
            edge['source_id'] = new_id
        if edge['target_id'] == old_id:
            edge['target_id'] = new_id


def validate_graph(seen_profiles: Dict, edges: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Validate that all edge endpoints exist in nodes.
    Returns (orphan_ids, valid_edges)
    """
    node_ids = set(seen_profiles.keys())
    orphan_ids = set()
    valid_edges = []

    for edge in edges:
        if edge['source_id'] in node_ids and edge['target_id'] in node_ids:
            valid_edges.append(edge)
        else:
            if edge['source_id'] not in node_ids:
                orphan_ids.add(edge['source_id'])
            if edge['target_id'] not in node_ids:
                orphan_ids.add(edge['target_id'])

    return sorted(orphan_ids), valid_edges


def test_network_connectivity() -> Tuple[bool, str]:
    """
    Test if enrichlayer.com is reachable.
    Returns (success, message)
    """
    try:
        # Test DNS resolution for the correct domain
        ip = socket.gethostbyname("enrichlayer.com")
        
        # Test HTTPS connection to the correct endpoint
        response = requests.get("https://enrichlayer.com/api/v2/profile", timeout=5)
        return True, f"✅ Network OK (resolved to {ip})"
    
    except socket.gaierror:
        return False, (
            "❌ DNS Resolution Failed\n\n"
            "Cannot reach enrichlayer.com. This indicates a network/firewall restriction.\n\n"
            "**Solutions:**\n"
            "1. Check your internet connection\n"
            "2. Try a different network\n"
            "3. Use Mock Mode for testing"
        )
    
    except requests.exceptions.ConnectionError:
        return False, (
            "❌ Connection Failed\n\n"
            "DNS works but cannot establish HTTPS connection.\n\n"
            "**Solutions:**\n"
            "1. Check EnrichLayer service status\n"
            "2. Try a different network\n"
            "3. Use Mock Mode for testing"
        )
    
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


# ============================================================================
# ENRICHLAYER API CLIENT
# ============================================================================

def call_enrichlayer_api(api_token: str, profile_url: str, mock_mode: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call EnrichLayer person profile endpoint with retry logic.
    
    Returns:
        (response_dict, error_message) tuple
        - If successful: (response, None)
        - If failed: (None, error_message)
    
    Implements exponential backoff for rate limit errors (429).
    """
    if mock_mode:
        # Return mock data for testing
        time.sleep(0.1)  # Small delay to simulate API call
        return get_mock_response(profile_url), None
    
    # Correct EnrichLayer API endpoint (v2)
    endpoint = "https://enrichlayer.com/api/v2/profile"
    headers = {
        "Authorization": f"Bearer {api_token}",
    }
    params = {
        "url": profile_url,
        "use_cache": "if-present",  # Use cache if available
        "live_fetch": "if-needed",   # Only fetch live if needed
    }
    
    for attempt in range(max_retries):
        try:
            # Use GET request with params (not POST with json)
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 401:
                return None, "Invalid API token"
            elif response.status_code == 403:
                # Out of credits - don't retry
                return None, "Out of credits (check your EnrichLayer balance)"
            elif response.status_code == 429:
                # Rate limit - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                    time.sleep(wait_time)
                    continue
                else:
                    return None, f"Rate limit exceeded (tried {max_retries} times)"
            elif response.status_code == 503:
                # Enrichment failed - can retry
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    return None, "Enrichment failed after retries"
            else:
                return None, f"API error {response.status_code}: {response.text}"
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"Request timed out (tried {max_retries} times)"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"Network error: {str(e)}"
    
    return None, "Failed after maximum retries"


def get_mock_response(profile_url: str) -> Dict:
    """
    Generate comprehensive mock API response for stress testing.
    
    Creates deterministic but varied fake profiles with:
    - 25-40 "people_also_viewed" connections per profile
    - Realistic names, titles, organizations, sectors
    - Enough data to stress test 1000 node / 1000 edge limits
    
    Uses hash of profile URL for deterministic randomness.
    """
    import hashlib
    
    # Deterministic seed based on profile URL
    url_hash = int(hashlib.md5(profile_url.encode()).hexdigest(), 16)
    
    # Mock data pools
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
        "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
        "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker"
    ]
    
    titles = [
        "CEO", "Founder", "Director", "VP", "Manager", "Consultant", "Partner",
        "Executive Director", "Chief Strategy Officer", "Program Director",
        "Senior Advisor", "Managing Director", "Principal", "Fellow", "Board Member",
        "Chief Impact Officer", "Head of Partnerships", "Director of Development",
        "Senior Program Officer", "Policy Director", "Research Director"
    ]
    
    organizations = [
        "World Resources Institute", "The Nature Conservancy", "WWF", "IUCN",
        "Conservation International", "Environmental Defense Fund", "Sierra Club",
        "Greenpeace", "Earthjustice", "Ocean Conservancy", "Wildlife Conservation Society",
        "Rainforest Alliance", "Global Water Partnership", "Water.org", "charity: water",
        "Pacific Institute", "Alliance for Water Stewardship", "CDP", "Ceres",
        "BSR", "World Economic Forum", "Aspen Institute", "Brookings Institution",
        "Carnegie Endowment", "Council on Foreign Relations", "RAND Corporation",
        "McKinsey & Company", "Boston Consulting Group", "Bain & Company", "Deloitte",
        "Accenture", "PwC", "EY", "KPMG", "Goldman Sachs", "JPMorgan Chase",
        "Bank of America", "Citigroup", "Morgan Stanley", "BlackRock", "Vanguard",
        "Ford Foundation", "Rockefeller Foundation", "MacArthur Foundation",
        "Gates Foundation", "Hewlett Foundation", "Packard Foundation", "Bloomberg Philanthropies",
        "Open Society Foundations", "Omidyar Network", "Skoll Foundation", "Toniic",
        "Stanford University", "Harvard University", "MIT", "Yale University",
        "Columbia University", "UC Berkeley", "Princeton University", "Oxford University"
    ]
    
    locations = [
        "San Francisco, CA", "New York, NY", "Washington, DC", "Boston, MA",
        "Los Angeles, CA", "Seattle, WA", "Chicago, IL", "Denver, CO",
        "Austin, TX", "Portland, OR", "Miami, FL", "Atlanta, GA",
        "London, UK", "Geneva, Switzerland", "Amsterdam, Netherlands",
        "Berlin, Germany", "Paris, France", "Singapore", "Hong Kong",
        "Tokyo, Japan", "Sydney, Australia", "Toronto, Canada", "Vancouver, Canada"
    ]
    
    sectors = [
        "Philanthropy", "Nonprofit", "Consulting", "Finance", "Technology",
        "Academia", "Government", "Social Impact", "Corporate", "Peacebuilding/Democracy"
    ]
    
    # Generate profile data based on URL hash
    temp_id = canonical_id_from_url(profile_url)
    
    # Use hash to pick attributes deterministically
    first_name = first_names[url_hash % len(first_names)]
    last_name = last_names[(url_hash // 100) % len(last_names)]
    title = titles[(url_hash // 1000) % len(titles)]
    org = organizations[(url_hash // 10000) % len(organizations)]
    location = locations[(url_hash // 100000) % len(locations)]
    sector = sectors[(url_hash // 1000000) % len(sectors)]
    
    full_name = f"{first_name} {last_name}"
    headline = f"{title} at {org}"
    occupation = headline
    
    # Generate 25-40 connections (deterministic based on hash)
    num_connections = 25 + (url_hash % 16)  # 25-40 connections
    
    people_also_viewed = []
    for i in range(num_connections):
        # Create unique but deterministic connection
        conn_hash = (url_hash + i * 7919) % (2**32)  # Prime multiplier for variety
        
        conn_first = first_names[conn_hash % len(first_names)]
        conn_last = last_names[(conn_hash // 100) % len(last_names)]
        conn_title = titles[(conn_hash // 1000) % len(titles)]
        conn_org = organizations[(conn_hash // 10000) % len(organizations)]
        conn_location = locations[(conn_hash // 100000) % len(locations)]
        
        conn_name = f"{conn_first} {conn_last}"
        conn_id = f"{conn_first.lower()}-{conn_last.lower()}-{conn_hash % 1000}"
        
        people_also_viewed.append({
            "link": f"https://www.linkedin.com/in/{conn_id}",
            "name": conn_name,
            "summary": f"{conn_title} at {conn_org}",
            "location": conn_location
        })
    
    # Build response matching EnrichLayer v2 API format
    return {
        "public_identifier": temp_id,
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "headline": headline,
        "occupation": occupation,
        "location_str": location,
        "summary": f"Experienced {title.lower()} with expertise in {sector.lower()}.",
        "experiences": [
            {
                "company": org,
                "title": title,
                "starts_at": {"year": 2020, "month": 1},
                "ends_at": None
            },
            {
                "company": organizations[(url_hash // 50000) % len(organizations)],
                "title": titles[(url_hash // 5000) % len(titles)],
                "starts_at": {"year": 2015, "month": 6},
                "ends_at": {"year": 2019, "month": 12}
            }
        ],
        "people_also_viewed": people_also_viewed
    }


# ============================================================================
# BFS CRAWLER
# ============================================================================

def run_crawler(
    seeds: List[Dict],
    api_token: str,
    max_degree: int,
    max_edges: int,
    max_nodes: int,
    status_container,
    mock_mode: bool = False,
    advanced_mode: bool = False,
    progress_bar = None,
    per_min_limit: int = PER_MIN_LIMIT
) -> Tuple[Dict, List, List, Dict]:
    """
    Run BFS crawler on seed profiles.
    
    Returns:
        (seen_profiles, edges, raw_profiles, stats)
    """
    # Initialize rate limiter
    rate_limiter = None
    if not mock_mode:
        rate_limiter = RateLimiter(per_min_limit=per_min_limit)
    
    # Initialize data structures
    queue = deque()
    seen_profiles = {}
    edges = []
    raw_profiles = []
    processed_nodes = 0  # Track progress
    
    # Statistics tracking
    stats = {
        'api_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'nodes_added': 0,
        'edges_added': 0,
        'max_degree_reached': 0,
        'stopped_reason': None,
        'profiles_with_no_neighbors': 0,
        'error_breakdown': {
            'rate_limit': 0,
            'out_of_credits': 0,
            'auth_error': 0,
            'not_found': 0,
            'enrichment_failed': 0,
            'other': 0,
            'consecutive_rate_limits': 0
        }
    }
    
    # Initialize seeds
    status_container.write("🌱 Initializing seed profiles...")
    for seed in seeds:
        temp_id = canonical_id_from_url(seed['profile_url'])
        node = {
            'id': temp_id,
            'name': seed['name'],
            'profile_url': seed['profile_url'],
            'headline': '',
            'location': '',
            'degree': 0,
            'source_type': 'seed'
        }
        seen_profiles[temp_id] = node
        queue.append(temp_id)
        stats['nodes_added'] += 1
    
    status_container.write(f"✅ Added {len(seeds)} seed profiles to queue")
    
    # BFS crawl
    while queue:
        # Check global limits
        if len(edges) >= max_edges:
            stats['stopped_reason'] = 'edge_limit'
            status_container.warning(f"⚠️ Reached edge limit ({max_edges}). Stopping crawl.")
            break
        
        if len(seen_profiles) >= max_nodes:
            stats['stopped_reason'] = 'node_limit'
            status_container.warning(f"⚠️ Reached node limit ({max_nodes}). Stopping crawl.")
            break
        
        current_id = queue.popleft()
        current_node = seen_profiles[current_id]
        processed_nodes += 1
        
        # Update progress bar (based on processed vs total known)
        if progress_bar is not None:
            total_known = processed_nodes + len(queue)
            if total_known > 0:
                progress = processed_nodes / total_known
                progress_bar.progress(
                    min(max(progress, 0.0), 0.99),  # Cap at 99% until complete
                    text=f"Processing... {processed_nodes} done, {len(queue)} remaining"
                )
        
        # Stop expanding if at max degree
        if current_node['degree'] >= max_degree:
            continue
        
        # Status update with real-time stats
        progress_text = f"🔍 Processing: {current_node['name']} (degree {current_node['degree']})"
        if stats['api_calls'] > 0:
            success_rate = (stats['successful_calls'] / stats['api_calls']) * 100
            progress_text += f" | ✅ {stats['successful_calls']}/{stats['api_calls']} ({success_rate:.0f}%)"
            if stats['failed_calls'] > 0:
                progress_text += f" | ❌ {stats['failed_calls']} failed"
                if stats['error_breakdown'].get('rate_limit', 0) > 0:
                    progress_text += f" (Rate limited: {stats['error_breakdown']['rate_limit']})"
        if rate_limiter is not None:
            progress_text += f" | ⏱️ {rate_limiter.get_status()}"
        status_container.write(progress_text)
        
        # Rate limiting: wait for a safe slot before calling the API
        if rate_limiter is not None:
            rate_limiter.wait_for_slot()
        
        # Call EnrichLayer API
        stats['api_calls'] += 1
        response, error = call_enrichlayer_api(api_token, current_node['profile_url'], mock_mode=mock_mode)
        
        # Record the API call for rate limiting
        if rate_limiter is not None:
            rate_limiter.record_call()
        
        # Tiny courtesy delay (the real throttle is per-minute via RateLimiter)
        if not mock_mode:
            time.sleep(0.2)
        
        if error:
            stats['failed_calls'] += 1
            status_container.error(f"❌ Failed to fetch {current_node['profile_url']}: {error}")
            
            # Classify error type
            if "Rate limit" in error:
                stats['error_breakdown']['rate_limit'] += 1
                
                # Check for excessive consecutive rate limits
                consecutive_rate_limits = stats['error_breakdown'].get('consecutive_rate_limits', 0) + 1
                stats['error_breakdown']['consecutive_rate_limits'] = consecutive_rate_limits
                
                if consecutive_rate_limits >= 10:
                    st.warning(f"""
                    **⏸️ Pausing: Too many consecutive rate limits ({consecutive_rate_limits})**
                    
                    Waiting 30 seconds before continuing...
                    Consider stopping and using Degree 1 instead.
                    """)
                    time.sleep(30)  # Long pause after many consecutive failures
                    stats['error_breakdown']['consecutive_rate_limits'] = 0  # Reset counter
            elif "Out of credits" in error:
                stats['error_breakdown']['out_of_credits'] += 1
                stats['stopped_reason'] = 'out_of_credits'
                st.error("🚫 **Out of Credits!** Check your EnrichLayer balance.")
                break
            elif "Invalid API token" in error:
                stats['error_breakdown']['auth_error'] += 1
                stats['stopped_reason'] = 'auth_error'
                break
            elif "Enrichment failed" in error:
                stats['error_breakdown']['enrichment_failed'] += 1
            elif "404" in error or "not found" in error.lower():
                stats['error_breakdown']['not_found'] += 1
            else:
                stats['error_breakdown']['other'] += 1
            
            # Continue with other profiles for non-fatal errors
            continue
        
        stats['successful_calls'] += 1
        stats['error_breakdown']['consecutive_rate_limits'] = 0  # Reset on success
        raw_profiles.append(response)
        
        # Update node with enriched data
        enriched_id = response.get('public_identifier', current_id)
        current_node['headline'] = response.get('headline', '')
        current_node['location'] = response.get('location_str') or response.get('location', '')
        
        # Advanced mode: Extract organization and sector
        if advanced_mode:
            occupation = response.get('occupation', '')
            experiences = response.get('experiences', [])
            organization = extract_organization(occupation, experiences)
            sector = infer_sector(organization, current_node['headline'])
            
            current_node['organization'] = organization
            current_node['sector'] = sector
        
        # Update canonical ID if different
        if enriched_id != current_id:
            update_canonical_ids(seen_profiles, edges, current_id, enriched_id)
            current_id = enriched_id
            current_node = seen_profiles[current_id]
        
        # Extract neighbors
        neighbors = response.get('people_also_viewed', [])
        
        # Improvement #4: Clear messaging for no neighbors
        if not neighbors:
            status_container.write("   └─ ⚠️ No 'people also viewed' connections found for this profile.")
            stats['profiles_with_no_neighbors'] += 1
        else:
            status_container.write(f"   └─ Found {len(neighbors)} connections")
        
        # Process each neighbor
        for neighbor in neighbors:
            if len(edges) >= max_edges:
                status_container.warning(f"⚠️ Reached edge limit ({max_edges}) while processing neighbors.")
                break
            
            # Handle both v2 API format and mock data format
            # v2 API uses: link, name, summary
            # Mock uses: profile_url, full_name, headline, public_identifier
            neighbor_url = neighbor.get('link') or neighbor.get('profile_url', '')
            neighbor_name = neighbor.get('name') or neighbor.get('full_name', '')
            neighbor_headline = neighbor.get('summary') or neighbor.get('headline', '')
            
            if not neighbor_url:
                continue
            
            # Use public_identifier if available (mock data), otherwise extract from URL
            neighbor_id = neighbor.get('public_identifier', canonical_id_from_url(neighbor_url))
            
            # Add edge
            edges.append({
                'source_id': current_id,
                'target_id': neighbor_id,
                'edge_type': 'people_also_viewed'
            })
            stats['edges_added'] += 1
            
            # Skip if already seen
            if neighbor_id in seen_profiles:
                continue
            
            # Check node limit
            if len(seen_profiles) >= max_nodes:
                status_container.warning(f"⚠️ Reached node limit ({max_nodes}) while processing neighbors.")
                break
            
            # Create new node
            neighbor_node = {
                'id': neighbor_id,
                'name': neighbor_name,
                'profile_url': neighbor_url,
                'headline': neighbor_headline,
                'location': neighbor.get('location', ''),
                'degree': current_node['degree'] + 1,
                'source_type': 'discovered'
            }
            
            # Advanced mode: Add organization and sector (will be populated when fetched)
            if advanced_mode:
                neighbor_node['organization'] = ''  # Will be filled when profile is fetched
                neighbor_node['sector'] = ''
            
            seen_profiles[neighbor_id] = neighbor_node
            stats['nodes_added'] += 1
            
            # Track max degree
            stats['max_degree_reached'] = max(stats['max_degree_reached'], neighbor_node['degree'])
            
            # Enqueue if can still be expanded
            if neighbor_node['degree'] < max_degree:
                queue.append(neighbor_id)
    
    if not stats['stopped_reason']:
        stats['stopped_reason'] = 'completed'
    
    return seen_profiles, edges, raw_profiles, stats


# ============================================================================
# NETWORK METRICS CALCULATION
# ============================================================================

def calculate_network_metrics(seen_profiles: Dict, edges: List) -> Dict:
    """
    Calculate network centrality metrics using NetworkX.
    
    Returns dict with:
        - node_metrics: {node_id: {metric_name: value, ...}, ...}
        - network_stats: {metric_name: value, ...}
        - top_nodes: {metric_name: [(node_id, value), ...], ...}
    """
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node_id, node_data in seen_profiles.items():
        G.add_node(node_id, **node_data)
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source_id'], edge['target_id'])
    
    # Initialize results
    node_metrics = {node_id: {} for node_id in seen_profiles.keys()}
    network_stats = {}
    top_nodes = {}
    
    # Skip calculations if graph is too small
    if len(G.nodes()) < 2 or len(G.edges()) < 1:
        return {
            'node_metrics': node_metrics,
            'network_stats': {'nodes': len(G.nodes()), 'edges': len(G.edges())},
            'top_nodes': {},
            'brokerage_roles': {},
            'communities': {}
        }
    
    # ----- DEGREE CENTRALITY -----
    # Number of connections (normalized by max possible)
    try:
        degree_centrality = nx.degree_centrality(G)
        for node_id, value in degree_centrality.items():
            if node_id in node_metrics:
                node_metrics[node_id]['degree_centrality'] = round(value, 4)
        
        # Also store raw degree count
        for node_id in G.nodes():
            if node_id in node_metrics:
                node_metrics[node_id]['degree'] = G.degree(node_id)
        
        # Top 10 by degree
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['degree'] = sorted_degree
        
        network_stats['avg_degree'] = round(sum(dict(G.degree()).values()) / len(G.nodes()), 2)
        network_stats['max_degree'] = max(dict(G.degree()).values())
    except Exception as e:
        pass  # Skip if calculation fails
    
    # ----- BETWEENNESS CENTRALITY -----
    # How often a node lies on shortest paths between other nodes (identifies brokers)
    try:
        betweenness = nx.betweenness_centrality(G)
        for node_id, value in betweenness.items():
            if node_id in node_metrics:
                node_metrics[node_id]['betweenness_centrality'] = round(value, 4)
        
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['betweenness'] = sorted_betweenness
        
        network_stats['avg_betweenness'] = round(sum(betweenness.values()) / len(betweenness), 4)
    except Exception as e:
        pass
    
    # ----- EIGENVECTOR CENTRALITY -----
    # Connected to well-connected people (influence)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        for node_id, value in eigenvector.items():
            if node_id in node_metrics:
                node_metrics[node_id]['eigenvector_centrality'] = round(value, 4)
        
        sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['eigenvector'] = sorted_eigenvector
    except Exception as e:
        # Eigenvector can fail on disconnected graphs
        pass
    
    # ----- CLOSENESS CENTRALITY -----
    # Average distance to all other nodes (accessibility)
    try:
        # Use connected components for disconnected graphs
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            # Calculate for largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subgraph)
        
        for node_id, value in closeness.items():
            if node_id in node_metrics:
                node_metrics[node_id]['closeness_centrality'] = round(value, 4)
        
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['closeness'] = sorted_closeness
    except Exception as e:
        pass
    
    # ----- COMMUNITY DETECTION & BROKERAGE ROLES -----
    communities = {}
    brokerage_roles = {}
    try:
        if len(G.nodes()) >= 3 and len(G.edges()) >= 2:
            communities = detect_communities(G)
            brokerage_roles = compute_brokerage_roles(G, communities)
            
            # Store in node metrics
            for node_id in node_metrics:
                if node_id in communities:
                    node_metrics[node_id]['community'] = communities[node_id]
                if node_id in brokerage_roles:
                    node_metrics[node_id]['brokerage_role'] = brokerage_roles[node_id]
            
            # Count communities
            if communities:
                network_stats['num_communities'] = len(set(communities.values()))
            
            # Count brokerage roles
            if brokerage_roles:
                role_counts = {}
                for role in brokerage_roles.values():
                    role_counts[role] = role_counts.get(role, 0) + 1
                network_stats['brokerage_role_counts'] = role_counts
    except Exception as e:
        pass  # Skip if community detection fails
    
    # ----- NETWORK-LEVEL STATS -----
    network_stats['nodes'] = len(G.nodes())
    network_stats['edges'] = len(G.edges())
    
    try:
        network_stats['density'] = round(nx.density(G), 4)
    except:
        pass
    
    try:
        if nx.is_connected(G):
            network_stats['diameter'] = nx.diameter(G)
            network_stats['avg_path_length'] = round(nx.average_shortest_path_length(G), 2)
        else:
            # For disconnected graphs, report on largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            network_stats['largest_component_size'] = len(largest_cc)
            network_stats['num_components'] = nx.number_connected_components(G)
            if len(subgraph) > 1:
                network_stats['diameter_largest_cc'] = nx.diameter(subgraph)
    except:
        pass
    
    try:
        network_stats['avg_clustering'] = round(nx.average_clustering(G), 4)
    except:
        pass
    
    return {
        'node_metrics': node_metrics,
        'network_stats': network_stats,
        'top_nodes': top_nodes,
        'brokerage_roles': brokerage_roles,
        'communities': communities
    }


def generate_network_analysis_json(network_metrics: Dict, seen_profiles: Dict) -> str:
    """Generate network_analysis.json with summary statistics and top nodes."""
    
    analysis = {
        'generated_at': datetime.now().isoformat(),
        'network_statistics': network_metrics.get('network_stats', {}),
        'top_connectors': [],
        'top_brokers': [],
        'top_influencers': [],
        'metric_definitions': {
            'degree_centrality': 'Number of direct connections (normalized). High = well-connected.',
            'betweenness_centrality': 'How often node lies on shortest paths. High = broker/bridge.',
            'eigenvector_centrality': 'Connected to influential people. High = influential.',
            'closeness_centrality': 'Average distance to all others. High = central/accessible.'
        }
    }
    
    top_nodes = network_metrics.get('top_nodes', {})
    
    # Add top connectors (by degree)
    if 'degree' in top_nodes:
        for node_id, score in top_nodes['degree']:
            if node_id in seen_profiles:
                analysis['top_connectors'].append({
                    'id': node_id,
                    'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'degree_centrality': score,
                    'connections': network_metrics['node_metrics'].get(node_id, {}).get('degree', 0)
                })
    
    # Add top brokers (by betweenness)
    if 'betweenness' in top_nodes:
        for node_id, score in top_nodes['betweenness']:
            if node_id in seen_profiles:
                analysis['top_brokers'].append({
                    'id': node_id,
                    'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'betweenness_centrality': score
                })
    
    # Add top influencers (by eigenvector)
    if 'eigenvector' in top_nodes:
        for node_id, score in top_nodes['eigenvector']:
            if node_id in seen_profiles:
                analysis['top_influencers'].append({
                    'id': node_id,
                    'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'eigenvector_centrality': score
                })
    
    return json.dumps(analysis, indent=2)


# ============================================================================
# CSV/JSON GENERATION
# ============================================================================

def generate_nodes_csv(seen_profiles: Dict, max_degree: int, max_edges: int, max_nodes: int, network_metrics: Dict = None) -> str:
    """Generate nodes.csv content with metadata header and optional network metrics."""
    nodes_data = []
    
    # Get node metrics if available
    node_metrics = {}
    if network_metrics:
        node_metrics = network_metrics.get('node_metrics', {})
    
    for node in seen_profiles.values():
        node_dict = {
            'id': node['id'],
            'name': node['name'],
            'profile_url': node['profile_url'],
            'headline': node.get('headline', ''),
            'location': node.get('location', ''),
            'degree': node['degree'],
            'source_type': node['source_type']
        }
        
        # Add organization and sector if present (advanced mode)
        if 'organization' in node:
            node_dict['organization'] = node.get('organization', '')
        if 'sector' in node:
            node_dict['sector'] = node.get('sector', '')
        
        # Add network metrics if available (advanced mode)
        if node['id'] in node_metrics:
            metrics = node_metrics[node['id']]
            node_dict['connections'] = metrics.get('degree', 0)
            node_dict['degree_centrality'] = metrics.get('degree_centrality', 0)
            node_dict['betweenness_centrality'] = metrics.get('betweenness_centrality', 0)
            node_dict['eigenvector_centrality'] = metrics.get('eigenvector_centrality', 0)
            node_dict['closeness_centrality'] = metrics.get('closeness_centrality', 0)
        
        nodes_data.append(node_dict)
    
    df = pd.DataFrame(nodes_data)
    csv_body = df.to_csv(index=False)
    
    # Improvement #5: Add metadata header
    meta = (
        f"# generated_at={datetime.utcnow().isoformat()}Z; "
        f"max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}\n"
    )
    
    return meta + csv_body


def generate_edges_csv(edges: List, max_degree: int, max_edges: int, max_nodes: int) -> str:
    """Generate edges.csv content with metadata header."""
    df = pd.DataFrame(edges)
    csv_body = df.to_csv(index=False)
    
    # Improvement #5: Add metadata header
    meta = (
        f"# generated_at={datetime.utcnow().isoformat()}Z; "
        f"max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}\n"
    )
    
    return meta + csv_body


def generate_raw_json(raw_profiles: List) -> str:
    """Generate raw_profiles.json content."""
    return json.dumps(raw_profiles, indent=2)


def generate_insights_report(
    network_stats: Dict,
    health_score: int,
    health_label: str,
    health_details: List[str],
    sector_analysis: 'SectorAnalysis',
    top_connectors: List[Tuple],
    top_brokers: List[Tuple],
    brokerage_roles: Dict[str, str],
    recommendations: str,
    seen_profiles: Dict,
    node_metrics: Dict = None,
    degree_pcts: Dict = None,
    betweenness_pcts: Dict = None,
    eigenvector_pcts: Dict = None,
    closeness_pcts: Dict = None,
) -> str:
    """
    Generate a comprehensive Markdown report of network insights.
    """
    from datetime import datetime
    
    lines = []
    
    # Header
    lines.append("# Network Analysis Report")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Nodes:** {network_stats.get('nodes', 0)} | **Edges:** {network_stats.get('edges', 0)}")
    lines.append("")
    
    # Network Health
    lines.append("---")
    lines.append(f"## 🏥 Network Health: {health_score}/100 ({health_label})")
    lines.append("")
    for detail in health_details:
        lines.append(f"- {detail}")
    lines.append("")
    
    # Key Statistics
    lines.append("---")
    lines.append("## 📊 Network Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Nodes | {network_stats.get('nodes', 0)} |")
    lines.append(f"| Edges | {network_stats.get('edges', 0)} |")
    lines.append(f"| Density | {network_stats.get('density', 0):.4f} |")
    lines.append(f"| Avg Degree | {network_stats.get('avg_degree', 0):.1f} |")
    lines.append(f"| Avg Clustering | {network_stats.get('avg_clustering', 0):.4f} |")
    lines.append(f"| Components | {network_stats.get('num_components', 1)} |")
    lines.append(f"| Largest Component | {network_stats.get('largest_component_size', 0)} nodes |")
    lines.append("")
    
    # Sector Distribution
    if sector_analysis:
        lines.append("---")
        lines.append("## 🏢 Sector Distribution")
        lines.append("")
        lines.append(f"**Diversity Score:** {sector_analysis.diversity_score:.2f} ({sector_analysis.diversity_label})")
        lines.append("")
        
        if sector_analysis.dominant_sectors:
            lines.append(f"**⚠️ Dominant Sectors:** {', '.join(sector_analysis.dominant_sectors)}")
        if sector_analysis.underrepresented_sectors:
            lines.append(f"**💡 Underrepresented:** {', '.join(sector_analysis.underrepresented_sectors)}")
        lines.append("")
        
        lines.append("| Sector | Count | % of Classified |")
        lines.append("|--------|-------|-----------------|")
        for _, row in sector_analysis.df.iterrows():
            lines.append(f"| {row['sector']} | {row['count']} | {row['pct_classified']:.1f}% |")
        lines.append("")
    
    # Brokerage Roles
    if brokerage_roles:
        lines.append("---")
        lines.append("## 🎭 Brokerage Roles")
        lines.append("")
        lines.append("*How people connect groups in the network*")
        lines.append("")
        
        role_counts = {}
        for role in brokerage_roles.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        
        role_labels = {
            "liaison": "🌉 Liaison (cross-group bridge)",
            "gatekeeper": "🚪 Gatekeeper (controls access)",
            "representative": "🔗 Representative (outbound voice)",
            "coordinator": "🧩 Coordinator (internal organizer)",
            "consultant": "🧠 Consultant (multi-group advisor)",
            "peripheral": "⚪ Peripheral (edge of network)",
        }
        
        lines.append("| Role | Count | Description |")
        lines.append("|------|-------|-------------|")
        for role_key in ["liaison", "gatekeeper", "representative", "coordinator", "consultant", "peripheral"]:
            count = role_counts.get(role_key, 0)
            label = role_labels.get(role_key, role_key)
            lines.append(f"| {label} | {count} | |")
        lines.append("")
    
    # Top Connectors
    if top_connectors:
        lines.append("---")
        lines.append("## 🔗 Top Connectors (by Degree)")
        lines.append("")
        lines.append("*People with the most direct connections*")
        lines.append("")
        
        for i, (node_id, score) in enumerate(top_connectors[:10], 1):
            profile = seen_profiles.get(node_id, {})
            name = profile.get('name', node_id)
            org = profile.get('organization', 'Unknown')
            sector = profile.get('sector', '')
            connections = node_metrics.get(node_id, {}).get('degree', 0) if node_metrics else int(score * 100)
            role = brokerage_roles.get(node_id, 'unknown') if brokerage_roles else 'unknown'
            
            lines.append(f"### {i}. {name} ({org})")
            lines.append(f"- **Connections:** {connections}")
            lines.append(f"- **Sector:** {sector or 'Unknown'}")
            lines.append(f"- **Role:** {role.title()}")
            
            # Generate recommendation if we have percentiles
            if degree_pcts and betweenness_pcts:
                dominant = set(sector_analysis.dominant_sectors) if sector_analysis else set()
                underrep = set(sector_analysis.underrepresented_sectors) if sector_analysis else set()
                is_dom = sector in dominant if sector else False
                is_under = sector in underrep if sector else False
                
                blurb, rec = describe_node_with_recommendation(
                    name=name,
                    organization=org,
                    role=role,
                    degree_pct=degree_pcts.get(node_id),
                    betweenness_pct=betweenness_pcts.get(node_id),
                    eigenvector_pct=eigenvector_pcts.get(node_id) if eigenvector_pcts else None,
                    closeness_pct=closeness_pcts.get(node_id) if closeness_pcts else None,
                    sector=sector,
                    is_dominant_sector=is_dom,
                    is_underrepresented_sector=is_under,
                )
                lines.append(f"- **Analysis:** {blurb}")
                if rec:
                    lines.append(f"- **💡 Suggested Focus:** {rec}")
            lines.append("")
    
    # Top Brokers
    if top_brokers:
        lines.append("---")
        lines.append("## 🌉 Top Brokers (by Betweenness)")
        lines.append("")
        lines.append("*People who bridge different parts of the network*")
        lines.append("")
        
        for i, (node_id, score) in enumerate(top_brokers[:10], 1):
            profile = seen_profiles.get(node_id, {})
            name = profile.get('name', node_id)
            org = profile.get('organization', 'Unknown')
            sector = profile.get('sector', '')
            role = brokerage_roles.get(node_id, 'unknown') if brokerage_roles else 'unknown'
            
            lines.append(f"### {i}. {name} ({org})")
            lines.append(f"- **Betweenness:** {score:.4f}")
            lines.append(f"- **Sector:** {sector or 'Unknown'}")
            lines.append(f"- **Role:** {role.title()}")
            
            # Generate recommendation if we have percentiles
            if degree_pcts and betweenness_pcts:
                dominant = set(sector_analysis.dominant_sectors) if sector_analysis else set()
                underrep = set(sector_analysis.underrepresented_sectors) if sector_analysis else set()
                is_dom = sector in dominant if sector else False
                is_under = sector in underrep if sector else False
                
                blurb, rec = describe_node_with_recommendation(
                    name=name,
                    organization=org,
                    role=role,
                    degree_pct=degree_pcts.get(node_id),
                    betweenness_pct=betweenness_pcts.get(node_id),
                    eigenvector_pct=eigenvector_pcts.get(node_id) if eigenvector_pcts else None,
                    closeness_pct=closeness_pcts.get(node_id) if closeness_pcts else None,
                    sector=sector,
                    is_dominant_sector=is_dom,
                    is_underrepresented_sector=is_under,
                )
                lines.append(f"- **Analysis:** {blurb}")
                if rec:
                    lines.append(f"- **💡 Suggested Focus:** {rec}")
            lines.append("")
    
    # Recommendations
    if recommendations:
        lines.append("---")
        lines.append("## 🚀 Strategic Recommendations")
        lines.append("")
        lines.append(recommendations)
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("*Report generated by C4C Network Seed Crawler*")
    lines.append("")
    
    return "\n".join(lines)


def create_brokerage_role_chart(brokerage_roles: Dict[str, str]) -> 'go.Figure':
    """
    Create a horizontal bar chart showing brokerage role distribution.
    Returns a Plotly Figure.
    """
    if not brokerage_roles:
        return None
    
    # Count roles
    role_counts = {}
    for role in brokerage_roles.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    
    # Define order and colors
    role_order = ["liaison", "gatekeeper", "representative", "coordinator", "consultant", "peripheral"]
    role_labels = {
        "liaison": "🌉 Liaison",
        "gatekeeper": "🚪 Gatekeeper", 
        "representative": "🔗 Representative",
        "coordinator": "🧩 Coordinator",
        "consultant": "🧠 Consultant",
        "peripheral": "⚪ Peripheral",
    }
    role_colors = {
        "liaison": "#D97706",      # Amber
        "gatekeeper": "#F97316",   # Orange
        "representative": "#10B981", # Green
        "coordinator": "#3B82F6",  # Blue
        "consultant": "#6366F1",   # Indigo
        "peripheral": "#9CA3AF",   # Gray
    }
    
    # Build data in order
    roles = []
    counts = []
    colors = []
    for role in role_order:
        if role in role_counts:
            roles.append(role_labels.get(role, role))
            counts.append(role_counts[role])
            colors.append(role_colors.get(role, "#9CA3AF"))
    
    # Create chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=roles,
        x=counts,
        orientation='h',
        marker_color=colors,
        text=counts,
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>%{x} people<extra></extra>"
    ))
    
    fig.update_layout(
        title="Brokerage Role Distribution",
        xaxis_title="Number of People",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    
    return fig


def fig_to_png_bytes(fig: 'go.Figure') -> Tuple[Optional[bytes], Optional[str]]:
    """
    Convert a Plotly figure to PNG bytes.
    Requires kaleido package.
    Returns (png_bytes, error_message)
    """
    try:
        import kaleido
        png_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        return png_bytes, None
    except ImportError:
        return None, "kaleido not installed"
    except Exception as e:
        return None, str(e)


def create_download_zip(
    nodes_csv: str, 
    edges_csv: str, 
    raw_json: str, 
    analysis_json: str = None,
    insights_report: str = None,
    sector_chart_png: bytes = None,
    brokerage_chart_png: bytes = None,
) -> bytes:
    """
    Create a ZIP file containing all output files.
    Returns ZIP file as bytes.
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add nodes.csv
        zip_file.writestr('nodes.csv', nodes_csv)
        # Add edges.csv
        zip_file.writestr('edges.csv', edges_csv)
        # Add raw_profiles.json
        zip_file.writestr('raw_profiles.json', raw_json)
        # Add network_analysis.json if available
        if analysis_json:
            zip_file.writestr('network_analysis.json', analysis_json)
        # Add insights report if available
        if insights_report:
            zip_file.writestr('network_insights_report.md', insights_report)
        # Add charts if available
        if sector_chart_png:
            zip_file.writestr('charts/sector_distribution.png', sector_chart_png)
        if brokerage_chart_png:
            zip_file.writestr('charts/brokerage_roles.png', brokerage_chart_png)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="C4C Network Seed Crawler",
        page_icon="https://static.wixstatic.com/media/275a3f_9c48d5079fcf4b688606c81d8f34d5a5~mv2.jpg",
        layout="wide"
    )
    
    # Initialize session state for preserving results
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = None
    
    # Header with C4C logo
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image(
            "https://static.wixstatic.com/media/275a3f_9c48d5079fcf4b688606c81d8f34d5a5~mv2.jpg",
            width=80
        )
    with col2:
        st.title("C4C Network Seed Crawler")
    
    st.markdown("Convert LinkedIn seed profiles into a Polinode-ready network using EnrichLayer")
    
    # ========================================================================
    # MODE SELECTION
    # ========================================================================
    
    st.markdown("---")
    
    st.subheader("🎛️ Select Mode")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        advanced_mode = st.toggle(
            "Advanced Mode",
            value=False,
            help="Enable network analysis and insights"
        )
    
    with col2:
        if advanced_mode:
            st.info("""
            **🔬 Advanced Mode** - Network Intelligence  
            Includes everything in Basic Mode plus:
            - Centrality metrics (degree, betweenness, eigenvector, closeness)
            - Community detection and clustering
            - Brokerage analysis (coordinators, gatekeepers, liaisons)
            - Key position identification (connectors, brokers, bridges)
            - Network insights and strategic recommendations
            
            *⏱️ Longer processing time, richer insights*
            """)
        else:
            st.success("""
            **📊 Basic Mode** - Quick Network Crawl  
            Perfect for rapid exploration:
            - Crawl LinkedIn networks (1 or 2 degrees)
            - Export nodes, edges, and raw profiles
            - Import directly to Polinode or other tools
            - Fast processing, clean data
            
            *⚡ Quick results, simple outputs*
            """)
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 1: INPUT
    # ========================================================================
    
    st.header("📥 Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Upload Seed Profiles")
        uploaded_file = st.file_uploader(
            "Upload CSV with columns: name, profile_url (max 5 rows)",
            type=['csv'],
            help="CSV must contain 'name' and 'profile_url' columns with 1-5 seed profiles"
        )
        
        seeds = []
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['name', 'profile_url']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                elif len(df) > 5:
                    st.error("❌ Prototype limit: max 5 seed profiles.")
                elif len(df) == 0:
                    st.error("❌ CSV file is empty.")
                else:
                    seeds = df.to_dict('records')
                    st.success(f"✅ Loaded {len(seeds)} seed profiles")
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with col2:
        st.subheader("2. EnrichLayer API Token")
        
        # Improvement #6: Optional auto-fill from secrets
        default_token = ""
        try:
            default_token = st.secrets.get("ENRICHLAYER_TOKEN", "")
        except:
            pass
        
        api_token = st.text_input(
            "Enter your API token",
            type="password",
            value=default_token,
            help="Get your token from EnrichLayer dashboard. Not stored, used only for this session."
        )
        
        # Network connectivity test
        if st.button("🔍 Test API Connection", help="Check if EnrichLayer API is reachable"):
            with st.spinner("Testing connection..."):
                success, message = test_network_connectivity()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Improvement #1: UI-based mock mode toggle
        mock_mode = st.toggle(
            "Run in mock mode (no real API calls)",
            value=DEFAULT_MOCK_MODE,
            help="Use mock responses for testing without consuming API credits."
        )
        
        if mock_mode:
            st.info("""
            🧪 **MOCK MODE** - No real API calls, no credits used!
            
            Generates realistic synthetic network data:
            - 25-40 connections per profile (deterministic)
            - Realistic names, titles, organizations
            - Perfect for stress testing node/edge limits
            - Use `mock_test_seeds.csv` or any seed file
            """)
    
    # ========================================================================
    # SECTION 2: CONFIGURATION
    # ========================================================================
    
    st.header("⚙️ Crawl Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_degree = st.radio(
            "Maximum Degree (hops)",
            options=[1, 2],
            index=0,  # Default to Degree 1 (safer)
            help="1 hop = direct connections only, 2 hops = connections of connections"
        )
        
        # Warning for degree 2
        if max_degree == 2:
            st.error("""
            **⚠️ Degree 2 Warning - Read Before Running!**
            
            Degree 2 crawls are **expensive and slow**:
            - 📊 10-50x more API calls than Degree 1
            - 💳 Uses 100-500 credits
            - ⏱️ Takes 30-90+ minutes (at 20 req/min)
            - 🚫 May still hit rate limits on large networks
            
            **💡 Recommendation:** Start with Degree 1 first!
            """, icon="🚨")
        else:
            st.success("""
            **✅ Degree 1 Selected - Good Choice!**
            
            - 🎯 Direct connections only
            - ⚡ ~1-2 minutes for 5 seeds
            - 💰 Uses ~5-10 credits
            - ✅ Reliable, low rate limit risk
            """, icon="👍")
    
    with col2:
        st.markdown("**Crawl Limits:**")
        st.metric("Max Edges", 10000)
        st.metric("Max Nodes", 7500)
    
    # Rate Limit Information (cleaner version per team feedback)
    st.caption(f"""
    ⏱️ **API pacing:** This prototype is tuned for up to **{PER_MIN_LIMIT} requests/minute**
    (EnrichLayer $49/mo plan). The app automatically throttles calls so we don't hit rate limits.
    Progress bar shows processed nodes vs. remaining queue.
    """)
    
    # ========================================================================
    # RUN BUTTON
    # ========================================================================
    
    can_run = len(seeds) > 0 and (api_token or mock_mode)
    
    if not can_run:
        if len(seeds) == 0:
            st.warning("⚠️ Please upload a valid seed CSV to continue.")
        elif not api_token and not mock_mode:
            st.warning("⚠️ Please enter your EnrichLayer API token to continue.")
    
    run_button = st.button(
        "🚀 Run Crawl",
        disabled=not can_run,
        type="primary",
        use_container_width=True
    )
    
    # ========================================================================
    # CRAWL EXECUTION
    # ========================================================================
    
    if run_button:
        st.header("🔄 Crawl Progress")
        
        # Progress bar
        progress_bar = st.progress(0.0, text="Starting crawl...")
        
        status_container = st.status("Running crawl...", expanded=True)
        
        # Run the crawler
        seen_profiles, edges, raw_profiles, stats = run_crawler(
            seeds=seeds,
            api_token=api_token,
            max_degree=max_degree,
            max_edges=10000,
            max_nodes=7500,
            status_container=status_container,
            mock_mode=mock_mode,
            advanced_mode=advanced_mode,
            progress_bar=progress_bar,
            per_min_limit=PER_MIN_LIMIT
        )
        
        progress_bar.progress(1.0, text="✅ Complete!")
        status_container.update(label="✅ Crawl Complete!", state="complete")
        
        # Improvement #3: Graph validation
        orphan_ids, valid_edges = validate_graph(seen_profiles, edges)
        
        if orphan_ids:
            st.warning(
                f"⚠️ Detected {len(orphan_ids)} orphan node IDs referenced in edges but "
                "not present in nodes. Edges involving these IDs have been excluded from the download."
            )
            edges = valid_edges
        
        # Improvement #4: Special message for empty results
        if len(edges) == 0:
            st.info(
                "ℹ️ Crawl completed, but no connections were found. "
                "This may mean the selected profiles have limited 'people also viewed' data, "
                "or that the crawl depth was too shallow."
            )
        
        # Calculate network metrics (only in advanced mode)
        network_metrics = None
        if advanced_mode and len(edges) > 0:
            with st.spinner("📊 Calculating network metrics..."):
                network_metrics = calculate_network_metrics(seen_profiles, edges)
        
        # Store results in session state so they persist across reruns (e.g., after downloads)
        st.session_state.crawl_results = {
            'seen_profiles': seen_profiles,
            'edges': edges,
            'raw_profiles': raw_profiles,
            'stats': stats,
            'max_degree': max_degree,
            'advanced_mode': advanced_mode,  # Store mode setting
            'network_metrics': network_metrics  # Store network metrics
        }
    
    # Display results if available (either from current run or session state)
    if st.session_state.crawl_results is not None:
        results = st.session_state.crawl_results
        seen_profiles = results['seen_profiles']
        edges = results['edges']
        raw_profiles = results['raw_profiles']
        stats = results['stats']
        max_degree = results['max_degree']
        was_advanced_mode = results.get('advanced_mode', False)
        network_metrics = results.get('network_metrics', None)
        
        # ====================================================================
        # RESULTS SUMMARY
        # ====================================================================
        
        st.header("📊 Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", len(seen_profiles))
        col2.metric("Total Edges", len(edges))
        col3.metric("Max Degree", stats['max_degree_reached'])
        col4.metric("API Calls", stats['api_calls'])
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Successful", stats['successful_calls'], delta_color="normal")
        col6.metric("Failed", stats['failed_calls'], delta_color="inverse")
        col7.metric("No Neighbors", stats['profiles_with_no_neighbors'])
        
        if stats['stopped_reason'] == 'completed':
            col8.success("✅ Completed")
        elif stats['stopped_reason'] == 'edge_limit':
            col8.warning("⚠️ Edge Limit")
        elif stats['stopped_reason'] == 'node_limit':
            col8.warning("⚠️ Node Limit")
        elif stats['stopped_reason'] == 'auth_error':
            col8.error("❌ Auth Error")
        elif stats['stopped_reason'] == 'out_of_credits':
            col8.error("❌ Out of Credits")
        
        # Show error breakdown if there were failures
        if stats['failed_calls'] > 0:
            st.markdown("---")
            st.subheader("❌ Error Breakdown")
            
            error_breakdown = stats.get('error_breakdown', {})
            if error_breakdown:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                if error_breakdown.get('rate_limit', 0) > 0:
                    col1.metric("Rate Limits", error_breakdown['rate_limit'], delta_color="off")
                if error_breakdown.get('out_of_credits', 0) > 0:
                    col2.metric("Out of Credits", error_breakdown['out_of_credits'], delta_color="off")
                if error_breakdown.get('auth_error', 0) > 0:
                    col3.metric("Auth Errors", error_breakdown['auth_error'], delta_color="off")
                if error_breakdown.get('not_found', 0) > 0:
                    col4.metric("Not Found", error_breakdown['not_found'], delta_color="off")
                if error_breakdown.get('enrichment_failed', 0) > 0:
                    col5.metric("Enrichment Failed", error_breakdown['enrichment_failed'], delta_color="off")
                if error_breakdown.get('other', 0) > 0:
                    col6.metric("Other Errors", error_breakdown['other'], delta_color="off")
                
                # Interpretation
                if error_breakdown.get('rate_limit', 0) > stats['failed_calls'] * 0.5:
                    st.warning("""
                    **⚠️ High Rate Limit Failures**
                    
                    Most failures were due to rate limiting. This suggests:
                    - Your crawl exceeded EnrichLayer's rate limits
                    - Consider using Degree 1 instead of Degree 2
                    - Space out large crawls over time
                    - Check your EnrichLayer plan limits
                    """)
                
                if error_breakdown.get('out_of_credits', 0) > 0:
                    st.error("""
                    **🚫 Out of Credits**
                    
                    You've exhausted your EnrichLayer credits. To continue:
                    1. Check your credit balance at EnrichLayer dashboard
                    2. Purchase more credits if needed
                    3. Resume your crawl
                    """)
            
            # ================================================================
            # DETAILED ERROR LOG (C4C Internal Use)
            # ================================================================
            with st.expander("🔧 Technical Details (C4C Internal)", expanded=False):
                st.markdown("### API Call Statistics")
                st.code(f"""
Total API Calls Attempted: {stats['api_calls']}
Successful Calls: {stats['successful_calls']}
Failed Calls: {stats['failed_calls']}
Success Rate: {(stats['successful_calls'] / max(stats['api_calls'], 1) * 100):.1f}%

Error Breakdown:
- Rate Limit (429): {error_breakdown.get('rate_limit', 0)}
- Out of Credits (403): {error_breakdown.get('out_of_credits', 0)}
- Auth Errors (401): {error_breakdown.get('auth_error', 0)}
- Not Found (404): {error_breakdown.get('not_found', 0)}
- Enrichment Failed (503): {error_breakdown.get('enrichment_failed', 0)}
- Other Errors: {error_breakdown.get('other', 0)}

Crawl Configuration:
- Max Degree: {max_degree}
- Max Edges Limit: 10000
- Max Nodes Limit: 7500
- API Delay: {API_DELAY} seconds between calls
- Stopped Reason: {stats.get('stopped_reason', 'unknown')}
                """, language="text")
                
                # Rate limit diagnosis
                if error_breakdown.get('rate_limit', 0) > 10:
                    st.markdown("### 🔍 Rate Limit Diagnosis")
                    st.warning(f"""
                    **High rate limit failures detected ({error_breakdown.get('rate_limit', 0)} errors)**
                    
                    **Possible causes:**
                    1. EnrichLayer has strict per-minute limits (likely 60/min or less)
                    2. Degree 2 crawls make too many rapid requests
                    3. Your plan tier may have lower limits
                    
                    **Current settings:**
                    - API Delay: {API_DELAY} seconds (should give ~{60/API_DELAY:.0f} calls/min)
                    - This run attempted: {stats['api_calls']} calls
                    
                    **Recommendations:**
                    - Use Degree 1 for most crawls
                    - Contact EnrichLayer support about rate limits
                    - Consider increasing API_DELAY to 3-4 seconds
                    """)
                
                st.markdown("### 📊 Network Statistics")
                st.code(f"""
Total Nodes Discovered: {len(seen_profiles)}
Total Edges Discovered: {len(edges)}
Nodes with Organization Data: {sum(1 for n in seen_profiles.values() if n.get('organization'))}
Max Degree Reached: {stats.get('max_degree_reached', 0)}
Profiles With No Neighbors: {stats.get('profiles_with_no_neighbors', 0)}
                """, language="text")

        
        # ====================================================================
        # ADVANCED ANALYTICS (if advanced mode was enabled)
        # ====================================================================
        
        if was_advanced_mode:
            st.markdown("---")
            st.header("🔬 Advanced Network Analytics")
            
            # Check if organization data is available
            has_org_data = any('organization' in node for node in seen_profiles.values())
            
            if has_org_data:
                st.success("✅ **Organization Extraction Active** - Enhanced data available")
                
                # Extract organization statistics
                orgs = {}
                sectors = {}
                for node in seen_profiles.values():
                    org = node.get('organization', '')
                    sector = node.get('sector', 'Unknown')
                    
                    if org:
                        orgs[org] = orgs.get(org, 0) + 1
                    if sector:
                        sectors[sector] = sectors.get(sector, 0) + 1
                
                # Display organization breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏢 Organizations Represented")
                    
                    if orgs:
                        # Sort by count
                        sorted_orgs = sorted(orgs.items(), key=lambda x: x[1], reverse=True)
                        
                        st.metric("Unique Organizations", len(orgs))
                        
                        # Show top 10
                        st.markdown("**Top Organizations:**")
                        for org, count in sorted_orgs[:10]:
                            st.markdown(f"- **{org}**: {count} {'person' if count == 1 else 'people'}")
                        
                        if len(sorted_orgs) > 10:
                            st.caption(f"...and {len(sorted_orgs) - 10} more organizations")
                    else:
                        st.info("No organization data extracted from profiles")
                
                with col2:
                    st.subheader("🎯 Sector Distribution")
                    render_sector_analysis(sectors, len(seen_profiles), seen_profiles)
                
                # Compute sector_analysis for recommendations
                sector_analysis = analyze_sectors(sectors, len(seen_profiles)) if sectors else None
                
                # Note about what this enables
                st.markdown("---")
                st.info("""
                **🎯 What Organization Data Enables:**
                
                With organization and sector information, you can now:
                - Identify cross-sector brokers
                - Detect organizational silos
                - Find inter-organizational bridges
                - Map influence across sectors
                
                **Coming Next:** Brokerage matrix showing who connects which organizations/sectors.
                """)
            
            else:
                st.warning("""
                **⚠️ Organization Data Not Available**
                
                Organization extraction requires full profile data from EnrichLayer API responses.
                This data may not be available for:
                - Discovered nodes that weren't fully fetched
                - Profiles with incomplete data
                - Mock mode tests
                
                **Tip:** Run with real API token and degree 1 or 2 to get organization data for fetched profiles.
                """)
            
            # ================================================================
            # NETWORK CENTRALITY METRICS (Advanced Mode)
            # ================================================================
            st.markdown("---")
            st.subheader("📊 Network Centrality Metrics")
            
            if network_metrics and network_metrics.get('top_nodes'):
                top_nodes = network_metrics['top_nodes']
                network_stats = network_metrics.get('network_stats', {})
                node_metrics = network_metrics.get('node_metrics', {})
                brokerage_roles = network_metrics.get('brokerage_roles', {})
                communities = network_metrics.get('communities', {})
                
                # Extract metric values for breakpoint calculation
                degree_values = [m.get('degree_centrality', 0) for m in node_metrics.values()]
                betweenness_values = [m.get('betweenness_centrality', 0) for m in node_metrics.values()]
                closeness_values = [m.get('closeness_centrality', 0) for m in node_metrics.values()]
                eigenvector_values = [m.get('eigenvector_centrality', 0) for m in node_metrics.values()]
                
                # Compute adaptive breakpoints
                deg_bp = compute_breakpoints(degree_values) if degree_values else None
                btw_bp = compute_breakpoints(betweenness_values) if betweenness_values else None
                clo_bp = compute_breakpoints(closeness_values) if closeness_values else None
                eig_bp = compute_breakpoints(eigenvector_values) if eigenvector_values else None
                
                # Compute percentiles for narrative generation
                degree_pcts = compute_percentiles({nid: m.get('degree_centrality', 0) for nid, m in node_metrics.items()})
                betweenness_pcts = compute_percentiles({nid: m.get('betweenness_centrality', 0) for nid, m in node_metrics.items()})
                closeness_pcts = compute_percentiles({nid: m.get('closeness_centrality', 0) for nid, m in node_metrics.items()})
                eigenvector_pcts = compute_percentiles({nid: m.get('eigenvector_centrality', 0) for nid, m in node_metrics.items()})
                
                # ----- NETWORK HEALTH SCORE -----
                health_stats = NetworkStats(
                    n_nodes=network_stats.get('nodes', 0),
                    n_edges=network_stats.get('edges', 0),
                    density=network_stats.get('density', 0),
                    avg_degree=network_stats.get('avg_degree', 0),
                    avg_clustering=network_stats.get('avg_clustering', 0),
                    n_components=network_stats.get('num_components', 1),
                    largest_component_size=network_stats.get('largest_component_size', network_stats.get('nodes', 0))
                )
                
                health_score, health_label = compute_network_health(
                    health_stats,
                    degree_values=degree_values,
                    betweenness_values=betweenness_values
                )
                
                render_health_summary(health_score, health_label)
                render_health_details(health_stats, degree_values, betweenness_values)
                
                st.markdown("---")
                
                # ----- NETWORK OVERVIEW STATS -----
                st.markdown("**Network Overview:**")
                stats_cols = st.columns(5)
                stats_cols[0].metric("Nodes", network_stats.get('nodes', 0))
                stats_cols[1].metric("Edges", network_stats.get('edges', 0))
                stats_cols[2].metric("Density", f"{network_stats.get('density', 0):.4f}")
                stats_cols[3].metric("Avg Degree", network_stats.get('avg_degree', 0))
                stats_cols[4].metric("Avg Clustering", f"{network_stats.get('avg_clustering', 0):.4f}")
                
                # Additional network stats if available
                if 'num_components' in network_stats:
                    st.caption(f"📈 Components: {network_stats['num_components']} | Largest component: {network_stats.get('largest_component_size', 'N/A')} nodes")
                
                st.markdown("---")
                
                # Get sector flags for recommendations
                dominant_sectors = set(sector_analysis.dominant_sectors) if sector_analysis else set()
                underrep_sectors = set(sector_analysis.underrepresented_sectors) if sector_analysis else set()
                
                def get_sector_flags(node_sector):
                    """Check if a sector is dominant or underrepresented."""
                    if not node_sector:
                        return False, False
                    return (node_sector in dominant_sectors, node_sector in underrep_sectors)
                
                # ----- TOP NODES WITH BADGES -----
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔗 Top Connectors** (by Degree)")
                    st.caption(METRIC_TOOLTIPS["degree"])
                    if 'degree' in top_nodes and deg_bp:
                        for i, (node_id, score) in enumerate(top_nodes['degree'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            sector = seen_profiles.get(node_id, {}).get('sector', '')
                            connections = node_metrics.get(node_id, {}).get('degree', 0)
                            
                            # Get level and badge
                            level = classify_value(score, deg_bp)
                            badge = render_badge("degree", level, small=True)
                            
                            # Get brokerage role
                            broker_role = brokerage_roles.get(node_id, 'peripheral')
                            
                            # Get sector flags
                            is_dom, is_under = get_sector_flags(sector)
                            
                            st.markdown(f"{i}. **{name}** ({org}) — {connections} connections {badge}", unsafe_allow_html=True)
                            
                            # Generate rich narrative with recommendation
                            blurb, rec = describe_node_with_recommendation(
                                name=name,
                                organization=org,
                                role=broker_role,
                                degree_pct=degree_pcts.get(node_id),
                                betweenness_pct=betweenness_pcts.get(node_id),
                                eigenvector_pct=eigenvector_pcts.get(node_id),
                                closeness_pct=closeness_pcts.get(node_id),
                                sector=sector,
                                is_dominant_sector=is_dom,
                                is_underrepresented_sector=is_under,
                            )
                            st.caption(blurb)
                            if rec:
                                st.markdown(f"**💡 Suggested focus:** {rec}")
                    else:
                        st.info("No degree data available")
                    
                    st.markdown("")
                    st.markdown("**📍 Top Accessible** (by Closeness)")
                    st.caption(METRIC_TOOLTIPS["closeness"])
                    if 'closeness' in top_nodes and clo_bp:
                        for i, (node_id, score) in enumerate(top_nodes['closeness'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            
                            level = classify_value(score, clo_bp)
                            badge = render_badge("closeness", level, small=True)
                            
                            st.markdown(f"{i}. **{name}** ({org}) — {score:.4f} {badge}", unsafe_allow_html=True)
                    else:
                        st.info("No closeness data available")
                
                with col2:
                    st.markdown("**🌉 Top Brokers** (by Betweenness)")
                    st.caption(METRIC_TOOLTIPS["betweenness"])
                    if 'betweenness' in top_nodes and btw_bp:
                        # Track critical brokers (extreme betweenness)
                        critical_broker_ids = []
                        
                        for i, (node_id, score) in enumerate(top_nodes['betweenness'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            sector = seen_profiles.get(node_id, {}).get('sector', '')
                            
                            level = classify_value(score, btw_bp)
                            badge = render_badge("betweenness", level, small=True)
                            
                            # Get brokerage role badge
                            broker_role = brokerage_roles.get(node_id, 'peripheral')
                            role_badge = render_broker_badge(broker_role, small=True)
                            
                            # Track critical brokers
                            if level in ('high', 'extreme'):
                                critical_broker_ids.append(node_id)
                            
                            # Get sector flags
                            is_dom, is_under = get_sector_flags(sector)
                            
                            st.markdown(f"{i}. **{name}** ({org}) — {score:.4f} {badge} {role_badge}", unsafe_allow_html=True)
                            
                            # Generate rich narrative with recommendation
                            blurb, rec = describe_node_with_recommendation(
                                name=name,
                                organization=org,
                                role=broker_role,
                                degree_pct=degree_pcts.get(node_id),
                                betweenness_pct=betweenness_pcts.get(node_id),
                                eigenvector_pct=eigenvector_pcts.get(node_id),
                                closeness_pct=closeness_pcts.get(node_id),
                                sector=sector,
                                is_dominant_sector=is_dom,
                                is_underrepresented_sector=is_under,
                            )
                            st.caption(blurb)
                            if rec:
                                st.markdown(f"**💡 Suggested focus:** {rec}")
                    else:
                        st.info("No betweenness data available")
                    
                    st.markdown("")
                    st.markdown("**⭐ Top Influencers** (by Eigenvector)")
                    st.caption(METRIC_TOOLTIPS["eigenvector"])
                    if 'eigenvector' in top_nodes and eig_bp:
                        for i, (node_id, score) in enumerate(top_nodes['eigenvector'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            
                            level = classify_value(score, eig_bp)
                            badge = render_badge("eigenvector", level, small=True)
                            
                            st.markdown(f"{i}. **{name}** ({org}) — {score:.4f} {badge}", unsafe_allow_html=True)
                    else:
                        st.info("No eigenvector data available")
                
                # Metric definitions
                with st.expander("ℹ️ What do these metrics mean?"):
                    st.markdown("""
                    | Metric | What It Measures | Identifies |
                    |--------|------------------|------------|
                    | **Degree Centrality** | Number of direct connections | **Connectors** — well-networked individuals |
                    | **Betweenness Centrality** | How often on shortest paths between others | **Brokers** — bridge different groups |
                    | **Eigenvector Centrality** | Connected to influential people | **Influencers** — access to power |
                    | **Closeness Centrality** | Average distance to everyone | **Accessible hubs** — can reach anyone quickly |
                    
                    **Badge Levels** (based on network distribution):
                    - ⚪ **Low** — Bottom 40% of network
                    - 🔹 **Medium** — 40th-80th percentile
                    - 🟢/🟠/💫/⭐ **High** — 80th-95th percentile
                    - 🔥/🚨/🚀/👑 **Extreme** — Top 5%
                    
                    **Network Health Score** (0-100):
                    - Combines connectivity, cohesion, fragmentation, and centralization
                    - 🟢 70+ = Healthy cohesion
                    - 🟡 40-69 = Mixed signals
                    - 🔴 0-39 = Fragile / at risk
                    """)
            else:
                st.info("Network metrics require edges to calculate. Run a crawl with connections to see centrality analysis.")
            
            # ================================================================
            # BROKERAGE ROLE SUMMARY
            # ================================================================
            if brokerage_roles:
                st.markdown("---")
                st.subheader("🎭 Brokerage Roles")
                st.caption("How people connect groups in the network")
                
                # Count roles
                role_counts = {}
                for role in brokerage_roles.values():
                    role_counts[role] = role_counts.get(role, 0) + 1
                
                # Display role counts as metrics
                role_cols = st.columns(6)
                role_order = ["liaison", "gatekeeper", "representative", "coordinator", "consultant", "peripheral"]
                
                for i, role in enumerate(role_order):
                    count = role_counts.get(role, 0)
                    cfg = BROKER_BADGES.get(role, {})
                    emoji = cfg.get('emoji', '⚪')
                    label = cfg.get('label', role.title())
                    role_cols[i].metric(f"{emoji} {label}", count)
                
                # Display brokerage chart
                brokerage_chart = create_brokerage_role_chart(brokerage_roles)
                if brokerage_chart:
                    st.plotly_chart(brokerage_chart, use_container_width=True, key="brokerage_chart_display")
                
                # Role definitions
                with st.expander("📖 What do these roles mean?"):
                    st.markdown("""
| Role | Description | Strategic Value |
|------|-------------|-----------------|
| 🌉 **Liaison** | Bridges groups they don't belong to | Extremely valuable — enables cross-sector flow |
| 🚪 **Gatekeeper** | Controls access into their group | Key for sector handoffs — needs backup |
| 🔗 **Representative** | Connects their group outward | Brings sector voice to other groups |
| 🧩 **Coordinator** | Connects people within their group | Keeps internal cohesion strong |
| 🧠 **Consultant** | Advises across multiple groups | Non-hierarchical cross-group connector |
| ⚪ **Peripheral** | Edge of network | Lower structural influence |
                    """)
            
            # ================================================================
            # RECOMMENDATIONS
            # ================================================================
            if network_metrics and network_metrics.get('top_nodes'):
                st.markdown("---")
                
                # Get critical brokers (top 5 with high/extreme betweenness)
                critical_broker_ids = []
                if 'betweenness' in top_nodes and btw_bp:
                    for node_id, score in top_nodes['betweenness'][:5]:
                        level = classify_value(score, btw_bp)
                        if level in ('high', 'extreme'):
                            critical_broker_ids.append(node_id)
                
                # Get or create sector_analysis
                if 'sector_analysis' not in dir() or sector_analysis is None:
                    # Fallback: create minimal sector analysis
                    sectors_temp = {}
                    for node in seen_profiles.values():
                        sector = node.get('sector', 'Unknown')
                        if sector:
                            sectors_temp[sector] = sectors_temp.get(sector, 0) + 1
                    sector_analysis = analyze_sectors(sectors_temp, len(seen_profiles)) if sectors_temp else None
                
                render_recommendations(
                    stats=health_stats,
                    sector_analysis=sector_analysis,
                    degree_values=degree_values,
                    betweenness_values=betweenness_values,
                    health_score=health_score,
                    health_label=health_label,
                    brokerage_roles=brokerage_roles,
                    critical_brokers=critical_broker_ids,
                )
        
        # ====================================================================
        # DOWNLOAD SECTION
        # ====================================================================
        
        st.header("💾 Download Results")
        
        # Generate files
        nodes_csv = generate_nodes_csv(seen_profiles, max_degree=max_degree, max_edges=10000, max_nodes=7500, network_metrics=network_metrics)
        edges_csv = generate_edges_csv(edges, max_degree=max_degree, max_edges=10000, max_nodes=7500)
        raw_json = generate_raw_json(raw_profiles)
        
        # Generate network analysis JSON if metrics available
        analysis_json = None
        insights_report = None
        brokerage_chart_fig = None
        
        if network_metrics and advanced_mode:
            analysis_json = generate_network_analysis_json(network_metrics, seen_profiles)
            
            # Generate insights report
            try:
                # Collect health details for report
                health_details = []
                cohesion = health_stats.largest_component_size / max(health_stats.n_nodes, 1)
                if cohesion >= 0.9:
                    health_details.append(f"✅ Highly unified — {cohesion*100:.0f}% of people can reach each other")
                elif cohesion >= 0.7:
                    health_details.append(f"🟡 Moderately unified — {cohesion*100:.0f}% in main component")
                else:
                    health_details.append(f"🔴 Fragmented — only {cohesion*100:.0f}% in main component")
                
                if health_stats.n_components == 1:
                    health_details.append("✅ Fully connected — everyone can reach everyone")
                elif health_stats.n_components <= 3:
                    health_details.append(f"🟡 {health_stats.n_components} separate clusters exist")
                else:
                    health_details.append(f"🔴 {health_stats.n_components} disconnected groups")
                
                if health_stats.avg_degree >= 4:
                    health_details.append(f"✅ Strong connectivity — avg {health_stats.avg_degree:.1f} connections per person")
                elif health_stats.avg_degree >= 2:
                    health_details.append(f"🟡 Moderate connectivity — avg {health_stats.avg_degree:.1f} connections")
                else:
                    health_details.append(f"🔴 Low connectivity — avg {health_stats.avg_degree:.1f} connections")
                
                # Generate recommendations text
                recommendations_text = generate_recommendations(
                    stats=health_stats,
                    sector_analysis=sector_analysis,
                    degree_values=degree_values,
                    betweenness_values=betweenness_values,
                    health_score=health_score,
                    health_label=health_label,
                    brokerage_roles=brokerage_roles,
                    critical_brokers=critical_broker_ids if 'critical_broker_ids' in dir() else [],
                )
                
                # Get top nodes
                top_connectors = top_nodes.get('degree', [])[:10] if top_nodes else []
                top_brokers_list = top_nodes.get('betweenness', [])[:10] if top_nodes else []
                
                insights_report = generate_insights_report(
                    network_stats=network_stats,
                    health_score=health_score,
                    health_label=health_label,
                    health_details=health_details,
                    sector_analysis=sector_analysis,
                    top_connectors=top_connectors,
                    top_brokers=top_brokers_list,
                    brokerage_roles=brokerage_roles,
                    recommendations=recommendations_text,
                    seen_profiles=seen_profiles,
                    node_metrics=node_metrics,
                    degree_pcts=degree_pcts,
                    betweenness_pcts=betweenness_pcts,
                    eigenvector_pcts=eigenvector_pcts,
                    closeness_pcts=closeness_pcts,
                )
            except Exception as e:
                st.warning(f"Could not generate insights report: {e}")
            
            # Create brokerage chart for display (not export)
            if brokerage_roles:
                try:
                    brokerage_chart_fig = create_brokerage_role_chart(brokerage_roles)
                except Exception as e:
                    st.warning(f"Could not create brokerage chart: {e}")
        
        # Primary action: Download all as ZIP
        st.markdown("### 📦 Download All Files")
        zip_data = create_download_zip(
            nodes_csv, 
            edges_csv, 
            raw_json, 
            analysis_json,
            insights_report,
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            zip_contents = "nodes.csv, edges.csv, raw_profiles.json"
            if analysis_json:
                zip_contents += ", network_analysis.json"
            if insights_report:
                zip_contents += ", network_insights_report.md"
            
            st.download_button(
                label="⬇️ Download All as ZIP (Recommended)",
                data=zip_data,
                file_name="c4c_network_crawl.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                help=f"Download all files ({zip_contents}) in one ZIP file"
            )
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True, help="Clear results to start a new crawl"):
                st.session_state.crawl_results = None
                st.rerun()
        
        # Individual downloads - Data Files
        st.markdown("### 📄 Download Data Files")
        st.caption("Download data files individually")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 nodes.csv",
                data=nodes_csv,
                file_name="nodes.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_nodes"
            )
        
        with col2:
            st.download_button(
                label="📥 edges.csv",
                data=edges_csv,
                file_name="edges.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_edges"
            )
        
        with col3:
            st.download_button(
                label="📥 raw_profiles.json",
                data=raw_json,
                file_name="raw_profiles.json",
                mime="application/json",
                use_container_width=True,
                key="download_raw"
            )
        
        # Advanced Mode Downloads - Insights Report
        if advanced_mode and insights_report:
            st.markdown("### 📊 Download Insights Report")
            st.caption("Comprehensive analysis in Markdown format")
            
            st.download_button(
                label="📝 Download Insights Report (.md)",
                data=insights_report,
                file_name="network_insights_report.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_report"
            )
            st.caption("💡 Tip: Charts are visible in the app above — use screenshots if needed for presentations.")
        
        # ====================================================================
        # DATA PREVIEW
        # ====================================================================
        
        with st.expander("👀 Preview Nodes"):
            st.dataframe(pd.DataFrame([node for node in seen_profiles.values()]))
        
        with st.expander("👀 Preview Edges"):
            if len(edges) > 0:
                st.dataframe(pd.DataFrame(edges))
            else:
                st.info("No edges to display")
        
        # Preview insights report
        if insights_report:
            with st.expander("📝 Preview Insights Report"):
                st.markdown(insights_report)


if __name__ == "__main__":
    main()
