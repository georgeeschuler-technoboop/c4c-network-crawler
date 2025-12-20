"""
ActorGraph — People-centered Network Graphs

Build network graphs from LinkedIn profile data using EnrichLayer API.
Compute centrality metrics, detect communities, and generate strategic insights.

Part of the C4C Network Intelligence Platform.
"""

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
from datetime import datetime, timezone
import socket
import zipfile
import networkx as nx
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go


# ============================================================================
# APP VERSION
# ============================================================================

APP_VERSION = "0.4.0"

# ============================================================================
# VERSION HISTORY (most recent first)
# ============================================================================
VERSION_HISTORY = [
    ("v0.4.0", [
        "Max 10 seed rows (people or companies); rejects mixed seed types",
        "Company crawl support (LinkedIn company URLs) + company-specific KPIs",
        "Polinode-ready company node fields in nodes.csv (keeps core crawl logic)",
        "Fix: generate_nodes_csv now accepts crawl_type without crashing",
    ]),
]


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
    """
    data = []
    total_classified = sum(sector_counts.values()) or 1

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
    dominant_sectors = df[df["pct_classified"] >= 40.0]["sector"].tolist()
    underrepresented_sectors = df[df["pct_classified"] <= 10.0]["sector"].tolist()

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
    
    analysis = analyze_sectors(sectors, total_nodes)
    df = analysis.df
    
    st.metric("Sectors Identified", len(sectors))
    
    def get_bar_color(pct):
        if pct >= 40:
            return "#F97316"
        elif pct <= 10:
            return "#9CA3AF"
        else:
            return "#10B981"
    
    df["color"] = df["pct_classified"].apply(get_bar_color)
    df_sorted = df.sort_values("pct_classified", ascending=True)
    
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
        height=max(200, len(df) * 40),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="% of Classified Actors",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, max(df["pct_classified"]) * 1.15]
        ),
        yaxis=dict(showgrid=False)
    )
    
    st.markdown("**Share of classified actors by sector:**")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 12px; margin-bottom: 10px;">
        <span><span style="color: #F97316;">●</span> Dominant (≥40%)</span>
        <span><span style="color: #10B981;">●</span> Healthy (10-40%)</span>
        <span><span style="color: #9CA3AF;">●</span> Underrepresented (≤10%)</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(analysis.summary_text)
    
    if analysis.diversity_score >= 0.75:
        st.success(f"🟢 **Sector Diversity: {analysis.diversity_label}**")
    elif analysis.diversity_score >= 0.45:
        st.warning(f"🟡 **Sector Diversity: {analysis.diversity_label}**")
    else:
        st.error(f"🔴 **Sector Diversity: {analysis.diversity_label}**")
    
    if analysis.dominant_sectors:
        st.warning(f"⚠️ **Sector Dominance** — Network shaped by **{', '.join(analysis.dominant_sectors)}**.")
    
    if analysis.underrepresented_sectors and len(analysis.underrepresented_sectors) != len(df):
        st.info(f"💡 **Underrepresented:** {', '.join(analysis.underrepresented_sectors)}")
    
    if seen_profiles:
        st.markdown("---")
        st.markdown("**👥 Explore People by Sector:**")
        
        profiles_by_sector = {}
        for node_id, profile in seen_profiles.items():
            sector = profile.get('sector', 'Unknown')
            if sector not in profiles_by_sector:
                profiles_by_sector[sector] = []
            profiles_by_sector[sector].append(profile)
        
        for _, row in df.iterrows():
            sector = row["sector"]
            count = int(row["count"])
            pct = row["pct_classified"]
            
            indicator = "🟠" if pct >= 40 else "⚪" if pct <= 10 else "🟢"
            profiles = profiles_by_sector.get(sector, [])
            
            with st.expander(f"{indicator} **{sector}** — {count} people ({pct:.1f}%)"):
                if profiles:
                    for p in profiles[:10]:
                        name = p.get('name', 'Unknown')
                        org = p.get('organization', '')
                        org_text = f" • {org}" if org else ""
                        st.markdown(f"**{name}**{org_text}")
                    if len(profiles) > 10:
                        st.caption(f"...and {len(profiles) - 10} more")


# ============================================================================
# BROKERAGE ROLES (Gould & Fernandez Classification)
# ============================================================================

def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Run Louvain community detection."""
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        return partition
    except ImportError:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition


def compute_brokerage_roles(G: nx.Graph, communities: Dict[str, int]) -> Dict[str, str]:
    """Implements simplified Gould & Fernandez brokerage classification."""
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
        neighbor_groups.discard(-1)

        same_group_ties = [n for n in neighbors if communities.get(n) == group]
        other_group_ties = [n for n in neighbors if communities.get(n) != group and n in communities]

        if len(neighbor_groups) <= 1 and group in neighbor_groups:
            roles[node] = "coordinator"
            continue

        if len(neighbor_groups) >= 2 and group not in neighbor_groups:
            roles[node] = "liaison"
            continue

        if len(other_group_ties) >= len(neighbors) * 0.5:
            roles[node] = "representative"
            continue

        if len(other_group_ties) > 0 and len(other_group_ties) <= len(neighbors) * 0.35:
            roles[node] = "gatekeeper"
            continue

        if len(neighbor_groups) > 2:
            roles[node] = "consultant"
            continue

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
    "liaison": "The rare 'true bridge' — connects groups they don't belong to.",
    "consultant": "Sits across multiple groups with non-hierarchical ties.",
    "peripheral": "Low influence, low connectivity — sits at the edge.",
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
    """Returns (blurb, recommendation) for a single node."""
    org_str = f" at {organization}" if organization else ""
    sector_str = f" in the {sector} space" if sector else ""

    deg_bucket = _bucket_from_percentile(degree_pct)
    btw_bucket = _bucket_from_percentile(betweenness_pct)
    eig_bucket = _bucket_from_percentile(eigenvector_pct)
    clo_bucket = _bucket_from_percentile(closeness_pct)

    role_key = (role or "").lower()

    if role_key == "liaison":
        role_sentence = f"{name}{org_str}{sector_str} acts as a cross-group liaison."
    elif role_key == "gatekeeper":
        role_sentence = f"{name}{org_str}{sector_str} functions as a gatekeeper."
    elif role_key == "representative":
        role_sentence = f"{name}{org_str}{sector_str} represents their group to others."
    elif role_key == "coordinator":
        role_sentence = f"{name}{org_str}{sector_str} coordinates within their group."
    elif role_key == "consultant":
        role_sentence = f"{name}{org_str}{sector_str} spans several groups as an advisor."
    elif role_key == "peripheral":
        role_sentence = f"{name}{org_str}{sector_str} sits at the edge of the network."
    else:
        role_sentence = f"{name}{org_str}{sector_str} holds a meaningful position."

    context_bits = []
    if deg_bucket in ("very high", "high"):
        context_bits.append("connected to many people")
    if btw_bucket in ("very high", "high"):
        context_bits.append("links otherwise disconnected groups")
    if eig_bucket in ("very high", "high"):
        context_bits.append("connected to influential actors")

    blurb = role_sentence
    if context_bits:
        blurb += " They " + "; ".join(context_bits) + "."

    if not include_recommendation:
        return blurb, None

    rec_bits = []
    if role_key == "liaison" and btw_bucket in ("very high", "high"):
        rec_bits.append("Protect from overload; involve early in cross-group work.")
    if role_key == "gatekeeper" and btw_bucket in ("very high", "high"):
        rec_bits.append("Ensure backup pathways exist.")
    if role_key == "coordinator" and deg_bucket in ("very high", "high"):
        rec_bits.append("Leverage as internal organizer.")
    if role_key in ("representative", "liaison") and is_underrepresented_sector:
        rec_bits.append("Key voice for underrepresented sector.")
    if (deg_bucket == "very high" or btw_bucket == "very high"):
        rec_bits.append("Monitor workload; consider sharing responsibilities.")

    if not rec_bits:
        rec_bits.append("Keep informed and connected to relevant conversations.")

    recommendation = " ".join(rec_bits)
    return blurb, recommendation


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
        name=name, organization=organization, role=role,
        degree_pct=degree_pct, betweenness_pct=betweenness_pct,
        eigenvector_pct=eigenvector_pct, closeness_pct=closeness_pct,
        sector=sector, include_recommendation=False,
    )
    return blurb


def compute_percentiles(values: Dict[str, float]) -> Dict[str, float]:
    """Convert raw metric values to percentiles (0-1)."""
    if not values:
        return {}
    sorted_values = sorted(values.values())
    n = len(sorted_values)
    percentiles = {}
    for node_id, value in values.items():
        rank = sorted_values.index(value)
        percentiles[node_id] = rank / max(n - 1, 1)
    return percentiles


def describe_broker_role(name: str, org: str, role: str, betweenness_level: str) -> str:
    """Generate a narrative description of a broker's role."""
    level_to_pct = {"low": 0.2, "medium": 0.5, "high": 0.85, "extreme": 0.97}
    btw_pct = level_to_pct.get(betweenness_level, 0.5)
    return describe_node_narrative(name=name, organization=org, role=role, betweenness_pct=btw_pct)


# ============================================================================
# RECOMMENDATIONS ENGINE
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
    """Generate rule-based recommendations based on network structure."""
    rec_sections: List[str] = []
    if brokerage_roles is None:
        brokerage_roles = {}
    if critical_brokers is None:
        critical_brokers = []

    if health_score >= 70:
        intro = "The network is structurally healthy. Focus on deepening strategic relationships."
    elif health_score >= 40:
        intro = "The network shows mixed signals. Targeted bridge-building could unlock value."
    else:
        intro = "The network appears fragile. Basic connectivity needs attention."

    rec_sections.append(f"### 🧭 How to Read This\n\n{intro}\n")

    connectivity_recs = []
    if stats.avg_degree < 3:
        connectivity_recs.append("**Increase direct connections:** Host small mixed-group sessions.")
    if stats.n_components > 1:
        connectivity_recs.append(f"**Bridge isolated groups:** {stats.n_components} disconnected clusters exist.")

    if connectivity_recs:
        rec_sections.append("### 🔗 Strengthen Basic Connections\n\n" +
                            "\n\n".join([f"- {r}" for r in connectivity_recs]) + "\n")

    sector_recs = []
    if sector_analysis and sector_analysis.dominant_sectors:
        dom = ", ".join(sector_analysis.dominant_sectors)
        sector_recs.append(f"**Rebalance who is at the table:** Network shaped by **{dom}**.")
    if sector_analysis and sector_analysis.underrepresented_sectors:
        under = ", ".join(sector_analysis.underrepresented_sectors)
        sector_recs.append(f"**Invite missing voices:** {under} are underrepresented.")

    if sector_recs:
        rec_sections.append("### 🎯 Balance Sectors\n\n" +
                            "\n\n".join([f"- {r}" for r in sector_recs]) + "\n")

    power_recs = []
    deg_cent = centralization_index(degree_values)
    btw_cent = centralization_index(betweenness_values)
    avg_cent = 0.5 * (deg_cent + btw_cent)

    if critical_brokers:
        power_recs.append("**Reduce dependence on critical brokers:** Identify backup connectors.")
    if avg_cent > 0.5:
        power_recs.append("**Distribute influence:** Rotate facilitation roles.")

    if power_recs:
        rec_sections.append("### 🧩 Work with Brokers\n\n" +
                            "\n\n".join([f"- {r}" for r in power_recs]) + "\n")

    if len(rec_sections) == 1:
        rec_sections.append("### ✨ No Structural Red Flags\n\nFocus on clarifying shared purpose.")

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
        stats=stats, sector_analysis=sector_analysis,
        degree_values=degree_values, betweenness_values=betweenness_values,
        health_score=health_score, health_label=health_label,
        brokerage_roles=brokerage_roles, critical_brokers=critical_brokers,
    )
    st.markdown(recommendations_md)


# ============================================================================
# CONFIGURATION
# ============================================================================

API_DELAY = 3.0
PER_MIN_LIMIT = 20
MAX_SEED_ROWS = 10
DEFAULT_MOCK_MODE = True


# ============================================================================
# RATE LIMITER CLASS
# ============================================================================

class RateLimiter:
    """Rate limiter that enforces a per-minute request limit."""
    def __init__(self, per_min_limit: int, buffer: float = 0.8):
        self.per_min_limit = per_min_limit
        self.allowed_per_min = max(1, int(per_min_limit * buffer))
        self.window_start = time.time()
        self.calls_in_window = 0

    def wait_for_slot(self):
        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= 60:
            self.window_start = now
            self.calls_in_window = 0
            return
        if self.calls_in_window >= self.allowed_per_min:
            sleep_for = 60 - elapsed
            time.sleep(sleep_for)
            self.window_start = time.time()
            self.calls_in_window = 0

    def record_call(self):
        self.calls_in_window += 1
    
    def get_status(self) -> str:
        return f"{self.calls_in_window}/{self.allowed_per_min} calls this minute"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_url_stub(profile_url: str) -> str:
    """Extract a temporary ID from LinkedIn URL."""
    clean_url = profile_url.rstrip('/').split('?')[0]
    match = re.search(r'/in/([^/]+)', clean_url)
    if match:
        return match.group(1)
    return clean_url.split('/')[-1]


def extract_organization(occupation: str = '', experiences: List = None) -> str:
    """Extract organization name from occupation string or experiences."""
    if occupation and ' at ' in occupation:
        org = occupation.split(' at ', 1)[1].strip()
        org = org.replace('|', '').strip()
        return org
    if experiences and len(experiences) > 0:
        recent = experiences[0]
        if 'company' in recent and recent['company']:
            return recent['company'].strip()
    return ''


def extract_org_from_summary(summary: str) -> str:
    """Extract organization name from a LinkedIn summary/headline string."""
    if not summary:
        return ''
    match = re.search(r'\bat\s+([A-Z][A-Za-z0-9\s&\-\'\.,()]+?)(?:\s*[|•,]|$)', summary)
    if match:
        org = match.group(1).strip()
        org = re.sub(r'\s+(and|for|in|of|the)\s*$', '', org, flags=re.IGNORECASE)
        org = re.sub(r'[,.]$', '', org).strip()
        if 2 < len(org) < 80:
            return org
    match = re.search(r'@\s*([A-Z][A-Za-z0-9\s&\-\']+?)(?:\s*[|•,]|$)', summary)
    if match:
        org = match.group(1).strip()
        if 2 < len(org) < 80:
            return org
    return ''


def extract_title_from_summary(summary: str) -> str:
    """Extract job title from a LinkedIn summary/headline string."""
    if not summary:
        return ''
    match = re.search(r'^([^|•\n]+?)\s+at\s+', summary)
    if match:
        title = match.group(1).strip()
        if len(title) < 80:
            return title
    match = re.search(r'^([^|•\n@]+?)\s*@\s*', summary)
    if match:
        title = match.group(1).strip()
        if len(title) < 80:
            return title
    parts = re.split(r'\s*[|•]\s*', summary)
    if parts:
        first = parts[0].strip()
        first = re.sub(r'\s+at\s+.+$', '', first, flags=re.IGNORECASE)
        if len(first) < 80:
            return first
    return ''


def infer_sector(organization: str, headline: str = '') -> str:
    """Infer sector/industry from organization name and headline."""
    combined = f"{organization} {headline}".lower()
    
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
    """Update all references to old_id with new_id after API enrichment."""
    if old_id in seen_profiles:
        node = seen_profiles[old_id]
        node['id'] = new_id
        seen_profiles[new_id] = node
        if old_id != new_id:
            del seen_profiles[old_id]
    for edge in edges:
        if edge['source_id'] == old_id:
            edge['source_id'] = new_id
        if edge['target_id'] == old_id:
            edge['target_id'] = new_id


def validate_graph(seen_profiles: Dict, edges: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Validate that all edge endpoints exist in nodes."""
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
    """Test if enrichlayer.com is reachable."""
    try:
        ip = socket.gethostbyname("enrichlayer.com")
        response = requests.get("https://enrichlayer.com/api/v2/profile", timeout=5)
        return True, f"✅ Network OK (resolved to {ip})"
    except socket.gaierror:
        return False, "❌ DNS Resolution Failed"
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection Failed"
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


# ============================================================================
# ENRICHLAYER API CLIENT
# ============================================================================

def call_enrichlayer_api(api_token: str, profile_url: str, mock_mode: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """Call EnrichLayer person profile endpoint with retry logic."""
    if mock_mode:
        time.sleep(0.1)
        return get_mock_response(profile_url), None
    
    endpoint = "https://enrichlayer.com/api/v2/profile"
    headers = {"Authorization": f"Bearer {api_token}"}
    params = {"url": profile_url, "use_cache": "if-present", "live_fetch": "if-needed"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 401:
                return None, "Invalid API token"
            elif response.status_code == 403:
                return None, "Out of credits"
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3
                    time.sleep(wait_time)
                    continue
                return None, f"Rate limit exceeded"
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None, "Enrichment failed"
            else:
                return None, f"API error {response.status_code}"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, "Request timed out"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"Network error: {str(e)}"
    
    return None, "Failed after maximum retries"


def get_mock_response(profile_url: str) -> Dict:
    """Generate comprehensive mock API response for stress testing."""
    import hashlib
    
    url_hash = int(hashlib.md5(profile_url.encode()).hexdigest(), 16)
    
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
                   "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"]
    titles = ["CEO", "Founder", "Director", "VP", "Manager", "Consultant", "Partner",
              "Executive Director", "Chief Strategy Officer", "Program Director"]
    organizations = ["World Resources Institute", "The Nature Conservancy", "WWF", "IUCN",
                     "Conservation International", "Environmental Defense Fund", "Sierra Club",
                     "Ford Foundation", "Rockefeller Foundation", "MacArthur Foundation",
                     "Stanford University", "Harvard University", "MIT", "McKinsey & Company"]
    locations = ["San Francisco, CA", "New York, NY", "Washington, DC", "Boston, MA",
                 "Los Angeles, CA", "Seattle, WA", "Chicago, IL", "Denver, CO"]
    
    temp_id = canonical_id_from_url(profile_url)
    first_name = first_names[url_hash % len(first_names)]
    last_name = last_names[(url_hash // 100) % len(last_names)]
    title = titles[(url_hash // 1000) % len(titles)]
    org = organizations[(url_hash // 10000) % len(organizations)]
    location = locations[(url_hash // 100000) % len(locations)]
    
    full_name = f"{first_name} {last_name}"
    headline = f"{title} at {org}"
    
    num_connections = 25 + (url_hash % 16)
    people_also_viewed = []
    for i in range(num_connections):
        conn_hash = (url_hash + i * 7919) % (2**32)
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
    
    return {
        "public_identifier": temp_id,
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "headline": headline,
        "occupation": headline,
        "location_str": location,
        "summary": f"Experienced {title.lower()} with broad expertise.",
        "experiences": [{"company": org, "title": title, "starts_at": {"year": 2020, "month": 1}, "ends_at": None}],
        "people_also_viewed": people_also_viewed
    }




# ============================================================================
# ENRICHLAYER COMPANY API CLIENT
# ============================================================================

def call_enrichlayer_company_api(api_token: str, company_url: str, mock_mode: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Company enrichment + neighbor discovery.
    - Seeds are LinkedIn company URLs (contain '/company/').
    """
    if mock_mode:
        time.sleep(0.1)
        return get_mock_company_response(company_url), None

    endpoint = "https://enrichlayer.com/api/v2/company"
    headers = {"Authorization": f"Bearer {api_token}"}
    params = {"url": company_url, "use_cache": "if-present", "live_fetch": "if-needed"}

    for attempt in range(max_retries):
        try:
            resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json(), None
            if resp.status_code == 401:
                return None, "Invalid API token"
            if resp.status_code == 403:
                return None, "Out of credits"
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 3)
                    continue
                return None, "Rate limit exceeded"
            if resp.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None, "Enrichment failed"
            return None, f"API error {resp.status_code}"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, "Request timed out"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"Network error: {str(e)}"

    return None, "Failed after maximum retries"


def get_mock_company_response(company_url: str) -> Dict:
    import hashlib
    url_hash = int(hashlib.md5(company_url.encode()).hexdigest(), 16)

    companies = [
        ("The Nature Conservancy", "https://www.nature.org"),
        ("World Wildlife Fund", "https://www.worldwildlife.org"),
        ("Environmental Defense Fund", "https://www.edf.org"),
        ("World Resources Institute", "https://www.wri.org"),
        ("Rockefeller Foundation", "https://www.rockefellerfoundation.org"),
        ("Ford Foundation", "https://www.fordfoundation.org"),
        ("MacArthur Foundation", "https://www.macfound.org"),
        ("C40 Cities", "https://www.c40.org"),
        ("Alliance for Water Stewardship", "https://a4ws.org"),
        ("Circle of Blue", "https://www.circleofblue.org"),
    ]
    base_name, base_site = companies[url_hash % len(companies)]
    n = 10 + (url_hash % 10)

    neighbors = []
    for i in range(n):
        h = (url_hash + i * 7919) % (2**32)
        name, site = companies[h % len(companies)]
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        neighbors.append({
            "link": f"https://www.linkedin.com/company/{slug}/",
            "name": name,
            "website": site,
            "summary": "Mock related company",
            "location": "—",
        })

    return {
        "company_name": base_name,
        "linkedin_url": company_url,
        "website": base_site,
        "industry": "Mock industry",
        "location": "—",
        "related_companies": neighbors,
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
    """Run BFS crawler on seed profiles."""
    rate_limiter = None if mock_mode else RateLimiter(per_min_limit=per_min_limit)
    
    queue = deque()
    seen_profiles = {}
    edges = []
    raw_profiles = []
    processed_nodes = 0
    
    stats = {
        'api_calls': 0, 'successful_calls': 0, 'failed_calls': 0,
        'nodes_added': 0, 'edges_added': 0, 'max_degree_reached': 0,
        'stopped_reason': None, 'profiles_with_no_neighbors': 0,
        'people_with_no_neighbors': 0,
        'companies_with_no_neighbors': 0,
        'error_breakdown': {'rate_limit': 0, 'out_of_credits': 0, 'auth_error': 0,
                           'not_found': 0, 'enrichment_failed': 0, 'other': 0,
                           'consecutive_rate_limits': 0}
    }
    
    status_container.write("🌱 Initializing seed profiles...")
    for seed in seeds:
        temp_id = canonical_id_from_url(seed['profile_url'])
        node = {
            'id': temp_id, 'name': seed['name'], 'profile_url': seed['profile_url'],
            'headline': '', 'location': '', 'degree': 0, 'source_type': 'seed',
            'crawl_type': crawl_type
        }
        seen_profiles[temp_id] = node
        queue.append(temp_id)
        stats['nodes_added'] += 1
    
    status_container.write(f"✅ Added {len(seeds)} seed profiles to queue")
    
    while queue:
        if len(edges) >= max_edges:
            stats['stopped_reason'] = 'edge_limit'
            break
        if len(seen_profiles) >= max_nodes:
            stats['stopped_reason'] = 'node_limit'
            break
        
        current_id = queue.popleft()
        current_node = seen_profiles[current_id]
        processed_nodes += 1
        
        if progress_bar is not None:
            total_known = processed_nodes + len(queue)
            if total_known > 0:
                progress_bar.progress(min(max(processed_nodes / total_known, 0.0), 0.99),
                                      text=f"Processing... {processed_nodes} done, {len(queue)} remaining")
        
        if current_node['degree'] >= max_degree:
            continue
        
        progress_text = f"🔍 Processing: {current_node['name']} (degree {current_node['degree']})"
        if rate_limiter:
            progress_text += f" | ⏱️ {rate_limiter.get_status()}"
        status_container.write(progress_text)
        
        if rate_limiter:
            rate_limiter.wait_for_slot()
        
        stats['api_calls'] += 1
        if crawl_type == 'company':
            response, error = call_enrichlayer_company_api(api_token, current_node['profile_url'], mock_mode=mock_mode)
        else:
            response, error = call_enrichlayer_api(api_token, current_node['profile_url'], mock_mode=mock_mode)
        
        if rate_limiter:
            rate_limiter.record_call()
        
        if not mock_mode:
            time.sleep(0.2)
        
        if error:
            stats['failed_calls'] += 1
            status_container.error(f"❌ Failed: {error}")
            if "Rate limit" in error:
                stats['error_breakdown']['rate_limit'] += 1
            elif "Out of credits" in error:
                stats['error_breakdown']['out_of_credits'] += 1
                stats['stopped_reason'] = 'out_of_credits'
                break
            elif "Invalid API token" in error:
                stats['error_breakdown']['auth_error'] += 1
                stats['stopped_reason'] = 'auth_error'
                break
            continue
        
        stats['successful_calls'] += 1
        raw_profiles.append(response)
        
        enriched_id = response.get('public_identifier', current_id)
        current_node['headline'] = response.get('headline', '')
        current_node['location'] = response.get('location_str') or response.get('location', '')
        
        if advanced_mode:
            occupation = response.get('occupation', '')
            experiences = response.get('experiences', [])
            organization = extract_organization(occupation, experiences)
            sector = infer_sector(organization, current_node['headline'])
            current_node['organization'] = organization
            current_node['sector'] = sector
        
        if enriched_id != current_id:
            update_canonical_ids(seen_profiles, edges, current_id, enriched_id)
            current_id = enriched_id
            current_node = seen_profiles[current_id]
        
        neighbors = (response.get('people_also_viewed')
                    or response.get('related_companies')
                    or response.get('companies_also_viewed')
                    or response.get('similar_companies')
                    or response.get('also_viewed')
                    or [])
        if not neighbors:
            stats['profiles_with_no_neighbors'] += 1
            if crawl_type == 'company':
                stats['companies_with_no_neighbors'] += 1
            else:
                stats['people_with_no_neighbors'] += 1
        else:
            status_container.write(f"   └─ Found {len(neighbors)} connections")
        
        for neighbor in neighbors:
            if len(edges) >= max_edges:
                break
            
            neighbor_url = neighbor.get('link') or neighbor.get('profile_url', '')
            neighbor_name = neighbor.get('name') or neighbor.get('full_name', '')
            neighbor_headline = neighbor.get('summary') or neighbor.get('headline', '')
            
            if not neighbor_url:
                continue
            
            neighbor_id = neighbor.get('public_identifier', canonical_id_from_url(neighbor_url))
            
            edges.append({'source_id': current_id, 'target_id': neighbor_id, 'edge_type': ('company_related' if crawl_type=='company' else 'people_also_viewed')})
            stats['edges_added'] += 1
            
            if neighbor_id in seen_profiles:
                continue
            if len(seen_profiles) >= max_nodes:
                break
            
            neighbor_node = {
                'id': neighbor_id, 'name': neighbor_name, 'profile_url': neighbor_url,
                'headline': neighbor_headline, 'location': neighbor.get('location', ''),
                'degree': current_node['degree'] + 1, 'source_type': 'discovered',
                'crawl_type': crawl_type
            }
            
            if advanced_mode:
                extracted_org = extract_org_from_summary(neighbor_headline)
                neighbor_node['organization'] = extracted_org
                neighbor_node['sector'] = infer_sector(extracted_org, neighbor_headline) if extracted_org else ''
            
            seen_profiles[neighbor_id] = neighbor_node
            stats['nodes_added'] += 1
            stats['max_degree_reached'] = max(stats['max_degree_reached'], neighbor_node['degree'])
            
            if neighbor_node['degree'] < max_degree:
                queue.append(neighbor_id)
    
    if not stats['stopped_reason']:
        stats['stopped_reason'] = 'completed'
    
    return seen_profiles, edges, raw_profiles, stats


# ============================================================================
# NETWORK METRICS CALCULATION
# ============================================================================

def calculate_network_metrics(seen_profiles: Dict, edges: List) -> Dict:
    """Calculate network centrality metrics using NetworkX."""
    G = nx.Graph()
    
    for node_id, node_data in seen_profiles.items():
        G.add_node(node_id, **node_data)
    for edge in edges:
        G.add_edge(edge['source_id'], edge['target_id'])
    
    node_metrics = {node_id: {} for node_id in seen_profiles.keys()}
    network_stats = {}
    top_nodes = {}
    
    if len(G.nodes()) < 2 or len(G.edges()) < 1:
        return {'node_metrics': node_metrics, 'network_stats': {'nodes': len(G.nodes()), 'edges': len(G.edges())},
                'top_nodes': {}, 'brokerage_roles': {}, 'communities': {}}
    
    try:
        degree_centrality = nx.degree_centrality(G)
        for node_id, value in degree_centrality.items():
            if node_id in node_metrics:
                node_metrics[node_id]['degree_centrality'] = round(value, 4)
        for node_id in G.nodes():
            if node_id in node_metrics:
                node_metrics[node_id]['degree'] = G.degree(node_id)
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['degree'] = sorted_degree
        network_stats['avg_degree'] = round(sum(dict(G.degree()).values()) / len(G.nodes()), 2)
        network_stats['max_degree'] = max(dict(G.degree()).values())
    except:
        pass
    
    try:
        betweenness = nx.betweenness_centrality(G)
        for node_id, value in betweenness.items():
            if node_id in node_metrics:
                node_metrics[node_id]['betweenness_centrality'] = round(value, 4)
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['betweenness'] = sorted_betweenness
        network_stats['avg_betweenness'] = round(sum(betweenness.values()) / len(betweenness), 4)
    except:
        pass
    
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        for node_id, value in eigenvector.items():
            if node_id in node_metrics:
                node_metrics[node_id]['eigenvector_centrality'] = round(value, 4)
        sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['eigenvector'] = sorted_eigenvector
    except:
        pass
    
    try:
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subgraph)
        for node_id, value in closeness.items():
            if node_id in node_metrics:
                node_metrics[node_id]['closeness_centrality'] = round(value, 4)
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['closeness'] = sorted_closeness
    except:
        pass
    
    communities = {}
    brokerage_roles = {}
    try:
        if len(G.nodes()) >= 3 and len(G.edges()) >= 2:
            communities = detect_communities(G)
            brokerage_roles = compute_brokerage_roles(G, communities)
            for node_id in node_metrics:
                if node_id in communities:
                    node_metrics[node_id]['community'] = communities[node_id]
                if node_id in brokerage_roles:
                    node_metrics[node_id]['brokerage_role'] = brokerage_roles[node_id]
            if communities:
                network_stats['num_communities'] = len(set(communities.values()))
    except:
        pass
    
    network_stats['nodes'] = len(G.nodes())
    network_stats['edges'] = len(G.edges())
    
    try:
        network_stats['density'] = round(nx.density(G), 4)
    except:
        pass
    
    try:
        if nx.is_connected(G):
            network_stats['diameter'] = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            network_stats['largest_component_size'] = len(largest_cc)
            network_stats['num_components'] = nx.number_connected_components(G)
    except:
        pass
    
    try:
        network_stats['avg_clustering'] = round(nx.average_clustering(G), 4)
    except:
        pass
    
    return {
        'node_metrics': node_metrics, 'network_stats': network_stats,
        'top_nodes': top_nodes, 'brokerage_roles': brokerage_roles, 'communities': communities
    }


# ============================================================================
# CSV/JSON GENERATION
# ============================================================================

def generate_nodes_csv(seen_profiles: Dict, max_degree: int, max_edges: int, max_nodes: int, network_metrics: Dict = None, crawl_type: str = "people") -> str:
    """Generate nodes.csv content."""
    nodes_data = []
    node_metrics = network_metrics.get('node_metrics', {}) if network_metrics else {}
    
    for node in seen_profiles.values():
        node_dict = {
            'id': node['id'], 'name': node['name'], 'profile_url': node['profile_url'],
            'headline': node.get('headline', ''), 'location': node.get('location', ''),
            'degree': node['degree'], 'source_type': node['source_type'],
            'label': node.get('name', ''),
            'type': 'Company' if crawl_type=='company' else 'Person',
            'linkedin_url': node.get('profile_url', ''),
        }
        if 'organization' in node:
            node_dict['organization'] = node.get('organization', '')
        if 'sector' in node:
            node_dict['sector'] = node.get('sector', '')
        if crawl_type == 'company':
            node_dict['website'] = node.get('website', '')
            node_dict['industry'] = node.get('industry', '')
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
    meta = f"# generated_at={datetime.now(timezone.utc).isoformat()}; max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}\n"
    return meta + csv_body


def generate_edges_csv(edges: List, max_degree: int, max_edges: int, max_nodes: int) -> str:
    """Generate edges.csv content."""
    df = pd.DataFrame(edges)
    csv_body = df.to_csv(index=False)
    meta = f"# generated_at={datetime.now(timezone.utc).isoformat()}; max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}\n"
    return meta + csv_body


def generate_raw_json(raw_profiles: List) -> str:
    """Generate raw_profiles.json content."""
    return json.dumps(raw_profiles, indent=2)


def generate_network_analysis_json(network_metrics: Dict, seen_profiles: Dict) -> str:
    """Generate network_analysis.json."""
    analysis = {
        'generated_at': datetime.now().isoformat(),
        'network_statistics': network_metrics.get('network_stats', {}),
        'top_connectors': [], 'top_brokers': [], 'top_influencers': []
    }
    
    top_nodes = network_metrics.get('top_nodes', {})
    
    if 'degree' in top_nodes:
        for node_id, score in top_nodes['degree']:
            if node_id in seen_profiles:
                analysis['top_connectors'].append({
                    'id': node_id, 'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'degree_centrality': score
                })
    
    if 'betweenness' in top_nodes:
        for node_id, score in top_nodes['betweenness']:
            if node_id in seen_profiles:
                analysis['top_brokers'].append({
                    'id': node_id, 'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'betweenness_centrality': score
                })
    
    return json.dumps(analysis, indent=2)


def generate_crawl_log(stats: Dict, seen_profiles: Dict, edges: List, max_degree: int,
                       max_edges: int, max_nodes: int, api_delay: float, mode: str, mock_mode: bool) -> str:
    """Generate a JSON log of the crawl session."""
    nodes_with_org = sum(1 for n in seen_profiles.values() if n.get('organization'))
    seed_count = sum(1 for n in seen_profiles.values() if n.get('source_type') == 'seed')
    
    log_data = {
        'crawl_metadata': {'timestamp': datetime.now(timezone.utc).isoformat(), 'mode': mode, 'mock_mode': mock_mode, 'version': APP_VERSION, 'crawl_type': crawl_type},
        'configuration': {'max_degree': max_degree, 'max_edges': max_edges, 'max_nodes': max_nodes, 'api_delay_seconds': api_delay},
        'api_statistics': {
            'total_calls': stats.get('api_calls', 0),
            'successful_calls': stats.get('successful_calls', 0),
            'failed_calls': stats.get('failed_calls', 0),
            'success_rate': round((stats.get('successful_calls', 0) / max(stats.get('api_calls', 1), 1)) * 100, 2),
        },
        'error_breakdown': stats.get('error_breakdown', {}),
        'network_statistics': {
            'total_nodes': len(seen_profiles), 'total_edges': len(edges),
            'seed_nodes': seed_count, 'nodes_with_organization': nodes_with_org,
        },
        'stop_reason': stats.get('stopped_reason', 'unknown'),
    }
    return json.dumps(log_data, indent=2)


def create_download_zip(nodes_csv: str, edges_csv: str, raw_json: str, analysis_json: str = None,
                        insights_report: str = None, crawl_log: str = None) -> bytes:
    """Create a ZIP file containing all output files."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('nodes.csv', nodes_csv)
        zip_file.writestr('edges.csv', edges_csv)
        zip_file.writestr('raw_profiles.json', raw_json)
        if analysis_json:
            zip_file.writestr('network_analysis.json', analysis_json)
        if insights_report:
            zip_file.writestr('network_insights_report.md', insights_report)
        if crawl_log:
            zip_file.writestr('crawl_log.json', crawl_log)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def create_brokerage_role_chart(brokerage_roles: Dict[str, str]) -> 'go.Figure':
    """Create a horizontal bar chart showing brokerage role distribution."""
    if not brokerage_roles:
        return None
    
    role_counts = {}
    for role in brokerage_roles.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    
    role_order = ["liaison", "gatekeeper", "representative", "coordinator", "consultant", "peripheral"]
    role_labels = {"liaison": "🌉 Liaison", "gatekeeper": "🚪 Gatekeeper", "representative": "🔗 Representative",
                   "coordinator": "🧩 Coordinator", "consultant": "🧠 Consultant", "peripheral": "⚪ Peripheral"}
    role_colors = {"liaison": "#D97706", "gatekeeper": "#F97316", "representative": "#10B981",
                   "coordinator": "#3B82F6", "consultant": "#6366F1", "peripheral": "#9CA3AF"}
    
    roles, counts, colors = [], [], []
    for role in role_order:
        if role in role_counts:
            roles.append(role_labels.get(role, role))
            counts.append(role_counts[role])
            colors.append(role_colors.get(role, "#9CA3AF"))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=roles, x=counts, orientation='h', marker_color=colors, text=counts, textposition='auto',
                         hovertemplate="<b>%{y}</b><br>%{x} people<extra></extra>"))
    fig.update_layout(title="Brokerage Role Distribution", xaxis_title="Number of People", yaxis_title="",
                      height=300, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(autorange="reversed"), showlegend=False)
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="ActorGraph",
        page_icon="https://static.wixstatic.com/media/275a3f_5747a8179bda42ab9b268accbdaf4ac2~mv2.png",
        layout="wide"
    )
    
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = None
    
    # Header with C4C logo
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("https://static.wixstatic.com/media/275a3f_5747a8179bda42ab9b268accbdaf4ac2~mv2.png", width=80)
    with col2:
        st.title("ActorGraph")
        st.markdown("People-centered network graphs from public profile data.")
        st.caption(f"v{APP_VERSION}")
        with st.expander("🧾 Version history"):
            for v, bullets in VERSION_HISTORY:
                if bullets:
                    st.markdown(f"**UPDATED {v}:** {bullets[0]}")
                    for b in bullets[1:]:
                        st.markdown(f"- {b}")
    
    st.markdown("---")
    
    # MODE SELECTION
    st.subheader("🎛️ Select Mode")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        advanced_mode = st.toggle("mode_toggle", value=False, label_visibility="collapsed", key="_advanced_mode")
    with col1:
        st.markdown("**📊 Seed Crawler**" if not advanced_mode else "📊 Seed Crawler")
    with col3:
        st.markdown("**🔬 Intelligence Engine**" if advanced_mode else "🔬 Intelligence Engine")
    
    if advanced_mode:
        st.info("**🔬 Network Intelligence Engine** — Full strategic analysis with centrality metrics, communities, and brokerage roles.")
    else:
        st.success("**📊 Network Seed Crawler** — Quick network mapping. Crawl, export, import to Polinode.")
    
    st.markdown("---")
    
    # INPUT SECTION
    st.header("📥 Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Upload Seed Profiles")
        uploaded_file = st.file_uploader("Upload CSV with columns: (people) name, profile_url OR (companies) org_name, linkedin_profile_url (max 10 rows)", type=['csv'])
        
        seeds = []
        crawl_type = None
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                cols = {c.strip(): c for c in df.columns}
                has_people = ('name' in cols and 'profile_url' in cols)
                has_company = ('org_name' in cols and 'linkedin_profile_url' in cols)

                if len(df) == 0:
                    st.error("❌ CSV file is empty.")
                elif len(df) > MAX_SEED_ROWS:
                    st.error(f"❌ Prototype limit: max {MAX_SEED_ROWS} seed rows.")
                elif not (has_people or has_company):
                    st.error("❌ Seed CSV must include either (name, profile_url) for people OR (org_name, linkedin_profile_url) for companies.")
                else:
                    seeds = []

                    if has_people:
                        tmp = df[[cols['name'], cols['profile_url']]].rename(columns={cols['name']: 'name', cols['profile_url']: 'profile_url'})
                        tmp = tmp.dropna(subset=['profile_url'])
                        tmp['profile_url'] = tmp['profile_url'].astype(str).str.strip()
                        tmp = tmp[tmp['profile_url'] != ""]
                        tmp = tmp[tmp['profile_url'].str.contains(r'/in/', na=False)]
                        if len(tmp) != len(df.dropna(subset=[cols['profile_url']])):
                            st.error("❌ People seed files must contain only LinkedIn profile URLs with '/in/'.")
                        else:
                            crawl_type = "people"
                            seeds = tmp.to_dict('records')

                    if has_company:
                        tmp2 = df[[cols['org_name'], cols['linkedin_profile_url']]].rename(columns={cols['org_name']: 'name', cols['linkedin_profile_url']: 'profile_url'})
                        tmp2 = tmp2.dropna(subset=['profile_url'])
                        tmp2['profile_url'] = tmp2['profile_url'].astype(str).str.strip()
                        tmp2 = tmp2[tmp2['profile_url'] != ""]
                        tmp2 = tmp2[tmp2['profile_url'].str.contains(r'/company/', na=False)]
                        if len(tmp2) != len(df.dropna(subset=[cols['linkedin_profile_url']])):
                            st.error("❌ Company seed files must contain only LinkedIn company URLs with '/company/'.")
                        else:
                            if has_people:
                                st.error("❌ This CSV contains both people and company seed columns. Please run one crawl type at a time (no mixed seed files).")
                            else:
                                crawl_type = "company"
                                seeds = tmp2.to_dict('records')

                    if seeds:
                        st.success(f"✅ Loaded {len(seeds)} seed rows ({crawl_type})")
                        st.dataframe(pd.DataFrame(seeds))

                    st.success(f"✅ Loaded {len(seeds)} seed profiles")
                    st.dataframe(df)
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with col2:
        st.subheader("2. EnrichLayer API Token")
        
        default_token = ""
        try:
            default_token = st.secrets.get("ENRICHLAYER_TOKEN", "")
        except:
            pass
        
        api_token = st.text_input("Enter your API token", type="password", value=default_token)
        
        if st.button("🔍 Test API Connection"):
            with st.spinner("Testing connection..."):
                success, message = test_network_connectivity()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        mock_mode = st.toggle("Run in mock mode (no real API calls)", value=DEFAULT_MOCK_MODE)
        
        if mock_mode:
            st.info("🧪 **MOCK MODE** - No real API calls, no credits used!")
    
    # CONFIGURATION
    st.header("⚙️ Crawl Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_degree = st.radio("Maximum Degree (hops)", options=[1, 2], index=0)
        if max_degree == 2:
            st.error("**⚠️ Degree 2 Warning** — 10-50x more API calls. Start with Degree 1!")
        else:
            st.success("**✅ Degree 1 Selected** — Direct connections only. Fast and reliable.")
    
    with col2:
        st.markdown("**Crawl Limits:**")
        st.metric("Max Edges", 10000)
        st.metric("Max Nodes", 7500)
    
    st.caption(f"⏱️ API pacing: up to **{PER_MIN_LIMIT} requests/minute**")
    
    # RUN BUTTON
    can_run = len(seeds) > 0 and (api_token or mock_mode) and (crawl_type in ('people','company'))
    
    if not can_run:
        if len(seeds) == 0:
            st.warning("⚠️ Please upload a valid seed CSV to continue.")
        elif not api_token and not mock_mode:
            st.warning("⚠️ Please enter your EnrichLayer API token to continue.")
    
    run_button = st.button("🚀 Run Crawl", disabled=not can_run, type="primary", use_container_width=True)
    
    # CRAWL EXECUTION
    if run_button:
        st.header("🔄 Crawl Progress")
        progress_bar = st.progress(0.0, text="Starting crawl...")
        status_container = st.status("Running crawl...", expanded=True)
        
        seen_profiles, edges, raw_profiles, stats = run_crawler(
            seeds=seeds, api_token=api_token, max_degree=max_degree, max_edges=10000, max_nodes=7500,
            status_container=status_container, mock_mode=mock_mode, advanced_mode=advanced_mode,
            progress_bar=progress_bar, per_min_limit=PER_MIN_LIMIT,
            crawl_type=crawl_type
        )
        
        progress_bar.progress(1.0, text="✅ Complete!")
        status_container.update(label="✅ Crawl Complete!", state="complete")
        
        orphan_ids, valid_edges = validate_graph(seen_profiles, edges)
        if orphan_ids:
            st.warning(f"⚠️ Detected {len(orphan_ids)} orphan node IDs. Excluded from download.")
            edges = valid_edges
        
        network_metrics = None
        if advanced_mode and len(edges) > 0:
            with st.spinner("📊 Calculating network metrics..."):
                network_metrics = calculate_network_metrics(seen_profiles, edges)
        
        st.session_state.crawl_results = {
            'seen_profiles': seen_profiles, 'edges': edges, 'raw_profiles': raw_profiles,
            'stats': stats, 'max_degree': max_degree, 'advanced_mode': advanced_mode,
            'mock_mode': mock_mode, 'network_metrics': network_metrics,
            'crawl_type': crawl_type
        }
    
    # DISPLAY RESULTS
    if st.session_state.crawl_results is not None:
        results = st.session_state.crawl_results
        seen_profiles = results['seen_profiles']
        edges = results['edges']
        raw_profiles = results['raw_profiles']
        stats = results['stats']
        was_max_degree = results['max_degree']
        was_advanced_mode = results.get('advanced_mode', False)
        was_mock_mode = results.get('mock_mode', False)
        was_crawl_type = results.get('crawl_type', 'people')
        network_metrics = results.get('network_metrics', None)
        
        st.header("📊 Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", len(seen_profiles))
        col2.metric("Total Edges", len(edges))
        col3.metric("Max Degree", stats['max_degree_reached'])
        col4.metric("API Calls", stats['api_calls'])
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Successful", stats['successful_calls'])
        col6.metric("Failed", stats['failed_calls'])
        col7.metric("No Neighbors", stats['companies_with_no_neighbors'] if was_crawl_type=='company' else stats['people_with_no_neighbors'])
        
        if stats['stopped_reason'] == 'completed':
            col8.success("✅ Completed")
        elif stats['stopped_reason'] in ('edge_limit', 'node_limit'):
            col8.warning(f"⚠️ {stats['stopped_reason'].replace('_', ' ').title()}")
        else:
            col8.error(f"❌ {stats['stopped_reason']}")
        
        # ADVANCED ANALYTICS
        if was_advanced_mode and network_metrics and network_metrics.get('top_nodes'):
            st.markdown("---")
            st.header("🔬 Network Intelligence")
            
            top_nodes = network_metrics['top_nodes']
            network_stats = network_metrics.get('network_stats', {})
            node_metrics = network_metrics.get('node_metrics', {})
            brokerage_roles = network_metrics.get('brokerage_roles', {})
            
            degree_values = [m.get('degree_centrality', 0) for m in node_metrics.values()]
            betweenness_values = [m.get('betweenness_centrality', 0) for m in node_metrics.values()]
            
            deg_bp = compute_breakpoints(degree_values) if degree_values else None
            btw_bp = compute_breakpoints(betweenness_values) if betweenness_values else None
            
            degree_pcts = compute_percentiles({nid: m.get('degree_centrality', 0) for nid, m in node_metrics.items()})
            betweenness_pcts = compute_percentiles({nid: m.get('betweenness_centrality', 0) for nid, m in node_metrics.items()})
            
            health_stats = NetworkStats(
                n_nodes=network_stats.get('nodes', 0), n_edges=network_stats.get('edges', 0),
                density=network_stats.get('density', 0), avg_degree=network_stats.get('avg_degree', 0),
                avg_clustering=network_stats.get('avg_clustering', 0), n_components=network_stats.get('num_components', 1),
                largest_component_size=network_stats.get('largest_component_size', network_stats.get('nodes', 0))
            )
            
            health_score, health_label = compute_network_health(health_stats, degree_values, betweenness_values)
            render_health_summary(health_score, health_label)
            render_health_details(health_stats, degree_values, betweenness_values)
            
            st.markdown("---")
            st.markdown("**Network Overview:**")
            stats_cols = st.columns(5)
            stats_cols[0].metric("Nodes", network_stats.get('nodes', 0))
            stats_cols[1].metric("Edges", network_stats.get('edges', 0))
            stats_cols[2].metric("Density", f"{network_stats.get('density', 0):.4f}")
            stats_cols[3].metric("Avg Degree", network_stats.get('avg_degree', 0))
            stats_cols[4].metric("Avg Clustering", f"{network_stats.get('avg_clustering', 0):.4f}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔗 Top Connectors** (by Degree)")
                if 'degree' in top_nodes and deg_bp:
                    for i, (node_id, score) in enumerate(top_nodes['degree'][:5], 1):
                        name = seen_profiles.get(node_id, {}).get('name', node_id)
                        org = seen_profiles.get(node_id, {}).get('organization', '')
                        connections = node_metrics.get(node_id, {}).get('degree', 0)
                        level = classify_value(score, deg_bp)
                        badge = render_badge("degree", level, small=True)
                        st.markdown(f"{i}. **{name}** ({org}) — {connections} connections {badge}", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🌉 Top Brokers** (by Betweenness)")
                if 'betweenness' in top_nodes and btw_bp:
                    for i, (node_id, score) in enumerate(top_nodes['betweenness'][:5], 1):
                        name = seen_profiles.get(node_id, {}).get('name', node_id)
                        org = seen_profiles.get(node_id, {}).get('organization', '')
                        level = classify_value(score, btw_bp)
                        badge = render_badge("betweenness", level, small=True)
                        broker_role = brokerage_roles.get(node_id, 'peripheral')
                        role_badge = render_broker_badge(broker_role, small=True)
                        st.markdown(f"{i}. **{name}** ({org}) — {score:.4f} {badge} {role_badge}", unsafe_allow_html=True)
            
            if brokerage_roles:
                st.markdown("---")
                st.subheader("🎭 Brokerage Roles")
                brokerage_chart = create_brokerage_role_chart(brokerage_roles)
                if brokerage_chart:
                    st.plotly_chart(brokerage_chart, use_container_width=True)
            
            # Sector analysis
            sectors = {}
            for node in seen_profiles.values():
                sector = node.get('sector', 'Unknown')
                if sector:
                    sectors[sector] = sectors.get(sector, 0) + 1
            
            if sectors:
                st.markdown("---")
                st.subheader("🎯 Sector Distribution")
                render_sector_analysis(sectors, len(seen_profiles), seen_profiles)
                sector_analysis = analyze_sectors(sectors, len(seen_profiles))
            else:
                sector_analysis = None
            
            st.markdown("---")
            render_recommendations(
                stats=health_stats, sector_analysis=sector_analysis,
                degree_values=degree_values, betweenness_values=betweenness_values,
                health_score=health_score, health_label=health_label,
                brokerage_roles=brokerage_roles, critical_brokers=[]
            )
        
        # DOWNLOAD SECTION
        st.header("💾 Download Results")
        
        nodes_csv = generate_nodes_csv(seen_profiles, max_degree=was_max_degree, max_edges=10000, max_nodes=7500, network_metrics=network_metrics, crawl_type=was_crawl_type)
        edges_csv = generate_edges_csv(edges, max_degree=was_max_degree, max_edges=10000, max_nodes=7500)
        raw_json = generate_raw_json(raw_profiles)
        
        analysis_json = None
        if network_metrics and was_advanced_mode:
            analysis_json = generate_network_analysis_json(network_metrics, seen_profiles)
        
        crawl_log = generate_crawl_log(
            stats=stats, seen_profiles=seen_profiles, edges=edges,
            max_degree=was_max_degree, max_edges=10000, max_nodes=7500,
            api_delay=1.0, mode='Intelligence Engine' if was_advanced_mode else 'Seed Crawler',
            mock_mode=was_mock_mode,
            crawl_type=was_crawl_type
        )
        
        zip_data = create_download_zip(nodes_csv, edges_csv, raw_json, analysis_json, None, crawl_log)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.download_button("⬇️ Download All as ZIP", data=zip_data, file_name="actorgraph_network.zip",
                              mime="application/zip", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.crawl_results = None
                st.rerun()
        
        st.markdown("### 📄 Individual Files")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button("📥 nodes.csv", data=nodes_csv, file_name="nodes.csv", mime="text/csv", use_container_width=True)
        with col2:
            st.download_button("📥 edges.csv", data=edges_csv, file_name="edges.csv", mime="text/csv", use_container_width=True)
        with col3:
            st.download_button("📥 raw_profiles.json", data=raw_json, file_name="raw_profiles.json", mime="application/json", use_container_width=True)
        with col4:
            st.download_button("📥 crawl_log.json", data=crawl_log, file_name="crawl_log.json", mime="application/json", use_container_width=True)
        
        with st.expander("👀 Preview Nodes"):
            st.dataframe(pd.DataFrame([node for node in seen_profiles.values()]))
        
        with st.expander("👀 Preview Edges"):
            if len(edges) > 0:
                st.dataframe(pd.DataFrame(edges))
            else:
                st.info("No edges to display")


if __name__ == "__main__":
    main()
