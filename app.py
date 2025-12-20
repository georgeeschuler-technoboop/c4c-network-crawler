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

APP_VERSION = "0.3.2"

# ============================================================================
# VERSION HISTORY
# ============================================================================
# UPDATED v0.3.2: Seed upload + company URL support
# - Seed CSV now accepts up to 10 rows (was 5)
# - EnrichLayer request params updated (removed invalid live_fetch value that caused 400s)
# - Added company URL handling (/company/ and /showcase/) via /api/v2/company
#
# UPDATED v0.3.1: Hotfix - syntax error recovery
# - Fixed a stray return outside function that prevented Streamlit from booting
#
# UPDATED v0.3.0: Network Intelligence Engine (advanced mode)
# - Network Health Score + breakdown
# - Brokerage roles + sector analysis + recommendations


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
    params = {"url": profile_url, "use_cache": "if-present", "fallback_to_cache": "on-error"}
    
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

def is_company_url(url: str) -> bool:
    """Heuristic: treat LinkedIn /company/ and /showcase/ URLs as companies."""
    if not url:
        return False
    u = url.lower()
    return ("/company/" in u) or ("/showcase/" in u)


def call_enrichlayer_company_api(
    api_token: str,
    company_url: str,
    mock_mode: bool = False,
    max_retries: int = 3
) -> Tuple[Optional[Dict], Optional[str]]:
    """Call EnrichLayer company endpoint with retry logic."""
    if mock_mode:
        time.sleep(0.1)
        return get_mock_company_response(company_url), None

    endpoint = "https://enrichlayer.com/api/v2/company"
    headers = {"Authorization": f"Bearer {api_token}"}
    params = {"url": company_url, "use_cache": "if-present", "fallback_to_cache": "on-error"}

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
                return None, "Rate limit exceeded"
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None, "Enrichment failed"
            else:
                # Bubble up body hint if present (helps debug 400s)
                try:
                    detail = response.text[:200]
                except Exception:
                    detail = ""
                suffix = f" ({detail})" if detail else ""
                return None, f"API error {response.status_code}{suffix}"
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
    """Generate mock company response (similar/affiliated companies)."""
    import hashlib

    url_hash = int(hashlib.md5(company_url.encode()).hexdigest(), 16)

    orgs = [
        "Great Lakes Fishery Commission", "Great Lakes Protection Fund", "The Joyce Foundation",
        "Cleveland Foundation", "Charles Stewart Mott Foundation", "Environment Funders Canada",
        "National Fish and Wildlife Foundation", "The Nature Conservancy", "WWF", "Ford Foundation",
        "Rockefeller Foundation", "MacArthur Foundation",
    ]
    industries = ["Nonprofit", "Philanthropy", "Government", "Consulting", "Technology", "Academia"]
    locations = ["Chicago, IL", "Toronto, ON", "Cleveland, OH", "Detroit, MI", "Washington, DC", "New York, NY"]

    name = orgs[url_hash % len(orgs)]
    industry = industries[(url_hash // 1000) % len(industries)]
    location = locations[(url_hash // 100000) % len(locations)]

    # Create neighbors
    n = 18 + (url_hash % 12)
    neighbors = []
    for i in range(n):
        h = (url_hash + i * 104729) % (2**32)
        n_name = orgs[h % len(orgs)]
        n_ind = industries[(h // 1000) % len(industries)]
        n_loc = locations[(h // 100000) % len(locations)]
        stub = re.sub(r"[^a-z0-9]+", "-", n_name.lower()).strip("-") + f"-{h % 1000}"
        neighbors.append({
            "link": f"https://www.linkedin.com/company/{stub}/",
            "name": n_name,
            "industry": n_ind,
            "location": n_loc,
        })

    # Split into two lists to mimic real API shapes
    half = max(1, len(neighbors)//2)
    return {
        "company_id": canonical_id_from_url(company_url),
        "name": name,
        "industry": industry,
        "location": location,
        "tagline": f"{name} — {industry}",
        "similar_companies": neighbors[:half],
        "affiliated_companies": neighbors[half:],
    }


def get_mock_response(profile_url: str) -> Dict:
    """Generate mock person response (people_also_viewed) for synthetic mode."""
    import hashlib

    url_hash = int(hashlib.md5(profile_url.encode()).hexdigest(), 16)

    first_names = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","William","Elizabeth","David","Barbara","Richard","Susan","Joseph","Jessica"]
    last_names = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas"]
    titles = ["CEO","Founder","Director","VP","Manager","Consultant","Partner","Executive Director","Chief Strategy Officer","Program Director"]
    organizations = ["World Resources Institute","The Nature Conservancy","WWF","IUCN","Conservation International","Environmental Defense Fund","Sierra Club","Ford Foundation","Rockefeller Foundation","MacArthur Foundation","Stanford University","Harvard University","MIT","McKinsey & Company"]
    locations = ["San Francisco, CA","New York, NY","Washington, DC","Boston, MA","Los Angeles, CA","Seattle, WA","Chicago, IL","Denver, CO"]

    temp_id = canonical_id_from_url(profile_url)
    first_name = first_names[url_hash % len(first_names)]
    last_name = last_names[(url_hash // 100) % len(last_names)]
    title = titles[(url_hash // 1000) % len(titles)]
    org = organizations[(url_hash // 10000) % len(organizations)]
    location = locations[(url_hash // 100000) % len(locations)]

    full_name = f"{first_name} {last_name}"
    headline = f"{title} at {org}"

    num_connections = 18 + (url_hash % 14)
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



