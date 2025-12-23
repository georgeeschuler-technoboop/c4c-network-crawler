"""
ActorGraph — People-centered Network Graphs

Build network graphs from LinkedIn profile data using EnrichLayer API.
Compute centrality metrics, detect communities, and generate strategic insights.

Part of the C4C Network Intelligence Platform.
"""

import sys
from pathlib import Path
# Ensure repo root is in path for c4c_utils import
sys.path.insert(0, str(Path(__file__).parent))

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

APP_VERSION = "0.5.3"

VERSION_HISTORY = [
    "FIXED v0.5.3: Moved cloud login to top of sidebar for consistency with other apps",
    "FIXED v0.5.2: Added sys.path fix for c4c_utils import on Streamlit Cloud",
    "FIXED v0.5.1: Restored CSV upload for seed profiles; added app icon; improved cloud login visibility",
    "CLOUD v0.5.0: Phase 2 - Project Store cloud integration; save bundles to Supabase Storage",
    "SCHEMA v0.4.0: CoreGraph v1 schema alignment for InsightGraph compatibility; unified bundle format with manifest.json",
    "ADDED v0.3.13: Polinode Excel export (single .xlsx with Nodes + Edges tabs); requires openpyxl",
    "HARDENED v0.3.12: Name canonicalization (Unicode NFKC, whitespace, quotes, hyphens); post-export validation for Polinode",
    "FIXED v0.3.11: Polinode edges now use display Names (not IDs); duplicate node names deduplicated (prefer seed)",
    "FIXED v0.3.10: URL parser now handles /about/, /jobs/, /people/ suffixes and properly extracts company IDs",
    "UPDATED v0.3.9: Separate C4C and Polinode schema outputs (nodes.csv + nodes_polinode.csv, edges.csv + edges_polinode.csv)",
    "FIXED v0.3.8: City/Region/Country now populated from API response and parsed from location strings",
]


# ============================================================================
# COREGRAPH SCHEMA CONSTANTS (v1)
# ============================================================================

COREGRAPH_VERSION = "c4c_coregraph_v1"
BUNDLE_VERSION = "1.0"
SOURCE_APP = "actorgraph"

# Valid node types (lowercase)
VALID_NODE_TYPES = frozenset({
    'person',        # Individual human
    'organization',  # Foundation / nonprofit
    'company',       # For-profit company
    'school',        # Educational institution
})

# Valid edge types (lowercase)
VALID_EDGE_TYPES = frozenset({
    'grant',         # Funding relationship
    'board',         # Board membership
    'employment',    # Works at
    'education',     # Attended
    'connection',    # Social/professional connection (LinkedIn)
    'affiliation',   # Generic membership
})

# ActorGraph edge type mapping
EDGE_TYPE_MAPPING = {
    'people_also_viewed': 'connection',
    'similar_companies': 'connection',
}


def namespace_id(node_id: str) -> str:
    """Namespace a node ID to prevent collisions across apps."""
    if not node_id:
        return node_id
    node_id_str = str(node_id)
    if ':' in node_id_str:
        return node_id_str
    return f"{SOURCE_APP}:{node_id_str}"


def normalize_edge_type(edge_type: str) -> str:
    """Normalize edge type to CoreGraph vocabulary."""
    if not edge_type:
        return 'connection'
    et = str(edge_type).strip().lower()
    return EDGE_TYPE_MAPPING.get(et, et)


def derive_node_type(node: dict) -> str:
    """Derive CoreGraph node_type from ActorGraph node data."""
    # Check explicit is_company flag first
    if node.get('is_company'):
        return 'company'
    # Check profile URL for /company/
    profile_url = str(node.get('profile_url', '')).lower()
    if '/company/' in profile_url:
        return 'company'
    if '/school/' in profile_url:
        return 'school'
    # Default to person
    return 'person'


# ============================================================================
# PROJECT STORE CLOUD INTEGRATION (Phase 2)
# ============================================================================

def init_project_store():
    """Initialize Project Store client for bundle storage (Phase 2)."""
    if "project_store" not in st.session_state:
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            
            # Import Project Store client
            from c4c_utils.c4c_project_store import ProjectStoreClient
            
            client = ProjectStoreClient(url, key)
            st.session_state.project_store = client
        except ImportError:
            st.session_state.project_store = None
        except Exception as e:
            st.session_state.project_store = None
    
    return st.session_state.get("project_store")


def get_project_store_authenticated():
    """Get authenticated Project Store client, or None."""
    client = st.session_state.get("project_store")
    if client and client.is_authenticated():
        return client
    return None


def render_cloud_status():
    """Render cloud connection status and login UI in sidebar."""
    st.sidebar.subheader("☁️ Cloud Storage")
    
    # Initialize Project Store
    init_project_store()
    
    client = st.session_state.get("project_store")
    
    if not client:
        st.sidebar.warning("☁️ Cloud unavailable")
        st.sidebar.caption("c4c_utils package not found")
        st.sidebar.markdown("---")
        return None
    
    if client.is_authenticated():
        user = client.get_current_user()
        
        # Get project count
        projects, _ = client.list_projects(source_app=SOURCE_APP)
        project_count = len(projects) if projects else 0
        
        # Logged in: show email + project count + logout button
        st.sidebar.success(f"✅ {user['email']}")
        st.sidebar.caption(f"📦 {project_count} cloud project(s)")
        
        if st.sidebar.button("Logout", key="cloud_logout", use_container_width=True):
            client.logout()
            st.rerun()
        st.sidebar.markdown("---")
        return client
    else:
        # Not logged in: show login form directly (not collapsed)
        st.sidebar.info("🔒 Not logged in")
        
        with st.sidebar.expander("🔑 Login / Sign Up", expanded=True):
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                email = st.text_input("Email", key="cloud_login_email")
                password = st.text_input("Password", type="password", key="cloud_login_pass")
                if st.button("Login", key="cloud_login_btn", use_container_width=True):
                    success, error = client.login(email, password)
                    if success:
                        st.success("✅ Logged in!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {error}")
            
            with tab2:
                st.caption("First time? Create an account.")
                signup_email = st.text_input("Email", key="cloud_signup_email")
                signup_pass = st.text_input("Password", type="password", key="cloud_signup_pass")
                if st.button("Sign Up", key="cloud_signup_btn", use_container_width=True):
                    success, error = client.signup(signup_email, signup_pass)
                    if success:
                        st.success("✅ Check email to confirm")
                    else:
                        st.error(f"Signup failed: {error}")
        
        st.sidebar.markdown("---")
        return client
        
        return client


def save_bundle_to_cloud(
    project_name: str,
    zip_data: bytes,
    node_count: int,
    edge_count: int,
    crawl_type: str = None
) -> tuple:
    """
    Save ActorGraph bundle to Project Store (Supabase Storage).
    
    Returns:
        Tuple of (success: bool, message: str, project_slug: str or None)
    """
    client = get_project_store_authenticated()
    
    if not client:
        return False, "Login required to save to cloud", None
    
    try:
        project, error = client.save_project(
            name=project_name,
            bundle_data=zip_data,
            source_app=SOURCE_APP,
            node_count=node_count,
            edge_count=edge_count,
            jurisdiction=None,  # LinkedIn networks aren't jurisdiction-specific
            region_preset=crawl_type,
            app_version=APP_VERSION,
            schema_version=COREGRAPH_VERSION,
            bundle_version=BUNDLE_VERSION
        )
        
        if error:
            return False, f"Upload failed: {error}", None
        
        return True, f"Saved to cloud: {project.slug}", project.slug
        
    except Exception as e:
        return False, f"Cloud save failed: {str(e)}", None


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
        fragments.append("often bridges different clusters")

    if levels.get("eigenvector") in ("high", "extreme"):
        fragments.append("connects to other influential people")

    if levels.get("closeness") in ("high", "extreme"):
        fragments.append("can reach most people quickly")

    if not fragments:
        return f"{name}{org_part} has a typical network position."

    return f"{name}{org_part} " + ", and ".join(fragments) + "."


# ============================================================================
# BROKERAGE ROLE ANALYSIS
# ============================================================================

BROKERAGE_BADGE_CONFIG: Dict[str, Dict[str, str]] = {
    "liaison":       {"emoji": "🌉", "label": "Liaison",       "color": "#D97706"},
    "gatekeeper":    {"emoji": "🚪", "label": "Gatekeeper",    "color": "#F97316"},
    "representative":{"emoji": "🔗", "label": "Representative","color": "#10B981"},
    "coordinator":   {"emoji": "🧩", "label": "Coordinator",   "color": "#3B82F6"},
    "consultant":    {"emoji": "🧠", "label": "Consultant",    "color": "#6366F1"},
    "peripheral":    {"emoji": "⚪", "label": "Peripheral",    "color": "#9CA3AF"},
}


def render_broker_badge(role: str, small: bool = False) -> str:
    """Return HTML snippet for a brokerage role badge."""
    cfg = BROKERAGE_BADGE_CONFIG.get(role, BROKERAGE_BADGE_CONFIG["peripheral"])
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


def compute_percentiles(values_dict: Dict[str, float]) -> Dict[str, int]:
    """Compute percentile rank for each key."""
    if not values_dict:
        return {}
    from scipy import stats as scipy_stats
    vals = list(values_dict.values())
    result = {}
    for k, v in values_dict.items():
        pct = scipy_stats.percentileofscore(vals, v, kind='rank')
        result[k] = int(round(pct))
    return result


def compute_brokerage_roles(G: nx.Graph, node_metrics: Dict) -> Dict[str, str]:
    """Classify nodes into brokerage roles based on community structure."""
    if G.number_of_nodes() < 3:
        return {}
    
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
    except:
        return {}
    
    roles = {}
    for node in G.nodes():
        node_community = partition.get(node)
        if node_community is None:
            roles[node] = "peripheral"
            continue
        
        neighbors = list(G.neighbors(node))
        if not neighbors:
            roles[node] = "peripheral"
            continue
        
        neighbor_communities = [partition.get(n) for n in neighbors if partition.get(n) is not None]
        if not neighbor_communities:
            roles[node] = "peripheral"
            continue
        
        same_community = sum(1 for c in neighbor_communities if c == node_community)
        diff_community = len(neighbor_communities) - same_community
        
        total = len(neighbor_communities)
        same_ratio = same_community / total if total > 0 else 0
        diff_ratio = diff_community / total if total > 0 else 0
        
        metrics = node_metrics.get(node, {})
        betweenness = metrics.get('betweenness_centrality', 0)
        
        if betweenness > 0.1 and diff_ratio > 0.5:
            roles[node] = "liaison"
        elif betweenness > 0.05 and diff_ratio > 0.3:
            roles[node] = "gatekeeper"
        elif same_ratio > 0.7 and diff_ratio > 0.1:
            roles[node] = "representative"
        elif same_ratio > 0.8:
            roles[node] = "coordinator"
        elif diff_ratio > 0.5:
            roles[node] = "consultant"
        else:
            roles[node] = "peripheral"
    
    return roles


# ============================================================================
# NETWORK HEALTH SCORING
# ============================================================================

def compute_network_health(stats: NetworkStats, degree_values: List[float], 
                          betweenness_values: List[float]) -> Tuple[float, str]:
    """Compute an overall network health score (0-100)."""
    scores = []
    
    # Connectivity score (based on density, scaled for network size)
    if stats.n_nodes > 0:
        expected_density = min(0.3, 10 / stats.n_nodes)
        density_score = min(100, (stats.density / expected_density) * 100)
        scores.append(density_score * 0.25)
    
    # Component fragmentation score
    if stats.n_nodes > 0:
        largest_ratio = stats.largest_component_size / stats.n_nodes
        component_score = largest_ratio * 100
        scores.append(component_score * 0.25)
    
    # Hub distribution score (Gini coefficient of degree)
    if degree_values and len(degree_values) > 1:
        sorted_degrees = sorted(degree_values)
        n = len(sorted_degrees)
        cumsum = np.cumsum(sorted_degrees)
        gini = (2 * sum((i + 1) * d for i, d in enumerate(sorted_degrees))) / (n * sum(sorted_degrees)) - (n + 1) / n
        gini = max(0, min(1, gini))
        hub_score = (1 - gini) * 100
        scores.append(hub_score * 0.25)
    
    # Broker concentration score
    if betweenness_values and len(betweenness_values) > 1:
        max_btw = max(betweenness_values)
        avg_btw = np.mean(betweenness_values)
        if max_btw > 0:
            concentration = avg_btw / max_btw
            broker_score = concentration * 100
        else:
            broker_score = 100
        scores.append(broker_score * 0.25)
    
    total_score = sum(scores) if scores else 50
    
    if total_score >= 80:
        label = "Excellent"
    elif total_score >= 60:
        label = "Good"
    elif total_score >= 40:
        label = "Fair"
    else:
        label = "Needs Attention"
    
    return round(total_score, 1), label


def render_health_summary(score: float, label: str):
    """Render network health score summary."""
    if score >= 80:
        color = "#10B981"
        emoji = "🟢"
    elif score >= 60:
        color = "#3B82F6"
        emoji = "🔵"
    elif score >= 40:
        color = "#F59E0B"
        emoji = "🟡"
    else:
        color = "#EF4444"
        emoji = "🔴"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}15, {color}05); 
                border: 2px solid {color}; border-radius: 12px; padding: 20px; margin: 10px 0;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="font-size: 48px;">{emoji}</div>
            <div>
                <div style="font-size: 14px; color: #6B7280; font-weight: 500;">Network Health Score</div>
                <div style="font-size: 36px; font-weight: 700; color: {color};">{score}</div>
                <div style="font-size: 16px; color: #374151;">{label}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_health_details(stats: NetworkStats, degree_values: List[float], betweenness_values: List[float]):
    """Render detailed health breakdown."""
    with st.expander("📊 Health Score Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Connectivity**")
            if stats.n_nodes > 0:
                expected = min(0.3, 10 / stats.n_nodes)
                pct = min(100, (stats.density / expected) * 100)
                st.progress(pct / 100, text=f"Density: {stats.density:.4f}")
            
            st.markdown("**Component Cohesion**")
            if stats.n_nodes > 0:
                ratio = stats.largest_component_size / stats.n_nodes
                st.progress(ratio, text=f"{ratio*100:.1f}% in largest component")
        
        with col2:
            st.markdown("**Hub Distribution**")
            if degree_values and len(degree_values) > 1:
                sorted_d = sorted(degree_values)
                n = len(sorted_d)
                gini = (2 * sum((i+1)*d for i,d in enumerate(sorted_d))) / (n * sum(sorted_d)) - (n+1)/n
                gini = max(0, min(1, gini))
                st.progress(1 - gini, text=f"Gini: {gini:.2f} (lower is better)")
            
            st.markdown("**Broker Spread**")
            if betweenness_values and max(betweenness_values) > 0:
                concentration = np.mean(betweenness_values) / max(betweenness_values)
                st.progress(concentration, text=f"Spread: {concentration:.2f}")


# ============================================================================
# SECTOR ANALYSIS
# ============================================================================

def analyze_sectors(sectors: Dict[str, int], total_nodes: int) -> Dict:
    """Analyze sector distribution and identify concentration."""
    if not sectors or total_nodes == 0:
        return {}
    
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
    top_sector, top_count = sorted_sectors[0]
    concentration = top_count / total_nodes
    
    return {
        'top_sector': top_sector,
        'top_sector_count': top_count,
        'concentration': concentration,
        'num_sectors': len(sectors),
        'is_concentrated': concentration > 0.5,
        'is_diverse': len(sectors) >= 5 and concentration < 0.3,
    }


def render_sector_analysis(sectors: Dict[str, int], total_nodes: int, seen_profiles: Dict):
    """Render sector distribution chart."""
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:10]
    
    labels = [s[0] for s in sorted_sectors]
    values = [s[1] for s in sorted_sectors]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color='#3B82F6',
        text=values, textposition='auto',
    ))
    fig.update_layout(
        title="Top Sectors",
        xaxis_title="Number of People",
        height=max(250, len(labels) * 30),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# RECOMMENDATIONS ENGINE
# ============================================================================

def render_recommendations(stats: NetworkStats, sector_analysis: Dict, 
                          degree_values: List[float], betweenness_values: List[float],
                          health_score: float, health_label: str,
                          brokerage_roles: Dict[str, str], critical_brokers: List[str]):
    """Render strategic recommendations based on network analysis."""
    st.subheader("💡 Strategic Recommendations")
    
    recommendations = []
    
    # Health-based recommendations
    if health_score < 40:
        recommendations.append({
            'priority': 'high',
            'title': 'Network Fragmentation Risk',
            'description': 'The network shows signs of fragmentation. Consider introducing connectors between isolated clusters.',
            'action': 'Identify and engage potential bridge-builders.',
        })
    
    # Broker concentration
    if betweenness_values:
        max_btw = max(betweenness_values)
        avg_btw = np.mean(betweenness_values)
        if max_btw > 0 and avg_btw / max_btw < 0.2:
            recommendations.append({
                'priority': 'medium',
                'title': 'Single Point of Failure Risk',
                'description': 'A small number of brokers control most information flow.',
                'action': 'Develop alternative pathways and backup connectors.',
            })
    
    # Sector concentration
    if sector_analysis and sector_analysis.get('is_concentrated'):
        recommendations.append({
            'priority': 'low',
            'title': 'Sector Concentration',
            'description': f"Over 50% of contacts are in {sector_analysis['top_sector']}.",
            'action': 'Consider expanding into adjacent sectors for resilience.',
        })
    
    # Component fragmentation
    if stats.n_components > 1 and stats.n_nodes > 10:
        recommendations.append({
            'priority': 'medium',
            'title': 'Disconnected Groups',
            'description': f"Network has {stats.n_components} disconnected components.",
            'action': 'Find common interests to bridge separate groups.',
        })
    
    if not recommendations:
        st.success("✅ Network appears healthy! No critical issues detected.")
        return
    
    priority_colors = {'high': '#EF4444', 'medium': '#F59E0B', 'low': '#3B82F6'}
    priority_emojis = {'high': '🔴', 'medium': '🟡', 'low': '🔵'}
    
    for rec in recommendations:
        color = priority_colors.get(rec['priority'], '#6B7280')
        emoji = priority_emojis.get(rec['priority'], '⚪')
        
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding: 12px 16px; margin: 8px 0; 
                    background: {color}10; border-radius: 0 8px 8px 0;">
            <div style="font-weight: 600; color: #111827;">{emoji} {rec['title']}</div>
            <div style="color: #4B5563; margin: 4px 0;">{rec['description']}</div>
            <div style="color: {color}; font-weight: 500;">→ {rec['action']}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# API CONFIGURATION
# ============================================================================

PER_MIN_LIMIT = 15


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, per_min_limit: int = PER_MIN_LIMIT):
        self.per_min_limit = per_min_limit
        self.call_times: deque = deque()
        self.window_seconds = 60
    
    def wait_for_slot(self):
        """Wait until a rate limit slot is available."""
        while True:
            now = time.time()
            while self.call_times and (now - self.call_times[0]) > self.window_seconds:
                self.call_times.popleft()
            
            if len(self.call_times) < self.per_min_limit:
                return
            
            oldest = self.call_times[0]
            sleep_time = self.window_seconds - (now - oldest) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def record_call(self):
        """Record that a call was made."""
        self.call_times.append(time.time())
    
    def get_status(self) -> str:
        """Get current rate limit status."""
        now = time.time()
        while self.call_times and (now - self.call_times[0]) > self.window_seconds:
            self.call_times.popleft()
        return f"{len(self.call_times)}/{self.per_min_limit} calls/min"


def call_enrichlayer_api(api_token: str, profile_url: str, crawl_type: str = "people", mock_mode: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    """Call EnrichLayer API to enrich a profile."""
    if mock_mode:
        return generate_mock_response(profile_url, crawl_type), None
    
    if crawl_type == "company":
        endpoint = "https://api.enrichlayer.com/v1/linkedin/company"
    else:
        endpoint = "https://api.enrichlayer.com/v1/linkedin/people"
    
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {"linkedin_url": profile_url}
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 429:
            return None, "Rate limit exceeded"
        elif response.status_code == 401:
            return None, "Invalid API token"
        elif response.status_code == 402:
            return None, "Out of credits"
        elif response.status_code == 404:
            return None, "Profile not found"
        elif response.status_code != 200:
            return None, f"API error: {response.status_code}"
        
        data = response.json()
        if crawl_type == "company":
            data['is_company'] = True
        return data, None
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except Exception as e:
        return None, f"Request failed: {str(e)}"


def generate_mock_response(profile_url: str, crawl_type: str = "people") -> Dict:
    """Generate mock API response for testing."""
    import random
    
    profile_id = canonical_id_from_url(profile_url)
    
    if crawl_type == "company":
        industries = ["Technology", "Healthcare", "Finance", "Education", "Manufacturing", "Retail"]
        return {
            "public_identifier": profile_id,
            "name": f"Company {profile_id[:8].title()}",
            "headline": random.choice(industries),
            "location_str": random.choice(["San Francisco, CA", "New York, NY", "Chicago, IL", "Austin, TX"]),
            "city": random.choice(["San Francisco", "New York", "Chicago", "Austin"]),
            "state": random.choice(["CA", "NY", "IL", "TX"]),
            "country": "US",
            "is_company": True,
            "website": f"https://{profile_id}.com",
            "industry": random.choice(industries),
            "company_size": [random.randint(10, 100), random.randint(100, 1000)],
            "founded_year": random.randint(1990, 2020),
            "people_also_viewed": [
                {"link": f"https://linkedin.com/company/mock-{i}", "name": f"Similar Co {i}"}
                for i in range(random.randint(2, 5))
            ],
        }
    else:
        titles = ["CEO", "CTO", "VP Engineering", "Director", "Manager", "Analyst"]
        companies = ["TechCorp", "HealthInc", "FinanceGroup", "EduStart", "ManufactureCo"]
        return {
            "public_identifier": profile_id,
            "full_name": f"Person {profile_id[:8].title()}",
            "headline": f"{random.choice(titles)} at {random.choice(companies)}",
            "location_str": random.choice(["San Francisco, CA", "New York, NY", "Chicago, IL"]),
            "city": random.choice(["San Francisco", "New York", "Chicago"]),
            "state": random.choice(["CA", "NY", "IL"]),
            "country": "US",
            "occupation": random.choice(titles),
            "experiences": [{"company": random.choice(companies), "title": random.choice(titles)}],
            "people_also_viewed": [
                {"link": f"https://linkedin.com/in/mock-person-{i}", "name": f"Mock Person {i}"}
                for i in range(random.randint(3, 8))
            ],
        }


# ============================================================================
# URL PARSING
# ============================================================================

def canonical_id_from_url(url: str) -> str:
    """Extract canonical ID from LinkedIn URL."""
    if not url:
        return ""
    
    url = str(url).strip()
    url = re.sub(r'\?.*$', '', url)
    url = re.sub(r'#.*$', '', url)
    url = url.rstrip('/')
    
    # Handle /about/, /jobs/, /people/ suffixes
    url = re.sub(r'/(about|jobs|people|posts|insights)/?$', '', url)
    
    # Extract company ID
    company_match = re.search(r'linkedin\.com/company/([^/?#]+)', url)
    if company_match:
        return company_match.group(1).lower()
    
    # Extract person ID
    person_match = re.search(r'linkedin\.com/in/([^/?#]+)', url)
    if person_match:
        return person_match.group(1).lower()
    
    # Extract school ID
    school_match = re.search(r'linkedin\.com/school/([^/?#]+)', url)
    if school_match:
        return school_match.group(1).lower()
    
    # Fallback: use last path segment
    parts = url.rstrip('/').split('/')
    return parts[-1].lower() if parts else url.lower()


def update_canonical_ids(seen_profiles: Dict, edges: List, old_id: str, new_id: str):
    """Update profile and edge IDs when canonical ID changes."""
    if old_id == new_id:
        return
    
    if old_id in seen_profiles:
        profile = seen_profiles.pop(old_id)
        profile['id'] = new_id
        seen_profiles[new_id] = profile
    
    for edge in edges:
        if edge.get('source_id') == old_id:
            edge['source_id'] = new_id
        if edge.get('target_id') == old_id:
            edge['target_id'] = new_id


def parse_location_string(location: str) -> Dict[str, str]:
    """Parse a location string into city, state, country components."""
    result = {'city': '', 'state': '', 'country': ''}
    if not location:
        return result
    
    parts = [p.strip() for p in location.split(',')]
    
    if len(parts) >= 3:
        result['city'] = parts[0]
        result['state'] = parts[1]
        result['country'] = parts[2]
    elif len(parts) == 2:
        result['city'] = parts[0]
        if len(parts[1]) == 2:
            result['state'] = parts[1]
        else:
            result['country'] = parts[1]
    elif len(parts) == 1:
        if len(parts[0]) == 2:
            result['country'] = parts[0]
        else:
            result['city'] = parts[0]
    
    return result


# ============================================================================
# ORGANIZATION EXTRACTION
# ============================================================================

def extract_organization(occupation: str, experiences: List) -> str:
    """Extract current organization from profile data."""
    if experiences:
        for exp in experiences:
            if isinstance(exp, dict):
                company = exp.get('company') or exp.get('company_name', '')
                if company:
                    return company
    
    if occupation:
        if ' at ' in occupation:
            return occupation.split(' at ')[-1].strip()
        if ' @ ' in occupation:
            return occupation.split(' @ ')[-1].strip()
    
    return ''


def extract_org_from_summary(headline: str) -> str:
    """Extract organization from headline/summary."""
    if not headline:
        return ''
    
    if ' at ' in headline:
        return headline.split(' at ')[-1].strip()
    if ' @ ' in headline:
        return headline.split(' @ ')[-1].strip()
    
    return ''


def infer_sector(organization: str, headline: str) -> str:
    """Infer sector from organization name and headline."""
    combined = f"{organization} {headline}".lower()
    
    sector_keywords = {
        'Technology': ['tech', 'software', 'ai', 'data', 'cloud', 'saas', 'startup'],
        'Finance': ['bank', 'financial', 'investment', 'capital', 'fund', 'fintech'],
        'Healthcare': ['health', 'medical', 'pharma', 'biotech', 'hospital', 'clinic'],
        'Education': ['university', 'college', 'school', 'education', 'learning'],
        'Nonprofit': ['foundation', 'nonprofit', 'charity', 'ngo', 'social impact'],
        'Government': ['government', 'federal', 'state', 'city', 'public sector'],
        'Consulting': ['consulting', 'advisory', 'strategy', 'mckinsey', 'bcg', 'bain'],
        'Media': ['media', 'news', 'publishing', 'entertainment', 'broadcast'],
    }
    
    for sector, keywords in sector_keywords.items():
        if any(kw in combined for kw in keywords):
            return sector
    
    return 'Other'


# ============================================================================
# NETWORK ANALYSIS
# ============================================================================

def compute_network_metrics(seen_profiles: Dict, edges: List) -> Dict:
    """Compute network metrics using NetworkX."""
    if not edges or len(seen_profiles) < 2:
        return {}
    
    G = nx.Graph()
    
    for node_id in seen_profiles:
        G.add_node(node_id)
    
    for edge in edges:
        source = edge.get('source_id')
        target = edge.get('target_id')
        if source and target and source in seen_profiles and target in seen_profiles:
            G.add_edge(source, target)
    
    if G.number_of_edges() == 0:
        return {}
    
    # Network-level stats
    network_stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'avg_clustering': nx.average_clustering(G),
        'num_components': nx.number_connected_components(G),
    }
    
    # Largest component
    if nx.number_connected_components(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        network_stats['largest_component_size'] = len(largest_cc)
    
    # Node-level metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = {n: 0 for n in G.nodes()}
    
    try:
        closeness_centrality = nx.closeness_centrality(G)
    except:
        closeness_centrality = {n: 0 for n in G.nodes()}
    
    node_metrics = {}
    for node_id in G.nodes():
        node_metrics[node_id] = {
            'degree': G.degree(node_id),
            'degree_centrality': degree_centrality.get(node_id, 0),
            'betweenness_centrality': betweenness_centrality.get(node_id, 0),
            'eigenvector_centrality': eigenvector_centrality.get(node_id, 0),
            'closeness_centrality': closeness_centrality.get(node_id, 0),
        }
    
    # Top nodes by each metric
    top_nodes = {
        'degree': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        'betweenness': sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        'eigenvector': sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        'closeness': sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
    }
    
    # Brokerage roles
    brokerage_roles = compute_brokerage_roles(G, node_metrics)
    
    return {
        'network_stats': network_stats,
        'node_metrics': node_metrics,
        'top_nodes': top_nodes,
        'brokerage_roles': brokerage_roles,
        'graph': G,
    }


# ============================================================================
# CRAWLER
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
    per_min_limit: int = PER_MIN_LIMIT,
    crawl_type: str = "people"
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
        'error_breakdown': {'rate_limit': 0, 'out_of_credits': 0, 'auth_error': 0,
                           'not_found': 0, 'enrichment_failed': 0, 'other': 0,
                           'consecutive_rate_limits': 0}
    }
    
    status_container.write("🌱 Initializing seed profiles...")
    for seed in seeds:
        temp_id = canonical_id_from_url(seed['profile_url'])
        node = {
            'id': temp_id, 'name': seed['name'], 'profile_url': seed['profile_url'],
            'headline': '', 'location': '', 'degree': 0, 'source_type': 'seed'
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
        response, error = call_enrichlayer_api(api_token, current_node['profile_url'], crawl_type=crawl_type, mock_mode=mock_mode)
        
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
        
        # Store company-specific fields if present
        if response.get('is_company'):
            current_node['is_company'] = True
            current_node['website'] = response.get('website', '')
            current_node['industry'] = response.get('industry', '')
            current_node['company_size'] = response.get('company_size')
            current_node['founded_year'] = response.get('founded_year')
            current_node['company_type'] = response.get('company_type', '')
        
        # Store location components if available
        if response.get('city'):
            current_node['city'] = response.get('city', '')
        if response.get('state'):
            current_node['state'] = response.get('state', '')
        if response.get('country'):
            current_node['country'] = response.get('country', '')
        
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
        
        neighbors = response.get('people_also_viewed', [])
        if not neighbors:
            stats['profiles_with_no_neighbors'] += 1
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
            
            edge_type = 'similar_companies' if crawl_type == 'company' else 'people_also_viewed'
            edges.append({'source_id': current_id, 'target_id': neighbor_id, 'edge_type': edge_type})
            stats['edges_added'] += 1
            
            if neighbor_id in seen_profiles:
                continue
            if len(seen_profiles) >= max_nodes:
                break
            
            neighbor_location = neighbor.get('location', '')
            neighbor_node = {
                'id': neighbor_id, 'name': neighbor_name, 'profile_url': neighbor_url,
                'headline': neighbor_headline, 'location': neighbor_location,
                'degree': current_node['degree'] + 1, 'source_type': 'discovered'
            }
            
            if neighbor_location:
                loc_parts = parse_location_string(neighbor_location)
                neighbor_node['city'] = loc_parts['city']
                neighbor_node['state'] = loc_parts['state']
                neighbor_node['country'] = loc_parts['country']
            
            if crawl_type == 'company':
                neighbor_node['is_company'] = True
                neighbor_node['industry'] = neighbor.get('summary') or neighbor.get('industry', '')
            
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
        stats['stopped_reason'] = 'queue_exhausted'
    
    return seen_profiles, edges, raw_profiles, stats


# ============================================================================
# CSV GENERATION - COREGRAPH SCHEMA (v1)
# ============================================================================

def generate_nodes_csv(seen_profiles: Dict, max_degree: int, max_edges: int, max_nodes: int, network_metrics: Dict = None) -> str:
    """Generate nodes.csv in CoreGraph v1 schema."""
    nodes_data = []
    node_metrics = network_metrics.get('node_metrics', {}) if network_metrics else {}
    
    for node in seen_profiles.values():
        # Derive node_type from profile data
        node_type = derive_node_type(node)
        
        # Format company size
        company_size = node.get('company_size')
        if company_size and isinstance(company_size, (list, tuple)) and len(company_size) >= 2:
            size_str = f"{company_size[0]}-{company_size[1]}" if company_size[1] else f"{company_size[0]}+"
        else:
            size_str = str(company_size) if company_size else ''
        
        node_dict = {
            # === CoreGraph Required ===
            'node_id': namespace_id(node['id']),
            'node_type': node_type,
            'label': node.get('name', ''),
            'source_app': SOURCE_APP,
            
            # === Location ===
            'city': node.get('city', ''),
            'region': node.get('state', ''),  # state → region
            'jurisdiction': node.get('country', ''),
            
            # === ActorGraph-specific ===
            'profile_url': node.get('profile_url', ''),
            'headline': node.get('headline', ''),
            'location': node.get('location', ''),
            'crawl_degree': node.get('degree', 0),
            'source_type': node.get('source_type', ''),
            'website': node.get('website', ''),
            'industry': node.get('industry', ''),
            'company_size': size_str,
            'founded_year': node.get('founded_year', ''),
        }
        
        # Add organization/sector if present (people crawls)
        if 'organization' in node:
            node_dict['organization'] = node.get('organization', '')
        if 'sector' in node:
            node_dict['sector'] = node.get('sector', '')
        
        # Add network metrics if available
        if node['id'] in node_metrics:
            metrics = node_metrics[node['id']]
            node_dict['connections'] = metrics.get('degree', 0)
            node_dict['degree_centrality'] = round(metrics.get('degree_centrality', 0), 6)
            node_dict['betweenness_centrality'] = round(metrics.get('betweenness_centrality', 0), 6)
            node_dict['eigenvector_centrality'] = round(metrics.get('eigenvector_centrality', 0), 6)
            node_dict['closeness_centrality'] = round(metrics.get('closeness_centrality', 0), 6)
        
        nodes_data.append(node_dict)
    
    df = pd.DataFrame(nodes_data)
    
    # Ensure CoreGraph columns come first
    coregraph_cols = ['node_id', 'node_type', 'label', 'source_app', 'city', 'region', 'jurisdiction']
    other_cols = [c for c in df.columns if c not in coregraph_cols]
    df = df[coregraph_cols + other_cols]
    
    return df.to_csv(index=False)


def generate_edges_csv(edges: List, max_degree: int, max_edges: int, max_nodes: int) -> str:
    """Generate edges.csv in CoreGraph v1 schema."""
    edges_data = []
    
    for i, edge in enumerate(edges):
        edge_type_raw = edge.get('edge_type', 'connection')
        edge_type = normalize_edge_type(edge_type_raw)
        
        edges_data.append({
            # === CoreGraph Required ===
            'edge_id': f"e-{i+1}",
            'from_id': namespace_id(edge['source_id']),
            'to_id': namespace_id(edge['target_id']),
            'edge_type': edge_type,
            'directed': False,  # LinkedIn connections are undirected
            'weight': 1,
            'source_app': SOURCE_APP,
            
            # === ActorGraph-specific ===
            'original_edge_type': edge_type_raw,
        })
    
    df = pd.DataFrame(edges_data)
    return df.to_csv(index=False)


# ============================================================================
# POLINODE EXPORT (unchanged from v0.3.x)
# ============================================================================

def canonicalize_name(name: str) -> str:
    """Canonicalize a display name for Polinode consistency."""
    import unicodedata
    if not name:
        return name
    name = unicodedata.normalize('NFKC', str(name))
    name = name.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    name = name.replace('–', '-').replace('—', '-').replace('−', '-')
    name = ' '.join(name.split())
    return name


def generate_nodes_polinode_csv(seen_profiles: Dict, max_degree: int, max_edges: int, max_nodes: int, network_metrics: Dict = None) -> Tuple[str, Dict[str, str]]:
    """Generate nodes_polinode.csv in Polinode-ready schema."""
    nodes_data = []
    node_metrics = network_metrics.get('node_metrics', {}) if network_metrics else {}
    
    id_to_name = {}
    name_to_node = {}
    
    for node in seen_profiles.values():
        node_id = node['id']
        display_name = canonicalize_name(node.get('name', node_id))
        
        if display_name in name_to_node:
            existing = name_to_node[display_name]
            if existing.get('source_type') == 'discovered' and node.get('source_type') == 'seed':
                name_to_node[display_name] = node
                id_to_name[node_id] = display_name
                id_to_name[existing['id']] = display_name
            else:
                id_to_name[node_id] = display_name
        else:
            name_to_node[display_name] = node
            id_to_name[node_id] = display_name
    
    for display_name, node in name_to_node.items():
        node_type = "Company" if "/company/" in str(node.get('profile_url','')).lower() else "Person"
        
        company_size = node.get('company_size')
        if company_size and isinstance(company_size, (list, tuple)) and len(company_size) >= 2:
            size_str = f"{company_size[0]}-{company_size[1]}" if company_size[1] else f"{company_size[0]}+"
        else:
            size_str = str(company_size) if company_size else ''
        
        node_dict = {
            'Name': display_name,
            'Type': node_type,
            'LinkedIn URL': node.get('profile_url', ''),
            'Headline': node.get('headline', ''),
            'Seed vs Discovered': node.get('source_type', ''),
            'City': node.get('city', ''),
            'Region': node.get('state', ''),
            'Country': node.get('country', ''),
        }
        
        if node.get('is_company') or node_type == "Company":
            node_dict['Website'] = node.get('website', '')
            node_dict['Industry'] = node.get('industry', '')
            node_dict['Company Size'] = size_str
            node_dict['Founded Year'] = node.get('founded_year', '') or ''
        
        if 'organization' in node:
            node_dict['Organization'] = node.get('organization', '')
        if 'sector' in node:
            node_dict['Sector'] = node.get('sector', '')
        
        if node['id'] in node_metrics:
            metrics = node_metrics[node['id']]
            node_dict['Connections'] = metrics.get('degree', 0)
            node_dict['Degree Centrality'] = round(metrics.get('degree_centrality', 0), 4)
            node_dict['Betweenness Centrality'] = round(metrics.get('betweenness_centrality', 0), 4)
            node_dict['Eigenvector Centrality'] = round(metrics.get('eigenvector_centrality', 0), 4)
            node_dict['Closeness Centrality'] = round(metrics.get('closeness_centrality', 0), 4)
        
        nodes_data.append(node_dict)
    
    df = pd.DataFrame(nodes_data)
    csv_body = df.to_csv(index=False)
    return csv_body, id_to_name


def generate_edges_polinode_csv(edges: List, id_to_name: Dict[str, str]) -> str:
    """Generate edges_polinode.csv in Polinode-ready schema."""
    edges_data = []
    
    for edge in edges:
        source_name = id_to_name.get(edge['source_id'])
        target_name = id_to_name.get(edge['target_id'])
        
        if not source_name or not target_name:
            continue
        
        edges_data.append({
            'Source': source_name,
            'Target': target_name,
            'Type': edge.get('edge_type', 'connection'),
        })
    
    df = pd.DataFrame(edges_data)
    return df.to_csv(index=False)


def validate_polinode_export(nodes_csv: str, edges_csv: str) -> Tuple[bool, List[str]]:
    """Validate Polinode export for consistency."""
    errors = []
    
    nodes_df = pd.read_csv(StringIO(nodes_csv))
    edges_df = pd.read_csv(StringIO(edges_csv))
    
    if 'Name' in nodes_df.columns:
        duplicates = nodes_df[nodes_df.duplicated(subset='Name', keep=False)]['Name'].unique()
        if len(duplicates) > 0:
            errors.append(f"Duplicate node names ({len(duplicates)}): {', '.join(str(d) for d in duplicates[:5])}")
    
    if 'Name' in nodes_df.columns and 'Source' in edges_df.columns:
        node_names = set(nodes_df['Name'].dropna())
        edge_sources = set(edges_df['Source'].dropna())
        edge_targets = set(edges_df['Target'].dropna())
        
        missing_sources = edge_sources - node_names
        missing_targets = edge_targets - node_names
        
        if missing_sources:
            errors.append(f"Edge sources not in nodes ({len(missing_sources)}): {', '.join(str(s) for s in list(missing_sources)[:5])}")
        if missing_targets:
            errors.append(f"Edge targets not in nodes ({len(missing_targets)}): {', '.join(str(t) for t in list(missing_targets)[:5])}")
    
    return len(errors) == 0, errors


def generate_polinode_excel(nodes_polinode_csv: str, edges_polinode_csv: str) -> bytes:
    """Generate Excel file with Nodes and Edges sheets for Polinode."""
    nodes_df = pd.read_csv(StringIO(nodes_polinode_csv))
    edges_df = pd.read_csv(StringIO(edges_polinode_csv))
    
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
        edges_df.to_excel(writer, sheet_name='Edges', index=False)
    
    excel_buffer.seek(0)
    return excel_buffer.getvalue()


# ============================================================================
# MANIFEST & BUNDLE GENERATION (CoreGraph v1)
# ============================================================================

def generate_manifest(seen_profiles: Dict, edges: List, crawl_type: str) -> str:
    """Generate manifest.json for CoreGraph bundle."""
    manifest = {
        "schema_version": COREGRAPH_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "source_app": SOURCE_APP,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(seen_profiles),
        "edge_count": len(edges),
        "crawl_type": crawl_type,
        "app_version": APP_VERSION,
    }
    return json.dumps(manifest, indent=2)


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
        'crawl_metadata': {'timestamp': datetime.now(timezone.utc).isoformat(), 'mode': mode, 'mock_mode': mock_mode, 'version': APP_VERSION},
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
            'seed_nodes': seed_count, 'discovered_nodes': len(seen_profiles) - seed_count,
            'nodes_with_organization': nodes_with_org,
        },
        'termination': {'reason': stats.get('stopped_reason', 'unknown')},
    }
    return json.dumps(log_data, indent=2)


def create_coregraph_bundle(nodes_csv: str, edges_csv: str, manifest_json: str,
                            nodes_polinode_csv: str, edges_polinode_csv: str,
                            polinode_excel: bytes,
                            raw_json: str, analysis_json: str = None,
                            crawl_log: str = None) -> bytes:
    """Create CoreGraph-compatible ZIP bundle."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # CoreGraph bundle structure (root level)
        zip_file.writestr('manifest.json', manifest_json)
        zip_file.writestr('nodes.csv', nodes_csv)
        zip_file.writestr('edges.csv', edges_csv)
        
        # Polinode subfolder
        zip_file.writestr('polinode/nodes.csv', nodes_polinode_csv)
        zip_file.writestr('polinode/edges.csv', edges_polinode_csv)
        if polinode_excel:
            zip_file.writestr('polinode/actorgraph_polinode.xlsx', polinode_excel)
        
        # Raw data and logs
        zip_file.writestr('raw_profiles.json', raw_json)
        if analysis_json:
            zip_file.writestr('network_analysis.json', analysis_json)
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
        page_icon="https://static.wixstatic.com/media/275a3f_87490929c29444a99f948f1e12cac9a8~mv2.png", 
        layout="wide"
    )
    
    st.title("🕸️ ActorGraph")
    st.markdown(f"*People-centered Network Graphs* — v{APP_VERSION}")
    
    # Sidebar
    with st.sidebar:
        # Cloud login at top (consistent with other apps)
        render_cloud_status()
        
        st.header("⚙️ Configuration")
        
        api_token = st.text_input("EnrichLayer API Token", type="password", 
                                  help="Get your token at enrichlayer.com")
        
        mock_mode = st.checkbox("🧪 Mock Mode (no API calls)", value=True,
                               help="Use fake data for testing")
        
        st.markdown("---")
        
        crawl_type = st.radio("Crawl Type", ["People", "Companies"], 
                             help="Choose what to crawl")
        crawl_type_value = "company" if crawl_type == "Companies" else "people"
        
        advanced_mode = st.checkbox("🔬 Advanced Analytics", value=True,
                                   help="Compute network metrics and insights")
        
        st.markdown("---")
        st.subheader("Crawl Limits")
        
        max_degree = st.slider("Max Degree", 1, 3, 1,
                              help="How many hops from seeds")
        max_nodes = st.slider("Max Nodes", 10, 500, 100,
                             help="Maximum profiles to collect")
        max_edges = st.slider("Max Edges", 10, 2000, 500,
                             help="Maximum connections")
        
        st.markdown("---")
        st.markdown(f"**Schema:** CoreGraph v1")
        st.markdown(f"**Bundle:** {BUNDLE_VERSION}")
    
    # Main content
    st.header("🌱 Seed Profiles")
    
    # Two input methods: Manual URLs or CSV upload
    input_method = st.radio(
        "Input method",
        ["📝 Manual URLs", "📁 Upload CSV"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    seeds = []
    
    if input_method == "📝 Manual URLs":
        seed_input = st.text_area(
            "Enter LinkedIn URLs (one per line)",
            placeholder="https://linkedin.com/in/johndoe\nhttps://linkedin.com/company/acme-corp",
            height=150
        )
        
        # Parse seeds from text input
        if seed_input:
            for line in seed_input.strip().split('\n'):
                url = line.strip()
                if url and ('linkedin.com/in/' in url or 'linkedin.com/company/' in url):
                    name = canonical_id_from_url(url).replace('-', ' ').title()
                    seeds.append({'name': name, 'profile_url': url})
    
    else:  # CSV upload
        st.caption("Upload a CSV with columns: `name`, `profile_url` (LinkedIn URL)")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV should have 'name' and 'profile_url' columns"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find the URL column (flexible naming)
                url_col = None
                for col in ['profile_url', 'linkedin_url', 'url', 'LinkedIn URL', 'linkedin']:
                    if col in df.columns:
                        url_col = col
                        break
                
                # Find the name column (flexible naming)
                name_col = None
                for col in ['name', 'Name', 'organization', 'Organization', 'company', 'Company']:
                    if col in df.columns:
                        name_col = col
                        break
                
                if url_col is None:
                    st.error("❌ CSV must have a URL column (profile_url, linkedin_url, or url)")
                else:
                    # Parse seeds from CSV
                    for _, row in df.iterrows():
                        url = str(row.get(url_col, '')).strip()
                        if url and ('linkedin.com/in/' in url or 'linkedin.com/company/' in url):
                            if name_col and pd.notna(row.get(name_col)):
                                name = str(row[name_col]).strip()
                            else:
                                name = canonical_id_from_url(url).replace('-', ' ').title()
                            seeds.append({'name': name, 'profile_url': url})
                    
                    # Show preview
                    if seeds:
                        st.success(f"✅ Loaded {len(seeds)} seed profile(s) from CSV")
                        with st.expander("Preview seeds", expanded=False):
                            preview_df = pd.DataFrame(seeds)
                            st.dataframe(preview_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No valid LinkedIn URLs found in CSV")
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    if seeds:
        st.success(f"✅ {len(seeds)} valid seed(s) detected")
    
    # Run crawler
    col1, col2 = st.columns([3, 1])
    with col1:
        run_button = st.button("🚀 Start Crawl", type="primary", use_container_width=True,
                               disabled=not seeds and not mock_mode)
    with col2:
        if st.button("🧪 Demo Data", use_container_width=True):
            seeds = [
                {'name': 'Demo Person 1', 'profile_url': 'https://linkedin.com/in/demo-person-1'},
                {'name': 'Demo Person 2', 'profile_url': 'https://linkedin.com/in/demo-person-2'},
            ]
            st.session_state.demo_seeds = seeds
            st.rerun()
    
    if 'demo_seeds' in st.session_state:
        seeds = st.session_state.demo_seeds
        del st.session_state.demo_seeds
        run_button = True
    
    if run_button and (seeds or mock_mode):
        if not seeds:
            seeds = [
                {'name': 'Mock Seed 1', 'profile_url': 'https://linkedin.com/in/mock-seed-1'},
                {'name': 'Mock Seed 2', 'profile_url': 'https://linkedin.com/in/mock-seed-2'},
            ]
        
        status_container = st.empty()
        progress_bar = st.progress(0, text="Initializing...")
        
        seen_profiles, edges, raw_profiles, stats = run_crawler(
            seeds=seeds,
            api_token=api_token,
            max_degree=max_degree,
            max_edges=max_edges,
            max_nodes=max_nodes,
            status_container=status_container,
            mock_mode=mock_mode,
            advanced_mode=advanced_mode,
            progress_bar=progress_bar,
            crawl_type=crawl_type_value,
        )
        
        progress_bar.progress(1.0, text="Complete!")
        status_container.success(f"✅ Crawl complete! {len(seen_profiles)} nodes, {len(edges)} edges")
        
        # Store results
        st.session_state.crawl_results = {
            'seen_profiles': seen_profiles,
            'edges': edges,
            'raw_profiles': raw_profiles,
            'stats': stats,
            'max_degree': max_degree,
            'max_edges': max_edges,
            'max_nodes': max_nodes,
            'advanced_mode': advanced_mode,
            'mock_mode': mock_mode,
            'crawl_type': crawl_type_value,
        }
    
    # Display results
    if 'crawl_results' in st.session_state and st.session_state.crawl_results:
        results = st.session_state.crawl_results
        seen_profiles = results['seen_profiles']
        edges = results['edges']
        raw_profiles = results['raw_profiles']
        stats = results['stats']
        was_max_degree = results['max_degree']
        was_advanced_mode = results['advanced_mode']
        was_mock_mode = results['mock_mode']
        was_crawl_type = results['crawl_type']
        
        st.markdown("---")
        st.header("📊 Results")
        
        # Compute network metrics if advanced mode
        network_metrics = {}
        if was_advanced_mode and edges:
            network_metrics = compute_network_metrics(seen_profiles, edges)
        
        # Display analytics
        if network_metrics:
            network_stats = network_metrics.get('network_stats', {})
            node_metrics = network_metrics.get('node_metrics', {})
            top_nodes = network_metrics.get('top_nodes', {})
            brokerage_roles = network_metrics.get('brokerage_roles', {})
            
            degree_values = [m.get('degree_centrality', 0) for m in node_metrics.values()]
            betweenness_values = [m.get('betweenness_centrality', 0) for m in node_metrics.values()]
            
            deg_bp = compute_breakpoints(degree_values) if degree_values else None
            btw_bp = compute_breakpoints(betweenness_values) if betweenness_values else None
            
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
            stats_cols[3].metric("Avg Degree", f"{network_stats.get('avg_degree', 0):.1f}")
            stats_cols[4].metric("Avg Clustering", f"{network_stats.get('avg_clustering', 0):.4f}")
            
            st.markdown("---")
            analytics_col1, analytics_col2 = st.columns(2)
            
            with analytics_col1:
                st.markdown("**🔗 Top Connectors** (by Degree)")
                if 'degree' in top_nodes and deg_bp:
                    for i, (node_id, score) in enumerate(top_nodes['degree'][:5], 1):
                        name = seen_profiles.get(node_id, {}).get('name', node_id)
                        org = seen_profiles.get(node_id, {}).get('organization', '')
                        connections = node_metrics.get(node_id, {}).get('degree', 0)
                        level = classify_value(score, deg_bp)
                        badge = render_badge("degree", level, small=True)
                        st.markdown(f"{i}. **{name}** ({org}) — {connections} connections {badge}", unsafe_allow_html=True)
            
            with analytics_col2:
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
        
        # Generate CoreGraph schema files
        nodes_csv = generate_nodes_csv(seen_profiles, max_degree=was_max_degree, max_edges=max_edges, max_nodes=max_nodes, network_metrics=network_metrics)
        edges_csv = generate_edges_csv(edges, max_degree=was_max_degree, max_edges=max_edges, max_nodes=max_nodes)
        manifest_json = generate_manifest(seen_profiles, edges, was_crawl_type)
        
        # Generate Polinode schema files
        nodes_polinode_csv, id_to_name = generate_nodes_polinode_csv(seen_profiles, max_degree=was_max_degree, max_edges=max_edges, max_nodes=max_nodes, network_metrics=network_metrics)
        edges_polinode_csv = generate_edges_polinode_csv(edges, id_to_name)
        
        # Validate Polinode export
        is_valid, validation_errors = validate_polinode_export(nodes_polinode_csv, edges_polinode_csv)
        if not is_valid:
            st.error("⚠️ **Polinode Export Validation Failed**")
            for err in validation_errors:
                st.warning(f"• {err}")
        
        raw_json = generate_raw_json(raw_profiles)
        
        analysis_json = None
        if network_metrics and was_advanced_mode:
            analysis_json = generate_network_analysis_json(network_metrics, seen_profiles)
        
        crawl_log = generate_crawl_log(
            stats=stats, seen_profiles=seen_profiles, edges=edges,
            max_degree=was_max_degree, max_edges=max_edges, max_nodes=max_nodes,
            api_delay=1.0, mode='Intelligence Engine' if was_advanced_mode else 'Seed Crawler',
            mock_mode=was_mock_mode
        )
        
        # Generate Excel file for Polinode
        polinode_excel = generate_polinode_excel(nodes_polinode_csv, edges_polinode_csv)
        
        # Create CoreGraph bundle
        zip_data = create_coregraph_bundle(
            nodes_csv, edges_csv, manifest_json,
            nodes_polinode_csv, edges_polinode_csv, polinode_excel,
            raw_json, analysis_json, crawl_log
        )
        
        # Primary download
        st.download_button(
            "⬇️ Download CoreGraph Bundle (.zip)", 
            data=zip_data, 
            file_name="actorgraph_bundle.zip",
            mime="application/zip", 
            type="primary", 
            use_container_width=True,
            help="CoreGraph v1 compatible bundle for InsightGraph"
        )
        
        # Cloud save and clear buttons
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            client = get_project_store_authenticated()
            cloud_enabled = client is not None
            
            # Get project name from first seed profile
            first_seed_name = list(seen_profiles.values())[0].get('name', 'Network') if seen_profiles else 'Network'
            project_name = f"{first_seed_name} Network"
            
            if st.button("☁️ Save to Cloud", 
                        disabled=not cloud_enabled,
                        use_container_width=True,
                        help="Login to enable cloud save" if not cloud_enabled else "Save bundle to Project Store"):
                
                with st.spinner("☁️ Uploading bundle..."):
                    success, message, slug = save_bundle_to_cloud(
                        project_name=project_name,
                        zip_data=zip_data,
                        node_count=len(seen_profiles),
                        edge_count=len(edges),
                        crawl_type=was_crawl_type
                    )
                    
                    if success:
                        st.success(f"☁️ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.crawl_results = None
                st.rerun()
        
        with col3:
            if not cloud_enabled:
                st.caption("☁️ Login to enable")
        
        # CoreGraph Schema Downloads
        with st.expander("📄 CoreGraph Schema Files"):
            st.markdown(f"**Schema:** `{COREGRAPH_VERSION}` | **Bundle:** `{BUNDLE_VERSION}`")
            c4c_col1, c4c_col2, c4c_col3 = st.columns(3)
            with c4c_col1:
                st.download_button("📥 nodes.csv", data=nodes_csv, file_name="nodes.csv", mime="text/csv", use_container_width=True)
            with c4c_col2:
                st.download_button("📥 edges.csv", data=edges_csv, file_name="edges.csv", mime="text/csv", use_container_width=True)
            with c4c_col3:
                st.download_button("📥 manifest.json", data=manifest_json, file_name="manifest.json", mime="application/json", use_container_width=True)
        
        # Polinode Schema Downloads
        with st.expander("🔗 Polinode Import Files"):
            st.download_button(
                "📊 Download Polinode Excel (Nodes + Edges tabs)", 
                data=polinode_excel, 
                file_name="actorgraph_polinode.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            poli_col1, poli_col2 = st.columns(2)
            with poli_col1:
                st.download_button("📥 nodes_polinode.csv", data=nodes_polinode_csv, file_name="nodes_polinode.csv", mime="text/csv", use_container_width=True)
            with poli_col2:
                st.download_button("📥 edges_polinode.csv", data=edges_polinode_csv, file_name="edges_polinode.csv", mime="text/csv", use_container_width=True)
        
        with st.expander("👀 Preview Nodes"):
            preview_df = pd.read_csv(StringIO(nodes_csv))
            st.dataframe(preview_df)
        
        with st.expander("👀 Preview Edges"):
            if edges:
                preview_df = pd.read_csv(StringIO(edges_csv))
                st.dataframe(preview_df)
            else:
                st.info("No edges to display")


if __name__ == "__main__":
    main()
