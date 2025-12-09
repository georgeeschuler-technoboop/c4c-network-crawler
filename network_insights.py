# network_insights.py
"""
Network Insights Module for C4C Network Crawler

Provides:
1. Breakpoints for centrality metrics (quantile-based, adaptive)
2. Badge system (emoji + color + label)
3. Explanatory tooltips
4. Node role description generator
5. Network Health Score (0-100)
6. Streamlit rendering helpers
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import streamlit as st


# ============================================================================
# 1. BREAKPOINTS & LEVELS
# ============================================================================

CENTRALITY_LEVELS = ["low", "medium", "high", "extreme"]


@dataclass
class MetricBreakpoints:
    """Quantile-based thresholds for a single centrality metric."""
    low: float      # up to this = "low"
    medium: float   # up to this = "medium"
    high: float     # up to this = "high"; above = "extreme"


def compute_breakpoints(values: Sequence[float]) -> MetricBreakpoints:
    """
    Compute adaptive thresholds for a centrality metric based on quantiles.
    40% / 80% / 95% = low / medium / high / extreme.

    This keeps interpretation stable across different network sizes.
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        # Fallback, shouldn't happen in practice
        return MetricBreakpoints(low=0.0, medium=0.0, high=0.0)

    q40, q80, q95 = np.quantile(arr, [0.40, 0.80, 0.95])
    return MetricBreakpoints(low=q40, medium=q80, high=q95)


def classify_value(value: float, bp: MetricBreakpoints) -> str:
    """
    Map a raw centrality value to a qualitative level:
    "low", "medium", "high", or "extreme".
    """
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


# ============================================================================
# 2. BADGE DESIGNS & TOOL TIPS
# ============================================================================

# Per metric, per level visual + short label.
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

# Longer tooltips (hover text / help icons) per metric.
METRIC_TOOLTIPS: Dict[str, str] = {
    "degree": (
        "Degree centrality counts how many direct connections a node has. "
        "Higher values indicate hubs that many others are directly linked to."
    ),
    "betweenness": (
        "Betweenness centrality measures how often a node sits on the shortest "
        "path between others. High values indicate brokers who bridge groups."
    ),
    "closeness": (
        "Closeness centrality captures how easily a node can reach everyone else. "
        "Higher values mean shorter average path distance across the network."
    ),
    "eigenvector": (
        "Eigenvector centrality reflects influence: being connected to other "
        "well-connected nodes. High scores indicate access to power centers."
    ),
}


def render_badge(metric: str, level: str, small: bool = False) -> str:
    """
    Return an HTML snippet for a colored badge that Streamlit can render
    via st.markdown(..., unsafe_allow_html=True).
    """
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
    """
    Return just the emoji + label for a badge (no HTML).
    Useful for simpler displays.
    """
    cfg = BADGE_CONFIG[metric][level]
    return f"{cfg['emoji']} {cfg['label']}"


# ============================================================================
# 3. NODE-LEVEL TEXT GENERATOR
# ============================================================================

def describe_node_role(
    name: str,
    organization: str,
    levels: Dict[str, str],
) -> str:
    """
    Generate a short, human-friendly description of a node's structural role
    based on its qualitative centrality levels.

    levels: {
        "degree": "high",
        "betweenness": "medium",
        "closeness": "low",
        "eigenvector": "high"
    }
    """
    org_part = f" at {organization}" if organization else ""

    # Build fragments
    fragments: List[str] = []

    # Degree
    if levels.get("degree") in ("high", "extreme"):
        fragments.append("acts as a hub for many direct relationships")
    elif levels.get("degree") == "low":
        fragments.append("sits on the edge of the network")

    # Betweenness
    if levels.get("betweenness") == "extreme":
        fragments.append("is a critical bridge between otherwise disconnected groups")
    elif levels.get("betweenness") == "high":
        fragments.append("often brokers interactions between different clusters")

    # Closeness
    if levels.get("closeness") in ("high", "extreme"):
        fragments.append("can quickly reach most other actors in the system")

    # Eigenvector
    if levels.get("eigenvector") in ("high", "extreme"):
        fragments.append("is connected to other highly influential actors")

    if not fragments:
        return (
            f"{name}{org_part} plays a more peripheral structural role in this network."
        )

    joined = "; ".join(fragments)
    return f"{name}{org_part} {joined}."


# ============================================================================
# 4. NETWORK HEALTH SCORE (0–100)
# ============================================================================

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


def centralization_index(values: Sequence[float]) -> float:
    """
    Rough centralization measure: how dominant the top node is compared
    to a uniform distribution. Returns 0–1, where 1 = fully centralized.
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size <= 1 or arr.sum() == 0:
        return 0.0

    top_share = arr.max() / arr.sum()
    n = arr.size
    min_share = 1.0 / n
    # Normalize between uniform (min_share) and fully centralized (1.0)
    denom = 1.0 - min_share
    if denom <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (top_share - min_share) / denom)))


def compute_network_health(
    stats: NetworkStats,
    degree_values: Sequence[float],
    betweenness_values: Sequence[float],
) -> Tuple[int, str]:
    """
    Compute a 0–100 health score and a qualitative label.

    Components:
    - Connectivity (avg degree)
    - Cohesion (largest component share)
    - Fragmentation (n_components)
    - Centralization penalty (degree + betweenness)
    
    Returns:
        (score, label) where label is one of:
        - "Healthy cohesion"
        - "Mixed signals"
        - "Fragile / at risk"
    """
    if stats.n_nodes == 0:
        return 0, "No data"

    # 1) Connectivity score (0–25)
    #   Target avg degree around 4–8 for collaborative networks.
    target_min, target_max = 2.0, 8.0
    deg = stats.avg_degree
    if deg <= target_min:
        connectivity = 0.0
    elif deg >= target_max:
        connectivity = 1.0
    else:
        connectivity = (deg - target_min) / (target_max - target_min)
    connectivity_score = connectivity * 25.0

    # 2) Cohesion score (0–25): share of nodes in largest component
    largest_share = stats.largest_component_size / stats.n_nodes
    cohesion_score = largest_share * 25.0

    # 3) Fragmentation penalty (0–15): many components => penalty
    if stats.n_components <= 1:
        fragmentation_penalty = 0.0
    else:
        # Cap at, say, 10 components
        frag = min(stats.n_components - 1, 9) / 9.0
        fragmentation_penalty = frag * 15.0

    # 4) Centralization penalty (0–25):
    #    high concentration of degree or betweenness reduces health.
    deg_cent = centralization_index(degree_values)
    btw_cent = centralization_index(betweenness_values)
    centralization = 0.5 * (deg_cent + btw_cent)  # simple average
    centralization_penalty = centralization * 25.0

    # Base score + contributions - penalties
    base = 30.0  # neutral baseline
    raw_score = base + connectivity_score + cohesion_score \
        - fragmentation_penalty - centralization_penalty

    score = int(max(0, min(100, round(raw_score))))

    # Label
    if score >= 70:
        label = "Healthy cohesion"
    elif score >= 40:
        label = "Mixed signals"
    else:
        label = "Fragile / at risk"

    return score, label


def render_health_summary(score: int, label: str):
    """
    Streamlit rendering helper for the network health score.
    """
    if label == "Healthy cohesion":
        color = "🟢"
    elif label == "Mixed signals":
        color = "🟡"
    else:
        color = "🔴"

    st.markdown(f"### {color} Network Health: **{score} / 100** — *{label}*")
    st.caption(
        "This score combines connectivity, cohesion, fragmentation, and how "
        "concentrated power is in a few hubs or brokers."
    )


def render_health_details(
    stats: NetworkStats,
    degree_values: Sequence[float],
    betweenness_values: Sequence[float],
):
    """
    Render detailed breakdown of health score components.
    """
    with st.expander("🔍 Health Score Breakdown"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Factors:**")
            
            # Connectivity
            if stats.avg_degree >= 4:
                st.markdown(f"- ✅ Good connectivity (avg {stats.avg_degree:.1f} connections)")
            else:
                st.markdown(f"- ⚠️ Low connectivity (avg {stats.avg_degree:.1f} connections)")
            
            # Cohesion
            largest_share = stats.largest_component_size / max(stats.n_nodes, 1) * 100
            if largest_share >= 80:
                st.markdown(f"- ✅ High cohesion ({largest_share:.0f}% in main component)")
            else:
                st.markdown(f"- ⚠️ Fragmented ({largest_share:.0f}% in main component)")
        
        with col2:
            st.markdown("**Risk Factors:**")
            
            # Components
            if stats.n_components > 3:
                st.markdown(f"- ⚠️ {stats.n_components} disconnected groups")
            else:
                st.markdown(f"- ✅ Only {stats.n_components} component(s)")
            
            # Centralization
            deg_cent = centralization_index(degree_values)
            btw_cent = centralization_index(betweenness_values)
            if deg_cent > 0.5 or btw_cent > 0.5:
                st.markdown("- ⚠️ Power concentrated in few nodes")
            else:
                st.markdown("- ✅ Power well distributed")
