# ActorGraph — People & Company Network Graphs
#
# Build network graphs from LinkedIn profile or company data using EnrichLayer API.
# Compute centrality metrics, detect communities, and generate strategic insights.
#
# Part of the C4C Network Intelligence Platform.
#
# ---------------------------------------------------------------------------
# VERSION HISTORY
# ---------------------------------------------------------------------------
# UPDATED v0.4.3: Download/export + seed upload hardening
# - Restored max 10 seed-row constraint (no mixed person+company seed files)
# - Fixed download crash (generate_nodes_csv now accepts crawl_type)
# - Cleaner Polinode-ready node fields for company crawls (label, url, industry, website)
# - Crawl KPIs adapt to crawl type (people vs companies)
# ---------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import json
import time
from io import BytesIO
from collections import deque
from typing import Dict, List, Tuple, Optional, Sequence
import re
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
APP_VERSION = "0.4.3"


# ============================================================================
# NETWORK INSIGHTS (Badges, Health Score, Breakpoints)
# ============================================================================

CENTRALITY_LEVELS = ["low", "medium", "high", "extreme"]


@dataclass
class MetricBreakpoints:
    low: float
    medium: float
    high: float


@dataclass
class NetworkStats:
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    avg_clustering: float
    n_components: int
    largest_component_size: int


def compute_breakpoints(values: Sequence[float]) -> MetricBreakpoints:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return MetricBreakpoints(low=0.0, medium=0.0, high=0.0)
    q40, q80, q95 = np.quantile(arr, [0.40, 0.80, 0.95])
    return MetricBreakpoints(low=float(q40), medium=float(q80), high=float(q95))


def classify_value(value: float, bp: MetricBreakpoints) -> str:
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


BADGE_CONFIG: Dict[str, Dict[str, Dict[str, str]]] = {
    "degree": {
        "low":     {"emoji": "⚪", "label": "Low connectivity",      "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Moderate connectivity", "color": "#3B82F6"},
        "high":    {"emoji": "🟢", "label": "Highly connected",      "color": "#10B981"},
        "extreme": {"emoji": "🔥", "label": "Super hub",             "color": "#F97316"},
    },
    "betweenness": {
        "low":     {"emoji": "⚪", "label": "Within cluster",        "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Occasional bridge",    "color": "#3B82F6"},
        "high":    {"emoji": "🟠", "label": "Key broker",           "color": "#F97316"},
        "extreme": {"emoji": "🚨", "label": "Critical bottleneck",  "color": "#DC2626"},
    },
    "closeness": {
        "low":     {"emoji": "⚪", "label": "Hard to reach",         "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Moderate reach",       "color": "#3B82F6"},
        "high":    {"emoji": "💫", "label": "Well positioned",      "color": "#10B981"},
        "extreme": {"emoji": "🚀", "label": "System-wide access",   "color": "#0EA5E9"},
    },
    "eigenvector": {
        "low":     {"emoji": "⚪", "label": "Peripheral influence",    "color": "#9CA3AF"},
        "medium":  {"emoji": "🔹", "label": "Connected to influence", "color": "#3B82F6"},
        "high":    {"emoji": "⭐", "label": "Influence hub",           "color": "#FACC15"},
        "extreme": {"emoji": "👑", "label": "Power center",            "color": "#D97706"},
    },
}


def render_badge(metric: str, level: str, small: bool = False) -> str:
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


def centralization_index(values: Sequence[float]) -> float:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size <= 1 or float(arr.sum()) == 0.0:
        return 0.0
    top_share = float(arr.max() / arr.sum())
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
    largest_share = stats.largest_component_size / max(stats.n_nodes, 1)
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
    if label == "Healthy cohesion":
        color = "🟢"
    elif label == "Mixed signals":
        color = "🟡"
    else:
        color = "🔴"
    st.markdown(f"### {color} Network Health: **{score} / 100** — *{label}*")
    st.caption("This score reflects connectivity, cohesion, fragmentation, and how concentrated influence is.")


# ============================================================================
# BROKERAGE ROLES (Gould & Fernandez Classification)
# ============================================================================

def detect_communities(G: nx.Graph) -> Dict[str, int]:
    try:
        import community as community_louvain  # type: ignore
        partition = community_louvain.best_partition(G)
        return partition
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        partition: Dict[str, int] = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition


def compute_brokerage_roles(G: nx.Graph, communities: Dict[str, int]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
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


def render_broker_badge(role: str, small: bool = True) -> str:
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


# ============================================================================
# CONFIGURATION
# ============================================================================

PER_MIN_LIMIT = 20
DEFAULT_SYNTHETIC_MODE = True

MAX_SEED_ROWS = 10


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
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
            time.sleep(max(0.0, 60 - elapsed))
            self.window_start = time.time()
            self.calls_in_window = 0

    def record_call(self):
        self.calls_in_window += 1

    def get_status(self) -> str:
        return f"{self.calls_in_window}/{self.allowed_per_min} calls this minute"


# ============================================================================
# UTILITIES
# ============================================================================

def is_linkedin_person_url(url: str) -> bool:
    u = (url or "").lower()
    return "linkedin.com/in/" in u


def is_linkedin_company_url(url: str) -> bool:
    u = (url or "").lower()
    return "linkedin.com/company/" in u


def detect_crawl_type_from_seeds(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Decide crawl type from the linkedin_profile_url column:
    - people: linkedin.com/in/...
    - company: linkedin.com/company/...
    Returns (crawl_type, error_message).
    """
    urls = [str(u).strip() for u in df.get("linkedin_profile_url", df.get("profile_url", pd.Series([], dtype=str))).tolist()]
    urls = [u for u in urls if u and u.lower() != "nan"]

    if not urls:
        return None, "No LinkedIn URLs found."

    person = [u for u in urls if is_linkedin_person_url(u)]
    company = [u for u in urls if is_linkedin_company_url(u)]
    other = [u for u in urls if (not is_linkedin_person_url(u) and not is_linkedin_company_url(u))]

    if other:
        return None, "Some URLs do not look like LinkedIn person (/in/) or company (/company/) URLs."

    if person and company:
        return None, "Mixed seed file detected (people + companies). Please run one type at a time."

    if person:
        return "people", None
    return "company", None


def extract_url_stub(profile_url: str) -> str:
    clean_url = (profile_url or "").rstrip('/').split('?')[0]
    m = re.search(r'/in/([^/]+)', clean_url)
    if m:
        return m.group(1)
    return clean_url.split('/')[-1]


def extract_company_stub(company_url: str) -> str:
    clean_url = (company_url or "").rstrip('/').split('?')[0]
    m = re.search(r'/company/([^/]+)', clean_url)
    if m:
        return m.group(1)
    return clean_url.split('/')[-1]


def canonical_id_from_url(url: str, crawl_type: str) -> str:
    return extract_url_stub(url) if crawl_type == "people" else extract_company_stub(url)


def test_network_connectivity() -> Tuple[bool, str]:
    try:
        ip = socket.gethostbyname("enrichlayer.com")
        requests.get("https://enrichlayer.com", timeout=5)
        return True, f"✅ Network OK (resolved to {ip})"
    except socket.gaierror:
        return False, "❌ DNS Resolution Failed"
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection Failed"
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


# ============================================================================
# ENRICHLAYER API CLIENTS
# ============================================================================

def call_enrichlayer_person_api(api_token: str, profile_url: str, synthetic_mode: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    if synthetic_mode:
        time.sleep(0.05)
        return get_mock_person_response(profile_url), None

    endpoint = "https://enrichlayer.com/api/v2/profile"
    headers = {"Authorization": f"Bearer {api_token}"}
    params = {"url": profile_url, "use_cache": "if-present", "live_fetch": "if-needed"}

    for attempt in range(max_retries):
        try:
            r = requests.get(endpoint, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 401:
                return None, "Invalid API token"
            if r.status_code == 403:
                return None, "Out of credits"
            if r.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 3)
                    continue
                return None, "Rate limit exceeded"
            if r.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None, "Enrichment failed"
            return None, f"API error {r.status_code}"
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


def call_enrichlayer_company_api(api_token: str, company_url: str, synthetic_mode: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    if synthetic_mode:
        time.sleep(0.05)
        return get_mock_company_response(company_url), None

    endpoint = "https://enrichlayer.com/api/v2/company"
    headers = {"Authorization": f"Bearer {api_token}"}
    params = {"url": company_url, "use_cache": "if-present", "live_fetch": "if-needed"}

    for attempt in range(max_retries):
        try:
            r = requests.get(endpoint, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 401:
                return None, "Invalid API token"
            if r.status_code == 403:
                return None, "Out of credits"
            if r.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 3)
                    continue
                return None, "Rate limit exceeded"
            if r.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None, "Enrichment failed"
            return None, f"API error {r.status_code}"
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


# ============================================================================
# SYNTHETIC RESPONSES
# ============================================================================

def get_mock_person_response(profile_url: str) -> Dict:
    import hashlib
    url_hash = int(hashlib.md5(profile_url.encode()).hexdigest(), 16)

    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
                   "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"]
    titles = ["CEO", "Founder", "Director", "VP", "Manager", "Consultant", "Partner",
              "Executive Director", "Chief Strategy Officer", "Program Director"]
    orgs = ["World Resources Institute", "The Nature Conservancy", "WWF", "IUCN",
            "Conservation International", "Environmental Defense Fund", "Sierra Club",
            "Ford Foundation", "Rockefeller Foundation", "MacArthur Foundation",
            "Stanford University", "Harvard University", "MIT", "McKinsey & Company"]
    locations = ["San Francisco, CA", "New York, NY", "Washington, DC", "Boston, MA",
                 "Los Angeles, CA", "Seattle, WA", "Chicago, IL", "Denver, CO"]

    temp_id = extract_url_stub(profile_url)
    first = first_names[url_hash % len(first_names)]
    last = last_names[(url_hash // 100) % len(last_names)]
    title = titles[(url_hash // 1000) % len(titles)]
    org = orgs[(url_hash // 10000) % len(orgs)]
    loc = locations[(url_hash // 100000) % len(locations)]

    people_also_viewed = []
    for i in range(25 + (url_hash % 12)):
        conn_hash = (url_hash + i * 7919) % (2**32)
        conn_first = first_names[conn_hash % len(first_names)]
        conn_last = last_names[(conn_hash // 100) % len(last_names)]
        conn_title = titles[(conn_hash // 1000) % len(titles)]
        conn_org = orgs[(conn_hash // 10000) % len(orgs)]
        conn_loc = locations[(conn_hash // 100000) % len(locations)]
        conn_id = f"{conn_first.lower()}-{conn_last.lower()}-{conn_hash % 1000}"
        people_also_viewed.append({
            "link": f"https://www.linkedin.com/in/{conn_id}",
            "name": f"{conn_first} {conn_last}",
            "summary": f"{conn_title} at {conn_org}",
            "location": conn_loc
        })

    headline = f"{title} at {org}"
    return {
        "public_identifier": temp_id,
        "full_name": f"{first} {last}",
        "headline": headline,
        "occupation": headline,
        "location_str": loc,
        "people_also_viewed": people_also_viewed
    }


def get_mock_company_response(company_url: str) -> Dict:
    import hashlib
    url_hash = int(hashlib.md5(company_url.encode()).hexdigest(), 16)
    industries = ["Nonprofit", "Consulting", "Technology", "Financial Services", "Higher Education", "Government"]
    geos = ["New York, NY", "Toronto, ON", "Chicago, IL", "Washington, DC", "San Francisco, CA", "Boston, MA"]

    slug = extract_company_stub(company_url)
    name = slug.replace("-", " ").title()
    industry = industries[url_hash % len(industries)]
    geo = geos[(url_hash // 1000) % len(geos)]
    website = f"https://www.{slug}.org"

    similars = []
    for i in range(15 + (url_hash % 10)):
        s_hash = (url_hash + i * 3571) % (2**32)
        s_slug = f"{slug}-{s_hash % 1000}"
        similars.append({
            "link": f"https://www.linkedin.com/company/{s_slug}/",
            "name": s_slug.replace("-", " ").title(),
            "industry": industries[s_hash % len(industries)],
            "location": geos[(s_hash // 100) % len(geos)]
        })

    return {
        "company_slug": slug,
        "name": name,
        "industry": industry,
        "website": website,
        "hq_location": geo,
        # different APIs use different keys; we support both in crawler
        "similar_companies": similars,
        "companies_also_viewed": similars,
    }


# ============================================================================
# CRAWLER (BFS)
# ============================================================================

def update_canonical_ids(seen: Dict, edges: List[Dict], old_id: str, new_id: str) -> None:
    if old_id in seen:
        node = seen[old_id]
        node["id"] = new_id
        seen[new_id] = node
        if old_id != new_id:
            del seen[old_id]
    for e in edges:
        if e["source_id"] == old_id:
            e["source_id"] = new_id
        if e["target_id"] == old_id:
            e["target_id"] = new_id


def validate_graph(seen: Dict, edges: List[Dict]) -> Tuple[List[str], List[Dict]]:
    node_ids = set(seen.keys())
    orphan_ids = set()
    valid_edges = []
    for e in edges:
        if e["source_id"] in node_ids and e["target_id"] in node_ids:
            valid_edges.append(e)
        else:
            if e["source_id"] not in node_ids:
                orphan_ids.add(e["source_id"])
            if e["target_id"] not in node_ids:
                orphan_ids.add(e["target_id"])
    return sorted(orphan_ids), valid_edges


def _extract_company_neighbors(response: Dict) -> List[Dict]:
    # support a few plausible shapes
    for key in ("similar_companies", "companies_also_viewed", "also_viewed", "similar"):
        val = response.get(key)
        if isinstance(val, list):
            return val
    return []


def run_crawler(
    seeds: List[Dict],
    api_token: str,
    crawl_type: str,
    max_degree: int,
    max_edges: int,
    max_nodes: int,
    status_container,
    synthetic_mode: bool = False,
    advanced_mode: bool = False,
    progress_bar=None,
    per_min_limit: int = PER_MIN_LIMIT,
) -> Tuple[Dict, List[Dict], List[Dict], Dict]:
    rate_limiter = None if synthetic_mode else RateLimiter(per_min_limit=per_min_limit)

    queue = deque()
    seen_profiles: Dict[str, Dict] = {}
    edges: List[Dict] = []
    raw_items: List[Dict] = []
    processed_nodes = 0

    stats = {
        "api_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "nodes_added": 0,
        "edges_added": 0,
        "max_degree_reached": 0,
        "stopped_reason": None,
        "items_with_no_neighbors": 0,  # people or companies
        "crawl_type": crawl_type,
        "error_breakdown": {
            "rate_limit": 0,
            "out_of_credits": 0,
            "auth_error": 0,
            "enrichment_failed": 0,
            "other": 0,
        },
    }

    status_container.write("🌱 Initializing seeds...")
    for seed in seeds:
        url = seed.get("profile_url") or seed.get("linkedin_profile_url") or ""
        name = seed.get("name") or seed.get("label") or seed.get("organization") or url
        temp_id = canonical_id_from_url(url, crawl_type=crawl_type)
        node = {
            "id": temp_id,
            "name": name,
            "profile_url": url,
            "degree": 0,
            "source_type": "seed",
            "crawl_type": crawl_type,
        }
        seen_profiles[temp_id] = node
        queue.append(temp_id)
        stats["nodes_added"] += 1

    status_container.write(f"✅ Added {len(seeds)} seed(s) to queue")

    while queue:
        if len(edges) >= max_edges:
            stats["stopped_reason"] = "edge_limit"
            break
        if len(seen_profiles) >= max_nodes:
            stats["stopped_reason"] = "node_limit"
            break

        current_id = queue.popleft()
        current_node = seen_profiles[current_id]
        processed_nodes += 1

        if progress_bar is not None:
            total_known = processed_nodes + len(queue)
            if total_known > 0:
                progress_bar.progress(
                    min(max(processed_nodes / total_known, 0.0), 0.99),
                    text=f"Processing... {processed_nodes} done, {len(queue)} remaining",
                )

        if current_node["degree"] >= max_degree:
            continue

        label = current_node.get("name", current_id)
        progress_text = f"🔍 Processing: {label} (degree {current_node['degree']})"
        if rate_limiter:
            progress_text += f" | ⏱️ {rate_limiter.get_status()}"
        status_container.write(progress_text)

        if rate_limiter:
            rate_limiter.wait_for_slot()

        stats["api_calls"] += 1
        if crawl_type == "people":
            response, error = call_enrichlayer_person_api(api_token, current_node["profile_url"], synthetic_mode=synthetic_mode)
        else:
            response, error = call_enrichlayer_company_api(api_token, current_node["profile_url"], synthetic_mode=synthetic_mode)

        if rate_limiter:
            rate_limiter.record_call()

        if error:
            stats["failed_calls"] += 1
            status_container.error(f"❌ Failed: {error}")
            if "Rate limit" in error:
                stats["error_breakdown"]["rate_limit"] += 1
            elif "Out of credits" in error:
                stats["error_breakdown"]["out_of_credits"] += 1
                stats["stopped_reason"] = "out_of_credits"
                break
            elif "Invalid API token" in error:
                stats["error_breakdown"]["auth_error"] += 1
                stats["stopped_reason"] = "auth_error"
                break
            else:
                stats["error_breakdown"]["other"] += 1
            continue

        stats["successful_calls"] += 1
        raw_items.append(response or {})

        # Enrich node fields
        if crawl_type == "people":
            enriched_id = (response or {}).get("public_identifier", current_id)
            current_node["headline"] = (response or {}).get("headline", "")
            current_node["location"] = (response or {}).get("location_str") or (response or {}).get("location", "")
            current_node["label"] = current_node.get("name") or (response or {}).get("full_name", "")
            neighbors = (response or {}).get("people_also_viewed", []) or []
        else:
            # company
            enriched_id = (response or {}).get("company_slug", current_id)
            current_node["industry"] = (response or {}).get("industry", "")
            current_node["website"] = (response or {}).get("website", "")
            current_node["location"] = (response or {}).get("hq_location") or (response or {}).get("location", "")
            current_node["label"] = (response or {}).get("name") or current_node.get("name")
            neighbors = _extract_company_neighbors(response or {})

        if enriched_id and enriched_id != current_id:
            update_canonical_ids(seen_profiles, edges, current_id, enriched_id)
            current_id = enriched_id
            current_node = seen_profiles[current_id]

        if not neighbors:
            stats["items_with_no_neighbors"] += 1
        else:
            status_container.write(f"   └─ Found {len(neighbors)} connections")

        for n in neighbors:
            if len(edges) >= max_edges:
                break

            n_url = n.get("link") or n.get("profile_url") or n.get("url") or ""
            if not n_url:
                continue

            # Filter neighbors to same crawl_type to avoid accidental mixing
            if crawl_type == "people" and not is_linkedin_person_url(n_url):
                continue
            if crawl_type == "company" and not is_linkedin_company_url(n_url):
                continue

            n_name = n.get("name") or n.get("full_name") or ""
            n_headline = n.get("summary") or n.get("headline") or ""
            n_id = n.get("public_identifier") or n.get("company_slug") or canonical_id_from_url(n_url, crawl_type=crawl_type)

            edge_type = "people_also_viewed" if crawl_type == "people" else "similar_company"
            edges.append({"source_id": current_id, "target_id": n_id, "edge_type": edge_type})
            stats["edges_added"] += 1

            if n_id in seen_profiles:
                continue
            if len(seen_profiles) >= max_nodes:
                break

            node = {
                "id": n_id,
                "name": n_name or n_id,
                "profile_url": n_url,
                "degree": current_node["degree"] + 1,
                "source_type": "discovered",
                "crawl_type": crawl_type,
                "label": n_name or n_id,
            }

            if crawl_type == "people":
                node["headline"] = n_headline
                node["location"] = n.get("location", "")
            else:
                node["industry"] = n.get("industry", "")
                node["location"] = n.get("location", "")
                node["website"] = n.get("website", "")

            seen_profiles[n_id] = node
            stats["nodes_added"] += 1
            stats["max_degree_reached"] = max(stats["max_degree_reached"], node["degree"])
            if node["degree"] < max_degree:
                queue.append(n_id)

    if not stats["stopped_reason"]:
        stats["stopped_reason"] = "completed"

    return seen_profiles, edges, raw_items, stats


# ============================================================================
# METRICS
# ============================================================================

def calculate_network_metrics(seen_profiles: Dict, edges: List[Dict]) -> Dict:
    G = nx.Graph()
    for node_id, node_data in seen_profiles.items():
        G.add_node(node_id, **node_data)
    for e in edges:
        G.add_edge(e["source_id"], e["target_id"])

    node_metrics = {node_id: {} for node_id in seen_profiles.keys()}
    network_stats: Dict[str, float] = {}
    top_nodes: Dict[str, List[Tuple[str, float]]] = {}

    if len(G.nodes()) < 2 or len(G.edges()) < 1:
        return {"node_metrics": node_metrics, "network_stats": {"nodes": len(G.nodes()), "edges": len(G.edges())},
                "top_nodes": {}, "brokerage_roles": {}, "communities": {}}

    try:
        degree_centrality = nx.degree_centrality(G)
        for node_id, value in degree_centrality.items():
            node_metrics[node_id]["degree_centrality"] = round(float(value), 4)
        for node_id in G.nodes():
            node_metrics[node_id]["degree"] = int(G.degree(node_id))
        top_nodes["degree"] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        network_stats["avg_degree"] = round(sum(dict(G.degree()).values()) / len(G.nodes()), 2)
    except Exception:
        pass

    try:
        betweenness = nx.betweenness_centrality(G)
        for node_id, value in betweenness.items():
            node_metrics[node_id]["betweenness_centrality"] = round(float(value), 4)
        top_nodes["betweenness"] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        pass

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        for node_id, value in eigenvector.items():
            node_metrics[node_id]["eigenvector_centrality"] = round(float(value), 4)
        top_nodes["eigenvector"] = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        pass

    try:
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subgraph)
        for node_id, value in closeness.items():
            node_metrics[node_id]["closeness_centrality"] = round(float(value), 4)
        top_nodes["closeness"] = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        pass

    communities: Dict[str, int] = {}
    brokerage_roles: Dict[str, str] = {}
    try:
        if len(G.nodes()) >= 3 and len(G.edges()) >= 2:
            communities = detect_communities(G)
            brokerage_roles = compute_brokerage_roles(G, communities)
            for node_id in node_metrics:
                if node_id in communities:
                    node_metrics[node_id]["community"] = communities[node_id]
                if node_id in brokerage_roles:
                    node_metrics[node_id]["brokerage_role"] = brokerage_roles[node_id]
            network_stats["num_communities"] = len(set(communities.values())) if communities else 0
    except Exception:
        pass

    network_stats["nodes"] = len(G.nodes())
    network_stats["edges"] = len(G.edges())
    try:
        network_stats["density"] = round(nx.density(G), 4)
    except Exception:
        pass
    try:
        if nx.is_connected(G):
            network_stats["num_components"] = 1
            network_stats["largest_component_size"] = len(G.nodes())
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            network_stats["largest_component_size"] = len(largest_cc)
            network_stats["num_components"] = nx.number_connected_components(G)
    except Exception:
        pass
    try:
        network_stats["avg_clustering"] = round(nx.average_clustering(G), 4)
    except Exception:
        pass

    return {
        "node_metrics": node_metrics,
        "network_stats": network_stats,
        "top_nodes": top_nodes,
        "brokerage_roles": brokerage_roles,
        "communities": communities,
    }


# ============================================================================
# EXPORTS (CSV/JSON/ZIP)
# ============================================================================

def _polinode_safe_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ").strip()
    return s


def generate_nodes_csv(
    seen_profiles: Dict,
    max_degree: int,
    max_edges: int,
    max_nodes: int,
    network_metrics: Optional[Dict] = None,
    crawl_type: Optional[str] = None,
    **_kwargs,
) -> str:
    """
    Polinode-friendly nodes.csv.

    NOTE: We accept crawl_type/**kwargs to avoid brittle download crashes when the UI evolves.
    """
    node_metrics = (network_metrics or {}).get("node_metrics", {})
    rows: List[Dict] = []

    # best-effort crawl_type detection if not provided
    if not crawl_type:
        ct = None
        for n in seen_profiles.values():
            if n.get("crawl_type") in ("people", "company"):
                ct = n.get("crawl_type")
                break
        crawl_type = ct or "people"

    for node in seen_profiles.values():
        node_id = node.get("id", "")
        url = node.get("profile_url", "")
        label = node.get("label") or node.get("name") or node_id

        base = {
            "id": _polinode_safe_text(node_id),
            "label": _polinode_safe_text(label),
            "name": _polinode_safe_text(node.get("name", label)),
            "linkedin_url": _polinode_safe_text(url),
            "source_type": _polinode_safe_text(node.get("source_type", "")),
            "crawl_type": _polinode_safe_text(node.get("crawl_type", crawl_type)),
            "degree": int(node.get("degree", 0)),
        }

        if crawl_type == "people":
            base.update({
                "headline": _polinode_safe_text(node.get("headline", "")),
                "location": _polinode_safe_text(node.get("location", "")),
            })
        else:
            base.update({
                "industry": _polinode_safe_text(node.get("industry", "")),
                "website": _polinode_safe_text(node.get("website", "")),
                "location": _polinode_safe_text(node.get("location", "")),
            })

        m = node_metrics.get(node_id, {})
        if m:
            base.update({
                "connections": m.get("degree", 0),
                "degree_centrality": m.get("degree_centrality", 0),
                "betweenness_centrality": m.get("betweenness_centrality", 0),
                "eigenvector_centrality": m.get("eigenvector_centrality", 0),
                "closeness_centrality": m.get("closeness_centrality", 0),
                "community": m.get("community", ""),
                "brokerage_role": m.get("brokerage_role", ""),
            })

        rows.append(base)

    df = pd.DataFrame(rows)
    meta = f"# generated_at={datetime.now(timezone.utc).isoformat()}; max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}; crawl_type={crawl_type}\n"
    return meta + df.to_csv(index=False)


def generate_edges_csv(edges: List[Dict], max_degree: int, max_edges: int, max_nodes: int, crawl_type: str = "") -> str:
    df = pd.DataFrame(edges)
    if "edge_type" not in df.columns:
        df["edge_type"] = ""
    meta = f"# generated_at={datetime.now(timezone.utc).isoformat()}; max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}; crawl_type={crawl_type}\n"
    return meta + df.to_csv(index=False)


def generate_raw_json(raw_items: List[Dict]) -> str:
    return json.dumps(raw_items, indent=2)


def generate_crawl_log(stats: Dict, seen_profiles: Dict, edges: List[Dict], max_degree: int,
                       max_edges: int, max_nodes: int, synthetic_mode: bool) -> str:
    seed_count = sum(1 for n in seen_profiles.values() if n.get("source_type") == "seed")
    log_data = {
        "crawl_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": APP_VERSION,
            "crawl_type": stats.get("crawl_type"),
            "synthetic_mode": synthetic_mode,
        },
        "configuration": {
            "max_degree": max_degree,
            "max_edges": max_edges,
            "max_nodes": max_nodes,
            "per_min_limit": PER_MIN_LIMIT,
        },
        "api_statistics": {
            "total_calls": stats.get("api_calls", 0),
            "successful_calls": stats.get("successful_calls", 0),
            "failed_calls": stats.get("failed_calls", 0),
            "success_rate": round((stats.get("successful_calls", 0) / max(stats.get("api_calls", 1), 1)) * 100, 2),
        },
        "network_statistics": {
            "total_nodes": len(seen_profiles),
            "total_edges": len(edges),
            "seed_nodes": seed_count,
            "items_with_no_neighbors": stats.get("items_with_no_neighbors", 0),
        },
        "stop_reason": stats.get("stopped_reason", "unknown"),
        "error_breakdown": stats.get("error_breakdown", {}),
    }
    return json.dumps(log_data, indent=2)


def create_download_zip(nodes_csv: str, edges_csv: str, raw_json: str, crawl_log: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("nodes.csv", nodes_csv)
        z.writestr("edges.csv", edges_csv)
        z.writestr("raw_items.json", raw_json)
        z.writestr("crawl_log.json", crawl_log)
    buf.seek(0)
    return buf.getvalue()


# ============================================================================
# SIMPLE CHARTS
# ============================================================================

def create_brokerage_role_chart(brokerage_roles: Dict[str, str]) -> Optional[go.Figure]:
    if not brokerage_roles:
        return None
    role_counts: Dict[str, int] = {}
    for r in brokerage_roles.values():
        role_counts[r] = role_counts.get(r, 0) + 1

    role_order = ["liaison", "gatekeeper", "representative", "coordinator", "consultant", "peripheral"]
    labels = {
        "liaison": "🌉 Liaison",
        "gatekeeper": "🚪 Gatekeeper",
        "representative": "🔗 Representative",
        "coordinator": "🧩 Coordinator",
        "consultant": "🧠 Consultant",
        "peripheral": "⚪ Peripheral",
    }
    colors = {
        "liaison": "#D97706",
        "gatekeeper": "#F97316",
        "representative": "#10B981",
        "coordinator": "#3B82F6",
        "consultant": "#6366F1",
        "peripheral": "#9CA3AF",
    }

    roles, counts, cols = [], [], []
    for r in role_order:
        if r in role_counts:
            roles.append(labels.get(r, r))
            counts.append(role_counts[r])
            cols.append(colors.get(r, "#9CA3AF"))

    fig = go.Figure()
    fig.add_trace(go.Bar(y=roles, x=counts, orientation="h", marker_color=cols, text=counts, textposition="auto"))
    fig.update_layout(
        title="Brokerage Role Distribution",
        xaxis_title="Count",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="ActorGraph",
        page_icon="https://static.wixstatic.com/media/275a3f_5747a8179bda42ab9b268accbdaf4ac2~mv2.png",
        layout="wide",
    )

    if "crawl_results" not in st.session_state:
        st.session_state.crawl_results = None

    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("https://static.wixstatic.com/media/275a3f_5747a8179bda42ab9b268accbdaf4ac2~mv2.png", width=80)
    with col2:
        st.title("ActorGraph")
        st.markdown("People- or company-centered network graphs from public LinkedIn data.")
        st.caption(f"v{APP_VERSION}")

    st.markdown("---")

    st.subheader("🎛️ Select Mode")
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        advanced_mode = st.toggle("mode_toggle", value=False, label_visibility="collapsed", key="_advanced_mode")
    with c1:
        st.markdown("**📊 Seed Crawler**" if not advanced_mode else "📊 Seed Crawler")
    with c3:
        st.markdown("**🔬 Intelligence Engine**" if advanced_mode else "🔬 Intelligence Engine")

    st.markdown("---")
    st.header("📥 Input")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("1. Upload Seed CSV")
        st.caption(f"Prototype limit: max {MAX_SEED_ROWS} rows. Mixed people+company files are blocked.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        seeds: List[Dict] = []
        crawl_type: Optional[str] = None

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                # Accept either (name, profile_url) OR (name, linkedin_profile_url)
                if "profile_url" not in df.columns and "linkedin_profile_url" in df.columns:
                    df = df.rename(columns={"linkedin_profile_url": "profile_url"})

                required = ["name", "profile_url"]
                missing = [c for c in required if c not in df.columns]

                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                else:
                    # ignore blank URLs
                    df["profile_url"] = df["profile_url"].astype(str).str.strip()
                    df = df[df["profile_url"].str.len() > 0]
                    df = df[~df["profile_url"].str.lower().isin(["nan", "none"])]

                    if len(df) == 0:
                        st.error("❌ No usable LinkedIn URLs found after filtering blanks.")
                    elif len(df) > MAX_SEED_ROWS:
                        st.error(f"❌ Prototype limit: max {MAX_SEED_ROWS} seed rows.")
                    else:
                        crawl_type, err = detect_crawl_type_from_seeds(df.rename(columns={"profile_url": "linkedin_profile_url"}))
                        if err:
                            st.error(f"❌ {err}")
                        else:
                            seeds = df[["name", "profile_url"]].to_dict("records")
                            ct_label = "People crawl" if crawl_type == "people" else "Company crawl"
                            st.success(f"✅ Loaded {len(seeds)} seeds — {ct_label}")
                            st.dataframe(df)

            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")

    with right:
        st.subheader("2. EnrichLayer API Token")
        default_token = ""
        try:
            default_token = st.secrets.get("ENRICHLAYER_TOKEN", "")
        except Exception:
            pass

        api_token = st.text_input("Enter your API token", type="password", value=default_token)

        if st.button("🔍 Test Connection"):
            with st.spinner("Testing..."):
                ok, msg = test_network_connectivity()
                st.success(msg) if ok else st.error(msg)

        synthetic_mode = st.toggle("Run in synthetic mode (no credits)", value=DEFAULT_SYNTHETIC_MODE)
        if synthetic_mode:
            st.info("🧪 Synthetic mode is ON — no real API calls, no credits used.")

    st.header("⚙️ Crawl Configuration")
    c1, c2 = st.columns(2)
    with c1:
        max_degree = st.radio("Maximum Degree (hops)", options=[1, 2], index=0)
        if max_degree == 2:
            st.warning("⚠️ Degree 2 can be 10–50× more calls. Start with Degree 1.")
        else:
            st.success("✅ Degree 1 selected — direct connections only.")
    with c2:
        st.markdown("**Crawl Limits (fixed):**")
        st.metric("Max Edges", 10000)
        st.metric("Max Nodes", 7500)

    st.caption(f"⏱️ API pacing: up to {PER_MIN_LIMIT} requests/minute")

    can_run = len(seeds) > 0 and (api_token or synthetic_mode) and (crawl_type in ("people", "company"))
    if not can_run:
        if len(seeds) == 0:
            st.warning("⚠️ Upload a valid seed CSV to continue.")
        elif not api_token and not synthetic_mode:
            st.warning("⚠️ Enter an API token or enable synthetic mode.")
        elif crawl_type not in ("people", "company"):
            st.warning("⚠️ Seed crawl type could not be detected.")

    run_button = st.button("🚀 Run Crawl", disabled=not can_run, type="primary", use_container_width=True)

    if run_button:
        st.header("🔄 Crawl Progress")
        progress_bar = st.progress(0.0, text="Starting crawl...")
        status_container = st.status("Running crawl...", expanded=True)

        seen_profiles, edges, raw_items, stats = run_crawler(
            seeds=seeds,
            api_token=api_token,
            crawl_type=crawl_type or "people",
            max_degree=max_degree,
            max_edges=10000,
            max_nodes=7500,
            status_container=status_container,
            synthetic_mode=synthetic_mode,
            advanced_mode=advanced_mode,
            progress_bar=progress_bar,
            per_min_limit=PER_MIN_LIMIT,
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
            "seen_profiles": seen_profiles,
            "edges": edges,
            "raw_items": raw_items,
            "stats": stats,
            "max_degree": max_degree,
            "advanced_mode": advanced_mode,
            "synthetic_mode": synthetic_mode,
            "crawl_type": crawl_type,
            "network_metrics": network_metrics,
        }

    # RESULTS / DOWNLOAD
    if st.session_state.crawl_results is not None:
        res = st.session_state.crawl_results
        seen_profiles = res["seen_profiles"]
        edges = res["edges"]
        raw_items = res["raw_items"]
        stats = res["stats"]
        max_degree = res["max_degree"]
        crawl_type = res.get("crawl_type") or stats.get("crawl_type") or "people"
        advanced_mode = res.get("advanced_mode", False)
        synthetic_mode = res.get("synthetic_mode", False)
        network_metrics = res.get("network_metrics")

        st.header("📊 Results Summary")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Nodes", len(seen_profiles))
        k2.metric("Total Edges", len(edges))
        k3.metric("API Calls", stats.get("api_calls", 0))
        k4.metric("Successful", stats.get("successful_calls", 0))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Failed", stats.get("failed_calls", 0))
        label_no = "People w/ no neighbors" if crawl_type == "people" else "Companies w/ no neighbors"
        k6.metric(label_no, stats.get("items_with_no_neighbors", 0))
        k7.metric("Max Degree Reached", stats.get("max_degree_reached", 0))
        if stats.get("stopped_reason") == "completed":
            k8.success("✅ Completed")
        else:
            k8.warning(str(stats.get("stopped_reason")))

        if advanced_mode and network_metrics and network_metrics.get("top_nodes"):
            st.markdown("---")
            st.header("🔬 Network Intelligence")

            node_metrics = network_metrics.get("node_metrics", {})
            net_stats = network_metrics.get("network_stats", {})
            top_nodes = network_metrics.get("top_nodes", {})
            brokerage_roles = network_metrics.get("brokerage_roles", {})

            degree_values = [m.get("degree_centrality", 0) for m in node_metrics.values()]
            betweenness_values = [m.get("betweenness_centrality", 0) for m in node_metrics.values()]
            deg_bp = compute_breakpoints(degree_values) if degree_values else MetricBreakpoints(0, 0, 0)
            btw_bp = compute_breakpoints(betweenness_values) if betweenness_values else MetricBreakpoints(0, 0, 0)

            health_stats = NetworkStats(
                n_nodes=int(net_stats.get("nodes", 0)),
                n_edges=int(net_stats.get("edges", 0)),
                density=float(net_stats.get("density", 0)),
                avg_degree=float(net_stats.get("avg_degree", 0)),
                avg_clustering=float(net_stats.get("avg_clustering", 0)),
                n_components=int(net_stats.get("num_components", 1)),
                largest_component_size=int(net_stats.get("largest_component_size", net_stats.get("nodes", 0) or 0)),
            )
            score, label = compute_network_health(health_stats, degree_values, betweenness_values)
            render_health_summary(score, label)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔗 Top Connectors** (Degree)")
                for i, (nid, sc) in enumerate(top_nodes.get("degree", [])[:5], 1):
                    name = seen_profiles.get(nid, {}).get("label") or seen_profiles.get(nid, {}).get("name") or nid
                    conn = node_metrics.get(nid, {}).get("degree", 0)
                    level = classify_value(sc, deg_bp)
                    st.markdown(f"{i}. **{name}** — {conn} connections {render_badge('degree', level, small=True)}", unsafe_allow_html=True)

            with c2:
                st.markdown("**🌉 Top Brokers** (Betweenness)")
                for i, (nid, sc) in enumerate(top_nodes.get("betweenness", [])[:5], 1):
                    name = seen_profiles.get(nid, {}).get("label") or seen_profiles.get(nid, {}).get("name") or nid
                    level = classify_value(sc, btw_bp)
                    role = brokerage_roles.get(nid, "peripheral")
                    st.markdown(f"{i}. **{name}** — {sc:.4f} {render_badge('betweenness', level, small=True)} {render_broker_badge(role, small=True)}", unsafe_allow_html=True)

            if brokerage_roles:
                st.markdown("---")
                st.subheader("🎭 Brokerage Roles")
                fig = create_brokerage_role_chart(brokerage_roles)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        st.header("💾 Download Results")

        nodes_csv = generate_nodes_csv(
            seen_profiles,
            max_degree=max_degree,
            max_edges=10000,
            max_nodes=7500,
            crawl_type=crawl_type,
            network_metrics=network_metrics,
        )
        edges_csv = generate_edges_csv(edges, max_degree=max_degree, max_edges=10000, max_nodes=7500, crawl_type=crawl_type)
        raw_json = generate_raw_json(raw_items)
        crawl_log = generate_crawl_log(
            stats=stats,
            seen_profiles=seen_profiles,
            edges=edges,
            max_degree=max_degree,
            max_edges=10000,
            max_nodes=7500,
            synthetic_mode=synthetic_mode,
        )

        zip_data = create_download_zip(nodes_csv, edges_csv, raw_json, crawl_log)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.download_button(
                "⬇️ Download All as ZIP",
                data=zip_data,
                file_name=f"actorgraph_{crawl_type}_network.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
        with c2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.crawl_results = None
                st.rerun()

        st.markdown("### 📄 Individual Files")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("📥 nodes.csv", data=nodes_csv, file_name="nodes.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("📥 edges.csv", data=edges_csv, file_name="edges.csv", mime="text/csv", use_container_width=True)
        with c3:
            st.download_button("📥 raw_items.json", data=raw_json, file_name="raw_items.json", mime="application/json", use_container_width=True)
        with c4:
            st.download_button("📥 crawl_log.json", data=crawl_log, file_name="crawl_log.json", mime="application/json", use_container_width=True)

        with st.expander("👀 Preview Nodes"):
            st.dataframe(pd.DataFrame([n for n in seen_profiles.values()]))

        with st.expander("👀 Preview Edges"):
            st.dataframe(pd.DataFrame(edges) if edges else pd.DataFrame(columns=["source_id", "target_id", "edge_type"]))


if __name__ == "__main__":
    main()
