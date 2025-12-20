# ActorGraph — Streamlit App
# C4C Network Crawler (People + Company modes)
#
# UPDATED v0.5.0: Seed auto-detect + company Polinode fields + KPI logging + max-10 enforcement
# - Auto-detect seed type from columns + URL patterns (/in/ vs /company/)
# - Disallow mixed seeds in a single CSV
# - Enforce max 10 seed rows (non-empty URL rows)
# - KPI tiles adapt to crawl type (people vs company)
# - Cleaner Polinode-ready company node fields (no crawl-logic changes)
#
# NOTE: This app assumes EnrichLayer endpoints:
#   - People:  POST {BASE_URL}/api/v2/profile   payload {"linkedin_url": "..."}
#   - Company: POST {BASE_URL}/api/v2/company   payload {"linkedin_url": "..."}
# If your EnrichLayer account uses different paths, set env vars:
#   ENRICHLAYER_BASE_URL, ENRICHLAYER_PROFILE_PATH, ENRICHLAYER_COMPANY_PATH

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Versioning
# -----------------------------
APP_VERSION = "v0.5.0"

# Keep a short human-readable changelog for sanity while you iterate quickly.
VERSION_HISTORY = [
    {
        "version": "v0.5.0",
        "title": "Seed auto-detect + Polinode company fields + KPI logging",
        "bullets": [
            "Auto-detect People vs Company seed files from columns + URL patterns",
            "Disallow mixed People + Company seeds in the same CSV",
            "Enforce max 10 seed rows (rows missing LinkedIn URL are ignored)",
            "KPI tiles and crawl log KPIs adapt to crawl type",
            "Cleaner Polinode-ready company node fields (Name, Type, LinkedIn URL, Industry, Location, Seed flag, !Internal ID)",
        ],
    },
    {
        "version": "v0.4.1",
        "title": "Company crawl support (first pass) + Polinode exports",
        "bullets": [
            "Company seed CSV support (LinkedIn /company/ URLs)",
            "Polinode-ready nodes/edges export buttons",
        ],
    },
    {
        "version": "v0.4.0",
        "title": "Stable People crawl + mock mode",
        "bullets": [
            "People seed CSV support (LinkedIn /in/ URLs)",
            "Mock mode (synthetic) for no-credit testing",
        ],
    },
]


# -----------------------------
# EnrichLayer config
# -----------------------------
DEFAULT_BASE_URL = "https://api.enrichlayer.com"
BASE_URL = os.getenv("ENRICHLAYER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
PROFILE_PATH = os.getenv("ENRICHLAYER_PROFILE_PATH", "/api/v2/profile")
COMPANY_PATH = os.getenv("ENRICHLAYER_COMPANY_PATH", "/api/v2/company")  # may differ for your plan/account

REQUEST_TIMEOUT = 60


# -----------------------------
# Helpers
# -----------------------------
def _clean_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s


def _is_linkedin_people_url(url: str) -> bool:
    u = url.lower()
    return "linkedin.com" in u and "/in/" in u


def _is_linkedin_company_url(url: str) -> bool:
    u = url.lower()
    return "linkedin.com" in u and "/company/" in u


def _normalize_linkedin_url(url: str) -> str:
    """Light normalization; we purposely keep it conservative."""
    u = _clean_str(url)
    if not u:
        return ""
    # add scheme if missing
    if u.startswith("www."):
        u = "https://" + u
    if u.startswith("linkedin.com"):
        u = "https://www." + u
    return u


@dataclass
class SeedLoadResult:
    ok: bool
    crawl_type: Optional[str]  # "people" | "company"
    df: Optional[pd.DataFrame]
    error: Optional[str]


def detect_seed_type_and_load(df: pd.DataFrame) -> SeedLoadResult:
    """
    Accepted seed CSV formats:

    People mode:
      - columns: name, profile_url
        (profile_url must be a LinkedIn /in/ url)

    Company mode:
      - columns: org_name, linkedin_profile_url
        (linkedin_profile_url must be a LinkedIn /company/ url)

    Friendly aliases also accepted:
      - name can stand in for org_name (for company files)
      - profile_url can stand in for linkedin_profile_url (for company files)

    Rules:
      - Ignore rows with blank URL
      - Max 10 non-blank URL rows
      - No mixed people+company URLs in same file
    """
    if df is None or df.empty:
        return SeedLoadResult(False, None, None, "Seed CSV is empty.")

    # normalize columns
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Identify candidate URL column
    url_col = None
    for candidate in ["profile_url", "linkedin_profile_url", "linkedin_url", "url"]:
        if candidate in df.columns:
            url_col = candidate
            break
    if not url_col:
        return SeedLoadResult(
            False, None, None,
            "Seed CSV must include a LinkedIn URL column: profile_url (people) or linkedin_profile_url (company)."
        )

    df[url_col] = df[url_col].apply(_normalize_linkedin_url)
    df_nonblank = df[df[url_col].astype(str).str.strip() != ""].copy()

    if df_nonblank.empty:
        return SeedLoadResult(False, None, None, "No valid LinkedIn URLs found (all URL rows were blank).")

    # max-10 enforcement (based on nonblank URLs)
    if len(df_nonblank) > 10:
        df_nonblank = df_nonblank.iloc[:10].copy()

    # Determine by URL pattern
    url_flags = []
    for u in df_nonblank[url_col].astype(str).tolist():
        is_people = _is_linkedin_people_url(u)
        is_company = _is_linkedin_company_url(u)
        url_flags.append((is_people, is_company, u))

    any_people = any(p for p, c, u in url_flags)
    any_company = any(c for p, c, u in url_flags)

    if any_people and any_company:
        return SeedLoadResult(
            False, None, None,
            "Mixed seed file detected: contains both People (/in/) and Company (/company/) LinkedIn URLs. "
            "Please upload one or the other."
        )
    if not any_people and not any_company:
        return SeedLoadResult(
            False, None, None,
            "Could not detect seed type: LinkedIn URLs must contain either '/in/' (people) or '/company/' (companies)."
        )

    crawl_type = "people" if any_people else "company"

    # Validate required name columns (flexible)
    if crawl_type == "people":
        # require name column
        if "name" not in df_nonblank.columns:
            return SeedLoadResult(False, None, None, "People seed CSV must include a 'name' column.")
        # enforce url integrity
        bad = [u for p, c, u in url_flags if not p]
        if bad:
            return SeedLoadResult(
                False, None, None,
                "People seed files must contain only LinkedIn profile URLs with '/in/'."
            )
        # standardize columns
        df_nonblank = df_nonblank.rename(columns={url_col: "profile_url"})
        df_nonblank["seed_name"] = df_nonblank["name"].astype(str).str.strip()
        df_nonblank["seed_url"] = df_nonblank["profile_url"].astype(str).str.strip()

    else:
        # company
        if "org_name" not in df_nonblank.columns and "name" not in df_nonblank.columns:
            return SeedLoadResult(False, None, None, "Company seed CSV must include 'org_name' (or 'name') column.")
        org_col = "org_name" if "org_name" in df_nonblank.columns else "name"
        bad = [u for p, c, u in url_flags if not c]
        if bad:
            return SeedLoadResult(
                False, None, None,
                "Company seed files must contain only LinkedIn company URLs with '/company/'."
            )
        df_nonblank = df_nonblank.rename(columns={url_col: "linkedin_profile_url"})
        df_nonblank["seed_name"] = df_nonblank[org_col].astype(str).str.strip()
        df_nonblank["seed_url"] = df_nonblank["linkedin_profile_url"].astype(str).str.strip()

    return SeedLoadResult(True, crawl_type, df_nonblank, None)


def enrichlayer_post(endpoint_path: str, api_token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint_path}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        # Try to surface response body for debugging (Streamlit Cloud will redact secrets anyway)
        try:
            detail = r.json()
        except Exception:
            detail = {"text": r.text}
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


# -----------------------------
# Crawl logic (kept simple)
# -----------------------------
def crawl_people(
    seeds: pd.DataFrame,
    api_token: str,
    max_degree: int,
    per_minute_limit: int = 16,
    mock_mode: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      - nodes_by_url: {profile_url: node_dict}
      - edges: list of {source, target, relationship, degree}
      - stats: dict of KPIs
      - raw_items: list of raw responses saved for fixture-like debugging
    """
    t0 = time.time()
    queue: List[Tuple[str, int]] = [(u, 0) for u in seeds["seed_url"].tolist()]
    seen: set[str] = set()
    nodes_by_url: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    raw_items: List[Dict[str, Any]] = []

    calls_this_minute = 0
    window_start = time.time()

    success = 0
    failed = 0
    no_neighbors = 0

    # NOTE: adjust these keys to match your EnrichLayer response schema for people.
    neighbor_keys = ["people_also_viewed", "similar_profiles", "connections", "neighbors"]

    while queue:
        url, degree = queue.pop(0)
        if url in seen:
            continue
        if degree > max_degree:
            continue

        seen.add(url)

        # rate-limit
        if not mock_mode:
            if time.time() - window_start >= 60:
                window_start = time.time()
                calls_this_minute = 0
            if calls_this_minute >= per_minute_limit:
                time.sleep(max(0.0, 60 - (time.time() - window_start)))
                window_start = time.time()
                calls_this_minute = 0

        try:
            if mock_mode:
                item = {
                    "full_name": url.split("/")[-2].replace("-", " ").title(),
                    "linkedin_profile_url": url,
                    "headline": "Synthetic profile",
                    "location": "Syntheticville",
                    "people_also_viewed": [],
                }
            else:
                item = enrichlayer_post(PROFILE_PATH, api_token, {"linkedin_url": url})
                calls_this_minute += 1

            raw_items.append({"seed_url": url, "degree": degree, "item": item})
            success += 1

            # node
            name = item.get("full_name") or item.get("name") or item.get("first_name", "") + " " + item.get("last_name", "")
            name = _clean_str(name) or url
            nodes_by_url[url] = {
                "name": name,
                "type": "PERSON",
                "linkedin_url": url,
                "headline": _clean_str(item.get("headline")),
                "location": _clean_str(item.get("location")),
                "seed": url in set(seeds["seed_url"].tolist()),
                "degree": degree,
            }

            # neighbors
            neighbors: List[str] = []
            for k in neighbor_keys:
                v = item.get(k)
                if isinstance(v, list):
                    # accept list of dicts or list of urls
                    for n in v:
                        if isinstance(n, str):
                            neighbors.append(_normalize_linkedin_url(n))
                        elif isinstance(n, dict):
                            u = _normalize_linkedin_url(n.get("linkedin_profile_url") or n.get("profile_url") or n.get("url") or "")
                            if u:
                                neighbors.append(u)

            neighbors = [u for u in neighbors if u]
            if not neighbors:
                no_neighbors += 1

            for nb in neighbors:
                if nb == url:
                    continue
                edges.append({"source": url, "target": nb, "relationship": "NEIGHBOR", "degree": degree + 1})
                if degree + 1 <= max_degree and nb not in seen:
                    queue.append((nb, degree + 1))

        except Exception as e:
            failed += 1
            raw_items.append({"seed_url": url, "degree": degree, "error": str(e)})

    stats = {
        "crawl_type": "people",
        "seeds": int(len(seeds)),
        "successful": success,
        "failed": failed,
        "people_w_no_neighbors": no_neighbors,
        "unique_nodes": len(nodes_by_url),
        "edges": len(edges),
        "seconds": round(time.time() - t0, 2),
    }
    return nodes_by_url, edges, stats, raw_items


def _company_linkedin_url_from_universal_id(universal_name_id: Any) -> str:
    uid = _clean_str(universal_name_id)
    if not uid:
        return ""
    return f"https://www.linkedin.com/company/{uid}/"


def crawl_companies(
    seeds: pd.DataFrame,
    api_token: str,
    max_degree: int,
    per_minute_limit: int = 16,
    mock_mode: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Company crawl uses EnrichLayer company endpoint (schema based on your recent raw_items.json upload).
    We treat `similar_companies` as neighbors.
    """
    t0 = time.time()
    queue: List[Tuple[str, int]] = [(u, 0) for u in seeds["seed_url"].tolist()]
    seen: set[str] = set()
    nodes_by_url: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    raw_items: List[Dict[str, Any]] = []

    calls_this_minute = 0
    window_start = time.time()

    success = 0
    failed = 0
    no_neighbors = 0

    while queue:
        url, degree = queue.pop(0)
        if url in seen:
            continue
        if degree > max_degree:
            continue

        seen.add(url)

        # rate-limit
        if not mock_mode:
            if time.time() - window_start >= 60:
                window_start = time.time()
                calls_this_minute = 0
            if calls_this_minute >= per_minute_limit:
                time.sleep(max(0.0, 60 - (time.time() - window_start)))
                window_start = time.time()
                calls_this_minute = 0

        try:
            if mock_mode:
                item = {
                    "name": url.split("/")[-2].replace("-", " ").title(),
                    "industry": "Synthetic",
                    "headquarters": "Synthetic City",
                    "website": "",
                    "universal_name_id": "",
                    "company_size": "",
                    "similar_companies": [],
                }
            else:
                item = enrichlayer_post(COMPANY_PATH, api_token, {"linkedin_url": url})
                calls_this_minute += 1

            raw_items.append({"seed_url": url, "degree": degree, "item": item})
            success += 1

            universal_id = item.get("universal_name_id")
            canonical_url = _company_linkedin_url_from_universal_id(universal_id) or url

            nodes_by_url[canonical_url] = {
                "name": _clean_str(item.get("name")) or url,
                "type": "COMPANY",
                "linkedin_url": canonical_url,
                "industry": _clean_str(item.get("industry")),
                "website": _clean_str(item.get("website")),
                "headquarters": _clean_str(item.get("headquarters")),
                "company_size": _clean_str(item.get("company_size")),
                "specialties": ", ".join(item.get("specialties", []) or []) if isinstance(item.get("specialties"), list) else _clean_str(item.get("specialties")),
                "seed": url in set(seeds["seed_url"].tolist()),
                "degree": degree,
            }

            # neighbors
            neighbors: List[str] = []
            sc = item.get("similar_companies")
            if isinstance(sc, list):
                for n in sc:
                    if isinstance(n, dict):
                        # best-effort: these objects sometimes only have name/location, so we can't always build a URL
                        # If EnrichLayer returns universal_name_id here, use it.
                        nb_uid = n.get("universal_name_id") or n.get("id") or ""
                        nb_url = _company_linkedin_url_from_universal_id(nb_uid)
                        if nb_url:
                            neighbors.append(nb_url)

            if not neighbors:
                no_neighbors += 1

            for nb in neighbors:
                if nb == canonical_url:
                    continue
                edges.append({"source": canonical_url, "target": nb, "relationship": "SIMILAR_COMPANY", "degree": degree + 1})
                if degree + 1 <= max_degree and nb not in seen:
                    queue.append((nb, degree + 1))

        except Exception as e:
            failed += 1
            raw_items.append({"seed_url": url, "degree": degree, "error": str(e)})

    stats = {
        "crawl_type": "company",
        "seeds": int(len(seeds)),
        "successful": success,
        "failed": failed,
        "companies_w_no_neighbors": no_neighbors,
        "unique_nodes": len(nodes_by_url),
        "edges": len(edges),
        "seconds": round(time.time() - t0, 2),
    }
    return nodes_by_url, edges, stats, raw_items


# -----------------------------
# Exports
# -----------------------------
def to_polinode_nodes(nodes_by_url: Dict[str, Dict[str, Any]], crawl_type: str) -> pd.DataFrame:
    rows = []
    for url, n in nodes_by_url.items():
        base = {
            "!Internal ID": url,
            "Name": n.get("name", url),
            "Type": n.get("type", "NODE"),
            "LinkedIn URL": n.get("linkedin_url", url),
            "Seed": "seed" if n.get("seed") else "discovered",
            "Degree": n.get("degree", ""),
        }
        if crawl_type == "people":
            base.update({
                "Headline": n.get("headline", ""),
                "Location": n.get("location", ""),
            })
        else:
            base.update({
                "Industry": n.get("industry", ""),
                "Headquarters": n.get("headquarters", ""),
                "Website": n.get("website", ""),
                "Company Size": n.get("company_size", ""),
                "Specialties": n.get("specialties", ""),
            })
        rows.append(base)

    df = pd.DataFrame(rows)
    # Put the polinode-required columns first if present
    first_cols = ["Name", "!Internal ID"]
    other_cols = [c for c in df.columns if c not in first_cols]
    return df[first_cols + other_cols]


def to_polinode_edges(edges: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in edges:
        rows.append({
            "Source": e["source"],
            "Target": e["target"],
            "Type": e.get("relationship", "EDGE"),
            "Degree": e.get("degree", ""),
        })
    return pd.DataFrame(rows)


# -----------------------------
# UI
# -----------------------------
def render_version_history() -> None:
    with st.expander("Version history", expanded=False):
        for entry in VERSION_HISTORY:
            st.markdown(f"**UPDATED {entry['version']}: {entry['title']}**")
            for b in entry["bullets"]:
                st.markdown(f"- {b}")


def main() -> None:
    st.set_page_config(page_title="ActorGraph", layout="wide")
    st.title("ActorGraph")
    st.caption(f"{APP_VERSION} • C4C Network Crawler (People + Company seeds)")

    render_version_history()

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.header("1. Upload Seed Profiles")
        st.write("Upload CSV with columns: **(people)** `name, profile_url` OR **(companies)** `org_name, linkedin_profile_url` (max 10 rows).")
        uploaded = st.file_uploader("Seed CSV", type=["csv"])

        seed_result: Optional[SeedLoadResult] = None
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                seed_result = detect_seed_type_and_load(raw_df)
            except Exception as e:
                seed_result = SeedLoadResult(False, None, None, f"Could not read CSV: {e}")

        if seed_result and not seed_result.ok:
            st.error(seed_result.error)
        elif seed_result and seed_result.ok:
            st.success(f"Loaded {len(seed_result.df)} seed rows ({seed_result.crawl_type})")
            st.dataframe(seed_result.df.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Upload a seed CSV to continue.")

    with col2:
        st.header("2. EnrichLayer API Token")
        api_token = st.text_input("Enter your API token", type="password")
        mock_mode = st.toggle("Run in mock mode (no real API calls)", value=False)

        if st.button("Test API Connection", disabled=(not api_token and not mock_mode)):
            try:
                if mock_mode:
                    st.success("Mock mode enabled — no API calls will be made.")
                else:
                    # lightweight probe
                    enrichlayer_post(PROFILE_PATH, api_token, {"linkedin_url": "https://www.linkedin.com/in/streamlit/"})
                    st.success("API request succeeded.")
            except Exception as e:
                st.error(f"API connection test failed: {e}")

    st.divider()

    st.header("Crawl Configuration")
    cfg_left, cfg_right = st.columns([1, 1], gap="large")
    with cfg_left:
        max_degree = st.radio("Maximum Degree (hops)", options=[1, 2], index=0, horizontal=False)
        st.caption("Degree 1 = direct neighbors only. Degree 2 = neighbors-of-neighbors (costs more credits).")
    with cfg_right:
        st.subheader("Crawl Limits")
        max_edges = st.number_input("Max Edges", min_value=100, max_value=50000, value=10000, step=100)
        max_nodes = st.number_input("Max Nodes", min_value=100, max_value=50000, value=7500, step=100)
        st.caption("These limits only apply to exports; the crawl may return fewer.")

    st.caption("API pacing: up to 16 requests/minute")

    can_run = bool(seed_result and seed_result.ok and (api_token or mock_mode))

    if not can_run:
        st.warning("Please upload a valid seed CSV and provide an API token (or enable mock mode) to continue.")
        st.button("🚀 Run Crawl", disabled=True)
        return

    st.divider()
    st.subheader("Run")

    if "run_state" not in st.session_state:
        st.session_state.run_state = {}

    if st.button("🚀 Run Crawl", disabled=False):
        crawl_type = seed_result.crawl_type
        seeds_df = seed_result.df

        with st.spinner("Crawling..."):
            if crawl_type == "people":
                nodes_by_url, edges_list, stats, raw_items = crawl_people(
                    seeds=seeds_df, api_token=api_token, max_degree=max_degree, mock_mode=mock_mode
                )
            else:
                nodes_by_url, edges_list, stats, raw_items = crawl_companies(
                    seeds=seeds_df, api_token=api_token, max_degree=max_degree, mock_mode=mock_mode
                )

        # Store
        st.session_state.run_state = {
            "crawl_type": crawl_type,
            "nodes_by_url": nodes_by_url,
            "edges": edges_list,
            "stats": stats,
            "raw_items": raw_items,
        }

    # Results
    run_state = st.session_state.get("run_state") or {}
    if not run_state:
        return

    crawl_type = run_state["crawl_type"]
    stats = run_state["stats"]

    # KPI tiles
    if crawl_type == "people":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Successful", stats.get("successful", 0))
        k2.metric("Failed", stats.get("failed", 0))
        k3.metric("People w/ no neighbors", stats.get("people_w_no_neighbors", 0))
        k4.metric("Edges", stats.get("edges", 0))
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Successful", stats.get("successful", 0))
        k2.metric("Failed", stats.get("failed", 0))
        k3.metric("Companies w/ no neighbors", stats.get("companies_w_no_neighbors", 0))
        k4.metric("Edges", stats.get("edges", 0))

    st.success("Completed")

    st.header("Download Results")

    nodes_df = to_polinode_nodes(run_state["nodes_by_url"], crawl_type=crawl_type)
    edges_df = to_polinode_edges(run_state["edges"])

    # Apply export caps (soft)
    nodes_df = nodes_df.head(int(max_nodes))
    edges_df = edges_df.head(int(max_edges))

    st.download_button(
        "⬇️ Download nodes (Polinode CSV)",
        data=nodes_df.to_csv(index=False).encode("utf-8"),
        file_name="nodes_polinode.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download edges (Polinode CSV)",
        data=edges_df.to_csv(index=False).encode("utf-8"),
        file_name="edges_polinode.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download crawl_log.json",
        data=json.dumps(stats, indent=2).encode("utf-8"),
        file_name="crawl_log.json",
        mime="application/json",
    )
    st.download_button(
        "⬇️ Download raw_items.json",
        data=json.dumps(run_state["raw_items"], indent=2).encode("utf-8"),
        file_name="raw_items.json",
        mime="application/json",
    )

    with st.expander("Preview exports"):
        st.subheader("Nodes (Polinode)")
        st.dataframe(nodes_df.head(50), use_container_width=True, hide_index=True)
        st.subheader("Edges (Polinode)")
        st.dataframe(edges_df.head(50), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
