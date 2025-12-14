import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# ---------------------------
# Scoring + query logic
# ---------------------------

REQUIRED_COLS = ["full_name", "country"]
OPTIONAL_COLS = ["city", "state_province", "metro_region", "company", "title"]

@dataclass
class Candidate:
    url: str
    title: str
    snippet: str
    score: float

def normalize_text(x: Optional[str]) -> str:
    return (x or "").strip()

def build_queries(row: Dict[str, str]) -> List[str]:
    name = normalize_text(row.get("full_name"))
    country = normalize_text(row.get("country"))
    city = normalize_text(row.get("city"))
    state = normalize_text(row.get("state_province"))
    metro = normalize_text(row.get("metro_region"))
    company = normalize_text(row.get("company"))
    title = normalize_text(row.get("title"))

    loc = " ".join([p for p in [metro, city, state] if p]).strip()

    # 3 passes: specific -> medium -> fallback
    q1_parts = [f'"{name}"', "site:linkedin.com/in"]
    if loc: q1_parts.append(f'"{loc}"')
    if country: q1_parts.append(f'"{country}"')
    if company: q1_parts.append(f'"{company}"')
    if title: q1_parts.append(f'"{title}"')
    q1 = " ".join(q1_parts)

    q2_parts = [f'"{name}"', "site:linkedin.com/in"]
    if loc: q2_parts.append(f'"{loc}"')
    if country: q2_parts.append(f'"{country}"')
    q2 = " ".join(q2_parts)

    q3_parts = [f'"{name}"', "site:linkedin.com/in"]
    if country: q3_parts.append(f'"{country}"')
    if company: q3_parts.append(f'"{company}"')
    q3 = " ".join(q3_parts)

    return [q1, q2, q3]

def fetch_serpapi_results(api_key: str, q: str, num: int = 10) -> Dict:
    params = {
        "engine": "google",
        "q": q,
        "num": num,
        "api_key": api_key,
    }
    r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def is_linkedin_profile_url(url: str) -> bool:
    url = url.lower()
    return ("linkedin.com/in/" in url) or ("linkedin.com/pub/" in url)

def score_candidate(row: Dict[str, str], cand_title: str, cand_snippet: str, url: str) -> float:
    name = normalize_text(row.get("full_name"))
    country = normalize_text(row.get("country"))
    city = normalize_text(row.get("city"))
    state = normalize_text(row.get("state_province"))
    metro = normalize_text(row.get("metro_region"))
    company = normalize_text(row.get("company"))
    title = normalize_text(row.get("title"))

    hay = f"{cand_title} {cand_snippet}".strip()

    # Name similarity: use token sort
    name_score = fuzz.token_sort_ratio(name, cand_title) / 100.0

    # Location hits
    loc_terms = [t for t in [metro, city, state, country] if t]
    loc_hits = sum(1 for t in loc_terms if t.lower() in hay.lower())
    loc_score = min(loc_hits / max(len(loc_terms), 1), 1.0) if loc_terms else 0.0

    # Company/title hits
    biz_terms = [t for t in [company, title] if t]
    biz_hits = sum(1 for t in biz_terms if t.lower() in hay.lower())
    biz_score = min(biz_hits / max(len(biz_terms), 1), 1.0) if biz_terms else 0.0

    # URL quality bonus (vanity URLs slightly preferred)
    url_bonus = 0.05 if "/in/" in url.lower() else 0.0

    # Weighted blend
    return (0.65 * name_score) + (0.25 * loc_score) + (0.10 * biz_score) + url_bonus

def pick_best(cands: List[Candidate]) -> Tuple[Optional[Candidate], float, bool]:
    if not cands:
        return None, 0.0, True

    cands_sorted = sorted(cands, key=lambda c: c.score, reverse=True)
    best = cands_sorted[0]
    second = cands_sorted[1] if len(cands_sorted) > 1 else None

    # Confidence heuristic: best relative to runner-up
    eps = 1e-6
    conf = best.score / (best.score + (second.score if second else 0.0) + eps)

    # Review rules (tune these as you go)
    review = (best.score < 0.55) or (conf < 0.75) or (second and (best.score - second.score) < 0.08)
    return best, float(round(conf, 3)), review

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="LinkedIn URL Resolver (CSV → CSV)", layout="wide")
st.title("LinkedIn Profile URL Resolver (CSV → CSV)")

st.caption("Upload a CSV with at least: full_name, country. Strongly recommended: city/state_province/metro_region. "
           "Returns best LinkedIn URL + confidence + review flag + top candidates.")

api_key = st.secrets.get("SERPAPI_API_KEY") if hasattr(st, "secrets") else None
if not api_key:
    st.warning("Missing SERPAPI_API_KEY. Add it in Streamlit Secrets before running.")
    st.stop()

uploaded = st.file_uploader("Upload input CSV", type=["csv"])

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    max_candidates = st.slider("Keep top candidates", 1, 10, 5)
with colB:
    per_row_pause = st.slider("Pause per query (sec)", 0.0, 1.0, 0.2, 0.1)
with colC:
    st.write("Tip: start small (25–100 rows) to tune thresholds and avoid burning searches.")

if uploaded is not None:
    df = pd.read_csv(uploaded)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Input CSV missing required columns: {missing}")
        st.stop()

    # Ensure optional cols exist
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""

    if st.button("Resolve LinkedIn URLs"):
        out_rows = []
        progress = st.progress(0)
        status = st.empty()

        total = len(df)
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            queries = build_queries(row_dict)

            all_candidates: List[Candidate] = []
            query_used = None

            for q in queries:
                query_used = q
                try:
                    data = fetch_serpapi_results(api_key, q, num=10)
                except Exception as e:
                    status.error(f"Row {i}: SerpApi error: {e}")
                    continue

                organic = data.get("organic_results", []) or []
                for r in organic:
                    url = r.get("link") or ""
                    if not url or not is_linkedin_profile_url(url):
                        continue
                    title = r.get("title") or ""
                    snippet = r.get("snippet") or ""
                    s = score_candidate(row_dict, title, snippet, url)
                    all_candidates.append(Candidate(url=url, title=title, snippet=snippet, score=s))

                # If we already have decent candidates, stop early
                all_candidates = sorted(all_candidates, key=lambda c: c.score, reverse=True)
                if all_candidates and all_candidates[0].score >= 0.70:
                    break

                if per_row_pause > 0:
                    time.sleep(per_row_pause)

            # Deduplicate by URL, keep best score per URL
            by_url = {}
            for c in all_candidates:
                if c.url not in by_url or c.score > by_url[c.url].score:
                    by_url[c.url] = c
            uniq_candidates = sorted(by_url.values(), key=lambda c: c.score, reverse=True)[:max_candidates]

            best, conf, review = pick_best(uniq_candidates)

            out_rows.append({
                "id": row_dict.get("id", i),
                "full_name": row_dict.get("full_name"),
                "country": row_dict.get("country"),
                "linkedin_url_best": best.url if best else "",
                "confidence": conf,
                "review_flag": review,
                "query_used": query_used or "",
                "candidates_json": json.dumps(
                    [{"url": c.url, "title": c.title, "snippet": c.snippet, "score": round(c.score, 3)} for c in uniq_candidates],
                    ensure_ascii=False
                ),
            })

            progress.progress(int(((i + 1) / total) * 100))
            status.write(f"Processed {i + 1}/{total}")

        out_df = pd.DataFrame(out_rows)

        st.success("Done.")
        st.dataframe(out_df, use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download output CSV",
            data=csv_bytes,
            file_name="linkedin_resolved_output.csv",
            mime="text/csv",
        )
