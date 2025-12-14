import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz

SEARCHAPI_ENDPOINT = "https://www.searchapi.io/api/v1/search"

# ---------------------------
# App metadata
# ---------------------------

APP_NAME = "Resolver"
APP_VERSION = "0.5.3"  # bump whenever query/scoring/output logic changes

# ---------------------------
# Page config with icon
# ---------------------------

ICON_URL = "https://static.wixstatic.com/media/275a3f_2ff6958b542640a7970937a336f883f1~mv2.png"

st.set_page_config(
    page_title="Resolver | LinkedIn Profile URL Resolver",
    page_icon=ICON_URL,
    layout="wide",
)

# ---------------------------
# Session state for persistence
# ---------------------------

if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "show_success" not in st.session_state:
    st.session_state.show_success = False

# ---------------------------
# Header with logo and title
# ---------------------------

col_logo, col_title = st.columns([0.08, 0.92], vertical_alignment="center")
with col_logo:
    st.image(ICON_URL, width=50)
with col_title:
    st.markdown(
        f"""
        <div>
            <span style="font-size: 2rem; font-weight: 700;">Resolver</span>
            <span style="font-size: 1rem; color: #666; margin-left: 0.5rem;">v{APP_VERSION}</span>
        </div>
        <div style="font-size: 1rem; color: #888; margin-top: -0.25rem;">
            LinkedIn Profile URL Resolver (CSV → CSV)
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------
# Guidance info box
# ---------------------------

st.info(
    """
**How to prepare your CSV:**

1. **Required columns:** `full_name`, `country`
2. **Recommended (sweet spot for accuracy):** `city` and/or `state` — these disambiguate common names far better than country alone
3. **Helpful but less reliable:** `company` or `organization`, `title` — only helps if it matches what's currently on their LinkedIn profile

*Also accepts: `province`, `state_province`, `metro`, `metro_area`, `metro_region`*
""",
    icon="📋",
)

# ---------------------------
# API key handling (manual first, secrets fallback)
# ---------------------------

st.sidebar.header("Settings")

api_key_manual = st.sidebar.text_input(
    "SearchAPI.io API key",
    type="password",
    help="Paste your SearchAPI.io key here for this session. If blank, the app will try Streamlit Secrets.",
)

# API Search Counter
st.sidebar.markdown("---")
st.sidebar.markdown("**API Searches**")

# Initialize the widget key if not exists
if "api_counter_input" not in st.session_state:
    st.session_state.api_counter_input = 0

api_count = st.sidebar.number_input(
    "Total searches",
    min_value=0,
    step=1,
    key="api_counter_input",
    help="Tracks cumulative API calls. Set to your dashboard value to sync, or reset to 0.",
    label_visibility="collapsed",
)

if st.sidebar.button("Reset to 0", use_container_width=True):
    st.session_state.api_counter_input = 0
    st.rerun()

st.sidebar.markdown("---")

api_key = api_key_manual.strip() if api_key_manual else ""

if not api_key:
    try:
        api_key = st.secrets["SEARCHAPI_API_KEY"].strip()
    except Exception:
        pass
    
    # Fallback to old key name for backwards compatibility
    if not api_key:
        try:
            api_key = st.secrets["SERPAPI_API_KEY"].strip()
        except Exception:
            api_key = ""

if not api_key:
    st.warning(
        "No SearchAPI.io API key provided.\n\n"
        "• Paste one into the sidebar, or\n"
        "• Add SEARCHAPI_API_KEY to Streamlit Secrets"
    )
    st.stop()

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
    if loc:
        q1_parts.append(f'"{loc}"')
    if country:
        q1_parts.append(f'"{country}"')
    if company:
        q1_parts.append(f'"{company}"')
    if title:
        q1_parts.append(f'"{title}"')
    q1 = " ".join(q1_parts)

    q2_parts = [f'"{name}"', "site:linkedin.com/in"]
    if loc:
        q2_parts.append(f'"{loc}"')
    if country:
        q2_parts.append(f'"{country}"')
    q2 = " ".join(q2_parts)

    q3_parts = [f'"{name}"', "site:linkedin.com/in"]
    if country:
        q3_parts.append(f'"{country}"')
    if company:
        q3_parts.append(f'"{company}"')
    q3 = " ".join(q3_parts)

    return [q1, q2, q3]


def fetch_searchapi_results(api_key: str, q: str, num: int = 10) -> Dict:
    params = {
        "engine": "google",
        "q": q,
        "num": num,
        "api_key": api_key,
    }
    r = requests.get(SEARCHAPI_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    
    # Increment API search counter (use widget key directly)
    if "api_counter_input" in st.session_state:
        st.session_state.api_counter_input += 1
    
    return r.json()


def is_linkedin_profile_url(url: str) -> bool:
    u = (url or "").lower()
    return ("linkedin.com/in/" in u) or ("linkedin.com/pub/" in u)


def score_candidate(row: Dict[str, str], cand_title: str, cand_snippet: str, url: str) -> float:
    name = normalize_text(row.get("full_name"))
    country = normalize_text(row.get("country"))
    city = normalize_text(row.get("city"))
    state = normalize_text(row.get("state_province"))
    metro = normalize_text(row.get("metro_region"))
    company = normalize_text(row.get("company"))
    title = normalize_text(row.get("title"))

    hay = f"{cand_title} {cand_snippet}".strip()

    # Name similarity: use token sort ratio against result title
    name_fuzz_score = fuzz.token_sort_ratio(name, cand_title) / 100.0
    
    # Name substring bonus: if full name appears in title, don't penalize for extra words
    # This prevents "George Schuler" matching better than "George Schuler - Global Sustainability Executive"
    name_in_title = 1.0 if name.lower() in cand_title.lower() else 0.0
    name_score = max(name_fuzz_score, name_in_title * 0.95)  # Substring match = 95% score

    # Location hits
    loc_terms = [t for t in [metro, city, state, country] if t]
    loc_hits = sum(1 for t in loc_terms if t.lower() in hay.lower())
    loc_score = min(loc_hits / max(len(loc_terms), 1), 1.0) if loc_terms else 0.0

    # Company/title hits
    biz_terms = [t for t in [company, title] if t]
    biz_hits = sum(1 for t in biz_terms if t.lower() in hay.lower())
    biz_score = min(biz_hits / max(len(biz_terms), 1), 1.0) if biz_terms else 0.0
    
    # Bonus: exact company name match in snippet (strong disambiguation signal)
    company_bonus = 0.0
    if company and company.lower() in hay.lower():
        company_bonus = 0.10  # Significant boost when company name found

    # URL quality bonus (vanity URLs slightly preferred)
    url_bonus = 0.05 if "/in/" in (url or "").lower() else 0.0

    # Weighted blend (rebalanced: less name weight, more biz weight)
    return (0.50 * name_score) + (0.25 * loc_score) + (0.25 * biz_score) + company_bonus + url_bonus


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
# Main UI
# ---------------------------

uploaded = st.file_uploader("Upload input CSV", type=["csv"])

# Detect new file upload and clear old results
if uploaded is not None:
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.last_uploaded_file != file_id:
        st.session_state.last_uploaded_file = file_id
        st.session_state.results_df = None
        st.session_state.summary_data = None
        st.session_state.show_success = False

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    max_candidates = st.slider("Keep top candidates", 1, 10, 5)
with colB:
    per_row_pause = st.slider("Pause per query (sec)", 0.0, 1.0, 0.2, 0.1)
with colC:
    st.caption("💡 Start small (25–100 rows) to tune thresholds and avoid burning API credits.")

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Normalize column names: accept common aliases
    # organization → company
    if "organization" in df.columns and "company" not in df.columns:
        df["company"] = df["organization"]
    
    # state or province → state_province
    if "state_province" not in df.columns:
        if "state" in df.columns:
            df["state_province"] = df["state"]
        elif "province" in df.columns:
            df["state_province"] = df["province"]
    
    # metro or metro_area → metro_region
    if "metro_region" not in df.columns:
        if "metro" in df.columns:
            df["metro_region"] = df["metro"]
        elif "metro_area" in df.columns:
            df["metro_region"] = df["metro_area"]
    
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Input CSV missing required columns: {missing}")
        st.stop()

    # Ensure optional cols exist
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""

    if st.button("🔍 Resolve LinkedIn URLs", type="primary"):
        out_rows = []
        progress = st.progress(0)
        status = st.empty()

        total = len(df)
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            queries = build_queries(row_dict)

            all_candidates: List[Candidate] = []
            query_used = ""

            for q in queries:
                query_used = q
                try:
                    data = fetch_searchapi_results(api_key, q, num=10)
                except Exception as e:
                    status.error(f"Row {i}: SearchAPI error: {e}")
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
            by_url: Dict[str, Candidate] = {}
            for c in all_candidates:
                if c.url not in by_url or c.score > by_url[c.url].score:
                    by_url[c.url] = c

            uniq_candidates = sorted(by_url.values(), key=lambda c: c.score, reverse=True)[:max_candidates]

            best, conf, review = pick_best(uniq_candidates)

            out_rows.append(
                {
                    "id": row_dict.get("id", i),
                    "full_name": row_dict.get("full_name"),
                    "country": row_dict.get("country"),
                    "linkedin_url_best": best.url if best else "",
                    "confidence": conf,
                    "review_flag": review,
                    "resolver_version": APP_VERSION,
                    "query_used": query_used or "",
                    "candidates_json": json.dumps(
                        [
                            {
                                "url": c.url,
                                "title": c.title,
                                "snippet": c.snippet,
                                "score": round(c.score, 3),
                            }
                            for c in uniq_candidates
                        ],
                        ensure_ascii=False,
                    ),
                }
            )

            progress.progress(int(((i + 1) / total) * 100))
            status.write(f"Processed {i + 1}/{total}")

        out_df = pd.DataFrame(out_rows)
        
        # Store in session state for persistence
        st.session_state.results_df = out_df
        st.session_state.show_success = True
        
        # Rerun to refresh sidebar counter display
        st.rerun()

# -------------------------------------------------
# Display Results (from session state - persists across reruns)
# -------------------------------------------------

if st.session_state.results_df is not None:
    out_df = st.session_state.results_df
    
    # Show success message only once after a fresh run
    if st.session_state.show_success:
        st.success("✅ Resolution complete!")
        st.session_state.show_success = False
    
    # -------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Run Summary")
    
    total_names = len(out_df)
    matches_found = len(out_df[out_df["linkedin_url_best"] != ""])
    no_matches = total_names - matches_found
    needs_review = len(out_df[out_df["review_flag"] == True])
    auto_accept = matches_found - len(out_df[(out_df["linkedin_url_best"] != "") & (out_df["review_flag"] == True)])
    avg_confidence = out_df[out_df["confidence"] > 0]["confidence"].mean() if matches_found > 0 else 0
    
    # Confidence bins
    high_conf = len(out_df[out_df["confidence"] > 0.7])
    med_conf = len(out_df[(out_df["confidence"] >= 0.5) & (out_df["confidence"] <= 0.7)])
    low_conf = len(out_df[(out_df["confidence"] > 0) & (out_df["confidence"] < 0.5)])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Names Processed", total_names)
    with col2:
        st.metric("Matches Found", f"{matches_found} ({100*matches_found//total_names}%)")
    with col3:
        st.metric("No Match", no_matches)
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}" if avg_confidence else "N/A")
    
    # Quality Distribution
    st.markdown("**Quality Distribution**")
    qual_col1, qual_col2, qual_col3 = st.columns(3)
    with qual_col1:
        st.markdown(f"🟢 **High** (>0.7): **{high_conf}**")
    with qual_col2:
        st.markdown(f"🟡 **Medium** (0.5–0.7): **{med_conf}**")
    with qual_col3:
        st.markdown(f"🔴 **Low** (<0.5): **{low_conf}**")
    
    # By Country Performance
    if "country" in out_df.columns:
        st.markdown("**Match Rate by Country**")
        country_stats = out_df.groupby("country").agg(
            total=("full_name", "count"),
            matched=("linkedin_url_best", lambda x: (x != "").sum()),
            avg_conf=("confidence", lambda x: x[x > 0].mean() if (x > 0).any() else 0)
        ).reset_index()
        country_stats["match_rate"] = (country_stats["matched"] / country_stats["total"] * 100).round(1)
        country_stats["avg_conf"] = (country_stats["avg_conf"] * 100).round(1)
        country_stats.columns = ["Country", "Total", "Matched", "Avg Conf %", "Match Rate %"]
        country_stats = country_stats.sort_values("Match Rate %", ascending=True)
        st.dataframe(country_stats, use_container_width=True, hide_index=True)
    
    # No Match List
    no_match_df = out_df[out_df["linkedin_url_best"] == ""][["full_name", "country"]]
    if len(no_match_df) > 0:
        with st.expander(f"🔍 No Match — Manual Lookup Needed ({len(no_match_df)})", expanded=False):
            for _, row in no_match_df.iterrows():
                st.write(f"• {row['full_name']} ({row['country']})")
    
    # Too Close to Call
    too_close_rows = []
    for _, row in out_df.iterrows():
        if row["linkedin_url_best"] and row["candidates_json"]:
            try:
                cands = json.loads(row["candidates_json"])
                if len(cands) >= 2:
                    top_score = cands[0]["score"]
                    second_score = cands[1]["score"]
                    if (top_score - second_score) < 0.08 and top_score < 0.75:
                        too_close_rows.append({
                            "name": row["full_name"],
                            "country": row["country"],
                            "picked": cands[0]["title"],
                            "picked_score": round(top_score, 3),
                            "runner_up": cands[1]["title"],
                            "runner_up_score": round(second_score, 3),
                        })
            except:
                pass
    
    if too_close_rows:
        with st.expander(f"⚠️ Too Close to Call! ({len(too_close_rows)})", expanded=False):
            st.caption("These matches have very similar scores — verify manually.")
            for item in too_close_rows:
                st.markdown(f"**{item['name']}** ({item['country']})")
                st.markdown(f"  - Picked: {item['picked']} ({item['picked_score']})")
                st.markdown(f"  - Runner-up: {item['runner_up']} ({item['runner_up_score']})")
                st.markdown("---")
    
    # -------------------------------------------------
    # Generate Summary Report (Markdown)
    # -------------------------------------------------
    report_lines = [
        f"# Resolver Run Summary",
        f"**Version:** {APP_VERSION}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        f"- **Total Names Processed:** {total_names}",
        f"- **Matches Found:** {matches_found} ({100*matches_found//total_names}%)",
        f"- **No Match:** {no_matches}",
        f"- **Average Confidence:** {avg_confidence:.1%}" if avg_confidence else "- **Average Confidence:** N/A",
        "",
        "## Quality Distribution",
        f"- 🟢 **High** (>0.7): {high_conf}",
        f"- 🟡 **Medium** (0.5–0.7): {med_conf}",
        f"- 🔴 **Low** (<0.5): {low_conf}",
        "",
    ]
    
    if "country" in out_df.columns:
        report_lines.append("## Match Rate by Country")
        report_lines.append("| Country | Total | Matched | Match Rate | Avg Conf |")
        report_lines.append("|---------|-------|---------|------------|----------|")
        for _, row in country_stats.iterrows():
            report_lines.append(f"| {row['Country']} | {row['Total']} | {row['Matched']} | {row['Match Rate %']}% | {row['Avg Conf %']}% |")
        report_lines.append("")
    
    if len(no_match_df) > 0:
        report_lines.append("## No Match — Manual Lookup Needed")
        for _, row in no_match_df.iterrows():
            report_lines.append(f"- {row['full_name']} ({row['country']})")
        report_lines.append("")
    
    if too_close_rows:
        report_lines.append("## Too Close to Call!")
        report_lines.append("These matches have very similar scores — verify manually.")
        report_lines.append("")
        for item in too_close_rows:
            report_lines.append(f"### {item['name']} ({item['country']})")
            report_lines.append(f"- **Picked:** {item['picked']} ({item['picked_score']})")
            report_lines.append(f"- **Runner-up:** {item['runner_up']} ({item['runner_up_score']})")
            report_lines.append("")
    
    summary_report = "\n".join(report_lines)
    
    # -------------------------------------------------
    # Downloads
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Downloads")
    
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    summary_bytes = summary_report.encode("utf-8")
    
    # Create ZIP with both files
    import io
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("linkedin_resolved_output.csv", csv_bytes)
        zf.writestr("resolver_summary.md", summary_bytes)
    zip_bytes = zip_buffer.getvalue()
    
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            "📦 Download All (ZIP)",
            data=zip_bytes,
            file_name="resolver_results.zip",
            mime="application/zip",
        )
    with dl_col2:
        st.download_button(
            "📄 Download CSV",
            data=csv_bytes,
            file_name="linkedin_resolved_output.csv",
            mime="text/csv",
        )
    with dl_col3:
        st.download_button(
            "📋 Download Summary",
            data=summary_bytes,
            file_name="resolver_summary.md",
            mime="text/markdown",
        )
    
    # -------------------------------------------------
    # Results table with color-coded confidence
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Results")
    
    # Color coding function for confidence
    def color_confidence(val):
        if pd.isna(val) or val == 0:
            return ""
        elif val > 0.7:
            return "background-color: #90EE90"  # Light green
        elif val >= 0.5:
            return "background-color: #FFEB99"  # Light yellow
        else:
            return "background-color: #FFB6B6"  # Light red/pink
    
    # Apply styling with color-coded confidence (applymap for pandas compatibility)
    try:
        styled_df = out_df.style.map(color_confidence, subset=["confidence"])
    except AttributeError:
        styled_df = out_df.style.applymap(color_confidence, subset=["confidence"])
    styled_df = styled_df.format({"confidence": "{:.1%}"})
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
