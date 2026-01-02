"""
C4C UI helpers (Streamlit)
Shared styling + components used across C4C Data Platform apps (OrgGraph / ActorGraph / InsightGraph).

Focus: clean "Lab Console" vibe + carded, stage-based pipeline layout.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Optional, Sequence

import streamlit as st


_CSS = r"""<style>
:root{
  --c4c-indigo:#2825be;
  --c4c-amber:#eb9001;
  --c4c-terracotta:#cf4c38;
  --c4c-teal:#0c7a7a;
  --c4c-ink:#0f172a;
  --c4c-muted:#64748b;
  --c4c-border:#e5e7eb;
  --c4c-card:#ffffff;
  --c4c-bg:#f8fafc;
  --c4c-soft:#eef2ff;
}

html, body, [data-testid="stAppViewContainer"]{
  background: var(--c4c-bg);
}

/* Safety: ensure Streamlit top toolbar/header doesn't overlap content */
[data-testid="stAppViewContainer"]{
  padding-top: 0px !important;
}

/* Main container padding (robust across Streamlit versions) */
.block-container, [data-testid="stMainBlockContainer"]{
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
}

/* Top utility bar */
.c4c-topbar{
  background: var(--c4c-card);
  border: 1px solid var(--c4c-border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  margin: 0.25rem 0 1.0rem 0;
}
.c4c-topbar .row{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
}
.c4c-topbar .left{
  display:flex;
  align-items:center;
  gap:14px;
  min-width:0;
}
.c4c-topbar .logo{
  width: 44px;
  height: 44px;
  border-radius: 14px;
  border: 1px solid var(--c4c-border);
  background: var(--c4c-card);
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
}
.c4c-topbar .title{
  font-size: 1.35rem;
  font-weight: 750;
  color: var(--c4c-ink);
  line-height: 1.1;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.c4c-topbar .subtitle{
  margin-top: 3px;
  color: var(--c4c-muted);
  font-size: 0.92rem;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.c4c-topbar .right{
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
  justify-content:flex-end;
}
.c4c-pill{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid var(--c4c-border);
  background: rgba(255,255,255,0.75);
  color: var(--c4c-muted);
  font-size: 0.86rem;
}
.c4c-pill strong{
  color: var(--c4c-ink);
  font-weight: 650;
}

/* Stage cards */
.c4c-card{
  background: var(--c4c-card);
  border: 1px solid var(--c4c-border);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  margin-bottom: 14px;
}
.c4c-card .card-head{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom: 10px;
}
.c4c-card .card-title{
  font-weight: 750;
  color: var(--c4c-ink);
}
.c4c-card .card-subtitle{
  color: var(--c4c-muted);
  font-size: 0.92rem;
  margin-top: 3px;
}
.c4c-stage{
  display:flex;
  align-items:center;
  gap:10px;
}
.c4c-stage .stage-label{
  font-size: 0.86rem;
  font-weight: 650;
  color: var(--c4c-muted);
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--c4c-border);
  background: var(--c4c-soft);
}
.c4c-status{
  font-size: 0.86rem;
  font-weight: 650;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--c4c-border);
  background: rgba(255,255,255,0.75);
  color: var(--c4c-muted);
}
.c4c-status.complete{ border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.08); color: #166534; }
.c4c-status.active{ border-color: rgba(40,37,190,0.35); background: rgba(40,37,190,0.07); color: #1e1b4b; }
.c4c-status.idle{}

/* Buttons */
.stDownloadButton button, .stButton button{
  border-radius: 12px !important;
}

/* Reduce visual noise of horizontal rule */
hr{
  border-top: 1px solid var(--c4c-border) !important;
}
</style>"""


def inject_c4c_theme(extra_css: str = "") -> None:
    """Inject shared C4C theme CSS. Call once after st.set_page_config()."""
    st.markdown(
        _CSS + (f"\n<style>\n{extra_css}\n</style>" if extra_css else ""),
        unsafe_allow_html=True
    )


def c4c_topbar(
    title: str,
    subtitle: str = "",
    icon_url: Optional[str] = None,
    right_pills: Sequence[str] | None = None,
) -> None:
    """Render the top utility bar. right_pills are HTML strings (already escaped)."""
    logo_html = (
        f"<img src='{escape(icon_url)}' style='width:34px;height:34px;object-fit:contain;'/>" if icon_url else ""
    )
    pills_html = "".join(right_pills or [])
    st.markdown(
        f"""
<div class='c4c-topbar'>
  <div class='row'>
    <div class='left'>
      <div class='logo'>{logo_html}</div>
      <div style='min-width:0;'>
        <div class='title'>{escape(title)}</div>
        <div class='subtitle'>{escape(subtitle) if subtitle else ''}</div>
      </div>
    </div>
    <div class='right'>{pills_html}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def c4c_pill(label: str, value: str = "", icon: str = "") -> str:
    """Return a pill HTML string."""
    icon_html = f"<span>{escape(icon)}</span>" if icon else ""
    value_html = f"<strong>{escape(value)}</strong>" if value else ""
    mid = " " if (value_html and label) else ""
    return f"<span class='c4c-pill'>{icon_html}{escape(label)}{mid}{value_html}</span>"


def c4c_stage_open(stage_num: int, title: str, status: str = "idle", subtitle: str = "") -> None:
    """
    Open a stage card. Call c4c_stage_close().

    status in: idle, active, complete
    """
    status_norm = (status or "idle").strip().lower()
    status_text = {"idle": "○ Not started", "active": "● Active", "complete": "✓ Complete"}.get(status_norm, status_norm)
    sub_html = f"<div class='card-subtitle'>{escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"""
<div class='c4c-card'>
  <div class='card-head'>
    <div class='c4c-stage'>
      <span class='stage-label'>Stage {stage_num}</span>
      <div>
        <div class='card-title'>{escape(title)}</div>
        {sub_html}
      </div>
    </div>
    <span class='c4c-status {escape(status_norm)}'>{escape(status_text)}</span>
  </div>
  <div class='card-body'>
""",
        unsafe_allow_html=True,
    )


def c4c_stage_close() -> None:
    st.markdown("</div></div>", unsafe_allow_html=True)
