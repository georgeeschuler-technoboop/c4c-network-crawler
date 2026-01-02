"""
C4C Console UI helpers (Streamlit)
Shared styling + components used across C4C Data Platform apps (OrgGraph / ActorGraph / InsightGraph).

Drop this file alongside app.py and import:
from console_ui import inject_c4c_console_theme, c4c_header, c4c_card_open, c4c_card_close, c4c_console
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
}

html, body, [data-testid="stAppViewContainer"]{
  background: var(--c4c-bg);
}

/* Safety: ensure Streamlit top toolbar/header doesn't overlap content */
[data-testid="stAppViewContainer"]{
  padding-top: 0px !important;
}

/* Tighten default padding slightly */
.block-container, [data-testid="stMainBlockContainer"]{
  padding-top: 3.5rem !important;
  padding-bottom: 2rem !important;
}

/* Inputs — higher contrast + clear focus affordance */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{
  background: #ffffff !important;
  border: 1px solid #d0d5dd !important;
  border-radius: 10px !important;
  padding: 0.55rem 0.75rem !important;
}

div[data-baseweb="select"] > div{
  background: #ffffff !important;
  border: 1px solid #d0d5dd !important;
  border-radius: 10px !important;
}

div[data-baseweb="input"]:focus-within input,
div[data-baseweb="textarea"]:focus-within textarea{
  border-color: #5b5ce2 !important;
  box-shadow: 0 0 0 2px rgba(91, 92, 226, 0.18) !important;
}

div[data-baseweb="select"]:focus-within > div{
  border-color: #5b5ce2 !important;
  box-shadow: 0 0 0 2px rgba(91, 92, 226, 0.18) !important;
}

/* App header */

.c4c-header{
  display:flex;
  gap:14px;
  align-items:center;
  margin: 0.25rem 0 1.0rem 0;
}
.c4c-header .logo{
  width: 52px;
  height: 52px;
  border-radius: 14px;
  border: 1px solid var(--c4c-border);
  background: var(--c4c-card);
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}
.c4c-header .title{
  font-size: 1.65rem;
  font-weight: 700;
  color: var(--c4c-ink);
  line-height: 1.1;
}
.c4c-header .subtitle{
  margin-top: 2px;
  color: var(--c4c-muted);
  font-size: 0.95rem;
}

/* Cards */
.c4c-card{
  background: var(--c4c-card);
  border: 1px solid var(--c4c-border);
  border-radius: 16px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  margin-bottom: 14px;
}
.c4c-card.primary{
  border-left: 4px solid var(--c4c-indigo);
  padding-left: 14px;
}
.c4c-card .card-title{
  font-weight: 700;
  color: var(--c4c-ink);
  margin-bottom: 6px;
}
.c4c-card .card-subtitle{
  color: var(--c4c-muted);
  font-size: 0.92rem;
  margin-bottom: 10px;
}
.c4c-kv{
  display:flex;
  flex-wrap:wrap;
  gap:10px 16px;
  color: var(--c4c-muted);
  font-size: 0.9rem;
}
.c4c-pill{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--c4c-border);
  background: rgba(255,255,255,0.6);
  color: var(--c4c-muted);
}

/* Console */
.c4c-console{
  background: #0b1020;
  border: 1px solid rgba(203,213,245,0.15);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.20);
  margin-bottom: 14px;
}
.c4c-console .card-title{
  color: #e6e9ff;
  font-weight: 700;
  margin-bottom: 6px;
}
.c4c-console pre{
  white-space: pre-wrap;
  color: #cbd5f5;
  margin: 0;
  font-size: 0.86rem;
  line-height: 1.35;
}

/* Buttons */
.stDownloadButton button, .stButton button{
  border-radius: 12px !important;
}

/* Reduce visual noise of horizontal rule */
hr{
  border-top: 1px solid var(--c4c-border) !important;
}

/* Sub-cards inside stages (used for Region + other grouped UI) */
.c4c-subcard{
  background: var(--c4c-card);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 16px;
  padding: 14px 16px;
  margin: 10px 0 14px 0;
  box-shadow: 0 6px 18px rgba(15,23,42,0.04);
}
.c4c-subcard-hd{
  display:flex;
  align-items:baseline;
  justify-content:space-between;
  margin-bottom:8px;
}
.c4c-subcard-title{
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--c4c-ink);
}
.c4c-subcard-sub{
  font-size: 0.85rem;
  color: rgba(15,23,42,0.55);
  padding: 2px 10px;
  border: 1px solid rgba(15,23,42,0.12);
  border-radius: 999px;
  background: rgba(248,250,252,0.9);
}
.c4c-subcard-body{ margin-top: 6px; }

</style>"""


def inject_c4c_console_theme(extra_css: str = "") -> None:
    """
    Inject the shared C4C console UI theme CSS.
    Call once near the top of the app (right after st.set_page_config).
    """
    st.markdown(
        _CSS + (f"\n<style>\n{{extra_css}}\n</style>" if extra_css else ""),
        unsafe_allow_html=True
    )


def c4c_header(title: str, subtitle: str = "", icon_url: Optional[str] = None, right_html: str = "") -> None:
    """Render the console-style header."""
    logo_html = (
        f"<img src='{escape(icon_url)}' style='width:42px;height:42px;object-fit:contain;'/>" if icon_url else ""
    )
    st.markdown(
        f"""
<div class='c4c-header'>
  <div class='logo'>{logo_html}</div>
  <div style='flex:1;min-width:0;'>
    <div class='title'>{escape(title)}</div>
    <div class='subtitle'>{escape(subtitle) if subtitle else ''}</div>
  </div>
  <div class='header-right'>{right_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def c4c_badge(text: str, tone: str = "default") -> str:
    """Return a badge HTML string.

    tone in: default, indigo, amber, terracotta, teal, success, warn, danger.
    """
    t = tone.strip().lower()
    return f"<span class='c4c-badge tone-{escape(t)}'>{escape(text)}</span>"


def c4c_card_open(title: str, subtitle: str = "", variant: str = "") -> None:
    """Open a card container. Remember to call c4c_card_close().

    variant examples: 'primary', 'console', 'muted'
    """
    v = variant.strip().lower()
    cls = "c4c-card" + (f" variant-{escape(v)}" if v else "")
    sub = f"<div class='card-subtitle'>{escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"""
<div class='{cls}'>
  <div class='card-title'>{escape(title)}</div>
  {sub}
  <div class='card-body'>
""",
        unsafe_allow_html=True,
    )


def c4c_card_close() -> None:
    st.markdown("</div></div>", unsafe_allow_html=True)

def c4c_console(title: str, lines: Sequence[str] | Iterable[str]) -> None:
    """Render a console-style log block."""
    safe_lines = "\n".join(escape(str(x)) for x in lines)
    st.markdown(
        f"""
<div class='c4c-console'>
  <div class='card-title'>{escape(title)}</div>
  <pre>{safe_lines}</pre>
</div>
""",
        unsafe_allow_html=True,
    )
