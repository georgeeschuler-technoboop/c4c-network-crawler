#!/usr/bin/env python3
"""
c4c_utils/lint_report.py — Validates InsightGraph HTML reports for quality regressions.

Part of the C4C Report Authoring Contract (see docs/REPORT_AUTHORING_GUIDE.md).

Usage (CLI):
    python -m c4c_utils.lint_report path/to/report.html
    python -m c4c_utils.lint_report path/to/report.html --require-signal-level
    python -m c4c_utils.lint_report path/to/report.html --require-dl-per-section
    python -m c4c_utils.lint_report path/to/report.html --require-dl-per-section --require-signal-level-per-section

Usage (as module):
    from c4c_utils.lint_report import lint_html, LintIssue
    issues = lint_html(html_content, enforce_signal_level=True)

Returns exit code 0 if OK, 1 if issues found.

VERSION HISTORY:
----------------
v1.2 (2025-12-22): Per-section Decision Lens + signal intensity enforcement
    - NEW: --require-dl-per-section flag
    - NEW: --require-signal-level-per-section flag
    - NEW: Visible text extraction for cleaner language checks
    - NEW: Inline code backtick leakage detection
    - IMPROVED: Section extraction for per-section validation
v1.1 (2025-12-22): Added signal intensity checks, semantic inflation patterns
v1.0 (2025-12-22): Initial version with raw markdown and prescriptive language checks
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class LintIssue:
    code: str
    message: str
    context: str


# ----------------------------
# Patterns
# ----------------------------

# Raw markdown patterns that should not appear in final HTML *text nodes*
RAW_MD_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("RAW_MD_ITALIC", re.compile(r"(?<!\w)_(?!_)([^_\n]{2,})_(?!\w)")),
    ("RAW_MD_BOLD", re.compile(r"\*\*[^*\n]{2,}\*\*")),
    ("RAW_MD_CODE", re.compile(r"`[^`\n]{2,}`")),              # Inline code leakage
    ("RAW_MD_HEADER", re.compile(r"(?m)^(#{1,6})\s+.+$")),
    ("RAW_MD_FENCE", re.compile(r"```")),
]

# Disallowed/prescriptive language (flag for manual review)
DISALLOWED_LANGUAGE: List[Tuple[str, re.Pattern]] = [
    ("PRESCRIPTIVE_SHOULD", re.compile(r"\bshould\b", re.IGNORECASE)),
    ("PRESCRIPTIVE_MUST", re.compile(r"\bmust\b", re.IGNORECASE)),
    ("PRESCRIPTIVE_RECOMMEND", re.compile(r"\brecommend(s|ed|ation)?\b", re.IGNORECASE)),
    ("PRESCRIPTIVE_BEST_PRACTICE", re.compile(r"\bbest practice(s)?\b", re.IGNORECASE)),
    # Semantic inflation - flag for manual review
    ("SEMANTIC_INFLATION_NATURAL", re.compile(r"\bnatural (partners?|hub)\b", re.IGNORECASE)),
    ("SEMANTIC_INFLATION_CONSENSUS", re.compile(r"\bstrong consensus\b", re.IGNORECASE)),
]

# Decision Lens presence checks
DECISION_LENS_CLASS = re.compile(r'class="[^"]*\bdecision-lens\b[^"]*"', re.IGNORECASE)
DECISION_LENS_TITLE = re.compile(r">Decision Lens<", re.IGNORECASE)

# Signal intensity badge presence (your contract language)
SIGNAL_LEVEL_BADGE = re.compile(r"\b(Low|Moderate|High)-intensity signal\b", re.IGNORECASE)

# Section detection
SECTION_OPEN = re.compile(r"<section\b[^>]*>", re.IGNORECASE)
SECTION_CLOSE = re.compile(r"</section>", re.IGNORECASE)

# Remove blocks that are not "visible text"
SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]+>")


def _snippet(text: str, start: int, end: int, radius: int = 100) -> str:
    """Extract a snippet of text around a match for context."""
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    snip = text[a:b].replace("\n", " ").strip()
    if a > 0:
        snip = "..." + snip
    if b < len(text):
        snip = snip + "..."
    return snip


def _visible_text(html: str) -> str:
    """
    Very lightweight "visible text" extraction without external deps.
    Removes <script>, <style>, and HTML tags.
    """
    stripped = SCRIPT_BLOCK.sub(" ", html)
    stripped = STYLE_BLOCK.sub(" ", stripped)
    stripped = TAG.sub(" ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def _extract_sections(html: str) -> List[str]:
    """
    Extract <section>...</section> blocks.
    Assumes sections are not deeply nested (true for our report templates).
    """
    sections: List[str] = []
    idx = 0
    while True:
        m_open = SECTION_OPEN.search(html, idx)
        if not m_open:
            break
        m_close = SECTION_CLOSE.search(html, m_open.end())
        if not m_close:
            break
        block = html[m_open.start():m_close.end()]
        sections.append(block)
        idx = m_close.end()
    return sections


def lint_html(
    html: str,
    enforce_decision_lens: bool = True,
    enforce_signal_level: bool = False,
    check_language: bool = True,
    require_dl_per_section: bool = False,
    require_signal_level_per_section: bool = False,
) -> List[LintIssue]:
    """
    Lint an HTML report for quality issues.
    
    Args:
        html: The HTML content to lint
        enforce_decision_lens: Check for Decision Lens presence (global)
        enforce_signal_level: Check for signal intensity badges (global)
        check_language: Check for prescriptive/inflated language
        require_dl_per_section: Require Decision Lens in every <section>
        require_signal_level_per_section: Require signal intensity in sections with Decision Lens
    
    Returns:
        List of LintIssue objects
    """
    issues: List[LintIssue] = []

    # 1) Raw markdown artifacts: scan full HTML (since leakage is typically in text nodes)
    for code, pat in RAW_MD_PATTERNS:
        for m in pat.finditer(html):
            issues.append(LintIssue(
                code,
                f"Found raw markdown artifact ({code}).",
                _snippet(html, m.start(), m.end())
            ))

    # 2) Language checks: scan visible text only (reduces false positives in attributes)
    if check_language:
        vis = _visible_text(html)
        for code, pat in DISALLOWED_LANGUAGE:
            for m in pat.finditer(vis):
                issues.append(LintIssue(
                    code,
                    f"Disallowed or review-needed language: '{m.group(0)}'.",
                    _snippet(vis, m.start(), m.end())
                ))

    # 3) Decision Lens presence (global)
    if enforce_decision_lens:
        has_sections = bool(SECTION_OPEN.search(html))
        has_dl = bool(DECISION_LENS_CLASS.search(html)) or bool(DECISION_LENS_TITLE.search(html))
        if has_sections and not has_dl:
            issues.append(LintIssue(
                "MISSING_DECISION_LENS",
                "No Decision Lens block detected in report HTML.",
                "Expected a div with class 'decision-lens' (or literal text 'Decision Lens')."
            ))

    # 4) Decision Lens per-section (optional strict mode)
    if require_dl_per_section:
        sections = _extract_sections(html)
        if sections:
            for i, sec in enumerate(sections, start=1):
                if not (DECISION_LENS_CLASS.search(sec) or DECISION_LENS_TITLE.search(sec)):
                    issues.append(LintIssue(
                        "MISSING_DECISION_LENS_IN_SECTION",
                        f"Section #{i} has no Decision Lens block.",
                        _snippet(sec, 0, min(len(sec), 200))
                    ))

    # 5) Signal intensity badge presence (global)
    if enforce_signal_level:
        if not SIGNAL_LEVEL_BADGE.search(html):
            issues.append(LintIssue(
                "MISSING_SIGNAL_LEVEL",
                "No signal intensity labels detected in report HTML.",
                "Expected: 'Low-intensity signal', 'Moderate-intensity signal', or 'High-intensity signal'."
            ))

    # 6) Signal intensity per-section (optional strict mode)
    if require_signal_level_per_section:
        sections = _extract_sections(html)
        if sections:
            for i, sec in enumerate(sections, start=1):
                if DECISION_LENS_CLASS.search(sec) and not SIGNAL_LEVEL_BADGE.search(sec):
                    issues.append(LintIssue(
                        "MISSING_SIGNAL_LEVEL_IN_SECTION",
                        f"Section #{i} has a Decision Lens but no signal intensity label.",
                        _snippet(sec, 0, min(len(sec), 240))
                    ))

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lint a generated InsightGraph HTML report for quality regressions."
    )
    ap.add_argument("html_path", type=str, help="Path to report HTML file")

    ap.add_argument("--no-decision-lens", action="store_true",
                    help="Disable Decision Lens global presence check")
    ap.add_argument("--require-dl-per-section", action="store_true",
                    help="Require a Decision Lens inside every <section> block")

    ap.add_argument("--require-signal-level", action="store_true",
                    help="Require signal intensity labels somewhere in the report")
    ap.add_argument("--require-signal-level-per-section", action="store_true",
                    help="Require signal intensity label inside each section that has a Decision Lens")

    ap.add_argument("--no-language-check", action="store_true",
                    help="Disable prescriptive/inflated language checks")
    ap.add_argument("--max-issues", type=int, default=50,
                    help="Max issues to print")

    args = ap.parse_args()

    path = pathlib.Path(args.html_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2

    html = path.read_text(encoding="utf-8", errors="replace")

    issues = lint_html(
        html,
        enforce_decision_lens=not args.no_decision_lens,
        enforce_signal_level=args.require_signal_level,
        check_language=not args.no_language_check,
        require_dl_per_section=args.require_dl_per_section,
        require_signal_level_per_section=args.require_signal_level_per_section,
    )

    if not issues:
        print("✅ Lint OK: no issues found.")
        return 0

    print(f"❌ Lint FAIL: {len(issues)} issue(s) found.\n")
    for i, issue in enumerate(issues[:args.max_issues], start=1):
        print(f"{i}. [{issue.code}] {issue.message}")
        print(f"   Context: {issue.context}\n")

    if len(issues) > args.max_issues:
        print(f"... truncated: showing first {args.max_issues} issues ...")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
