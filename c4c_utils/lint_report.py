#!/usr/bin/env python3
"""
lint_report.py — Validates InsightGraph HTML reports for quality regressions.

Usage:
    python lint_report.py path/to/report.html
    python lint_report.py path/to/report.html --require-signal-level

Returns exit code 0 if OK, 1 if issues found.
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


# Raw markdown patterns that should not appear in HTML
RAW_MD_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("RAW_MD_ITALIC", re.compile(r"(?<!\w)_(?!_)([^_\n]{2,})_(?!\w)")),
    ("RAW_MD_BOLD", re.compile(r"\*\*[^*\n]{2,}\*\*")),
    ("RAW_MD_HEADER", re.compile(r"(?m)^(#{1,6})\s+.+$")),
    ("RAW_MD_FENCE", re.compile(r"```")),
]

# Disallowed/prescriptive language
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

# Signal intensity badge presence
SIGNAL_LEVEL_BADGE = re.compile(r"\b(Low|Moderate|High)-intensity signal\b", re.IGNORECASE)

# Section detection
SECTION_TAG = re.compile(r"<section\b", re.IGNORECASE)


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


def lint_html(
    html: str, 
    enforce_decision_lens: bool = True, 
    enforce_signal_level: bool = False,
    check_language: bool = True
) -> List[LintIssue]:
    """
    Lint an HTML report for quality issues.
    
    Args:
        html: The HTML content to lint
        enforce_decision_lens: Check for Decision Lens presence
        enforce_signal_level: Check for signal intensity badges
        check_language: Check for prescriptive/inflated language
    
    Returns:
        List of LintIssue objects
    """
    issues: List[LintIssue] = []

    # Raw markdown artifacts
    for code, pat in RAW_MD_PATTERNS:
        for m in pat.finditer(html):
            issues.append(LintIssue(
                code, 
                f"Found raw markdown artifact ({code}).",
                _snippet(html, m.start(), m.end())
            ))

    # Disallowed/prescriptive language
    if check_language:
        for code, pat in DISALLOWED_LANGUAGE:
            for m in pat.finditer(html):
                # Skip if inside <code> or <pre> blocks
                before = html[:m.start()]
                if before.count("<code") > before.count("</code"):
                    continue
                if before.count("<pre") > before.count("</pre"):
                    continue
                    
                issues.append(LintIssue(
                    code, 
                    f"Disallowed or review-needed language: '{m.group(0)}'.",
                    _snippet(html, m.start(), m.end())
                ))

    # Decision Lens presence
    if enforce_decision_lens:
        has_sections = bool(SECTION_TAG.search(html))
        has_dl = bool(DECISION_LENS_CLASS.search(html))
        if has_sections and not has_dl:
            issues.append(LintIssue(
                "MISSING_DECISION_LENS",
                "No Decision Lens block detected in report HTML.",
                "Expected a div with class 'decision-lens'."
            ))

    # Signal level badge presence
    if enforce_signal_level:
        if not SIGNAL_LEVEL_BADGE.search(html):
            issues.append(LintIssue(
                "MISSING_SIGNAL_LEVEL",
                "No signal intensity labels detected in report HTML.",
                "Expected: 'Low-intensity signal', 'Moderate-intensity signal', or 'High-intensity signal'."
            ))

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lint a generated InsightGraph HTML report for quality regressions."
    )
    ap.add_argument("html_path", type=str, help="Path to report HTML file")
    ap.add_argument(
        "--no-decision-lens", 
        action="store_true", 
        help="Disable Decision Lens presence check"
    )
    ap.add_argument(
        "--require-signal-level", 
        action="store_true", 
        help="Require signal intensity labels to appear"
    )
    ap.add_argument(
        "--no-language-check",
        action="store_true",
        help="Disable prescriptive language checks"
    )
    ap.add_argument(
        "--max-issues", 
        type=int, 
        default=50, 
        help="Max issues to print (default: 50)"
    )
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
        check_language=not args.no_language_check
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
