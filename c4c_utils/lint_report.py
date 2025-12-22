#!/usr/bin/env python3
"""
lint_report.py — Lints InsightGraph HTML reports for authoring and interpretation regressions.
"""

from __future__ import annotations
import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LintIssue:
    code: str
    message: str
    context: str


# --- Patterns ---

RAW_MARKDOWN = [
    ("RAW_MD_ITALIC", re.compile(r"(?<!\w)_(?!_)([^_\n]{2,})_(?!\w)")),
    ("RAW_MD_BOLD", re.compile(r"\*\*[^*\n]{2,}\*\*")),
    ("RAW_MD_CODE", re.compile(r"`[^`\n]+`")),
    ("RAW_MD_FENCE", re.compile(r"```")),
]

DISALLOWED_LANGUAGE = [
    ("PRESCRIPTIVE_SHOULD", re.compile(r"\bshould\b", re.I)),
    ("PRESCRIPTIVE_MUST", re.compile(r"\bmust\b", re.I)),
    ("PRESCRIPTIVE_RECOMMEND", re.compile(r"\brecommend(s|ed|ation)?\b", re.I)),
    ("SEMANTIC_INFLATION_NATURAL", re.compile(r"\bnatural (partners?|hub)\b", re.I)),
]

DECISION_LENS = re.compile(r'class="[^"]*decision-lens[^"]*"', re.I)
SIGNAL_INTENSITY = re.compile(r"\b(Low|Moderate|High)-intensity signal\b", re.I)
SECTION = re.compile(r"<section\b", re.I)

TAG = re.compile(r"<[^>]+>")


def visible_text(html: str) -> str:
    return TAG.sub(" ", html)


def lint_html(html: str, require_per_section: bool = False) -> List[LintIssue]:
    issues: List[LintIssue] = []

    # Raw markdown leakage
    for code, pat in RAW_MARKDOWN:
        for m in pat.finditer(html):
            issues.append(LintIssue(code, "Raw markdown artifact found.", m.group(0)))

    # Language checks (visible text only)
    text = visible_text(html)
    for code, pat in DISALLOWED_LANGUAGE:
        for m in pat.finditer(text):
            issues.append(LintIssue(code, f"Disallowed language: '{m.group(0)}'", m.group(0)))

    # Global Decision Lens check
    if SECTION.search(html) and not DECISION_LENS.search(html):
        issues.append(LintIssue(
            "MISSING_DECISION_LENS",
            "No Decision Lens detected in report.",
            "Expected at least one Decision Lens block."
        ))

    # Signal intensity check
    if DECISION_LENS.search(html) and not SIGNAL_INTENSITY.search(html):
        issues.append(LintIssue(
            "MISSING_SIGNAL_INTENSITY",
            "Decision Lens present but no signal intensity label found.",
            "Expected Low / Moderate / High-intensity signal."
        ))

    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html_path")
    args = ap.parse_args()

    path = pathlib.Path(args.html_path)
    if not path.exists():
        print("File not found:", path, file=sys.stderr)
        return 2

    html = path.read_text(encoding="utf-8", errors="replace")
    issues = lint_html(html)

    if not issues:
        print("✅ Lint OK")
        return 0

    print(f"❌ Lint failed with {len(issues)} issue(s):\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. [{issue.code}] {issue.message}")
        print(f"   Context: {issue.context}\n")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
