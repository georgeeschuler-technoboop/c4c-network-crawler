"""
c4c_utils/irs_return_qa.py

Quality assurance utilities for IRS return parsing.

Provides:
- compute_confidence(): Compute confidence score (0-100) from diagnostics
- render_return_qa_panel(): Streamlit UI component for per-return QA summary

Usage:
    from c4c_utils.irs_return_qa import compute_confidence, render_return_qa_panel
    
    # After parsing
    conf = compute_confidence(diagnostics)
    diagnostics["confidence_score"] = conf.score
    diagnostics["confidence_grade"] = conf.grade
    
    # In Streamlit UI
    render_return_qa_panel(st, foundation_meta, diagnostics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# =============================================================================
# Confidence Scoring
# =============================================================================

@dataclass
class ConfidenceResult:
    """Result of confidence scoring."""
    score: int                        # 0-100
    grade: str                        # "high" | "medium" | "low" | "failed"
    reasons: List[str]                # top reasons (bullets)
    penalties: List[Tuple[str, int]]  # (reason, -points)


def compute_confidence(diagnostics: Dict[str, Any]) -> ConfidenceResult:
    """
    Compute a confidence score (0-100) + grade + reasons from diagnostics.
    
    This is:
      - deterministic
      - transparent (penalties list explains the score)
      - robust to missing fields
    
    Expected diagnostics keys (not all required):
    
    Universal:
        - form_type_detected
        - page_count
        - empty_page_ratio
        - warnings (list[str])
    
    990-PF:
        - reported_total_3a, computed_total_3a, total_mismatch_3a
        - reported_total_3b, computed_total_3b, total_mismatch_3b
        - grants_3a_count
    
    990:
        - schedule_i_pages_detected
        - schedule_i_part_ii_found
        - schedule_i_reason
        - reported_total_part1_line13
        - computed_total_schedule_i_part_ii
    
    Returns:
        ConfidenceResult with score, grade, reasons, and penalties
    """
    form_type = (
        diagnostics.get("form_type_detected") 
        or diagnostics.get("form_type") 
        or ""
    ).strip()
    
    if not form_type:
        form_type = "unknown"

    # If unknown form type, fail fast
    if form_type.lower() in {"unknown", "?", "unclassified"}:
        return ConfidenceResult(
            score=0,
            grade="failed",
            reasons=["Form type could not be confidently detected."],
            penalties=[("Unknown form type", -100)],
        )

    score = 100
    penalties: List[Tuple[str, int]] = []
    reasons: List[str] = []

    def penalize(reason: str, points: int) -> None:
        nonlocal score
        score -= points
        penalties.append((reason, -points))

    # -------------------------------------------------------------------------
    # Universal extraction health
    # -------------------------------------------------------------------------
    empty_ratio = diagnostics.get("empty_page_ratio", None)
    if isinstance(empty_ratio, (int, float)):
        if empty_ratio > 0.50:
            penalize("High empty-page ratio (>50%) suggests text extraction failure.", 50)
        elif empty_ratio > 0.30:
            penalize("Moderate empty-page ratio (>30%) suggests partial text extraction.", 30)

    # -------------------------------------------------------------------------
    # Form-specific scoring
    # -------------------------------------------------------------------------
    ft = form_type.lower()

    if ft in {"990pf", "990-pf"}:
        # Primary: 3a totals reconciliation
        rep_3a = diagnostics.get("reported_total_3a", None)
        # Support both computed_total_3a (contract) and grants_3a_total (current parser)
        comp_3a = diagnostics.get("computed_total_3a") or diagnostics.get("grants_3a_total")
        grants_3a_count = diagnostics.get("grants_3a_count", None)

        # If we have reported total 3a, mismatch is a big deal
        if isinstance(rep_3a, (int, float)) and isinstance(comp_3a, (int, float)):
            diff = abs(int(rep_3a) - int(comp_3a))
            pct = (diff / float(rep_3a) * 100) if rep_3a else 0
            
            # Percentage-based thresholds (more forgiving for large foundations)
            if pct <= 0.5:
                # Excellent match (within 0.5%)
                reasons.append(f"3a computed total matches reported total ({100-pct:.1f}% match).")
            elif pct <= 1.0:
                # Good match (within 1%)
                reasons.append(f"3a totals within 1% ({100-pct:.1f}% match).")
            elif pct <= 2.0:
                # Minor mismatch
                penalize(f"3a totals mismatch: {pct:.1f}% variance (${diff:,}).", 15)
            elif pct <= 5.0:
                # Moderate mismatch
                penalize(f"3a totals mismatch: {pct:.1f}% variance (${diff:,}).", 30)
            else:
                # Serious mismatch
                penalize(f"3a totals mismatch: {pct:.1f}% variance (${diff:,}).", 50)
        else:
            # Missing reported 3a is less severe, but reduces QA certainty
            penalize("Reported 3a total not found; cannot reconcile computed totals.", 15)

        # If reported 3a > 0 but parsed 0 grants, that's likely a failure
        if isinstance(rep_3a, (int, float)) and rep_3a > 0:
            if isinstance(grants_3a_count, int) and grants_3a_count == 0:
                penalize("Reported 3a total > 0 but parsed 0 3a grants.", 60)

        # Secondary: 3b mismatch is informative but less important
        rep_3b = diagnostics.get("reported_total_3b", None)
        # Support both computed_total_3b (contract) and grants_3b_total (current parser)
        comp_3b = diagnostics.get("computed_total_3b") or diagnostics.get("grants_3b_total")
        if isinstance(rep_3b, (int, float)) and isinstance(comp_3b, (int, float)):
            diff_b = abs(int(rep_3b) - int(comp_3b))
            pct_b = (diff_b / float(rep_3b) * 100) if rep_3b else 0
            
            # 3b is secondary, so lighter penalties
            if pct_b > 2.0:
                penalize(f"3b totals mismatch: {pct_b:.1f}% variance [secondary].", 10)

    elif ft == "990":
        pages = diagnostics.get("schedule_i_pages_detected", None)
        part_ii_found = diagnostics.get("schedule_i_part_ii_found", None)
        reason = (diagnostics.get("schedule_i_reason") or "").strip()

        # Schedule I coverage penalties
        if isinstance(pages, int):
            if pages == 0:
                penalize("Schedule I not detected (0 pages).", 60)
            elif pages == 1:
                penalize("Only 1 Schedule I page detected (possible partial PDF).", 25)
            else:
                reasons.append(f"Schedule I detected on {pages} pages.")

        # Part II presence
        if part_ii_found is False:
            penalize("Schedule I found but Part II not found (domestic org grant detail missing).", 30)
        elif part_ii_found is True:
            reasons.append("Schedule I Part II detected.")

        # Totals reconciliation (where available)
        rep = diagnostics.get("reported_total_part1_line13", None)
        comp = diagnostics.get("computed_total_schedule_i_part_ii", None)

        if isinstance(rep, (int, float)) and isinstance(comp, (int, float)):
            diff = abs(int(rep) - int(comp))
            if diff <= 100:
                reasons.append("Schedule I computed total matches Part I line 13 (within $100).")
            else:
                penalize(f"Schedule I computed total mismatches Part I line 13 (diff ${diff:,}).", 40)
        else:
            penalize("Reported total (Part I line 13) not found; limited QA reconciliation.", 15)

        # Strong hint: totals > 0 but no schedule I records parsed
        if isinstance(rep, (int, float)) and rep > 0:
            if isinstance(comp, (int, float)) and comp == 0:
                penalize("Reported grants total > 0 but computed Schedule I total is 0.", 50)

        if reason and reason != "ok":
            penalize(f"Schedule I parsing status: {reason}.", 5)

    else:
        # Unknown-but-not-unknown: conservative penalty
        penalize(f"Unrecognized form type '{form_type}' for scoring rubric.", 30)

    # Clamp score
    score = max(0, min(100, int(round(score))))

    # Grade thresholds
    if score >= 85:
        grade = "high"
    elif score >= 65:
        grade = "medium"
    elif score >= 35:
        grade = "low"
    else:
        grade = "failed"

    # If no positive reasons were added, add a generic one
    if not reasons:
        reasons.append("Confidence derived from available diagnostics; see penalties for details.")

    # Keep reasons concise (top 6)
    reasons = reasons[:6]

    return ConfidenceResult(score=score, grade=grade, reasons=reasons, penalties=penalties)


# =============================================================================
# Streamlit QA Panel
# =============================================================================

def render_return_qa_panel(
    st,
    foundation_meta: Dict[str, Any],
    diagnostics: Dict[str, Any],
    expanded: bool = True
) -> None:
    """
    Streamlit UI component: per-return QA summary panel.
    
    Displays:
    - Organization header (name, EIN, year, form type)
    - Confidence score and grade with reasons
    - Extraction health metrics
    - Form-specific QA checks (totals reconciliation)
    - Warnings
    
    Args:
        st: Streamlit module (passed in for testability)
        foundation_meta: Foundation metadata dict
        diagnostics: Diagnostics dict from parser
        expanded: Whether expander starts expanded (default True)
    
    Usage:
        from c4c_utils.irs_return_qa import render_return_qa_panel
        render_return_qa_panel(st, result["foundation_meta"], result["diagnostics"])
    """
    org = (
        foundation_meta.get("foundation_name") 
        or diagnostics.get("org_name") 
        or "Unknown organization"
    )
    ein = foundation_meta.get("foundation_ein") or foundation_meta.get("ein") or ""
    year = foundation_meta.get("tax_year") or ""
    form_type = (
        diagnostics.get("form_type_detected") 
        or diagnostics.get("form_type") 
        or "unknown"
    )
    source_type = (
        diagnostics.get("source_type_detected") 
        or diagnostics.get("source_type") 
        or "unknown"
    )

    # Compute confidence if not present
    if "confidence_score" not in diagnostics or "confidence_grade" not in diagnostics:
        conf = compute_confidence(diagnostics)
        diagnostics["confidence_score"] = conf.score
        diagnostics["confidence_grade"] = conf.grade
        diagnostics["confidence_reasons"] = conf.reasons
        diagnostics["confidence_penalties"] = conf.penalties

    score = diagnostics.get("confidence_score", 0)
    grade = diagnostics.get("confidence_grade", "failed")
    reasons = diagnostics.get("confidence_reasons", [])
    warnings = diagnostics.get("warnings", [])

    # Grade to emoji mapping
    grade_emoji = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🟠",
        "failed": "🔴"
    }
    emoji = grade_emoji.get(grade, "⚪")

    with st.expander(f"{emoji} QA Summary — {org}", expanded=expanded):
        # Header
        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown(f"**Organization:** {org}")
        cols[1].markdown(f"**Year:** {year or '—'}")
        cols[2].markdown(f"**Form:** {form_type}")
        cols[3].markdown(f"**Source:** {source_type}")
        if ein:
            st.markdown(f"**EIN:** {ein}")

        st.divider()

        # Confidence block
        st.markdown("### 📊 Parser Confidence")
        st.markdown(f"**Grade:** `{grade.upper()}`  &nbsp;&nbsp; **Score:** `{score}/100`")
        if reasons:
            st.markdown("**Why:**")
            for r in reasons:
                st.markdown(f"- {r}")

        # Extraction health
        page_count = diagnostics.get("page_count") or diagnostics.get("pages_processed")
        empty_ratio = diagnostics.get("empty_page_ratio")
        if page_count is not None or empty_ratio is not None:
            st.markdown("### 📄 Extraction Health")
            if page_count is not None:
                st.markdown(f"- Pages processed: **{page_count}**")
            if isinstance(empty_ratio, (int, float)):
                st.markdown(f"- Empty-page ratio: **{empty_ratio:.0%}**")

        # Form-specific QA
        ft = str(form_type).lower()
        st.markdown("### 🔍 Form Checks")

        if ft in {"990pf", "990-pf"}:
            _render_990pf_checks(st, diagnostics)
        elif ft == "990":
            _render_990_checks(st, diagnostics)
        else:
            st.markdown("- No form-specific QA available (unknown form type).")

        # Warnings last
        if warnings:
            st.markdown("### ⚠️ Warnings")
            for w in warnings:
                st.warning(w)


def _render_990pf_checks(st, diagnostics: Dict[str, Any]) -> None:
    """Render 990-PF specific QA checks."""
    rep_3a = diagnostics.get("reported_total_3a")
    comp_3a = diagnostics.get("computed_total_3a") or diagnostics.get("grants_3a_total")
    c3a = diagnostics.get("grants_3a_count")

    st.markdown("**990-PF Part XV (primary: 3a paid during year)**")
    st.markdown(f"- 3a grants count: **{c3a if c3a is not None else '—'}**")
    st.markdown(f"- 3a reported total: **{_fmt_money(rep_3a)}**")
    st.markdown(f"- 3a computed total: **{_fmt_money(comp_3a)}**")
    
    if rep_3a is not None and comp_3a is not None:
        diff = abs(int(rep_3a) - int(comp_3a))
        pct = (diff / float(rep_3a) * 100) if rep_3a else 0
        match_status = "✅" if diff <= 100 else "⚠️"
        st.markdown(f"- 3a difference: **{_fmt_money(diff)}** ({pct:.1f}%) {match_status}")

    st.markdown("**990-PF Part XV (secondary: 3b approved for future)**")
    rep_3b = diagnostics.get("reported_total_3b")
    comp_3b = diagnostics.get("computed_total_3b") or diagnostics.get("grants_3b_total")
    c3b = diagnostics.get("grants_3b_count")
    
    st.markdown(f"- 3b grants count: **{c3b if c3b is not None else '—'}**")
    st.markdown(f"- 3b reported total: **{_fmt_money(rep_3b)}**")
    st.markdown(f"- 3b computed total: **{_fmt_money(comp_3b)}**")


def _render_990_checks(st, diagnostics: Dict[str, Any]) -> None:
    """Render Form 990 specific QA checks."""
    pages = diagnostics.get("schedule_i_pages_detected")
    part_ii = diagnostics.get("schedule_i_part_ii_found")
    reason = diagnostics.get("schedule_i_reason", "")
    rep = diagnostics.get("reported_total_part1_line13")
    comp = diagnostics.get("computed_total_schedule_i_part_ii")
    count = diagnostics.get("schedule_i_part_ii_grants_count")

    st.markdown("**Form 990 Schedule I (primary detail source)**")
    st.markdown(f"- Schedule I pages detected: **{pages if pages is not None else '—'}**")
    st.markdown(f"- Schedule I Part II found: **{part_ii if part_ii is not None else '—'}**")
    if reason:
        st.markdown(f"- Schedule I status: **{reason}**")
    st.markdown(f"- Schedule I parsed grants count: **{count if count is not None else '—'}**")

    st.markdown("**Totals reconciliation (QA)**")
    st.markdown(f"- Part I line 13 reported total: **{_fmt_money(rep)}**")
    st.markdown(f"- Computed total from Schedule I Part II: **{_fmt_money(comp)}**")
    
    if rep is not None and comp is not None:
        diff = abs(int(rep) - int(comp))
        match_status = "✅" if diff <= 100 else "⚠️"
        st.markdown(f"- Difference: **{_fmt_money(diff)}** {match_status}")


def _fmt_money(x: Any) -> str:
    """Format a number as currency, handling None/empty gracefully."""
    try:
        if x is None or x == "":
            return "—"
        val = int(float(x))
        return f"${val:,}"
    except Exception:
        return str(x)


# =============================================================================
# Utility: Enrich diagnostics with confidence
# =============================================================================

def enrich_diagnostics_with_confidence(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute confidence and add to diagnostics dict (in-place + return).
    
    Adds:
        - confidence_score
        - confidence_grade
        - confidence_reasons
        - confidence_penalties
    
    Usage:
        diagnostics = enrich_diagnostics_with_confidence(diagnostics)
    """
    conf = compute_confidence(diagnostics)
    diagnostics["confidence_score"] = conf.score
    diagnostics["confidence_grade"] = conf.grade
    diagnostics["confidence_reasons"] = conf.reasons
    diagnostics["confidence_penalties"] = conf.penalties
    return diagnostics
