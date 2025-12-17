# c4c_utils/irs_return_dispatcher.py
"""
Unified dispatcher for IRS return parsing (PDF and XML).

Routes:
- XML 990-PF → parse_990pf_xml
- XML 990 → parse_990_xml  
- PDF 990-PF → parse_990_pdf
- PDF 990 → not yet supported (returns empty with warning)

Also handles:
- Scanned/image PDF detection
- Attachment reference detection ("See attached detail")
- Region tagging (optional)
- Confidence scoring

v2.7 Changes:
- Added scanned PDF detection (empty_ratio > 0.9)
- Fixed "See Statement" warning: check grant COUNT not just totals
- Improved form type detection for IRS-sourced PDFs
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import re
import io
import xml.etree.ElementTree as ET
import pypdf
import pandas as pd

from c4c_utils.irs_return_result import ReturnParseResult, empty_result
from c4c_utils.irs_return_qa import compute_confidence
from c4c_utils.region_tagger import apply_region_tagging

from c4c_utils.irs990_parser import parse_990_pdf          # current PF PDF parser
from c4c_utils.irs990pf_xml_parser import parse_990pf_xml  # PF XML
from c4c_utils.irs990_xml_parser import parse_990_xml      # 990 XML (Schedule I grants)

# IRS namespace for XML parsing
IRS_NS = {"irs": "http://www.irs.gov/efile"}

ATTACHMENT_PHRASES = [
    "see attached", "see attached detail", "see statement", "see schedule",
    "attached detail", "see additional information", "see attached list",
]

# =============================================================================
# XML Helpers
# =============================================================================

def _detect_xml_form_type(xml_bytes: bytes) -> str:
    """
    Detect form type from XML structure before parsing.
    
    Returns: "990PF", "990", or "unknown"
    """
    try:
        # Handle BOM
        if xml_bytes.startswith(b'\xef\xbb\xbf'):
            xml_bytes = xml_bytes[3:]
        
        root = ET.fromstring(xml_bytes)
        
        # Try ReturnTypeCd first (most reliable)
        return_type_el = root.find(".//irs:ReturnHeader/irs:ReturnTypeCd", IRS_NS)
        if return_type_el is not None and return_type_el.text:
            return_type = return_type_el.text.strip().upper()
            if return_type in ("990PF", "990-PF"):
                return "990PF"
            if return_type == "990":
                return "990"
        
        # Fallback: check for form-specific elements
        if root.find(".//irs:IRS990PF", IRS_NS) is not None:
            return "990PF"
        if root.find(".//irs:IRS990", IRS_NS) is not None:
            return "990"
        
        return "unknown"
    except ET.ParseError:
        return "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# PDF Helpers
# =============================================================================

def _extract_pdf_pages(file_bytes: bytes) -> Tuple[List[str], int, int, float]:
    """Extract per-page text from PDF."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    page_count = len(pages)
    empty_pages = sum(1 for t in pages if not t.strip())
    empty_ratio = (empty_pages / page_count) if page_count else 0.0
    return pages, page_count, empty_pages, empty_ratio


def _detect_scanned_pdf(page_texts: List[str], empty_ratio: float) -> Tuple[bool, str]:
    """
    Detect if PDF is a scanned/image PDF with no extractable text.
    
    Returns: (is_scanned, reason_message)
    """
    # If more than 90% of pages have no text, likely scanned
    if empty_ratio > 0.9:
        return True, (
            "Scanned/image PDF detected. No extractable text found. "
            "This appears to be a TIFF or image-wrapped PDF. "
            "OCR integration would be required to parse this document. "
            "Consider sourcing the XML version from ProPublica or IRS e-file."
        )
    
    # Check total text length across all pages
    total_text = "".join(page_texts)
    if len(total_text.strip()) < 500 and len(page_texts) > 5:
        return True, (
            "Scanned/image PDF detected. Minimal extractable text found. "
            "This document may be an image-based scan. "
            "Consider sourcing the XML version from ProPublica or IRS e-file."
        )
    
    return False, ""


def _detect_source_type(page_texts: List[str]) -> str:
    """Detect PDF source (propublica, irs, unknown)."""
    blob = "\n".join(page_texts[:3]).lower()
    if "propublica" in blob or "nonprofit explorer" in blob:
        return "propublica"
    if "irs" in blob and "e-file" in blob:
        return "irs"
    return "unknown"


def _detect_pdf_form_type(page_texts: List[str]) -> str:
    """Detect form type from PDF text."""
    blob = "\n".join(page_texts[:5])
    if re.search(r"Form\s+990-PF", blob, re.IGNORECASE):
        return "990PF"
    if re.search(r"Form\s+990\b", blob, re.IGNORECASE) and not re.search(r"990-PF", blob, re.IGNORECASE):
        return "990"
    return "unknown"


def _schedule_i_pages_detected(page_texts: List[str]) -> int:
    """Count pages containing Schedule I references."""
    return sum(1 for t in page_texts if re.search(r"\bSchedule\s+I\b", t, re.IGNORECASE))


def _detect_attachment_reference(text: str) -> Tuple[bool, List[str], Optional[int]]:
    """
    Detect 'See attached detail' style placeholders.
    
    Returns: (detected, list of phrases found, optional amount)
    """
    low = text.lower()
    hits = [p for p in ATTACHMENT_PHRASES if p in low]
    if not hits:
        return False, [], None
    
    # Try to capture nearby amount
    amt = None
    m = re.search(r"(?:see attached|see statement)[^\n]{0,80}(\d[\d,]{2,})", text, re.IGNORECASE)
    if m:
        try:
            amt = int(m.group(1).replace(",", ""))
        except:
            amt = None
    return True, hits, amt


def _check_see_statement_grants(grants_df: pd.DataFrame, attach_detected: bool) -> Tuple[bool, str]:
    """
    Check if grants are actually parsed or just "(SEE STATEMENT)" placeholders.
    
    Returns: (has_real_grants, warning_message)
    """
    if not attach_detected:
        return True, ""
    
    if grants_df.empty:
        return False, (
            "No grants extracted. Form contains 'See Statement' references - "
            "detailed grants are likely in supplemental pages. "
            "Check IRS TEOS or ProPublica for complete filing with attachments."
        )
    
    # Check if we only have placeholder grants
    if 'grantee_name' in grants_df.columns:
        grantee_names = grants_df['grantee_name'].astype(str).str.upper()
        placeholder_count = grantee_names.str.contains(
            r'SEE\s+STATEMENT|SEE\s+ATTACHED|SEE\s+SCHEDULE', 
            regex=True
        ).sum()
        
        total_grants = len(grants_df)
        
        # If most grants are placeholders, warn
        if placeholder_count > 0 and placeholder_count >= total_grants * 0.5:
            return False, (
                f"Only {total_grants - placeholder_count} real grants extracted; "
                f"{placeholder_count} placeholder entries found. "
                "Detailed grants are in supplemental pages (Part XIV continuation). "
                "Parser may need to extract from continuation pages."
            )
        
        # If we have very few grants but high totals, warn
        if total_grants < 5 and attach_detected:
            return False, (
                f"Only {total_grants} grants extracted but 'See Statement' detected. "
                "Detailed grant list may be in supplemental pages."
            )
    
    return True, ""


# =============================================================================
# Main Dispatcher
# =============================================================================

def parse_irs_return(
    file_bytes: bytes, 
    filename: str, 
    tax_year_override: str = "", 
    region_spec: Optional[dict] = None
) -> dict:
    """
    Unified entry point for parsing IRS 990/990-PF returns.
    
    Args:
        file_bytes: Raw file content
        filename: Original filename (used for extension detection)
        tax_year_override: Optional tax year override
        region_spec: Optional region definition for tagging
    
    Returns:
        dict with keys: grants_df, people_df, foundation_meta, diagnostics, debug
    """
    try:
        ext = filename.lower().split(".")[-1]

        # =====================================================================
        # XML Route
        # =====================================================================
        if ext == "xml":
            # Detect form type BEFORE parsing
            form_type = _detect_xml_form_type(file_bytes)
            
            if form_type == "990PF":
                out = parse_990pf_xml(file_bytes, filename, tax_year_override)
            elif form_type == "990":
                out = parse_990_xml(file_bytes, filename, tax_year_override)
            else:
                # Unknown form type
                out = empty_result(filename, tax_year_override).to_public_dict()
                out["diagnostics"]["warnings"] = out["diagnostics"].get("warnings", [])
                out["diagnostics"]["warnings"].append(
                    f"Could not detect form type from XML. Expected 990-PF or 990."
                )
                out["diagnostics"]["form_type_detected"] = "unknown"

            # Ensure diagnostics exists
            diag = out.setdefault("diagnostics", {})
            diag["file_ext"] = "xml"
            diag["form_type_detected"] = diag.get("form_type_detected") or form_type
            
            # Region tagging (optional)
            if region_spec and isinstance(out.get("grants_df"), pd.DataFrame) and not out["grants_df"].empty:
                out["grants_df"] = apply_region_tagging(out["grants_df"], region_spec)

            # Confidence scoring
            conf = compute_confidence(diag)
            diag["confidence_score"] = conf.score
            diag["confidence_grade"] = conf.grade
            diag["confidence_reasons"] = conf.reasons
            diag["confidence_penalties"] = conf.penalties
            
            return out

        # =====================================================================
        # PDF Route
        # =====================================================================
        page_texts, page_count, empty_pages, empty_ratio = _extract_pdf_pages(file_bytes)
        
        # =====================================================================
        # Scanned PDF Detection (v2.7)
        # =====================================================================
        is_scanned, scanned_message = _detect_scanned_pdf(page_texts, empty_ratio)
        if is_scanned:
            out = empty_result(filename, tax_year_override).to_public_dict()
            diag = out.setdefault("diagnostics", {})
            diag["file_ext"] = ext
            diag["form_type_detected"] = "unknown"
            diag["is_scanned_pdf"] = True
            diag["page_count"] = page_count
            diag["empty_page_count"] = empty_pages
            diag["empty_page_ratio"] = empty_ratio
            diag.setdefault("warnings", []).append(scanned_message)
            diag.setdefault("errors", []).append("Cannot parse scanned/image PDF without OCR.")
            
            # Still compute confidence (will be 0 due to errors)
            conf = compute_confidence(diag)
            diag["confidence_score"] = conf.score
            diag["confidence_grade"] = conf.grade
            diag["confidence_reasons"] = conf.reasons
            diag["confidence_penalties"] = conf.penalties
            
            return out
        
        source_type = _detect_source_type(page_texts)
        form_type = _detect_pdf_form_type(page_texts)
        schedule_i_count = _schedule_i_pages_detected(page_texts)
        full_text = "\n".join(page_texts)

        # Attachment reference detection
        attach_detected, attach_hits, attach_amt = _detect_attachment_reference(full_text)

        # Route based on form type
        if form_type == "990PF":
            out = parse_990_pdf(file_bytes, filename, tax_year_override)
        else:
            # 990 PDF parsing not yet supported
            out = empty_result(filename, tax_year_override).to_public_dict()
            out["diagnostics"]["warnings"] = out["diagnostics"].get("warnings", [])
            out["diagnostics"]["warnings"].append(
                "990 PDF grant parsing not implemented yet. "
                "This appears to be a Form 990 (not 990-PF). "
                "Prefer XML format for Form 990 grant extraction (Schedule I)."
            )
            out["diagnostics"]["form_type_detected"] = form_type or "990"

        # Enrich diagnostics
        diag = out.setdefault("diagnostics", {})
        diag["file_ext"] = ext
        diag["form_type_detected"] = diag.get("form_type_detected") or form_type
        diag["source_type_detected"] = diag.get("source_type_detected") or source_type
        diag["page_count"] = page_count
        diag["empty_page_count"] = empty_pages
        diag["empty_page_ratio"] = empty_ratio
        diag["schedule_i_pages_detected"] = schedule_i_count
        diag["is_scanned_pdf"] = False

        # Attachment sentinel fields
        diag["attachment_reference_phrases"] = attach_hits
        diag["attachment_reference_amount"] = attach_amt
        
        # v2.7: Check for actual grants, not just totals matching
        grants_df = out.get("grants_df", pd.DataFrame())
        has_real_grants, grants_warning = _check_see_statement_grants(grants_df, attach_detected)
        
        if attach_detected and not has_real_grants:
            diag["attachment_reference_detected"] = True
            diag.setdefault("warnings", []).append(grants_warning)
            diag["grants_table_incomplete"] = True
        elif attach_detected and has_real_grants:
            # We detected "see statement" but also extracted real grants
            # Check if totals reconcile
            rep_3a = diag.get("reported_total_3a", 0)
            comp_3a = diag.get("grants_3a_total", 0)
            rep_3b = diag.get("reported_total_3b", 0)
            comp_3b = diag.get("grants_3b_total", 0)
            
            match_3a = (comp_3a / rep_3a * 100) if rep_3a > 0 else 0
            match_3b = (comp_3b / rep_3b * 100) if rep_3b > 0 else 0
            
            # Only suppress if >98% match AND we have more than placeholder grants
            if match_3a >= 98 or match_3b >= 98:
                diag["attachment_reference_detected"] = False  # Suppress warning
            else:
                diag["attachment_reference_detected"] = True
                diag.setdefault("warnings", []).append(
                    f"'See Statement' detected. Totals match: 3a={match_3a:.1f}%, 3b={match_3b:.1f}%. "
                    "Some grants may be in supplemental pages."
                )
        else:
            diag["attachment_reference_detected"] = False

        # Region tagging
        if region_spec and isinstance(out.get("grants_df"), pd.DataFrame) and not out["grants_df"].empty:
            out["grants_df"] = apply_region_tagging(out["grants_df"], region_spec)

        # Confidence scoring
        conf = compute_confidence(diag)
        diag["confidence_score"] = conf.score
        diag["confidence_grade"] = conf.grade
        diag["confidence_reasons"] = conf.reasons
        diag["confidence_penalties"] = conf.penalties

        return out

    except Exception as e:
        return empty_result(filename, tax_year_override, str(e)).to_public_dict()
