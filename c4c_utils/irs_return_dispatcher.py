# c4c_utils/irs_return_dispatcher.py
"""
Unified dispatcher for IRS return parsing (PDF and XML).

Routes:
- XML 990-PF → parse_990pf_xml
- XML 990 → parse_990_xml  
- PDF 990-PF → parse_990_pdf
- PDF 990 → not yet supported (returns empty with warning)

Also handles:
- Attachment reference detection ("See attached detail")
- Region tagging (optional)
- Confidence scoring
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
                "990 PDF grant parsing not implemented yet. Prefer XML for Form 990 grant extraction (Schedule I)."
            )

        # Enrich diagnostics
        diag = out.setdefault("diagnostics", {})
        diag["file_ext"] = ext
        diag["form_type_detected"] = diag.get("form_type_detected") or form_type
        diag["source_type_detected"] = diag.get("source_type_detected") or source_type
        diag["page_count"] = page_count
        diag["empty_page_count"] = empty_pages
        diag["empty_page_ratio"] = empty_ratio
        diag["schedule_i_pages_detected"] = schedule_i_count

        # Attachment sentinel fields
        # But suppress if totals match well (>98%) — means we got the grants despite the phrase
        suppress_attachment_warning = False
        if attach_detected:
            # Check totals reconciliation
            rep_3a = diag.get("reported_total_3a")
            comp_3a = diag.get("grants_3a_total", 0)
            rep_3b = diag.get("reported_total_3b")
            comp_3b = diag.get("grants_3b_total", 0)
            
            # Calculate match percentages
            match_3a = None
            match_3b = None
            
            if rep_3a and rep_3a > 0:
                match_3a = (comp_3a / rep_3a) * 100
            if rep_3b and rep_3b > 0:
                match_3b = (comp_3b / rep_3b) * 100
            
            # Suppress warning if either total matches >98%
            # (means we successfully extracted grants despite "see statement" phrase)
            if match_3a is not None and match_3a >= 98:
                suppress_attachment_warning = True
            elif match_3b is not None and match_3b >= 98:
                suppress_attachment_warning = True
        
        diag["attachment_reference_phrases"] = attach_hits
        diag["attachment_reference_amount"] = attach_amt
        
        if attach_detected and not suppress_attachment_warning:
            diag["attachment_reference_detected"] = True
            diag.setdefault("warnings", [])
            diag["warnings"].append(
                "Return references attached grant detail / statement. "
                "Source file may be missing attachments. Consider IRS TEOS for full filing."
            )
            diag["grants_table_incomplete"] = True
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
