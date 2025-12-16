# c4c_utils/irs_return_dispatcher.py
from __future__ import annotations
from typing import Optional, Tuple, List
import re
import io
import pypdf
import pandas as pd

from c4c_utils.irs_return_result import ReturnParseResult, empty_result
from c4c_utils.irs_return_qa import compute_confidence
from c4c_utils.region_tagger import apply_region_tagging

from c4c_utils.irs990_parser import parse_990_pdf          # current PF PDF parser (refactor later)
from c4c_utils.irs990pf_xml_parser import parse_990pf_xml  # PF XML
from c4c_utils.irs990_xml_parser import parse_990_xml      # 990 XML (Schedule I grants)

ATTACHMENT_PHRASES = [
    "see attached", "see attached detail", "see statement", "see schedule",
    "attached detail", "see additional information", "see attached list",
]

def _extract_pdf_pages(file_bytes: bytes) -> Tuple[List[str], int, int, float]:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    page_count = len(pages)
    empty_pages = sum(1 for t in pages if not t.strip())
    empty_ratio = (empty_pages / page_count) if page_count else 0.0
    return pages, page_count, empty_pages, empty_ratio

def _detect_source_type(page_texts: List[str]) -> str:
    blob = "\n".join(page_texts[:3]).lower()
    if "propublica" in blob or "nonprofit explorer" in blob:
        return "propublica"
    if "irs" in blob and "e-file" in blob:
        return "irs"
    return "unknown"

def _detect_form_type(page_texts: List[str]) -> str:
    blob = "\n".join(page_texts[:5])
    if re.search(r"Form\s+990-PF", blob, re.IGNORECASE):
        return "990PF"
    if re.search(r"Form\s+990\b", blob, re.IGNORECASE) and not re.search(r"990-PF", blob, re.IGNORECASE):
        return "990"
    return "unknown"

def _schedule_i_pages_detected(page_texts: List[str]) -> int:
    return sum(1 for t in page_texts if re.search(r"\bSchedule\s+I\b", t, re.IGNORECASE))

def _detect_attachment_reference(text: str) -> Tuple[bool, list[str], Optional[int]]:
    low = text.lower()
    hits = [p for p in ATTACHMENT_PHRASES if p in low]
    if not hits:
        return False, [], None
    # try to capture nearby amount
    amt = None
    m = re.search(r"(?:see attached|see statement)[^\n]{0,80}(\d[\d,]{2,})", text, re.IGNORECASE)
    if m:
        try:
            amt = int(m.group(1).replace(",", ""))
        except:
            amt = None
    return True, hits, amt

def parse_irs_return(file_bytes: bytes, filename: str, tax_year_override: str = "", region_spec: Optional[dict] = None) -> dict:
    try:
        ext = filename.lower().split(".")[-1]

        # XML route
        if ext == "xml":
            # quick sniff: PF XMLs contain ReturnTypeCd like "990PF" sometimes; your XML parsers already handle it.
            try:
                result = parse_990pf_xml(file_bytes, filename, tax_year_override)
                # if parser returns “not PF”, fall through to 990 XML
                if result.get("diagnostics", {}).get("form_type_detected") not in (None, "", "unknown") and result["diagnostics"]["form_type_detected"] != "990PF":
                    raise ValueError("Not 990PF XML")
                out = result
            except Exception:
                out = parse_990_xml(file_bytes, filename, tax_year_override)

            # region tagging (optional)
            if region_spec and isinstance(out.get("grants_df"), pd.DataFrame) and not out["grants_df"].empty:
                out["grants_df"] = apply_region_tagging(out["grants_df"], region_spec)

            # confidence
            conf = compute_confidence(out.get("diagnostics", {}))
            out["diagnostics"]["confidence_score"] = conf.score
            out["diagnostics"]["confidence_grade"] = conf.grade
            out["diagnostics"]["confidence_reasons"] = conf.reasons
            out["diagnostics"]["confidence_penalties"] = conf.penalties
            return out

        # PDF route (dispatcher)
        page_texts, page_count, empty_pages, empty_ratio = _extract_pdf_pages(file_bytes)
        source_type = _detect_source_type(page_texts)
        form_type = _detect_form_type(page_texts)

        schedule_i_count = _schedule_i_pages_detected(page_texts)
        full_text = "\n".join(page_texts)

        # attachment reference sentinel
        attach_detected, attach_hits, attach_amt = _detect_attachment_reference(full_text)

        # current state: parse_990_pdf is PF-only; expand later
        if form_type == "990PF":
            out = parse_990_pdf(file_bytes, filename, tax_year_override)
        else:
            out = empty_result(filename, tax_year_override).to_public_dict()
            out["diagnostics"]["warnings"] = out["diagnostics"].get("warnings", [])
            out["diagnostics"]["warnings"].append("990 PDF grant parsing not implemented yet. Prefer XML for Form 990 grant extraction (Schedule I).")

        # enrich diagnostics universally
        diag = out.setdefault("diagnostics", {})
        diag["file_ext"] = ext
        diag["form_type_detected"] = diag.get("form_type_detected") or form_type
        diag["source_type_detected"] = diag.get("source_type_detected") or source_type
        diag["page_count"] = page_count
        diag["empty_page_count"] = empty_pages
        diag["empty_page_ratio"] = empty_ratio
        diag["schedule_i_pages_detected"] = schedule_i_count

        # attachment sentinel fields
        diag["attachment_reference_detected"] = attach_detected
        diag["attachment_reference_phrases"] = attach_hits
        diag["attachment_reference_amount"] = attach_amt
        if attach_detected:
            diag.setdefault("warnings", [])
            diag["warnings"].append(
                "Return references attached grant detail / statement. Source file may be missing attachments. Consider IRS TEOS for full filing."
            )
            diag["grants_table_incomplete"] = True

        # region tagging
        if region_spec and isinstance(out.get("grants_df"), pd.DataFrame) and not out["grants_df"].empty:
            out["grants_df"] = apply_region_tagging(out["grants_df"], region_spec)

        # confidence
        conf = compute_confidence(diag)
        diag["confidence_score"] = conf.score
        diag["confidence_grade"] = conf.grade
        diag["confidence_reasons"] = conf.reasons
        diag["confidence_penalties"] = conf.penalties

        return out

    except Exception as e:
        return empty_result(filename, tax_year_override, str(e)).to_public_dict()
