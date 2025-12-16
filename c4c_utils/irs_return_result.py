# c4c_utils/irs_return_result.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass
class FoundationMeta:
    foundation_name: str = ""
    foundation_ein: str = ""
    tax_year: str = ""
    source_file: str = ""
    form_type: str = ""     # "990PF" | "990" | "unknown"
    source_type: str = ""   # "propublica" | "irs" | "unknown"

@dataclass
class BaseDiagnostics:
    # universal
    form_type_detected: str = "unknown"
    source_type_detected: str = "unknown"
    file_ext: str = ""
    page_count: int = 0
    empty_page_count: int = 0
    empty_page_ratio: float = 0.0

    # schedule coverage
    schedule_i_pages_detected: int = 0
    schedule_i_grants_detected: int = 0

    # attachment / “see statement” detection
    attachment_reference_detected: bool = False
    attachment_reference_phrases: List[str] = field(default_factory=list)
    attachment_reference_amount: Optional[int] = None
    grants_table_incomplete: bool = False

    # totals
    reported_total_3a: Optional[int] = None
    reported_total_3b: Optional[int] = None
    computed_total_3a: Optional[int] = None
    computed_total_3b: Optional[int] = None
    total_mismatch_3a: bool = False
    total_mismatch_3b: bool = False

    # confidence
    confidence_score: int = 0
    confidence_grade: str = "?"
    confidence_reasons: List[str] = field(default_factory=list)
    confidence_penalties: List[str] = field(default_factory=list)

    # messaging
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class ReturnParseResult:
    grants_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    people_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    foundation_meta: FoundationMeta = field(default_factory=FoundationMeta)
    diagnostics: BaseDiagnostics = field(default_factory=BaseDiagnostics)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "grants_df": self.grants_df,
            "people_df": self.people_df,
            "foundation_meta": asdict(self.foundation_meta),
            "diagnostics": asdict(self.diagnostics),
            "debug": self.debug,
        }

def empty_result(filename: str, tax_year: str = "", error: str = "") -> ReturnParseResult:
    res = ReturnParseResult()
    res.foundation_meta.source_file = filename
    res.foundation_meta.tax_year = tax_year
    res.diagnostics.file_ext = filename.split(".")[-1].lower() if "." in filename else ""
    if error:
        res.diagnostics.errors.append(f"Parse error: {error}")
    res.diagnostics.warnings.append("No structured grant data extracted.")
    return res
