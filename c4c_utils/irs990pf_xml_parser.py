"""
c4c_utils/irs990pf_xml_parser.py

Parse IRS Form 990-PF XML files (e-filed returns from ProPublica).

Advantages over PDF parsing:
- No page-break errors
- Structured data with consistent field names
- Exact totals for QA reconciliation
- Better address data (especially for Canada)

Usage:
    from c4c_utils.irs990pf_xml_parser import parse_990pf_xml
    
    result = parse_990pf_xml(xml_bytes, filename)
    # Returns: {grants_df, people_df, foundation_meta, diagnostics}
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Parser version
PARSER_VERSION = "1.0-xml"

# IRS e-file namespace
IRS_NS = {"irs": "http://www.irs.gov/efile"}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Grant:
    """Represents a single grant."""
    grantee_name: str
    grantee_city: str = ""
    grantee_state: str = ""
    grantee_zip: str = ""
    grantee_country: str = "US"
    grantee_province_raw: str = ""
    grantee_country_raw: str = ""
    amount: int = 0
    purpose: str = ""
    relationship: str = ""
    foundation_status: str = ""
    grant_bucket: str = "3a"  # "3a" or "3b"


@dataclass
class BoardMember:
    """Represents a board member/officer."""
    name: str
    title: str = ""
    hours_per_week: float = 0.0
    compensation: int = 0
    benefits: int = 0
    expense_allowance: int = 0
    city: str = ""
    state: str = ""


# =============================================================================
# XML Helper Functions
# =============================================================================

def _find(element: ET.Element, xpath: str, default: str = "") -> str:
    """Find element text with namespace support."""
    el = element.find(xpath, IRS_NS)
    if el is not None and el.text:
        return el.text.strip()
    return default


def _find_int(element: ET.Element, xpath: str, default: int = 0) -> int:
    """Find element text and convert to int."""
    text = _find(element, xpath, "")
    if text:
        # Remove commas and convert
        text = text.replace(",", "")
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _find_float(element: ET.Element, xpath: str, default: float = 0.0) -> float:
    """Find element text and convert to float."""
    text = _find(element, xpath, "")
    if text:
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _extract_us_address(addr_el: Optional[ET.Element]) -> Dict[str, str]:
    """Extract US address fields."""
    if addr_el is None:
        return {"city": "", "state": "", "zip": "", "country": "US"}
    
    return {
        "city": _find(addr_el, "irs:CityNm"),
        "state": _find(addr_el, "irs:StateAbbreviationCd"),
        "zip": _find(addr_el, "irs:ZIPCd"),
        "country": "US",
    }


def _extract_foreign_address(addr_el: Optional[ET.Element]) -> Dict[str, str]:
    """Extract foreign address fields."""
    if addr_el is None:
        return {"city": "", "state": "", "zip": "", "country": "", "province_raw": "", "country_raw": ""}
    
    country_code = _find(addr_el, "irs:CountryCd")
    province = _find(addr_el, "irs:ProvinceOrStateNm")
    
    # Normalize Canada provinces
    state = ""
    if country_code.upper() in ("CA", "CAN", "CANADA"):
        state = _normalize_canada_province(province)
        country_code = "CA"
    
    return {
        "city": _find(addr_el, "irs:CityNm"),
        "state": state,
        "zip": _find(addr_el, "irs:ForeignPostalCd"),
        "country": country_code,
        "province_raw": province,
        "country_raw": _find(addr_el, "irs:CountryCd"),
    }


def _normalize_canada_province(province: str) -> str:
    """Normalize Canadian province to 2-letter code."""
    province = province.upper().strip()
    
    # Already a code
    CA_CODES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}
    if province in CA_CODES:
        return province
    
    # Full name mapping
    name_map = {
        "ALBERTA": "AB",
        "BRITISH COLUMBIA": "BC",
        "MANITOBA": "MB",
        "NEW BRUNSWICK": "NB",
        "NEWFOUNDLAND": "NL",
        "NEWFOUNDLAND AND LABRADOR": "NL",
        "NOVA SCOTIA": "NS",
        "NORTHWEST TERRITORIES": "NT",
        "NUNAVUT": "NU",
        "ONTARIO": "ON",
        "PRINCE EDWARD ISLAND": "PE",
        "QUEBEC": "QC",
        "SASKATCHEWAN": "SK",
        "YUKON": "YT",
    }
    
    return name_map.get(province, province[:2] if len(province) >= 2 else "")


# =============================================================================
# Grant Extraction
# =============================================================================

def _extract_grants_3a(root: ET.Element) -> Tuple[List[Grant], int]:
    """
    Extract grants paid during the year (Part XV, line 3a).
    
    Returns: (list of grants, reported total from XML)
    """
    grants = []
    
    # Find all 3a grant groups
    for grp in root.findall(".//irs:GrantOrContributionPdDurYrGrp", IRS_NS):
        grant = _parse_grant_group(grp, grant_bucket="3a")
        if grant:
            grants.append(grant)
    
    # Get reported total
    reported_total = _find_int(root, ".//irs:TotalGrantOrContriPdDurYrAmt", 0)
    
    return grants, reported_total


def _extract_grants_3b(root: ET.Element) -> Tuple[List[Grant], int]:
    """
    Extract grants approved for future payment (Part XV, line 3b).
    
    Returns: (list of grants, reported total from XML)
    """
    grants = []
    
    # Find all 3b grant groups
    for grp in root.findall(".//irs:GrantOrContriApprvForFutGrp", IRS_NS):
        grant = _parse_grant_group(grp, grant_bucket="3b")
        if grant:
            grants.append(grant)
    
    # Get reported total
    reported_total = _find_int(root, ".//irs:TotalGrantOrContriApprvFutAmt", 0)
    
    return grants, reported_total


def _parse_grant_group(grp: ET.Element, grant_bucket: str) -> Optional[Grant]:
    """Parse a single grant group element."""
    
    # Get grantee name
    name = _find(grp, "irs:RecipientBusinessName/irs:BusinessNameLine1Txt")
    if not name:
        # Try person name (rare for 990-PF)
        name = _find(grp, "irs:RecipientPersonNm")
    
    if not name:
        return None
    
    # Get address (US or foreign)
    us_addr = grp.find("irs:RecipientUSAddress", IRS_NS)
    foreign_addr = grp.find("irs:RecipientForeignAddress", IRS_NS)
    
    if us_addr is not None:
        addr = _extract_us_address(us_addr)
    elif foreign_addr is not None:
        addr = _extract_foreign_address(foreign_addr)
    else:
        addr = {"city": "", "state": "", "zip": "", "country": ""}
    
    # Get amount
    amount = _find_int(grp, "irs:Amt", 0)
    
    # Get other fields
    purpose = _find(grp, "irs:GrantOrContributionPurposeTxt")
    relationship = _find(grp, "irs:RecipientRelationshipTxt")
    foundation_status = _find(grp, "irs:RecipientFoundationStatusTxt")
    
    return Grant(
        grantee_name=name,
        grantee_city=addr.get("city", ""),
        grantee_state=addr.get("state", ""),
        grantee_zip=addr.get("zip", ""),
        grantee_country=addr.get("country", "US"),
        grantee_province_raw=addr.get("province_raw", ""),
        grantee_country_raw=addr.get("country_raw", ""),
        amount=amount,
        purpose=purpose,
        relationship=relationship,
        foundation_status=foundation_status,
        grant_bucket=grant_bucket,
    )


# =============================================================================
# Board Member Extraction
# =============================================================================

def _extract_board_members(root: ET.Element) -> List[BoardMember]:
    """Extract officers, directors, trustees, and key employees."""
    members = []
    
    for grp in root.findall(".//irs:OfficerDirTrstKeyEmplGrp", IRS_NS):
        # Name is in BusinessName/BusinessNameLine1Txt (yes, weird but true)
        name = _find(grp, "irs:BusinessName/irs:BusinessNameLine1Txt")
        if not name:
            name = _find(grp, "irs:PersonNm")
        
        if not name:
            continue
        
        # Get address
        us_addr = grp.find("irs:USAddress", IRS_NS)
        addr = _extract_us_address(us_addr) if us_addr is not None else {}
        
        members.append(BoardMember(
            name=name,
            title=_find(grp, "irs:TitleTxt"),
            hours_per_week=_find_float(grp, "irs:AverageHrsPerWkDevotedToPosRt"),
            compensation=_find_int(grp, "irs:CompensationAmt"),
            benefits=_find_int(grp, "irs:EmployeeBenefitProgramAmt"),
            expense_allowance=_find_int(grp, "irs:ExpenseAccountOtherAllwncAmt"),
            city=addr.get("city", ""),
            state=addr.get("state", ""),
        ))
    
    return members


# =============================================================================
# Foundation Metadata
# =============================================================================

def _extract_foundation_meta(root: ET.Element) -> Dict[str, Any]:
    """Extract foundation metadata from ReturnHeader."""
    
    # EIN - try multiple paths
    ein = _find(root, ".//irs:ReturnHeader/irs:Filer/irs:EIN")
    if not ein:
        ein = _find(root, ".//irs:Filer/irs:EIN")
    
    # Format EIN with dash
    if ein and len(ein) == 9:
        ein = f"{ein[:2]}-{ein[2:]}"
    
    # Foundation name
    name = _find(root, ".//irs:ReturnHeader/irs:Filer/irs:BusinessName/irs:BusinessNameLine1Txt")
    if not name:
        name = _find(root, ".//irs:Filer/irs:BusinessName/irs:BusinessNameLine1Txt")
    
    # Tax year
    tax_year = _find(root, ".//irs:ReturnHeader/irs:TaxYr")
    
    # Tax period end date (as fallback for year)
    tax_period_end = _find(root, ".//irs:ReturnHeader/irs:TaxPeriodEndDt")
    if not tax_year and tax_period_end:
        tax_year = tax_period_end[:4]
    
    return {
        "foundation_name": name,
        "foundation_ein": ein,
        "tax_year": tax_year,
        "tax_period_end": tax_period_end,
    }


# =============================================================================
# Main Parser Function
# =============================================================================

def parse_990pf_xml(
    xml_bytes: bytes,
    filename: str = "",
    tax_year_override: str = ""
) -> Dict[str, Any]:
    """
    Parse IRS Form 990-PF XML file.
    
    Args:
        xml_bytes: Raw XML content
        filename: Source filename (for diagnostics)
        tax_year_override: Override tax year (if needed)
    
    Returns:
        Dict with keys:
            - foundation_meta: Foundation info
            - grants_df: DataFrame of grants
            - people_df: DataFrame of board members
            - diagnostics: Parsing diagnostics including QA totals
    """
    diagnostics = {
        "parser_version": PARSER_VERSION,
        "source_file": filename,
        "source_type": "xml",
        "form_type_detected": "990-PF",
        "warnings": [],
        "errors": [],
    }
    
    try:
        # Parse XML
        # Handle BOM if present
        if xml_bytes.startswith(b'\xef\xbb\xbf'):
            xml_bytes = xml_bytes[3:]
        
        root = ET.fromstring(xml_bytes)
        
        # Verify this is a 990-PF
        return_type = _find(root, ".//irs:ReturnHeader/irs:ReturnTypeCd")
        if return_type and return_type.upper() not in ("990PF", "990-PF"):
            diagnostics["warnings"].append(f"Expected 990-PF but found ReturnTypeCd={return_type}")
        
        # Extract foundation metadata
        meta = _extract_foundation_meta(root)
        
        # Apply tax year override if provided
        if tax_year_override:
            meta["tax_year"] = tax_year_override
        
        # Extract grants
        grants_3a, reported_total_3a = _extract_grants_3a(root)
        grants_3b, reported_total_3b = _extract_grants_3b(root)
        
        # Compute totals
        computed_total_3a = sum(g.amount for g in grants_3a)
        computed_total_3b = sum(g.amount for g in grants_3b)
        
        # Extract board members
        board_members = _extract_board_members(root)
        
        # Build diagnostics
        diagnostics.update({
            "grants_3a_count": len(grants_3a),
            "grants_3a_total": computed_total_3a,
            "grants_3b_count": len(grants_3b),
            "grants_3b_total": computed_total_3b,
            "reported_total_3a": reported_total_3a,
            "reported_total_3b": reported_total_3b,
            "total_mismatch_3a": abs(computed_total_3a - reported_total_3a) > 100 if reported_total_3a else False,
            "total_mismatch_3b": abs(computed_total_3b - reported_total_3b) > 100 if reported_total_3b else False,
            "board_count": len(board_members),
        })
        
        # Compute confidence (QA reconciliation)
        if reported_total_3a and reported_total_3a > 0:
            match_pct_3a = (computed_total_3a / reported_total_3a) * 100
            diagnostics["confidence_3a"] = {
                "match_pct": round(match_pct_3a, 1),
                "variance_pct": round(abs(100 - match_pct_3a), 1),
                "status": "excellent" if abs(100 - match_pct_3a) <= 1 else "good" if abs(100 - match_pct_3a) <= 5 else "warning",
                "confidence": "high" if abs(100 - match_pct_3a) <= 1 else "medium" if abs(100 - match_pct_3a) <= 5 else "low",
            }
        
        if reported_total_3b and reported_total_3b > 0:
            match_pct_3b = (computed_total_3b / reported_total_3b) * 100
            diagnostics["confidence_3b"] = {
                "match_pct": round(match_pct_3b, 1),
                "variance_pct": round(abs(100 - match_pct_3b), 1),
                "status": "excellent" if abs(100 - match_pct_3b) <= 1 else "good" if abs(100 - match_pct_3b) <= 5 else "warning",
                "confidence": "high" if abs(100 - match_pct_3b) <= 1 else "medium" if abs(100 - match_pct_3b) <= 5 else "low",
            }
        
        # Build grants DataFrame
        all_grants = grants_3a + grants_3b
        
        if all_grants:
            grants_df = pd.DataFrame([{
                'grantee_name': g.grantee_name,
                'grantee_city': g.grantee_city,
                'grantee_state': g.grantee_state,
                'grantee_zip': g.grantee_zip,
                'grantee_country': g.grantee_country,
                'grant_amount': g.amount,
                'grant_purpose_raw': g.purpose,
                'relationship': g.relationship,
                'foundation_status': g.foundation_status,
                'grant_bucket': g.grant_bucket,
            } for g in all_grants])
        else:
            grants_df = pd.DataFrame()
        
        # Build people DataFrame
        if board_members:
            people_df = pd.DataFrame([{
                'name': m.name,
                'title': m.title,
                'hours': m.hours_per_week,
                'compensation': m.compensation,
                'benefits': m.benefits,
                'expense_allowance': m.expense_allowance,
                'person_city': m.city,
                'person_state': m.state,
            } for m in board_members])
        else:
            people_df = pd.DataFrame()
        
        return {
            'foundation_meta': meta,
            'grants_df': grants_df,
            'people_df': people_df,
            'diagnostics': diagnostics,
        }
        
    except ET.ParseError as e:
        diagnostics["errors"].append(f"XML parse error: {str(e)}")
        return {
            'foundation_meta': {},
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'diagnostics': diagnostics,
        }
    except Exception as e:
        diagnostics["errors"].append(f"Unexpected error: {str(e)}")
        return {
            'foundation_meta': {},
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'diagnostics': diagnostics,
        }


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python irs990pf_xml_parser.py <990pf.xml>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    with open(filepath, "rb") as f:
        xml_bytes = f.read()
    
    result = parse_990pf_xml(xml_bytes, filepath)
    
    meta = result["foundation_meta"]
    diag = result["diagnostics"]
    grants_df = result["grants_df"]
    people_df = result["people_df"]
    
    print(f"\n{'='*60}")
    print(f"🏛️  {meta.get('foundation_name', 'Unknown')}")
    print(f"{'='*60}")
    print(f"EIN: {meta.get('foundation_ein', 'N/A')}")
    print(f"Tax Year: {meta.get('tax_year', 'N/A')}")
    print(f"Parser: {diag.get('parser_version', 'N/A')}")
    
    print(f"\n📊 GRANTS SUMMARY")
    print(f"  3a (paid): {diag.get('grants_3a_count', 0):,} grants = ${diag.get('grants_3a_total', 0):,}")
    print(f"     Reported: ${diag.get('reported_total_3a', 0):,}")
    if diag.get('confidence_3a'):
        print(f"     Match: {diag['confidence_3a']['match_pct']}% ({diag['confidence_3a']['status']})")
    
    print(f"  3b (future): {diag.get('grants_3b_count', 0):,} grants = ${diag.get('grants_3b_total', 0):,}")
    print(f"     Reported: ${diag.get('reported_total_3b', 0):,}")
    if diag.get('confidence_3b'):
        print(f"     Match: {diag['confidence_3b']['match_pct']}% ({diag['confidence_3b']['status']})")
    
    print(f"\n👥 BOARD MEMBERS: {diag.get('board_count', 0)}")
    if not people_df.empty:
        for _, row in people_df.head(5).iterrows():
            print(f"  - {row['name']} ({row['title']})")
    
    print(f"\n📋 SAMPLE GRANTS")
    if not grants_df.empty:
        for _, row in grants_df.head(5).iterrows():
            print(f"  - {row['grantee_name'][:40]}: ${row['grant_amount']:,}")
    
    if diag.get("warnings"):
        print(f"\n⚠️  Warnings: {diag['warnings']}")
    if diag.get("errors"):
        print(f"\n❌ Errors: {diag['errors']}")
