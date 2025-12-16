"""
c4c_utils/irs990_xml_parser.py

Parse IRS Form 990 XML files (e-filed returns from ProPublica).
Extracts grants from Schedule I (Grants and Other Assistance to Organizations, 
Governments, and Individuals in the United States).

This parser handles public charities (Form 990), not private foundations (990-PF).

Advantages over PDF parsing:
- No page-break errors
- Structured data with consistent field names
- Better address data
- EIN of grantees included

Usage:
    from c4c_utils.irs990_xml_parser import parse_990_xml
    
    result = parse_990_xml(xml_bytes, filename)
    # Returns: {grants_df, people_df, foundation_meta, diagnostics}
"""

from __future__ import annotations

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
class ScheduleIGrant:
    """Represents a single grant from Schedule I."""
    grantee_name: str
    grantee_ein: str = ""
    grantee_city: str = ""
    grantee_state: str = ""
    grantee_zip: str = ""
    grantee_country: str = "US"
    grantee_province_raw: str = ""
    grantee_country_raw: str = ""
    cash_grant_amount: int = 0
    non_cash_amount: int = 0
    purpose: str = ""
    irc_section: str = ""  # e.g., "501(C)(3)"


@dataclass
class BoardMember:
    """Represents a board member/officer from Part VII."""
    name: str
    title: str = ""
    hours_per_week: float = 0.0
    compensation_from_org: int = 0
    compensation_from_related: int = 0
    other_compensation: int = 0


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
    
    CA_CODES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}
    if province in CA_CODES:
        return province
    
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
# Schedule I Grant Extraction
# =============================================================================

def _extract_schedule_i_grants(root: ET.Element) -> Tuple[List[ScheduleIGrant], Dict[str, Any]]:
    """
    Extract grants from Schedule I (IRS990ScheduleI).
    
    Schedule I Part II contains: RecipientTable entries for organizations.
    
    Returns: (list of grants, schedule_i_diagnostics)
    """
    grants = []
    diag = {
        "schedule_i_found": False,
        "schedule_i_part_ii_count": 0,
        "total_501c3_org_count": 0,
        "total_other_org_count": 0,
    }
    
    # Find Schedule I
    schedule_i = root.find(".//irs:IRS990ScheduleI", IRS_NS)
    if schedule_i is None:
        diag["schedule_i_reason"] = "missing_schedule_i"
        return grants, diag
    
    diag["schedule_i_found"] = True
    
    # Get org counts from Schedule I summary
    diag["total_501c3_org_count"] = _find_int(schedule_i, "irs:Total501c3OrgCnt", 0)
    diag["total_other_org_count"] = _find_int(schedule_i, "irs:TotalOtherOrgCnt", 0)
    
    # Extract each RecipientTable entry (Part II - grants to orgs)
    for recipient in schedule_i.findall("irs:RecipientTable", IRS_NS):
        grant = _parse_recipient_table(recipient)
        if grant:
            grants.append(grant)
    
    diag["schedule_i_part_ii_count"] = len(grants)
    
    if not grants:
        diag["schedule_i_reason"] = "schedule_i_found_but_no_records"
    else:
        diag["schedule_i_reason"] = "ok"
    
    return grants, diag


def _parse_recipient_table(recipient: ET.Element) -> Optional[ScheduleIGrant]:
    """Parse a single RecipientTable entry from Schedule I."""
    
    # Get grantee name
    name = _find(recipient, "irs:RecipientBusinessName/irs:BusinessNameLine1Txt")
    if not name:
        name = _find(recipient, "irs:RecipientPersonNm")
    
    if not name:
        return None
    
    # Get EIN (big advantage of XML!)
    ein = _find(recipient, "irs:RecipientEIN")
    if ein and len(ein) == 9:
        ein = f"{ein[:2]}-{ein[2:]}"
    
    # Get address (US or foreign)
    us_addr = recipient.find("irs:USAddress", IRS_NS)
    foreign_addr = recipient.find("irs:ForeignAddress", IRS_NS)
    
    if us_addr is not None:
        addr = _extract_us_address(us_addr)
    elif foreign_addr is not None:
        addr = _extract_foreign_address(foreign_addr)
    else:
        addr = {"city": "", "state": "", "zip": "", "country": "US"}
    
    # Get amounts
    cash_amount = _find_int(recipient, "irs:CashGrantAmt", 0)
    non_cash_amount = _find_int(recipient, "irs:NonCashAssistanceAmt", 0)
    
    # Get purpose and IRC section
    purpose = _find(recipient, "irs:PurposeOfGrantTxt")
    irc_section = _find(recipient, "irs:IRCSectionDesc")
    
    return ScheduleIGrant(
        grantee_name=name,
        grantee_ein=ein,
        grantee_city=addr.get("city", ""),
        grantee_state=addr.get("state", ""),
        grantee_zip=addr.get("zip", ""),
        grantee_country=addr.get("country", "US"),
        grantee_province_raw=addr.get("province_raw", ""),
        grantee_country_raw=addr.get("country_raw", ""),
        cash_grant_amount=cash_amount,
        non_cash_amount=non_cash_amount,
        purpose=purpose,
        irc_section=irc_section,
    )


# =============================================================================
# Board Member Extraction (Part VII Section A)
# =============================================================================

def _extract_board_members(root: ET.Element) -> List[BoardMember]:
    """Extract officers, directors, trustees from Part VII Section A."""
    members = []
    
    for grp in root.findall(".//irs:Form990PartVIISectionAGrp", IRS_NS):
        name = _find(grp, "irs:PersonNm")
        if not name:
            # Sometimes it's in BusinessName
            name = _find(grp, "irs:BusinessName/irs:BusinessNameLine1Txt")
        
        if not name:
            continue
        
        members.append(BoardMember(
            name=name,
            title=_find(grp, "irs:TitleTxt"),
            hours_per_week=_find_float(grp, "irs:AverageHoursPerWeekRt"),
            compensation_from_org=_find_int(grp, "irs:ReportableCompFromOrgAmt"),
            compensation_from_related=_find_int(grp, "irs:ReportableCompFromRltdOrgAmt"),
            other_compensation=_find_int(grp, "irs:OtherCompensationAmt"),
        ))
    
    return members


# =============================================================================
# Organization Metadata
# =============================================================================

def _extract_org_meta(root: ET.Element) -> Dict[str, Any]:
    """Extract organization metadata from ReturnHeader."""
    
    # EIN
    ein = _find(root, ".//irs:ReturnHeader/irs:Filer/irs:EIN")
    if not ein:
        ein = _find(root, ".//irs:Filer/irs:EIN")
    
    if ein and len(ein) == 9:
        ein = f"{ein[:2]}-{ein[2:]}"
    
    # Organization name
    name = _find(root, ".//irs:ReturnHeader/irs:Filer/irs:BusinessName/irs:BusinessNameLine1Txt")
    if not name:
        name = _find(root, ".//irs:Filer/irs:BusinessName/irs:BusinessNameLine1Txt")
    
    # Tax year
    tax_year = _find(root, ".//irs:ReturnHeader/irs:TaxYr")
    
    # Tax period end date
    tax_period_end = _find(root, ".//irs:ReturnHeader/irs:TaxPeriodEndDt")
    if not tax_year and tax_period_end:
        tax_year = tax_period_end[:4]
    
    # Grants paid amount from Part I (line 13)
    grants_paid = _find_int(root, ".//irs:IRS990/irs:CYGrantsAndSimilarPaidAmt", 0)
    
    return {
        "foundation_name": name,
        "foundation_ein": ein,
        "tax_year": tax_year,
        "tax_period_end": tax_period_end,
        "reported_grants_paid": grants_paid,
    }


# =============================================================================
# Main Parser Function
# =============================================================================

def parse_990_xml(
    xml_bytes: bytes,
    filename: str = "",
    tax_year_override: str = ""
) -> Dict[str, Any]:
    """
    Parse IRS Form 990 XML file (Schedule I grants).
    
    Args:
        xml_bytes: Raw XML content
        filename: Source filename (for diagnostics)
        tax_year_override: Override tax year (if needed)
    
    Returns:
        Dict with keys:
            - foundation_meta: Organization info
            - grants_df: DataFrame of Schedule I grants
            - people_df: DataFrame of board members
            - diagnostics: Parsing diagnostics
    """
    diagnostics = {
        "parser_version": PARSER_VERSION,
        "source_file": filename,
        "source_type": "xml",
        "form_type_detected": "990",
        "warnings": [],
        "errors": [],
    }
    
    try:
        # Handle BOM if present
        if xml_bytes.startswith(b'\xef\xbb\xbf'):
            xml_bytes = xml_bytes[3:]
        
        root = ET.fromstring(xml_bytes)
        
        # Verify this is a Form 990 (not 990-PF or 990-EZ)
        return_type = _find(root, ".//irs:ReturnHeader/irs:ReturnTypeCd")
        if return_type and return_type.upper() not in ("990",):
            diagnostics["warnings"].append(f"Expected 990 but found ReturnTypeCd={return_type}")
        
        # Extract organization metadata
        meta = _extract_org_meta(root)
        
        if tax_year_override:
            meta["tax_year"] = tax_year_override
        
        # Extract Schedule I grants
        grants, schedule_i_diag = _extract_schedule_i_grants(root)
        
        # Compute totals
        computed_cash_total = sum(g.cash_grant_amount for g in grants)
        computed_non_cash_total = sum(g.non_cash_amount for g in grants)
        computed_total = computed_cash_total + computed_non_cash_total
        
        # Extract board members
        board_members = _extract_board_members(root)
        
        # Build diagnostics
        diagnostics.update(schedule_i_diag)
        diagnostics.update({
            "grants_count": len(grants),
            "grants_cash_total": computed_cash_total,
            "grants_non_cash_total": computed_non_cash_total,
            "grants_total": computed_total,
            "reported_grants_paid": meta.get("reported_grants_paid", 0),
            "board_count": len(board_members),
        })
        
        # Note: Schedule I may not include all grants (only those to orgs, not individuals)
        # So comparing to Part I line 13 may show discrepancy
        reported = meta.get("reported_grants_paid", 0)
        if reported and reported > 0:
            coverage_pct = (computed_total / reported) * 100
            diagnostics["schedule_i_coverage_pct"] = round(coverage_pct, 1)
            diagnostics["schedule_i_coverage_note"] = (
                "Schedule I covers grants to organizations; "
                "Part III individual grants may account for difference"
            )
        
        # Build grants DataFrame
        if grants:
            grants_df = pd.DataFrame([{
                'grantee_name': g.grantee_name,
                'grantee_ein': g.grantee_ein,
                'grantee_city': g.grantee_city,
                'grantee_state': g.grantee_state,
                'grantee_zip': g.grantee_zip,
                'grantee_country': g.grantee_country,
                'grant_amount': g.cash_grant_amount + g.non_cash_amount,
                'cash_grant_amount': g.cash_grant_amount,
                'non_cash_amount': g.non_cash_amount,
                'grant_purpose_raw': g.purpose,
                'irc_section': g.irc_section,
                'grant_bucket': 'schedule_i',
            } for g in grants])
        else:
            grants_df = pd.DataFrame()
        
        # Build people DataFrame
        if board_members:
            people_df = pd.DataFrame([{
                'name': m.name,
                'title': m.title,
                'hours': m.hours_per_week,
                'compensation': m.compensation_from_org + m.compensation_from_related,
                'compensation_from_org': m.compensation_from_org,
                'compensation_from_related': m.compensation_from_related,
                'other_compensation': m.other_compensation,
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
        print("Usage: python irs990_xml_parser.py <990.xml>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    with open(filepath, "rb") as f:
        xml_bytes = f.read()
    
    result = parse_990_xml(xml_bytes, filepath)
    
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
    print(f"Form Type: {diag.get('form_type_detected', 'N/A')}")
    
    print(f"\n📊 SCHEDULE I GRANTS")
    print(f"  Schedule I found: {diag.get('schedule_i_found', False)}")
    print(f"  Grants extracted: {diag.get('grants_count', 0)}")
    print(f"  Cash total: ${diag.get('grants_cash_total', 0):,}")
    print(f"  Non-cash total: ${diag.get('grants_non_cash_total', 0):,}")
    print(f"  Combined total: ${diag.get('grants_total', 0):,}")
    
    reported = diag.get('reported_grants_paid', 0)
    if reported:
        print(f"  Part I reported: ${reported:,}")
        print(f"  Coverage: {diag.get('schedule_i_coverage_pct', 0)}%")
    
    print(f"\n👥 BOARD/OFFICERS: {diag.get('board_count', 0)}")
    if not people_df.empty:
        for _, row in people_df.head(5).iterrows():
            comp = row.get('compensation', 0)
            print(f"  - {row['name']} ({row['title']})" + (f" ${comp:,}" if comp else ""))
    
    print(f"\n📋 SAMPLE GRANTS")
    if not grants_df.empty:
        for _, row in grants_df.head(5).iterrows():
            ein_str = f" [{row['grantee_ein']}]" if row.get('grantee_ein') else ""
            print(f"  - {row['grantee_name'][:40]}{ein_str}: ${row['grant_amount']:,}")
    
    if diag.get("warnings"):
        print(f"\n⚠️  Warnings: {diag['warnings']}")
    if diag.get("errors"):
        print(f"\n❌ Errors: {diag['errors']}")
