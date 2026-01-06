"""
IRS 990-EZ XML Parser

Extracts officers, directors, trustees, and basic grant information from
IRS Form 990-EZ electronic filings.

990-EZ is the "Short Form" used by small exempt organizations with:
- Gross receipts < $200,000, AND
- Total assets < $500,000

Key differences from 990 and 990-PF:
- No Schedule I (detailed grants)
- Grant total only on Line 10 (GrantsAndSimilarAmountsPaidAmt)
- Grant recipients may be listed in Schedule O (supplemental information)
- Officers/Directors in Part IV (OfficerDirectorTrusteeEmplGrp)

VERSION HISTORY:
----------------
v1.0.0 (2026-01-06): Initial release
- Extract officers/directors/trustees from Part IV
- Extract grant total from Line 10
- Parse Schedule O for grant recipient names/amounts
- Full diagnostic output matching other parsers
"""

import xml.etree.ElementTree as ET
import pandas as pd
import re
from typing import Optional
from datetime import datetime

PARSER_VERSION = "1.0.0"

# IRS e-file namespace
NS = {'irs': 'http://www.irs.gov/efile'}


def _get_text(element, xpath: str, ns: dict = NS) -> str:
    """Safely get text from an XML element."""
    if element is None:
        return ""
    found = element.find(xpath, ns)
    return found.text.strip() if found is not None and found.text else ""


def _get_float(element, xpath: str, ns: dict = NS) -> float:
    """Safely get float from an XML element."""
    text = _get_text(element, xpath, ns)
    if not text:
        return 0.0
    try:
        return float(text.replace(',', ''))
    except ValueError:
        return 0.0


def detect_990ez(file_bytes: bytes) -> bool:
    """
    Detect if XML file is a 990-EZ form.
    
    Checks for:
    - <ReturnTypeCd>990EZ</ReturnTypeCd> in header
    - <IRS990EZ> element in body
    """
    # Handle BOM
    if file_bytes.startswith(b'\xef\xbb\xbf'):
        file_bytes = file_bytes[3:]
    
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError:
        return False
    
    # Check ReturnTypeCd
    return_type = root.find('.//irs:ReturnTypeCd', NS)
    if return_type is not None and return_type.text == '990EZ':
        return True
    
    # Check for IRS990EZ element
    form_990ez = root.find('.//irs:IRS990EZ', NS)
    return form_990ez is not None


def _extract_foundation_meta(root) -> dict:
    """Extract organization metadata from 990-EZ header."""
    header = root.find('.//irs:ReturnHeader', NS)
    filer = root.find('.//irs:Filer', NS) if header is not None else None
    form = root.find('.//irs:IRS990EZ', NS)
    
    # Get tax year
    tax_year = _get_text(header, 'irs:TaxYr', NS)
    if not tax_year:
        # Try to extract from TaxPeriodEndDt
        end_date = _get_text(header, 'irs:TaxPeriodEndDt', NS)
        if end_date:
            tax_year = end_date[:4]
    
    # Get EIN
    ein = _get_text(filer, 'irs:EIN', NS) if filer is not None else ""
    if ein and len(ein) == 9:
        ein = f"{ein[:2]}-{ein[2:]}"
    
    # Get organization name
    org_name = _get_text(filer, './/irs:BusinessNameLine1Txt', NS) if filer is not None else ""
    if not org_name:
        org_name = _get_text(filer, './/irs:BusinessNameLine1', NS) if filer is not None else ""
    
    # Get address
    address = filer.find('.//irs:USAddress', NS) if filer is not None else None
    city = _get_text(address, 'irs:CityNm', NS) or _get_text(address, 'irs:City', NS)
    state = _get_text(address, 'irs:StateAbbreviationCd', NS) or _get_text(address, 'irs:State', NS)
    zip_code = _get_text(address, 'irs:ZIPCd', NS) or _get_text(address, 'irs:ZIPCode', NS)
    
    # Get assets
    assets_eoy = _get_float(form, './/irs:Form990TotalAssetsGrp/irs:EOYAmt', NS)
    
    # Get primary exempt purpose
    exempt_purpose = _get_text(form, 'irs:PrimaryExemptPurposeTxt', NS)
    
    return {
        'foundation_name': org_name,
        'foundation_ein': ein,
        'tax_year': tax_year,
        'city': city,
        'state': state,
        'zip': zip_code,
        'country': 'US',
        'form_type': '990-EZ',
        'total_assets': assets_eoy,
        'exempt_purpose': exempt_purpose,
        'source_type': 'xml',
    }


def _extract_officers_directors(root) -> pd.DataFrame:
    """
    Extract officers, directors, trustees, and key employees from Part IV.
    
    XML Path: //IRS990EZ/OfficerDirectorTrusteeEmplGrp
    """
    people = []
    
    # Find all officer/director/trustee groups
    groups = root.findall('.//irs:IRS990EZ/irs:OfficerDirectorTrusteeEmplGrp', NS)
    
    for grp in groups:
        name = _get_text(grp, 'irs:PersonNm', NS)
        if not name:
            # Try business name (sometimes entities serve on boards)
            name = _get_text(grp, './/irs:BusinessNameLine1Txt', NS)
        
        if not name:
            continue
        
        title = _get_text(grp, 'irs:TitleTxt', NS)
        hours = _get_text(grp, 'irs:AverageHrsPerWkDevotedToPosRt', NS)
        compensation = _get_float(grp, 'irs:CompensationAmt', NS)
        benefits = _get_float(grp, 'irs:EmployeeBenefitProgramAmt', NS)
        expense_allowance = _get_float(grp, 'irs:ExpenseAccountOtherAllwncAmt', NS)
        
        # Try to parse hours as float
        hours_float = None
        if hours:
            try:
                hours_float = float(hours)
            except ValueError:
                pass
        
        people.append({
            'name': name,
            'title': title,
            'hours_per_week': hours_float,
            'compensation': compensation,
            'benefits': benefits,
            'expense_allowance': expense_allowance,
        })
    
    return pd.DataFrame(people)


def _parse_schedule_o_grants(root) -> list:
    """
    Parse Schedule O for grant recipient information.
    
    990-EZ doesn't have Schedule I, but grant details are often
    provided in Schedule O (supplemental information) with references
    to "Part I, Line 10".
    
    Example text:
    "includes pass-through grants to: Hancock County Soil & Water 
    Conservation District - $ 17,400 St. John Valley Soil & Water 
    Conservation District - $ 6,000 Spruce Mountain High School - $ 3,000"
    
    Returns list of dicts with grantee_name, grant_amount.
    """
    grants = []
    
    # Find Schedule O entries
    schedule_o = root.find('.//irs:IRS990ScheduleO', NS)
    if schedule_o is None:
        return grants
    
    # Look for entries referencing Line 10 (grants)
    details = schedule_o.findall('.//irs:SupplementalInformationDetail', NS)
    
    for detail in details:
        ref = _get_text(detail, 'irs:FormAndLineReferenceDesc', NS).lower()
        explanation = _get_text(detail, 'irs:ExplanationTxt', NS)
        
        # Check if this is grant-related (Line 10)
        if 'line 10' not in ref and 'grants' not in ref.lower():
            continue
        
        if not explanation:
            continue
        
        # Try to parse grant recipients and amounts
        # Pattern: "Organization Name - $ Amount" or "Organization Name $Amount"
        # Also handles: "Organization Name - Amount" (no dollar sign)
        
        # Split on common delimiters that separate grants
        # Look for patterns like "Org Name - $ 17,400"
        pattern = r'([A-Za-z][^$\-\n]+?)\s*[-–—]\s*\$?\s*([\d,]+(?:\.\d{2})?)'
        matches = re.findall(pattern, explanation)
        
        for match in matches:
            org_name = match[0].strip()
            amount_str = match[1].replace(',', '')
            
            # Clean up org name - remove common prefixes
            # First, if there's a colon, take only the part after the last colon
            if ':' in org_name:
                org_name = org_name.split(':')[-1].strip()
            
            # Then remove any remaining grant-related prefixes
            org_name = re.sub(
                r'^(includes?\s*)?(pass[-\s]?through\s*)?(grants?\s*)?(to:?\s*)?',
                '', 
                org_name, 
                flags=re.IGNORECASE
            )
            org_name = org_name.strip()
            
            if not org_name:
                continue
            
            try:
                amount = float(amount_str)
            except ValueError:
                continue
            
            if amount > 0:
                grants.append({
                    'grantee_name': org_name,
                    'grant_amount': amount,
                    'grant_bucket': 'schedule_o',  # 990-EZ specific bucket
                    'grant_purpose_raw': '',
                })
    
    return grants


def parse_990ez_xml(file_bytes: bytes, source_file: str = "", 
                    region_spec: dict = None) -> dict:
    """
    Parse IRS 990-EZ XML file and extract organizations and people.
    
    Args:
        file_bytes: Raw XML file content
        source_file: Original filename for tracking
        region_spec: Optional region filter definition
    
    Returns:
        Dictionary with:
        - grants_df: DataFrame of grants (may be empty or from Schedule O)
        - people_df: DataFrame of officers/directors/trustees
        - foundation_meta: Organization metadata
        - diagnostics: Parsing diagnostics
    """
    diagnostics = {
        'form_type_detected': '990-EZ',
        'source_type_detected': 'xml',
        'file_ext': 'xml',
        'parser_version': PARSER_VERSION,
        'errors': [],
        'warnings': [],
        'notes': [],
    }
    
    # Handle BOM
    if file_bytes.startswith(b'\xef\xbb\xbf'):
        file_bytes = file_bytes[3:]
    
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError as e:
        diagnostics['errors'].append(f"XML parse error: {e}")
        return {
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'foundation_meta': {'source_file': source_file, 'form_type': '990-EZ'},
            'diagnostics': diagnostics,
        }
    
    # Verify this is a 990-EZ
    if not detect_990ez(file_bytes):
        diagnostics['errors'].append("File does not appear to be a 990-EZ form")
        diagnostics['form_type_detected'] = 'unknown'
    
    # Extract foundation metadata
    meta = _extract_foundation_meta(root)
    meta['source_file'] = source_file
    
    # Extract officers/directors/trustees
    people_df = _extract_officers_directors(root)
    diagnostics['board_count'] = len(people_df)
    
    # Get grant total from Line 10
    form = root.find('.//irs:IRS990EZ', NS)
    grants_total = _get_float(form, 'irs:GrantsAndSimilarAmountsPaidAmt', NS)
    diagnostics['grants_total_line_10'] = grants_total
    
    # Try to parse Schedule O for grant details
    schedule_o_grants = _parse_schedule_o_grants(root)
    diagnostics['schedule_o_grants_parsed'] = len(schedule_o_grants)
    
    # Build grants DataFrame
    if schedule_o_grants:
        grants_df = pd.DataFrame(schedule_o_grants)
        # Add foundation info
        grants_df['foundation_name'] = meta.get('foundation_name', '')
        grants_df['foundation_ein'] = meta.get('foundation_ein', '')
        grants_df['tax_year'] = meta.get('tax_year', '')
        grants_df['grantee_city'] = ''
        grants_df['grantee_state'] = ''
        grants_df['grantee_country'] = 'US'
        grants_df['source_file'] = source_file
        grants_df['source_system'] = 'IRS_990EZ'
        grants_df['fiscal_year'] = meta.get('tax_year', '')
        
        # Compute parsed total vs reported
        parsed_total = grants_df['grant_amount'].sum()
        if grants_total > 0:
            match_pct = (parsed_total / grants_total) * 100
            diagnostics['grants_match_pct'] = round(match_pct, 1)
            if match_pct < 90:
                diagnostics['warnings'].append(
                    f"Schedule O grants (${parsed_total:,.0f}) account for "
                    f"{match_pct:.0f}% of Line 10 total (${grants_total:,.0f})"
                )
        
        # Apply region tagging if specified
        if region_spec and region_spec.get('id') != 'none':
            # Note: 990-EZ typically doesn't have grantee addresses
            # Default to True since we can't verify
            grants_df['region_relevant'] = True
            diagnostics['notes'].append("Region tagging skipped - no grantee addresses in 990-EZ")
    else:
        grants_df = pd.DataFrame()
        if grants_total > 0:
            diagnostics['warnings'].append(
                f"${grants_total:,.0f} in grants reported on Line 10, "
                f"but no detail found in Schedule O"
            )
    
    diagnostics['grants_count'] = len(grants_df)
    
    # Compute confidence score
    confidence_score = 100
    confidence_reasons = []
    confidence_penalties = []
    
    if people_df.empty:
        confidence_score -= 30
        confidence_penalties.append(("No officers/directors found", -30))
    else:
        confidence_reasons.append(f"Found {len(people_df)} officers/directors/trustees")
    
    if grants_total > 0 and grants_df.empty:
        confidence_score -= 20
        confidence_penalties.append(("Grants reported but no detail parsed", -20))
    elif not grants_df.empty:
        confidence_reasons.append(f"Parsed {len(grants_df)} grants from Schedule O")
    
    if not meta.get('foundation_name'):
        confidence_score -= 20
        confidence_penalties.append(("Organization name not found", -20))
    
    if not meta.get('foundation_ein'):
        confidence_score -= 10
        confidence_penalties.append(("EIN not found", -10))
    
    diagnostics['confidence_score'] = max(0, confidence_score)
    diagnostics['confidence_grade'] = (
        'high' if confidence_score >= 80 else
        'medium' if confidence_score >= 60 else
        'low' if confidence_score >= 40 else
        'failed'
    )
    diagnostics['confidence_reasons'] = confidence_reasons
    diagnostics['confidence_penalties'] = confidence_penalties
    
    return {
        'grants_df': grants_df,
        'people_df': people_df,
        'foundation_meta': meta,
        'diagnostics': diagnostics,
    }


# =============================================================================
# Convenience function for testing
# =============================================================================

def parse_file(filepath: str) -> dict:
    """Parse a 990-EZ XML file from disk."""
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    return parse_990ez_xml(file_bytes, source_file=filepath)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python irs990ez_xml_parser.py <990ez_file.xml>")
        sys.exit(1)
    
    result = parse_file(sys.argv[1])
    
    print("\n=== Foundation Metadata ===")
    for k, v in result['foundation_meta'].items():
        print(f"  {k}: {v}")
    
    print(f"\n=== Officers/Directors ({len(result['people_df'])}) ===")
    if not result['people_df'].empty:
        print(result['people_df'].to_string())
    
    print(f"\n=== Grants ({len(result['grants_df'])}) ===")
    if not result['grants_df'].empty:
        print(result['grants_df'][['grantee_name', 'grant_amount']].to_string())
    
    print("\n=== Diagnostics ===")
    diag = result['diagnostics']
    print(f"  Confidence: {diag['confidence_grade']} ({diag['confidence_score']}/100)")
    print(f"  Board count: {diag['board_count']}")
    print(f"  Grants total (Line 10): ${diag.get('grants_total_line_10', 0):,.0f}")
    print(f"  Schedule O grants parsed: {diag.get('schedule_o_grants_parsed', 0)}")
    
    if diag['warnings']:
        print("\n  Warnings:")
        for w in diag['warnings']:
            print(f"    - {w}")
