"""
c4c_utils/irs_xml_dispatcher.py

Unified dispatcher for IRS XML files.
Detects form type (990-PF vs 990) and routes to the appropriate parser.

Usage:
    from c4c_utils.irs_xml_dispatcher import parse_irs_xml
    
    result = parse_irs_xml(xml_bytes, filename)
    # Returns: {grants_df, people_df, foundation_meta, diagnostics}
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple
import pandas as pd

# Import parsers
from c4c_utils.irs990pf_xml_parser import parse_990pf_xml, PARSER_VERSION as PF_VERSION
from c4c_utils.irs990_xml_parser import parse_990_xml, PARSER_VERSION as F990_VERSION

# IRS namespace
IRS_NS = {"irs": "http://www.irs.gov/efile"}


def detect_form_type(xml_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Detect the IRS form type from XML content.
    
    Returns:
        Tuple of (form_type, error_message)
        form_type: "990-PF", "990", "990-EZ", or "unknown"
    """
    try:
        # Handle BOM
        if xml_bytes.startswith(b'\xef\xbb\xbf'):
            xml_bytes = xml_bytes[3:]
        
        # Parse just enough to get the return type
        root = ET.fromstring(xml_bytes)
        
        # Look for ReturnTypeCd in header
        return_type_el = root.find(".//irs:ReturnHeader/irs:ReturnTypeCd", IRS_NS)
        if return_type_el is not None and return_type_el.text:
            return_type = return_type_el.text.strip().upper()
            
            if return_type in ("990PF", "990-PF"):
                return "990-PF", None
            elif return_type == "990":
                return "990", None
            elif return_type == "990EZ":
                return "990-EZ", None
            else:
                return return_type, None
        
        # Fallback: check for form-specific elements
        if root.find(".//irs:IRS990PF", IRS_NS) is not None:
            return "990-PF", None
        if root.find(".//irs:IRS990", IRS_NS) is not None:
            return "990", None
        if root.find(".//irs:IRS990EZ", IRS_NS) is not None:
            return "990-EZ", None
        
        return "unknown", "Could not detect form type from XML structure"
        
    except ET.ParseError as e:
        return "unknown", f"XML parse error: {str(e)}"
    except Exception as e:
        return "unknown", f"Error detecting form type: {str(e)}"


def parse_irs_xml(
    xml_bytes: bytes,
    filename: str = "",
    tax_year_override: str = ""
) -> Dict[str, Any]:
    """
    Parse an IRS XML file, automatically detecting form type.
    
    Args:
        xml_bytes: Raw XML content
        filename: Source filename (for diagnostics)
        tax_year_override: Override tax year (if needed)
    
    Returns:
        Dict with keys:
            - foundation_meta: Organization info
            - grants_df: DataFrame of grants
            - people_df: DataFrame of board members
            - diagnostics: Parsing diagnostics
    """
    # Detect form type
    form_type, detect_error = detect_form_type(xml_bytes)
    
    if detect_error:
        return {
            'foundation_meta': {},
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'diagnostics': {
                "parser_version": "dispatcher-1.0",
                "source_file": filename,
                "source_type": "xml",
                "form_type_detected": "unknown",
                "errors": [detect_error],
                "warnings": [],
            }
        }
    
    # Route to appropriate parser
    if form_type == "990-PF":
        result = parse_990pf_xml(xml_bytes, filename, tax_year_override)
    elif form_type == "990":
        result = parse_990_xml(xml_bytes, filename, tax_year_override)
    elif form_type == "990-EZ":
        # 990-EZ typically doesn't have detailed grant schedules
        return {
            'foundation_meta': {},
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'diagnostics': {
                "parser_version": "dispatcher-1.0",
                "source_file": filename,
                "source_type": "xml",
                "form_type_detected": "990-EZ",
                "warnings": ["Form 990-EZ does not typically include detailed grant schedules"],
                "errors": [],
            }
        }
    else:
        return {
            'foundation_meta': {},
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'diagnostics': {
                "parser_version": "dispatcher-1.0",
                "source_file": filename,
                "source_type": "xml",
                "form_type_detected": form_type,
                "warnings": [f"Unsupported form type: {form_type}"],
                "errors": [],
            }
        }
    
    # Add dispatcher info to diagnostics
    result['diagnostics']['dispatcher_version'] = "1.0"
    result['diagnostics']['routed_to'] = form_type
    
    return result


def is_xml_file(filename: str) -> bool:
    """Check if filename indicates an XML file."""
    return filename.lower().endswith('.xml')


def parse_irs_file(
    file_bytes: bytes,
    filename: str,
    tax_year_override: str = ""
) -> Dict[str, Any]:
    """
    Parse an IRS file (XML or PDF), automatically detecting format.
    
    This is the main entry point for the unified parser.
    
    Args:
        file_bytes: Raw file content
        filename: Source filename
        tax_year_override: Override tax year (if needed)
    
    Returns:
        Standardized result dict
    """
    if is_xml_file(filename):
        return parse_irs_xml(file_bytes, filename, tax_year_override)
    else:
        # Assume PDF - import and call PDF parser
        # This import is here to avoid circular dependencies
        from c4c_utils.irs990_parser import parse_990_pdf
        return parse_990_pdf(file_bytes, filename)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python irs_xml_dispatcher.py <file.xml>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    
    # Detect form type
    form_type, error = detect_form_type(file_bytes)
    print(f"\n📋 Detected form type: {form_type}")
    if error:
        print(f"   Error: {error}")
    
    # Parse
    result = parse_irs_xml(file_bytes, filepath)
    
    meta = result["foundation_meta"]
    diag = result["diagnostics"]
    grants_df = result["grants_df"]
    people_df = result["people_df"]
    
    print(f"\n{'='*60}")
    print(f"🏛️  {meta.get('foundation_name', 'Unknown')}")
    print(f"{'='*60}")
    print(f"EIN: {meta.get('foundation_ein', 'N/A')}")
    print(f"Tax Year: {meta.get('tax_year', 'N/A')}")
    print(f"Form Type: {diag.get('form_type_detected', 'N/A')}")
    print(f"Parser: {diag.get('parser_version', 'N/A')}")
    
    if not grants_df.empty:
        total = grants_df['grant_amount'].sum() if 'grant_amount' in grants_df.columns else 0
        print(f"\n📊 Grants: {len(grants_df)} records, ${total:,} total")
    else:
        print("\n📊 Grants: None extracted")
    
    if not people_df.empty:
        print(f"👥 Board/Officers: {len(people_df)}")
    
    if diag.get("warnings"):
        print(f"\n⚠️  Warnings: {diag['warnings']}")
    if diag.get("errors"):
        print(f"\n❌ Errors: {diag['errors']}")
