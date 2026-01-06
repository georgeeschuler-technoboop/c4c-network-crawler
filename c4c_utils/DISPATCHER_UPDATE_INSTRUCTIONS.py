"""
IRS Return Dispatcher Update for 990-EZ Support
================================================

This file shows the changes needed to add 990-EZ support to irs_return_dispatcher.py

OVERVIEW:
---------
1. Import the new 990-EZ parser
2. Add 990-EZ detection in the form type detection logic
3. Route 990-EZ XMLs to the new parser

CHANGES TO MAKE:
----------------

=== 1. ADD IMPORT ===

Near the top of irs_return_dispatcher.py, add:

    from c4c_utils.irs990ez_xml_parser import parse_990ez_xml, detect_990ez

=== 2. UPDATE detect_xml_form_type() FUNCTION ===

Add 990-EZ detection BEFORE the 990-PF and 990 checks:

    def detect_xml_form_type(file_bytes: bytes) -> str:
        '''
        Detect whether XML is 990-PF, 990, or 990-EZ.
        Returns: '990-PF', '990', '990-EZ', or 'unknown'
        '''
        # Handle BOM
        if file_bytes.startswith(b'\\xef\\xbb\\xbf'):
            file_bytes = file_bytes[3:]
        
        try:
            root = ET.fromstring(file_bytes)
        except ET.ParseError:
            return 'unknown'
        
        ns = {'irs': 'http://www.irs.gov/efile'}
        
        # Check ReturnTypeCd first (most reliable)
        return_type = root.find('.//irs:ReturnTypeCd', ns)
        if return_type is not None:
            if return_type.text == '990EZ':
                return '990-EZ'
            elif return_type.text == '990PF':
                return '990-PF'
            elif return_type.text == '990':
                return '990'
        
        # Fallback: check for form-specific elements
        if root.find('.//irs:IRS990EZ', ns) is not None:
            return '990-EZ'
        if root.find('.//irs:IRS990PF', ns) is not None:
            return '990-PF'
        if root.find('.//irs:IRS990', ns) is not None:
            return '990'
        
        return 'unknown'

=== 3. UPDATE parse_irs_return() ROUTING ===

In the main parse_irs_return() function, add the 990-EZ route:

    def parse_irs_return(file_bytes: bytes, filename: str, 
                         tax_year_override: str = "", 
                         region_spec: dict = None) -> dict:
        '''
        Unified dispatcher for IRS return parsing.
        '''
        ext = filename.lower().split('.')[-1]
        
        if ext == 'xml':
            form_type = detect_xml_form_type(file_bytes)
            
            if form_type == '990-EZ':
                # Route to 990-EZ parser
                return parse_990ez_xml(file_bytes, filename, region_spec)
            
            elif form_type == '990-PF':
                # Route to 990-PF parser
                return parse_990pf_xml(file_bytes, filename, region_spec)
            
            elif form_type == '990':
                # Route to 990 parser (Schedule I grants)
                return parse_990_xml(file_bytes, filename, region_spec)
            
            else:
                # Unknown form type
                return {
                    'grants_df': pd.DataFrame(),
                    'people_df': pd.DataFrame(),
                    'foundation_meta': {
                        'source_file': filename,
                        'form_type': 'unknown'
                    },
                    'diagnostics': {
                        'form_type_detected': 'unknown',
                        'errors': [f'Could not detect form type from XML. Expected 990-PF, 990, or 990-EZ.'],
                        'warnings': [],
                    }
                }
        
        elif ext == 'pdf':
            # PDF parsing (existing logic)
            ...
        
        else:
            # Unsupported format
            ...

=== 4. FILE PLACEMENT ===

Copy irs990ez_xml_parser.py to:
    c4c_utils/irs990ez_xml_parser.py

=== 5. TESTING ===

After making these changes, test with:

    # In Python
    from c4c_utils.irs_return_dispatcher import parse_irs_return
    
    with open('test_990ez.xml', 'rb') as f:
        result = parse_irs_return(f.read(), 'test_990ez.xml')
    
    print(f"Form type: {result['foundation_meta'].get('form_type')}")
    print(f"Board members: {len(result['people_df'])}")
    print(f"Grants: {len(result['grants_df'])}")

=== NOTES ===

- 990-EZ is the "Short Form" for small organizations (<$200K revenue)
- It does NOT have Schedule I (detailed grants)
- Grant recipients are sometimes listed in Schedule O (supplemental info)
- The parser extracts Schedule O grant details when available
- Officers/Directors/Trustees are in Part IV (OfficerDirectorTrusteeEmplGrp)
- This enables board interlock detection even for small charities
"""
