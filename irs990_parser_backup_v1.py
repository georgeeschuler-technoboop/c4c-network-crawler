"""
IRS 990-PF Parser for C4C Network Intelligence Engine

Extracts:
- Foundation metadata (name, EIN, tax year)
- Grants schedule (Part XIV, Section 3)
- Board/officer information (Part VII)

Designed to work with 990-PF filings, particularly those formatted
like ProPublica Nonprofit Explorer exports.
"""

import re
import pandas as pd
import pdfplumber
from typing import Optional
from io import BytesIO


def parse_990_pdf(file_bytes: bytes, source_file: str, tax_year_override: str = "") -> dict:
    """
    Parse a 990-PF PDF and extract foundation metadata, grants, and board members.
    
    Args:
        file_bytes: Raw bytes of the PDF file
        source_file: Filename for tracking/debugging
        tax_year_override: Optional tax year to use instead of auto-detection
    
    Returns:
        dict with keys:
            - foundation_meta: dict with foundation_name, foundation_ein, tax_year
            - grants_df: DataFrame with grant records
            - people_df: DataFrame with board/officer records
    """
    # Initialize empty results
    foundation_meta = {
        "foundation_name": "",
        "foundation_ein": "",
        "tax_year": None,
    }
    grants_df = pd.DataFrame()
    people_df = pd.DataFrame()
    
    try:
        # Open PDF from bytes
        pdf_stream = BytesIO(file_bytes)
        with pdfplumber.open(pdf_stream) as pdf:
            # Extract all text for searching
            all_text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"
            
            # 1. Extract foundation metadata from first few pages
            foundation_meta = _extract_foundation_meta(all_text, tax_year_override)
            
            # 2. Extract grants from Part XIV Section 3
            grants_df = _extract_grants(pdf, all_text, foundation_meta, source_file)
            
            # 3. Extract board/officers from Part VII
            people_df = _extract_people(pdf, all_text, foundation_meta, source_file)
    
    except Exception as e:
        print(f"Error parsing {source_file}: {e}")
        # Return empty results on error
    
    return {
        "foundation_meta": foundation_meta,
        "grants_df": grants_df,
        "people_df": people_df,
    }


def _extract_foundation_meta(text: str, tax_year_override: str = "") -> dict:
    """Extract foundation name, EIN, and tax year from 990-PF text."""
    meta = {
        "foundation_name": "",
        "foundation_ein": "",
        "tax_year": None,
    }
    
    # Try to extract EIN - pattern: XX-XXXXXXX
    ein_patterns = [
        r'TIN:\s*(\d{2}-\d{7})',
        r'Employer identification number\s*(\d{2}-\d{7})',
        r'EIN[:\s]+(\d{2}-\d{7})',
        r'(\d{2}-\d{7})',  # Fallback - any EIN pattern
    ]
    
    for pattern in ein_patterns:
        match = re.search(pattern, text)
        if match:
            meta["foundation_ein"] = match.group(1)
            break
    
    # Try to extract foundation name
    # Look for patterns like "Name of foundation\nTHE PORTER FAMILY FOUNDATION"
    name_patterns = [
        r'Name of foundation[^\n]*\n([A-Z][A-Z\s&\.\,\-\']+(?:FOUNDATION|FUND|TRUST|CHARITABLE))',
        r'(?:THE\s+)?([A-Z][A-Z\s&\.\,\-\']+(?:FOUNDATION|FUND|TRUST))',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            if len(name) > 5 and len(name) < 100:  # Sanity check
                meta["foundation_name"] = name.upper()
                break
    
    # Try to extract tax year
    if tax_year_override:
        try:
            meta["tax_year"] = int(tax_year_override)
        except ValueError:
            pass
    
    if not meta["tax_year"]:
        # Look for "For calendar year XXXX" or "tax year beginning XX-XX-XXXX"
        year_patterns = [
            r'Form\s*990-PF.*?(\d{4})',
            r'For calendar year\s*(\d{4})',
            r'tax year beginning\s*\d{2}-\d{2}-(\d{4})',
            r'Return of Private Foundation.*?(\d{4})',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 2000 <= year <= 2030:  # Sanity check
                    meta["tax_year"] = year
                    break
    
    return meta


def _extract_grants(pdf, all_text: str, foundation_meta: dict, source_file: str) -> pd.DataFrame:
    """
    Extract grants from Part XIV Section 3 of the 990-PF.
    
    Uses text-based parsing since grants are often in multi-line format.
    """
    grants = []
    
    # Find the grants section - Part XIV, Section 3 "Grants and Contributions Paid"
    # Better approach: find "a Paid during the year" and look for actual grant data after
    
    grants_section = ""
    
    # Method 1: Look for "Paid during the year" followed by actual grant data
    # The grants appear after "a Paid during the year" and end at "Total ... 3a"
    paid_match = re.search(r'a\s+Paid during the year\s*\n', all_text, re.IGNORECASE)
    if paid_match:
        start_pos = paid_match.end()
        
        # Find the end of grants section - "Total ... 3a" with the total amount
        end_match = re.search(r'Total\s*\.+\s*3a\s+[\d,]+', all_text[start_pos:])
        if end_match:
            end_pos = start_pos + end_match.end()
        else:
            # Fallback: look for "b Approved for future payment"
            end_match = re.search(r'\bb\s+Approved for future payment', all_text[start_pos:], re.IGNORECASE)
            if end_match:
                end_pos = start_pos + end_match.start()
            else:
                # Last fallback: take next 5000 chars
                end_pos = start_pos + 5000
        
        grants_section = all_text[start_pos:end_pos]
    
    # Method 2: If method 1 failed, look for known grantee patterns
    if not grants_section or len(grants_section) < 100:
        # Look for common foundation/nonprofit name patterns in the grants area
        org_patterns = [
            r'(?:COUNCIL|FOUNDATION|NETWORK|CENTER|INSTITUTE|UNIVERSITY|WILDLIFE)',
        ]
        for pattern in org_patterns:
            match = re.search(pattern, all_text[30000:])  # Start searching after front matter
            if match:
                # Found an org, now find section boundaries
                search_start = 30000 + match.start() - 200
                total_match = re.search(r'Total\s*\.+\s*3a', all_text[search_start:])
                if total_match:
                    grants_section = all_text[search_start:search_start + total_match.end()]
                    break
    
    if not grants_section:
        return pd.DataFrame(columns=[
            'foundation_name', 'foundation_ein', 'tax_year',
            'grantee_name', 'grantee_city', 'grantee_state',
            'grant_amount', 'grant_purpose_raw', 'source_file'
        ])
    
    # Parse grants from the section
    # Actual format observed in 990-PFs:
    # [status] [501(c)3] [purpose] [AMOUNT]  <- status/purpose line for NEXT grantee
    # ORG NAME
    # ADDRESS
    # CITY, STATE ZIP
    # [purpose line with amount for following grantee]
    #
    # So amounts appear BEFORE the org they belong to, not after
    
    lines = grants_section.strip().split('\n')
    
    # Clean lines - remove page headers/footers
    cleaned_lines = []
    seen_lines = set()  # Track seen lines to remove duplicates from page breaks
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip page navigation
        if 'propublica.org' in line.lower():
            continue
        if 'Page' in line and 'of 17' in line:
            continue
        if 'Porter Family Foundation - Full Filing' in line:
            continue
        # Stop at Total line
        if re.match(r'^Total\s*\.', line):
            break
        if 'Approved for future payment' in line:
            break
        if 'Form 990-PF' in line:
            continue
        
        # Skip duplicate lines (often from page breaks)
        # Only skip if it's a short line that we've seen before
        line_key = line.upper().strip()
        if len(line_key) < 20 and line_key in seen_lines:
            continue
        seen_lines.add(line_key)
        
        cleaned_lines.append(line)
    
    # Now parse using state machine
    # The first purpose/status line contains info for the FIRST grantee
    # So we need to capture that properly
    
    current_grant = None
    pending_purpose = ""
    pending_amount = None
    
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        
        # Extract amount if present at end of line
        amount_match = re.search(r'([\d,]+)\s*$', line)
        line_amount = None
        if amount_match:
            try:
                amt_str = amount_match.group(1).replace(',', '')
                amt = int(amt_str)
                if amt >= 100:  # Minimum grant amount
                    line_amount = amt
            except ValueError:
                pass
        
        # Check line type
        is_purpose_status_line = (
            line_amount is not None and 
            (
                re.search(r'501\([cC]\)', line) or
                re.search(r'\b(charitable|education|environmental|charity|promote|purposes)\b', line, re.I) or
                re.match(r'^(NONE|None|none)\s', line)
            )
        )
        
        # Multi-line org name detection: look for ALL CAPS continuation that's
        # NOT an address (no numbers at start) and NOT a known non-name pattern
        # Also check for short single-word continuations
        is_name_continuation = False
        if current_grant is not None and not is_purpose_status_line:
            # Short ALL CAPS word could be continuation (like "ACTION")
            is_short_caps = (
                line.isupper() and
                len(line.split()) <= 2 and  # 1-2 words
                not line_amount and
                not re.match(r'^\d+\s', line) and
                not re.match(r'^(SUITE|STE|PO BOX|P\.O\.)', line, re.I) and
                len(line) > 2 and len(line) < 30
            )
            
            # Longer ALL CAPS that looks like name continuation
            is_longer_caps_continuation = (
                line.isupper() and
                not line_amount and
                not re.match(r'^\d+\s', line) and
                not re.match(r'^(SUITE|STE|PO BOX|P\.O\.)', line, re.I) and
                len(line) > 2 and
                # Check next line - if it's an address, this is likely a name continuation
                (i + 1 < len(cleaned_lines) and 
                 (re.match(r'^\d+\s', cleaned_lines[i+1]) or 
                  re.match(r'^(SUITE|STE|PO BOX)', cleaned_lines[i+1], re.I)))
            )
            
            is_name_continuation = is_short_caps or is_longer_caps_continuation
        
        is_org_name = (
            not is_purpose_status_line and
            not is_name_continuation and
            not line_amount and  # Org names don't end with amounts
            len(line) > 3 and
            not re.match(r'^\d+\s+\w', line) and  # Not address starting with number
            not re.match(r'^(SUITE|STE|Suite|FLOOR|ROOM)\b', line, re.I) and
            not re.match(r'^(PO BOX|P\.O\. BOX)', line, re.I) and
            not re.match(r'^[A-Za-z]+,\s*[A-Z]{2}\s*\d{5}', line) and  # Not city/state
            # Not an address line (contains street indicators)
            not re.search(r'\b(AVE|AVENUE|STREET|ST|ROAD|RD|DRIVE|DR|BLVD|LANE|LN|WAY|PLACE|PL)\s', line, re.I)
        )
        
        is_address = (
            re.match(r'^\d+\s+\w', line) or
            re.match(r'^(PO BOX|P\.O\. BOX)', line, re.I) or
            re.match(r'^(SUITE|STE|Suite)\s', line, re.I) or
            # Catch addresses like "ONE SOUTH HARBOR AVE" 
            re.search(r'\b(AVE|AVENUE|STREET|ST|ROAD|RD|DRIVE|DR|BLVD|LANE|LN|WAY|PLACE|PL)\b', line, re.I)
        )
        
        city_state_match = re.match(r'^([A-Za-z\s\.]+)[,\s]+([A-Z]{2})\s*(\d{5})?', line)
        is_city_state = city_state_match is not None and not is_purpose_status_line
        
        # Process line based on type
        if is_purpose_status_line:
            # Save current grant if we have one with data
            if current_grant and current_grant.get('name'):
                grants.append(current_grant)
            
            # Extract purpose (remove status indicators and amount)
            purpose = line
            purpose = re.sub(r'(?:^|\s)(NONE|None|none|N/A)\s*', ' ', purpose)
            purpose = re.sub(r'501\([cC]\)\d?\s*', '', purpose)
            purpose = re.sub(r'[\d,]+\s*$', '', purpose).strip()
            
            pending_purpose = purpose
            pending_amount = line_amount
            current_grant = None
            
        elif is_name_continuation and current_grant:
            # Append to current grant name
            current_grant['name'] += ' ' + line
            
        elif is_org_name:
            # Start new grant with pending purpose/amount
            current_grant = {
                'name': line,
                'address_lines': [],
                'city': '',
                'state': '',
                'purpose': pending_purpose,
                'amount': pending_amount,
            }
            pending_purpose = ""
            pending_amount = None
            
        elif current_grant is not None:
            if is_address:
                current_grant['address_lines'].append(line)
            elif is_city_state:
                current_grant['city'] = city_state_match.group(1).strip()
                current_grant['state'] = city_state_match.group(2)
        
        i += 1
    
    # Don't forget the last grant
    if current_grant and current_grant.get('name'):
        grants.append(current_grant)
    
    # Convert to final format
    final_grants = []
    for g in grants:
        final_grants.append(_finalize_grant(g, foundation_meta, source_file))
    
    # Create DataFrame
    if final_grants:
        df = pd.DataFrame(final_grants)
    else:
        df = pd.DataFrame(columns=[
            'foundation_name', 'foundation_ein', 'tax_year',
            'grantee_name', 'grantee_city', 'grantee_state',
            'grant_amount', 'grant_purpose_raw', 'source_file'
        ])
    
    return df


def _finalize_grant(grant_data: dict, foundation_meta: dict, source_file: str) -> dict:
    """Convert raw grant data to final format."""
    # Clean up name
    name = grant_data['name'].strip()
    name = re.sub(r'\s+', ' ', name)
    
    # Clean up purpose
    purpose = grant_data.get('purpose', '')
    # Remove status indicators from purpose
    purpose = re.sub(r'(?:NONE|none|N/A|501\([Cc]\)\d?)', '', purpose).strip()
    
    return {
        'foundation_name': foundation_meta.get('foundation_name', ''),
        'foundation_ein': foundation_meta.get('foundation_ein', ''),
        'tax_year': foundation_meta.get('tax_year'),
        'grantee_name': name,
        'grantee_city': grant_data.get('city', ''),
        'grantee_state': grant_data.get('state', ''),
        'grant_amount': grant_data.get('amount'),
        'grant_purpose_raw': purpose,
        'source_file': source_file,
    }


def _extract_people(pdf, all_text: str, foundation_meta: dict, source_file: str) -> pd.DataFrame:
    """
    Extract board members and officers from Part VII.
    
    Uses table extraction since Part VII is typically well-formatted as a table.
    """
    people = []
    
    # Search each page for Part VII table
    for page_num, page in enumerate(pdf.pages):
        page_text = page.extract_text() or ""
        
        # Look for Part VII on this page - specifically "List all officers, directors"
        if "Part VII" not in page_text:
            continue
        if "List all officers" not in page_text and "officers, directors" not in page_text.lower():
            continue
        
        # Extract tables from this page
        tables = page.extract_tables()
        
        for table in tables:
            if not table:
                continue
            
            # Look for table with officer/director info
            # Should have columns like "Name and address", "Title", "Compensation"
            
            # First check if this looks like the right table
            table_text = str(table)
            if 'Name and address' not in table_text and 'Title' not in table_text:
                continue
            
            for row_idx, row in enumerate(table):
                if not row or len(row) < 2:
                    continue
                
                # Get first two columns (name and title)
                name_cell = str(row[0] or "").strip()
                title_cell = str(row[1] or "").strip()
                
                # Skip header rows
                if any(x in name_cell.lower() for x in ['name and address', '(a)', 'recipient']):
                    continue
                if not name_cell:
                    continue
                
                # Skip rows that are clearly not names
                # Names don't contain digits at start, URLs, or common non-name words
                if re.match(r'^\d', name_cell):
                    continue
                if 'http' in name_cell.lower():
                    continue
                if any(x in name_cell.lower() for x in ['total number', 'page', 'part', 'form 990', 'more than']):
                    continue
                if any(x in name_cell for x in ['$', '%', '.'*3]):
                    continue
                
                # Skip address lines (contain state abbreviations or zip codes in typical positions)
                if re.search(r',\s*[A-Z]{2}\s*\d{5}', name_cell):
                    continue
                if re.match(r'.*(STREET|ROAD|AVE|DRIVE|SUITE|STE)\b', name_cell.upper()):
                    continue
                
                # Check if this looks like a person's name (letters, spaces, maybe periods for initials)
                # Should be mostly letters and spaces
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in name_cell) / max(len(name_cell), 1)
                if alpha_ratio < 0.8:
                    continue
                
                # Name should be reasonable length
                if len(name_cell) < 3 or len(name_cell) > 50:
                    continue
                
                # Extract role from title cell
                role = ""
                if title_cell:
                    # Parse title - often contains "DIRECTOR\n0" (title + hours)
                    title_lines = title_cell.split('\n')
                    for tl in title_lines:
                        tl = tl.strip().upper()
                        if tl in ['DIRECTOR', 'TRUSTEE', 'OFFICER', 'PRESIDENT', 'SECRETARY', 
                                  'TREASURER', 'VP', 'VICE PRESIDENT', 'CHAIRMAN', 'CHAIR', 'CEO', 'CFO']:
                            role = tl
                            break
                        elif 'DIRECTOR' in tl:
                            role = 'DIRECTOR'
                            break
                        elif 'TRUSTEE' in tl:
                            role = 'TRUSTEE'
                            break
                
                # Only add if we haven't seen this person
                if name_cell not in [p['person_name'] for p in people]:
                    people.append({
                        'person_name': name_cell,
                        'person_city': '',
                        'person_state': '',
                        'role': role,
                        'org_name': foundation_meta.get('foundation_name', ''),
                        'org_ein': foundation_meta.get('foundation_ein', ''),
                        'tax_year': foundation_meta.get('tax_year'),
                        'source_file': source_file,
                    })
    
    # Create DataFrame
    if people:
        df = pd.DataFrame(people)
    else:
        df = pd.DataFrame(columns=[
            'person_name', 'person_city', 'person_state', 'role',
            'org_name', 'org_ein', 'tax_year', 'source_file'
        ])
    
    return df
