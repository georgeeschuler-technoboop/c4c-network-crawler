"""
990-PF Parser v2.1 - Complete Module for OrgGraph (US)
======================================================
Fixes:
1. Better handling of inline amounts (Joyce Foundation style)
2. Better org name detection (doesn't require Inc/Foundation/etc.)
3. Handles multi-line purposes
4. Robust ProPublica artifact removal (headers, footers, URLs)
5. Filters address fragments captured as org names

Exports:
- parse_990_pdf(file_bytes, filename, tax_year_override) -> dict
- IRS990PFParser class for direct use
"""

import re
import io
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import pypdf


@dataclass
class Grant:
    """Represents a single grant."""
    recipient_name: str
    recipient_address: str = ""
    recipient_city: str = ""
    recipient_state: str = ""
    foundation_status: str = ""  # PC, NC, PF, GOV, SO, POF
    purpose: str = ""
    amount: int = 0


@dataclass  
class Person:
    """Represents a board member/officer."""
    name: str
    title: str = ""
    compensation: float = 0


class IRS990PFParser:
    """Parse 990-PF forms to extract grants and board members."""
    
    # ProPublica artifacts to remove - order matters (most specific first)
    PROPUBLICA_PATTERNS = [
        # Concatenated Page + URL (no space between) - must come first
        r'Page\s+\d+\s+of\s+\d+https?://[^\s]+',
        # Standard header with date
        r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}\s*:\s*\d{2}\s*[AP]M\s*.+?-\s*Full Filing\s*-\s*Nonprofit Explorer\s*-\s*ProPublica',
        # Header without date but with foundation name
        r'[A-Z][A-Za-z\s]+(?:Foundation|Fund|Trust|Inc\.?)\s*-\s*Full Filing\s*-\s*Nonprofit Explorer\s*-\s*ProPublica',
        # Any line containing "Full Filing - Nonprofit Explorer - ProPublica"
        r'.*Full Filing\s*-\s*Nonprofit Explorer\s*-\s*ProPublica.*',
        # URL pattern
        r'https?://projects\.propublica\.org/[^\s\n]+',
        # Page marker
        r'Page\s+\d+\s+of\s+\d+',
    ]
    
    # Foundation status codes (sorted by length for proper matching)
    STATUS_CODES = ['SO II', 'SO I', 'GOV', 'POF', 'PC', 'NC', 'PF', 'SO']
    
    # Lines to skip (form headers, column headers, etc.)
    SKIP_PATTERNS = [
        r'^Form\s+990',
        r'^Page\s+\d',
        r'^Part\s+[XIV]+',
        r'^a\s+Paid during the year',
        r'^b\s+Approved for future',
        r'^Recipient',
        r'^Name and address',
        r'^Foundation\s*$',
        r'^status of',
        r'^recipient\s*$',
        r'^Purpose of grant',
        r'^contribution',
        r'^Amount\s*$',
        r'^Total\s*\.+',
    ]
    
    def __init__(self):
        self.compiled_propublica = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.PROPUBLICA_PATTERNS]
        self.compiled_skip = [re.compile(p, re.IGNORECASE) for p in self.SKIP_PATTERNS]
    
    def parse_bytes(self, pdf_bytes: bytes) -> tuple[list[Grant], list[Person], dict]:
        """Parse PDF from bytes. Returns (grants, people, metadata)."""
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self._parse_text(all_text)
    
    def parse_file(self, pdf_path: str) -> tuple[list[Grant], list[Person], dict]:
        """Parse PDF from file path. Returns (grants, people, metadata)."""
        reader = pypdf.PdfReader(pdf_path)
        all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self._parse_text(all_text)
    
    def _parse_text(self, all_text: str) -> tuple[list[Grant], list[Person], dict]:
        """Parse extracted text. Returns (grants, people, metadata)."""
        # Extract metadata first (before cleaning)
        metadata = self._extract_metadata(all_text)
        
        # Clean text
        cleaned = self._clean_text(all_text)
        
        # Find grants section
        section = self._find_grants_section(cleaned)
        grants = []
        if section:
            raw_grants = self._extract_grants(section)
            grants = [g for g in raw_grants if self._is_valid_grant(g)]
        
        # Extract people (board members)
        people = self._extract_people(cleaned)
        
        return grants, people, metadata
    
    def _extract_metadata(self, text: str) -> dict:
        """Extract foundation metadata from the form."""
        metadata = {
            'foundation_name': '',
            'ein': '',
            'tax_year': '',
            'is_990pf': True,
            'form_type': '990-PF',
        }
        
        # Check form type
        if re.search(r'Form\s+990\s*\n', text) and not re.search(r'Form\s+990-PF', text):
            metadata['is_990pf'] = False
            metadata['form_type'] = '990'
        
        # Find EIN
        ein_match = re.search(r'Employer identification number\s*(\d{2}[-\s]?\d{7})', text, re.IGNORECASE)
        if ein_match:
            metadata['ein'] = ein_match.group(1).replace('-', '').replace(' ', '')
        
        # Find organization name (usually near top)
        name_patterns = [
            r'Name of foundation\s*\n\s*([A-Z][A-Za-z\s&,\.]+(?:Foundation|Fund|Trust|Inc\.?|LLC))',
            r'^([A-Z][A-Z\s&,\.]+(?:FOUNDATION|FUND|TRUST|INC\.?))',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                metadata['foundation_name'] = match.group(1).strip()
                break
        
        # Find tax year
        year_match = re.search(r'(?:Tax year|Calendar year|For calendar year)\s*(\d{4})', text, re.IGNORECASE)
        if year_match:
            metadata['tax_year'] = year_match.group(1)
        else:
            # Try to find year in form header
            year_match = re.search(r'Form 990-PF\s*\((\d{4})\)', text)
            if year_match:
                metadata['tax_year'] = year_match.group(1)
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Remove ProPublica artifacts and normalize whitespace."""
        cleaned = text
        
        # Apply ProPublica patterns
        for pattern in self.compiled_propublica:
            cleaned = pattern.sub('', cleaned)
        
        # Additional line-by-line cleaning for stubborn artifacts
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines that are clearly ProPublica artifacts
            if 'Nonprofit Explorer' in line:
                continue
            if 'ProPublica' in line:
                continue
            if re.match(r'^\d{1,2}/\d{1,2}/\d{2}', line) and 'PM' in line or 'AM' in line:
                continue
            filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        return cleaned.strip()
    
    def _find_grants_section(self, text: str) -> Optional[str]:
        """Find the Part XV 3a grants section."""
        # Look for "a Paid during the year"
        paid_match = re.search(r'a\s+Paid during the year', text, re.IGNORECASE)
        if not paid_match:
            return None
        
        start = paid_match.end()
        
        # Find end at "b Approved for future payment" or "Total ... 3a"
        end_patterns = [
            r'\bb\s+Approved for future payment',
            r'Total\s*\.+\s*3a',
        ]
        
        end = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text[start:], re.IGNORECASE)
            if match and start + match.start() < end:
                end = start + match.start()
        
        return text[start:end]
    
    def _extract_grants(self, section: str) -> list[Grant]:
        """Extract grants from section text."""
        grants = []
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip header/boilerplate lines
            if self._is_skip_line(line):
                i += 1
                continue
            
            # Check for status line (NONE PC/NC/etc.)
            status_match = self._parse_status_line(line)
            
            if status_match:
                status, purpose, inline_amount = status_match
                
                # Walk backwards to find org name and address
                name, address, city, state = self._extract_name_address_backwards(lines, i)
                
                if inline_amount is not None:
                    # Inline amount (Joyce style)
                    if name:
                        grants.append(Grant(
                            recipient_name=name,
                            recipient_address=address,
                            recipient_city=city,
                            recipient_state=state,
                            foundation_status=status,
                            purpose=purpose,
                            amount=inline_amount
                        ))
                    i += 1
                else:
                    # Look forward for purpose continuation and amount (Mott style)
                    purpose_lines = [purpose] if purpose else []
                    j = i + 1
                    amount = None
                    
                    while j < len(lines):
                        if self._is_standalone_amount(lines[j]):
                            amount = self._parse_amount(lines[j])
                            j += 1
                            break
                        if self._parse_status_line(lines[j]):
                            break
                        if lines[j].upper().strip() == 'NONE':
                            break
                        purpose_lines.append(lines[j])
                        j += 1
                    
                    if name and amount:
                        grants.append(Grant(
                            recipient_name=name,
                            recipient_address=address,
                            recipient_city=city,
                            recipient_state=state,
                            foundation_status=status,
                            purpose=" ".join(purpose_lines),
                            amount=amount
                        ))
                    i = j
            
            # Check for split NONE pattern (NONE on its own line)
            elif line.upper().strip() == 'NONE':
                if i + 1 < len(lines):
                    next_status = self._parse_bare_status_line(lines[i + 1])
                    if next_status:
                        status, purpose_start = next_status
                        name, address, city, state = self._extract_name_address_backwards(lines, i)
                        
                        purpose_lines = [purpose_start] if purpose_start else []
                        j = i + 2
                        amount = None
                        
                        while j < len(lines):
                            if self._is_standalone_amount(lines[j]):
                                amount = self._parse_amount(lines[j])
                                j += 1
                                break
                            if self._parse_status_line(lines[j]) or lines[j].upper().strip() == 'NONE':
                                break
                            purpose_lines.append(lines[j])
                            j += 1
                        
                        if name and amount:
                            grants.append(Grant(
                                recipient_name=name,
                                recipient_address=address,
                                recipient_city=city,
                                recipient_state=state,
                                foundation_status=status,
                                purpose=" ".join(purpose_lines),
                                amount=amount
                            ))
                        i = j
                        continue
                i += 1
            else:
                i += 1
        
        return grants
    
    def _parse_status_line(self, line: str) -> Optional[tuple[str, str, Optional[int]]]:
        """
        Parse a status line like 'NONE PC Education 160,000'.
        Returns (status, purpose, inline_amount) or None.
        """
        for status in self.STATUS_CODES:
            match = re.match(rf'^NONE\s+({status})\s*(.*)', line, re.IGNORECASE)
            if match:
                status_code = match.group(1).upper()
                rest = match.group(2).strip()
                
                # Check for inline amount at end
                amount_match = re.search(r'([\d,]+)\s*$', rest)
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '')
                    if amount_str.isdigit():
                        amount = int(amount_str)
                        if 50 <= amount <= 100_000_000:
                            purpose = rest[:amount_match.start()].strip()
                            return (status_code, purpose, amount)
                
                return (status_code, rest, None)
        return None
    
    def _parse_bare_status_line(self, line: str) -> Optional[tuple[str, str]]:
        """Parse a line that starts with just a status code (after NONE on prev line)."""
        for status in self.STATUS_CODES:
            match = re.match(rf'^({status})\s*(.*)', line, re.IGNORECASE)
            if match:
                return (match.group(1).upper(), match.group(2).strip())
        return None
    
    def _extract_name_address_backwards(self, lines: list[str], status_idx: int) -> tuple[str, str, str, str]:
        """
        Walk backwards from status line to extract org name and address.
        Returns (name, address, city, state).
        """
        name_lines = []
        address_lines = []
        city = ""
        state = ""
        
        j = status_idx - 1
        phase = "address"  # Start collecting address, then switch to name
        
        while j >= 0:
            line = lines[j]
            
            # Stop if we hit another status line
            if self._parse_status_line(line) or line.upper().strip() == 'NONE':
                break
            
            # Stop if we hit an amount (previous grant's amount)
            if self._is_standalone_amount(line):
                break
            
            # Skip header lines
            if self._is_skip_line(line):
                j -= 1
                continue
            
            if phase == "address":
                if self._is_address_line(line):
                    address_lines.insert(0, line)
                    # Extract city/state from "City, ST ZIP" pattern
                    city_state_match = re.match(r'^([A-Za-z\s]+),\s*([A-Z]{2})\s*\d*', line)
                    if city_state_match:
                        city = city_state_match.group(1).strip()
                        state = city_state_match.group(2)
                else:
                    # This is probably the org name
                    phase = "name"
                    name_lines.insert(0, line)
            else:
                # Collecting name lines
                if self._is_address_line(line):
                    # We've gone too far back
                    break
                name_lines.insert(0, line)
                # Most org names are 1-2 lines
                if len(name_lines) >= 2:
                    break
            
            j -= 1
        
        name = " ".join(name_lines).strip()
        address = " | ".join(address_lines).strip()
        
        return name, address, city, state
    
    def _is_address_line(self, line: str) -> bool:
        """Check if a line looks like an address component."""
        # City, State ZIP pattern
        if re.match(r'^[A-Za-z\s]+,\s*[A-Z]{2}\s*\d*', line):
            return True
        
        # International postal codes
        if re.match(r'^[A-Za-z\s]+\s+\d{4,}$', line):
            return True
        if re.match(r'^[A-Za-z\s]+\s+[A-Z]{2}\d+[A-Z]*\d*$', line):
            return True
        
        # 2-letter country code only
        if re.match(r'^[A-Z]{2}\s*$', line):
            return True
        
        # Suite/Floor/Office
        if re.match(r'^(Suite|Floor|Ste|Fl|Room|Office|Unit|Apt)\s+', line, re.IGNORECASE):
            return True
        
        # PO Box
        if re.match(r'^P\.?O\.?\s*Box', line, re.IGNORECASE):
            return True
        
        # Street address patterns
        street_types = r'(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Circle|Cir|Court|Ct|Place|Pl|Highway|Hwy|Parkway|Pkwy|Floor|Fl)'
        
        # Number + words + street type
        if re.match(rf'^\d+\s+.*{street_types}', line, re.IGNORECASE):
            # But don't match if it looks like an org name with numbers
            org_indicators = r'\b(Inc|LLC|Foundation|Association|Center|Centre|Institute|University|College|School|Church|Ministry|Network|Alliance|Council|Society|Trust|Fund|Organization|Corporation|Corp|Company|Co|Group|Partners|Services)\b'
            if re.search(org_indicators, line, re.IGNORECASE):
                return False
            return True
        
        return False
    
    def _is_standalone_amount(self, line: str) -> bool:
        """Check if line is only a dollar amount."""
        cleaned = line.replace(',', '').strip()
        if cleaned.isdigit():
            val = int(cleaned)
            return 50 <= val <= 100_000_000
        return False
    
    def _parse_amount(self, line: str) -> int:
        """Parse amount from line."""
        return int(line.replace(',', '').strip())
    
    def _is_skip_line(self, line: str) -> bool:
        """Check if line should be skipped."""
        for pattern in self.compiled_skip:
            if pattern.match(line):
                return True
        return False
    
    def _is_valid_grant(self, grant: Grant) -> bool:
        """Validate a parsed grant."""
        name = grant.recipient_name.strip()
        
        # Filter empty/short names
        if len(name) < 3:
            return False
        
        # Filter pure numbers (including amounts like "50,000")
        if name.replace(',', '').replace(' ', '').replace('-', '').replace('/', '').isdigit():
            return False
        
        # Filter things that look like amounts
        if re.match(r'^[\d,]+$', name):
            return False
        
        # Filter 2-letter country codes
        if len(name) == 2 and name.isupper():
            return False
        
        # Filter suite/office numbers
        if re.match(r'^\d+[-/]?\d*$', name):
            return False
        
        # Filter address fragments
        address_fragments = [
            r'^(Fl|Floor|Ste|Suite|Apt|Room|Unit)\s*\d',  # "Fl 3", "Suite 100"
            r'^Box\s+\d',  # "Box 21195"
            r'^\d+\s*(st|nd|rd|th)\s+(Fl|Floor)',  # "11th Floor"
            r'^\d+\s+N\.?\s+Michigan',  # Address fragments like "150 N Michigan Ave"
        ]
        for pattern in address_fragments:
            if re.match(pattern, name, re.IGNORECASE):
                return False
        
        # Filter form artifacts and ProPublica remnants
        artifacts = [
            'Form 990', 'Page ', 'Part ', 'NONE', 'Amount', 'Recipient',
            'ProPublica', 'Nonprofit Explorer', 'Full Filing',
        ]
        for artifact in artifacts:
            if artifact.lower() in name.lower():
                return False
        
        # Filter lines that look like date/time stamps (ProPublica headers)
        if re.match(r'^\d{1,2}/\d{1,2}/\d{2}', name):
            return False
        
        return True
    
    def _extract_people(self, text: str) -> list[Person]:
        """Extract board members/officers from Part VII or Part VIII."""
        people = []
        
        # Look for Part VII "Officers, Directors, Trustees" section
        part_vii_match = re.search(
            r'Part\s+VII[A-Z\s]*(?:Officers|Directors|Trustees|Key Employees)',
            text, re.IGNORECASE
        )
        
        if not part_vii_match:
            return people
        
        # Get section text
        start = part_vii_match.end()
        end_match = re.search(r'Part\s+VIII', text[start:], re.IGNORECASE)
        end = start + end_match.start() if end_match else min(start + 5000, len(text))
        
        section = text[start:end]
        
        # Look for name patterns - typically ALL CAPS names
        # Pattern: NAME ... TITLE ... NUMBER
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for lines with names (typically capitalized) followed by title
            name_match = re.match(r'^([A-Z][A-Za-z\s\.,]+?)(?:\s{2,}|\t)(.+)', line)
            if name_match:
                potential_name = name_match.group(1).strip()
                rest = name_match.group(2).strip()
                
                # Filter out obvious non-names
                if len(potential_name) < 3:
                    continue
                if potential_name.upper() in ['NAME', 'TITLE', 'HOURS', 'COMPENSATION']:
                    continue
                
                # Look for title keywords
                title_keywords = ['director', 'trustee', 'officer', 'president', 'secretary', 
                                 'treasurer', 'chairman', 'ceo', 'cfo', 'vice']
                
                title = ""
                for keyword in title_keywords:
                    if keyword in rest.lower():
                        # Extract title
                        title_match = re.search(rf'({keyword}[A-Za-z\s]*)', rest, re.IGNORECASE)
                        if title_match:
                            title = title_match.group(1).strip()
                            break
                
                if title or 'X' in rest:  # 'X' often marks position checkboxes
                    people.append(Person(
                        name=potential_name,
                        title=title
                    ))
        
        return people


# =============================================================================
# PUBLIC API - Used by OrgGraph app
# =============================================================================

def parse_990_pdf(file_bytes: bytes, filename: str, tax_year_override: str = "") -> dict:
    """
    Parse a 990-PF PDF file and return structured data.
    
    This is the main entry point used by the OrgGraph Streamlit app.
    
    Args:
        file_bytes: Raw PDF bytes
        filename: Original filename (for logging)
        tax_year_override: Optional override for tax year
    
    Returns:
        dict with keys:
            - grants_df: DataFrame of grants
            - people_df: DataFrame of board members
            - foundation_meta: dict of foundation metadata
            - diagnostics: dict of parsing diagnostics
    """
    parser = IRS990PFParser()
    
    try:
        grants, people, metadata = parser.parse_bytes(file_bytes)
    except Exception as e:
        # Return empty results on parse failure
        return {
            'grants_df': pd.DataFrame(),
            'people_df': pd.DataFrame(),
            'foundation_meta': {
                'foundation_name': '',
                'foundation_ein': '',
                'tax_year': tax_year_override or '',
                'source_file': filename,
            },
            'diagnostics': {
                'org_name': '',
                'is_990pf': False,
                'form_type': 'unknown',
                'error': str(e),
            }
        }
    
    # Override tax year if provided
    tax_year = tax_year_override or metadata.get('tax_year', '')
    
    # Build grants DataFrame
    if grants:
        grants_data = []
        for g in grants:
            grants_data.append({
                'foundation_name': metadata.get('foundation_name', ''),
                'foundation_ein': metadata.get('ein', ''),
                'tax_year': tax_year,
                'grantee_name': g.recipient_name,
                'grantee_city': g.recipient_city,
                'grantee_state': g.recipient_state,
                'grant_amount': float(g.amount),
                'grant_purpose_raw': g.purpose,
                'source_file': filename,
            })
        grants_df = pd.DataFrame(grants_data)
    else:
        grants_df = pd.DataFrame(columns=[
            'foundation_name', 'foundation_ein', 'tax_year',
            'grantee_name', 'grantee_city', 'grantee_state',
            'grant_amount', 'grant_purpose_raw', 'source_file'
        ])
    
    # Build people DataFrame
    if people:
        people_data = []
        for p in people:
            people_data.append({
                'foundation_name': metadata.get('foundation_name', ''),
                'foundation_ein': metadata.get('ein', ''),
                'person_name': p.name,
                'title': p.title,
                'source_file': filename,
            })
        people_df = pd.DataFrame(people_data)
    else:
        people_df = pd.DataFrame(columns=[
            'foundation_name', 'foundation_ein', 'person_name', 'title', 'source_file'
        ])
    
    return {
        'grants_df': grants_df,
        'people_df': people_df,
        'foundation_meta': {
            'foundation_name': metadata.get('foundation_name', ''),
            'foundation_ein': metadata.get('ein', ''),
            'tax_year': tax_year,
            'source_file': filename,
        },
        'diagnostics': {
            'org_name': metadata.get('foundation_name', ''),
            'is_990pf': metadata.get('is_990pf', True),
            'form_type': metadata.get('form_type', '990-PF'),
        }
    }


# =============================================================================
# CLI TEST HARNESS
# =============================================================================

def test_parser(pdf_path: str):
    """Run parser and show results (for command-line testing)."""
    print("=" * 80)
    print("990-PF PARSER v2.1 TEST")
    print("=" * 80)
    print(f"\nInput: {pdf_path}\n")
    
    parser = IRS990PFParser()
    grants, people, metadata = parser.parse_file(pdf_path)
    
    print(f"Foundation: {metadata.get('foundation_name', 'Unknown')}")
    print(f"EIN: {metadata.get('ein', 'Unknown')}")
    print(f"Tax Year: {metadata.get('tax_year', 'Unknown')}")
    print(f"\nGrants extracted: {len(grants)}")
    print(f"Total amount: ${sum(g.amount for g in grants):,}")
    print(f"Board members: {len(people)}")
    
    print("\n" + "-" * 80)
    print("First 15 grants:")
    print("-" * 80)
    
    for i, grant in enumerate(grants[:15], 1):
        print(f"\n{i}. {grant.recipient_name}")
        if grant.recipient_city or grant.recipient_state:
            print(f"   Location: {grant.recipient_city}, {grant.recipient_state}")
        print(f"   Status: {grant.foundation_status}")
        purpose = grant.purpose[:60] + "..." if len(grant.purpose) > 60 else grant.purpose
        print(f"   Purpose: {purpose}")
        print(f"   Amount: ${grant.amount:,}")
    
    # Summary by purpose category
    print("\n" + "=" * 80)
    print("SUMMARY BY PURPOSE")
    print("=" * 80)
    
    purpose_totals = {}
    for g in grants:
        purpose = g.purpose.strip()
        if purpose:
            purpose_totals[purpose] = purpose_totals.get(purpose, 0) + g.amount
    
    sorted_purposes = sorted(purpose_totals.items(), key=lambda x: -x[1])[:15]
    for purpose, total in sorted_purposes:
        label = purpose[:50] + "..." if len(purpose) > 50 else purpose
        print(f"  ${total:>12,}  {label}")
    
    return grants, people, metadata


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python irs990_parser.py <pdf_path>")
        print("       python irs990_parser.py /path/to/990-pf.pdf")
    else:
        test_parser(sys.argv[1])
