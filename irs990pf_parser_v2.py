"""
990-PF Parser v2 - Improved Grant Extraction
=============================================
Fixes:
1. Better handling of inline amounts (Joyce Foundation style)
2. Better org name detection (doesn't require Inc/Foundation/etc.)
3. Handles multi-line purposes

Usage: python irs990pf_parser_v2.py <pdf_path>
"""

import re
import sys
from dataclasses import dataclass
from typing import Optional
import pypdf


@dataclass
class Grant:
    """Represents a single grant."""
    recipient_name: str
    recipient_address: str = ""
    foundation_status: str = ""  # PC, NC, PF, GOV, SO, POF
    purpose: str = ""
    amount: int = 0


class IRS990PFParser:
    """Parse 990-PF forms to extract grants."""
    
    # ProPublica artifacts to remove
    PROPUBLICA_PATTERNS = [
        r'\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}\s*:\d{2}\s*[AP]M.+?-\s*Full Filing\s*-\s*Nonprofit Explorer\s*-\s*ProPublica',
        r'https?://projects\.propublica\.org/[^\s]+',
        r'Page\s+\d+\s+of\s+\d+',
        # Handle concatenated Page + URL (no space between)
        r'Page\s+\d+\s+of\s+\d+https?://[^\s]+',
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
        self.compiled_propublica = [re.compile(p, re.IGNORECASE) for p in self.PROPUBLICA_PATTERNS]
        self.compiled_skip = [re.compile(p, re.IGNORECASE) for p in self.SKIP_PATTERNS]
    
    def parse(self, pdf_path: str) -> list[Grant]:
        """Parse a 990-PF PDF and return list of grants."""
        # Read PDF
        reader = pypdf.PdfReader(pdf_path)
        all_text = "\n".join(page.extract_text() for page in reader.pages)
        
        # Clean text
        cleaned = self._clean_text(all_text)
        
        # Find grants section
        section = self._find_grants_section(cleaned)
        if not section:
            return []
        
        # Extract grants
        grants = self._extract_grants(section)
        
        # Validate grants
        valid_grants = [g for g in grants if self._is_valid_grant(g)]
        
        return valid_grants
    
    def _clean_text(self, text: str) -> str:
        """Remove ProPublica artifacts and normalize whitespace."""
        cleaned = text
        for pattern in self.compiled_propublica:
            cleaned = pattern.sub('', cleaned)
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
                name, address = self._extract_name_address_backwards(lines, i)
                
                if inline_amount is not None:
                    # Inline amount (Joyce style)
                    if name:
                        grants.append(Grant(
                            recipient_name=name,
                            recipient_address=address,
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
                        name, address = self._extract_name_address_backwards(lines, i)
                        
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
    
    def _extract_name_address_backwards(self, lines: list[str], status_idx: int) -> tuple[str, str]:
        """
        Walk backwards from status line to extract org name and address.
        Returns (name, address).
        """
        name_lines = []
        address_lines = []
        
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
        
        return name, address
    
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
        ]
        for pattern in address_fragments:
            if re.match(pattern, name, re.IGNORECASE):
                return False
        
        # Filter form artifacts
        artifacts = ['Form 990', 'Page ', 'Part ', 'NONE', 'Amount', 'Recipient']
        for artifact in artifacts:
            if artifact.lower() in name.lower():
                return False
        
        return True


def test_parser(pdf_path: str):
    """Run parser and show results."""
    print("=" * 80)
    print("990-PF PARSER v2 TEST")
    print("=" * 80)
    print(f"\nInput: {pdf_path}\n")
    
    parser = IRS990PFParser()
    grants = parser.parse(pdf_path)
    
    print(f"Grants extracted: {len(grants)}")
    print(f"Total amount: ${sum(g.amount for g in grants):,}")
    
    print("\n" + "-" * 80)
    print("First 15 grants:")
    print("-" * 80)
    
    for i, grant in enumerate(grants[:15], 1):
        print(f"\n{i}. {grant.recipient_name}")
        if grant.recipient_address:
            addr = grant.recipient_address[:60] + "..." if len(grant.recipient_address) > 60 else grant.recipient_address
            print(f"   Address: {addr}")
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
        # Extract first few words as category
        purpose = g.purpose.strip()
        if purpose:
            purpose_totals[purpose] = purpose_totals.get(purpose, 0) + g.amount
    
    # Sort by total amount
    sorted_purposes = sorted(purpose_totals.items(), key=lambda x: -x[1])[:15]
    for purpose, total in sorted_purposes:
        label = purpose[:50] + "..." if len(purpose) > 50 else purpose
        print(f"  ${total:>12,}  {label}")
    
    return grants


if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path = "/mnt/user-data/uploads/TEST_Joyce_Foundation_-_Full_Filing_-_Nonprofit_Explorer_-_ProPublica_990-PF.pdf"
    else:
        pdf_path = sys.argv[1]
    
    test_parser(pdf_path)
