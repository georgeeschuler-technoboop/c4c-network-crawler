"""
990-PF Parser v2.7 - Complete Module for OrgGraph
==================================================
Changes in v2.7:
- Fixed BoardExtractor to find Part VII continuation pages ("Part VII, Line 1")
- Added pattern for hours embedded in title (e.g., "CHAIR, 18.0")
- Added Part XIV continuation page grant extraction ("Part XIV, Line 3a/3b")
- Improved handling of multi-page supplemental grants
- Better detection of "(SEE STATEMENT)" placeholders

Changes in v2.6:
- REMOVED Great Lakes hardcoding (GL_KEYWORDS, gl_relevant)
- Region tagging is now handled separately via region_tagger.py
- Added grantee_address_raw for address fallback parsing
- Renamed output columns for consistency with spec:
  org_name → grantee_name, city_state_zip → split to grantee_state

Changes in v2.5:
- Added format detection diagnostics (Erb-style vs Joyce-style extraction order)
- Added parsing confidence score based on total matching
- Added sample grant logging for verification
- Better error reporting when totals don't match

Exports:
- parse_990_pdf(file_bytes, filename, tax_year_override) -> dict
- IRS990PFParser class for direct use
"""

PARSER_VERSION = "2.7"

import re
import io
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import pandas as pd
import pdfplumber


@dataclass
class Grant:
    """Represents a single grant."""
    recipient_name: str
    recipient_address: str = ""
    recipient_city_state_zip: str = ""
    relationship: str = ""
    foundation_status: str = ""
    purpose: str = ""
    amount: int = 0
    grant_bucket: str = "3a"  # "3a" = paid, "3b" = approved future
    raw_text: str = ""


@dataclass
class BoardMember:
    """Represents a board member/officer."""
    name: str
    title: str = ""
    hours_per_week: float = 0.0
    compensation: int = 0
    benefits: int = 0
    expense_account: int = 0


class TextCleaner:
    """Removes ProPublica PDF artifacts from extracted text."""
    
    PROPUBLICA_PATTERNS = [
        r'Fred A And Barbara M Erb Family Foundation - Full Filing.*?Page \d+ of \d+',
        r'Charles Stewart Mott Foundation - Full Filing.*?Page \d+ of \d+',
        r'Joyce Foundation - Full Filing.*?Page \d+ of \d+',
        r'[A-Za-z\s\-&]+ - Full Filing - Nonprofit Explorer - ProPublica.*?\d+:\d+ [AP]M',
        r'https://projects\.propublica\.org/nonprofits/organizations/\d+/\d+/full',
        r'Page \d+ of \d+$',
        r'^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[AP]M$',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) 
                                   for p in self.PROPUBLICA_PATTERNS]
    
    def clean(self, text: str) -> str:
        """Remove ProPublica artifacts from text."""
        cleaned = text
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()


class FoundationMetaExtractor:
    """Extracts foundation metadata from 990-PF."""
    
    def extract(self, text: str, tax_year_override: str = "") -> dict:
        """Extract foundation name, EIN, and tax year."""
        meta = {
            "foundation_name": "",
            "foundation_ein": "",
            "tax_year": tax_year_override if tax_year_override else None,
        }
        
        # Foundation name extraction - multiple patterns
        
        # Pattern 1: Look for "NAME FOUNDATION" followed immediately by EIN (most reliable)
        # This appears in the header: "FREY FOUNDATION 23-7094777"
        name_with_ein = re.search(
            r'\b([A-Z][A-Z\s&\-\.]{2,40}(?:FOUNDATION|TRUST|FUND))\s+(\d{2}-\d{7})',
            text[:5000]
        )
        if name_with_ein:
            name = name_with_ein.group(1).strip()
            name = re.sub(r'\s+', ' ', name)
            # Skip if it looks like a form title
            if 'Return of Private' not in name and 'Form' not in name:
                meta["foundation_name"] = name
                meta["foundation_ein"] = name_with_ein.group(2)
        
        # Pattern 2: Look for "THE ... FOUNDATION" pattern  
        if not meta["foundation_name"]:
            name_match = re.search(
                r'\b(THE\s+[A-Z][A-Z\s&\-\.]{2,40}(?:FOUNDATION|TRUST|FUND))\b',
                text[:5000]
            )
            if name_match:
                name = name_match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if 'Return of Private' not in name:
                    meta["foundation_name"] = name
        
        # Pattern 3: Look at individual lines for foundation name
        if not meta["foundation_name"]:
            lines = text.split('\n')[:60]
            for line in lines:
                line = line.strip()
                # Skip short lines or form labels
                if len(line) < 10:
                    continue
                if line.startswith('Form') or line.startswith('efile') or line.startswith('990'):
                    continue
                # Skip rotated text artifacts (reversed words)
                if re.match(r'^[a-z]+$', line):  # All lowercase = likely artifact
                    continue
                    
                # Look for "NAME FOUNDATION" pattern at start of line  
                match = re.match(r'^([A-Z][A-Z\s&\-\.]{2,40}(?:FOUNDATION|TRUST|FUND))\b', line)
                if match:
                    name = match.group(1).strip()
                    if 'Return of Private' not in name:
                        meta["foundation_name"] = name
                        break
        
        # EIN (if not already captured)
        if not meta["foundation_ein"]:
            ein_match = re.search(r'Employer identification number\s*\n?\s*(\d{2}-\d{7})', text)
            if ein_match:
                meta["foundation_ein"] = ein_match.group(1)
            else:
                ein_match = re.search(r'\b(\d{2}-\d{7})\b', text[:5000])
                if ein_match:
                    meta["foundation_ein"] = ein_match.group(1)
        
        # Tax year
        if not meta["tax_year"]:
            year_match = re.search(r'calendar year (\d{4})', text, re.IGNORECASE)
            if year_match:
                meta["tax_year"] = year_match.group(1)
            else:
                year_match = re.search(r'tax year beginning.*?(\d{4})', text, re.IGNORECASE)
                if year_match:
                    meta["tax_year"] = year_match.group(1)
        
        return meta


class SectionFinder:
    """Finds Part XIV grants section boundaries."""
    
    # v2.4 fix: Pattern for spaced dots (. . . .) or consecutive dots (......)
    SPACED_DOTS = r'(?:\.[\s\.]*){3,}'  # 3+ dots with optional spaces between
    
    def find_grants_section_3a(self, text: str) -> tuple:
        """Find the 3a section (Paid during the year)."""
        # Find start: "a Paid during the year"
        start_match = re.search(r'a\s+Paid\s+during\s+the\s+year', text, re.IGNORECASE)
        if not start_match:
            return None, None
        
        start_pos = start_match.end()
        
        # Find end: "Total ... 3a" with amount
        # v2.4: Handle both ". . . . . 3a" and "......3a"
        end_pattern = rf'{self.SPACED_DOTS}\s*(?:►?\s*)?3a\s+([\d,]+)'
        end_match = re.search(end_pattern, text[start_pos:])
        
        if end_match:
            end_pos = start_pos + end_match.end()
            total_3a = int(end_match.group(1).replace(',', ''))
            return start_pos, end_pos, total_3a
        
        # Fallback: look for "b Approved for future payment"
        fallback = re.search(r'b\s+Approved\s+for\s+future\s+payment', text[start_pos:], re.IGNORECASE)
        if fallback:
            return start_pos, start_pos + fallback.start(), 0
        
        return start_pos, len(text), 0
    
    def find_grants_section_3b(self, text: str) -> tuple:
        """Find the 3b section (Approved for future payment)."""
        # Find start: "b Approved for future payment"
        start_match = re.search(r'b\s+Approved\s+for\s+future\s+payment', text, re.IGNORECASE)
        if not start_match:
            return None, None, 0
        
        start_pos = start_match.end()
        
        # Find end: "Total ... 3b" with amount
        # v2.4: Handle both spaced and consecutive dots
        end_pattern = rf'{self.SPACED_DOTS}\s*(?:►?\s*)?3b\s+([\d,]+)'
        end_match = re.search(end_pattern, text[start_pos:])
        
        if end_match:
            end_pos = start_pos + end_match.end()
            total_3b = int(end_match.group(1).replace(',', ''))
            return start_pos, end_pos, total_3b
        
        # Fallback: look for "Part XV" or "Form 990-PF"
        for pattern in [r'Part\s+XV', r'Form\s+990-PF\s*\(', r'Page\s+\d+']:
            fallback = re.search(pattern, text[start_pos:], re.IGNORECASE)
            if fallback:
                return start_pos, start_pos + fallback.start(), 0
        
        return start_pos, len(text), 0
    
    def find_continuation_sections(self, text: str) -> dict:
        """
        v2.7: Find Part XIV continuation pages.
        
        Returns dict with keys:
        - '3a_sections': list of (start, end) tuples for Part XIV, Line 3a continuations
        - '3b_sections': list of (start, end) tuples for Part XIV, Line 3b continuations
        """
        result = {'3a_sections': [], '3b_sections': []}
        
        # Find all "Part XIV, Line 3a" continuation headers
        # Pattern: "Part XIV, Line 3a" followed by "Grants and Contributions Paid During the Year"
        pattern_3a = re.compile(
            r'Part\s+XIV,?\s+Line\s+3a\s+.*?Grants\s+and\s+Contributions\s+Paid',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in pattern_3a.finditer(text):
            start = match.end()
            # Find end: next Part header or page footer
            end_match = re.search(
                r'Part\s+XIV,?\s+Line\s+3[ab]|Part\s+XV|Form\s+990-PF\s*\(',
                text[start:], re.IGNORECASE
            )
            end = start + end_match.start() if end_match else len(text)
            result['3a_sections'].append((start, end))
        
        # Find all "Part XIV, Line 3b" continuation headers
        pattern_3b = re.compile(
            r'Part\s+XIV,?\s+Line\s+3b\s+.*?(?:Approved\s+[Ff]or\s+[Ff]uture|Future\s+Payment)',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in pattern_3b.finditer(text):
            start = match.end()
            end_match = re.search(
                r'Part\s+XIV,?\s+Line\s+3[ab]|Part\s+XV|Form\s+990-PF\s*\(',
                text[start:], re.IGNORECASE
            )
            end = start + end_match.start() if end_match else len(text)
            result['3b_sections'].append((start, end))
        
        return result


class GrantExtractor:
    """Extracts individual grants from section text."""
    
    def __init__(self):
        # Track extraction format for diagnostics
        self.format_a_count = 0  # Erb-style: status/amount BEFORE org name
        self.format_b_count = 0  # Joyce-style: org name BEFORE status/amount
        self.format_c_count = 0  # v2.7: Continuation page format
        self.sample_grants = []  # Store first few for verification
    
    def get_format_diagnostics(self) -> dict:
        """Return diagnostics about which extraction format was detected."""
        total = self.format_a_count + self.format_b_count + self.format_c_count
        if total == 0:
            dominant = "unknown"
            confidence = 0
        elif self.format_a_count >= self.format_b_count and self.format_a_count >= self.format_c_count:
            dominant = "format_a_erb_style"
            confidence = self.format_a_count / total
        elif self.format_b_count >= self.format_c_count:
            dominant = "format_b_joyce_style"
            confidence = self.format_b_count / total
        else:
            dominant = "format_c_continuation"
            confidence = self.format_c_count / total
        
        return {
            "dominant_format": dominant,
            "format_a_erb_style_count": self.format_a_count,
            "format_b_joyce_style_count": self.format_b_count,
            "format_c_continuation_count": self.format_c_count,
            "format_confidence": round(confidence, 2),
            "sample_grants": self.sample_grants[:5]  # First 5 for verification
        }
    
    def extract_grants(self, section_text: str, grant_bucket: str = "3a") -> List[Grant]:
        """Extract grants from the section text."""
        grants = []
        
        # Check for "(SEE STATEMENT)" placeholder - don't extract as a grant
        if re.search(r'\(\s*SEE\s+STATEMENT\s*\)', section_text, re.IGNORECASE):
            # This is a placeholder, not actual grants
            # Return empty - the continuation pages will have the real grants
            pass
        
        # Split into potential grant blocks
        # Each grant typically has: ORG_NAME, ADDRESS, CITY/STATE/ZIP, PC/NC, PURPOSE, AMOUNT
        
        # Pattern for amount at end of line - 3+ digits, possibly with comma
        amount_pattern = re.compile(r'([\d,]{3,})\s*$')
        
        # Pattern for foundation status codes
        status_pattern = re.compile(r'\b(PC|NC|PF|GOV|SO)\b')
        
        # Pattern for city/state/zip lines (should NOT be treated as amount lines)
        # Matches: "CITY,ST 12345" or "CITY, ST 12345-6789" or "CITY,ST 123456789" (malformed zip)
        city_state_zip_pattern = re.compile(r',\s*[A-Z]{2}\s+\d{5,9}$', re.IGNORECASE)
        
        lines = section_text.split('\n')
        
        current_grant = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Skip header lines and form artifacts
            if self._is_header_line(line):
                i += 1
                continue
            
            # Skip "(SEE STATEMENT)" lines
            if re.search(r'\(\s*SEE\s+STATEMENT\s*\)', line, re.IGNORECASE):
                i += 1
                continue
            
            # Check if line has an amount
            amount_match = amount_pattern.search(line)
            
            if amount_match:
                amount = int(amount_match.group(1).replace(',', ''))
                
                # Two formats:
                # Format A (Erb style): "PC PROGRAM SUPPORT 12,500" then "ORG NAME" on next line
                # Format B (Joyce style): "ORG NAME" then "CITY, STATE ZIP" then "NONE PC General purposes 12,500"
                
                # Check if this line has status code (Format A or B with inline status)
                status_match = status_pattern.search(line)
                
                if status_match:
                    status = status_match.group(1)
                    # Get purpose (text between status and amount)
                    purpose_text = line[status_match.end():amount_match.start()].strip()
                    
                    # Format A: status/purpose/amount first, org name follows
                    # Look at next lines for org name
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        # If next line looks like an org name (starts with capital, not an address)
                        if next_line and not self._is_address_line(next_line) and not status_pattern.match(next_line):
                            org_name = next_line
                            # Get address lines
                            address_lines = []
                            j = i + 2
                            while j < len(lines) and j < i + 5:
                                addr_line = lines[j].strip()
                                if not addr_line or status_pattern.search(addr_line):
                                    break
                                # Check if this looks like an amount line but NOT a city/state/zip
                                if amount_pattern.search(addr_line) and not city_state_zip_pattern.search(addr_line):
                                    break
                                if self._is_header_line(addr_line):
                                    j += 1
                                    continue
                                address_lines.append(addr_line)
                                j += 1
                            
                            grant = Grant(
                                recipient_name=org_name,
                                recipient_address=address_lines[0] if address_lines else "",
                                recipient_city_state_zip=address_lines[1] if len(address_lines) > 1 else "",
                                foundation_status=status,
                                purpose=purpose_text,
                                amount=amount,
                                grant_bucket=grant_bucket
                            )
                            grants.append(grant)
                            self.format_a_count += 1  # Track Format A
                            
                            # Store sample for verification
                            if len(self.sample_grants) < 5:
                                self.sample_grants.append({
                                    "org": org_name,
                                    "amount": amount,
                                    "format": "A_erb_style",
                                    "raw_line": line[:50]
                                })
                            
                            i = j
                            continue
                
                # Format B: We already have org info, this line completes the grant
                elif current_grant:
                    # This line is: "NONE PC General purposes 12,500"
                    status_match = status_pattern.search(line)
                    if status_match:
                        current_grant.foundation_status = status_match.group(1)
                        current_grant.purpose = line[status_match.end():amount_match.start()].strip()
                    current_grant.amount = amount
                    grants.append(current_grant)
                    self.format_b_count += 1  # Track Format B
                    
                    # Store sample for verification
                    if len(self.sample_grants) < 5:
                        self.sample_grants.append({
                            "org": current_grant.recipient_name,
                            "amount": amount,
                            "format": "B_joyce_style",
                            "raw_line": line[:50]
                        })
                    
                    current_grant = None
                    i += 1
                    continue
            
            # No amount on this line - could be start of a new grant
            if not amount_pattern.search(line):
                # If it looks like an org name, start accumulating
                if self._looks_like_org_name(line):
                    current_grant = Grant(
                        recipient_name=line,
                        grant_bucket=grant_bucket
                    )
                    # Check next lines for address
                    j = i + 1
                    while j < len(lines) and j < i + 4:
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        if self._is_header_line(next_line):
                            j += 1
                            continue
                        if amount_pattern.search(next_line) or status_pattern.search(next_line):
                            break
                        if self._is_address_line(next_line):
                            if not current_grant.recipient_address:
                                current_grant.recipient_address = next_line
                            else:
                                current_grant.recipient_city_state_zip = next_line
                        j += 1
                    i = j
                    continue
            
            i += 1
        
        return self._filter_valid_grants(grants)
    
    def extract_continuation_grants(self, section_text: str, grant_bucket: str = "3a") -> List[Grant]:
        """
        v2.7: Extract grants from continuation page format.
        
        Continuation pages have a different format:
        NAME AND ADDRESS | RELATIONSHIP | FOUNDATION STATUS | PURPOSE | AMOUNT
        
        Example from Frey Foundation:
        ACCESS OF WEST MICHIGAN     NONE                            PC                MINDSET MEALS—WALK FOR
        1700 28TH ST SE                                                               GOOD FOOD                                   500
        GRAND RAPIDS, MI 49508-1414
        """
        grants = []
        
        # Pattern for amount at end of line
        amount_pattern = re.compile(r'([\d,]{3,})\s*$')
        
        # Pattern for foundation status codes
        status_pattern = re.compile(r'\b(PC|NC|PF|GOV|SO)\b')
        
        # Pattern for relationship (typically NONE)
        relationship_pattern = re.compile(r'\bNONE\b', re.IGNORECASE)
        
        lines = section_text.split('\n')
        
        current_grant = None
        current_purpose_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Skip headers
            if self._is_header_line(line):
                i += 1
                continue
            
            # Check for amount at end of line
            amount_match = amount_pattern.search(line)
            
            if amount_match:
                amount = int(amount_match.group(1).replace(',', ''))
                
                # This line contains the amount - extract other info
                line_before_amount = line[:amount_match.start()].strip()
                
                # Check for status code
                status_match = status_pattern.search(line_before_amount)
                
                if current_grant:
                    # Finalize the current grant
                    if status_match:
                        current_grant.foundation_status = status_match.group(1)
                        # Purpose is between status and amount
                        current_grant.purpose = line_before_amount[status_match.end():].strip()
                    else:
                        # Purpose might be the whole line before amount
                        current_grant.purpose = " ".join(current_purpose_lines + [line_before_amount]).strip()
                    
                    current_grant.amount = amount
                    grants.append(current_grant)
                    self.format_c_count += 1
                    
                    if len(self.sample_grants) < 5:
                        self.sample_grants.append({
                            "org": current_grant.recipient_name,
                            "amount": amount,
                            "format": "C_continuation",
                            "raw_line": line[:50]
                        })
                    
                    current_grant = None
                    current_purpose_lines = []
                
                i += 1
                continue
            
            # No amount - could be org name, address, or purpose continuation
            if self._looks_like_org_name(line) and not current_grant:
                # Start new grant - look for NONE and status in same line
                rel_match = relationship_pattern.search(line)
                stat_match = status_pattern.search(line)
                
                if rel_match:
                    # Format: "ORG NAME     NONE     PC     PURPOSE"
                    org_end = rel_match.start()
                    org_name = line[:org_end].strip()
                else:
                    org_name = line
                
                current_grant = Grant(
                    recipient_name=org_name,
                    grant_bucket=grant_bucket
                )
                
                if rel_match:
                    current_grant.relationship = "NONE"
                if stat_match:
                    current_grant.foundation_status = stat_match.group(1)
                    # Everything after status is start of purpose
                    purpose_start = line[stat_match.end():].strip()
                    if purpose_start:
                        current_purpose_lines.append(purpose_start)
                
            elif self._is_address_line(line) and current_grant:
                # Address line
                if not current_grant.recipient_address:
                    current_grant.recipient_address = line
                elif not current_grant.recipient_city_state_zip:
                    current_grant.recipient_city_state_zip = line
            
            elif current_grant:
                # Continuation of purpose or other info
                # Check for status code
                stat_match = status_pattern.search(line)
                if stat_match:
                    current_grant.foundation_status = stat_match.group(1)
                    purpose_part = line[stat_match.end():].strip()
                    if purpose_part:
                        current_purpose_lines.append(purpose_part)
                else:
                    current_purpose_lines.append(line)
            
            i += 1
        
        return self._filter_valid_grants(grants)
    
    def _is_header_line(self, line: str) -> bool:
        """Check if line is a header/form artifact."""
        artifacts = [
            'Recipient', 'Name and address', 'Foundation status',
            'Purpose of grant', 'If recipient is', 'contribution',
            'Paid during the year', 'Approved for future',
            'Form 990-PF', 'Part XIV', 'Supplementary Information',
            'https://', 'propublica', 'Page', 'show any relationship',
            'Grants and Contributions', 'Line 3a', 'Line 3b',
            '(continued)', 'Relationship', 'Amount'
        ]
        line_lower = line.lower().strip()
        # Also skip single-word header fragments like "status"
        if line_lower in ['status', 'amount', 'purpose', 'relationship']:
            return True
        return any(a.lower() in line_lower for a in artifacts)
    
    def _is_address_line(self, line: str) -> bool:
        """Check if line looks like an address."""
        # Street address patterns
        if re.match(r'^\d+\s+\w', line):  # Starts with number
            return True
        if re.match(r'^P\.?O\.?\s*Box', line, re.IGNORECASE):
            return True
        if re.match(r'^Suite\s+\d', line, re.IGNORECASE):
            return True
        # City, State ZIP
        if re.search(r',\s*[A-Z]{2}\s+\d{5}', line):
            return True
        # Just state and zip
        if re.match(r'^[A-Z]{2}\s+\d{5}', line):
            return True
        return False
    
    def _looks_like_org_name(self, line: str) -> bool:
        """Check if line looks like an organization name."""
        # Must be mostly letters
        if not re.search(r'[A-Za-z]{3,}', line):
            return False
        # Shouldn't start with numbers (address)
        if re.match(r'^\d+\s', line):
            return False
        # Shouldn't start with ordinal + address word
        if re.match(r'^(\d+(?:ST|ND|RD|TH)\s+(?:FLOOR|STREET|AVE|AVENUE|DRIVE|ROAD))', line, re.IGNORECASE):
            return False
        # Shouldn't start with address indicators
        if re.match(r'^(SUITE|FLOOR|ROOM|P\.?O\.?\s*BOX|ONE\s+|TWO\s+|THREE\s+)', line, re.IGNORECASE):
            return False
        # Shouldn't be a status/purpose line
        if re.match(r'^(PC|NC|PF|GOV|SO|NONE)\s', line):
            return False
        # Should be reasonably long
        if len(line) < 3:
            return False
        # Shouldn't look like a city/state line
        if re.search(r',\s*[A-Z]{2}\s+\d{5}', line):
            return False
        return True
    
    def _filter_valid_grants(self, grants: List[Grant]) -> List[Grant]:
        """Filter out invalid/artifact grants."""
        valid = []
        for g in grants:
            # Skip if amount is 0 or unreasonably high
            if g.amount <= 0 or g.amount > 500_000_000:
                continue
            # Skip if name is too short or looks like address
            if len(g.recipient_name) < 3:
                continue
            if re.match(r'^\d+\s', g.recipient_name):  # Starts with address number
                continue
            # Skip ordinal + address type (e.g., "6TH FLOOR")
            if re.match(r'^\d+(?:ST|ND|RD|TH)\s+(?:FLOOR|STREET|AVE|AVENUE)', g.recipient_name, re.IGNORECASE):
                continue
            # Skip if name starts with address indicators
            if re.match(r'^(SUITE|FLOOR|ROOM|P\.?O\.?\s*BOX)\b', g.recipient_name, re.IGNORECASE):
                continue
            # Skip if name is just a road/street type
            if re.match(r'^(ROAD|STREET|AVENUE|DRIVE|LANE|WAY|BOULEVARD|BLVD)$', g.recipient_name, re.IGNORECASE):
                continue
            # Skip if name contains ProPublica artifacts
            if 'propublica' in g.recipient_name.lower():
                continue
            if 'Full Filing' in g.recipient_name:
                continue
            # Skip "(SEE STATEMENT)" placeholders
            if re.search(r'SEE\s+STATEMENT', g.recipient_name, re.IGNORECASE):
                continue
            valid.append(g)
        return valid


class BoardExtractor:
    """Extracts board members from Part VII."""
    
    def extract(self, text: str) -> List[BoardMember]:
        """Extract board members from Part VII, including continuation pages."""
        members = []
        
        # Find all Part VII sections (main and continuations)
        sections = self._find_all_part_vii_sections(text)
        
        for section in sections:
            section_members = self._extract_from_section(section)
            # Deduplicate by name
            for m in section_members:
                if not any(existing.name == m.name for existing in members):
                    members.append(m)
        
        return members
    
    def _find_all_part_vii_sections(self, text: str) -> List[str]:
        """
        v2.7: Find main Part VII AND continuation pages.
        """
        sections = []
        
        # Pattern 1: Main Part VII header
        main_pattern = re.compile(
            r'Part\s+VII\s+Information\s+About\s+Officers',
            re.IGNORECASE
        )
        
        # Pattern 2: Part VII Line 1 continuation
        continuation_pattern = re.compile(
            r'Part\s+VII,?\s+Line\s+1\s+.*?(?:officers|directors|trustees|compensation)',
            re.IGNORECASE | re.DOTALL
        )
        
        # Find main Part VII section
        main_match = main_pattern.search(text)
        if main_match:
            start = main_match.start()
            end_patterns = [
                r'2\s+Compensation\s+of\s+five\s+highest',
                r'Part\s+VIII',
                r'Part\s+VII,?\s+Line\s+1'
            ]
            end = start + 10000
            for pattern in end_patterns:
                end_match = re.search(pattern, text[start:], re.IGNORECASE)
                if end_match:
                    end = min(end, start + end_match.start())
            sections.append(text[start:end])
        
        # Find all Part VII Line 1 continuations
        for match in continuation_pattern.finditer(text):
            start = match.start()
            end_patterns = [
                r'Part\s+VIII',
                r'Part\s+XIV',
                r'Part\s+XV',
                r'Form\s+990-PF\s*\(\d',
            ]
            end = start + 10000
            for pattern in end_patterns:
                end_match = re.search(pattern, text[start:], re.IGNORECASE)
                if end_match:
                    end = min(end, start + end_match.start())
            sections.append(text[start:end])
        
        return sections
    
    def _extract_from_section(self, section: str) -> List[BoardMember]:
        """Extract board members from a single section."""
        members = []
        lines = section.split('\n')
        
        # Track which names we've already found
        found_names = set()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Skip header lines
            if any(skip in line.upper() for skip in ['NAME AND ADDRESS', 'TITLE, AND AVERAGE', 
                                                       'COMPENSATION', 'DEVOTED TO POSITION',
                                                       'CONTRIBUTIONS TO', 'EXPENSE ACCOUNT',
                                                       'PART VII', 'FORM 990', 'PAGE ', 
                                                       'CONTRACTORS', 'FOUNDATION-']):
                continue
            
            # Pattern A: Main Part VII format (Frey style)
            # "DAVID G. FREY, JR. CHAIR, 18.0" (pdfplumber normalizes whitespace)
            # Next line has compensation: "46,318 0 0"
            match_a = re.match(
                r'^([A-Z][A-Z\.\s,\-\']+?)\s+'  # Name (greedy but lazy)
                r'(CHAIR|CEO|PRESIDENT|TRUSTEE|DIRECTOR|VP|VICE\s*CHAIR|VICE\s*PRESIDENT|SECRETARY|TREASURER)'
                r'(?:\s+AND\s+CEO)?'  # Optional "AND CEO" suffix
                r',?\s*(\d+\.?\d*)\s*$',  # Title with hours at end
                line, re.IGNORECASE
            )
            
            if match_a:
                name = match_a.group(1).strip().rstrip(',').strip()
                title = match_a.group(2).strip()
                hours = float(match_a.group(3)) if match_a.group(3) else 0.0
                
                if self._is_valid_name(name) and name not in found_names:
                    # Look for compensation on next lines
                    comp, benefits = self._find_compensation_in_lines(lines, i + 1)
                    members.append(BoardMember(
                        name=name,
                        title=title,
                        hours_per_week=hours,
                        compensation=comp,
                        benefits=benefits
                    ))
                    found_names.add(name)
                continue
            
            # Pattern B: Continuation page format (Frey Part VII, Line 1)
            # "CAMPBELL W. FREY SUITE 1100, GRAND TRUSTEE, 3.0 17,000 0 0"
            # Note: Sometimes has space before comma like "TRUSTEE , 3.0"
            match_b = re.match(
                r'^([A-Z][A-Z\.\s\-\']+?)\s+'  # Name
                r'(?:SUITE|FLOOR|\d+\s+[A-Z]).*?\s+'  # Address fragment
                r'(CHAIR|CEO|PRESIDENT|TRUSTEE|DIRECTOR|VP|VICE\s*CHAIR|VICE\s*PRESIDENT|SECRETARY|TREASURER)'
                r'\s*,?\s*(\d+\.?\d*)\s+'  # Title with optional space before comma, then hours
                r'([\d,]+)\s+([\d,]+)\s+(\d+)',  # Comp, benefits, expenses
                line, re.IGNORECASE
            )
            
            if match_b:
                name = match_b.group(1).strip().rstrip(',').strip()
                title = match_b.group(2).strip()
                hours = float(match_b.group(3)) if match_b.group(3) else 0.0
                comp = int(match_b.group(4).replace(',', ''))
                benefits = int(match_b.group(5).replace(',', ''))
                
                if self._is_valid_name(name) and name not in found_names:
                    members.append(BoardMember(
                        name=name,
                        title=title,
                        hours_per_week=hours,
                        compensation=comp,
                        benefits=benefits
                    ))
                    found_names.add(name)
                continue
            
            # Pattern C: Simple all-in-one format with just NAME TITLE HOURS COMP...
            # "JOHN SMITH TRUSTEE 10.0 50000 5000 0"
            match_c = re.match(
                r'^([A-Z][A-Z\.\s,\-\']+?)\s+'
                r'(CHAIR|CEO|PRESIDENT|TRUSTEE|DIRECTOR|VP|VICE|SECRETARY|TREASURER|OFFICER|MANAGER)'
                r'[A-Z\s&,]*?\s*'
                r'(\d+\.?\d*)\s+'  # Hours
                r'([\d,]+)\s+([\d,]+)\s+(\d+)',  # Comp, benefits, expenses
                line, re.IGNORECASE
            )
            
            if match_c:
                name = match_c.group(1).strip().rstrip(',').strip()
                title = match_c.group(2).strip()
                hours = float(match_c.group(3))
                comp = int(match_c.group(4).replace(',', ''))
                benefits = int(match_c.group(5).replace(',', ''))
                
                if self._is_valid_name(name) and name not in found_names:
                    members.append(BoardMember(
                        name=name,
                        title=title,
                        hours_per_week=hours,
                        compensation=comp,
                        benefits=benefits
                    ))
                    found_names.add(name)
        
        return members
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted name is valid."""
        if 'NAME' in name.upper() or 'ADDRESS' in name.upper():
            return False
        if len(name) < 4:
            return False
        if re.match(r'^\d', name):  # Starts with number
            return False
        if re.match(r'^(SUITE|FLOOR|ROOM|PO BOX)', name, re.IGNORECASE):
            return False
        # Must have at least 2 words (first and last name)
        if len(name.split()) < 2:
            return False
        return True
    
    def _find_compensation_in_lines(self, lines: List[str], start_idx: int) -> Tuple[int, int]:
        """Look for compensation numbers in subsequent lines."""
        for i in range(start_idx, min(start_idx + 3, len(lines))):
            line = lines[i].strip()
            # Look for pattern like "46,318    0    0" or "46,318                                    0                     0"
            match = re.search(r'^\s*([\d,]+)\s+([\d,]+)\s+(\d+)\s*$', line)
            if match:
                return int(match.group(1).replace(',', '')), int(match.group(2).replace(',', ''))
        return 0, 0


class IRS990PFParser:
    """Main parser class for 990-PF forms."""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.meta_extractor = FoundationMetaExtractor()
        self.section_finder = SectionFinder()
        self.grant_extractor = GrantExtractor()
        self.board_extractor = BoardExtractor()
    
    def _calculate_confidence(self, parsed_total: int, reported_total: int) -> dict:
        """Calculate parsing confidence based on total matching."""
        if reported_total == 0:
            return {"match_pct": 0, "status": "no_reported_total", "confidence": "low"}
        
        match_pct = (parsed_total / reported_total) * 100
        variance = abs(100 - match_pct)
        
        if variance <= 1:
            status = "excellent"
            confidence = "high"
        elif variance <= 5:
            status = "good"
            confidence = "medium-high"
        elif variance <= 10:
            status = "acceptable"
            confidence = "medium"
        elif variance <= 20:
            status = "needs_review"
            confidence = "low"
        else:
            status = "poor"
            confidence = "very_low"
        
        return {
            "match_pct": round(match_pct, 1),
            "variance_pct": round(variance, 1),
            "status": status,
            "confidence": confidence
        }
    
    def parse(self, file_bytes: bytes, source_file: str = "", tax_year_override: str = "") -> dict:
        """
        Parse a 990-PF PDF file.
        
        Returns dict with:
        - foundation_meta: name, EIN, tax year
        - grants_df: DataFrame of grants
        - people_df: DataFrame of board members
        - diagnostics: parsing stats and warnings
        """
        diagnostics = {
            "parser_version": PARSER_VERSION,
            "source_file": source_file,
            "grants_3a_count": 0,
            "grants_3a_total": 0,
            "grants_3b_count": 0,
            "grants_3b_total": 0,
            "reported_total_3a": 0,
            "reported_total_3b": 0,
            "board_count": 0,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Extract text from PDF
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            
            full_text = "\n".join(pages)
            cleaned_text = self.cleaner.clean(full_text)
            
            # Extract metadata
            meta = self.meta_extractor.extract(cleaned_text, tax_year_override)
            
            # Initialize grant lists
            grants = []
            
            # Extract grants from main 3a section
            result_3a = self.section_finder.find_grants_section_3a(cleaned_text)
            if result_3a[0] is not None:
                section_3a = cleaned_text[result_3a[0]:result_3a[1]]
                grants_3a = self.grant_extractor.extract_grants(section_3a, "3a")
                grants.extend(grants_3a)
                diagnostics["grants_3a_count"] = len(grants_3a)
                diagnostics["grants_3a_total"] = sum(g.amount for g in grants_3a)
                if len(result_3a) > 2:
                    diagnostics["reported_total_3a"] = result_3a[2]
                    # Calculate confidence
                    diagnostics["confidence_3a"] = self._calculate_confidence(
                        diagnostics["grants_3a_total"],
                        diagnostics["reported_total_3a"]
                    )
                    
                    # Add warning if confidence is low
                    if diagnostics["confidence_3a"].get("confidence") in ["low", "very_low"]:
                        diagnostics["warnings"].append(
                            f"3a parsing variance is {diagnostics['confidence_3a'].get('variance_pct')}% - "
                            f"parsed ${diagnostics['grants_3a_total']:,} vs reported ${diagnostics['reported_total_3a']:,}"
                        )
            
            # v2.7: Extract grants from Part XIV continuation pages
            continuation_sections = self.section_finder.find_continuation_sections(cleaned_text)
            
            continuation_3a_grants = []
            for start, end in continuation_sections.get('3a_sections', []):
                section_text = cleaned_text[start:end]
                cont_grants = self.grant_extractor.extract_continuation_grants(section_text, "3a")
                continuation_3a_grants.extend(cont_grants)
            
            if continuation_3a_grants:
                # If main section had no/few grants but continuation has many, use continuation
                if diagnostics["grants_3a_count"] < 5 and len(continuation_3a_grants) > 5:
                    grants = [g for g in grants if g.grant_bucket != "3a"]  # Remove main 3a
                    grants.extend(continuation_3a_grants)
                    diagnostics["grants_3a_count"] = len(continuation_3a_grants)
                    diagnostics["grants_3a_total"] = sum(g.amount for g in continuation_3a_grants)
                    diagnostics["grants_from_continuation_3a"] = True
                    
                    # Recalculate confidence
                    if diagnostics.get("reported_total_3a"):
                        diagnostics["confidence_3a"] = self._calculate_confidence(
                            diagnostics["grants_3a_total"],
                            diagnostics["reported_total_3a"]
                        )
            
            # Extract grants from 3b section
            result_3b = self.section_finder.find_grants_section_3b(cleaned_text)
            if result_3b[0] is not None:
                section_3b = cleaned_text[result_3b[0]:result_3b[1]]
                grants_3b = self.grant_extractor.extract_grants(section_3b, "3b")
                grants.extend(grants_3b)
                diagnostics["grants_3b_count"] = len(grants_3b)
                diagnostics["grants_3b_total"] = sum(g.amount for g in grants_3b)
                if len(result_3b) > 2:
                    diagnostics["reported_total_3b"] = result_3b[2]
                    # Calculate confidence
                    diagnostics["confidence_3b"] = self._calculate_confidence(
                        diagnostics["grants_3b_total"],
                        diagnostics["reported_total_3b"]
                    )
                    
                    # Add warning if confidence is low
                    if diagnostics["confidence_3b"].get("confidence") in ["low", "very_low"]:
                        diagnostics["warnings"].append(
                            f"3b parsing variance is {diagnostics['confidence_3b'].get('variance_pct')}% - "
                            f"parsed ${diagnostics['grants_3b_total']:,} vs reported ${diagnostics['reported_total_3b']:,}"
                        )
            
            # v2.7: Extract grants from Part XIV 3b continuation pages
            continuation_3b_grants = []
            for start, end in continuation_sections.get('3b_sections', []):
                section_text = cleaned_text[start:end]
                cont_grants = self.grant_extractor.extract_continuation_grants(section_text, "3b")
                continuation_3b_grants.extend(cont_grants)
            
            if continuation_3b_grants:
                if diagnostics["grants_3b_count"] < 5 and len(continuation_3b_grants) > 5:
                    grants = [g for g in grants if g.grant_bucket != "3b"]
                    grants.extend(continuation_3b_grants)
                    diagnostics["grants_3b_count"] = len(continuation_3b_grants)
                    diagnostics["grants_3b_total"] = sum(g.amount for g in continuation_3b_grants)
                    diagnostics["grants_from_continuation_3b"] = True
                    
                    if diagnostics.get("reported_total_3b"):
                        diagnostics["confidence_3b"] = self._calculate_confidence(
                            diagnostics["grants_3b_total"],
                            diagnostics["reported_total_3b"]
                        )
            
            # Get format diagnostics
            diagnostics["extraction_format"] = self.grant_extractor.get_format_diagnostics()
            diagnostics["sample_grants"] = diagnostics["extraction_format"].get("sample_grants", [])
            
            # Extract board members (v2.7: includes continuation pages)
            board = self.board_extractor.extract(cleaned_text)
            diagnostics["board_count"] = len(board)
            
            # Overall parsing quality
            if not grants:
                diagnostics["warnings"].append("No grants extracted - check PDF format")
            
            # Helper to parse city/state from city_state_zip
            def parse_city_state(city_state_zip: str) -> tuple:
                """Parse 'CITY,ST ZIP' or 'CITY, ST ZIP' into (city, state)."""
                if not city_state_zip:
                    return "", ""
                text = city_state_zip.upper().strip()
                
                # Pattern: CITY,ST ZIP or CITY, ST ZIP (handle comma directly before state)
                # Match: "DETROIT,MI 48201" or "CHICAGO, IL 60601"
                match = re.search(r'^(.+?),\s*([A-Z]{2})\s+\d', text)
                if match:
                    return match.group(1).strip(), match.group(2)
                
                # Pattern: CITY ST ZIP (no comma)
                match = re.search(r'^(.+?)\s+([A-Z]{2})\s+\d{5}', text)
                if match:
                    return match.group(1).strip(), match.group(2)
                
                # Try just state code anywhere
                match = re.search(r'\b([A-Z]{2})\s+\d{5}', text)
                if match:
                    return "", match.group(1)
                    
                return "", ""
            
            # Convert to DataFrames with updated column names
            grants_data = []
            for g in grants:
                city, state = parse_city_state(g.recipient_city_state_zip)
                # Build raw address for fallback parsing
                addr_parts = [g.recipient_address, g.recipient_city_state_zip]
                raw_addr = ", ".join(p for p in addr_parts if p)
                
                grants_data.append({
                    'grantee_name': g.recipient_name,
                    'grantee_address': g.recipient_address,
                    'grantee_city': city,
                    'grantee_state': state,
                    'grantee_address_raw': raw_addr,
                    'status': g.foundation_status,
                    'purpose': g.purpose,
                    'grant_amount': g.amount,
                    'grant_bucket': g.grant_bucket,
                })
            
            grants_df = pd.DataFrame(grants_data) if grants_data else pd.DataFrame()
            
            people_df = pd.DataFrame([{
                'name': b.name,
                'title': b.title,
                'hours': b.hours_per_week,
                'compensation': b.compensation,
                'benefits': b.benefits
            } for b in board]) if board else pd.DataFrame()
            
            return {
                'foundation_meta': meta,
                'grants_df': grants_df,
                'people_df': people_df,
                'diagnostics': diagnostics
            }
            
        except Exception as e:
            diagnostics["errors"].append(str(e))
            return {
                'foundation_meta': {"foundation_name": "", "foundation_ein": "", "tax_year": None},
                'grants_df': pd.DataFrame(),
                'people_df': pd.DataFrame(),
                'diagnostics': diagnostics
            }


def parse_990_pdf(file_bytes: bytes, source_file: str = "", tax_year_override: str = "") -> dict:
    """
    Main entry point for parsing 990-PF PDFs.
    
    Args:
        file_bytes: Raw PDF bytes
        source_file: Filename for diagnostics
        tax_year_override: Optional tax year to use
    
    Returns:
        dict with keys: foundation_meta, grants_df, people_df, diagnostics
    """
    parser = IRS990PFParser()
    return parser.parse(file_bytes, source_file, tax_year_override)


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print(f"990-PF Parser v{PARSER_VERSION}")
        print("Usage: python irs990_parser.py <path_to_990pf.pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    result = parse_990_pdf(pdf_bytes, pdf_path)
    diag = result['diagnostics']
    
    print(f"\n{'='*60}")
    print(f"990-PF Parser v{PARSER_VERSION} Results")
    print('='*60)
    
    print(f"\n📋 FOUNDATION INFO")
    print(f"   Name: {result['foundation_meta']['foundation_name']}")
    print(f"   EIN: {result['foundation_meta']['foundation_ein']}")
    print(f"   Tax Year: {result['foundation_meta']['tax_year']}")
    
    print(f"\n📊 GRANTS SECTION 3a (Paid During Year)")
    print(f"   Parsed: {diag['grants_3a_count']} grants, ${diag['grants_3a_total']:,}")
    print(f"   Reported: ${diag['reported_total_3a']:,}")
    if diag.get('grants_from_continuation_3a'):
        print(f"   Source: Part XIV continuation pages")
    if diag.get('confidence_3a'):
        conf = diag['confidence_3a']
        print(f"   Match: {conf.get('match_pct', 0)}% ({conf.get('status', 'unknown')}) - Confidence: {conf.get('confidence', 'unknown')}")
    
    print(f"\n📊 GRANTS SECTION 3b (Approved Future)")
    print(f"   Parsed: {diag['grants_3b_count']} grants, ${diag['grants_3b_total']:,}")
    print(f"   Reported: ${diag['reported_total_3b']:,}")
    if diag.get('grants_from_continuation_3b'):
        print(f"   Source: Part XIV continuation pages")
    if diag.get('confidence_3b'):
        conf = diag['confidence_3b']
        print(f"   Match: {conf.get('match_pct', 0)}% ({conf.get('status', 'unknown')}) - Confidence: {conf.get('confidence', 'unknown')}")
    
    print(f"\n👥 BOARD MEMBERS: {diag['board_count']}")
    if not result['people_df'].empty:
        for _, row in result['people_df'].head(5).iterrows():
            print(f"   - {row['name']}: {row['title']}")
    
    print(f"\n🔍 EXTRACTION FORMAT DETECTION")
    fmt = diag.get('extraction_format', {})
    print(f"   Dominant format: {fmt.get('dominant_format', 'unknown')}")
    print(f"   Format A (Erb-style): {fmt.get('format_a_erb_style_count', 0)} grants")
    print(f"   Format B (Joyce-style): {fmt.get('format_b_joyce_style_count', 0)} grants")
    print(f"   Format C (Continuation): {fmt.get('format_c_continuation_count', 0)} grants")
    print(f"   Format confidence: {fmt.get('format_confidence', 0)*100:.0f}%")
    
    if diag.get('sample_grants'):
        print(f"\n📝 SAMPLE GRANTS (for verification)")
        for i, sg in enumerate(diag['sample_grants'][:5], 1):
            print(f"   {i}. {sg.get('org', 'Unknown')[:40]} - ${sg.get('amount', 0):,} [{sg.get('format', '')}]")
    
    if diag.get('warnings'):
        print(f"\n⚠️  WARNINGS")
        for w in diag['warnings']:
            print(f"   - {w}")
    
    if diag.get('errors'):
        print(f"\n❌ ERRORS")
        for e in diag['errors']:
            print(f"   - {e}")
    
    print(f"\n{'='*60}")
