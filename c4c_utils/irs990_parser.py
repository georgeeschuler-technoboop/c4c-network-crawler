"""
990-PF Parser v2.5 - Complete Module for OrgGraph
==================================================
Changes in v2.5:
- Added format detection diagnostics (Erb-style vs Joyce-style extraction order)
- Added parsing confidence score based on total matching
- Added sample grant logging for verification
- Better error reporting when totals don't match

Changes in v2.4:
- Fixed regex pattern for "Total ... 3a/3b" to handle spaced dots (. . . . .)
  Previously only matched consecutive dots (......), now matches both styles

Changes in v2.3:
- Fixed org name detection to capture foundation name from Part I header
- Fixed board member extraction from Part VII
- Handles inline amounts (Joyce style) and separate-line amounts (Erb style)
- Robust ProPublica artifact removal
- Filters address fragments captured as org names

Exports:
- parse_990_pdf(file_bytes, filename, tax_year_override) -> dict
- IRS990PFParser class for direct use
"""

PARSER_VERSION = "2.5"

import re
import io
from dataclasses import dataclass, field
from typing import Optional, List
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
    gl_relevant: bool = False
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
        
        # Foundation name - look in the header section
        # Pattern 1: Look for "THE ... FOUNDATION" pattern early in document
        name_match = re.search(
            r'\b(THE\s+[A-Z][A-Z\s&\-\.]+(?:FOUNDATION|TRUST|FUND))\b',
            text[:3000], re.IGNORECASE
        )
        if name_match:
            name = name_match.group(1).strip()
            # Clean up extra whitespace
            name = re.sub(r'\s+', ' ', name)
            meta["foundation_name"] = name
        
        # Pattern 2: Look at very beginning of document for foundation name
        if not meta["foundation_name"]:
            # First substantive text line often has foundation name
            lines = text.split('\n')[:30]
            for line in lines:
                line = line.strip()
                # Skip short lines or form labels
                if len(line) < 10 or line.startswith('Form') or line.startswith('efile'):
                    continue
                # Look for "THE ... FOUNDATION" pattern
                if re.search(r'\b(THE\s+)?[A-Z][A-Z\s&\-\.]+FOUNDATION\b', line, re.IGNORECASE):
                    name = re.search(r'((?:THE\s+)?[A-Z][A-Z\s&\-\.]+(?:FOUNDATION|TRUST|FUND))', 
                                    line, re.IGNORECASE)
                    if name:
                        meta["foundation_name"] = name.group(1).strip()
                        break
        
        # EIN
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


class GrantExtractor:
    """Extracts individual grants from section text."""
    
    # Great Lakes keywords for tagging
    GL_KEYWORDS = [
        'great lakes', 'lake michigan', 'lake superior', 'lake huron', 
        'lake erie', 'lake ontario', 'watershed', 'water quality',
        'freshwater', 'clean water', 'river', 'wetland'
    ]
    
    def __init__(self):
        # Track extraction format for diagnostics
        self.format_a_count = 0  # Erb-style: status/amount BEFORE org name
        self.format_b_count = 0  # Joyce-style: org name BEFORE status/amount
        self.sample_grants = []  # Store first few for verification
    
    def get_format_diagnostics(self) -> dict:
        """Return diagnostics about which extraction format was detected."""
        total = self.format_a_count + self.format_b_count
        if total == 0:
            dominant = "unknown"
            confidence = 0
        elif self.format_a_count > self.format_b_count:
            dominant = "format_a_erb_style"
            confidence = self.format_a_count / total
        else:
            dominant = "format_b_joyce_style"
            confidence = self.format_b_count / total
        
        return {
            "dominant_format": dominant,
            "format_a_erb_style_count": self.format_a_count,
            "format_b_joyce_style_count": self.format_b_count,
            "format_confidence": round(confidence, 2),
            "sample_grants": self.sample_grants[:5]  # First 5 for verification
        }
    
    def extract_grants(self, section_text: str, grant_bucket: str = "3a") -> List[Grant]:
        """Extract grants from the section text."""
        grants = []
        
        # Reset counters for this section
        self.format_a_count = 0
        self.format_b_count = 0
        self.sample_grants = []
        
        # Split into potential grant blocks
        # Each grant typically has: ORG_NAME, ADDRESS, CITY/STATE/ZIP, PC/NC, PURPOSE, AMOUNT
        
        # Pattern for amount at end of line
        amount_pattern = re.compile(r'([\d,]{3,})\s*$')
        
        # Pattern for foundation status codes
        status_pattern = re.compile(r'\b(PC|NC|PF|GOV|SO)\b')
        
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
                                if not addr_line or status_pattern.search(addr_line) or amount_pattern.search(addr_line):
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
                                grant_bucket=grant_bucket,
                                gl_relevant=self._is_gl_relevant(org_name + " " + purpose_text)
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
                    current_grant.gl_relevant = self._is_gl_relevant(
                        current_grant.recipient_name + " " + current_grant.purpose
                    )
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
    
    def _is_header_line(self, line: str) -> bool:
        """Check if line is a header/form artifact."""
        artifacts = [
            'Recipient', 'Name and address', 'Foundation status',
            'Purpose of grant', 'If recipient is', 'contribution',
            'Paid during the year', 'Approved for future',
            'Form 990-PF', 'Part XIV', 'Supplementary Information',
            'https://', 'propublica', 'Page', 'show any relationship'
        ]
        line_lower = line.lower()
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
        # Shouldn't be a status/purpose line
        if re.match(r'^(PC|NC|PF|GOV|SO|NONE)\s', line):
            return False
        # Should be reasonably long
        if len(line) < 3:
            return False
        return True
    
    def _is_gl_relevant(self, text: str) -> bool:
        """Check if grant is Great Lakes relevant."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.GL_KEYWORDS)
    
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
            # Skip if name contains ProPublica artifacts
            if 'propublica' in g.recipient_name.lower():
                continue
            if 'Full Filing' in g.recipient_name:
                continue
            valid.append(g)
        return valid


class BoardExtractor:
    """Extracts board members from Part VII."""
    
    def extract(self, text: str) -> List[BoardMember]:
        """Extract board members from Part VII."""
        members = []
        
        # Find Part VII section
        part7_match = re.search(
            r'Part\s+VII.*?Information\s+About\s+Officers',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if not part7_match:
            return members
        
        # Find the section text - from Part VII to Part VIII or "2 Compensation of five highest"
        start = part7_match.start()
        # Look for end markers
        end_markers = [
            r'2\s+Compensation\s+of\s+five\s+highest',
            r'Part\s+VIII'
        ]
        end = start + 10000
        for marker in end_markers:
            end_match = re.search(marker, text[start:], re.IGNORECASE)
            if end_match:
                end = min(end, start + end_match.start())
        
        section = text[start:end]
        
        # Pattern for Erb-style: NAME TITLE COMP BENEFITS EXPENSES (with hours on next line)
        # Example: "JOHN M ERB CHAIR AND CEO 253,523 62,198 0"
        # Or: "DEBORAH D ERB TRUSTEE 0 0 0"
        
        # Pattern: NAME (all caps, 2+ words) + TITLE + 3 numbers
        pattern1 = re.compile(
            r'^([A-Z][A-Z\s]+?)\s+'  # Name (all caps)
            r'((?:CHAIR|CEO|PRESIDENT|TRUSTEE|DIRECTOR|VP|VICE\s+PRESIDENT|SECRETARY|TREASURER)[A-Z\s]*?)\s+'  # Title
            r'([\d,]+)\s+([\d,]+)\s+(\d+)',  # Comp, benefits, expenses
            re.MULTILINE
        )
        
        for match in pattern1.finditer(section):
            name = match.group(1).strip()
            title = match.group(2).strip()
            comp = int(match.group(3).replace(',', ''))
            benefits = int(match.group(4).replace(',', ''))
            
            # Skip header lines
            if 'NAME' in name or 'ADDRESS' in name:
                continue
            if len(name) < 4:
                continue
                
            members.append(BoardMember(
                name=name,
                title=title,
                compensation=comp,
                benefits=benefits
            ))
        
        # Also try Joyce-style pattern: NAME TITLE HOURS COMP BENEFITS EXPENSES (all on one line)
        pattern2 = re.compile(
            r'([A-Z][A-Z\s]+?)\s+'
            r'((?:CHAIR|CEO|PRESIDENT|TRUSTEE|DIRECTOR|VP|VICE|SECRETARY|TREASURER|OFFICER|MANAGER)[A-Z\s]*?)\s+'
            r'(\d+\.?\d*)\s+'  # Hours
            r'([\d,]+)\s+([\d,]+)\s+(\d+)',
            re.MULTILINE
        )
        
        for match in pattern2.finditer(section):
            name = match.group(1).strip()
            # Check we didn't already capture this person
            if any(m.name == name for m in members):
                continue
            
            title = match.group(2).strip()
            hours = float(match.group(3))
            comp = int(match.group(4).replace(',', ''))
            benefits = int(match.group(5).replace(',', ''))
            
            if 'NAME' in name or 'ADDRESS' in name:
                continue
            if len(name) < 4:
                continue
                
            members.append(BoardMember(
                name=name,
                title=title,
                hours_per_week=hours,
                compensation=comp,
                benefits=benefits
            ))
        
        return members


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
        """Parse 990-PF and return structured data."""
        
        diagnostics = {
            "parser_version": PARSER_VERSION,
            "source_file": source_file,
            "pages_processed": 0,
            "grants_3a_count": 0,
            "grants_3a_total": 0,
            "grants_3b_count": 0,
            "grants_3b_total": 0,
            "board_count": 0,
            "reported_total_3a": 0,
            "reported_total_3b": 0,
            "confidence_3a": {},
            "confidence_3b": {},
            "extraction_format": {},
            "sample_grants": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # Extract text from PDF
            pdf_stream = io.BytesIO(file_bytes)
            with pdfplumber.open(pdf_stream) as pdf:
                diagnostics["pages_processed"] = len(pdf.pages)
                
                all_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    all_text += page_text + "\n"
            
            # Clean ProPublica artifacts
            cleaned_text = self.cleaner.clean(all_text)
            
            # Extract metadata
            meta = self.meta_extractor.extract(cleaned_text, tax_year_override)
            
            # Extract grants from 3a section
            grants = []
            result_3a = self.section_finder.find_grants_section_3a(cleaned_text)
            if result_3a[0] is not None:
                section_3a = cleaned_text[result_3a[0]:result_3a[1]]
                grants_3a = self.grant_extractor.extract_grants(section_3a, "3a")
                grants.extend(grants_3a)
                diagnostics["grants_3a_count"] = len(grants_3a)
                diagnostics["grants_3a_total"] = sum(g.amount for g in grants_3a)
                
                # Get format diagnostics from 3a extraction
                format_diag = self.grant_extractor.get_format_diagnostics()
                diagnostics["extraction_format"] = format_diag
                diagnostics["sample_grants"] = format_diag.get("sample_grants", [])
                
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
            
            # Extract board members
            board = self.board_extractor.extract(cleaned_text)
            diagnostics["board_count"] = len(board)
            
            # Overall parsing quality
            if not grants:
                diagnostics["warnings"].append("No grants extracted - check PDF format")
            
            # Convert to DataFrames
            grants_df = pd.DataFrame([{
                'org_name': g.recipient_name,
                'address': g.recipient_address,
                'city_state_zip': g.recipient_city_state_zip,
                'status': g.foundation_status,
                'purpose': g.purpose,
                'amount': g.amount,
                'grant_bucket': g.grant_bucket,
                'gl_relevant': g.gl_relevant
            } for g in grants]) if grants else pd.DataFrame()
            
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
    if diag.get('confidence_3a'):
        conf = diag['confidence_3a']
        print(f"   Match: {conf.get('match_pct', 0)}% ({conf.get('status', 'unknown')}) - Confidence: {conf.get('confidence', 'unknown')}")
    
    print(f"\n📊 GRANTS SECTION 3b (Approved Future)")
    print(f"   Parsed: {diag['grants_3b_count']} grants, ${diag['grants_3b_total']:,}")
    print(f"   Reported: ${diag['reported_total_3b']:,}")
    if diag.get('confidence_3b'):
        conf = diag['confidence_3b']
        print(f"   Match: {conf.get('match_pct', 0)}% ({conf.get('status', 'unknown')}) - Confidence: {conf.get('confidence', 'unknown')}")
    
    print(f"\n👥 BOARD MEMBERS: {diag['board_count']}")
    
    print(f"\n🔍 EXTRACTION FORMAT DETECTION")
    fmt = diag.get('extraction_format', {})
    print(f"   Dominant format: {fmt.get('dominant_format', 'unknown')}")
    print(f"   Format A (Erb-style): {fmt.get('format_a_erb_style_count', 0)} grants")
    print(f"   Format B (Joyce-style): {fmt.get('format_b_joyce_style_count', 0)} grants")
    print(f"   Format confidence: {fmt.get('format_confidence', 0)*100:.0f}%")
    
    if diag.get('sample_grants'):
        print(f"\n📝 SAMPLE GRANTS (for verification)")
        for i, sg in enumerate(diag['sample_grants'][:3], 1):
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
