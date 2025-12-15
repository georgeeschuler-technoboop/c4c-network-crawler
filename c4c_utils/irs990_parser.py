"""
990-PF Parser v2.3 - Complete Module for OrgGraph (US)
======================================================
Based on team review of v2.1 and v2.2. Key changes:

P0 Fixes (v2.2):
- Fixed operator precedence bug in AM/PM line filtering
- Changed re.DOTALL to re.MULTILINE for safer pattern matching

P1 Fixes (v2.2):
- Separates 3a (paid during year) from 3b (approved for future)
- Adds reported vs computed totals for QA reconciliation
- Adds Great Lakes state relevance tagging
- Additional ProPublica artifact patterns

P1 Patches (v2.3):
1. Anchor Part XIV search before finding "a Paid during the year"
2. Totals regex handles newlines between Total and number
3. GL state inference falls back to recipient_address
4. Mismatch threshold reports actual diffs (strict + lenient)
5. Dedup pass to remove page-break duplicates
6. Export grantee_address_raw for audit

Exports:
- parse_990_pdf(file_bytes, filename, tax_year_override) -> dict
- IRS990PFParser class for direct use
"""

import re
import io
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import pypdf


# Great Lakes states for relevance tagging
GL_STATES = {"MI", "OH", "MN", "WI", "IN", "IL", "NY"}


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
    grant_bucket: str = "3a_paid"  # "3a_paid" or "3b_future"
    gl_relevant: bool = False  # True if grantee in Great Lakes state


@dataclass  
class Person:
    """Represents a board member/officer."""
    name: str
    title: str = ""
    compensation: float = 0


class IRS990PFParser:
    """Parse 990-PF forms to extract grants and board members."""
    
    # ProPublica artifacts to remove
    # NOTE: Using MULTILINE not DOTALL to prevent cross-line matching
    PROPUBLICA_PATTERNS = [
        # Concatenated Page + URL (no space between) - must come first
        r'Page\s+\d+\s+of\s+\d+https?://[^\n]+',
        # Standard header with date
        r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}\s*:\s*\d{2}\s*[AP]M\s*[^\n]*Full Filing[^\n]*ProPublica[^\n]*',
        # Header without date but with foundation name
        r'[A-Z][A-Za-z\s]+(?:Foundation|Fund|Trust|Inc\.?)\s*-\s*Full Filing\s*-\s*Nonprofit Explorer\s*-\s*ProPublica[^\n]*',
        # Any line containing "Full Filing - Nonprofit Explorer - ProPublica"
        r'[^\n]*Full Filing\s*-\s*Nonprofit Explorer\s*-\s*ProPublica[^\n]*',
        # URL pattern
        r'https?://projects\.propublica\.org/[^\s\n]+',
        # Page marker
        r'Page\s+\d+\s+of\s+\d+',
        # P1.1: Additional ProPublica/efile markers
        r'<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>[^\n]*',
        r'efile Public Visual Render ObjectId:[^\n]*',
        r'TY\s+\d{4}\s+IRS\s+990\s+e-File\s+Render[^\n]*',
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
    
    # Regex for reported totals - Patch 2: Allow newlines with tight window
    TOTAL_3A_RE = re.compile(r'\bTotal\b[\s\S]{0,120}\b3a\b[\s\S]{0,40}?([\d,]+)', re.IGNORECASE)
    TOTAL_3B_RE = re.compile(r'\bTotal\b[\s\S]{0,120}\b3b\b[\s\S]{0,40}?([\d,]+)', re.IGNORECASE)
    
    def __init__(self):
        # P0.2 fix: Use MULTILINE instead of DOTALL
        self.compiled_propublica = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.PROPUBLICA_PATTERNS]
        self.compiled_skip = [re.compile(p, re.IGNORECASE) for p in self.SKIP_PATTERNS]
    
    def parse_bytes(self, pdf_bytes: bytes) -> tuple[list[Grant], list[Grant], list[Person], dict]:
        """Parse PDF from bytes. Returns (grants_3a, grants_3b, people, metadata)."""
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self._parse_text(all_text)
    
    def parse_file(self, pdf_path: str) -> tuple[list[Grant], list[Grant], list[Person], dict]:
        """Parse PDF from file path. Returns (grants_3a, grants_3b, people, metadata)."""
        reader = pypdf.PdfReader(pdf_path)
        all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self._parse_text(all_text)
    
    def _parse_text(self, all_text: str) -> tuple[list[Grant], list[Grant], list[Person], dict]:
        """Parse extracted text. Returns (grants_3a, grants_3b, people, metadata)."""
        # Extract metadata first (before cleaning)
        metadata = self._extract_metadata(all_text)
        
        # Clean text
        cleaned = self._clean_text(all_text)
        
        # Initialize results
        grants_3a = []
        grants_3b = []
        reported_total_3a = None
        reported_total_3b = None
        
        # Find Part XIV Line 3 block (contains both 3a and 3b)
        block = self._find_part_xiv_line3_block(cleaned)
        
        if block:
            # Split into 3a (paid) and 3b (future)
            paid_text, future_text = self._split_paid_future(block)
            
            # Parse reported totals
            reported_total_3a = self._parse_reported_total(paid_text, "3a")
            reported_total_3b = self._parse_reported_total(future_text, "3b") if future_text else None
            
            # Extract 3a grants
            raw_3a = self._extract_grants(paid_text)
            grants_3a = [g for g in raw_3a if self._is_valid_grant(g)]
            grants_3a = self._dedup_grants(grants_3a)  # Patch 5: dedup
            for g in grants_3a:
                g.grant_bucket = "3a_paid"
                self._tag_gl_relevance(g)
            
            # Extract 3b grants (if section exists)
            if future_text:
                raw_3b = self._extract_grants(future_text)
                grants_3b = [g for g in raw_3b if self._is_valid_grant(g)]
                grants_3b = self._dedup_grants(grants_3b)  # Patch 5: dedup
                for g in grants_3b:
                    g.grant_bucket = "3b_future"
                    self._tag_gl_relevance(g)
        
        # Compute totals for QA
        computed_total_3a = sum(g.amount for g in grants_3a)
        computed_total_3b = sum(g.amount for g in grants_3b)
        
        # Patch 4: Calculate actual diffs for diagnostics
        diff_3a = abs((reported_total_3a or 0) - computed_total_3a) if reported_total_3a else None
        diff_3b = abs((reported_total_3b or 0) - computed_total_3b) if reported_total_3b else None
        diff_pct_3a = (diff_3a / reported_total_3a * 100) if reported_total_3a and diff_3a else None
        diff_pct_3b = (diff_3b / reported_total_3b * 100) if reported_total_3b and diff_3b else None
        
        # Add totals to metadata for diagnostics
        metadata['reported_total_3a'] = reported_total_3a
        metadata['reported_total_3b'] = reported_total_3b
        metadata['computed_total_3a'] = computed_total_3a
        metadata['computed_total_3b'] = computed_total_3b
        metadata['diff_3a'] = diff_3a
        metadata['diff_3b'] = diff_3b
        metadata['diff_pct_3a'] = diff_pct_3a
        metadata['diff_pct_3b'] = diff_pct_3b
        # Strict mismatch: any difference > $1
        metadata['total_mismatch_3a_strict'] = (diff_3a is not None and diff_3a > 1)
        metadata['total_mismatch_3b_strict'] = (diff_3b is not None and diff_3b > 1)
        # Lenient mismatch: difference > $100 (allows for rounding)
        metadata['total_mismatch_3a'] = (diff_3a is not None and diff_3a > 100)
        metadata['total_mismatch_3b'] = (diff_3b is not None and diff_3b > 100)
        
        # Extract people (board members)
        people = self._extract_people(cleaned)
        
        return grants_3a, grants_3b, people, metadata
    
    def _extract_metadata(self, text: str) -> dict:
        """Extract foundation metadata from the form."""
        metadata = {
            'foundation_name': '',
            'ein': '',
            'tax_year': '',
            'is_990pf': True,
            'form_type': '990-PF',
            'notes': [
                "Primary metric is Part XIV line 3a (paid during the year). "
                "Part XIV line 3b is secondary (approved for future payment / pipeline)."
            ],
        }
        
        # Check form type
        if re.search(r'Form\s+990\s*\n', text) and not re.search(r'Form\s+990-PF', text):
            metadata['is_990pf'] = False
            metadata['form_type'] = '990'
        
        # Find EIN - look for pattern near "Employer identification number"
        ein_match = re.search(r'Employer identification number\s*\n?\s*(\d{2}[-\s]?\d{7})', text, re.IGNORECASE)
        if ein_match:
            metadata['ein'] = ein_match.group(1).replace('-', '').replace(' ', '')
        else:
            # Also try standalone EIN pattern
            ein_match = re.search(r'\b(\d{2}-\d{7})\b', text[:3000])
            if ein_match:
                metadata['ein'] = ein_match.group(1).replace('-', '')
        
        # Find organization name - multiple strategies
        # Strategy 1: Look for "Name of foundation" followed by the name on next line
        name_match = re.search(r'Name of (?:foundation|organization)\s*\n\s*([A-Za-z][A-Za-z\s&,\.\-\']+?)(?:\s*\n|\s{2,})', text, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            # Clean up - remove trailing address fragments
            name = re.sub(r'\s+\d+\s*$', '', name)  # Remove trailing numbers
            if len(name) > 3 and 'Number and street' not in name:
                metadata['foundation_name'] = name
        
        # Strategy 2: Look for foundation name in ProPublica header (often accurate)
        if not metadata['foundation_name']:
            header_match = re.search(r'PM([A-Za-z][A-Za-z\s&,\.\-\']+(?:Foundation|Fund|Trust|Inc\.?))\s*-\s*Full Filing', text)
            if header_match:
                metadata['foundation_name'] = header_match.group(1).strip()
        
        # Strategy 3: Look for ALL CAPS foundation name near top
        if not metadata['foundation_name']:
            caps_match = re.search(r'\n([A-Z][A-Z\s&,\.]+(?:FOUNDATION|FUND|TRUST))\s*\n', text[:3000])
            if caps_match:
                metadata['foundation_name'] = caps_match.group(1).strip()
        
        # Find tax year
        year_match = re.search(r'(?:calendar year|tax year beginning)\s*(\d{4})', text, re.IGNORECASE)
        if year_match:
            metadata['tax_year'] = year_match.group(1)
        else:
            # Try to find year in form header
            year_match = re.search(r'Form 990-PF\s*\((\d{4})\)', text)
            if year_match:
                metadata['tax_year'] = year_match.group(1)
            else:
                # Look for year near "ending" date
                year_match = re.search(r'ending\s*\d{1,2}[-/]\d{1,2}[-/](\d{4})', text)
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
            # P0.1 fix: Correct operator precedence for AM/PM check
            if re.match(r'^\d{1,2}/\d{1,2}/\d{2}', line) and ('PM' in line or 'AM' in line):
                continue
            if '<PARSED TEXT FOR PAGE:' in line:
                continue
            filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        return cleaned.strip()
    
    def _find_part_xiv_line3_block(self, text: str) -> Optional[str]:
        """
        Find the Part XIV Line 3 block containing both 3a and 3b.
        Returns the entire block from "a Paid during the year" to Part XV.
        
        Patch 1: Anchor to Part XIV first to avoid matching wrong "a Paid" marker.
        """
        # First find Part XIV
        part_xiv = re.search(r'\bPart\s+XIV\b', text, re.IGNORECASE)
        if not part_xiv:
            # Fallback: try Part XV (some forms label it differently)
            part_xiv = re.search(r'\bPart\s+XV\b.*Supplementary', text, re.IGNORECASE)
            if not part_xiv:
                return None
        
        # Now find "a Paid during the year" AFTER Part XIV
        sub_text = text[part_xiv.start():]
        paid_match = re.search(r'\ba\s+Paid during the year\b', sub_text, re.IGNORECASE)
        if not paid_match:
            return None
        
        start = part_xiv.start() + paid_match.start()  # Absolute position
        
        # Find end at Part XV (or next Part after our section)
        remaining = text[start:]
        end_match = re.search(r'\bPart\s+XV\b(?!\s*Supplementary)', remaining, re.IGNORECASE)
        if not end_match:
            # Try Part XVI as fallback
            end_match = re.search(r'\bPart\s+XVI\b', remaining, re.IGNORECASE)
        
        end = start + end_match.start() if end_match else len(text)
        
        return text[start:end]
    
    def _split_paid_future(self, block: str) -> tuple[str, str]:
        """
        Split Part XIV Line 3 block into 3a (paid) and 3b (future) sections.
        Returns (paid_text, future_text).
        """
        # Look for "b Approved for future payment"
        future_match = re.search(r'^\s*b\s+Approved for future payment', block, re.IGNORECASE | re.MULTILINE)
        if not future_match:
            return block, ""
        
        return block[:future_match.start()], block[future_match.start():]
    
    def _parse_reported_total(self, block: str, which: str) -> Optional[int]:
        """Parse the reported total from 3a or 3b section."""
        rx = self.TOTAL_3A_RE if which == "3a" else self.TOTAL_3B_RE
        m = rx.search(block)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                return None
        return None
    
    # Regex for extracting state from address (fallback)
    STATE_FROM_ADDR_RE = re.compile(r',\s*([A-Z]{2})\s+\d{5}(?:-\d{4})?\b')
    
    def _dedup_grants(self, grants: list[Grant]) -> list[Grant]:
        """
        Patch 5: Remove duplicate grants caused by page breaks.
        Keys on (name, amount, state, purpose prefix).
        """
        seen = set()
        deduped = []
        for g in grants:
            key = (
                g.recipient_name.strip().upper(),
                g.amount,
                g.recipient_state,
                (g.purpose or "")[:50].strip().upper()
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(g)
        return deduped
    
    def _tag_gl_relevance(self, grant: Grant) -> None:
        """
        Tag grant as Great Lakes relevant based on grantee state.
        Patch 3: Falls back to parsing recipient_address if state not set.
        """
        st = grant.recipient_state
        
        # Fallback: try to extract state from full address
        if not st and grant.recipient_address:
            m = self.STATE_FROM_ADDR_RE.search(grant.recipient_address.upper())
            if m:
                st = m.group(1)
                grant.recipient_state = st
        
        grant.gl_relevant = st in GL_STATES
    
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
                        # Stop at Total lines (prevents purpose corruption)
                        if re.search(r'\bTotal\b.*\b3[ab]\b', lines[j], re.IGNORECASE):
                            break
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
                            # Stop at Total lines
                            if re.search(r'\bTotal\b.*\b3[ab]\b', lines[j], re.IGNORECASE):
                                break
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
                    # Extract city/state - improved regex for edge cases
                    city_state_match = re.match(r'^([A-Za-z\.\'\-\s]+),\s*([A-Z]{2})\s*\d{0,10}', line)
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
        # City, State ZIP pattern - improved for edge cases (St. Paul, Winston-Salem, etc.)
        if re.match(r'^[A-Za-z\.\'\-\s]+,\s*[A-Z]{2}\s*\d{0,10}', line):
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
            'ProPublica', 'Nonprofit Explorer', 'Full Filing', 'PARSED TEXT',
        ]
        for artifact in artifacts:
            if artifact.lower() in name.lower():
                return False
        
        # Filter lines that look like date/time stamps (ProPublica headers)
        if re.match(r'^\d{1,2}/\d{1,2}/\d{2}', name):
            return False
        
        return True
    
    def _extract_people(self, text: str) -> list[Person]:
        """Extract board members/officers from Part VII."""
        people = []
        
        # Look for Part VII section
        part_vii_match = re.search(
            r'Part\s+VII[A-Z\s\-]*Information About Officers',
            text, re.IGNORECASE
        )
        
        if not part_vii_match:
            # Try alternate pattern
            part_vii_match = re.search(r'Part\s+VII', text, re.IGNORECASE)
            if not part_vii_match:
                return people
        
        # Get section text - Part VII to Part VIII (or next 10000 chars)
        start = part_vii_match.end()
        end_match = re.search(r'Part\s+VIII', text[start:], re.IGNORECASE)
        end = start + end_match.start() if end_match else min(start + 10000, len(text))
        
        section = text[start:end]
        
        # Clean ProPublica artifacts from section
        section = re.sub(r'\d{1,2}/\d{1,2}/\d{2},?\s*\d{1,2}\s*:\s*\d{2}\s*[AP]M.*?ProPublica', '', section, flags=re.IGNORECASE)
        section = re.sub(r'Page\s+\d+\s+of\s+\d+.*?full', '', section, flags=re.IGNORECASE)
        section = re.sub(r'https?://[^\s]+', '', section)
        
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        # Title keywords to identify officer/director lines
        title_keywords = [
            'director', 'trustee', 'president', 'vice president', 'v.p.',
            'secretary', 'treasurer', 'chairman', 'chair', 'ceo', 'cfo', 'coo',
            'chief', 'officer', 'manager', 'executive'
        ]
        
        # Skip patterns (addresses, headers, etc.)
        skip_patterns = [
            r'^[\d,\.\s]+$',  # Pure numbers
            r'^\d+\s+[NSEW]\s+\w+',  # Street addresses
            r'^Chicago|^New York|^Washington|^Los Angeles',  # City names
            r'^IL\s+\d|^NY\s+\d|^CA\s+\d|^DC\s+\d',  # State + ZIP
            r'^\(\d{3}\)',  # Phone numbers
            r'^Name and address',  # Column headers
            r'^\(a\)\s|^\(b\)\s|^\(c\)\s|^\(d\)\s|^\(e\)\s',  # Column labels
            r'^Title,|^Compensation|^Contributions|^Expense',  # Column headers
            r'^hours per week',  # Column header fragment
            r'^Officer/|^Treasurer|^Secretary',  # Partial titles (continuation lines)
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip header/address lines
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                i += 1
                continue
            
            # Check if this line contains a title keyword
            line_lower = line.lower()
            has_title = any(kw in line_lower for kw in title_keywords)
            
            if has_title:
                # Try to split name from title
                for kw in title_keywords:
                    kw_match = re.search(rf'\b({kw})', line, re.IGNORECASE)
                    if kw_match:
                        name_part = line[:kw_match.start()].strip()
                        title_part = line[kw_match.start():].strip()
                        
                        # Clean up name - remove trailing punctuation
                        name_part = re.sub(r'[\s/\-:]+$', '', name_part)
                        
                        # Validate name
                        if name_part and len(name_part) >= 3:
                            # Must start with capital letter (proper name)
                            if re.match(r'^[A-Z][a-z]', name_part):
                                # Check it's not an address or other garbage
                                if not re.search(r'\d{5}|\bStreet\b|\bAve\b|\bRoad\b|\bSuite\b', name_part, re.IGNORECASE):
                                    # Must have at least a first and last name (space in between)
                                    if ' ' in name_part:
                                        # Clean up the title
                                        title_clean = re.sub(r'\s*\(.*?\)', '', title_part)
                                        title_clean = title_clean.strip()
                                        
                                        # Don't add duplicates
                                        if not any(p.name == name_part for p in people):
                                            people.append(Person(
                                                name=name_part,
                                                title=title_clean
                                            ))
                                        break
            
            i += 1
        
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
            - grants_df: DataFrame of grants (3a + 3b combined, with grant_bucket column)
            - people_df: DataFrame of board members
            - foundation_meta: dict of foundation metadata
            - diagnostics: dict of parsing diagnostics including totals
    """
    parser = IRS990PFParser()
    
    try:
        grants_3a, grants_3b, people, metadata = parser.parse_bytes(file_bytes)
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
    
    # Combine 3a and 3b grants (but keep grant_bucket to distinguish)
    all_grants = grants_3a + grants_3b
    
    # Build grants DataFrame
    if all_grants:
        grants_data = []
        for g in all_grants:
            grants_data.append({
                'foundation_name': metadata.get('foundation_name', ''),
                'foundation_ein': metadata.get('ein', ''),
                'tax_year': tax_year,
                'grantee_name': g.recipient_name,
                'grantee_city': g.recipient_city,
                'grantee_state': g.recipient_state,
                'grantee_address_raw': g.recipient_address,  # Patch 6: for audit
                'grant_amount': float(g.amount),
                'grant_purpose_raw': g.purpose,
                'grant_bucket': g.grant_bucket,
                'gl_relevant': g.gl_relevant,
                'source_file': filename,
            })
        grants_df = pd.DataFrame(grants_data)
    else:
        grants_df = pd.DataFrame(columns=[
            'foundation_name', 'foundation_ein', 'tax_year',
            'grantee_name', 'grantee_city', 'grantee_state', 'grantee_address_raw',
            'grant_amount', 'grant_purpose_raw', 'grant_bucket',
            'gl_relevant', 'source_file'
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
    
    # Build diagnostics with totals
    diagnostics = {
        'org_name': metadata.get('foundation_name', ''),
        'is_990pf': metadata.get('is_990pf', True),
        'form_type': metadata.get('form_type', '990-PF'),
        'reported_total_3a': metadata.get('reported_total_3a'),
        'reported_total_3b': metadata.get('reported_total_3b'),
        'computed_total_3a': metadata.get('computed_total_3a'),
        'computed_total_3b': metadata.get('computed_total_3b'),
        'diff_3a': metadata.get('diff_3a'),
        'diff_3b': metadata.get('diff_3b'),
        'diff_pct_3a': metadata.get('diff_pct_3a'),
        'diff_pct_3b': metadata.get('diff_pct_3b'),
        'total_mismatch_3a': metadata.get('total_mismatch_3a', False),
        'total_mismatch_3b': metadata.get('total_mismatch_3b', False),
        'total_mismatch_3a_strict': metadata.get('total_mismatch_3a_strict', False),
        'total_mismatch_3b_strict': metadata.get('total_mismatch_3b_strict', False),
        'grants_3a_count': len(grants_3a),
        'grants_3b_count': len(grants_3b),
        'gl_relevant_count': sum(1 for g in all_grants if g.gl_relevant),
        'gl_relevant_amount': sum(g.amount for g in all_grants if g.gl_relevant),
        'notes': metadata.get('notes', []),
    }
    
    return {
        'grants_df': grants_df,
        'people_df': people_df,
        'foundation_meta': {
            'foundation_name': metadata.get('foundation_name', ''),
            'foundation_ein': metadata.get('ein', ''),
            'tax_year': tax_year,
            'source_file': filename,
        },
        'diagnostics': diagnostics
    }


# =============================================================================
# CLI TEST HARNESS
# =============================================================================

def test_parser(pdf_path: str):
    """Run parser and show results (for command-line testing)."""
    print("=" * 80)
    print("990-PF PARSER v2.3 TEST")
    print("=" * 80)
    print(f"\nInput: {pdf_path}\n")
    
    parser = IRS990PFParser()
    grants_3a, grants_3b, people, metadata = parser.parse_file(pdf_path)
    
    print(f"Foundation: {metadata.get('foundation_name', 'Unknown')}")
    print(f"EIN: {metadata.get('ein', 'Unknown')}")
    print(f"Tax Year: {metadata.get('tax_year', 'Unknown')}")
    
    print("\n" + "-" * 80)
    print("3a GRANTS (Paid During Year) - PRIMARY")
    print("-" * 80)
    print(f"Count: {len(grants_3a)}")
    computed_3a = metadata.get('computed_total_3a', 0)
    reported_3a = metadata.get('reported_total_3a')
    print(f"Computed total: ${computed_3a:,}")
    if reported_3a:
        print(f"Reported total: ${reported_3a:,}")
        diff = metadata.get('diff_3a', 0)
        diff_pct = metadata.get('diff_pct_3a', 0)
        if diff and diff > 1:
            print(f"Difference: ${diff:,} ({diff_pct:.2f}%)")
            status = "✓ MATCH (lenient)" if diff <= 100 else f"⚠ MISMATCH"
        else:
            status = "✓ EXACT MATCH"
        print(f"Status: {status}")
    
    # Great Lakes stats
    gl_grants = [g for g in grants_3a if g.gl_relevant]
    print(f"\nGreat Lakes relevant: {len(gl_grants)} grants (${sum(g.amount for g in gl_grants):,})")
    
    print("\n" + "-" * 80)
    print("3b GRANTS (Approved for Future) - SECONDARY")
    print("-" * 80)
    print(f"Count: {len(grants_3b)}")
    computed_3b = metadata.get('computed_total_3b', 0)
    reported_3b = metadata.get('reported_total_3b')
    print(f"Computed total: ${computed_3b:,}")
    if reported_3b:
        print(f"Reported total: ${reported_3b:,}")
        diff = metadata.get('diff_3b', 0)
        if diff:
            print(f"Difference: ${diff:,}")
    
    print("\n" + "-" * 80)
    print("BOARD MEMBERS")
    print("-" * 80)
    print(f"Count: {len(people)}")
    for p in people[:10]:
        print(f"  - {p.name}: {p.title}")
    if len(people) > 10:
        print(f"  ... and {len(people) - 10} more")
    
    print("\n" + "-" * 80)
    print("SAMPLE 3a GRANTS (first 10)")
    print("-" * 80)
    for i, g in enumerate(grants_3a[:10], 1):
        gl_tag = " [GL]" if g.gl_relevant else ""
        print(f"{i}. {g.recipient_name}{gl_tag}")
        loc = f"{g.recipient_city}, {g.recipient_state}" if g.recipient_city else g.recipient_state or "(no location)"
        print(f"   {loc} | ${g.amount:,}")
    
    if grants_3b:
        print("\n" + "-" * 80)
        print("SAMPLE 3b GRANTS (first 5)")
        print("-" * 80)
        for i, g in enumerate(grants_3b[:5], 1):
            gl_tag = " [GL]" if g.gl_relevant else ""
            print(f"{i}. {g.recipient_name}{gl_tag}")
            loc = f"{g.recipient_city}, {g.recipient_state}" if g.recipient_city else g.recipient_state or "(no location)"
            print(f"   {loc} | ${g.amount:,}")
    
    return grants_3a, grants_3b, people, metadata


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python irs990_parser.py <pdf_path>")
    else:
        test_parser(sys.argv[1])
