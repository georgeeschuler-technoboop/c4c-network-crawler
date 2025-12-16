"""
c4c_utils/board_extractor.py

Extract board members, officers, and trustees from IRS Form 990-PF Part VII.
Handles multiple PDF formats including:
- Erb-style: ALL CAPS names with titles on separate lines
- Joyce-style: Mixed case names with inline titles

Usage:
    from c4c_utils.board_extractor import BoardExtractor
    
    extractor = BoardExtractor(pdf_path)
    members = extractor.extract()
    
    for member in members:
        print(f"{member.name} - {member.title} - ${member.compensation:,.0f}")
"""

import pdfplumber
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoardMember:
    """Represents a board member, officer, or trustee from Form 990-PF Part VII"""
    name: str
    title: str
    compensation: Optional[float] = None
    hours_per_week: Optional[float] = None
    benefits: Optional[float] = None
    expense_allowance: Optional[float] = None
    address: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'title': self.title,
            'compensation': self.compensation,
            'hours_per_week': self.hours_per_week,
            'benefits': self.benefits,
            'expense_allowance': self.expense_allowance,
            'address': self.address
        }
    
    def __repr__(self) -> str:
        comp = f"${self.compensation:,.0f}" if self.compensation else "N/A"
        return f"BoardMember({self.name}, {self.title}, {comp})"


class BoardExtractor:
    """
    Extract board members from IRS Form 990-PF Part VII.
    
    Handles two main PDF format styles:
    1. Erb-style (ALL CAPS names, titles on separate lines)
    2. Joyce-style (mixed case names, inline titles with compensation)
    
    Example:
        extractor = BoardExtractor("/path/to/990pf.pdf")
        members = extractor.extract()
        print(extractor.summary())
    """
    
    # Joyce-style title patterns (appear on same line as name, before numbers)
    # ORDER MATTERS - more specific patterns first to avoid partial matches
    INLINE_TITLE_PATTERNS = [
        # Specific compound titles first
        r'Executive V\.?P\.?/?Chief Strategy Officer',
        r'Executive V\.?P\.?/Chief Strategy',
        r'Chief Investment Officer/?Treasurer',
        r'Chief Investment Officer',
        r'Chief Executive Officer',
        r'Chief Operating Officer',
        r'Chief Financial Officer',
        r'President/?Director',
        r'Executive Vice President',
        r'V\.?P\.? of \w+(?:\s*&\s*\w+)*',  # V.P. of Strategy & Programs
        r'Vice President',
        # "Program Director" BEFORE generic "Director"
        r'Program Director',
        r'MG\.?\s*DIR\.?\s*COMMUN\.?',  # Managing Director Communications
        r'Managing Director',
        r'General Counsel',
        # Director with qualifier
        r'Director\s*\([^)]+\)',  # Director (thru 4/2023)
        # Simple titles last
        r'President',
        r'CEO',
        r'COO',
        r'CFO',
        r'Treasurer',
        r'Secretary',
        r'Director',
        r'Trustee',
    ]
    
    # Erb-style title keywords (on separate line, usually ALL CAPS)
    TITLE_KEYWORDS = [
        'CHAIR', 'CHAIRMAN', 'CHAIRWOMAN', 'CHAIRPERSON',
        'PRESIDENT', 'VICE PRESIDENT', 'VP',
        'CEO', 'COO', 'CFO', 'CIO',
        'CHIEF EXECUTIVE', 'CHIEF OPERATING', 'CHIEF FINANCIAL',
        'SECRETARY', 'TREASURER', 'ASSISTANT TREASURER',
        'TRUSTEE', 'DIRECTOR', 'BOARD MEMBER',
        'EXECUTIVE DIRECTOR', 'MANAGING DIRECTOR',
        'GENERAL COUNSEL',
    ]
    
    def __init__(self, pdf_path: str):
        """
        Initialize the extractor.
        
        Args:
            pdf_path: Path to the 990-PF PDF file
        """
        self.pdf_path = pdf_path
        self.members: List[BoardMember] = []
        self._extraction_method: Optional[str] = None
        
    def extract(self) -> List[BoardMember]:
        """
        Extract board members from the PDF.
        
        Returns:
            List of BoardMember objects
        """
        logger.info(f"Extracting board members from: {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            # Find Part VII pages
            part_vii_text = self._find_part_vii(pdf)
            
            if not part_vii_text:
                logger.warning("Could not find Part VII section")
                return []
            
            # Try both extraction methods
            members_joyce = self._extract_joyce_style(part_vii_text)
            members_erb = self._extract_erb_style(part_vii_text)
            
            # Use whichever found more members
            if len(members_joyce) >= len(members_erb):
                self.members = members_joyce
                self._extraction_method = "joyce"
                logger.info(f"Used Joyce-style extraction: {len(members_joyce)} members")
            else:
                self.members = members_erb
                self._extraction_method = "erb"
                logger.info(f"Used Erb-style extraction: {len(members_erb)} members")
            
            return self.members
    
    def _find_part_vii(self, pdf) -> str:
        """Find and extract Part VII text from PDF."""
        part_vii_pages = []
        in_part_vii = False
        
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            
            # Check for Part VII start
            if "Part VII" in text and ("Officers" in text or "Directors" in text or "Trustees" in text):
                in_part_vii = True
            
            # Check for Part VIII (end of Part VII)
            if in_part_vii and "Part VIII" in text:
                part_vii_pages.append(text)
                break
            
            if in_part_vii:
                part_vii_pages.append(text)
                
            # Safety limit - Part VII usually within first 15 pages
            if page_num > 15 and not in_part_vii:
                break
        
        return "\n".join(part_vii_pages)
    
    def _extract_joyce_style(self, text: str) -> List[BoardMember]:
        """
        Extract from Joyce-style format:
        Name Title Compensation Benefits Expenses
        Hours
        Address lines
        """
        members = []
        lines = text.split('\n')
        
        # Build regex for inline titles
        title_pattern = '|'.join(self.INLINE_TITLE_PATTERNS)
        
        # Pattern: Name + Title + Numbers (compensation data)
        # Using non-greedy name matching (.+?) so title patterns get priority
        name_title_pattern = re.compile(
            r'^(.+?)\s+'  # Name (non-greedy)
            r'(' + title_pattern + r')\s+'  # Title
            r'([\d,]+)\s+'  # Compensation
            r'([\d,]+)\s+'  # Benefits
            r'([\d,]+)',    # Expenses
            re.IGNORECASE
        )
        
        # Handle wrapped titles (title continues on next line)
        partial_title_pattern = re.compile(
            r'^(.+?)\s+'  # Name (non-greedy)
            r'(Chief\s+\w+|Executive\s+V\.?P\.?/?(?:Chief\s+)?(?:Strategy)?)\s+'
            r'([\d,]+)\s+'  # Compensation
            r'([\d,]+)\s+'  # Benefits  
            r'([\d,]+)',    # Expenses
            re.IGNORECASE
        )
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and headers
            if not line or 'Page' in line or 'https://' in line:
                i += 1
                continue
            
            # Try full title match
            match = name_title_pattern.match(line)
            if match:
                name = match.group(1).strip()
                title = match.group(2).strip()
                comp = self._parse_number(match.group(3))
                benefits = self._parse_number(match.group(4))
                expenses = self._parse_number(match.group(5))
                
                # Validate name
                if not self._is_valid_name(name):
                    i += 1
                    continue
                
                # Get hours from next line if available
                hours = self._get_hours(lines, i + 1)
                
                # Get address from subsequent lines
                address = self._extract_address(lines, i + 2)
                
                members.append(BoardMember(
                    name=name,
                    title=title,
                    compensation=comp,
                    benefits=benefits,
                    expense_allowance=expenses,
                    hours_per_week=hours,
                    address=address
                ))
                i += 1
                continue
            
            # Try partial title match (title wraps to next line)
            match = partial_title_pattern.match(line)
            if match:
                name = match.group(1).strip()
                partial_title = match.group(2).strip()
                comp = self._parse_number(match.group(3))
                benefits = self._parse_number(match.group(4))
                expenses = self._parse_number(match.group(5))
                
                if not self._is_valid_name(name):
                    i += 1
                    continue
                
                # Look for title continuation
                full_title = self._complete_wrapped_title(partial_title, lines, i + 1)
                hours = self._get_hours(lines, i + 1)
                
                members.append(BoardMember(
                    name=name,
                    title=full_title,
                    compensation=comp,
                    benefits=benefits,
                    expense_allowance=expenses,
                    hours_per_week=hours
                ))
            
            i += 1
        
        return members
    
    def _extract_erb_style(self, text: str) -> List[BoardMember]:
        """
        Extract from Erb-style format:
        ALL CAPS NAME
        Title Line
        Address
        Hours | Compensation | etc
        """
        members = []
        lines = text.split('\n')
        
        # Pattern for ALL CAPS names
        all_caps_name = re.compile(r'^([A-Z][A-Z\s\.]+[A-Z])\s*$')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            match = all_caps_name.match(line)
            if match:
                potential_name = match.group(1).strip()
                
                # Skip headers/labels
                if any(skip in potential_name for skip in ['TOTAL', 'FORM', 'PART', 'PAGE', 'NONE']):
                    i += 1
                    continue
                
                # Look for title in next few lines
                title = None
                for j in range(i + 1, min(i + 4, len(lines))):
                    check_line = lines[j].strip().upper()
                    for keyword in self.TITLE_KEYWORDS:
                        if keyword in check_line:
                            title = lines[j].strip()
                            break
                    if title:
                        break
                
                if title:
                    comp_data = self._find_compensation_data(lines, i, i + 6)
                    
                    members.append(BoardMember(
                        name=potential_name.title(),  # Convert to title case
                        title=title,
                        compensation=comp_data.get('compensation'),
                        hours_per_week=comp_data.get('hours'),
                        benefits=comp_data.get('benefits')
                    ))
            
            i += 1
        
        return members
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate that a string looks like a person's name."""
        name_parts = name.split()
        if len(name_parts) < 2 or len(name_parts) > 4:
            return False
        if any(c.isdigit() for c in name):
            return False
        if not all(part[0].isupper() for part in name_parts if part):
            return False
        return True
    
    def _get_hours(self, lines: List[str], start_idx: int) -> Optional[float]:
        """Extract hours from lines following a name/title."""
        if start_idx < len(lines):
            next_line = lines[start_idx].strip()
            hours_match = re.match(r'^(\d+\.?\d*)\s*$', next_line)
            if hours_match:
                return float(hours_match.group(1))
        return None
    
    def _complete_wrapped_title(self, partial: str, lines: List[str], idx: int) -> str:
        """Complete a title that wraps to the next line."""
        full_title = partial
        if idx < len(lines):
            next_line = lines[idx].strip()
            if not re.match(r'^\d+\.?\d*\s*$', next_line) and not re.match(r'^\d+\s+\w', next_line):
                continuations = ['Officer', 'Treasurer', 'Director', 'Secretary', 'President']
                if any(kw.lower() in next_line.lower() for kw in continuations):
                    full_title = f"{partial} {next_line}"
                    full_title = re.sub(r'/\s*/+', '/', full_title)
        return full_title
    
    def _extract_address(self, lines: List[str], start_idx: int) -> Optional[str]:
        """Extract address from subsequent lines."""
        address_parts = []
        for i in range(start_idx, min(start_idx + 3, len(lines))):
            line = lines[i].strip()
            if re.match(r'^\d+\s+\w', line) or re.match(r'^[A-Z][a-z]+,?\s*[A-Z]{2}', line):
                address_parts.append(line)
            elif line and not re.match(r'^[\d,]+\s*$', line):
                if ',' in line or re.search(r'\d{5}', line):
                    address_parts.append(line)
                else:
                    break
        return '\n'.join(address_parts) if address_parts else None
    
    def _find_compensation_data(self, lines: List[str], start: int, end: int) -> Dict:
        """Find compensation numbers in a range of lines."""
        data = {}
        for i in range(start, min(end, len(lines))):
            line = lines[i].strip()
            numbers = re.findall(r'[\d,]+', line)
            if len(numbers) >= 2:
                nums = [self._parse_number(n) for n in numbers]
                for n in nums:
                    if n and n < 100:
                        data['hours'] = n
                    elif n and n > 1000:
                        if 'compensation' not in data:
                            data['compensation'] = n
                        elif 'benefits' not in data:
                            data['benefits'] = n
                break
        return data
    
    def _parse_number(self, s: str) -> Optional[float]:
        """Parse number string, handling commas."""
        if not s:
            return None
        try:
            return float(s.replace(',', ''))
        except ValueError:
            return None
    
    def to_json(self) -> str:
        """Export members to JSON string."""
        return json.dumps([m.to_dict() for m in self.members], indent=2)
    
    def to_dicts(self) -> List[Dict]:
        """Export members as list of dictionaries."""
        return [m.to_dict() for m in self.members]
    
    def summary(self) -> str:
        """Generate a text summary of extracted members."""
        lines = [
            f"\n{'='*60}",
            f"BOARD MEMBERS EXTRACTED: {len(self.members)}",
            f"Extraction method: {self._extraction_method}",
            '='*60
        ]
        for m in self.members:
            comp_str = f"${m.compensation:,.0f}" if m.compensation else "N/A"
            lines.append(f"\n{m.name}")
            lines.append(f"  Title: {m.title}")
            lines.append(f"  Compensation: {comp_str}")
            if m.hours_per_week:
                lines.append(f"  Hours/Week: {m.hours_per_week}")
        return '\n'.join(lines)


# Convenience function for quick extraction
def extract_board_members(pdf_path: str) -> List[Dict]:
    """
    Quick extraction of board members from a 990-PF PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries with member data
    """
    extractor = BoardExtractor(pdf_path)
    extractor.extract()
    return extractor.to_dicts()


if __name__ == "__main__":
    # Test with a PDF if run directly
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "/mnt/user-data/uploads/TEST_Joyce_Foundation_-_Full_Filing_-_Nonprofit_Explorer_-_ProPublica_990-PF.pdf"
    
    extractor = BoardExtractor(pdf_path)
    members = extractor.extract()
    print(extractor.summary())
