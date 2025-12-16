# =============================================================================
# ADD TO c4c_utils/__init__.py
# =============================================================================

# Add this import line with your other imports:
from . import irs_return_qa

# Your full __init__.py should look like:
"""
# C4C Utilities Package
# Network analysis utilities for Connecting for Change

from . import irs990_parser
from . import network_export
from . import board_extractor
from . import irs_return_qa

__version__ = "0.1.0"
"""


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# Option 1: Compute confidence after parsing
from c4c_utils.irs_return_qa import compute_confidence

result = parse_990_pdf(file_bytes, filename)
conf = compute_confidence(result["diagnostics"])
result["diagnostics"]["confidence_score"] = conf.score
result["diagnostics"]["confidence_grade"] = conf.grade
result["diagnostics"]["confidence_reasons"] = conf.reasons


# Option 2: One-liner to enrich diagnostics
from c4c_utils.irs_return_qa import enrich_diagnostics_with_confidence

diagnostics = enrich_diagnostics_with_confidence(result["diagnostics"])


# Option 3: Render QA panel in Streamlit
import streamlit as st
from c4c_utils.irs_return_qa import render_return_qa_panel

render_return_qa_panel(st, result["foundation_meta"], result["diagnostics"])
