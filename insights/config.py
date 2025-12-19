"""
Insight Engine — Configuration

Single source of truth for all thresholds, limits, and settings.
Used by: UI, report generator, exports.

Version: 1.0.0
"""

# =============================================================================
# Evidence Display Limits
# =============================================================================
TOP_N_SHARED_GRANTEES = 5
TOP_N_FUNDER_PAIRS = 5
TOP_N_BROKERS = 5
TOP_N_BRIDGES = 5

# =============================================================================
# Broker Detection (Opportunity)
# =============================================================================
# Hidden Broker = high betweenness + low visibility
# IMPORTANT: Percentiles computed WITHIN node_type (funder, grantee, person)
# to avoid cross-type threshold artifacts
#
# Betweenness: top X percentile (within node_type)
BROKER_BETWEENNESS_PERCENTILE = 95
# Visibility (degree): bottom X percentile (within node_type)
BROKER_VISIBILITY_PERCENTILE = 40
# Optional: use weighted_degree (sum of edge weights) when weights are stable
BROKER_USE_WEIGHTED_DEGREE = False  # Set True when edge weights are reliable

# =============================================================================
# Bridge Detection (Risk)
# =============================================================================
# Single-Point Bridges are articulation points (no threshold needed)
# Computed via nx.articulation_points()
# No configuration required — structural property of the graph

# =============================================================================
# Funding Concentration Thresholds
# =============================================================================
CONCENTRATION_HIGH_THRESHOLD = 70    # top5_share_pct > 70% = "highly concentrated"
CONCENTRATION_MODERATE_THRESHOLD = 50  # > 50% = "moderate concentration"

# =============================================================================
# Overlap Thresholds
# =============================================================================
MULTI_FUNDER_HIGH_THRESHOLD = 30   # > 30% multi-funder grantees = "strong overlap"
MULTI_FUNDER_LOW_THRESHOLD = 10    # < 10% = "weak overlap"

# =============================================================================
# Network Health Score Weights (v1)
# =============================================================================
HEALTH_WEIGHT_MULTI_FUNDER = 0.30
HEALTH_WEIGHT_CONNECTIVITY = 0.25
HEALTH_WEIGHT_CONCENTRATION = 0.25
HEALTH_WEIGHT_GOVERNANCE = 0.20

# Health score labels
HEALTH_LABEL_THRESHOLDS = {
    80: "Strong",
    60: "Moderate", 
    40: "Developing",
    0: "Fragile"
}

# =============================================================================
# Governance Confidence
# =============================================================================
GOVERNANCE_COVERAGE_THRESHOLD = 0.30  # Below 30% coverage = "low confidence"

# =============================================================================
# Confidence Level Definitions
# =============================================================================
# high: metric coverage ≥ threshold AND signal detected
# medium: metric present but partial or borderline
# low: metric present but below coverage threshold
# unavailable: metric missing or data absent

CONFIDENCE_LEVELS = ["high", "medium", "low", "unavailable"]

# =============================================================================
# Graceful Degradation — Fallback Copy
# =============================================================================
FALLBACK_COPY = {
    "governance": "Governance data unavailable or incomplete; governance connectivity cannot be assessed reliably.",
    "brokers": "No hidden brokers detected under current thresholds.",
    "bridges": "No single-point bridges detected; the network shows redundancy at this level.",
    "overlap": "No multi-funder grantees detected; coordination anchors are absent in this dataset.",
    "portfolio_twins": "No funder pairs with meaningful portfolio overlap detected.",
    "general": "Insufficient data to assess this metric.",
}

# =============================================================================
# Report Defaults
# =============================================================================
RECOMMENDATIONS_COUNT = 4  # Always exactly 4 recommendations

STRATEGIC_INTRO = (
    "The following recommendations focus on strengthening coordination and resilience "
    "based on observed network structure. They are intended as starting points, not prescriptions."
)

# =============================================================================
# App Metadata
# =============================================================================
APP_VERSION = "1.0.0"
APP_NAME = "Insight Engine"
GENERATOR_CREDIT = "C4C Network Insight Engine"

# =============================================================================
# Reading Contract
# =============================================================================
READING_CONTRACT = {
    "skim": {
        "time": "5 minutes",
        "sections": ["System Summary", "Network Health", "Strategic Recommendations"]
    },
    "read": {
        "time": "20 minutes", 
        "sections": ["All signal sections with selected evidence"]
    },
    "explore": {
        "time": "Optional",
        "sections": ["Grant themes", "Data tables", "Downloads"]
    }
}
