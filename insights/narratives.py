"""
Insight Engine — Narratives Module

Rule-based interpretation logic for generating signal narratives.
Each function returns a SignalInterpretation object with:
- signal: what is observed (factual)
- interpretation: what it means (analytical)  
- why_it_matters: opportunity or risk (implication)
- evidence: top examples, metrics, tables
- confidence: high, medium, low, unavailable

VERSION HISTORY:
----------------
v1.0.0 (2025-12-19): Initial release
- SignalInterpretation dataclass
- Interpretation functions for all signal sections
- generate_system_summary() with headline + positives/gaps
- generate_recommendations() always returns exactly 4

v1.0.1 (2025-12-19): Fixed recommendations bug
- Default recommendations now have unique triggers (default_info_sharing, 
  default_thematic, default_visibility, default_evolution)
- Removed duplicate trigger check that was blocking defaults
- Guarantees exactly 4 recommendations in all cases
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import config as cfg


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SignalInterpretation:
    """Structured interpretation for a single signal section."""
    signal: str
    interpretation: str
    why_it_matters: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "high"  # high, medium, low, unavailable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "interpretation": self.interpretation,
            "why_it_matters": self.why_it_matters,
            "evidence": self.evidence,
            "confidence": self.confidence
        }


@dataclass 
class SystemSummary:
    """Structured system summary for the report header."""
    headline: str
    summary_paragraph: str
    positives: List[str]  # What's working (2 items)
    gaps: List[str]       # What's missing (2 items)
    why_it_matters: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "summary_paragraph": self.summary_paragraph,
            "positives": self.positives,
            "gaps": self.gaps,
            "why_it_matters": self.why_it_matters
        }


@dataclass
class Recommendation:
    """A single strategic recommendation."""
    title: str
    text: str
    trigger: str  # Which signal triggered this recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "trigger": self.trigger
        }


# =============================================================================
# Interpretation Functions
# =============================================================================

def interpret_funding_concentration(
    top5_share_pct: float,
    total_funding: float,
    funder_count: int,
    grantee_count: int
) -> SignalInterpretation:
    """
    Interpret funding concentration based on top 5 funder share.
    
    Thresholds:
    - > 70%: highly concentrated
    - > 50%: moderate concentration  
    - <= 50%: distributed
    """
    # Determine interpretation based on thresholds
    if top5_share_pct > cfg.CONCENTRATION_HIGH_THRESHOLD:
        signal = f"The top 5 funders account for {top5_share_pct:.0f}% of total funding."
        interpretation = "Funding is highly concentrated among a small number of funders."
        why_it_matters = "This creates dependency risk if major funders shift priorities or exit the field."
        confidence = "high"
    elif top5_share_pct > cfg.CONCENTRATION_MODERATE_THRESHOLD:
        signal = f"The top 5 funders account for {top5_share_pct:.0f}% of total funding."
        interpretation = "Funding shows moderate concentration."
        why_it_matters = "The system has some diversification but remains sensitive to top funder decisions."
        confidence = "high"
    else:
        signal = f"The top 5 funders account for {top5_share_pct:.0f}% of total funding."
        interpretation = "Funding is relatively distributed across funders."
        why_it_matters = "This diversification provides resilience against individual funder changes."
        confidence = "high"
    
    evidence = {
        "total_funding": total_funding,
        "funder_count": funder_count,
        "grantee_count": grantee_count,
        "top5_share_pct": top5_share_pct
    }
    
    return SignalInterpretation(
        signal=signal,
        interpretation=interpretation,
        why_it_matters=why_it_matters,
        evidence=evidence,
        confidence=confidence
    )


def interpret_funder_overlap(
    multi_funder_pct: float,
    multi_funder_count: int,
    total_grantees: int,
    top_shared_grantees: List[Dict[str, Any]]
) -> SignalInterpretation:
    """
    Interpret funder overlap based on multi-funder grantee percentage.
    
    Thresholds:
    - > 30%: strong overlap (coordination anchors present)
    - > 10%: moderate overlap
    - <= 10%: weak overlap (siloed funding)
    """
    if multi_funder_pct > cfg.MULTI_FUNDER_HIGH_THRESHOLD:
        signal = f"{multi_funder_count} grantees ({multi_funder_pct:.0f}%) receive funding from multiple funders."
        interpretation = "Strong funder overlap suggests natural coordination anchors exist."
        why_it_matters = "These shared grantees can serve as bridges for funder alignment without new infrastructure."
        opportunity = "Convene top shared grantees to identify coordination opportunities across their funder relationships."
        confidence = "high"
    elif multi_funder_pct > cfg.MULTI_FUNDER_LOW_THRESHOLD:
        signal = f"{multi_funder_count} grantees ({multi_funder_pct:.0f}%) receive funding from multiple funders."
        interpretation = "Moderate funder overlap indicates some natural connection points."
        why_it_matters = "Coordination potential exists but may require intentional cultivation."
        opportunity = "Identify and strengthen relationships with existing multi-funder grantees."
        confidence = "high"
    elif multi_funder_count == 0:
        signal = cfg.FALLBACK_COPY["overlap"]
        interpretation = "Funders operate in silos with no shared grantees."
        why_it_matters = "Coordination will require building new bridges; no natural anchors exist."
        opportunity = "Consider joint funding initiatives to create shared interests."
        confidence = "low"
    else:
        signal = f"Only {multi_funder_count} grantees ({multi_funder_pct:.0f}%) receive funding from multiple funders."
        interpretation = "Weak funder overlap suggests largely siloed funding portfolios."
        why_it_matters = "Limited natural coordination points may require intentional bridge-building."
        opportunity = "Prioritize the few existing multi-funder grantees as coordination seeds."
        confidence = "medium"
    
    evidence = {
        "multi_funder_count": multi_funder_count,
        "multi_funder_pct": multi_funder_pct,
        "total_grantees": total_grantees,
        "top_shared_grantees": top_shared_grantees[:cfg.TOP_N_SHARED_GRANTEES]
    }
    
    return SignalInterpretation(
        signal=signal,
        interpretation=interpretation,
        why_it_matters=opportunity,  # Using opportunity as why_it_matters for this section
        evidence=evidence,
        confidence=confidence
    )


def interpret_portfolio_twins(
    top_funder_pairs: List[Dict[str, Any]],
    max_overlap_pct: float
) -> SignalInterpretation:
    """
    Interpret portfolio similarity between funder pairs.
    """
    if not top_funder_pairs:
        return SignalInterpretation(
            signal=cfg.FALLBACK_COPY["portfolio_twins"],
            interpretation="Funders have largely distinct portfolios with minimal grantee overlap.",
            why_it_matters="This suggests diverse funding strategies but limited coordination potential through shared investments.",
            evidence={"top_funder_pairs": []},
            confidence="low"
        )
    
    top_pair = top_funder_pairs[0]
    pair_count = len(top_funder_pairs)
    
    if max_overlap_pct > 50:
        signal = f"{pair_count} funder pairs share significant portfolio overlap (up to {max_overlap_pct:.0f}%)."
        interpretation = "Some funders have highly aligned investment strategies."
        why_it_matters = "These 'portfolio twins' are natural partners for coordinated strategy or joint initiatives."
    elif max_overlap_pct > 20:
        signal = f"{pair_count} funder pairs share moderate portfolio overlap (up to {max_overlap_pct:.0f}%)."
        interpretation = "Some funders have partially aligned portfolios."
        why_it_matters = "These pairs may benefit from information sharing about shared grantees."
    else:
        signal = f"Even the closest funder pairs share limited overlap ({max_overlap_pct:.0f}% at most)."
        interpretation = "Funders maintain largely distinct portfolios."
        why_it_matters = "Coordination will need to focus on complementarity rather than overlap."
    
    evidence = {
        "top_funder_pairs": top_funder_pairs[:cfg.TOP_N_FUNDER_PAIRS],
        "max_overlap_pct": max_overlap_pct
    }
    
    return SignalInterpretation(
        signal=signal,
        interpretation=interpretation,
        why_it_matters=why_it_matters,
        evidence=evidence,
        confidence="high" if top_funder_pairs else "low"
    )


def interpret_governance(
    governance_data_available: bool,
    shared_board_count: int,
    pct_with_interlocks: float,
    governance_coverage_pct: float = 1.0,
    top_board_connectors: List[Dict[str, Any]] = None
) -> SignalInterpretation:
    """
    Interpret governance connectivity based on board interlocks.
    """
    if not governance_data_available:
        return SignalInterpretation(
            signal=cfg.FALLBACK_COPY["governance"],
            interpretation="Board and governance relationships could not be analyzed.",
            why_it_matters="If governance data becomes available, this section can reveal hidden coordination channels.",
            evidence={},
            confidence="unavailable"
        )
    
    # Check confidence based on coverage
    if governance_coverage_pct < cfg.GOVERNANCE_COVERAGE_THRESHOLD:
        confidence = "low"
        coverage_note = f" (based on {governance_coverage_pct*100:.0f}% coverage)"
    else:
        confidence = "high"
        coverage_note = ""
    
    if shared_board_count > 5 and pct_with_interlocks > 30:
        signal = f"{shared_board_count} individuals serve on multiple boards; {pct_with_interlocks:.0f}% of funders share board connections{coverage_note}."
        interpretation = "Strong governance connectivity suggests informal coordination channels likely exist."
        why_it_matters = "Board relationships can facilitate trust and information flow outside formal grant processes."
        opportunity = "Map key multi-board individuals to understand existing influence pathways."
    elif shared_board_count > 0:
        signal = f"{shared_board_count} individuals serve on multiple boards; {pct_with_interlocks:.0f}% of funders share board connections{coverage_note}."
        interpretation = "Limited governance connectivity indicates funders operate relatively independently."
        why_it_matters = "Coordination may need to rely on grantee connections rather than board relationships."
        opportunity = "Consider whether deeper governance mapping would reveal hidden connections."
    else:
        signal = f"No shared board members detected{coverage_note}."
        interpretation = "Funders appear to operate with fully independent governance."
        why_it_matters = "Coordination must be built through other channels; no governance shortcuts exist."
        opportunity = "Focus coordination efforts on grantee-level connections instead."
    
    evidence = {
        "shared_board_count": shared_board_count,
        "pct_with_interlocks": pct_with_interlocks,
        "governance_coverage_pct": governance_coverage_pct,
        "top_board_connectors": top_board_connectors or []
    }
    
    return SignalInterpretation(
        signal=signal,
        interpretation=interpretation,
        why_it_matters=opportunity,
        evidence=evidence,
        confidence=confidence
    )


def interpret_hidden_brokers(
    broker_count: int,
    top_brokers: List[Dict[str, Any]]
) -> SignalInterpretation:
    """
    Interpret hidden brokers (high betweenness, low visibility).
    
    Framing: OPPORTUNITY lens
    """
    if broker_count == 0:
        return SignalInterpretation(
            signal=cfg.FALLBACK_COPY["brokers"],
            interpretation="Either the network lacks structurally critical low-visibility nodes, or thresholds need adjustment.",
            why_it_matters="This is neutral; coordination may already flow through visible channels.",
            evidence={"broker_count": 0, "top_brokers": []},
            confidence="medium"
        )
    
    signal = f"{broker_count} hidden broker{'s' if broker_count != 1 else ''} identified."
    interpretation = "These organizations sit on many coordination pathways but are not widely recognized as connectors."
    why_it_matters = "Engaging these brokers could accelerate coordination without requiring new infrastructure."
    
    evidence = {
        "broker_count": broker_count,
        "top_brokers": top_brokers[:cfg.TOP_N_BROKERS]
    }
    
    return SignalInterpretation(
        signal=signal,
        interpretation=interpretation,
        why_it_matters=why_it_matters,
        evidence=evidence,
        confidence="high"
    )


def interpret_single_point_bridges(
    bridge_count: int,
    top_bridges: List[Dict[str, Any]]
) -> SignalInterpretation:
    """
    Interpret single-point bridges (articulation points).
    
    Framing: RISK lens
    """
    if bridge_count == 0:
        return SignalInterpretation(
            signal=cfg.FALLBACK_COPY["bridges"],
            interpretation="The network has multiple pathways connecting its components.",
            why_it_matters="This structural redundancy provides resilience against disruption.",
            evidence={"bridge_count": 0, "top_bridges": []},
            confidence="high"
        )
    
    signal = f"{bridge_count} single-point bridge{'s' if bridge_count != 1 else ''} identified."
    interpretation = "These nodes are critical connectors; their removal would fragment the network."
    risk = "If any of these bridges exit or reduce engagement, entire network segments could become isolated."
    mitigation = "Consider building redundant connections around these critical nodes to reduce fragility."
    
    evidence = {
        "bridge_count": bridge_count,
        "top_bridges": top_bridges[:cfg.TOP_N_BRIDGES]
    }
    
    return SignalInterpretation(
        signal=signal,
        interpretation=interpretation,
        why_it_matters=f"**Risk:** {risk}\n\n**Mitigation:** {mitigation}",
        evidence=evidence,
        confidence="high"
    )


def interpret_network_health(
    health_score: float,
    multi_funder_pct: float,
    connectivity_pct: float,
    top5_share_pct: float,
    governance_connectivity_label: str
) -> SignalInterpretation:
    """
    Interpret the overall network health score.
    """
    # Determine label based on score
    health_label = "Fragile"
    for threshold, label in sorted(cfg.HEALTH_LABEL_THRESHOLDS.items(), reverse=True):
        if health_score >= threshold:
            health_label = label
            break
    
    if health_score >= 80:
        interpretation = "This network shows strong coordination infrastructure."
        implications = "The system is well-positioned for collective action; focus on maintaining and leveraging existing strengths."
    elif health_score >= 60:
        interpretation = "This network has moderate coordination capacity with room for improvement."
        implications = "Targeted investments in weak areas could significantly improve collective effectiveness."
    elif health_score >= 40:
        interpretation = "This network is developing but lacks strong coordination infrastructure."
        implications = "Foundational work is needed before coordinated initiatives will succeed."
    else:
        interpretation = "This network shows fragile coordination capacity."
        implications = "Significant investment in relationship-building is needed before pursuing collective goals."
    
    evidence = {
        "health_score": health_score,
        "health_label": health_label,
        "multi_funder_pct": multi_funder_pct,
        "connectivity_pct": connectivity_pct,
        "top5_share_pct": top5_share_pct,
        "governance_connectivity_label": governance_connectivity_label
    }
    
    return SignalInterpretation(
        signal=f"Network Health Score: {health_score:.0f}/100 — {health_label}",
        interpretation=interpretation,
        why_it_matters=implications,
        evidence=evidence,
        confidence="high"
    )


# =============================================================================
# System Summary Generator
# =============================================================================

def generate_system_summary(
    all_signals: Dict[str, SignalInterpretation],
    project_name: str = "Network"
) -> SystemSummary:
    """
    Generate the system summary from all signal interpretations.
    
    Always returns:
    - 1 headline
    - 1 summary paragraph
    - 2 positives ("What's working")
    - 2 gaps ("What's missing")  
    - 1 "Why this matters" sentence
    """
    # Extract key metrics for headline generation
    funding = all_signals.get("funding_concentration")
    overlap = all_signals.get("funder_overlap")
    governance = all_signals.get("governance")
    health = all_signals.get("network_health")
    
    # Generate headline based on dominant patterns
    positives = []
    gaps = []
    
    # Analyze funding concentration
    if funding and funding.confidence != "unavailable":
        top5 = funding.evidence.get("top5_share_pct", 50)
        if top5 <= 50:
            positives.append("Funding is distributed across multiple sources, providing resilience")
        else:
            gaps.append("Funding is concentrated among few funders, creating dependency risk")
    
    # Analyze overlap
    if overlap and overlap.confidence != "unavailable":
        multi_pct = overlap.evidence.get("multi_funder_pct", 0)
        if multi_pct > cfg.MULTI_FUNDER_HIGH_THRESHOLD:
            positives.append("Strong funder overlap creates natural coordination anchors")
        elif multi_pct < cfg.MULTI_FUNDER_LOW_THRESHOLD:
            gaps.append("Limited funder overlap means few natural coordination points exist")
        else:
            positives.append("Moderate funder overlap provides some coordination potential")
    
    # Analyze governance
    if governance and governance.confidence not in ["unavailable", "low"]:
        shared_boards = governance.evidence.get("shared_board_count", 0)
        if shared_boards > 3:
            positives.append("Board interlocks suggest informal coordination channels exist")
        else:
            gaps.append("Limited board connectivity means coordination must be built through other channels")
    elif governance and governance.confidence == "unavailable":
        gaps.append("Governance relationships are unmapped, leaving potential connections invisible")
    
    # Analyze brokers/bridges
    brokers = all_signals.get("hidden_brokers")
    bridges = all_signals.get("single_point_bridges")
    
    if brokers and brokers.evidence.get("broker_count", 0) > 0:
        positives.append("Hidden brokers offer untapped coordination leverage")
    
    if bridges and bridges.evidence.get("bridge_count", 0) > 0:
        gaps.append("Single-point bridges create structural fragility")
    
    # Ensure exactly 2 positives and 2 gaps
    default_positive = "Network structure provides a foundation for coordination"
    default_gap = "Opportunities for strengthened coordination remain unexplored"
    
    while len(positives) < 2:
        positives.append(default_positive)
    while len(gaps) < 2:
        gaps.append(default_gap)
    
    positives = positives[:2]
    gaps = gaps[:2]
    
    # Generate headline
    health_score = health.evidence.get("health_score", 50) if health else 50
    if health_score >= 70 and len([p for p in positives if "coordination" in p.lower()]) > 0:
        headline = "Connected and Coordinated"
    elif health_score >= 50:
        headline = "Connected but Not Yet Coordinated"
    elif health_score >= 30:
        headline = "Fragmented with Coordination Potential"
    else:
        headline = "Siloed with Limited Coordination Infrastructure"
    
    # Generate summary paragraph
    funder_count = funding.evidence.get("funder_count", 0) if funding else 0
    grantee_count = funding.evidence.get("grantee_count", 0) if funding else 0
    
    summary_paragraph = (
        f"This {project_name.lower()} encompasses {funder_count} funders and {grantee_count} grantees. "
        f"{positives[0]}. However, {gaps[0].lower()}. "
        f"The analysis below identifies specific leverage points and risks."
    )
    
    # Why it matters
    why_it_matters = (
        "Understanding network structure reveals where coordination is possible, "
        "where it requires investment, and where fragility demands attention."
    )
    
    return SystemSummary(
        headline=headline,
        summary_paragraph=summary_paragraph,
        positives=positives,
        gaps=gaps,
        why_it_matters=why_it_matters
    )


# =============================================================================
# Recommendations Generator
# =============================================================================

def generate_recommendations(
    all_signals: Dict[str, SignalInterpretation]
) -> List[Recommendation]:
    """
    Generate exactly 4 strategic recommendations based on signal triggers.
    
    Trigger → Recommendation mapping:
    - Low overlap → "Start with anchor grantees"
    - Missing governance → "Consider capturing governance ties"
    - Single-point bridges present → "Build redundancy around critical connectors"
    - Brokers abundant → "Engage hidden brokers as coordination catalysts"
    - High concentration → "Diversify funding relationships"
    - Strong overlap → "Leverage existing coordination anchors"
    """
    recommendations = []
    
    # Check overlap signal
    overlap = all_signals.get("funder_overlap")
    if overlap:
        multi_pct = overlap.evidence.get("multi_funder_pct", 0)
        if multi_pct < cfg.MULTI_FUNDER_LOW_THRESHOLD:
            recommendations.append(Recommendation(
                title="Identify Anchor Grantees",
                text=(
                    "With limited natural overlap, prioritize identifying grantees that could serve as "
                    "coordination anchors. Even a few shared investments can seed broader alignment."
                ),
                trigger="low_overlap"
            ))
        elif multi_pct > cfg.MULTI_FUNDER_HIGH_THRESHOLD:
            recommendations.append(Recommendation(
                title="Leverage Existing Coordination Anchors",
                text=(
                    "Strong funder overlap already exists. Convene top shared grantees to surface "
                    "coordination opportunities and build on existing relationships."
                ),
                trigger="high_overlap"
            ))
    
    # Check governance signal
    governance = all_signals.get("governance")
    if governance and governance.confidence in ["unavailable", "low"]:
        recommendations.append(Recommendation(
            title="Map Governance Relationships",
            text=(
                "Governance connectivity is currently unmapped or incomplete. Consider investing in "
                "board relationship mapping to reveal hidden coordination channels."
            ),
            trigger="missing_governance"
        ))
    elif governance and governance.evidence.get("shared_board_count", 0) > 3:
        recommendations.append(Recommendation(
            title="Activate Board Networks",
            text=(
                "Board interlocks suggest informal coordination channels already exist. "
                "Engage multi-board individuals as bridges for strategic conversations."
            ),
            trigger="strong_governance"
        ))
    
    # Check bridges signal
    bridges = all_signals.get("single_point_bridges")
    if bridges and bridges.evidence.get("bridge_count", 0) > 0:
        recommendations.append(Recommendation(
            title="Build Redundancy Around Critical Connectors",
            text=(
                "Single-point bridges create structural fragility. Invest in building alternative "
                "connections around these critical nodes to improve network resilience."
            ),
            trigger="bridges_present"
        ))
    
    # Check brokers signal
    brokers = all_signals.get("hidden_brokers")
    if brokers and brokers.evidence.get("broker_count", 0) > 0:
        recommendations.append(Recommendation(
            title="Engage Hidden Brokers",
            text=(
                "Hidden brokers offer untapped coordination leverage. These low-visibility, "
                "high-connectivity organizations can accelerate alignment without new infrastructure."
            ),
            trigger="brokers_present"
        ))
    
    # Check concentration signal
    funding = all_signals.get("funding_concentration")
    if funding:
        top5 = funding.evidence.get("top5_share_pct", 50)
        if top5 > cfg.CONCENTRATION_HIGH_THRESHOLD:
            recommendations.append(Recommendation(
                title="Diversify Funding Relationships",
                text=(
                    "High funding concentration creates dependency risk. Consider strategies to "
                    "broaden the funder base or deepen relationships with secondary funders."
                ),
                trigger="high_concentration"
            ))
    
    # Ensure exactly 4 recommendations
    default_recommendations = [
        Recommendation(
            title="Strengthen Information Sharing",
            text=(
                "Regular information exchange between funders and key grantees can surface "
                "coordination opportunities without requiring formal structures."
            ),
            trigger="default_info_sharing"
        ),
        Recommendation(
            title="Map Thematic Alignment",
            text=(
                "Analyze grant purposes to identify thematic clusters where coordination "
                "could amplify impact across multiple funders."
            ),
            trigger="default_thematic"
        ),
        Recommendation(
            title="Invest in Network Visibility",
            text=(
                "This analysis provides a structural view. Consider complementing it with "
                "stakeholder interviews to surface relationship dynamics not visible in data."
            ),
            trigger="default_visibility"
        ),
        Recommendation(
            title="Monitor Network Evolution",
            text=(
                "Network structure changes over time. Consider repeating this analysis "
                "periodically to track progress and identify emerging patterns."
            ),
            trigger="default_evolution"
        )
    ]
    
    # Add defaults until we have exactly 4
    for rec in default_recommendations:
        if len(recommendations) >= cfg.RECOMMENDATIONS_COUNT:
            break
        recommendations.append(rec)
    
    return recommendations[:cfg.RECOMMENDATIONS_COUNT]

