"""
Insight Engine — Report Generator

Generates the canonical Markdown report from network metrics and narratives.
Output matches the exact structure defined in the spec.

VERSION HISTORY:
----------------
v1.0.0 (2025-12-19): Initial release
- ReportData container class
- Table formatters for evidence sections
- generate_report() producing canonical Markdown

v1.0.1 (2025-12-19): Updated visibility display
- Broker table now shows visibility_rank (lower = less visible)
- Shows raw degree in parentheses to ground metric in real data
- Updated footnote to explain calculation
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

import config as cfg
from narratives import (
    SignalInterpretation,
    SystemSummary,
    Recommendation,
    interpret_funding_concentration,
    interpret_funder_overlap,
    interpret_portfolio_twins,
    interpret_governance,
    interpret_hidden_brokers,
    interpret_single_point_bridges,
    interpret_network_health,
    generate_system_summary,
    generate_recommendations
)


# =============================================================================
# Report Data Container
# =============================================================================

class ReportData:
    """Container for all data needed to generate a report."""
    
    def __init__(self, metrics: Dict[str, Any], project_name: str = "Network"):
        self.metrics = metrics
        self.project_name = project_name
        self.generated_timestamp = datetime.now().isoformat()
        
        # Generate all signal interpretations
        self.signals: Dict[str, SignalInterpretation] = {}
        self._generate_all_signals()
        
        # Generate summary and recommendations
        self.system_summary = generate_system_summary(self.signals, project_name)
        self.recommendations = generate_recommendations(self.signals)
    
    def _generate_all_signals(self):
        """Generate interpretations for all signal sections."""
        m = self.metrics
        
        # Funding Concentration
        self.signals["funding_concentration"] = interpret_funding_concentration(
            top5_share_pct=m.get("top5_share_pct", 0),
            total_funding=m.get("total_funding", 0),
            funder_count=m.get("funder_count", 0),
            grantee_count=m.get("grantee_count", 0)
        )
        
        # Funder Overlap
        self.signals["funder_overlap"] = interpret_funder_overlap(
            multi_funder_pct=m.get("multi_funder_pct", 0),
            multi_funder_count=m.get("multi_funder_count", 0),
            total_grantees=m.get("grantee_count", 0),
            top_shared_grantees=m.get("top_shared_grantees", [])
        )
        
        # Portfolio Twins
        self.signals["portfolio_twins"] = interpret_portfolio_twins(
            top_funder_pairs=m.get("top_funder_pairs", []),
            max_overlap_pct=m.get("max_funder_overlap_pct", 0)
        )
        
        # Governance
        self.signals["governance"] = interpret_governance(
            governance_data_available=m.get("governance_data_available", False),
            shared_board_count=m.get("shared_board_count", 0),
            pct_with_interlocks=m.get("pct_with_interlocks", 0),
            governance_coverage_pct=m.get("governance_coverage_pct", 0),
            top_board_connectors=m.get("top_board_connectors", [])
        )
        
        # Hidden Brokers
        self.signals["hidden_brokers"] = interpret_hidden_brokers(
            broker_count=m.get("broker_count", 0),
            top_brokers=m.get("top_brokers", [])
        )
        
        # Single-Point Bridges
        self.signals["single_point_bridges"] = interpret_single_point_bridges(
            bridge_count=m.get("bridge_count", 0),
            top_bridges=m.get("top_bridges", [])
        )
        
        # Network Health
        health_score = m.get("network_health_score", 50)
        governance_label = "Available" if m.get("governance_data_available") else "Unavailable"
        
        self.signals["network_health"] = interpret_network_health(
            health_score=health_score,
            multi_funder_pct=m.get("multi_funder_pct", 0),
            connectivity_pct=m.get("connectivity_pct", 0),
            top5_share_pct=m.get("top5_share_pct", 0),
            governance_connectivity_label=governance_label
        )


# =============================================================================
# Table Formatters
# =============================================================================

def format_shared_grantees_table(grantees: List[Dict[str, Any]]) -> str:
    """Format shared grantees as Markdown table rows."""
    if not grantees:
        return "| *No shared grantees detected* | — | — | — |"
    
    rows = []
    for g in grantees[:cfg.TOP_N_SHARED_GRANTEES]:
        name = g.get("grantee_name", "Unknown")
        funder_count = g.get("funder_count", 0)
        top_funders = g.get("top_funders", [])
        funders_str = ", ".join(top_funders[:3]) + ("..." if len(top_funders) > 3 else "")
        total = g.get("total_received", 0)
        total_str = f"${total:,.0f}" if total else "—"
        rows.append(f"| {name} | {funder_count} | {funders_str} | {total_str} |")
    
    return "\n".join(rows)


def format_funder_pairs_table(pairs: List[Dict[str, Any]]) -> str:
    """Format funder pairs as Markdown table rows."""
    if not pairs:
        return "| *No significant funder pairs detected* | — | — | — |"
    
    rows = []
    for p in pairs[:cfg.TOP_N_FUNDER_PAIRS]:
        funder_a = p.get("funder_a", "Unknown")
        funder_b = p.get("funder_b", "Unknown")
        shared_count = p.get("shared_grantee_count", 0)
        overlap_pct = p.get("overlap_pct", 0)
        rows.append(f"| {funder_a} | {funder_b} | {shared_count} | {overlap_pct:.0f}% |")
    
    return "\n".join(rows)


def format_brokers_table(brokers: List[Dict[str, Any]]) -> str:
    """Format hidden brokers as Markdown table rows."""
    if not brokers:
        return "| *No hidden brokers detected* | — | — | — |"
    
    rows = []
    for b in brokers[:cfg.TOP_N_BROKERS]:
        name = b.get("org_name", "Unknown")
        betweenness = b.get("betweenness", 0)
        visibility_rank = b.get("visibility_rank", b.get("visibility_percentile", 0))
        degree = b.get("degree", 0)
        reason = b.get("broker_reason", "High structural importance, low visibility")
        # Show visibility rank with degree in parentheses for grounding
        rows.append(f"| {name} | {betweenness:.3f} | {visibility_rank:.0f}% ({degree} connections) | {reason} |")
    
    return "\n".join(rows)


def format_bridges_table(bridges: List[Dict[str, Any]]) -> str:
    """Format single-point bridges as Markdown table rows."""
    if not bridges:
        return "| *No single-point bridges detected* | — | — |"
    
    rows = []
    for b in bridges[:cfg.TOP_N_BRIDGES]:
        name = b.get("node_name", "Unknown")
        node_type = b.get("node_type", "Unknown")
        impact = b.get("impact_if_removed", "Network fragmentation")
        rows.append(f"| {name} | {node_type} | {impact} |")
    
    return "\n".join(rows)


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.0f}K"
    else:
        return f"${amount:,.0f}"


# =============================================================================
# Report Generator
# =============================================================================

def generate_report(report_data: ReportData) -> str:
    """
    Generate the complete Markdown report.
    
    Returns the report as a string matching the canonical structure.
    """
    m = report_data.metrics
    s = report_data.signals
    summary = report_data.system_summary
    recs = report_data.recommendations
    
    # Network Health details
    health = s["network_health"]
    health_score = health.evidence.get("health_score", 50)
    health_label = health.evidence.get("health_label", "Moderate")
    
    # Format tables
    shared_grantee_rows = format_shared_grantees_table(
        s["funder_overlap"].evidence.get("top_shared_grantees", [])
    )
    funder_pair_rows = format_funder_pairs_table(
        s["portfolio_twins"].evidence.get("top_funder_pairs", [])
    )
    broker_rows = format_brokers_table(
        s["hidden_brokers"].evidence.get("top_brokers", [])
    )
    bridge_rows = format_bridges_table(
        s["single_point_bridges"].evidence.get("top_bridges", [])
    )
    
    # Build the report
    report = f"""# Network Insight Report

**Project:** {report_data.project_name}  
**Generated:** {report_data.generated_timestamp}

**Nodes:** {m.get('node_count', 0):,} | **Edges:** {m.get('edge_count', 0):,} | **Total Funding:** {format_currency(m.get('total_funding', 0))}

---

## How to Read This Report

- **5 minutes:** System Summary → Network Health → Strategic Recommendations
- **20 minutes:** Read all signal sections and expand selected evidence
- **Optional:** Explore grant themes or download data files

*If Governance Connectivity shows "data unavailable," you may skip that section.*

---

## System Summary

**{summary.headline}**

{summary.summary_paragraph}

**What's working**
- {summary.positives[0]}
- {summary.positives[1]}

**What's missing**
- {summary.gaps[0]}
- {summary.gaps[1]}

**Why this matters**  
{summary.why_it_matters}

---

## Network Health

**Overall Score:** {health_score:.0f} / 100 — {health_label}

### Key Signals

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Multi-Funder Grantees | {m.get('multi_funder_pct', 0):.0f}% | {_short_interpretation(s['funder_overlap'])} |
| Network Connectivity | {m.get('connectivity_pct', 0):.0f}% | {_connectivity_interpretation(m.get('connectivity_pct', 0))} |
| Funding Concentration (Top 5) | {m.get('top5_share_pct', 0):.0f}% | {_short_interpretation(s['funding_concentration'])} |
| Governance Connectivity | {health.evidence.get('governance_connectivity_label', 'Unknown')} | {_short_interpretation(s['governance'])} |

**Why this matters**  
{health.why_it_matters}

---

## Funding Concentration

**Signal**  
{s['funding_concentration'].signal}

**Interpretation**  
{s['funding_concentration'].interpretation}

**Why it matters**  
{s['funding_concentration'].why_it_matters}

**Evidence**

| Metric | Value |
|--------|-------|
| Total Funding | {format_currency(m.get('total_funding', 0))} |
| Funders | {m.get('funder_count', 0):,} |
| Grantees | {m.get('grantee_count', 0):,} |
| Top 5 Share | {m.get('top5_share_pct', 0):.0f}% |

---

## Funder Overlap Clusters

**Signal**  
{s['funder_overlap'].signal}

**Interpretation**  
{s['funder_overlap'].interpretation}

**Opportunity**  
{s['funder_overlap'].why_it_matters}

**Evidence (Top {cfg.TOP_N_SHARED_GRANTEES} shared grantees)**

| Grantee | Funder Count | Top Funders | Total Received |
|---------|--------------|-------------|----------------|
{shared_grantee_rows}

---

## Portfolio Twins

**Signal**  
{s['portfolio_twins'].signal}

**Interpretation**  
{s['portfolio_twins'].interpretation}

**Why it matters**  
{s['portfolio_twins'].why_it_matters}

**Evidence (Top {cfg.TOP_N_FUNDER_PAIRS} funder pairs)**

| Funder A | Funder B | Shared Grantees | Overlap % |
|----------|----------|-----------------|-----------|
{funder_pair_rows}

---

## Governance Connectivity

**Signal**  
{s['governance'].signal}

**Interpretation**  
{s['governance'].interpretation}

**Why it matters**  
{s['governance'].why_it_matters}

**Evidence**
- Shared board members detected: {m.get('shared_board_count', 0)}
- Funders with board interlocks: {m.get('pct_with_interlocks', 0):.0f}%

---

## Hidden Brokers

> **Lens: Opportunity**

**Signal**  
{s['hidden_brokers'].signal}

**Interpretation**  
{s['hidden_brokers'].interpretation}

**Opportunity**  
{s['hidden_brokers'].why_it_matters}

**Evidence (Top {cfg.TOP_N_BROKERS} hidden brokers)**

| Organization | Betweenness | Visibility | Why They Matter |
|--------------|-------------|------------|-----------------|
{broker_rows}

*Visibility shows what % of structurally important nodes have fewer connections than this one. Lower % = less visible. Raw connection count shown in parentheses.*

---

## Single-Point Bridges

> **Lens: Risk**

**Signal**  
{s['single_point_bridges'].signal}

**Interpretation**  
{s['single_point_bridges'].interpretation}

{s['single_point_bridges'].why_it_matters}

**Evidence (Top {cfg.TOP_N_BRIDGES} critical bridges)**

| Node | Type | Impact if Removed |
|------|------|-------------------|
{bridge_rows}

---

## Strategic Recommendations

{cfg.STRATEGIC_INTRO}

1. **{recs[0].title}**  
   {recs[0].text}

2. **{recs[1].title}**  
   {recs[1].text}

3. **{recs[2].title}**  
   {recs[2].text}

4. **{recs[3].title}**  
   {recs[3].text}

---

## Appendix

### Method Notes

**Network Health Score (v1):** Weighted composite of coordination (multi-funder %), connectivity, concentration (inverse), and governance connectivity proxies.

**Hidden Broker Definition:** Nodes in top {cfg.BROKER_BETWEENNESS_PERCENTILE}th percentile of betweenness centrality AND bottom {cfg.BROKER_VISIBILITY_PERCENTILE}th percentile of degree, computed within each node type (funder, grantee, person) to avoid cross-type artifacts. Optionally uses weighted degree when edge weights are stable.

**Single-Point Bridge Definition:** Articulation points — nodes whose removal disconnects the network graph.

### Data Notes

- **Data Sources:** {m.get('data_sources', 'IRS 990-PF filings')}
- **Date Range:** {m.get('date_range', 'Not specified')}
- **Processing Timestamp:** {report_data.generated_timestamp}
- **Thresholds Applied:** TOP_N={cfg.TOP_N_BROKERS}, Concentration High={cfg.CONCENTRATION_HIGH_THRESHOLD}%

---

*Report generated by {cfg.GENERATOR_CREDIT} v{cfg.APP_VERSION}*
"""
    
    return report


def _short_interpretation(signal: SignalInterpretation) -> str:
    """Extract a short version of interpretation for table cells."""
    interp = signal.interpretation
    # Truncate if too long for table
    if len(interp) > 60:
        return interp[:57] + "..."
    return interp


def _connectivity_interpretation(connectivity_pct: float) -> str:
    """Generate short interpretation for connectivity."""
    if connectivity_pct > 80:
        return "Strong network connectivity"
    elif connectivity_pct > 50:
        return "Moderate connectivity"
    elif connectivity_pct > 20:
        return "Limited connectivity"
    else:
        return "Fragmented network"


# =============================================================================
# File I/O
# =============================================================================

def load_metrics_from_json(filepath: Path) -> Dict[str, Any]:
    """Load network metrics from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_report(report: str, filepath: Path) -> None:
    """Save the generated report to a file."""
    with open(filepath, 'w') as f:
        f.write(report)


def generate_report_from_file(
    metrics_path: Path,
    output_path: Path,
    project_name: str = "Network"
) -> str:
    """
    Complete workflow: load metrics, generate report, save to file.
    
    Returns the generated report string.
    """
    metrics = load_metrics_from_json(metrics_path)
    report_data = ReportData(metrics, project_name)
    report = generate_report(report_data)
    save_report(report, output_path)
    return report


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python report_generator.py <metrics.json> <output.md> [project_name]")
        sys.exit(1)
    
    metrics_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    project_name = sys.argv[3] if len(sys.argv) > 3 else "Network"
    
    report = generate_report_from_file(metrics_path, output_path, project_name)
    print(f"Report generated: {output_path}")
