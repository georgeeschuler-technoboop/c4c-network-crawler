"""
Insight Engine v2 — Streamlit App

A systems briefing generator, not a dashboard.
Generates narrative reports with expandable evidence.

The UI is an interface to the briefing, not the product itself.

Version: 2.0.0
Based on: insight-engine-spec.md (December 2025)
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
import zipfile

# =============================================================================
# Import Insight Engine modules
# =============================================================================
import config as cfg
from narratives import (
    SignalInterpretation,
    interpret_funding_concentration,
    interpret_funder_overlap,
    interpret_portfolio_twins,
    interpret_governance,
    interpret_hidden_brokers,
    interpret_single_point_bridges,
    interpret_network_health,
    generate_system_summary,
    generate_recommendations,
)
from report_generator import ReportData, generate_report

# =============================================================================
# App Configuration
# =============================================================================

APP_VERSION = "2.0.0"
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_ed8e76c8495d4799a5d7575822009e93~mv2.png"

# Paths
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"

st.set_page_config(
    page_title="Insight Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Session State Initialization
# =============================================================================

if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "generated_report" not in st.session_state:
    st.session_state.generated_report = None


# =============================================================================
# Data Loading
# =============================================================================

def get_projects() -> list[dict]:
    """Scan demo_data directory for available projects."""
    projects = []
    
    if not DEMO_DATA_DIR.exists():
        return projects
    
    for folder in DEMO_DATA_DIR.iterdir():
        if not folder.is_dir():
            continue
        
        # Check for required files
        has_nodes = (folder / "nodes.csv").exists()
        has_edges = (folder / "edges.csv").exists()
        has_metrics = (folder / "network_metrics.json").exists()
        has_grants = (folder / "grants_detail.csv").exists()
        
        if has_nodes and has_edges:
            projects.append({
                "id": folder.name,
                "path": folder,
                "has_nodes": has_nodes,
                "has_edges": has_edges,
                "has_metrics": has_metrics,
                "has_grants": has_grants,
                "is_complete": has_nodes and has_edges and has_metrics
            })
    
    return sorted(projects, key=lambda x: x["id"])


def load_metrics(project_path: Path) -> dict:
    """Load network metrics from JSON file or compute from CSVs."""
    metrics_path = project_path / "network_metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    # Fallback: compute basic metrics from nodes/edges
    return compute_basic_metrics(project_path)


def compute_basic_metrics(project_path: Path) -> dict:
    """Compute basic metrics from nodes.csv and edges.csv."""
    nodes_df = pd.read_csv(project_path / "nodes.csv")
    edges_df = pd.read_csv(project_path / "edges.csv")
    
    # Basic counts
    node_count = len(nodes_df)
    edge_count = len(edges_df)
    
    # Node type counts
    funder_count = len(nodes_df[nodes_df.get("node_type", "") == "funder"]) if "node_type" in nodes_df.columns else 0
    grantee_count = len(nodes_df[nodes_df.get("node_type", "") == "grantee"]) if "node_type" in nodes_df.columns else 0
    
    # Funding totals
    total_funding = edges_df["grant_amount"].sum() if "grant_amount" in edges_df.columns else 0
    
    # Multi-funder calculation
    if "target_id" in edges_df.columns and "source_id" in edges_df.columns:
        grantee_funder_counts = edges_df.groupby("target_id")["source_id"].nunique()
        multi_funder_count = (grantee_funder_counts > 1).sum()
        multi_funder_pct = (multi_funder_count / len(grantee_funder_counts) * 100) if len(grantee_funder_counts) > 0 else 0
    else:
        multi_funder_count = 0
        multi_funder_pct = 0
    
    # Top 5 concentration
    if "source_id" in edges_df.columns and "grant_amount" in edges_df.columns:
        funder_totals = edges_df.groupby("source_id")["grant_amount"].sum().sort_values(ascending=False)
        top5_funding = funder_totals.head(5).sum()
        top5_share_pct = (top5_funding / total_funding * 100) if total_funding > 0 else 0
    else:
        top5_share_pct = 0
    
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "funder_count": funder_count,
        "grantee_count": grantee_count,
        "total_funding": total_funding,
        "multi_funder_count": multi_funder_count,
        "multi_funder_pct": multi_funder_pct,
        "top5_share_pct": top5_share_pct,
        "connectivity_pct": 50,  # Placeholder - needs graph analysis
        "network_health_score": 50,  # Placeholder
        "governance_data_available": False,
        "shared_board_count": 0,
        "pct_with_interlocks": 0,
        "broker_count": 0,
        "bridge_count": 0,
        "top_shared_grantees": [],
        "top_funder_pairs": [],
        "top_brokers": [],
        "top_bridges": [],
    }


def load_evidence_data(project_path: Path, metrics: dict) -> dict:
    """Load additional evidence data (shared grantees, pairs, etc.) if available."""
    # Try to load pre-computed evidence files
    evidence_files = {
        "shared_grantees": "shared_grantees.json",
        "funder_pairs": "funder_pairs.json",
        "brokers": "brokers.json",
        "bridges": "bridges.json",
    }
    
    for key, filename in evidence_files.items():
        filepath = project_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                if key == "shared_grantees":
                    metrics["top_shared_grantees"] = data
                elif key == "funder_pairs":
                    metrics["top_funder_pairs"] = data
                elif key == "brokers":
                    metrics["top_brokers"] = data
                elif key == "bridges":
                    metrics["top_bridges"] = data
    
    return metrics


# =============================================================================
# UI Helper Functions
# =============================================================================

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


def confidence_badge(confidence: str) -> str:
    """Return emoji badge for confidence level."""
    badges = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🟠",
        "unavailable": "⚪"
    }
    return badges.get(confidence, "⚪")


def render_signal_section(
    title: str,
    signal: SignalInterpretation,
    lens: str = None,
    show_evidence: bool = True
):
    """Render a signal section with consistent structure."""
    
    # Section header with optional lens
    if lens:
        st.markdown(f"### {title}")
        if lens == "opportunity":
            st.info("**Lens: Opportunity**")
        elif lens == "risk":
            st.warning("**Lens: Risk**")
    else:
        st.markdown(f"### {title}")
    
    # Confidence indicator
    st.caption(f"{confidence_badge(signal.confidence)} Confidence: {signal.confidence}")
    
    # Signal → Interpretation → Why it matters
    st.markdown(f"**Signal**  \n{signal.signal}")
    st.markdown(f"**Interpretation**  \n{signal.interpretation}")
    st.markdown(f"**Why it matters**  \n{signal.why_it_matters}")
    
    # Evidence (collapsible if there's data)
    if show_evidence and signal.evidence:
        with st.expander("📊 View Evidence", expanded=False):
            render_evidence(signal.evidence)


def render_evidence(evidence: dict):
    """Render evidence data as appropriate UI elements."""
    
    # Handle different evidence types
    if "top_shared_grantees" in evidence and evidence["top_shared_grantees"]:
        st.markdown("**Top Shared Grantees**")
        grantees = evidence["top_shared_grantees"]
        df = pd.DataFrame(grantees)
        if not df.empty:
            display_cols = ["grantee_name", "funder_count", "total_received"]
            display_cols = [c for c in display_cols if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    
    if "top_funder_pairs" in evidence and evidence["top_funder_pairs"]:
        st.markdown("**Top Funder Pairs**")
        pairs = evidence["top_funder_pairs"]
        df = pd.DataFrame(pairs)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    if "top_brokers" in evidence and evidence["top_brokers"]:
        st.markdown("**Hidden Brokers**")
        brokers = evidence["top_brokers"]
        df = pd.DataFrame(brokers)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    if "top_bridges" in evidence and evidence["top_bridges"]:
        st.markdown("**Single-Point Bridges**")
        bridges = evidence["top_bridges"]
        df = pd.DataFrame(bridges)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Render simple key-value metrics
    simple_keys = ["total_funding", "funder_count", "grantee_count", "top5_share_pct", 
                   "multi_funder_count", "multi_funder_pct", "broker_count", "bridge_count"]
    
    metrics_to_show = {k: v for k, v in evidence.items() if k in simple_keys and v is not None}
    
    if metrics_to_show:
        cols = st.columns(min(4, len(metrics_to_show)))
        for i, (key, value) in enumerate(metrics_to_show.items()):
            col = cols[i % len(cols)]
            label = key.replace("_", " ").title()
            if "funding" in key.lower() or "received" in key.lower():
                col.metric(label, format_currency(value))
            elif "pct" in key.lower():
                col.metric(label, f"{value:.1f}%")
            else:
                col.metric(label, f"{value:,}" if isinstance(value, (int, float)) else value)


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.title("🔬 Insight Engine")
    st.caption(f"Systems Briefing Generator • v{APP_VERSION}")
    
    # =========================================================================
    # Reading Contract Banner
    # =========================================================================
    with st.expander("📖 How to Read This Report", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⏱️ 5 minutes (Skim)**")
            st.markdown("""
            - System Summary
            - Network Health
            - Strategic Recommendations
            """)
        
        with col2:
            st.markdown("**📚 20 minutes (Read)**")
            st.markdown("""
            - All signal sections
            - Selected evidence
            """)
        
        with col3:
            st.markdown("**🔍 Optional (Explore)**")
            st.markdown("""
            - Grant themes
            - Data tables
            - Downloads
            """)
        
        st.caption("*If Governance Connectivity shows 'data unavailable,' you may skip that section.*")
    
    st.divider()
    
    # =========================================================================
    # Project Selection
    # =========================================================================
    st.markdown("### 🗂️ Project")
    
    projects = get_projects()
    
    if not projects:
        st.warning(f"""
        **No projects found.**
        
        Expected location: `{DEMO_DATA_DIR}`
        
        Each project folder should contain:
        - `nodes.csv` (required)
        - `edges.csv` (required)
        - `network_metrics.json` (recommended)
        """)
        st.stop()
    
    # Project selector
    project_options = {p["id"]: p for p in projects}
    selected_id = st.selectbox(
        "Select project",
        options=list(project_options.keys()),
        format_func=lambda x: f"{'✅' if project_options[x]['is_complete'] else '⚠️'} {x}",
        label_visibility="collapsed"
    )
    
    project = project_options[selected_id]
    
    # Status indicator
    if project["is_complete"]:
        st.success(f"📂 **{selected_id}** — Complete dataset with pre-computed metrics")
    else:
        st.info(f"📂 **{selected_id}** — Basic dataset (some metrics will be estimated)")
    
    # Generate button
    if st.button("🚀 Generate Briefing", type="primary", use_container_width=True):
        with st.spinner("Generating systems briefing..."):
            # Load metrics
            metrics = load_metrics(project["path"])
            metrics = load_evidence_data(project["path"], metrics)
            
            # Generate report data
            report_data = ReportData(metrics, project_name=selected_id)
            report_markdown = generate_report(report_data)
            
            # Store in session state
            st.session_state.report_data = report_data
            st.session_state.generated_report = report_markdown
            st.session_state.project_id = selected_id
        
        st.rerun()
    
    # =========================================================================
    # Report Display
    # =========================================================================
    
    if st.session_state.report_data is None:
        st.info("👆 Select a project and click **Generate Briefing** to create your systems briefing.")
        st.stop()
    
    report_data = st.session_state.report_data
    signals = report_data.signals
    summary = report_data.system_summary
    recommendations = report_data.recommendations
    metrics = report_data.metrics
    
    st.divider()
    
    # =========================================================================
    # 1. System Summary
    # =========================================================================
    st.markdown("## System Summary")
    
    # Headline
    st.markdown(f"### {summary.headline}")
    
    # Summary paragraph
    st.markdown(summary.summary_paragraph)
    
    # What's working / What's missing
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ What's working**")
        for positive in summary.positives:
            st.markdown(f"- {positive}")
    
    with col2:
        st.markdown("**⚠️ What's missing**")
        for gap in summary.gaps:
            st.markdown(f"- {gap}")
    
    # Why it matters
    st.markdown(f"**Why this matters**  \n{summary.why_it_matters}")
    
    st.divider()
    
    # =========================================================================
    # 2. Network Health
    # =========================================================================
    st.markdown("## Network Health")
    
    health = signals["network_health"]
    health_score = health.evidence.get("health_score", 50)
    health_label = health.evidence.get("health_label", "Moderate")
    
    # Score display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="Overall Score",
            value=f"{health_score:.0f} / 100",
            delta=health_label
        )
    
    # Key signals table
    st.markdown("### Key Signals")
    
    health_data = {
        "Metric": [
            "Multi-Funder Grantees",
            "Network Connectivity", 
            "Funding Concentration (Top 5)",
            "Governance Connectivity"
        ],
        "Value": [
            f"{metrics.get('multi_funder_pct', 0):.0f}%",
            f"{metrics.get('connectivity_pct', 0):.0f}%",
            f"{metrics.get('top5_share_pct', 0):.0f}%",
            "Available" if metrics.get("governance_data_available") else "Unavailable"
        ],
        "Status": [
            "🟢" if metrics.get('multi_funder_pct', 0) > 30 else "🟡" if metrics.get('multi_funder_pct', 0) > 10 else "🔴",
            "🟢" if metrics.get('connectivity_pct', 0) > 70 else "🟡" if metrics.get('connectivity_pct', 0) > 40 else "🔴",
            "🟢" if metrics.get('top5_share_pct', 0) < 50 else "🟡" if metrics.get('top5_share_pct', 0) < 70 else "🔴",
            "🟢" if metrics.get("governance_data_available") else "⚪"
        ]
    }
    
    st.dataframe(
        pd.DataFrame(health_data),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown(f"**Why this matters**  \n{health.why_it_matters}")
    
    st.divider()
    
    # =========================================================================
    # 3. Funding Concentration
    # =========================================================================
    render_signal_section(
        "Funding Concentration",
        signals["funding_concentration"]
    )
    
    st.divider()
    
    # =========================================================================
    # 4. Funder Overlap Clusters
    # =========================================================================
    render_signal_section(
        "Funder Overlap Clusters",
        signals["funder_overlap"]
    )
    
    st.divider()
    
    # =========================================================================
    # 5. Portfolio Twins
    # =========================================================================
    render_signal_section(
        "Portfolio Twins",
        signals["portfolio_twins"]
    )
    
    st.divider()
    
    # =========================================================================
    # 6. Governance Connectivity
    # =========================================================================
    render_signal_section(
        "Governance Connectivity",
        signals["governance"]
    )
    
    st.divider()
    
    # =========================================================================
    # 7. Hidden Brokers (Opportunity)
    # =========================================================================
    render_signal_section(
        "Hidden Brokers",
        signals["hidden_brokers"],
        lens="opportunity"
    )
    
    st.divider()
    
    # =========================================================================
    # 8. Single-Point Bridges (Risk)
    # =========================================================================
    render_signal_section(
        "Single-Point Bridges",
        signals["single_point_bridges"],
        lens="risk"
    )
    
    st.divider()
    
    # =========================================================================
    # 9. Strategic Recommendations
    # =========================================================================
    st.markdown("## Strategic Recommendations")
    
    st.markdown(cfg.STRATEGIC_INTRO)
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f"**{i}. {rec.title}**")
            st.markdown(rec.text)
            st.caption(f"Trigger: {rec.trigger}")
    
    st.divider()
    
    # =========================================================================
    # 10. Appendix / Optional Deep Dive
    # =========================================================================
    st.markdown("## Appendix")
    
    # Grant Purpose Explorer (collapsed)
    if project.get("has_grants"):
        with st.expander("🎯 Grant Purpose Explorer (Optional Deep Dive)", expanded=False):
            st.info("Grant purpose analysis available. This feature helps explore thematic patterns in funding.")
            # Placeholder for Grant Purpose Explorer integration
            grants_path = project["path"] / "grants_detail.csv"
            if grants_path.exists():
                grants_df = pd.read_csv(grants_path)
                st.dataframe(grants_df.head(20), use_container_width=True, hide_index=True)
    
    # Method Notes
    with st.expander("📋 Method Notes", expanded=False):
        st.markdown(f"""
**Network Health Score (v1):** Weighted composite of coordination (multi-funder %), 
connectivity, concentration (inverse), and governance connectivity proxies.

**Hidden Broker Definition:** Nodes in top {cfg.BROKER_BETWEENNESS_PERCENTILE}th percentile 
of betweenness centrality AND bottom {cfg.BROKER_VISIBILITY_PERCENTILE}th percentile of degree, 
computed within each node type (funder, grantee, person) to avoid cross-type artifacts.

**Single-Point Bridge Definition:** Articulation points — nodes whose removal disconnects 
the network graph.
        """)
    
    # Data Notes
    with st.expander("📊 Data Notes", expanded=False):
        st.markdown(f"""
- **Data Sources:** {metrics.get('data_sources', 'IRS 990-PF filings')}
- **Date Range:** {metrics.get('date_range', 'Not specified')}
- **Processing Timestamp:** {report_data.generated_timestamp}
- **Thresholds Applied:** TOP_N={cfg.TOP_N_BROKERS}, Concentration High={cfg.CONCENTRATION_HIGH_THRESHOLD}%
        """)
    
    st.divider()
    
    # =========================================================================
    # Project Outputs (Downloads)
    # =========================================================================
    st.markdown("## 📥 Project Outputs")
    
    st.markdown("Download your briefing and supporting data.")
    
    # Organize by audience
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👔 Executive**")
        st.download_button(
            label="📄 Download Executive Briefing",
            data=st.session_state.generated_report,
            file_name=f"{selected_id}_insight_report.md",
            mime="text/markdown",
            help="Markdown format, convertible to PDF",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**📊 Analyst**")
        
        # Create metrics JSON
        metrics_json = json.dumps(metrics, indent=2, default=str)
        st.download_button(
            label="📊 Download Metrics (JSON)",
            data=metrics_json,
            file_name=f"{selected_id}_metrics.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        st.markdown("**🛠️ Developer**")
        
        # Create full package ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{selected_id}_insight_report.md", st.session_state.generated_report)
            zf.writestr(f"{selected_id}_metrics.json", metrics_json)
            
            # Add source files if available
            for filename in ["nodes.csv", "edges.csv", "grants_detail.csv"]:
                filepath = project["path"] / filename
                if filepath.exists():
                    zf.write(filepath, f"data/{filename}")
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 Download Full Package (ZIP)",
            data=zip_buffer,
            file_name=f"{selected_id}_full_package.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # =========================================================================
    # Footer
    # =========================================================================
    st.divider()
    st.caption(f"*Report generated by {cfg.GENERATOR_CREDIT} v{cfg.APP_VERSION}*")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
