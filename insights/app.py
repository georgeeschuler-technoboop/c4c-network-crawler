"""
Insight Engine — Streamlit App

Structured insight from complex networks.
Reads exported data from OrgGraph US/CA projects.

UPDATED v0.5.0: Aligned with OrgGraph US/CA visual patterns
- Multi-project support (scans demo_data/)
- Loads pre-exported artifacts when available
- Consistent section headers and metric labels
- Clean separation: Network Results → Insight Cards → Downloads
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from io import BytesIO
import zipfile
import sys
import importlib.util

# =============================================================================
# Config
# =============================================================================

APP_VERSION = "0.5.0"
C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_ed8e76c8495d4799a5d7575822009e93~mv2.png"

# Get paths
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DEMO_DATA_DIR = REPO_ROOT / "demo_data"

st.set_page_config(
    page_title="Insight Engine",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)

# =============================================================================
# Dynamic Import of run.py (for compute fallback)
# =============================================================================

def load_run_module():
    """Load run.py module for computing insights when pre-exports don't exist."""
    run_path = APP_DIR / "run.py"
    if not run_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("run", run_path)
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)
        return run_module
    except Exception as e:
        st.warning(f"Could not load run.py: {e}")
        return None


# =============================================================================
# Project Discovery
# =============================================================================

def get_projects() -> list:
    """Discover available projects in demo_data/."""
    if not DEMO_DATA_DIR.exists():
        return []
    
    projects = []
    for item in DEMO_DATA_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check what files exist
            has_nodes = (item / "nodes.csv").exists()
            has_edges = (item / "edges.csv").exists()
            has_summary = (item / "project_summary.json").exists()
            has_cards = (item / "insight_cards.json").exists()
            has_report = (item / "insight_report.md").exists()
            has_metrics = (item / "node_metrics.csv").exists()
            
            if has_nodes or has_edges or has_summary:
                projects.append({
                    "id": item.name,
                    "path": item,
                    "has_nodes": has_nodes,
                    "has_edges": has_edges,
                    "has_summary": has_summary,
                    "has_cards": has_cards,
                    "has_report": has_report,
                    "has_metrics": has_metrics,
                    "is_complete": has_summary and has_cards and has_report,
                })
    
    # Sort: complete projects first, then alphabetically
    projects.sort(key=lambda x: (not x["is_complete"], x["id"].lower()))
    return projects


def load_project_data(project: dict) -> dict:
    """Load all available data for a project."""
    path = project["path"]
    data = {
        "project_id": project["id"],
        "nodes_df": None,
        "edges_df": None,
        "project_summary": None,
        "insight_cards": None,
        "markdown_report": None,
        "metrics_df": None,
    }
    
    # Load CSVs
    if project["has_nodes"]:
        data["nodes_df"] = pd.read_csv(path / "nodes.csv")
    if project["has_edges"]:
        data["edges_df"] = pd.read_csv(path / "edges.csv")
    if project["has_metrics"]:
        data["metrics_df"] = pd.read_csv(path / "node_metrics.csv")
    
    # Load JSON artifacts
    if project["has_summary"]:
        with open(path / "project_summary.json", "r") as f:
            data["project_summary"] = json.load(f)
    if project["has_cards"]:
        with open(path / "insight_cards.json", "r") as f:
            data["insight_cards"] = json.load(f)
    
    # Load markdown report
    if project["has_report"]:
        with open(path / "insight_report.md", "r") as f:
            data["markdown_report"] = f.read()
    
    return data


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    if "current_project" not in st.session_state:
        st.session_state.current_project = None
    if "project_data" not in st.session_state:
        st.session_state.project_data = None


# =============================================================================
# Rendering Functions (Aligned with OrgGraph US/CA)
# =============================================================================

def render_network_results(summary: dict):
    """
    Render Network Results from project_summary.json.
    Uses same icons/labels as OrgGraph US/CA.
    """
    st.subheader("📊 Network Results")
    st.caption("These metrics describe the merged exported network for this project.")
    
    # Row 1: Node and Edge counts
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏛️ Organizations", f"{summary['node_counts']['organizations']:,}")
    col2.metric("👤 People", f"{summary['node_counts']['people']:,}")
    col3.metric("💰 Grant Edges", f"{summary['edge_counts']['grants']:,}")
    col4.metric("🪪 Board Edges", f"{summary['edge_counts']['board_memberships']:,}")
    
    # Row 2: Funding metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Total Funding", f"${summary['funding']['total_amount']:,.0f}")
    col2.metric("🎁 Funders", summary['funding']['funder_count'])
    col3.metric("🎯 Grantees", summary['funding']['grantee_count'])
    
    # Governance metric if available
    if "governance" in summary and "multi_board_people" in summary["governance"]:
        col4.metric("🔗 Multi-Board People", summary['governance']['multi_board_people'])
    
    # Top funders share if available
    if "funding" in summary and "top5_share" in summary["funding"]:
        top5 = summary["funding"]["top5_share"]
        st.caption(f"*Top 5 funders account for {top5:.1%} of total funding*")


def render_health_banner(insight_cards: dict):
    """Render the Network Health banner."""
    health = insight_cards.get("health", {})
    health_score = health.get("score", 0)
    health_label = health.get("label", "Unknown")
    
    # Color based on score
    if health_score >= 70:
        health_color = "🟢"
    elif health_score >= 40:
        health_color = "🟡"
    else:
        health_color = "🔴"
    
    st.markdown(f"### {health_color} Network Health: **{health_score}/100** — *{health_label}*")
    st.caption("This score reflects funder coordination, governance ties, and funding concentration.")
    
    # Health factors in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Positive Factors**")
        for f in health.get("positive", []):
            st.markdown(f"- {f}")
        if not health.get("positive"):
            st.caption("*No strong positive factors*")
    with col2:
        st.markdown("**⚠️ Risk Factors**")
        for f in health.get("risk", []):
            st.markdown(f"- {f}")
        if not health.get("risk"):
            st.caption("*No significant risks*")


def render_insight_card(card: dict):
    """Render a single insight card with narratives."""
    with st.container():
        st.markdown(f"#### {card['title']}")
        st.caption(f"Use Case: {card['use_case']}")
        
        # Render summary
        st.markdown(card['summary'])
        
        # Health factors (special handling for network health card)
        if "health_factors" in card and card["health_factors"]:
            factors = card["health_factors"]
            if factors.get("positive") or factors.get("risk"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Positive Factors**")
                    for f in factors.get("positive", []):
                        st.markdown(f"- {f}")
                with col2:
                    st.markdown("**⚠️ Risk Factors**")
                    for f in factors.get("risk", []):
                        st.markdown(f"- {f}")
        
        # Ranked rows with narratives
        ranked_rows = card.get("ranked_rows", [])
        if ranked_rows:
            has_narratives = any(r.get("narrative") for r in ranked_rows)
            has_interpretations = any(r.get("interpretation") for r in ranked_rows)
            
            if has_interpretations and not has_narratives:
                # Health-style indicators
                st.markdown("---")
                for row in ranked_rows:
                    indicator = row.get("indicator", "")
                    value = row.get("value", "")
                    interpretation = row.get("interpretation", "")
                    st.markdown(f"**{indicator}:** {value}")
                    if interpretation:
                        st.caption(f"↳ {interpretation}")
                    
            elif has_narratives:
                # Render as expandable narrative sections
                for row in ranked_rows:
                    entity_name = (
                        row.get("grantee") or 
                        row.get("person") or 
                        row.get("org") or 
                        row.get("funder") or 
                        row.get("node") or 
                        row.get("pair") or
                        f"#{row.get('rank', '')}"
                    )
                    
                    with st.expander(f"**{row.get('rank', '')}. {entity_name}**"):
                        if row.get("narrative"):
                            st.markdown(row["narrative"])
                        if row.get("recommendation"):
                            st.markdown(row["recommendation"])
                        
                        # Show key metrics
                        metrics_to_show = ["funders", "boards", "amount", "outflow", "shared", "jaccard", "betweenness"]
                        metric_parts = []
                        for m in metrics_to_show:
                            if m in row:
                                val = row[m]
                                if isinstance(val, float):
                                    val = f"{val:.3f}" if val < 1 else f"{val:,.0f}"
                                metric_parts.append(f"**{m.title()}:** {val}")
                        if metric_parts:
                            st.caption(" • ".join(metric_parts))
            else:
                # Render as simple table
                df = pd.DataFrame(ranked_rows)
                for col in df.columns:
                    if "amount" in col.lower() or "received" in col.lower() or "outflow" in col.lower():
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                
                display_cols = [c for c in df.columns if not c.endswith("_id") and not c.endswith("_ids") and c != "rank"]
                if display_cols:
                    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        
        # Evidence summary
        evidence = card.get("evidence", {})
        node_count = len(evidence.get("node_ids", []))
        edge_count = len(evidence.get("edge_ids", []))
        if node_count > 0 or edge_count > 0:
            st.caption(f"📎 Evidence: {node_count} nodes, {edge_count} edges")
        
        st.divider()


def render_insight_cards_section(insight_cards: dict):
    """Render the Insight Cards section."""
    st.subheader("🧠 Insight Cards")
    st.caption("Cards are generated from the exported network metrics. Each card is a narrative interpretation with ranked evidence.")
    
    cards = insight_cards.get("cards", [])
    
    if not cards:
        st.info("No insight cards available for this project.")
        return
    
    # Card filter
    use_cases = list(set(c["use_case"] for c in cards))
    selected_use_case = st.selectbox(
        "Filter by Use Case",
        ["All"] + sorted(use_cases),
        index=0
    )
    
    filtered_cards = cards if selected_use_case == "All" else [c for c in cards if c["use_case"] == selected_use_case]
    
    for card in filtered_cards:
        render_insight_card(card)


def render_node_metrics(metrics_df: pd.DataFrame):
    """Render node metrics in expanders (aligned with OrgGraph pattern)."""
    with st.expander("👀 Preview Node Metrics", expanded=False):
        tab1, tab2 = st.tabs(["Organizations", "People"])
        
        with tab1:
            org_metrics = metrics_df[metrics_df["node_type"] == "ORG"]
            display_cols = ["label", "degree", "grant_in_degree", "grant_out_degree", 
                          "grant_outflow_total", "betweenness", "shared_board_count",
                          "is_connector", "is_broker", "is_hidden_broker", "is_capital_hub"]
            display_cols = [c for c in display_cols if c in org_metrics.columns]
            
            if not org_metrics.empty and display_cols:
                sort_col = "grant_outflow_total" if "grant_outflow_total" in display_cols else display_cols[0]
                st.dataframe(
                    org_metrics[display_cols].sort_values(sort_col, ascending=False), 
                    use_container_width=True, 
                    hide_index=True
                )
                st.caption(f"{len(org_metrics)} organizations")
            else:
                st.info("No organization metrics available")
        
        with tab2:
            person_metrics = metrics_df[metrics_df["node_type"] == "PERSON"]
            display_cols = ["label", "boards_served", "betweenness", "is_connector"]
            display_cols = [c for c in display_cols if c in person_metrics.columns]
            
            if not person_metrics.empty and display_cols:
                sort_col = "boards_served" if "boards_served" in display_cols else display_cols[0]
                st.dataframe(
                    person_metrics[display_cols].sort_values(sort_col, ascending=False), 
                    use_container_width=True, 
                    hide_index=True
                )
                st.caption(f"{len(person_metrics)} people")
            else:
                st.info("No people metrics available")


def render_downloads(data: dict):
    """Render download section (aligned with OrgGraph pattern)."""
    st.subheader("📥 Downloads")
    st.caption("These are the exact artifacts produced for this project export.")
    
    # Check what's available
    has_report = data.get("markdown_report") is not None
    has_cards = data.get("insight_cards") is not None
    has_summary = data.get("project_summary") is not None
    has_metrics = data.get("metrics_df") is not None
    
    def create_zip():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if has_metrics:
                zf.writestr("node_metrics.csv", data["metrics_df"].to_csv(index=False))
            if has_cards:
                zf.writestr("insight_cards.json", json.dumps(data["insight_cards"], indent=2))
            if has_summary:
                zf.writestr("project_summary.json", json.dumps(data["project_summary"], indent=2))
            if has_report:
                zf.writestr("insight_report.md", data["markdown_report"])
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # Primary downloads
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if has_report:
            st.download_button(
                "📝 Download Insight Report (Markdown)",
                data=data["markdown_report"],
                file_name="insight_report.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("No insight report available")
    
    with col2:
        if has_report or has_cards or has_summary or has_metrics:
            st.download_button(
                "📦 Download All (ZIP)",
                data=create_zip(),
                file_name=f"{data['project_id']}_insights.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    # Individual files
    st.caption("Individual data files:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if has_metrics:
            st.download_button(
                "📄 node_metrics.csv",
                data=data["metrics_df"].to_csv(index=False),
                file_name="node_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if has_cards:
            st.download_button(
                "📄 insight_cards.json",
                data=json.dumps(data["insight_cards"], indent=2),
                file_name="insight_cards.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        if has_summary:
            st.download_button(
                "📄 project_summary.json",
                data=json.dumps(data["project_summary"], indent=2),
                file_name="project_summary.json",
                mime="application/json",
                use_container_width=True
            )


def render_technical_details(data: dict):
    """Render technical details in expander."""
    with st.expander("🛠️ Technical Details", expanded=False):
        st.caption("Developer-facing details for troubleshooting.")
        
        if data.get("project_summary"):
            st.markdown("**project_summary.json:**")
            st.json(data["project_summary"])
        
        if data.get("insight_cards"):
            st.markdown("**insight_cards.json (health + metadata):**")
            cards_preview = {
                "health": data["insight_cards"].get("health", {}),
                "card_count": len(data["insight_cards"].get("cards", [])),
                "generated_at": data["insight_cards"].get("generated_at", "unknown"),
            }
            st.json(cards_preview)


# =============================================================================
# Compute Fallback (when pre-exports don't exist)
# =============================================================================

def compute_insights(project: dict, project_id: str) -> dict:
    """Compute insights from nodes/edges when pre-exports don't exist."""
    run_module = load_run_module()
    if not run_module:
        st.error("Cannot compute insights: run.py not found")
        return None
    
    path = project["path"]
    nodes_path = path / "nodes.csv"
    edges_path = path / "edges.csv"
    
    with st.spinner("Computing insights from network data..."):
        try:
            # Load and validate
            nodes_df, edges_df = run_module.load_and_validate(nodes_path, edges_path)
            
            # Build graphs
            grant_graph = run_module.build_grant_graph(nodes_df, edges_df)
            board_graph = run_module.build_board_graph(nodes_df, edges_df)
            interlock_graph = run_module.build_interlock_graph(nodes_df, edges_df)
            
            # Compute metrics
            metrics_df = run_module.compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
            metrics_df = run_module.compute_derived_signals(metrics_df)
            
            # Flow stats
            flow_stats = run_module.compute_flow_stats(edges_df, metrics_df)
            overlap_df = run_module.compute_portfolio_overlap(edges_df)
            
            # Generate insights
            insight_cards = run_module.generate_insight_cards(
                nodes_df, edges_df, metrics_df,
                interlock_graph, flow_stats, overlap_df,
                project_id=project_id
            )
            
            # Project summary
            project_summary = run_module.generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
            
            # Generate markdown report
            markdown_report = run_module.generate_markdown_report(insight_cards, project_summary, project_id)
            
            return {
                "project_id": project_id,
                "nodes_df": nodes_df,
                "edges_df": edges_df,
                "metrics_df": metrics_df,
                "insight_cards": insight_cards,
                "project_summary": project_summary,
                "markdown_report": markdown_report,
            }
            
        except Exception as e:
            st.error(f"Error computing insights: {e}")
            return None


# =============================================================================
# Main App
# =============================================================================

def main():
    init_session_state()
    
    # ==========================================================================
    # Header
    # ==========================================================================
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("Insight Engine")
    
    st.markdown("Structured insight from complex networks.")
    st.caption(f"v{APP_VERSION}")
    
    st.divider()
    
    # ==========================================================================
    # Section 1: Project Selection
    # ==========================================================================
    st.subheader("🗂️ Project")
    st.caption("Select a prepared demo dataset (exported from OrgGraph). Insight Engine reads the exported network + metrics.")
    
    projects = get_projects()
    
    if not projects:
        st.warning(f"""
        **No projects found.**
        
        Expected location: `{DEMO_DATA_DIR}`
        
        Each project folder should contain at minimum `nodes.csv` and `edges.csv`.
        For full functionality, also include: `project_summary.json`, `insight_cards.json`, `insight_report.md`, `node_metrics.csv`
        """)
        st.stop()
    
    # Build project selector
    project_options = []
    for p in projects:
        status = "✅" if p["is_complete"] else "⚠️"
        project_options.append(f"{status} {p['id']}")
    
    selected_option = st.selectbox(
        "Select project",
        project_options,
        label_visibility="collapsed"
    )
    
    # Find selected project
    selected_id = selected_option.split(" ", 1)[1]  # Remove status emoji
    selected_project = next((p for p in projects if p["id"] == selected_id), None)
    
    if not selected_project:
        st.error("Project not found")
        st.stop()
    
    # Show project status
    if selected_project["is_complete"]:
        st.success(f"📂 **{selected_id}** — Complete dataset (pre-computed insights available)")
    else:
        missing = []
        if not selected_project["has_summary"]:
            missing.append("project_summary.json")
        if not selected_project["has_cards"]:
            missing.append("insight_cards.json")
        if not selected_project["has_report"]:
            missing.append("insight_report.md")
        st.info(f"📂 **{selected_id}** — Missing: {', '.join(missing)}")
    
    # Load or compute data
    if selected_project["is_complete"]:
        # Load pre-exported data
        data = load_project_data(selected_project)
    else:
        # Need to compute
        col1, col2 = st.columns([1, 3])
        with col1:
            compute_btn = st.button("🚀 Compute Insights", type="primary")
        
        if compute_btn:
            data = compute_insights(selected_project, selected_id)
            if data:
                st.session_state.project_data = data
                st.rerun()
        
        if st.session_state.project_data and st.session_state.project_data.get("project_id") == selected_id:
            data = st.session_state.project_data
        else:
            st.info("Click 'Compute Insights' to generate analysis from the network data.")
            st.stop()
    
    st.divider()
    
    # ==========================================================================
    # Section 2: Network Results (from project_summary.json)
    # ==========================================================================
    if data.get("project_summary"):
        render_network_results(data["project_summary"])
        st.divider()
    
    # ==========================================================================
    # Section 3: Health Banner + Insight Cards
    # ==========================================================================
    if data.get("insight_cards"):
        render_health_banner(data["insight_cards"])
        st.divider()
        render_insight_cards_section(data["insight_cards"])
        st.divider()
    
    # ==========================================================================
    # Section 4: Node Metrics (optional)
    # ==========================================================================
    if data.get("metrics_df") is not None and not data["metrics_df"].empty:
        render_node_metrics(data["metrics_df"])
        st.divider()
    
    # ==========================================================================
    # Section 5: Downloads
    # ==========================================================================
    render_downloads(data)
    
    st.divider()
    
    # ==========================================================================
    # Section 6: Technical Details
    # ==========================================================================
    render_technical_details(data)


if __name__ == "__main__":
    main()
