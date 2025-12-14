"""
Insight Cards — Streamlit App

Structured insight from complex networks.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from io import BytesIO
import zipfile
import sys
import importlib.util

# Get paths
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Direct import of run.py to avoid package import issues
run_path = APP_DIR / "run.py"
spec = importlib.util.spec_from_file_location("run", run_path)
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)

# Get functions from the module
load_and_validate = run_module.load_and_validate
build_grant_graph = run_module.build_grant_graph
build_board_graph = run_module.build_board_graph
build_interlock_graph = run_module.build_interlock_graph
compute_base_metrics = run_module.compute_base_metrics
compute_derived_signals = run_module.compute_derived_signals
compute_flow_stats = run_module.compute_flow_stats
compute_portfolio_overlap = run_module.compute_portfolio_overlap
generate_insight_cards = run_module.generate_insight_cards
generate_project_summary = run_module.generate_project_summary
generate_markdown_report = run_module.generate_markdown_report

# =============================================================================
# Config
# =============================================================================

C4C_LOGO_URL = "https://static.wixstatic.com/media/275a3f_c94bb96428e843ab828bed6ea6c43ead~mv2.png"

GLFN_DATA_DIR = REPO_ROOT / "demo_data" / "glfn"

st.set_page_config(
    page_title="Insight Cards",
    page_icon=C4C_LOGO_URL,
    layout="wide"
)

# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    if "insights_run" not in st.session_state:
        st.session_state.insights_run = False
    if "metrics_df" not in st.session_state:
        st.session_state.metrics_df = None
    if "insight_cards" not in st.session_state:
        st.session_state.insight_cards = None
    if "project_summary" not in st.session_state:
        st.session_state.project_summary = None
    if "markdown_report" not in st.session_state:
        st.session_state.markdown_report = None


# =============================================================================
# Insight Card Rendering
# =============================================================================

def render_card(card: dict):
    """Render a single insight card with narratives."""
    with st.container():
        st.markdown(f"### {card['title']}")
        st.caption(f"Use Case: {card['use_case']}")
        
        # Render summary (supports markdown)
        st.markdown(card['summary'])
        
        # Health factors (special handling for network health card)
        if "health_factors" in card and card["health_factors"]:
            factors = card["health_factors"]
            if factors.get("positive") or factors.get("risk"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Positive Factors**")
                    for f in factors.get("positive", []):
                        st.markdown(f)
                with col2:
                    st.markdown("**⚠️ Risk Factors**")
                    for f in factors.get("risk", []):
                        st.markdown(f)
        
        # Ranked rows with narratives
        ranked_rows = card.get("ranked_rows", [])
        if ranked_rows:
            # Check what type of rows we have
            has_narratives = any(r.get("narrative") for r in ranked_rows)
            has_interpretations = any(r.get("interpretation") for r in ranked_rows)
            
            if has_interpretations and not has_narratives:
                # Health-style indicators: render as vertical metric blocks
                st.markdown("---")
                for row in ranked_rows:
                    indicator = row.get("indicator", "")
                    value = row.get("value", "")
                    interpretation = row.get("interpretation", "")
                    
                    # Metric with value prominent
                    st.markdown(f"**{indicator}:** {value}")
                    if interpretation:
                        st.caption(f"↳ {interpretation}")
                    st.markdown("")  # spacing
                    
            elif has_narratives:
                # Render as expandable narrative sections
                for row in ranked_rows:
                    # Determine the entity name
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
                # Render as simple table for non-narrative rows
                df = pd.DataFrame(ranked_rows)
                
                # Format currency columns
                for col in df.columns:
                    if "amount" in col.lower() or "received" in col.lower() or "outflow" in col.lower():
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                
                # Drop ID columns
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


# =============================================================================
# Main App
# =============================================================================

def main():
    init_session_state()
    
    # Header
    col_logo, col_title = st.columns([0.08, 0.92])
    with col_logo:
        st.image(C4C_LOGO_URL, width=60)
    with col_title:
        st.title("Insight Cards")
    
    st.markdown("Structured insight from complex networks.")
    st.caption(f"v{APP_VERSION}")
    
    st.divider()
    
    # Check for data
    nodes_path = GLFN_DATA_DIR / "nodes.csv"
    edges_path = GLFN_DATA_DIR / "edges.csv"
    
    if not nodes_path.exists() or not edges_path.exists():
        st.error(f"""
        **GLFN data not found.**
        
        Expected files:
        - `{nodes_path}`
        - `{edges_path}`
        
        Please ensure GLFN data is committed to the repo.
        """)
        st.stop()
    
    # Show data status
    nodes_df_raw = pd.read_csv(nodes_path)
    edges_df_raw = pd.read_csv(edges_path)
    
    st.success(f"📂 **GLFN data found:** {len(nodes_df_raw)} nodes, {len(edges_df_raw)} edges")
    
    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("🚀 Run Insights Engine", type="primary", use_container_width=True)
    with col2:
        project_id = st.text_input("Project ID", value="glfn", label_visibility="collapsed")
    
    if run_button:
        with st.spinner("Generating Insight Cards..."):
            try:
                # Load and validate
                nodes_df, edges_df = load_and_validate(nodes_path, edges_path)
                
                # Build graphs
                grant_graph = build_grant_graph(nodes_df, edges_df)
                board_graph = build_board_graph(nodes_df, edges_df)
                interlock_graph = build_interlock_graph(nodes_df, edges_df)
                
                # Compute metrics
                metrics_df = compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
                metrics_df = compute_derived_signals(metrics_df)
                
                # Flow stats
                flow_stats = compute_flow_stats(edges_df, metrics_df)
                overlap_df = compute_portfolio_overlap(edges_df)
                
                # Generate insights
                insight_cards = generate_insight_cards(
                    nodes_df, edges_df, metrics_df,
                    interlock_graph, flow_stats, overlap_df,
                    project_id=project_id
                )
                
                # Project summary
                project_summary = generate_project_summary(nodes_df, edges_df, metrics_df, flow_stats)
                
                # Generate markdown report
                markdown_report = generate_markdown_report(insight_cards, project_summary, project_id)
                
                # Store in session state
                st.session_state.insights_run = True
                st.session_state.metrics_df = metrics_df
                st.session_state.insight_cards = insight_cards
                st.session_state.project_summary = project_summary
                st.session_state.markdown_report = markdown_report
                
            except Exception as e:
                st.error(f"Error running insights: {e}")
                st.stop()
        
        st.rerun()
    
    # Display results if available
    if st.session_state.insights_run:
        st.divider()
        
        # Network Health Banner
        insight_data = st.session_state.insight_cards
        health = insight_data.get("health", {})
        health_score = health.get("score", 0)
        health_label = health.get("label", "Unknown")
        
        if health_score >= 70:
            health_color = "🟢"
        elif health_score >= 40:
            health_color = "🟡"
        else:
            health_color = "🔴"
        
        st.markdown(f"## {health_color} Network Health: **{health_score}/100** — *{health_label}*")
        st.caption("This score reflects funder coordination, governance ties, and funding concentration.")
        
        # Health factors in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Positive Factors**")
            for f in health.get("positive", []):
                st.markdown(f)
            if not health.get("positive"):
                st.caption("*No strong positive factors*")
        with col2:
            st.markdown("**⚠️ Risk Factors**")
            for f in health.get("risk", []):
                st.markdown(f)
            if not health.get("risk"):
                st.caption("*No significant risks*")
        
        st.divider()
        
        # Project Summary
        summary = st.session_state.project_summary
        st.subheader("📊 Project Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Organizations", summary["node_counts"]["organizations"])
        with col2:
            st.metric("People", summary["node_counts"]["people"])
        with col3:
            st.metric("Grants", summary["edge_counts"]["grants"])
        with col4:
            st.metric("Board Memberships", summary["edge_counts"]["board_memberships"])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Funding", f"${summary['funding']['total_amount']:,.0f}")
        with col2:
            st.metric("Funders", summary["funding"]["funder_count"])
        with col3:
            st.metric("Grantees", summary["funding"]["grantee_count"])
        with col4:
            st.metric("Multi-Board People", summary["governance"]["multi_board_people"])
        
        st.divider()
        
        # Insight Cards
        st.subheader("💡 Insight Cards")
        
        cards = st.session_state.insight_cards.get("cards", [])
        
        # Card filter
        use_cases = list(set(c["use_case"] for c in cards))
        selected_use_case = st.selectbox(
            "Filter by Use Case",
            ["All"] + sorted(use_cases),
            index=0
        )
        
        filtered_cards = cards if selected_use_case == "All" else [c for c in cards if c["use_case"] == selected_use_case]
        
        for card in filtered_cards:
            render_card(card)
        
        st.divider()
        
        # Node Metrics Preview
        st.subheader("📋 Node Metrics Preview")
        
        metrics_df = st.session_state.metrics_df
        
        tab1, tab2 = st.tabs(["Organizations", "People"])
        
        with tab1:
            org_metrics = metrics_df[metrics_df["node_type"] == "ORG"]
            display_cols = ["label", "degree", "grant_in_degree", "grant_out_degree", 
                          "grant_outflow_total", "betweenness", "shared_board_count",
                          "is_connector", "is_broker", "is_hidden_broker", "is_capital_hub"]
            display_cols = [c for c in display_cols if c in org_metrics.columns]
            st.dataframe(org_metrics[display_cols].sort_values("grant_outflow_total", ascending=False), 
                        use_container_width=True, hide_index=True)
        
        with tab2:
            person_metrics = metrics_df[metrics_df["node_type"] == "PERSON"]
            display_cols = ["label", "boards_served", "betweenness", "is_connector"]
            display_cols = [c for c in display_cols if c in person_metrics.columns]
            st.dataframe(person_metrics[display_cols].sort_values("boards_served", ascending=False), 
                        use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Downloads
        st.subheader("📥 Download Outputs")
        
        def create_zip():
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("node_metrics.csv", st.session_state.metrics_df.to_csv(index=False))
                zf.writestr("insight_cards.json", json.dumps(st.session_state.insight_cards, indent=2))
                zf.writestr("project_summary.json", json.dumps(st.session_state.project_summary, indent=2))
                zf.writestr("insight_report.md", st.session_state.markdown_report)
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
        
        # First row: Report download (prominent)
        col_report, col_zip = st.columns([2, 1])
        
        with col_report:
            st.download_button(
                "📝 Download Insight Report (Markdown)",
                data=st.session_state.markdown_report,
                file_name="insight_report.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        
        with col_zip:
            st.download_button(
                "📦 Download All (ZIP)",
                data=create_zip(),
                file_name="network_insights.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        # Second row: Individual files
        st.caption("Individual data files:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 node_metrics.csv",
                data=st.session_state.metrics_df.to_csv(index=False),
                file_name="node_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "📄 insight_cards.json",
                data=json.dumps(st.session_state.insight_cards, indent=2),
                file_name="insight_cards.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                "📄 project_summary.json",
                data=json.dumps(st.session_state.project_summary, indent=2),
                file_name="project_summary.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
