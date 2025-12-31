"""
InsightGraph — Network Analysis Dispatcher

Detects network type and routes to the appropriate analyzer.
This is the main entry point for network analysis.

VERSION HISTORY:
----------------
v4.0.1 (2025-12-31): Fixed imports for Streamlit Cloud compatibility
- Use absolute imports with sys.path manipulation
- Works when loaded via importlib (app.py dynamic loading)

v4.0.0 (2025-12-31): Network-type-aware architecture
- Detects network type from data: funder, social, or hybrid
- Routes to FunderAnalyzer or SocialAnalyzer
- Returns standardized AnalysisResult
- Backwards-compatible run() function

USAGE:
    from insights.run import run, analyze_network
    
    # Simple usage (auto-detect type)
    result = analyze_network(nodes_df, edges_df, project_id="my_project")
    
    # File-based usage (backwards compatible)
    summary, report = run(nodes_path, edges_path, output_dir, project_id)
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for dynamic loading compatibility
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import pandas as pd
from datetime import datetime, timezone

# Now import analyzers (works both as package and when dynamically loaded)
try:
    from analyzers import (
        detect_network_type,
        detect_source_app,
        FunderAnalyzer,
        SocialAnalyzer,
        AnalysisResult,
    )
except ImportError:
    # Fallback for package import
    from .analyzers import (
        detect_network_type,
        detect_source_app,
        FunderAnalyzer,
        SocialAnalyzer,
        AnalysisResult,
    )

# =============================================================================
# Version
# =============================================================================

ENGINE_VERSION = "4.0.0"
BUNDLE_FORMAT_VERSION = "1.1"

# =============================================================================
# Default Paths (for backwards compatibility)
# =============================================================================

DEFAULT_NODES = Path("demo_data/glfn/nodes.csv")
DEFAULT_EDGES = Path("demo_data/glfn/edges.csv")
DEFAULT_OUTPUT = Path("output/glfn")


# =============================================================================
# Main Entry Points
# =============================================================================

def analyze_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    project_id: str = "project"
) -> AnalysisResult:
    """
    Analyze a network with automatic type detection.
    
    This is the primary programmatic entry point.
    
    Args:
        nodes_df: DataFrame with node_id, node_type, label, and type-specific columns
        edges_df: DataFrame with edge_id, from_id, to_id, edge_type
        project_id: Project identifier
        
    Returns:
        AnalysisResult with health, cards, metrics, report, etc.
        
    Note:
        For unknown network types, uses SocialAnalyzer as fallback but
        sets health.score to indicate uncertainty and adds a warning.
    """
    # Detect network type
    network_type = detect_network_type(nodes_df, edges_df)
    source_app = detect_source_app(nodes_df)
    
    print(f"\n{'='*60}")
    print(f"C4C InsightGraph — Network Analysis")
    print(f"{'='*60}")
    print(f"\nNetwork type detected: {network_type}")
    print(f"Source app detected: {source_app}")
    print(f"Nodes: {len(nodes_df)}, Edges: {len(edges_df)}")
    
    # Route to appropriate analyzer
    if network_type == 'social':
        print("\n→ Using Social Network Analyzer")
        analyzer = SocialAnalyzer(nodes_df, edges_df, project_id)
    elif network_type == 'funder':
        print("\n→ Using Funder Network Analyzer")
        analyzer = FunderAnalyzer(nodes_df, edges_df, project_id)
    elif network_type == 'hybrid':
        # For now, use funder analyzer for hybrid networks
        # Future: create HybridAnalyzer that combines both
        print("\n→ Hybrid network detected, using Funder Network Analyzer")
        print("   (Hybrid analysis coming in Phase 5)")
        analyzer = FunderAnalyzer(nodes_df, edges_df, project_id)
    elif network_type == 'unknown':
        # Unknown type - warn user and use social as fallback
        print("\n⚠️  WARNING: Unknown network type detected")
        print("   No recognized edge_type found (grant, board, connection, etc.)")
        print("   Using Social Network Analyzer as fallback")
        print("   Results may be incomplete or inaccurate")
        analyzer = SocialAnalyzer(nodes_df, edges_df, project_id)
    else:
        # Defensive fallback
        print(f"\n⚠️  WARNING: Unexpected network type: {network_type}")
        print("   Using Social Network Analyzer as fallback")
        analyzer = SocialAnalyzer(nodes_df, edges_df, project_id)
    
    # Run analysis
    print("\nRunning analysis...")
    result = analyzer.analyze()
    
    # Add warning to health if unknown type
    if network_type == 'unknown':
        result.health.risk.insert(0, "⚠️ **Unknown network type** — analysis may be incomplete")
    
    print(f"\n✅ Analysis complete!")
    print(f"   Health: {result.health.score}/100 ({result.health.label})")
    print(f"   Cards: {len(result.cards)}")
    print(f"   Report: {len(result.markdown_report)} chars")
    
    return result


def load_and_validate(nodes_path: Path, edges_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate nodes and edges CSVs."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    # Basic validation
    required_node_cols = ['node_id']
    required_edge_cols = ['from_id', 'to_id']
    
    for col in required_node_cols:
        if col not in nodes_df.columns:
            raise ValueError(f"Missing required node column: {col}")
    
    for col in required_edge_cols:
        if col not in edges_df.columns:
            raise ValueError(f"Missing required edge column: {col}")
    
    # Add defaults if missing
    if 'node_type' not in nodes_df.columns:
        nodes_df['node_type'] = 'unknown'
    if 'label' not in nodes_df.columns:
        nodes_df['label'] = nodes_df['node_id']
    if 'edge_type' not in edges_df.columns:
        edges_df['edge_type'] = 'connection'
    
    print(f"✓ Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    return nodes_df, edges_df


def run(
    nodes_path: Path | str,
    edges_path: Path | str,
    output_dir: Path | str,
    project_id: str = "project"
) -> tuple[dict, str]:
    """
    Run analysis from file paths and write outputs.
    
    Backwards-compatible with original run.py interface.
    
    Args:
        nodes_path: Path to nodes.csv
        edges_path: Path to edges.csv
        output_dir: Directory for output files
        project_id: Project identifier
        
    Returns:
        (project_summary_dict, markdown_report_str)
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    output_dir = Path(output_dir)
    
    # Load data
    nodes_df, edges_df = load_and_validate(nodes_path, edges_path)
    
    # Run analysis
    result = analyze_network(nodes_df, edges_df, project_id)
    
    # Write outputs
    print("\nWriting outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Node metrics
    result.metrics_df.to_csv(output_dir / "node_metrics.csv", index=False)
    
    # Insight cards JSON
    with open(output_dir / "insight_cards.json", "w") as f:
        json.dump(result.to_insight_cards_dict(), f, indent=2)
    
    # Project summary JSON
    with open(output_dir / "project_summary.json", "w") as f:
        json.dump(result.to_project_summary_dict(), f, indent=2)
    
    # Markdown report
    with open(output_dir / "insight_report.md", "w") as f:
        f.write(result.markdown_report)
    
    print(f"\n✅ Done! Outputs in {output_dir}")
    print(f"   📄 node_metrics.csv")
    print(f"   📄 insight_cards.json")
    print(f"   📄 project_summary.json")
    print(f"   📄 insight_report.md")
    
    return result.to_project_summary_dict(), result.markdown_report


# =============================================================================
# HTML Report Rendering
# =============================================================================

def render_html_report(
    markdown_content: str = None,
    project_summary: dict = None,
    insight_cards: dict = None,
    project_id: str = "report"
) -> str:
    """
    Render HTML report from markdown or analysis outputs.
    
    Args:
        markdown_content: Pre-generated markdown report string
        project_summary: Project summary dict (fallback if no markdown)
        insight_cards: Insight cards dict (fallback if no markdown)
        project_id: Project identifier for title
        
    Returns:
        HTML string
    """
    # Try to use markdown library if available
    try:
        import markdown
        has_markdown = True
    except ImportError:
        has_markdown = False
        print("Warning: markdown library not installed. Using basic HTML conversion.")
    
    # Generate HTML from markdown report
    if markdown_content:
        if has_markdown:
            html_content = markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code']
            )
        else:
            # Basic conversion without markdown library
            html_content = _basic_markdown_to_html(markdown_content)
    else:
        # Fallback: generate basic HTML from project_summary
        html_content = f"<h1>{project_id} — Network Analysis Report</h1>"
        if project_summary:
            html_content += f"<p>Network Type: {project_summary.get('network_type', 'unknown')}</p>"
            node_counts = project_summary.get('node_counts', {})
            html_content += f"<p>Nodes: {node_counts.get('total', 0)}</p>"
    
    # Wrap in full HTML document with styling
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_id} — InsightGraph Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #1a1a2e; border-bottom: 2px solid #4a90d9; padding-bottom: 0.5rem; }}
        h2 {{ color: #16213e; margin-top: 2rem; }}
        h3 {{ color: #0f3460; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 0.75rem;
            text-align: left;
        }}
        th {{ background-color: #f5f5f5; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
        code {{
            background: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        blockquote {{
            border-left: 4px solid #4a90d9;
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            background: #f8f9fa;
        }}
        .health-score {{
            font-size: 2rem;
            font-weight: bold;
            color: #4a90d9;
        }}
        @media print {{
            body {{ padding: 1rem; }}
        }}
    </style>
</head>
<body>
{html_content}
<footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee; color: #666; font-size: 0.9rem;">
    Generated by InsightGraph • {datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</body>
</html>"""
    
    return html


def _basic_markdown_to_html(md: str) -> str:
    """Basic markdown to HTML conversion without external libraries."""
    import re
    
    html = md
    
    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Code
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
    
    # Links
    html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
    
    # Line breaks (preserve paragraph structure)
    html = re.sub(r'\n\n', '</p><p>', html)
    html = f'<p>{html}</p>'
    
    # Clean up empty paragraphs
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'<p>(<h[123]>)', r'\1', html)
    html = re.sub(r'(</h[123]>)</p>', r'\1', html)
    
    return html


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="C4C InsightGraph — Network Analysis & Briefing Generator"
    )
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES,
                        help="Path to nodes.csv")
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES,
                        help="Path to edges.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory")
    parser.add_argument("--project", type=str, default="project",
                        help="Project ID")
    
    args = parser.parse_args()
    
    summary, report = run(args.nodes, args.edges, args.out, args.project)
    
    print(f"\nNodes: {summary['node_counts']['total']}")
    print(f"Network Type: {summary.get('network_type', 'unknown')}")


if __name__ == "__main__":
    main()
