"""
InsightGraph — Network Analysis Dispatcher

Detects network type and routes to the appropriate analyzer.
This is the main entry point for network analysis.

VERSION HISTORY:
----------------
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
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

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
