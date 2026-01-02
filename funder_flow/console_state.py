"""
Console State Helpers for OrgGraph US
Compute live console lines from session state.

Add to app.py or import as needed.
"""

from typing import List, Tuple
import streamlit as st


def get_console_state() -> Tuple[List[str], str]:
    """
    Compute console lines and current stage from session state.
    
    Returns:
        (lines, current_stage) where current_stage is one of:
        'project', 'data', 'process', 'export', 'done'
    """
    lines = []
    current_stage = 'project'
    
    # --- Stage 1: Project ---
    project = st.session_state.get('current_project')
    if project:
        lines.append(f"✓ Project: {project}")
        current_stage = 'data'
    else:
        lines.append("● Waiting for project selection...")
        return lines, current_stage
    
    # --- Stage 2: Data ---
    # Check for uploaded files or existing data
    has_files = st.session_state.get('uploaded_files') or st.session_state.get('parsed_returns')
    file_count = len(st.session_state.get('parsed_returns', []))
    
    if file_count > 0:
        lines.append(f"✓ Data: {file_count} return(s) loaded")
        current_stage = 'process'
    elif has_files:
        lines.append("● Processing uploaded files...")
        return lines, current_stage
    else:
        lines.append("○ Awaiting data upload")
        return lines, current_stage
    
    # --- Stage 3: Processing ---
    nodes_df = st.session_state.get('nodes_df')
    edges_df = st.session_state.get('edges_df')
    
    if nodes_df is not None and len(nodes_df) > 0:
        node_count = len(nodes_df)
        edge_count = len(edges_df) if edges_df is not None else 0
        lines.append(f"✓ Network: {node_count} nodes, {edge_count} edges")
        current_stage = 'export'
    else:
        lines.append("○ Ready to run ingestion")
        return lines, current_stage
    
    # --- Stage 4: Export ---
    last_export = st.session_state.get('last_export_time')
    if last_export:
        lines.append(f"✓ Exported: {last_export}")
        current_stage = 'done'
    else:
        lines.append("● Ready to export bundle")
    
    return lines, current_stage


def get_cloud_status() -> Tuple[bool, str]:
    """
    Check cloud login status.
    
    Returns:
        (is_logged_in, status_text)
    """
    user = st.session_state.get('user')
    if user:
        email = user.get('email', 'Connected')
        return True, f"☁️ {email}"
    return False, "☁️ Not connected"


def render_live_console():
    """
    Render the console with live state.
    Call this instead of the static c4c_console().
    """
    from console_ui import c4c_console
    
    lines, stage = get_console_state()
    
    # Add cloud status
    is_cloud, cloud_text = get_cloud_status()
    if is_cloud:
        lines.insert(0, cloud_text)
    
    c4c_console("Console", lines)
    
    return stage  # Return current stage for flow control


# =============================================================================
# Stage visibility helpers
# =============================================================================

def should_show_stage(stage_name: str, current_stage: str) -> bool:
    """
    Determine if a stage should be visible/expanded.
    
    Stages: project → data → process → export → done
    """
    stage_order = ['project', 'data', 'process', 'export', 'done']
    
    try:
        stage_idx = stage_order.index(stage_name)
        current_idx = stage_order.index(current_stage)
        
        # Show current and previous stages
        return stage_idx <= current_idx
    except ValueError:
        return True  # Unknown stage, show it


def is_stage_complete(stage_name: str, current_stage: str) -> bool:
    """Check if a stage is complete (current stage is past it)."""
    stage_order = ['project', 'data', 'process', 'export', 'done']
    
    try:
        stage_idx = stage_order.index(stage_name)
        current_idx = stage_order.index(current_stage)
        return stage_idx < current_idx
    except ValueError:
        return False
