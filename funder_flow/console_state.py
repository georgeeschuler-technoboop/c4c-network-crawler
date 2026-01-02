"""
Console State Helpers for OrgGraph US
Compute live console lines from session state.

VERSION HISTORY:
v1.0.0 (2026-01-01): Initial version for console-driven flow
"""

from typing import List, Tuple, Optional
import streamlit as st


def get_project_store_client():
    """Get project store client if available."""
    return st.session_state.get("project_store")


def get_cloud_status() -> Tuple[bool, str]:
    """
    Check cloud login status.
    
    Returns:
        (is_logged_in, status_text)
    """
    client = get_project_store_client()
    if client and hasattr(client, 'is_authenticated') and client.is_authenticated():
        user = client.get_current_user()
        if user:
            email = user.get('email', 'Connected')
            return True, f"☁️ {email}"
        return True, "☁️ Connected"
    return False, "☁️ Offline"


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
        # Clean up display name
        display_name = project.replace('_', ' ').title() if project != '_demo' else 'Demo'
        lines.append(f"✓ Project: {display_name}")
        current_stage = 'data'
    else:
        lines.append("● Select or create a project to begin")
        return lines, current_stage
    
    # --- Stage 2: Data ---
    parse_results = st.session_state.get('parse_results', [])
    
    if len(parse_results) > 0:
        lines.append(f"✓ Data: {len(parse_results)} return(s) loaded")
        current_stage = 'process'
    else:
        lines.append("○ Upload 990-PF files or use demo data")
        return lines, current_stage
    
    # --- Stage 3: Processing ---
    nodes_df = st.session_state.get('nodes_df')
    edges_df = st.session_state.get('edges_df')
    processed = st.session_state.get('processed', False)
    
    if processed and nodes_df is not None and len(nodes_df) > 0:
        node_count = len(nodes_df)
        edge_count = len(edges_df) if edges_df is not None else 0
        lines.append(f"✓ Network: {node_count} nodes, {edge_count} edges")
        current_stage = 'export'
    else:
        lines.append("○ Ready to process")
        return lines, current_stage
    
    # --- Stage 4: Export ready ---
    lines.append("● Ready to export bundle")
    
    return lines, current_stage


def render_live_console() -> str:
    """
    Render the console with live state.
    Call this instead of the static c4c_console().
    
    Returns:
        current_stage for flow control
    """
    from console_ui import c4c_console
    
    lines, stage = get_console_state()
    c4c_console("Console", lines)
    
    return stage


def is_stage_complete(stage_name: str, current_stage: str) -> bool:
    """Check if a stage is complete (current stage is past it)."""
    stage_order = ['project', 'data', 'process', 'export', 'done']
    
    try:
        stage_idx = stage_order.index(stage_name)
        current_idx = stage_order.index(current_stage)
        return stage_idx < current_idx
    except ValueError:
        return False


def should_show_stage(stage_name: str, current_stage: str) -> bool:
    """
    Determine if a stage should be visible.
    
    Stages: project → data → process → export → done
    """
    stage_order = ['project', 'data', 'process', 'export', 'done']
    
    try:
        stage_idx = stage_order.index(stage_name)
        current_idx = stage_order.index(current_stage)
        return stage_idx <= current_idx
    except ValueError:
        return True
