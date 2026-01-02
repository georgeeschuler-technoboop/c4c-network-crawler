"""
Console State Helpers for OrgGraph US
Cloud status helper for header display.

VERSION HISTORY:
v1.1.0 (2026-01-01): Simplified - removed unused console functions
v1.0.0 (2026-01-01): Initial version
"""

from typing import Tuple
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
