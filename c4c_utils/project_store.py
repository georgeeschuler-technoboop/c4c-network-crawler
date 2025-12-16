"""
c4c_utils/project_store.py

Unified project config storage.

Config lives at: demo_data/{project_id}/config.json

This replaces the separate regions_catalog.py approach with a unified
per-project config that can hold region settings and future project-level options.

Usage:
    from c4c_utils.project_store import (
        list_projects,
        load_project_config,
        save_project_config,
        get_region_from_config,
    )
    
    # List available projects
    projects = list_projects()  # Returns [("great_lakes_demo", "Great Lakes Demo"), ...]
    
    # Load config for a project
    cfg = load_project_config("great_lakes_demo")
    
    # Save config
    save_project_config("great_lakes_demo", cfg)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default base directory for projects
DEFAULT_BASE_DIR = "demo_data"


# =============================================================================
# Core IO Functions
# =============================================================================

def list_projects(base_dir: str = DEFAULT_BASE_DIR) -> List[Tuple[str, str]]:
    """
    Discover projects from subdirectories under base_dir.
    
    Returns:
        List of (project_id, project_name) tuples, sorted by name.
        project_id is the directory name.
        project_name comes from config.json or is derived from directory name.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    projects = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            project_id = item.name
            
            # Try to get display name from config
            config_path = item / "config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text(encoding="utf-8"))
                    project_name = cfg.get("project_name", _id_to_display_name(project_id))
                except Exception:
                    project_name = _id_to_display_name(project_id)
            else:
                project_name = _id_to_display_name(project_id)
            
            projects.append((project_id, project_name))
    
    return sorted(projects, key=lambda x: x[1].lower())


def load_project_config(project_id: str, base_dir: str = DEFAULT_BASE_DIR) -> Dict[str, Any]:
    """
    Load project config from demo_data/{project_id}/config.json.
    
    If config.json doesn't exist, returns default config with region_filter.mode="off".
    
    Also handles legacy migration: if regions.json exists but config.json doesn't,
    attempts to import region settings from regions.json.
    """
    base_path = Path(base_dir)
    project_path = base_path / project_id
    config_path = project_path / "config.json"
    legacy_regions_path = project_path / "regions.json"
    
    # If config.json exists, load it
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    # Create default config
    cfg = default_project_config(project_id)
    
    # Legacy migration: check for regions.json
    if legacy_regions_path.exists():
        try:
            legacy = json.loads(legacy_regions_path.read_text(encoding="utf-8"))
            cfg = _migrate_legacy_regions(cfg, legacy)
        except Exception:
            pass
    
    return cfg


def save_project_config(project_id: str, cfg: Dict[str, Any], base_dir: str = DEFAULT_BASE_DIR) -> None:
    """
    Save project config to demo_data/{project_id}/config.json.
    
    Creates the project directory if it doesn't exist.
    """
    base_path = Path(base_dir)
    project_path = base_path / project_id
    config_path = project_path / "config.json"
    
    # Ensure directory exists
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure project_id is set in config
    cfg["project_id"] = project_id
    
    # Write config
    config_path.write_text(
        json.dumps(cfg, indent=2, sort_keys=True),
        encoding="utf-8"
    )


def project_exists(project_id: str, base_dir: str = DEFAULT_BASE_DIR) -> bool:
    """Check if a project directory exists."""
    return (Path(base_dir) / project_id).is_dir()


# =============================================================================
# Default Config
# =============================================================================

def default_project_config(project_id: str) -> Dict[str, Any]:
    """
    Create default project config.
    
    IMPORTANT: New projects default to region_filter.mode="off".
    This ensures no region tagging is applied unless explicitly set.
    """
    return {
        "project_id": project_id,
        "project_name": _id_to_display_name(project_id),
        "region_filter": {
            "mode": "off",  # "off", "preset", or "custom"
            "preset_key": "",
            "custom_admin1_codes": [],
            "custom_country_codes": [],
        },
        # Room for future project-level settings:
        # "parsing": {},
        # "outputs": {},
    }


# =============================================================================
# Region Helpers
# =============================================================================

def get_region_from_config(cfg: Dict[str, Any], region_presets: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a region definition from project config.
    
    Args:
        cfg: Project config dict
        region_presets: Dict of preset regions (from regions_presets.py)
    
    Returns:
        Region definition dict, or None if mode is "off"
    """
    rf = cfg.get("region_filter", {})
    mode = rf.get("mode", "off")
    
    if mode == "off":
        return None
    
    if mode == "preset":
        preset_key = rf.get("preset_key", "")
        if preset_key and preset_key in region_presets:
            return region_presets[preset_key]
        return None
    
    if mode == "custom":
        admin1_codes = set(rf.get("custom_admin1_codes", []) or [])
        country_codes = set(rf.get("custom_country_codes", []) or [])
        
        if not admin1_codes and not country_codes:
            return None
        
        # Build region definition in the format expected by region_tagger
        us_states = [c for c in admin1_codes if len(c) == 2 and c.isalpha() and c.upper() == c]
        ca_provinces = [c for c in admin1_codes if c in {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}]
        
        # Filter to only include actual US states vs CA provinces
        from c4c_utils.regions_presets import US_STATES, CA_PROVINCES
        us_states = [c for c in admin1_codes if c in US_STATES]
        ca_provinces = [c for c in admin1_codes if c in CA_PROVINCES]
        
        return {
            "id": "custom",
            "name": cfg.get("project_name", "Custom") + " Region",
            "source": "project_config",
            "include_us_states": us_states,
            "include_ca_provinces": ca_provinces,
            "include_countries": list(country_codes),
            "notes": "Custom region from project config",
        }
    
    return None


def update_region_in_config(
    cfg: Dict[str, Any],
    mode: str,
    preset_key: str = "",
    custom_admin1_codes: List[str] = None,
    custom_country_codes: List[str] = None,
) -> Dict[str, Any]:
    """
    Update region_filter in project config.
    
    Args:
        cfg: Project config dict
        mode: "off", "preset", or "custom"
        preset_key: Key for preset region (if mode="preset")
        custom_admin1_codes: List of state/province codes (if mode="custom")
        custom_country_codes: List of country codes (if mode="custom")
    
    Returns:
        Updated config dict
    """
    cfg["region_filter"] = {
        "mode": mode,
        "preset_key": preset_key if mode == "preset" else "",
        "custom_admin1_codes": list(custom_admin1_codes or []) if mode == "custom" else [],
        "custom_country_codes": list(custom_country_codes or []) if mode == "custom" else [],
    }
    return cfg


# =============================================================================
# Internal Helpers
# =============================================================================

def _id_to_display_name(project_id: str) -> str:
    """Convert project_id to human-readable display name."""
    return project_id.replace("_", " ").replace("-", " ").title()


def _migrate_legacy_regions(cfg: Dict[str, Any], legacy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy regions.json format to unified config.
    
    Legacy format stored custom regions as a dict of region definitions.
    We'll import the first one if it exists.
    """
    if not legacy:
        return cfg
    
    # Legacy format: {"region_id": {"name": ..., "include_us_states": [...], ...}}
    # Try to find a region to import
    for region_id, region_def in legacy.items():
        if isinstance(region_def, dict):
            us_states = region_def.get("include_us_states", [])
            ca_provinces = region_def.get("include_ca_provinces", [])
            
            if us_states or ca_provinces:
                cfg["region_filter"] = {
                    "mode": "custom",
                    "preset_key": "",
                    "custom_admin1_codes": list(us_states) + list(ca_provinces),
                    "custom_country_codes": ["US", "CA"] if ca_provinces else ["US"],
                }
                cfg["project_name"] = region_def.get("name", cfg.get("project_name", ""))
                break
    
    return cfg


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_DIR
    
    print(f"\n📁 Projects in {base_dir}/")
    print("=" * 50)
    
    projects = list_projects(base_dir)
    
    if not projects:
        print("  (no projects found)")
    else:
        for pid, pname in projects:
            cfg = load_project_config(pid, base_dir)
            rf = cfg.get("region_filter", {})
            mode = rf.get("mode", "off")
            
            mode_display = {
                "off": "🔘 Off",
                "preset": f"📍 Preset: {rf.get('preset_key', '')}",
                "custom": f"🎯 Custom: {len(rf.get('custom_admin1_codes', []))} codes",
            }.get(mode, mode)
            
            print(f"  [{pid}] {pname}")
            print(f"      Region: {mode_display}")
