# regions_catalog.py
"""
Project Regions Catalog for OrgGraph
====================================
Load/save custom project-defined regions to JSON file.
Regions persist across sessions and can be shared across projects.
"""

import json
from pathlib import Path
from typing import Dict, Any

# Default location for project regions file
PROJECT_REGIONS_PATH = Path("orggraph_project_regions.json")


def load_project_regions(path: Path = None) -> Dict[str, Dict[str, Any]]:
    """
    Load project-defined regions from JSON file.
    
    Args:
        path: Optional custom path to regions file
        
    Returns:
        Dict mapping region_id -> RegionDef dict
    """
    file_path = path or PROJECT_REGIONS_PATH
    
    if not file_path.exists():
        return {}
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate structure
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_project_regions(regions: Dict[str, Dict[str, Any]], path: Path = None) -> bool:
    """
    Save project-defined regions to JSON file.
    
    Args:
        regions: Dict mapping region_id -> RegionDef dict
        path: Optional custom path to regions file
        
    Returns:
        True if saved successfully, False otherwise
    """
    file_path = path or PROJECT_REGIONS_PATH
    
    try:
        file_path.write_text(
            json.dumps(regions, indent=2, sort_keys=True),
            encoding="utf-8"
        )
        return True
    except IOError:
        return False


def add_project_region(region_def: Dict[str, Any], path: Path = None) -> bool:
    """
    Add or update a single project region.
    
    Args:
        region_def: RegionDef dict (must include 'id')
        path: Optional custom path to regions file
        
    Returns:
        True if saved successfully
    """
    region_id = region_def.get("id")
    if not region_id:
        return False
    
    regions = load_project_regions(path)
    regions[region_id] = region_def
    return save_project_regions(regions, path)


def delete_project_region(region_id: str, path: Path = None) -> bool:
    """
    Delete a project region by ID.
    
    Args:
        region_id: ID of region to delete
        path: Optional custom path to regions file
        
    Returns:
        True if deleted successfully
    """
    regions = load_project_regions(path)
    if region_id in regions:
        del regions[region_id]
        return save_project_regions(regions, path)
    return False
