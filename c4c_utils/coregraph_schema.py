"""
C4C CoreGraph Schema v1 — Shared Constants and Utilities

This module provides the canonical vocabulary and utilities for C4C network data,
enabling interoperability between OrgGraph (US/CA), ActorGraph, and InsightGraph.

VERSION HISTORY:
----------------
v1.1.0 (2025-12-22): Phase 1b - Unified CSV column ordering
- Revised UNIFIED_NODE_COLUMNS to practical OrgGraph set
- Revised UNIFIED_EDGE_COLUMNS to practical OrgGraph set
- Column order standardized for cross-app compatibility

v1.0.0 (2025-12-22): Initial Phase 1a implementation
- Canonical node_type vocabulary (lowercase)
- Canonical edge_type vocabulary (lowercase)
- org_type for funder/grantee roles
- Namespace functions for ID collision prevention
- Export utilities for unified CSV format

SCHEMA PRINCIPLES:
------------------
1. Extensible core: Small semantic core (node_type, edge_type) with optional metadata
2. Semantic clarity: InsightGraph treats node_type and edge_type as semantic truth
3. Namespaced identity: All exported node IDs are namespaced by source app
4. Source provenance: source_app field tracks data origin
5. Backwards tolerant: Readers accept legacy values case-insensitively
"""

from typing import Optional, List, Dict, Any
import pandas as pd

# =============================================================================
# Schema Version
# =============================================================================

COREGRAPH_VERSION = "c4c_coregraph_v1"

# =============================================================================
# Canonical Node Type Vocabulary (Semantic Identity)
# =============================================================================

# Valid node types (lowercase, structural identity)
VALID_NODE_TYPES = frozenset({
    'person',        # Individual human
    'organization',  # Foundation / nonprofit
    'company',       # For-profit company (ActorGraph)
    'school',        # Educational institution (ActorGraph)
})

# Legacy mappings (uppercase → lowercase)
NODE_TYPE_NORMALIZATION = {
    'ORG': 'organization',
    'PERSON': 'person',
    'COMPANY': 'company',
    'SCHOOL': 'school',
    'FUNDER': 'organization',   # Legacy: funder is now org_type
    'GRANTEE': 'organization',  # Legacy: grantee is now org_type
}

# =============================================================================
# Canonical Edge Type Vocabulary (Semantic Meaning)
# =============================================================================

# Valid edge types (lowercase)
VALID_EDGE_TYPES = frozenset({
    'grant',         # Funding relationship (funder → grantee)
    'board',         # Board membership (person → org)
    'employment',    # Works at (person → org)
    'education',     # Attended (person → school)
    'connection',    # Social connection (person ↔ person)
    'affiliation',   # Generic membership (person → org)
})

# Legacy mappings (uppercase → lowercase)
EDGE_TYPE_NORMALIZATION = {
    'GRANT': 'grant',
    'BOARD_MEMBERSHIP': 'board',
    'BOARD': 'board',
    'EMPLOYMENT': 'employment',
    'EDUCATION': 'education',
    'CONNECTION': 'connection',
    'AFFILIATION': 'affiliation',
}

# Default directed value by edge type
EDGE_DIRECTED_DEFAULTS = {
    'grant': True,
    'board': True,
    'employment': True,
    'education': True,
    'affiliation': True,
    'connection': False,  # Only connection is undirected
}

# =============================================================================
# Canonical Org Type Vocabulary (Functional Role)
# =============================================================================

VALID_ORG_TYPES = frozenset({
    'funder',    # Gives grants
    'grantee',   # Receives grants
    'both',      # Both gives and receives grants
})

# =============================================================================
# Source App Identifiers
# =============================================================================

VALID_SOURCE_APPS = frozenset({
    'orggraph_us',
    'orggraph_ca',
    'actorgraph',
    'manual',
})

# =============================================================================
# Unified Node Schema (Phase 1b)
# =============================================================================

# Column order matters for consistent exports
# This is the practical set for OrgGraph US/CA - ActorGraph fields can be added later
UNIFIED_NODE_COLUMNS = [
    # === Identity (Required) ===
    'node_id',        # Required: Namespaced ID (orggraph_us:org-123)
    'node_type',      # Required: person, organization
    'label',          # Required: Display name
    
    # === Organization Fields ===
    'org_type',       # Optional: funder, grantee, both
    'org_slug',       # Optional: URL-friendly org name
    'tax_id',         # Optional: CRA BN (CA) or EIN (US)
    'assets_latest',  # Optional: Latest assets value
    'assets_year',    # Optional: Year of assets value
    
    # === Person Fields ===
    'first_name',     # Optional: Person first name
    'last_name',      # Optional: Person last name
    
    # === Location ===
    'city',           # Optional
    'region',         # Optional: State/province code
    'jurisdiction',   # Optional: US, CA
    
    # === Provenance ===
    'source_app',     # Required: orggraph_us, orggraph_ca, actorgraph
    'source_system',  # Optional: IRS_990, CHARITYDATA_CA
    'source_ref',     # Optional: Reference to source document
]

# =============================================================================
# Unified Edge Schema (Phase 1b)
# =============================================================================

# Column order matters for consistent exports
# This is the practical set for OrgGraph US/CA
UNIFIED_EDGE_COLUMNS = [
    # === Identity (Required) ===
    'edge_id',        # Optional: Unique edge identifier
    'from_id',        # Required: Source node (namespaced)
    'to_id',          # Required: Target node (namespaced)
    'edge_type',      # Required: grant, board
    
    # === Graph Properties ===
    'directed',       # Optional: Default True for grant/board
    'weight',         # Optional: Default 1
    
    # === Grant Fields ===
    'amount',         # Optional: Total grant amount
    'amount_cash',    # Optional: Cash portion
    'amount_in_kind', # Optional: In-kind portion
    'currency',       # Optional: USD, CAD
    'fiscal_year',    # Optional: Fiscal year
    'reporting_period', # Optional: Reporting period string
    'purpose',        # Optional: Grant purpose/description
    
    # === Board/Employment Fields ===
    'role',           # Optional: Job title, board role
    'start_date',     # Optional: Start date
    'end_date',       # Optional: End date
    'at_arms_length', # Optional: Board member arm's length flag
    
    # === Location ===
    'city',           # Optional: Grantee city
    'region',         # Optional: Grantee state/province
    
    # === Provenance ===
    'source_app',     # Required: orggraph_us, orggraph_ca
    'source_system',  # Optional: IRS_990, CHARITYDATA_CA
    'source_ref',     # Optional: Reference to source document
]

# =============================================================================
# ID Namespacing Functions
# =============================================================================

def namespace_id(node_id: str, source_app: str) -> str:
    """
    Namespace a node ID to prevent collisions across apps.
    
    If the ID already contains ':', it's assumed to be namespaced
    and is returned unchanged (prevents double-prefixing).
    
    Args:
        node_id: Original node ID (e.g., 'org-123')
        source_app: Source app identifier (e.g., 'orggraph_us')
    
    Returns:
        Namespaced ID (e.g., 'orggraph_us:org-123')
    
    Examples:
        >>> namespace_id('org-123', 'orggraph_us')
        'orggraph_us:org-123'
        
        >>> namespace_id('orggraph_us:org-123', 'orggraph_us')
        'orggraph_us:org-123'  # Already namespaced, unchanged
    """
    if not node_id:
        return node_id
    
    node_id_str = str(node_id)
    
    # Already namespaced? Return as-is
    if ':' in node_id_str:
        return node_id_str
    
    return f"{source_app}:{node_id_str}"


def extract_namespace(node_id: str) -> tuple:
    """
    Extract namespace and local ID from a namespaced node ID.
    
    Args:
        node_id: Namespaced ID (e.g., 'orggraph_us:org-123')
    
    Returns:
        Tuple of (namespace, local_id)
        If not namespaced, returns (None, node_id)
    
    Examples:
        >>> extract_namespace('orggraph_us:org-123')
        ('orggraph_us', 'org-123')
        
        >>> extract_namespace('org-123')
        (None, 'org-123')
    """
    if not node_id or ':' not in str(node_id):
        return (None, str(node_id) if node_id else '')
    
    parts = str(node_id).split(':', 1)
    return (parts[0], parts[1])

# =============================================================================
# Type Normalization Functions
# =============================================================================

def normalize_node_type(node_type: str) -> str:
    """
    Normalize node_type to canonical lowercase vocabulary.
    
    Args:
        node_type: Original node type (e.g., 'ORG', 'PERSON')
    
    Returns:
        Normalized type (e.g., 'organization', 'person')
    """
    if not node_type:
        return ''
    
    nt = str(node_type).strip().upper()
    
    # Check legacy mapping first
    if nt in NODE_TYPE_NORMALIZATION:
        return NODE_TYPE_NORMALIZATION[nt]
    
    # Already lowercase and valid?
    nt_lower = str(node_type).strip().lower()
    if nt_lower in VALID_NODE_TYPES:
        return nt_lower
    
    # Unknown type - return lowercase
    return nt_lower


def normalize_edge_type(edge_type: str) -> str:
    """
    Normalize edge_type to canonical lowercase vocabulary.
    
    Args:
        edge_type: Original edge type (e.g., 'GRANT', 'BOARD_MEMBERSHIP')
    
    Returns:
        Normalized type (e.g., 'grant', 'board')
    """
    if not edge_type:
        return ''
    
    et = str(edge_type).strip().upper()
    
    # Check legacy mapping first
    if et in EDGE_TYPE_NORMALIZATION:
        return EDGE_TYPE_NORMALIZATION[et]
    
    # Already lowercase and valid?
    et_lower = str(edge_type).strip().lower()
    if et_lower in VALID_EDGE_TYPES:
        return et_lower
    
    # Unknown type - return lowercase
    return et_lower


def get_directed_default(edge_type: str) -> bool:
    """
    Get default directed value for an edge type.
    
    Args:
        edge_type: Normalized edge type
    
    Returns:
        True for directed, False for undirected
    """
    et = normalize_edge_type(edge_type)
    return EDGE_DIRECTED_DEFAULTS.get(et, True)

# =============================================================================
# DataFrame Transformation Functions
# =============================================================================

def normalize_nodes_df(
    nodes_df: pd.DataFrame,
    source_app: str,
    infer_org_type: bool = True
) -> pd.DataFrame:
    """
    Normalize a nodes DataFrame to CoreGraph v1 schema.
    
    Applies:
    - Lowercase node_type
    - Namespaced node_id
    - source_app field
    - org_type inference from legacy node_type
    
    Args:
        nodes_df: Original nodes DataFrame
        source_app: Source app identifier
        infer_org_type: If True, infer org_type from legacy funder/grantee node_type
    
    Returns:
        Normalized DataFrame
    """
    if nodes_df is None or nodes_df.empty:
        return nodes_df
    
    df = nodes_df.copy()
    
    # Infer org_type from legacy node_type before normalization
    if infer_org_type and 'node_type' in df.columns:
        if 'org_type' not in df.columns:
            df['org_type'] = ''
        
        # Legacy funder/grantee → org_type
        mask_funder = df['node_type'].str.upper() == 'FUNDER'
        mask_grantee = df['node_type'].str.upper() == 'GRANTEE'
        df.loc[mask_funder, 'org_type'] = 'funder'
        df.loc[mask_grantee, 'org_type'] = 'grantee'
    
    # Normalize node_type
    if 'node_type' in df.columns:
        df['node_type'] = df['node_type'].apply(normalize_node_type)
    
    # Namespace node_id
    if 'node_id' in df.columns:
        df['node_id'] = df['node_id'].apply(lambda x: namespace_id(x, source_app))
    
    # Set source_app
    df['source_app'] = source_app
    
    return df


def normalize_edges_df(
    edges_df: pd.DataFrame,
    source_app: str
) -> pd.DataFrame:
    """
    Normalize an edges DataFrame to CoreGraph v1 schema.
    
    Applies:
    - Lowercase edge_type
    - Namespaced from_id and to_id
    - source_app field
    - directed default based on edge_type
    - weight default (1)
    
    Args:
        edges_df: Original edges DataFrame
        source_app: Source app identifier
    
    Returns:
        Normalized DataFrame
    """
    if edges_df is None or edges_df.empty:
        return edges_df
    
    df = edges_df.copy()
    
    # Normalize edge_type
    if 'edge_type' in df.columns:
        df['edge_type'] = df['edge_type'].apply(normalize_edge_type)
    
    # Namespace from_id and to_id
    if 'from_id' in df.columns:
        df['from_id'] = df['from_id'].apply(lambda x: namespace_id(x, source_app))
    if 'to_id' in df.columns:
        df['to_id'] = df['to_id'].apply(lambda x: namespace_id(x, source_app))
    
    # Set directed default based on edge_type
    if 'directed' not in df.columns:
        df['directed'] = df['edge_type'].apply(get_directed_default)
    
    # Set weight default
    if 'weight' not in df.columns:
        df['weight'] = 1
    else:
        df['weight'] = df['weight'].fillna(1)
    
    # Set source_app
    df['source_app'] = source_app
    
    return df

# =============================================================================
# Validation Functions
# =============================================================================

def validate_node(node: dict) -> List[str]:
    """
    Validate a single node against CoreGraph v1 schema.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Required fields
    if not node.get('node_id'):
        errors.append("Missing required field: node_id")
    elif ':' not in str(node['node_id']):
        errors.append(f"node_id not namespaced: {node['node_id']}")
    
    if not node.get('label'):
        errors.append("Missing required field: label")
    
    if not node.get('node_type'):
        errors.append("Missing required field: node_type")
    elif str(node['node_type']).lower() not in VALID_NODE_TYPES:
        errors.append(f"Invalid node_type: {node['node_type']}")
    
    # Optional field validation
    if node.get('org_type'):
        if str(node['org_type']).lower() not in VALID_ORG_TYPES:
            errors.append(f"Invalid org_type: {node['org_type']}")
    
    return errors


def validate_edge(edge: dict, node_ids: set = None) -> List[str]:
    """
    Validate a single edge against CoreGraph v1 schema.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Required fields
    if not edge.get('from_id'):
        errors.append("Missing required field: from_id")
    elif node_ids and str(edge['from_id']) not in node_ids:
        errors.append(f"from_id not in nodes: {edge['from_id']}")
    
    if not edge.get('to_id'):
        errors.append("Missing required field: to_id")
    elif node_ids and str(edge['to_id']) not in node_ids:
        errors.append(f"to_id not in nodes: {edge['to_id']}")
    
    if not edge.get('edge_type'):
        errors.append("Missing required field: edge_type")
    elif str(edge['edge_type']).lower() not in VALID_EDGE_TYPES:
        errors.append(f"Invalid edge_type: {edge['edge_type']}")
    
    return errors

# =============================================================================
# Export Utilities (Phase 1b)
# =============================================================================

def prepare_unified_nodes_csv(
    nodes_df: pd.DataFrame,
    source_app: str
) -> pd.DataFrame:
    """
    Prepare nodes DataFrame for unified CSV export.
    
    Ensures:
    - All columns present in correct order
    - Missing columns filled with empty string
    - source_app populated
    - IDs namespaced
    - Types normalized
    """
    # Normalize first
    df = normalize_nodes_df(nodes_df, source_app)
    
    # Ensure all columns present
    for col in UNIFIED_NODE_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    
    # Return with correct column order
    return df[UNIFIED_NODE_COLUMNS]


def prepare_unified_edges_csv(
    edges_df: pd.DataFrame,
    source_app: str
) -> pd.DataFrame:
    """
    Prepare edges DataFrame for unified CSV export.
    
    Ensures:
    - All columns present in correct order
    - Missing columns filled with empty string (except defaults)
    - source_app populated
    - IDs namespaced
    - Types normalized
    - directed and weight have defaults
    """
    # Normalize first
    df = normalize_edges_df(edges_df, source_app)
    
    # Ensure all columns present
    for col in UNIFIED_EDGE_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    
    # Return with correct column order
    return df[UNIFIED_EDGE_COLUMNS]
