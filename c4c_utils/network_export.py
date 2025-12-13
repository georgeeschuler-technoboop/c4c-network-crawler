"""
Network Export utilities for C4C Network Intelligence Engine

Converts parsed 990 data into canonical node and edge CSVs.

Canonical Schema v1 (MVP):
- nodes.csv: ORG and PERSON nodes
- edges.csv: GRANT and BOARD_MEMBERSHIP edges

Compatible with both US IRS 990 and CA charitydata.ca sources.
"""

import re
import hashlib
import pandas as pd
from typing import List

# =============================================================================
# Constants
# =============================================================================

SOURCE_SYSTEM = "IRS_990"
JURISDICTION = "US"
CURRENCY = "USD"

NODE_COLUMNS = [
    "node_id", "node_type", "label", "org_slug", "jurisdiction", "tax_id",
    "city", "region", "source_system", "source_ref", "assets_latest", "assets_year",
    "first_name", "last_name"
]

EDGE_COLUMNS = [
    "edge_id", "from_id", "to_id", "edge_type",
    "amount", "amount_cash", "amount_in_kind", "currency",
    "fiscal_year", "reporting_period", "purpose",
    "role", "start_date", "end_date", "at_arms_length",
    "city", "region",
    "source_system", "source_ref"
]


# =============================================================================
# Helper Functions
# =============================================================================

def slugify_loose(text: str) -> str:
    """
    Convert text to a lowercase slug for org_slug field.
    
    Rules:
    - Lowercase
    - Non-alphanumeric characters → hyphen
    - Trim leading/trailing hyphens
    - Collapse multiple hyphens
    """
    if not text:
        return ""
    
    text = text.strip().lower()
    text = re.sub(r"&|\+", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def generate_edge_hash(s: str) -> str:
    """Generate short hash for edge ID when amount is 0."""
    return hashlib.md5(s.encode()).hexdigest()[:8]


def format_ein(ein: str) -> str:
    """Normalize EIN to XX-XXXXXXX format."""
    ein = (ein or "").replace("-", "").strip()
    if len(ein) == 9:
        return f"{ein[:2]}-{ein[2:]}"
    return ein


# =============================================================================
# Build Nodes DataFrame
# =============================================================================

def build_nodes_df(
    grants_df: pd.DataFrame,
    people_df: pd.DataFrame,
    foundations_meta_list: List[dict]
) -> pd.DataFrame:
    """
    Build unified nodes DataFrame from grants, people, and foundation metadata.
    
    Node types:
    - ORG: org:<org_slug> — foundations and grantees
    - PERSON: person:<context>:<name_key> — board members
    
    Returns DataFrame with canonical columns.
    """
    nodes = []
    seen_ids = set()
    
    # 1. Foundation nodes (ORG)
    for meta in foundations_meta_list:
        ein = meta.get('foundation_ein', '').replace('-', '')
        if not ein:
            continue
        
        org_name = meta.get('foundation_name', '')
        org_slug = slugify_loose(org_name) if org_name else f"ein-{ein}"
        node_id = f"org:{org_slug}"
        
        if node_id not in seen_ids:
            nodes.append({
                "node_id": node_id,
                "node_type": "ORG",
                "label": org_name,
                "org_slug": org_slug,
                "jurisdiction": JURISDICTION,
                "tax_id": format_ein(ein),
                "city": "",
                "region": "",
                "source_system": SOURCE_SYSTEM,
                "source_ref": format_ein(ein),
                "assets_latest": meta.get('total_assets'),
                "assets_year": meta.get('tax_year'),
                "first_name": "",
                "last_name": "",
            })
            seen_ids.add(node_id)
    
    # 2. Grantee nodes (ORG) from grants_df
    if not grants_df.empty:
        for _, row in grants_df.iterrows():
            grantee_name = row.get('grantee_name', '')
            if not grantee_name:
                continue
            
            grantee_slug = f"grantee-{slugify_loose(grantee_name)}"
            state = row.get('grantee_state', '')
            if state:
                grantee_slug = f"{grantee_slug}-{state.lower()}"
            
            node_id = f"org:{grantee_slug}"
            
            if node_id not in seen_ids:
                nodes.append({
                    "node_id": node_id,
                    "node_type": "ORG",
                    "label": grantee_name,
                    "org_slug": grantee_slug,
                    "jurisdiction": JURISDICTION,
                    "tax_id": "",
                    "city": row.get('grantee_city', ''),
                    "region": row.get('grantee_state', ''),
                    "source_system": SOURCE_SYSTEM,
                    "source_ref": row.get('source_file', ''),
                    "assets_latest": None,
                    "assets_year": None,
                    "first_name": "",
                    "last_name": "",
                })
                seen_ids.add(node_id)
    
    # 3. Person nodes (PERSON) from people_df
    if not people_df.empty:
        for _, row in people_df.iterrows():
            person_name = row.get('person_name', '')
            if not person_name:
                continue
            
            org_ein = row.get('org_ein', '').replace('-', '')
            org_name = row.get('org_name', '')
            org_slug = slugify_loose(org_name) if org_name else f"ein-{org_ein}"
            tax_year = row.get('tax_year', '')
            
            # Contextual person ID (same pattern as CA adapter)
            # person:<context_org_slug>:<name_slug>|<year>
            name_slug = slugify_loose(person_name)
            person_key = f"{org_slug}:{name_slug}|{tax_year}"
            node_id = f"person:{person_key}"
            
            if node_id not in seen_ids:
                # Try to split first/last name
                name_parts = person_name.strip().split()
                first_name = name_parts[0] if name_parts else ""
                last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
                
                nodes.append({
                    "node_id": node_id,
                    "node_type": "PERSON",
                    "label": person_name,
                    "org_slug": "",
                    "jurisdiction": "",
                    "tax_id": "",
                    "city": row.get('person_city', ''),
                    "region": row.get('person_state', ''),
                    "source_system": SOURCE_SYSTEM,
                    "source_ref": row.get('source_file', ''),
                    "assets_latest": None,
                    "assets_year": None,
                    "first_name": first_name,
                    "last_name": last_name,
                })
                seen_ids.add(node_id)
    
    # Create DataFrame with canonical column order
    if nodes:
        df = pd.DataFrame(nodes).reindex(columns=NODE_COLUMNS)
    else:
        df = pd.DataFrame(columns=NODE_COLUMNS)
    
    return df


# =============================================================================
# Build Edges DataFrame
# =============================================================================

def build_edges_df(
    grants_df: pd.DataFrame,
    people_df: pd.DataFrame,
    foundations_meta_list: List[dict]
) -> pd.DataFrame:
    """
    Build unified edges DataFrame combining grants and board memberships.
    
    Edge types:
    - GRANT: org (foundation) → org (grantee)
    - BOARD_MEMBERSHIP: person → org (foundation)
    
    Returns DataFrame with canonical columns.
    """
    edges = []
    
    # Build foundation EIN → org_slug lookup
    ein_to_slug = {}
    for meta in foundations_meta_list:
        ein = meta.get('foundation_ein', '').replace('-', '')
        org_name = meta.get('foundation_name', '')
        if ein:
            ein_to_slug[ein] = slugify_loose(org_name) if org_name else f"ein-{ein}"
    
    # 1. Grant edges
    if not grants_df.empty:
        # Track edge base IDs to add sequence numbers for duplicates
        edge_base_counts = {}
        
        for _, row in grants_df.iterrows():
            ein = row.get('foundation_ein', '').replace('-', '')
            grantee_name = row.get('grantee_name', '')
            
            if not ein or not grantee_name:
                continue
            
            # Foundation org_slug
            foundation_slug = ein_to_slug.get(ein, f"ein-{ein}")
            from_id = f"org:{foundation_slug}"
            
            # Grantee org_slug
            grantee_slug = f"grantee-{slugify_loose(grantee_name)}"
            state = row.get('grantee_state', '')
            if state:
                grantee_slug = f"{grantee_slug}-{state.lower()}"
            to_id = f"org:{grantee_slug}"
            
            # Amount
            amount = row.get('grant_amount', 0)
            try:
                amount = float(amount) if pd.notna(amount) else 0
            except:
                amount = 0
            
            # Fiscal year
            fiscal_year = row.get('tax_year')
            try:
                fiscal_year = int(fiscal_year) if pd.notna(fiscal_year) else None
            except:
                fiscal_year = None
            
            # Build edge base ID
            if amount > 0:
                edge_base = f"gr:{from_id}->{to_id}:{fiscal_year}:{int(amount)}"
            else:
                hash_input = f"{from_id}{to_id}{fiscal_year}{grantee_name}"
                edge_base = f"gr:{from_id}->{to_id}:{fiscal_year}:h{generate_edge_hash(hash_input)}"
            
            # Add sequence number to preserve multiple identical grants
            if edge_base not in edge_base_counts:
                edge_base_counts[edge_base] = 0
            edge_base_counts[edge_base] += 1
            seq = edge_base_counts[edge_base]
            edge_id = f"{edge_base}:{seq}"
            
            edges.append({
                "edge_id": edge_id,
                "from_id": from_id,
                "to_id": to_id,
                "edge_type": "GRANT",
                "amount": amount,
                "amount_cash": None,
                "amount_in_kind": None,
                "currency": CURRENCY,
                "fiscal_year": fiscal_year,
                "reporting_period": "",
                "purpose": row.get('grant_purpose_raw', ''),
                "role": "",
                "start_date": "",
                "end_date": "",
                "at_arms_length": "",
                "city": row.get('grantee_city', ''),
                "region": row.get('grantee_state', ''),
                "source_system": SOURCE_SYSTEM,
                "source_ref": row.get('source_file', ''),
            })
    
    # 2. Board membership edges
    if not people_df.empty:
        for _, row in people_df.iterrows():
            person_name = row.get('person_name', '')
            org_ein = row.get('org_ein', '').replace('-', '')
            
            if not person_name or not org_ein:
                continue
            
            # Person node_id
            org_name = row.get('org_name', '')
            org_slug = slugify_loose(org_name) if org_name else f"ein-{org_ein}"
            tax_year = row.get('tax_year', '')
            name_slug = slugify_loose(person_name)
            person_key = f"{org_slug}:{name_slug}|{tax_year}"
            from_id = f"person:{person_key}"
            
            # Foundation node_id
            to_id = f"org:{org_slug}"
            
            # Deterministic edge_id
            edge_id = f"bm:{from_id}->{to_id}:{tax_year}"
            
            edges.append({
                "edge_id": edge_id,
                "from_id": from_id,
                "to_id": to_id,
                "edge_type": "BOARD_MEMBERSHIP",
                "amount": None,
                "amount_cash": None,
                "amount_in_kind": None,
                "currency": "",
                "fiscal_year": tax_year if tax_year else None,
                "reporting_period": "",
                "purpose": "",
                "role": row.get('role', ''),
                "start_date": "",
                "end_date": "",
                "at_arms_length": "",
                "city": "",
                "region": "",
                "source_system": SOURCE_SYSTEM,
                "source_ref": row.get('source_file', ''),
            })
    
    # Create DataFrame with canonical column order
    if edges:
        df = pd.DataFrame(edges).reindex(columns=EDGE_COLUMNS)
    else:
        df = pd.DataFrame(columns=EDGE_COLUMNS)
    
    return df


# =============================================================================
# Helper Functions
# =============================================================================

def get_existing_foundations(nodes_df: pd.DataFrame) -> list:
    """
    Extract list of foundation names already in GLFN data.
    Foundations are ORG nodes that have a tax_id.
    Returns: list of (label, source_system) tuples
    """
    if nodes_df.empty or "node_type" not in nodes_df.columns:
        return []
    
    # Filter to ORG nodes with tax_id (foundations have tax_ids, donees don't)
    orgs = nodes_df[nodes_df["node_type"] == "ORG"].copy()
    
    if "tax_id" not in orgs.columns:
        return []
    
    foundations = orgs[orgs["tax_id"].notna() & (orgs["tax_id"] != "")]
    
    if foundations.empty:
        return []
    
    result = []
    for _, row in foundations.iterrows():
        label = row.get("label", "Unknown")
        source = row.get("source_system", "")
        result.append((label, source))
    
    return result
