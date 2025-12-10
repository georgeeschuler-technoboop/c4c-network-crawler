"""
Network Export utilities for C4C Network Intelligence Engine

Converts parsed 990 data into Polinode-ready node and edge CSVs.
"""

import re
import pandas as pd
from typing import List


def slugify_name(name: str) -> str:
    """
    Convert a name to a slug for use as node ID.
    
    Rules:
    - Uppercase
    - Non-alphanumeric characters → underscore
    - Trim leading/trailing underscores
    - Collapse multiple underscores
    """
    if not name:
        return ""
    
    slug = name.upper()
    slug = re.sub(r'[^A-Z0-9]', '_', slug)
    slug = re.sub(r'_+', '_', slug)
    slug = slug.strip('_')
    return slug


def build_nodes_df(
    grants_df: pd.DataFrame,
    people_df: pd.DataFrame,
    foundations_meta_list: List[dict]
) -> pd.DataFrame:
    """
    Build unified nodes DataFrame from grants, people, and foundation metadata.
    
    Node types:
    - foundation: FNDN_<EIN>
    - grantee: ORG_<SLUG(name)>
    - person: PERSON_<SLUG(name)>
    
    Returns DataFrame with columns:
        node_id, label, type, city, state, country, source
    """
    nodes = []
    seen_ids = set()
    
    # 1. Foundation nodes
    for meta in foundations_meta_list:
        ein = meta.get('foundation_ein', '').replace('-', '')
        if not ein:
            continue
        
        node_id = f"FNDN_{ein}"
        if node_id not in seen_ids:
            nodes.append({
                'node_id': node_id,
                'label': meta.get('foundation_name', ''),
                'type': 'foundation',
                'city': '',
                'state': '',
                'country': 'US',
                'source': 'irs_990',
            })
            seen_ids.add(node_id)
    
    # 2. Grantee nodes from grants_df
    if not grants_df.empty:
        for _, row in grants_df.iterrows():
            grantee_name = row.get('grantee_name', '')
            if not grantee_name:
                continue
            
            slug = slugify_name(grantee_name)
            node_id = f"ORG_{slug}"
            
            if node_id not in seen_ids:
                nodes.append({
                    'node_id': node_id,
                    'label': grantee_name,
                    'type': 'grantee',
                    'city': row.get('grantee_city', ''),
                    'state': row.get('grantee_state', ''),
                    'country': 'US',
                    'source': 'irs_990',
                })
                seen_ids.add(node_id)
    
    # 3. Person nodes from people_df
    if not people_df.empty:
        for _, row in people_df.iterrows():
            person_name = row.get('person_name', '')
            if not person_name:
                continue
            
            slug = slugify_name(person_name)
            node_id = f"PERSON_{slug}"
            
            if node_id not in seen_ids:
                nodes.append({
                    'node_id': node_id,
                    'label': person_name,
                    'type': 'person',
                    'city': row.get('person_city', ''),
                    'state': row.get('person_state', ''),
                    'country': 'US',
                    'source': 'irs_990_board',
                })
                seen_ids.add(node_id)
    
    # Create DataFrame
    if nodes:
        df = pd.DataFrame(nodes)
    else:
        df = pd.DataFrame(columns=['node_id', 'label', 'type', 'city', 'state', 'country', 'source'])
    
    return df


def build_edges_df(
    grants_df: pd.DataFrame,
    people_df: pd.DataFrame,
    foundations_meta_list: List[dict]
) -> pd.DataFrame:
    """
    Build unified edges DataFrame combining grants and board memberships.
    
    Edge types:
    - grant: foundation → grantee
    - board_membership: person → foundation
    
    Returns DataFrame with columns:
        from_id, to_id, edge_type, grant_amount, tax_year, grant_purpose_raw,
        role, start_year, end_year, foundation_name, grantee_name, source_file
    """
    edges = []
    
    # 1. Grant edges
    if not grants_df.empty:
        for _, row in grants_df.iterrows():
            ein = row.get('foundation_ein', '').replace('-', '')
            grantee_name = row.get('grantee_name', '')
            
            if not ein or not grantee_name:
                continue
            
            from_id = f"FNDN_{ein}"
            to_id = f"ORG_{slugify_name(grantee_name)}"
            
            edges.append({
                'from_id': from_id,
                'to_id': to_id,
                'edge_type': 'grant',
                'grant_amount': row.get('grant_amount'),
                'tax_year': row.get('tax_year'),
                'grant_purpose_raw': row.get('grant_purpose_raw', ''),
                'role': '',
                'start_year': None,
                'end_year': None,
                'foundation_name': row.get('foundation_name', ''),
                'grantee_name': grantee_name,
                'source_file': row.get('source_file', ''),
            })
    
    # 2. Board membership edges
    if not people_df.empty:
        for _, row in people_df.iterrows():
            person_name = row.get('person_name', '')
            org_ein = row.get('org_ein', '').replace('-', '')
            
            if not person_name or not org_ein:
                continue
            
            from_id = f"PERSON_{slugify_name(person_name)}"
            to_id = f"FNDN_{org_ein}"
            
            edges.append({
                'from_id': from_id,
                'to_id': to_id,
                'edge_type': 'board_membership',
                'grant_amount': None,
                'tax_year': row.get('tax_year'),
                'grant_purpose_raw': '',
                'role': row.get('role', ''),
                'start_year': None,
                'end_year': None,
                'foundation_name': row.get('org_name', ''),
                'grantee_name': '',
                'source_file': row.get('source_file', ''),
            })
    
    # Create DataFrame
    if edges:
        df = pd.DataFrame(edges)
    else:
        df = pd.DataFrame(columns=[
            'from_id', 'to_id', 'edge_type', 'grant_amount', 'tax_year',
            'grant_purpose_raw', 'role', 'start_year', 'end_year',
            'foundation_name', 'grantee_name', 'source_file'
        ])
    
    return df
