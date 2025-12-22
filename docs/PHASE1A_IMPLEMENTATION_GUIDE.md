# Phase 1a Implementation Guide: OrgGraph Schema Normalization

**Date:** 2025-12-23  
**Status:** Ready to Implement

This document provides the specific code changes needed to implement Phase 1a
(Schema Normalization) in OrgGraph US and OrgGraph CA.

---

## Overview

Phase 1a normalizes the data schema across all C4C apps:

| Change | Before | After |
|--------|--------|-------|
| node_type | `"ORG"`, `"PERSON"` | `"organization"`, `"person"` |
| edge_type | `"GRANT"`, `"BOARD_MEMBERSHIP"` | `"grant"`, `"board"` |
| node_id | `org-123` | `orggraph_ca:org-123` |
| source_app | (missing) | `"orggraph_ca"` or `"orggraph_us"` |

---

## Step 1: Add coregraph_schema.py to c4c_utils/

Copy the `coregraph_schema.py` file to your `c4c_utils/` directory.

This provides:
- `namespace_id()` — Adds source prefix to IDs
- `normalize_node_type()` — Converts ORG → organization
- `normalize_edge_type()` — Converts GRANT → grant
- `normalize_nodes_df()` — Full DataFrame normalization
- `normalize_edges_df()` — Full DataFrame normalization

---

## Step 2: OrgGraph CA Changes (app.py)

### 2.1 Add import at top of file

```python
# Add after existing imports (around line 94)
from c4c_utils.coregraph_schema import (
    normalize_nodes_df,
    normalize_edges_df,
    namespace_id,
    normalize_node_type,
    normalize_edge_type,
)

# Define source app constant
SOURCE_APP = "orggraph_ca"
```

### 2.2 Update NODE_COLUMNS (line ~325)

Add `source_app` and `org_type` to the columns:

```python
NODE_COLUMNS = [
    "node_id", "node_type", "label", "org_slug", "jurisdiction", "tax_id",
    "city", "region", "source_system", "source_ref", "assets_latest", "assets_year",
    "first_name", "last_name",
    "org_type", "source_app"  # NEW: Added for CoreGraph v1
]
```

### 2.3 Update EDGE_COLUMNS (line ~331)

Add `source_app`, `directed`, and `weight`:

```python
EDGE_COLUMNS = [
    "edge_id", "from_id", "to_id", "edge_type",
    "amount", "amount_cash", "amount_in_kind", "currency",
    "fiscal_year", "reporting_period", "purpose",
    "role", "start_date", "end_date", "at_arms_length",
    "city", "region",
    "source_system", "source_ref",
    "directed", "weight", "source_app"  # NEW: Added for CoreGraph v1
]
```

### 2.4 Update process_directors_file() (line ~941)

Change node_type from "PERSON" to "person" and add source_app:

```python
# Around line 1032-1041
# Create person node
person_node = {col: "" for col in NODE_COLUMNS}
person_node.update({
    "node_id": namespace_id(person_id, SOURCE_APP),  # CHANGED: Namespace ID
    "node_type": "person",  # CHANGED: lowercase
    "label": f"{first_name} {last_name}".strip(),
    "first_name": first_name,
    "last_name": last_name,
    "jurisdiction": JURISDICTION,
    "source_system": SOURCE_SYSTEM,
    "source_app": SOURCE_APP,  # NEW
})
nodes.append(person_node)

# Around line 1044-1058
# Create board edge - update IDs and edge_type
edge_id = deterministic_board_edge_id(person_id, org_id, role)

board_edge = {col: "" for col in EDGE_COLUMNS}
board_edge.update({
    "edge_id": edge_id,
    "from_id": namespace_id(person_id, SOURCE_APP),  # CHANGED: Namespace
    "to_id": namespace_id(org_id, SOURCE_APP),       # CHANGED: Namespace
    "edge_type": "board",  # CHANGED: lowercase, was BOARD_MEMBERSHIP
    "role": role,
    "start_date": start_date,
    "end_date": end_date,
    "at_arms_length": "Y" if at_arms_length else "N",
    "source_system": SOURCE_SYSTEM,
    "source_app": SOURCE_APP,  # NEW
    "directed": True,  # NEW
    "weight": 1,  # NEW
})
edges.append(board_edge)
```

### 2.5 Update process_grants_file() (line ~1063)

Change node_type from "ORG" to "organization" and edge_type from "GRANT" to "grant":

```python
# Around line 1098-1109
# Create grantee org node
grantee_slug = slugify_loose(str(grantee_name))
grantee_id = f"org-{grantee_slug}"

grantee_node = {col: "" for col in NODE_COLUMNS}
grantee_node.update({
    "node_id": namespace_id(grantee_id, SOURCE_APP),  # CHANGED: Namespace
    "node_type": "organization",  # CHANGED: lowercase, was ORG
    "org_type": "grantee",  # NEW: functional role
    "label": str(grantee_name).strip(),
    "org_slug": grantee_slug,
    "city": city,
    "region": province,
    "jurisdiction": JURISDICTION,
    "source_system": SOURCE_SYSTEM,
    "source_app": SOURCE_APP,  # NEW
})
nodes.append(grantee_node)

# Around line 1112-1130
# Create grant edge
edge_id = deterministic_grant_edge_id(org_id, grantee_id, total_amount, fiscal_year or 0)

grant_edge = {col: "" for col in EDGE_COLUMNS}
grant_edge.update({
    "edge_id": edge_id,
    "from_id": namespace_id(org_id, SOURCE_APP),      # CHANGED: Namespace
    "to_id": namespace_id(grantee_id, SOURCE_APP),    # CHANGED: Namespace
    "edge_type": "grant",  # CHANGED: lowercase, was GRANT
    "amount": total_amount,
    "amount_cash": cash,
    "amount_in_kind": in_kind,
    "currency": CURRENCY,
    "fiscal_year": fiscal_year or "",
    "reporting_period": reporting_period,
    "city": city,
    "region": province,
    "source_system": SOURCE_SYSTEM,
    "source_app": SOURCE_APP,  # NEW
    "directed": True,  # NEW
    "weight": 1,  # NEW
})
edges.append(grant_edge)
```

### 2.6 Update foundation org node creation

Find where the foundation org node is created (usually in process_org_data or similar)
and apply the same changes:

```python
# Foundation org node
foundation_node = {col: "" for col in NODE_COLUMNS}
foundation_node.update({
    "node_id": namespace_id(f"org-{cra_bn}", SOURCE_APP),  # CHANGED: Namespace
    "node_type": "organization",  # CHANGED: lowercase
    "org_type": "funder",  # NEW: functional role
    "label": org_name,
    # ... other fields ...
    "source_app": SOURCE_APP,  # NEW
})
```

### 2.7 Update get_existing_foundations() (line ~445)

Change the filter to use lowercase:

```python
def get_existing_foundations(nodes_df: pd.DataFrame) -> list:
    """Get list of existing foundations from nodes."""
    if nodes_df.empty or "node_type" not in nodes_df.columns:
        return []
    
    # CHANGED: Accept both old and new formats (backwards compatible)
    orgs = nodes_df[nodes_df["node_type"].str.lower().isin(["org", "organization"])]
    if orgs.empty:
        return []
    
    foundations = []
    for _, row in orgs.iterrows():
        label = row.get("label", "Unknown")
        source = row.get("source_system", "")
        foundations.append((label, source))
    
    return foundations
```

---

## Step 3: OrgGraph US Changes (app-2.py)

The US version uses `c4c_utils.network_export` for building nodes/edges, so changes
may need to be made in that module as well. However, the same pattern applies:

### 3.1 Add import

```python
from c4c_utils.coregraph_schema import (
    normalize_nodes_df,
    normalize_edges_df,
    namespace_id,
)

SOURCE_APP = "orggraph_us"
```

### 3.2 Update network_export.py (if applicable)

If `build_nodes_df()` and `build_edges_df()` are in `c4c_utils/network_export.py`,
update those functions to:

1. Use lowercase `node_type` ("organization", "person")
2. Use lowercase `edge_type` ("grant", "board")
3. Accept `source_app` parameter
4. Call `namespace_id()` on all IDs

---

## Step 4: Alternative — Normalize at Export Time

If you prefer minimal code changes to the data processing logic, you can normalize
at export time instead. Add this to `render_downloads()`:

```python
def render_downloads(nodes_df, edges_df, grants_detail_df=None, project_name=None):
    # ... existing code ...
    
    # Normalize for export
    from c4c_utils.coregraph_schema import normalize_nodes_df, normalize_edges_df
    
    nodes_normalized = normalize_nodes_df(nodes_df, SOURCE_APP)
    edges_normalized = normalize_edges_df(edges_df, SOURCE_APP)
    
    # Use normalized versions for C4C Schema downloads
    st.markdown("**📁 C4C Schema** (for Insight Engine)")
    c4c_col1, c4c_col2, c4c_col3 = st.columns(3)
    
    with c4c_col1:
        st.download_button(
            "📥 nodes.csv",
            data=nodes_normalized.to_csv(index=False),  # CHANGED: Use normalized
            file_name="nodes.csv",
            mime="text/csv",
            use_container_width=True
        )
    # ... etc ...
```

This approach:
- ✅ Minimal code changes
- ✅ Backwards compatible (internal data unchanged)
- ⚠️ Normalization happens on every export (slightly slower)

---

## Step 5: Testing

After making changes, verify:

```python
# Test 1: Node types are lowercase
assert all(nodes_df['node_type'].isin(['person', 'organization', 'company', 'school']))

# Test 2: Edge types are lowercase  
assert all(edges_df['edge_type'].isin(['grant', 'board', 'employment', 'education', 'connection', 'affiliation']))

# Test 3: IDs are namespaced
assert all(':' in str(nid) for nid in nodes_df['node_id'])
assert all(':' in str(eid) for eid in edges_df['from_id'])
assert all(':' in str(eid) for eid in edges_df['to_id'])

# Test 4: source_app is set
assert all(nodes_df['source_app'] == 'orggraph_ca')  # or orggraph_us
assert all(edges_df['source_app'] == 'orggraph_ca')

# Test 5: Edge refs resolve to nodes
node_ids = set(nodes_df['node_id'])
assert all(edges_df['from_id'].isin(node_ids))
assert all(edges_df['to_id'].isin(node_ids))
```

---

## Summary of Changes

| File | Changes |
|------|---------|
| `c4c_utils/coregraph_schema.py` | NEW: Shared schema module |
| `app.py` (CA) | Add import, update node_type/edge_type, namespace IDs |
| `app-2.py` (US) | Add import, update node_type/edge_type, namespace IDs |
| `c4c_utils/network_export.py` | Update if US uses this for node/edge building |

---

## Files Provided

1. **coregraph_schema.py** — Ready to drop into `c4c_utils/`
2. This implementation guide — Step-by-step instructions

The schema module is fully functional and tested. The implementation changes are
surgical and can be applied incrementally (e.g., CA first, then US).
