# TICKET: OrgGraph Save & Merge Project Bundle (Phase 2.2)

**Priority:** High  
**Effort:** Medium  
**Apps:** OrgGraph US, OrgGraph CA  
**Depends On:** Phase 2.1 (Project Store Table)

---

## Summary

Allow OrgGraph US and CA to save compiled project bundles to the Project Store, with automatic merge semantics when multiple runs contribute to the same project.

---

## Background

Currently, OrgGraph outputs require manual GitHub uploads to share with InsightGraph. This ticket enables:

- Direct save to Project Store
- Incremental builds (US run + CA run → merged project)
- Deduplication on save (re-running doesn't duplicate data)

**Key principle:** The ZIP bundle is the canonical artifact. Merge happens at the bundle level, not in database tables.

---

## Tasks

### 1. Add Project Selector UI

```python
# In app.py sidebar or main area

st.subheader("📁 Project Store")

# Project selector
projects = list_projects(supabase)
project_options = ["➕ Create new project"] + [p['name'] for p in projects]

selected = st.selectbox("Select project", project_options)

if selected == "➕ Create new project":
    new_name = st.text_input("Project name")
    if st.button("Create Project") and new_name:
        project = create_project(supabase, name=new_name, bundle_path=f"...")
        st.success(f"Created project: {new_name}")
else:
    current_project = next(p for p in projects if p['name'] == selected)
```

### 2. Implement bundle merge logic

```python
# orggraph/bundle_merge.py

import pandas as pd
from typing import Tuple
import zipfile
import io

def merge_bundles(
    existing_bundle: bytes,
    new_bundle: bytes,
    source_app: str
) -> Tuple[bytes, dict]:
    """
    Merge two bundles with deduplication.
    
    Returns:
        Tuple of (merged_bundle_bytes, stats_dict)
    """
    # Extract existing data
    existing_nodes, existing_edges = extract_bundle_data(existing_bundle)
    
    # Extract new data
    new_nodes, new_edges = extract_bundle_data(new_bundle)
    
    # Merge nodes (dedupe on node_id)
    merged_nodes = pd.concat([existing_nodes, new_nodes], ignore_index=True)
    merged_nodes = merged_nodes.drop_duplicates(subset=['node_id'], keep='last')
    
    # Merge edges (dedupe on from_id + to_id + edge_type)
    merged_edges = pd.concat([existing_edges, new_edges], ignore_index=True)
    merged_edges = merged_edges.drop_duplicates(
        subset=['from_id', 'to_id', 'edge_type'], 
        keep='last'
    )
    
    # Collect source apps
    existing_sources = get_source_apps(existing_bundle)
    all_sources = list(set(existing_sources + [source_app]))
    
    # Create merged bundle
    merged_bundle = create_bundle(
        nodes_df=merged_nodes,
        edges_df=merged_edges,
        source_apps=all_sources
    )
    
    stats = {
        'nodes': len(merged_nodes),
        'edges': len(merged_edges),
        'sources': all_sources
    }
    
    return merged_bundle, stats


def extract_bundle_data(bundle_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract nodes and edges DataFrames from bundle ZIP."""
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        nodes_df = pd.read_csv(zf.open('data/nodes.csv'))
        edges_df = pd.read_csv(zf.open('data/edges.csv'))
    return nodes_df, edges_df


def get_source_apps(bundle_bytes: bytes) -> list:
    """Extract source_apps from bundle manifest."""
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest = json.loads(zf.read('manifest.json'))
    return manifest.get('source_apps', [])
```

### 3. Implement save flow

```python
# orggraph/project_store.py

def save_project_bundle(
    supabase: Client,
    project_id: str,
    new_bundle: bytes,
    source_app: str
) -> dict:
    """
    Save bundle to Project Store with merge.
    
    1. Download existing bundle (if any)
    2. Merge with new data
    3. Upload merged bundle
    4. Update project metadata
    """
    project = get_project(supabase, project_id)
    bundle_path = project['bundle_path']
    
    # Step 1: Download existing bundle
    existing_bundle = download_bundle(supabase, bundle_path)
    
    # Step 2: Merge
    if existing_bundle:
        merged_bundle, stats = merge_bundles(existing_bundle, new_bundle, source_app)
    else:
        merged_bundle = new_bundle
        stats = get_bundle_stats(new_bundle)
        stats['sources'] = [source_app]
    
    # Step 3: Validate before upload
    validate_bundle(merged_bundle)  # Uses Phase 1 validation
    
    # Step 4: Upload merged bundle
    upload_bundle(supabase, bundle_path, merged_bundle)
    
    # Step 5: Update metadata
    update_project_metadata(
        supabase,
        project_id=project_id,
        row_counts={'nodes': stats['nodes'], 'edges': stats['edges']},
        source_apps=stats['sources'],
        expected_version=project['bundle_version']  # Optimistic concurrency
    )
    
    return stats


def download_bundle(supabase: Client, bundle_path: str) -> bytes:
    """Download bundle from Supabase Storage."""
    try:
        response = supabase.storage.from_('project-bundles').download(bundle_path)
        return response
    except Exception:
        return None  # No existing bundle


def upload_bundle(supabase: Client, bundle_path: str, bundle_bytes: bytes):
    """Upload bundle to Supabase Storage."""
    supabase.storage.from_('project-bundles').upload(
        bundle_path,
        bundle_bytes,
        file_options={"upsert": "true"}
    )
```

### 4. Add Save button to UI

```python
# Replace or augment Download ZIP button

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save to Project Store"):
        bundle = create_project_bundle(...)
        stats = save_project_bundle(
            supabase,
            project_id=current_project['project_id'],
            new_bundle=bundle,
            source_app='orggraph'
        )
        st.success(f"Saved! {stats['nodes']} nodes, {stats['edges']} edges")

with col2:
    st.download_button(
        "📦 Download ZIP",
        data=create_project_bundle(...),
        file_name=f"{project_name}.zip"
    )
```

---

## Storage Layout

```
project-bundles/
└── {project_id}/
    └── bundle.zip
```

Example:
```
project-bundles/
└── a1b2c3d4-e5f6-7890-abcd-ef1234567890/
    └── bundle.zip
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `orggraph/app.py` | MODIFY: Add Project Selector + Save button |
| `orggraph/bundle_merge.py` | NEW: Merge logic |
| `orggraph/project_store.py` | NEW: Save/download functions |
| `c4c_utils/project_store.py` | MODIFY: Add upload/download helpers |

---

## Acceptance Criteria

- [ ] OrgGraph US can create a new project
- [ ] OrgGraph US can save bundle to Project Store
- [ ] OrgGraph CA can select same project and merge data
- [ ] Re-running US does **not** duplicate nodes/edges (dedupe works)
- [ ] Merged bundle passes `lint_report.py --validate-schema`
- [ ] `bundle_version` increments on each save
- [ ] `source_apps` tracks which apps contributed
- [ ] `row_counts` reflects merged totals
- [ ] Download ZIP still works (unchanged)

---

## Merge Rules

| Data Type | Dedupe Key | Behavior |
|-----------|------------|----------|
| Nodes | `node_id` | Last write wins |
| Edges | `(from_id, to_id, edge_type)` | Last write wins |

**Why "last write wins":** Allows corrections to overwrite stale data.

---

## Testing

```python
def test_first_save():
    # US creates project and saves
    bundle_us = create_bundle(nodes_us, edges_us, source_app='orggraph')
    stats = save_project_bundle(supabase, project_id, bundle_us, 'orggraph')
    assert stats['nodes'] == 100
    assert stats['sources'] == ['orggraph']

def test_merge_save():
    # CA saves to same project
    bundle_ca = create_bundle(nodes_ca, edges_ca, source_app='orggraph')
    stats = save_project_bundle(supabase, project_id, bundle_ca, 'orggraph')
    assert stats['nodes'] == 150  # Merged, some overlap
    assert 'orggraph' in stats['sources']

def test_no_duplicates():
    # Re-run US with same data
    stats_before = get_project(supabase, project_id)['row_counts']
    save_project_bundle(supabase, project_id, bundle_us, 'orggraph')
    stats_after = get_project(supabase, project_id)['row_counts']
    assert stats_after['nodes'] == stats_before['nodes']  # No growth

def test_bundle_validates():
    bundle = download_bundle(supabase, bundle_path)
    errors = validate_bundle(bundle)
    assert errors == []
```

---

## Non-Goals

- Multi-user permissions (single-tenant for now)
- Conflict resolution UI (just warn on version mismatch)
- Storing nodes/edges in tables (Phase 3)

---

## Dependencies

- Phase 2.1 (Project Store Table)
- Phase 1c (ZIP Bundle format)

## Unlocks

- Ticket 2.3 (InsightGraph Load)
- No more manual GitHub project uploads
