# TICKET: InsightGraph Load from Project Store (Phase 2.3)

**Priority:** High  
**Effort:** Medium  
**Apps:** InsightGraph  
**Depends On:** Phase 2.1 (Project Store Table), Phase 2.2 (OrgGraph Save)

---

## Summary

Allow InsightGraph to load project bundles directly from the Project Store, eliminating dependency on local files or hardcoded GitHub demo paths.

---

## Background

Currently, InsightGraph requires:
- Manual file upload, or
- Hardcoded paths to GitHub demo data

This ticket enables:
- Project picker from Project Store
- Direct bundle download and analysis
- Same report quality, no local files needed

**Key principle:** InsightGraph consumes the same bundle format it always has — the only change is where the bundle comes from.

---

## Tasks

### 1. Add Project Picker UI

```python
# In insightgraph/app.py

st.subheader("📁 Load Project")

# Data source selector
data_source = st.radio(
    "Data source",
    ["Project Store", "Upload files", "Demo data"],
    horizontal=True
)

if data_source == "Project Store":
    projects = list_projects(supabase)
    
    if not projects:
        st.info("No projects in store. Use OrgGraph to create one.")
    else:
        project_options = {p['name']: p for p in projects}
        selected_name = st.selectbox(
            "Select project",
            options=list(project_options.keys())
        )
        selected_project = project_options[selected_name]
        
        # Show project info
        st.caption(f"Last updated: {selected_project['updated_at']}")
        st.caption(f"Sources: {', '.join(selected_project['source_apps'])}")
        st.caption(f"Nodes: {selected_project['row_counts'].get('nodes', '?')}, "
                   f"Edges: {selected_project['row_counts'].get('edges', '?')}")
        
        if st.button("📥 Load Project"):
            with st.spinner("Loading from Project Store..."):
                nodes_df, edges_df, manifest = load_project_from_store(
                    supabase, 
                    selected_project['project_id']
                )
            st.success(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges")
            st.session_state['nodes_df'] = nodes_df
            st.session_state['edges_df'] = edges_df
            st.session_state['project_manifest'] = manifest

elif data_source == "Upload files":
    # Existing upload UI
    nodes_file = st.file_uploader("nodes.csv")
    edges_file = st.file_uploader("edges.csv")
    # ...

elif data_source == "Demo data":
    # Existing demo data loader
    # ...
```

### 2. Implement project loader

```python
# insightgraph/project_loader.py

import pandas as pd
import zipfile
import io
import json
from typing import Tuple

def load_project_from_store(
    supabase,
    project_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load project bundle from Project Store.
    
    Returns:
        Tuple of (nodes_df, edges_df, manifest)
    """
    # Get project metadata
    project = get_project(supabase, project_id)
    bundle_path = project['bundle_path']
    
    # Download bundle
    bundle_bytes = download_bundle(supabase, bundle_path)
    
    if bundle_bytes is None:
        raise ValueError(f"Bundle not found: {bundle_path}")
    
    # Validate bundle
    errors = validate_bundle_bytes(bundle_bytes)
    if errors:
        raise ValueError(f"Invalid bundle: {errors}")
    
    # Parse bundle
    nodes_df, edges_df, manifest = parse_bundle(bundle_bytes)
    
    return nodes_df, edges_df, manifest


def parse_bundle(bundle_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse bundle ZIP into DataFrames and manifest."""
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        # Read nodes
        nodes_df = pd.read_csv(zf.open('data/nodes.csv'))
        
        # Read edges
        edges_df = pd.read_csv(zf.open('data/edges.csv'))
        
        # Read manifest
        manifest = json.loads(zf.read('manifest.json'))
        
        # Read grants_detail if present
        if 'data/grants_detail.csv' in zf.namelist():
            grants_df = pd.read_csv(zf.open('data/grants_detail.csv'))
        else:
            grants_df = None
    
    return nodes_df, edges_df, manifest


def validate_bundle_bytes(bundle_bytes: bytes) -> list:
    """Validate bundle against CoreGraph v1 schema."""
    errors = []
    
    try:
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
            # Check required files
            required = ['data/nodes.csv', 'data/edges.csv', 'manifest.json']
            for f in required:
                if f not in zf.namelist():
                    errors.append(f"Missing required file: {f}")
            
            if errors:
                return errors
            
            # Validate manifest
            manifest = json.loads(zf.read('manifest.json'))
            if manifest.get('schema_version') != 'c4c_coregraph_v1':
                errors.append(f"Invalid schema_version: {manifest.get('schema_version')}")
            
            # Use existing validation from Phase 1
            # (Can import from lint_report.py)
            
    except zipfile.BadZipFile:
        errors.append("Invalid ZIP file")
    
    return errors
```

### 3. Update report pipeline to accept loaded data

```python
# insightgraph/report_pipeline.py

def run_analysis(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    project_id: str,
    manifest: dict = None
):
    """
    Run InsightGraph analysis on loaded data.
    
    This is the same analysis as before — the only change is
    data can now come from Project Store instead of local files.
    """
    # Existing analysis pipeline
    grant_graph = build_grant_graph(nodes_df, edges_df)
    board_graph = build_board_graph(nodes_df, edges_df)
    interlock_graph = build_interlock_graph(nodes_df, edges_df)
    
    metrics_df = compute_base_metrics(nodes_df, grant_graph, board_graph, interlock_graph)
    metrics_df = compute_derived_signals(metrics_df)
    
    # ... rest of analysis
    
    return results
```

### 4. Remove hardcoded demo paths

```python
# Before
DEFAULT_NODES = Path("data/glfn/nodes.csv")
DEFAULT_EDGES = Path("data/glfn/edges.csv")

# After
# Demo data loaded from Project Store or embedded sample
DEMO_PROJECT_ID = "glfn-demo"  # Or load from config
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `insightgraph/app.py` | MODIFY: Add Project Picker UI |
| `insightgraph/project_loader.py` | NEW: Load from Project Store |
| `insightgraph/report_pipeline.py` | MODIFY: Accept loaded data (minimal change) |
| `c4c_utils/project_store.py` | USE: Shared helpers from 2.1 |

---

## UX Flow

```
┌─────────────────────────────────────────────┐
│  📁 Load Project                            │
├─────────────────────────────────────────────┤
│  ○ Project Store  ○ Upload files  ○ Demo    │
│                                             │
│  Select project: [Great Lakes Funders ▼]    │
│                                             │
│  Last updated: 2025-12-22                   │
│  Sources: orggraph                          │
│  Nodes: 2,920 | Edges: 5,000                │
│                                             │
│  [📥 Load Project]                          │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  ✅ Loaded 2,920 nodes, 5,000 edges         │
│                                             │
│  [▶ Generate Report]                        │
└─────────────────────────────────────────────┘
```

---

## Acceptance Criteria

- [ ] InsightGraph shows Project Picker when "Project Store" selected
- [ ] Can list all projects from `c4c_projects` table
- [ ] Can load GLFN project saved by OrgGraph
- [ ] Bundle validation runs before analysis
- [ ] Invalid bundles show clear error message
- [ ] Report output is **identical** to GitHub demo output (same data)
- [ ] No local filesystem dependency for Project Store path
- [ ] "Upload files" and "Demo data" options still work
- [ ] Can export new ZIP after analysis (unchanged)

---

## Testing

```python
def test_load_from_store():
    # Assumes OrgGraph has saved GLFN project
    nodes_df, edges_df, manifest = load_project_from_store(
        supabase, 
        project_id="glfn-project-id"
    )
    assert len(nodes_df) > 0
    assert len(edges_df) > 0
    assert manifest['schema_version'] == 'c4c_coregraph_v1'

def test_report_identical():
    # Load from store
    nodes_store, edges_store, _ = load_project_from_store(supabase, project_id)
    
    # Load from local demo
    nodes_local = pd.read_csv('data/glfn/nodes.csv')
    edges_local = pd.read_csv('data/glfn/edges.csv')
    
    # Compare (after normalizing)
    assert len(nodes_store) == len(nodes_local)
    assert len(edges_store) == len(edges_local)

def test_invalid_bundle_rejected():
    # Create invalid bundle
    invalid_bundle = create_invalid_bundle()
    upload_bundle(supabase, "test/invalid.zip", invalid_bundle)
    
    with pytest.raises(ValueError, match="Invalid"):
        load_project_from_store(supabase, "test-project")

def test_upload_still_works():
    # Existing upload flow should be unchanged
    nodes_df = pd.read_csv(uploaded_nodes_file)
    edges_df = pd.read_csv(uploaded_edges_file)
    results = run_analysis(nodes_df, edges_df, "uploaded")
    assert results is not None
```

---

## Non-Goals

- Writing back to Project Store from InsightGraph (read-only for now)
- Real-time sync with OrgGraph
- Project permissions/sharing

---

## Dependencies

- Phase 2.1 (Project Store Table)
- Phase 2.2 (OrgGraph Save) — so there's data to load

## Unlocks

- End-to-end GLFN demo without manual steps
- Any project created by OrgGraph is instantly analyzable
