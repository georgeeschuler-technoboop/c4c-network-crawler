# TICKET: Developer Documentation (Phase 2.5)

**Priority:** Low  
**Effort:** Small  
**Apps:** Documentation  
**Depends On:** Phase 2.1-2.4 (all implementation complete)

---

## Summary

Document the Project Store architecture and Phase 2 usage so new developers can understand where project data lives, how it's merged, and how apps interact.

---

## Background

With Phase 2 complete, we have a working bundle-first Project Store. This ticket ensures the architecture is documented for:

- New developers joining the project
- Future maintainers
- Planning Phase 3 (full tables)

---

## Tasks

### 1. Create PROJECT_STORE.md

```markdown
# C4C Project Store Architecture

## Overview

The Project Store is a shared data layer that allows C4C apps to read and write
project data without manual file transfers.

**Phase 2 (Current):** Bundle-first architecture
- ZIP bundles stored in Supabase Storage
- Metadata index in `c4c_projects` table
- Merge semantics for incremental builds

**Phase 3 (Future):** Full table architecture
- Nodes/edges stored in Supabase tables
- Real-time queries
- Cross-project analysis

## Architecture Diagram

```
┌─────────────┐     ┌─────────────┐
│ OrgGraph US │     │ OrgGraph CA │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────┐
│         Project Store            │
├──────────────────────────────────┤
│  c4c_projects (metadata index)   │
│  project-bundles/ (ZIP storage)  │
└──────────────────────────────────┘
                │
                ▼
       ┌─────────────────┐
       │  InsightGraph   │
       └─────────────────┘
```

## Data Flow

### OrgGraph Save

1. User selects or creates a project
2. OrgGraph creates ZIP bundle (Phase 1 format)
3. If project exists, download + merge with existing bundle
4. Upload merged bundle to Supabase Storage
5. Update `c4c_projects` metadata (with version check)

### InsightGraph Load

1. User selects project from picker
2. Download bundle from Supabase Storage
3. Validate bundle (CoreGraph v1 schema)
4. Parse into DataFrames
5. Run analysis pipeline (unchanged from before)

## Key Concepts

### Bundle-First Philosophy

The ZIP bundle is the **canonical artifact**. Benefits:
- Portable (can be shared via email, Slack, etc.)
- Self-contained (includes README, manifest)
- Versionable (bundle_version tracks changes)
- Offline-capable (download and work locally)

### Merge Semantics

When multiple OrgGraph runs save to the same project:
- Nodes: dedupe on `node_id` (last write wins)
- Edges: dedupe on `(from_id, to_id, edge_type)` (last write wins)
- Source apps: accumulated in `source_apps[]`

### Optimistic Concurrency

To prevent silent overwrites:
- Each project has a `bundle_version` counter
- Save checks expected version before writing
- Version mismatch → conflict error → user reloads

## Database Schema

### c4c_projects

| Column | Type | Description |
|--------|------|-------------|
| project_id | UUID | Primary key |
| name | TEXT | Display name |
| schema_version | TEXT | CoreGraph schema version |
| bundle_path | TEXT | Path in Supabase Storage |
| row_counts | JSONB | `{nodes: N, edges: M}` |
| source_apps | TEXT[] | Apps that contributed data |
| bundle_version | INTEGER | Optimistic concurrency counter |
| created_at | TIMESTAMPTZ | Creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

### Storage Layout

```
project-bundles/
└── {project_id}/
    └── bundle.zip
```

## Migration Path to Phase 3

Phase 3 will add:
- `c4c_nodes` table (per-record storage)
- `c4c_edges` table (per-record storage)
- Real-time sync between bundle and tables
- Cross-project queries

The bundle remains the source of truth during migration.

## Common Tasks

### Create a new project (OrgGraph)

```python
from c4c_utils.project_store import create_project

project = create_project(
    supabase,
    name="My Network Analysis",
    bundle_path=f"project-bundles/{uuid4()}/bundle.zip"
)
```

### Save to project (OrgGraph)

```python
from orggraph.project_store import save_project_bundle

stats = save_project_bundle(
    supabase,
    project_id=project['project_id'],
    new_bundle=bundle_bytes,
    source_app='orggraph'
)
```

### Load from project (InsightGraph)

```python
from insightgraph.project_loader import load_project_from_store

nodes_df, edges_df, manifest = load_project_from_store(
    supabase,
    project_id=project['project_id']
)
```

## Troubleshooting

### "Version conflict" error

Another user saved while you were working. Click "Reload Project" and try again.

### "Invalid bundle" error

Bundle failed CoreGraph v1 validation. Check:
- All node_ids are namespaced (`orggraph:xxx`)
- All node_types are valid (`person`, `organization`, etc.)
- manifest.json has `schema_version: "c4c_coregraph_v1"`

### Project not appearing in InsightGraph

- Ensure OrgGraph saved successfully
- Check Supabase dashboard for `c4c_projects` row
- Verify bundle exists in `project-bundles/` bucket
```

### 2. Update main README.md

Add section to existing README:

```markdown
## Project Store

C4C apps share data through the Project Store, a Supabase-backed data layer.

- **OrgGraph** creates and saves projects
- **InsightGraph** loads and analyzes projects
- **ZIP bundles** are the canonical artifact format

See [docs/PROJECT_STORE.md](docs/PROJECT_STORE.md) for architecture details.

### Quick Start

1. In OrgGraph: Create a project and save your data
2. In InsightGraph: Select the project from the picker
3. Generate your report

No manual file transfers required!
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `docs/PROJECT_STORE.md` | NEW: Full architecture doc |
| `README.md` | MODIFY: Add Project Store section |
| `docs/SCHEMA_SPEC.md` | MODIFY: Link to Project Store doc |

---

## Acceptance Criteria

- [ ] `docs/PROJECT_STORE.md` exists and covers:
  - [ ] Architecture diagram
  - [ ] Data flow for save and load
  - [ ] Bundle-first philosophy explanation
  - [ ] Merge semantics
  - [ ] Optimistic concurrency
  - [ ] Database schema
  - [ ] Storage layout
  - [ ] Code examples
  - [ ] Troubleshooting guide
  - [ ] Migration path to Phase 3
- [ ] Main README.md updated with Project Store section
- [ ] New developer can answer:
  - [ ] Where does project data live?
  - [ ] How is data merged?
  - [ ] How do OrgGraph and InsightGraph interact?
  - [ ] What happens if two people save at once?

---

## Testing

```markdown
### Documentation Review Checklist

- [ ] Architecture diagram is accurate
- [ ] Code examples work (copy-paste test)
- [ ] Troubleshooting covers common issues
- [ ] No references to deprecated paths/methods
- [ ] Links to other docs are valid
```

---

## Non-Goals

- User-facing documentation (this is developer docs)
- API reference (code is self-documenting)
- Tutorial/walkthrough (just reference material)

---

## Dependencies

- Phase 2.1-2.4 must be complete (documenting what exists)

## Unlocks

- Onboarding new developers
- Planning Phase 3 with shared understanding
