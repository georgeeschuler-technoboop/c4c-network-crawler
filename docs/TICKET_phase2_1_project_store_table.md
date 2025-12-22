# TICKET: Project Store Index Table (Phase 2.1)

**Priority:** High  
**Effort:** Small  
**Apps:** Infrastructure / Supabase  
**Blocks:** Phase 2.2, 2.3

---

## Summary

Create `c4c_projects` table as a Project Store index (metadata only), providing a stable project selector and metadata registry without storing nodes/edges in database tables.

---

## Background

Current workflow requires manual GitHub uploads to share compiled projects between apps. The Project Store provides:

- Central registry of projects
- Metadata for bundle discovery
- Foundation for OrgGraph save and InsightGraph load

**Key principle:** Bundle blob is stored in Supabase Storage, not in tables. The index table only tracks metadata.

---

## Tasks

### 1. Create Supabase table

```sql
CREATE TABLE IF NOT EXISTS c4c_projects (
  project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  description TEXT,
  schema_version TEXT NOT NULL DEFAULT 'c4c_coregraph_v1',
  bundle_path TEXT NOT NULL,
  row_counts JSONB DEFAULT '{}',
  source_apps TEXT[] DEFAULT '{}',
  bundle_version INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_projects_updated ON c4c_projects(updated_at DESC);
CREATE INDEX idx_projects_name ON c4c_projects(name);

-- Enable RLS
ALTER TABLE c4c_projects ENABLE ROW LEVEL SECURITY;

-- Policy: authenticated users can read/write
CREATE POLICY "Users can manage projects"
  ON c4c_projects
  FOR ALL
  USING (auth.role() = 'authenticated')
  WITH CHECK (auth.role() = 'authenticated');
```

### 2. Create Supabase Storage bucket

```sql
-- Create storage bucket for project bundles
INSERT INTO storage.buckets (id, name, public)
VALUES ('project-bundles', 'project-bundles', false);

-- Policy: authenticated users can read/write
CREATE POLICY "Users can manage bundles"
  ON storage.objects
  FOR ALL
  USING (bucket_id = 'project-bundles' AND auth.role() = 'authenticated')
  WITH CHECK (bucket_id = 'project-bundles' AND auth.role() = 'authenticated');
```

### 3. Create shared helper module (optional but recommended)

```python
# c4c_utils/project_store.py

from supabase import Client
from typing import Optional, List
import json

def list_projects(supabase: Client) -> List[dict]:
    """List all projects ordered by updated_at."""
    response = supabase.table('c4c_projects') \
        .select('*') \
        .order('updated_at', desc=True) \
        .execute()
    return response.data

def get_project(supabase: Client, project_id: str) -> Optional[dict]:
    """Get a single project by ID."""
    response = supabase.table('c4c_projects') \
        .select('*') \
        .eq('project_id', project_id) \
        .single() \
        .execute()
    return response.data

def create_project(
    supabase: Client,
    name: str,
    bundle_path: str,
    schema_version: str = 'c4c_coregraph_v1',
    description: str = None
) -> dict:
    """Create a new project."""
    response = supabase.table('c4c_projects').insert({
        'name': name,
        'description': description,
        'schema_version': schema_version,
        'bundle_path': bundle_path,
        'row_counts': {},
        'source_apps': [],
        'bundle_version': 0
    }).execute()
    return response.data[0]

def update_project_metadata(
    supabase: Client,
    project_id: str,
    row_counts: dict = None,
    source_apps: List[str] = None,
    expected_version: int = None
) -> dict:
    """
    Update project metadata.
    If expected_version provided, uses optimistic concurrency.
    """
    update_data = {'updated_at': 'NOW()'}
    
    if row_counts is not None:
        update_data['row_counts'] = row_counts
    if source_apps is not None:
        update_data['source_apps'] = source_apps
    
    query = supabase.table('c4c_projects') \
        .update(update_data) \
        .eq('project_id', project_id)
    
    if expected_version is not None:
        query = query.eq('bundle_version', expected_version)
    
    # Increment version
    # Note: This requires a separate RPC call for atomic increment
    # See Ticket 2.4 for full implementation
    
    response = query.execute()
    return response.data[0] if response.data else None
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `supabase/migrations/003_project_store.sql` | NEW: Create table + bucket |
| `c4c_utils/project_store.py` | NEW: Shared helper module |

---

## Acceptance Criteria

- [ ] `c4c_projects` table exists in Supabase
- [ ] `project-bundles` storage bucket exists
- [ ] Can create a project row with name + bundle_path
- [ ] Can update `row_counts` and `source_apps`
- [ ] Projects can be listed ordered by `updated_at`
- [ ] RLS policies allow authenticated users to read/write
- [ ] **No nodes/edges stored in Postgres** (bundle blob only)

---

## Testing

```python
def test_create_project():
    project = create_project(
        supabase,
        name="Great Lakes Funders Network",
        bundle_path="project-bundles/glfn/bundle.zip"
    )
    assert project['name'] == "Great Lakes Funders Network"
    assert project['bundle_version'] == 0

def test_list_projects():
    projects = list_projects(supabase)
    assert isinstance(projects, list)
    # Should be ordered by updated_at desc
    if len(projects) > 1:
        assert projects[0]['updated_at'] >= projects[1]['updated_at']

def test_update_metadata():
    update_project_metadata(
        supabase,
        project_id=project['project_id'],
        row_counts={'nodes': 2920, 'edges': 5000},
        source_apps=['orggraph']
    )
    updated = get_project(supabase, project['project_id'])
    assert updated['row_counts']['nodes'] == 2920
```

---

## Non-Goals

- Storing node/edge data in tables (Phase 3)
- Complex access control (single-tenant for now)
- Project sharing/permissions UI

---

## Dependencies

None — this is the foundation ticket for Phase 2.

## Unlocks

- Ticket 2.2 (OrgGraph Save & Merge)
- Ticket 2.3 (InsightGraph Load)
