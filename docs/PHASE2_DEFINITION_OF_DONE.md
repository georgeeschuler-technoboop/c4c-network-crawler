# Phase 2 Definition of Done

**C4C Project Store (Bundle-First) — Completion Checklist**

Use this checklist to verify Phase 2 is complete before moving to Phase 3 (full tables) or declaring the GLFN workflow production-ready.

---

## 🔲 Phase 2.1: Project Store Index Table

### Supabase Infrastructure
- [ ] `c4c_projects` table exists
- [ ] `project-bundles` storage bucket exists
- [ ] RLS policies allow authenticated users to read/write
- [ ] Index on `updated_at` exists

### Table Schema
- [ ] `project_id` (UUID, PK)
- [ ] `name` (TEXT, NOT NULL)
- [ ] `schema_version` (TEXT, default 'c4c_coregraph_v1')
- [ ] `bundle_path` (TEXT, NOT NULL)
- [ ] `row_counts` (JSONB)
- [ ] `source_apps` (TEXT[])
- [ ] `bundle_version` (INTEGER, default 0)
- [ ] `created_at` / `updated_at` (TIMESTAMPTZ)

### Shared Helpers
- [ ] `list_projects()` returns projects ordered by updated_at
- [ ] `get_project()` returns single project by ID
- [ ] `create_project()` creates new project row
- [ ] `update_project_metadata()` updates row_counts, source_apps

---

## 🔲 Phase 2.2: OrgGraph Save & Merge

### UI Components
- [ ] Project Selector dropdown exists
- [ ] "Create new project" option works
- [ ] "Save to Project Store" button exists
- [ ] "Download ZIP" button still works

### Save Flow
- [ ] Creates new bundle on first save
- [ ] Downloads existing bundle on subsequent saves
- [ ] Merges nodes (dedupe on `node_id`)
- [ ] Merges edges (dedupe on `from_id + to_id + edge_type`)
- [ ] Validates bundle before upload
- [ ] Uploads merged bundle to Storage
- [ ] Updates project metadata

### Merge Behavior
- [ ] US run creates project + bundle
- [ ] CA run merges into same project
- [ ] Re-running US does **not** duplicate nodes/edges
- [ ] `source_apps` tracks all contributing apps
- [ ] `row_counts` reflects merged totals
- [ ] `bundle_version` increments on each save

### Validation
- [ ] Merged bundle passes `lint_report.py --validate-schema`

---

## 🔲 Phase 2.3: InsightGraph Load

### UI Components
- [ ] Data source selector (Project Store / Upload / Demo)
- [ ] Project Picker dropdown when "Project Store" selected
- [ ] Project metadata shown (updated_at, sources, counts)
- [ ] "Load Project" button works

### Load Flow
- [ ] Lists projects from `c4c_projects`
- [ ] Downloads bundle from Storage
- [ ] Validates bundle (CoreGraph v1 schema)
- [ ] Parses into DataFrames
- [ ] Passes to existing analysis pipeline

### Compatibility
- [ ] Report output identical to local file input
- [ ] "Upload files" option still works
- [ ] "Demo data" option still works
- [ ] Can export new ZIP after analysis

### Error Handling
- [ ] Missing bundle shows clear error
- [ ] Invalid bundle shows validation errors
- [ ] Empty project list shows helpful message

---

## 🔲 Phase 2.4: Overwrite Protection

### Infrastructure
- [ ] `increment_bundle_version` RPC function exists
- [ ] Function uses `FOR UPDATE` lock
- [ ] Function returns success/failure + current version

### Save Flow
- [ ] Reads `bundle_version` before save
- [ ] Checks version in atomic RPC call
- [ ] Fails gracefully on version mismatch
- [ ] Auto-retries once before failing

### UX
- [ ] User sees clear error on conflict
- [ ] Error shows expected vs current version
- [ ] "Reload Project" button available
- [ ] Reload recovers gracefully

### Safety
- [ ] Two concurrent saves cannot overwrite silently
- [ ] No corrupted bundles created
- [ ] Version only increments on successful save

---

## 🔲 Phase 2.5: Developer Documentation

### PROJECT_STORE.md
- [ ] Architecture diagram
- [ ] Data flow (save and load)
- [ ] Bundle-first philosophy
- [ ] Merge semantics explained
- [ ] Optimistic concurrency explained
- [ ] Database schema documented
- [ ] Storage layout documented
- [ ] Code examples (create, save, load)
- [ ] Troubleshooting guide
- [ ] Migration path to Phase 3

### README.md
- [ ] Project Store section added
- [ ] Quick start instructions

### Developer Test
- [ ] New developer can explain where data lives
- [ ] New developer can explain merge behavior
- [ ] New developer can explain app interaction

---

## 🎯 End-to-End Demo Checkpoint

After all Phase 2 tickets complete, verify full workflow:

### Setup
- [ ] Supabase project configured
- [ ] Storage bucket exists
- [ ] RLS policies active

### OrgGraph US Flow
- [ ] Create "GLFN" project
- [ ] Load US foundation data
- [ ] Save to Project Store
- [ ] Verify bundle in Storage
- [ ] Verify project metadata

### OrgGraph CA Flow
- [ ] Select existing "GLFN" project
- [ ] Load CA foundation data
- [ ] Save (merge) to Project Store
- [ ] Verify node count increased
- [ ] Verify no duplicates
- [ ] Verify `source_apps` updated

### InsightGraph Flow
- [ ] Select "GLFN" from Project Picker
- [ ] Load project (see merged data)
- [ ] Generate report
- [ ] Verify report quality matches expectations
- [ ] Export ZIP bundle

### Smoke Test Commands

```bash
# 1. Verify Supabase table
supabase db execute "SELECT * FROM c4c_projects"

# 2. Verify Storage bucket
supabase storage list project-bundles

# 3. Download and validate bundle
supabase storage download project-bundles/{project_id}/bundle.zip
python lint_report.py bundle/ --validate-schema
# Expected: ✅ Schema validation passed

# 4. Compare InsightGraph output
# Load same data from local files and Project Store
# Reports should be identical
```

---

## ⚠️ Known Risks / Watch Items

| Risk | Mitigation |
|------|------------|
| Bundle size limits | Supabase Storage has 50MB default limit; monitor bundle sizes |
| Merge conflicts | Optimistic concurrency prevents silent overwrites |
| Stale metadata | Always update metadata after bundle upload |
| RLS misconfiguration | Test with non-admin user |
| Network failures mid-save | Bundle uploaded but metadata not updated; next save will re-merge |

---

## Sign-Off

| Phase | Completed | Verified By | Date |
|-------|-----------|-------------|------|
| 2.1: Project Store Table | ☐ | | |
| 2.2: OrgGraph Save & Merge | ☐ | | |
| 2.3: InsightGraph Load | ☐ | | |
| 2.4: Overwrite Protection | ☐ | | |
| 2.5: Developer Docs | ☐ | | |
| **End-to-End Demo** | ☐ | | |

---

## What Phase 2 Unlocks

With Phase 2 complete:

✅ **No manual GitHub workflow** — Save directly to Project Store  
✅ **Incremental builds** — US + CA merge naturally  
✅ **InsightGraph integration** — Load any saved project  
✅ **Multi-user safety** — Optimistic concurrency prevents overwrites  
✅ **Portable bundles** — ZIP remains canonical, can still download  

**Not yet unlocked (Phase 3+):**
- Full node/edge tables in Supabase
- Real-time queries across projects
- Cross-project analysis
- ActorGraph integration (people data)

---

## Phase 2 → Phase 3 Transition

When ready for Phase 3:

1. Add `c4c_nodes` and `c4c_edges` tables (per CoreGraph v1 schema)
2. On bundle save, also write to tables (dual-write)
3. InsightGraph can query tables directly (faster for large projects)
4. Bundle remains source of truth during transition
5. Eventually, tables become primary, bundle becomes export format

Phase 2's bundle-first approach makes this migration safe and incremental.
