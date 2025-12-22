# TICKET: Overwrite Protection (Phase 2.4)

**Priority:** Medium  
**Effort:** Small  
**Apps:** OrgGraph US, OrgGraph CA  
**Depends On:** Phase 2.2 (OrgGraph Save)

---

## Summary

Prevent silent overwrites when two users save to the same project simultaneously using optimistic concurrency control on `bundle_version`.

---

## Background

Without protection, this race condition can occur:

```
User A: Read bundle (version 1)
User B: Read bundle (version 1)
User A: Merge + Upload (version 2)
User B: Merge + Upload (version 2) ← OVERWRITES User A's changes!
```

This ticket adds version checking to prevent silent data loss.

---

## Tasks

### 1. Create atomic version increment function

```sql
-- Supabase RPC function for atomic version check + increment
CREATE OR REPLACE FUNCTION increment_bundle_version(
  p_project_id UUID,
  p_expected_version INTEGER,
  p_row_counts JSONB,
  p_source_apps TEXT[]
)
RETURNS TABLE (
  success BOOLEAN,
  new_version INTEGER,
  current_version INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
  v_current_version INTEGER;
  v_new_version INTEGER;
BEGIN
  -- Get current version with lock
  SELECT bundle_version INTO v_current_version
  FROM c4c_projects
  WHERE project_id = p_project_id
  FOR UPDATE;
  
  -- Check if version matches expected
  IF v_current_version != p_expected_version THEN
    RETURN QUERY SELECT 
      FALSE as success,
      NULL::INTEGER as new_version,
      v_current_version as current_version;
    RETURN;
  END IF;
  
  -- Increment version and update metadata
  v_new_version := v_current_version + 1;
  
  UPDATE c4c_projects
  SET 
    bundle_version = v_new_version,
    row_counts = p_row_counts,
    source_apps = p_source_apps,
    updated_at = NOW()
  WHERE project_id = p_project_id;
  
  RETURN QUERY SELECT 
    TRUE as success,
    v_new_version as new_version,
    v_current_version as current_version;
END;
$$;
```

### 2. Update save flow with version check

```python
# orggraph/project_store.py

class VersionConflictError(Exception):
    """Raised when bundle version has changed since read."""
    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(
            f"Version conflict: expected {expected}, found {current}. "
            "Project was updated by another user."
        )


def save_project_bundle(
    supabase: Client,
    project_id: str,
    new_bundle: bytes,
    source_app: str,
    max_retries: int = 1
) -> dict:
    """
    Save bundle to Project Store with optimistic concurrency.
    
    Raises:
        VersionConflictError: If project was modified since read
    """
    for attempt in range(max_retries + 1):
        try:
            return _save_with_version_check(
                supabase, project_id, new_bundle, source_app
            )
        except VersionConflictError as e:
            if attempt < max_retries:
                # Retry once: re-read, re-merge, re-save
                continue
            raise


def _save_with_version_check(
    supabase: Client,
    project_id: str,
    new_bundle: bytes,
    source_app: str
) -> dict:
    """Internal save with version check."""
    
    # Step 1: Read current state
    project = get_project(supabase, project_id)
    expected_version = project['bundle_version']
    bundle_path = project['bundle_path']
    
    # Step 2: Download and merge
    existing_bundle = download_bundle(supabase, bundle_path)
    
    if existing_bundle:
        merged_bundle, stats = merge_bundles(existing_bundle, new_bundle, source_app)
    else:
        merged_bundle = new_bundle
        stats = get_bundle_stats(new_bundle)
        stats['sources'] = [source_app]
    
    # Step 3: Validate
    validate_bundle(merged_bundle)
    
    # Step 4: Upload bundle (this is safe - blob storage)
    upload_bundle(supabase, bundle_path, merged_bundle)
    
    # Step 5: Atomic version check + increment
    result = supabase.rpc('increment_bundle_version', {
        'p_project_id': project_id,
        'p_expected_version': expected_version,
        'p_row_counts': {'nodes': stats['nodes'], 'edges': stats['edges']},
        'p_source_apps': stats['sources']
    }).execute()
    
    row = result.data[0]
    
    if not row['success']:
        # Version mismatch - another user saved first
        # Note: Bundle was uploaded but metadata not updated
        # This is acceptable - next save will re-merge
        raise VersionConflictError(
            expected=expected_version,
            current=row['current_version']
        )
    
    stats['bundle_version'] = row['new_version']
    return stats
```

### 3. Add UI handling for conflicts

```python
# In app.py save button handler

try:
    stats = save_project_bundle(
        supabase,
        project_id=current_project['project_id'],
        new_bundle=bundle,
        source_app='orggraph',
        max_retries=1  # Auto-retry once
    )
    st.success(f"✅ Saved! {stats['nodes']} nodes, {stats['edges']} edges")
    
except VersionConflictError as e:
    st.error(
        f"⚠️ Project was updated by another user.\n\n"
        f"Your version: {e.expected}, Current version: {e.current}\n\n"
        f"Please reload the project and try again."
    )
    
    if st.button("🔄 Reload Project"):
        st.experimental_rerun()
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `supabase/migrations/004_version_rpc.sql` | NEW: Create RPC function |
| `orggraph/project_store.py` | MODIFY: Add version check logic |
| `orggraph/app.py` | MODIFY: Handle VersionConflictError |

---

## Sequence Diagram

```
User A                    User B                    Supabase
  │                         │                          │
  ├─── Read (v1) ──────────────────────────────────────►
  │                         ├─── Read (v1) ───────────►│
  │                         │                          │
  ├─── Merge locally        │                          │
  │                         ├─── Merge locally         │
  │                         │                          │
  ├─── Upload blob ─────────────────────────────────────►
  ├─── increment_version(v1) ──────────────────────────►
  │    ◄─── success, v2 ───────────────────────────────┤
  │                         │                          │
  │                         ├─── Upload blob ──────────►
  │                         ├─── increment_version(v1) ►
  │                         │    ◄─── FAIL, current=v2 ┤
  │                         │                          │
  │                         ├─── Show error ───────────│
  │                         │                          │
```

---

## Acceptance Criteria

- [ ] `increment_bundle_version` RPC function exists
- [ ] Save fails gracefully if version changed
- [ ] User sees clear error message on conflict
- [ ] Auto-retry once before failing
- [ ] No corrupted bundles created (blob upload is safe)
- [ ] `bundle_version` increments only on successful save
- [ ] Reload button allows user to recover

---

## Testing

```python
def test_version_conflict_detected():
    # Simulate race condition
    project = get_project(supabase, project_id)
    v1 = project['bundle_version']
    
    # Another "user" saves first
    _force_increment_version(supabase, project_id)
    
    # Our save should fail
    with pytest.raises(VersionConflictError) as exc:
        save_project_bundle(supabase, project_id, bundle, 'orggraph')
    
    assert exc.value.expected == v1
    assert exc.value.current == v1 + 1

def test_auto_retry_succeeds():
    # First attempt fails, retry succeeds
    # (Mock the conflict to resolve on second try)
    stats = save_project_bundle(
        supabase, project_id, bundle, 'orggraph',
        max_retries=1
    )
    assert stats is not None

def test_no_silent_overwrite():
    # User A and B both read v1
    project_v1 = get_project(supabase, project_id)
    
    # User A saves successfully
    save_project_bundle(supabase, project_id, bundle_a, 'orggraph')
    
    # User B tries to save with stale version
    # Should fail, not silently overwrite
    with pytest.raises(VersionConflictError):
        _save_with_version_check(
            supabase, project_id, bundle_b, 'orggraph'
        )
```

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| First save (version 0) | Works normally, version becomes 1 |
| Same user re-saves | Works (version increments) |
| Two users, one saves first | Second user gets conflict error |
| Conflict + auto-retry | Re-reads, re-merges, succeeds |
| Conflict + max retries exceeded | Shows error, user must reload |

---

## Non-Goals

- Real-time collaboration / locking
- Merge conflict resolution UI
- Per-record locking (whole bundle is atomic)

---

## Dependencies

- Phase 2.2 (OrgGraph Save) — this adds safety to that flow

## Unlocks

- Safe multi-user editing
- No silent data loss
