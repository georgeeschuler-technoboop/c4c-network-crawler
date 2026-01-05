# Sprint Backlog — January 5, 2026

## Phase Status Overview

| Phase        | Description                            | Status         |
| ------------ | -------------------------------------- | -------------- |
| **Phase 1**  | CoreGraph Schema Standardization       | ✅ Complete     |
| **Phase 2**  | Cloud Storage (Supabase)               | ✅ Complete     |
| **Phase 3A** | Multi-Project Merge                    | ✅ Complete     |
| **Phase 3B** | Project Management UI                  | ✅ Complete     |
| **Phase 4**  | Entity Linking (ActorGraph ↔ OrgGraph) | ✅ Complete     |
| **Phase 5**  | Cross-Network Analysis (People × Orgs) | 🔲 Not Started |

---

## Recent Completed Work (Jan 5, 2026)

### ✅ YAML Copy Manager Integration — COMPLETE

**Apps:** InsightGraph  
**Versions:** `funder_analyzer.py` v1.2.0, `social_analyzer.py` v1.2.0, `run.py` v4.1.0  
**Status:** ✅ Deployed and verified

**What was done:**
- Implemented YAML-driven copy management for health labels, role definitions, tooltips, and report templates
- Single source of truth: `INSIGHTGRAPH_COPY_MAP_v1.yaml`
- Health vocabulary now uses: Fragile / Constrained / Moderate / Strong (from YAML)

**Files deployed:**
```
insights/
├── __init__.py                       ← NEW (package marker)
├── run.py                            ← v4.1.0 (clean package imports)
├── copy_manager.py                   ← NEW (YAML loader)
├── INSIGHTGRAPH_COPY_MAP_v1.yaml    ← NEW (copy strings)
└── analyzers/
    ├── __init__.py                   ← Unchanged
    ├── funder_analyzer.py            ← v1.2.0
    └── social_analyzer.py            ← v1.2.0
```

**Bug fixes resolved during integration:**

| Issue | Root Cause | Fix |
|-------|------------|-----|
| `DiGraph.add_edge() got multiple values for 'weight'` | `row.to_dict()` included weight key | Exclude 'weight' from row dict |
| `InsightCard.__init__() unexpected keyword 'details'` | Wrong field names | Use `evidence`, `health_factors` |
| `ProjectSummary.__init__() unexpected keyword 'project_id'` | Field doesn't exist | Use `generated_at`, `network_type`, `source_app` |
| `AnalysisResult.__init__() unexpected keyword 'brokerage_data'` | Wrong field name | Use `brokerage` |
| `BrokerageData` invalid `summary` field | Field is `interpretation` | Use defaults for disabled state |
| `ImportError: attempted relative import beyond top-level package` | `run.py` sys.path manipulation broke package structure | Removed sys.path hack, use absolute imports |
| `ImportError: No module named 'yaml'` | PyYAML not installed | Added `pyyaml` to requirements.txt |

**Verification markers:**
- Report header shows: `**Analyzer:** v1.2.0-2026-01-06-package-fix`
- Report footer shows: `(YAML)` instead of `(fallback: ...)`
- Health label shows: `Fragile` instead of `Fragmented / siloed`

---

## Recent Bug Fixes (Dec 25 - Jan 5)

| Item | App | Version | Status |
|------|-----|---------|--------|
| Hidden broker calculation | InsightGraph V1 | v3.0.23 | ✅ Done |
| Funding metrics calculation | InsightGraph V2 | v0.15.6 | ✅ Done |
| Blank page / API issues | ActorGraph | v0.5.8 | ✅ Done |
| iPad Safari XML guardrail | OrgGraph US | v0.23.1 | ✅ Done |
| YAML Copy Manager Integration | InsightGraph | v1.2.0 | ✅ Done |
| Package structure fix | InsightGraph | run.py v4.1.0 | ✅ Done |
| Dataclass signature mismatches | InsightGraph | v1.1.3-v1.2.0 | ✅ Done |

---

## 📋 Full Backlog — Connecting for Change Apps

### 1. Download Simplification ✅ COMPLETE

**Apps:** ActorGraph, OrgGraph US, OrgGraph CA, InsightGraph  
**Priority:** High  
**Effort:** ~1-2 hours per app  
**Status:** 
- ✅ OrgGraph US v0.24.0 — Done
- ✅ ActorGraph v0.5.6 — Done
- ✅ InsightGraph v0.15.9 — Done
- ✅ OrgGraph CA v0.16.0 — Done

**Description:** Collapse 6+ download buttons into two clear actions:

| Action | Button | What it does |
|--------|--------|--------------|
| **Download** | 📦 Download ZIP | Everything in one bundle |
| **Cloud** | ☁️ Save to Cloud | Upload to Supabase |

**ZIP contents (with clear naming):**
```
{project_name}_export.zip
├── README.md                    # What each file is, column definitions
├── {project_name}_nodes.csv
├── {project_name}_edges.csv
├── {project_name}_grants_detail.csv
├── manifest.json
└── polinode/
    ├── {project_name}_polinode_nodes.csv
    ├── {project_name}_polinode_edges.csv
    └── {project_name}_polinode.xlsx
```

**Why:** Current UI is cluttered and confusing. Users just want their data.

---

### 2. Project Browser (Supabase Front-End) ⭐ NEW

**Apps:** New standalone app ("C4C Project Hub")  
**Priority:** High  
**Effort:** ~4-6 hours to MVP  
**Description:** Lightweight Streamlit app for George and Sarah to manage cloud projects without diving into Supabase dashboard.

**Core features:**

| Feature | Description |
|---------|-------------|
| Browse projects | List all projects with metadata (app, date, node/edge counts) |
| Filter/search | By source app, date range, project name |
| Preview data | View nodes/edges in a table without downloading |
| Download bundle | Grab the ZIP for any project |
| Manage | Rename, delete, toggle public/private |
| Cross-references | Show linked projects (e.g., ActorGraph ↔ OrgGraph) |

**Future considerations:**
- Notes/comments field for collaboration
- Tags or status (draft/reviewed/published)
- Client access (read-only view)?

---

### 3. Board Detail CSV Export and Interlock Detection

**Apps:** OrgGraph US, OrgGraph CA  
**Priority:** High  
**Effort:** ~3-4 hours  
**Description:** Export board member details as CSV. Detect board interlocks (same person on multiple boards) and flag in output. Currently board data is parsed but not fully surfaced.

**Outputs:**
- `{project_name}_board_members.csv` — all board members with org affiliations
- `{project_name}_interlocks.csv` — people serving on 2+ boards

---

### 4. Move Advanced Analytics to InsightGraph

**Apps:** ActorGraph → InsightGraph  
**Priority:** High  
**Effort:** ~4-6 hours  
**Description:** Transfer advanced analytics functions (broker detection, centrality analysis, hidden influencer identification) from ActorGraph to InsightGraph.

- **ActorGraph** becomes: Ingestion + basic stats + cloud save
- **OrgGraph** becomes: Ingestion + basic stats + cloud save
- **InsightGraph** becomes: The dedicated tool for in-depth network analysis

This aligns with the "systems briefing generator" product definition.

**DoD:** ActorGraph/OrgGraph produce clean CoreGraph + basic stats only; InsightGraph owns interpretation + narrative.

---

### 5. GLFN Demo Workflow Validation

**Apps:** All three  
**Priority:** High  
**Effort:** ~2 hours  
**Description:** Run complete end-to-end workflow and validate all outputs.

**Pass/Fail Checklist:**

| # | Check | Status |
|---|-------|--------|
| 1 | OrgGraph US: ingest → ZIP + cloud save works | 🔲 |
| 2 | OrgGraph CA: ingest → ZIP + cloud save works | 🔲 |
| 3 | ActorGraph: crawl → ZIP + cloud save works | 🔲 |
| 4 | InsightGraph: loads cloud projects → links entities → generates report | 🔲 |
| 5 | InsightGraph: report shows `(YAML)` footer + correct vocab labels | 🔲 |
| 6 | End-to-end: manifest + README presence and correct filenames in ZIPs | 🔲 |

**DoD:** All 6 checks pass. This becomes the baseline regression suite.

---

### 6. Improved File Naming and README in ZIP Downloads

**Apps:** InsightGraph, OrgGraph US, OrgGraph CA  
**Priority:** Medium  
**Status:** Partially addressed by #1 (Download Simplification)  
**Description:**
- Use descriptive filenames: `{project_name}_nodes.csv` instead of `nodes.csv`
- Include README.md in ZIP with: project metadata, column definitions, data provenance, generation timestamp

---

### 7. UI Design Cleanup and Modernization

**Apps:** All  
**Priority:** Medium  
**Effort:** ~1 day per app  
**Description:**
- Overhaul UI to be cleaner and more modern
- Add hover instructions or `?` icons for guidance
- Clarify user flow: obvious where to start and why each step matters
- Simplify color scheme and layout to reduce clutter
- Apply glassmorphism design language from mockups

---

### 8. Overview and Quick Start Documents

**Apps:** ActorGraph, OrgGraph, InsightGraph  
**Priority:** Medium  
**Description:**
- **Overview doc** for each app: purpose, key features, what it produces
- **Quick start guide** for each app: step-by-step to first output
- Could be embedded in app (collapsible "Getting Started" section) or as downloadable PDF

---

### 9. Metrics Calculation Document for Data Science Review

**Apps:** InsightGraph (primary)  
**Priority:** High  
**Status:** ✅ v1.0 Complete — Awaiting Sarah's review  

---

### 10. Phase 5: Cross-Network Analysis (People × Orgs)

**Apps:** InsightGraph  
**Priority:** Medium  
**Effort:** ~1 day  
**Description:** People × Organizations insights:
- Shared board members across funders/grantees
- Boundary spanners connecting different clusters
- Funder influence paths through governance ties

Analyze how board members connect organizations, identify influential individuals spanning multiple entities.

---

### 11. AI Assistant as Multiple Experts (Research Item)

**Apps:** InsightGraph  
**Priority:** Low (Research)  
**Description:** Explore multi-expert AI personas for diverse network interpretation perspectives.

---

### 12. YAML Copy Vocabulary Refinement ⭐ NEW

**Apps:** InsightGraph  
**Priority:** Low  
**Status:** Infrastructure complete, vocabulary refinement deferred  
**Description:** Replace remaining technical terminology with accessible language in `INSIGHTGRAPH_COPY_MAP_v1.yaml`. Infrastructure is now in place; copy refinement can happen anytime by editing the YAML file.

**Vocabulary mapping opportunity:**
| Current Term | Accessible Alternative |
|--------------|------------------------|
| Betweenness centrality | Bridging influence |
| Eigenvector centrality | Network prominence |
| Louvain communities | Natural clusters |

**DoD:** No jargon leakage in report and UI tooltips for the same term (copy map covers both).

---

## Backlog Priority Matrix

| Priority | Items |
|----------|-------|
| **High** | Project Browser (#2), Board Detail Export (#3), Analytics Migration (#4), GLFN Demo (#5), Metrics Doc Review (#9) |
| **Medium** | File Naming (#6), UI Modernization (#7), Quick Start Docs (#8), Phase 5 (#10) |
| **Low/Research** | AI Multiple Experts (#11), YAML Vocabulary Refinement (#12) |
| **Complete** | Download Simplification (#1), YAML Copy Manager Integration |

---

## Recommended Sprint Sequence

```
Sprint N (Current - Jan 5):
  ├── ✅ YAML Copy Manager Integration (COMPLETE)
  └── 🔲 #5 GLFN Demo Validation (6-point checklist)

Sprint N+1:
  ├── #2 Project Browser MVP ← Platform glue, enables Sarah collaboration
  │     (creates test harness for downloads, previews, metadata)
  └── #3 Board Detail Export (client-visible feature)

Sprint N+2:
  ├── #4 Advanced Analytics Migration (architectural cleanup)
  └── #10 Phase 5 Cross-Network Analysis (People × Orgs)

Sprint N+3:
  ├── #7 UI Modernization
  └── #8 Quick Start Docs
```

**Note:** Project Browser (#2) before Board Interlocks (#3) because it creates the shared surface for Sarah and becomes the test harness for all multi-project work.

---

## App Version Summary

| App | Current Version | Last Update |
|-----|-----------------|-------------|
| OrgGraph US | v0.24.0 | Download simplification, BOM fix |
| OrgGraph CA | v0.16.0 | Download simplification |
| ActorGraph | v0.5.6 | Download simplification |
| InsightGraph (app.py) | v0.15.9 | Download simplification |
| InsightGraph (run.py) | v4.1.0 | Package structure fix |
| InsightGraph (funder_analyzer) | v1.2.0 | YAML copy integration |
| InsightGraph (social_analyzer) | v1.2.0 | YAML copy integration |

---

## Test Data Available

| Dataset | Source | Stats |
|---------|--------|-------|
| Great Lakes Funders | OrgGraph US | 2,781 nodes |
| Great Lakes Funders (CA) | OrgGraph CA | 495 nodes, 863 edges |
| Healing Our Waters Coalition | ActorGraph | 109 nodes, 116 edges |
| Entity-linked network | InsightGraph | ~70 matches |

---

## Technical Debt Resolved (Jan 5)

| Issue | Resolution |
|-------|------------|
| Hardcoded health labels | Now YAML-driven via `copy_manager.py` |
| Hardcoded `source_app='orggraph_us'` | Now detected from base class |
| Local timestamps | Now UTC (`datetime.now(timezone.utc)`) |
| `sys.path` manipulation in run.py | Removed; clean package imports |
| Missing `insights/__init__.py` | Added for proper package structure |

### Packaging Guardrail (Permanent)

**Smoke test to prevent regression:**
```bash
python -c "import insights; import insights.analyzers"
```

Add to CI or verify in Streamlit start logs. This catches `sys.path` hacks that break relative imports.
