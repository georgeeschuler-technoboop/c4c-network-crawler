# C4C Lab Console — System Guide

**LAB_CONSOLE_SYSTEM_GUIDE.md**

Last updated: January 1, 2026  
Audience: Internal users (Sarah, George), developers, future maintainers  
Status: Canonical reference  
Console version: v1.7 | Manifest version: v1.8 | CloudProjects version: v0.5.0

---

## 1. What the C4C Lab Console Is

The C4C Lab Console is the single canonical interface for accessing:

- Live C4C network analysis apps (ActorGraph, OrgGraph, InsightGraph, etc.)
- Shared demo inputs and outputs (stored in Supabase)
- Quick Start guides and technical documentation
- System status (what's stable, beta, alpha, or in active development)

Its purpose is to eliminate:

- "Which link is the right one?"
- Out-of-date PDFs floating around Slack or email
- Re-explaining workflows to collaborators and developers

**If something is not visible in the Lab Console, it should be assumed non-canonical or deprecated.**

---

## 2. Two Access Points: Lab Console + CloudProjects

The system now has two complementary interfaces:

| Interface | Purpose | Auth Required | Best For |
|-----------|---------|---------------|----------|
| **Lab Console** | Dashboard view — status, apps, docs, artifacts | No | Quick access, overview, sharing with stakeholders |
| **CloudProjects** | Workspace view — projects, docs, artifacts | Projects tab: Yes / Docs tab: No | Hands-on work, downloading files, managing saved projects |

Both read from the same `_manifest.json`, ensuring consistency.

**Lab Console** = "What do we have?"  
**CloudProjects** = "Let me work with it."

---

## 3. High-Level System Architecture

The system is intentionally simple and split into clear responsibilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                         │
├────────────────────────────┬────────────────────────────────────┤
│   Lab Console (Wix embed)  │   CloudProjects (Streamlit)        │
│   - Dashboard view         │   - Projects tab (auth required)   │
│   - Status, apps, docs     │   - Docs & Artifacts tab (public)  │
└────────────────────────────┴────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   _manifest.json    │  ← SOURCE OF TRUTH
                    │   (Supabase)        │
                    └─────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Supabase Storage│  │ Wix Pages       │  │ Streamlit Apps  │
│ - demo/*/inputs │  │ - Quick Starts  │  │ - ActorGraph    │
│ - demo/*/outputs│  │ - /actorgraph   │  │ - OrgGraph US   │
│ - docs/*.md     │  │ - /orggraph-us  │  │ - OrgGraph CA   │
│ - docs/*.pdf    │  │ - etc.          │  │ - InsightGraph  │
│ - schema/       │  │                 │  │ - Seed Resolver │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Key Components

| Component | Responsibility |
|-----------|----------------|
| **Lab Console (HTML)** | Static page that loads `_manifest.json` and renders apps, artifacts, docs dynamically. Contains no hard-coded data. |
| **CloudProjects (Streamlit)** | Interactive app for managing saved projects + browsing docs/artifacts. Also reads from `_manifest.json`. |
| **_manifest.json** | Single source of truth for apps, docs, artifacts, versions, and status. |
| **Supabase Storage** | Stores demo inputs/outputs, MD files, PDFs. Not used to render HTML pages. |
| **Wix** | Hosts rendered HTML documentation (Quick Starts). Flat URLs only. |
| **Streamlit Apps** | Execution layer. Linked from console, not embedded. |

---

## 4. The Manifest Contract (Critical)

The Lab Console and CloudProjects do not decide anything.

They read from `_manifest.json`.

If something is missing from the UI, it is almost always because:

- It is missing from the manifest
- Or the manifest schema was changed without updating the renderer

### The Manifest Governs

- App cards (name, description, icon, URL, status, version)
- Quick Start guide links (both Wix and MD versions)
- Docs & Guides section
- Shared Project Artifacts section
- Schema references

### Design Rule (Non-Negotiable)

**Nothing should be hard-coded in the HTML that represents data.**

The HTML/Streamlit code should only define:

- Layout
- Styling
- Rendering logic

All content lives in `_manifest.json`.

---

## 5. Supabase Storage Structure

```
c4c-artifacts/
├── demo/
│   ├── _manifest.json          ← THE SOURCE OF TRUTH
│   ├── glfn/
│   │   ├── inputs/
│   │   │   ├── glfn_ca_charity_csv.zip
│   │   │   └── glfn_us_irs990_xml.zip
│   │   ├── outputs/
│   │   │   ├── glfn_polinode_export.xlsx
│   │   │   └── glfn_insightgraph_report.html
│   │   └── README.md
│   └── how/
│       ├── inputs/
│       │   └── how_seeds_6_and_10.zip
│       └── README.md
├── docs/
│   ├── quickstart_actorgraph.md
│   ├── quickstart_actorgraph.html      ← Raw backup (not used for rendering)
│   ├── quickstart_orggraph_us.md
│   ├── quickstart_orggraph_ca.md
│   ├── quickstart_insightgraph.md
│   ├── quickstart_seed_resolver.md
│   ├── quickstart_cloudprojects.md
│   └── insightgraph_metrics_v2.pdf
└── schema/
    └── README.md
```

### File Type Routing

| File Type | Stored In | Served From | Used By |
|-----------|-----------|-------------|---------|
| Quick Start HTML | Supabase (backup) | **Wix** | Lab Console, browser |
| Quick Start MD | Supabase | Supabase (direct download) | CloudProjects |
| Metrics PDF | Supabase | Supabase | Both |
| Demo ZIPs/XLSX | Supabase | Supabase | Both |
| Schema README | Supabase | Supabase | Both |

---

## 6. Apps: How They Are Defined and Rendered

Each app entry in the manifest includes:

```json
{
  "actorgraph": {
    "name": "ActorGraph",
    "description": "People-centered network graphs from LinkedIn data...",
    "url": "https://c4c-network-crawler-actorgraph.streamlit.app/",
    "icon": "https://static.wixstatic.com/media/...",
    "status": "stable",
    "version": "v0.5.6",
    "quickstart_id": "qs_actorgraph"
  }
}
```

### Status Values (Controlled Vocabulary)

| Status | Meaning | Visual |
|--------|---------|--------|
| `stable` | Safe to use, default recommendation | Blue pill |
| `beta` | Usable, still evolving | Purple pill |
| `alpha` | Experimental, unstable | Orange pill |
| `active_dev` | In heavy development | Indigo pill |
| `internal` | Internal utility only | Gray pill |

These drive visual pills, ordering, and user expectations.

---

## 7. Docs: Dual-URL Pattern

Each Quick Start has two URLs in the manifest:

```json
{
  "id": "qs_actorgraph",
  "type": "quickstart",
  "title": "Quick Start — ActorGraph",
  "notes": "Build social networks from LinkedIn. ~15 min.",
  "web_url": "https://www.connectingforchangellc.com/actorgraph",
  "md_url": "https://igbzclkhwnxnypjssdwz.supabase.co/.../quickstart_actorgraph.md"
}
```

| Field | Purpose | Used By |
|-------|---------|---------|
| `web_url` | Rendered HTML page on Wix | Lab Console (⚡ Quick Start buttons) |
| `md_url` | Raw Markdown file on Supabase | CloudProjects (⬇ Download MD buttons) |

### Why Two URLs?

- **Wix** renders nicely in browsers but isn't downloadable
- **Supabase MD** is portable and works offline

This gives users both options without duplicating content.

---

## 8. Artifacts vs Docs (This Matters)

These two sections are intentionally separate.

### Shared Project Artifacts

**What they are:**
- Demo inputs (CSV, XML, ZIP)
- Demo outputs (XLSX, HTML reports)

**Purpose:**
- Reproducible examples
- Known-good reference files
- Cross-app testing assets

**Location:** Supabase storage under `demo/<project>/inputs/` and `demo/<project>/outputs/`

**Rendered from:** `projects[*].inputs` and `projects[*].outputs`

---

### Docs & Guides

**What they are:**
- Quick Start guides (one per app)
- Metrics calculation documentation
- System and schema documentation

**Purpose:**
- Onboarding
- Trust and transparency
- Developer alignment

**Rendered from:** Top-level `docs[]` and `schema` entry

Docs are rendered similarly to artifacts for visual consistency, but they are conceptually different.

---

## 9. Why Docs Are Hosted on Wix (Important Constraint)

### The Problem

Supabase Storage:
- Serves HTML files as `text/plain`
- Does not reliably render HTML pages
- Has security restrictions by design

Edge Functions:
- Add complexity
- Have auth, CORS, and maintenance overhead
- Were intentionally avoided

### The Solution

- Render HTML documentation as Wix pages
- Use those Wix URLs in the manifest (`web_url`)
- Treat Wix as the "presentation layer" for docs only

### Consequence

- URLs are flat (e.g., `/actorgraph`, not `/documents/actorgraph`)
- Logical hierarchy exists in navigation, not URLs
- This is acceptable and intentional

### Wix Page Inventory

| Page | URL |
|------|-----|
| Documents hub | `connectingforchangellc.com/documents` |
| ActorGraph | `connectingforchangellc.com/actorgraph` |
| OrgGraph US | `connectingforchangellc.com/orggraph-us` |
| OrgGraph CA | `connectingforchangellc.com/orggraph-ca` |
| InsightGraph | `connectingforchangellc.com/insightgraph` |
| Seed Resolver | `connectingforchangellc.com/seed-resolver` |
| CloudProjects | `connectingforchangellc.com/cloudprojects` |

---

## 10. Adding or Updating Things (Developer Playbook)

### Add a New App

1. Add an entry under `apps` in `_manifest.json`
2. Include `icon`, `status`, `version`, `url`
3. (Optional) Create a Quick Start and link via `quickstart_id`
4. Upload manifest to Supabase
5. Reload console

**No HTML changes required.**

---

### Add or Update a Quick Start

1. Create or update the HTML page on Wix
2. Upload the MD version to Supabase (`docs/quickstart_<app>.md`)
3. Copy both URLs
4. Add/update entry in `docs[]`:
   - `type`: `"quickstart"`
   - `web_url`: Wix page URL
   - `md_url`: Supabase MD URL
5. Link it to an app via `quickstart_id`

---

### Add a New Demo Project

1. Create folder structure in Supabase: `demo/<project>/inputs/`, `demo/<project>/outputs/`
2. Upload input and output files
3. Create a README.md in the project folder
4. Add a new entry under `projects[]` in the manifest
5. Register inputs, outputs, and readme URL
6. Console and CloudProjects update automatically

---

### Update Version Numbers or Status

- Change only `_manifest.json`
- Never hard-code versions in HTML or Streamlit code

---

### Add a New Doc (Non-Quick Start)

1. Upload file to Supabase (`docs/` folder)
2. Add entry to `docs[]` with appropriate `type` (e.g., `"metrics"`)
3. Use `url` field (not `web_url`) for direct Supabase downloads

---

## 11. Known Constraints & Design Decisions

These are intentional, not bugs:

| Constraint | Reason |
|------------|--------|
| Flat URLs for docs | Wix limitation |
| No embedded iframes of Streamlit apps | Performance, complexity |
| No HTML rendering from Supabase | Security, reliability |
| No server-side logic in the console | Simplicity, portability |
| Manifest-driven everything | Single source of truth |
| Two access points (Console + CloudProjects) | Different use cases |

**The goal is clarity, not cleverness.**

---

## 12. If Something Breaks: Debug Checklist

### Sections Are Blank

- [ ] Does `_manifest.json` load successfully? (Check browser console for `[Lab Console]` logs)
- [ ] Is the JSON valid? (Use a JSON validator)
- [ ] Did the structure change without updating the renderer?
- [ ] Is there a network/CORS error?

### Apps Show but Buttons Don't Work

- [ ] Check `url` fields in manifest
- [ ] Check `quickstart_id` references match actual `docs[].id` values

### Docs Open as Raw HTML

- [ ] The URL points to Supabase, not Wix
- [ ] Update the manifest to use the Wix `web_url`

### Artifacts Missing

- [ ] Confirm files exist in Supabase at the specified paths
- [ ] Confirm they are registered under the correct `project` in the manifest
- [ ] Check for typos in URLs

### CloudProjects Docs Tab Empty

- [ ] Manifest fetch working? (Check Streamlit logs)
- [ ] `docs[]` array present in manifest?
- [ ] `projects[]` array present for artifacts?

---

## 13. Version History

| Date | Console | Manifest | CloudProjects | Changes |
|------|---------|----------|---------------|---------|
| 2026-01-01 | v1.7 | v1.8 | v0.5.0 | Flat Wix URLs, `md_url` fields, CloudProjects Docs & Artifacts tab |
| 2025-12-31 | v1.6 | v1.7 | v0.4.2 | Added `web_url` for Wix-hosted docs |
| 2025-12-31 | v1.5 | v1.5 | v0.4.2 | Fixed UTF-8 encoding, error handling |
| Earlier | v1.4.x | v1.x | v0.4.x | Initial manifest-driven architecture |

---

## 14. What This System Optimizes For

- **Clarity over cleverness**
- **Single source of truth**
- **Low cognitive overhead**
- **Easy handoff**
- **Future extensibility**

If the console ever feels confusing, something has drifted from these principles.

---

## 15. Quick Reference: Where Things Live

| Thing | Location |
|-------|----------|
| Manifest | `supabase/c4c-artifacts/demo/_manifest.json` |
| Console HTML | Wix embed (Lab Console page) |
| CloudProjects | Streamlit Cloud (`c4c-network-crawler-test`) |
| Quick Starts (rendered) | Wix pages (`/actorgraph`, `/orggraph-us`, etc.) |
| Quick Starts (MD) | `supabase/c4c-artifacts/docs/*.md` |
| Demo inputs | `supabase/c4c-artifacts/demo/<project>/inputs/` |
| Demo outputs | `supabase/c4c-artifacts/demo/<project>/outputs/` |
| Metrics PDF | `supabase/c4c-artifacts/docs/insightgraph_metrics_v2.pdf` |
| Schema | `supabase/c4c-artifacts/schema/README.md` |

---

## 16. Final Notes

- This file is the canonical system explanation
- App-specific behavior lives in app READMEs and Quick Starts
- The manifest is the contract
- The console and CloudProjects are lenses

If you're unsure where something belongs, ask:

**"Is this data, or is this presentation?"**

- If it's data → manifest
- If it's presentation → HTML/Streamlit

---

*End of document.*
