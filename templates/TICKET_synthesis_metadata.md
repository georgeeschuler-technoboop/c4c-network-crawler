# 🎟️ Developer Ticket: Add Synthesis Metadata + Artifacts to InsightGraph Export

## Title
Add synthesis metadata and guidance artifacts to InsightGraph project export (manifest.json)

---

## Context / Why

We want InsightGraph exports to support downstream synthesis (infographics, slides, AI tools like NotebookLM) in a non-prescriptive, decision-oriented way.

This ticket adds descriptive metadata + guidance files to the export bundle so:
- humans (George / Sarah) are reminded how to synthesize responsibly
- AI tools have guardrails
- future tools can consume the manifest consistently

This is **metadata only** — no change to analytics, scoring, or report logic.

---

## Files to Touch

| File | Action |
|------|--------|
| `insights/run.py` | Modify `generate_manifest()` to include synthesis block |
| `c4c_utils/lint_report.py` | Add optional synthesis validation check |
| `templates/guides/VISUAL_SYNTHESIS_GUIDE.md` | **NEW** - copy from provided file |
| `templates/guides/SYNTHESIS_MODE_PROMPT.md` | **NEW** - copy from provided file |
| `templates/guides/SYNTHESIS_CHECKLIST.md` | **NEW** - copy from provided file |

---

## What to Implement

### 1️⃣ Extend `generate_manifest()` in `insights/run.py`

The `generate_manifest()` function (around line 2242) already builds the manifest dict. Add a `synthesis` block before the `return manifest` statement.

**Location:** `insights/run.py`, function `generate_manifest()`

**Add this block** (already implemented in v3.0.22):

```python
# Add synthesis metadata (guidance for downstream tools)
manifest["synthesis"] = {
    "purpose": (
        "Guidance for generating non-prescriptive visual or narrative "
        "summaries of this report."
    ),
    "visual_synthesis_guide": "guides/VISUAL_SYNTHESIS_GUIDE.md",
    "synthesis_mode_prompt": "guides/SYNTHESIS_MODE_PROMPT.md",
    "synthesis_checklist": "guides/SYNTHESIS_CHECKLIST.md",
    "intended_use": [
        "NotebookLM",
        "slide generation tools",
        "infographic drafting",
        "facilitated discussion prep"
    ],
    "constraints": [
        "Do not imply recommendations from structural signals",
        "Preserve signal intensity labels (Low/Moderate/High)",
        "Include 'no action may be appropriate' framing",
        "Use 'teams often use this to decide' phrasing"
    ]
}
```

Also update the `outputs` section to include synthesis_guides:

```python
"synthesis_guides": {
    "visual_synthesis_guide": "guides/VISUAL_SYNTHESIS_GUIDE.md",
    "synthesis_mode_prompt": "guides/SYNTHESIS_MODE_PROMPT.md",
    "synthesis_checklist": "guides/SYNTHESIS_CHECKLIST.md"
}
```

---

### 2️⃣ Copy synthesis guide files to export directory

In the export function (wherever the bundle is written to disk), copy the three guide files:

**Option A: In `insights/app.py` export logic**

```python
import shutil
import os

# After writing manifest.json, copy synthesis guides
guides_dir = os.path.join(export_dir, "guides")
os.makedirs(guides_dir, exist_ok=True)

template_guides = Path(__file__).parent.parent / "templates" / "guides"
for guide_file in ["VISUAL_SYNTHESIS_GUIDE.md", "SYNTHESIS_MODE_PROMPT.md", "SYNTHESIS_CHECKLIST.md"]:
    src = template_guides / guide_file
    if src.exists():
        shutil.copy(src, guides_dir / guide_file)
```

**Option B: Embed in run.py as constants (simpler)**

If you prefer not to manage separate template files, the guide content can be embedded as constants in run.py and written directly during export.

---

### 3️⃣ Add synthesis guide files to repo

Create the directory `templates/guides/` and add these three files (provided separately):

```
c4c-network-crawler/
├── templates/
│   └── guides/
│       ├── VISUAL_SYNTHESIS_GUIDE.md
│       ├── SYNTHESIS_MODE_PROMPT.md
│       └── SYNTHESIS_CHECKLIST.md
```

---

### 4️⃣ (Optional) Update `c4c_utils/lint_report.py`

Add a validation check that ensures synthesis metadata exists when a report is present.

**Add this function:**

```python
def check_synthesis_metadata(export_dir: str) -> List[LintIssue]:
    """Validate synthesis metadata and files exist in export bundle."""
    issues = []
    manifest_path = Path(export_dir) / "manifest.json"
    
    if not manifest_path.exists():
        return issues  # No manifest, nothing to check
    
    manifest = json.loads(manifest_path.read_text())
    
    # Check if report exists
    has_report = (Path(export_dir) / "index.html").exists() or \
                 (Path(export_dir) / "report.md").exists()
    
    if not has_report:
        return issues  # No report, synthesis not required
    
    # Check synthesis block
    if "synthesis" not in manifest:
        issues.append(LintIssue(
            "MISSING_SYNTHESIS_METADATA",
            "manifest.json missing 'synthesis' block",
            "Export bundles with reports should include synthesis metadata"
        ))
        return issues
    
    # Check referenced files exist
    synthesis = manifest["synthesis"]
    for key in ["visual_synthesis_guide", "synthesis_mode_prompt", "synthesis_checklist"]:
        if key in synthesis:
            file_path = Path(export_dir) / synthesis[key]
            if not file_path.exists():
                issues.append(LintIssue(
                    "MISSING_SYNTHESIS_FILE",
                    f"Referenced synthesis file not found: {synthesis[key]}",
                    f"Expected at: {file_path}"
                ))
    
    return issues
```

---

## Acceptance Criteria ✅

- [ ] `manifest.json` includes a top-level `synthesis` object with the exact keys above
- [ ] `manifest.json` `outputs` section includes `synthesis_guides` paths
- [ ] All referenced `.md` files exist in the export `guides/` directory
- [ ] No existing export behavior is broken
- [ ] No analytics or report logic is modified
- [ ] Export bundle is self-describing and tool-agnostic

---

## Non-Goals ❌

- No changes to InsightGraph scoring or insights
- No UI changes
- No hard-coding for specific tools (NotebookLM, etc.)
- No embedding long prompt text inside JSON

---

## Testing

1. Run InsightGraph export on GLFN dataset
2. Verify `manifest.json` contains `synthesis` block
3. Verify `guides/` folder contains all three `.md` files
4. Run `python -m c4c_utils.lint_report path/to/export/index.html`
5. (Optional) Run synthesis check on export directory

---

## Files Provided

The following files are ready to drop into `templates/guides/`:

1. `VISUAL_SYNTHESIS_GUIDE.md` - Visual element guidelines
2. `SYNTHESIS_MODE_PROMPT.md` - AI tool instructions
3. `SYNTHESIS_CHECKLIST.md` - Pre-publication review checklist

---

## Why This Matters

This enables:
- consistent infographic + slide drafts across projects
- safer AI-assisted synthesis
- a scalable workflow as synthesis tools evolve

Think of this as **epistemic metadata**, not presentation logic.

---

*Ticket created: 2025-12-22*
*Related: C4C Report Authoring Contract v1.2*
