# C4C Report Authoring Guide

**Applies to:** InsightGraph / OrgGraph-derived reports  
**Version:** 1.1  
**Last updated:** 2025-12-22

---

## 1. What these reports are

C4C reports surface **decision-relevant structural signals** in a funding/network ecosystem. They are designed to support:
- shared understanding
- prioritization
- coordination conversations
- deeper inquiry with expert facilitation

## 2. What these reports are not

Reports are **not**:
- evaluations of performance
- prescriptive recommendations
- proof of intent, alignment, or impact
- complete representations of reality (data is partial)

---

## 3. Canonical Report Structure (required)

Every report must follow this structure:

1. **Title**
2. **Executive Summary**
3. **Table of Contents**
4. **Sections** (each section must include: intro + Decision Lens + analysis)
5. **Data Notes / Methodology**
6. **Manifest / Traceability**

---

## 4. Executive Summary Contract (required)

Executive Summary must include:
- One paragraph: what this report does / doesn't do
- 4–6 "Key Signals" bullets (plain language)
- One sentence: "How teams typically use this report"

Rules:
- No section-by-section repetition
- No recommendations
- Minimize metrics (include only if essential)

---

## 5. Section Intro Rules (required)

Each section begins with a 1–2 sentence intro that answers:
- What is this about?
- Why does it matter?

Avoid:
- "This section analyzes…"
- dense method descriptions
- numeric claims without context

---

## 6. Decision Lens Contract (required for every analytical section)

Each analytical section MUST include a Decision Lens block that answers:

1. **Signal intensity** (Low / Moderate / High)  
2. **What this tells you** (plain-language interpretation)  
3. **Why it matters** (decision relevance)  
4. **What teams often do next** (descriptive actions, not prescriptions)  
5. **What not to over-interpret** (guardrail)  
6. **Permission for no action** (explicit normalization — included globally in component)

### Language constraints

**Allowed phrasing:**
- "signals suggest…"
- "may indicate…"
- "can be used to assess…"
- "teams often use this to decide…"
- "could be worth exploring"
- "a potential touchpoint"
- "worth a conversation"
- "Teams often use this signal to decide whether…"
- "This signal is commonly used to assess whether…"

**Disallowed phrasing (hard):**
- "should"
- "must"
- "we recommend"
- "best practice"
- "natural partners" (unless qualified with uncertainty)
- "natural hub" (unless qualified)
- "strong consensus" (use "shared investment priorities" instead)

**Disallowed phrasing (soft guidance drift):**
- "This suggests…"
- "This points to…"
- "This highlights an opportunity…"
- "This reveals an opportunity…"
- "This indicates potential for…"
- "Opportunity to…"

**Required replacement pattern:**
Use descriptive usage framing:
- ❌ "This suggests you should coordinate"
- ✅ "Teams often use this signal to decide whether coordination would add value"

---

## 7. Signal Intensity Framework

Every section must declare a signal intensity level:

| Level | Meaning | Reader Action |
|-------|---------|---------------|
| **Low** | Primarily confirmatory / contextual | No action typically required |
| **Moderate** | Worth discussion or light exploration | Consider a brief check-in |
| **High** | Merits active follow-up or strategy review | Prioritize for discussion |

Sections currently assigned:
- **Low:** Portfolio Twins, Roles × Region, Board Conduits, Isolated Funders
- **Moderate:** Network Health, Funding Concentration, Multi-Funder Grantees, Hidden Brokers, Bridges
- **High:** (reserved for urgent/dynamic findings)

---

## 8. Portfolio Overlap / "Portfolio Twins" Specific Contract

This section must explicitly state:
- shared touchpoints ≠ aligned strategy
- low similarity is normal and often healthy
- primary practical use is to decide where coordination is NOT needed

Every pair listing must include:
- Shared grantees (count)
- Portfolio sizes (A size, B size)
- Similarity score (Jaccard or equivalent)
- A plain label: "low / moderate / high overlap"

Never imply coordination necessity from overlap alone.

---

## 9. Markdown → HTML Rendering Contract

Markdown is the source of truth.

The HTML must never contain raw markdown artifacts such as:
- `_italic_` → must become `<em>italic</em>`
- `**bold**` → must become `<strong>bold</strong>`
- backticks for inline code when unintended

All markdown content must be rendered to HTML before insertion into the HTML template.

**Known patterns to catch:**
- Section subtitles (use `<p class="section-subtitle">` directly)
- Decision Lens text (preprocess before markdown library)

---

## 10. Definition of Done (report quality)

A report is "valid" when:

- [ ] Every section contains intro + Decision Lens + analysis
- [ ] No prescriptive language appears ("should", "must", "recommend")
- [ ] No semantic inflation ("natural partners", "strong consensus")
- [ ] No soft guidance drift ("this suggests", "this indicates", "opportunity to")
- [ ] Signal intensity is present everywhere
- [ ] "No action required" is normalized where appropriate
- [ ] No raw markdown artifacts appear in the final HTML output
- [ ] "Strategic Recommendations" renamed to "Decision Options"
- [ ] A non-technical reader can answer:
  - What is this telling me?
  - Why does it matter?
  - What kind of decision could this inform?
  - (Without being told what to do)

---

## 11. Traceability Contract

Each report must link to:
- manifest (inputs, config, version)
- source markdown (report.md)
- outputs (csv/xlsx/json artifacts)

---

## 12. Quick Reference: Find & Replace

| Find | Replace With |
|------|--------------|
| "natural partners" | "potential coordination touchpoints" |
| "natural hub" | "shared investment node (context needed)" |
| "strong consensus" | "shared investment priorities" |
| "should" | "could" / "may" / "consider" |
| "must" | "may want to" / "could" |
| "recommend" | "some teams find value in" |

---

*End of guide.*
