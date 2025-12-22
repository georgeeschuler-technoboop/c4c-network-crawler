# C4C InsightGraph — Report Authoring Guide
Version: 1.0  
Status: Canonical  
Applies to: InsightGraph / OrgGraph / Network Intelligence reports

---

## 1. Purpose of InsightGraph Reports

InsightGraph reports surface **structural signals** in funding and organizational networks to support:
- strategic sensemaking
- prioritization
- coordination conversations
- expert-facilitated decision processes

These reports are designed to **inform judgment**, not replace it.

---

## 2. What These Reports Are NOT

InsightGraph reports do NOT:
- evaluate funder or grantee performance
- prescribe actions or “best practices”
- prove intent, alignment, or impact
- provide complete or causal explanations

All signals are **directional and contextual**.

---

## 3. Canonical Report Structure (Required)

Every report MUST include:

1. Title + metadata
2. Executive summary
3. Table of contents
4. Analytical sections  
5. Data notes / methodology
6. Manifest & traceability links

---

## 4. Executive Summary Contract

The Executive Summary must:
- State clearly what the report *does and does not* do
- Present 4–6 **plain-language signals**
- Include one sentence answering:  
  *“How do teams typically use this report?”*

Avoid:
- metrics without interpretation
- section-by-section repetition
- recommendations

---

## 5. Section Authoring Rules (Critical)

Each analytical section MUST include:

1. **Section intro (1–2 sentences)**  
   - What this section examines  
   - Why it matters at a high level  

2. **Decision Lens block (required)**  
3. **Supporting analysis / tables / visuals**

No section with metrics may ship without a Decision Lens.

---

## 6. Decision Lens Contract (Required Everywhere)

The Decision Lens translates analysis into **decision context**.

Each Decision Lens MUST include:

### Required fields
- **Signal intensity**: Low / Moderate / High
- **What this tells you**  
- **Why it matters**
- **What teams often do next**
- **What not to over-interpret**

### Language rules
Use:
- “signals suggest…”
- “may indicate…”
- “teams often use this to decide…”

Avoid:
- “should / must”
- “we recommend”
- “best practice”
- “natural partners / hubs” (unless explicitly qualified)

### Normalization rule
Every Decision Lens must explicitly permit **no action** as a valid outcome.

---

## 7. Signal Intensity Guidance

- **Low-intensity signal**  
  Contextual or confirmatory; often used to rule things out

- **Moderate-intensity signal**  
  Worth discussion or light exploration

- **High-intensity signal**  
  Merits closer review or facilitated conversation

---

## 8. Portfolio Twins (Special Handling)

This section MUST explicitly state:

- Shared grantees ≠ aligned strategy
- Low overlap is normal and often healthy
- Primary use is deciding **where coordination is NOT needed**

Never imply action from overlap alone.

---

## 9. Markdown → HTML Rendering Contract

Markdown is the source of truth.

The final HTML MUST NOT contain raw markdown artifacts:
- `_italic_`
- `**bold**`
- backticks
- fenced code blocks

All markdown must be rendered before injection into HTML templates.

---

## 10. Definition of Done (Quality Gate)

A report is valid when:
- Every analytical section has a Decision Lens
- No prescriptive language appears
- Signal intensity is present and appropriate
- “No action” is normalized where relevant
- No raw markdown appears in HTML
- A non-technical reader can answer:
  - What does this tell me?
  - Why does it matter?
  - What kind of decision could this inform?

---
End of guide.
