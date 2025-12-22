# C4C Report Authoring Guide  
**Version:** 1.0  
**Applies to:** InsightGraph, OrgGraph-derived reports  
**Audience:** Developers, analysts, facilitators, report authors

---

## 1. Purpose of This Guide

This guide defines the **authoring contract** for all C4C Network Intelligence reports.

Its goals are to ensure that every report:
- is accessible to non-technical readers
- clearly explains *why patterns matter*
- supports real-world decision-making
- can be generated consistently from Markdown into HTML, PDF, or other formats

Markdown is the **single source of truth**.

---

## 2. Core Design Principle

> **C4C reports surface decision-relevant signals, not prescriptions.**

Reports:
- explain *what is happening* and *why it matters*
- do **not** tell clients what they “should” do
- create a shared foundation for expert interpretation, facilitation, and strategy work

---

## 3. Canonical Report Structure (Required)

Every report must follow this structure.

```markdown
# {{ Project Name }}

## Executive Summary
<paragraph>

- Key signal 1
- Key signal 2
- Key signal 3
- Key signal 4

---

## {{ Section Title }}

<Section intro paragraph>

:::decision-lens
**What this tells you**  
...

**Why it matters for decisions**  
...

**What teams often do next**  
...
:::

<Analysis content>
