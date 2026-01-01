INSIGHTGRAPH v1.0 — Product Promise

Status: Active Development → Targeting Beta
Scope: v1.0
Audience: Internal users, collaborators, and pilot partners

Purpose

InsightGraph exists to turn network data into interpretable, decision-ready insight.
Version 1.0 defines a stable analytical contract: what InsightGraph guarantees it can do, what it explicitly does not attempt to do, and how results should be interpreted.

This document is the authoritative reference for:
	•	Feature expectations
	•	Bug vs enhancement decisions
	•	Documentation accuracy
	•	Lab Console status (Active Dev → Beta → Stable)

If functionality is not listed here, it is not guaranteed in v1.0.

⸻

What InsightGraph v1.0 Guarantees

1. Network-Type-Aware Analysis

InsightGraph v1.0 correctly distinguishes and analyzes different network types, including:
	•	Social / affiliation networks (ActorGraph)
	•	Funding and governance networks (OrgGraph US / CA)
	•	Coalition or advocacy networks
	•	Hybrid datasets produced via entity linking

Each network is analyzed according to its structure, not with a one-size-fits-all metric set.

⸻

2. Deterministic Core Metrics

For any valid input network, InsightGraph v1.0 reliably computes:
	•	Node and edge counts
	•	Degree and weighted degree
	•	Component structure
	•	Density
	•	Centrality measures appropriate to network type
	•	Brokerage and intermediary indicators

Metric definitions, assumptions, and formulas are documented in
InsightGraph — Metrics Calculation Guide v2.0.

Given the same inputs and configuration, InsightGraph produces repeatable results.

⸻

3. Brokerage Role Identification

InsightGraph v1.0 identifies structural roles within a network, including:
	•	Brokers / boundary spanners
	•	Highly central actors
	•	Bridge nodes connecting communities
	•	Peripheral but strategically positioned actors

These roles are derived from network structure, not inferred intent or qualitative judgment.

InsightGraph does not assign value judgments; interpretation remains the responsibility of the analyst.

⸻

4. Entity-Linked Network Overlap Analysis

When provided with multiple networks sharing common entities (e.g., organizations appearing in both funding and advocacy networks), InsightGraph v1.0 can:
	•	Match exact and near-exact entities
	•	Surface overlaps and non-overlaps
	•	Quantify alignment between networks
	•	Flag entities requiring manual review

This capability supports questions such as:
	•	“Which advocacy actors are foundation-funded?”
	•	“Where do influence and resources fail to overlap?”

⸻

5. Human-Readable Insight Outputs

InsightGraph v1.0 produces clear, structured outputs designed for interpretation, including:
	•	Summary tables
	•	Highlighted actors and roles
	•	Interpretable narrative cues
	•	Shareable HTML reports

Outputs are designed to support analysis and storytelling, not automated decision-making.

⸻

What InsightGraph v1.0 Explicitly Does Not Promise

InsightGraph v1.0 does not guarantee:
	•	Predictive modeling or forecasting
	•	Causal inference
	•	Sentiment analysis
	•	Automated recommendations or prescriptive actions
	•	Real-time or streaming analysis
	•	Exhaustive data completeness (results depend on inputs)

InsightGraph is an analysis and sense-making tool, not an AI oracle.

⸻

Stability & Support Expectations

Under the v1.0 promise:
	•	Metric definitions will not change without versioning
	•	Output schemas remain stable
	•	Breaking changes require a major version increment
	•	Bugs that violate this promise are treated as defects, not enhancements

⸻

Version Status
	•	Current: Active Development
	•	Next: Beta (upon scope lock)
	•	Stable: After Phase 5 lands or after a defined soak period with no promise violations

⸻

Summary

InsightGraph v1.0 is a trustworthy analytical layer for understanding complex networks.
It prioritizes transparency, interpretability, and structural insight over automation or prediction.

⸻

2. Phase 5 Guardrails — What Can and Cannot Change

This is the part that prevents future pain.

Phase 5 May Introduce

✅ Allowed without breaking v1.0:
	•	New analytical modules layered on top of existing metrics
	•	Cross-network path analysis
	•	Scenario views or comparative lenses
	•	Optional AI-assisted interpretation clearly labeled as assistive
	•	Performance improvements
	•	UX enhancements
	•	Additional output formats

⸻

Phase 5 May Not Change (Without v2.0)

🚫 Not allowed under v1.0:
	•	Changing existing metric definitions
	•	Reinterpreting brokerage roles without versioning
	•	Altering core schemas silently
	•	Replacing deterministic metrics with probabilistic ones
	•	Auto-generating recommendations presented as “answers”
	•	Removing interpretability in favor of black-box outputs

If any of the above are desired, that is v2.0 territory.

⸻

Phase 5 Rule of Thumb

Phase 5 may add lenses, not rewrite the foundation.

If it changes how someone would interpret a v1.0 result, it must be:
	•	Versioned
	•	Documented
	•	Explicitly opt-in

⸻

3. Developer Handoff Reminders (Actionable)

You can copy-paste this directly to your developer.

⸻

🔒 Backlog Update Required

Add:
Phase 4F — InsightGraph v1.0 Scope Lock & Stabilization

Type: Governance / Scope Lock
Not Feature Work

Includes:
	•	Finalize INSIGHTGRAPH_V1_PROMISE.md
	•	Confirm alignment with Metrics Guide v2.0
	•	Confirm Quick Start claims match promise
	•	Flag any known deviations

⸻

🧭 Lab Console Status Rules
	•	Current: active_dev
	•	Move to beta:
	•	Once Phase 4F is complete
	•	Promise document finalized
	•	Move to stable:
	•	After Phase 5 lands or
	•	After agreed soak period (e.g., 30–60 days) with no promise violations

⸻

🔁 Ongoing Rule

If a change:
	•	Breaks a promise → bug
	•	Adds capability → Phase 5
	•	Rewrites assumptions → new major version

⸻
