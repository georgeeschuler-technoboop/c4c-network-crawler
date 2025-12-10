Below is a polished, repo-ready README for app_990.py, written so future you, Sarah, or any developer instantly understands the purpose, structure, workflow, and how to extend it later.

You can copy/paste directly into README.md or into a dedicated README_990.md.

⸻

C4C 990 Funder Flow Prototype

(app_990.py)

The C4C 990 Funder Flow Prototype is a Streamlit-based tool that extracts funder–grantee relationships and board interlocks from IRS 990/990-PF filings and converts them into network-ready CSV files (nodes + edges).
These files can be immediately imported into network-mapping tools like Polinode and eventually integrated into the C4C Network Intelligence Engine.

This app represents a second data rail alongside the existing LinkedIn Seed Crawler:
	•	Rail 1 → Professional networks (LinkedIn via EnrichLayer)
	•	Rail 2 → Philanthropic & governance networks (IRS 990 + board interlocks)

⸻

🚀 What This Tool Does

Given one or more IRS 990/990-PF PDF filings, the app:
	1.	Parses foundation metadata
	•	Foundation name
	•	EIN
	•	Tax year
	2.	Extracts the grants schedule
	•	Grantee name + location
	•	Grant amount
	•	Grant purpose text
	3.	Extracts the board/officer table
	•	Trustee/Director names
	•	Roles
	•	Optional city/state
	4.	Builds three downloadable CSVs:

1. grants.csv

Flat table of all grant line items across uploaded filings.

2. nodes.csv

Unified node list including:
	•	Foundations
	•	Grantees
	•	People (board members)

Each node gets a stable unique ID:

FNDN_<EIN>  
ORG_<SLUG(grantee name)>  
PERSON_<SLUG(person name)>

3. edges.csv

Two edge types:
	•	grant → FNDN → ORG
	•	board_membership → PERSON → FNDN

Ready to drop into Polinode for immediate network visualization.

⸻

📁 Repo Structure

repo/
  app.py                     # C4C LinkedIn Seed Crawler
  app_990.py                 # 990 Funder Flow Prototype (this tool)
  c4c_utils/
    __init__.py
    irs990_parser.py         # PDF parsing → grants_df, people_df, foundation_meta
    network_export.py        # Build nodes_df + edges_df from parsed data
  README.md (or README_990.md)
  requirements.txt


⸻

🧠 How It Works

1. Upload 990 PDF(s)

Users upload one or more filings (e.g., Porter Family Foundation).

2. Parsing

irs990_parser.py extracts:
	•	Foundation header
	•	Grants schedule
	•	Board/officer table

Uses pdfplumber or camelot/tabula-py for table extraction.

3. Normalization

Grant rows and people rows are standardized into consistent schemas.

4. Network Construction

network_export.py generates:

nodes.csv
Columns:

node_id, label, type, city, state, country, source

edges.csv
Columns:

from_id, to_id, edge_type, grant_amount, tax_year,
grant_purpose_raw, role, start_year, end_year,
foundation_name, grantee_name, source_file


⸻

🖥️ Running the App

From within the repo:

streamlit run app_990.py

Requirements (add to requirements.txt)

streamlit
pandas
pdfplumber      # or camelot-py[cv] / tabula-py depending on implementation


⸻

✔ Current MVP Capabilities
	•	Parse one or more filings
	•	Extract grants schedule
	•	Extract board/officer names and roles
	•	Build consistent nodes and edges tables
	•	Show previews of all tables inside Streamlit
	•	Provide download buttons for CSVs
	•	Graceful fallback if tables are missing (e.g., malformed PDF)

⸻

🧭 Future Extensions

This prototype is designed to scale into several future features:

1. Multi-funder ecosystem maps

Upload dozens of filings → instantly visualize:
	•	Co-funding patterns
	•	Overlaps and gaps
	•	Geographic clusters
	•	Potential snowball fundraising opportunities

2. Board interlocks across multiple orgs

Identify:
	•	Power brokers
	•	Highly connected trustees
	•	Governance bottlenecks
	•	Recruitment candidates based on adjacency

3. Integration with LinkedIn data

Cross-walk 990 board members with LinkedIn profiles to:
	•	Fill missing attributes
	•	Map professional pathways
	•	Suggest recruitment candidates

4. Integration with the C4C Seed Crawler

Full Intelligence Engine =
People networks + Funding networks + Organizational networks
from a single unified schema.

⸻

🧪 Test Filings

We recommend beginning with:
	•	Porter Family Foundation (Great Lakes Water Funder Network member)
	•	Another 1–2 Great Lakes funders for variation in table layout

The goal is robust heuristics, not perfect coverage of all 990s.

⸻

🤝 For Developers

Key functions:

irs990_parser.parse_990_pdf()
Returns:

{
  "foundation_meta": {...},
  "grants_df": pd.DataFrame,
  "people_df": pd.DataFrame,
}

network_export.build_nodes_df()
network_export.build_edges_df()

Testing Approach
	•	Unit tests for slugify_name
	•	Component tests on grants extraction
	•	Visual inspection of nodes/edges in Polinode

⸻

📣 Credits & Context

This prototype is inspired by work with:
	•	Great Lakes Water Funder Network
	•	Circle of Blue & Jon Allan
	•	Polinode team (Chad Taberna + Nat Bulkley)

The vision is to map where money, trust, and expertise already flow — and use that to guide:
	•	Funding strategy
	•	Board recruitment
	•	Basin partnerships
	•	Philanthropy alignment

This tool becomes one of the “keys to the castle.”

## 🗺️ Roadmap (Short-Term)

This prototype is being built in small, testable steps. The immediate priorities are:

1. **Get one 990 working end-to-end (Porter)**
   - Parse foundation metadata, grants, and board members.
   - Generate valid `grants.csv`, `nodes.csv`, and `edges.csv`.
   - Confirm that `nodes.csv` + `edges.csv` load cleanly in Polinode.

2. **Handle a second 990 with a slightly different layout**
   - Add a second Great Lakes funder as a test case.
   - Refine parsing heuristics so both filings work without code changes.

3. **Improve UX and error handling**
   - Clear status messages when parsing succeeds or fails.
   - Graceful handling of malformed or unusual PDFs (no crashes).

4. **Prepare for integration with the Network Intelligence Engine**
   - Keep the `nodes.csv` and `edges.csv` schema stable.
   - Ensure IDs and `edge_type` values are consistent with other rails
     (e.g., LinkedIn-based networks).
     
     ## 🤝 Working with a Developer

If a developer is helping on this project, here’s how to get them oriented quickly:

1. **Start with the goal**
   - Share this README and the idea in one sentence:
     > “We want to turn IRS 990 filings into network CSVs that show who funds whom and who sits on which boards.”

2. **Point them to the core files**
   - `app_990.py` – Streamlit UI
   - `c4c_utils/irs990_parser.py` – PDF → grants + people
   - `c4c_utils/network_export.py` – grants + people → nodes + edges

3. **Give them a test file**
   - Provide at least one real 990-PF PDF (e.g., Porter Family Foundation).
   - Tell them: “If this one works end-to-end, we’re happy for now.”

4. **Use small, concrete tasks**
   - Parse foundation + grants (no people yet).
   - Then add board/officer parsing.
   - Then build `nodes.csv` and `edges.csv`.
   - Then refine for a second foundation.

5. **How we’ll review work**
   - We’ll run `streamlit run app_990.py`.
   - We’ll upload the test 990.
   - We’ll check:
     - Do the tables look roughly correct?
     - Do the CSVs open cleanly?
     - Do `nodes.csv` + `edges.csv` import into Polinode without ID mismatches?
     

