# C4C Network Intelligence Engine

Tools for parsing nonprofit data and generating network graphs for analysis.

## Quick Start

### Run the US 990 Funder Flow App

```bash
cd funder_flow
streamlit run app.py
```

### Run the CA Charity Ingestion App

```bash
cd ca_ingestion
streamlit run app.py
```

## Project Modes

Both apps support two modes:

| Mode | Description |
|------|-------------|
| **GLFN Demo (pre-loaded)** | Loads pre-built canonical graph from `demo_data/glfn/`. No uploads needed. |
| **New Project (upload)** | Upload source files to generate a new graph. |

## Demo Data Structure

For the GLFN Demo to work, ensure these files exist:

```
demo_data/
└── glfn/
    ├── nodes.csv          # Canonical ORG + PERSON nodes
    ├── edges.csv          # Canonical GRANT + BOARD_MEMBERSHIP edges
    └── org_attributes.json  # (optional) Org metadata
```

## Canonical Schema (v1 MVP)

Both adapters output the same schema for cross-source compatibility.

**nodes.csv columns:**
```
node_id, node_type, label, org_slug, jurisdiction, tax_id,
city, region, source_system, source_ref, assets_latest, assets_year,
first_name, last_name
```

**edges.csv columns:**
```
edge_id, from_id, to_id, edge_type,
amount, amount_cash, amount_in_kind, currency,
fiscal_year, reporting_period, purpose,
role, start_date, end_date, at_arms_length,
city, region, source_system, source_ref
```

**Node ID Formats:**
- Organizations: `org:<org_slug>`
- People: `person:<context_org>:<name_key>`

**Edge Types:**
- `GRANT` — foundation → grantee
- `BOARD_MEMBERSHIP` — person → foundation

**Source Systems:**
- `IRS_990` — US private foundation filings
- `CHARITYDATA_CA` — Canadian charity data

## Repository Structure

```
c4c-network-crawler/
├── app.py                    # LinkedIn network crawler (Streamlit)
├── requirements.txt
├── c4c_utils/
│   ├── irs990_parser.py      # 990-PF PDF parser
│   └── network_export.py     # Canonical schema builder
├── funder_flow/
│   └── app.py                # US 990-PF Funder Flow (Streamlit)
├── ca_ingestion/
│   └── app.py                # CA charity ingestion (Streamlit)
└── demo_data/
    └── glfn/
        ├── nodes.csv         # Pre-built GLFN graph
        └── edges.csv
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with:

**For US 990 Funder Flow:**
- **Main file path:** `funder_flow/app.py`

**For CA Charity Ingestion:**
- **Main file path:** `ca_ingestion/app.py`

### Local Development

```bash
pip install -r requirements.txt

# Run US 990 app
streamlit run funder_flow/app.py

# Run CA app
streamlit run ca_ingestion/app.py
```

## Data Sources

| Source | Jurisdiction | Data | App |
|--------|--------------|------|-----|
| IRS 990-PF PDFs | USA | Foundation grants, board | funder_flow |
| [charitydata.ca](https://charitydata.ca) | Canada | Assets, directors, grants | ca_ingestion |

## License

Internal use only — Connecting for Change LLC
