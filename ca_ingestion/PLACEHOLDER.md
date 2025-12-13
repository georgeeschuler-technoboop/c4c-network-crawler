# GLFN Demo Data

This folder should contain the pre-built canonical graph for the GLFN demo.

## Required Files

1. **nodes.csv** — Canonical ORG + PERSON nodes
2. **edges.csv** — Canonical GRANT + BOARD_MEMBERSHIP edges
3. **org_attributes.json** — (optional) Org metadata

## How to Populate

### Option A: Use the Upload Mode

1. Run the CA ingestion app: `streamlit run ca_ingestion/app.py`
2. Select "New Project (upload)"
3. Upload charitydata.ca CSVs for each GLFN org
4. Download the ZIP output
5. Unzip and copy `nodes.csv` and `edges.csv` here

### Option B: Merge Multiple Orgs

If processing multiple organizations:

```python
import pandas as pd

# Load individual org outputs
nodes1 = pd.read_csv("org1/nodes.csv")
nodes2 = pd.read_csv("org2/nodes.csv")

# Concatenate and deduplicate
all_nodes = pd.concat([nodes1, nodes2]).drop_duplicates(subset=["node_id"])
all_nodes.to_csv("demo_data/glfn/nodes.csv", index=False)

# Same for edges
```

## Schema Reference

See `README.md` in repo root for canonical schema columns.
