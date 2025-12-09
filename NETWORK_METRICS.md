# Phase 2: Network Centrality Metrics

## Overview

Network centrality metrics are now calculated automatically for **all crawls** (not just advanced mode). These metrics help identify:

- **Connectors** — People with the most connections
- **Brokers** — People who bridge different groups  
- **Influencers** — People connected to influential others
- **Accessible Hubs** — People who can reach anyone quickly

---

## Metrics Calculated

### Node-Level Metrics

| Metric | What It Measures | Identifies |
|--------|------------------|------------|
| **Degree Centrality** | Number of direct connections (normalized) | Connectors |
| **Betweenness Centrality** | How often node lies on shortest paths | Brokers/Bridges |
| **Eigenvector Centrality** | Connected to well-connected people | Influencers |
| **Closeness Centrality** | Average distance to all other nodes | Accessible hubs |

### Network-Level Statistics

| Metric | Description |
|--------|-------------|
| **Nodes** | Total people in network |
| **Edges** | Total connections |
| **Density** | Proportion of possible connections that exist (0-1) |
| **Avg Degree** | Average connections per person |
| **Avg Clustering** | How much nodes cluster together |
| **Components** | Number of disconnected subgroups |
| **Diameter** | Longest shortest path in network |

---

## Where Metrics Appear

### 1. UI Display (Always Shown)

After any crawl, you'll see:

```
📊 Network Centrality Metrics

Network Overview:
[Nodes: 964] [Edges: 1000] [Density: 0.0021] [Avg Degree: 2.07] [Avg Clustering: 0.0234]

📈 Components: 3 | Largest component: 950 nodes

---

🔗 Top Connectors (by Degree)        🌉 Top Brokers (by Betweenness)
Most direct connections               Bridge between groups

1. Sarah Chen (WWF) — 45 connections  1. Michael Brown (TNC) — 0.1234
2. John Smith (WRI) — 38 connections  2. Lisa Park (Ceres) — 0.0987
...                                    ...

📍 Top Accessible (by Closeness)      ⭐ Top Influencers (by Eigenvector)
Shortest average distance to others   Connected to well-connected people

1. Julia Roig (Pacific Inst) — 0.45   1. Dara Parker (CDP) — 0.3421
...                                    ...
```

### 2. nodes.csv (Enhanced)

New columns added to nodes.csv:

```csv
id,name,profile_url,headline,location,degree,source_type,organization,sector,connections,degree_centrality,betweenness_centrality,eigenvector_centrality,closeness_centrality
alice-chen,Alice Chen,https://...,Director at WWF,DC,0,seed,WWF,Nonprofit,45,0.0234,0.1234,0.3421,0.4532
```

### 3. network_analysis.json (New File)

New JSON file included in ZIP download:

```json
{
  "generated_at": "2025-12-08T16:50:00.000000",
  "network_statistics": {
    "nodes": 964,
    "edges": 1000,
    "density": 0.0021,
    "avg_degree": 2.07,
    "avg_clustering": 0.0234,
    "num_components": 3,
    "largest_component_size": 950
  },
  "top_connectors": [
    {"id": "alice-chen", "name": "Alice Chen", "organization": "WWF", "degree_centrality": 0.0234, "connections": 45},
    ...
  ],
  "top_brokers": [
    {"id": "michael-brown", "name": "Michael Brown", "organization": "TNC", "betweenness_centrality": 0.1234},
    ...
  ],
  "top_influencers": [
    {"id": "dara-parker", "name": "Dara Parker", "organization": "CDP", "eigenvector_centrality": 0.3421},
    ...
  ],
  "metric_definitions": {
    "degree_centrality": "Number of direct connections (normalized). High = well-connected.",
    "betweenness_centrality": "How often node lies on shortest paths. High = broker/bridge.",
    "eigenvector_centrality": "Connected to influential people. High = influential.",
    "closeness_centrality": "Average distance to all others. High = central/accessible."
  }
}
```

---

## Technical Implementation

### Dependencies

Added to `requirements.txt`:
```
networkx>=3.0
```

### Functions Added

```python
def calculate_network_metrics(seen_profiles: Dict, edges: List) -> Dict:
    """
    Calculate network centrality metrics using NetworkX.
    
    Returns:
        node_metrics: {node_id: {metric: value, ...}, ...}
        network_stats: {metric: value, ...}
        top_nodes: {metric: [(node_id, value), ...], ...}
    """

def generate_network_analysis_json(network_metrics: Dict, seen_profiles: Dict) -> str:
    """Generate network_analysis.json with summary statistics and top nodes."""
```

### Error Handling

- Gracefully handles disconnected graphs
- Eigenvector centrality falls back if graph is problematic
- Closeness centrality calculated on largest connected component if needed
- All calculations wrapped in try/except to prevent crashes

---

## Interpreting Results

### High Degree Centrality
- **Connectors** with many direct relationships
- May be well-networked leaders or public figures
- Risk: Bottleneck if they leave

### High Betweenness Centrality  
- **Brokers** who bridge different groups
- Often cross-sector or cross-organizational
- Strategic for spreading information
- C4C focus: These are your key network weavers!

### High Eigenvector Centrality
- **Influencers** connected to other influencers
- Access to power and resources
- May not have most connections but have best connections

### High Closeness Centrality
- **Accessible hubs** who can reach anyone quickly
- Good for rapid information dissemination
- Central to the network structure

---

## Use Cases for C4C

### Identifying Network Weavers
Look for people high in **betweenness** who also have moderate **eigenvector** scores — these are your cross-sector bridges with access to influence.

### Finding Hidden Influencers
People with high **eigenvector** but lower **degree** — connected to the right people but not obviously networked.

### Detecting Network Vulnerabilities
People high in multiple metrics — if they leave, the network fragments.

### Cross-Sector Analysis (with Advanced Mode)
Combine centrality metrics with organization/sector data to identify:
- Cross-sector brokers (high betweenness + multiple sectors)
- Organizational bridges (connect different orgs)
- Sector influencers (high eigenvector within sector)

---

## Performance Notes

- Metrics calculated after crawl completes
- For 1000+ nodes, calculation takes 1-3 seconds
- For 5000+ nodes, may take 5-10 seconds
- UI shows spinner during calculation

---

## Files Updated

- **app.py** — Core metrics calculation and display
- **requirements.txt** — Added networkx>=3.0
- **NETWORK_METRICS.md** — This documentation

---

## What's Next (Roadmap)

### Community Detection (Phase 3)
- Algorithmic cluster identification (Louvain, etc.)
- Modularity scores
- Visualize communities

### Brokerage Matrix (Phase 4)
- Gould & Fernandez brokerage roles
- Coordinators, gatekeepers, representatives, liaisons
- Structural hole analysis

### Strategic Insights (Phase 5)
- AI-generated narrative analysis
- Gap identification
- Collaboration opportunity recommendations

---

## Quick Test

1. Run mock mode with Degree 2
2. Scroll to "📊 Network Centrality Metrics"
3. Verify:
   - Network Overview shows 5 metrics
   - Top 5 shown for each category
   - Metric definitions expandable
4. Download ZIP and verify network_analysis.json included
