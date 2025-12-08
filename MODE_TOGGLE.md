# Mode Toggle Implementation - Basic vs Advanced

## What Was Added

A clear mode selection interface that lets users choose between Basic and Advanced modes before starting their crawl.

---

## UI Layout

### Location
Positioned **immediately after the header**, before any input fields. This ensures users make the mode choice first, setting expectations for the entire workflow.

```
[C4C Logo] C4C Network Seed Crawler
Convert LinkedIn seed profiles...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎛️ Select Mode

[Toggle: Advanced Mode]  [Explanation box]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📥 Input
[Upload CSV...]
```

---

## Mode Explanations

### Basic Mode (Default) ✅
```
📊 Basic Mode - Quick Network Crawl

Perfect for rapid exploration:
- Crawl LinkedIn networks (1 or 2 degrees)
- Export nodes, edges, and raw profiles
- Import directly to Polinode or other tools
- Fast processing, clean data

⚡ Quick results, simple outputs
```

**When to use:**
- Quick network mapping
- Simple data export
- Testing seed lists
- Regular crawls

---

### Advanced Mode 🔬
```
🔬 Advanced Mode - Network Intelligence

Includes everything in Basic Mode plus:
- Centrality metrics (degree, betweenness, eigenvector, closeness)
- Community detection and clustering
- Brokerage analysis (coordinators, gatekeepers, liaisons)
- Key position identification (connectors, brokers, bridges)
- Network insights and strategic recommendations

⏱️ Longer processing time, richer insights
```

**When to use:**
- Strategic analysis
- Identifying key actors
- Understanding network structure
- Finding collaboration opportunities

---

## Implementation Details

### Code Structure

```python
# 1. Mode selection (top of page)
advanced_mode = st.toggle("Advanced Mode", value=False)

# Shows contextual explanation based on selection
if advanced_mode:
    st.info("🔬 Advanced Mode explanation...")
else:
    st.success("📊 Basic Mode explanation...")

# 2. Store mode with results
st.session_state.crawl_results = {
    'seen_profiles': seen_profiles,
    'edges': edges,
    'advanced_mode': advanced_mode  # Stored for later
}

# 3. Display advanced section if mode was enabled
if was_advanced_mode:
    st.header("🔬 Advanced Network Analytics")
    st.info("🚧 Coming Soon! Features include...")
```

### Session State
Mode setting is preserved with crawl results so:
- ✅ Persists across page reruns
- ✅ Survives file downloads
- ✅ Linked to specific crawl results
- ✅ Can compare different modes

---

## Current Behavior

### Basic Mode (Implemented) ✅
**Input:**
- Upload seed CSV
- Enter API token
- Choose degree (1 or 2)
- Run crawl

**Output:**
- nodes.csv
- edges.csv
- raw_profiles.json

**All features work as before**

---

### Advanced Mode (Placeholder) 🚧
**Input:**
- Same as Basic Mode

**Output:**
- Same basic files as Basic Mode
- Shows "Coming Soon" section with planned features

**Message to users:**
```
🚧 Advanced Analytics - Coming Soon!

The following features are currently in development:

Network Metrics (Next Release)
Community Detection (In Progress)
Brokerage Analysis (Planned)
Strategic Insights (Future)

For now, download basic files and import to Polinode.
```

---

## Development Roadmap

### Phase 1: Organization Extraction (Next)
**What changes:**
- Advanced mode extracts organization from API responses
- Adds `organization` column to nodes.csv
- Sets foundation for group-based analysis

**User sees:**
- Enhanced nodes.csv with org data
- "Organization data extracted" message

---

### Phase 2: Basic Metrics (Week 2)
**What changes:**
- Calculate centrality metrics
- Add metrics to nodes.csv
- Show top connectors/brokers in UI

**User sees:**
```
🔬 Network Metrics

Top Connectors (by degree):
1. Dara Parker (66 connections)
2. Nick Rossi (83 connections)

Top Brokers (by betweenness):
1. Shea Gopaul (0.234)
2. Julia Roig (0.189)
```

**New files:**
- nodes.csv (with metric columns)
- network_analysis.json

---

### Phase 3: Community Detection (Week 3)
**What changes:**
- Identify network clusters
- Label communities
- Calculate modularity

**User sees:**
```
🔬 Community Structure

Detected 5 communities:
- Cluster 1: 45 people (Social Impact)
- Cluster 2: 32 people (Finance)
- Cluster 3: 28 people (Academia)

Modularity: 0.456 (well-defined communities)
```

**Enhanced files:**
- nodes.csv (with cluster column)
- network_analysis.json (with community stats)

---

### Phase 4: Brokerage Analysis (Week 4)
**What changes:**
- Calculate brokerage roles
- Generate brokerage matrix
- Identify structural positions

**User sees:**
```
🔬 Brokerage Analysis

Key Roles:
- Coordinators: 23 people (within-group)
- Gatekeepers: 12 people (control inflow)
- Liaisons: 8 people (bridge groups)

Critical Brokers:
1. Dara Parker - Connects Social Impact ↔ Finance
2. Julia Roig - Bridges Peacebuilding ↔ Democracy
```

**New files:**
- brokerage_matrix.csv
- key_positions.csv

---

### Phase 5: Strategic Insights (Future)
**What changes:**
- AI-generated narrative insights
- Gap analysis
- Collaboration recommendations

**User sees:**
```
🔬 Strategic Insights

Hidden Brokers:
- Person X connects disparate groups (high betweenness, low visibility)

Alignment Gaps:
- Philanthropy ↔ Government sectors isolated
- Only 2 bridge connections

Collaboration Opportunities:
- Nonprofit ↔ Enterprise weak ties (3 shared contacts)
- High potential for partnership
```

---

## File Outputs by Mode

### Basic Mode Files:
```
basic_crawl.zip
├── nodes.csv (7 columns)
│   ├── id
│   ├── name
│   ├── profile_url
│   ├── headline
│   ├── location
│   ├── degree
│   └── source_type
├── edges.csv (3 columns)
│   ├── source_id
│   ├── target_id
│   └── edge_type
└── raw_profiles.json
```

### Advanced Mode Files (When Complete):
```
advanced_analysis.zip
├── nodes.csv (15+ columns)
│   ├── Basic columns (7)
│   ├── organization
│   ├── sector
│   ├── degree_centrality
│   ├── betweenness_centrality
│   ├── eigenvector_centrality
│   ├── closeness_centrality
│   ├── cluster_id
│   └── brokerage_role
├── edges.csv (same as basic)
├── raw_profiles.json (same as basic)
├── network_analysis.json (NEW)
│   ├── network_summary
│   ├── communities
│   ├── centrality_rankings
│   └── structural_metrics
├── key_positions.csv (NEW)
│   ├── role
│   ├── id
│   ├── name
│   ├── score
│   └── reason
└── brokerage_matrix.csv (NEW)
    ├── from_group
    ├── to_group
    ├── broker_id
    ├── broker_name
    └── brokerage_type
```

---

## Benefits of This Approach

### For Testing
- ✅ Basic mode always works (safe fallback)
- ✅ Advanced mode can break without affecting core
- ✅ Easy to compare outputs between modes
- ✅ Clear scope for each development phase

### For Users
- ✅ Choose complexity level upfront
- ✅ Clear expectations set early
- ✅ No confusion about features
- ✅ Can use basic mode while advanced develops

### For Development
- ✅ Ship features incrementally
- ✅ Gather feedback per feature
- ✅ No breaking changes to basic mode
- ✅ Professional development practice

---

## User Experience Flow

### First-Time User (Basic Mode)
```
1. See mode selection → Read both descriptions
2. Choose Basic (default) → Green success box
3. Upload CSV → Enter token → Run crawl
4. Get results → Download files → Import to Polinode
5. ✅ Complete workflow, clear outputs
```

### Advanced User (Advanced Mode)
```
1. See mode selection → Read both descriptions
2. Toggle Advanced → Blue info box with features
3. Upload CSV → Enter token → Run crawl
4. Get basic results (same as basic mode)
5. See "Coming Soon" section with roadmap
6. Download files → Wait for future features
7. ✅ Knows what's coming, can plan accordingly
```

### Returning User (After Features Launch)
```
1. Toggle Advanced → Already knows what it does
2. Run crawl → Get enhanced outputs
3. See network metrics, communities, brokerage
4. Download advanced files → Rich analysis ready
5. ✅ Full network intelligence platform
```

---

## Testing Instructions

### Test Basic Mode:
1. Leave toggle OFF (default)
2. Run a crawl
3. Should see: Green "Basic Mode" message
4. Should get: 3 files (nodes, edges, raw)
5. Should NOT see: Advanced analytics section

### Test Advanced Mode:
1. Turn toggle ON
2. Run a crawl
3. Should see: Blue "Advanced Mode" message
4. Should get: Same 3 files (for now)
5. Should see: "Coming Soon" section with roadmap

### Test Mode Persistence:
1. Run crawl in Advanced mode
2. Download a file
3. Should still see: Advanced analytics section
4. Mode setting should persist

---

## Future Enhancements

### Optional: Mode Comparison
Add ability to run same seeds in both modes and compare:
```
Compare Modes:
Basic:  300 nodes, 280 edges | 5 seconds
Advanced: 300 nodes, 280 edges | 12 seconds
         + Metrics, Communities, Brokerage
```

### Optional: Save Mode Preference
```python
# Remember user's preferred mode
if 'preferred_mode' not in st.session_state:
    st.session_state.preferred_mode = False

advanced_mode = st.toggle(
    "Advanced Mode",
    value=st.session_state.preferred_mode
)
```

### Optional: Mode-Specific Limits
```python
if advanced_mode:
    # Higher limits for detailed analysis
    max_edges = 2500
    max_nodes = 1000
else:
    # Quick crawls
    max_edges = 1000
    max_nodes = 500
```

---

## Summary

**Status:** ✅ Mode toggle implemented  
**Default:** Basic mode (safe and familiar)  
**Advanced features:** Placeholder (coming soon)  
**User experience:** Clear, informative, no confusion  
**Development:** Ready for incremental feature additions  

The foundation is set for building the advanced analytics platform while keeping the basic crawler stable and reliable! 🎉
