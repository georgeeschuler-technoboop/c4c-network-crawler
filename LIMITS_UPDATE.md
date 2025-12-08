# Crawl Limits Updated - Moderate Scenario

## What Changed

Updated from **prototype limits** to **moderate production limits**:

| Limit | Before (Prototype) | After (Moderate) | Change |
|-------|-------------------|------------------|--------|
| **Max Edges** | 100 | 1000 | 10× increase |
| **Max Nodes** | 150 | 500 | 3.3× increase |

---

## Why the Change

### What Happened
Your real crawl with 5 seeds hit the edge limit after only 3 profiles:
- Dara Parker: 66 connections
- Julia Roig: 19 connections  
- Nick Rossi: 83 connections
- **Total: 168 potential edges, stopped at 100**
- James Dalton & Emma Iezzoni: Not even processed!

### The Problem
100 edges was way too small for your highly-connected seed profiles in the social impact sector.

---

## New Limits Explained

### 1000 Edges
**What it covers:**
- All direct connections from 5-10 seeds
- Average of 50-80 connections per seed
- Room for some degree-2 exploration

**Expected behavior:**
- Degree 1: Will capture ALL connections from your seeds ✅
- Degree 2: Will capture many (but not all) secondary connections
- May still hit limit on large degree-2 crawls

**Crawl time:**
- ~8-15 minutes (depending on connections found)
- 1 second delay between API calls
- Includes processing time

### 500 Nodes
**What it covers:**
- 5 seed profiles
- ~200-300 degree-1 connections
- ~200 degree-2 connections

**Network size:**
- Perfect for Polinode visualization
- Not too big, not too small
- Good balance of detail and performance

---

## Expected Results Now

### For Your 5 Seeds (Degree 1):
```
Estimated output:
- Nodes: ~300 (5 seeds + ~60 connections each)
- Edges: ~300 (all degree-1 connections)
- Crawl time: ~5 minutes
- API calls: ~300
```

### For Your 5 Seeds (Degree 2):
```
Estimated output:
- Nodes: 500 (will hit node limit)
- Edges: 1000 (may hit edge limit)
- Crawl time: ~8-10 minutes
- API calls: ~500
```

---

## What You'll See

### UI Changes
**Before:**
```
Prototype Limits:
Max Edges: 100
Max Nodes: 150
```

**After:**
```
Crawl Limits:
Max Edges: 1000
Max Nodes: 500
```

### Crawl Messages
You might see:
- ✅ "✅ Crawl Complete!" (if finished naturally)
- ⚠️ "Reached edge limit (1000)" (if network is huge)
- ⚠️ "Reached node limit (500)" (if network is huge)

---

## Files Updated

**app.py** - Three changes:
1. UI display: Shows 1000/500 instead of 100/150
2. Crawler call: Passes 1000/500 to run_crawler()
3. CSV metadata: Records 1000/500 in file headers

---

## Next Steps

1. **Update app.py** on Streamlit Cloud with the new version
2. **Re-run your crawl** with the same 5 seeds
3. **Results you should get:**
   - All 5 seeds processed ✅
   - ~300-500 nodes
   - ~300-1000 edges
   - Complete network!

---

## CSV Metadata

Your downloaded files will now show:
```csv
# generated_at=2025-12-08T...; max_degree=2; max_edges=1000; max_nodes=500
```

This helps track what limits were used for each crawl.

---

## API Usage

**Cost estimate:**
- ~300-500 API calls per crawl
- At 1 second/call = 5-8 minutes
- Check your EnrichLayer plan if running many crawls

---

## Future Scaling

If you need more later:

**For bigger networks (10+ seeds):**
```python
max_edges = 2500
max_nodes = 1000
```

**For comprehensive mapping:**
```python
max_edges = 5000
max_nodes = 2000
```

**For unlimited (careful!):**
```python
max_edges = 50000
max_nodes = 10000
```

But start with 1000/500 - it's a good sweet spot! ✅

---

## Performance

### Polinode Import
- 500 nodes + 1000 edges = **Fast and smooth** ✅
- Creates rich, detailed network
- Not overwhelming to analyze

### Crawl Speed
- Moderate limits = Reasonable wait time
- Not too fast (rate limits)
- Not too slow (frustrating)

### Network Quality
- Captures meaningful connections
- Shows community structure
- Good for analysis and visualization

---

## Summary

**Old:** 100 edges, 150 nodes (too small - hit limit after 3 profiles)  
**New:** 1000 edges, 500 nodes (perfect for 5-10 seeds)  
**Result:** Complete networks, all seeds processed ✅

Ready to test! 🚀
