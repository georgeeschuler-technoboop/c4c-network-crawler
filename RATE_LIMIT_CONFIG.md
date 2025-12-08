# Rate Limit Configuration & Progress Bar

## 🔍 Key Discovery: EnrichLayer Rate Limits

From EnrichLayer documentation:

| Plan | Rate Limit |
|------|------------|
| Trial/PAYG | 2 requests/min |
| $49/mo | **20 requests/min** |
| $299/mo | 50 requests/min |
| $899/mo | 100 requests/min |
| $1899/mo | 300 requests/min |

**Your issue:** Running at 180+ req/min against a 2 req/min limit = 96% failures!

---

## ✅ Changes Made

### 1. API Delay Set for $49/mo Plan

```python
API_DELAY = 3.0  # 20 requests/min = 1 every 3 seconds
```

**Rate calculation:**
- 60 seconds / 3 seconds per call = 20 calls/minute ✅
- Matches $49/mo plan limit exactly

---

### 2. Rate Limit Info Box

Added clear information in the UI:

```
⏱️ Rate Limit: 20 requests/minute (EnrichLayer $49/mo plan)

Estimated time for 5 seeds:
- Degree 1: ~0 min 15 sec (5 API calls)
- Degree 2: ~12-25 min (250-500 API calls, varies by network size)

Each API call takes ~3 seconds to respect rate limits.
```

---

### 3. Progress Bar with Time Estimate

**During crawl, you'll see:**

```
[████████████░░░░░░░░] 60%
Processing... 30/50 profiles (60%)

⏱️ ~2 min 30 sec remaining | Elapsed: 1m 30s | Rate: 20 req/min
```

**Features:**
- Real-time progress percentage
- Profiles processed count
- Estimated time remaining
- Elapsed time
- Current rate display

---

### 4. Updated Time Estimates

**Degree 1 (accurate):**
```
5 seeds × 3 sec/call = 15 seconds
10 seeds × 3 sec/call = 30 seconds
```

**Degree 2 (estimated range):**
```
5 seeds × ~50-100 connections each × 3 sec/call = 12-25 minutes
10 seeds × ~50-100 connections each × 3 sec/call = 25-50 minutes
```

---

## 📊 Rate Limit Comparison

| Old Setting | New Setting |
|-------------|-------------|
| ~300ms delay | 3000ms delay |
| ~180 req/min | 20 req/min |
| 96% failure rate | ~0% failure rate (expected) |

---

## 🎯 Future: Plan Selector

For future flexibility, could add a plan selector:

```python
plan_options = {
    "Trial/PAYG (2/min)": 30.0,      # 1 every 30 seconds
    "$49/mo (20/min)": 3.0,           # 1 every 3 seconds
    "$299/mo (50/min)": 1.2,          # 1 every 1.2 seconds
    "$899/mo (100/min)": 0.6,         # 1 every 0.6 seconds
    "$1899/mo (300/min)": 0.2         # 1 every 0.2 seconds
}
```

**For now:** Hardcoded to $49/mo plan as George specified.

---

## 📋 UI Flow

### Before Crawl:
```
⚙️ Crawl Configuration

[Degree 1] [Degree 2]

✅ Degree 1 Selected - Good Choice!
- Direct connections only
- ~1-2 minutes for 5 seeds
- Uses ~5-10 credits
- Reliable, low rate limit risk

Crawl Limits:
Max Edges: 1000
Max Nodes: 500

⏱️ Rate Limit: 20 requests/minute (EnrichLayer $49/mo plan)
Estimated time for 5 seeds:
- Degree 1: ~0 min 15 sec
- Degree 2: ~12-25 min

[🚀 Run Crawl]
```

### During Crawl:
```
🔄 Crawl Progress

[████████████░░░░░░░░] 60%
Processing... 3/5 profiles (60%)

⏱️ ~6 sec remaining | Elapsed: 0m 9s | Rate: 20 req/min

🔍 Processing: Julia Roig (degree 0) | ✅ 3/3 (100%)
```

### After Crawl:
```
📊 Results Summary

Total Nodes: 150
Total Edges: 145
...

🔧 Technical Details (C4C Internal) [expandable]
```

---

## 🧪 Expected Results

### Degree 1 (5 seeds):
```
API calls: 5
Time: ~15 seconds
Success rate: 95-100%
Rate limit errors: 0
```

### Degree 2 (5 seeds):
```
API calls: ~250-500
Time: ~12-25 minutes
Success rate: 90-95%
Rate limit errors: Minimal (if any)
```

---

## 💡 Why This Works

**The math:**
- Your limit: 20 requests/minute
- Our rate: 20 requests/minute (exactly)
- Buffer: Retry logic handles occasional 429s
- Result: Should be sustainable!

**Compared to before:**
- Old rate: 180+ requests/minute
- Old limit: 2 requests/minute (trial)
- Result: 96% failure rate

---

## 📁 Files Updated

- **app.py**
  - `API_DELAY = 3.0` (20 req/min)
  - Rate limit info box added
  - Progress bar with time estimate
  - Real-time progress display

---

## 🚀 Test Instructions

1. **Deploy updated app.py**
2. **Run Degree 1 with 5 seeds**
3. **Watch the progress bar** - should complete in ~15 seconds
4. **Verify 95-100% success rate**
5. **Check Technical Details** for confirmation

---

## Future Enhancements

### Plan Selector (Optional)
```python
st.selectbox("EnrichLayer Plan", [
    "$49/mo (20 req/min)",
    "$299/mo (50 req/min)",
    ...
])
```

### Adaptive Rate Limiting
```python
# If hitting 429s, automatically slow down
if rate_limit_hit:
    current_delay *= 1.5
```

### Credit Tracking
```python
# Show estimated credit usage
st.info(f"This crawl will use ~{estimated_calls} credits")
```

---

## Summary

| Feature | Status |
|---------|--------|
| 20 req/min rate limit | ✅ Implemented |
| Rate limit info display | ✅ Implemented |
| Progress bar | ✅ Implemented |
| Time estimate | ✅ Implemented |
| Real-time progress | ✅ Implemented |
| Plan selector | 🔮 Future |

**Ready to test!** 🎉
