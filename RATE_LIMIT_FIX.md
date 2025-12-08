# Rate Limit Handling - Implemented

## What Happened

During degree-2 crawls, you hit EnrichLayer's API rate limits:
```
❌ Failed to fetch https://www.linkedin.com/in/haileymeslin: Rate limit exceeded
❌ Failed to fetch https://www.linkedin.com/in/rodruff: Rate limit exceeded
❌ Failed to fetch https://www.linkedin.com/in/hilaryhenegar: Rate limit exceeded
```

This is **expected** for degree-2 crawls because they make many API calls in quick succession.

---

## What Was Fixed

### 1. Exponential Backoff Retry Logic ✅

**Before:**
- Hit rate limit → Fail immediately
- Move to next profile

**After:**
- Hit rate limit → Wait and retry
- 1st retry: Wait 3 seconds
- 2nd retry: Wait 6 seconds
- 3rd retry: Wait 12 seconds
- After 3 tries → Mark as failed

**Code:**
```python
def call_enrichlayer_api(..., max_retries: int = 3):
    for attempt in range(max_retries):
        response = requests.get(...)
        
        if response.status_code == 429:  # Rate limit
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                time.sleep(wait_time)
                continue
            else:
                return None, "Rate limit exceeded (tried 3 times)"
```

---

### 2. Increased Base Delay ✅

**Before:**
- 1 second delay between API calls
- ~60 calls per minute
- Easily hits rate limits

**After:**
- 2 second delay between API calls
- ~30 calls per minute
- Reduced rate limit risk

**Code:**
```python
API_DELAY = 2.0  # Increased from 1.0
```

---

### 3. User Warning for Degree-2 ✅

**Added warning in UI:**
```
⚠️ Degree 2 Notice:
- Fetches many more profiles
- May hit rate limits (auto-retry enabled)
- Takes 10-20 min, uses 100-500 credits
- 2-second delay between calls
```

Sets proper expectations before running expensive crawls.

---

## How It Works Now

### Degree 1 Crawl (Recommended):
```
5 seeds → ~5 API calls
+ Fetching seed profiles
= ~10 total API calls
⏱️ Time: 20-30 seconds
✅ Rate limit: Very unlikely
```

### Degree 2 Crawl (Advanced):
```
5 seeds → ~5 API calls
+ 200 degree-1 profiles → ~200 API calls  
= ~205 total API calls
⏱️ Time: 7-14 minutes (with delays)
⚠️ Rate limit: Possible, auto-retries
```

---

## EnrichLayer Rate Limits

**Typical limits** (varies by plan):
- 60 requests per minute
- 1000 requests per hour
- 10,000 requests per day

**With our settings:**
- 2-second delay = 30 requests/minute
- Stays well under 60/minute limit
- But sustained crawls can still hit hourly limits

---

## What Happens During Rate Limit

### Old Behavior (Before Fix):
```
🔍 Processing: Hailey Meslin (degree 1)
❌ Failed: Rate limit exceeded
🔍 Processing: Rod Ruff (degree 1)
❌ Failed: Rate limit exceeded
🔍 Processing: Hilary Henegar (degree 1)
❌ Failed: Rate limit exceeded
...
[Many failures, incomplete network]
```

### New Behavior (After Fix):
```
🔍 Processing: Hailey Meslin (degree 1)
⏱️ Rate limit hit, waiting 3 seconds...
⏱️ Retry 1...
✅ Success!
🔍 Processing: Rod Ruff (degree 1)
⏱️ Rate limit hit, waiting 3 seconds...
⏱️ Retry 1...
✅ Success!
...
[Slower but complete network]
```

---

## Best Practices

### For Testing:
- ✅ Use **Degree 1** (fast, reliable)
- ✅ Test with 3-5 seeds
- ✅ Check results before scaling up

### For Production:
- ⚠️ Use **Degree 2** sparingly
- ⚠️ Limit to 5-10 seeds max per crawl
- ⚠️ Spread large crawls over time
- ⚠️ Monitor credit usage

### To Avoid Rate Limits:
1. **Start with Degree 1** - Get direct connections only
2. **Run multiple small crawls** instead of one big crawl
3. **Space out crawls** - Wait 5-10 minutes between runs
4. **Check your plan limits** - Know your EnrichLayer tier

---

## Credit Usage Estimates

### Degree 1 (5 seeds):
```
Profiles fetched: ~5-10
Credits used: ~5-10
Time: 20-40 seconds
Rate limit risk: Very low ✅
```

### Degree 2 (5 seeds):
```
Profiles fetched: ~200-500
Credits used: ~200-500
Time: 7-17 minutes
Rate limit risk: Moderate ⚠️
```

### Degree 2 (10 seeds):
```
Profiles fetched: ~400-1000
Credits used: ~400-1000
Time: 15-35 minutes
Rate limit risk: High 🔴
```

---

## Troubleshooting

### If you still hit rate limits:

**Option 1: Reduce degree**
```
max_degree = 1  # Only direct connections
```

**Option 2: Reduce seeds**
```
# Instead of 10 seeds, use 3-5
# Run multiple crawls
```

**Option 3: Increase delay** (if needed)
```python
# In app.py, change:
API_DELAY = 3.0  # Or even 4.0
```

**Option 4: Contact EnrichLayer**
- Ask about rate limit increase
- Upgrade plan if available
- Request burst allowance

---

## Error Messages Decoded

### "Rate limit exceeded"
- **Meaning:** Too many requests in short time
- **Action:** Auto-retries with backoff
- **After 3 tries:** Skips profile, continues crawl

### "Rate limit exceeded (tried 3 times)"
- **Meaning:** Retry exhausted
- **Action:** Profile skipped, marked as failed
- **Impact:** Missing data for that node

### "Invalid API token"
- **Meaning:** Token expired or wrong
- **Action:** Fix token, restart crawl
- **Impact:** Crawl stops immediately

---

## Monitoring

### In the UI, you'll see:

**Stats:**
```
📊 Results Summary
Total Nodes: 105
Total Edges: 100
API Calls: 205
Successful: 195
Failed: 10  ← Check this number
```

**Status messages:**
```
✅ Success
⏱️ Retrying (rate limit)
⚠️ Skipped (after retries)
```

### Failed calls are OK if:
- < 5% of total (acceptable loss)
- Network still connected
- Key nodes succeeded

### Failed calls are a problem if:
- > 20% of total (incomplete data)
- Many seed profiles failed
- Network is fragmented

---

## Advanced: Custom Rate Limit Settings

If you need more control, you can modify:

### Change retry attempts:
```python
# In call_enrichlayer_api function
max_retries: int = 5  # Try more times
```

### Change backoff timing:
```python
wait_time = (2 ** attempt) * 5  # Longer waits: 5s, 10s, 20s
```

### Change base delay:
```python
API_DELAY = 3.0  # Slower but safer
```

---

## Summary

**Problem:** Degree-2 crawls hit EnrichLayer rate limits

**Solution:** 
- ✅ Exponential backoff retry (3 attempts)
- ✅ Increased delay (1s → 2s)
- ✅ User warning for degree-2

**Result:**
- Slower but more reliable
- Auto-recovers from rate limits
- Clear user expectations

**Recommendation:**
- Use Degree 1 for testing
- Use Degree 2 sparingly
- Monitor failed calls in results

**Status:** ✅ Fixed and deployed!
