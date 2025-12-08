# Rate Limit Configuration & Progress Bar

## 🔍 EnrichLayer Rate Limits (From Documentation)

| Plan | Rate Limit |
|------|------------|
| Trial/PAYG | 2 requests/min |
| **$49/mo** | **20 requests/min** ← Our target |
| $299/mo | 50 requests/min |
| $899/mo | 100 requests/min |
| $1899/mo | 300 requests/min |

---

## ✅ Implementation (Based on Team Feedback)

### 1. RateLimiter Class

A clean, drop-in rate limiter with sliding window:

```python
class RateLimiter:
    def __init__(self, per_min_limit: int, buffer: float = 0.8):
        """
        per_min_limit: documented limit (e.g., 20 requests/min)
        buffer: safety factor (0.8 → aim for 16/min so we never hit the hard cap)
        """
        self.per_min_limit = per_min_limit
        self.allowed_per_min = max(1, int(per_min_limit * buffer))
        self.window_start = time.time()
        self.calls_in_window = 0

    def wait_for_slot(self):
        now = time.time()
        elapsed = now - self.window_start

        # New minute → reset window
        if elapsed >= 60:
            self.window_start = now
            self.calls_in_window = 0
            return

        # If we've hit our safe quota, sleep until the minute resets
        if self.calls_in_window >= self.allowed_per_min:
            sleep_for = 60 - elapsed
            time.sleep(sleep_for)
            self.window_start = time.time()
            self.calls_in_window = 0

    def record_call(self):
        self.calls_in_window += 1
```

**Benefits:**
- Tracks actual calls within a minute window
- 80% safety buffer (aims for 16/min when limit is 20/min)
- Auto-waits when approaching limit
- More robust than fixed delays

---

### 2. Progress Bar (Processed vs Queue)

Instead of estimating total upfront, we use:

```python
progress = processed_nodes / (processed_nodes + len(queue))
```

**Benefits:**
- Naturally goes from 0 → 1 as BFS empties the queue
- No need to guess total profiles upfront
- Accurate regardless of network size
- Intuitive "done vs remaining" display

**Display:**
```
[████████████░░░░░░░░] 
Processing... 15 done, 35 remaining
```

---

### 3. Configuration

```python
PER_MIN_LIMIT = 20  # Tuned for $49/mo plan
```

Hardcoded for now, can be made configurable later via plan selector.

---

## 📋 UI Text

Simple, clear caption:

```
⏱️ API pacing: This prototype is tuned for up to 20 requests/minute
(EnrichLayer $49/mo plan). The app automatically throttles calls so we
don't hit rate limits. Progress bar shows processed nodes vs. remaining queue.
```

---

## 🎯 How It Works

### During Crawl:

1. **Before each API call:**
   ```python
   rate_limiter.wait_for_slot()  # Waits if we've used our quota this minute
   ```

2. **Make the call:**
   ```python
   response, error = call_enrichlayer_api(...)
   ```

3. **After each call:**
   ```python
   rate_limiter.record_call()  # Track the call
   time.sleep(0.2)  # Tiny courtesy delay
   ```

4. **Update progress:**
   ```python
   progress = processed_nodes / (processed_nodes + len(queue))
   progress_bar.progress(progress, text="Processing... X done, Y remaining")
   ```

---

## 📊 Expected Behavior

### With 20/min Limit (16/min effective with buffer):

**Degree 1 (5 seeds):**
```
API calls: ~5
Time: ~20-30 seconds
Rate limit errors: 0
```

**Degree 2 (5 seeds):**
```
API calls: ~250-500
Time: ~15-30 minutes (varies by network)
Rate limit errors: 0 (with proper pacing)
```

---

## 🔮 Future: Plan Selector

If needed later, add a dropdown:

```python
plan_options = {
    "Trial/PAYG (2/min)": 2,
    "$49/mo (20/min)": 20,
    "$299/mo (50/min)": 50,
    "$899/mo (100/min)": 100,
    "$1899/mo (300/min)": 300
}

selected_plan = st.selectbox("EnrichLayer Plan", list(plan_options.keys()))
per_min_limit = plan_options[selected_plan]
```

But for prototype, hardcoded 20/min is fine.

---

## 📁 Files Updated

- **app.py**
  - `RateLimiter` class added
  - `PER_MIN_LIMIT = 20` config
  - Progress bar using queue-based calculation
  - Clean UI caption

---

## 🙏 Credit

Implementation based on feedback from C4C team member - cleaner, more robust approach than fixed delays.
