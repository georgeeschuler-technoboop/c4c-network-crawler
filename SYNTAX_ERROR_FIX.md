# Syntax Error Fix - Indentation Error

## The Error

When deploying the updated app.py, you got:
```
File "/mount/src/c4c-network-crawler/app.py", line 251
    return None, "Request timeout"
    ^
IndentationError: unexpected indent
```

---

## Root Cause

When I added the retry logic to `call_enrichlayer_api()`, I didn't fully remove the old exception handling code. Lines 251-253 were duplicate leftovers:

```python
# Line 250: New code (correct)
return None, "Failed after maximum retries"

# Lines 251-253: Old code (leftover - WRONG!)
    return None, "Request timeout"  # ← Wrong indentation!
except requests.exceptions.RequestException as e:
    return None, f"Network error: {str(e)}"
```

The old code had improper indentation and was unreachable dead code.

---

## The Fix

### Removed duplicate lines:
```python
# BEFORE (broken):
return None, "Failed after maximum retries"
    return None, "Request timeout"  # ← Duplicate!
except requests.exceptions.RequestException as e:
    return None, f"Network error: {str(e)}"


def get_mock_response(...):

# AFTER (fixed):
return None, "Failed after maximum retries"


def get_mock_response(...):
```

### Also fixed f-string bug:
```python
# BEFORE:
return None, "Request timed out (tried {max_retries} times)"  # ← Not an f-string!

# AFTER:
return None, f"Request timed out (tried {max_retries} times)"  # ← Fixed!
```

---

## Why This Happened

When doing complex code replacements with str_replace, it's easy to:
1. Not fully match the old code
2. Leave fragments behind
3. Create indentation issues

**Lesson:** Always view the surrounding context after large changes to verify clean replacement.

---

## How to Prevent

### Better approach for complex changes:
```python
# Instead of:
str_replace(old_entire_function, new_entire_function)

# Do:
1. View the function first
2. Make targeted replacements
3. View again to verify
4. Check for duplicates
```

---

## Testing

After this fix:
1. ✅ Python syntax is valid
2. ✅ Function properly closed
3. ✅ No duplicate code
4. ✅ F-strings work correctly
5. ✅ App should deploy cleanly

---

## Status

✅ **Fixed and ready to deploy**

The indentation error is resolved. You can now update your Streamlit Cloud app with the corrected app.py.

---

## Summary

**Problem:** Duplicate leftover code with wrong indentation  
**Cause:** Incomplete code replacement  
**Fix:** Removed lines 251-253 (duplicates)  
**Also fixed:** Missing f-string prefix on line 243  
**Status:** ✅ Ready to deploy
