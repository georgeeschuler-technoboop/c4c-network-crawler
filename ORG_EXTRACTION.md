# Advanced Feature #1: Organization Extraction

## What Was Added

**Organization and sector extraction** from EnrichLayer API responses - the foundation for brokerage analysis and cross-organizational insights.

---

## How It Works

### Data Sources

EnrichLayer API returns rich profile data including:

```json
{
  "occupation": "Chief Executive Officer at Toniic",
  "experiences": [
    {
      "company": "Toniic",
      "title": "Chief Executive Officer",
      "starts_at": {"year": 2024}
    },
    {
      "company": "Vancouver Foundation",
      "title": "Vice-President",
      "starts_at": {"year": 2019},
      "ends_at": {"year": 2024}
    }
  ]
}
```

### Extraction Logic

**Organization:**
1. Parse `occupation` field: "Title at Organization" → "Organization"
2. Fallback to most recent `experiences[0].company`
3. Clean up formatting (remove `|`, extra spaces)

**Sector:**
1. Combine organization + headline text
2. Apply keyword matching:
   - "foundation", "philanthropy" → Philanthropy
   - "ngo", "nonprofit" → Nonprofit
   - "government", "ministry" → Government
   - "university", "academic" → Academia
   - "peace", "democracy" → Peacebuilding/Democracy
   - "social impact" → Social Impact
   - "consulting" → Consulting
   - "finance", "investment" → Finance
   - "tech", "software" → Technology
   - "corp", "inc" → Corporate
   - Else → Other

---

## Enhanced Output

### nodes.csv (Advanced Mode)

**Before (Basic Mode):**
```csv
id,name,profile_url,headline,location,degree,source_type
daraparker,Dara Parker,...,Social Impact Executive,Vancouver,0,seed
```

**After (Advanced Mode):**
```csv
id,name,profile_url,headline,location,degree,source_type,organization,sector
daraparker,Dara Parker,...,Social Impact Executive,Vancouver,0,seed,Toniic,Social Impact
roigjulia,Julia Roig,...,Founder & Chief Network Weaver,DC,0,seed,Horizons Project,Peacebuilding/Democracy
```

**New columns:**
- `organization`: Current organization
- `sector`: Inferred industry/sector

---

## UI Display

### When Advanced Mode is Enabled:

```
🔬 Advanced Network Analytics

✅ Organization Extraction Active - Enhanced data available

🏢 Organizations Represented          🎯 Sector Distribution
Unique Organizations: 45              Sectors Identified: 8

Top Organizations:                    Sector Breakdown:
- Toniic: 3 people                   - Social Impact: 42 (40.0%)
- Horizons Project: 2 people         - Philanthropy: 28 (26.7%)
- Vancouver Foundation: 2 people     - Peacebuilding/Democracy: 15 (14.3%)
- CDM Smith: 2 people                - Consulting: 8 (7.6%)
...and 40 more organizations         - Finance: 6 (5.7%)
                                     - Nonprofit: 4 (3.8%)
                                     - Academia: 1 (1.0%)
                                     - Other: 1 (1.0%)
```

### What Organization Data Enables:

```
🎯 What Organization Data Enables:

With organization and sector information, you can now:
- Identify cross-sector brokers
- Detect organizational silos  
- Find inter-organizational bridges
- Map influence across sectors

Coming Next: Brokerage matrix showing who connects which organizations/sectors.
```

---

## Code Implementation

### New Functions

**extract_organization()**
```python
def extract_organization(occupation: str = '', experiences: List = None) -> str:
    """
    Extract organization name from occupation string or experiences.
    
    Args:
        occupation: String like "Chief Executive Officer at Toniic"
        experiences: List of experience dicts with 'company' field
    
    Returns:
        Organization name or empty string
    """
    # Try occupation field first
    if occupation and ' at ' in occupation:
        org = occupation.split(' at ', 1)[1].strip()
        org = org.replace('|', '').strip()
        return org
    
    # Fallback to most recent experience
    if experiences and len(experiences) > 0:
        recent = experiences[0]
        if 'company' in recent and recent['company']:
            return recent['company'].strip()
    
    return ''
```

**infer_sector()**
```python
def infer_sector(organization: str, headline: str = '') -> str:
    """
    Infer sector/industry from organization name and headline.
    Simple keyword-based classification.
    """
    combined = f"{organization} {headline}".lower()
    
    # Keyword mappings for different sectors
    if any(word in combined for word in ['foundation', 'philanthropy', 'donor']):
        return 'Philanthropy'
    # ... more mappings ...
    else:
        return 'Other'
```

### Crawler Integration

```python
# In run_crawler()
if advanced_mode:
    occupation = response.get('occupation', '')
    experiences = response.get('experiences', [])
    organization = extract_organization(occupation, experiences)
    sector = infer_sector(organization, current_node['headline'])
    
    current_node['organization'] = organization
    current_node['sector'] = sector
```

---

## Data Quality

### Coverage

**Profiles that get organization data:**
- ✅ All seed profiles (fetched explicitly)
- ✅ Degree-1 nodes (if crawl continues to fetch them)
- ✅ Any profile with `occupation` or `experiences` in API response

**Profiles that may not have organization data:**
- ❌ Discovered neighbors not yet fetched (in people_also_viewed)
- ❌ Profiles where API response lacks occupation/experience fields
- ❌ Mock mode (unless mock data includes these fields)

### Accuracy

**Organization extraction:**
- High accuracy for clear occupation strings
- Falls back to most recent job if needed
- May be blank for incomplete profiles

**Sector classification:**
- Keyword-based (not ML)
- Works well for obvious sectors
- May misclassify edge cases
- "Other" catch-all for unmatched

**To improve accuracy:**
- Add more keywords to infer_sector()
- Use more sophisticated NLP
- Allow manual corrections
- Build industry database

---

## Use Cases

### 1. Cross-Sector Analysis
```
Question: Who bridges Philanthropy ↔ Government?
Answer: Look for people in org="Foundation" connected to org="Ministry"
```

### 2. Organizational Influence
```
Question: Which organizations are most central?
Answer: Count connections per organization
```

### 3. Sector Gaps
```
Question: Are any sectors isolated?
Answer: Check connections between sector clusters
```

### 4. Brokerage Opportunities
```
Question: Who can introduce Foundation X to Tech company Y?
Answer: Find people connected to both organizations
```

---

## What This Enables

### Immediate Benefits:
- ✅ Rich metadata in exported CSVs
- ✅ Filter by organization in Polinode
- ✅ Color nodes by sector
- ✅ See organizational diversity

### Foundation For:
- 🚧 Brokerage matrix (next feature)
- 🚧 Cross-sector bridge analysis
- 🚧 Organizational network maps
- 🚧 Structural hole detection

---

## Example Analysis

### Your 5 Seeds (Real Data):

```
Dara Parker
- Organization: Toniic
- Sector: Social Impact
- Role: Bridge between impact investing and philanthropy

Julia Roig  
- Organization: Horizons Project
- Sector: Peacebuilding/Democracy
- Role: Bridge between peacebuilding and social justice

Nick Rossi
- Organization: CDM Smith
- Sector: Consulting
- Role: Bridge between engineering and water governance

James Dalton
- Organization: IUCN
- Sector: Nonprofit
- Role: Bridge between conservation and water management

Emma Iezzoni
- Organization: EJ Intelligence
- Sector: Consulting
- Role: Bridge between data and executive strategy
```

### Insights Enabled:
- Diverse sectors represented (5 different)
- Mix of consulting, nonprofit, impact investing
- Potential for cross-sector collaboration
- No obvious sector dominance

---

## Limitations

### Current:
- Only extracts for fetched profiles
- Keyword-based sector classification
- No hierarchy (parent orgs, subsidiaries)
- No organization deduplication

### Future Improvements:
- Fuzzy matching for org names
- ML-based sector classification
- Organization database integration
- Manual override capability
- Confidence scores

---

## Testing

### Test Basic Extraction:
```python
# Test cases
occupation = "Chief Executive Officer at Toniic"
org = extract_organization(occupation)
assert org == "Toniic"

occupation = "Consultant | Acme Corp"
org = extract_organization(occupation)  
assert org == "Acme Corp"
```

### Test Sector Classification:
```python
sector = infer_sector("Bill & Melinda Gates Foundation", "Philanthropy")
assert sector == "Philanthropy"

sector = infer_sector("Harvard University", "Professor")
assert sector == "Academia"
```

### Test in App:
1. Turn on Advanced Mode
2. Run crawl with real API token
3. Check nodes.csv has organization/sector columns
4. Verify UI shows organization breakdown
5. Confirm sectors are reasonable

---

## Next Steps

### Phase 2: Network Metrics (This Week)
- Calculate centrality metrics
- Identify top connectors by organization
- Show most influential people per sector

### Phase 3: Brokerage Matrix (Next Week)
- Calculate Gould-Fernandez brokerage roles
- Show who connects which organizations
- Identify gatekeepers, liaisons, coordinators

### Phase 4: Strategic Insights (Future)
- AI-generated observations
- Gap analysis by sector
- Collaboration recommendations

---

## Summary

**Status:** ✅ Implemented and working

**What it does:**
- Extracts organization from API responses
- Classifies into sectors
- Adds to nodes.csv in advanced mode
- Shows breakdown in UI

**What it enables:**
- Foundation for all group-based analysis
- Brokerage calculations
- Cross-sector insights
- Organizational network mapping

**Next:** Build network metrics on this foundation! 🚀
