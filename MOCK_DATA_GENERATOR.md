# Mock Data Generator for Stress Testing

## Purpose

Save API credits by using realistic synthetic data to:
- Test node and edge limits (1000 nodes, 1000 edges)
- Verify progress bar behavior
- Test organization extraction
- Validate export functionality
- Develop new features without cost

---

## How It Works

### Deterministic Generation

Each profile URL generates a **consistent, unique** fake profile:

```python
# URL hash seeds the randomness
url_hash = int(hashlib.md5(profile_url.encode()).hexdigest(), 16)

# Same URL always produces same profile
first_name = first_names[url_hash % len(first_names)]
last_name = last_names[(url_hash // 100) % len(last_names)]
...
```

**Benefits:**
- Reproducible results for debugging
- No randomness between runs
- Can share specific test cases

---

### Network Structure

Each mock profile includes:
- **25-40 "people_also_viewed" connections** (varies by URL)
- Realistic names (64 first names × 56 last names = 3,584 combinations)
- Realistic titles (21 options)
- Realistic organizations (60+ options)
- Realistic locations (23 options)
- Sector classification data

---

### Expected Network Sizes

**Degree 1 (5 seeds):**
```
Seeds: 5
Connections per seed: ~30
Total nodes: ~155 (5 + 150 connections)
Total edges: ~150
```

**Degree 2 (5 seeds):**
```
Seeds: 5
Degree-1 nodes: ~150
Connections per degree-1: ~30
Potential nodes: 4,500+
Actual nodes: 1,000 (hits limit) ✅
Actual edges: 1,000+ (hits limit) ✅
```

**This is exactly what you need for stress testing!**

---

## How to Use

### 1. Enable Mock Mode

In the app UI:
```
[✓] Run in mock mode (no real API calls)
```

### 2. Upload Any Seed File

Use the provided `mock_test_seeds.csv`:
```csv
name,profile_url
Alice Chen,https://www.linkedin.com/in/alice-chen-water
Bob Martinez,https://www.linkedin.com/in/bob-martinez-impact
Carol Williams,https://www.linkedin.com/in/carol-williams-foundation
David Lee,https://www.linkedin.com/in/david-lee-strategy
Emma Johnson,https://www.linkedin.com/in/emma-johnson-consulting
```

Or use your real seed file - mock mode will generate fake data for any URL.

### 3. Run Crawl

- **Degree 1:** Quick test, ~155 nodes
- **Degree 2:** Stress test, hits 1000 node limit ✅

### 4. Verify Results

Check that:
- [x] Progress bar works smoothly
- [x] Node limit (1000) is respected
- [x] Edge limit (1000) is respected
- [x] Organization extraction works
- [x] CSV exports are valid
- [x] ZIP download works

---

## Sample Mock Data

### Profile Response:
```json
{
  "public_identifier": "alice-chen-water",
  "full_name": "Alice Chen",
  "headline": "Director at World Resources Institute",
  "occupation": "Director at World Resources Institute",
  "location_str": "Washington, DC",
  "experiences": [
    {
      "company": "World Resources Institute",
      "title": "Director",
      "starts_at": {"year": 2020, "month": 1}
    }
  ],
  "people_also_viewed": [
    {
      "link": "https://www.linkedin.com/in/james-smith-123",
      "name": "James Smith",
      "summary": "CEO at The Nature Conservancy",
      "location": "San Francisco, CA"
    },
    // ... 24-39 more connections
  ]
}
```

### Organization Extraction:
```
Organization: World Resources Institute
Sector: Nonprofit (inferred from keywords)
```

---

## Data Pools

### First Names (64)
James, Mary, John, Patricia, Robert, Jennifer, Michael, Linda, William, Elizabeth, David, Barbara, Richard, Susan, Joseph, Jessica, Thomas, Sarah, Charles, Karen, Christopher, Nancy, Daniel, Lisa, Matthew, Betty, Anthony, Margaret, Mark, Sandra, Donald, Ashley, Steven, Kimberly, Paul, Emily, Andrew, Donna, Joshua, Michelle, Kenneth, Dorothy, Kevin, Carol, Brian, Amanda, George, Melissa, Edward, Deborah, Ronald, Stephanie, Timothy, Rebecca, Jason, Sharon, Jeffrey, Laura, Ryan, Cynthia, Jacob, Kathleen, Gary, Amy

### Last Names (56)
Smith, Johnson, Williams, Brown, Jones, Garcia, Miller, Davis, Rodriguez, Martinez, Hernandez, Lopez, Gonzalez, Wilson, Anderson, Thomas, Taylor, Moore, Jackson, Martin, Lee, Perez, Thompson, White, Harris, Sanchez, Clark, Ramirez, Lewis, Robinson, Walker, Young, Allen, King, Wright, Scott, Torres, Nguyen, Hill, Flores, Green, Adams, Nelson, Baker, Hall, Rivera, Campbell, Mitchell, Carter, Roberts, Gomez, Phillips, Evans, Turner, Diaz, Parker

### Organizations (60+)
- **Nonprofits:** World Resources Institute, The Nature Conservancy, WWF, IUCN, Conservation International, Environmental Defense Fund, Sierra Club, Greenpeace, Earthjustice, Ocean Conservancy, Wildlife Conservation Society, Rainforest Alliance, Global Water Partnership, Water.org, charity: water, Pacific Institute, Alliance for Water Stewardship, CDP, Ceres, BSR
- **Think Tanks:** World Economic Forum, Aspen Institute, Brookings Institution, Carnegie Endowment, Council on Foreign Relations, RAND Corporation
- **Consulting:** McKinsey & Company, Boston Consulting Group, Bain & Company, Deloitte, Accenture, PwC, EY, KPMG
- **Finance:** Goldman Sachs, JPMorgan Chase, Bank of America, Citigroup, Morgan Stanley, BlackRock, Vanguard
- **Foundations:** Ford Foundation, Rockefeller Foundation, MacArthur Foundation, Gates Foundation, Hewlett Foundation, Packard Foundation, Bloomberg Philanthropies, Open Society Foundations, Omidyar Network, Skoll Foundation, Toniic
- **Academia:** Stanford University, Harvard University, MIT, Yale University, Columbia University, UC Berkeley, Princeton University, Oxford University

### Titles (21)
CEO, Founder, Director, VP, Manager, Consultant, Partner, Executive Director, Chief Strategy Officer, Program Director, Senior Advisor, Managing Director, Principal, Fellow, Board Member, Chief Impact Officer, Head of Partnerships, Director of Development, Senior Program Officer, Policy Director, Research Director

### Locations (23)
San Francisco, CA | New York, NY | Washington, DC | Boston, MA | Los Angeles, CA | Seattle, WA | Chicago, IL | Denver, CO | Austin, TX | Portland, OR | Miami, FL | Atlanta, GA | London, UK | Geneva, Switzerland | Amsterdam, Netherlands | Berlin, Germany | Paris, France | Singapore | Hong Kong | Tokyo, Japan | Sydney, Australia | Toronto, Canada | Vancouver, Canada

---

## Comparison: Mock vs Real

| Aspect | Mock Mode | Real API |
|--------|-----------|----------|
| Credits used | 0 | 1 per call |
| Speed | Instant (0.1s delay) | Rate limited (20/min) |
| Data quality | Synthetic but realistic | Real LinkedIn data |
| Network structure | Deterministic | Real relationships |
| Use case | Testing, development | Production crawls |

---

## Testing Checklist

### Basic Functionality:
- [ ] Degree 1 crawl completes
- [ ] Degree 2 crawl hits node limit
- [ ] Progress bar updates correctly
- [ ] Status messages display
- [ ] Results summary accurate

### Limits:
- [ ] 1000 node limit enforced
- [ ] 1000 edge limit enforced
- [ ] Crawl stops gracefully at limit

### Exports:
- [ ] nodes.csv downloads correctly
- [ ] edges.csv downloads correctly
- [ ] raw_profiles.json downloads correctly
- [ ] ZIP file contains all 3 files

### Advanced Mode:
- [ ] Organization extraction works
- [ ] Sector classification works
- [ ] Organization breakdown displays
- [ ] Sector distribution displays

---

## Files

- **app.py** - Updated with mock data generator
- **mock_test_seeds.csv** - Sample seed file for testing

---

## Summary

**Mock mode now generates:**
- Realistic, deterministic fake profiles
- 25-40 connections per profile
- Enough data to hit 1000 node/edge limits
- Organization and sector data for advanced mode testing

**Perfect for:**
- Stress testing without burning credits
- Developing new features
- Debugging edge cases
- Demo purposes

**To use:** Toggle mock mode ON, upload any seed file, run crawl! 🧪
