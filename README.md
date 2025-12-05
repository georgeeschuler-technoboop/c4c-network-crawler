
# C4C Network Seed Crawler

A Streamlit prototype that converts LinkedIn seed profiles into Polinode-ready network data using EnrichLayer's API.

## Features

- ✅ Upload 1-5 seed LinkedIn profiles
- ✅ Bounded BFS crawl (1 or 2 degrees)
- ✅ Automatic rate limiting (1s delay between calls)
- ✅ Real-time crawl status updates
- ✅ Export to nodes.csv, edges.csv, raw_profiles.json
- ✅ Prototype limits: 100 edges, 150 nodes

## Quick Start

### Prerequisites

- Python 3.10 or higher
- EnrichLayer API token ([get one here](https://enrichlayer.com))

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

### Testing with Mock Data

To test without API calls, edit `app.py` and set:
```python
MOCK_MODE = True
```

Then run normally. The app will use simulated API responses.

## Usage

1. **Upload CSV**: Prepare a CSV with columns `name` and `profile_url` (max 5 rows)
2. **Enter API Token**: Your EnrichLayer API token (not stored)
3. **Configure**: Choose 1 or 2 degree crawl
4. **Run**: Click "Run Crawl"
5. **Download**: Get your nodes.csv, edges.csv, and raw_profiles.json

## File Formats

### nodes.csv
| Column | Description |
|--------|-------------|
| id | Canonical LinkedIn identifier |
| name | Full name |
| profile_url | LinkedIn URL |
| headline | Professional headline |
| location | Geographic location |
| degree | Network distance from seeds (0-2) |
| source_type | "seed" or "discovered" |

### edges.csv
| Column | Description |
|--------|-------------|
| source_id | Source profile ID |
| target_id | Target profile ID |
| edge_type | Always "people_also_viewed" |

### raw_profiles.json
Array of raw EnrichLayer API responses for all successfully fetched profiles.

## Deployment

### Streamlit Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from repository
4. Add EnrichLayer API token to app secrets

### Local Hosting
```bash
streamlit run app.py --server.port 8501
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Rate Limiting

The app includes a 1-second delay between API calls to respect EnrichLayer's limits. For 30-40 profiles, expect:
- 30-40 seconds crawl time
- ~30-40 API calls
- Safe from rate limiting

## Troubleshooting

**"Invalid API token"**
- Check your EnrichLayer dashboard for the correct token
- Ensure no extra spaces when pasting

**"Rate limit exceeded"**
- Wait a few minutes and try again
- Reduce number of seed profiles

**"No neighbors found"**
- Some profiles have limited "people also viewed" data
- Try different seed profiles

## Support

For questions or issues:
- Email: [your-email]
- GitHub Issues: [repo-link]

## License

[Your License]
## Pilot-Ready Features

### Mock Mode Toggle
Test without API calls using the mock mode toggle in the UI:
- Toggle "Run in mock mode" checkbox
- Or set environment variable: `export C4C_MOCK_MODE=true`
- Uses realistic mock data from `mock_personal_profile_response.json`

### Graph Validation
The app automatically validates that all edges reference existing nodes:
- Orphan edges are detected and excluded
- Warning message shows if any inconsistencies are found
- Ensures Polinode compatibility

### Enhanced Logging
Clear feedback during crawl:
- "No neighbors" alerts for profiles without connections
- Special message when crawl produces empty results
- Real-time status updates with emoji indicators

### CSV Metadata
Downloaded CSV files include generation metadata as comments:
```csv
# generated_at=2024-12-05T10:30:00Z; max_degree=2; max_edges=100; max_nodes=150
id,name,profile_url,...
```

### Streamlit Cloud Deployment
For shared use between George and Sarah:

1. Add API token to Streamlit secrets:
   - Go to App Settings → Secrets
   - Add: `ENRICHLAYER_TOKEN = "your-token-here"`
   - Token auto-fills for team members

2. Set default mock mode via secrets (optional):
```toml
   ENRICHLAYER_TOKEN = "your-token"
   C4C_MOCK_MODE = "false"
```