import streamlit as st
import pandas as pd
import requests
import json
import time
from io import StringIO, BytesIO
from collections import deque
from typing import Dict, List, Tuple, Optional
import re
import os
import pathlib
from datetime import datetime
import socket
import zipfile

# ============================================================================
# CONFIGURATION
# ============================================================================

API_DELAY = 1.0  # Seconds between API calls
DEFAULT_MOCK_MODE = os.getenv("C4C_MOCK_MODE", "false").lower() == "true"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_url_stub(profile_url: str) -> str:
    """
    Extract a temporary ID from LinkedIn URL.
    Example: https://www.linkedin.com/in/john-doe → john-doe
    """
    # Remove trailing slashes and query parameters
    clean_url = profile_url.rstrip('/').split('?')[0]
    
    # Extract the username part
    match = re.search(r'/in/([^/]+)', clean_url)
    if match:
        return match.group(1)
    
    # Fallback: use last part of URL
    return clean_url.split('/')[-1]


def canonical_id_from_url(profile_url: str) -> str:
    """Generate temporary canonical ID from URL before API enrichment."""
    return extract_url_stub(profile_url)


def update_canonical_ids(seen_profiles: Dict, edges: List, old_id: str, new_id: str) -> None:
    """
    Update all references to old_id with new_id after API enrichment.
    """
    # Update the node entry
    if old_id in seen_profiles:
        node = seen_profiles[old_id]
        node['id'] = new_id
        seen_profiles[new_id] = node
        if old_id != new_id:
            del seen_profiles[old_id]
    
    # Update all edge references
    for edge in edges:
        if edge['source_id'] == old_id:
            edge['source_id'] = new_id
        if edge['target_id'] == old_id:
            edge['target_id'] = new_id


def validate_graph(seen_profiles: Dict, edges: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Validate that all edge endpoints exist in nodes.
    Returns (orphan_ids, valid_edges)
    """
    node_ids = set(seen_profiles.keys())
    orphan_ids = set()
    valid_edges = []

    for edge in edges:
        if edge['source_id'] in node_ids and edge['target_id'] in node_ids:
            valid_edges.append(edge)
        else:
            if edge['source_id'] not in node_ids:
                orphan_ids.add(edge['source_id'])
            if edge['target_id'] not in node_ids:
                orphan_ids.add(edge['target_id'])

    return sorted(orphan_ids), valid_edges


def test_network_connectivity() -> Tuple[bool, str]:
    """
    Test if enrichlayer.com is reachable.
    Returns (success, message)
    """
    try:
        # Test DNS resolution for the correct domain
        ip = socket.gethostbyname("enrichlayer.com")
        
        # Test HTTPS connection to the correct endpoint
        response = requests.get("https://enrichlayer.com/api/v2/profile", timeout=5)
        return True, f"✅ Network OK (resolved to {ip})"
    
    except socket.gaierror:
        return False, (
            "❌ DNS Resolution Failed\n\n"
            "Cannot reach enrichlayer.com. This indicates a network/firewall restriction.\n\n"
            "**Solutions:**\n"
            "1. Check your internet connection\n"
            "2. Try a different network\n"
            "3. Use Mock Mode for testing"
        )
    
    except requests.exceptions.ConnectionError:
        return False, (
            "❌ Connection Failed\n\n"
            "DNS works but cannot establish HTTPS connection.\n\n"
            "**Solutions:**\n"
            "1. Check EnrichLayer service status\n"
            "2. Try a different network\n"
            "3. Use Mock Mode for testing"
        )
    
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


# ============================================================================
# ENRICHLAYER API CLIENT
# ============================================================================

def call_enrichlayer_api(api_token: str, profile_url: str, mock_mode: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call EnrichLayer person profile endpoint.
    
    Returns:
        (response_dict, error_message) tuple
        - If successful: (response, None)
        - If failed: (None, error_message)
    """
    if mock_mode:
        # Return mock data for testing
        time.sleep(0.1)  # Small delay to simulate API call
        return get_mock_response(profile_url), None
    
    # Correct EnrichLayer API endpoint (v2)
    endpoint = "https://enrichlayer.com/api/v2/profile"
    headers = {
        "Authorization": f"Bearer {api_token}",
    }
    params = {
        "url": profile_url,
        "use_cache": "if-present",  # Use cache if available
        "live_fetch": "if-needed",   # Only fetch live if needed
    }
    
    try:
        # Use GET request with params (not POST with json)
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            return None, "Invalid API token"
        elif response.status_code == 429:
            return None, "Rate limit exceeded"
        else:
            return None, f"API error {response.status_code}: {response.text}"
    
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"


def get_mock_response(profile_url: str) -> Dict:
    """
    Return mock API response from mock_personal_profile_response.json.
    Falls back to synthetic response if file not found.
    """
    try:
        mock_path = pathlib.Path(__file__).parent / "mock_personal_profile_response.json"
        with open(mock_path, "r") as f:
            base_response = json.load(f)
            # Vary the response slightly based on the profile URL
            temp_id = canonical_id_from_url(profile_url)
            base_response["public_identifier"] = temp_id
            return base_response
    except FileNotFoundError:
        # Fallback to synthetic response matching v2 API format
        temp_id = canonical_id_from_url(profile_url)
        return {
            "public_identifier": temp_id,
            "full_name": f"Mock User ({temp_id})",
            "headline": "Mock Professional",
            "location": "Mock City",
            "people_also_viewed": [
                {
                    "link": f"https://www.linkedin.com/in/mock-connection-1-{temp_id}",
                    "name": "Mock Connection 1",
                    "summary": "Mock Title 1",
                    "location": "Mock City"
                },
                {
                    "link": f"https://www.linkedin.com/in/mock-connection-2-{temp_id}",
                    "name": "Mock Connection 2",
                    "summary": "Mock Title 2",
                    "location": "Mock City"
                }
            ]
        }


# ============================================================================
# BFS CRAWLER
# ============================================================================

def run_crawler(
    seeds: List[Dict],
    api_token: str,
    max_degree: int,
    max_edges: int,
    max_nodes: int,
    status_container,
    mock_mode: bool = False
) -> Tuple[Dict, List, List, Dict]:
    """
    Run BFS crawler on seed profiles.
    
    Returns:
        (seen_profiles, edges, raw_profiles, stats)
    """
    # Initialize data structures
    queue = deque()
    seen_profiles = {}
    edges = []
    raw_profiles = []
    
    # Statistics tracking
    stats = {
        'api_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'nodes_added': 0,
        'edges_added': 0,
        'max_degree_reached': 0,
        'stopped_reason': None,
        'profiles_with_no_neighbors': 0
    }
    
    # Initialize seeds
    status_container.write("🌱 Initializing seed profiles...")
    for seed in seeds:
        temp_id = canonical_id_from_url(seed['profile_url'])
        node = {
            'id': temp_id,
            'name': seed['name'],
            'profile_url': seed['profile_url'],
            'headline': '',
            'location': '',
            'degree': 0,
            'source_type': 'seed'
        }
        seen_profiles[temp_id] = node
        queue.append(temp_id)
        stats['nodes_added'] += 1
    
    status_container.write(f"✅ Added {len(seeds)} seed profiles to queue")
    
    # BFS crawl
    while queue:
        # Check global limits
        if len(edges) >= max_edges:
            stats['stopped_reason'] = 'edge_limit'
            status_container.warning(f"⚠️ Reached edge limit ({max_edges}). Stopping crawl.")
            break
        
        if len(seen_profiles) >= max_nodes:
            stats['stopped_reason'] = 'node_limit'
            status_container.warning(f"⚠️ Reached node limit ({max_nodes}). Stopping crawl.")
            break
        
        current_id = queue.popleft()
        current_node = seen_profiles[current_id]
        
        # Stop expanding if at max degree
        if current_node['degree'] >= max_degree:
            continue
        
        # Status update
        status_container.write(f"🔍 Processing: {current_node['name']} (degree {current_node['degree']})")
        
        # Call EnrichLayer API
        stats['api_calls'] += 1
        response, error = call_enrichlayer_api(api_token, current_node['profile_url'], mock_mode=mock_mode)
        
        if error:
            stats['failed_calls'] += 1
            status_container.error(f"❌ Failed to fetch {current_node['profile_url']}: {error}")
            
            # Stop if authentication fails
            if "Invalid API token" in error:
                stats['stopped_reason'] = 'auth_error'
                break
            
            # Continue with other profiles for other errors
            continue
        
        stats['successful_calls'] += 1
        raw_profiles.append(response)
        
        # Update node with enriched data
        enriched_id = response.get('public_identifier', current_id)
        current_node['headline'] = response.get('headline', '')
        current_node['location'] = response.get('location', '')
        
        # Update canonical ID if different
        if enriched_id != current_id:
            update_canonical_ids(seen_profiles, edges, current_id, enriched_id)
            current_id = enriched_id
            current_node = seen_profiles[current_id]
        
        # Extract neighbors
        neighbors = response.get('people_also_viewed', [])
        
        # Improvement #4: Clear messaging for no neighbors
        if not neighbors:
            status_container.write("   └─ ⚠️ No 'people also viewed' connections found for this profile.")
            stats['profiles_with_no_neighbors'] += 1
        else:
            status_container.write(f"   └─ Found {len(neighbors)} connections")
        
        # Process each neighbor
        for neighbor in neighbors:
            if len(edges) >= max_edges:
                status_container.warning(f"⚠️ Reached edge limit ({max_edges}) while processing neighbors.")
                break
            
            # Handle both v2 API format and mock data format
            # v2 API uses: link, name, summary
            # Mock uses: profile_url, full_name, headline, public_identifier
            neighbor_url = neighbor.get('link') or neighbor.get('profile_url', '')
            neighbor_name = neighbor.get('name') or neighbor.get('full_name', '')
            neighbor_headline = neighbor.get('summary') or neighbor.get('headline', '')
            
            if not neighbor_url:
                continue
            
            # Use public_identifier if available (mock data), otherwise extract from URL
            neighbor_id = neighbor.get('public_identifier', canonical_id_from_url(neighbor_url))
            
            # Add edge
            edges.append({
                'source_id': current_id,
                'target_id': neighbor_id,
                'edge_type': 'people_also_viewed'
            })
            stats['edges_added'] += 1
            
            # Skip if already seen
            if neighbor_id in seen_profiles:
                continue
            
            # Check node limit
            if len(seen_profiles) >= max_nodes:
                status_container.warning(f"⚠️ Reached node limit ({max_nodes}) while processing neighbors.")
                break
            
            # Create new node
            neighbor_node = {
                'id': neighbor_id,
                'name': neighbor_name,
                'profile_url': neighbor_url,
                'headline': neighbor_headline,
                'location': neighbor.get('location', ''),
                'degree': current_node['degree'] + 1,
                'source_type': 'discovered'
            }
            seen_profiles[neighbor_id] = neighbor_node
            stats['nodes_added'] += 1
            
            # Track max degree
            stats['max_degree_reached'] = max(stats['max_degree_reached'], neighbor_node['degree'])
            
            # Enqueue if can still be expanded
            if neighbor_node['degree'] < max_degree:
                queue.append(neighbor_id)
        
        # Rate limiting delay
        if queue and not mock_mode:  # Only delay if there are more profiles and not in mock mode
            time.sleep(API_DELAY)
    
    if not stats['stopped_reason']:
        stats['stopped_reason'] = 'completed'
    
    return seen_profiles, edges, raw_profiles, stats


# ============================================================================
# CSV/JSON GENERATION
# ============================================================================

def generate_nodes_csv(seen_profiles: Dict, max_degree: int, max_edges: int, max_nodes: int) -> str:
    """Generate nodes.csv content with metadata header."""
    nodes_data = []
    for node in seen_profiles.values():
        nodes_data.append({
            'id': node['id'],
            'name': node['name'],
            'profile_url': node['profile_url'],
            'headline': node.get('headline', ''),
            'location': node.get('location', ''),
            'degree': node['degree'],
            'source_type': node['source_type']
        })
    
    df = pd.DataFrame(nodes_data)
    csv_body = df.to_csv(index=False)
    
    # Improvement #5: Add metadata header
    meta = (
        f"# generated_at={datetime.utcnow().isoformat()}Z; "
        f"max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}\n"
    )
    
    return meta + csv_body


def generate_edges_csv(edges: List, max_degree: int, max_edges: int, max_nodes: int) -> str:
    """Generate edges.csv content with metadata header."""
    df = pd.DataFrame(edges)
    csv_body = df.to_csv(index=False)
    
    # Improvement #5: Add metadata header
    meta = (
        f"# generated_at={datetime.utcnow().isoformat()}Z; "
        f"max_degree={max_degree}; max_edges={max_edges}; max_nodes={max_nodes}\n"
    )
    
    return meta + csv_body


def generate_raw_json(raw_profiles: List) -> str:
    """Generate raw_profiles.json content."""
    return json.dumps(raw_profiles, indent=2)


def create_download_zip(nodes_csv: str, edges_csv: str, raw_json: str) -> bytes:
    """
    Create a ZIP file containing all three output files.
    Returns ZIP file as bytes.
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add nodes.csv
        zip_file.writestr('nodes.csv', nodes_csv)
        # Add edges.csv
        zip_file.writestr('edges.csv', edges_csv)
        # Add raw_profiles.json
        zip_file.writestr('raw_profiles.json', raw_json)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="C4C Network Seed Crawler",
        page_icon="https://static.wixstatic.com/media/275a3f_9c48d5079fcf4b688606c81d8f34d5a5~mv2.jpg",
        layout="wide"
    )
    
    # Initialize session state for preserving results
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = None
    
    # Header with C4C logo
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image(
            "https://static.wixstatic.com/media/275a3f_9c48d5079fcf4b688606c81d8f34d5a5~mv2.jpg",
            width=80
        )
    with col2:
        st.title("C4C Network Seed Crawler")
    
    st.markdown("Convert LinkedIn seed profiles into a Polinode-ready network using EnrichLayer")
    
    # ========================================================================
    # MODE SELECTION
    # ========================================================================
    
    st.markdown("---")
    
    st.subheader("🎛️ Select Mode")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        advanced_mode = st.toggle(
            "Advanced Mode",
            value=False,
            help="Enable network analysis and insights"
        )
    
    with col2:
        if advanced_mode:
            st.info("""
            **🔬 Advanced Mode** - Network Intelligence  
            Includes everything in Basic Mode plus:
            - Centrality metrics (degree, betweenness, eigenvector, closeness)
            - Community detection and clustering
            - Brokerage analysis (coordinators, gatekeepers, liaisons)
            - Key position identification (connectors, brokers, bridges)
            - Network insights and strategic recommendations
            
            *⏱️ Longer processing time, richer insights*
            """)
        else:
            st.success("""
            **📊 Basic Mode** - Quick Network Crawl  
            Perfect for rapid exploration:
            - Crawl LinkedIn networks (1 or 2 degrees)
            - Export nodes, edges, and raw profiles
            - Import directly to Polinode or other tools
            - Fast processing, clean data
            
            *⚡ Quick results, simple outputs*
            """)
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 1: INPUT
    # ========================================================================
    
    st.header("📥 Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Upload Seed Profiles")
        uploaded_file = st.file_uploader(
            "Upload CSV with columns: name, profile_url (max 5 rows)",
            type=['csv'],
            help="CSV must contain 'name' and 'profile_url' columns with 1-5 seed profiles"
        )
        
        seeds = []
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['name', 'profile_url']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                elif len(df) > 5:
                    st.error("❌ Prototype limit: max 5 seed profiles.")
                elif len(df) == 0:
                    st.error("❌ CSV file is empty.")
                else:
                    seeds = df.to_dict('records')
                    st.success(f"✅ Loaded {len(seeds)} seed profiles")
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with col2:
        st.subheader("2. EnrichLayer API Token")
        
        # Improvement #6: Optional auto-fill from secrets
        default_token = ""
        try:
            default_token = st.secrets.get("ENRICHLAYER_TOKEN", "")
        except:
            pass
        
        api_token = st.text_input(
            "Enter your API token",
            type="password",
            value=default_token,
            help="Get your token from EnrichLayer dashboard. Not stored, used only for this session."
        )
        
        # Network connectivity test
        if st.button("🔍 Test API Connection", help="Check if EnrichLayer API is reachable"):
            with st.spinner("Testing connection..."):
                success, message = test_network_connectivity()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Improvement #1: UI-based mock mode toggle
        mock_mode = st.toggle(
            "Run in mock mode (no real API calls)",
            value=DEFAULT_MOCK_MODE,
            help="Use mock responses for testing without consuming API credits."
        )
        
        if mock_mode:
            st.info("🧪 Running in MOCK MODE (no real API calls)")
    
    # ========================================================================
    # SECTION 2: CONFIGURATION
    # ========================================================================
    
    st.header("⚙️ Crawl Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_degree = st.radio(
            "Maximum Degree (hops)",
            options=[1, 2],
            index=1,
            help="1 hop = direct connections only, 2 hops = connections of connections"
        )
    
    with col2:
        st.markdown("**Crawl Limits:**")
        st.metric("Max Edges", 1000)
        st.metric("Max Nodes", 500)
    
    # ========================================================================
    # RUN BUTTON
    # ========================================================================
    
    can_run = len(seeds) > 0 and (api_token or mock_mode)
    
    if not can_run:
        if len(seeds) == 0:
            st.warning("⚠️ Please upload a valid seed CSV to continue.")
        elif not api_token and not mock_mode:
            st.warning("⚠️ Please enter your EnrichLayer API token to continue.")
    
    run_button = st.button(
        "🚀 Run Crawl",
        disabled=not can_run,
        type="primary",
        use_container_width=True
    )
    
    # ========================================================================
    # CRAWL EXECUTION
    # ========================================================================
    
    if run_button:
        st.header("🔄 Crawl Progress")
        
        status_container = st.status("Running crawl...", expanded=True)
        
        # Run the crawler
        seen_profiles, edges, raw_profiles, stats = run_crawler(
            seeds=seeds,
            api_token=api_token,
            max_degree=max_degree,
            max_edges=1000,
            max_nodes=500,
            status_container=status_container,
            mock_mode=mock_mode
        )
        
        status_container.update(label="✅ Crawl Complete!", state="complete")
        
        # Improvement #3: Graph validation
        orphan_ids, valid_edges = validate_graph(seen_profiles, edges)
        
        if orphan_ids:
            st.warning(
                f"⚠️ Detected {len(orphan_ids)} orphan node IDs referenced in edges but "
                "not present in nodes. Edges involving these IDs have been excluded from the download."
            )
            edges = valid_edges
        
        # Improvement #4: Special message for empty results
        if len(edges) == 0:
            st.info(
                "ℹ️ Crawl completed, but no connections were found. "
                "This may mean the selected profiles have limited 'people also viewed' data, "
                "or that the crawl depth was too shallow."
            )
        
        # Store results in session state so they persist across reruns (e.g., after downloads)
        st.session_state.crawl_results = {
            'seen_profiles': seen_profiles,
            'edges': edges,
            'raw_profiles': raw_profiles,
            'stats': stats,
            'max_degree': max_degree,
            'advanced_mode': advanced_mode  # Store mode setting
        }
    
    # Display results if available (either from current run or session state)
    if st.session_state.crawl_results is not None:
        results = st.session_state.crawl_results
        seen_profiles = results['seen_profiles']
        edges = results['edges']
        raw_profiles = results['raw_profiles']
        stats = results['stats']
        max_degree = results['max_degree']
        was_advanced_mode = results.get('advanced_mode', False)
        
        # ====================================================================
        # RESULTS SUMMARY
        # ====================================================================
        
        st.header("📊 Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", len(seen_profiles))
        col2.metric("Total Edges", len(edges))
        col3.metric("Max Degree", stats['max_degree_reached'])
        col4.metric("API Calls", stats['api_calls'])
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Successful", stats['successful_calls'], delta_color="normal")
        col6.metric("Failed", stats['failed_calls'], delta_color="inverse")
        col7.metric("No Neighbors", stats['profiles_with_no_neighbors'])
        
        if stats['stopped_reason'] == 'completed':
            col8.success("✅ Completed")
        elif stats['stopped_reason'] == 'edge_limit':
            col8.warning("⚠️ Edge Limit")
        elif stats['stopped_reason'] == 'node_limit':
            col8.warning("⚠️ Node Limit")
        elif stats['stopped_reason'] == 'auth_error':
            col8.error("❌ Auth Error")
        
        # ====================================================================
        # ADVANCED ANALYTICS (if advanced mode was enabled)
        # ====================================================================
        
        if was_advanced_mode:
            st.markdown("---")
            st.header("🔬 Advanced Network Analytics")
            
            st.info("""
            **🚧 Advanced Analytics - Coming Soon!**
            
            The following features are currently in development:
            
            **Network Metrics** (Next Release)
            - Degree centrality (in/out/total)
            - Betweenness centrality (identify brokers)
            - Eigenvector centrality (identify influencers)
            - Closeness centrality (identify connectors)
            - Clustering coefficient
            
            **Community Detection** (In Progress)
            - Identify network clusters
            - Calculate modularity scores
            - Label communities by organization/sector
            
            **Brokerage Analysis** (Planned)
            - Coordinators (within-group brokers)
            - Gatekeepers (control inflow)
            - Representatives (control outflow)
            - Liaisons (connect unrelated groups)
            - Structural hole positions
            
            **Strategic Insights** (Future)
            - Hidden brokers and key connectors
            - Alignment gaps across sectors
            - Collaboration opportunities
            - Minimum viable coalition identification
            
            For now, download your basic files below and import to Polinode for visualization and analysis.
            """)
            
            st.markdown("**Note:** When these features are ready, you'll see:")
            st.markdown("- 📊 Enhanced nodes.csv with centrality metrics")
            st.markdown("- 📈 network_analysis.json with summary statistics")
            st.markdown("- 🎯 key_positions.csv identifying important actors")
            st.markdown("- 🔗 brokerage_matrix.csv showing structural roles")
        
        # ====================================================================
        # DOWNLOAD SECTION
        # ====================================================================
        
        st.header("💾 Download Results")
        
        # Generate files
        nodes_csv = generate_nodes_csv(seen_profiles, max_degree=max_degree, max_edges=1000, max_nodes=500)
        edges_csv = generate_edges_csv(edges, max_degree=max_degree, max_edges=1000, max_nodes=500)
        raw_json = generate_raw_json(raw_profiles)
        
        # Primary action: Download all as ZIP
        st.markdown("### 📦 Download All Files")
        zip_data = create_download_zip(nodes_csv, edges_csv, raw_json)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.download_button(
                label="⬇️ Download All as ZIP (Recommended)",
                data=zip_data,
                file_name="c4c_network_crawl.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                help="Download all three files (nodes.csv, edges.csv, raw_profiles.json) in one ZIP file"
            )
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True, help="Clear results to start a new crawl"):
                st.session_state.crawl_results = None
                st.rerun()
        
        # Individual downloads
        st.markdown("### 📄 Download Individual Files")
        st.caption("Or download files individually (results will stay available)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download nodes.csv",
                data=nodes_csv,
                file_name="nodes.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_nodes"
            )
        
        with col2:
            st.download_button(
                label="📥 Download edges.csv",
                data=edges_csv,
                file_name="edges.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_edges"
            )
        
        with col3:
            st.download_button(
                label="📥 Download raw_profiles.json",
                data=raw_json,
                file_name="raw_profiles.json",
                mime="application/json",
                use_container_width=True,
                key="download_raw"
            )
        
        # ====================================================================
        # DATA PREVIEW
        # ====================================================================
        
        with st.expander("👀 Preview Nodes"):
            st.dataframe(pd.DataFrame([node for node in seen_profiles.values()]))
        
        with st.expander("👀 Preview Edges"):
            if len(edges) > 0:
                st.dataframe(pd.DataFrame(edges))
            else:
                st.info("No edges to display")


if __name__ == "__main__":
    main()
