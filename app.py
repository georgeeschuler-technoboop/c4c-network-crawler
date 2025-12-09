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
import networkx as nx

# ============================================================================
# CONFIGURATION
# ============================================================================

API_DELAY = 3.0  # Seconds between API calls (20 requests/min for $49/mo EnrichLayer plan)
PER_MIN_LIMIT = 20  # Tuned for $49/mo plan (can be changed for other tiers)


# ============================================================================
# RATE LIMITER CLASS
# ============================================================================

class RateLimiter:
    """
    Rate limiter that enforces a per-minute request limit.
    Uses a sliding window approach with a safety buffer.
    """
    def __init__(self, per_min_limit: int, buffer: float = 0.8):
        """
        Args:
            per_min_limit: documented limit (e.g., 20 requests/min)
            buffer: safety factor (0.8 → aim for 16/min so we never hit the hard cap)
        """
        self.per_min_limit = per_min_limit
        self.allowed_per_min = max(1, int(per_min_limit * buffer))
        self.window_start = time.time()
        self.calls_in_window = 0

    def wait_for_slot(self):
        """Wait until we have a safe slot to make a request."""
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
        """Record that we made a call."""
        self.calls_in_window += 1
    
    def get_status(self) -> str:
        """Get current rate limiter status for display."""
        return f"{self.calls_in_window}/{self.allowed_per_min} calls this minute"
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


def extract_organization(occupation: str = '', experiences: List = None) -> str:
    """
    Extract organization name from occupation string or experiences.
    
    Args:
        occupation: String like "Chief Executive Officer at Toniic"
        experiences: List of experience dicts with 'company' field
    
    Returns:
        Organization name or empty string
    """
    # Try occupation field first (most reliable for current role)
    if occupation and ' at ' in occupation:
        # "Chief Executive Officer at Toniic" → "Toniic"
        org = occupation.split(' at ', 1)[1].strip()
        # Clean up common suffixes
        org = org.replace('|', '').strip()
        return org
    
    # Fallback to most recent experience
    if experiences and len(experiences) > 0:
        # Get most recent (first in list, usually current)
        recent = experiences[0]
        if 'company' in recent and recent['company']:
            return recent['company'].strip()
    
    return ''


def infer_sector(organization: str, headline: str = '') -> str:
    """
    Infer sector/industry from organization name and headline.
    Simple keyword-based classification.
    """
    combined = f"{organization} {headline}".lower()
    
    # Keyword mappings
    if any(word in combined for word in ['foundation', 'philanthropy', 'donor', 'giving']):
        return 'Philanthropy'
    elif any(word in combined for word in ['ngo', 'nonprofit', 'charity', 'humanitarian']):
        return 'Nonprofit'
    elif any(word in combined for word in ['government', 'ministry', 'federal', 'state', 'public sector']):
        return 'Government'
    elif any(word in combined for word in ['university', 'college', 'academic', 'research', 'professor']):
        return 'Academia'
    elif any(word in combined for word in ['peace', 'conflict', 'democracy', 'justice', 'rights']):
        return 'Peacebuilding/Democracy'
    elif any(word in combined for word in ['social impact', 'social change', 'community development']):
        return 'Social Impact'
    elif any(word in combined for word in ['consulting', 'consultant', 'advisory']):
        return 'Consulting'
    elif any(word in combined for word in ['finance', 'investment', 'capital', 'fund', 'bank']):
        return 'Finance'
    elif any(word in combined for word in ['tech', 'software', 'digital', 'platform']):
        return 'Technology'
    elif any(word in combined for word in ['corp', 'inc', 'llc', 'ltd', 'company']):
        return 'Corporate'
    else:
        return 'Other'


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

def call_enrichlayer_api(api_token: str, profile_url: str, mock_mode: bool = False, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call EnrichLayer person profile endpoint with retry logic.
    
    Returns:
        (response_dict, error_message) tuple
        - If successful: (response, None)
        - If failed: (None, error_message)
    
    Implements exponential backoff for rate limit errors (429).
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
    
    for attempt in range(max_retries):
        try:
            # Use GET request with params (not POST with json)
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 401:
                return None, "Invalid API token"
            elif response.status_code == 403:
                # Out of credits - don't retry
                return None, "Out of credits (check your EnrichLayer balance)"
            elif response.status_code == 429:
                # Rate limit - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                    time.sleep(wait_time)
                    continue
                else:
                    return None, f"Rate limit exceeded (tried {max_retries} times)"
            elif response.status_code == 503:
                # Enrichment failed - can retry
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    return None, "Enrichment failed after retries"
            else:
                return None, f"API error {response.status_code}: {response.text}"
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"Request timed out (tried {max_retries} times)"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"Network error: {str(e)}"
    
    return None, "Failed after maximum retries"


def get_mock_response(profile_url: str) -> Dict:
    """
    Generate comprehensive mock API response for stress testing.
    
    Creates deterministic but varied fake profiles with:
    - 25-40 "people_also_viewed" connections per profile
    - Realistic names, titles, organizations, sectors
    - Enough data to stress test 1000 node / 1000 edge limits
    
    Uses hash of profile URL for deterministic randomness.
    """
    import hashlib
    
    # Deterministic seed based on profile URL
    url_hash = int(hashlib.md5(profile_url.encode()).hexdigest(), 16)
    
    # Mock data pools
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
        "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
        "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker"
    ]
    
    titles = [
        "CEO", "Founder", "Director", "VP", "Manager", "Consultant", "Partner",
        "Executive Director", "Chief Strategy Officer", "Program Director",
        "Senior Advisor", "Managing Director", "Principal", "Fellow", "Board Member",
        "Chief Impact Officer", "Head of Partnerships", "Director of Development",
        "Senior Program Officer", "Policy Director", "Research Director"
    ]
    
    organizations = [
        "World Resources Institute", "The Nature Conservancy", "WWF", "IUCN",
        "Conservation International", "Environmental Defense Fund", "Sierra Club",
        "Greenpeace", "Earthjustice", "Ocean Conservancy", "Wildlife Conservation Society",
        "Rainforest Alliance", "Global Water Partnership", "Water.org", "charity: water",
        "Pacific Institute", "Alliance for Water Stewardship", "CDP", "Ceres",
        "BSR", "World Economic Forum", "Aspen Institute", "Brookings Institution",
        "Carnegie Endowment", "Council on Foreign Relations", "RAND Corporation",
        "McKinsey & Company", "Boston Consulting Group", "Bain & Company", "Deloitte",
        "Accenture", "PwC", "EY", "KPMG", "Goldman Sachs", "JPMorgan Chase",
        "Bank of America", "Citigroup", "Morgan Stanley", "BlackRock", "Vanguard",
        "Ford Foundation", "Rockefeller Foundation", "MacArthur Foundation",
        "Gates Foundation", "Hewlett Foundation", "Packard Foundation", "Bloomberg Philanthropies",
        "Open Society Foundations", "Omidyar Network", "Skoll Foundation", "Toniic",
        "Stanford University", "Harvard University", "MIT", "Yale University",
        "Columbia University", "UC Berkeley", "Princeton University", "Oxford University"
    ]
    
    locations = [
        "San Francisco, CA", "New York, NY", "Washington, DC", "Boston, MA",
        "Los Angeles, CA", "Seattle, WA", "Chicago, IL", "Denver, CO",
        "Austin, TX", "Portland, OR", "Miami, FL", "Atlanta, GA",
        "London, UK", "Geneva, Switzerland", "Amsterdam, Netherlands",
        "Berlin, Germany", "Paris, France", "Singapore", "Hong Kong",
        "Tokyo, Japan", "Sydney, Australia", "Toronto, Canada", "Vancouver, Canada"
    ]
    
    sectors = [
        "Philanthropy", "Nonprofit", "Consulting", "Finance", "Technology",
        "Academia", "Government", "Social Impact", "Corporate", "Peacebuilding/Democracy"
    ]
    
    # Generate profile data based on URL hash
    temp_id = canonical_id_from_url(profile_url)
    
    # Use hash to pick attributes deterministically
    first_name = first_names[url_hash % len(first_names)]
    last_name = last_names[(url_hash // 100) % len(last_names)]
    title = titles[(url_hash // 1000) % len(titles)]
    org = organizations[(url_hash // 10000) % len(organizations)]
    location = locations[(url_hash // 100000) % len(locations)]
    sector = sectors[(url_hash // 1000000) % len(sectors)]
    
    full_name = f"{first_name} {last_name}"
    headline = f"{title} at {org}"
    occupation = headline
    
    # Generate 25-40 connections (deterministic based on hash)
    num_connections = 25 + (url_hash % 16)  # 25-40 connections
    
    people_also_viewed = []
    for i in range(num_connections):
        # Create unique but deterministic connection
        conn_hash = (url_hash + i * 7919) % (2**32)  # Prime multiplier for variety
        
        conn_first = first_names[conn_hash % len(first_names)]
        conn_last = last_names[(conn_hash // 100) % len(last_names)]
        conn_title = titles[(conn_hash // 1000) % len(titles)]
        conn_org = organizations[(conn_hash // 10000) % len(organizations)]
        conn_location = locations[(conn_hash // 100000) % len(locations)]
        
        conn_name = f"{conn_first} {conn_last}"
        conn_id = f"{conn_first.lower()}-{conn_last.lower()}-{conn_hash % 1000}"
        
        people_also_viewed.append({
            "link": f"https://www.linkedin.com/in/{conn_id}",
            "name": conn_name,
            "summary": f"{conn_title} at {conn_org}",
            "location": conn_location
        })
    
    # Build response matching EnrichLayer v2 API format
    return {
        "public_identifier": temp_id,
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "headline": headline,
        "occupation": occupation,
        "location_str": location,
        "summary": f"Experienced {title.lower()} with expertise in {sector.lower()}.",
        "experiences": [
            {
                "company": org,
                "title": title,
                "starts_at": {"year": 2020, "month": 1},
                "ends_at": None
            },
            {
                "company": organizations[(url_hash // 50000) % len(organizations)],
                "title": titles[(url_hash // 5000) % len(titles)],
                "starts_at": {"year": 2015, "month": 6},
                "ends_at": {"year": 2019, "month": 12}
            }
        ],
        "people_also_viewed": people_also_viewed
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
    mock_mode: bool = False,
    advanced_mode: bool = False,
    progress_bar = None,
    per_min_limit: int = PER_MIN_LIMIT
) -> Tuple[Dict, List, List, Dict]:
    """
    Run BFS crawler on seed profiles.
    
    Returns:
        (seen_profiles, edges, raw_profiles, stats)
    """
    # Initialize rate limiter
    rate_limiter = None
    if not mock_mode:
        rate_limiter = RateLimiter(per_min_limit=per_min_limit)
    
    # Initialize data structures
    queue = deque()
    seen_profiles = {}
    edges = []
    raw_profiles = []
    processed_nodes = 0  # Track progress
    
    # Statistics tracking
    stats = {
        'api_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'nodes_added': 0,
        'edges_added': 0,
        'max_degree_reached': 0,
        'stopped_reason': None,
        'profiles_with_no_neighbors': 0,
        'error_breakdown': {
            'rate_limit': 0,
            'out_of_credits': 0,
            'auth_error': 0,
            'not_found': 0,
            'enrichment_failed': 0,
            'other': 0,
            'consecutive_rate_limits': 0
        }
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
        processed_nodes += 1
        
        # Update progress bar (based on processed vs total known)
        if progress_bar is not None:
            total_known = processed_nodes + len(queue)
            if total_known > 0:
                progress = processed_nodes / total_known
                progress_bar.progress(
                    min(max(progress, 0.0), 0.99),  # Cap at 99% until complete
                    text=f"Processing... {processed_nodes} done, {len(queue)} remaining"
                )
        
        # Stop expanding if at max degree
        if current_node['degree'] >= max_degree:
            continue
        
        # Status update with real-time stats
        progress_text = f"🔍 Processing: {current_node['name']} (degree {current_node['degree']})"
        if stats['api_calls'] > 0:
            success_rate = (stats['successful_calls'] / stats['api_calls']) * 100
            progress_text += f" | ✅ {stats['successful_calls']}/{stats['api_calls']} ({success_rate:.0f}%)"
            if stats['failed_calls'] > 0:
                progress_text += f" | ❌ {stats['failed_calls']} failed"
                if stats['error_breakdown'].get('rate_limit', 0) > 0:
                    progress_text += f" (Rate limited: {stats['error_breakdown']['rate_limit']})"
        if rate_limiter is not None:
            progress_text += f" | ⏱️ {rate_limiter.get_status()}"
        status_container.write(progress_text)
        
        # Rate limiting: wait for a safe slot before calling the API
        if rate_limiter is not None:
            rate_limiter.wait_for_slot()
        
        # Call EnrichLayer API
        stats['api_calls'] += 1
        response, error = call_enrichlayer_api(api_token, current_node['profile_url'], mock_mode=mock_mode)
        
        # Record the API call for rate limiting
        if rate_limiter is not None:
            rate_limiter.record_call()
        
        # Tiny courtesy delay (the real throttle is per-minute via RateLimiter)
        if not mock_mode:
            time.sleep(0.2)
        
        if error:
            stats['failed_calls'] += 1
            status_container.error(f"❌ Failed to fetch {current_node['profile_url']}: {error}")
            
            # Classify error type
            if "Rate limit" in error:
                stats['error_breakdown']['rate_limit'] += 1
                
                # Check for excessive consecutive rate limits
                consecutive_rate_limits = stats['error_breakdown'].get('consecutive_rate_limits', 0) + 1
                stats['error_breakdown']['consecutive_rate_limits'] = consecutive_rate_limits
                
                if consecutive_rate_limits >= 10:
                    st.warning(f"""
                    **⏸️ Pausing: Too many consecutive rate limits ({consecutive_rate_limits})**
                    
                    Waiting 30 seconds before continuing...
                    Consider stopping and using Degree 1 instead.
                    """)
                    time.sleep(30)  # Long pause after many consecutive failures
                    stats['error_breakdown']['consecutive_rate_limits'] = 0  # Reset counter
            elif "Out of credits" in error:
                stats['error_breakdown']['out_of_credits'] += 1
                stats['stopped_reason'] = 'out_of_credits'
                st.error("🚫 **Out of Credits!** Check your EnrichLayer balance.")
                break
            elif "Invalid API token" in error:
                stats['error_breakdown']['auth_error'] += 1
                stats['stopped_reason'] = 'auth_error'
                break
            elif "Enrichment failed" in error:
                stats['error_breakdown']['enrichment_failed'] += 1
            elif "404" in error or "not found" in error.lower():
                stats['error_breakdown']['not_found'] += 1
            else:
                stats['error_breakdown']['other'] += 1
            
            # Continue with other profiles for non-fatal errors
            continue
        
        stats['successful_calls'] += 1
        stats['error_breakdown']['consecutive_rate_limits'] = 0  # Reset on success
        raw_profiles.append(response)
        
        # Update node with enriched data
        enriched_id = response.get('public_identifier', current_id)
        current_node['headline'] = response.get('headline', '')
        current_node['location'] = response.get('location_str') or response.get('location', '')
        
        # Advanced mode: Extract organization and sector
        if advanced_mode:
            occupation = response.get('occupation', '')
            experiences = response.get('experiences', [])
            organization = extract_organization(occupation, experiences)
            sector = infer_sector(organization, current_node['headline'])
            
            current_node['organization'] = organization
            current_node['sector'] = sector
        
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
            
            # Advanced mode: Add organization and sector (will be populated when fetched)
            if advanced_mode:
                neighbor_node['organization'] = ''  # Will be filled when profile is fetched
                neighbor_node['sector'] = ''
            
            seen_profiles[neighbor_id] = neighbor_node
            stats['nodes_added'] += 1
            
            # Track max degree
            stats['max_degree_reached'] = max(stats['max_degree_reached'], neighbor_node['degree'])
            
            # Enqueue if can still be expanded
            if neighbor_node['degree'] < max_degree:
                queue.append(neighbor_id)
    
    if not stats['stopped_reason']:
        stats['stopped_reason'] = 'completed'
    
    return seen_profiles, edges, raw_profiles, stats


# ============================================================================
# NETWORK METRICS CALCULATION
# ============================================================================

def calculate_network_metrics(seen_profiles: Dict, edges: List) -> Dict:
    """
    Calculate network centrality metrics using NetworkX.
    
    Returns dict with:
        - node_metrics: {node_id: {metric_name: value, ...}, ...}
        - network_stats: {metric_name: value, ...}
        - top_nodes: {metric_name: [(node_id, value), ...], ...}
    """
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node_id, node_data in seen_profiles.items():
        G.add_node(node_id, **node_data)
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source_id'], edge['target_id'])
    
    # Initialize results
    node_metrics = {node_id: {} for node_id in seen_profiles.keys()}
    network_stats = {}
    top_nodes = {}
    
    # Skip calculations if graph is too small
    if len(G.nodes()) < 2 or len(G.edges()) < 1:
        return {
            'node_metrics': node_metrics,
            'network_stats': {'nodes': len(G.nodes()), 'edges': len(G.edges())},
            'top_nodes': {}
        }
    
    # ----- DEGREE CENTRALITY -----
    # Number of connections (normalized by max possible)
    try:
        degree_centrality = nx.degree_centrality(G)
        for node_id, value in degree_centrality.items():
            if node_id in node_metrics:
                node_metrics[node_id]['degree_centrality'] = round(value, 4)
        
        # Also store raw degree count
        for node_id in G.nodes():
            if node_id in node_metrics:
                node_metrics[node_id]['degree'] = G.degree(node_id)
        
        # Top 10 by degree
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['degree'] = sorted_degree
        
        network_stats['avg_degree'] = round(sum(dict(G.degree()).values()) / len(G.nodes()), 2)
        network_stats['max_degree'] = max(dict(G.degree()).values())
    except Exception as e:
        pass  # Skip if calculation fails
    
    # ----- BETWEENNESS CENTRALITY -----
    # How often a node lies on shortest paths between other nodes (identifies brokers)
    try:
        betweenness = nx.betweenness_centrality(G)
        for node_id, value in betweenness.items():
            if node_id in node_metrics:
                node_metrics[node_id]['betweenness_centrality'] = round(value, 4)
        
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['betweenness'] = sorted_betweenness
        
        network_stats['avg_betweenness'] = round(sum(betweenness.values()) / len(betweenness), 4)
    except Exception as e:
        pass
    
    # ----- EIGENVECTOR CENTRALITY -----
    # Connected to well-connected people (influence)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
        for node_id, value in eigenvector.items():
            if node_id in node_metrics:
                node_metrics[node_id]['eigenvector_centrality'] = round(value, 4)
        
        sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['eigenvector'] = sorted_eigenvector
    except Exception as e:
        # Eigenvector can fail on disconnected graphs
        pass
    
    # ----- CLOSENESS CENTRALITY -----
    # Average distance to all other nodes (accessibility)
    try:
        # Use connected components for disconnected graphs
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            # Calculate for largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subgraph)
        
        for node_id, value in closeness.items():
            if node_id in node_metrics:
                node_metrics[node_id]['closeness_centrality'] = round(value, 4)
        
        sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes['closeness'] = sorted_closeness
    except Exception as e:
        pass
    
    # ----- NETWORK-LEVEL STATS -----
    network_stats['nodes'] = len(G.nodes())
    network_stats['edges'] = len(G.edges())
    
    try:
        network_stats['density'] = round(nx.density(G), 4)
    except:
        pass
    
    try:
        if nx.is_connected(G):
            network_stats['diameter'] = nx.diameter(G)
            network_stats['avg_path_length'] = round(nx.average_shortest_path_length(G), 2)
        else:
            # For disconnected graphs, report on largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            network_stats['largest_component_size'] = len(largest_cc)
            network_stats['num_components'] = nx.number_connected_components(G)
            if len(subgraph) > 1:
                network_stats['diameter_largest_cc'] = nx.diameter(subgraph)
    except:
        pass
    
    try:
        network_stats['avg_clustering'] = round(nx.average_clustering(G), 4)
    except:
        pass
    
    return {
        'node_metrics': node_metrics,
        'network_stats': network_stats,
        'top_nodes': top_nodes
    }


def generate_network_analysis_json(network_metrics: Dict, seen_profiles: Dict) -> str:
    """Generate network_analysis.json with summary statistics and top nodes."""
    
    analysis = {
        'generated_at': datetime.now().isoformat(),
        'network_statistics': network_metrics.get('network_stats', {}),
        'top_connectors': [],
        'top_brokers': [],
        'top_influencers': [],
        'metric_definitions': {
            'degree_centrality': 'Number of direct connections (normalized). High = well-connected.',
            'betweenness_centrality': 'How often node lies on shortest paths. High = broker/bridge.',
            'eigenvector_centrality': 'Connected to influential people. High = influential.',
            'closeness_centrality': 'Average distance to all others. High = central/accessible.'
        }
    }
    
    top_nodes = network_metrics.get('top_nodes', {})
    
    # Add top connectors (by degree)
    if 'degree' in top_nodes:
        for node_id, score in top_nodes['degree']:
            if node_id in seen_profiles:
                analysis['top_connectors'].append({
                    'id': node_id,
                    'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'degree_centrality': score,
                    'connections': network_metrics['node_metrics'].get(node_id, {}).get('degree', 0)
                })
    
    # Add top brokers (by betweenness)
    if 'betweenness' in top_nodes:
        for node_id, score in top_nodes['betweenness']:
            if node_id in seen_profiles:
                analysis['top_brokers'].append({
                    'id': node_id,
                    'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'betweenness_centrality': score
                })
    
    # Add top influencers (by eigenvector)
    if 'eigenvector' in top_nodes:
        for node_id, score in top_nodes['eigenvector']:
            if node_id in seen_profiles:
                analysis['top_influencers'].append({
                    'id': node_id,
                    'name': seen_profiles[node_id].get('name', ''),
                    'organization': seen_profiles[node_id].get('organization', ''),
                    'eigenvector_centrality': score
                })
    
    return json.dumps(analysis, indent=2)


# ============================================================================
# CSV/JSON GENERATION
# ============================================================================

def generate_nodes_csv(seen_profiles: Dict, max_degree: int, max_edges: int, max_nodes: int, network_metrics: Dict = None) -> str:
    """Generate nodes.csv content with metadata header and optional network metrics."""
    nodes_data = []
    
    # Get node metrics if available
    node_metrics = {}
    if network_metrics:
        node_metrics = network_metrics.get('node_metrics', {})
    
    for node in seen_profiles.values():
        node_dict = {
            'id': node['id'],
            'name': node['name'],
            'profile_url': node['profile_url'],
            'headline': node.get('headline', ''),
            'location': node.get('location', ''),
            'degree': node['degree'],
            'source_type': node['source_type']
        }
        
        # Add organization and sector if present (advanced mode)
        if 'organization' in node:
            node_dict['organization'] = node.get('organization', '')
        if 'sector' in node:
            node_dict['sector'] = node.get('sector', '')
        
        # Add network metrics if available (advanced mode)
        if node['id'] in node_metrics:
            metrics = node_metrics[node['id']]
            node_dict['connections'] = metrics.get('degree', 0)
            node_dict['degree_centrality'] = metrics.get('degree_centrality', 0)
            node_dict['betweenness_centrality'] = metrics.get('betweenness_centrality', 0)
            node_dict['eigenvector_centrality'] = metrics.get('eigenvector_centrality', 0)
            node_dict['closeness_centrality'] = metrics.get('closeness_centrality', 0)
        
        nodes_data.append(node_dict)
    
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


def create_download_zip(nodes_csv: str, edges_csv: str, raw_json: str, analysis_json: str = None) -> bytes:
    """
    Create a ZIP file containing all output files.
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
        # Add network_analysis.json if available
        if analysis_json:
            zip_file.writestr('network_analysis.json', analysis_json)
    
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
            st.info("""
            🧪 **MOCK MODE** - No real API calls, no credits used!
            
            Generates realistic synthetic network data:
            - 25-40 connections per profile (deterministic)
            - Realistic names, titles, organizations
            - Perfect for stress testing node/edge limits
            - Use `mock_test_seeds.csv` or any seed file
            """)
    
    # ========================================================================
    # SECTION 2: CONFIGURATION
    # ========================================================================
    
    st.header("⚙️ Crawl Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_degree = st.radio(
            "Maximum Degree (hops)",
            options=[1, 2],
            index=0,  # Default to Degree 1 (safer)
            help="1 hop = direct connections only, 2 hops = connections of connections"
        )
        
        # Warning for degree 2
        if max_degree == 2:
            st.error("""
            **⚠️ Degree 2 Warning - Read Before Running!**
            
            Degree 2 crawls are **expensive and slow**:
            - 📊 10-50x more API calls than Degree 1
            - 💳 Uses 100-500 credits
            - ⏱️ Takes 30-90+ minutes (at 20 req/min)
            - 🚫 May still hit rate limits on large networks
            
            **💡 Recommendation:** Start with Degree 1 first!
            """, icon="🚨")
        else:
            st.success("""
            **✅ Degree 1 Selected - Good Choice!**
            
            - 🎯 Direct connections only
            - ⚡ ~1-2 minutes for 5 seeds
            - 💰 Uses ~5-10 credits
            - ✅ Reliable, low rate limit risk
            """, icon="👍")
    
    with col2:
        st.markdown("**Crawl Limits:**")
        st.metric("Max Edges", 5000)
        st.metric("Max Nodes", 2500)
    
    # Rate Limit Information (cleaner version per team feedback)
    st.caption(f"""
    ⏱️ **API pacing:** This prototype is tuned for up to **{PER_MIN_LIMIT} requests/minute**
    (EnrichLayer $49/mo plan). The app automatically throttles calls so we don't hit rate limits.
    Progress bar shows processed nodes vs. remaining queue.
    """)
    
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
        
        # Progress bar
        progress_bar = st.progress(0.0, text="Starting crawl...")
        
        status_container = st.status("Running crawl...", expanded=True)
        
        # Run the crawler
        seen_profiles, edges, raw_profiles, stats = run_crawler(
            seeds=seeds,
            api_token=api_token,
            max_degree=max_degree,
            max_edges=5000,
            max_nodes=2500,
            status_container=status_container,
            mock_mode=mock_mode,
            advanced_mode=advanced_mode,
            progress_bar=progress_bar,
            per_min_limit=PER_MIN_LIMIT
        )
        
        progress_bar.progress(1.0, text="✅ Complete!")
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
        
        # Calculate network metrics (only in advanced mode)
        network_metrics = None
        if advanced_mode and len(edges) > 0:
            with st.spinner("📊 Calculating network metrics..."):
                network_metrics = calculate_network_metrics(seen_profiles, edges)
        
        # Store results in session state so they persist across reruns (e.g., after downloads)
        st.session_state.crawl_results = {
            'seen_profiles': seen_profiles,
            'edges': edges,
            'raw_profiles': raw_profiles,
            'stats': stats,
            'max_degree': max_degree,
            'advanced_mode': advanced_mode,  # Store mode setting
            'network_metrics': network_metrics  # Store network metrics
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
        network_metrics = results.get('network_metrics', None)
        
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
        elif stats['stopped_reason'] == 'out_of_credits':
            col8.error("❌ Out of Credits")
        
        # Show error breakdown if there were failures
        if stats['failed_calls'] > 0:
            st.markdown("---")
            st.subheader("❌ Error Breakdown")
            
            error_breakdown = stats.get('error_breakdown', {})
            if error_breakdown:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                if error_breakdown.get('rate_limit', 0) > 0:
                    col1.metric("Rate Limits", error_breakdown['rate_limit'], delta_color="off")
                if error_breakdown.get('out_of_credits', 0) > 0:
                    col2.metric("Out of Credits", error_breakdown['out_of_credits'], delta_color="off")
                if error_breakdown.get('auth_error', 0) > 0:
                    col3.metric("Auth Errors", error_breakdown['auth_error'], delta_color="off")
                if error_breakdown.get('not_found', 0) > 0:
                    col4.metric("Not Found", error_breakdown['not_found'], delta_color="off")
                if error_breakdown.get('enrichment_failed', 0) > 0:
                    col5.metric("Enrichment Failed", error_breakdown['enrichment_failed'], delta_color="off")
                if error_breakdown.get('other', 0) > 0:
                    col6.metric("Other Errors", error_breakdown['other'], delta_color="off")
                
                # Interpretation
                if error_breakdown.get('rate_limit', 0) > stats['failed_calls'] * 0.5:
                    st.warning("""
                    **⚠️ High Rate Limit Failures**
                    
                    Most failures were due to rate limiting. This suggests:
                    - Your crawl exceeded EnrichLayer's rate limits
                    - Consider using Degree 1 instead of Degree 2
                    - Space out large crawls over time
                    - Check your EnrichLayer plan limits
                    """)
                
                if error_breakdown.get('out_of_credits', 0) > 0:
                    st.error("""
                    **🚫 Out of Credits**
                    
                    You've exhausted your EnrichLayer credits. To continue:
                    1. Check your credit balance at EnrichLayer dashboard
                    2. Purchase more credits if needed
                    3. Resume your crawl
                    """)
            
            # ================================================================
            # DETAILED ERROR LOG (C4C Internal Use)
            # ================================================================
            with st.expander("🔧 Technical Details (C4C Internal)", expanded=False):
                st.markdown("### API Call Statistics")
                st.code(f"""
Total API Calls Attempted: {stats['api_calls']}
Successful Calls: {stats['successful_calls']}
Failed Calls: {stats['failed_calls']}
Success Rate: {(stats['successful_calls'] / max(stats['api_calls'], 1) * 100):.1f}%

Error Breakdown:
- Rate Limit (429): {error_breakdown.get('rate_limit', 0)}
- Out of Credits (403): {error_breakdown.get('out_of_credits', 0)}
- Auth Errors (401): {error_breakdown.get('auth_error', 0)}
- Not Found (404): {error_breakdown.get('not_found', 0)}
- Enrichment Failed (503): {error_breakdown.get('enrichment_failed', 0)}
- Other Errors: {error_breakdown.get('other', 0)}

Crawl Configuration:
- Max Degree: {max_degree}
- Max Edges Limit: 5000
- Max Nodes Limit: 2500
- API Delay: {API_DELAY} seconds between calls
- Stopped Reason: {stats.get('stopped_reason', 'unknown')}
                """, language="text")
                
                # Rate limit diagnosis
                if error_breakdown.get('rate_limit', 0) > 10:
                    st.markdown("### 🔍 Rate Limit Diagnosis")
                    st.warning(f"""
                    **High rate limit failures detected ({error_breakdown.get('rate_limit', 0)} errors)**
                    
                    **Possible causes:**
                    1. EnrichLayer has strict per-minute limits (likely 60/min or less)
                    2. Degree 2 crawls make too many rapid requests
                    3. Your plan tier may have lower limits
                    
                    **Current settings:**
                    - API Delay: {API_DELAY} seconds (should give ~{60/API_DELAY:.0f} calls/min)
                    - This run attempted: {stats['api_calls']} calls
                    
                    **Recommendations:**
                    - Use Degree 1 for most crawls
                    - Contact EnrichLayer support about rate limits
                    - Consider increasing API_DELAY to 3-4 seconds
                    """)
                
                st.markdown("### 📊 Network Statistics")
                st.code(f"""
Total Nodes Discovered: {len(seen_profiles)}
Total Edges Discovered: {len(edges)}
Nodes with Organization Data: {sum(1 for n in seen_profiles.values() if n.get('organization'))}
Max Degree Reached: {stats.get('max_degree_reached', 0)}
Profiles With No Neighbors: {stats.get('profiles_with_no_neighbors', 0)}
                """, language="text")

        
        # ====================================================================
        # ADVANCED ANALYTICS (if advanced mode was enabled)
        # ====================================================================
        
        if was_advanced_mode:
            st.markdown("---")
            st.header("🔬 Advanced Network Analytics")
            
            # Check if organization data is available
            has_org_data = any('organization' in node for node in seen_profiles.values())
            
            if has_org_data:
                st.success("✅ **Organization Extraction Active** - Enhanced data available")
                
                # Extract organization statistics
                orgs = {}
                sectors = {}
                for node in seen_profiles.values():
                    org = node.get('organization', '')
                    sector = node.get('sector', 'Unknown')
                    
                    if org:
                        orgs[org] = orgs.get(org, 0) + 1
                    if sector:
                        sectors[sector] = sectors.get(sector, 0) + 1
                
                # Display organization breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏢 Organizations Represented")
                    
                    if orgs:
                        # Sort by count
                        sorted_orgs = sorted(orgs.items(), key=lambda x: x[1], reverse=True)
                        
                        st.metric("Unique Organizations", len(orgs))
                        
                        # Show top 10
                        st.markdown("**Top Organizations:**")
                        for org, count in sorted_orgs[:10]:
                            st.markdown(f"- **{org}**: {count} {'person' if count == 1 else 'people'}")
                        
                        if len(sorted_orgs) > 10:
                            st.caption(f"...and {len(sorted_orgs) - 10} more organizations")
                    else:
                        st.info("No organization data extracted from profiles")
                
                with col2:
                    st.subheader("🎯 Sector Distribution")
                    
                    if sectors:
                        # Sort by count
                        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
                        
                        st.metric("Sectors Identified", len(sectors))
                        
                        st.markdown("**Sector Breakdown:**")
                        for sector, count in sorted_sectors:
                            percentage = (count / len(seen_profiles)) * 100
                            st.markdown(f"- **{sector}**: {count} ({percentage:.1f}%)")
                    else:
                        st.info("No sector classification available")
                
                # Note about what this enables
                st.markdown("---")
                st.info("""
                **🎯 What Organization Data Enables:**
                
                With organization and sector information, you can now:
                - Identify cross-sector brokers
                - Detect organizational silos
                - Find inter-organizational bridges
                - Map influence across sectors
                
                **Coming Next:** Brokerage matrix showing who connects which organizations/sectors.
                """)
            
            else:
                st.warning("""
                **⚠️ Organization Data Not Available**
                
                Organization extraction requires full profile data from EnrichLayer API responses.
                This data may not be available for:
                - Discovered nodes that weren't fully fetched
                - Profiles with incomplete data
                - Mock mode tests
                
                **Tip:** Run with real API token and degree 1 or 2 to get organization data for fetched profiles.
                """)
            
            # ================================================================
            # NETWORK CENTRALITY METRICS (Advanced Mode)
            # ================================================================
            st.markdown("---")
            st.subheader("📊 Network Centrality Metrics")
            
            if network_metrics and network_metrics.get('top_nodes'):
                top_nodes = network_metrics['top_nodes']
                network_stats = network_metrics.get('network_stats', {})
                
                # Network-level statistics
                st.markdown("**Network Overview:**")
                stats_cols = st.columns(5)
                stats_cols[0].metric("Nodes", network_stats.get('nodes', 0))
                stats_cols[1].metric("Edges", network_stats.get('edges', 0))
                stats_cols[2].metric("Density", f"{network_stats.get('density', 0):.4f}")
                stats_cols[3].metric("Avg Degree", network_stats.get('avg_degree', 0))
                stats_cols[4].metric("Avg Clustering", f"{network_stats.get('avg_clustering', 0):.4f}")
                
                # Additional network stats if available
                if 'num_components' in network_stats:
                    st.caption(f"📈 Components: {network_stats['num_components']} | Largest component: {network_stats.get('largest_component_size', 'N/A')} nodes")
                
                st.markdown("---")
                
                # Top nodes by different centrality measures
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔗 Top Connectors** (by Degree)")
                    st.caption("Most direct connections")
                    if 'degree' in top_nodes:
                        for i, (node_id, score) in enumerate(top_nodes['degree'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            connections = network_metrics['node_metrics'].get(node_id, {}).get('degree', 0)
                            org_text = f" ({org})" if org else ""
                            st.markdown(f"{i}. **{name}**{org_text} — {connections} connections")
                    else:
                        st.info("No degree data available")
                    
                    st.markdown("")
                    st.markdown("**📍 Top Accessible** (by Closeness)")
                    st.caption("Shortest average distance to others")
                    if 'closeness' in top_nodes:
                        for i, (node_id, score) in enumerate(top_nodes['closeness'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            org_text = f" ({org})" if org else ""
                            st.markdown(f"{i}. **{name}**{org_text} — {score:.4f}")
                    else:
                        st.info("No closeness data available")
                
                with col2:
                    st.markdown("**🌉 Top Brokers** (by Betweenness)")
                    st.caption("Bridge between groups")
                    if 'betweenness' in top_nodes:
                        for i, (node_id, score) in enumerate(top_nodes['betweenness'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            org_text = f" ({org})" if org else ""
                            st.markdown(f"{i}. **{name}**{org_text} — {score:.4f}")
                    else:
                        st.info("No betweenness data available")
                    
                    st.markdown("")
                    st.markdown("**⭐ Top Influencers** (by Eigenvector)")
                    st.caption("Connected to well-connected people")
                    if 'eigenvector' in top_nodes:
                        for i, (node_id, score) in enumerate(top_nodes['eigenvector'][:5], 1):
                            name = seen_profiles.get(node_id, {}).get('name', node_id)
                            org = seen_profiles.get(node_id, {}).get('organization', '')
                            org_text = f" ({org})" if org else ""
                            st.markdown(f"{i}. **{name}**{org_text} — {score:.4f}")
                    else:
                        st.info("No eigenvector data available")
                
                # Metric definitions
                with st.expander("ℹ️ What do these metrics mean?"):
                    st.markdown("""
                    | Metric | What It Measures | Identifies |
                    |--------|------------------|------------|
                    | **Degree Centrality** | Number of direct connections | **Connectors** — well-networked individuals |
                    | **Betweenness Centrality** | How often on shortest paths between others | **Brokers** — bridge different groups |
                    | **Eigenvector Centrality** | Connected to influential people | **Influencers** — access to power |
                    | **Closeness Centrality** | Average distance to everyone | **Accessible hubs** — can reach anyone quickly |
                    
                    **Network-level metrics:**
                    - **Density**: Proportion of possible connections that exist (0-1)
                    - **Clustering**: How much nodes cluster together
                    - **Components**: Disconnected subgroups in the network
                    """)
            else:
                st.info("Network metrics require edges to calculate. Run a crawl with connections to see centrality analysis.")
            
            # Roadmap for future features
            st.markdown("---")
            st.subheader("📋 Coming Soon")
            
            st.markdown("""
            **Community Detection** (In Progress)  
            - Algorithmic cluster identification
            - Modularity scores
            
            **Brokerage Matrix** (Planned)
            - Coordinators, gatekeepers, representatives, liaisons
            - Structural hole analysis
            - Cross-sector bridge identification
            
            **Strategic Insights** (Future)
            - AI-generated narrative analysis
            - Gap identification
            - Collaboration opportunities
            """)
        
        # ====================================================================
        # DOWNLOAD SECTION
        # ====================================================================
        
        st.header("💾 Download Results")
        
        # Generate files
        nodes_csv = generate_nodes_csv(seen_profiles, max_degree=max_degree, max_edges=5000, max_nodes=2500, network_metrics=network_metrics)
        edges_csv = generate_edges_csv(edges, max_degree=max_degree, max_edges=5000, max_nodes=2500)
        raw_json = generate_raw_json(raw_profiles)
        
        # Generate network analysis JSON if metrics available
        analysis_json = None
        if network_metrics:
            analysis_json = generate_network_analysis_json(network_metrics, seen_profiles)
        
        # Primary action: Download all as ZIP
        st.markdown("### 📦 Download All Files")
        zip_data = create_download_zip(nodes_csv, edges_csv, raw_json, analysis_json)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.download_button(
                label="⬇️ Download All as ZIP (Recommended)",
                data=zip_data,
                file_name="c4c_network_crawl.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                help="Download all files (nodes.csv, edges.csv, raw_profiles.json, network_analysis.json) in one ZIP file"
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
