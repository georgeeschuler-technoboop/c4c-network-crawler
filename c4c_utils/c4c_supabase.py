"""
C4C Graph Platform — Supabase Integration

Shared module for OrgGraph US, OrgGraph CA, InsightGraph, ActorGraph.
Handles authentication, project CRUD, and data sync.

Usage:
    from c4c_supabase import C4CSupabase
    
    db = C4CSupabase()
    if db.login(email, password):
        projects = db.list_projects()
        db.save_project(name, nodes_df, edges_df, ...)

VERSION HISTORY:
----------------
v1.0.0 (2025-12-21): Initial implementation
  - Auth: login, logout, signup, get_user
  - Projects: create, list, get, update, delete
  - Data: save_nodes, save_edges, save_grants_detail
  - Artifacts: save_artifact, get_artifact
  - Sharing: add_member, remove_member, list_members

SETUP:
------
1. pip install supabase
2. Set environment variables:
   - SUPABASE_URL=https://your-project.supabase.co
   - SUPABASE_KEY=your-anon-key

Or pass directly to C4CSupabase(url=..., key=...)
"""

import os
import json
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase package not installed. Run: pip install supabase")


class C4CSupabase:
    """
    Supabase client wrapper for C4C Graph Platform.
    
    Handles authentication and all database operations.
    """
    
    def __init__(self, url: str = None, key: str = None):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL (or set SUPABASE_URL env var)
            key: Supabase anon key (or set SUPABASE_KEY env var)
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase package not installed. Run: pip install supabase")
        
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key required. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables, "
                "or pass url= and key= to C4CSupabase()"
            )
        
        self.client: Client = create_client(self.url, self.key)
        self._user = None
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def login(self, email: str, password: str) -> bool:
        """
        Log in with email and password.
        
        Returns True on success, False on failure.
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            self._user = response.user
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    
    def signup(self, email: str, password: str) -> bool:
        """
        Create new account with email and password.
        
        Returns True on success, False on failure.
        Note: May require email confirmation depending on Supabase settings.
        """
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            self._user = response.user
            return True
        except Exception as e:
            print(f"Signup failed: {e}")
            return False
    
    def logout(self):
        """Log out current user."""
        try:
            self.client.auth.sign_out()
            self._user = None
        except Exception:
            pass
    
    def get_user(self) -> Optional[Dict]:
        """Get current authenticated user, or None if not logged in."""
        try:
            response = self.client.auth.get_user()
            self._user = response.user
            return {
                "id": self._user.id,
                "email": self._user.email,
                "created_at": self._user.created_at
            } if self._user else None
        except Exception:
            return None
    
    @property
    def user_id(self) -> Optional[str]:
        """Get current user's ID."""
        user = self.get_user()
        return user["id"] if user else None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is logged in."""
        return self.get_user() is not None
    
    # =========================================================================
    # Projects
    # =========================================================================
    
    def create_project(
        self,
        name: str,
        slug: str,
        source_app: str,
        description: str = None,
        config: Dict = None
    ) -> Optional[Dict]:
        """
        Create a new project.
        
        Args:
            name: Display name (e.g., "Great Lakes Funders Network")
            slug: URL-safe identifier (e.g., "glfn-2025")
            source_app: One of 'orggraph_us', 'orggraph_ca', 'insightgraph', 'actorgraph'
            description: Optional description
            config: Optional config dict (region_lens, settings, etc.)
        
        Returns:
            Created project dict, or None on failure.
        """
        uid = self.user_id
        if not uid:
            self._last_error = "Not authenticated (user_id is None)"
            return None
        
        try:
            data = {
                "user_id": uid,
                "name": name,
                "slug": slug,
                "source_app": source_app,
                "description": description,
                "config": config or {}
            }
            print(f"DEBUG: Inserting project with user_id={uid}, slug={slug}")
            
            response = self.client.table("projects").insert(data).execute()
            
            print(f"DEBUG: Response data={response.data}")
            self._last_error = None
            return response.data[0] if response.data else None
        except Exception as e:
            self._last_error = str(e)
            print(f"Error creating project: {e}")
            return None
    
    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return getattr(self, '_last_error', None)
    
    def list_projects(self, source_app: str = None) -> List[Dict]:
        """
        List all projects accessible to current user.
        
        Args:
            source_app: Optional filter by source app
        
        Returns:
            List of project dicts.
        """
        if not self.user_id:
            return []
        
        try:
            query = self.client.table("projects").select("*")
            
            if source_app:
                query = query.eq("source_app", source_app)
            
            response = query.order("updated_at", desc=True).execute()
            return response.data or []
        except Exception as e:
            print(f"Error listing projects: {e}")
            return []
    
    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get a project by ID."""
        try:
            response = self.client.table("projects").select("*").eq("id", project_id).single().execute()
            return response.data
        except Exception as e:
            print(f"Error getting project: {e}")
            return None
    
    def get_project_by_slug(self, slug: str) -> Optional[Dict]:
        """Get a project by slug (for current user)."""
        if not self.user_id:
            return None
        
        try:
            response = self.client.table("projects").select("*").eq("slug", slug).single().execute()
            return response.data
        except Exception:
            return None
    
    def update_project(self, project_id: str, **updates) -> Optional[Dict]:
        """
        Update project fields.
        
        Args:
            project_id: Project UUID
            **updates: Fields to update (name, description, config, node_count, etc.)
        
        Returns:
            Updated project dict, or None on failure.
        """
        try:
            response = self.client.table("projects").update(updates).eq("id", project_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error updating project: {e}")
            return None
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and all its data.
        
        Returns True on success.
        """
        try:
            self.client.table("projects").delete().eq("id", project_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting project: {e}")
            return False
    
    # =========================================================================
    # Nodes
    # =========================================================================
    
    def save_nodes(self, project_id: str, nodes_df: pd.DataFrame) -> int:
        """
        Save nodes DataFrame to Supabase.
        
        Upserts based on (project_id, node_id).
        
        Args:
            project_id: Project UUID
            nodes_df: DataFrame with columns: node_id, node_type, label, ...
        
        Returns:
            Number of rows saved.
        """
        if nodes_df.empty:
            return 0
        
        # Required columns
        required = {"node_id", "node_type", "label"}
        if not required.issubset(nodes_df.columns):
            raise ValueError(f"nodes_df missing required columns: {required - set(nodes_df.columns)}")
        
        # Build records
        records = []
        attr_cols = [c for c in nodes_df.columns if c not in {"node_id", "node_type", "label", "network_role_code"}]
        
        for _, row in nodes_df.iterrows():
            record = {
                "project_id": project_id,
                "node_id": row["node_id"],
                "node_type": row["node_type"],
                "label": row["label"],
                "network_role_code": row.get("network_role_code"),
                "attributes": {col: _clean_value(row.get(col)) for col in attr_cols if pd.notna(row.get(col))}
            }
            records.append(record)
        
        # Upsert in batches
        return self._upsert_batch("nodes", records, ["project_id", "node_id"])
    
    def get_nodes(self, project_id: str) -> pd.DataFrame:
        """Get all nodes for a project as DataFrame."""
        try:
            response = self.client.table("nodes").select("*").eq("project_id", project_id).execute()
            if not response.data:
                return pd.DataFrame()
            
            # Flatten attributes
            records = []
            for row in response.data:
                record = {
                    "node_id": row["node_id"],
                    "node_type": row["node_type"],
                    "label": row["label"],
                    "network_role_code": row.get("network_role_code"),
                }
                record.update(row.get("attributes", {}))
                records.append(record)
            
            return pd.DataFrame(records)
        except Exception as e:
            print(f"Error getting nodes: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Edges
    # =========================================================================
    
    def save_edges(self, project_id: str, edges_df: pd.DataFrame) -> int:
        """
        Save edges DataFrame to Supabase.
        
        Args:
            project_id: Project UUID
            edges_df: DataFrame with columns: edge_id, edge_type, from_id, to_id, ...
        
        Returns:
            Number of rows saved.
        """
        if edges_df.empty:
            return 0
        
        required = {"edge_id", "edge_type", "from_id", "to_id"}
        if not required.issubset(edges_df.columns):
            raise ValueError(f"edges_df missing required columns: {required - set(edges_df.columns)}")
        
        records = []
        attr_cols = [c for c in edges_df.columns if c not in required]
        
        for _, row in edges_df.iterrows():
            record = {
                "project_id": project_id,
                "edge_id": row["edge_id"],
                "edge_type": row["edge_type"],
                "from_id": row["from_id"],
                "to_id": row["to_id"],
                "attributes": {col: _clean_value(row.get(col)) for col in attr_cols if pd.notna(row.get(col))}
            }
            records.append(record)
        
        return self._upsert_batch("edges", records, ["project_id", "edge_id"])
    
    def get_edges(self, project_id: str) -> pd.DataFrame:
        """Get all edges for a project as DataFrame."""
        try:
            response = self.client.table("edges").select("*").eq("project_id", project_id).execute()
            if not response.data:
                return pd.DataFrame()
            
            records = []
            for row in response.data:
                record = {
                    "edge_id": row["edge_id"],
                    "edge_type": row["edge_type"],
                    "from_id": row["from_id"],
                    "to_id": row["to_id"],
                }
                record.update(row.get("attributes", {}))
                records.append(record)
            
            return pd.DataFrame(records)
        except Exception as e:
            print(f"Error getting edges: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Grants Detail
    # =========================================================================
    
    def save_grants_detail(self, project_id: str, grants_df: pd.DataFrame) -> int:
        """
        Save grants_detail DataFrame to Supabase.
        
        Args:
            project_id: Project UUID
            grants_df: DataFrame with grant details
        
        Returns:
            Number of rows saved.
        """
        if grants_df is None:
            print("DEBUG save_grants_detail: grants_df is None")
            return 0
        
        if grants_df.empty:
            print("DEBUG save_grants_detail: grants_df is empty")
            return 0
        
        print(f"DEBUG save_grants_detail: {len(grants_df)} rows, columns: {list(grants_df.columns)}")
        
        # Column aliases (OrgGraph exports → Supabase schema)
        # Use alias value if target column is missing OR empty
        column_aliases = {
            "foundation_ein": "funder_id",
            "foundation_name": "funder_name",
            "grant_amount": "amount",
            "grant_purpose_raw": "grant_purpose",
            "tax_year": "fiscal_year",
        }
        
        # Apply aliases - overwrite if target is missing or all null
        df = grants_df.copy()
        for old_col, new_col in column_aliases.items():
            if old_col in df.columns:
                # Check if target column is missing or all empty/null
                target_empty = new_col not in df.columns or df[new_col].isna().all() or (df[new_col].astype(str).str.strip() == '').all()
                if target_empty:
                    df[new_col] = df[old_col]
                    print(f"DEBUG: Aliased {old_col} → {new_col}")
        
        # Generate grant_id if not present
        if "grant_id" not in df.columns:
            # Create composite key from funder + grantee + amount + year + row index
            # Row index ensures uniqueness even for duplicate grants
            def make_grant_id(idx, row):
                funder = str(row.get('funder_id', '') or '')
                grantee = str(row.get('grantee_name', '') or '')[:30]
                amt = str(row.get('amount', '') or '')
                year = str(row.get('fiscal_year', '') or '')
                return f"{funder}_{grantee}_{amt}_{year}_{idx}".replace(" ", "_")[:100]
            
            df["grant_id"] = [make_grant_id(i, row) for i, row in df.iterrows()]
            print(f"DEBUG: Generated grant_id with row index, sample: {df['grant_id'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        # Core columns that map to Supabase schema
        core_cols = {
            "grant_id", "funder_id", "funder_name", "grantee_id", "grantee_name",
            "amount", "fiscal_year", "grant_purpose",
            "grantee_city", "grantee_state", "grantee_country",
            "source_system", "grant_bucket", "region_relevant"
        }
        
        records = []
        attr_cols = [c for c in df.columns if c not in core_cols and c not in column_aliases.keys()]
        
        for _, row in df.iterrows():
            record = {"project_id": project_id}
            
            # Map core columns
            for col in core_cols:
                if col in df.columns and pd.notna(row.get(col)):
                    record[col] = _clean_value(row[col])
            
            # Everything else goes to attributes
            attrs = {
                col: _clean_value(row.get(col)) 
                for col in attr_cols 
                if pd.notna(row.get(col))
            }
            if attrs:
                record["attributes"] = attrs
            
            records.append(record)
        
        print(f"DEBUG: Built {len(records)} records, sample keys: {list(records[0].keys()) if records else 'N/A'}")
        
        result = self._upsert_batch("grants_detail", records, ["project_id", "grant_id"])
        print(f"DEBUG: Upserted {result} grants_detail rows")
        return result
    
    def get_grants_detail(self, project_id: str) -> pd.DataFrame:
        """Get all grants_detail for a project as DataFrame."""
        try:
            response = self.client.table("grants_detail").select("*").eq("project_id", project_id).execute()
            if not response.data:
                return pd.DataFrame()
            
            records = []
            for row in response.data:
                record = {k: v for k, v in row.items() if k not in {"id", "project_id", "attributes", "created_at"}}
                record.update(row.get("attributes", {}))
                records.append(record)
            
            return pd.DataFrame(records)
        except Exception as e:
            print(f"Error getting grants_detail: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Artifacts
    # =========================================================================
    
    def save_artifact(
        self,
        project_id: str,
        artifact_type: str,
        content: Any,
        generator_version: str = None
    ) -> bool:
        """
        Save an artifact (report, summary, etc.).
        
        Args:
            project_id: Project UUID
            artifact_type: One of 'project_summary', 'insight_cards', 'insight_report', 'node_metrics', 'parse_log'
            content: Dict/list for JSON types, string for text types
            generator_version: Optional version string (e.g., 'run.py v3.0.5')
        
        Returns:
            True on success.
        """
        try:
            record = {
                "project_id": project_id,
                "artifact_type": artifact_type,
                "generator_version": generator_version
            }
            
            if artifact_type == "insight_report":
                record["content_text"] = content
            else:
                record["content_json"] = content
            
            self.client.table("artifacts").upsert(
                record,
                on_conflict="project_id,artifact_type"
            ).execute()
            
            return True
        except Exception as e:
            print(f"Error saving artifact: {e}")
            return False
    
    def get_artifact(self, project_id: str, artifact_type: str) -> Any:
        """
        Get an artifact by type.
        
        Returns:
            Content (dict/list for JSON, string for text), or None if not found.
        """
        try:
            response = self.client.table("artifacts").select("*").eq("project_id", project_id).eq("artifact_type", artifact_type).single().execute()
            
            if not response.data:
                return None
            
            if artifact_type == "insight_report":
                return response.data.get("content_text")
            else:
                return response.data.get("content_json")
        except Exception:
            return None
    
    # =========================================================================
    # Project Members (Sharing)
    # =========================================================================
    
    def add_member(self, project_id: str, user_email: str, role: str = "viewer") -> bool:
        """
        Add a member to a project by email.
        
        Args:
            project_id: Project UUID
            user_email: Email of user to add
            role: 'viewer' or 'editor'
        
        Returns:
            True on success.
        """
        # First, find user by email (requires admin or custom function)
        # For now, we'll need the user_id directly
        print("Note: add_member by email requires looking up user ID. Consider using add_member_by_id().")
        return False
    
    def add_member_by_id(self, project_id: str, user_id: str, role: str = "viewer") -> bool:
        """
        Add a member to a project by user ID.
        
        Args:
            project_id: Project UUID
            user_id: User UUID
            role: 'viewer' or 'editor'
        
        Returns:
            True on success.
        """
        try:
            self.client.table("project_members").upsert({
                "project_id": project_id,
                "user_id": user_id,
                "role": role
            }, on_conflict="project_id,user_id").execute()
            return True
        except Exception as e:
            print(f"Error adding member: {e}")
            return False
    
    def remove_member(self, project_id: str, user_id: str) -> bool:
        """Remove a member from a project."""
        try:
            self.client.table("project_members").delete().eq("project_id", project_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            print(f"Error removing member: {e}")
            return False
    
    def list_members(self, project_id: str) -> List[Dict]:
        """List all members of a project."""
        try:
            response = self.client.table("project_members").select("*").eq("project_id", project_id).execute()
            return response.data or []
        except Exception as e:
            print(f"Error listing members: {e}")
            return []
    
    # =========================================================================
    # Bulk Save (Convenience)
    # =========================================================================
    
    def save_project_data(
        self,
        project_id: str,
        nodes_df: pd.DataFrame = None,
        edges_df: pd.DataFrame = None,
        grants_df: pd.DataFrame = None,
        project_summary: Dict = None,
        insight_cards: Dict = None,
        insight_report: str = None,
        node_metrics: Dict = None
    ) -> Dict[str, int]:
        """
        Save all project data in one call.
        
        Returns:
            Dict with counts of saved items.
        """
        results = {}
        
        if nodes_df is not None:
            results["nodes"] = self.save_nodes(project_id, nodes_df)
        
        if edges_df is not None:
            results["edges"] = self.save_edges(project_id, edges_df)
        
        print(f"DEBUG save_project_data: grants_df is None={grants_df is None}, empty={grants_df.empty if grants_df is not None else 'N/A'}")
        if grants_df is not None:
            print(f"DEBUG save_project_data: calling save_grants_detail with {len(grants_df)} rows")
            results["grants"] = self.save_grants_detail(project_id, grants_df)
            print(f"DEBUG save_project_data: save_grants_detail returned {results.get('grants', 'MISSING')}")
        
        if project_summary is not None:
            self.save_artifact(project_id, "project_summary", project_summary)
            results["project_summary"] = 1
        
        if insight_cards is not None:
            self.save_artifact(project_id, "insight_cards", insight_cards)
            results["insight_cards"] = 1
        
        if insight_report is not None:
            self.save_artifact(project_id, "insight_report", insight_report)
            results["insight_report"] = 1
        
        if node_metrics is not None:
            self.save_artifact(project_id, "node_metrics", node_metrics)
            results["node_metrics"] = 1
        
        # Update project stats
        self.update_project(
            project_id,
            node_count=results.get("nodes", 0),
            edge_count=results.get("edges", 0)
        )
        
        return results
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _upsert_batch(self, table: str, records: List[Dict], conflict_cols: List[str], batch_size: int = 500) -> int:
        """Upsert records in batches."""
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                response = self.client.table(table).upsert(
                    batch,
                    on_conflict=",".join(conflict_cols)
                ).execute()
                total += len(batch)
                print(f"DEBUG _upsert_batch: {table} batch {i//batch_size + 1}, {len(batch)} records OK")
            except Exception as e:
                print(f"ERROR _upsert_batch {table} batch {i//batch_size + 1}: {e}")
                # Print sample record for debugging
                if batch:
                    print(f"DEBUG: Sample record keys: {list(batch[0].keys())}")
        return total


# =============================================================================
# Helpers
# =============================================================================

def _clean_value(val):
    """Clean a value for JSON serialization."""
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, (int, float, str, bool, list, dict)):
        return val
    return str(val)


# =============================================================================
# Streamlit Session Helpers
# =============================================================================

def init_supabase_session(st_session_state) -> Optional[C4CSupabase]:
    """
    Initialize Supabase client in Streamlit session state.
    
    Usage:
        db = init_supabase_session(st.session_state)
        if db and db.is_authenticated:
            projects = db.list_projects()
    """
    if "supabase" not in st_session_state:
        try:
            st_session_state.supabase = C4CSupabase()
        except Exception as e:
            print(f"Could not initialize Supabase: {e}")
            st_session_state.supabase = None
    
    return st_session_state.supabase


def render_login_form(st, db: C4CSupabase) -> bool:
    """
    Render a simple login/signup form in Streamlit.
    
    Returns True if user is authenticated.
    
    Usage:
        if not render_login_form(st, db):
            st.stop()
    """
    if db.is_authenticated:
        user = db.get_user()
        st.sidebar.success(f"Logged in as {user['email']}")
        if st.sidebar.button("Logout"):
            db.logout()
            st.rerun()
        return True
    
    st.sidebar.subheader("Login")
    
    tab1, tab2 = st.sidebar.tabs(["Login", "Sign Up"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if db.login(email, password):
                st.rerun()
            else:
                st.error("Login failed. Check your credentials.")
    
    with tab2:
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            if db.signup(email, password):
                st.success("Account created! Check your email to confirm.")
            else:
                st.error("Signup failed.")
    
    return False
