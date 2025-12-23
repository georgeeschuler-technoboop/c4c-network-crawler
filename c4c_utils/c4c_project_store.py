"""
C4C Project Store Client — Supabase Integration

This module provides a unified interface for saving and loading network analysis
projects to/from the C4C cloud storage (Supabase).

Used by: OrgGraph US, OrgGraph CA, ActorGraph, InsightGraph

VERSION HISTORY:
----------------
v1.0.0 (2025-12-22): Initial implementation
  - ProjectStoreClient class for CRUD operations
  - save_project() for uploading bundles
  - load_project() for downloading bundles
  - list_projects() for browsing available projects
  - Slug generation and validation

USAGE:
------
    from c4c_project_store import ProjectStoreClient
    
    # Initialize client
    client = ProjectStoreClient(supabase_url, supabase_key)
    
    # Login user
    client.login(email, password)
    
    # Save a project
    project = client.save_project(
        name="Great Lakes Funders Network",
        bundle_data=zip_bytes,
        source_app="orggraph_us",
        node_count=2781,
        edge_count=3314,
        jurisdiction="US",
        region_preset="great_lakes"
    )
    
    # List user's projects
    projects = client.list_projects()
    
    # Load a project
    bundle_data = client.load_project(project_id="...")
"""

import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from io import BytesIO


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Project:
    """Represents a project in the store."""
    id: str
    slug: str
    name: str
    description: Optional[str]
    source_app: str
    schema_version: str
    bundle_version: str
    app_version: Optional[str]
    node_count: int
    edge_count: int
    jurisdiction: Optional[str]
    region_preset: Optional[str]
    bundle_path: str
    bundle_size_bytes: Optional[int]
    created_at: str
    updated_at: str
    created_by: str
    is_public: bool
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Project':
        """Create Project from dictionary."""
        return cls(
            id=data.get('id', ''),
            slug=data.get('slug', ''),
            name=data.get('name', ''),
            description=data.get('description'),
            source_app=data.get('source_app', ''),
            schema_version=data.get('schema_version', 'c4c_coregraph_v1'),
            bundle_version=data.get('bundle_version', '1.0'),
            app_version=data.get('app_version'),
            node_count=data.get('node_count', 0),
            edge_count=data.get('edge_count', 0),
            jurisdiction=data.get('jurisdiction'),
            region_preset=data.get('region_preset'),
            bundle_path=data.get('bundle_path', ''),
            bundle_size_bytes=data.get('bundle_size_bytes'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            created_by=data.get('user_id', ''),  # DB column is user_id
            is_public=data.get('is_public', False),
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProjectSummary:
    """Lightweight project info for list views."""
    id: str
    slug: str
    name: str
    source_app: str
    node_count: int
    edge_count: int
    jurisdiction: Optional[str]
    updated_at: str
    is_public: bool
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProjectSummary':
        return cls(
            id=data.get('id', ''),
            slug=data.get('slug', ''),
            name=data.get('name', ''),
            source_app=data.get('source_app', ''),
            node_count=data.get('node_count', 0),
            edge_count=data.get('edge_count', 0),
            jurisdiction=data.get('jurisdiction'),
            updated_at=data.get('updated_at', ''),
            is_public=data.get('is_public', False),
        )


# =============================================================================
# Slug Generation
# =============================================================================

def generate_slug(name: str) -> str:
    """
    Generate a URL-friendly slug from a project name.
    
    Examples:
        "Great Lakes Funders Network" → "great-lakes-funders-network-20251222"
        "Ontario Foundations" → "ontario-foundations-20251222"
    """
    # Convert to lowercase
    slug = name.lower()
    
    # Replace non-alphanumeric with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    
    # Truncate to max 50 chars
    slug = slug[:50]
    
    # Add date suffix
    date_suffix = datetime.now().strftime('%Y%m%d')
    slug = f"{slug}-{date_suffix}"
    
    return slug


def validate_slug(slug: str) -> bool:
    """Validate that a slug is well-formed."""
    if not slug:
        return False
    if len(slug) > 100:
        return False
    if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', slug):
        return False
    return True


# =============================================================================
# Project Store Client
# =============================================================================

class ProjectStoreClient:
    """
    Client for C4C Project Store (Supabase).
    
    Provides methods for saving, loading, and listing network analysis projects.
    """
    
    BUCKET_NAME = 'project_bundles'
    TABLE_NAME = 'projects'
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the client.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/public key
        """
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(supabase_url, supabase_key)
        except ImportError:
            raise ImportError("supabase-py not installed. Run: pip install supabase")
        
        self.user_id: Optional[str] = None
        self.user_email: Optional[str] = None
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def login(self, email: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Login with email and password.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if response.user:
                self.user_id = response.user.id
                self.user_email = response.user.email
                return True, None
            return False, "Login failed"
        except Exception as e:
            return False, str(e)
    
    def signup(self, email: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Create a new account.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            if response.user:
                self.user_id = response.user.id
                self.user_email = response.user.email
                return True, None
            return False, "Signup failed"
        except Exception as e:
            return False, str(e)
    
    def logout(self):
        """Logout current user."""
        try:
            self.client.auth.sign_out()
        except:
            pass
        self.user_id = None
        self.user_email = None
    
    def is_authenticated(self) -> bool:
        """Check if user is logged in."""
        return self.user_id is not None
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user info."""
        if not self.user_id:
            return None
        return {
            'id': self.user_id,
            'email': self.user_email
        }
    
    # =========================================================================
    # Project Operations
    # =========================================================================
    
    def save_project(
        self,
        name: str,
        bundle_data: bytes,
        source_app: str,
        node_count: int,
        edge_count: int,
        jurisdiction: Optional[str] = None,
        region_preset: Optional[str] = None,
        description: Optional[str] = None,
        app_version: Optional[str] = None,
        schema_version: str = 'c4c_coregraph_v1',
        bundle_version: str = '1.0',
        slug: Optional[str] = None,
        overwrite_slug: Optional[str] = None,
    ) -> Tuple[Optional[Project], Optional[str]]:
        """
        Save a project bundle to cloud storage.
        
        Args:
            name: Human-readable project name
            bundle_data: ZIP file bytes
            source_app: Source application ('orggraph_us', 'orggraph_ca', 'actorgraph')
            node_count: Number of nodes in the network
            edge_count: Number of edges in the network
            jurisdiction: Geographic jurisdiction ('US', 'CA', or None)
            region_preset: Region preset used (e.g., 'great_lakes')
            description: Optional description
            app_version: Version of the source app
            schema_version: CoreGraph schema version
            bundle_version: Bundle format version
            slug: Custom slug (auto-generated if not provided)
            overwrite_slug: If provided, update existing project with this slug
        
        Returns:
            Tuple of (Project, error_message)
        """
        if not self.is_authenticated():
            return None, "Not authenticated"
        
        # Generate slug if not provided
        if not slug:
            slug = generate_slug(name)
        
        if not validate_slug(slug):
            return None, f"Invalid slug: {slug}"
        
        # Check for existing project if overwriting
        existing_project = None
        if overwrite_slug:
            existing = self._get_project_by_slug(overwrite_slug)
            if existing:
                if existing.created_by != self.user_id:
                    return None, "Cannot overwrite another user's project"
                existing_project = existing
                slug = overwrite_slug
        
        # Upload bundle to storage
        bundle_path = f"{self.user_id}/{slug}.zip"
        
        try:
            # Delete existing file if overwriting
            if existing_project:
                try:
                    self.client.storage.from_(self.BUCKET_NAME).remove([existing_project.bundle_path])
                except:
                    pass  # Ignore if file doesn't exist
            
            # Upload new file
            self.client.storage.from_(self.BUCKET_NAME).upload(
                bundle_path,
                bundle_data,
                file_options={"content-type": "application/zip"}
            )
        except Exception as e:
            return None, f"Storage upload failed: {str(e)}"
        
        # Prepare project record
        project_data = {
            'slug': slug,
            'name': name,
            'description': description,
            'source_app': source_app,
            'schema_version': schema_version,
            'bundle_version': bundle_version,
            'app_version': app_version,
            'node_count': node_count,
            'edge_count': edge_count,
            'jurisdiction': jurisdiction,
            'region_preset': region_preset,
            'bundle_path': bundle_path,
            'bundle_size_bytes': len(bundle_data),
            'user_id': self.user_id,  # DB column is user_id
            'is_public': False,
        }
        
        try:
            if existing_project:
                # Update existing record
                response = self.client.table(self.TABLE_NAME).update(project_data).eq('id', existing_project.id).execute()
            else:
                # Insert new record
                response = self.client.table(self.TABLE_NAME).insert(project_data).execute()
            
            if response.data and len(response.data) > 0:
                return Project.from_dict(response.data[0]), None
            else:
                return None, "Database insert failed"
        except Exception as e:
            # Try to clean up uploaded file on failure
            try:
                self.client.storage.from_(self.BUCKET_NAME).remove([bundle_path])
            except:
                pass
            return None, f"Database error: {str(e)}"
    
    def load_project(self, project_id: Optional[str] = None, slug: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Load a project bundle from cloud storage.
        
        Args:
            project_id: Project UUID (preferred)
            slug: Project slug (alternative)
        
        Returns:
            Tuple of (bundle_bytes, error_message)
        """
        if not self.is_authenticated():
            return None, "Not authenticated"
        
        # Get project metadata
        project = None
        if project_id:
            project = self._get_project_by_id(project_id)
        elif slug:
            project = self._get_project_by_slug(slug)
        
        if not project:
            return None, "Project not found"
        
        # Check access (owner or public)
        if project.created_by != self.user_id and not project.is_public:
            return None, "Access denied"
        
        # Download bundle
        try:
            response = self.client.storage.from_(self.BUCKET_NAME).download(project.bundle_path)
            return response, None
        except Exception as e:
            return None, f"Download failed: {str(e)}"
    
    def delete_project(self, project_id: Optional[str] = None, slug: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Delete a project (metadata and bundle).
        
        Returns:
            Tuple of (success, error_message)
        """
        if not self.is_authenticated():
            return False, "Not authenticated"
        
        # Get project
        project = None
        if project_id:
            project = self._get_project_by_id(project_id)
        elif slug:
            project = self._get_project_by_slug(slug)
        
        if not project:
            return False, "Project not found"
        
        if project.created_by != self.user_id:
            return False, "Cannot delete another user's project"
        
        # Delete storage file
        try:
            self.client.storage.from_(self.BUCKET_NAME).remove([project.bundle_path])
        except:
            pass  # Continue even if file doesn't exist
        
        # Delete database record
        try:
            self.client.table(self.TABLE_NAME).delete().eq('id', project.id).execute()
            return True, None
        except Exception as e:
            return False, f"Delete failed: {str(e)}"
    
    def list_projects(
        self,
        include_public: bool = True,
        source_app: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 100
    ) -> Tuple[List[ProjectSummary], Optional[str]]:
        """
        List available projects.
        
        Args:
            include_public: Include public projects from other users
            source_app: Filter by source app
            jurisdiction: Filter by jurisdiction
            limit: Maximum number of results
        
        Returns:
            Tuple of (list of ProjectSummary, error_message)
        """
        if not self.is_authenticated():
            return [], "Not authenticated"
        
        try:
            # Build query
            query = self.client.table(self.TABLE_NAME).select(
                'id, slug, name, source_app, node_count, edge_count, jurisdiction, updated_at, is_public, user_id'
            )
            
            # Filter by ownership or public
            if include_public:
                query = query.or_(f"user_id.eq.{self.user_id},is_public.eq.true")
            else:
                query = query.eq('user_id', self.user_id)
            
            # Apply filters
            if source_app:
                query = query.eq('source_app', source_app)
            if jurisdiction:
                query = query.eq('jurisdiction', jurisdiction)
            
            # Order and limit
            query = query.order('updated_at', desc=True).limit(limit)
            
            response = query.execute()
            
            projects = [ProjectSummary.from_dict(p) for p in response.data]
            return projects, None
        except Exception as e:
            return [], f"Query failed: {str(e)}"
    
    def get_project(self, project_id: Optional[str] = None, slug: Optional[str] = None) -> Tuple[Optional[Project], Optional[str]]:
        """
        Get full project details.
        
        Returns:
            Tuple of (Project, error_message)
        """
        if not self.is_authenticated():
            return None, "Not authenticated"
        
        project = None
        if project_id:
            project = self._get_project_by_id(project_id)
        elif slug:
            project = self._get_project_by_slug(slug)
        
        if not project:
            return None, "Project not found"
        
        # Check access
        if project.created_by != self.user_id and not project.is_public:
            return None, "Access denied"
        
        return project, None
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by UUID."""
        try:
            response = self.client.table(self.TABLE_NAME).select('*').eq('id', project_id).single().execute()
            if response.data:
                return Project.from_dict(response.data)
        except:
            pass
        return None
    
    def _get_project_by_slug(self, slug: str) -> Optional[Project]:
        """Get project by slug."""
        try:
            response = self.client.table(self.TABLE_NAME).select('*').eq('slug', slug).single().execute()
            if response.data:
                return Project.from_dict(response.data)
        except:
            pass
        return None
    
    def check_slug_available(self, slug: str) -> bool:
        """Check if a slug is available."""
        try:
            response = self.client.table(self.TABLE_NAME).select('id').eq('slug', slug).execute()
            return len(response.data) == 0
        except:
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client_from_env() -> ProjectStoreClient:
    """
    Create a ProjectStoreClient from environment variables.
    
    Expects:
        SUPABASE_URL: Supabase project URL
        SUPABASE_KEY: Supabase anon key
    """
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
    
    return ProjectStoreClient(url, key)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Quick test
    print("C4C Project Store Client v1.0.0")
    print("=" * 40)
    print(f"generate_slug('Great Lakes Funders') = {generate_slug('Great Lakes Funders')}")
    print(f"validate_slug('glfn-2024') = {validate_slug('glfn-2024')}")
    print(f"validate_slug('Invalid Slug!') = {validate_slug('Invalid Slug!')}")
