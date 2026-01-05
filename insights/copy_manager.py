"""
InsightGraph Copy Manager

Loads copy strings from INSIGHTGRAPH_COPY_MAP_v1.yaml and provides
typed access for reports, UI, and exports.

Single source of truth for all narrative copy.

VERSION HISTORY:
----------------
v1.0.0 (2026-01-06): Initial release
- Load YAML copy map at startup
- Health score band lookups
- Role definitions (hidden broker, gatekeeper, etc.)
- Report section templates with variable substitution
- Glossary access
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HealthBand:
    """Health score band with label and description."""
    range: str
    label: str
    description: str


@dataclass
class RoleDefinition:
    """Structural role definition."""
    name: str
    definition: str


# =============================================================================
# Copy Manager
# =============================================================================

class CopyManager:
    """
    Manages copy strings loaded from YAML.
    
    Usage:
        copy = CopyManager()
        
        # Health score bands
        band = copy.get_health_band(75)  # Returns HealthBand for "Moderate"
        
        # Role definitions
        broker_def = copy.get_role("hidden_broker")
        
        # Report templates
        template = copy.get_report_template("executive_summary")
        
        # Glossary
        definition = copy.get_glossary_term("network_health")
    """
    
    def __init__(self, yaml_path: Optional[Path] = None):
        """
        Initialize copy manager.
        
        Args:
            yaml_path: Path to YAML file. If None, uses default location.
        """
        if yaml_path is None:
            # Default: look in same directory as this module
            yaml_path = Path(__file__).parent / "INSIGHTGRAPH_COPY_MAP_v1.yaml"
        
        self._yaml_path = yaml_path
        self._data: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        """Load YAML file."""
        if not self._yaml_path.exists():
            # Fallback: try docs folder
            alt_path = Path(__file__).parent.parent / "docs" / "INSIGHTGRAPH_COPY_MAP_v1.yaml"
            if alt_path.exists():
                self._yaml_path = alt_path
            else:
                raise FileNotFoundError(f"Copy map not found: {self._yaml_path}")
        
        with open(self._yaml_path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)
    
    def reload(self):
        """Reload YAML file (for hot reloading during development)."""
        self._load()
    
    # =========================================================================
    # Global Positioning
    # =========================================================================
    
    @property
    def positioning_primary(self) -> str:
        """Primary positioning statement."""
        return self._data.get("global", {}).get("positioning", {}).get(
            "primary", 
            "InsightGraph reveals how coordination actually works inside complex ecosystems."
        )
    
    @property
    def structural_pivot(self) -> str:
        """Structural pivot statement."""
        return self._data.get("global", {}).get("positioning", {}).get(
            "structural_pivot",
            "To understand performance, we look past totals and examine structure."
        )
    
    @property
    def coordination_question(self) -> str:
        """Core coordination question."""
        return self._data.get("global", {}).get("positioning", {}).get(
            "coordination_question",
            "The key question is not whether actors are connected, but whether the system supports coordinated action."
        )
    
    # =========================================================================
    # Health Score
    # =========================================================================
    
    @property
    def health_score_helper(self) -> str:
        """Health score interpretive guardrail."""
        return self._data.get("ui", {}).get("health_score", {}).get(
            "helper",
            "Network Health reflects coordination capacity — not impact, effectiveness, or intent."
        )
    
    def get_health_band(self, score: float) -> HealthBand:
        """
        Get health band for a given score.
        
        Args:
            score: Health score (0-100)
            
        Returns:
            HealthBand with range, label, and description
        """
        bands = self._data.get("ui", {}).get("health_score", {}).get("scale_labels", {})
        
        if score >= 80:
            band_data = bands.get("strong", {})
        elif score >= 60:
            band_data = bands.get("moderate", {})
        elif score >= 40:
            band_data = bands.get("constrained", {})
        else:
            band_data = bands.get("fragile", {})
        
        return HealthBand(
            range=band_data.get("range", ""),
            label=band_data.get("label", "Unknown"),
            description=band_data.get("description", "")
        )
    
    def get_health_label(self, score: float) -> str:
        """Get just the health label (Strong/Moderate/Constrained/Fragile)."""
        return self.get_health_band(score).label
    
    def get_health_description(self, score: float) -> str:
        """Get the health band description."""
        return self.get_health_band(score).description
    
    # =========================================================================
    # Structural Roles
    # =========================================================================
    
    def get_role(self, role_key: str) -> RoleDefinition:
        """
        Get role definition.
        
        Args:
            role_key: One of: hidden_broker, gatekeeper, bridge, coordinator, peripheral
            
        Returns:
            RoleDefinition with name and definition
        """
        roles = self._data.get("ui", {}).get("roles", {})
        role_data = roles.get(role_key, {})
        
        return RoleDefinition(
            name=role_data.get("name", role_key.replace("_", " ").title()),
            definition=role_data.get("definition", "")
        )
    
    @property
    def hidden_broker_definition(self) -> str:
        """Canonical hidden broker definition (for verbatim reuse)."""
        return self.get_role("hidden_broker").definition
    
    @property
    def gatekeeper_definition(self) -> str:
        """Gatekeeper definition."""
        return self.get_role("gatekeeper").definition
    
    # =========================================================================
    # Metric Tooltips
    # =========================================================================
    
    def get_metric_tooltip(self, metric: str) -> str:
        """
        Get tooltip text for a metric.
        
        Args:
            metric: One of: degree, betweenness, eigenvector, density
        """
        tooltips = self._data.get("ui", {}).get("metrics_tooltips", {})
        return tooltips.get(metric, "")
    
    # =========================================================================
    # Report Templates
    # =========================================================================
    
    def get_report_template(self, section_key: str) -> str:
        """
        Get report section template.
        
        Args:
            section_key: One of: executive_summary, scale_snapshot, health_interpretation,
                        coordination_findings, governance_findings, brokers_and_roles,
                        risk_and_fragility, structural_options
                        
        Returns:
            Template string with ${variable} placeholders
        """
        sections = self._data.get("reports", {}).get("sections", {})
        section = sections.get(section_key, {})
        return section.get("template", "")
    
    def get_report_title(self, section_key: str) -> str:
        """Get report section title."""
        sections = self._data.get("reports", {}).get("sections", {})
        section = sections.get(section_key, {})
        return section.get("title", section_key.replace("_", " ").title())
    
    def render_template(self, section_key: str, **variables) -> str:
        """
        Render a report template with variables.
        
        Args:
            section_key: Template key
            **variables: Variables to substitute (e.g., nodes_total=500)
            
        Returns:
            Rendered template string
        """
        template = self.get_report_template(section_key)
        
        # Simple ${var} substitution
        for key, value in variables.items():
            template = template.replace(f"${{{key}}}", str(value))
        
        return template
    
    # =========================================================================
    # Glossary
    # =========================================================================
    
    def get_glossary_term(self, term: str) -> str:
        """Get glossary definition for a term."""
        glossary = self._data.get("glossary", {})
        return glossary.get(term, "")
    
    def get_all_glossary_terms(self) -> Dict[str, str]:
        """Get all glossary terms and definitions."""
        return self._data.get("glossary", {}).copy()
    
    # =========================================================================
    # UI Labels
    # =========================================================================
    
    def get_ui_label(self, *path: str) -> str:
        """
        Get UI label by path.
        
        Args:
            *path: Path components (e.g., "coordination", "header")
        """
        current = self._data.get("ui", {})
        for key in path:
            if isinstance(current, dict):
                current = current.get(key, "")
            else:
                return ""
        return current if isinstance(current, str) else ""
    
    @property
    def governance_helper(self) -> str:
        """Governance helper text."""
        return self._data.get("ui", {}).get("governance", {}).get(
            "helper",
            "Governance ties indicate embedded connective tissue. When absent, coordination must be intentional."
        )
    
    @property
    def risk_fragility_helper(self) -> str:
        """Risk/fragility helper text."""
        return self._data.get("ui", {}).get("risk_fragility", {}).get(
            "helper",
            "Fragility highlights where the network depends on single bridges or small sets of connectors."
        )


# =============================================================================
# Module-level singleton for easy import
# =============================================================================

# Lazy initialization — created on first access
_copy_manager: Optional[CopyManager] = None


def get_copy_manager() -> CopyManager:
    """Get the global CopyManager instance."""
    global _copy_manager
    if _copy_manager is None:
        try:
            _copy_manager = CopyManager()
        except FileNotFoundError:
            # Return a fallback manager with defaults
            _copy_manager = CopyManager.__new__(CopyManager)
            _copy_manager._data = {}
    return _copy_manager


# Convenience aliases
def get_health_label(score: float) -> str:
    """Get health label for score."""
    return get_copy_manager().get_health_label(score)


def get_health_description(score: float) -> str:
    """Get health description for score."""
    return get_copy_manager().get_health_description(score)


def get_hidden_broker_definition() -> str:
    """Get canonical hidden broker definition."""
    return get_copy_manager().hidden_broker_definition


def get_health_guardrail() -> str:
    """Get health score interpretive guardrail."""
    return get_copy_manager().health_score_helper
