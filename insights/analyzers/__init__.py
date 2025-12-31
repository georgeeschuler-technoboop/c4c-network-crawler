"""
InsightGraph Analyzers Package

Network-type-specific analyzers that inherit from NetworkAnalyzer base class.
"""

from .base import (
    NetworkAnalyzer,
    AnalysisResult,
    HealthScore,
    InsightCard,
    BrokerageData,
    ProjectSummary,
    NetworkType,
    SourceApp,
    detect_network_type,
    detect_source_app,
    compute_brokerage_roles,
    get_top_brokers,
    get_brokerage_badge,
    BROKERAGE_ROLE_CONFIG,
    MIN_NODES_FOR_BROKERAGE,
    INSIGHT_RESULT_SCHEMA_VERSION,
)

from .funder_analyzer import FunderAnalyzer
from .social_analyzer import SocialAnalyzer

__all__ = [
    # Base classes and types
    'NetworkAnalyzer',
    'AnalysisResult',
    'HealthScore',
    'InsightCard',
    'BrokerageData',
    'ProjectSummary',
    'NetworkType',
    'SourceApp',
    # Detection functions
    'detect_network_type',
    'detect_source_app',
    # Brokerage utilities
    'compute_brokerage_roles',
    'get_top_brokers',
    'get_brokerage_badge',
    'BROKERAGE_ROLE_CONFIG',
    'MIN_NODES_FOR_BROKERAGE',
    # Schema version
    'INSIGHT_RESULT_SCHEMA_VERSION',
    # Analyzers
    'FunderAnalyzer',
    'SocialAnalyzer',
]
