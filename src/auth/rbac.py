"""
Role-Based Access Control (RBAC) Manager for Healthcare Q&A Tool.

This module provides fine-grained permission management for healthcare
professionals with different access levels and responsibilities.
"""

from enum import Enum
from typing import Dict, List, Set

from loguru import logger

from .auth_manager import Role, User


class Permission(str, Enum):
    """System permissions for fine-grained access control."""
    
    # Document Management
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    EXPORT_DATA = "export_data"
    
    # Search and Query
    BASIC_SEARCH = "basic_search"
    ADVANCED_SEARCH = "advanced_search"
    ASK_QUESTIONS = "ask_questions"
    VIEW_SOURCES = "view_sources"
    
    # Analytics and Insights
    VIEW_ANALYTICS = "view_analytics"
    VIEW_DETAILED_ANALYTICS = "view_detailed_analytics"
    GENERATE_REPORTS = "generate_reports"
    
    # Collection Management
    MANAGE_COLLECTIONS = "manage_collections"
    RESET_COLLECTIONS = "reset_collections"
    BACKUP_COLLECTIONS = "backup_collections"
    
    # System Administration
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    CONFIGURE_SETTINGS = "configure_settings"
    
    # Special Access
    ACCESS_ALL_FEATURES = "access_all_features"
    BULK_OPERATIONS = "bulk_operations"
    API_ACCESS = "api_access"


class RBACManager:
    """Role-Based Access Control manager with healthcare-specific permissions."""
    
    def __init__(self):
        """Initialize RBAC manager with role-permission mappings."""
        self.role_permissions = self._initialize_role_permissions()
        self.feature_permissions = self._initialize_feature_permissions()
        
        logger.info("Initialized RBAC Manager with healthcare role permissions")
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize role-to-permission mappings."""
        return {
            Role.ADMIN: {
                # Full system access
                Permission.READ_DOCUMENTS,
                Permission.WRITE_DOCUMENTS,
                Permission.DELETE_DOCUMENTS,
                Permission.EXPORT_DATA,
                Permission.BASIC_SEARCH,
                Permission.ADVANCED_SEARCH,
                Permission.ASK_QUESTIONS,
                Permission.VIEW_SOURCES,
                Permission.VIEW_ANALYTICS,
                Permission.VIEW_DETAILED_ANALYTICS,
                Permission.GENERATE_REPORTS,
                Permission.MANAGE_COLLECTIONS,
                Permission.RESET_COLLECTIONS,
                Permission.BACKUP_COLLECTIONS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SYSTEM,
                Permission.VIEW_SYSTEM_LOGS,
                Permission.CONFIGURE_SETTINGS,
                Permission.ACCESS_ALL_FEATURES,
                Permission.BULK_OPERATIONS,
                Permission.API_ACCESS
            },
            
            Role.RESEARCHER: {
                # Research-focused permissions
                Permission.READ_DOCUMENTS,
                Permission.WRITE_DOCUMENTS,
                Permission.EXPORT_DATA,
                Permission.BASIC_SEARCH,
                Permission.ADVANCED_SEARCH,
                Permission.ASK_QUESTIONS,
                Permission.VIEW_SOURCES,
                Permission.VIEW_ANALYTICS,
                Permission.VIEW_DETAILED_ANALYTICS,
                Permission.GENERATE_REPORTS,
                Permission.MANAGE_COLLECTIONS,
                Permission.BACKUP_COLLECTIONS,
                Permission.BULK_OPERATIONS,
                Permission.API_ACCESS
            },
            
            Role.CLINICIAN: {
                # Clinical practice permissions
                Permission.READ_DOCUMENTS,
                Permission.BASIC_SEARCH,
                Permission.ADVANCED_SEARCH,
                Permission.ASK_QUESTIONS,
                Permission.VIEW_SOURCES,
                Permission.VIEW_ANALYTICS,
                Permission.EXPORT_DATA
            },
            
            Role.VIEWER: {
                # Read-only permissions
                Permission.READ_DOCUMENTS,
                Permission.BASIC_SEARCH,
                Permission.ASK_QUESTIONS,
                Permission.VIEW_SOURCES
            }
        }
    
    def _initialize_feature_permissions(self) -> Dict[str, Set[Permission]]:
        """Initialize feature-to-permission mappings for UI components."""
        return {
            "research_ingest": {
                Permission.WRITE_DOCUMENTS,
                Permission.MANAGE_COLLECTIONS,
                Permission.ADVANCED_SEARCH
            },
            
            "ask_questions": {
                Permission.ASK_QUESTIONS,
                Permission.READ_DOCUMENTS
            },
            
            "analytics_dashboard": {
                Permission.VIEW_ANALYTICS
            },
            
            "detailed_analytics": {
                Permission.VIEW_DETAILED_ANALYTICS
            },
            
            "export_functionality": {
                Permission.EXPORT_DATA
            },
            
            "collection_management": {
                Permission.MANAGE_COLLECTIONS
            },
            
            "system_settings": {
                Permission.CONFIGURE_SETTINGS,
                Permission.MANAGE_SYSTEM
            },
            
            "user_management": {
                Permission.MANAGE_USERS
            },
            
            "bulk_operations": {
                Permission.BULK_OPERATIONS
            },
            
            "reset_collections": {
                Permission.RESET_COLLECTIONS
            },
            
            "advanced_search": {
                Permission.ADVANCED_SEARCH
            }
        }
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user_permissions = self.role_permissions.get(user.role, set())
        return permission in user_permissions
    
    def has_feature_access(self, user: User, feature: str) -> bool:
        """
        Check if user has access to a specific feature.
        
        Args:
            user: User object
            feature: Feature name to check
            
        Returns:
            True if user has access to feature
        """
        required_permissions = self.feature_permissions.get(feature, set())
        user_permissions = self.role_permissions.get(user.role, set())
        
        # User needs at least one of the required permissions
        return bool(required_permissions.intersection(user_permissions))
    
    def get_user_permissions(self, user: User) -> List[str]:
        """
        Get all permissions for a user.
        
        Args:
            user: User object
            
        Returns:
            List of permission strings
        """
        permissions = self.role_permissions.get(user.role, set())
        return [perm.value for perm in permissions]
    
    def get_accessible_features(self, user: User) -> List[str]:
        """
        Get all features accessible to a user.
        
        Args:
            user: User object
            
        Returns:
            List of accessible feature names
        """
        accessible_features = []
        
        for feature in self.feature_permissions:
            if self.has_feature_access(user, feature):
                accessible_features.append(feature)
        
        return accessible_features
    
    def get_role_description(self, role: Role) -> Dict[str, str]:
        """
        Get detailed description of a role and its capabilities.
        
        Args:
            role: Role to describe
            
        Returns:
            Dictionary with role information
        """
        descriptions = {
            Role.ADMIN: {
                "title": "System Administrator",
                "description": "Full system access with user and system management capabilities",
                "capabilities": [
                    "Manage all users and permissions",
                    "Configure system settings",
                    "Access all features and data",
                    "Perform bulk operations",
                    "Reset and backup collections",
                    "View system logs and analytics"
                ],
                "use_cases": [
                    "System maintenance and configuration",
                    "User account management",
                    "Data backup and recovery",
                    "System monitoring and troubleshooting"
                ]
            },
            
            Role.RESEARCHER: {
                "title": "Healthcare Researcher",
                "description": "Advanced research capabilities with data analysis and export features",
                "capabilities": [
                    "Advanced search and filtering",
                    "Detailed analytics and reporting",
                    "Data export and bulk operations",
                    "Collection management",
                    "API access for automation"
                ],
                "use_cases": [
                    "Literature review and meta-analysis",
                    "Research data collection",
                    "Trend analysis and reporting",
                    "Academic research projects"
                ]
            },
            
            Role.CLINICIAN: {
                "title": "Healthcare Clinician",
                "description": "Clinical practice focused with Q&A and evidence-based search",
                "capabilities": [
                    "Clinical question answering",
                    "Evidence-based search",
                    "Source verification",
                    "Basic analytics viewing",
                    "Limited data export"
                ],
                "use_cases": [
                    "Clinical decision support",
                    "Patient care research",
                    "Evidence-based practice",
                    "Medical education"
                ]
            },
            
            Role.VIEWER: {
                "title": "Healthcare Viewer",
                "description": "Read-only access for basic search and information retrieval",
                "capabilities": [
                    "Basic search functionality",
                    "Question answering",
                    "Source viewing",
                    "Read-only document access"
                ],
                "use_cases": [
                    "Medical education",
                    "Information lookup",
                    "Basic research queries",
                    "Learning and training"
                ]
            }
        }
        
        return descriptions.get(role, {})
    
    def check_bulk_operation_permission(self, user: User, operation_type: str) -> bool:
        """
        Check if user can perform bulk operations.
        
        Args:
            user: User object
            operation_type: Type of bulk operation
            
        Returns:
            True if user has permission for bulk operation
        """
        if not self.has_permission(user, Permission.BULK_OPERATIONS):
            return False
        
        # Additional checks for specific bulk operations
        restricted_operations = ["delete_all", "reset_system", "bulk_delete"]
        
        if operation_type in restricted_operations:
            return self.has_permission(user, Permission.MANAGE_SYSTEM)
        
        return True
    
    def get_permission_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        Get a complete permission matrix for all roles.
        
        Returns:
            Dictionary mapping roles to their permissions
        """
        matrix = {}
        
        for role in Role:
            role_perms = self.role_permissions.get(role, set())
            matrix[role.value] = {
                perm.value: perm in role_perms 
                for perm in Permission
            }
        
        return matrix
