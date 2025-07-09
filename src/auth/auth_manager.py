"""
Authentication Manager for Healthcare Q&A Tool.

This module provides comprehensive user authentication and session management
with role-based access control (RBAC) for healthcare professionals.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import bcrypt
import jwt
from loguru import logger
from pydantic import BaseModel

from ..config import get_settings


class Role(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"
    VIEWER = "viewer"


class User(BaseModel):
    """User model with healthcare-specific attributes."""
    username: str
    email: str
    full_name: str
    role: Role
    department: Optional[str] = None
    institution: Optional[str] = None
    license_number: Optional[str] = None
    specialization: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    password_hash: str = ""
    
    class Config:
        use_enum_values = True


class AuthManager:
    """Comprehensive authentication manager with RBAC support."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.settings = get_settings()
        self.users_db = self._initialize_demo_users()
        self.active_sessions: Dict[str, Dict] = {}
        
        logger.info("Initialized AuthManager with demo users")
    
    def _initialize_demo_users(self) -> Dict[str, User]:
        """Initialize demo users for different roles."""
        demo_users = {}
        
        # Admin user - Full system access
        admin_user = User(
            username="admin",
            email="admin@mediinsight.com",
            full_name="Dr. Sarah Chen",
            role=Role.ADMIN,
            department="IT Administration",
            institution="MediInsight Health Solutions",
            specialization="Healthcare Informatics",
            is_active=True,
            created_at=datetime.now(),
            password_hash=self._hash_password("admin123")
        )
        demo_users["admin"] = admin_user
        
        # Researcher user - Research and analytics access
        researcher_user = User(
            username="researcher",
            email="researcher@mediinsight.com",
            full_name="Dr. Michael Rodriguez",
            role=Role.RESEARCHER,
            department="Research & Development",
            institution="MediInsight Health Solutions",
            license_number="MD-12345",
            specialization="Endocrinology",
            is_active=True,
            created_at=datetime.now(),
            password_hash=self._hash_password("research123")
        )
        demo_users["researcher"] = researcher_user
        
        # Clinician user - Clinical query access
        clinician_user = User(
            username="clinician",
            email="clinician@mediinsight.com",
            full_name="Dr. Emily Johnson",
            role=Role.CLINICIAN,
            department="Clinical Practice",
            institution="MediInsight Health Solutions",
            license_number="MD-67890",
            specialization="Internal Medicine",
            is_active=True,
            created_at=datetime.now(),
            password_hash=self._hash_password("clinic123")
        )
        demo_users["clinician"] = clinician_user
        
        # Viewer user - Read-only access
        viewer_user = User(
            username="viewer",
            email="viewer@mediinsight.com",
            full_name="John Smith",
            role=Role.VIEWER,
            department="Medical Education",
            institution="MediInsight Health Solutions",
            specialization="Medical Student",
            is_active=True,
            created_at=datetime.now(),
            password_hash=self._hash_password("view123")
        )
        demo_users["viewer"] = viewer_user
        
        return demo_users
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.users_db.get(username.lower())
        
        if not user:
            logger.warning(f"Authentication failed: User '{username}' not found")
            return None
        
        if not user.is_active:
            logger.warning(f"Authentication failed: User '{username}' is inactive")
            return None
        
        if not self._verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: Invalid password for user '{username}'")
            return None
        
        # Update last login
        user.last_login = datetime.now()
        logger.info(f"User '{username}' authenticated successfully")
        
        return user
    
    def create_session_token(self, user: User) -> str:
        """
        Create a JWT session token for authenticated user.
        
        Args:
            user: Authenticated user
            
        Returns:
            JWT token string
        """
        payload = {
            "username": user.username,
            "email": user.email,
            "role": user.role,  # Already a string due to use_enum_values = True
            "full_name": user.full_name,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.settings.session_timeout_hours)
        }
        
        token = jwt.encode(
            payload,
            self.settings.jwt_secret_key,
            algorithm="HS256"
        )
        
        # Store active session
        session_id = hashlib.md5(f"{user.username}{datetime.now()}".encode()).hexdigest()
        self.active_sessions[session_id] = {
            "user": user,
            "token": token,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        logger.info(f"Created session token for user '{user.username}'")
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate a JWT token and return the user.
        
        Args:
            token: JWT token string
            
        Returns:
            User object if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=["HS256"]
            )
            
            username = payload.get("username")
            user = self.users_db.get(username)
            
            if user and user.is_active:
                return user
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token expired")
        except jwt.InvalidTokenError:
            logger.warning("Token validation failed: Invalid token")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
        
        return None
    
    def get_user_permissions(self, user: User) -> List[str]:
        """
        Get user permissions based on role.
        
        Args:
            user: User object
            
        Returns:
            List of permission strings
        """
        role_permissions = {
            Role.ADMIN: [
                "read_documents", "write_documents", "delete_documents",
                "manage_users", "manage_system", "view_analytics",
                "export_data", "manage_collections", "access_all_features"
            ],
            Role.RESEARCHER: [
                "read_documents", "write_documents", "view_analytics",
                "export_data", "manage_collections", "advanced_search"
            ],
            Role.CLINICIAN: [
                "read_documents", "view_analytics", "basic_search",
                "ask_questions", "view_sources"
            ],
            Role.VIEWER: [
                "read_documents", "basic_search", "ask_questions"
            ]
        }
        
        return role_permissions.get(user.role, [])
    
    def has_permission(self, user: User, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: User object
            permission: Permission string to check
            
        Returns:
            True if user has permission, False otherwise
        """
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def get_demo_credentials(self) -> Dict[str, Dict[str, str]]:
        """
        Get demo user credentials for testing.
        
        Returns:
            Dictionary of demo credentials
        """
        return {
            "admin": {
                "username": "admin",
                "password": "admin123",
                "role": "Administrator",
                "description": "Full system access - manage users, collections, and all features"
            },
            "researcher": {
                "username": "researcher", 
                "password": "research123",
                "role": "Researcher",
                "description": "Research access - advanced search, analytics, data export"
            },
            "clinician": {
                "username": "clinician",
                "password": "clinic123", 
                "role": "Clinician",
                "description": "Clinical access - Q&A, basic search, view sources"
            },
            "viewer": {
                "username": "viewer",
                "password": "view123",
                "role": "Viewer", 
                "description": "Read-only access - basic search and Q&A only"
            }
        }
    
    def logout_user(self, token: str) -> bool:
        """
        Logout user by invalidating their session.
        
        Args:
            token: JWT token to invalidate
            
        Returns:
            True if logout successful
        """
        # Find and remove session
        for session_id, session_data in list(self.active_sessions.items()):
            if session_data.get("token") == token:
                del self.active_sessions[session_id]
                logger.info(f"User logged out successfully")
                return True
        
        return False
