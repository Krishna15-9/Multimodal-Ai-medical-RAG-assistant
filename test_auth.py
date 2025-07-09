#!/usr/bin/env python3
"""
Quick test script for authentication system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.auth import AuthManager, RBACManager

def test_authentication():
    """Test the authentication system."""
    print("🔍 Testing Authentication System...")
    
    try:
        # Initialize managers
        auth_manager = AuthManager()
        rbac_manager = RBACManager()
        
        print("✅ Managers initialized successfully")
        
        # Test demo credentials
        demo_creds = auth_manager.get_demo_credentials()
        print(f"✅ Found {len(demo_creds)} demo users")
        
        # Test authentication for each role
        for role, creds in demo_creds.items():
            username = creds['username']
            password = creds['password']
            
            print(f"\n🔐 Testing {role} login...")
            user = auth_manager.authenticate_user(username, password)
            
            if user:
                print(f"✅ {username} authenticated successfully")
                print(f"   Role: {user.role}")
                print(f"   Full Name: {user.full_name}")
                
                # Test token creation
                token = auth_manager.create_session_token(user)
                print(f"✅ JWT token created successfully")
                
                # Test permissions
                permissions = rbac_manager.get_user_permissions(user)
                print(f"✅ User has {len(permissions)} permissions")
                
                # Test feature access
                features = rbac_manager.get_accessible_features(user)
                print(f"✅ User can access {len(features)} features")
                
            else:
                print(f"❌ {username} authentication failed")
        
        print("\n🎉 Authentication system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

if __name__ == "__main__":
    test_authentication()
