#!/usr/bin/env python3
"""
Healthcare Q&A Tool - Application Launcher

Professional launcher script for the Healthcare Q&A Tool with
environment validation and startup optimization.
"""

import os
import subprocess
import sys
from pathlib import Path


def validate_environment():
    """Validate the environment setup."""
    print("🔍 Validating environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        template_file = Path(".env.template")
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print("📝 Please edit .env file with your API keys")
        else:
            print("❌ .env.template not found")
            return False
    
    # Check required directories
    required_dirs = ["data", "logs"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Environment validation complete")
    return True


def launch_streamlit():
    """Launch the Streamlit application."""
    app_file = Path("streamlit_app.py")
    
    if not app_file.exists():
        print("❌ streamlit_app.py not found!")
        return False
    
    print("🚀 Launching Healthcare Q&A Tool...")
    print("📱 Application will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_file),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--theme.base", "light",
        "--theme.primaryColor", "#1f77b4"
    ]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
        return True
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False


def main():
    """Main launcher function."""
    print("🏥 Healthcare Q&A Tool - Professional Launcher")
    print("=" * 50)
    
    if not validate_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    if not launch_streamlit():
        print("❌ Application launch failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
