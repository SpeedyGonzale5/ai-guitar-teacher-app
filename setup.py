#!/usr/bin/env python3
"""
Setup script for AI Sports Analysis Tool
Automates the installation of dependencies and setup process.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required Python packages."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def verify_installation():
    """Verify that all packages are installed correctly."""
    try:
        import cv2
        import mediapipe
        import numpy
        print("✅ All dependencies verified successfully")
        return True
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🏓 AI Sports Analysis Tool Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed during verification")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your video files to the project directory")
    print("2. Add corresponding JSON analysis files")
    print("3. Update CURRENT_CONFIG in pickleball.py to select your video")
    print("4. Run: python pickleball.py")

if __name__ == "__main__":
    main()
