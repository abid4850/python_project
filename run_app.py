"""
🚀 Automated Startup Script for EV Market Forecasting Dashboard
Handles dependency installation and app launch
"""

import subprocess
import sys
import os
import time

def print_header():
    """Print fancy header"""
    print("\n" + "="*80)
    print("🚀 EV MARKET FORECASTING DASHBOARD - STARTUP")
    print("   Kaggle Grandmaster Edition")
    print("="*80 + "\n")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  WARNING: Python 3.8+ recommended")
        return False
    return True

def install_dependencies():
    """Install required packages"""
    print("\n" + "-"*80)
    print("📦 Installing Dependencies...")
    print("-"*80)
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ ERROR: {requirements_file} not found!")
        return False
    
    try:
        # Upgrade pip first
        print("\n📥 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install from requirements
        print("\n📥 Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        print("\n✅ All dependencies installed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to install dependencies")
        print(f"   Error: {e}")
        return False

def verify_imports():
    """Verify key imports are available"""
    print("\n" + "-"*80)
    print("🔍 Verifying Packages...")
    print("-"*80)
    
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("streamlit", "Streamlit"),
        ("prophet", "Prophet"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-Learn"),
        ("plotly", "Plotly"),
    ]
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - NOT FOUND")
            all_ok = False
    
    return all_ok

def launch_app():
    """Launch Streamlit app"""
    print("\n" + "-"*80)
    print("🚀 Launching Application...")
    print("-"*80)
    
    app_file = "ev_market_app.py"
    
    if not os.path.exists(app_file):
        print(f"❌ ERROR: {app_file} not found!")
        return False
    
    print(f"\n✅ Starting {app_file}...")
    print("\n📊 Dashboard URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   • First load trains all models (2-3 minutes)")
    print("   • Use sidebar to navigate sections")
    print("   • Hover on charts for detailed data")
    print("   • Close this terminal to stop the app")
    print("\n" + "="*80 + "\n")
    
    try:
        subprocess.call([sys.executable, "-m", "streamlit", "run", app_file])
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to launch app")
        print(f"   Error: {e}")
        return False

def main():
    """Main execution"""
    print_header()
    
    # Check Python
    if not check_python_version():
        input("Press Enter to continue anyway...")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        print("   Please install manually: pip install -r requirements.txt")
        return False
    
    # Verify imports
    if not verify_imports():
        print("\n⚠️  Some packages may not be installed correctly")
        response = input("   Continue anyway? (y/n): ").lower()
        if response != 'y':
            return False
    
    # Launch app
    return launch_app()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Application launch failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
