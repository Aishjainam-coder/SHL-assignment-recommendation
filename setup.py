import subprocess
import sys
import os
import platform

def install_requirements():
    print("Installing requirements...")
    try:
        # First, upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install wheel first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wheel"])
        
        # Install numpy and pandas using the latest versions
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        
        # Install remaining requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print("\nIf you're still having issues, try installing Python 3.11 instead of Python 3.13")
        print("You can download Python 3.11 from: https://www.python.org/downloads/release/python-3119/")
        sys.exit(1)

def create_data_directory():
    print("Creating data directory...")
    os.makedirs("data", exist_ok=True)
    print("Data directory created!")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    install_requirements()
    create_data_directory()
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Run the scraper to collect SHL assessment data:")
    print("   python scraper.py")
    print("2. Start the API server:")
    print("   python api.py")
    print("3. In a new terminal, start the Streamlit frontend:")
    print("   streamlit run app.py") 