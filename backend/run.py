import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_server():
    """Run the FastAPI server"""
    print("Starting Signature Verification API server...")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    # Create the backend directory if it doesn't exist
    os.makedirs("backend", exist_ok=True)
    os.chdir("backend")
    
    try:
        install_requirements()
        run_server()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Python and pip are installed")