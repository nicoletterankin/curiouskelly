#!/usr/bin/env python3
"""
Kelly25 API Launcher
Easy startup script for the Kelly25 Voice API
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import torch
        import soundfile
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_api.txt"])
        return True

def check_model():
    """Check if the trained model exists"""
    model_path = Path("kelly25_model_output/best_model.pth")
    if not model_path.exists():
        print("❌ Trained model not found!")
        print("Please run the training first: python basic_kelly25_trainer.py")
        return False
    
    print("✅ Trained model found")
    return True

def start_api():
    """Start the Kelly25 API server"""
    print("🚀 Starting Kelly25 Voice API Server...")
    print("=" * 50)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Examples: http://localhost:8000/examples")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    # Start the server
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "kelly25_api:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

def main():
    """Main launcher function"""
    print("🎤 Kelly25 Voice API Launcher")
    print("=" * 30)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model
    if not check_model():
        return
    
    # Start API
    start_api()

if __name__ == "__main__":
    main()





































