#!/usr/bin/env python3
"""
Run script for Stock Forecast AI Agent

This script provides a simple way to run the modular version of the
Stock Forecast AI Agent. It will launch the Streamlit app and provide
instructions for accessing it through a web browser.
"""

import subprocess
import os
import sys
import webbrowser
from time import sleep

def main():
    print("Starting Stock Forecast AI Agent...")
    print("-----------------------------------")
    
    # Determine the script path
    script_path = os.path.abspath("app_modular.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        import yfinance
        import torch
        from chronos import ChronosPipeline
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the app with Streamlit
    print("Launching Streamlit app...")
    port = 8505  # Use a specific port to avoid conflicts
    
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", script_path, "--server.port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for Streamlit to start
        sleep(2)
        
        # Open browser
        url = f"http://localhost:{port}"
        print(f"Opening browser to {url}")
        webbrowser.open(url)
        
        # Print instructions
        print("\nStock Forecast AI Agent is running!")
        print("------------------------------------")
        print(f"Access the app at: {url}")
        print("Press Ctrl+C to stop the application")
        
        # Wait for the process
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping Stock Forecast AI Agent...")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        
if __name__ == "__main__":
    main() 