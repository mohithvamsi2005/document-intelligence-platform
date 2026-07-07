"""
Main Entry Point
-----------------
Run the application:
  Without Docker: streamlit run app/main.py
  With Docker:    docker-compose up

This file exists for documentation and direct execution.
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("Starting Document Intelligence Platform...")
    print("Access at: http://localhost:8501")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/main.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])