#!/usr/bin/env python3
"""
Orchestration script for building all Kelly assets.
This can be called directly or imported.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly_pack.cli import main

if __name__ == "__main__":
    sys.exit(main())


