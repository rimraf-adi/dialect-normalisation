"""
Main CLI launcher for dialect-norm package.
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialect_norm.cli import main

if __name__ == "__main__":
    main()
