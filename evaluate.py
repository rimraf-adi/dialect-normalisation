"""
CLI Entry Point for Indic ASR Evaluation & Dialect Benchmarking Suite.
Usage: python evaluate.py [--lang marathi|all] [--decoder ctc|rnnt|both]
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialect_norm.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ["pipeline", "evaluate", "-h", "--help"]):
        sys.argv.insert(1, "evaluate")
    main()
