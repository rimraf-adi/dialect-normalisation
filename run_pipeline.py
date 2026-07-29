"""
CLI Entry Point for Synthetic Parallel Data Generation Pipeline.
Usage: python run_pipeline.py [--provider lmstudio|ollama] [--model-name NAME]
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialect_norm.cli import main

if __name__ == "__main__":
    # Ensure sub-command default is 'pipeline' if invoked directly
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ["pipeline", "evaluate", "-h", "--help"]):
        sys.argv.insert(1, "pipeline")
    main()
