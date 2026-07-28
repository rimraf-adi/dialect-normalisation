"""
Unified CLI Entry Point for Gemma Synthetic Parallel Data Generation Pipeline with Ollama Health & Safety Check.
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialect_norm.gemma_pipeline import check_ollama_health, execute_generation_pipeline
from dialect_norm.sampler import load_and_sample_dialect_dataset


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Gemma 4:12b Synthetic Data Generation Pipeline (10k samples per dialect: D1, D2, D4)."
    )
    parser.add_argument("--meta-path", type=str, default=None, help="Path to meta_train_mr_clean.json metadata file")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma4:12b",
        help="Ollama target model name (default: gemma4:12b)",
    )
    parser.add_argument(
        "--samples-per-dialect",
        type=int,
        default=10000,
        help="Number of distorted samples to extract per dialect (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic_parallel",
        help="Directory to save 1,000-row output CSV files",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save detailed log files (default: logs)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility")

    args = parser.parse_args()

    # CRITICAL SAFETY CHECK FIRST (FAILS FAST IN <0.5s IF OLLAMA IS DOWN)
    is_healthy, health_msg = check_ollama_health(model_name=args.model_name)
    if not is_healthy:
        print(f"\n❌ OLLAMA SAFETY CHECK FAILED: {health_msg}", file=sys.stderr)
        print("--> Make sure Ollama server is running ('ollama serve') and model is pulled.", file=sys.stderr)
        sys.exit(1)

    meta_path1 = Path("IISc_RESPIN_train_mr_clean/IISc_RESPIN_train_mr_clean/meta_train_mr_clean.json")
    meta_path2 = Path("IISc_RESPIN_train_mr_clean/meta_train_mr_clean.json")

    if args.meta_path:
        meta_path = Path(args.meta_path)
    elif meta_path1.exists():
        meta_path = meta_path1
    elif meta_path2.exists():
        meta_path = meta_path2
    else:
        print(f"Error: Could not find meta_train_mr_clean.json at {meta_path1} or {meta_path2}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    # Step 1: Sample 10,000 random/distorted utterances per dialect (D1, D2, D4)
    sampled_dataset = load_and_sample_dialect_dataset(
        meta_path=meta_path,
        samples_per_dialect=args.samples_per_dialect,
        target_dialects=["D1", "D2", "D4"],
        seed=args.seed,
    )

    # Step 2: Execute Gemma generation pipeline (5 sentences/prompt, 1,000 rows/CSV)
    execute_generation_pipeline(
        sampled_dataset=sampled_dataset,
        output_dir=output_dir,
        log_dir=log_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
