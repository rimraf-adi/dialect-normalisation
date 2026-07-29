"""
Unified CLI Entrypoint for Dialect Normalisation and ASR Benchmarking Suite.
"""

import argparse
import sys
from pathlib import Path

from dialect_norm.gemma_pipeline import check_llm_health, execute_generation_pipeline
from dialect_norm.sampler import load_and_sample_dialect_dataset
from dialect_norm.engine import evaluate_language


def run_pipeline_cmd(args):
    provider = args.provider.lower()
    if provider == "lmstudio":
        model_name = args.model_name if args.model_name else "google/gemma-4-e2b"
        base_url = args.base_url if args.base_url else "http://localhost:1234/v1"
    else:  # ollama
        model_name = args.model_name if args.model_name else "gemma4:12b"
        base_url = args.base_url if args.base_url else "http://localhost:11434"

    is_healthy, health_msg = check_llm_health(provider=provider, model_name=model_name, base_url=base_url)
    if not is_healthy:
        print(f"\n❌ LLM SAFETY CHECK FAILED: {health_msg}", file=sys.stderr)
        if provider == "lmstudio":
            print("--> Ensure LM Studio local server is running on port 1234 with model 'google/gemma-4-e2b'.", file=sys.stderr)
        else:
            print("--> Ensure Ollama server is running ('ollama serve').", file=sys.stderr)
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

    sampled_dataset = load_and_sample_dialect_dataset(
        meta_path=meta_path,
        samples_per_dialect=args.samples_per_dialect,
        target_dialects=["D1", "D2", "D4"],
        seed=args.seed,
    )

    execute_generation_pipeline(
        sampled_dataset=sampled_dataset,
        output_dir=output_dir,
        log_dir=log_dir,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )


def run_evaluate_cmd(args):
    from dialect_norm.evaluators.run_all import AVAILABLE_LANGUAGES, run_all_evaluations

    output_dir = Path(args.output_dir)
    target_langs = list(AVAILABLE_LANGUAGES.keys()) if args.lang.lower() == "all" else [args.lang.lower()]

    for l_key in target_langs:
        if l_key not in AVAILABLE_LANGUAGES:
            print(f"Error: Language '{l_key}' is invalid. Options are: {', '.join(list(AVAILABLE_LANGUAGES.keys()))}", file=sys.stderr)
            sys.exit(1)

        info = AVAILABLE_LANGUAGES[l_key]
        dataset_dir = Path(info["dir"])
        candidate1 = dataset_dir / info["dir"] / f"meta_test_{info['dataset_code']}.json"
        candidate2 = dataset_dir / f"meta_test_{info['dataset_code']}.json"

        if candidate1.exists():
            meta_file, base_dir = candidate1, dataset_dir / info["dir"]
        elif candidate2.exists():
            meta_file, base_dir = candidate2, dataset_dir
        else:
            print(f"Warning: Metadata file for {info['name']} not found at {candidate1}. Skipping.", file=sys.stderr)
            continue

        evaluate_language(
            language_name=info["name"],
            dataset_lang_code=info["dataset_code"],
            model_lang_code=info["model_code"],
            meta_file=meta_file,
            base_dir=base_dir,
            decoder_modes=["ctc", "rnnt"] if args.decoder == "both" else [args.decoder],
            device=args.device,
            max_samples=args.max_samples,
            output_dir=output_dir,
            token=args.token,
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Dialect Normalisation & ASR Benchmarking Suite CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Subparser: pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Run synthetic parallel dataset generation pipeline")
    pipe_parser.add_argument("--meta-path", type=str, default=None, help="Path to meta_train_mr_clean.json metadata file")
    pipe_parser.add_argument("--provider", type=str, default="lmstudio", choices=["lmstudio", "ollama"], help="LLM provider")
    pipe_parser.add_argument("--model-name", type=str, default=None, help="Target model name")
    pipe_parser.add_argument("--base-url", type=str, default=None, help="Base URL endpoint")
    pipe_parser.add_argument("--samples-per-dialect", type=int, default=10000, help="Utterances per dialect")
    pipe_parser.add_argument("--output-dir", type=str, default="data/synthetic_parallel", help="Output directory")
    pipe_parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    pipe_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Subparser: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run IndicConformer ASR benchmarking evaluation")
    eval_parser.add_argument("--lang", type=str, default="marathi", help="Language to evaluate or 'all'")
    eval_parser.add_argument("--decoder", type=str, choices=["ctc", "rnnt", "both"], default="both", help="Decoder mode")
    eval_parser.add_argument("--device", type=str, default=None, help="Inference device")
    eval_parser.add_argument("--max-samples", type=int, default=None, help="Max samples per language")
    eval_parser.add_argument("--output-dir", type=str, default="baseline-indic-conformer", help="Output directory")
    eval_parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")

    args = parser.parse_args()

    if args.command == "pipeline":
        run_pipeline_cmd(args)
    elif args.command == "evaluate":
        run_evaluate_cmd(args)
    else:
        # Default behavior: run pipeline if no command given
        pipe_parser.print_help()


if __name__ == "__main__":
    main()
