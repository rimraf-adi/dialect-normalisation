"""
Unified Evaluation CLI for Dialect Normalisation and ASR Benchmarking.
"""

import argparse
import sys
from pathlib import Path
from dialect_norm.engine import evaluate_language

LANG_CONFIGS = {
    "bhojpuri": {"name": "Bhojpuri", "dataset_code": "bh", "model_code": "hi", "dir": "IISc_RESPIN_test_bh"},
    "bengali": {"name": "Bengali", "dataset_code": "bn", "model_code": "bn", "dir": "IISc_RESPIN_test_bn"},
    "chhattisgarhi": {"name": "Chhattisgarhi", "dataset_code": "ch", "model_code": "hi", "dir": "IISc_RESPIN_test_ch"},
    "hindi": {"name": "Hindi", "dataset_code": "hi", "model_code": "hi", "dir": "IISc_RESPIN_test_hi"},
    "kannada": {"name": "Kannada", "dataset_code": "kn", "model_code": "kn", "dir": "IISc_RESPIN_test_kn"},
    "magahi": {"name": "Magahi", "dataset_code": "mg", "model_code": "hi", "dir": "IISc_RESPIN_test_mg"},
    "marathi": {"name": "Marathi", "dataset_code": "mr", "model_code": "mr", "dir": "IISc_RESPIN_test_mr"},
    "maithili": {"name": "Maithili", "dataset_code": "mt", "model_code": "mai", "dir": "IISc_RESPIN_test_mt"},
    "telugu": {"name": "Telugu", "dataset_code": "te", "model_code": "te", "dir": "IISc_RESPIN_test_te"},
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Unified ASR Evaluation CLI for RESPIN Indic Datasets using IndicConformer."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="marathi",
        help=f"Language to evaluate. Options: {', '.join(list(LANG_CONFIGS.keys()))} or 'all'",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["ctc", "rnnt", "both"],
        default="both",
        help="Decoding mode: ctc, rnnt, or both (default: both)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum utterances to evaluate per language")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline-indic-conformer",
        help="Directory to save output Detailed and Summary YAML files",
    )
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token for gated models")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    target_langs = list(LANG_CONFIGS.keys()) if args.lang.lower() == "all" else [args.lang.lower()]

    for l_key in target_langs:
        if l_key not in LANG_CONFIGS:
            print(f"Error: Language '{l_key}' is invalid. Options are: {', '.join(list(LANG_CONFIGS.keys()))}", file=sys.stderr)
            sys.exit(1)

        info = LANG_CONFIGS[l_key]
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


if __name__ == "__main__":
    main()
