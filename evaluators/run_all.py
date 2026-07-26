import argparse
import sys
import torch
from pathlib import Path
from dialect_norm.engine import evaluate_language

AVAILABLE_LANGUAGES = {
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

    parser = argparse.ArgumentParser(description="Evaluate IndicConformer on all available RESPIN Datasets.")
    parser.add_argument("--langs", nargs="+", default=list(AVAILABLE_LANGUAGES.keys()), help="Languages to evaluate (default: all)")
    parser.add_argument("--decoder", type=str, choices=["ctc", "rnnt", "both"], default="both", help="Decoding mode")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of utterances per language")
    parser.add_argument("--output-dir", type=str, default="baseline-indic-conformer", help="Directory to save output YAML files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")

    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    decoder_modes = ["ctc", "rnnt"] if args.decoder == "both" else [args.decoder]
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("RUNNING INDIC-CONFORMER EVALUATIONS ACROSS ALL LANGUAGES")
    print(f"Target Directory: {output_dir.resolve()}")
    print(f"Selected Languages: {', '.join(args.langs)}")
    print("=" * 70)

    for lang in args.langs:
        lang_key = lang.lower().strip()
        if lang_key not in AVAILABLE_LANGUAGES:
            print(f"Warning: Language '{lang}' not recognized. Skipping.", file=sys.stderr)
            continue

        info = AVAILABLE_LANGUAGES[lang_key]
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
            decoder_modes=decoder_modes,
            device=device,
            max_samples=args.max_samples,
            output_dir=output_dir,
            token=args.token,
        )


if __name__ == "__main__":
    main()
