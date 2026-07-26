import argparse
import sys
import torch
from pathlib import Path
from eval_engine import evaluate_dataset


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate IndicConformer on RESPIN Magahi Dataset.")
    parser.add_argument("--meta-file", type=str, default=None, help="Path to meta_test_mg.json metadata file")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory containing test audio files")
    parser.add_argument("--decoder", type=str, choices=["ctc", "rnnt", "both"], default="both", help="Decoding mode: ctc, rnnt, or both (default: both)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of utterances to evaluate")
    parser.add_argument("--output-dir", type=str, default="baseline-indic-conformer", help="Directory to save output YAML files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token for gated models")

    args = parser.parse_args()

    # Default paths for Magahi
    if args.meta_file:
        meta_file = Path(args.meta_file)
        base_dir = Path(args.base_dir) if args.base_dir else meta_file.parent
    else:
        candidate1 = Path("IISc_RESPIN_test_mg/IISc_RESPIN_test_mg/meta_test_mg.json")
        candidate2 = Path("IISc_RESPIN_test_mg/meta_test_mg.json")
        if candidate1.exists():
            meta_file, base_dir = candidate1, Path("IISc_RESPIN_test_mg/IISc_RESPIN_test_mg")
        elif candidate2.exists():
            meta_file, base_dir = candidate2, Path("IISc_RESPIN_test_mg")
        else:
            meta_file, base_dir = candidate1, Path("IISc_RESPIN_test_mg/IISc_RESPIN_test_mg")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    decoder_modes = ["ctc", "rnnt"] if args.decoder == "both" else [args.decoder]

    evaluate_dataset(
        language_name="Magahi",
        dataset_lang_code="mg",
        model_lang_code="hi",
        meta_file=meta_file,
        base_dir=base_dir,
        decoder_modes=decoder_modes,
        device=device,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        token=args.token,
    )


if __name__ == "__main__":
    main()
