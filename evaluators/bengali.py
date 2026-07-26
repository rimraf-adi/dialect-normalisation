import argparse
import sys
import torch
from pathlib import Path
from dialect_norm.engine import evaluate_language


def evaluate_bengali(
    meta_file: Path = None,
    base_dir: Path = None,
    decoder: str = "both",
    device: str = None,
    max_samples: int = None,
    output_dir: Path = Path("baseline-indic-conformer"),
    token: str = None,
):
    if not meta_file:
        candidate1 = Path("IISc_RESPIN_test_bn/IISc_RESPIN_test_bn/meta_test_bn.json")
        candidate2 = Path("IISc_RESPIN_test_bn/meta_test_bn.json")
        if candidate1.exists():
            meta_file, base_dir = candidate1, Path("IISc_RESPIN_test_bn/IISc_RESPIN_test_bn")
        elif candidate2.exists():
            meta_file, base_dir = candidate2, Path("IISc_RESPIN_test_bn")
        else:
            meta_file, base_dir = candidate1, Path("IISc_RESPIN_test_bn/IISc_RESPIN_test_bn")

    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    decoder_modes = ["ctc", "rnnt"] if decoder == "both" else [decoder]

    return evaluate_language(
        language_name="Bengali",
        dataset_lang_code="bn",
        model_lang_code="bn",
        meta_file=meta_file,
        base_dir=base_dir,
        decoder_modes=decoder_modes,
        device=device,
        max_samples=max_samples,
        output_dir=output_dir,
        token=token,
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate IndicConformer on RESPIN Bengali Dataset.")
    parser.add_argument("--meta-file", type=str, default=None, help="Path to meta_test_bn.json metadata file")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory containing test audio files")
    parser.add_argument("--decoder", type=str, choices=["ctc", "rnnt", "both"], default="both", help="Decoding mode")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of utterances to evaluate")
    parser.add_argument("--output-dir", type=str, default="baseline-indic-conformer", help="Directory to save output YAML files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")

    args = parser.parse_args()
    meta_file = Path(args.meta_file) if args.meta_file else None
    base_dir = Path(args.base_dir) if args.base_dir else None

    evaluate_bengali(
        meta_file=meta_file,
        base_dir=base_dir,
        decoder=args.decoder,
        device=args.device,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        token=args.token,
    )


if __name__ == "__main__":
    main()
