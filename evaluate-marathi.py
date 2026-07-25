import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
import yaml
import jiwer
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel
from transformers.utils import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def normalize_text(text: str) -> str:
    """Normalizes text by removing Marathi/Devanagari punctuation and extra whitespace."""
    if not text:
        return ""
    punct_pattern = r'[।॥?!.,;:"\'\-\(\)\[\]\{\}—«»“”‘’]'
    text = re.sub(punct_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics_for_set(references_raw, hypotheses_raw, references_norm, hypotheses_norm):
    """Computes WER, CER, SER, and edit distance breakdowns for raw and normalized text."""
    if not references_raw:
        return {}

    raw_wer = jiwer.wer(references_raw, hypotheses_raw)
    raw_cer = jiwer.cer(references_raw, hypotheses_raw)
    raw_word_output = jiwer.process_words(references_raw, hypotheses_raw)
    raw_ser = sum(r != h for r, h in zip(references_raw, hypotheses_raw)) / len(references_raw)

    norm_wer = jiwer.wer(references_norm, hypotheses_norm)
    norm_cer = jiwer.cer(references_norm, hypotheses_norm)
    norm_word_output = jiwer.process_words(references_norm, hypotheses_norm)
    norm_ser = sum(r != h for r, h in zip(references_norm, hypotheses_norm)) / len(references_norm)

    return {
        "num_samples": len(references_raw),
        "raw": {
            "wer": float(round(raw_wer, 4)),
            "wer_percentage": float(round(raw_wer * 100, 2)),
            "cer": float(round(raw_cer, 4)),
            "cer_percentage": float(round(raw_cer * 100, 2)),
            "ser": float(round(raw_ser, 4)),
            "ser_percentage": float(round(raw_ser * 100, 2)),
            "exact_match_acc_percentage": float(round((1.0 - raw_ser) * 100, 2)),
            "substitutions": int(raw_word_output.substitutions),
            "deletions": int(raw_word_output.deletions),
            "insertions": int(raw_word_output.insertions),
            "hits": int(raw_word_output.hits),
            "total_ref_words": int(sum(len(ref.split()) for ref in references_raw)),
        },
        "normalized": {
            "wer": float(round(norm_wer, 4)),
            "wer_percentage": float(round(norm_wer * 100, 2)),
            "cer": float(round(norm_cer, 4)),
            "cer_percentage": float(round(norm_cer * 100, 2)),
            "ser": float(round(norm_ser, 4)),
            "ser_percentage": float(round(norm_ser * 100, 2)),
            "exact_match_acc_percentage": float(round((1.0 - norm_ser) * 100, 2)),
            "substitutions": int(norm_word_output.substitutions),
            "deletions": int(norm_word_output.deletions),
            "insertions": int(norm_word_output.insertions),
            "hits": int(norm_word_output.hits),
            "total_ref_words": int(sum(len(ref.split()) for ref in references_norm)),
        },
    }


def to_python_types(obj):
    if isinstance(obj, dict):
        return {str(k): to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_types(x) for x in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def load_and_preprocess_audio(audio_path: str, target_sample_rate: int = 16000):
    """Loads an audio file, converts to mono, and resamples to target_sample_rate."""
    wav, sr = torchaudio.load(audio_path)
    if wav.ndim > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
    return wav


def run_evaluation_for_decoder(
    decoder_name: str,
    model,
    samples: list,
    device: str,
    lang_code: str = "mr",
):
    print(f"\n--- Running Inference with Decoder: {decoder_name.upper()} ---")
    predictions = []
    start_time = time.time()

    for sample in tqdm(samples, desc=f"Transcribing ({decoder_name.upper()})"):
        wav_path = sample["full_wav_path"]
        try:
            wav = load_and_preprocess_audio(wav_path).to(device)
            with torch.no_grad():
                transcription = model(wav, lang_code, decoder_name)
            if isinstance(transcription, (list, tuple)):
                transcription = transcription[0]
            predictions.append(str(transcription))
        except Exception as e:
            print(f"\nWarning: Error processing {wav_path}: {e}", file=sys.stderr)
            predictions.append("")

    elapsed_time = time.time() - start_time
    print(f"Inference ({decoder_name.upper()}) completed in {elapsed_time:.2f}s ({len(samples)/elapsed_time:.2f} UTT/s).")

    all_ref_raw, all_hyp_raw = [], []
    all_ref_norm, all_hyp_norm = [], []

    for idx, sample in enumerate(samples):
        ref_raw = sample["text"].strip()
        hyp_raw = predictions[idx].strip()
        ref_norm = normalize_text(ref_raw)
        hyp_norm = normalize_text(hyp_raw)

        all_ref_raw.append(ref_raw)
        all_hyp_raw.append(hyp_raw)
        all_ref_norm.append(ref_norm)
        all_hyp_norm.append(hyp_norm)

    overall_metrics = compute_metrics_for_set(all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm)

    def compute_group_metrics(group_key):
        groups = {}
        for idx, sample in enumerate(samples):
            val = sample.get(group_key, "Unknown")
            if val not in groups:
                groups[val] = {"ref_raw": [], "hyp_raw": [], "ref_norm": [], "hyp_norm": [], "duration": 0.0}
            groups[val]["ref_raw"].append(all_ref_raw[idx])
            groups[val]["hyp_raw"].append(all_hyp_raw[idx])
            groups[val]["ref_norm"].append(all_ref_norm[idx])
            groups[val]["hyp_norm"].append(all_hyp_norm[idx])
            groups[val]["duration"] += sample.get("duration", 0.0)

        result_group = {}
        for name, data in sorted(groups.items()):
            res = compute_metrics_for_set(data["ref_raw"], data["hyp_raw"], data["ref_norm"], data["hyp_norm"])
            res["total_duration_hours"] = float(round(data["duration"] / 3600.0, 3))
            result_group[str(name)] = res
        return result_group

    def compute_composite_group_metrics(key1, key2):
        groups = {}
        for idx, sample in enumerate(samples):
            val1 = sample.get(key1, "Unknown")
            val2 = sample.get(key2, "Unknown")
            comp_key = f"{val1}_{val2}"
            if comp_key not in groups:
                groups[comp_key] = {"ref_raw": [], "hyp_raw": [], "ref_norm": [], "hyp_norm": [], "duration": 0.0}
            groups[comp_key]["ref_raw"].append(all_ref_raw[idx])
            groups[comp_key]["hyp_raw"].append(all_hyp_raw[idx])
            groups[comp_key]["ref_norm"].append(all_ref_norm[idx])
            groups[comp_key]["hyp_norm"].append(all_hyp_norm[idx])
            groups[comp_key]["duration"] += sample.get("duration", 0.0)

        result_group = {}
        for name, data in sorted(groups.items()):
            res = compute_metrics_for_set(data["ref_raw"], data["hyp_raw"], data["ref_norm"], data["hyp_norm"])
            res["total_duration_hours"] = float(round(data["duration"] / 3600.0, 3))
            result_group[str(name)] = res
        return result_group

    dialect_metrics = compute_group_metrics("dialect")
    domain_metrics = compute_group_metrics("domain")
    gender_metrics = compute_group_metrics("gender")
    slab_metrics = compute_group_metrics("slab")
    age_group_metrics = compute_group_metrics("age_group")
    dialect_x_domain_metrics = compute_composite_group_metrics("dialect", "domain")

    # Complete samplewise analytics for ALL samples
    samplewise_analytics = []
    for idx, s in enumerate(samples):
        r_raw = all_ref_raw[idx]
        h_raw = all_hyp_raw[idx]
        r_norm = all_ref_norm[idx]
        h_norm = all_hyp_norm[idx]

        r_wer = float(round(jiwer.wer(r_raw, h_raw), 4)) if r_raw else 0.0
        n_wer = float(round(jiwer.wer(r_norm, h_norm), 4)) if r_norm else 0.0
        r_cer = float(round(jiwer.cer(r_raw, h_raw), 4)) if r_raw else 0.0
        n_cer = float(round(jiwer.cer(r_norm, h_norm), 4)) if r_norm else 0.0

        samplewise_analytics.append({
            "utterance_id": s.get("utterance_id", s.get("utt_key")),
            "dialect": s.get("dialect"),
            "domain": s.get("domain"),
            "gender": s.get("gender"),
            "slab": s.get("slab"),
            "age_group": s.get("age_group"),
            "speaker_id": s.get("speaker_id"),
            "duration_seconds": s.get("duration"),
            "reference_raw": r_raw,
            "hypothesis_raw": h_raw,
            "reference_normalized": r_norm,
            "hypothesis_normalized": h_norm,
            "raw_wer": r_wer,
            "normalized_wer": n_wer,
            "raw_cer": r_cer,
            "normalized_cer": n_cer,
            "exact_match": (r_norm == h_norm),
        })

    return {
        "overall_metrics": overall_metrics,
        "dialect_breakdown": dialect_metrics,
        "domain_breakdown": domain_metrics,
        "gender_breakdown": gender_metrics,
        "slab_breakdown": slab_metrics,
        "age_group_breakdown": age_group_metrics,
        "dialect_x_domain_breakdown": dialect_x_domain_metrics,
        "samplewise_analytics": samplewise_analytics,
    }


def evaluate_marathi(
    meta_file: Path,
    base_dir: Path,
    decoder_modes: list = ["ctc", "rnnt"],
    model_name: str = "ai4bharat/indic-conformer-600m-multilingual",
    device: str = "cuda",
    max_samples: int = None,
    output_detailed_yaml_path: Path = None,
    output_summary_yaml_path: Path = None,
    token: str = None,
):
    print("\n" + "=" * 70)
    print("INDIC-CONFORMER EVALUATION ON MARATHI (MR) TEST SET")
    print("=" * 70)

    if not meta_file.exists():
        print(f"Error: Metadata file not found at {meta_file.resolve()}", file=sys.stderr)
        return None

    print(f"Loading metadata from: {meta_file}...")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    total_in_meta = len(meta_data)
    print(f"Total utterances available in metadata: {total_in_meta}")

    samples = []
    for idx, (key, item) in enumerate(meta_data.items()):
        if max_samples and idx >= max_samples:
            break
        rel_wav = item["wav_path"]
        full_wav_path = base_dir / rel_wav
        item["utt_key"] = key
        item["full_wav_path"] = str(full_wav_path)
        samples.append(item)

    print(f"Evaluating on {len(samples)} utterances using device '{device}'...")

    total_duration_sec = sum(s.get("duration", 0.0) for s in samples)
    total_duration_hrs = float(round(total_duration_sec / 3600.0, 3))

    print(f"Loading model: {model_name}...")
    try:
        model_kwargs = {"trust_remote_code": True}
        if token:
            model_kwargs["token"] = token
        elif os.environ.get("HF_TOKEN"):
            model_kwargs["token"] = os.environ.get("HF_TOKEN")

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            print(
                f"\n[ERROR] Model '{model_name}' requires Hugging Face authentication/gated repo access.\n"
                "Please accept model terms at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual\n"
                "and pass your access token via `--token YOUR_HF_TOKEN` or set `set HF_TOKEN=YOUR_HF_TOKEN`.\n",
                file=sys.stderr,
            )
        raise e

    model = model.to(device)
    model.eval()

    decoder_results = {}
    for dec in decoder_modes:
        dec_res = run_evaluation_for_decoder(dec, model, samples, device, lang_code="mr")
        decoder_results[dec] = dec_res

    # 1. Build DETAILED Report (Sample-wise, Dialect-wise, Domain-wise, Gender, Slab, Age analytics)
    detailed_report = {
        "model_info": {
            "model_name": model_name,
            "language": "mr",
            "device": device,
        },
        "dataset_info": {
            "split": "test",
            "metadata_file": str(meta_file),
            "total_utterances_evaluated": len(samples),
            "total_utterances_in_meta": total_in_meta,
            "total_duration_hours": total_duration_hrs,
            "avg_duration_seconds": float(round(total_duration_sec / len(samples), 2)) if samples else 0.0,
        },
        "decoder_results": decoder_results,
    }

    # 2. Build SMALL SUMMARY Report (Key overall metrics & high-level breakdowns only)
    summary_decoder_results = {}
    for dec, res in decoder_results.items():
        om = res["overall_metrics"]
        summary_decoder_results[dec] = {
            "overall_summary": {
                "raw_wer_percentage": om["raw"]["wer_percentage"],
                "normalized_wer_percentage": om["normalized"]["wer_percentage"],
                "raw_cer_percentage": om["raw"]["cer_percentage"],
                "normalized_cer_percentage": om["normalized"]["cer_percentage"],
                "raw_ser_percentage": om["raw"]["ser_percentage"],
                "normalized_ser_percentage": om["normalized"]["ser_percentage"],
                "normalized_exact_match_acc_percentage": om["normalized"]["exact_match_acc_percentage"],
                "total_ref_words": om["normalized"]["total_ref_words"],
                "substitutions": om["normalized"]["substitutions"],
                "deletions": om["normalized"]["deletions"],
                "insertions": om["normalized"]["insertions"],
                "hits": om["normalized"]["hits"],
            },
            "dialect_normalized_wer": {
                d: {
                    "normalized_wer_percentage": ddata["normalized"]["wer_percentage"],
                    "normalized_cer_percentage": ddata["normalized"]["cer_percentage"],
                    "utterances": ddata["num_samples"],
                }
                for d, ddata in res["dialect_breakdown"].items()
            },
            "domain_normalized_wer": {
                dm: {
                    "normalized_wer_percentage": dmdata["normalized"]["wer_percentage"],
                    "normalized_cer_percentage": dmdata["normalized"]["cer_percentage"],
                    "utterances": dmdata["num_samples"],
                }
                for dm, dmdata in res["domain_breakdown"].items()
            },
        }

    summary_report = {
        "model_info": detailed_report["model_info"],
        "dataset_info": detailed_report["dataset_info"],
        "key_metrics_summary": summary_decoder_results,
    }

    clean_detailed_report = to_python_types(detailed_report)
    clean_summary_report = to_python_types(summary_report)

    # Save Detailed YAML Report
    if output_detailed_yaml_path:
        out_detailed_file = Path(output_detailed_yaml_path).resolve()
        with open(out_detailed_file, "w", encoding="utf-8") as f:
            yaml.dump(clean_detailed_report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"\n[Detailed YAML Report] Saved to: {out_detailed_file}")

    # Save Small Summary YAML Report
    if output_summary_yaml_path:
        out_summary_file = Path(output_summary_yaml_path).resolve()
        with open(out_summary_file, "w", encoding="utf-8") as f:
            yaml.dump(clean_summary_report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"[Small Summary YAML Report] Saved to: {out_summary_file}")

    print("\n" + "=" * 70)
    print("INDIC-CONFORMER MARATHI EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Utterances evaluated : {len(samples)}")
    print(f"Total Audio Duration      : {total_duration_hrs} hours")

    for dec, res in decoder_results.items():
        om = res["overall_metrics"]
        print("-" * 70)
        print(f"DECODER MODE: {dec.upper()}")
        print(f"  RAW WER                    : {om['raw']['wer_percentage']}% (WER={om['raw']['wer']})")
        print(f"  NORMALIZED WER             : {om['normalized']['wer_percentage']}% (WER={om['normalized']['wer']})")
        print(f"  RAW CER                    : {om['raw']['cer_percentage']}%")
        print(f"  NORMALIZED CER             : {om['normalized']['cer_percentage']}%")
        print(f"  NORMALIZED Exact Match Acc : {om['normalized']['exact_match_acc_percentage']}%")
        print("  DIALECT-WISE NORMALIZED WER:")
        for dialect, ddata in res["dialect_breakdown"].items():
            print(f"    - {dialect:4s}: WER = {ddata['normalized']['wer_percentage']:5.2f}% | CER = {ddata['normalized']['cer_percentage']:5.2f}% | Utts = {ddata['num_samples']}")
        print("  DOMAIN-WISE NORMALIZED WER:")
        for domain, dmdata in res["domain_breakdown"].items():
            print(f"    - {domain:12s}: WER = {dmdata['normalized']['wer_percentage']:5.2f}% | CER = {dmdata['normalized']['cer_percentage']:5.2f}% | Utts = {dmdata['num_samples']}")
    print("=" * 70)

    return detailed_report, summary_report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate IndicConformer on RESPIN Marathi Dataset.")
    parser.add_argument("--meta-file", type=str, default=None, help="Path to meta_test_mr.json metadata file")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory containing test audio files")
    parser.add_argument("--decoder", type=str, choices=["ctc", "rnnt", "both"], default="both", help="Decoding mode: ctc, rnnt, or both (default: both)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of utterances to evaluate")
    parser.add_argument("--output-detailed-yaml", type=str, default="indic_conformer_marathi_detailed.yaml", help="Path to detailed output YAML file")
    parser.add_argument("--output-summary-yaml", type=str, default="indic_conformer_marathi_summary.yaml", help="Path to summary output YAML file")
    parser.add_argument("--output-yaml", type=str, default=None, help="Legacy output YAML flag (sets detailed output path)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token for gated models")

    args = parser.parse_args()

    detailed_path = Path(args.output_yaml) if args.output_yaml else Path(args.output_detailed_yaml)
    summary_path = Path(args.output_summary_yaml)

    # Determine default paths if not explicitly provided
    if args.meta_file:
        meta_file = Path(args.meta_file)
        base_dir = Path(args.base_dir) if args.base_dir else meta_file.parent
    else:
        candidate_meta1 = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr/meta_test_mr.json")
        candidate_meta2 = Path("IISc_RESPIN_test_mr/meta_test_mr.json")
        if candidate_meta1.exists():
            meta_file = candidate_meta1
            base_dir = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr")
        elif candidate_meta2.exists():
            meta_file = candidate_meta2
            base_dir = Path("IISc_RESPIN_test_mr")
        else:
            meta_file = candidate_meta1
            base_dir = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    decoder_modes = ["ctc", "rnnt"] if args.decoder == "both" else [args.decoder]

    evaluate_marathi(
        meta_file=meta_file,
        base_dir=base_dir,
        decoder_modes=decoder_modes,
        device=device,
        max_samples=args.max_samples,
        output_detailed_yaml_path=detailed_path,
        output_summary_yaml_path=summary_path,
        token=args.token,
    )


if __name__ == "__main__":
    main()
