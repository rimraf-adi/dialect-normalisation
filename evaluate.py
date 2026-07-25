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
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.utils import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


from concurrent.futures import ThreadPoolExecutor


def load_single_audio(path):
    speech, sr = sf.read(path)
    return {"raw": speech, "sampling_rate": sr}


def threaded_audio_generator(audio_paths, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for audio_dict in executor.map(load_single_audio, audio_paths, chunksize=16):
            yield audio_dict


def normalize_text(text: str) -> str:
    """Normalizes text by removing Hindi and general punctuation and extra whitespace."""
    if not text:
        return ""
    punct_pattern = r'[।॥?!.,;:"\'\-\(\)\[\]\{\}—«»“”‘’]'
    text = re.sub(punct_pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
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


def evaluate_split(
    split_name: str,
    meta_file: Path,
    base_dir: Path,
    whisper_asr,
    batch_size: int = 32,
    num_workers: int = 8,
    max_samples: int = None,
    output_yaml_paths: list = None,
):
    print("\n" + "=" * 70)
    print(f"EVALUATING SPLIT: {split_name.upper()}")
    print("=" * 70)

    if not meta_file.exists():
        print(f"Error: Metadata file not found at {meta_file.resolve()}", file=sys.stderr)
        return None

    print(f"Loading metadata from: {meta_file}...")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    total_in_meta = len(meta_data)
    print(f"Total utterances available in {split_name} metadata: {total_in_meta}")

    samples = []
    audio_paths = []
    for idx, (key, item) in enumerate(meta_data.items()):
        if max_samples and idx >= max_samples:
            break
        rel_wav = item["wav_path"]
        full_wav_path = base_dir / rel_wav
        item["utt_key"] = key
        item["full_wav_path"] = str(full_wav_path)
        samples.append(item)
        audio_paths.append(str(full_wav_path))

    print(f"Evaluating on {len(samples)} utterances (batch_size={batch_size}, max_workers={num_workers}, FP16 GPU)...")
    start_time = time.time()

    predictions = []
    pipe_outputs = whisper_asr(threaded_audio_generator(audio_paths, max_workers=num_workers), batch_size=batch_size)
    for out in tqdm(pipe_outputs, total=len(audio_paths), desc=f"Transcribing ({split_name})"):
        predictions.append(out["text"])

    elapsed_time = time.time() - start_time
    print(f"Inference for {split_name} completed in {elapsed_time:.2f} seconds ({len(samples)/elapsed_time:.2f} UTT/s).")

    all_ref_raw, all_hyp_raw = [], []
    all_ref_norm, all_hyp_norm = [], []

    for idx, sample in enumerate(samples):
        ref_raw = sample["text"].strip()
        hyp_raw = predictions[idx].strip()
        ref_norm = normalize_text(ref_raw)
        hyp_norm = normalize_text(hyp_raw)

        sample["ref_raw"] = ref_raw
        sample["hyp_raw"] = hyp_raw
        sample["ref_norm"] = ref_norm
        sample["hyp_norm"] = hyp_norm

        all_ref_raw.append(ref_raw)
        all_hyp_raw.append(hyp_raw)
        all_ref_norm.append(ref_norm)
        all_hyp_norm.append(hyp_norm)

    total_duration_sec = sum(s.get("duration", 0.0) for s in samples)
    total_duration_hrs = float(round(total_duration_sec / 3600.0, 3))

    overall_metrics = compute_metrics_for_set(all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm)

    def compute_group_metrics(group_key):
        groups = {}
        for sample in samples:
            val = sample.get(group_key, "Unknown")
            if val not in groups:
                groups[val] = {"ref_raw": [], "hyp_raw": [], "ref_norm": [], "hyp_norm": [], "duration": 0.0}
            groups[val]["ref_raw"].append(sample["ref_raw"])
            groups[val]["hyp_raw"].append(sample["hyp_raw"])
            groups[val]["ref_norm"].append(sample["ref_norm"])
            groups[val]["hyp_norm"].append(sample["hyp_norm"])
            groups[val]["duration"] += sample.get("duration", 0.0)

        result_group = {}
        for name, data in sorted(groups.items()):
            res = compute_metrics_for_set(data["ref_raw"], data["hyp_raw"], data["ref_norm"], data["hyp_norm"])
            res["total_duration_hours"] = float(round(data["duration"] / 3600.0, 3))
            result_group[str(name)] = res
        return result_group

    def compute_composite_group_metrics(key1, key2):
        groups = {}
        for sample in samples:
            val1 = sample.get(key1, "Unknown")
            val2 = sample.get(key2, "Unknown")
            comp_key = f"{val1}_{val2}"
            if comp_key not in groups:
                groups[comp_key] = {"ref_raw": [], "hyp_raw": [], "ref_norm": [], "hyp_norm": [], "duration": 0.0}
            groups[comp_key]["ref_raw"].append(sample["ref_raw"])
            groups[comp_key]["hyp_raw"].append(sample["hyp_raw"])
            groups[comp_key]["ref_norm"].append(sample["ref_norm"])
            groups[comp_key]["hyp_norm"].append(sample["hyp_norm"])
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

    sample_predictions = []
    for s in samples[:10]:
        sample_predictions.append({
            "utterance_id": s.get("utterance_id", s.get("utt_key")),
            "dialect": s.get("dialect"),
            "domain": s.get("domain"),
            "reference": s["ref_raw"],
            "hypothesis": s["hyp_raw"],
            "normalized_reference": s["ref_norm"],
            "normalized_hypothesis": s["hyp_norm"],
            "raw_wer": float(round(jiwer.wer(s["ref_raw"], s["hyp_raw"]), 4)),
            "norm_wer": float(round(jiwer.wer(s["ref_norm"], s["hyp_norm"]), 4)),
        })

    report = {
        "model_info": {
            "model_name": "IndicWhisper (whisper-medium-hi_alldata_multigpu)",
            "model_path": "hindi_models/whisper-medium-hi_alldata_multigpu",
            "language": "hi",
            "device": "cuda",
            "precision": "float16",
        },
        "dataset_info": {
            "split": split_name,
            "metadata_file": str(meta_file),
            "total_utterances_evaluated": len(samples),
            "total_utterances_in_meta": total_in_meta,
            "total_duration_hours": total_duration_hrs,
            "avg_duration_seconds": float(round(total_duration_sec / len(samples), 2)) if samples else 0.0,
        },
        "evaluation_summary": {
            "raw_wer_percentage": overall_metrics["raw"]["wer_percentage"],
            "normalized_wer_percentage": overall_metrics["normalized"]["wer_percentage"],
            "raw_cer_percentage": overall_metrics["raw"]["cer_percentage"],
            "normalized_cer_percentage": overall_metrics["normalized"]["cer_percentage"],
            "raw_ser_percentage": overall_metrics["raw"]["ser_percentage"],
            "normalized_ser_percentage": overall_metrics["normalized"]["ser_percentage"],
            "normalized_exact_match_acc_percentage": overall_metrics["normalized"]["exact_match_acc_percentage"],
        },
        "overall_metrics": overall_metrics,
        "dialect_breakdown": dialect_metrics,
        "domain_breakdown": domain_metrics,
        "gender_breakdown": gender_metrics,
        "slab_breakdown": slab_metrics,
        "age_group_breakdown": age_group_metrics,
        "dialect_x_domain_breakdown": dialect_x_domain_metrics,
        "sample_predictions": sample_predictions,
    }

    clean_report = to_python_types(report)

    if output_yaml_paths:
        for out_path in output_yaml_paths:
            out_file = Path(out_path).resolve()
            with open(out_file, "w", encoding="utf-8") as f:
                yaml.dump(clean_report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            print(f"Saved YAML report to: {out_file}")

    print("\n" + "=" * 70)
    print(f"INDICWHISPER EVALUATION REPORT ({split_name.upper()} SPLIT)")
    print("=" * 70)
    print(f"Total Utterances evaluated : {len(samples)}")
    print(f"Total Audio Duration      : {total_duration_hrs} hours")
    print(f"RAW WER                    : {overall_metrics['raw']['wer_percentage']}% (WER={overall_metrics['raw']['wer']})")
    print(f"NORMALIZED WER             : {overall_metrics['normalized']['wer_percentage']}% (WER={overall_metrics['normalized']['wer']})")
    print(f"RAW CER                    : {overall_metrics['raw']['cer_percentage']}%")
    print(f"NORMALIZED CER             : {overall_metrics['normalized']['cer_percentage']}%")
    print(f"NORMALIZED Exact Match Acc : {overall_metrics['normalized']['exact_match_acc_percentage']}%")
    print("-" * 70)
    print("DIALECT-WISE NORMALIZED WER:")
    for dialect, ddata in dialect_metrics.items():
        print(f"  - {dialect:4s}: WER = {ddata['normalized']['wer_percentage']:5.2f}% | CER = {ddata['normalized']['cer_percentage']:5.2f}% | Utts = {ddata['num_samples']}")
    print("-" * 70)
    print("DOMAIN-WISE NORMALIZED WER:")
    for domain, dmdata in domain_metrics.items():
        print(f"  - {domain:12s}: WER = {dmdata['normalized']['wer_percentage']:5.2f}% | CER = {dmdata['normalized']['cer_percentage']:5.2f}% | Utts = {dmdata['num_samples']}")
    print("=" * 70)

    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate IndicWhisper on RESPIN Hindi Dataset.")
    parser.add_argument("--test", action="store_true", help="Evaluate on the test set split")
    parser.add_argument("--train", action="store_true", help="Evaluate on the train set split")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for pipeline inference (default: 16)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of PyTorch DataLoader worker processes (default: 0 for Windows compatibility)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of utterances to evaluate per split")

    args = parser.parse_args()

    if not args.train and not args.test:
        args.test = True

    model_path = "hindi_models/whisper-medium-hi_alldata_multigpu"
    device = "cuda"
    lang_code = "hi"

    print("=" * 70)
    print("IndicWhisper Accelerated Evaluation Pipeline")
    print("=" * 70)
    print(f"Loading FP16 model pipeline from: {model_path} on device: {device}...")
    generate_kwargs = {"task": "transcribe", "language": lang_code if lang_code != "or" else None}
    whisper_asr = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        dtype=torch.float16,
        generate_kwargs=generate_kwargs,
    )

    if hasattr(whisper_asr.model.config, "forced_decoder_ids"):
        whisper_asr.model.config.forced_decoder_ids = None
    if hasattr(whisper_asr.model, "generation_config") and whisper_asr.model.generation_config:
        whisper_asr.model.generation_config.forced_decoder_ids = None

    if lang_code == 'or':
        whisper_asr.model.config.forced_decoder_ids = (
            whisper_asr.tokenizer.get_decoder_prompt_ids(
                language=None, task="transcribe"
            )
        )

    if args.test:
        meta_test = Path("IISc_RESPIN_test_hi/meta_test_hi.json")
        base_test = Path("IISc_RESPIN_test_hi")
        out_paths = ["indicwhisper_test.yaml", "indicwhisper.yaml"]
        evaluate_split("test", meta_test, base_test, whisper_asr, batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples, output_yaml_paths=out_paths)

    if args.train:
        meta_train = Path("IISc_RESPIN_train_hi_clean/meta_train_hi_clean.json")
        base_train = Path("IISc_RESPIN_train_hi_clean")
        out_paths = ["indicwhisper_train.yaml"]
        evaluate_split("train", meta_train, base_train, whisper_asr, batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples, output_yaml_paths=out_paths)


if __name__ == "__main__":
    main()
