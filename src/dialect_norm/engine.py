import json
import sys
from pathlib import Path
from .metrics import (
    normalize_text,
    compute_metrics_for_set,
    compute_group_metrics,
    compute_composite_group_metrics,
    compute_samplewise_analytics,
)
from .model import load_indic_conformer_model, run_decoder_inference
from .reporter import generate_yaml_reports


def evaluate_language(
    language_name: str,
    dataset_lang_code: str,
    model_lang_code: str,
    meta_file: Path,
    base_dir: Path,
    decoder_modes: list = ["ctc", "rnnt"],
    model_name: str = "ai4bharat/indic-conformer-600m-multilingual",
    device: str = "cuda",
    max_samples: int = None,
    output_dir: Path = Path("baseline-indic-conformer"),
    output_detailed_yaml_filename: str = None,
    output_summary_yaml_filename: str = None,
    token: str = None,
):
    """Unified engine to evaluate IndicConformer for any given language dataset."""
    print("\n" + "=" * 70)
    print(f"INDIC-CONFORMER EVALUATION: {language_name.upper()} ({dataset_lang_code.upper()})")
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

    model = load_indic_conformer_model(model_name=model_name, token=token, device=device)

    decoder_results = {}
    for dec in decoder_modes:
        predictions = run_decoder_inference(
            model=model,
            samples=samples,
            decoder_name=dec,
            model_lang_code=model_lang_code,
            device=device,
        )

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
        dialect_metrics = compute_group_metrics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm, "dialect")
        domain_metrics = compute_group_metrics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm, "domain")
        gender_metrics = compute_group_metrics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm, "gender")
        slab_metrics = compute_group_metrics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm, "slab")
        age_group_metrics = compute_group_metrics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm, "age_group")
        dialect_x_domain_metrics = compute_composite_group_metrics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm, "dialect", "domain")
        samplewise_analytics = compute_samplewise_analytics(samples, all_ref_raw, all_hyp_raw, all_ref_norm, all_hyp_norm)

        decoder_results[dec] = {
            "overall_metrics": overall_metrics,
            "dialect_breakdown": dialect_metrics,
            "domain_breakdown": domain_metrics,
            "gender_breakdown": gender_metrics,
            "slab_breakdown": slab_metrics,
            "age_group_breakdown": age_group_metrics,
            "dialect_x_domain_breakdown": dialect_x_domain_metrics,
            "samplewise_analytics": samplewise_analytics,
        }

    detailed_report, summary_report = generate_yaml_reports(
        language_name=language_name,
        dataset_lang_code=dataset_lang_code,
        model_lang_code=model_lang_code,
        model_name=model_name,
        meta_file=meta_file,
        samples=samples,
        total_in_meta=total_in_meta,
        total_duration_sec=total_duration_sec,
        decoder_results=decoder_results,
        output_dir=output_dir,
        output_detailed_filename=output_detailed_yaml_filename,
        output_summary_filename=output_summary_yaml_filename,
        device=device,
    )

    print("\n" + "=" * 70)
    print(f"INDIC-CONFORMER {language_name.upper()} EVALUATION SUMMARY REPORT")
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
