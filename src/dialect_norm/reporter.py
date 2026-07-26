from pathlib import Path
import yaml


def to_python_types(obj):
    """Converts numpy / PyTorch types to native Python types for clean YAML serialization."""
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


def generate_yaml_reports(
    language_name: str,
    dataset_lang_code: str,
    model_lang_code: str,
    model_name: str,
    meta_file: Path,
    samples: list,
    total_in_meta: int,
    total_duration_sec: float,
    decoder_results: dict,
    output_dir: Path,
    output_detailed_filename: str = None,
    output_summary_filename: str = None,
    device: str = "cuda",
) -> tuple:
    """Builds and writes Detailed and Summary YAML reports to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = language_name.lower().replace(" ", "_")
    if not output_detailed_filename:
        output_detailed_filename = f"indic_conformer_{slug}_detailed.yaml"
    if not output_summary_filename:
        output_summary_filename = f"indic_conformer_{slug}_summary.yaml"

    output_detailed_path = output_dir / output_detailed_filename
    output_summary_path = output_dir / output_summary_filename

    total_duration_hrs = float(round(total_duration_sec / 3600.0, 3))

    # 1. DETAILED Report
    detailed_report = {
        "model_info": {
            "model_name": model_name,
            "language_name": language_name,
            "dataset_lang_code": dataset_lang_code,
            "model_lang_code": model_lang_code,
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

    # 2. SMALL SUMMARY Report
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
    with open(output_detailed_path, "w", encoding="utf-8") as f:
        yaml.dump(clean_detailed_report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"\n[Detailed YAML Report] Saved to: {output_detailed_path}")

    # Save Small Summary YAML Report
    with open(output_summary_path, "w", encoding="utf-8") as f:
        yaml.dump(clean_summary_report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"[Small Summary YAML Report] Saved to: {output_summary_path}")

    return detailed_report, summary_report
