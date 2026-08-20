"""
Fast and comprehensive evaluation script to benchmark IndicBART and mT5 model variants
on the official IISc_RESPIN_test_mr test set with granular per-dialect breakdowns.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import evaluate
import jiwer
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration

from dialect_norm.metrics import normalize_text

logger = logging.getLogger("dialect_norm.respin_eval")


def setup_logger():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_respin_test_data() -> Dict[str, List[Dict]]:
    """Loads and groups IISc_RESPIN_test_mr metadata into dialect partitions."""
    p1 = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr/meta_test_mr.json")
    p2 = Path("IISc_RESPIN_test_mr/meta_test_mr.json")
    meta_path = p1 if p1.exists() else p2

    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot locate RESPIN test metadata at {p1} or {p2}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    groups = {"D1": [], "D2": [], "D3": [], "D4": []}
    for item in meta.values():
        d_code = item.get("dialect", "").upper()
        if d_code in groups:
            groups[d_code].append(item)

    logger.info(f"Loaded RESPIN Test Set: D1 ({len(groups['D1'])}), D2 ({len(groups['D2'])}), D3 ({len(groups['D3'])}), D4 ({len(groups['D4'])}) => Total {sum(len(v) for v in groups.values())} utterances")
    return groups


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Computes BLEU, chrF++, WER, CER, Exact Match."""
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    norm_preds = [normalize_text(p) for p in predictions]
    norm_refs = [normalize_text(r) for r in references]

    refs_nested = [[r] for r in norm_refs]
    bleu_res = sacrebleu.compute(predictions=norm_preds, references=refs_nested)
    chrf_res = chrf.compute(predictions=norm_preds, references=refs_nested, word_order=2)

    wer_val = float(jiwer.wer(norm_refs, norm_preds)) * 100.0
    cer_val = float(jiwer.cer(norm_refs, norm_preds)) * 100.0

    exact_matches = sum(1 for p, r in zip(norm_preds, norm_refs) if p == r)
    exact_acc = round((exact_matches / max(1, len(norm_refs))) * 100.0, 2)

    return {
        "num_samples": len(norm_refs),
        "bleu": round(bleu_res["score"], 2),
        "chrf": round(chrf_res["score"], 2),
        "wer": round(wer_val, 2),
        "cer": round(cer_val, 2),
        "exact_match_acc": exact_acc,
    }


def generate_predictions_batch(model, tokenizer, texts: List[str], is_indicbart: bool, device: str, batch_size: int = 64) -> List[str]:
    """Generates batch predictions with high GPU throughput and exact tokenizer conditioning."""
    predictions = []
    model.eval()

    target_lang_id = None
    if is_indicbart:
        target_lang_id = tokenizer.convert_tokens_to_ids("<2mr>")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if is_indicbart:
            # IndicBART requires <2mr> input tags exactly matching training format
            batch_inputs = [f"<2mr> {t} <2mr>" for t in batch]
            enc = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outs = model.generate(
                    **enc,
                    forced_bos_token_id=target_lang_id,
                    max_length=128,
                    num_beams=4,
                )
            dec = [tokenizer.decode(o, skip_special_tokens=True).replace("<2mr>", "").strip() for o in outs]
        else:
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outs = model.generate(
                    **enc,
                    max_length=128,
                    num_beams=4,
                )
            dec = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outs]

        predictions.extend(dec)

    return predictions


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Fast RESPIN Evaluation for IndicBART and mT5 Models.")
    parser.add_argument("--output-dir", type=str, default="reports/respin_eval", help="Output directory for reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    respin_groups = load_respin_test_data()

    models_to_eval = [
        {
            "name": "IndicBART Combined 16k",
            "key": "indicbart_combined_16k",
            "path": "models/indicbart_combined/fold_1/best_model",
            "is_indicbart": True,
        },
        {
            "name": "IndicBART Combined 32k",
            "key": "indicbart_combined_32k",
            "path": "models/indicbart_combined_32k/fold_1/best_model",
            "is_indicbart": True,
        },
        {
            "name": "IndicBART Raw Unverified 32k",
            "key": "indicbart_raw_unverified_32k",
            "path": "models/indicbart_raw_unverified_32k/fold_1/best_model",
            "is_indicbart": True,
        },
        {
            "name": "mT5-Small Combined 16k",
            "key": "mt5_combined_16k",
            "path": "models/mt5_combined_16k/best_model",
            "is_indicbart": False,
        },
        {
            "name": "mT5-Small Combined 32k",
            "key": "mt5_combined_32k",
            "path": "models/mt5_combined_32k/best_model",
            "is_indicbart": False,
        },
        {
            "name": "mT5-Small Raw Unverified 32k",
            "key": "mt5_raw_unverified_32k",
            "path": "models/mt5_raw_unverified_32k/best_model",
            "is_indicbart": False,
        },
    ]

    all_results = {}

    for m_cfg in models_to_eval:
        m_name = m_cfg["name"]
        m_path = Path(m_cfg["path"])

        if not m_path.exists():
            logger.warning(f"Model path {m_path} does not exist. Skipping {m_name}...")
            continue

        logger.info("=" * 80)
        logger.info(f"EVALUATING MODEL: {m_name} ({m_path})")
        logger.info("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(str(m_path))
        if m_cfg["is_indicbart"]:
            model = MBartForConditionalGeneration.from_pretrained(str(m_path)).to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(m_path)).to(device)

        m_results = {}
        all_preds = []
        all_refs = []

        for d_code in ["D1", "D2", "D3", "D4"]:
            items = respin_groups[d_code]
            texts = [it["text"] for it in items]
            preds = generate_predictions_batch(model, tokenizer, texts, m_cfg["is_indicbart"], device)

            res = compute_metrics(preds, texts)
            m_results[d_code] = res
            logger.info(f"  --> {d_code} ({len(items)} samples): BLEU: {res['bleu']} | chrF++: {res['chrf']} | WER: {res['wer']}% | Exact Match: {res['exact_match_acc']}%")

            all_preds.extend(preds)
            all_refs.extend(texts)

        # Compute Combined metrics across all 2,170 samples instantly from concatenated predictions
        comb_res = compute_metrics(all_preds, all_refs)
        m_results["Combined"] = comb_res
        logger.info(f"  --> Combined (2,170 samples): BLEU: {comb_res['bleu']} | chrF++: {comb_res['chrf']} | WER: {comb_res['wer']}% | Exact Match: {comb_res['exact_match_acc']}%")

        all_results[m_cfg["key"]] = {
            "name": m_name,
            "path": str(m_path),
            "results": m_results,
        }

        # Clean GPU memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save JSON and YAML
    json_path = output_dir / "respin_all_models_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    yaml_path = output_dir / "respin_all_models_eval.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(all_results, f, allow_unicode=True)

    # Generate Comprehensive Markdown Table Report
    md_file = output_dir / "respin_models_benchmark_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Official IISc RESPIN Held-Out Spoken Test Set Benchmark Report\n\n")
        f.write("**Test Dataset**: `IISc_RESPIN_test_mr` (2,170 Total Spoken Utterances)\n")
        f.write("**Evaluation Metrics**: BLEU (sacrebleu), chrF++ (word_order=2), Word Error Rate (WER %), Character Error Rate (CER %), Exact Match Accuracy (%)\n\n")
        f.write("---\n\n")

        # 1. Overall Combined Test Performance
        f.write("## 1. Overall Combined Test Performance (2,170 Utterances)\n\n")
        f.write("| Model Variant | Training Scale / Quality | Checkpoint Path | BLEU Score | chrF++ Score | Word Error Rate (WER %) | Char Error Rate (CER %) | Exact Match Acc (%) |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for k, data in all_results.items():
            comb = data["results"].get("Combined", {})
            f.write(f"| **{data['name']}** | `{k}` | `{data['path']}` | **{comb.get('bleu', 'N/A')}** | **{comb.get('chrf', 'N/A')}** | **{comb.get('wer', 'N/A')}%** | {comb.get('cer', 'N/A')}% | {comb.get('exact_match_acc', 'N/A')}% |\n")

        f.write("\n---\n\n")

        # 2. Per-Dialect Granular Breakdown (D1 Malvani, D2 Ahirani, D3 Standard, D4 Varhadi)
        f.write("## 2. Granular Per-Dialect Breakdown across all Models\n\n")
        f.write("| Model Variant | D1 Malvani (559 utts)<br>BLEU / WER / chrF++ | D2 Ahirani (540 utts)<br>BLEU / WER / chrF++ | D3 Standard (555 utts)<br>BLEU / WER / chrF++ | D4 Varhadi (516 utts)<br>BLEU / WER / chrF++ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for k, data in all_results.items():
            d1 = data["results"].get("D1", {})
            d2 = data["results"].get("D2", {})
            d3 = data["results"].get("D3", {})
            d4 = data["results"].get("D4", {})
            f.write(f"| **{data['name']}** | {d1.get('bleu', 'N/A')} / {d1.get('wer', 'N/A')}% / {d1.get('chrf', 'N/A')} | {d2.get('bleu', 'N/A')} / {d2.get('wer', 'N/A')}% / {d2.get('chrf', 'N/A')} | {d3.get('bleu', 'N/A')} / {d3.get('wer', 'N/A')}% / {d3.get('chrf', 'N/A')} | {d4.get('bleu', 'N/A')} / {d4.get('wer', 'N/A')}% / {d4.get('chrf', 'N/A')} |\n")

        f.write("\n---\n\n")

    logger.info("=" * 80)
    logger.info(f"RESPIN MODEL EVALUATION COMPLETE! Saved to: {md_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
