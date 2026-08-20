"""
Extracts highly divergent original spoken RESPIN dialect transcripts (D1, D2, D4)
by computing normalization edit distance (WER/CER) via mT5 and Standard Marathi lexical divergence.
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dialect_norm.metrics import normalize_text

logger = logging.getLogger("dialect_norm.respin_analysis")


def setup_logger():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_respin_metadata() -> Tuple[List[Dict], List[str]]:
    """Loads all RESPIN test utterances and extracts D3 Standard Marathi reference vocab."""
    meta1 = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr/meta_test_mr.json")
    meta2 = Path("IISc_RESPIN_test_mr/meta_test_mr.json")
    meta_path = meta1 if meta1.exists() else meta2

    if not meta_path.exists():
        logger.error(f"Metadata file not found at {meta_path}")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    utterances = []
    d3_words = set()

    for key, item in meta.items():
        item["key"] = key
        utterances.append(item)
        if item.get("dialect") == "D3":
            norm_t = normalize_text(item.get("text", ""))
            d3_words.update(norm_t.split())

    return utterances, d3_words


def compute_standard_oov_ratio(text: str, d3_vocab: set) -> float:
    """Computes the ratio of words in text that do not appear in Standard Marathi (D3) corpus."""
    words = normalize_text(text).split()
    if not words:
        return 0.0
    oov_count = sum(1 for w in words if w not in d3_vocab)
    return round((oov_count / len(words)) * 100.0, 2)


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Extract highly divergent original RESPIN dialect transcripts.")
    parser.add_argument("--model-path", type=str, default="models/mt5_combined_32k/best_model", help="Path to fine-tuned normalizer model")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top samples to extract per dialect")
    parser.add_argument("--output-dir", type=str, default="reports/respin_divergent_samples", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("RESPIN ORIGINAL TRANSCRIPTS: COMPUTATIONAL DIALECT DIVERGENCE PIPELINE")
    logger.info("=" * 80)

    utterances, d3_vocab = load_respin_metadata()
    logger.info(f"Loaded {len(utterances):,} total RESPIN utterances. Built Standard Marathi D3 Vocab ({len(d3_vocab):,} words).")

    # Load fine-tuned normalizer model if available, else fall back to lexical distance
    model_path = Path(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = None
    model = None

    if model_path.exists():
        logger.info(f"Loading fine-tuned normalizer model from {model_path} on {device}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            model.eval()
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}. Running lexical divergence scoring.")

    # Group utterances by dialect
    dialect_groups = {"D1": [], "D2": [], "D4": []}
    for item in utterances:
        d_code = item.get("dialect", "").upper()
        if d_code in dialect_groups:
            dialect_groups[d_code].append(item)

    summary_stats = {}
    all_top = []

    dialect_names = {
        "D1": "D1 Malvani (South Konkan: Ratnagiri / Sindhudurg)",
        "D2": "D2 Ahirani (North Konkan / Khandesh: Palghar / Thane / Jalgaon)",
        "D4": "D4 Varhadi (Vidarbha: Amravati / Akola)",
    }

    for d_code, items in dialect_groups.items():
        logger.info(f"Analyzing {len(items)} original RESPIN transcripts for {d_code}...")

        # Deduplicate by unique text to show distinct linguistic utterances
        seen_texts = set()
        unique_items = []
        for it in items:
            t_norm = normalize_text(it["text"])
            if t_norm and t_norm not in seen_texts:
                seen_texts.add(t_norm)
                unique_items.append(it)

        logger.info(f"  -> {len(unique_items)} unique spoken sentences found in {d_code}")

        texts = [it["text"] for it in unique_items]

        # Generate normalized Standard Marathi predictions if model is loaded
        normalized_preds = []
        if model and tokenizer:
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_txts = texts[i : i + batch_size]
                enc = tokenizer(batch_txts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    outs = model.generate(**enc, max_length=128)
                dec = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
                normalized_preds.extend(dec)
        else:
            normalized_preds = texts

        scored_items = []
        for idx, item in enumerate(unique_items):
            orig_txt = item["text"]
            norm_pred = normalized_preds[idx]

            d_norm = normalize_text(orig_txt)
            s_norm = normalize_text(norm_pred)

            d_words = d_norm.split()
            s_words = s_norm.split()

            if not d_words or not s_words:
                continue

            # Standard Levenshtein word edit distance bounded in [0, 100%]
            word_ops = jiwer.process_words(s_norm, d_norm)
            edits = word_ops.substitutions + word_ops.deletions + word_ops.insertions
            bounded_wer = min(100.0, round((edits / max(len(d_words), len(s_words))) * 100.0, 2))

            char_ops = jiwer.process_characters(s_norm, d_norm)
            char_edits = char_ops.substitutions + char_ops.deletions + char_ops.insertions
            bounded_cer = min(100.0, round((char_edits / max(len(d_norm), len(s_norm))) * 100.0, 2))

            oov_ratio = compute_standard_oov_ratio(orig_txt, d3_vocab)

            # Composite Divergence Score bounded in [0, 100]
            div_score = round((0.45 * bounded_wer) + (0.30 * bounded_cer) + (0.25 * oov_ratio), 2)

            scored_items.append({
                "key": item["key"],
                "dialect": d_code,
                "domain": item.get("domain", ""),
                "gender": item.get("gender", ""),
                "divergence_score": div_score,
                "wer_percentage": bounded_wer,
                "cer_percentage": bounded_cer,
                "standard_oov_ratio": oov_ratio,
                "original_transcript": orig_txt,
                "normalized_standard_output": norm_pred,
                "wav_path": item.get("wav_path", ""),
            })

        # Sort in descending order of divergence
        scored_items.sort(key=lambda x: x["divergence_score"], reverse=True)
        top_k = scored_items[: args.top_k]
        all_top.extend(top_k)

        # Save CSV
        csv_path = output_dir / f"respin_{d_code.lower()}_top_divergent.csv"
        fieldnames = [
            "key",
            "dialect",
            "domain",
            "gender",
            "divergence_score",
            "wer_percentage",
            "cer_percentage",
            "standard_oov_ratio",
            "original_transcript",
            "normalized_standard_output",
            "wav_path",
        ]

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_k)

        avg_score = round(sum(p["divergence_score"] for p in top_k) / max(1, len(top_k)), 2)
        avg_wer = round(sum(p["wer_percentage"] for p in top_k) / max(1, len(top_k)), 2)
        avg_oov = round(sum(p["standard_oov_ratio"] for p in top_k) / max(1, len(top_k)), 2)

        summary_stats[d_code] = {
            "name": dialect_names.get(d_code, d_code),
            "total_utterances": len(items),
            "extracted_top_k": len(top_k),
            "avg_divergence_score": avg_score,
            "avg_wer": avg_wer,
            "avg_oov_ratio": avg_oov,
            "csv_path": str(csv_path),
            "top_samples": top_k[:10],
        }

        logger.info(f"  --> Extracted Top {len(top_k)} RESPIN transcripts for {d_code} (Avg WER: {avg_wer}%, Avg OOV: {avg_oov}%) Saved to: {csv_path}")

    # Generate Markdown Summary Digest Report
    md_file = output_dir / "respin_divergence_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# RESPIN Spoken Dialect Transcripts: Divergence Analysis Report\n\n")
        f.write("**Objective**: Computational extraction and ranking of original spoken RESPIN test transcripts (**D1 Malvani**, **D2 Ahirani**, and **D4 Varhadi**) that diverge most significantly from Standard Pune Marathi norms.\n\n")
        f.write("## 1. Summary Statistics & Extraction Metrics\n\n")
        f.write("| Dialect Code | Dialect Region Name | Total Spoken Utterances | Extracted Top Chunk | Avg Divergence Score | Normalization WER (%) | Standard OOV Ratio (%) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for d_code, s in summary_stats.items():
            f.write(f"| **{d_code}** | {s['name']} | {s['total_utterances']} | **{s['extracted_top_k']}** | **{s['avg_divergence_score']}** | **{s['avg_wer']}%** | **{s['avg_oov_ratio']}%** |\n")

        f.write("\n---\n\n")
        f.write("## 2. Top Most Divergent RESPIN Transcripts per Dialect\n\n")

        for d_code, s in summary_stats.items():
            f.write(f"### 📍 {s['name']} (Top 10 Spoken Transcripts)\n\n")
            f.write("| Rank | Score | Normalization WER (%) | OOV Ratio (%) | Original Spoken RESPIN Transcript | Normalized Standard Output |\n")
            f.write("| :---: | :---: | :---: | :---: | :--- | :--- |\n")
            for idx, item in enumerate(s["top_samples"]):
                f.write(f"| **{idx+1}** | **{item['divergence_score']}** | {item['wer_percentage']}% | {item['standard_oov_ratio']}% | `{item['original_transcript']}` | `{item['normalized_standard_output']}` |\n")
            f.write("\n---\n\n")

    logger.info("=" * 80)
    logger.info(f"RESPIN PIPELINE COMPLETE! Report generated at: {md_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
