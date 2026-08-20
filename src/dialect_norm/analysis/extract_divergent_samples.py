"""
Pipeline script to score, rank, and extract parallel Marathi dialect samples (D1, D2, D4)
that exhibit maximum linguistic divergence / edit distance from Standard Pune Marathi.
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

from dialect_norm.metrics import normalize_text

logger = logging.getLogger("dialect_norm.analysis")


def setup_logger():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def extract_word_diffs(dialect_norm: str, standard_norm: str) -> Tuple[str, float]:
    """
    Compares normalized dialect words vs standard words to produce
    a human-readable difference string and calculate lexical substitution ratio.
    """
    d_words = dialect_norm.split()
    s_words = standard_norm.split()

    diff_tokens = []
    substitutions = 0

    # Fast word mismatch check
    d_set = set(d_words)
    s_set = set(s_words)
    diff_count = len(d_set.symmetric_difference(s_set))

    # Detailed alignment via jiwer
    try:
        word_output = jiwer.process_words(standard_norm, dialect_norm)
        alignment = word_output.alignments[0] if word_output.alignments else []

        for chunk in alignment:
            if chunk.type == "substitute":
                ref_w = " ".join([s_words[i] for i in range(chunk.ref_start_idx, min(chunk.ref_end_idx, len(s_words)))])
                hyp_w = " ".join([d_words[i] for i in range(chunk.hyp_start_idx, min(chunk.hyp_end_idx, len(d_words)))])
                diff_tokens.append(f"[{hyp_w} → {ref_w}]")
                substitutions += max(chunk.ref_end_idx - chunk.ref_start_idx, chunk.hyp_end_idx - chunk.hyp_start_idx)
            elif chunk.type == "delete":
                ref_w = " ".join([s_words[i] for i in range(chunk.ref_start_idx, min(chunk.ref_end_idx, len(s_words)))])
                diff_tokens.append(f"[DEL: {ref_w}]")
                substitutions += (chunk.ref_end_idx - chunk.ref_start_idx)
            elif chunk.type == "insert":
                hyp_w = " ".join([d_words[i] for i in range(chunk.hyp_start_idx, min(chunk.hyp_end_idx, len(d_words)))])
                diff_tokens.append(f"[INS: {hyp_w}]")
                substitutions += (chunk.hyp_end_idx - chunk.hyp_start_idx)
    except Exception:
        substitutions = diff_count

    diff_str = " ".join(diff_tokens[:8]) if diff_tokens else "Exact Match"
    lexical_ratio = (substitutions / max(1, len(s_words))) * 100.0
    return diff_str, round(lexical_ratio, 2)


def score_pair(dialect_raw: str, standard_raw: str, dialect_code: str, source_file: str) -> Dict:
    """Computes comprehensive divergence metrics for a dialect-standard parallel pair."""
    d_norm = normalize_text(dialect_raw)
    s_norm = normalize_text(standard_raw)

    if not d_norm or not s_norm:
        return None

    # Fast character and word length check
    if d_norm == s_norm:
        return None

    d_words = d_norm.split()
    s_words = s_norm.split()

    # Fast approximate WER calculation
    overlap = len(set(d_words).intersection(set(s_words)))
    approx_wer = ((len(s_words) - overlap + len(d_words) - overlap) / max(1, len(s_words))) * 100.0

    if approx_wer < 25.0:
        return None

    # Detailed WER and CER
    wer_val = float(jiwer.wer(s_norm, d_norm)) * 100.0
    cer_val = float(jiwer.cer(s_norm, d_norm)) * 100.0
    char_lev_dist = abs(len(d_norm) - len(s_norm))

    word_diff_summary, lex_ratio = extract_word_diffs(d_norm, s_norm)

    # Composite Divergence Score
    divergence_score = (0.50 * wer_val) + (0.30 * cer_val) + (0.20 * lex_ratio)

    return {
        "dialect": dialect_code,
        "divergence_score": round(divergence_score, 2),
        "wer_percentage": round(wer_val, 2),
        "cer_percentage": round(cer_val, 2),
        "lexical_diff_ratio": lex_ratio,
        "char_edit_distance": char_lev_dist,
        "dialect_text": dialect_raw,
        "standard_text": standard_raw,
        "word_diffs": word_diff_summary,
        "source_file": source_file,
    }


def load_dataset_pairs() -> Dict[str, List[Dict]]:
    """Loads parallel pairs grouped by dialect from all available repository data sources."""
    dialect_pairs = {"D1": [], "D2": [], "D4": []}

    # 1. 16k Original Synthetic Parallel Datasets
    files_16k = {
        "D1": Path("data/synthetic_parallel/d1.csv"),
        "D2": Path("data/synthetic_parallel/d2.csv"),
        "D4": Path("data/synthetic_parallel/d4.csv"),
    }

    for d_code, fpath in files_16k.items():
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d_txt = row.get("dialect_text", "").strip()
                    s_txt = row.get("standard_text", "").strip()
                    if d_txt and s_txt:
                        scored = score_pair(d_txt, s_txt, d_code, fpath.name)
                        if scored:
                            dialect_pairs[d_code].append(scored)

    # 2. 32k Expanded Augmented Datasets
    files_aug = {
        "D1": Path("data/synthetic-data/d1_aug.csv"),
        "D2": Path("data/synthetic-data/d2_aug.csv"),
        "D4": Path("data/synthetic-data/d4_aug.csv"),
    }

    for d_code, fpath in files_aug.items():
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d_txt = row.get("dialect_text", "").strip()
                    s_txt = row.get("standard_text", "").strip()
                    if d_txt and s_txt:
                        scored = score_pair(d_txt, s_txt, d_code, fpath.name)
                        if scored:
                            dialect_pairs[d_code].append(scored)

    # 3. Official RESPIN Test Set
    respin_meta1 = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr/meta_test_mr.json")
    respin_meta2 = Path("IISc_RESPIN_test_mr/meta_test_mr.json")
    respin_file = respin_meta1 if respin_meta1.exists() else respin_meta2

    if respin_file.exists():
        with open(respin_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            for item in meta.values():
                d_code = item.get("dialect", "").upper()
                txt = item.get("text", "").strip()
                if d_code in dialect_pairs and txt:
                    # For RESPIN, compare text vs normalized standard text
                    scored = score_pair(txt, txt, d_code, "meta_test_mr.json")
                    if scored and scored["wer_percentage"] > 0:
                        dialect_pairs[d_code].append(scored)

    return dialect_pairs


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Extract highly divergent Marathi dialect samples.")
    parser.add_argument("--top-k", type=int, default=150, help="Number of top divergent samples to extract per dialect")
    parser.add_argument("--min-wer", type=float, default=30.0, help="Minimum WER percentage threshold")
    parser.add_argument("--output-dir", type=str, default="reports/divergent_samples", help="Output directory for reports & CSVs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPUTATIONAL PIPELINE: HIGHLY DIVERGENT MARATHI DIALECT EXTRACTION")
    logger.info("=" * 80)

    dataset_pairs = load_dataset_pairs()

    summary_stats = {}
    all_top_samples = []

    dialect_names = {
        "D1": "D1 Malvani (South Konkan: Ratnagiri / Sindhudurg)",
        "D2": "D2 Ahirani (North Konkan / Khandesh: Palghar / Thane / Jalgaon)",
        "D4": "D4 Varhadi (Vidarbha: Amravati / Akola)",
    }

    for d_code, pairs in dataset_pairs.items():
        logger.info(f"Processing {len(pairs):,} parallel pairs for {d_code}...")

        # Filter by minimum WER threshold
        filtered = [p for p in pairs if p["wer_percentage"] >= args.min_wer]

        # Sort in descending order of divergence score
        filtered.sort(key=lambda x: x["divergence_score"], reverse=True)

        top_k = filtered[: args.top_k]
        all_top_samples.extend(top_k)

        # Save dialect-specific CSV
        csv_file = output_dir / f"{d_code.lower()}_top_divergent.csv"
        fieldnames = [
            "dialect",
            "divergence_score",
            "wer_percentage",
            "cer_percentage",
            "lexical_diff_ratio",
            "char_edit_distance",
            "dialect_text",
            "standard_text",
            "word_diffs",
            "source_file",
        ]

        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_k)

        avg_score = round(sum(p["divergence_score"] for p in top_k) / max(1, len(top_k)), 2)
        avg_wer = round(sum(p["wer_percentage"] for p in top_k) / max(1, len(top_k)), 2)
        avg_cer = round(sum(p["cer_percentage"] for p in top_k) / max(1, len(top_k)), 2)

        summary_stats[d_code] = {
            "name": dialect_names.get(d_code, d_code),
            "total_candidates": len(pairs),
            "threshold_candidates": len(filtered),
            "extracted_top_k": len(top_k),
            "avg_divergence_score": avg_score,
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "csv_path": str(csv_file),
            "top_samples": top_k[:10],  # For Markdown summary table
        }

        logger.info(f"  --> Extracted Top {len(top_k)} samples for {d_code} (Avg WER: {avg_wer}%, Avg CER: {avg_cer}%) Saved to: {csv_file}")

    # Save combined CSV
    all_top_samples.sort(key=lambda x: x["divergence_score"], reverse=True)
    all_csv_file = output_dir / "all_dialects_top_divergent.csv"
    with open(all_csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_top_samples)

    # Generate Markdown Summary Digest Report
    md_file = output_dir / "divergence_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Highly Divergent Marathi Dialect Extraction Report\n\n")
        f.write("**Pipeline Objective**: Computational extraction and ranking of sentence pairs from **D1 Malvani**, **D2 Ahirani**, and **D4 Varhadi** that exhibit maximum linguistic divergence / edit distance from Standard Pune Marathi for manual verification.\n\n")
        f.write("## 1. Summary Statistics & Extraction Metrics\n\n")
        f.write("| Dialect Code | Dialect Region Name | Total Candidate Pairs | Pairs meeting WER ≥ 30% | Extracted Chunk | Avg Divergence Score | Avg WER (%) | Avg CER (%) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for d_code, s in summary_stats.items():
            f.write(f"| **{d_code}** | {s['name']} | {s['total_candidates']:,} | {s['threshold_candidates']:,} | **{s['extracted_top_k']}** | **{s['avg_divergence_score']}** | **{s['avg_wer']}%** | **{s['avg_cer']}%** |\n")

        f.write("\n---\n\n")
        f.write("## 2. Top Divergent Samples per Dialect for Manual Verification\n\n")

        for d_code, s in summary_stats.items():
            f.write(f"### 📍 {s['name']} (Top 10 Most Divergent Samples)\n\n")
            f.write("| Rank | Score | WER (%) | CER (%) | Dialect Text (Input) | Standard Marathi (Target) | Highlighted Word Differences |\n")
            f.write("| :---: | :---: | :---: | :---: | :--- | :--- | :--- |\n")
            for idx, item in enumerate(s["top_samples"]):
                f.write(f"| **{idx+1}** | **{item['divergence_score']}** | {item['wer_percentage']}% | {item['cer_percentage']}% | `{item['dialect_text']}` | `{item['standard_text']}` | `{item['word_diffs']}` |\n")
            f.write("\n---\n\n")

        f.write("## 3. Exported Dataset Files for Annotation & Testing\n\n")
        f.write(f"* **D1 Malvani Top CSV**: [{summary_stats['D1']['csv_path']}](file:///{Path(summary_stats['D1']['csv_path']).resolve().as_posix()})\n")
        f.write(f"* **D2 Ahirani Top CSV**: [{summary_stats['D2']['csv_path']}](file:///{Path(summary_stats['D2']['csv_path']).resolve().as_posix()})\n")
        f.write(f"* **D4 Varhadi Top CSV**: [{summary_stats['D4']['csv_path']}](file:///{Path(summary_stats['D4']['csv_path']).resolve().as_posix()})\n")
        f.write(f"* **Combined All Dialects Top CSV**: [{all_csv_file}](file:///{all_csv_file.resolve().as_posix()})\n")

    logger.info("=" * 80)
    logger.info(f"COMPUTATIONAL PIPELINE COMPLETE! Report generated at: {md_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
