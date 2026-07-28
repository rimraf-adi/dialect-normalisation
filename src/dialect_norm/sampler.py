"""
Dataset Sampler for Marathi Dialects (D1, D2, D4).
Samples 10,000 random/distorted utterances per dialect (30,000 total).
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List
from .distortion import compute_dialect_distortion_score

logger = logging.getLogger("dialect_norm.sampler")


def load_and_sample_dialect_dataset(
    meta_path: Path,
    samples_per_dialect: int = 10000,
    target_dialects: List[str] = None,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Loads RESPIN Marathi Train Set metadata, computes distortion scores for sentences,
    and samples `samples_per_dialect` items for each specified dialect with detailed logging.
    """
    if target_dialects is None:
        target_dialects = ["D1", "D2", "D4"]

    logger.info(f"[Sampler] Reading metadata file: {meta_path.resolve()}...")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = list(data.values()) if isinstance(data, dict) else data
    logger.info(f"[Sampler] Loaded {len(entries):,} total utterances from metadata.")

    # Group unique sentences per dialect
    dialect_buckets: Dict[str, Dict[str, dict]] = {d: {} for d in target_dialects}

    for e in entries:
        dial = e.get("dialect", "")
        txt = e.get("text", "").strip()
        tid = e.get("text_id", "")
        dom = e.get("domain", "")

        if dial in dialect_buckets and txt and txt not in dialect_buckets[dial]:
            score = compute_dialect_distortion_score(txt, dialect_code=dial)
            dialect_buckets[dial][txt] = {
                "text_id": tid,
                "dialect": dial,
                "domain": dom,
                "dialect_text": txt,
                "distortion_score": score,
            }

    sampled_dataset: Dict[str, List[dict]] = {}
    random.seed(seed)

    logger.info(f"[Sampler] Target: Sampling {samples_per_dialect:,} items per dialect (Target Dialects: {', '.join(target_dialects)})...")

    for dial in target_dialects:
        available_items = list(dialect_buckets[dial].values())
        logger.info(f"  --> Dialect {dial}: {len(available_items):,} total unique sentences extracted.")

        # Sort by distortion score (descending) to prioritize distorted dialect sentences
        available_items.sort(key=lambda x: x["distortion_score"], reverse=True)

        if len(available_items) >= samples_per_dialect:
            # Take top 60% by distortion score + random sample remaining 40% for diversity
            top_cutoff = int(samples_per_dialect * 0.6)
            top_items = available_items[:top_cutoff]
            remaining = available_items[top_cutoff:]

            random.shuffle(remaining)
            random_items = remaining[:(samples_per_dialect - top_cutoff)]

            selected = top_items + random_items
            random.shuffle(selected)
            sampled_dataset[dial] = selected
        else:
            logger.warning(f"      Dialect {dial} has fewer than {samples_per_dialect} unique sentences. Selecting all {len(available_items)}.")
            sampled_dataset[dial] = available_items

        avg_dist = sum(x['distortion_score'] for x in sampled_dataset[dial]) / len(sampled_dataset[dial])
        max_dist = max(x['distortion_score'] for x in sampled_dataset[dial])
        min_dist = min(x['distortion_score'] for x in sampled_dataset[dial])

        logger.info(f"  --> Dialect {dial} Selected: {len(sampled_dataset[dial]):,} samples | Distortion Score (Avg: {avg_dist:.2f}, Min: {min_dist:.2f}, Max: {max_dist:.2f})")

        # Log 2 sample sentences per dialect
        for idx_sample, sample_item in enumerate(sampled_dataset[dial][:2], 1):
            logger.info(f"      [Sample {dial}-{idx_sample}] (Score: {sample_item['distortion_score']:.1f}) [{sample_item['domain']}]: '{sample_item['dialect_text']}'")

    total_sampled = sum(len(v) for v in sampled_dataset.values())
    logger.info(f"[Sampler] Final Dataset Sampled Total: {total_sampled:,} samples across {len(target_dialects)} dialects.")
    return sampled_dataset
