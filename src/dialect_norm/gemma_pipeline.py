"""
Gemma 4:12b Synthetic Data Generation Pipeline Engine with Detailed Logging.
Batches 5 sentences per prompt and saves output in 1,000-row CSV files.
"""

import csv
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:12b"
ITEMS_PER_PROMPT = 5
ROWS_PER_CSV = 1000

logger = logging.getLogger("dialect_norm.gemma_pipeline")

PROMPT_TEMPLATE = """You are an expert linguist specializing in Marathi dialects.
Translate the following 5 regional Marathi dialect sentences into Standard Pune Marathi (शुद्ध पुणेरी मराठी).
Preserve the exact meaning, but convert all regional dialectal words, suffixes, and non-standard grammar into standard formal Marathi.

Input Dialect Sentences:
1. {text_1}
2. {text_2}
3. {text_3}
4. {text_4}
5. {text_5}

Respond ONLY with a valid JSON array of 5 strings containing the Standard Marathi translations in order:
[
  "Standard sentence 1",
  "Standard sentence 2",
  "Standard sentence 3",
  "Standard sentence 4",
  "Standard sentence 5"
]"""


def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "gemma_pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)

    logger.info(f"Detailed logging initialized. Log file: {log_file.resolve()}")


def query_ollama_batch(batch_items: List[dict], batch_idx: int, total_batches: int, model: str = MODEL_NAME, base_url: str = OLLAMA_URL) -> List[str]:
    """Sends 5 sentences in a single prompt to Ollama gemma4:12b and parses response with detailed logging."""
    texts = [item["dialect_text"] for item in batch_items]
    while len(texts) < ITEMS_PER_PROMPT:
        texts.append(texts[-1])

    prompt_content = PROMPT_TEMPLATE.format(
        text_1=texts[0],
        text_2=texts[1],
        text_3=texts[2],
        text_4=texts[3],
        text_5=texts[4],
    )

    payload = {
        "model": model,
        "prompt": prompt_content,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    start_time = time.time()
    logger.debug(f"[Batch {batch_idx}/{total_batches}] Sending prompt with {len(batch_items)} dialect items to Ollama ({model})...")

    try:
        req = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            elapsed = time.time() - start_time
            if resp.status == 200:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw_response = res_data.get("response", "").strip()

                # Attempt JSON array parsing
                json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        if isinstance(parsed, list) and len(parsed) >= len(batch_items):
                            logger.info(f"[Batch {batch_idx}/{total_batches}] SUCCESS in {elapsed:.2f}s | Parsed 5 JSON translations cleanly.")
                            return [str(x).strip() for x in parsed[:len(batch_items)]]
                    except json.JSONDecodeError:
                        logger.warning(f"[Batch {batch_idx}/{total_batches}] JSON regex matched but decode failed. Falling back to line parsing.")

                # Fallback line parsing
                lines = [line.strip() for line in raw_response.split("\n") if line.strip() and not line.startswith("[") and not line.startswith("]")]
                cleaned_lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip(' "') for line in lines]
                if len(cleaned_lines) >= len(batch_items):
                    logger.info(f"[Batch {batch_idx}/{total_batches}] SUCCESS (Fallback Line Parse) in {elapsed:.2f}s | Extracted {len(cleaned_lines)} lines.")
                    return cleaned_lines[:len(batch_items)]
                else:
                    logger.warning(f"[Batch {batch_idx}/{total_batches}] Response length mismatch ({len(cleaned_lines)} lines vs {len(batch_items)} expected). Raw response snippet: '{raw_response[:100]}...'")

    except urllib.error.URLError as e:
        elapsed = time.time() - start_time
        logger.error(f"[Batch {batch_idx}/{total_batches}] Connection Error after {elapsed:.2f}s: {e}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Batch {batch_idx}/{total_batches}] Unexpected Error after {elapsed:.2f}s: {e}")

    return None


def execute_generation_pipeline(
    sampled_dataset: Dict[str, List[dict]],
    output_dir: Path,
    log_dir: Path = Path("logs"),
    rows_per_csv: int = ROWS_PER_CSV,
):
    setup_logger(log_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "pipeline_checkpoint.json"

    # Flatten and interleave dialect items (D1, D2, D4)
    flattened_items = []
    dialects = list(sampled_dataset.keys())
    max_len = max(len(v) for v in sampled_dataset.values())

    for idx in range(max_len):
        for dial in dialects:
            if idx < len(sampled_dataset[dial]):
                flattened_items.append(sampled_dataset[dial][idx])

    total_target = len(flattened_items)
    total_batches = (total_target + ITEMS_PER_PROMPT - 1) // ITEMS_PER_PROMPT

    # Checkpoint recovery
    processed_count = 0
    current_csv_index = 1
    current_csv_rows = []

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                processed_count = ckpt.get("processed_count", 0)
                current_csv_index = ckpt.get("current_csv_index", 1)
                logger.info(f"[Checkpoint] Loaded existing checkpoint: {processed_count:,} pairs processed. Starting at CSV Part {current_csv_index:03d}.")
        except Exception as e:
            logger.error(f"[Checkpoint] Failed to load checkpoint file: {e}")

    remaining_items = flattened_items[processed_count:]

    logger.info("=" * 70)
    logger.info("GEMMA 4:12B SYNTHETIC PARALLEL DATA PIPELINE INITIALIZED")
    logger.info("=" * 70)
    logger.info(f"Target Total Pairs    : {total_target:,}")
    logger.info(f"Total Batches Needed  : {total_batches:,} (5 items/batch)")
    logger.info(f"CSV Capacity Limit    : {rows_per_csv:,} rows/file")
    logger.info(f"Output Directory      : {output_dir.resolve()}")
    logger.info(f"Log File Location     : {(log_dir / 'gemma_pipeline.log').resolve()}")
    logger.info("=" * 70)

    def flush_csv(csv_idx, rows):
        csv_path = output_dir / f"marathi_parallel_part_{csv_idx:03d}.csv"
        file_exists = csv_path.exists()

        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        logger.info(f"[CSV Saved] Successfully saved {len(rows):,} rows to: {csv_path.resolve()}")

    batch_idx = (processed_count // ITEMS_PER_PROMPT) + 1
    i = 0
    total_remaining = len(remaining_items)

    pbar = tqdm(total=total_target, initial=processed_count, desc="Generating Parallel CSVs")

    while i < total_remaining:
        batch = remaining_items[i : i + ITEMS_PER_PROMPT]
        translations = query_ollama_batch(batch, batch_idx, total_batches)

        if translations:
            batch_added = 0
            for item, trans in zip(batch, translations):
                if trans and trans != item["dialect_text"]:
                    processed_count += 1
                    batch_added += 1
                    row = {
                        "id": processed_count,
                        "text_id": item["text_id"],
                        "dialect": item["dialect"],
                        "domain": item["domain"],
                        "distortion_score": item["distortion_score"],
                        "dialect_text": item["dialect_text"],
                        "standard_text": trans,
                    }
                    current_csv_rows.append(row)
                    pbar.update(1)

                    if len(current_csv_rows) >= rows_per_csv:
                        flush_csv(current_csv_index, current_csv_rows)
                        current_csv_rows = []
                        current_csv_index += 1

            # Save checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({
                    "processed_count": processed_count,
                    "current_csv_index": current_csv_index,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)

            pct = (processed_count / total_target) * 100.0
            logger.info(f"[Progress] Batch {batch_idx}/{total_batches} completed. Added {batch_added} pairs. Total progress: {processed_count:,}/{total_target:,} ({pct:.2f}%).")

            i += len(batch)
            batch_idx += 1
        else:
            logger.warning(f"[Retry] Batch {batch_idx}/{total_batches} failed or returned empty response. Retrying in 3 seconds...")
            time.sleep(3)

    if current_csv_rows:
        flush_csv(current_csv_index, current_csv_rows)
        current_csv_rows = []

    pbar.close()
    logger.info("=" * 70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Final Output Count : {processed_count:,} parallel pairs")
    logger.info(f"Total CSV Files    : {current_csv_index if not current_csv_rows else current_csv_index}")
    logger.info("=" * 70)
