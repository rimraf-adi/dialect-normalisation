"""
Gemma 4:12b Synthetic Data Generation Pipeline Engine.
Batches 5 sentences per prompt and saves output in 1,000-row CSV files.
"""

import csv
import json
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


def query_ollama_batch(batch_items: List[dict], model: str = MODEL_NAME, base_url: str = OLLAMA_URL) -> List[str]:
    """Sends 5 sentences in a single prompt to Ollama gemma4:12b and parses response."""
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

    try:
        req = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw_response = res_data.get("response", "").strip()

                json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list) and len(parsed) >= len(batch_items):
                        return [str(x).strip() for x in parsed[:len(batch_items)]]

                lines = [line.strip() for line in raw_response.split("\n") if line.strip() and not line.startswith("[") and not line.startswith("]")]
                cleaned_lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip(' "') for line in lines]
                if len(cleaned_lines) >= len(batch_items):
                    return cleaned_lines[:len(batch_items)]
    except Exception as e:
        print(f"Warning: Ollama API request failed: {e}", file=sys.stderr)

    return None


def execute_generation_pipeline(
    sampled_dataset: Dict[str, List[dict]],
    output_dir: Path,
    rows_per_csv: int = ROWS_PER_CSV,
):
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
                print(f"[Pipeline] Resuming from checkpoint: {processed_count:,} pairs processed.")
        except Exception as e:
            print(f"[Pipeline] Could not load checkpoint: {e}")

    remaining_items = flattened_items[processed_count:]

    print("\n" + "=" * 70)
    print("STARTING GEMMA 4:12B PARALLEL DATA GENERATION PIPELINE")
    print("=" * 70)
    print(f"Total Sentences Target : {total_target:,}")
    print(f"Prompt Batch Size      : {ITEMS_PER_PROMPT} sentences per call")
    print(f"CSV Capacity           : {rows_per_csv:,} rows per file")
    print(f"Output Directory       : {output_dir.resolve()}\n")

    pbar = tqdm(total=total_target, initial=processed_count, desc="Generating Parallel CSVs")

    def flush_csv(csv_idx, rows):
        csv_path = output_dir / f"marathi_parallel_part_{csv_idx:03d}.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\n[CSV Saved] Written {len(rows)} rows to: {csv_path.name}")

    i = 0
    total_remaining = len(remaining_items)

    while i < total_remaining:
        batch = remaining_items[i : i + ITEMS_PER_PROMPT]
        translations = query_ollama_batch(batch)

        if translations:
            for item, trans in zip(batch, translations):
                if trans and trans != item["dialect_text"]:
                    processed_count += 1
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

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"processed_count": processed_count, "current_csv_index": current_csv_index}, f)

            i += len(batch)
        else:
            time.sleep(2)

    if current_csv_rows:
        flush_csv(current_csv_index, current_csv_rows)
        current_csv_rows = []

    pbar.close()
    print("\n" + "=" * 70)
    print("✅ GEMMA GENERATION PIPELINE FINISHED!")
    print(f"Total Parallel Pairs Generated: {processed_count:,}")
    print("=" * 70)
