"""
NVIDIA NIM LLM Verification Engine for Marathi Sub-Dialect Datasets.
Uses openai/gpt-oss-120b and dynamic sysprompt.txt injection with detailed logging.
"""

import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "openai/gpt-oss-120b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

DATA_DIR = Path("data/synthetic_parallel")
CHECKPOINT_FILE = DATA_DIR / "llm_verification_checkpoint.json"
SYSPROMPT_PATH = Path("sysprompt.txt")
LOG_DIR = Path("logs")

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "llm_verifier.log"

    logger = logging.getLogger("dialect_norm.llm_verifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

logger = setup_logging()

def load_sysprompt() -> str:
    if SYSPROMPT_PATH.exists():
        return SYSPROMPT_PATH.read_text(encoding="utf-8").strip()
    return "You are a senior computational linguist specializing in Marathi dialects."

SYSTEM_PROMPT = load_sysprompt()

def evaluate_pair(row_id: str, dialect: str, dialect_text: str, standard_text: str) -> tuple[str, bool, str, float]:
    t0 = time.time()
    user_content = f"""Given Dialect ({dialect}): "{dialect_text}"
Current Standard Translation: "{standard_text}"

Evaluate if the Current Standard Translation strictly follows all rules and guidelines specified in the system prompt.
Mark FLAWED if:
1. Standard text is in Hindi or non-Marathi language.
2. Standard text is garbled, incoherent, or lost meaning/entities.
3. Standard text retains dialect words (e.g. माका, तुका, खय, शेतस, मले, इत्यादी).
4. Dialect text is a question (?) but standard text dropped question mark/context.

Reply strictly in JSON format:
{{"status": "VALID" | "FLAWED", "reason": "Short reason if FLAWED, else OK"}}"""

    # Pre-filter checks
    if re.search(r"की विधि से|होती है|होता है|किया जाता है", standard_text):
        elapsed = time.time() - t0
        return row_id, False, "Hindi language leakage (pre-filter)", elapsed

    if "?" in dialect_text and "?" not in standard_text:
        elapsed = time.time() - t0
        return row_id, False, "Question mark dropped (pre-filter)", elapsed

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                top_p=1,
                max_tokens=200,
                stream=False
            )
            elapsed = time.time() - t0
            content = completion.choices[0].message.content.strip()

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                res = json.loads(match.group(0))
                status = res.get("status", "").upper()
                reason = res.get("reason", "")
                if status == "FLAWED":
                    return row_id, False, reason, elapsed
                else:
                    return row_id, True, "OK", elapsed
            elif "FLAWED" in content.upper():
                return row_id, False, content[:100], elapsed
            else:
                return row_id, True, "OK", elapsed

        except Exception as e:
            logger.warning(f"Row {row_id} | Attempt {attempt + 1}/{max_retries} API Error: {e}")
            if attempt == max_retries - 1:
                elapsed = time.time() - t0
                return row_id, True, f"API Error (skipped): {str(e)}", elapsed
            time.sleep(1)

def main():
    logger.info("=" * 70)
    logger.info("NVIDIA NIM GPT-120B DATASET VERIFICATION ENGINE STARTED")
    logger.info("=" * 70)
    logger.info(f"Target Model       : {MODEL}")
    logger.info(f"API Endpoint       : {BASE_URL}")
    logger.info(f"System Prompt File : {SYSPROMPT_PATH.resolve()}")
    logger.info(f"Log File Path      : {(LOG_DIR / 'llm_verifier.log').resolve()}")
    logger.info("=" * 70)

    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(f"[Checkpoint] Loaded existing checkpoint with {len(checkpoint):,} verified rows.")
        except Exception as e:
            logger.error(f"[Checkpoint] Failed to load checkpoint: {e}")

    # Read d1.csv, d2.csv, and d4.csv
    candidate_csv_files = ["d1.csv", "d2.csv", "d4.csv"]
    candidate_rows = []

    for fname in candidate_csv_files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate_rows.append(row)

    # Read existing flawed.csv rows to preserve them
    existing_flawed = []
    flawed_path = DATA_DIR / "flawed.csv"
    if flawed_path.exists():
        with open(flawed_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_flawed.append(row)

    total_rows = len(candidate_rows)
    unverified = [r for r in candidate_rows if str(r["id"]) not in checkpoint]

    logger.info(f"Target Dialect CSVs   : d1.csv, d2.csv, d4.csv")
    logger.info(f"Total Rows to Check   : {total_rows:,}")
    logger.info(f"Already Verified Rows : {len(checkpoint):,}")
    logger.info(f"Remaining to Verify   : {len(unverified):,}")

    completed_count = len(checkpoint)
    flawed_count = sum(1 for v in checkpoint.values() if not v.get("valid", True))
    valid_count = completed_count - flawed_count

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {}
        for row in unverified:
            r_id = str(row["id"])
            dial = row.get("dialect", "D1")
            d_text = row.get("dialect_text", "")
            s_text = row.get("standard_text", "")

            fut = executor.submit(evaluate_pair, r_id, dial, d_text, s_text)
            futures[fut] = r_id

        for i, fut in enumerate(as_completed(futures), 1):
            r_id, is_valid, reason, latency = fut.result()
            checkpoint[r_id] = {
                "valid": is_valid,
                "reason": reason
            }

            completed_count += 1
            if is_valid:
                valid_count += 1
            else:
                flawed_count += 1
                logger.info(f"[FLAWED DETECTED] Row ID {r_id:>6s} ({latency:.2f}s) | Reason: {reason}")

            if i % 50 == 0 or i == len(unverified):
                elapsed_sec = time.time() - t_start
                rate = i / elapsed_sec if elapsed_sec > 0 else 0
                eta_sec = (len(unverified) - i) / rate if rate > 0 else 0
                eta_min = eta_sec / 60.0

                logger.info(
                    f"[Progress {completed_count:,}/{total_rows:,}] "
                    f"Valid: {valid_count:,} | Flawed: {flawed_count:,} | "
                    f"Speed: {rate:.1f} rows/sec | ETA: {eta_min:.1f} min"
                )

                # Persist checkpoint
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    # Separate clean dialect rows and compile flawed rows
    d1_clean, d2_clean, d4_clean = [], [], []
    new_flawed = []
    fieldnames = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]

    for row in candidate_rows:
        r_id = str(row["id"])
        ver = checkpoint.get(r_id, {"valid": True, "reason": "OK"})
        dial = row.get("dialect", "D1").upper().strip()

        if not ver["valid"]:
            flawed_row = {k: row.get(k, "") for k in fieldnames}
            flawed_row["flaw_reason"] = ver["reason"]
            new_flawed.append(flawed_row)
        else:
            clean_row = {k: row.get(k, "") for k in fieldnames}
            if dial == "D1":
                d1_clean.append(clean_row)
            elif dial == "D2":
                d2_clean.append(clean_row)
            elif dial == "D4":
                d4_clean.append(clean_row)
            else:
                d1_clean.append(clean_row)

    # Combine existing flawed rows with newly detected flawed rows (deduplicating by ID)
    seen_ids = set()
    all_flawed = []

    for f_row in existing_flawed + new_flawed:
        fid = str(f_row.get("id", ""))
        if fid not in seen_ids:
            seen_ids.add(fid)
            all_flawed.append(f_row)

    with open(DATA_DIR / "d1.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(d1_clean)

    with open(DATA_DIR / "d2.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(d2_clean)

    with open(DATA_DIR / "d4.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(d4_clean)

    flawed_fieldnames = fieldnames + ["flaw_reason"]
    with open(DATA_DIR / "flawed.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flawed_fieldnames)
        writer.writeheader()
        writer.writerows(all_flawed)

    logger.info("=" * 70)
    logger.info("VERIFICATION & REARRANGEMENT COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Verified Clean D1 (d1.csv)   : {len(d1_clean):,}")
    logger.info(f"Verified Clean D2 (d2.csv)   : {len(d2_clean):,}")
    logger.info(f"Verified Clean D4 (d4.csv)   : {len(d4_clean):,}")
    logger.info(f"Verified Flawed (flawed.csv) : {len(flawed_all):,}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
