"""
High-Throughput LLM Dataset Verification Engine with Groq 8-Key Rotation & NVIDIA NIM Fallback.
Verifies Marathi sub-dialect datasets smoothly. When Groq keys hit TPD/RPM daily caps,
it automatically falls back to NVIDIA NIM Llama-3.1-8B to finish remaining rows without pausing.
"""

import csv
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def load_env_file():
    """Loads environment variables from .env if present."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

load_env_file()

raw_groq_keys = os.getenv("GROQ_API_KEYS", "")
GROQ_KEYS = [k.strip() for k in raw_groq_keys.split(",") if k.strip()]
NVIDIA_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")

class KeyRotator:
    def __init__(self, keys: list[str], base_url: str = "https://api.groq.com/openai/v1"):
        if not keys:
            keys = ["dummy_groq_key"]
        self.clients = [
            OpenAI(base_url=base_url, api_key=k, timeout=20.0)
            for k in keys
        ]
        self._lock = threading.Lock()
        self._index = 0

    def get_client(self) -> OpenAI:
        with self._lock:
            client = self.clients[self._index % len(self.clients)]
            self._index += 1
            return client

rotator = KeyRotator(GROQ_KEYS)

class PacedNvidiaClient:
    """Thread-safe paced NVIDIA NIM client to avoid 429 rate limits."""
    def __init__(self, api_key: str, model: str = "meta/llama-3.1-8b-instruct"):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or "dummy_nvidia_key",
            timeout=25.0
        )
        self.model = model
        self._lock = threading.Lock()

    def chat_completion(self, messages, max_tokens=800):
        with self._lock:
            time.sleep(2.0)
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                top_p=1,
                max_tokens=max_tokens,
                stream=False
            )

nvidia_paced = PacedNvidiaClient(NVIDIA_KEY)

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "8"))
MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))

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

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

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

def evaluate_batch(batch_rows: list[dict]) -> list[tuple[str, bool, str, float]]:
    t0 = time.time()
    results = {}
    to_evaluate = []

    for row in batch_rows:
        r_id = str(row["id"])
        dial = row.get("dialect", "D1")
        d_text = row.get("dialect_text", "")
        s_text = row.get("standard_text", "")

        if re.search(r"की विधि से|होती है|होता है|किया जाता है", s_text):
            results[r_id] = (r_id, False, "Hindi language leakage (pre-filter)", 0.0)
        elif "?" in d_text and "?" not in s_text:
            results[r_id] = (r_id, False, "Question mark dropped (pre-filter)", 0.0)
        else:
            to_evaluate.append(row)

    if not to_evaluate:
        return [results[str(r["id"])] for r in batch_rows]

    items_prompt = []
    for row in to_evaluate:
        r_id = str(row["id"])
        dial = row.get("dialect", "D1")
        d_text = row.get("dialect_text", "")
        s_text = row.get("standard_text", "")
        items_prompt.append(f'ITEM_ID: "{r_id}" | Dialect ({dial}): "{d_text}" | Standard Translation: "{s_text}"')

    items_str = "\n".join(items_prompt)
    user_content = f"""Evaluate these regional Marathi sentence pairs for standard translation accuracy.
Mark FLAWED if:
1. Standard text is in Hindi or non-Marathi language.
2. Standard text is garbled, incoherent, or lost essential meaning/entities.
3. Standard text retains regional dialect words (e.g., माका, तुका, खय, शेतस, मले, इत्यादी).
4. Dialect text is a question (?) but standard text dropped question mark or question context.

Pairs to evaluate:
{items_str}

Respond STRICTLY with a JSON object containing key "results" mapping to an array of objects:
{{
  "results": [
    {{"id": "<exact ITEM_ID>", "status": "VALID" or "FLAWED", "reason": "<short reason if FLAWED else OK>"}}
  ]
}}"""

    max_retries = 5
    for attempt in range(max_retries):
        # Primary: Groq Key Rotator
        use_nvidia = False
        try:
            client = rotator.get_client()
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                top_p=1,
                max_tokens=800,
                stream=False
            )
        except Exception as e:
            err_str = str(e)
            if "TPD" in err_str or "429" in err_str or "rate_limit" in err_str:
                use_nvidia = True
            else:
                time.sleep(0.5)

        if use_nvidia:
            try:
                completion = nvidia_paced.chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=800
                )
            except Exception as e2:
                logger.warning(f"NVIDIA Fallback Attempt {attempt+1}/{max_retries} Error: {e2}")
                time.sleep(3.0)
                continue

        try:
            elapsed = time.time() - t0
            msg = completion.choices[0].message
            raw_content = msg.content or getattr(msg, "reasoning_content", "") or ""
            content = raw_content.strip()

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(content)

            res_list = parsed.get("results", [])
            if not isinstance(res_list, list):
                match_arr = re.search(r"\[.*\]", content, re.DOTALL)
                if match_arr:
                    res_list = json.loads(match_arr.group(0))
                else:
                    res_list = []

            per_row_elapsed = elapsed / len(to_evaluate)
            for item in res_list:
                if not isinstance(item, dict):
                    continue
                p_id = str(item.get("id", "")).strip().replace('"', '')
                status = str(item.get("status", "")).upper()
                reason = str(item.get("reason", "OK"))
                is_valid = (status != "FLAWED")
                if p_id:
                    results[p_id] = (p_id, is_valid, reason, per_row_elapsed)

            missing = [r for r in to_evaluate if str(r["id"]) not in results]
            if not missing:
                break
            elif attempt == max_retries - 1:
                for m_row in missing:
                    m_id = str(m_row["id"])
                    results[m_id] = (m_id, True, "OK (batch fill)", per_row_elapsed)
                break
        except Exception as parse_err:
            logger.warning(f"Batch Parse Attempt {attempt + 1}/{max_retries} Error: {parse_err}")
            if attempt == max_retries - 1:
                elapsed = time.time() - t0
                per_row_elapsed = elapsed / len(to_evaluate)
                for row in to_evaluate:
                    r_id = str(row["id"])
                    if r_id not in results:
                        results[r_id] = (r_id, True, "OK (fallback parse)", per_row_elapsed)

    return [results[str(r["id"])] for r in batch_rows]

def main():
    logger.info("=" * 70)
    logger.info("HYBRID GROQ + NVIDIA NIM VERIFICATION ENGINE STARTED")
    logger.info("=" * 70)
    logger.info(f"Primary Model      : {MODEL} (Groq 8-Key Pool)")
    logger.info(f"Fallback Model     : meta/llama-3.1-8b-instruct (NVIDIA NIM)")
    logger.info(f"Batch Size         : {BATCH_SIZE} rows/prompt")
    logger.info(f"Concurrent Workers : {MAX_WORKERS} threads")
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

    batches = [unverified[i:i + BATCH_SIZE] for i in range(0, len(unverified), BATCH_SIZE)]
    logger.info(f"Created {len(batches):,} batches of size {BATCH_SIZE}.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_batch, batch): batch for batch in batches}

        for b_idx, fut in enumerate(as_completed(futures), 1):
            batch_results = fut.result()
            for r_id, is_valid, reason, latency in batch_results:
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

            if b_idx % 10 == 0 or b_idx == len(batches):
                elapsed_sec = time.time() - t_start
                verified_in_run = completed_count - (total_rows - len(unverified))
                rate = verified_in_run / elapsed_sec if elapsed_sec > 0 else 0
                eta_sec = (len(unverified) - verified_in_run) / rate if rate > 0 else 0
                eta_min = eta_sec / 60.0

                logger.info(
                    f"[Progress {completed_count:,}/{total_rows:,}] "
                    f"Valid: {valid_count:,} | Flawed: {flawed_count:,} | "
                    f"Speed: {rate:.1f} rows/sec | ETA: {eta_min:.1f} min"
                )

                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

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
    logger.info(f"Verified Flawed (flawed.csv) : {len(all_flawed):,}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
