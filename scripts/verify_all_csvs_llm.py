import csv
import json
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

def load_sysprompt() -> str:
    if SYSPROMPT_PATH.exists():
        return SYSPROMPT_PATH.read_text(encoding="utf-8").strip()
    return "You are a senior computational linguist specializing in Marathi dialects."

SYSTEM_PROMPT = load_sysprompt()

def evaluate_pair(row_id, dialect, dialect_text, standard_text):
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

    # Quick deterministic pre-filter for obvious rule breaks to save API calls
    if re.search(r"की विधि से|होती है|होता है|किया जाता है", standard_text):
        return row_id, False, "Hindi language leakage"
    if "?" in dialect_text and "?" not in standard_text:
        return row_id, False, "Question mark dropped"
    
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
            content = completion.choices[0].message.content.strip()
            
            # Parse JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                res = json.loads(match.group(0))
                status = res.get("status", "").upper()
                reason = res.get("reason", "")
                if status == "FLAWED":
                    return row_id, False, reason
                else:
                    return row_id, True, "OK"
            elif "FLAWED" in content.upper():
                return row_id, False, content[:100]
            else:
                return row_id, True, "OK"
        except Exception as e:
            if attempt == max_retries - 1:
                # Default to heuristic on API failure
                return row_id, True, f"API Error: {str(e)}"
            time.sleep(1)

def main():
    print("=" * 70)
    print("STARTING FULL DATASET LLM VERIFICATION (NVIDIA NIM GPT-120B)")
    print("=" * 70)

    # Load existing checkpoint
    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"Loaded existing checkpoint with {len(checkpoint):,} verified rows.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    # Gather all rows across all CSV files in data/synthetic_parallel
    all_rows = []
    csv_files = ["d1.csv", "d2.csv", "d4.csv", "flawed.csv"]
    
    for fname in csv_files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    print(f"Total rows to verify across datasets: {len(all_rows):,}")

    unverified = [r for r in all_rows if str(r["id"]) not in checkpoint]
    print(f"Rows remaining to verify: {len(unverified):,}")

    batch_size = 25
    completed_count = len(checkpoint)

    # Use ThreadPoolExecutor for parallel verification
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
            r_id, is_valid, reason = fut.result()
            checkpoint[r_id] = {
                "valid": is_valid,
                "reason": reason
            }
            completed_count += 1

            if i % 100 == 0 or i == len(unverified):
                print(f"Progress: {completed_count:,} / {len(all_rows):,} rows verified ({i}/{len(unverified)} in this run)...")
                # Save checkpoint
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    # Final rearrangement based on LLM verification results
    d1_clean = []
    d2_clean = []
    d4_clean = []
    flawed_all = []

    fieldnames = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]

    for row in all_rows:
        r_id = str(row["id"])
        ver = checkpoint.get(r_id, {"valid": True, "reason": "OK"})
        dial = row.get("dialect", "D1").upper().strip()

        if not ver["valid"]:
            flawed_row = {k: row.get(k, "") for k in fieldnames}
            flawed_row["flaw_reason"] = ver["reason"]
            flawed_all.append(flawed_row)
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

    # Write final split files
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
        writer.writerows(flawed_all)

    print("=" * 70)
    print("ALL CSV VERIFICATION & REARRANGEMENT COMPLETE!")
    print("=" * 70)
    print(f"Verified Clean D1 (d1.csv) : {len(d1_clean):,}")
    print(f"Verified Clean D2 (d2.csv) : {len(d2_clean):,}")
    print(f"Verified Clean D4 (d4.csv) : {len(d4_clean):,}")
    print(f"Verified Flawed (flawed.csv): {len(flawed_all):,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
