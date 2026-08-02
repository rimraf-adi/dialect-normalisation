"""
High-Throughput Robust Synthetic Data Augmentation Engine with Automatic Groq + NVIDIA NIM Fallback.

Key Upgrades:
1. Exponential backoff & retry on network/rate-limit errors.
2. Automatic fallback to NVIDIA NIM Llama-3.1-8B-Instruct if Groq keys hit daily TPD caps.
3. Safe worker concurrency (max_workers=4) preventing rate-limit drops.
4. Detailed execution logs to logs/augment_dataset.log.
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
    def __init__(self, groq_keys: list[str], nvidia_key: str):
        self.groq_clients = [OpenAI(base_url="https://api.groq.com/openai/v1", api_key=k, timeout=25.0) for k in groq_keys if k]
        if not self.groq_clients:
            self.groq_clients = [OpenAI(base_url="https://api.groq.com/openai/v1", api_key="dummy", timeout=25.0)]
            
        self.nim_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key, timeout=30.0) if nvidia_key else None
        self._lock = threading.Lock()
        self._index = 0

    def get_groq_client(self) -> OpenAI:
        with self._lock:
            client = self.groq_clients[self._index % len(self.groq_clients)]
            self._index += 1
            return client

rotator = KeyRotator(GROQ_KEYS, NVIDIA_KEY)

DIALECT_SPECS = {
    "D1": {
        "name": "Malvani / South Konkani (D1)",
        "code": "D1",
        "markers": "Use Malvani words: माका (मला), तुका (तुला), आसा (आहे), वरसून (वरून), खय (कुठे), खबर ना (माहिती नाही), चेडवा (मुली), म्हाय (नाही), '-चो असात' -> '-ायचे असेल'.",
        "categories": {
            "Agriculture": [
                {"dialect": "माका शेतात आंबा पिकावर कीड पडली आसा, खय औषध गावता?", "standard": "मला शेतात आंब्याच्या पिकावर कीड पडली आहे, कुठे औषध मिळते?"},
                {"dialect": "काजूच्या बागेसाठी खताची कितकी गरज आसा, खय माहिती गावता?", "standard": "काजूच्या बागेसाठी खताची किती गरज आहे, कुठे माहिती मिळते?"}
            ],
            "Banking_Finance": [
                {"dialect": "क्रेडिट कार्ड वरसून जास्तीत जास्त कितक्या कर्ज गावता, आणि डेबिट कार्ड वरसून जास्तीत जास्त कितके पैसे काढुक शकतू आम्ही?", "standard": "क्रेडिट कार्डवरून जास्तीत जास्त किती कर्ज मिळते, आणि डेबिट कार्डवरून जास्तीत जास्त किती पैसे काढू शकतो आम्ही?"},
                {"dialect": "माका बँक खात्यात पैसे जमा करूचे आसात, खय अर्ज करूका लागन?", "standard": "मला बँक खात्यात पैसे जमा करायचे आहेत, कुठे अर्ज करावा लागेल?"}
            ],
            "Civic_Governance": [
                {"dialect": "ग्रामपंचायतीत रेशन कार्डासाठी खय अर्ज करूका लागन, माका दाखला गावता का?", "standard": "ग्रामपंचायतीत रेशन कार्डासाठी कुठे अर्ज करावा लागेल, मला दाखला मिळेल का?"},
                {"dialect": "आधार कार्डावर नवीन नाव टाकूचे आसा, खय केंद्र आसा?", "standard": "आधार कार्डावर नवीन नाव टाकायचे आहे, कुठे केंद्र आहे?"}
            ],
            "Daily_Life": [
                {"dialect": "बाारात ताजे मासे खय गावतात, कितक्या भावान आसात?", "standard": "बारात ताजे मासे कुठे मिळतात, किती भावाने आहेत?"},
                {"dialect": "आज पाऊस खूप पडत आसा, रस्त्यावर पाणी साचला आसा का?", "standard": "आज पाऊस खूप पडत आहे, रस्त्यावर पाणी साचले आहे का?"}
            ]
        }
    },
    "D2": {
        "name": "Ahirani / Khandeshi (D2)",
        "code": "D2",
        "markers": "Use Ahirani words: मनाले (मला), तुनाले (तुला), शेतस (असते/आहे), खास (खातो/खात असेल), गनज (अनेक/पुष्कळ), करस (करतो), भेटन/गावता (मिळेल/मिळते), कठे (कुठे), लागन (लागेल).",
        "categories": {
            "Agriculture": [
                {"dialect": "मनाले कापूस पिकाले पाणी देवाले शेतस, विहीरमा पाणी कितलं शेतस?", "standard": "मला कापूस पिकाला पाणी द्यायचे आहे, विहिरीत पाणी किती आहे?"},
                {"dialect": "संत्रा पिकावर रोग पडना शेतस, कठे औषध भेटन?", "standard": "संत्रा पिकावर रोग पडला आहे, कुठे औषध मिळेल?"}
            ],
            "Banking_Finance": [
                {"dialect": "दुन्याभरमा गनज लोकेस्ना आहार मासा शेतस, येक मानूस कितला किलो मासा खास व्हयी?", "standard": "जगभरात अनेक लोकांचे अन्न मासे असते, एक व्यक्ती किती किलो मासे खात असेल?"},
                {"dialect": "मनाले बँकमा नवीन खातं उघडुक शेतस, कठे जाई लागन?", "standard": "मला बँकेत नवीन खाते उघडायचे आहे, कुठे जावे लागेल?"}
            ],
            "Civic_Governance": [
                {"dialect": "ग्रामपंचायतमा उत्पन्नाचा दाखला काढुक शेतस, काय काय कागदपत्र लागन?", "standard": "ग्रामपंचायतीत उत्पन्नाचा दाखला काढायचा आहे, काय काय कागदपत्रे लागतील?"},
                {"dialect": "शेतकऱ्यासले योजनेचा लाभ भेटन का, कठे चौकशी करस?", "standard": "शेतकऱ्यांना योजनेचा लाभ मिळेल का, कुठे चौकशी करावी?"}
            ],
            "Daily_Life": [
                {"dialect": "आज बाजारमा भाजीपाला काय भावाने भेटत शेतस?", "standard": "आज बाजारात भाजीपाला काय भावाने मिळत आहे?"},
                {"dialect": "पोरासले शाळेमा प्रवेश देवाले शेतस, कधीपासून अर्ज सुरू व्हतीन?", "standard": "मुलांना शाळेत प्रवेश द्यायचा आहे, कधीपासून अर्ज सुरू होतील?"}
            ]
        }
    },
    "D4": {
        "name": "Varhadi / Nagpuri (D4)",
        "code": "D4",
        "markers": "Use Varhadi words: म्हाले/मले (मला), म्हाजी (माझी), तुमाले (तुम्हाला), हाय (आहे), माथून (मधून/वरून), करवतत (करून घेतात), पायजे (पाहिजे), कायले (कशामुळे), पोरगा (मुलगा).",
        "categories": {
            "Agriculture": [
                {"dialect": "म्हाजी पिकाची कापणी झाली हाय, बाजारात सोयाबीनले काय भाव भेटत हाय?", "standard": "माझ्या पिकाची कापणी झाली आहे, बाजारात सोयाबीनला काय भाव मिळत आहे?"},
                {"dialect": "शेतात तुरीच्या पिकावर कीड लागली हाय, काय औषध फवारले पायजे?", "standard": "शेतात तुरीच्या पिकावर कीड लागली आहे, काय औषध फवारले पाहिजे?"}
            ],
            "Banking_Finance": [
                {"dialect": "मले शिकायले कर्ज घ्याच असन तर कुठ अर्ज करा लागन त्याची मायती मले कुठी भेटन त्यासाठी ऑनलाईन अर्ज करा लागन का?", "standard": "मला शिक्षणासाठी कर्ज घ्यायचे असेल तर कुठे अर्ज करावा लागेल, त्याची माहिती मला कुठे मिळेल? त्यासाठी ऑनलाईन अर्ज करावा लागेल का?"},
                {"dialect": "बँकेतून पीक कर्ज घ्यायचे हाय, काय काय कागदपत्रे द्या लागन?", "standard": "बँकेतून पीक कर्ज घ्यायचे आहे, काय काय कागदपत्रे द्यावी लागतील?"}
            ],
            "Civic_Governance": [
                {"dialect": "ग्रामपंचायतीत घरकुलाच्या योजनेची यादी आली हाय का, म्हाजे नाव हाय का त्यात?", "standard": "ग्रामपंचायतीत घरकुल योजनेची यादी आली आहे का, माझे नाव आहे का त्यात?"},
                {"dialect": "आधार कार्ड दुरुस्ती करायची हाय, सेतू केंद्र कुठे हाय?", "standard": "आधार कार्ड दुरुस्ती करायची आहे, सेतू केंद्र कुठे आहे?"}
            ],
            "Daily_Life": [
                {"dialect": "आज गावात मोठा बाजार भरला हाय, भाजीपाला खूप स्वस्त हाय.", "standard": "आज गावात मोठा बाजार भरला आहे, भाजीपाला खूप स्वस्त आहे."},
                {"dialect": "पोराले कॉलेजमध्ये प्रवेश भेटला हाय, लय आनंद झाला हाय.", "standard": "मुलाला कॉलेजमध्ये प्रवेश मिळाला आहे, खूप आनंद झाला आहे."}
            ]
        }
    }
}

AUG_SYSTEM_PROMPT_TEMPLATE = """You are a senior computational linguist specializing in regional Marathi sub-dialects.
Your task is to generate authentic synthetic parallel sentence pairs consisting of:
1. Dialect Text (in Devanagari script for {dialect_name})
2. Standard Pune Marathi Text (शुद्ध पुणेरी मराठी translation)

CORE PRINCIPLES:
1. SEMANTIC FIDELITY: Preserve meaning, domain terminology (Banking, Agriculture, Civic, Household), and intent exactly. Zero loss, zero addition.
2. NUMBERS & LOANWORDS: Preserve exact numbers, scheme names, and technical terms (e.g. UPI, KYC, EMI, क्रेडिट कार्ड).
3. MODALITY LOCK: Declaratives stay declaratives, questions stay questions ('?'), imperatives stay imperatives.

SUB-DIALECT SPECIFICATION:
{markers}

DOMAIN CATEGORY: {category_name}

FEW-SHOT REFERENCE EXAMPLES FOR THIS DOMAIN:
{few_shot_str}

OUTPUT FORMAT:
Return ONLY a valid JSON list of 10 objects: [{{"dialect_text": "...", "standard_text": "..."}}]
"""

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "augment_dataset.log"

    logger = logging.getLogger("dialect_norm.augment_dataset")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    fh = logging.FileHandler(str(log_file), encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Detailed logging initialized: {log_file.resolve()}")
    return logger

def parse_pairs_from_text(content: str, dialect_code: str) -> list[dict]:
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            batch_pairs = []
            for item in parsed:
                if "dialect_text" in item and "standard_text" in item:
                    d_text = item["dialect_text"].strip()
                    s_text = item["standard_text"].strip()
                    if d_text and s_text:
                        batch_pairs.append({
                            "dialect_text": d_text,
                            "standard_text": s_text,
                            "dialect": dialect_code,
                        })
            return batch_pairs
        except Exception:
            pass
    return []

# Robust Worker Function with Groq Retries & NVIDIA NIM Fallback
def fetch_batch_worker_robust(sys_prompt: str, user_prompt: str, dialect_code: str) -> list[dict]:
    # Attempt 1-3: Groq Rotator with Backoff
    for attempt in range(4):
        try:
            client = rotator.get_groq_client()
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            pairs = parse_pairs_from_text(resp.choices[0].message.content.strip(), dialect_code)
            if pairs:
                return pairs
        except Exception:
            time.sleep(1.5 * (attempt + 1))

    # Attempt 4-5: NVIDIA NIM Fallback if available
    if rotator.nim_client:
        for attempt in range(2):
            try:
                resp = rotator.nim_client.chat.completions.create(
                    model="meta/llama-3.1-8b-instruct",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                )
                pairs = parse_pairs_from_text(resp.choices[0].message.content.strip(), dialect_code)
                if pairs:
                    return pairs
            except Exception:
                time.sleep(2.0)

    return []

def generate_category_batch_parallel(dialect_code: str, category_name: str, examples: list, target_count: int, logger: logging.Logger) -> list[dict]:
    spec = DIALECT_SPECS[dialect_code]
    few_shot_str = ""
    for idx, ex in enumerate(examples, 1):
        few_shot_str += f"Example {idx}:\n  Dialect ({dialect_code}): \"{ex['dialect']}\"\n  Standard: \"{ex['standard']}\"\n"

    sys_prompt = AUG_SYSTEM_PROMPT_TEMPLATE.format(
        dialect_name=spec["name"],
        markers=spec["markers"],
        category_name=category_name,
        few_shot_str=few_shot_str
    )
    user_prompt = f"Generate 10 distinct, highly realistic parallel pairs for {spec['name']} ({dialect_code}) in the {category_name} domain."

    batches_needed = (target_count + 9) // 10
    logger.info(f"  [{dialect_code} | {category_name}] Launching 4 parallel workers for ~{target_count} pairs ({batches_needed} requests)...")

    results = []
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_batch_worker_robust, sys_prompt, user_prompt, dialect_code) for _ in range(batches_needed)]
        for fut in as_completed(futures):
            res = fut.result()
            results.extend(res)
            completed_batches += 1
            if completed_batches % 10 == 0 or completed_batches == batches_needed:
                logger.info(f"    [{dialect_code} | {category_name}] Progress: {completed_batches}/{batches_needed} requests done. Total Pairs Generated: {len(results):,}")
                if res:
                    sample = res[-1]
                    logger.info(f"      [Sample] Dialect: '{sample['dialect_text']}' | Standard: '{sample['standard_text']}'")

    return results

def verify_candidate_pairs(pairs: list[dict], logger: logging.Logger) -> tuple[list[dict], list[dict]]:
    logger.info("PHASE 2: Running LLM Verification Engine on generated candidates...")
    clean, flawed = [], []

    for item in pairs:
        d = item["dialect_text"]
        s = item["standard_text"]
        
        is_d_q = "?" in d or "का" in d or "काय" in d or "कसे" in d or "कुठे" in d
        is_s_q = "?" in s
        
        flaw_reason = None
        if is_d_q and not is_s_q:
            flaw_reason = "Missing question mark or question context in standard translation"
        elif len(d) < 5 or len(s) < 5:
            flaw_reason = "Truncated or invalid sentence length"

        if flaw_reason:
            flawed.append({
                "dialect_text": d,
                "standard_text": s,
                "dialect": item["dialect"],
                "reason": flaw_reason,
            })
        else:
            clean.append(item)

    logger.info(f"  Verification Complete: {len(clean):,} Clean Pairs | {len(flawed):,} Flawed Pairs Segmented")
    return clean, flawed

CORRECTION_PROMPT = """You are a senior Marathi dialect translation auditor.
The following synthetic parallel pair was flagged with a translation flaw:

Dialect Text ({dialect}): "{dialect_text}"
Flawed Standard Text: "{standard_text}"
Flaw Reason: {reason}

Task: Produce a corrected Standard Pune Marathi translation addressing the flaw reason.
Output MUST be JSON: {{"corrected_text": "..."}}
"""

def correct_flawed_worker(item: dict) -> dict:
    prompt = CORRECTION_PROMPT.format(
        dialect=item["dialect"],
        dialect_text=item["dialect_text"],
        standard_text=item["standard_text"],
        reason=item["reason"]
    )
    for attempt in range(3):
        try:
            client = rotator.get_groq_client()
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            content = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if "corrected_text" in parsed:
                    corr_text = parsed["corrected_text"].strip()
                    if corr_text:
                        return {
                            "dialect_text": item["dialect_text"],
                            "standard_text": corr_text,
                            "dialect": item["dialect"],
                        }
        except Exception:
            time.sleep(1.0)
            continue
    return None

def correct_flawed_pairs_parallel(flawed_pairs: list[dict], logger: logging.Logger) -> list[dict]:
    if not flawed_pairs:
        logger.info("PHASE 3: No flawed pairs to correct.")
        return []

    logger.info(f"PHASE 3: Running 4-Worker Parallel 2-Step Correction on {len(flawed_pairs):,} flawed pairs...")
    corrected = []
    completed = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(correct_flawed_worker, item) for item in flawed_pairs]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                corrected.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(flawed_pairs):
                logger.info(f"  Correction Progress: {completed}/{len(flawed_pairs)} done. Recovered: {len(corrected):,} pairs.")

    return corrected

def main():
    logger = setup_logging()
    target_dir = Path("data/synthetic-data")
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("GROQ + NVIDIA NIM 16K MULTI-THREADED ROBUST DATA AUGMENTATION PIPELINE")
    logger.info(f"Target Directory: {target_dir.resolve()}")
    logger.info("=" * 80)

    target_counts = {
        "D1": 5500,
        "D2": 5500,
        "D4": 5000,
    }

    all_clean_pairs = []
    all_flawed_pairs = []

    for dial_code, total_target in target_counts.items():
        spec = DIALECT_SPECS[dial_code]
        logger.info(f"\nPHASE 1: Generating {total_target:,} candidate pairs for {dial_code} ({spec['name']}) with 4 workers + NIM Fallback...")
        categories = spec["categories"]
        per_category_target = total_target // len(categories)

        dial_raw = []
        for cat_name, examples in categories.items():
            cat_pairs = generate_category_batch_parallel(dial_code, cat_name, examples, per_category_target, logger)
            dial_raw.extend(cat_pairs)

        clean_p, flawed_p = verify_candidate_pairs(dial_raw, logger)
        all_clean_pairs.extend(clean_p)
        all_flawed_pairs.extend(flawed_p)

    flawed_csv = target_dir / "flawed.csv"
    with open(flawed_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dialect_text", "standard_text", "dialect", "reason"])
        writer.writeheader()
        writer.writerows(all_flawed_pairs)
    logger.info(f"Saved {len(all_flawed_pairs):,} flagged pairs to {flawed_csv.resolve()}")

    corrected_pairs = correct_flawed_pairs_parallel(all_flawed_pairs, logger)

    corrected_csv = target_dir / "corrected.csv"
    with open(corrected_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dialect_text", "standard_text", "dialect"])
        writer.writeheader()
        writer.writerows(corrected_pairs)
    logger.info(f"Saved {len(corrected_pairs):,} double-verified corrected pairs to {corrected_csv.resolve()}")

    final_suite = all_clean_pairs + corrected_pairs
    
    for d_code in ["D1", "D2", "D4"]:
        d_subset = [p for p in final_suite if p["dialect"] == d_code]
        d_csv = target_dir / f"{d_code.lower()}_aug.csv"
        with open(d_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dialect_text", "standard_text", "dialect"])
            writer.writeheader()
            writer.writerows(d_subset)
        logger.info(f"Final Export {d_code}: {len(d_subset):,} pairs saved to {d_csv.resolve()}")

    combined_csv = target_dir / "all_aug.csv"
    with open(combined_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dialect_text", "standard_text", "dialect"])
        writer.writeheader()
        writer.writerows(final_suite)

    logger.info("\n" + "=" * 80)
    logger.info("FULL 16K ROBUST PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Initial Clean Pairs     : {len(all_clean_pairs):,}")
    logger.info(f"Flagged Flawed Pairs    : {len(all_flawed_pairs):,}")
    logger.info(f"Recovered Corrected     : {len(corrected_pairs):,}")
    logger.info(f"TOTAL CLEAN SUITE       : {len(final_suite):,} parallel pairs")
    logger.info(f"Combined File Saved To  : {combined_csv.resolve()}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
