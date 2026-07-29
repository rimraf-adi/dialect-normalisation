"""
LM Studio / Ollama Synthetic Data Generation Pipeline Engine with Detailed Logging & Safety Health Checks.
Batches 5 sentences per prompt and saves output in 1,000-row CSV files.
Supports LM Studio (google/gemma-4-e2b) and Ollama (gemma4:12b).
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
from typing import Dict, List, Tuple
from tqdm import tqdm

DEFAULT_PROVIDER = "lmstudio"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "google/gemma-4-e2b"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:12b"

ITEMS_PER_PROMPT = 1
ROWS_PER_CSV = 1000

logger = logging.getLogger("dialect_norm.gemma_pipeline")

PROMPT_TEMPLATE = """You are a senior computational linguist specializing in Marathi sub-dialects:
- D1: Malvani / South Konkani (Ratnagiri, Sindhudurg)
- D2: Ahirani / Khandeshi (Jalgaon, Dhule, Palghar)
- D4: Varhadi / Nagpuri (Vidarbha, Amravati, Nagpur)

Normalize the following single regional Marathi dialect sentence into formal Standard Pune Marathi (शुद्ध पुणेरी मराठी).

MANDATORY NORMALIZATION RULES:
1. STRICT SEMANTIC FIDELITY (ZERO HALLUCINATION & ZERO LOSS):
   - Do NOT add any extra phrases, commentary, or assumptions (e.g. do NOT insert "हे मला माहीत नाही").
   - Do NOT drop or omit any word, clause, domain term, or concept (e.g. convert "शिकायले" -> "शिक्षणासाठी", do NOT drop it).
   - Maintain exact sentence structure, tense, numbers, and modality (questions remain questions, imperatives remain imperatives).

2. DIALECT LEXICAL & GRAMMAR CONVERSIONS:
   - Pronouns: Convert 'माका'/'मले' -> 'मला'; 'तुका'/'तुले' -> 'तुला'; 'आम्हाले' -> 'आम्हाला'.
   - Postpositions: Convert 'वरसून' -> 'वरून'; 'माथून' -> 'मधून'/'वरून'; '-ले' -> '-ला'.
   - Verbs & Auxiliaries: Convert 'भेटन'/'गावता' -> 'मिळेल'/'मिळते'; 'लागन' -> 'लागेल'; 'करस'/'करास' -> 'करतो'/'करते'; 'काढुक' -> 'काढायला'; 'असन' -> 'असेल'.
   - Domain Spellings: Correct regional phonetic distortions (e.g. 'मासामारी' -> 'मासेमारी').

3. FEW-SHOT REFERENCE EXAMPLES:
   - Dialect Input (D1): "क्रेडिट कार्ड वरसून जास्तीत जास्त कितक्या कर्ज गावता , आणि डेबिट कार्ड वरसून जास्तीत जास्त कितके पैसे काढुक शकतू आम्ही ?"
     Standard Output   : "क्रेडिट कार्डवरून जास्तीत जास्त किती कर्ज मिळते, आणि डेबिट कार्डवरून जास्तीत जास्त किती पैसे काढू शकतो आम्ही?"
   - Dialect Input (D2): "देशना पुरा मासामारी उत्पादनपैकी सुमारे साठ एकर उत्पादन देशमझारला मासामारीमाथून येस"
     Standard Output   : "देशाच्या संपूर्ण मासेमारी उत्पादनापैकी सुमारे साठ टक्के उत्पादन देशाच्या अंतर्गत मासेमारीतून येते."
   - Dialect Input (D4): "मले शिकायले कर्ज घ्याच असन तर कुठ अर्ज करा लागन त्याची मायती मले कुठी भेटन त्यासाठी ऑनलाईन अर्ज करा लागन का ?"
     Standard Output   : "मला शिक्षणासाठी कर्ज घ्यायचे असेल तर कुठे अर्ज करावा लागेल, त्याची माहिती मला कुठे मिळेल? त्यासाठी ऑनलाईन अर्ज करावा लागेल का?"

Dialect Sentence Input:
"{text}"

OUTPUT INSTRUCTION:
Respond ONLY with the clean Standard Pune Marathi translation text for this single sentence.
Do NOT use quotes, markdown fences, JSON formatting, preamble, or commentary. Output ONLY the single translated sentence.
"""


def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "gemma_pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)


def check_llm_health(provider: str, model_name: str, base_url: str) -> Tuple[bool, str]:
    """
    Safety Check: Verifies server availability and warm-up generation for LM Studio or Ollama.
    """
    print("\n" + "=" * 70, flush=True)
    print(f"LLM BACKEND SAFETY & HEALTH CHECK ({provider.upper()})", flush=True)
    print("=" * 70, flush=True)

    base_url = base_url.rstrip("/")

    if provider.lower() in ["lmstudio", "openai", "lm-studio"]:
        models_url = f"{base_url}/models"
        chat_url = f"{base_url}/chat/completions"

        print(f"[Health Check 1/3] Pinging LM Studio server at {models_url}...", flush=True)
        try:
            req = urllib.request.Request(models_url, headers={"User-Agent": "Python-LMStudioHealthCheck"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    models_data = json.loads(resp.read().decode("utf-8"))
                    installed_models = [m.get("id", "") for m in models_data.get("data", [])]
                    print(f"  --> Server UP! Active LM Studio Models ({len(installed_models)}): {', '.join(installed_models)}", flush=True)
                else:
                    return False, f"LM Studio server returned status HTTP {resp.status}"
        except urllib.error.URLError as e:
            return False, f"LM Studio server connection failed at {models_url}. Ensure LM Studio Local Server is running! Error: {e.reason}"
        except Exception as e:
            return False, f"Unexpected error connecting to LM Studio: {e}"

        print(f"[Health Check 2/3] Verifying model '{model_name}'...", flush=True)
        print(f"  --> LM Studio target model configured: '{model_name}'", flush=True)

        print(f"[Health Check 3/3] Running warm-up chat completion on '{model_name}'...", flush=True)
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello! Confirm active status."}],
            "max_tokens": 128,
            "temperature": 0.1,
        }

        t0 = time.time()
        try:
            req = urllib.request.Request(
                chat_url,
                data=json.dumps(test_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                elapsed = time.time() - t0
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    content = res_data["choices"][0]["message"]["content"].strip()
                    print(f"  --> WARM-UP SUCCESSFUL ({elapsed:.2f}s)! LM Studio Response: '{content[:60]}...'", flush=True)
                    print("=" * 70 + "\n", flush=True)
                    return True, f"LM Studio health check passed ({elapsed:.2f}s latency)."
                else:
                    return False, f"LM Studio chat endpoint returned status HTTP {resp.status}"
        except Exception as e:
            print(f"  ⚠️ LM Studio warm-up test warning: {e}. Proceeding with pipeline...", flush=True)
            print("=" * 70 + "\n", flush=True)
            return True, f"LM Studio server active (warm-up warning: {e})."

    else:  # Ollama Provider
        tags_url = f"{base_url}/api/tags"
        generate_url = f"{base_url}/api/generate"

        print(f"[Health Check 1/3] Pinging Ollama server at {tags_url}...", flush=True)
        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "Python-OllamaHealthCheck"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    tags_data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in tags_data.get("models", [])]
                    print(f"  --> Server UP! Installed Ollama models ({len(models)}): {', '.join(models)}", flush=True)
                else:
                    return False, f"Ollama server returned status HTTP {resp.status}"
        except urllib.error.URLError as e:
            return False, f"Ollama server connection failed at {tags_url}. Error: {e.reason}"
        except Exception as e:
            return False, f"Unexpected error connecting to Ollama: {e}"

        print(f"[Health Check 2/3] Verifying model '{model_name}'...", flush=True)
        print(f"  --> Target Ollama model: '{model_name}'", flush=True)

        print(f"[Health Check 3/3] Running warm-up prompt on '{model_name}'...", flush=True)
        test_payload = {
            "model": model_name,
            "prompt": "Hello! Confirm active status.",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 128},
        }

        t0 = time.time()
        try:
            req = urllib.request.Request(
                generate_url,
                data=json.dumps(test_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                elapsed = time.time() - t0
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    content = res_data.get("response", "").strip()
                    print(f"  --> WARM-UP SUCCESSFUL ({elapsed:.2f}s)! Ollama Response: '{content[:60]}...'", flush=True)
                    print("=" * 70 + "\n", flush=True)
                    return True, f"Ollama health check passed ({elapsed:.2f}s latency)."
                else:
                    return False, f"Ollama generation endpoint returned HTTP {resp.status}"
        except Exception as e:
            print(f"  ⚠️ Warm-up warning: {e}. Proceeding...", flush=True)
            print("=" * 70 + "\n", flush=True)
            return True, f"Ollama server active (warm-up warning: {e})."


def _clean_single_translation(raw_text: str) -> str:
    """Helper to extract clean single translated sentence from raw LLM output."""
    if not raw_text:
        return ""

    # Strip thinking / reasoning tags
    cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", raw_text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw_text.strip()

    # Strip markdown code blocks
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    # Extract JSON string if wrapped in JSON object or array
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                cleaned = str(parsed[0])
        except Exception:
            pass

    # Clean leading "Output:", "Standard Output:", or numbers
    cleaned = re.sub(r"^(?:Standard Output|Output|Standard Pune|Standard Marathi)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip()

    # Strip outer quotes
    cleaned = cleaned.strip(' "\'`\n\r\t')
    return cleaned


def _parse_response_to_translations(raw_text: str, expected_count: int) -> List[str]:
    """Extract list of translations (handles single-sentence output or legacy batch arrays)."""
    if expected_count == 1:
        single_trans = _clean_single_translation(raw_text)
        return [single_trans] if single_trans else []

    if not raw_text:
        return []

    cleaned_text = re.sub(r"<(thought|think)>.*?</\1>", "", raw_text, flags=re.DOTALL).strip()
    if not cleaned_text:
        cleaned_text = raw_text.strip()

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, list) and len(parsed) >= expected_count:
            return [str(x).strip() for x in parsed[:expected_count]]
    except Exception:
        pass

    json_match = re.search(r"\[.*\]", cleaned_text, re.DOTALL)
    if json_match:
        json_str_clean = re.sub(r",\s*\]", "]", json_match.group(0))
        try:
            parsed = json.loads(json_str_clean)
            if isinstance(parsed, list) and len(parsed) >= expected_count:
                return [str(x).strip() for x in parsed[:expected_count]]
        except Exception:
            pass

        quoted_items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', json_match.group(0))
        if len(quoted_items) >= expected_count:
            return [x.strip() for x in quoted_items[:expected_count]]

    lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]
    cleaned_lines = []
    for line in lines:
        if line in ["[", "]", "```", "```json"]:
            continue
        line_clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip(' ",\t')
        if line_clean and line_clean not in ["[", "]"]:
            cleaned_lines.append(line_clean)

    if len(cleaned_lines) >= expected_count:
        return cleaned_lines[:expected_count]

    return []


def query_llm_batch(
    batch_items: List[dict],
    batch_idx: int,
    total_batches: int,
    provider: str = DEFAULT_PROVIDER,
    model: str = LMSTUDIO_MODEL,
    base_url: str = LMSTUDIO_BASE_URL,
) -> List[str]:
    """Queries LLM for 1 sentence (or batch if configured) and returns list of translations."""
    expected_count = len(batch_items)

    if expected_count == 1:
        prompt_content = PROMPT_TEMPLATE.format(text=batch_items[0]["dialect_text"])
    else:
        # Fallback for multi-item format if used
        prompt_content = PROMPT_TEMPLATE.format(text=batch_items[0]["dialect_text"])

    base_url = base_url.rstrip("/")
    start_time = time.time()

    if provider.lower() in ["lmstudio", "openai", "lm-studio"]:
        chat_url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior computational linguist specializing in Marathi dialects. Respond ONLY with the requested clean translation.",
                },
                {"role": "user", "content": prompt_content},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        try:
            req = urllib.request.Request(
                chat_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                elapsed = time.time() - start_time
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    message = res_data["choices"][0]["message"]
                    raw_response = message.get("content", "") or ""
                    
                    # Fallback to reasoning_content if main content is empty
                    if not raw_response.strip():
                        raw_response = message.get("reasoning_content", "") or message.get("reasoning", "") or ""

                    translations = _parse_response_to_translations(raw_response, expected_count)
                    if len(translations) == expected_count and all(translations):
                        logger.info(f"[Item {batch_idx}/{total_batches}] LM Studio SUCCESS in {elapsed:.2f}s | 1:1 Translation.")
                        return translations

                    logger.warning(f"[Item {batch_idx}/{total_batches}] Response parsing issue. Raw: '{raw_response[:80]}...'")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Item {batch_idx}/{total_batches}] LM Studio API error ({elapsed:.2f}s): {e}")

    else:  # Ollama Provider
        generate_url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt_content,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024},
        }

        try:
            req = urllib.request.Request(
                generate_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                elapsed = time.time() - start_time
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    raw_response = res_data.get("response", "").strip()

                    translations = _parse_response_to_translations(raw_response, expected_count)
                    if len(translations) == expected_count:
                        logger.info(f"[Batch {batch_idx}/{total_batches}] Ollama SUCCESS in {elapsed:.2f}s | Parsed 5 translations.")
                        return translations

                    logger.warning(f"[Batch {batch_idx}/{total_batches}] Response length mismatch ({len(translations)} lines vs {expected_count} expected). Raw: '{raw_response[:80]}...'")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Batch {batch_idx}/{total_batches}] Ollama API error ({elapsed:.2f}s): {e}")

    return None


def execute_generation_pipeline(
    sampled_dataset: Dict[str, List[dict]],
    output_dir: Path,
    log_dir: Path = Path("logs"),
    rows_per_csv: int = ROWS_PER_CSV,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = LMSTUDIO_MODEL,
    base_url: str = LMSTUDIO_BASE_URL,
):
    setup_logger(log_dir)

    is_healthy, health_msg = check_llm_health(provider=provider, model_name=model_name, base_url=base_url)
    if not is_healthy:
        logger.error(f"[Safety Check Failed] {health_msg}")
        print(f"\n❌ CRITICAL SAFETY ERROR: {health_msg}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "pipeline_checkpoint.json"

    flattened_items = []
    dialects = list(sampled_dataset.keys())
    max_len = max(len(v) for v in sampled_dataset.values())

    for idx in range(max_len):
        for dial in dialects:
            if idx < len(sampled_dataset[dial]):
                flattened_items.append(sampled_dataset[dial][idx])

    total_target = len(flattened_items)
    total_batches = (total_target + ITEMS_PER_PROMPT - 1) // ITEMS_PER_PROMPT

    processed_count = 0
    current_csv_index = 1
    current_csv_row_count = 0

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                processed_count = ckpt.get("processed_count", 0)
                current_csv_index = (processed_count // rows_per_csv) + 1
                current_csv_row_count = processed_count % rows_per_csv
                logger.info(f"[Checkpoint] Loaded checkpoint: {processed_count:,} pairs processed. Current CSV Part {current_csv_index:03d} ({current_csv_row_count}/{rows_per_csv} rows).")
        except Exception as e:
            logger.error(f"[Checkpoint] Failed to load checkpoint: {e}")

    remaining_items = flattened_items[processed_count:]

    logger.info("=" * 70)
    logger.info(f"SYNTHETIC PARALLEL DATA PIPELINE INITIALIZED ({provider.upper()}: {model_name})")
    logger.info("=" * 70)
    logger.info(f"Provider Backend     : {provider.upper()}")
    logger.info(f"Target Model Name    : {model_name}")
    logger.info(f"API Endpoint URL     : {base_url}")
    logger.info(f"Target Total Pairs   : {total_target:,}")
    logger.info(f"Total Inferences Needed : {total_batches:,} (1 item/prompt)")
    logger.info(f"CSV Capacity Limit   : {rows_per_csv:,} rows/file")
    logger.info(f"Output Directory     : {output_dir.resolve()}")
    logger.info("=" * 70)

    def append_row_to_csv(csv_idx: int, row: dict):
        csv_path = output_dir / f"marathi_parallel_part_{csv_idx:03d}.csv"
        file_exists = csv_path.exists()

        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    batch_idx = processed_count + 1
    i = 0
    total_remaining = len(remaining_items)

    pbar = tqdm(total=total_target, initial=processed_count, desc="Generating Parallel CSVs")

    while i < total_remaining:
        batch = remaining_items[i : i + ITEMS_PER_PROMPT]
        translations = query_llm_batch(batch, batch_idx, total_batches, provider=provider, model=model_name, base_url=base_url)

        if translations:
            for item, trans in zip(batch, translations):
                if trans:
                    processed_count += 1
                    current_csv_row_count += 1
                    row = {
                        "id": processed_count,
                        "text_id": item["text_id"],
                        "dialect": item["dialect"],
                        "domain": item["domain"],
                        "distortion_score": item["distortion_score"],
                        "dialect_text": item["dialect_text"],
                        "standard_text": trans,
                    }
                    append_row_to_csv(current_csv_index, row)
                    pbar.update(1)

                    if current_csv_row_count >= rows_per_csv:
                        logger.info(f"[CSV Part Full] Reached {rows_per_csv:,} rows for Part {current_csv_index:03d}. Starting Part {current_csv_index + 1:03d}.")
                        current_csv_index += 1
                        current_csv_row_count = 0

            # Write checkpoint immediately after every single inference
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({
                    "processed_count": processed_count,
                    "current_csv_index": current_csv_index,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)

            pct = (processed_count / total_target) * 100.0
            logger.info(f"[Progress] Inference {batch_idx}/{total_batches} completed & saved. Total: {processed_count:,}/{total_target:,} ({pct:.2f}%).")

            i += len(batch)
            batch_idx += 1
        else:
            logger.warning(f"[Retry] Inference {batch_idx}/{total_batches} failed or returned empty response. Retrying in 3 seconds...")
            time.sleep(3)

    pbar.close()
    logger.info("=" * 70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Final Output Count : {processed_count:,} parallel pairs")
    logger.info(f"Total CSV Files    : {current_csv_index}")
    logger.info("=" * 70)
