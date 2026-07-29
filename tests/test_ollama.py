"""
Test script to verify if Ollama server is running and model 'gemma4:12b' is active.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def check_ollama_status(
    base_url: str = "http://localhost:11434",
    target_model: str = "gemma4:12b",
    test_prompt: str = "Hello! Please confirm you are active and briefly state your model identity.",
) -> bool:
    base_url = base_url.rstrip("/")
    tags_url = f"{base_url}/api/tags"
    generate_url = f"{base_url}/api/generate"

    print("=" * 65)
    print("OLLAMA MODEL ACTIVITY TEST")
    print("=" * 65)
    print(f"Target Server URL : {base_url}")
    print(f"Target Model      : {target_model}")
    print("-" * 65)

    print("[Step 1/3] Checking Ollama server connection...")
    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "Python-Ollama-Test"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                raw_data = response.read().decode("utf-8")
                tags_data = json.loads(raw_data)
                available_models = [m.get("name", "") for m in tags_data.get("models", [])]
                print(f"  --> Server Status: ONLINE (HTTP 200)")
                print(f"  --> Installed Models ({len(available_models)}): {', '.join(available_models)}")
            else:
                print(f"  --> [ERROR] Unexpected HTTP status code: {response.status}")
                return False
    except urllib.error.URLError as e:
        print(f"  --> [FAIL] Unable to connect to Ollama at {base_url}.")
        print(f"      Reason: {e.reason}")
        print("      Please ensure Ollama is installed and running (`ollama serve`).")
        return False
    except Exception as e:
        print(f"  --> [FAIL] Error pinging Ollama server: {e}")
        return False

    print("\n[Step 2/3] Verifying target model presence...")
    is_model_present = any(
        target_model.lower() in m.lower() or m.lower().startswith(target_model.lower())
        for m in available_models
    )
    if is_model_present:
        print(f"  --> [SUCCESS] Model '{target_model}' found in local Ollama repository.")
    else:
        print(f"  --> [WARNING] Model '{target_model}' was NOT found in listed models.")
        print(f"      Available models: {available_models}")

    print(f"\n[Step 3/3] Sending test inference request to '{target_model}'...")
    payload = {
        "model": target_model,
        "prompt": test_prompt,
        "stream": False,
        "options": {
            "num_predict": 100,
            "temperature": 0.2,
        },
    }

    start_time = time.time()
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            generate_url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            elapsed = time.time() - start_time
            if response.status == 200:
                res_json = json.loads(response.read().decode("utf-8"))
                model_reply = res_json.get("response", "").strip()
                eval_count = res_json.get("eval_count", 0)
                eval_duration_ns = res_json.get("eval_duration", 1)
                tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

                print(f"  --> Response Status: SUCCESS (HTTP 200)")
                print(f"  --> Inference Time : {elapsed:.2f} seconds")
                if eval_count > 0:
                    print(f"  --> Token Speed    : {tokens_per_sec:.2f} tokens/sec ({eval_count} tokens generated)")
                print("\n" + "=" * 65)
                print(f"MODEL RESPONSE ({target_model}):")
                print("-" * 65)
                print(model_reply)
                print("=" * 65)
                print(f"\n✅ RESULT: Ollama server is ONLINE and model '{target_model}' is ACTIVE!")
                return True
            else:
                print(f"  --> [FAIL] Model inference failed with HTTP status code {response.status}")
                return False
    except Exception as e:
        print(f"  --> [FAIL] Error during inference: {e}")
        return False


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Check if Ollama server is running and model gemma4:12b is active.")
    parser.add_argument("--url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--model", type=str, default="gemma4:12b", help="Model name to check")
    parser.add_argument("--prompt", type=str, default="Hello! Please confirm active status.", help="Test prompt")

    args = parser.parse_args()
    success = check_ollama_status(base_url=args.url, target_model=args.model, test_prompt=args.prompt)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
