r"""
Test script to verify 1-prompt-per-sentence translation quality on problematic dataset samples.
"""
import sys
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dialect_norm.gemma_pipeline import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    check_llm_health,
    query_llm_batch,
)


def run_pipeline_test():
    print("=== SINGLE SENTENCE (1 PROMPT PER SENTENCE) TRANSLATION QUALITY TEST ===", flush=True)

    healthy, msg = check_llm_health("lmstudio", LMSTUDIO_MODEL, LMSTUDIO_BASE_URL)
    print(f"Health status: {healthy} ({msg})", flush=True)
    if not healthy:
        return False

    problematic_samples = [
        {"dialect": "D2", "text_id": "712936", "dialect_text": "सघन शेती आणि जास्तीनी जमीन यानामान सधन शेतीले जास्तिनी मेहनत आणि जास्तीना पैसा लागस"},
        {"dialect": "D2", "text_id": "714055", "dialect_text": "देशना पुरा मासामारी उत्पादनपैकी सुमारे साठ एकर उत्पादन देशमझारला मासामारीमाथून येस"},
        {"dialect": "D1", "text_id": "712275", "dialect_text": "क्रेडिट कार्ड वरसून जास्तीत जास्त कितक्या कर्ज गावता , आणि डेबिट कार्ड वरसून जास्तीत जास्त कितके पैसे काढुक शकतू आम्ही ?"},
        {"dialect": "D2", "text_id": "713155", "dialect_text": "एस बँक नी देवान घेवानना इतिहास देखानी ऑनलाईन प्रणाली बराबर नही शे"},
        {"dialect": "D4", "text_id": "724336", "dialect_text": "मले शिकायले कर्ज घ्याच असन तर कुठ अर्ज करा लागन त्याची मायती मले कुठी भेटन त्यासाठी ऑनलाईन अर्ज करा लागन का ?"},
    ]

    print("\n--- Running 1 Prompt per Sentence Sequential Inferences ---", flush=True)
    results = []

    for i, item in enumerate(problematic_samples, 1):
        print(f"\n[{i}/5] Translating Dialect [{item['dialect']}]: {item['dialect_text']}", flush=True)
        res = query_llm_batch(
            batch_items=[item],
            batch_idx=i,
            total_batches=len(problematic_samples),
            provider="lmstudio",
            model=LMSTUDIO_MODEL,
            base_url=LMSTUDIO_BASE_URL,
        )
        if res and res[0]:
            print(f"   --> Standard Pune: {res[0]}", flush=True)
            results.append((item, res[0]))
        else:
            print(f"   ❌ Failed to translate item {i}", flush=True)

    print("\n=== SUMMARY RESULTS (1 PROMPT PER SENTENCE) ===", flush=True)
    for i, (orig, trans) in enumerate(results, 1):
        print(f"\nSample #{i} [{orig['dialect']}]:")
        print(f"  Dialect Text  : {orig['dialect_text']}")
        print(f"  Standard Pune : {trans}")

    return len(results) == len(problematic_samples)


if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
