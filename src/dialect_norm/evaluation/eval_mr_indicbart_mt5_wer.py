import csv
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import jiwer
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configure logger
logger = logging.getLogger("dialect_norm.evaluation")

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger(log_path: Path):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    fh = FlushingFileHandler(str(log_path), encoding="utf-8", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_dialect_data(csv_paths: List[Path]) -> List[Dict[str, str]]:
    all_data = []
    for path in csv_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_text = row.get("dialect_text", "").strip()
                s_text = row.get("standard_text", "").strip()
                if d_text and s_text:
                    all_data.append({
                        "dialect_text": d_text,
                        "standard_text": s_text,
                        "dialect": row.get("dialect", ""),
                        "domain": row.get("domain", ""),
                    })
    return all_data

def get_test_set(csv_paths: List[Path], test_ratio: float = 0.15, seed: int = 42) -> List[Dict[str, str]]:
    data = load_dialect_data(csv_paths)
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    return [data[i] for i in test_indices]

def clean_text(text: str) -> str:
    text = re.sub(r"<2[a-z]{2}>", "", text)
    text = re.sub(r"[^\u0900-\u097F\s.,!?-]", "", text)
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"\.+", ".", text)
    text = re.sub(r"-+", "-", text)
    return re.sub(r"\s+", " ", text).strip()

def run_model_inference(
    model_path: Path,
    test_data: List[Dict[str, str]],
    is_indicbart: bool = False,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[str]:
    logger.info(f"Loading checkpoint: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)
    model.eval()

    inputs = [d["dialect_text"] for d in test_data]
    if is_indicbart:
        inputs = [f"{t} <2mr>" for t in inputs]

    predictions = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_texts = inputs[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            gen_tokens = model.generate(
                **encoded,
                max_length=128,
                num_beams=2,
                early_stopping=True,
            )

            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            predictions.extend([clean_text(d) for d in decoded])

            if (i // batch_size + 1) % 10 == 0 or (i + batch_size >= len(inputs)):
                logger.info(f"  Processed {min(i + batch_size, len(inputs)):,}/{len(inputs):,} sentences...")

    return predictions

def compute_detailed_metrics(
    inputs: List[str],
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    refs_nested = [[r] for r in references]

    bleu_res = sacrebleu.compute(predictions=predictions, references=refs_nested)
    chrf_res = chrf.compute(predictions=predictions, references=refs_nested, word_order=2)

    raw_wer = float(jiwer.wer(references, inputs)) * 100.0
    raw_cer = float(jiwer.cer(references, inputs)) * 100.0

    norm_wer = float(jiwer.wer(references, predictions)) * 100.0
    norm_cer = float(jiwer.cer(references, predictions)) * 100.0

    wer_reduction = ((raw_wer - norm_wer) / raw_wer) * 100.0 if raw_wer > 0 else 0.0

    return {
        "bleu": round(bleu_res["score"], 2),
        "chrf": round(chrf_res["score"], 2),
        "raw_wer": round(raw_wer, 2),
        "raw_cer": round(raw_cer, 2),
        "norm_wer": round(norm_wer, 2),
        "norm_cer": round(norm_cer, 2),
        "wer_reduction": round(wer_reduction, 2),
    }

def main():
    log_file = Path("logs/eval_mr_indicbart_mt5.log")
    setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MARATHI TEST EVALUATION: INDICBART vs mT5-SMALL")
    logger.info("Metrics: BLEU, chrF++, WER (%), CER (%), and Baseline WER Reduction (%)")
    logger.info("=" * 80)

    def find_best_model(base_dir: Path) -> Path:
        if not base_dir.exists():
            return None
        matches = list(base_dir.glob("**/best_model"))
        if matches:
            return matches[0]
        if (base_dir / "config.json").exists():
            return base_dir
        return None

    tasks = [
        # 16k Original Datasets
        ("D1 Malvani (16k Original)", [Path("data/synthetic_parallel/d1.csv")], find_best_model(Path("models/indicbart_d1")), find_best_model(Path("models/mt5_d1_16k"))),
        ("D2 Ahirani (16k Original)", [Path("data/synthetic_parallel/d2.csv")], find_best_model(Path("models/indicbart_d2")), find_best_model(Path("models/mt5_d2_16k"))),
        ("D4 Varhadi (16k Original)", [Path("data/synthetic_parallel/d4.csv")], find_best_model(Path("models/indicbart_d4")), find_best_model(Path("models/mt5_d4_16k"))),
        ("D124 Combined (16k Original)", [Path("data/synthetic_parallel/d1.csv"), Path("data/synthetic_parallel/d2.csv"), Path("data/synthetic_parallel/d4.csv")], find_best_model(Path("models/indicbart_combined")), find_best_model(Path("models/mt5_combined_16k"))),
        
        # 32k Expanded Datasets
        ("D1 Malvani (32k Expanded)", [Path("data/synthetic_parallel/d1.csv"), Path("data/synthetic-data/all_aug.csv")], find_best_model(Path("models/indicbart_d1_32k")), find_best_model(Path("models/mt5_d1_32k"))),
        ("D2 Ahirani (32k Expanded)", [Path("data/synthetic_parallel/d2.csv"), Path("data/synthetic-data/all_aug.csv")], find_best_model(Path("models/indicbart_d2_32k")), find_best_model(Path("models/mt5_d2_32k"))),
        ("D4 Varhadi (32k Expanded)", [Path("data/synthetic_parallel/d4.csv"), Path("data/synthetic-data/all_aug.csv")], find_best_model(Path("models/indicbart_d4_32k")), find_best_model(Path("models/mt5_d4_32k"))),
        ("D124 Combined (32k Expanded)", [Path("data/synthetic_parallel/d1.csv"), Path("data/synthetic_parallel/d2.csv"), Path("data/synthetic_parallel/d4.csv"), Path("data/synthetic-data/all_aug.csv")], find_best_model(Path("models/indicbart_combined_32k")), find_best_model(Path("models/mt5_combined_32k"))),
    ]

    summary_rows = []

    for name, files, indicbart_model, mt5_model in tasks:
        logger.info("\n" + "=" * 80)
        logger.info(f"EVALUATING VARIANT: {name}")
        logger.info("=" * 80)

        test_data = get_test_set(files)
        logger.info(f"Test Set Size: {len(test_data):,} pairs")

        dialect_inputs = [clean_text(d["dialect_text"]) for d in test_data]
        targets = [clean_text(d["standard_text"]) for d in test_data]

        # Calculate Baseline (Unnormalized Dialect Text vs Standard Target)
        raw_wer = round(float(jiwer.wer(targets, dialect_inputs)) * 100.0, 2)
        raw_cer = round(float(jiwer.cer(targets, dialect_inputs)) * 100.0, 2)

        logger.info(f"RAW DIALECT BASELINE (Unnormalized): WER = {raw_wer:.2f}% | CER = {raw_cer:.2f}%")

        # 1. IndicBART Inference / Metric Retrieval
        indicbart_preds = []
        indicbart_metrics = None
        if indicbart_model and indicbart_model.exists():
            logger.info(f"\n--- Running IndicBART (244M) Test Inference ---")
            indicbart_preds = run_model_inference(indicbart_model, test_data, is_indicbart=True)
            indicbart_metrics = compute_detailed_metrics(dialect_inputs, indicbart_preds, targets)
            logger.info(f"IndicBART Metrics: BLEU = {indicbart_metrics['bleu']} | chrF++ = {indicbart_metrics['chrf']} | WER = {indicbart_metrics['norm_wer']}% | CER = {indicbart_metrics['norm_cer']}% | WER Reduction = -{indicbart_metrics['wer_reduction']}%")
        else:
            # Fallback to recorded cv_summary.yaml metrics for 16k IndicBART runs
            yaml_map = {
                "D1 Malvani (16k Original)": Path("models/indicbart_d1/cv_summary.yaml"),
                "D2 Ahirani (16k Original)": Path("models/indicbart_d2/cv_summary.yaml"),
                "D4 Varhadi (16k Original)": Path("models/indicbart_d4/cv_summary.yaml"),
                "D124 Combined (16k Original)": Path("models/indicbart_combined/cv_summary.yaml"),
            }
            yaml_path = yaml_map.get(name)
            if yaml_path and yaml_path.exists():
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    summary_data = yaml.safe_load(f)
                b = round(float(summary_data.get("avg_test_bleu", 0.0)), 2)
                c = round(float(summary_data.get("avg_test_chrf", 0.0)), 2)
                logger.info(f"IndicBART Recorded CV Metrics ({yaml_path.parent.name}): BLEU = {b} | chrF++ = {c}")
                indicbart_metrics = {"bleu": b, "chrf": c, "norm_wer": "N/A", "norm_cer": "N/A", "wer_reduction": "N/A"}
            else:
                logger.info(f"IndicBART model checkpoint not found for {name}. Skipping.")

        # 2. mT5-small Inference
        mt5_preds = []
        mt5_metrics = None
        if mt5_model and mt5_model.exists():
            logger.info(f"\n--- Running mT5-Small (300M) Test Inference ---")
            mt5_preds = run_model_inference(mt5_model, test_data, is_indicbart=False)
            mt5_metrics = compute_detailed_metrics(dialect_inputs, mt5_preds, targets)
            logger.info(f"mT5-Small Metrics: BLEU = {mt5_metrics['bleu']} | chrF++ = {mt5_metrics['chrf']} | WER = {mt5_metrics['norm_wer']}% | CER = {mt5_metrics['norm_cer']}% | WER Reduction = -{mt5_metrics['wer_reduction']}%")
        else:
            logger.info(f"mT5-Small model checkpoint not found for {name}. Skipping.")

        # Log Sample Predictions
        logger.info(f"\n--- Sample Qualtitative Prediction Inspection for {name} ---")
        for s_idx in range(min(3, len(test_data))):
            logger.info(f"[Sample {s_idx + 1}]")
            logger.info(f"  Dialect Input : '{dialect_inputs[s_idx]}'")
            logger.info(f"  Ground Truth  : '{targets[s_idx]}'")
            if indicbart_preds:
                logger.info(f"  IndicBART Pred: '{indicbart_preds[s_idx]}'")
            if mt5_preds:
                logger.info(f"  mT5-Small Pred: '{mt5_preds[s_idx]}'")

        summary_rows.append({
            "variant": name,
            "samples": len(test_data),
            "raw_wer": raw_wer,
            "raw_cer": raw_cer,
            "ib_bleu": indicbart_metrics["bleu"] if indicbart_metrics else "N/A",
            "ib_chrf": indicbart_metrics["chrf"] if indicbart_metrics else "N/A",
            "ib_wer": indicbart_metrics["norm_wer"] if indicbart_metrics else "N/A",
            "ib_cer": indicbart_metrics["norm_cer"] if indicbart_metrics else "N/A",
            "ib_wer_drop": indicbart_metrics["wer_reduction"] if indicbart_metrics else "N/A",
            "mt5_bleu": mt5_metrics["bleu"] if mt5_metrics else "N/A",
            "mt5_chrf": mt5_metrics["chrf"] if mt5_metrics else "N/A",
            "mt5_wer": mt5_metrics["norm_wer"] if mt5_metrics else "N/A",
            "mt5_cer": mt5_metrics["norm_cer"] if mt5_metrics else "N/A",
            "mt5_wer_drop": mt5_metrics["wer_reduction"] if mt5_metrics else "N/A",
        })

    logger.info("\n" + "=" * 100)
    logger.info("FINAL SUMMARY COMPARISON TABLE: MARATHI TEST WER / CER / BLEU / chrF++")
    logger.info("=" * 100)
    logger.info(f"{'Variant':<32} | {'Raw WER':<8} | {'IB BLEU':<8} | {'IB WER':<8} | {'IB Drop':<8} | {'mT5 BLEU':<8} | {'mT5 WER':<8} | {'mT5 Drop':<8}")
    logger.info("-" * 100)
    for r in summary_rows:
        logger.info(f"{r['variant']:<32} | {r['raw_wer']:<8} | {r['ib_bleu']:<8} | {r['ib_wer']:<8} | -{r['ib_wer_drop']}%   | {r['mt5_bleu']:<8} | {r['mt5_wer']:<8} | -{r['mt5_wer_drop']}%")
    logger.info("=" * 100)

if __name__ == "__main__":
    main()
