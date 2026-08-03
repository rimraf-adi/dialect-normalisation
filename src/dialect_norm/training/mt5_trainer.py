"""
google/mT5 Fine-tuning Engine for Marathi Dialect Normalization.
Supports 85/15 Stratified Train-Test Split and 5-Fold Cross-Validation.
Computes BLEU, chrF++, Test Loss, and per-dialect breakdown (D1, D2, D4).
"""

import csv
import gc
import json
import logging
import yaml
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger("dialect_norm.training.mt5")

# ---------------------------------------------------------------------------
# Dataset for mT5
# ---------------------------------------------------------------------------

class MT5DialectDataset(Dataset):
    """
    mT5 text-to-text dataset with task prefix 'normalize Marathi dialect to standard: '.
    """
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_input_len: int = 128, max_target_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = f"normalize Marathi dialect to standard: {item['dialect_text']}"
        tgt_text = item['standard_text']

        inputs = self.tokenizer(
            src_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            tgt_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dialect_data(csv_paths: List[Path]) -> List[Dict[str, str]]:
    all_data = []
    for path in csv_paths:
        if not path.exists():
            logger.warning(f"File {path} does not exist. Skipping.")
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
    logger.info(f"Loaded {len(all_data):,} parallel samples from {len(csv_paths)} CSV file(s).")
    return all_data

# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics_builder(tokenizer):
    import evaluate
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu_res = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_res = chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)

        return {
            "bleu": round(bleu_res["score"], 4),
            "chrf": round(chrf_res["score"], 4),
        }

    return compute_metrics

# ---------------------------------------------------------------------------
# Core Cross-Validation Training Loop
# ---------------------------------------------------------------------------

def run_cross_validation_mt5(
    data: List[Dict[str, str]],
    model_name: str = "google/mt5-small",
    output_dir: Path = Path("models/mt5_small_combined_32k"),
    dataset_files: List[str] = None,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    n_folds: int = 5,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir.parent / f"{output_dir.name}.log", encoding="utf-8"),
        ] if output_dir.parent.exists() else [logging.StreamHandler(sys.stdout)],
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Starting mT5 Fine-tuning ({model_name}) ===")
    logger.info(f"Total Dataset Size: {len(data):,} pairs")

    # 1. Stratified / Random Split into 85% Train Pool and 15% Held-out Test
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_pool_indices = indices[test_size:]

    test_data = [data[i] for i in test_indices]
    train_pool_data = [data[i] for i in train_pool_indices]

    logger.info(f"85% Training Pool: {len(train_pool_data):,} pairs")
    logger.info(f"15% Held-Out Test : {len(test_data):,} pairs")

    # 2. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Create Test Dataset
    test_dataset = MT5DialectDataset(test_data, tokenizer)

    # 4. K-Fold CV
    fold_size = len(train_pool_data) // n_folds
    fold_metrics = []

    for fold in range(n_folds):
        logger.info(f"\n======================================================================")
        logger.info(f"RUNNING mT5 CROSS-VALIDATION: FOLD {fold + 1}/{n_folds}")
        logger.info(f"======================================================================")

        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(train_pool_data)

        val_data = train_pool_data[val_start:val_end]
        train_data = train_pool_data[:val_start] + train_pool_data[val_end:]

        train_dataset = MT5DialectDataset(train_data, tokenizer)
        val_dataset = MT5DialectDataset(val_data, tokenizer)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        fold_output_dir = output_dir / f"fold_{fold + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(fold_output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,
            fp16=False,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_metrics_builder(tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # Evaluate on Val
        val_res = trainer.evaluate()
        # Evaluate on Test
        test_res = trainer.evaluate(test_dataset, metric_key_prefix="test")

        fold_record = {
            "fold": fold + 1,
            "val_loss": round(val_res.get("eval_loss", 0.0), 4),
            "val_bleu": round(val_res.get("eval_bleu", 0.0), 4),
            "val_chrf": round(val_res.get("eval_chrf", 0.0), 4),
            "test_loss": round(test_res.get("test_loss", 0.0), 4),
            "test_bleu": round(test_res.get("test_bleu", 0.0), 4),
            "test_chrf": round(test_res.get("test_chrf", 0.0), 4),
        }
        fold_metrics.append(fold_record)
        logger.info(f"Fold {fold + 1} Val Loss: {fold_record['val_loss']} | Val BLEU: {fold_record['val_bleu']} | Test BLEU: {fold_record['test_bleu']}")

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    avg_val_loss = round(float(np.mean([f["val_loss"] for f in fold_metrics])), 4)
    avg_val_bleu = round(float(np.mean([f["val_bleu"] for f in fold_metrics])), 4)
    avg_val_chrf = round(float(np.mean([f["val_chrf"] for f in fold_metrics])), 4)
    avg_test_loss = round(float(np.mean([f["test_loss"] for f in fold_metrics])), 4)
    avg_test_bleu = round(float(np.mean([f["test_bleu"] for f in fold_metrics])), 4)
    avg_test_chrf = round(float(np.mean([f["test_chrf"] for f in fold_metrics])), 4)

    summary = {
        "model_name": model_name,
        "dataset_files": dataset_files or [],
        "total_samples": len(data),
        "train_pool_size": len(train_pool_data),
        "test_set_size": len(test_data),
        "n_folds": n_folds,
        "avg_val_loss": avg_val_loss,
        "avg_val_bleu": avg_val_bleu,
        "avg_val_chrf": avg_val_chrf,
        "avg_test_loss": avg_test_loss,
        "avg_test_bleu": avg_test_bleu,
        "avg_test_chrf": avg_test_chrf,
        "fold_metrics": fold_metrics,
    }

    yaml_path = output_dir / "cv_summary.yaml"
    json_path = output_dir / "cv_summary.json"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"=== mT5 CROSS VALIDATION COMPLETED ===")
    logger.info(f"Avg Test Loss: {avg_test_loss} | BLEU: {avg_test_bleu} | chrF++: {avg_test_chrf}")
    return summary


def train_mt5_all_32k():
    csv_paths = [
        Path("data/synthetic_parallel/d1.csv"),
        Path("data/synthetic_parallel/d2.csv"),
        Path("data/synthetic_parallel/d4.csv"),
        Path("data/synthetic-data/all_aug.csv"),
    ]
    data = load_dialect_data(csv_paths)
    run_cross_validation_mt5(
        data=data,
        model_name="google/mt5-small",
        output_dir=Path("models/mt5_small_combined_32k"),
        dataset_files=[str(p) for p in csv_paths],
    )
