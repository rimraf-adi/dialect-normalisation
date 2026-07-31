"""
IndicBART Fine-tuning Engine with 75/25 Train-Test Split and 3-Fold Cross-Validation.
Supports D1 (Malvani), D2 (Ahirani), D4 (Varhadi), or Combined (D1+D2+D4) datasets.
"""

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger("dialect_norm.training")

class DialectDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_input_len: int = 128, max_target_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item["dialect_text"]
        tgt_text = item["standard_text"]

        inputs = self.tokenizer(
            src_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            tgt_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

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
                        "domain": row.get("domain", "")
                    })
    logger.info(f"Loaded {len(all_data):,} parallel samples from {len(csv_paths)} CSV file(s).")
    return all_data

def train_indicbart_3fold_cv(
    csv_paths: List[Path],
    output_dir: Path,
    model_name: str = "ai4bharat/IndicBART",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    seed: int = 42,
):
    """
    Executes 75/25 Train-Test Split with 3-Fold Cross-Validation on the 75% training pool.
    Evaluates each fold on the held-out 25% test set.
    """
    logging.basicConfig(level=logging.INFO)
    random.seed(seed)
    torch.manual_seed(seed)

    logger.info("=" * 70)
    logger.info(f"INDICBART FINE-TUNING (75/25 SPLIT + 3-FOLD CROSS VALIDATION)")
    logger.info("=" * 70)
    logger.info(f"Target Model       : {model_name}")
    logger.info(f"Output Directory   : {output_dir.resolve()}")
    logger.info(f"Epochs per Fold    : {epochs}")
    logger.info(f"Batch Size         : {batch_size}")
    logger.info(f"Learning Rate      : {learning_rate}")
    logger.info(f"Random Seed        : {seed}")
    logger.info("=" * 70)

    raw_data = load_dialect_data(csv_paths)
    if not raw_data:
        logger.error("No valid dataset samples found! Exiting.")
        sys.exit(1)

    # 1. 75% / 25% Train-Test Split
    shuffled_data = list(raw_data)
    random.shuffle(shuffled_data)

    split_idx = int(len(shuffled_data) * 0.75)
    train_pool = shuffled_data[:split_idx]
    test_set = shuffled_data[split_idx:]

    logger.info(f"Total Dataset Size : {len(shuffled_data):,} pairs")
    logger.info(f"75% Training Pool  : {len(train_pool):,} pairs")
    logger.info(f"25% Held-Out Test  : {len(test_set):,} pairs")

    # 2. Prepare 3-Fold CV on 75% Training Pool
    n_folds = 3
    fold_size = len(train_pool) // n_folds
    fold_metrics = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for fold in range(1, n_folds + 1):
        logger.info("\n" + "=" * 70)
        logger.info(f"RUNNING CROSS-VALIDATION: FOLD {fold}/{n_folds}")
        logger.info("=" * 70)

        val_start = (fold - 1) * fold_size
        val_end = val_start + fold_size if fold < n_folds else len(train_pool)

        fold_val_data = train_pool[val_start:val_end]
        fold_train_data = train_pool[:val_start] + train_pool[val_end:]

        logger.info(f"Fold {fold} Training Samples   : {len(fold_train_data):,}")
        logger.info(f"Fold {fold} Validation Samples : {len(fold_val_data):,}")

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        train_dataset = DialectDataset(fold_train_data, tokenizer)
        val_dataset = DialectDataset(fold_val_data, tokenizer)
        test_dataset = DialectDataset(test_set, tokenizer)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        fold_output_dir = output_dir / f"fold_{fold}"

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(fold_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            predict_with_generate=True,
            logging_dir=str(fold_output_dir / "logs"),
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train fold model
        trainer.train()

        # Evaluate fold model on 25% held-out test set
        logger.info(f"Evaluating Fold {fold} Model on 25% Held-Out Test Set ({len(test_set):,} samples)...")
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        
        logger.info(f"Fold {fold} Test Loss: {test_results.get('test_loss', 0.0):.4f}")
        
        # Save fold model
        fold_model_path = fold_output_dir / "model"
        model.save_pretrained(fold_model_path)
        tokenizer.save_pretrained(fold_model_path)
        
        fold_metrics.append({
            "fold": fold,
            "train_samples": len(fold_train_data),
            "val_samples": len(fold_val_data),
            "test_loss": test_results.get("test_loss", 0.0),
            "eval_loss": test_results.get("eval_loss", 0.0),
        })

    # Summary Report across 3 Folds
    avg_test_loss = sum(m["test_loss"] for m in fold_metrics) / n_folds
    
    summary = {
        "dataset_files": [str(p) for p in csv_paths],
        "total_samples": len(shuffled_data),
        "train_pool_size": len(train_pool),
        "test_set_size": len(test_set),
        "n_folds": n_folds,
        "avg_test_loss": avg_test_loss,
        "fold_metrics": fold_metrics
    }

    summary_file = output_dir / "cv_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("3-FOLD CROSS VALIDATION & TEST EVALUATION COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"Average Held-Out Test Loss across 3 Folds: {avg_test_loss:.4f}")
    logger.info(f"CV Summary Report saved to: {summary_file.resolve()}")
    logger.info("=" * 70)

def cli_train_d1():
    data_dir = Path("data/synthetic_parallel")
    train_indicbart_3fold_cv(csv_paths=[data_dir / "d1.csv"], output_dir=Path("models/indicbart_d1"))

def cli_train_d2():
    data_dir = Path("data/synthetic_parallel")
    train_indicbart_3fold_cv(csv_paths=[data_dir / "d2.csv"], output_dir=Path("models/indicbart_d2"))

def cli_train_d4():
    data_dir = Path("data/synthetic_parallel")
    train_indicbart_3fold_cv(csv_paths=[data_dir / "d4.csv"], output_dir=Path("models/indicbart_d4"))

def cli_train_all():
    data_dir = Path("data/synthetic_parallel")
    csv_paths = [data_dir / "d1.csv", data_dir / "d2.csv", data_dir / "d4.csv"]
    train_indicbart_3fold_cv(csv_paths=csv_paths, output_dir=Path("models/indicbart_combined"))
