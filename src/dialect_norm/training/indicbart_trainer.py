"""
IndicBART Fine-tuning Engine with 75/25 Train-Test Split and 3-Fold Cross-Validation.
Supports D1 (Malvani), D2 (Ahirani), D4 (Varhadi), D1+D2, or Combined (D1+D2+D4) datasets.
Computes BLEU, chrF++, and evaluation loss across all folds.
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

logger = logging.getLogger("dialect_norm.training")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
        # Prepend and append IndicBART Marathi language tag <2mr>
        src_text = f"<2mr> {item['dialect_text']} <2mr>"
        tgt_text = f"<2mr> {item['standard_text']} <2mr>"

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
# Stratified 75/25 Split (preserves dialect distribution)
# ---------------------------------------------------------------------------

def stratified_split(data: List[Dict], test_ratio: float = 0.25, seed: int = 42):
    """Split data into train_pool and test_set, stratified by dialect."""
    rng = random.Random(seed)
    by_dialect: Dict[str, list] = defaultdict(list)
    for item in data:
        by_dialect[item.get("dialect", "UNK")].append(item)

    train_pool, test_set = [], []
    for dialect, items in by_dialect.items():
        rng.shuffle(items)
        split_idx = int(len(items) * (1.0 - test_ratio))
        train_pool.extend(items[:split_idx])
        test_set.extend(items[split_idx:])

    rng.shuffle(train_pool)
    rng.shuffle(test_set)
    return train_pool, test_set


# ---------------------------------------------------------------------------
# Metrics: BLEU + chrF++
# ---------------------------------------------------------------------------

def build_compute_metrics(tokenizer):
    """Returns a compute_metrics function that calculates BLEU and chrF++."""
    import evaluate

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        preds = np.where(preds >= 0, preds, pad_id)
        labels = np.where(labels >= 0, labels, pad_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        import re

        clean_preds = []
        for p in decoded_preds:
            p_clean = re.sub(r"<2[a-z]{2}>", "", p)
            p_clean = re.sub(r"[^\u0900-\u097F\s.,!?-]", "", p_clean)
            p_clean = re.sub(r"\?+", "?", p_clean)
            p_clean = re.sub(r"\.+", ".", p_clean)
            p_clean = re.sub(r"-+", "-", p_clean)
            clean_preds.append(re.sub(r"\s+", " ", p_clean).strip())

        clean_labels = []
        for l in decoded_labels:
            l_clean = re.sub(r"<2[a-z]{2}>", "", l)
            l_clean = re.sub(r"[^\u0900-\u097F\s.,!?-]", "", l_clean)
            l_clean = re.sub(r"\?+", "?", l_clean)
            l_clean = re.sub(r"\.+", ".", l_clean)
            l_clean = re.sub(r"-+", "-", l_clean)
            clean_labels.append([re.sub(r"\s+", " ", l_clean).strip()])

        bleu_result = bleu_metric.compute(predictions=clean_preds, references=clean_labels)
        chrf_result = chrf_metric.compute(predictions=clean_preds, references=clean_labels)

        # Track length ratio (pred length vs ref length)
        avg_pred_len = sum(len(p.split()) for p in clean_preds) / max(1, len(clean_preds))
        avg_ref_len = sum(len(l[0].split()) for l in clean_labels) / max(1, len(clean_labels))
        logger.info(f"Generation Length Check: Avg Pred Words = {avg_pred_len:.1f} | Avg Ref Words = {avg_ref_len:.1f}")

        # Log sample generations to verify output quality
        for i in range(min(3, len(clean_preds))):
            logger.info(f"[Sample {i+1}] Pred: '{clean_preds[i]}' | Ref: '{clean_labels[i][0]}'")

        return {
            "bleu": round(bleu_result["score"], 4),
            "chrf": round(chrf_result["score"], 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Main Training Loop: 75/25 + 3-Fold CV
# ---------------------------------------------------------------------------

def train_indicbart_3fold_cv(
    csv_paths: List[Path],
    output_dir: Path,
    model_name: str = "ai4bharat/IndicBART",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 2,
    max_input_len: int = 128,
    max_target_len: int = 128,
    early_stopping_patience: int = 2,
    test_ratio: float = 0.15,
    n_folds: int = 5,
    seed: int = 42,
):
    """
    Executes Option B: 85/15 Stratified Train-Test Split with 5-Fold Cross-Validation
    on the 85% training pool. Evaluates each fold on the held-out 15% test set.
    Reports BLEU, chrF++, and loss metrics.
    """
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging: console + file inside logs/
    log_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file = logs_dir / f"train_{output_dir.name}.log"

    # Always clear previous log file and remove existing file handlers on restart
    if log_file.exists():
        try:
            log_file.write_text("", encoding="utf-8")
        except Exception:
            pass

    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8", mode="w")
    file_handler.setFormatter(log_fmt)
    root_logger.addHandler(file_handler)

    # Console handler (if not already present)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_fmt)
        root_logger.addHandler(console_handler)

    logger.info(f"Training log file: {log_file.resolve()}")

    logger.info("=" * 70)
    logger.info(f"INDICBART FINE-TUNING (OPTION B: {int((1-test_ratio)*100)}/{int(test_ratio*100)} STRATIFIED SPLIT + {n_folds}-FOLD CV)")
    logger.info("=" * 70)
    logger.info(f"Target Model            : {model_name}")
    logger.info(f"Output Directory        : {output_dir.resolve()}")
    logger.info(f"Epochs per Fold         : {epochs}")
    logger.info(f"Batch Size              : {batch_size}")
    logger.info(f"Gradient Accum Steps    : {gradient_accumulation_steps}")
    logger.info(f"Effective Batch Size    : {batch_size * gradient_accumulation_steps}")
    logger.info(f"Learning Rate           : {learning_rate}")
    logger.info(f"Weight Decay            : {weight_decay}")
    logger.info(f"Warmup Ratio            : {warmup_ratio}")
    logger.info(f"Early Stopping Patience : {early_stopping_patience}")
    logger.info(f"Max Input / Target Len  : {max_input_len} / {max_target_len}")
    logger.info(f"Random Seed             : {seed}")
    logger.info("=" * 70)

    raw_data = load_dialect_data(csv_paths)
    if not raw_data:
        logger.error("No valid dataset samples found! Exiting.")
        sys.exit(1)

    # 1. Stratified 85% / 15% Train-Test Split
    train_pool, test_set = stratified_split(raw_data, test_ratio=test_ratio, seed=seed)

    logger.info(f"Total Dataset Size      : {len(raw_data):,} pairs")
    logger.info(f"{int((1-test_ratio)*100)}% Training Pool       : {len(train_pool):,} pairs")
    logger.info(f"{int(test_ratio*100)}% Held-Out Test       : {len(test_set):,} pairs")

    # Log dialect distribution
    for label, subset in [("Train Pool", train_pool), ("Test Set", test_set)]:
        dist = defaultdict(int)
        for item in subset:
            dist[item.get("dialect", "UNK")] += 1
        logger.info(f"  {label} distribution: {dict(dist)}")

    # 2. Download and cache base model once locally to avoid repeated HuggingFace network calls
    base_model_dir = output_dir / "base_model"
    if not base_model_dir.exists() or not any(base_model_dir.iterdir()):
        logger.info(f"Downloading and caching base model '{model_name}' to {base_model_dir.resolve()}...")
        init_tokenizer = AutoTokenizer.from_pretrained(model_name)
        init_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        base_model_dir.mkdir(parents=True, exist_ok=True)
        init_tokenizer.save_pretrained(base_model_dir)
        init_model.save_pretrained(base_model_dir)
        del init_model, init_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    compute_metrics_fn = build_compute_metrics(tokenizer)

    # 3. Prepare Cross-Validation on Training Pool
    fold_size = len(train_pool) // n_folds
    fold_metrics = []

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

        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_dir, local_files_only=True)
        # Force decoder to generate Marathi (<2mr> token ID)
        mr_token_id = tokenizer.convert_tokens_to_ids("<2mr>")
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.forced_bos_token_id = mr_token_id
            model.generation_config.decoder_start_token_id = mr_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.num_beams = 4
            model.generation_config.length_penalty = 0.8
            model.generation_config.repetition_penalty = 1.2
            model.generation_config.no_repeat_ngram_size = 4
            model.generation_config.early_stopping = True
        else:
            model.config.forced_bos_token_id = mr_token_id

        train_dataset = DialectDataset(fold_train_data, tokenizer, max_input_len, max_target_len)
        val_dataset = DialectDataset(fold_val_data, tokenizer, max_input_len, max_target_len)
        test_dataset = DialectDataset(test_set, tokenizer, max_input_len, max_target_len)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        fold_output_dir = output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(fold_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=max_target_len,
            generation_num_beams=4,
            logging_dir=str(fold_output_dir / "logs"),
            logging_steps=50,
            fp16=False,
            report_to="none",
            seed=seed,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        # Train fold model
        train_result = trainer.train()
        train_loss = train_result.metrics.get("train_loss", train_result.training_loss)
        logger.info(f"Fold {fold} Step-Averaged Training Loss: {train_loss:.4f}")

        # Evaluate fold model on CV validation split
        logger.info(f"Evaluating Fold {fold} on CV Validation Split ({len(fold_val_data):,} samples)...")
        val_results = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
        logger.info(f"Fold {fold} Val Loss: {val_results.get('eval_loss', 0.0):.4f}  "
                     f"Val BLEU: {val_results.get('eval_bleu', 0.0):.2f}  "
                     f"Val chrF: {val_results.get('eval_chrf', 0.0):.2f}")

        # Evaluate fold model on held-out test set
        logger.info(f"Evaluating Fold {fold} on Held-Out Test Set ({len(test_set):,} samples)...")
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        logger.info(f"Fold {fold} Test Loss: {test_results.get('test_loss', 0.0):.4f}  "
                     f"Test BLEU: {test_results.get('test_bleu', 0.0):.2f}  "
                     f"Test chrF: {test_results.get('test_chrf', 0.0):.2f}")

        # Per-dialect breakdown evaluation if multiple dialects present in test set
        dialect_groups = defaultdict(list)
        for item in test_set:
            dialect_groups[item.get("dialect", "UNK").upper()].append(item)

        per_dialect_test = {}
        if len(dialect_groups) > 1:
            logger.info("Running per-dialect test evaluation breakdown...")
            for dial_code, dial_samples in sorted(dialect_groups.items()):
                dial_dataset = DialectDataset(dial_samples, tokenizer, max_input_len, max_target_len)
                prefix = f"test_{dial_code.lower()}"
                dial_res = trainer.evaluate(eval_dataset=dial_dataset, metric_key_prefix=prefix)
                b_score = dial_res.get(f"{prefix}_bleu", 0.0)
                c_score = dial_res.get(f"{prefix}_chrf", 0.0)
                l_score = dial_res.get(f"{prefix}_loss", 0.0)
                per_dialect_test[dial_code] = {
                    "test_samples": len(dial_samples),
                    "test_loss": round(l_score, 4),
                    "test_bleu": round(b_score, 4),
                    "test_chrf": round(c_score, 4),
                }
                logger.info(f"  [{dial_code}] Test Samples: {len(dial_samples):,} | Loss: {l_score:.4f} | BLEU: {b_score:.2f} | chrF: {c_score:.2f}")

        # Save best fold model
        fold_model_path = fold_output_dir / "best_model"
        trainer.save_model(str(fold_model_path))
        tokenizer.save_pretrained(fold_model_path)

        metric_entry = {
            "fold": fold,
            "train_samples": len(fold_train_data),
            "val_samples": len(fold_val_data),
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_results.get("eval_loss", 0.0), 4),
            "val_bleu": round(val_results.get("eval_bleu", 0.0), 4),
            "val_chrf": round(val_results.get("eval_chrf", 0.0), 4),
            "test_loss": round(test_results.get("test_loss", 0.0), 4),
            "test_bleu": round(test_results.get("test_bleu", 0.0), 4),
            "test_chrf": round(test_results.get("test_chrf", 0.0), 4),
        }
        if per_dialect_test:
            metric_entry["per_dialect_test_metrics"] = per_dialect_test

        fold_metrics.append(metric_entry)

        # Free GPU memory between folds
        del model, trainer, train_dataset, val_dataset, test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Fold {fold} complete. GPU memory freed.")

    # -----------------------------------------------------------------------
    # Summary Report across 3 Folds
    # -----------------------------------------------------------------------
    avg_test_loss = sum(m["test_loss"] for m in fold_metrics) / n_folds
    avg_test_bleu = sum(m["test_bleu"] for m in fold_metrics) / n_folds
    avg_test_chrf = sum(m["test_chrf"] for m in fold_metrics) / n_folds
    avg_val_loss = sum(m["val_loss"] for m in fold_metrics) / n_folds
    avg_val_bleu = sum(m["val_bleu"] for m in fold_metrics) / n_folds
    avg_val_chrf = sum(m["val_chrf"] for m in fold_metrics) / n_folds

    summary = {
        "dataset_files": [str(p) for p in csv_paths],
        "total_samples": len(raw_data),
        "train_pool_size": len(train_pool),
        "test_set_size": len(test_set),
        "model_name": model_name,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "early_stopping_patience": early_stopping_patience,
        },
        "n_folds": n_folds,
        "avg_val_loss": round(avg_val_loss, 4),
        "avg_val_bleu": round(avg_val_bleu, 4),
        "avg_val_chrf": round(avg_val_chrf, 4),
        "avg_test_loss": round(avg_test_loss, 4),
        "avg_test_bleu": round(avg_test_bleu, 4),
        "avg_test_chrf": round(avg_test_chrf, 4),
        "fold_metrics": fold_metrics,
    }

    summary_file_json = output_dir / "cv_summary.json"
    with open(summary_file_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_file_yaml = output_dir / "cv_summary.yaml"
    with open(summary_file_yaml, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("\n" + "=" * 70)
    logger.info("3-FOLD CROSS VALIDATION & TEST EVALUATION COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"Avg Val  Loss: {avg_val_loss:.4f}  |  BLEU: {avg_val_bleu:.2f}  |  chrF: {avg_val_chrf:.2f}")
    logger.info(f"Avg Test Loss: {avg_test_loss:.4f}  |  BLEU: {avg_test_bleu:.2f}  |  chrF: {avg_test_chrf:.2f}")
    logger.info(f"CV Summary saved to: {summary_file_json.resolve()}")
    logger.info(f"CV Summary saved to: {summary_file_yaml.resolve()}")
    logger.info("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# CLI Entrypoints: D1, D2, D4, D1+D2, D1+D2+D4
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/synthetic_parallel")


def cli_train_d1():
    """Train IndicBART on D1 (Malvani) dialect only."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d1.csv"],
        output_dir=Path("models/indicbart_d1"),
    )


def cli_train_d2():
    """Train IndicBART on D2 (Ahirani) dialect only."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d2.csv"],
        output_dir=Path("models/indicbart_d2"),
    )


def cli_train_d4():
    """Train IndicBART on D4 (Varhadi) dialect only."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d4.csv"],
        output_dir=Path("models/indicbart_d4"),
    )


def cli_train_d1d2():
    """Train IndicBART on D1 (Malvani) + D2 (Ahirani) combined."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d1.csv", DATA_DIR / "d2.csv"],
        output_dir=Path("models/indicbart_d1d2"),
    )


def cli_train_all():
    """Train IndicBART on all three dialects: D1 + D2 + D4 combined."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d1.csv", DATA_DIR / "d2.csv", DATA_DIR / "d4.csv"],
        output_dir=Path("models/indicbart_combined"),
    )


def cli_train_d1_32k():
    """Train IndicBART on D1 Malvani Original (5,576) + Synthetic (5,569) Combined (11,145 clean pairs)."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d1.csv", Path("data/synthetic-data/d1_aug.csv")],
        output_dir=Path("models/indicbart_d1_32k"),
    )


def cli_train_d2_32k():
    """Train IndicBART on D2 Ahirani Original (5,501) + Synthetic (5,534) Combined (11,035 clean pairs)."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d2.csv", Path("data/synthetic-data/d2_aug.csv")],
        output_dir=Path("models/indicbart_d2_32k"),
    )


def cli_train_d4_32k():
    """Train IndicBART on D4 Varhadi Original (5,086) + Synthetic (5,069) Combined (10,155 clean pairs)."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "d4.csv", Path("data/synthetic-data/d4_aug.csv")],
        output_dir=Path("models/indicbart_d4_32k"),
    )


def cli_train_all_32k():
    """Train IndicBART on D1 + D2 + D4 Original (16,163) + Synthetic (16,172) Combined (32,335 clean pairs)."""
    train_indicbart_3fold_cv(
        csv_paths=[
            DATA_DIR / "d1.csv",
            DATA_DIR / "d2.csv",
            DATA_DIR / "d4.csv",
            Path("data/synthetic-data/all_aug.csv"),
        ],
        output_dir=Path("models/indicbart_combined_32k"),
    )


def cli_train_raw_unverified_32k():
    """Train IndicBART on Raw Unverified Synthetic Dataset (Original Clean + Raw Unverified Gemma-2 Synthetic including flawed pairs)."""
    train_indicbart_3fold_cv(
        csv_paths=[DATA_DIR / "raw_unverified_combined.csv"],
        output_dir=Path("models/indicbart_raw_unverified_32k"),
    )



