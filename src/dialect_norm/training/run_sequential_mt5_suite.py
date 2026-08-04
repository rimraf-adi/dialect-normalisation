"""
Sequential Runner Suite for google/mt5-small across all 16k and 32k variants (D1, D2, D4, D124).
Skips variants that already have a completed cv_summary.yaml.
"""

import sys
import logging
from pathlib import Path
from dialect_norm.training.mt5_trainer import (
    train_mt5_d1_16k,
    train_mt5_d2_16k,
    train_mt5_d4_16k,
    train_mt5_all_16k,
    train_mt5_d1_32k,
    train_mt5_d2_32k,
    train_mt5_d4_32k,
    train_mt5_all_32k,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("sequential_mt5_suite")

def main():
    variants = [
        ("D1 16k (Malvani Original)", train_mt5_d1_16k, Path("models/mt5_d1_16k/cv_summary.yaml")),
        ("D2 16k (Ahirani Original)", train_mt5_d2_16k, Path("models/mt5_d2_16k/cv_summary.yaml")),
        ("D4 16k (Varhadi Original)", train_mt5_d4_16k, Path("models/mt5_d4_16k/cv_summary.yaml")),
        ("D124 16k (Combined Original)", train_mt5_all_16k, Path("models/mt5_combined_16k/cv_summary.yaml")),
        ("D1 32k (Malvani Expanded)", train_mt5_d1_32k, Path("models/mt5_d1_32k/cv_summary.yaml")),
        ("D2 32k (Ahirani Expanded)", train_mt5_d2_32k, Path("models/mt5_d2_32k/cv_summary.yaml")),
        ("D4 32k (Varhadi Expanded)", train_mt5_d4_32k, Path("models/mt5_d4_32k/cv_summary.yaml")),
        ("D124 32k (Combined Expanded)", train_mt5_all_32k, Path("models/mt5_combined_32k/cv_summary.yaml")),
    ]

    total = len(variants)
    logger.info(f"================================================================================")
    logger.info(f"STARTING mT5-small SEQUENTIAL FINETUNING SUITE ({total} VARIANTS)")
    logger.info(f"================================================================================")

    completed = 0
    for name, func, summary_path in variants:
        if summary_path.exists():
            logger.info(f"---> Skipping already completed variant: {name} (summary found at {summary_path})")
            completed += 1
            continue
        logger.info(f"\n---> Starting Variant {completed + 1}/{total}: {name}")
        try:
            func()
            completed += 1
            logger.info(f"--- [OK] Successfully completed: {name}")
        except Exception as e:
            logger.error(f"--- [FAILED] Error running {name}: {e}", exc_info=True)

    logger.info(f"\n================================================================================")
    logger.info(f"mT5-small SUITE FINISHED: {completed}/{total} variants completed cleanly")
    logger.info(f"================================================================================")

if __name__ == "__main__":
    main()
