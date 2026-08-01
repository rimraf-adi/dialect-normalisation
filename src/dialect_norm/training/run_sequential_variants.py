"""
Sequential Runner for IndicBART Fine-Tuning across requested variants:
1. D2 (Ahirani)
2. D4 (Varhadi)
3. D1 + D2 Combined
4. D1 + D2 + D4 Combined (All dialects - 16,163 clean pairs)
"""

import logging
import sys
from pathlib import Path

from dialect_norm.training.indicbart_trainer import (
    cli_train_d2,
    cli_train_d4,
    cli_train_d1d2,
    cli_train_all,
)

logger = logging.getLogger("dialect_norm.training_runner")

def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    variants = [
        ("D2 (Ahirani)", cli_train_d2),
        ("D4 (Varhadi)", cli_train_d4),
        ("D1 + D2 Combined", cli_train_d1d2),
        ("All Dialects (D1 + D2 + D4 Combined)", cli_train_all),
    ]

    logger.info("=" * 80)
    logger.info("STARTING SEQUENTIAL TRAINING FOR: D2, D4, D1+D2, and D1+D2+D4")
    logger.info("=" * 80)

    completed = []
    failed = []

    for idx, (name, train_fn) in enumerate(variants, 1):
        logger.info("\n" + "#" * 80)
        logger.info(f"VARIANT {idx}/{len(variants)}: Running training for {name}")
        logger.info("#" * 80 + "\n")

        try:
            train_fn()
            completed.append(name)
            logger.info(f"SUCCESS: Completed variant {name}")
        except Exception as e:
            logger.error(f"FAILURE: Variant {name} encountered an error: {e}", exc_info=True)
            failed.append((name, str(e)))

    logger.info("\n" + "=" * 80)
    logger.info("SEQUENTIAL TRAINING SUITE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Variants Executed: {len(variants)}")
    logger.info(f"Successfully Completed : {len(completed)} / {len(variants)}")
    for c in completed:
        logger.info(f"  [OK] {c}")

    if failed:
        logger.info(f"Failed Variants        : {len(failed)}")
        for f_name, f_err in failed:
            logger.info(f"  [FAILED] {f_name}: {f_err}")
    else:
        logger.info("ALL VARIANTS COMPLETED WITH ZERO ERRORS!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
