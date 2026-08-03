"""
Sequential Runner for IndicBART Fine-Tuning across requested 32k (Original + Synthetic) variants:
1. D1 Malvani 32k (11,145 pairs)
2. D2 Ahirani 32k (11,035 pairs)
3. D4 Varhadi 32k (10,155 pairs)
4. D124 All Dialects Combined 32k (32,335 clean parallel pairs)
"""

import logging
import sys
from pathlib import Path

from dialect_norm.training.indicbart_trainer import (
    cli_train_d1_32k,
    cli_train_d2_32k,
    cli_train_d4_32k,
    cli_train_all_32k,
)

logger = logging.getLogger("dialect_norm.training_runner_32k")

def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    variants = [
        ("D1 (Malvani 32k - 11,145 pairs)", cli_train_d1_32k),
        ("D2 (Ahirani 32k - 11,035 pairs)", cli_train_d2_32k),
        ("D4 (Varhadi 32k - 10,155 pairs)", cli_train_d4_32k),
        ("D124 All Dialects Combined 32k (32,335 pairs)", cli_train_all_32k),
    ]

    logger.info("=" * 80)
    logger.info("STARTING SEQUENTIAL 32K FINE-TUNING SUITE: D1, D2, D4, and D124")
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
    logger.info("SEQUENTIAL 32K TRAINING SUITE SUMMARY")
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
        logger.info("ALL 32K VARIANTS COMPLETED WITH ZERO ERRORS!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
