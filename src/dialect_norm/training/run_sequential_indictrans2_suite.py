"""
Sequential Runner Suite for IndicTrans2 across all 16k and 32k variants (D1, D2, D4, D124).
"""

import sys
import logging
from dialect_norm.training.indictrans2_trainer import (
    train_indictrans2_d1_16k,
    train_indictrans2_d2_16k,
    train_indictrans2_d4_16k,
    train_indictrans2_all_16k,
    train_indictrans2_d1_32k,
    train_indictrans2_d2_32k,
    train_indictrans2_d4_32k,
    train_indictrans2_all_32k,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("sequential_indictrans2_suite")

def main():
    variants = [
        ("D1 16k (Malvani Original)", train_indictrans2_d1_16k),
        ("D2 16k (Ahirani Original)", train_indictrans2_d2_16k),
        ("D4 16k (Varhadi Original)", train_indictrans2_d4_16k),
        ("D124 16k (Combined Original)", train_indictrans2_all_16k),
        ("D1 32k (Malvani Expanded)", train_indictrans2_d1_32k),
        ("D2 32k (Ahirani Expanded)", train_indictrans2_d2_32k),
        ("D4 32k (Varhadi Expanded)", train_indictrans2_d4_32k),
        ("D124 32k (Combined Expanded)", train_indictrans2_all_32k),
    ]

    total = len(variants)
    logger.info(f"================================================================================")
    logger.info(f"STARTING IndicTrans2 SEQUENTIAL FINETUNING SUITE ({total} VARIANTS)")
    logger.info(f"================================================================================")

    completed = 0
    for name, func in variants:
        logger.info(f"\n---> Starting Variant {completed + 1}/{total}: {name}")
        try:
            func()
            completed += 1
            logger.info(f"--- [OK] Successfully completed: {name}")
        except Exception as e:
            logger.error(f"--- [FAILED] Error running {name}: {e}", exc_info=True)

    logger.info(f"\n================================================================================")
    logger.info(f"IndicTrans2 SUITE FINISHED: {completed}/{total} variants completed cleanly")
    logger.info(f"================================================================================")

if __name__ == "__main__":
    main()
