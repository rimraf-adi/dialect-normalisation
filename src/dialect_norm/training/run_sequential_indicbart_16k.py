import logging
import sys
import time
from pathlib import Path

from dialect_norm.training.indicbart_trainer import (
    cli_train_d1,
    cli_train_d2,
    cli_train_d4,
    cli_train_all,
)

logger = logging.getLogger("dialect_norm.training.indicbart_runner")

def setup_logger():
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

def main():
    setup_logger()
    logger.info("=" * 80)
    logger.info("STARTING SEQUENTIAL IndicBART 16k RE-TRAINING SUITE (4 VARIANTS)")
    logger.info("Variants: D1 Malvani, D2 Ahirani, D4 Varhadi, D124 Combined")
    logger.info("=" * 80)

    tasks = [
        ("D1 Malvani 16k", cli_train_d1),
        ("D2 Ahirani 16k", cli_train_d2),
        ("D4 Varhadi 16k", cli_train_d4),
        ("D124 Combined 16k", cli_train_all),
    ]

    start_suite = time.time()
    completed = 0

    for name, func in tasks:
        logger.info(f"\n" + "=" * 80)
        logger.info(f"STARTING IndicBART 16k VARIANT: {name}")
        logger.info("=" * 80)
        t0 = time.time()

        try:
            func()
            elapsed = time.time() - t0
            completed += 1
            logger.info(f"--- [SUCCESS] {name} completed in {elapsed / 60:.2f} minutes ---")
        except Exception as e:
            logger.error(f"--- [FAILED] Error running {name}: {e}", exc_info=True)

    total_elapsed = time.time() - start_suite
    logger.info("\n" + "=" * 80)
    logger.info(f"IndicBART 16k SUITE FINISHED: {completed}/{len(tasks)} variants completed in {total_elapsed / 60:.2f} minutes")
    logger.info("=" * 80)

    # Run final held-out test evaluation suite for IndicBART and mT5
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING FINAL HELDOUT EVALUATION SUITE FOR ALL VARIANTS (INDICBART & mT5)")
    logger.info("=" * 80)
    try:
        from dialect_norm.evaluation.eval_mr_indicbart_mt5_wer import main as run_eval
        run_eval()
        logger.info("--- [SUCCESS] Final Heldout Evaluation Suite Finished! ---")
    except Exception as e:
        logger.error(f"--- [FAILED] Error running final heldout evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
