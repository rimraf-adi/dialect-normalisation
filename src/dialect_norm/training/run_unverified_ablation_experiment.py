"""
Execution runner for Verification Engine Impact Ablation Study.
Trains IndicBART and mT5-Small on Raw Unverified Synthetic Data and evaluates metrics against Filtered Clean benchmarks.
Configured with auto-flushing logging handlers for live streaming updates.
"""

import logging
import sys
import time
from pathlib import Path

from dialect_norm.training.indicbart_trainer import cli_train_raw_unverified_32k
from dialect_norm.training.mt5_trainer import train_mt5_raw_unverified_32k

log_file = Path("logs/train_raw_unverified_ablation.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger = logging.getLogger("dialect_norm.unverified_ablation")

def main():
    handlers = [
        FlushingFileHandler(log_file, encoding="utf-8"),
        FlushingStreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    # Force root logger handlers to flush
    for h in logging.getLogger().handlers:
        h.flush()

    logger.info("=" * 80)
    logger.info("STARTING VERIFICATION ENGINE ABLATION EXPERIMENT (RAW VS FILTERED SYNTHETIC DATA)")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # 1. IndicBART Raw Unverified Synthetic Training
    logger.info("\n>>> PHASE 1: Training IndicBART on Raw Unverified Synthetic Data (19,914 parallel pairs)")
    try:
        cli_train_raw_unverified_32k()
        logger.info("[SUCCESS] IndicBART Raw Unverified Training Complete.")
    except Exception as e:
        logger.error(f"[ERROR] IndicBART Raw Unverified Training failed: {e}", exc_info=True)

    # 2. mT5-Small Raw Unverified Synthetic Training
    logger.info("\n>>> PHASE 2: Training mT5-Small on Raw Unverified Synthetic Data (19,914 parallel pairs)")
    try:
        train_mt5_raw_unverified_32k()
        logger.info("[SUCCESS] mT5-Small Raw Unverified Training Complete.")
    except Exception as e:
        logger.error(f"[ERROR] mT5-Small Raw Unverified Training failed: {e}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"VERIFICATION ENGINE ABLATION COMPLETED IN {elapsed/60:.2f} MINUTES")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
