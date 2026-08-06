"""
Build Raw Unverified Synthetic Dataset for Ablation Study.
Combines original 16k clean pairs with all raw unverified Gemma-2 synthetic parallel output parts
(including flawed/rejected pairs in flawed.csv).
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger("dialect_norm.build_unverified")

def build_unverified_dataset(data_dir: Path = Path("data/synthetic_parallel")):
    out_file = data_dir / "raw_unverified_combined.csv"
    
    clean_paths = [data_dir / "d1.csv", data_dir / "d2.csv", data_dir / "d4.csv"]
    raw_parts = sorted(data_dir.glob("marathi_parallel_part_*.csv"))
    
    fieldnames = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]
    
    rows = []
    seen = set()
    
    # 1. Load clean 16k original dataset
    clean_count = 0
    for p in clean_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_text = row.get("dialect_text", "").strip()
                s_text = row.get("standard_text", "").strip()
                if d_text and s_text:
                    key = (d_text, s_text)
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
                        clean_count += 1
                        
    # 2. Load all raw unverified Gemma-2 output parts (including flawed rows)
    raw_count = 0
    for p in raw_parts:
        with open(p, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_text = row.get("dialect_text", "").strip()
                s_text = row.get("standard_text", "").strip()
                if d_text and s_text:
                    key = (d_text, s_text)
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
                        raw_count += 1

    with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"Created raw unverified combined dataset at: {out_file}")
    logger.info(f"Total samples: {len(rows):,} (Clean: {clean_count:,}, Raw Unverified Synthetic: {raw_count:,})")
    print(f"[OK] Unverified Dataset Created: {len(rows):,} total parallel pairs.")
    return out_file

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_unverified_dataset()
