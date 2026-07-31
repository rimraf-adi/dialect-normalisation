"""
Merges double-verified clean sentences from corrected.csv into d1.csv, d2.csv, and d4.csv,
and updates flawed.csv to retain ONLY remaining unresolved flawed rows.
"""

import csv
from pathlib import Path

def main():
    data_dir = Path("data/synthetic_parallel")
    corrected_file = data_dir / "corrected.csv"
    fieldnames = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]

    if not corrected_file.exists():
        print(f"File {corrected_file.resolve()} does not exist.")
        return

    corrected_clean_rows = []
    remaining_flawed_rows = []

    with open(corrected_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            is_clean = (str(r.get("verified_clean", "")).lower() == "true")
            clean_row = {k: r.get(k, "") for k in fieldnames}
            if is_clean:
                corrected_clean_rows.append(clean_row)
            else:
                flawed_row = dict(clean_row)
                audit_reason = r.get("audit_reason", "Failed audit")
                flawed_row["flaw_reason"] = f"Unresolved: {audit_reason}"
                remaining_flawed_rows.append(flawed_row)

    print(f"Total Corrected CSV Rows Processed: {len(corrected_clean_rows) + len(remaining_flawed_rows):,}")
    print(f"Clean rows to merge into datasets: {len(corrected_clean_rows):,}")
    print(f"Remaining flawed rows to keep in flawed.csv: {len(remaining_flawed_rows):,}")

    d1_rows, d2_rows, d4_rows = [], [], []

    for dial_code, target_list in [("D1", d1_rows), ("D2", d2_rows), ("D4", d4_rows)]:
        csv_file = data_dir / f"{dial_code.lower()}.csv"
        if csv_file.exists():
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    target_list.append({k: r.get(k, "") for k in fieldnames})

    before_d1, before_d2, before_d4 = len(d1_rows), len(d2_rows), len(d4_rows)

    for r in corrected_clean_rows:
        dial = r.get("dialect", "D1").upper().strip()
        if dial == "D1":
            d1_rows.append(r)
        elif dial == "D2":
            d2_rows.append(r)
        elif dial == "D4":
            d4_rows.append(r)
        else:
            d1_rows.append(r)

    # Write updated target CSV files
    for dial_code, target_list in [("D1", d1_rows), ("D2", d2_rows), ("D4", d4_rows)]:
        csv_file = data_dir / f"{dial_code.lower()}.csv"
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(target_list)

    # Overwrite flawed.csv with ONLY remaining unresolved flawed rows
    flawed_fieldnames = fieldnames + ["flaw_reason"]
    with open(data_dir / "flawed.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flawed_fieldnames)
        writer.writeheader()
        writer.writerows(remaining_flawed_rows)

    print("=" * 70)
    print(f"d1.csv updated: {before_d1:,} -> {len(d1_rows):,} clean rows (+{len(d1_rows)-before_d1:,})")
    print(f"d2.csv updated: {before_d2:,} -> {len(d2_rows):,} clean rows (+{len(d2_rows)-before_d2:,})")
    print(f"d4.csv updated: {before_d4:,} -> {len(d4_rows):,} clean rows (+{len(d4_rows)-before_d4:,})")
    print(f"flawed.csv updated: {len(remaining_flawed_rows):,} remaining unresolved rows")
    print("=" * 70)

if __name__ == "__main__":
    main()
